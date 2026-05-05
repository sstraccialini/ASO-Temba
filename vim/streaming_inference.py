"""
streaming_inference.py
======================
Streaming (online causal) inference and evaluation for MS-Temba.

Accuracy vs. latency
--------------------
mAP is always evaluated using the fast offline causal forward (single parallel
scan over the full sequence).  This is numerically identical to the streaming
forward (same causal SSM weights, same computation, just batched differently),
so the mAP numbers are valid for any chunk size.

The streaming forward is used only on a small sample of videos (--streaming_demo_n,
default 10) specifically to measure per-frame latency and real-time factor.
This avoids spending hours on 2500 × kernel-launch overhead per video.

Chunk size modes  (for the streaming demo)
------------------------------------------
0   Not used for streaming demo – all timing comes from offline causal.
    Use this when you only want the accuracy numbers.

1   Frame-by-frame streaming (maximum latency visibility, slowest).
    Demonstrates zero-lookahead, minimum buffering.

N   N-frame chunk streaming (throughput / latency trade-off).
    E.g. N=25 mirrors ~1 s buffer at 25 fps.

Usage examples
--------------
# Accuracy only (fastest):
python streaming_inference.py \\
    --weights ../outputs/causal-tsu_i3d/best_model.pth \\
    --dataset tsu --backbone i3d \\
    --rgb_root /path/to/tsu_features_i3d \\
    --stream_chunk_size 0

# Accuracy + streaming latency demo on 10 videos with chunk_size=1:
python streaming_inference.py \\
    --weights ../outputs/causal-tsu_i3d/best_model.pth \\
    --dataset tsu --backbone i3d \\
    --rgb_root /path/to/tsu_features_i3d \\
    --stream_chunk_size 1 --streaming_demo_n 10

# Same with 25-frame chunks:
python streaming_inference.py \\
    ... --stream_chunk_size 25 --streaming_demo_n 10

Saved outputs
-------------
evaluation_metrics.csv   Accuracy metrics matching MSTemba_main.py evaluation
evaluation_metrics.pkl   Same dict as pickle
eval_full_probs.pkl      Per-video probability tensors
offline_latency.csv      Per-video timing from the offline causal pass (all videos)
streaming_latency.csv    Per-video timing from the streaming pass (demo subset only)
"""

import argparse
import csv
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from timm.models import create_model

import models_MSTemba  # noqa: F401 – registers mstemba with timm
from apmeter import APMeter
from charades_dataloader import Charades as Dataset, collate_fn_unisize
from utils import sampled_25, mask_probs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apm_map(meter):
    """Compute mAP from an APMeter, handling the empty-meter (returns int 0) case."""
    ap = meter.value()
    if not torch.is_tensor(ap):
        return 0.0
    ap100 = ap * 100
    nz = torch.nonzero(ap100)
    if nz.numel() == 0:
        return 0.0
    return float(torch.sum(ap100) / nz.size()[0])


def _save_csv(path, rows):
    """Save a list-of-dicts to CSV."""
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MS-Temba streaming / causal inference")
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--dataset', type=str, default='tsu', choices=['charades', 'tsu'])
    p.add_argument('--backbone', type=str, default='i3d', choices=['i3d', 'clip'])
    p.add_argument('--rgb_root', type=str, required=True)
    p.add_argument('--mode', type=str, default='rgb')
    p.add_argument('--num_clips', type=int, default=2500)
    p.add_argument('--skip', type=int, default=0)
    p.add_argument('--fuser', type=str, default='sum',
                   choices=['sum', 'weighted', 'token-attention',
                            'cross-token-attention', 'attention'])
    p.add_argument('--stream_chunk_size', type=int, default=1,
                   help='Streaming chunk size in frames (0 = offline causal only).')
    p.add_argument('--streaming_demo_n', type=int, default=10,
                   help=(
                       'Number of videos to run in true streaming mode for latency '
                       'measurement. mAP always uses the fast offline causal forward. '
                       'Set -1 to stream every video (very slow for large datasets).'
                   ))
    p.add_argument('--output_dir', type=str, default='./output_streaming')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        # Prefer EMA weights (what training val_map is computed on) over raw model.
        # Priority matches MSTemba_main._select_checkpoint_state_dict.
        state_dict = None
        for key in ('model_ema_state_dict', 'model_ema', 'model', 'state_dict'):
            if key in ckpt and ckpt[key] is not None:
                state_dict = ckpt[key]
                print(f"[INFO] Loading weights from checkpoint key: '{key}'")
                break
        if state_dict is None:
            raise ValueError(f"No recognised weight key found in checkpoint: {list(ckpt.keys())}")
    else:
        state_dict = ckpt
    state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
                  for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys: {missing}")
    if unexpected:
        print(f"[WARNING] Unexpected keys: {unexpected}")


# ---------------------------------------------------------------------------
# Inference helpers
# Both receive inputs already squeezed to [B, C, T] and on GPU.
# Both return raw logits [B, T, num_classes] (before sigmoid / masking).
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_offline_causal(model, inputs_ct):
    """Full-sequence causal forward (single parallel scan)."""
    outputs, block_outputs, _, _ = model(inputs_ct)
    return outputs, block_outputs


@torch.no_grad()
def run_streaming(model, inputs_ct, chunk_size):
    """
    Chunk-by-chunk streaming forward.

    Returns
    -------
    logits             : [B, T, num_classes]  raw logits
    block_logits       : list of 3 × [B, T, num_classes]
    chunk_times        : per-chunk wall-clock seconds
    first_chunk_s      : latency to first output
    """
    B = inputs_ct.shape[0]
    T = inputs_ct.shape[2]
    state = model.init_stream_state(batch_size=B, device=inputs_ct.device)

    all_logits = []
    all_block_logits = [[] for _ in range(3)]
    chunk_times = []
    first_chunk_s = None

    step = max(chunk_size, 1)
    for start in range(0, T, step):
        chunk = inputs_ct[:, :, start: start + step]
        t0 = time.time()
        logits_chunk, block_preds_chunk, state = model.stream_forward(chunk, state)
        elapsed = time.time() - t0
        chunk_times.append(elapsed)
        if first_chunk_s is None:
            first_chunk_s = elapsed
        all_logits.append(logits_chunk)
        for i, bp in enumerate(block_preds_chunk):
            all_block_logits[i].append(bp)

    logits = torch.cat(all_logits, dim=1)
    block_logits = [torch.cat(bl, dim=1) for bl in all_block_logits]
    return logits, block_logits, chunk_times, first_chunk_s


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, args, classes, gpu):
    """
    Compute accuracy metrics (all videos, offline causal) and streaming latency
    metrics (first streaming_demo_n videos, true streaming).

    Rationale
    ---------
    The offline causal forward and the streaming forward produce numerically
    identical outputs (same causal SSM, different scheduling).  Running mAP
    evaluation through the streaming loop would be 100x–1000x slower with no
    change in numbers.  We therefore decouple accuracy from latency measurement.
    """
    model.eval()

    apm = APMeter()
    sampled_apm = APMeter()
    block_apms = [APMeter() for _ in range(3)]
    block_sampled_apms = [APMeter() for _ in range(3)]

    full_probs = {}
    tot_loss = 0.0
    num_iter = 0

    offline_rows = []    # one row per video – offline causal timing
    stream_rows = []     # one row per demo video – streaming timing

    do_streaming = args.stream_chunk_size > 0
    demo_limit = args.streaming_demo_n   # -1 = unlimited

    for data in dataloader:
        inputs, mask, labels, other, hm = data
        vid_id = other[0][0]
        vid_duration = float(other[1][0])

        # Squeeze [B, C, T, 1, 1] → [B, C, T] and move to GPU
        inputs_ct = inputs.squeeze(3).squeeze(3).cuda(gpu)
        mask_gpu = mask.cuda(gpu)
        labels_gpu = labels.cuda(gpu)
        T = inputs_ct.shape[2]
        video_fps = T / vid_duration if vid_duration > 0 else float('nan')

        # ── Offline causal forward (always – for mAP and offline timing) ───────
        t0 = time.time()
        logits, block_logits = run_offline_causal(model, inputs_ct)
        offline_elapsed = time.time() - t0

        # ── Apply sigmoid + mask ─────────────────────────────────────────────
        probs = torch.sigmoid(logits) * mask_gpu.unsqueeze(2)
        block_probs = [torch.sigmoid(bl) * mask_gpu.unsqueeze(2) for bl in block_logits]

        # ── BCE loss ──────────────────────────────────────────────────────────
        loss = F.binary_cross_entropy_with_logits(logits, labels_gpu, reduction='sum')
        tot_loss += (loss / torch.sum(mask_gpu)).item()
        num_iter += 1

        # ── Update accuracy meters ────────────────────────────────────────────
        p_np = probs.data.cpu().numpy()[0]
        l_np = labels.numpy()[0]
        m_np = mask.numpy()[0]

        apm.add(p_np, l_np)
        if m_np.sum() > 25:
            p1, l1 = sampled_25(p_np, l_np, m_np)
            sampled_apm.add(p1, l1)

        for i, bp in enumerate(block_probs):
            bp_np = bp.data.cpu().numpy()[0]
            block_apms[i].add(bp_np, l_np)
            if m_np.sum() > 25:
                p1, l1 = sampled_25(bp_np, l_np, m_np)
                block_sampled_apms[i].add(p1, l1)

        probs_masked = mask_probs(p_np, m_np).squeeze()
        full_probs[vid_id] = probs_masked.T

        # ── Offline latency row ───────────────────────────────────────────────
        offline_rows.append({
            'vid_id': vid_id,
            'num_frames': T,
            'video_duration_s': vid_duration,
            'video_fps': round(video_fps, 2) if np.isfinite(video_fps) else 'nan',
            'inference_time_s': offline_elapsed,
            'per_frame_latency_ms': offline_elapsed / T * 1000 if T > 0 else 'nan',
            'real_time_factor': offline_elapsed / vid_duration if vid_duration > 0 else 'nan',
        })

        # ── Streaming demo (first demo_limit videos only) ─────────────────────
        if do_streaming:
            
            # verify_equivalence(model, inputs_ct)
            
            want_demo = (demo_limit == -1) or (len(stream_rows) < demo_limit)
            if want_demo:
                _, _, chunk_times, first_chunk_s = run_streaming(
                    model, inputs_ct, args.stream_chunk_size
                )
                stream_elapsed = sum(chunk_times)
                stream_rows.append({
                    'vid_id': vid_id,
                    'num_frames': T,
                    'video_duration_s': vid_duration,
                    'video_fps': round(video_fps, 2) if np.isfinite(video_fps) else 'nan',
                    'chunk_size': args.stream_chunk_size,
                    'num_chunks': len(chunk_times),
                    'total_stream_time_s': stream_elapsed,
                    'per_frame_latency_ms': stream_elapsed / T * 1000 if T > 0 else 'nan',
                    'mean_chunk_time_ms': float(np.mean(chunk_times)) * 1000,
                    'max_chunk_time_ms': float(np.max(chunk_times)) * 1000,
                    'first_chunk_latency_ms': first_chunk_s * 1000,
                    'real_time_factor': stream_elapsed / vid_duration if vid_duration > 0 else 'nan',
                })

    # ── Warn if nothing was evaluated ────────────────────────────────────────
    if num_iter == 0:
        print("[WARNING] No videos were evaluated. "
              "Check --rgb_root and that feature files exist.")

    # ── Aggregate accuracy metrics ────────────────────────────────────────────
    val_loss = tot_loss / max(num_iter, 1)
    val_map = _apm_map(apm)
    sample_val_map = _apm_map(sampled_apm)
    block_val_maps = [_apm_map(block_apms[i]) for i in range(3)]
    block_sample_val_maps = [_apm_map(block_sampled_apms[i]) for i in range(3)]

    # ── Aggregate timing summaries ────────────────────────────────────────────
    def _mean(rows, key):
        vals = [r[key] for r in rows if isinstance(r[key], float) and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float('nan')

    offline_summary = {
        'num_videos': len(offline_rows),
        'avg_inference_time_s': _mean(offline_rows, 'inference_time_s'),
        'avg_per_frame_latency_ms': _mean(offline_rows, 'per_frame_latency_ms'),
        'avg_real_time_factor': _mean(offline_rows, 'real_time_factor'),
    }
    stream_summary = {
        'num_demo_videos': len(stream_rows),
        'chunk_size': args.stream_chunk_size,
        'avg_stream_time_s': _mean(stream_rows, 'total_stream_time_s'),
        'avg_per_frame_latency_ms': _mean(stream_rows, 'per_frame_latency_ms'),
        'avg_first_chunk_latency_ms': _mean(stream_rows, 'first_chunk_latency_ms'),
        'avg_real_time_factor': _mean(stream_rows, 'real_time_factor'),
    }

    return (full_probs, val_loss, val_map, sample_val_map,
            block_val_maps, block_sample_val_maps,
            offline_rows, stream_rows, offline_summary, stream_summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == 'charades':
        split_file = '../data/charades.json'
        classes = 157
    else:
        split_file = '../data/smarthome.json'
        classes = 51

    # Authors did not implement variable-length (non-padded) collate;
    # unisize padding is always used.
    collate_fn_obj = collate_fn_unisize(args.num_clips)
    collate_fn = collate_fn_obj.charades_collate_fn_unisize

    dataset = Dataset(split_file, 'testing', args.rgb_root,
                      args.batch_size, classes, args.num_clips, args.skip)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    in_feat_dim = 1024 if args.backbone == 'i3d' else 768
    model = create_model(
        'mstemba',
        pretrained=False,
        num_classes=classes,
        in_feat_dim=in_feat_dim,
        fuser=args.fuser,
        causal=True,
        causal_consistency_loss_weight=0.0,
        causal_consistency_margin=0.1,
    )
    _load_checkpoint(model, args.weights)
    model.cuda(args.gpu)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    demo_desc = (
        'no streaming demo' if args.stream_chunk_size == 0
        else f'streaming demo on {args.streaming_demo_n} videos '
             f'(chunk_size={args.stream_chunk_size})'
    )
    print(f"Loaded model: {n_params:,} trainable parameters")
    print(f"Evaluation mode: offline causal for all videos + {demo_desc}")

    t_start = time.time()
    (full_probs, val_loss, val_map, sample_val_map,
     block_val_maps, block_sample_val_maps,
     offline_rows, stream_rows,
     offline_summary, stream_summary) = evaluate(
        model, dataloader, args, classes, args.gpu
    )
    eval_time = time.time() - t_start

    # ── Build metrics dict ────────────────────────────────────────────────────
    metrics = {
        'model_path':            args.weights,
        'dataset':               args.dataset,
        'mode':                  args.mode,
        'num_classes':           classes,
        'num_parameters':        n_params,
        'eval_time_seconds':     round(eval_time, 2),
        'val_loss':              val_loss,

        'per_frame_mAP':         val_map,
        'sampled_mAP_25':        sample_val_map,

        'block1_per_frame_mAP':  block_val_maps[0],
        'block2_per_frame_mAP':  block_val_maps[1],
        'block3_per_frame_mAP':  block_val_maps[2],
        'block1_sampled_mAP_25': block_sample_val_maps[0],
        'block2_sampled_mAP_25': block_sample_val_maps[1],
        'block3_sampled_mAP_25': block_sample_val_maps[2],

        # Offline causal timing (all videos)
        'offline_avg_inference_time_s':       offline_summary['avg_inference_time_s'],
        'offline_avg_per_frame_latency_ms':   offline_summary['avg_per_frame_latency_ms'],
        'offline_avg_real_time_factor':       offline_summary['avg_real_time_factor'],

        # Streaming timing (demo subset)
        'stream_chunk_size':                  stream_summary['chunk_size'],
        'stream_num_demo_videos':             stream_summary['num_demo_videos'],
        'stream_avg_per_frame_latency_ms':    stream_summary['avg_per_frame_latency_ms'],
        'stream_avg_first_chunk_latency_ms':  stream_summary['avg_first_chunk_latency_ms'],
        'stream_avg_real_time_factor':        stream_summary['avg_real_time_factor'],
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        fmt = f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v)
        print(f"  {k:<44s}: {fmt}")
    print("=" * 60)

    # ── Save files ────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, v])
    print(f"Metrics CSV  : {csv_path}")

    pkl_path = os.path.join(args.output_dir, 'evaluation_metrics.pkl')
    pickle.dump(metrics, open(pkl_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"Metrics PKL  : {pkl_path}")

    probs_path = os.path.join(args.output_dir, 'eval_full_probs.pkl')
    pickle.dump(full_probs, open(probs_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"Probs PKL    : {probs_path}")

    if offline_rows:
        p = os.path.join(args.output_dir, 'offline_latency.csv')
        _save_csv(p, offline_rows)
        print(f"Offline lat. : {p}")

    if stream_rows:
        p = os.path.join(args.output_dir, 'streaming_latency.csv')
        _save_csv(p, stream_rows)
        print(f"Stream lat.  : {p}")


@torch.no_grad()
def verify_equivalence(model, inputs_ct, atol=1e-3):
    print("Verifying numerical equivalence of offline causal and streaming outputs...")
    off_logits, _ = run_offline_causal(model, inputs_ct)
    str_logits_1, _, _, _ = run_streaming(model, inputs_ct, chunk_size=1)
    str_logits_25, _, _, _ = run_streaming(model, inputs_ct, chunk_size=25)
    d1  = (off_logits - str_logits_1 ).abs().max().item()
    d25 = (off_logits - str_logits_25).abs().max().item()
    print(f"max|offline - stream(1)|  = {d1:.2e}")
    print(f"max|offline - stream(25)| = {d25:.2e}")
    assert d1 < atol and d25 < atol, "Causality violation!"

if __name__ == '__main__':
    main()