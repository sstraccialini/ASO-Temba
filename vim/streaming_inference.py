"""
streaming_inference.py
======================
Streaming (online causal) inference and evaluation for MS-Temba.

Usage examples
--------------
# Full-sequence causal inference (offline-style but with causal SSMs):
python streaming_inference.py \\
    --weights path/to/causal_best_model.pth \\
    --dataset tsu \\
    --backbone i3d \\
    --rgb_root /path/to/features \\
    --mode rgb \\
    --num_clips 2500 \\
    --stream_chunk_size 0

# True chunk-by-chunk streaming inference:
python streaming_inference.py \\
    --weights path/to/causal_best_model.pth \\
    --dataset tsu \\
    --backbone i3d \\
    --rgb_root /path/to/features \\
    --mode rgb \\
    --num_clips 2500 \\
    --stream_chunk_size 25

Chunk size modes
----------------
0   Full-sequence offline causal forward (single call, fastest).
    Numerically identical to chunk_size=1 but no streaming loop overhead.
    Use this to benchmark causal vs. bidirectional performance delta.

1   True frame-by-frame streaming via stream_forward().
    Maximum per-frame latency visibility; slowest due to loop overhead.
    Use this to demonstrate minimum buffering / zero lookahead.

N   N-frame chunk streaming.
    Trade-off between batch efficiency and online latency.
    E.g. N=25 mirrors a ~1 s buffer at 25 fps; N=50 mirrors ~2 s buffer.

All three modes produce numerically equivalent outputs (up to float32 rounding)
because they share the same causal SSM weights.

Saved outputs
-------------
evaluation_metrics.csv   Same headline numbers as MSTemba_main.py evaluation
evaluation_metrics.pkl   Same dict saved as pickle for downstream analysis
eval_full_probs.pkl      Per-video probability tensors (matches MSTemba_main.py)
streaming_latency.csv    Per-video inference timing and real-time factor
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MS-Temba streaming / causal inference")
    p.add_argument('--weights', type=str, required=True, help='Path to trained checkpoint')
    p.add_argument('--dataset', type=str, default='tsu', choices=['charades', 'tsu'])
    p.add_argument('--backbone', type=str, default='i3d', choices=['i3d', 'clip'])
    p.add_argument('--rgb_root', type=str, required=True)
    p.add_argument('--mode', type=str, default='rgb')
    p.add_argument('--num_clips', type=int, default=2500)
    p.add_argument('--skip', type=int, default=0)
    p.add_argument('--fuser', type=str, default='token-attention',
                   choices=['sum', 'weighted', 'token-attention', 'cross-token-attention', 'attention'])
    p.add_argument('--stream_chunk_size', type=int, default=1,
                   help=(
                       'Frames per streaming chunk. '
                       '0 = full sequence (single causal forward, fastest). '
                       '1 = frame-by-frame (true streaming, most latency detail). '
                       'N = N-frame chunk (balance between throughput and latency).'
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
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
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
# Each function receives inputs already squeezed to [B, C, T] and on GPU.
# Returns raw logits [B, T, num_classes] (before sigmoid, before masking).
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_offline_causal(model, inputs_ct, gpu):
    """Full-sequence causal forward.  inputs_ct: [B, C, T] on GPU."""
    outputs, block_outputs, _, _ = model(inputs_ct)
    return outputs, block_outputs


@torch.no_grad()
def run_streaming(model, inputs_ct, chunk_size, gpu):
    """
    Chunk-by-chunk streaming forward.  inputs_ct: [B, C, T] on GPU.

    Returns
    -------
    logits       : [B, T, num_classes]  raw (no sigmoid)
    block_logits : list of [B, T, num_classes]
    chunk_times  : list of per-chunk wall-clock seconds
    first_chunk_latency : seconds to complete the very first chunk
    """
    B = inputs_ct.shape[0]
    T = inputs_ct.shape[2]
    state = model.init_stream_state(batch_size=B, device=inputs_ct.device)

    all_logits = []
    all_block_logits = [[] for _ in range(3)]
    chunk_times = []
    first_chunk_latency = None

    step = max(chunk_size, 1)
    for start in range(0, T, step):
        chunk = inputs_ct[:, :, start: start + step]   # [B, C, step]
        t0 = time.time()
        logits_chunk, block_preds_chunk, state = model.stream_forward(chunk, state)
        elapsed = time.time() - t0
        chunk_times.append(elapsed)
        if first_chunk_latency is None:
            first_chunk_latency = elapsed

        all_logits.append(logits_chunk)
        for i, bp in enumerate(block_preds_chunk):
            all_block_logits[i].append(bp)

    logits = torch.cat(all_logits, dim=1)                            # [B, T, classes]
    block_logits = [torch.cat(bl, dim=1) for bl in all_block_logits]
    return logits, block_logits, chunk_times, first_chunk_latency


# ---------------------------------------------------------------------------
# Evaluation loop  (matches val_step in MSTemba_main.py)
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, args, classes, gpu):
    """
    Run evaluation matching the metrics reported in MSTemba_main.py val_step.

    Extra streaming metrics collected here:
      - per-video wall-clock inference time
      - per-frame inference time (ms)
      - real-time factor  (inference_time / video_duration;  <1 = faster than RT)
      - first-chunk latency (streaming modes only)
    """
    model.eval()

    apm = APMeter()
    sampled_apm = APMeter()
    block_apms = [APMeter() for _ in range(3)]
    block_sampled_apms = [APMeter() for _ in range(3)]

    full_probs = {}
    tot_loss = 0.0
    num_iter = 0

    # Per-video timing records for the streaming latency CSV
    latency_rows = []

    for data in dataloader:
        inputs, mask, labels, other, hm = data
        vid_id = other[0][0]
        vid_duration = float(other[1][0])          # seconds

        # Move to GPU; squeeze spatial singleton dims [B, C, T, 1, 1] → [B, C, T]
        inputs_ct = inputs.squeeze(3).squeeze(3).cuda(gpu)   # [B, C, T]
        mask_gpu = mask.cuda(gpu)
        labels_gpu = labels.cuda(gpu)

        T = inputs_ct.shape[2]
        video_fps = T / vid_duration if vid_duration > 0 else float('nan')

        # ── Inference ─────────────────────────────────────────────────────────
        t_start = time.time()
        chunk_times = None
        first_chunk_latency = None

        if args.stream_chunk_size == 0:
            # Full-sequence offline causal forward
            logits, block_logits = run_offline_causal(model, inputs_ct, gpu)
        else:
            # Chunk-by-chunk streaming forward
            logits, block_logits, chunk_times, first_chunk_latency = run_streaming(
                model, inputs_ct, args.stream_chunk_size, gpu
            )

        elapsed = time.time() - t_start

        # ── Apply sigmoid + mask to get probabilities ─────────────────────────
        probs = torch.sigmoid(logits) * mask_gpu.unsqueeze(2)
        block_probs = [torch.sigmoid(bl) * mask_gpu.unsqueeze(2) for bl in block_logits]

        # ── BCE loss (matches run_network in MSTemba_main.py) ─────────────────
        loss = F.binary_cross_entropy_with_logits(logits, labels_gpu, reduction='sum')
        loss = loss / torch.sum(mask_gpu)
        tot_loss += loss.item()
        num_iter += 1

        # ── Numpy views ───────────────────────────────────────────────────────
        p_np = probs.data.cpu().numpy()[0]          # [T, classes]
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

        # ── Latency record ────────────────────────────────────────────────────
        per_frame_ms = (elapsed / T * 1000) if T > 0 else float('nan')
        real_time_factor = (elapsed / vid_duration) if vid_duration > 0 else float('nan')

        row = {
            'vid_id': vid_id,
            'num_frames': T,
            'video_duration_s': vid_duration,
            'video_fps': video_fps,
            'inference_time_s': elapsed,
            'per_frame_latency_ms': per_frame_ms,
            'real_time_factor': real_time_factor,
            'first_chunk_latency_ms': (first_chunk_latency * 1000
                                       if first_chunk_latency is not None
                                       else elapsed * 1000),
            'stream_chunk_size': args.stream_chunk_size,
        }
        if chunk_times is not None:
            row['mean_chunk_time_ms'] = float(np.mean(chunk_times)) * 1000
            row['max_chunk_time_ms'] = float(np.max(chunk_times)) * 1000
        latency_rows.append(row)

    # ── Aggregate mAP metrics ──────────────────────────────────────────────────
    val_loss = tot_loss / max(num_iter, 1)

    val_map = (torch.sum(100 * apm.value()) /
               torch.nonzero(100 * apm.value()).size()[0])
    sample_val_map = (torch.sum(100 * sampled_apm.value()) /
                      torch.nonzero(100 * sampled_apm.value()).size()[0])

    block_val_maps = []
    block_sample_val_maps = []
    for i in range(3):
        bm = (torch.sum(100 * block_apms[i].value()) /
              torch.nonzero(100 * block_apms[i].value()).size()[0])
        bsm = (torch.sum(100 * block_sampled_apms[i].value()) /
               torch.nonzero(100 * block_sampled_apms[i].value()).size()[0])
        block_val_maps.append(float(bm))
        block_sample_val_maps.append(float(bsm))

    # ── Aggregate timing ──────────────────────────────────────────────────────
    all_times = [r['inference_time_s'] for r in latency_rows]
    all_rtf = [r['real_time_factor'] for r in latency_rows
               if not (isinstance(r['real_time_factor'], float)
                       and np.isnan(r['real_time_factor']))]
    all_pf_ms = [r['per_frame_latency_ms'] for r in latency_rows
                 if not (isinstance(r['per_frame_latency_ms'], float)
                         and np.isnan(r['per_frame_latency_ms']))]
    all_fc_ms = [r['first_chunk_latency_ms'] for r in latency_rows]

    return (full_probs, val_loss, float(val_map), float(sample_val_map),
            block_val_maps, block_sample_val_maps, latency_rows,
            {
                'avg_inference_time_s':        float(np.mean(all_times)),
                'avg_per_frame_latency_ms':    float(np.mean(all_pf_ms)) if all_pf_ms else float('nan'),
                'avg_real_time_factor':        float(np.mean(all_rtf)) if all_rtf else float('nan'),
                'avg_first_chunk_latency_ms':  float(np.mean(all_fc_ms)),
                'stream_chunk_size':           args.stream_chunk_size,
                'mode': ('frame-by-frame' if args.stream_chunk_size == 1
                         else 'offline_causal' if args.stream_chunk_size == 0
                         else f'{args.stream_chunk_size}-frame_chunks'),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset / split file ─────────────────────────────────────────────────
    if args.dataset == 'charades':
        split_file = '../data/charades.json'
        classes = 157
    else:
        split_file = '../data/smarthome.json'
        classes = 51

    # collate_fn_unisize pads all sequences to args.num_clips.
    # The authors did not implement the variable-length (non-padded) collate,
    # so unisize padding is always used.
    collate_fn_obj = collate_fn_unisize(args.num_clips)
    collate_fn = collate_fn_obj.charades_collate_fn_unisize

    dataset = Dataset(split_file, 'testing', args.rgb_root,
                      args.batch_size, classes, args.num_clips, args.skip)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    in_feat_dim = 1024 if args.backbone == 'i3d' else 768
    model = create_model(
        'mstemba',
        pretrained=False,
        num_classes=classes,
        in_feat_dim=in_feat_dim,
        fuser=args.fuser,
        causal=True,                         # streaming requires causal mode
        causal_consistency_loss_weight=0.0,  # no training loss during inference
        causal_consistency_margin=0.1,
    )
    _load_checkpoint(model, args.weights)
    model.cuda(args.gpu)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode_str = ('frame-by-frame' if args.stream_chunk_size == 1
                else 'offline causal' if args.stream_chunk_size == 0
                else f'{args.stream_chunk_size}-frame chunks')
    print(f"Loaded model: {n_params:,} trainable parameters")
    print(f"Streaming chunk size: {args.stream_chunk_size}  ({mode_str})")

    # ── Evaluation ────────────────────────────────────────────────────────────
    t_eval_start = time.time()
    (full_probs, val_loss, val_map, sample_val_map,
     block_val_maps, block_sample_val_maps,
     latency_rows, timing_summary) = evaluate(
        model, dataloader, args, classes, args.gpu
    )
    eval_time = time.time() - t_eval_start

    # ── Build metrics dict (matches MSTemba_main.py evaluation section) ───────
    metrics = {
        # ── Provenance ────────────────────────────────────────────────────────
        'model_path':             args.weights,
        'dataset':                args.dataset,
        'mode':                   args.mode,
        'num_classes':            classes,
        'num_parameters':         n_params,
        'eval_time_seconds':      round(eval_time, 2),
        'val_loss':               val_loss,

        # ── Headline numbers (same as MSTemba_main.py) ────────────────────────
        'per_frame_mAP':          val_map,
        'sampled_mAP_25':         sample_val_map,

        # ── Per-block mAPs ────────────────────────────────────────────────────
        'block1_per_frame_mAP':   block_val_maps[0],
        'block2_per_frame_mAP':   block_val_maps[1],
        'block3_per_frame_mAP':   block_val_maps[2],
        'block1_sampled_mAP_25':  block_sample_val_maps[0],
        'block2_sampled_mAP_25':  block_sample_val_maps[1],
        'block3_sampled_mAP_25':  block_sample_val_maps[2],

        # ── Streaming-specific ────────────────────────────────────────────────
        'stream_chunk_size':               timing_summary['stream_chunk_size'],
        'streaming_mode':                  timing_summary['mode'],
        'avg_inference_time_s':            timing_summary['avg_inference_time_s'],
        'avg_per_frame_latency_ms':        timing_summary['avg_per_frame_latency_ms'],
        'avg_real_time_factor':            timing_summary['avg_real_time_factor'],
        'avg_first_chunk_latency_ms':      timing_summary['avg_first_chunk_latency_ms'],
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STREAMING INFERENCE SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<40s}: {v:.4f}")
        else:
            print(f"  {k:<40s}: {v}")
    print("=" * 60)

    # ── Save evaluation_metrics.csv ───────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, v])
    print(f"Metrics CSV saved to: {csv_path}")

    # ── Save evaluation_metrics.pkl ───────────────────────────────────────────
    pkl_path = os.path.join(args.output_dir, 'evaluation_metrics.pkl')
    pickle.dump(metrics, open(pkl_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"Metrics pickle saved to: {pkl_path}")

    # ── Save eval_full_probs.pkl (matches MSTemba_main.py naming) ────────────
    probs_path = os.path.join(args.output_dir, 'eval_full_probs.pkl')
    pickle.dump(full_probs, open(probs_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"Per-video probabilities saved to: {probs_path}")

    # ── Save streaming_latency.csv (per-video timing detail) ─────────────────
    if latency_rows:
        lat_path = os.path.join(args.output_dir, 'streaming_latency.csv')
        fieldnames = list(latency_rows[0].keys())
        with open(lat_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(latency_rows)
        print(f"Per-video latency saved to: {lat_path}")


if __name__ == '__main__':
    main()