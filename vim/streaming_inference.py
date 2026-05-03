"""
streaming_inference.py
======================
Streaming (online causal) inference for MS-Temba.

Usage examples
--------------
# Full-sequence causal inference (offline-style but with causal SSMs):
python streaming_inference.py \
    --weights path/to/causal_best_model.pth \
    --dataset tsu \
    --backbone i3d \
    --rgb_root /path/to/features \
    --mode rgb \
    --num_clips 2500 \
    --stream_chunk_size 0          # 0 = full sequence at once (causal, no streaming loop)

# True chunk-by-chunk streaming inference:
python streaming_inference.py \
    --weights path/to/causal_best_model.pth \
    --dataset tsu \
    --backbone i3d \
    --rgb_root /path/to/features \
    --mode rgb \
    --num_clips 2500 \
    --stream_chunk_size 25         # emit predictions every 25 frames

Notes
-----
- The checkpoint must have been trained with --causal flag.
- stream_chunk_size=0 runs the full sequence in one call (uses offline forward),
  which is faster but does not demonstrate frame-by-frame streaming.
- stream_chunk_size=N > 0 runs true frame-by-frame streaming via stream_forward().
  The output is numerically equivalent to the full-sequence causal forward.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from timm.models import create_model

import models_MSTemba  # noqa: F401 – registers mstemba with timm
from apmeter import APMeter
from charades_dataloader import Charades as Dataset, mt_collate_fn as collate_fn
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
                   help='Frames per streaming chunk. 0 = full sequence (offline causal forward).')
    p.add_argument('--output_dir', type=str, default='./output_streaming')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
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


@torch.no_grad()
def run_streaming(model, inputs, mask, chunk_size, gpu):
    """
    Run causal streaming inference on a single video.

    Args:
        model: causal MSTemba (model.causal must be True).
        inputs: [1, C, T] feature tensor (channels-first).
        mask: [1, T] validity mask.
        chunk_size: int, frames per chunk (1 = frame-by-frame).
        gpu: CUDA device index.

    Returns:
        logits: [1, T, num_classes]
        block_logits: list of [1, T, num_classes]
    """
    inputs = inputs.cuda(gpu)
    mask = mask.cuda(gpu)

    T = inputs.shape[2]
    state = model.init_stream_state(batch_size=inputs.shape[0], device=inputs.device)

    all_logits = []
    all_block_logits = [[] for _ in range(3)]

    step = max(chunk_size, 1)
    for start in range(0, T, step):
        chunk = inputs[:, :, start : start + step]  # [B, C, chunk]
        logits_chunk, block_preds_chunk, state = model.stream_forward(chunk, state)
        all_logits.append(logits_chunk)
        for i, bp in enumerate(block_preds_chunk):
            all_block_logits[i].append(bp)

    logits = torch.cat(all_logits, dim=1)                          # [B, T, classes]
    block_logits = [torch.cat(bl, dim=1) for bl in all_block_logits]

    # Apply mask
    logits = F.sigmoid(logits) * mask.unsqueeze(2)
    block_logits = [F.sigmoid(bl) * mask.unsqueeze(2) for bl in block_logits]

    return logits, block_logits


@torch.no_grad()
def run_offline_causal(model, inputs, mask, gpu):
    """
    Full-sequence causal forward (single call, no streaming loop).
    Faster than streaming but identical in output.
    """
    inputs = inputs.cuda(gpu)
    mask = mask.cuda(gpu)
    inputs = inputs.squeeze(3).squeeze(3)
    outputs, block_outputs, _, _ = model(inputs)
    probs = F.sigmoid(outputs) * mask.unsqueeze(2)
    block_probs = [F.sigmoid(bo) * mask.unsqueeze(2) for bo in block_outputs]
    return probs, block_probs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, args, classes):
    model.eval()
    apm = APMeter()
    sampled_apm = APMeter()
    block_apms = [APMeter() for _ in range(3)]
    block_sampled_apms = [APMeter() for _ in range(3)]

    full_probs = {}
    total_time = 0.0
    num_vids = 0

    for data in dataloader:
        inputs, mask, labels, other, hm = data
        vid_id = other[0][0]

        t0 = time.time()

        if args.stream_chunk_size == 0:
            # Offline causal forward (faster, same result)
            probs, block_probs = run_offline_causal(model, inputs, mask, args.gpu)
        else:
            # True streaming
            inputs_streaming = inputs.squeeze(3).squeeze(3)      # [B, T, C]
            inputs_streaming = inputs_streaming.permute(0, 2, 1) # [B, C, T]
            probs, block_probs = run_streaming(
                model, inputs_streaming, mask, args.stream_chunk_size, args.gpu
            )

        elapsed = time.time() - t0
        total_time += elapsed
        num_vids += 1

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

    # Aggregate metrics
    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    sval_map = torch.sum(100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    metrics = {
        'per_frame_mAP': float(val_map),
        'sampled_mAP_25': float(sval_map),
        'avg_inference_time_s': total_time / max(num_vids, 1),
        'stream_chunk_size': args.stream_chunk_size,
        'mode': 'streaming' if args.stream_chunk_size > 0 else 'offline_causal',
    }
    for i in range(3):
        bm = torch.sum(100 * block_apms[i].value()) / torch.nonzero(100 * block_apms[i].value()).size()[0]
        bsm = torch.sum(100 * block_sampled_apms[i].value()) / torch.nonzero(100 * block_sampled_apms[i].value()).size()[0]
        metrics[f'block{i+1}_per_frame_mAP'] = float(bm)
        metrics[f'block{i+1}_sampled_mAP_25'] = float(bsm)

    return metrics, full_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    if args.dataset == 'charades':
        split_file = '../data/charades.json'
        classes = 157
    else:
        split_file = '../data/smarthome.json'
        classes = 51

    dataset = Dataset(split_file, 'testing', args.rgb_root, args.batch_size,
                      classes, args.num_clips, args.skip)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # Model
    in_feat_dim = 1024 if args.backbone == 'i3d' else 768
    model = create_model(
        'mstemba',
        pretrained=False,
        num_classes=classes,
        in_feat_dim=in_feat_dim,
        fuser=args.fuser,
        causal=True,   # streaming inference requires causal mode
        causal_consistency_loss_weight=0.0,
        causal_consistency_margin=0.1,
    )
    _load_checkpoint(model, args.weights)
    model.cuda(args.gpu)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Loaded model: {n_params:,} trainable parameters")
    print(f"Streaming chunk size: {args.stream_chunk_size} "
          f"({'frame-by-frame' if args.stream_chunk_size == 1 else 'offline causal' if args.stream_chunk_size == 0 else f'{args.stream_chunk_size}-frame chunks'})")

    # Run evaluation
    t_start = time.time()
    metrics, full_probs = evaluate(model, dataloader, args, classes)
    t_total = time.time() - t_start

    # Print summary
    print("\n" + "=" * 60)
    print("STREAMING INFERENCE SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<36s}: {v}")
    print(f"  {'total_eval_time_s':<36s}: {t_total:.1f}")
    print("=" * 60)

    # Save
    import csv
    csv_path = os.path.join(args.output_dir, 'streaming_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, v])
    print(f"Metrics saved to {csv_path}")

    pkl_path = os.path.join(args.output_dir, 'streaming_probs.pkl')
    pickle.dump(full_probs, open(pkl_path, 'wb'), pickle.HIGHEST_PROTOCOL)
    print(f"Probabilities saved to {pkl_path}")


if __name__ == '__main__':
    main()