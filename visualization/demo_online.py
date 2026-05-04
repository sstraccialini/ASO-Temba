"""
visualization/demo_online.py
============================
Live online-inference demo for MS-Temba (causal streaming mode).

Because the pipeline operates on pre-extracted I3D / CLIP features (no raw
pixel frames are stored), the visualisation renders:

  ┌───────────────────────────────────────────────────────────────┐
  │  MS-Temba · Online Inference Demo                            │
  │  Video: XXXXX   ──────────────────────────●───────  123/500  │
  ├───────────────────────────────────────────────────────────────┤
  │                                                               │
  │   Feature activation heatmap  (sliding 64-frame window)      │
  │   Columns = 128 feature channels · Rows = recent frames      │
  │                                                               │
  ├────────────────────────────────┬──────────────────────────────┤
  │  LIVE PREDICTIONS              │  GROUND TRUTH                │
  │                                │                              │
  │  ████████████  Sitting  (0.87) │  • Sitting on a chair        │
  │  ████████      Reading  (0.61) │  • Holding a book            │
  │  ██████        Writing  (0.44) │                              │
  │  █████         Walking  (0.32) │                              │
  │  ███           Eating   (0.18) │                              │
  └────────────────────────────────┴──────────────────────────────┘

Usage
-----
# Save to MP4 (works on headless HPC nodes — recommended):
  python visualization/demo_online.py \\
      --weights ../outputs/causal-tsu_i3d/best_model.pth \\
      --dataset tsu --backbone i3d \\
      --rgb_root /path/to/tsu_features_i3d \\
      --save demo_tsu.mp4

# Show live (requires a display / X11 forwarding):
  python visualization/demo_online.py \\
      --weights ../outputs/causal-tsu_i3d/best_model.pth \\
      --dataset tsu --backbone i3d \\
      --rgb_root /path/to/tsu_features_i3d \\
      --display

# Choose a specific video:
  python visualization/demo_online.py ... --video_id P11T15C01

# Process every N-th frame (faster render, e.g. --step 5):
  python visualization/demo_online.py ... --step 3

# Change animation speed (default 60 ms / frame):
  python visualization/demo_online.py ... --interval 40

# Provide custom class-name file (one name per line, 0-indexed):
  python visualization/demo_online.py ... --class_names my_classes.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ── Make vim/ importable regardless of cwd ──────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_VIM  = os.path.join(_HERE, '..', 'vim')
if _VIM not in sys.path:
    sys.path.insert(0, _VIM)

from timm.models import create_model
import models_MSTemba  # noqa: F401  — registers 'mstemba' with timm
from charades_dataloader import Charades as Dataset, collate_fn_unisize


# ── Class-name dictionaries ──────────────────────────────────────────────────

# TSU / Toyota SmartHome – 51 activity classes (0-indexed)
TSU_CLASSES: dict[int, str] = {
     0: "Cook · Clean dishes",    1: "Cook · Cleanup",
     2: "Cook · Cut",             3: "Cook · Stir",
     4: "Cook · Use stove",       5: "Cut bread",
     6: "Drink",                  7: "Dry utensils",
     8: "Eat · At table",         9: "Eat · Snack",
    10: "Enter",                  11: "Get up",
    12: "Lay down",               13: "Leave",
    14: "Make coffee · Pour grains",  15: "Make coffee · Pour milk",
    16: "Make coffee · Pour water",   17: "Make coffee · Switch on",
    18: "Make tea · Insert teabag",   19: "Make tea · Pour milk",
    20: "Make tea · Pour water",      21: "Make tea · Switch on",
    22: "Pour · From can",        23: "Pour · From bottle",
    24: "Pour · From kettle",     25: "Pour · From milk box",
    26: "Pour · From pot",        27: "Pour · From teapot",
    28: "Read book",              29: "Sit down",
    30: "Stand up",               31: "Take · From bottom shelf",
    32: "Take · From drawer",     33: "Take · From sink",
    34: "Take · From top shelf",  35: "Take pill",
    36: "Use laptop",             37: "Use tablet",
    38: "Use telephone",          39: "Use TV · Pause",
    40: "Use TV · Play",          41: "Walk",
    42: "Wash · Cutlery",         43: "Wash · Dishes",
    44: "Wash · Hands",           45: "Wash · Pot",
    46: "Watch · PC",             47: "Watch · TV",
    48: "Write · Keyboard",       49: "Write · Whiteboard",
    50: "Write · Paper",
}

# Charades – 157 activity classes  (0-indexed; displayed as c001…c157 when no
# custom name file is provided – matches the official Charades annotation IDs).
CHARADES_CLASSES: dict[int, str] = {i: f"c{i+1:03d}" for i in range(157)}

# Partial human-readable names for the most frequent Charades classes
_CHARADES_READABLE: dict[int, str] = {
     0: "Holding a blanket",       1: "Smiling at a blanket",
     2: "Lying on a bed",          3: "Sitting on a bed",
     4: "Standing on a bed",       5: "Lying on the floor",
     6: "Sitting on the floor",    7: "Holding a book",
     8: "Reading a book",          9: "Opening a book",
    10: "Closing a book",         11: "Sitting at a table",
    12: "Eating at a table",       13: "Holding a broom",
    14: "Sweeping the floor",      15: "Holding a chair",
    16: "Sitting on a chair",      17: "Standing on a chair",
    18: "Moving a chair",          19: "Holding a cup",
    20: "Drinking from a cup",     21: "Opening a door",
    22: "Closing a door",          23: "Holding a doorknob",
    24: "Washing dishes",          25: "Drying dishes",
    26: "Holding a dish",          27: "Pouring from a dish",
    28: "Opening a drawer",        29: "Closing a drawer",
    30: "Holding a laptop",        31: "Typing on a laptop",
    32: "Opening a laptop",        33: "Closing a laptop",
    34: "Holding a paper",         35: "Reading a paper",
    36: "Writing on paper",        37: "Folding a paper",
    38: "Holding a phone",         39: "Talking on a phone",
    40: "Holding a pillow",        41: "Lying on a pillow",
    42: "Sitting with a pillow",   43: "Throwing a pillow",
    44: "Opening the refrigerator",45: "Closing the refrigerator",
    46: "Holding the refrigerator",47: "Taking from refrigerator",
    48: "Holding shoes",           49: "Putting on shoes",
    50: "Taking off shoes",        51: "Tying shoes",
    52: "Holding a sandwich",      53: "Eating a sandwich",
    54: "Making a sandwich",       55: "Holding a towel",
    56: "Washing hands with towel",57: "Wiping face with towel",
    58: "Watching TV",             59: "Turning on TV",
    60: "Turning off TV",          61: "Holding a vacuum",
    62: "Vacuuming the floor",     63: "Opening a window",
    64: "Closing a window",        65: "Looking out of window",
}
CHARADES_CLASSES.update(_CHARADES_READABLE)


def _get_class_map(dataset: str, class_names_file: str | None) -> dict[int, str]:
    if class_names_file is not None:
        with open(class_names_file) as fh:
            names = [l.strip() for l in fh if l.strip()]
        return {i: n for i, n in enumerate(names)}
    return TSU_CLASSES if dataset == 'tsu' else CHARADES_CLASSES


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(args, num_classes: int, device):
    in_feat_dim = 1024 if args.backbone == 'i3d' else 768
    model = create_model(
        'mstemba',
        pretrained=False,
        num_classes=num_classes,
        in_feat_dim=in_feat_dim,
        fuser=args.fuser,
        causal=True,
        causal_consistency_loss_weight=0.0,
        causal_consistency_margin=0.1,
    )
    ckpt = torch.load(args.weights, map_location='cpu')
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARNING] Missing keys: {missing[:5]} …")
    if unexpected:
        print(f"[WARNING] Unexpected keys: {unexpected[:5]} …")
    model.to(device)
    model.eval()
    return model


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _pick_video(args, num_classes: int) -> tuple[str, np.ndarray, np.ndarray, float]:
    """
    Return (video_id, features [T, feat_dim], labels [T, C], duration_s).
    Features are loaded directly from the .npy file so we don't need the
    collate / padding machinery – the raw variable-length sequence is more
    appropriate for a streaming demo.
    """
    split_file = (
        os.path.join(_HERE, '..', 'data', 'charades.json')
        if args.dataset == 'charades'
        else os.path.join(_HERE, '..', 'data', 'smarthome.json')
    )

    import json
    with open(split_file) as fh:
        meta = json.load(fh)

    test_vids = [
        vid for vid, info in meta.items()
        if info.get('subset') == 'testing'
        and os.path.exists(os.path.join(args.rgb_root, vid + '.npy'))
        and len(info.get('actions', [])) >= 1
    ]
    if not test_vids:
        raise RuntimeError(
            f"No test videos found in {args.rgb_root}. "
            "Check --rgb_root and that the .npy feature files exist."
        )

    if args.video_id is not None:
        if args.video_id not in test_vids:
            raise ValueError(
                f"--video_id '{args.video_id}' not found in testing split or missing .npy."
            )
        vid = args.video_id
    else:
        vid = random.choice(test_vids)

    feat_path = os.path.join(args.rgb_root, vid + '.npy')
    features = np.load(feat_path).astype(np.float32)   # [T, feat_dim]
    T = features.shape[0]
    duration = float(meta[vid]['duration'])
    fps = T / duration if duration > 0 else 1.0

    labels = np.zeros((T, num_classes), np.float32)
    for ann in meta[vid]['actions']:
        cls_idx, t_start, t_end = ann[0], float(ann[1]), float(ann[2])
        if t_end < t_start:
            continue
        for fr in range(T):
            if fr / fps > t_start and fr / fps < t_end:
                labels[fr, cls_idx] = 1.0

    print(f"[demo] Selected video : {vid}  ({T} frames, {duration:.1f} s)")
    return vid, features, labels, duration


# ── Streaming inference (one frame at a time) ─────────────────────────────────

@torch.no_grad()
def _run_streaming(model, features: np.ndarray, device) -> np.ndarray:
    """
    Run frame-by-frame streaming inference.

    Returns
    -------
    probs : np.ndarray  [T, num_classes]
    """
    T, C = features.shape
    # [1, C, T] – batch=1
    feat_tensor = torch.from_numpy(features).T.unsqueeze(0).to(device)  # [1, C, T]

    state = model.init_stream_state(batch_size=1, device=device)
    all_probs = []

    print(f"[demo] Running streaming inference over {T} frames …")
    t0 = time.time()
    for t in range(T):
        chunk = feat_tensor[:, :, t : t + 1]               # [1, C, 1]
        logits_t, _, state = model.stream_forward(chunk, state)
        prob_t = torch.sigmoid(logits_t).cpu().numpy()[0, 0]  # [num_classes]
        all_probs.append(prob_t)

    elapsed = time.time() - t0
    print(f"[demo] Done in {elapsed:.2f}s  ({elapsed/T*1000:.1f} ms/frame, "
          f"RTF={elapsed/max((T/25), 1e-6):.3f})")
    return np.stack(all_probs, axis=0)   # [T, num_classes]


# ── Matplotlib animation ──────────────────────────────────────────────────────

def _build_animation(
    vid_id: str,
    features: np.ndarray,     # [T, feat_dim]
    probs: np.ndarray,         # [T, num_classes]
    labels: np.ndarray,        # [T, num_classes]
    duration: float,
    class_map: dict[int, str],
    args,
):
    """Build and return a matplotlib FuncAnimation."""
    import matplotlib
    if args.display:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            matplotlib.use('Qt5Agg')
    else:
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation

    T, feat_dim = features.shape
    num_classes  = probs.shape[1]
    topk         = 5
    win_frames   = 64          # scrolling heatmap window (frames)
    vis_dims     = 128         # feature channels to display
    step         = max(1, args.step)
    frames_idx   = list(range(0, T, step))

    # Pre-normalise features for heatmap display
    feat_vis = features[:, :vis_dims].copy()
    feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-8)

    fps_approx = T / duration if duration > 0 else 25.0

    # ── Figure / axes layout ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    fig.suptitle(
        "MS-Temba  ·  Online Causal Inference Demo",
        fontsize=14, fontweight='bold', color='#1a1a2e', y=0.98,
    )

    gs_outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.6, 1.0],
        top=0.92, bottom=0.04,
        hspace=0.38,
    )

    # Top: heatmap + progress bar
    gs_top = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs_outer[0],
        height_ratios=[0.06, 1.0],
        hspace=0.18,
    )
    ax_info   = fig.add_subplot(gs_top[0])
    ax_heatmap = fig.add_subplot(gs_top[1])

    # Bottom: predictions (left) + ground truth (right)
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs_outer[1],
        wspace=0.08,
    )
    ax_pred = fig.add_subplot(gs_bot[0])
    ax_gt   = fig.add_subplot(gs_bot[1])

    # ── Colour palette ────────────────────────────────────────────────────────
    PRED_COLORS = ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626']
    GT_COLOR    = '#16a34a'
    BAR_BG      = '#e5e7eb'

    # ── Static axes setup ─────────────────────────────────────────────────────

    # — Info bar (text only, no frame) —
    ax_info.set_axis_off()
    txt_info = ax_info.text(
        0.0, 0.5, '', transform=ax_info.transAxes,
        fontsize=11, color='#374151', va='center',
        fontfamily='monospace',
    )

    # — Heatmap —
    heat_data = np.zeros((win_frames, vis_dims))
    im_heat = ax_heatmap.imshow(
        heat_data, aspect='auto', origin='lower',
        cmap='plasma', vmin=0.0, vmax=1.0,
        interpolation='nearest',
    )
    ax_heatmap.set_title(
        "I3D Feature Activations  (sliding window · recent frames bottom→top)",
        fontsize=9.5, color='#4b5563', pad=4,
    )
    ax_heatmap.set_xlabel("Feature channel", fontsize=8.5, color='#6b7280')
    ax_heatmap.set_ylabel("Frame offset", fontsize=8.5, color='#6b7280')
    ax_heatmap.tick_params(labelsize=7.5, colors='#9ca3af')
    for spine in ax_heatmap.spines.values():
        spine.set_edgecolor('#d1d5db')
    cbar = fig.colorbar(im_heat, ax=ax_heatmap, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Activation", fontsize=8, color='#6b7280')

    # — Prediction panel —
    ax_pred.set_xlim(0, 1)
    ax_pred.set_ylim(-0.5, topk - 0.5)
    ax_pred.set_facecolor('white')
    ax_pred.set_title(
        "Live Predictions  (top 5)",
        fontsize=10.5, fontweight='bold', color='#1f2937', pad=6,
    )
    ax_pred.set_yticks([])
    ax_pred.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_pred.set_xticklabels(['0', '', '0.5', '', '1.0'], fontsize=8, color='#9ca3af')
    ax_pred.set_xlabel("Confidence", fontsize=8.5, color='#6b7280')
    for spine in ax_pred.spines.values():
        spine.set_edgecolor('#e5e7eb')

    # Background bars (full width, light grey)
    for k in range(topk):
        ax_pred.barh(
            k, 1.0, height=0.55, color=BAR_BG, left=0, zorder=1,
        )

    pred_bars  = []
    pred_texts = []
    pred_scores= []
    for k in range(topk):
        bar, = ax_pred.barh(
            k, 0.0, height=0.55,
            color=PRED_COLORS[k], left=0, zorder=2, alpha=0.85,
        )
        # Label: fixed at left edge, dark colour — readable on bar and background
        lbl = ax_pred.text(
            0.01, k, '', va='center', ha='left',
            fontsize=9.0, color='#1f2937', fontweight='bold', zorder=4,
        )
        # Score: fixed at right edge
        scr = ax_pred.text(
            0.99, k, '', va='center', ha='right',
            fontsize=8.5, color='#374151', zorder=4,
        )
        pred_bars.append(bar)
        pred_texts.append(lbl)
        pred_scores.append(scr)

    # — Ground truth panel —
    ax_gt.set_axis_off()
    ax_gt.set_facecolor('#f9fafb')
    gt_title = ax_gt.text(
        0.5, 0.93, "Ground Truth", transform=ax_gt.transAxes,
        fontsize=10.5, fontweight='bold', color='#1f2937',
        ha='center', va='top',
    )
    gt_texts = [
        ax_gt.text(
            0.06, 0.76 - i * 0.18, '', transform=ax_gt.transAxes,
            fontsize=9.0, color=GT_COLOR, va='top',
        )
        for i in range(6)
    ]
    gt_none_txt = ax_gt.text(
        0.06, 0.76, 'No active labels', transform=ax_gt.transAxes,
        fontsize=9.0, color='#9ca3af', va='top',
    )
    # ── Progress bar ─────────────────────────────────────────────────────────
    # Thin coloured strip across the bottom of the info axis
    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    pg_bg   = ax_info.barh(0.5, 1.0, height=0.4, color='#e5e7eb', left=0, zorder=1)
    pg_bar, = ax_info.barh(
        0.5, 0.0, height=0.4, color='#2563EB', left=0, zorder=2, alpha=0.8
    )

    # ── Animation update function ─────────────────────────────────────────────

    def update(fi):
        t = frames_idx[fi]
        progress = t / max(T - 1, 1)
        time_s   = t / fps_approx

        # — info text —
        txt_info.set_text(
            f"  Video: {vid_id}      "
            f"Frame: {t+1:>4d} / {T}      "
            f"Time: {time_s:>5.1f} s / {duration:.1f} s"
        )
        pg_bar.set_width(progress)

        # — heatmap: sliding window ending at t —
        t_start = max(0, t - win_frames + 1)
        window  = feat_vis[t_start : t + 1]              # up to win_frames rows
        pad_len = win_frames - window.shape[0]
        if pad_len > 0:
            window = np.vstack([np.zeros((pad_len, vis_dims)), window])
        im_heat.set_data(window)

        # — predictions —
        prob_t = probs[t]                                # [num_classes]
        top_k_idx = np.argsort(prob_t)[::-1][:topk]
        for k in range(topk):
            c   = top_k_idx[k]
            p   = float(prob_t[c])
            name = class_map.get(c, f'class {c}')
            # Truncate long names
            if len(name) > 26:
                name = name[:24] + '…'
            pred_bars[k].set_width(p)
            pred_texts[k].set_text(f"  {name}")
            pred_scores[k].set_text(f"{p:.2f}  ")

        # — ground truth —
        active = np.where(labels[t] > 0.5)[0]
        if len(active) == 0:
            gt_none_txt.set_visible(True)
            for gtxt in gt_texts:
                gtxt.set_text('')
        else:
            gt_none_txt.set_visible(False)
            for i, gtxt in enumerate(gt_texts):
                if i < len(active):
                    name = class_map.get(int(active[i]), f'class {active[i]}')
                    if len(name) > 30:
                        name = name[:28] + '…'
                    gtxt.set_text(f"  ✓ {name}")
                else:
                    gtxt.set_text('')

        return (im_heat, pg_bar, txt_info,
                *pred_bars, *pred_texts, *pred_scores,
                *gt_texts, gt_none_txt)

    anim = FuncAnimation(
        fig, update,
        frames=len(frames_idx),
        interval=args.interval,
        blit=True,
    )

    return fig, anim


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MS-Temba live online-inference visualisation demo."
    )
    p.add_argument('--weights',     required=True, help='Path to best_model.pth checkpoint.')
    p.add_argument('--dataset',     default='tsu', choices=['charades', 'tsu'],
                   help='Dataset identifier (determines num_classes and JSON split file).')
    p.add_argument('--backbone',    default='i3d', choices=['i3d', 'clip'],
                   help='Feature extractor used to produce the .npy files.')
    p.add_argument('--rgb_root',    required=True,
                   help='Directory containing per-video <vid_id>.npy feature files.')
    p.add_argument('--fuser',       default='token-attention',
                   choices=['sum', 'weighted', 'token-attention',
                            'cross-token-attention', 'attention'],
                   help='Fuser variant (must match the checkpoint).')
    p.add_argument('--video_id',    default=None,
                   help='Specific video ID to use; random if omitted.')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--step',        type=int, default=1,
                   help='Process every N-th frame (1 = all frames, 5 = 5× faster).')
    p.add_argument('--interval',    type=int, default=60,
                   help='Animation delay between frames in ms (default 60).')
    p.add_argument('--class_names', default=None,
                   help='Path to a text file with one class name per line (0-indexed).')
    p.add_argument('--save',        default=None,
                   help='Save animation to this file (e.g. demo.mp4). '
                        'Requires ffmpeg on PATH.')
    p.add_argument('--display',     action='store_true',
                   help='Show animation in an interactive window '
                        '(requires a display / X11 forwarding).')
    p.add_argument('--gpu',         type=int, default=0)
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save is None and not args.display:
        print("[demo] Neither --save nor --display specified. "
              "Defaulting to --save demo_output.mp4")
        args.save = 'demo_output.mp4'

    num_classes = 157 if args.dataset == 'charades' else 51
    class_map   = _get_class_map(args.dataset, args.class_names)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[demo] Device : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[demo] Loading model from: {args.weights}")
    model = _load_model(args, num_classes, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[demo] Model   : {n_params:,} trainable params")

    # ── Pick a video ──────────────────────────────────────────────────────────
    vid_id, features, labels, duration = _pick_video(args, num_classes)

    # ── Run streaming inference ───────────────────────────────────────────────
    probs = _run_streaming(model, features, device)   # [T, num_classes]

    # ── Build animation ───────────────────────────────────────────────────────
    print("[demo] Building animation …")
    fig, anim = _build_animation(
        vid_id, features, probs, labels, duration, class_map, args,
    )

    if args.save is not None:
        # Resolve output path relative to the script's parent (project root)
        save_path = args.save
        if not os.path.isabs(save_path):
            save_path = os.path.join(_HERE, save_path)
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(
            fps=max(1, 1000 // args.interval),
            metadata={'title': f'MS-Temba online demo – {vid_id}'},
            bitrate=1800,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        )
        print(f"[demo] Saving to: {save_path}")
        anim.save(save_path, writer=writer, dpi=120)
        print(f"[demo] Saved  ✓  {save_path}")

    if args.display:
        import matplotlib.pyplot as plt
        plt.show()

    print("[demo] Done.")


if __name__ == '__main__':
    main()