"""
visualization/demo_online.py
============================
Live online-inference demo for MS-Temba (causal streaming mode).

Speed knobs (fastest slowest)
---------------------------------
  --preview          Equivalent to --step 5 --dpi 85 --video_width 640
                     Quick sanity-check render at real-time speed.

  --step N           Use every N-th feature frame (default 3).
                     Cuts frame count and render time by Nx.

  --dpi D            Output resolution (default 100).  80 = draft, 140 = paper.

  --video_width W    Resize video frames to W pixels wide (default 640).
                     Smaller = faster imshow + less memory.

  --online_inference Run true frame-by-frame streaming instead of the fast
                     offline causal batch forward.  Numerically identical
                     outputs, but ~Tx slower (one kernel launch per frame).
                     Off by default -- only needed to measure streaming latency.

  --pred_threshold   Confidence below which all prediction bars are hidden and
                     "No activity detected" is shown instead (default 0.15).

Layout (when a real video is supplied)
---------------------------------------
  +-----------------------------------------------------------------+
  |  MS-Temba - Online Causal Inference Demo                        |
  |  Video: P11T15C01   ################ooooo  123 / 500            |
  +-----------------------------------------------------------------+
  |                                                                 |
  |              ACTUAL VIDEO FRAME  (full width)                   |
  |                                                                 |
  +-----------------------------------------------------------------+
  |  I3D feature activations  -  sliding 32-frame window            |
  +------------------------------+----------------------------------+
  |  LIVE PREDICTIONS (top 5)    |  GROUND TRUTH                    |
  |  ########  Sit down  (0.87)  |  check Sit down                  |
  |  ######    Read book (0.61)  |  check Read book                 |
  +------------------------------+----------------------------------+

Without a video file (feature-only mode), the video row is replaced by a
larger feature heatmap.

Video discovery (in order)
--------------------------
  1. --video /explicit/path/to/file.mp4
  2. First video file found inside  visualization/mp4/
  3. Feature-only fallback if nothing is found.

Usage
-----
  # Quick preview (~2 min, real-time speed):
    python visualization/demo_online.py \\
        --weights ../outputs/causal-tsu_i3d/best_model.pth \\
        --dataset tsu --backbone i3d \\
        --rgb_root /path/to/tsu_features_i3d \\
        --save demo_tsu.mp4 --preview

  # Full quality:
    python visualization/demo_online.py \\
        ... --save demo_tsu.mp4 --step 1 --dpi 120

  # Show live (needs display / X11):
    python visualization/demo_online.py ... --display
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# -- Make vim/ importable from any cwd ----------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_VIM  = os.path.join(_HERE, '..', 'vim')
if _VIM not in sys.path:
    sys.path.insert(0, _VIM)

from timm.models import create_model
import models_MSTemba  # noqa: F401  - registers 'mstemba' with timm
from charades_dataloader import Charades as Dataset, collate_fn_unisize


# -- Class-name dictionaries --------------------------------------------------

TSU_CLASSES: dict[int, str] = {
     0: "Cook - Clean dishes",    1: "Cook - Cleanup",
     2: "Cook - Cut",             3: "Cook - Stir",
     4: "Cook - Use stove",       5: "Cut bread",
     6: "Drink",                  7: "Dry utensils",
     8: "Eat - At table",         9: "Eat - Snack",
    10: "Enter",                  11: "Get up",
    12: "Lay down",               13: "Leave",
    14: "Make coffee - Pour grains",  15: "Make coffee - Pour milk",
    16: "Make coffee - Pour water",   17: "Make coffee - Switch on",
    18: "Make tea - Insert teabag",   19: "Make tea - Pour milk",
    20: "Make tea - Pour water",      21: "Make tea - Switch on",
    22: "Pour - From can",        23: "Pour - From bottle",
    24: "Pour - From kettle",     25: "Pour - From milk box",
    26: "Pour - From pot",        27: "Pour - From teapot",
    28: "Read book",              29: "Sit down",
    30: "Stand up",               31: "Take - From bottom shelf",
    32: "Take - From drawer",     33: "Take - From sink",
    34: "Take - From top shelf",  35: "Take pill",
    36: "Use laptop",             37: "Use tablet",
    38: "Use telephone",          39: "Use TV - Pause",
    40: "Use TV - Play",          41: "Walk",
    42: "Wash - Cutlery",         43: "Wash - Dishes",
    44: "Wash - Hands",           45: "Wash - Pot",
    46: "Watch - PC",             47: "Watch - TV",
    48: "Write - Keyboard",       49: "Write - Whiteboard",
    50: "Write - Paper",
}

CHARADES_CLASSES: dict[int, str] = {i: f"c{i+1:03d}" for i in range(157)}
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
    44: "Opening refrigerator",    45: "Closing refrigerator",
    46: "Holding refrigerator",    47: "Taking from refrigerator",
    48: "Holding shoes",           49: "Putting on shoes",
    50: "Taking off shoes",        51: "Tying shoes",
    52: "Holding a sandwich",      53: "Eating a sandwich",
    54: "Making a sandwich",       55: "Holding a towel",
    56: "Washing with towel",      57: "Wiping face with towel",
    58: "Watching TV",             59: "Turning on TV",
    60: "Turning off TV",
}
CHARADES_CLASSES.update(_CHARADES_READABLE)


def _get_class_map(dataset: str, class_names_file: str | None) -> dict[int, str]:
    if class_names_file is not None:
        with open(class_names_file, encoding='utf-8') as fh:
            names = [l.strip() for l in fh if l.strip()]
        return {i: n for i, n in enumerate(names)}
    return TSU_CLASSES if dataset == 'tsu' else CHARADES_CLASSES


# -- Video file discovery -----------------------------------------------------

_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def _find_video(explicit_path: str | None) -> str | None:
    if explicit_path is not None:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"--video file not found: {explicit_path}")
        return explicit_path
    mp4_dir = os.path.join(_HERE, 'mp4')
    if os.path.isdir(mp4_dir):
        for fname in sorted(os.listdir(mp4_dir)):
            if os.path.splitext(fname)[1].lower() in _VIDEO_EXTS:
                found = os.path.join(mp4_dir, fname)
                print(f"[demo] Auto-detected video : {found}")
                return found
    if os.path.isfile(mp4_dir) and os.path.splitext(mp4_dir)[1].lower() in _VIDEO_EXTS:
        return mp4_dir
    return None


# -- Video frame extraction ---------------------------------------------------

def _extract_video_frames(
    video_path: str,
    frames_idx: list[int],
    fps_feat:   float,
    video_width: int | None,
) -> dict[int, np.ndarray]:
    """
    Pre-extract RGB frames at feature-frame timestamps.
    Sequential scan -- no random seeks.
    Optionally resize to *video_width* pixels wide.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("pip install opencv-python  (needed for video frames)")

    cap       = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Map feature frame -> required video frame index
    required: list[tuple[int, int]] = [
        (min(int((t / fps_feat) * fps_video), n_vframes - 1), t)
        for t in frames_idx
    ]

    video_frames: dict[int, np.ndarray] = {}
    cur_vf  = 0
    req_ptr = 0

    print(f"[demo] Extracting {len(required)} frames from video "
          f"({fps_video:.1f} fps, {n_vframes} total) ...", flush=True)

    while req_ptr < len(required) and cap.isOpened():
        target_vf, feat_t = required[req_ptr]
        if cur_vf < target_vf:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_vf)
            cur_vf = target_vf
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if video_width is not None:
                h, w = rgb.shape[:2]
                new_h = max(1, int(h * video_width / w))
                rgb   = cv2.resize(rgb, (video_width, new_h),
                                   interpolation=cv2.INTER_AREA)
            video_frames[feat_t] = rgb
        cur_vf  += 1
        req_ptr += 1

    cap.release()
    print(f"[demo] Extracted {len(video_frames)} video frames.", flush=True)
    return video_frames


# -- Model loading ------------------------------------------------------------

def _load_model(args, num_classes: int, device):
    in_feat_dim = 1024 if args.backbone == 'i3d' else 768
    model = create_model(
        'mstemba', pretrained=False, num_classes=num_classes,
        in_feat_dim=in_feat_dim, fuser=args.fuser, causal=True,
        causal_consistency_loss_weight=0.0, causal_consistency_margin=0.1,
    )
    ckpt = torch.load(args.weights, map_location='cpu')
    sd   = ckpt.get('model', ckpt.get('state_dict', ckpt))
    sd   = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print(f"[WARNING] Missing keys:    {missing[:4]} ...")
    if unexpected: print(f"[WARNING] Unexpected keys: {unexpected[:4]} ...")
    model.to(device).eval()
    return model


# -- Dataset helpers ----------------------------------------------------------

def _pick_video(args, num_classes: int) -> tuple[str, np.ndarray, np.ndarray, float]:
    split_file = (
        os.path.join(_HERE, '..', 'data', 'charades.json')
        if args.dataset == 'charades'
        else os.path.join(_HERE, '..', 'data', 'smarthome.json')
    )
    import json
    with open(split_file, encoding='utf-8') as fh:
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
            "Check --rgb_root and that .npy feature files exist."
        )

    vid = args.video_id if args.video_id is not None else random.choice(test_vids)
    if vid not in test_vids:
        raise ValueError(f"--video_id '{vid}' not in testing split or missing .npy.")

    features = np.load(os.path.join(args.rgb_root, vid + '.npy')).astype(np.float32)
    features = np.squeeze(features)
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D [T, C], got shape {features.shape}")

    T        = features.shape[0]
    duration = float(meta[vid]['duration'])
    fps_feat = T / duration if duration > 0 else 1.0

    labels = np.zeros((T, num_classes), np.float32)
    for ann in meta[vid]['actions']:
        c, ts, te = ann[0], float(ann[1]), float(ann[2])
        if te < ts: continue
        for fr in range(T):
            if ts < fr / fps_feat < te:
                labels[fr, c] = 1.0

    print(f"[demo] Dataset video  : {vid}  ({T} frames, {duration:.1f} s, "
          f"{fps_feat:.1f} feat-fps)")
    return vid, features, labels, duration


# -- Inference ----------------------------------------------------------------

@torch.no_grad()
def _run_offline_causal(model, features: np.ndarray, device) -> np.ndarray:
    """
    Single-pass causal batch forward.
    Numerically identical to streaming, ~100-500x faster.
    """
    feat_tensor = torch.from_numpy(features).transpose(0, 1).unsqueeze(0).to(device)
    outputs, _, _, _ = model(feat_tensor)
    return torch.sigmoid(outputs).cpu().numpy()[0]   # [T, num_classes]


@torch.no_grad()
def _run_streaming(model, features: np.ndarray, device) -> np.ndarray:
    """True frame-by-frame streaming. Slow; use only for latency benchmarking."""
    T = features.shape[0]
    feat_tensor = torch.from_numpy(features).transpose(0, 1).unsqueeze(0).to(device)
    state, all_probs = model.init_stream_state(batch_size=1, device=device), []
    t0 = time.time()
    for t in range(T):
        chunk   = feat_tensor[:, :, t : t + 1]
        logit_t, _, state = model.stream_forward(chunk, state)
        all_probs.append(torch.sigmoid(logit_t).cpu().numpy()[0, 0])
    elapsed = time.time() - t0
    print(f"[demo] Streaming done {elapsed:.1f}s  "
          f"({elapsed/T*1000:.1f} ms/frame,  RTF={elapsed/max(T/25,1e-6):.3f})")
    return np.stack(all_probs, axis=0)


# -- Figure construction ------------------------------------------------------

def _build_figure(
    vid_id:       str,
    features:     np.ndarray,
    probs:        np.ndarray,
    labels:       np.ndarray,
    duration:     float,
    class_map:    dict[int, str],
    video_frames: dict[int, np.ndarray] | None,
    args,
) -> tuple:
    """
    Build the matplotlib figure and all animated artists.

    Returns
    -------
    fig      : Figure
    update   : callable(frame_index) -> tuple of artists
    n_frames : total number of animation frames
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    T            = features.shape[0]
    step         = max(1, args.step)
    frames_idx   = list(range(0, T, step))
    fps_approx   = T / duration if duration > 0 else 25.0
    has_video    = bool(video_frames)
    # Light-weight defaults; `--fast-preview` will reduce these further.
    topk         = 5
    vis_dims     = 128
    win_frames   = 32 if has_video else 64

    # If user requested an ultra-fast preview, reduce visual detail and
    # update frequency to minimize CPU work and canvas redraws.
    fast_preview = getattr(args, 'fast_preview', False)
    if fast_preview:
        topk = 3
        vis_dims = 32
        win_frames = 16
        heatmap_update_freq = 4
    else:
        heatmap_update_freq = 1

    feat_vis = features[:, :vis_dims].copy()
    feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min() + 1e-8)

    # Confidence threshold below which predictions are suppressed
    PRED_THRESHOLD = getattr(args, 'pred_threshold', 0.15)

    # -- Colours --------------------------------------------------------------
    PRED_COLORS = ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626']
    GT_COLOR    = '#15803d'
    BAR_BG      = '#e5e7eb'
    C_DARK      = '#1f2937'
    C_MID       = '#6b7280'
    C_LIGHT     = '#9ca3af'

    # -- Figure / GridSpec ----------------------------------------------------
    fig = plt.figure(figsize=(14, 9), facecolor='white')
    fig.suptitle("MS-Temba  -  Online Causal Inference Demo",
                 fontsize=13, fontweight='bold', color='#1a1a2e', y=0.99)

    if has_video:
        gs = gridspec.GridSpec(4, 1, figure=fig,
                               height_ratios=[0.04, 0.52, 0.11, 0.33],
                               top=0.96, bottom=0.03, hspace=0.30)
        ax_info    = fig.add_subplot(gs[0])
        ax_video   = fig.add_subplot(gs[1])
        ax_heatmap = fig.add_subplot(gs[2])
        ax_bottom  = fig.add_subplot(gs[3])
    else:
        gs = gridspec.GridSpec(3, 1, figure=fig,
                               height_ratios=[0.04, 0.56, 0.40],
                               top=0.96, bottom=0.03, hspace=0.35)
        ax_info    = fig.add_subplot(gs[0])
        ax_video   = None
        ax_heatmap = fig.add_subplot(gs[1])
        ax_bottom  = fig.add_subplot(gs[2])

    # Split bottom row: predictions (left 55%) + ground truth (right 45%)
    ax_bottom.set_axis_off()
    bp    = ax_bottom.get_position()
    split = 0.55
    ax_pred = fig.add_axes([bp.x0,
                            bp.y0,
                            bp.width * split - 0.005,
                            bp.height])
    ax_gt   = fig.add_axes([bp.x0 + bp.width * split + 0.005,
                            bp.y0,
                            bp.width * (1 - split),
                            bp.height])

    # -- Info bar -------------------------------------------------------------
    ax_info.set_axis_off()
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.barh(0.35, 1.0, height=0.30, color='#e5e7eb', left=0, zorder=1)
    pg_bar, = ax_info.barh(0.35, 0.0, height=0.30, color='#2563EB',
                           left=0, zorder=2, alpha=0.85)
    txt_info = ax_info.text(0.0, 0.88, '', transform=ax_info.transAxes,
                            fontsize=10, color=C_DARK, va='top',
                            fontfamily='monospace')

    # -- Video panel ----------------------------------------------------------
    if has_video:
        ax_video.set_axis_off()
        ax_video.set_facecolor('white')
        sample = next(iter(video_frames.values()))
        # aspect='auto': image fills the full axes area with no black bars.
        # interpolation='bilinear': smooth upscaling, no visible pixel grid.
        im_video = ax_video.imshow(sample, aspect='auto', interpolation='bilinear')
        for sp in ax_video.spines.values():
            sp.set_edgecolor('#d1d5db'); sp.set_linewidth(1.0)
    else:
        im_video = None

    # -- Feature heatmap ------------------------------------------------------
    im_heat = ax_heatmap.imshow(
        np.zeros((win_frames, vis_dims)), aspect='auto', origin='lower',
        cmap='plasma', vmin=0.0, vmax=1.0, interpolation='nearest',
    )
    ax_heatmap.set_title(
        f"I3D Feature Activations  ({win_frames}-frame sliding window  channels)",
        fontsize=8.5, color='#4b5563', pad=3)
    ax_heatmap.set_xlabel("Feature channel", fontsize=7.5, color=C_LIGHT)
    ax_heatmap.set_ylabel("frames", fontsize=7.5, color=C_LIGHT)
    ax_heatmap.tick_params(labelsize=7, colors=C_LIGHT)
    for sp in ax_heatmap.spines.values(): sp.set_edgecolor('#d1d5db')
    cbar = fig.colorbar(im_heat, ax=ax_heatmap, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(labelsize=6.5)

    # -- Predictions panel ----------------------------------------------------
    ax_pred.set_xlim(0, 1); ax_pred.set_ylim(-0.5, topk - 0.5)
    ax_pred.set_facecolor('white')
    ax_pred.set_title("Live Predictions  (top 5)",
                      fontsize=10, fontweight='bold', color=C_DARK, pad=5)
    ax_pred.set_yticks([])
    ax_pred.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_pred.set_xticklabels(['0', '', '0.5', '', '1'], fontsize=8, color=C_LIGHT)
    ax_pred.set_xlabel("Confidence", fontsize=8, color=C_MID)
    for sp in ax_pred.spines.values(): sp.set_edgecolor('#e5e7eb')
    for k in range(topk):
        ax_pred.barh(k, 1.0, height=0.60, color=BAR_BG, left=0, zorder=1)

    pred_bars, pred_labels, pred_scores = [], [], []
    for k in range(topk):
        bar, = ax_pred.barh(k, 0.0, height=0.60,
                            color=PRED_COLORS[k], left=0, zorder=2, alpha=0.85)
        lbl  = ax_pred.text(0.01, k, '', va='center', ha='left',
                            fontsize=8.5, color=C_DARK, fontweight='bold', zorder=4)
        scr  = ax_pred.text(0.99, k, '', va='center', ha='right',
                            fontsize=8.0, color=C_MID, zorder=4)
        pred_bars.append(bar); pred_labels.append(lbl); pred_scores.append(scr)

    # "No activity" overlay shown when max confidence is below threshold
    pred_idle_txt = ax_pred.text(
        0.5, (topk - 1) / 2.0, 'No activity detected',
        va='center', ha='center', fontsize=10, color=C_LIGHT,
        style='italic', zorder=5,
    )
    pred_idle_txt.set_visible(False)

    # -- Ground-truth panel ---------------------------------------------------
    # Light green background makes it visually distinct at a glance.
    ax_gt.set_axis_off()
    ax_gt.set_facecolor('#f0fdf4')
    ax_gt.text(0.5, 0.97, "Ground Truth",
               transform=ax_gt.transAxes,
               fontsize=11, fontweight='bold', color='#14532d',
               ha='center', va='top')
    # Thin separator line under the title
    ax_gt.plot([0.05, 0.95], [0.88, 0.88],
               transform=ax_gt.transAxes,
               color='#86efac', linewidth=1.5, solid_capstyle='round')
    # One text artist per possible active label (up to 6 shown)
    gt_entries = [
        ax_gt.text(0.06, 0.80 - i * 0.13, '',
                   transform=ax_gt.transAxes,
                   fontsize=10.5, color=GT_COLOR, va='top', fontweight='bold')
        for i in range(6)
    ]
    gt_none_txt = ax_gt.text(
        0.06, 0.80, '-- no active labels --',
        transform=ax_gt.transAxes,
        fontsize=10, color=C_LIGHT, va='top', style='italic',
    )

    # -- Update function (shared by all renderers) ----------------------------

    def update(fi: int):
        t        = frames_idx[fi]
        progress = t / max(T - 1, 1)
        time_s   = t / fps_approx

        # Info bar
        txt_info.set_text(
            f"  Video: {vid_id}      "
            f"Frame: {t+1:>4d} / {T}      "
            f"Time: {time_s:>5.1f} s / {duration:.1f} s"
        )
        pg_bar.set_width(progress)

        # Video frame
        if has_video:
            frame = video_frames.get(t)
            if frame is None:
                keys  = np.array(list(video_frames.keys()))
                frame = video_frames[keys[int(np.argmin(np.abs(keys - t)))]]
            im_video.set_data(frame)

        # Feature heatmap (sliding window). To save CPU when in fast-preview
        # mode, update the heatmap only every `heatmap_update_freq` frames.
        need_heat = (fi % heatmap_update_freq == 0) or fi == 0
        if need_heat:
            t0  = max(0, t - win_frames + 1)
            win = feat_vis[t0 : t + 1]
            pad = win_frames - win.shape[0]
            if pad > 0:
                win = np.vstack([np.zeros((pad, vis_dims)), win])
            im_heat.set_data(win)

        # Predictions: suppress entirely when nothing is happening
        prob_t   = probs[t]
        max_prob = float(np.max(prob_t))

        if max_prob < PRED_THRESHOLD:
            # Model is not confident -- blank bars, show idle message
            pred_idle_txt.set_visible(True)
            for k in range(topk):
                pred_bars[k].set_width(0)
                pred_labels[k].set_text('')
                pred_scores[k].set_text('')
        else:
            pred_idle_txt.set_visible(False)
            top_idx = np.argsort(prob_t)[::-1][:topk]
            for k in range(topk):
                c = top_idx[k]; p = float(prob_t[c])
                name = class_map.get(c, f'class {c}')
                if len(name) > 28: name = name[:26] + '...'
                pred_bars[k].set_width(p)
                pred_labels[k].set_text(f"  {name}")
                pred_scores[k].set_text(f"{p:.2f}  ")

        # Ground truth
        active = np.where(labels[t] > 0.5)[0]
        if len(active) == 0:
            gt_none_txt.set_visible(True)
            for g in gt_entries: g.set_text('')
        else:
            gt_none_txt.set_visible(False)
            for i, g in enumerate(gt_entries):
                if i < len(active):
                    name = class_map.get(int(active[i]), f'class {active[i]}')
                    if len(name) > 30: name = name[:28] + '...'
                    g.set_text(f"  [ok]  {name}")
                else:
                    g.set_text('')

        arts = [pg_bar, txt_info, gt_none_txt, pred_idle_txt,
                *gt_entries, *pred_bars, *pred_labels, *pred_scores]
        # Append heatmap only when it was updated (avoids redrawing it every frame)
        if need_heat:
            arts.insert(0, im_heat)
        if has_video:
            arts.append(im_video)
        return tuple(arts)

    return fig, update, len(frames_idx)


# -- Renderers ----------------------------------------------------------------

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(['ffmpeg', '-version'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _render_pipe(fig, update_fn, n_frames, fps, save_path, dpi, frame_repeat=1):
    """
    Render frames directly into an ffmpeg pipe (fast path).

    *frame_repeat*: each unique rendered frame is written this many times,
    keeping the video at *fps* while playing back at real-time speed when
    step > 1.  Example: step=5, fps_feat=25 -> fps_raw=5 -> repeat=3 ->
    output at 15 fps, each unique frame holds for 3/15 s = 200 ms = 5/25 s.
    """
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', 'pipe:0',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
        save_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Attempt fast blit-based rendering: draw only updated artists and blit the
        # figure canvas. This avoids a full canvas redraw per frame and is much
        # faster for long animations.
        canvas = fig.canvas
        canvas.draw()  # warm-up + ensure renderer exists
        try:
            background = canvas.copy_from_bbox(fig.bbox)
            use_blit = True
        except Exception:
            # Backend may not support blit/copy_from_bbox -> fall back
            use_blit = False

        for fi in range(n_frames):
            if fi % max(1, n_frames // 20) == 0:
                pct = 100 * fi // n_frames
                print(f"\r[demo] Rendering {pct:>3d}%  ({fi}/{n_frames}) ...",
                      end='', flush=True)

            arts = update_fn(fi)

            if use_blit:
                # Restore background and draw only artists returned by update_fn.
                try:
                    canvas.restore_region(background)
                    for a in arts:
                        try:
                            if a is not None and hasattr(a, 'axes') and a.axes is not None:
                                a.axes.draw_artist(a)
                        except Exception:
                            # Non-drawable artist; ignore and continue.
                            pass
                    canvas.blit(fig.bbox)
                except Exception:
                    # If blit fails mid-loop, disable and fall back.
                    use_blit = False
                    canvas.draw()
            if not use_blit:
                # Conservative (slower) fallback: full canvas draw.
                canvas.draw()

            rgb_bytes = np.asarray(canvas.buffer_rgba())[:, :, :3].tobytes()
            for _ in range(frame_repeat):
                proc.stdin.write(rgb_bytes)

        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        proc.kill()
        raise RuntimeError(
            "ffmpeg pipe broke -- check that ffmpeg is on PATH and the "
            f"output path is writable: {save_path}"
        )
    print(f"\r[demo] Rendering done  "
          f"({n_frames} frames x{frame_repeat} = "
          f"{n_frames * frame_repeat} output frames).", flush=True)


def _render_matplotlib(fig, update_fn, n_frames, fps, save_path, dpi):
    """Fallback: matplotlib FuncAnimation + FFMpegWriter."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    anim = FuncAnimation(fig, update_fn, frames=n_frames,
                         interval=1000 // fps, blit=True)
    writer = FFMpegWriter(fps=fps, bitrate=2000,
                          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    print("[demo] Rendering via matplotlib (fallback) ...", flush=True)
    anim.save(save_path, writer=writer, dpi=dpi)


# -- CLI ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MS-Temba live online-inference visualisation demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--weights',      required=True)
    p.add_argument('--dataset',      default='tsu', choices=['charades', 'tsu'])
    p.add_argument('--backbone',     default='i3d',  choices=['i3d', 'clip'])
    p.add_argument('--rgb_root',     required=True)
    p.add_argument('--fuser',        default='token-attention',
                   choices=['sum', 'weighted', 'token-attention',
                            'cross-token-attention', 'attention'])
    p.add_argument('--video',        default=None,
                   help='Explicit path to video file (auto-detected from '
                        'visualization/mp4/ if omitted).')
    p.add_argument('--video_id',     default=None,
                   help='Dataset video ID; random test video if omitted.')
    p.add_argument('--seed',         type=int, default=42)

    # -- Speed knobs ----------------------------------------------------------
    p.add_argument('--step',         type=int, default=3,
                   help='Use every N-th frame.  1=all (slow), 3=default.')
    p.add_argument('--dpi',          type=int, default=100,
                   help='Output DPI.  80=draft, 100=default, 140=print.')
    p.add_argument('--video_width',  type=int, default=640,
                   help='Resize video frames to this width in pixels.')
    p.add_argument('--preview',      action='store_true',
                   help='Shorthand for --step 5 --dpi 85 --video_width 640. '
                        'Renders at real-time speed in ~2-3 min.')
    p.add_argument('--fast-preview', action='store_true',
                   help='Very fast preview: aggressive subsample, skip video, '
                        'minimal visuals (good for quick qualitative checks).')
    p.add_argument('--max_display_frames', type=int, default=None,
                   help='Maximum number of frames to display/save (subsamples features).')
    p.add_argument('--online_inference', action='store_true', default=False,
                   help='Use true frame-by-frame streaming (slow, for latency '
                        'benchmarking only).')
    p.add_argument('--pred_threshold', type=float, default=0.15,
                   help='Hide prediction bars when max confidence is below '
                        'this value (shows "No activity detected" instead).')

    p.add_argument('--class_names',  default=None,
                   help='Text file with one class name per line (0-indexed).')
    p.add_argument('--save',         default=None,
                   help='Save animation to this MP4 path.')
    p.add_argument('--display',      action='store_true',
                   help='Open interactive window (needs display / X11).')
    p.add_argument('--gpu',          type=int, default=0)
    return p.parse_args()


# -- Main ---------------------------------------------------------------------

def main():
    args = parse_args()

    # Apply --preview shorthand
    if args.preview:
        args.step        = 5
        args.dpi         = 85
        args.video_width = 640
        print("[demo] --preview mode: step=5, dpi=85, video_width=640")

    # -- Fast-preview: extreme speedup for qualitative checks -------------
    if getattr(args, 'fast_preview', False):
        # Aggressive defaults for a very fast qualitative preview. Skip video
        # extraction, subsample heavily, and reduce visual complexity.
        args.step = max(args.step, 8)
        if args.max_display_frames is None:
            args.max_display_frames = 120
        # Skip extracting video frames (faster) by forcing video arg to None
        args.video = None
        args.dpi = min(args.dpi, 80)
        print("[demo] --fast-preview: step=8+, max_display_frames=120, skip video")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.save is None and not args.display:
        print("[demo] Neither --save nor --display -> defaulting to --save demo_output.mp4")
        args.save = 'demo_output.mp4'

    num_classes = 157 if args.dataset == 'charades' else 51
    class_map   = _get_class_map(args.dataset, args.class_names)
    device      = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[demo] Device             : {device}")

    # -- Locate video file ----------------------------------------------------
    video_path = _find_video(args.video)
    if video_path is None:
        print("[demo] No video file found -> feature-only mode.")

    # -- Model ----------------------------------------------------------------
    print(f"[demo] Loading checkpoint : {args.weights}")
    model    = _load_model(args, num_classes, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[demo] Parameters         : {n_params:,}")

    # -- Dataset video --------------------------------------------------------
    vid_id, features_orig, labels_orig, duration = _pick_video(args, num_classes)
    orig_T = features_orig.shape[0]

    # Subsample for display/inference if requested to speed up rendering
    if args.max_display_frames is not None and orig_T > args.max_display_frames:
        display_indices = np.linspace(0, orig_T - 1, args.max_display_frames, dtype=int)
        features = features_orig[display_indices]
        labels = labels_orig[display_indices]
        print(f"[demo] Subsampled {orig_T} -> {features.shape[0]} frames for display")
    else:
        features = features_orig
        labels = labels_orig
        display_indices = np.arange(orig_T, dtype=int)

    # -- Inference ------------------------------------------------------------
    if args.online_inference:
        print("[demo] Inference mode     : streaming (per-frame, slow)")
        probs = _run_streaming(model, features, device)
    else:
        print("[demo] Inference mode     : offline causal batch (fast)")
        t0 = time.time()
        probs = _run_offline_causal(model, features, device)
        print(f"[demo] Inference done     : {time.time()-t0:.2f}s")

    # -- Extract video frames -------------------------------------------------
    step = max(1, args.step)
    display_T = features.shape[0]
    frames_idx_display = list(range(0, display_T, step))
    # Map displayed indices back to original indices for video extraction
    frames_idx_original = display_indices[frames_idx_display]
    fps_feat_orig = orig_T / duration if duration > 0 else 25.0

    if video_path is not None:
        video_frames = _extract_video_frames(
            video_path, frames_idx_original, fps_feat_orig, args.video_width
        )
    else:
        video_frames = None

    # -- Compute output fps for real-time playback ----------------------------
    # fps_raw = unique rendered frames per second of real time.
    # We repeat each frame so the output stays at >= TARGET_FPS for smoothness,
    # while the video plays at 1x real-time speed.
    #
    #   step=1, fps_feat=25 -> fps_raw=25, repeat=1, fps_out=25  (full speed)
    #   step=3, fps_feat=25 -> fps_raw=8,  repeat=2, fps_out=16  (real speed)
    #   step=5, fps_feat=25 -> fps_raw=5,  repeat=3, fps_out=15  (real speed)
    # Estimate unique rendered fps in real time using original timeline spacing
    if len(display_indices) > 1:
        # average original-index stride between consecutive rendered frames
        orig_strides = np.diff(display_indices[::step]) if len(display_indices[::step]) > 1 else np.diff(display_indices)
        avg_orig_stride = float(np.mean(orig_strides)) if orig_strides.size > 0 else 1.0
    else:
        avg_orig_stride = 1.0
    fps_raw = (orig_T / duration) / avg_orig_stride if duration > 0 else (orig_T / 1.0) / avg_orig_stride
    TARGET_FPS   = 15
    frame_repeat = max(1, round(TARGET_FPS / fps_raw))
    fps_out      = round(fps_raw * frame_repeat)
    print(f"[demo] Playback           : {fps_raw:.1f} unique fps x "
          f"{frame_repeat} repeat = {fps_out} fps output  "
          f"(1x real-time, step={step})")

    # -- Build figure ---------------------------------------------------------
    print("[demo] Building figure ...")
    fig, update_fn, n_frames = _build_figure(
        vid_id, features, probs, labels, duration,
        class_map, video_frames, args,
    )
    print(f"[demo] Animation          : {n_frames} frames  (dpi={args.dpi})")

    # -- Save -----------------------------------------------------------------
    if args.save is not None:
        save_path = args.save
        if not os.path.isabs(save_path):
            save_path = os.path.join(_HERE, save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        t0 = time.time()
        if _ffmpeg_available():
            print("[demo] Renderer           : ffmpeg pipe (fast)")
            _render_pipe(fig, update_fn, n_frames, fps_out, save_path,
                         args.dpi, frame_repeat=frame_repeat)
        else:
            print("[demo] Renderer           : matplotlib (ffmpeg not on PATH)")
            _render_matplotlib(fig, update_fn, n_frames, fps_out, save_path,
                               args.dpi)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"[demo] Saved  {save_path}  "
              f"({size_mb:.1f} MB, {elapsed:.1f}s, "
              f"{n_frames/elapsed:.1f} unique frames/s)")

    # -- Display --------------------------------------------------------------
    if args.display:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        # interval_ms matches real-time playback speed (approx)
        if duration > 0:
            interval_ms = int(1000 * avg_orig_stride / (orig_T / duration))
        else:
            interval_ms = int(1000 * step)
        anim = FuncAnimation(fig, update_fn, frames=n_frames,
                             interval=interval_ms, blit=True)
        plt.show()

    print("[demo] Done.")


if __name__ == '__main__':
    main()
