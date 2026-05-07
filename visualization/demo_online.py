"""
visualization/demo_online.py
============================
Live online-inference demo for MS-Temba (causal streaming mode).

Speed knobs
---------------------------------
  --step N           Use every N-th feature frame (default 1 = all frames).
                     Output runs at step * real-time speed, consistently.

  --dpi D            Output resolution (default 100).  80 = draft, 140 = paper.

  --preview          Shorthand for --step 4 --dpi 85.

  --fast-preview     Very fast: step=8, dpi=72.

  --online_inference Run true frame-by-frame streaming instead of the fast
                     offline causal batch forward.  Numerically identical
                     outputs, but much slower.  Off by default.

  --pred_threshold   Confidence below which a ground-truth label is considered
                     NOT predicted (turns grey).  Default 0.15.

Speed model
-----------
  Output FPS = fps_feat / step.   All frames are unique; no frame repetition.
  This means with step=5 the animation plays 5× faster than real time, which
  is consistent and predictable so you can overlay the real video on top.

Layout
------
  +-----------------------------------------------------------------+
  |  MS-Temba - Online Causal Inference Demo                        |
  |  Video: P11T15C01   ################ooooo  123 / 500            |
  +-----------------------------------------------------------------+
  |                                                                 |
  |   [ Video — upload separately and sync at the same speed ]      |
  |                                                                 |
  +------------------------------+----------------------------------+
  |  LIVE PREDICTIONS (top 5)    |  GROUND TRUTH                    |
  |  shown only when GT active   |  ✓ Sit down   (green=predicted)  |
  |                              |    Read book  (grey=missed)       |
  +------------------------------+----------------------------------+

Usage
-----
  # All frames, real-time output speed:
    python visualization/demo_online.py \\
        --weights ../outputs/causal-tsu_i3d/best_model.pth \\
        --dataset tsu --backbone i3d \\
        --rgb_root /path/to/tsu_features_i3d \\
        --save demo_tsu.mp4

  # 4x faster than real time (step=4):
    python visualization/demo_online.py ... --step 4 --save demo_tsu.mp4

  # Quick preview:
    python visualization/demo_online.py ... --preview --save demo_tsu.mp4
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
    0: "Enter",
    1: "Walk",
    2: "Make_coffee",
    3: "Get_water",
    4: "Make_coffee.Pour_water",
    5: "Use_Drawer",
    6: "Make_coffee.Pour_grains",
    7: "Use_telephone",
    8: "Leave",
    9: "Put_something_on_table",
    10: "Take_something_off_table",
    11: "Pour.From_kettle",
    12: "Stir_coffee/tea",
    13: "Drink.From_cup",
    14: "Dump_in_trash",
    15: "Make_tea",
    16: "Make_tea.Boil_water",
    17: "Use_cupboard",
    18: "Make_tea.Insert_tea_bag",
    19: "Read",
    20: "Take_pills",
    21: "Use_fridge",
    22: "Clean_dishes",
    23: "Clean_dishes.Put_something_in_sink",
    24: "Eat_snack",
    25: "Sit_down",
    26: "Watch_TV",
    27: "Use_laptop",
    28: "Get_up",
    29: "Drink.From_bottle",
    30: "Pour.From_bottle",
    31: "Drink.From_glass",
    32: "Lay_down",
    33: "Drink.From_can",
    34: "Write",
    35: "Breakfast",
    36: "Breakfast.Spread_jam_or_butter",
    37: "Breakfast.Cut_bread",
    38: "Breakfast.Eat_at_table",
    39: "Breakfast.Take_ham",
    40: "Clean_dishes.Dry_up",
    41: "Wipe_table",
    42: "Cook",
    43: "Cook.Cut",
    44: "Cook.Use_stove",
    45: "Cook.Stir",
    46: "Cook.Use_oven",
    47: "Clean_dishes.Clean_with_water",
    48: "Use_tablet",
    49: "Use_glasses",
    50: "Pour.From_can",
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
    """Single-pass causal batch forward. Numerically identical to streaming."""
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
    vid_id:    str,
    probs:     np.ndarray,
    labels:    np.ndarray,
    duration:  float,
    class_map: dict[int, str],
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

    T          = probs.shape[0]
    step       = max(1, args.step)
    frames_idx = list(range(0, T, step))
    fps_approx = T / duration if duration > 0 else 25.0
    topk       = 5

    PRED_THRESHOLD = getattr(args, 'pred_threshold', 0.15)

    # Colours
    PRED_COLORS = ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626']
    GT_CORRECT  = '#16a34a'   # bright green: model predicts this GT label
    GT_MISSED   = '#9ca3af'   # grey: GT label not predicted
    BAR_BG      = '#e5e7eb'
    C_DARK      = '#1f2937'
    C_MID       = '#6b7280'
    C_LIGHT     = '#9ca3af'

    # -------------------------------------------------------------------------
    # Figure layout (3 rows):
    #   [0] info bar    (progress + frame counter)
    #   [1] video placeholder  (user overlays real video here)
    #   [2] predictions (left) + ground truth (right)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 9), facecolor='white')
    fig.suptitle("MS-Temba  -  Online Causal Inference Demo",
                 fontsize=13, fontweight='bold', color='#1a1a2e', y=0.99)

    gs = gridspec.GridSpec(3, 1, figure=fig,
                           height_ratios=[0.04, 0.48, 0.48],
                           top=0.96, bottom=0.03, hspace=0.22)
    ax_info   = fig.add_subplot(gs[0])
    ax_video  = fig.add_subplot(gs[1])
    ax_bottom = fig.add_subplot(gs[2])

    # -- Info bar -------------------------------------------------------------
    ax_info.set_axis_off()
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.barh(0.35, 1.0, height=0.30, color='#e5e7eb', left=0, zorder=1)
    pg_bar, = ax_info.barh(0.35, 0.0, height=0.30, color='#2563EB',
                           left=0, zorder=2, alpha=0.85)
    txt_info = ax_info.text(0.0, 0.88, '', transform=ax_info.transAxes,
                            fontsize=10, color=C_DARK, va='top',
                            fontfamily='monospace')

    # -- Video placeholder (plain white space — user overlays real video here) -
    ax_video.set_facecolor('white')
    ax_video.set_axis_off()

    # -- Bottom row: predictions (left 55%) + ground truth (right 45%) --------
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

    # -- Predictions panel ----------------------------------------------------
    ax_pred.set_xlim(0, 1); ax_pred.set_ylim(-0.5, topk - 0.5)
    ax_pred.set_facecolor('white')
    ax_pred.set_title("Live Predictions  (top 5)",
                      fontsize=9.5, fontweight='bold', color=C_DARK, pad=5)
    ax_pred.set_yticks([])
    ax_pred.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_pred.set_xticklabels(['0', '', '0.5', '', '1'], fontsize=8, color=C_LIGHT)
    ax_pred.set_xlabel("Confidence", fontsize=8, color=C_MID)
    for sp in ax_pred.spines.values(): sp.set_edgecolor('#e5e7eb')
    # static background bars
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

    # -- Ground-truth panel ---------------------------------------------------
    ax_gt.set_axis_off()
    ax_gt.set_facecolor('#f0fdf4')
    ax_gt.text(0.5, 0.97, "Ground Truth",
               transform=ax_gt.transAxes,
               fontsize=11, fontweight='bold', color='#14532d',
               ha='center', va='top')
    ax_gt.plot([0.05, 0.95], [0.88, 0.88],
               transform=ax_gt.transAxes,
               color='#86efac', linewidth=1.5, solid_capstyle='round')
    # Legend hint
    ax_gt.text(0.06, 0.85, '✓ = predicted   ·   grey = missed',
               transform=ax_gt.transAxes,
               fontsize=7.5, color=C_LIGHT, va='top', style='italic')

    gt_entries = [
        ax_gt.text(0.06, 0.76 - i * 0.12, '',
                   transform=ax_gt.transAxes,
                   fontsize=10.5, color=GT_CORRECT, va='top', fontweight='bold')
        for i in range(6)
    ]
    # Initialized empty; set_text() is used to show/hide (blit-safe: no visibility toggling)
    gt_none_txt = ax_gt.text(
        0.06, 0.76, '',
        transform=ax_gt.transAxes,
        fontsize=10, color=C_LIGHT, va='top', style='italic',
    )

    # -- Update function ------------------------------------------------------

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

        # Always show top-k predictions
        prob_t  = probs[t]
        top_idx = np.argsort(prob_t)[::-1][:topk]
        for k in range(topk):
            c = top_idx[k]; p = float(prob_t[c])
            name = class_map.get(c, f'class {c}')
            if len(name) > 28: name = name[:26] + '...'
            pred_bars[k].set_width(p)
            pred_labels[k].set_text(f"  {name}")
            pred_scores[k].set_text(f"{p:.2f}  ")

        # GT panel: use set_text to show/hide (set_visible breaks blit)
        active = np.where(labels[t] > 0.5)[0]
        if len(active) == 0:
            gt_none_txt.set_text('-- no active labels --')
            for g in gt_entries:
                g.set_text('')
        else:
            gt_none_txt.set_text('')
            for i, g in enumerate(gt_entries):
                if i < len(active):
                    c    = int(active[i])
                    name = class_map.get(c, f'class {c}')
                    if len(name) > 28: name = name[:26] + '...'
                    if float(prob_t[c]) > PRED_THRESHOLD:
                        g.set_text(f"✓  {name}")
                        g.set_color(GT_CORRECT)
                    else:
                        g.set_text(f"   {name}")
                        g.set_color(GT_MISSED)
                else:
                    g.set_text('')

        return (pg_bar, txt_info, gt_none_txt,
                *gt_entries, *pred_bars, *pred_labels, *pred_scores)

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
    Render frames directly into an ffmpeg pipe.

    frame_repeat: each unique rendered frame is written this many times so the
    output stays at *fps* while playing back at real-time speed when
    fps_feat < fps.  E.g. fps_feat=6.25, fps=25 -> frame_repeat=4.
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
        canvas = fig.canvas
        canvas.draw()
        try:
            background = canvas.copy_from_bbox(fig.bbox)
            use_blit = True
        except Exception:
            use_blit = False

        for fi in range(n_frames):
            if fi % max(1, n_frames // 20) == 0:
                pct = 100 * fi // n_frames
                print(f"\r[demo] Rendering {pct:>3d}%  ({fi}/{n_frames}) ...",
                      end='', flush=True)

            arts = update_fn(fi)

            if use_blit:
                try:
                    canvas.restore_region(background)
                    for a in arts:
                        try:
                            if a is not None and hasattr(a, 'axes') and a.axes is not None:
                                a.axes.draw_artist(a)
                        except Exception:
                            pass
                    canvas.blit(fig.bbox)
                except Exception:
                    use_blit = False
                    canvas.draw()
            if not use_blit:
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
          f"({n_frames} unique x{frame_repeat} = {n_frames*frame_repeat} output frames).",
          flush=True)


def _render_matplotlib(fig, update_fn, n_frames, fps, save_path, dpi, frame_repeat=1):
    """Fallback: matplotlib FuncAnimation + FFMpegWriter."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    total = n_frames * frame_repeat
    def _wrapped(fi):
        return update_fn(fi // frame_repeat)
    anim = FuncAnimation(fig, _wrapped, frames=total,
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
    p.add_argument('--video_id',     default=None,
                   help='Dataset video ID; random test video if omitted.')
    p.add_argument('--seed',         type=int, default=42)

    # Speed knobs
    p.add_argument('--step',         type=int, default=1,
                   help='Use every N-th feature frame.  Output plays at N× '
                        'real-time speed (consistent scaling).  1=all frames.')
    p.add_argument('--dpi',          type=int, default=100,
                   help='Output DPI.  80=draft, 100=default, 140=print.')
    p.add_argument('--preview',      action='store_true',
                   help='Shorthand for --step 4 --dpi 85.')
    p.add_argument('--fast-preview', action='store_true',
                   help='Very fast: --step 8 --dpi 72.')
    p.add_argument('--online_inference', action='store_true', default=False,
                   help='True frame-by-frame streaming (slow, latency benchmarking).')
    p.add_argument('--pred_threshold', type=float, default=0.15,
                   help='Probability threshold: GT label shown green when model '
                        'exceeds this confidence for that class.')
    p.add_argument('--output_fps',    type=int, default=25,
                   help='Output video fps.  Each unique animation frame is repeated '
                        'as needed so the output plays at exactly this fps and at '
                        '1× real-time speed.  Match your source video fps here.')

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

    if args.preview:
        args.step = max(args.step, 4)
        args.dpi  = min(args.dpi, 85)
        print(f"[demo] --preview: step={args.step}, dpi={args.dpi}")

    if getattr(args, 'fast_preview', False):
        args.step = max(args.step, 8)
        args.dpi  = min(args.dpi, 72)
        print(f"[demo] --fast-preview: step={args.step}, dpi={args.dpi}")

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

    # -- Model ----------------------------------------------------------------
    print(f"[demo] Loading checkpoint : {args.weights}")
    model    = _load_model(args, num_classes, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[demo] Parameters         : {n_params:,}")

    # -- Dataset video --------------------------------------------------------
    vid_id, features, labels, duration = _pick_video(args, num_classes)
    T = features.shape[0]

    # -- Inference ------------------------------------------------------------
    if args.online_inference:
        print("[demo] Inference mode     : streaming (per-frame, slow)")
        probs = _run_streaming(model, features, device)
    else:
        print("[demo] Inference mode     : offline causal batch (fast)")
        t0 = time.time()
        probs = _run_offline_causal(model, features, device)
        print(f"[demo] Inference done     : {time.time()-t0:.2f}s")

    # -- Output fps -----------------------------------------------------------
    # The output is always encoded at args.output_fps (default 25) so it can be
    # directly overlaid on the real source video without any speed adjustment.
    # Each unique rendered frame is repeated frame_repeat times to fill the gap
    # between the feature frame rate and the output fps.
    #
    #   fps_feat=6.25, step=1, output_fps=25  -> frame_repeat=4  (1× real time)
    #   fps_feat=6.25, step=2, output_fps=25  -> frame_repeat=8  (1× real time)
    fps_feat     = T / duration if duration > 0 else 25.0
    step         = max(1, args.step)
    fps_unique   = fps_feat / step          # unique animation frames per real second
    fps_out      = args.output_fps
    frame_repeat = max(1, round(fps_out / fps_unique))
    n_frames     = len(range(0, T, step))

    print(f"[demo] Feature fps        : {fps_feat:.2f} fps  (step={step} "
          f"-> {fps_unique:.2f} unique fps)")
    print(f"[demo] Output             : {fps_out} fps, {frame_repeat}× repeat per frame  "
          f"(1× real-time speed — overlay source video as-is)")
    print(f"[demo] Animation frames   : {n_frames} unique  x{frame_repeat} "
          f"= {n_frames*frame_repeat} output frames  (~{n_frames*frame_repeat/fps_out:.0f}s)")
    print(f"")
    print(f"[demo] *** VIDEO TO OVERLAY: {vid_id}  ***")
    print(f"")

    # -- Build figure ---------------------------------------------------------
    print("[demo] Building figure ...")
    fig, update_fn, n_frames = _build_figure(
        vid_id, probs, labels, duration, class_map, args,
    )
    print(f"[demo] Figure ready       : {n_frames} frames  (dpi={args.dpi})")

    # -- Save -----------------------------------------------------------------
    if args.save is not None:
        save_path = args.save
        if not os.path.isabs(save_path):
            save_path = os.path.join(_HERE, save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        t0 = time.time()
        if _ffmpeg_available():
            print("[demo] Renderer           : ffmpeg pipe (fast)")
            _render_pipe(fig, update_fn, n_frames, fps_out, save_path, args.dpi)
        else:
            print("[demo] Renderer           : matplotlib (ffmpeg not on PATH)")
            _render_matplotlib(fig, update_fn, n_frames, fps_out, save_path, args.dpi)

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
        interval_ms = max(1, int(1000 / fps_unique))
        anim = FuncAnimation(fig, update_fn, frames=n_frames,
                             interval=interval_ms, blit=True)
        plt.show()

    print("[demo] Done.")


if __name__ == '__main__':
    main()
