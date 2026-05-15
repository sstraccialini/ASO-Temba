<div align="center" style="display:flex;flex-direction:column;align-items:center;gap:8px;">
    <h1>ASO-Temba: Adaptive-Scale Online Mamba for Understanding Long Untrimmed Videos</h1>
    <p>
        <a href="https://github.com/thearkaprava/MS-Temba" target="_blank">
            <img src="https://img.shields.io/badge/Base-MS--Temba-1f6feb?style=flat-square" alt="Base repo">
        </a>
    </p>
</div>

## About the project

This repository extends **MS-Temba** (*Multi-Scale Temporal Mamba for Understanding Long Untrimmed Videos*, Sinha et al., 2025 — [arXiv:2501.06138](https://arxiv.org/abs/2501.06138), [original code](https://github.com/thearkaprava/MS-Temba)) for Temporal Action Detection (TAD) on densely labeled, long, untrimmed videos (Charades, Toyota Smarthome Untrimmed).

Our project investigates **two main modifications** of the original architecture. They are unified within this repository so they can be evaluated individually or together:

| Modification | Idea |
|---|---|
| **Fuser redesign** | Replace the original SSM-based Multi-Scale Mamba Fuser with alternative aggregation modules: learned **weighted** sums and several **attention-based** fusers (`weighted`, `token-attention`, `cross-token-attention`, `attention`) to study how multi-scale temporal features are best combined. |
| **Causal / online streaming** | Convert MS-Temba into a **causal**, **streaming** detector: forward-only SSMs (no bidirectional branch), a causal multi-scale fuser, and a frame-/chunk-level streaming inference pipeline suitable for **online** TAD. The original C-state diversity loss is replaced by a causal consistency loss (`L_caus_cons`). |

Additional context (motivation, qualitative results, benchmarks) is available in the project **poster** (`GameOfTones-Poster.pdf`) and **presentation** (`GameOfTones-presentation.pptx`) in this repository.

## Repository structure

```
ASO-Temba/
├── data/                         Dataset annotation JSONs
│   ├── charades.json
│   ├── multithumos.json
│   ├── smarthome.json            (Toyota Smarthome Untrimmed)
│   └── smarthome-10fps.json
├── vim/                          Main code (training, models, eval)
│   ├── models_MSTemba.py         Model + dilated SSM blocks + fuser variants
│   ├── MSTemba_main.py           Train / eval entry point
│   ├── streaming_inference.py    Online / chunked causal evaluation
│   ├── losses.py                 Action-detection + diversity / consistency losses
│   ├── charades_dataloader.py    Per-dataset dataloaders
│   ├── datasets.py, augment.py, samplers.py, apmeter.py, utils.py, engine.py, rope.py
│   ├── clip_feature_extraction.py
│   └── scripts/
│       ├── training/             SLURM jobs for training: job-{tsu,char}_{i3d,clip}.sh
│       │                         and job-causal-*.sh (causal streaming scripts)
│       └── evaluation/           SLURM jobs for evaluation: job-eval-{tsu,char}_{i3d,clip}.sh
├── visualization/
│   ├── demo_online.py            Live online-inference demo
│   └── job-demo-online.sh
├── causal-conv1d/                Vendored Dao-AILab/causal-conv1d (CUDA kernels)
├── mamba-1p1p1/                  Vendored mamba_ssm package (modified Mamba w/ C-state access)
├── det/, seg/                    Detectron2 / segmentation code inherited from
│                                 Vision-Mamba (not used by the TAD pipeline)
├── misc                          Various helpers and notebooks to download data, check annotations, extract frames, etc.
└── README.md
```

## Installation

The environment is sensitive to CUDA / PyTorch / `causal_conv1d` / `mamba_ssm` version pinning. The full procedure used in our experiments is reproduced below:

```bash
# 1. Conda env + CUDA 11.8 toolkit
conda create -n mstemba python=3.10.13 -y
conda activate mstemba

conda install -y --override-channels \
  -c "nvidia/label/cuda-11.8.0" \
  cuda-toolkit

nvcc --version

# 2. Base Python deps
pip install --upgrade pip setuptools wheel ninja packaging
pip install "numpy<2"

# 3. PyTorch 2.1.1 / cu118
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118

# 4. CUDA env vars (re-export in each shell / SLURM job)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 5. Pin setuptools/wheel for the legacy build system used by causal-conv1d
pip install "setuptools<70" "wheel<0.45"

# 6. Build vendored CUDA extensions
cd MS-Temba   # this repo's root
pip install --no-build-isolation --no-deps -e ./causal-conv1d
pip install --no-build-isolation        -e ./mamba-1p1p1

# 7. Remaining Python deps
pip install scikit-learn==1.3.2 tensorboard timm==0.4.12 \
            transformers==4.35.2 matplotlib==3.8.2
```

Sanity check:

```bash
python -c "
import torch, causal_conv1d, mamba_ssm
from mamba_ssm.modules.mamba_simple_getC import Mamba
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('Mamba imported OK')
"
```

Notice we replaced the `causal_conv1d` v1.0.0 used in the original paper with v1.1.1 from [https://github.com/Dao-AILab/causal-conv1d.git](https://github.com/Dao-AILab/causal-conv1d.git) due to CUDA compatibility issues in our machine, as suggested by the authors.

## Data

Like the original MS-Temba, we train on top of pre-extracted snippet features.

- **Charades** (24 fps): https://prior.allenai.org/projects/charades
- **Toyota Smarthome Untrimmed (TSU)**: https://project.inria.fr/toyotasmarthome/

As indicated by the authors, the features can be extracted using the following scripts as indicated from the MS-Temba authors:
- **I3D features**: extract with https://github.com/piergiaj/pytorch-i3d
- **CLIP features**: extract frames (`extract_frames.sh` / `serial_extract.sh`), then run `vim/clip_feature_extraction.py`

For convenience, we used the pre-extracted features provided by the authors of MS-Temba, which are available at https://huggingface.co/datasets/thearkaprava/Temporal_Action_Detection. Annotation JSONs are shipped under `data/`.

## Training & evaluation

The entry point is `vim/MSTemba_main.py`. Ready-to-run SLURM scripts live in `vim/scripts/training/` and `vim/scripts/evaluation/`; update the paths at the top of each script to your environment before submitting.

**Offline / fuser ablations.** Pick the fuser via `--fuser`:

```bash
python vim/MSTemba_main.py \
  -dataset tsu -mode rgb -backbone i3d -model mstemba \
  -train True -rgb_root /path/to/tsu_features_i3d \
  -num_clips 2500 -unisize True -batch_size 1 \
  --fuser weighted \
  -output_dir ./outputs/tsu_i3d-weighted
```

Supported values: `sum` (original), `weighted`, `token-attention`, `cross-token-attention`, `attention`.

**Causal / streaming.** Add the causal flags:

```bash
python vim/MSTemba_main.py \
  -dataset tsu -backbone i3d -model mstemba -train True \
  -rgb_root /path/to/tsu_features_i3d \
  --causal \
  --causal_consistency_loss_weight 1.0 \
  --causal_consistency_margin 0.1 \
  -output_dir ./outputs/causal-tsu_i3d
```

Online evaluation (per-frame or N-frame chunks) is run via `vim/streaming_inference.py`:

```bash
python vim/streaming_inference.py \
  --weights ./outputs/causal-tsu_i3d/best_model.pth \
  --dataset tsu --backbone i3d \
  --rgb_root /path/to/tsu_features_i3d \
  --stream_chunk_size 1        # 1 = frame-by-frame, N = N-frame buffer, 0 = mAP only
```

A live qualitative demo of the streaming model is provided in `visualization/demo_online.py`.

---
---
> Project carried out for the **Computer Vision and Image Processing** course, M.Sc. in Artificial Intelligence - Bocconi University, 2025/26.