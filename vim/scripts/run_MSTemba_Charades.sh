#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python MSTemba_main.py \
-dataset charades \
-mode rgb \
-backbone clip/i3d/ \
-model mstemba \
-train True \
-rgb_root /path/to/charades_features/ \
-num_clips 256 \
-skip 0 \
-comp_info False \
-epochs 50 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 5 \
-output_dir /path/to/output_folder/
