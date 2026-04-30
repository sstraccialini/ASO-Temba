#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python MSTemba_main.py \
-dataset tsu \
-mode rgb \
-backbone clip/i3d/ \
-model mstemba \
-train True \
-rgb_root /path/to/TSU_features/ \
-num_clips 2500 \
-skip 0 \
--lr 4.5e-4 \
-comp_info False \
-epochs 140 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 1 \
-output_dir /path/to/output_folder/