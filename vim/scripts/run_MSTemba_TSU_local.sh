#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python MSTemba_main.py \
  -dataset tsu \
  -mode rgb \
  -backbone i3d \
  -train True \
  -rgb_root /home/3185670/i3d_features \
  -epochs 140 \
  -batch_size 1 \
  -num_clips 2500 \
  -skip 0 \
  -comp_info False \
  -unisize True \
  -alpha_l 1 \
  -beta_l 0.05 \
  -output_dir /home/3185670/ASO-Temba/outputs/tsu_i3d
