#!/bin/bash
#SBATCH --account=3199302
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-tsu_clip-%j.out
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ====== USER CONFIG ======
MYID=3199302
BASE_HOME=/mnt/beegfsstudents/home/$MYID
USER_HOME=/home/$MYID

# ====== ENV SETUP ======
source ~/.bashrc
conda activate mstemba

cd $BASE_HOME/ASO-Temba/vim

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ====== DEBUG INFO ======
hostname
which python
echo "MYID=$MYID"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python -m pip show mamba-ssm || true
echo "======================"


python MSTemba_main.py \
  -dataset charades \
  -mode rgb \
  -backbone clip \
  -model mstemba \
  -train True \
  -rgb_root $BASE_HOME/ASO-Temba/data/charades_features_clip \
  -num_clips 256 \
  -skip 0 \
  -comp_info False \
  -epochs 50 \
  -unisize True \
  -alpha_l 1 \
  -beta_l 0.05 \
  -batch_size 5 \
  -output_dir $BASE_HOME/ASO-Temba/outputs/charades_clip



echo "Training done."
