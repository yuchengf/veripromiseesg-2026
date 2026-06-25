#!/bin/bash
set -e
cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BACKBONE="FacebookAI/xlm-roberta-large"
COMMON="--mode kfold --approach A1 --backbone $BACKBONE --data_dir final_data_ml --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 384 --seed 42 --rdrop_alpha 0.5 --swa_start_epoch 7"
NAME="MX1R_xlmr_ml_dc_kfold_t3nc3"
echo "=== $(date) Training $NAME (multilingual ZH+EN/FR/JA) ==="
if ! python -u esg_main.py $COMMON --run_dir "runs/$NAME" --batch_size 4; then
  echo "=== bs=4 failed, retry bs=2 ==="
  python -u esg_main.py $COMMON --run_dir "runs/$NAME" --batch_size 2
fi
echo "=== $(date) $NAME done ==="
