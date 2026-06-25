#!/bin/bash
set -e; cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--mode kfold --approach A1 --backbone hfl/chinese-roberta-wwm-ext-large --data_dir final_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 512 --seed 42 --rdrop_alpha 0.5 --swa_start_epoch 7"
NAME="FC1R512_roberta_dc_kfold_t3nc3"
echo "=== $(date) Training $NAME (max_length 512) ==="
if ! python -u esg_main.py $COMMON --run_dir "runs/$NAME" --batch_size 4; then
  echo "=== bs4 OOM, retry bs2 ==="; python -u esg_main.py $COMMON --run_dir "runs/$NAME" --batch_size 2
fi
python -u eval_single_run.py "$NAME"
echo "=== done $(date) ==="
