#!/bin/bash
set -e; cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--mode kfold --approach A1 --backbone hfl/chinese-roberta-wwm-ext-large --data_dir final_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 384 --rdrop_alpha 0.5 --swa_start_epoch 7 --t3_cond_weight 0.5"
NAME="COND_FC1R_final"
echo "=== $(date) Training $NAME (conditional T3 aux, seed 42) ==="
if [ ! -f "runs/$NAME/fold5/best.pt" ]; then
  python -u esg_main.py $COMMON --seed 42 --run_dir "runs/$NAME" --batch_size 8 \
   || python -u esg_main.py $COMMON --seed 42 --run_dir "runs/$NAME" --batch_size 4
fi
python -u eval_single_run.py "$NAME" || true
echo "=== $(date) DONE ==="
