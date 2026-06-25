#!/bin/bash
set -e
cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKBONE="IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"
DATA="final_data"
COMMON="--mode kfold --approach A1 --backbone $BACKBONE --data_dir $DATA --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 384"

echo "=== DeBERTa training: $(date) ==="

# FC1R (R-Drop + SWA) x 3 seeds — bs=4 for R-Drop
for PAIR in "42:FD1R_deberta_dc_kfold_t3nc3" "123:FD1R_deberta_dc_kfold_t3nc3_s1" "456:FD1R_deberta_dc_kfold_t3nc3_s2"; do
  SEED="${PAIR%%:*}"; NAME="${PAIR##*:}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME (seed=$SEED) ==="
  python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 4 --rdrop_alpha 0.5 --swa_start_epoch 7
done

# FC1 baseline x 3 seeds — bs=8
for PAIR in "42:FD1_deberta_dc_kfold_t3nc3" "123:FD1_deberta_dc_kfold_t3nc3_s1" "456:FD1_deberta_dc_kfold_t3nc3_s2"; do
  SEED="${PAIR%%:*}"; NAME="${PAIR##*:}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME (seed=$SEED) ==="
  python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 8
done

echo "=== DeBERTa done: $(date) ==="
