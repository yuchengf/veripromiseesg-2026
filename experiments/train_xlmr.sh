#!/bin/bash
set -e
cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKBONE="FacebookAI/xlm-roberta-large"
DATA="final_data"
COMMON="--mode kfold --approach A1 --backbone $BACKBONE --data_dir $DATA --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 384"

echo "=== XLM-R training: $(date) ==="

# FX1R (R-Drop + SWA, strongest recipe) x 3 seeds
for PAIR in "42:FX1R_xlmr_dc_kfold_t3nc3" "123:FX1R_xlmr_dc_kfold_t3nc3_s1" "456:FX1R_xlmr_dc_kfold_t3nc3_s2"; do
  SEED="${PAIR%%:*}"; NAME="${PAIR##*:}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME (seed=$SEED) ==="
  if ! python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 4 --rdrop_alpha 0.5 --swa_start_epoch 7; then
    echo "=== bs=4 failed (likely OOM), retry bs=2 ==="
    python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 2 --rdrop_alpha 0.5 --swa_start_epoch 7
  fi
done

echo "=== XLM-R done: $(date) ==="
