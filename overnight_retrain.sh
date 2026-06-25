#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Overnight Retrain: full_train_data (2000 rows) → predict test_2000
# Best architecture: 12-way ensemble + Qwen3 kNN
#   FC1  (baseline)       × 3 seeds
#   FC1R (R-Drop+SWA)     × 3 seeds
#   FC1S (SharpReCL)      × 3 seeds
#   FC1M (MR2 margin)     × 3 seeds
# ═══════════════════════════════════════════════════════════════
set -e
cd /home/yucheng/Desktop/ESG

BACKBONE="hfl/chinese-roberta-wwm-ext-large"
DATA="retrain_data"
COMMON="--mode kfold --approach A1 --backbone $BACKBONE --data_dir $DATA --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0"

echo "═══════════════════════════════════════════════════════"
echo "  Retrain started: $(date)"
echo "  Data: $DATA (2000 train → predict 2000 test)"
echo "═══════════════════════════════════════════════════════"

# ── FC1 baseline × 3 seeds ──
for PAIR in "42:RT_FC1_s0" "123:RT_FC1_s1" "456:RT_FC1_s2"; do
  SEED="${PAIR%%:*}"; NAME="${PAIR##*:}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME (seed=$SEED) ==="
  python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 16
done

# ── FC1R (R-Drop + SWA) × 3 seeds ──
for PAIR in "42:RT_FC1R_s0" "123:RT_FC1R_s1" "456:RT_FC1R_s2"; do
  SEED="${PAIR%%:*}"; NAME="${PAIR##*:}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME (seed=$SEED) ==="
  python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 8 --rdrop_alpha 0.5 --swa_start_epoch 7
done

# ── FC1S (SharpReCL) × 3 seeds ──
for PAIR in "42:RT_FC1S_s0" "123:RT_FC1S_s1" "456:RT_FC1S_s2"; do
  SEED="${PAIR%%:*}"; NAME="${PAIR##*:}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME (seed=$SEED) ==="
  python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 16 --sharp_recl_weight 0.10
done

# ── FC1M (MR2 margin) × 3 seeds ──
for PAIR in "42:RT_FC1M_s0" "123:RT_FC1M_s1" "456:RT_FC1M_s2"; do
  SEED="${PAIR%%:*}"; NAME="${PAIR##*:}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME (seed=$SEED) ==="
  python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 16 --mr2_weight 0.05
done

echo "═══════════════════════════════════════════════════════"
echo "  All retrain done: $(date)"
echo "  Models: runs/RT_FC1*"
echo "═══════════════════════════════════════════════════════"
