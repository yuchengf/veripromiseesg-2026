#!/bin/bash
# Overnight v4.1 training: C1R (R-Drop+SWA), C1S (SharpReCL), C1M (MR2), C1RS (R-Drop+SharpReCL)
# Then gen_submissions for s440-s449
set -e
cd /home/yucheng/Desktop/ESG

echo "=== $(date) === Overnight v4.1 training ==="

# Common args
BACKBONE="hfl/chinese-roberta-wwm-ext-large"
COMMON="--mode kfold --approach A1 --backbone $BACKBONE --data_dir 3rd_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0"

# R-Drop needs smaller batch_size (double forward pass → 2x VRAM)
BS_RDROP="--batch_size 8"
BS_NORMAL="--batch_size 16"

train_model() {
  local NAME=$1 SEED=$2 EXTRA=$3 BS=$4
  if [ -f "runs/$NAME/fold5/best.pt" ]; then
    echo "  [SKIP] $NAME: all 5 folds complete"
    return
  fi
  echo ""
  echo "=== $(date) === Training $NAME (seed=$SEED) ==="
  python -u esg_main.py $COMMON --seed $SEED --run_dir runs/$NAME $BS $EXTRA || \
    echo "  [FAIL] $NAME failed with exit code $?"
  echo "=== $(date) === $NAME done ==="
}

# ═══ C1R: R-Drop α=0.5 + SWA (batch_size=8 to avoid OOM) ═══
train_model C1R_roberta_dc_kfold_t3nc3    42  "--rdrop_alpha 0.5 --swa_start_epoch 7" "$BS_RDROP"
train_model C1R_roberta_dc_kfold_t3nc3_s1 123 "--rdrop_alpha 0.5 --swa_start_epoch 7" "$BS_RDROP"
train_model C1R_roberta_dc_kfold_t3nc3_s2 456 "--rdrop_alpha 0.5 --swa_start_epoch 7" "$BS_RDROP"

# ═══ C1S: SharpReCL (normal batch_size) ═══
train_model C1S_roberta_dc_kfold_t3nc3    42  "--sharp_recl_weight 0.10" "$BS_NORMAL"
train_model C1S_roberta_dc_kfold_t3nc3_s1 123 "--sharp_recl_weight 0.10" "$BS_NORMAL"
train_model C1S_roberta_dc_kfold_t3nc3_s2 456 "--sharp_recl_weight 0.10" "$BS_NORMAL"

# ═══ C1M: MR2 Adaptive Margin (normal batch_size) ═══
train_model C1M_roberta_dc_kfold_t3nc3    42  "--mr2_weight 0.05" "$BS_NORMAL"
train_model C1M_roberta_dc_kfold_t3nc3_s1 123 "--mr2_weight 0.05" "$BS_NORMAL"
train_model C1M_roberta_dc_kfold_t3nc3_s2 456 "--mr2_weight 0.05" "$BS_NORMAL"

# ═══ C1RS: R-Drop + SharpReCL combined (batch_size=8) ═══
train_model C1RS_roberta_dc_kfold_t3nc3    42  "--rdrop_alpha 0.5 --swa_start_epoch 7 --sharp_recl_weight 0.10" "$BS_RDROP"
train_model C1RS_roberta_dc_kfold_t3nc3_s1 123 "--rdrop_alpha 0.5 --swa_start_epoch 7 --sharp_recl_weight 0.10" "$BS_RDROP"
train_model C1RS_roberta_dc_kfold_t3nc3_s2 456 "--rdrop_alpha 0.5 --swa_start_epoch 7 --sharp_recl_weight 0.10" "$BS_RDROP"

# ═══ Generate all new submissions ═══
echo ""
echo "=== $(date) === Generating submissions ==="
python -u esg_main.py --mode gen_submissions --data_dir 3rd_data --skip_existing

echo ""
echo "=== $(date) === All done ==="
echo "Check submissions/s44*.csv for new files"
