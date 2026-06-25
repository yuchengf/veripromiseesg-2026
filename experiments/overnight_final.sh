#!/bin/bash
# Final stage training on final_data/
# Train C1 + C1R (best from v4.1) on final_data, predict valid_data for Kaggle
set -e
cd /home/yucheng/Desktop/ESG

echo "=== $(date) === Final stage training ==="

BACKBONE="hfl/chinese-roberta-wwm-ext-large"
COMMON="--mode kfold --approach A1 --backbone $BACKBONE --data_dir final_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0"
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

# ═══ FC1: C1 baseline on final_data (3 seeds) ═══
train_model FC1_roberta_dc_kfold_t3nc3    42  "" "$BS_NORMAL"
train_model FC1_roberta_dc_kfold_t3nc3_s1 123 "" "$BS_NORMAL"
train_model FC1_roberta_dc_kfold_t3nc3_s2 456 "" "$BS_NORMAL"

# ═══ FC1R: C1 + R-Drop + SWA on final_data (3 seeds) ═══
train_model FC1R_roberta_dc_kfold_t3nc3    42  "--rdrop_alpha 0.5 --swa_start_epoch 7" "$BS_RDROP"
train_model FC1R_roberta_dc_kfold_t3nc3_s1 123 "--rdrop_alpha 0.5 --swa_start_epoch 7" "$BS_RDROP"
train_model FC1R_roberta_dc_kfold_t3nc3_s2 456 "--rdrop_alpha 0.5 --swa_start_epoch 7" "$BS_RDROP"

# ═══ FC1S: C1 + SharpReCL on final_data (3 seeds) ═══
train_model FC1S_roberta_dc_kfold_t3nc3    42  "--sharp_recl_weight 0.10" "$BS_NORMAL"
train_model FC1S_roberta_dc_kfold_t3nc3_s1 123 "--sharp_recl_weight 0.10" "$BS_NORMAL"
train_model FC1S_roberta_dc_kfold_t3nc3_s2 456 "--sharp_recl_weight 0.10" "$BS_NORMAL"

# ═══ FC1M: C1 + MR2 on final_data (3 seeds) ═══
train_model FC1M_roberta_dc_kfold_t3nc3    42  "--mr2_weight 0.05" "$BS_NORMAL"
train_model FC1M_roberta_dc_kfold_t3nc3_s1 123 "--mr2_weight 0.05" "$BS_NORMAL"
train_model FC1M_roberta_dc_kfold_t3nc3_s2 456 "--mr2_weight 0.05" "$BS_NORMAL"

echo ""
echo "=== $(date) === All training done ==="

# ═══ Generate submissions for Kaggle (valid_data, 399 rows) ═══
echo "=== Generating submissions ==="
python -u esg_main.py --mode gen_submissions --data_dir final_data --skip_existing

echo "=== $(date) === All done ==="
