#!/bin/bash
cd /home/yucheng/Desktop/ESG
BACKBONE="hfl/chinese-roberta-wwm-ext-large"
COMMON="--mode kfold --approach A1 --backbone $BACKBONE --data_dir final_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0"

echo "=== $(date) === Makeup training ==="

for PAIR in "42:FC1_roberta_dc_kfold_t3nc3:" "123:FC1_roberta_dc_kfold_t3nc3_s1:" "456:FC1_roberta_dc_kfold_t3nc3_s2:"; do
  SEED="${PAIR%%:*}"; REST="${PAIR#*:}"; NAME="${REST%%:*}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME ===" && python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 16
done

[ -f "runs/FC1R_roberta_dc_kfold_t3nc3/fold5/best.pt" ] || {
  echo "=== $(date) Training FC1R seed42 ===" && python -u esg_main.py $COMMON --seed 42 --run_dir runs/FC1R_roberta_dc_kfold_t3nc3 --batch_size 8 --rdrop_alpha 0.5 --swa_start_epoch 7
}

for PAIR in "123:FC1M_roberta_dc_kfold_t3nc3_s1:" "456:FC1M_roberta_dc_kfold_t3nc3_s2:"; do
  SEED="${PAIR%%:*}"; REST="${PAIR#*:}"; NAME="${REST%%:*}"
  [ -f "runs/$NAME/fold5/best.pt" ] && echo "[SKIP] $NAME" && continue
  echo "=== $(date) Training $NAME ===" && python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 16 --mr2_weight 0.05
done

echo "=== $(date) === All makeup done ==="
