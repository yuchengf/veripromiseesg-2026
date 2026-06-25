#!/bin/bash
# New backbone smoke runs (seed 42 only, FC1R recipe). Promote to 3 seeds only if
# single-seed valid >= FC1R single-seed baseline.
set -e
cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA="final_data"
COMMON="--mode kfold --approach A1 --data_dir $DATA --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 384 --seed 42 --rdrop_alpha 0.5 --swa_start_epoch 7"

echo "=== new-backbone smoke: $(date) ==="

# mmBERT-base (307M, JHU-CLSP) — modern multilingual, beats XLM-R on classification
NAME="NB1R_mmbert_dc_kfold_t3nc3"
if [ ! -f "runs/$NAME/fold5/best.pt" ]; then
  echo "=== $(date) Training $NAME ==="
  if ! python -u esg_main.py $COMMON --backbone jhu-clsp/mmBERT-base --run_dir "runs/$NAME" --batch_size 4; then
    echo "=== bs=4 failed, retry bs=2 ==="
    python -u esg_main.py $COMMON --backbone jhu-clsp/mmBERT-base --run_dir "runs/$NAME" --batch_size 2
  fi
fi
python -u eval_single_run.py "$NAME" || true

# BGE-M3 backbone (568M, XLM-R-large arch + 100-lang retrieval pretraining)
NAME="NG1R_bgem3_dc_kfold_t3nc3"
if [ ! -f "runs/$NAME/fold5/best.pt" ]; then
  echo "=== $(date) Training $NAME ==="
  if ! python -u esg_main.py $COMMON --backbone BAAI/bge-m3 --run_dir "runs/$NAME" --batch_size 4; then
    echo "=== bs=4 failed, retry bs=2 ==="
    python -u esg_main.py $COMMON --backbone BAAI/bge-m3 --run_dir "runs/$NAME" --batch_size 2
  fi
fi
python -u eval_single_run.py "$NAME" || true

echo "=== new-backbone smoke done: $(date) ==="
