#!/bin/bash
# 全 12-way @512 (final_data, valid 用). FC1R×3 另由 train_512* 處理.
# 等目前 FC1R512 s1/s2 (PID 由參數傳入) 跑完再開始,避免 GPU 爭用.
set -e; cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WAIT_PID="$1"
if [ -n "$WAIT_PID" ]; then echo "waiting for PID $WAIT_PID..."; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
COMMON="--mode kfold --approach A1 --backbone hfl/chinese-roberta-wwm-ext-large --data_dir final_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 512"
run() {  # name seed extra
  NAME="$1"; SEED="$2"; EXTRA="$3"
  [ -f "runs/$NAME/fold5/best.pt" ] && { echo "[SKIP] $NAME"; return; }
  echo "=== $(date) Training $NAME (512) extra=$EXTRA ==="
  if ! python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 8 $EXTRA; then
    python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 4 $EXTRA
  fi
  python -u eval_single_run.py "$NAME" || true
}
# FC1 base ×3
run FC1512_roberta_dc_kfold_t3nc3    42  ""
run FC1512_roberta_dc_kfold_t3nc3_s1 123 ""
run FC1512_roberta_dc_kfold_t3nc3_s2 456 ""
# FC1S sharprecl ×3
run FC1S512_roberta_dc_kfold_t3nc3    42  "--sharp_recl_weight 0.10"
run FC1S512_roberta_dc_kfold_t3nc3_s1 123 "--sharp_recl_weight 0.10"
run FC1S512_roberta_dc_kfold_t3nc3_s2 456 "--sharp_recl_weight 0.10"
# FC1M mr2 ×3
run FC1M512_roberta_dc_kfold_t3nc3    42  "--mr2_weight 0.05"
run FC1M512_roberta_dc_kfold_t3nc3_s1 123 "--mr2_weight 0.05"
run FC1M512_roberta_dc_kfold_t3nc3_s2 456 "--mr2_weight 0.05"
echo "=== 12-way @512 final_data done $(date) ==="
