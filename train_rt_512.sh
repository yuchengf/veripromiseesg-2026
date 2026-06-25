#!/bin/bash
# 全 12-way @512 on retrain_data(2000, AIdea 用)→ 推論 + 生成所有提交檔.
# 等 final_data 佇列(PID 參數)跑完再開始.
set -e; cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WAIT_PID="$1"
if [ -n "$WAIT_PID" ]; then echo "waiting for final_data queue PID $WAIT_PID..."; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
COMMON="--mode kfold --approach A1 --backbone hfl/chinese-roberta-wwm-ext-large --data_dir retrain_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 512"
run() {  # name seed extra
  NAME="$1"; SEED="$2"; EXTRA="$3"
  [ -f "runs/$NAME/fold5/best.pt" ] && { echo "[SKIP] $NAME"; return; }
  echo "=== $(date) Training $NAME (512, retrain) extra=$EXTRA ==="
  python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 8 $EXTRA \
    || python -u esg_main.py $COMMON --seed "$SEED" --run_dir "runs/$NAME" --batch_size 4 $EXTRA
}
for i in 0 1 2; do S=(42 123 456); SEED=${S[$i]}
  run RT_FC1512_s$i  $SEED ""
  run RT_FC1R512_s$i $SEED "--rdrop_alpha 0.5 --swa_start_epoch 7"
  run RT_FC1S512_s$i $SEED "--sharp_recl_weight 0.10"
  run RT_FC1M512_s$i $SEED "--mr2_weight 0.05"
done
echo "=== $(date) RT 12-way @512 done → 推論+生成提交檔 ==="
python -u gen_rt512.py
echo "=== $(date) ALL DONE ==="
