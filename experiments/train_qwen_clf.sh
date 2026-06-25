#!/bin/bash
set -e; cd /home/yucheng/Desktop/ESG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME="BQ1_qwen8b_dc_kfold_t3nc3"
COMMON="--mode kfold --approach B_lora --backbone Qwen/Qwen3-8B --data_dir final_data --kfold 5 --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 --max_length 384 --seed 42"
echo "=== $(date) Training $NAME (Qwen3-8B LoRA classifier) ==="
if [ ! -f "runs/$NAME/fold5/best.pt" ]; then
  python -u esg_main.py $COMMON --run_dir "runs/$NAME" --batch_size 4 \
   || python -u esg_main.py $COMMON --run_dir "runs/$NAME" --batch_size 2 \
   || python -u esg_main.py $COMMON --run_dir "runs/$NAME" --batch_size 1
fi
echo "=== $(date) eval_single (cache valid probs + standalone score) ==="
python -u eval_single_run.py "$NAME" || true
echo "=== $(date) DONE: 若 standalone valid << 0.665 → 和其他 backbone 一樣無助益(天花板再確認) ==="
