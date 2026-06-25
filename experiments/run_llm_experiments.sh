#!/usr/bin/env bash
# run_llm_experiments.sh
# Stage 1: format × RAG combinations, single 80/20 run each
# Runs sequentially (one GPU). Each ~20-30 min.
# Usage: bash run_llm_experiments.sh 2>&1 | tee /tmp/llm_experiments.log

set -eo pipefail
PYTHON=/home/yucheng/miniconda3/envs/AICUP/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

BASE_ARGS="--mode single --backbone Qwen/Qwen3-8B \
  --max_length 2800 --batch_size 1 --grad_accum 16 \
  --epochs 3"

declare -A EXPERIMENTS=(
  ["pc_rag3"]="--output_format promptcast --rag_k 3 --run_dir runs/LLM_single_pc_rag3"
  ["pc_norag"]="--output_format promptcast --rag_k 0 --run_dir runs/LLM_single_pc_norag"
  ["json_rag3"]="--output_format json      --rag_k 3 --run_dir runs/LLM_single_json_rag3"
  ["json_norag"]="--output_format json      --rag_k 0 --run_dir runs/LLM_single_json_norag"
)

RESULTS=()

for name in pc_rag3 pc_norag json_rag3 json_norag; do
  args="${EXPERIMENTS[$name]}"
  log="/tmp/llm_${name}.log"

  echo ""
  echo "============================================================"
  echo "  START: $name"
  echo "  Args : $args"
  echo "  Log  : $log"
  echo "  Time : $(date '+%H:%M:%S')"
  echo "============================================================"

  $PYTHON -u llm_lora.py $BASE_ARGS $args 2>&1 | tee "$log"

  # Extract val score from log
  score=$(grep "Val weighted F1:" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "N/A")
  RESULTS+=("$name: $score")

  echo ""
  echo "  DONE: $name  val_f1=$score  $(date '+%H:%M:%S')"
done

echo ""
echo "============================================================"
echo "  EXPERIMENT SUMMARY"
echo "============================================================"
for r in "${RESULTS[@]}"; do
  echo "  $r"
done
echo "============================================================"
