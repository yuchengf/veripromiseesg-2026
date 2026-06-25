#!/bin/bash
# Overnight pipeline 2026-06-13:
#   1. kNN fine sweep (CPU, immediate)
#   2. wait for gen_rt_more (GPU) to finish
#   3. Misleading hunter: Qwen3-14B panel + scan 2000 AIdea test rows
#   4. XLM-R-large FX1R x3 seeds training
#   5. rerun pertask_valid_eval (caches FX1R valid/OOF, re-evaluates ensembles)
cd /home/yucheng/Desktop/ESG
LOG=agent_cache
mkdir -p "$LOG"

echo "=== overnight start: $(date) ===" | tee "$LOG/overnight.log"

echo "--- [1/5] kNN sweep (CPU) $(date) ---" | tee -a "$LOG/overnight.log"
python -u knn_sweep.py > "$LOG/knn_sweep.log" 2>&1 \
  && echo "knn_sweep OK" >> "$LOG/overnight.log" \
  || echo "knn_sweep FAILED" >> "$LOG/overnight.log"

echo "--- [2/5] waiting for gen_rt_more $(date) ---" | tee -a "$LOG/overnight.log"
while pgrep -f "gen_rt_more.py" > /dev/null; do sleep 60; done
echo "gen_rt_more finished $(date)" >> "$LOG/overnight.log"

echo "--- [3/5] misleading hunter $(date) ---" | tee -a "$LOG/overnight.log"
python -u misleading_hunter.py > "$LOG/misleading_hunter.log" 2>&1 \
  && echo "misleading_hunter OK" >> "$LOG/overnight.log" \
  || echo "misleading_hunter FAILED" >> "$LOG/overnight.log"

echo "--- [4/5] XLM-R training $(date) ---" | tee -a "$LOG/overnight.log"
bash train_xlmr.sh > "$LOG/train_xlmr.log" 2>&1 \
  && echo "train_xlmr OK" >> "$LOG/overnight.log" \
  || echo "train_xlmr FAILED" >> "$LOG/overnight.log"

echo "--- [5/5] pertask re-eval with FX1R $(date) ---" | tee -a "$LOG/overnight.log"
python -u pertask_valid_eval.py > "$LOG/pertask_eval_fx1r.log" 2>&1 \
  && echo "pertask_eval OK" >> "$LOG/overnight.log" \
  || echo "pertask_eval FAILED" >> "$LOG/overnight.log"

echo "=== overnight done: $(date) ===" | tee -a "$LOG/overnight.log"
