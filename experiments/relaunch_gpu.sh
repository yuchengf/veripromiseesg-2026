#!/bin/bash
# Relaunch the GPU work the watcher-deadlock skipped. GPU is free; run
# sequentially in one process (NO pgrep watcher -> no self-match deadlock).
cd /home/yucheng/Desktop/ESG
{
  echo "=== relaunch_gpu start: $(date) ==="
  echo "--- train_newbb (mmBERT + BGE-M3 smoke) ---"
  bash train_newbb.sh
  echo "--- knn_supcon contrastive finetune ---"
  python -u knn_supcon_finetune.py
  echo "=== relaunch_gpu done: $(date) ==="
} > agent_cache/relaunch_gpu.log 2>&1
