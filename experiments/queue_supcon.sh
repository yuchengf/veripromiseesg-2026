#!/bin/bash
# Launch knn_supcon_finetune.py after the overnight chain
# (overnight.sh -> train_newbb.sh) is fully done.
# "Done" = no pipeline process seen for 5 consecutive minutes
# (covers the 120s gap between overnight.sh end and train_newbb.sh start).
cd /home/yucheng/Desktop/ESG
PAT='overnight[.]sh|train_newbb[.]sh|esg_main[.]py|misleading_hunter[.]py'
quiet=0
while [ "$quiet" -lt 5 ]; do
  sleep 60
  if pgrep -f "$PAT" > /dev/null; then quiet=0; else quiet=$((quiet+1)); fi
done
echo "=== chain quiet, starting supcon: $(date) ===" >> agent_cache/overnight.log
python -u knn_supcon_finetune.py > agent_cache/knn_supcon.log 2>&1 \
  && echo "knn_supcon OK $(date)" >> agent_cache/overnight.log \
  || echo "knn_supcon FAILED $(date)" >> agent_cache/overnight.log
