#!/bin/bash
# Overnight training script — Phase 1 (LGBM) + Phase 2 (Attention Pool + Feature Prepend)
# Run: nohup bash overnight_train.sh > /tmp/overnight_log.txt 2>&1 &

set -e
cd /home/yucheng/Desktop/ESG

echo "=== $(date) === Starting overnight training ==="

# ── Phase 1: LGBM Stacking ──────────────────────────────────────────────────
echo ""
echo "=== Phase 1A: LGBM v2 (C1 only + hand features) ==="
python -u lgbm_stack.py \
  --data_dir 3rd_data \
  --gen_submission --local_score \
  --out submissions/s420_lgbm_v2.csv \
  --ncbias 0.3

echo ""
echo "=== Phase 1B: LGBM v3 (C1 + G1 backbone OOF) ==="
python -u lgbm_stack.py \
  --data_dir 3rd_data \
  --extra_kfold_groups \
    "runs/G1_macbert_dc_kfold_t3nc3,runs/G1_macbert_dc_kfold_t3nc3_s1,runs/G1_macbert_dc_kfold_t3nc3_s2" \
  --gen_submission --local_score \
  --out submissions/s421_lgbm_c1g1.csv \
  --ncbias 0.3

echo ""
echo "=== Phase 1C: LGBM v4 (C1 + G1 + G2 + G5 backbone OOF) ==="
python -u lgbm_stack.py \
  --data_dir 3rd_data \
  --extra_kfold_groups \
    "runs/G1_macbert_dc_kfold_t3nc3,runs/G1_macbert_dc_kfold_t3nc3_s1,runs/G1_macbert_dc_kfold_t3nc3_s2" \
    "runs/G2_lert_dc_kfold_t3nc3,runs/G2_lert_dc_kfold_t3nc3_s1,runs/G2_lert_dc_kfold_t3nc3_s2" \
    "runs/G5_roberta_dc_kfold_t3nc3_cemb64,runs/G5_roberta_dc_kfold_t3nc3_cemb64_s1,runs/G5_roberta_dc_kfold_t3nc3_cemb64_s2" \
  --gen_submission --local_score \
  --out submissions/s422_lgbm_c1g1g2g5.csv \
  --ncbias 0.3

echo ""
echo "=== Phase 1D: LGBM no NC bias (let LGBM learn) ==="
python -u lgbm_stack.py \
  --data_dir 3rd_data \
  --extra_kfold_groups \
    "runs/G1_macbert_dc_kfold_t3nc3,runs/G1_macbert_dc_kfold_t3nc3_s1,runs/G1_macbert_dc_kfold_t3nc3_s2" \
    "runs/G2_lert_dc_kfold_t3nc3,runs/G2_lert_dc_kfold_t3nc3_s1,runs/G2_lert_dc_kfold_t3nc3_s2" \
    "runs/G5_roberta_dc_kfold_t3nc3_cemb64,runs/G5_roberta_dc_kfold_t3nc3_cemb64_s1,runs/G5_roberta_dc_kfold_t3nc3_cemb64_s2" \
  --gen_submission --local_score \
  --out submissions/s423_lgbm_c1g1g2g5_nb0.csv \
  --ncbias 0.0

# ── Phase 2: Neural model training (GPU) ────────────────────────────────────
echo ""
echo "=== Phase 2: Training H1/H2 neural models ==="
python -u esg_main.py --mode run_v4 --data_dir 3rd_data --skip_existing

echo ""
echo "=== Phase 2: Generating submissions ==="
python -u esg_main.py --mode gen_submissions --data_dir 3rd_data --skip_existing

echo ""
echo "=== $(date) === Overnight training complete ==="
