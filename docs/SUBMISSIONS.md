# Submission Log — Final Stage Kaggle

| File | Val CV | LB Public | Δ vs prev | Hypothesis |
|------|--------|-----------|-----------|-----------|
| submission_sample | — | 0.27055 | baseline | all-Yes/Clear/already dummy |
| s500_c1r_seed42 | 0.626 | 0.61055 | +0.340 | C1R single seed (3rd_data model) |
| s506_c1_seed42 | 0.615 | 0.60208 | -0.008 | C1 baseline (3rd_data) |
| s505_c1rs_seed42 | — | 0.61007 | -0.001 | C1RS (3rd_data) |
| s504_c1m_seed42 | — | 0.60214 | — | C1M (3rd_data) |
| s503_c1s_seed42 | — | 0.59359 | — | C1S (3rd_data) |
| s510_fc1r_s1 | 0.612 | 0.61410 | +0.004 | FC1R s1 final_data (retrain helps) |
| s513_fc1m | 0.618 | 0.61147 | — | FC1M final_data |
| s511_fc1r_s2 | 0.619 | 0.60127 | — | FC1R s2 final_data |
| s450_fc1r_s1 | — | 0.61410 | 0 | FC1R s1 (gen_submissions) |
| s452_fc1r_2seed | — | 0.60987 | -0.004 | 2-seed avg worse than best single |
| s453_fc1r_3seed | — | 0.61616 | +0.002 | **3-seed ensemble helps** |
| s454_fc1_single | — | 0.60936 | — | FC1 baseline final_data |
| s458_fc1s_3seed | — | 0.59627 | — | FC1S 3-seed (worst) |
| s459_fc1m_single | — | 0.61147 | — | FC1M single |
| s461_fc1r2_fc1s | — | 0.61211 | — | cross-type 3-way |
| s462_fc1r2_fc1m | — | 0.61117 | — | cross-type 3-way |
| s463_fc1r3_fc1s3 6way | — | 0.61653 | +0.001 | **6-way cross helps** |
| s464_fc1r3_fc1m 4way | — | 0.60634 | — | FC1M hurts in ensemble |
| **s466_fc1r3_knn** | — | **0.62025** | **+0.004** | **kNN helps! T3 α=0.40** |
| **s467_6way_knn** | — | **0.62039** | **+0.000** | **6-way + kNN = BEST** |

## Key Findings
1. **Final_data retrain > 3rd_data models** (+0.004 on single seed)
2. **R-Drop+SWA (FC1R) is strongest single variant** (0.614 single, 0.616 3-seed)
3. **FC1S adds diversity in ensemble** (+0.001 from 3-seed to 6-way)
4. **FC1M hurts in ensemble** (s464 < s453)
5. **kNN definitively helps** (+0.004, s466 vs s453)
6. **Best = 6-way + kNN = 0.62039** (FC1R×3 + FC1S×3 + Qwen3 kNN)

## Next Hypotheses to Test
- [ ] kNN alpha sweep (0.20, 0.30, 0.50) — find optimal T3 alpha
- [ ] NC logit bias sweep (0.0, 0.1, 0.2, 0.5) — find optimal bias
- [ ] FC1R+FC1 cross ensemble — add vanilla baseline for diversity
- [ ] Threshold calibration on OOF — per-class optimal thresholds
