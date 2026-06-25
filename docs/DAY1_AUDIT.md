# Day 1 Audit — Final Stage 2026 ESG Classification

## Metric
**Weighted Macro-F1** across 4 tasks:
```
score = 0.20 × MacroF1(T1) + 0.30 × MacroF1(T2) + 0.35 × MacroF1(T3) + 0.15 × MacroF1(T4)
```
- Macro F1: each class equally weighted regardless of support
- `zero_division=0` in sklearn
- Only classes present in solution split are scored (absent classes don't penalize)

## Tasks
| Task | Column | Classes | Weight |
|------|--------|---------|--------|
| T1 | promise_status | Yes, No | 20% |
| T2 | evidence_status | Yes, No, N/A | 30% |
| T3 | evidence_quality | Clear, Not Clear, Misleading, N/A | **35%** |
| T4 | verification_timeline | already, within_2_years, between_2_and_5_years, more_than_5_years, N/A | 15% |

**T3 is the bottleneck** — highest weight, most classes, worst F1.

## Data Files (final_data/)
| File | Rows | Has Labels | Purpose |
|------|------|-----------|---------|
| train_data.csv | 1601 | Yes | Training |
| valid_data.csv | 399 | No | Kaggle eval (Public/Private split) |
| full_train_data.csv | 2000 | Yes | AIdea retrain (train+valid) |
| vpesg4k_test_2000.csv | 2000 | No | AIdea final submission |
| valid_solution_data.csv | 399 | Yes + Usage | Scoring reference (has Public/Private split) |

## ID Ranges
- Train + Valid: 10001–12000 (2000 unique)
- Test (AIdea): 12001–14000 (2000 unique, NEW data)

## Class Imbalance (train_data, 1601 rows)
| Task | Rarest Class | Count | % |
|------|-------------|-------|---|
| T1 | No | ~300 | 19% |
| T2 | N/A | ~298 | 19% |
| T3 | Misleading | ~2 | **0.1%** |
| T3 | Not Clear | ~180 | 11% |
| T4 | within_2_years | 29 | **1.8%** |

**Misleading (T3) is effectively absent** — only ~2 samples. within_2_years (T4) is also very rare.

## Kaggle Submission Format
- 4 columns: id, promise_status, verification_timeline, evidence_status, evidence_quality
- **N/A must be "-1"** (Kaggle pd.read_csv bug converts "N/A" to NaN)
- AIdea uses "N/A" as-is

## Structural Findings
1. valid_data (399) = exact same IDs as old 3rd_data test
2. T4 label changed: `longer_than_5_years` → `more_than_5_years`
3. Dependency: T1=No → T2=N/A, T3=N/A, T4=N/A
4. Solution has Public/Private split — some rows only count for private score
