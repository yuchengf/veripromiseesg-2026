"""
Threshold search for T3 "Not Clear" class.

Finds optimal tau such that: predict "Not Clear" if p(Not Clear) >= tau
instead of always using argmax. Evaluated on OOF predictions from the
rob_aug + lert_aug kfold models (trained on augmented 969-sample data).

Only evaluates on original 800 samples (aug samples have no ground truth value).
"""

import ast
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
import esg_main
# torch.load uses pickle which looks for Config in __main__; register it here
sys.modules["__main__"].Config = esg_main.Config
from esg_main import (
    extract_oof_probs, load_dataframes,
    IDX2LABEL, LABEL2IDX, TASK_WEIGHTS, NUM_LABELS,
)

DATA_DIR = "2026-esg-classification-challenge"
ROB_AUG_DIR = "runs/A1_roberta_dc_kfold_llmaug"
LERT_AUG_DIR = "runs/A1_lert_dc_kfold_llmaug"
BEST_SUBMISSION = "submissions/s119_v16_knn_k5_a055_t2.csv"
OUT_DIR = Path("submissions")

NOT_CLEAR_IDX = LABEL2IDX["t3"]["Not Clear"]  # 1
NA_IDX        = LABEL2IDX["t3"]["N/A"]        # 3


def main():
    # Load augmented train (969 samples) — must match what the aug models were trained on
    train_aug, test_df = load_dataframes(DATA_DIR, use_augmented=True)
    # Load original 800 samples for evaluation ground truth
    train_orig, _ = load_dataframes(DATA_DIR, use_augmented=False)
    n_orig = len(train_orig)  # 800

    print(f"Train (aug): {len(train_aug)}  |  Train (orig): {n_orig}")

    print("\nExtracting OOF probs from rob_aug kfold (969 samples)...")
    oof_rob, _ = extract_oof_probs(ROB_AUG_DIR, train_aug, n_splits=5, seed=42)
    print("Extracting OOF probs from lert_aug kfold (969 samples)...")
    oof_lert, _ = extract_oof_probs(LERT_AUG_DIR, train_aug, n_splits=5, seed=42)

    # Average ensemble probs — (969, 7): T2(3) + T3(4)
    oof_avg = (oof_rob + oof_lert) / 2.0

    # Only evaluate on first 800 rows (original samples with real labels)
    t3_probs = oof_avg[:n_orig, 3:]  # (800, 4)

    # Ground truth
    y_true_t3 = train_orig["evidence_quality"].tolist()
    y_true_t1 = train_orig["promise_status"].tolist()
    y_true_t2 = train_orig["evidence_status"].tolist()

    # N/A mask: cascade forces T3=N/A when T1=No or T2=No
    na_mask = np.array([
        y_true_t1[i] == "No" or y_true_t2[i] == "No"
        for i in range(n_orig)
    ])

    # Baseline argmax
    baseline_preds = np.argmax(t3_probs, axis=1)
    baseline_t3 = [IDX2LABEL["t3"][int(p)] for p in baseline_preds]
    nc_f1_base = f1_score(y_true_t3, baseline_t3,
                           labels=["Not Clear"], average="macro", zero_division=0)
    overall_f1_base = f1_score(y_true_t3, baseline_t3,
                                labels=sorted(set(y_true_t3)), average="macro", zero_division=0)
    print(f"\nBaseline argmax — Not Clear F1: {nc_f1_base:.4f}  |  T3 macro F1: {overall_f1_base:.4f}")

    # Search tau
    print("\nThreshold search (tau = 0.05 to 0.60):")
    results = []
    for tau_int in range(5, 61, 1):
        tau = tau_int / 100.0
        preds = np.argmax(t3_probs, axis=1).copy()
        nc_prob = t3_probs[:, NOT_CLEAR_IDX]
        override = (nc_prob >= tau) & (~na_mask)
        preds[override] = NOT_CLEAR_IDX
        pred_labels = [IDX2LABEL["t3"][int(p)] for p in preds]
        nc_f1 = f1_score(y_true_t3, pred_labels,
                          labels=["Not Clear"], average="macro", zero_division=0)
        t3_macro = f1_score(y_true_t3, pred_labels,
                             labels=sorted(set(y_true_t3)), average="macro", zero_division=0)
        results.append((tau, nc_f1, t3_macro))

    results_by_nc = sorted(results, key=lambda x: -x[1])
    print(f"{'tau':>6}  {'NC F1':>8}  {'T3 macro':>9}")
    for tau, nc_f1, t3_macro in results_by_nc[:15]:
        marker = " ←" if tau == results_by_nc[0][0] else ""
        print(f"{tau:>6.2f}  {nc_f1:>8.4f}  {t3_macro:>9.4f}{marker}")

    best_tau, best_nc_f1, best_t3 = results_by_nc[0]
    print(f"\nBest tau={best_tau:.2f}: Not Clear F1={best_nc_f1:.4f}, T3 macro={best_t3:.4f}")
    print(f"Improvement: Not Clear +{best_nc_f1 - nc_f1_base:+.4f}, T3 macro {best_t3 - overall_f1_base:+.4f}")

    # Apply to test submission
    print(f"\nApplying tau={best_tau:.2f} to: {BEST_SUBMISSION}")
    sub_df = pd.read_csv(BEST_SUBMISSION)

    # We need test T3 probs — re-run gen_submissions is complex.
    # Instead: report the best tau so user can add it to gen_submissions COMBOS.
    print(f"\n{'='*60}")
    print(f"ACTION: Add threshold tau={best_tau:.2f} to gen_submissions for T3.")
    print(f"Expected Not Clear F1 improvement (OOF estimate): +{best_nc_f1 - nc_f1_base:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
