"""
Company-level NC calibration: use per-company NC rate from training as Bayesian prior.
Applied as a post-processing step on neural model probability outputs.

For T3 Not Clear: P_adj(NC) ∝ P_neural(NC) * (co_nc_rate / global_nc_rate)^beta

Usage:
  python company_nc_calib.py --data_dir 3rd_data --beta 0.5 --local_score
"""
import argparse
import ast
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

# ── Label definitions ─────────────────────────────────────────────────────────
LABELS = {
    "t1": ["Yes", "No"],
    "t2": ["Yes", "No", "N/A"],
    "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "t4": ["already", "within_2_years", "between_2_and_5_years",
           "longer_than_5_years", "N/A"],
}
IDX2LABEL = {t: {i: l for i, l in enumerate(v)} for t, v in LABELS.items()}
LABEL2IDX = {t: {l: i for i, l in enumerate(v)} for t, v in LABELS.items()}
WEIGHTS = {"T1": 0.20, "T2": 0.30, "T3": 0.35, "T4": 0.15}
NC_IDX = LABEL2IDX["t3"]["Not Clear"]


def get_oof_probs_from_kfold(kfold_dirs, train_df, dir_seeds):
    """Extract averaged OOF probs from C1 kfold models."""
    import sys
    sys.path.insert(0, ".")

    # Inject esg_main classes so torch.load can unpickle Config
    import esg_main as em
    import sys as _sys
    main_mod = _sys.modules["__main__"]
    for name in ["Config", "ApproachA1", "ESGDataset"]:
        if hasattr(em, name) and not hasattr(main_mod, name):
            setattr(main_mod, name, getattr(em, name))

    all_oof = []
    for kd in kfold_dirs:
        if not Path(kd).exists():
            continue
        seed = dir_seeds.get(kd, 42)
        print(f"  OOF from {kd} (seed={seed}) ...")
        probs = em.extract_oof_teacher_probs(kd, train_df, n_splits=5, seed=seed)
        all_oof.append(probs)

    # Average across seeds
    avg = {}
    for task in LABELS:
        n = len(all_oof[0][task])
        avg[task] = [
            [sum(ap[task][i][c] for ap in all_oof) / len(all_oof)
             for c in range(len(LABELS[task]))]
            for i in range(n)
        ]
    return avg


def get_test_probs_from_kfold(kfold_dirs):
    """Extract averaged test probs from all folds of all kfold_dirs."""
    import sys
    sys.path.insert(0, ".")
    import esg_main as em
    import sys as _sys
    main_mod = _sys.modules["__main__"]
    for name in ["Config", "ApproachA1", "ESGDataset"]:
        if hasattr(em, name) and not hasattr(main_mod, name):
            setattr(main_mod, name, getattr(em, name))

    all_fold_probs = []
    for kd in kfold_dirs:
        if not Path(kd).exists():
            continue
        fold_idx = 1
        while True:
            ckpt = Path(kd) / f"fold{fold_idx}" / "best.pt"
            if not ckpt.exists():
                break
            result = em._predict_checkpoint(str(ckpt), test_df_global)
            if result is not None:
                all_fold_probs.append(result[1])
            fold_idx += 1

    avg = {}
    for task in LABELS:
        n = len(all_fold_probs[0][task])
        avg[task] = [
            [sum(fp[task][i][c] for fp in all_fold_probs) / len(all_fold_probs)
             for c in range(len(LABELS[task]))]
            for i in range(n)
        ]
    return avg


def apply_company_nc_calib(
    probs_t3: list[list[float]],
    companies: list[str],
    co_nc_rate: dict,
    global_nc_rate: float,
    beta: float = 0.5,
    base_nc_bias: float = 0.0,
) -> list[list[float]]:
    """Apply per-company NC calibration to T3 probability distributions.

    For each sample:
      bias = beta * log((co_nc_rate + eps) / (global_nc_rate + eps))
      P_adj(NC) *= exp(bias)
      then renormalize.
    Also applies base_nc_bias (global NC logit bias) additively.
    """
    eps = 1e-3
    result = []
    for i, (p, co) in enumerate(zip(probs_t3, companies)):
        p2 = list(p)
        co_rate = co_nc_rate.get(co, global_nc_rate)
        company_bias = beta * math.log((co_rate + eps) / (global_nc_rate + eps))
        total_nc_bias = base_nc_bias + company_bias
        p2[NC_IDX] *= math.exp(total_nc_bias)
        total = sum(p2)
        result.append([x / total for x in p2])
    return result


def score_local(sub_path: str, sol_path: str) -> float:
    sol = pd.read_csv(sol_path)
    pub = sol[sol["Usage"] == "Public"].copy()
    pub["label"] = pub["label"].apply(ast.literal_eval)
    for i, t in enumerate(["T1", "T2", "T3", "T4"]):
        pub[t] = pub["label"].apply(lambda x, i=i: x[i])

    sub = pd.read_csv(sub_path)
    sub["label"] = sub["label"].apply(ast.literal_eval)
    for i, t in enumerate(["T1", "T2", "T3", "T4"]):
        sub[t] = sub["label"].apply(lambda x, i=i: x[i])

    m = pub.merge(sub[["id", "T1", "T2", "T3", "T4"]], on="id",
                  suffixes=("_true", "_pred"))
    total = sum(
        WEIGHTS[t] * f1_score(m[f"{t}_true"], m[f"{t}_pred"], average="macro")
        for t in ["T1", "T2", "T3", "T4"]
    )
    nc_f1 = f1_score(m["T3_true"], m["T3_pred"], labels=["Not Clear"], average="macro")
    return total, nc_f1


# Global for get_test_probs_from_kfold
test_df_global = None


def main():
    global test_df_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="3rd_data")
    ap.add_argument("--kfold_dirs", nargs="+",
                    default=["runs/C1_roberta_dc_kfold_t3nc3",
                             "runs/C1_roberta_dc_kfold_t3nc3_s1",
                             "runs/C1_roberta_dc_kfold_t3nc3_s2"])
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Strength of company NC calibration (0=no effect)")
    ap.add_argument("--base_nc_bias", type=float, default=0.3,
                    help="Global NC logit bias (same as t3_nc_logit_bias in gen_submissions)")
    ap.add_argument("--out_prefix", default="submissions/s391_co_nc")
    ap.add_argument("--local_score", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep beta and base_nc_bias values")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train_data.csv")
    test_df_global = pd.read_csv(data_dir / "test_data.csv")
    test_df = test_df_global

    # Parse training labels
    train_df["label"] = train_df["label"].apply(ast.literal_eval)
    for i, t in enumerate(["t1", "t2", "t3", "t4"]):
        train_df[t] = train_df["label"].apply(lambda x, i=i: x[i])

    # Compute per-company NC rate from training
    global_nc_rate = (train_df["t3"] == "Not Clear").mean()
    co_nc_rate = train_df.groupby("company")["t3"].apply(
        lambda x: (x == "Not Clear").mean()
    ).to_dict()
    print(f"Global NC rate: {global_nc_rate:.4f}")
    print(f"Companies with >20% NC: {sum(v>0.2 for v in co_nc_rate.values())}")

    dir_seeds = {
        "runs/C1_roberta_dc_kfold_t3nc3":    42,
        "runs/C1_roberta_dc_kfold_t3nc3_s1": 123,
        "runs/C1_roberta_dc_kfold_t3nc3_s2": 456,
    }

    # Extract OOF probs (for tuning on training set)
    print("\nExtracting OOF probs...")
    oof_probs = get_oof_probs_from_kfold(args.kfold_dirs, train_df, dir_seeds)

    train_companies = train_df["company"].tolist()
    y_t3_true = train_df["t3"].map(LABEL2IDX["t3"]).values

    def eval_calib(beta, base_bias):
        t3_calib = apply_company_nc_calib(
            oof_probs["t3"], train_companies, co_nc_rate, global_nc_rate,
            beta=beta, base_nc_bias=base_bias,
        )
        t3_pred = [IDX2LABEL["t3"][np.argmax(p)] for p in t3_calib]
        return f1_score(train_df["t3"], t3_pred, average="macro")

    if args.sweep:
        print("\nSweeping beta × base_nc_bias on OOF training data...")
        best_f1, best_beta, best_bias = 0, 0, 0
        for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]:
            for base_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                f1 = eval_calib(beta, base_bias)
                if f1 > best_f1:
                    best_f1, best_beta, best_bias = f1, beta, base_bias
                    print(f"  beta={beta:.1f} bias={base_bias:.1f} → OOF T3 F1={f1:.4f} (NEW BEST)")
        print(f"\nBest: beta={best_beta}, base_bias={best_bias}, OOF T3 F1={best_f1:.4f}")
        args.beta = best_beta
        args.base_nc_bias = best_bias

    # Extract test probs
    print("\nExtracting test probs from kfold models...")
    test_probs = get_test_probs_from_kfold(args.kfold_dirs)
    test_companies = test_df["company"].tolist()

    # Apply calibration
    t3_calib = apply_company_nc_calib(
        test_probs["t3"], test_companies, co_nc_rate, global_nc_rate,
        beta=args.beta, base_nc_bias=args.base_nc_bias,
    )

    # Build submission (use argmax for T1/T2/T4, calibrated for T3)
    preds_t1 = [IDX2LABEL["t1"][int(np.argmax(p))] for p in test_probs["t1"]]
    preds_t2 = [IDX2LABEL["t2"][int(np.argmax(p))] for p in test_probs["t2"]]
    preds_t3 = [IDX2LABEL["t3"][int(np.argmax(p))] for p in t3_calib]
    preds_t4 = [IDX2LABEL["t4"][int(np.argmax(p))] for p in test_probs["t4"]]

    rows = []
    for i, row in test_df.iterrows():
        label = [preds_t1[i], preds_t2[i], preds_t3[i], preds_t4[i]]
        rows.append({"id": row["id"], "label": str(label)})
    sub = pd.DataFrame(rows)

    out = f"{args.out_prefix}_b{int(args.beta*10):02d}_nb{int(args.base_nc_bias*10):02d}.csv"
    sub.to_csv(out, index=False)
    print(f"\nSubmission saved → {out}")

    if args.local_score:
        sol_path = data_dir / "solution_data.csv"
        if sol_path.exists():
            total, nc_f1 = score_local(out, str(sol_path))
            print(f"Public local score: {total:.5f}  (NC F1={nc_f1:.4f})")


if __name__ == "__main__":
    main()
