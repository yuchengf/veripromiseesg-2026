"""
LGBM Stacking: Neural model OOF predictions + company/industry features.
Teacher's insight: company_id + industry category helps significantly.

Usage:
  python lgbm_stack.py --data_dir 3rd_data [--kfold_dir runs/C1_roberta_dc_kfold_t3nc3]
  python lgbm_stack.py --data_dir 3rd_data --gen_submission --out submissions/s390_lgbm_stack.csv
"""
import argparse
import ast
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ── Label definitions (matching esg_main.py) ─────────────────────────────────
LABELS = {
    "t1": ["Yes", "No"],
    "t2": ["Yes", "No", "N/A"],
    "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "t4": ["already", "within_2_years", "between_2_and_5_years",
           "longer_than_5_years", "N/A"],
}
IDX2LABEL = {task: {i: l for i, l in enumerate(labels)}
             for task, labels in LABELS.items()}
LABEL2IDX = {task: {l: i for i, l in enumerate(labels)}
             for task, labels in LABELS.items()}
WEIGHTS = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}

# ── Per-task LGBM hyperparams ─────────────────────────────────────────────────
LGBM_PARAMS = {
    "t3": dict(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        class_weight="balanced", random_state=42, n_jobs=4,
        verbose=-1,
    ),
    "default": dict(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=4,
        verbose=-1,
    ),
}


# ── Feature engineering ───────────────────────────────────────────────────────

def _extract_hand_features_np(text: str) -> np.ndarray:
    """Extract 13-dim hand features (numpy version)."""
    return _get_hand_fn()(text).numpy()


def _get_hand_fn():
    """Lazy-load extract_hand_features from esg_main (cached after first call)."""
    if not hasattr(_get_hand_fn, "_fn"):
        sys.path.insert(0, str(Path(__file__).parent))
        from esg_main import extract_hand_features
        _get_hand_fn._fn = extract_hand_features
    return _get_hand_fn._fn


def build_features(df: pd.DataFrame, company_enc: LabelEncoder,
                   use_hand: bool = True) -> np.ndarray:
    """Build non-neural feature matrix for one split (train or test)."""
    feats = []

    # Company label-encoded ID
    company_ids = company_enc.transform(df["company"].values)
    feats.append(company_ids.reshape(-1, 1))

    # Ticker industry proxy: first digit * 1000 floor
    ticker_prefix = (df["ticker"].astype(int) // 1000).values
    feats.append(ticker_prefix.reshape(-1, 1))

    # Numeric ticker
    feats.append(df["ticker"].astype(float).values.reshape(-1, 1))

    # Page number (log-scaled)
    page = np.log1p(df["page_number"].fillna(0).astype(float).values)
    feats.append(page.reshape(-1, 1))

    # ESG type one-hot (E/S/G → 3 dims); fill zeros if not available (test set)
    if "esg_type" in df.columns:
        esg_e = df["esg_type"].str.contains("E", na=False).astype(float).values
        esg_s = df["esg_type"].str.contains("S", na=False).astype(float).values
        esg_g = df["esg_type"].str.contains("G", na=False).astype(float).values
        feats.append(np.stack([esg_e, esg_s, esg_g], axis=1))
    else:
        feats.append(np.zeros((len(df), 3), dtype=float))

    # Text length
    text_len = df["data"].str.len().values.astype(float) / 500.0
    feats.append(text_len.reshape(-1, 1))

    # Hand-crafted linguistic features (13 dims)
    if use_hand:
        hf = np.stack([_extract_hand_features_np(t) for t in df["data"].tolist()])
        feats.append(hf)

    return np.hstack(feats).astype(np.float32)


def build_neural_features(probs: dict[str, list]) -> np.ndarray:
    """Flatten all task probabilities into a feature vector."""
    parts = []
    for task in ["t1", "t2", "t3", "t4"]:
        arr = np.array(probs[task], dtype=np.float32)
        parts.append(arr)
    return np.hstack(parts)  # (N, 2+3+4+5=14)


def weighted_f1(y_true, y_pred_dict: dict[str, np.ndarray]) -> float:
    total = 0.0
    for task, w in WEIGHTS.items():
        t_true = y_pred_dict[f"{task}_true"]
        t_pred = y_pred_dict[f"{task}_pred"]
        total += w * f1_score(t_true, t_pred, average="macro")
    return total


# ── OOF extraction via esg_main ──────────────────────────────────────────────

def _ensure_esg_main():
    """Import esg_main and inject its classes into __main__ so torch.load can unpickle."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import esg_main as em
    # Checkpoints were saved when esg_main ran as __main__, so Config etc.
    # are pickled as __main__.Config.  Inject them so unpickling works.
    main_mod = sys.modules["__main__"]
    for name in ["Config", "ApproachA1", "ESGDataset"]:
        if hasattr(em, name) and not hasattr(main_mod, name):
            setattr(main_mod, name, getattr(em, name))
    return em


def get_oof_probs(kfold_dirs: list[str], train_df: pd.DataFrame,
                  seed: int = 42) -> dict[str, list]:
    """Average OOF probs from multiple kfold_dirs (multi-seed ensemble)."""
    em = _ensure_esg_main()

    all_probs_list = []
    for kd in kfold_dirs:
        print(f"  Extracting OOF from {kd} ...")
        probs = em.extract_oof_teacher_probs(kd, train_df, n_splits=5, seed=seed)
        all_probs_list.append(probs)

    # Average across seeds
    avg: dict[str, list] = {}
    for task in LABELS:
        n = len(all_probs_list[0][task])
        avg[task] = [
            [sum(ap[task][i][c] for ap in all_probs_list) / len(all_probs_list)
             for c in range(len(LABELS[task]))]
            for i in range(n)
        ]
    return avg


def get_test_probs(kfold_dirs: list[str], test_df: pd.DataFrame,
                   seed: int = 42) -> dict[str, list]:
    """Average test probs from all folds × all kfold_dirs."""
    em = _ensure_esg_main()

    all_fold_probs = []
    for kd in kfold_dirs:
        fold_idx = 1
        while True:
            ckpt = Path(kd) / f"fold{fold_idx}" / "best.pt"
            if not ckpt.exists():
                break
            result = em._predict_checkpoint(str(ckpt), test_df)
            if result is not None:
                all_fold_probs.append(result[1])
                print(f"  {kd}/fold{fold_idx} OK")
            fold_idx += 1

    avg: dict[str, list] = {}
    for task in LABELS:
        n = len(all_fold_probs[0][task])
        avg[task] = [
            [sum(fp[task][i][c] for fp in all_fold_probs) / len(all_fold_probs)
             for c in range(len(LABELS[task]))]
            for i in range(n)
        ]
    return avg


# ── Per-task LGBM training ────────────────────────────────────────────────────

def train_task_lgbm(
    X: np.ndarray, y: np.ndarray, task: str, cv_folds: int = 5
) -> tuple[list[LGBMClassifier], float]:
    """Train per-task LGBM with CV. Returns list of fold models and OOF macro-F1."""
    params = LGBM_PARAMS.get(task, LGBM_PARAMS["default"])
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_preds = np.full(len(y), -1, dtype=int)
    models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        clf = LGBMClassifier(**params)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_preds[va_idx] = clf.predict(X[va_idx])
        models.append(clf)
        print(f"    [{task}] fold{fold+1} F1={f1_score(y[va_idx], oof_preds[va_idx], average='macro'):.4f}")

    oof_f1 = f1_score(y, oof_preds, average="macro")
    print(f"  [{task}] OOF macro-F1: {oof_f1:.4f}")
    return models, oof_f1


def ensemble_predict(models: list[LGBMClassifier], X: np.ndarray, n_classes: int) -> np.ndarray:
    """Average probability predictions from all fold models."""
    probs = np.zeros((len(X), n_classes), dtype=np.float32)
    for clf in models:
        p = clf.predict_proba(X)
        # Align class order
        for ci, cls_idx in enumerate(clf.classes_):
            probs[:, cls_idx] += p[:, ci]
    return probs / len(models)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="3rd_data")
    ap.add_argument("--kfold_dirs", nargs="+",
                    default=["runs/C1_roberta_dc_kfold_t3nc3",
                             "runs/C1_roberta_dc_kfold_t3nc3_s1",
                             "runs/C1_roberta_dc_kfold_t3nc3_s2"])
    ap.add_argument("--extra_kfold_groups", nargs="*", default=[],
                    help="Additional kfold dir groups (comma-sep per group) for diversity features")
    ap.add_argument("--seed_for_oof", type=int, default=42,
                    help="Seed used by neural kfold (must match training seed)")
    ap.add_argument("--ncbias", type=float, default=0.3,
                    help="T3 NC logit bias applied after LGBM")
    ap.add_argument("--no_hand", action="store_true", help="Disable hand features")
    ap.add_argument("--gen_submission", action="store_true")
    ap.add_argument("--out", default="submissions/s420_lgbm_v2.csv")
    ap.add_argument("--save_models", default="runs/lgbm_stack_models.pkl")
    ap.add_argument("--local_score", action="store_true",
                    help="Score against solution_data.csv if available")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")

    # Parse labels
    train_df["label"] = train_df["label"].apply(ast.literal_eval)
    for i, t in enumerate(["t1", "t2", "t3", "t4"]):
        train_df[t] = train_df["label"].apply(lambda x, i=i: x[i])

    # Company label encoder — fit on ALL companies (train+test)
    all_companies = sorted(set(train_df["company"].unique()) |
                           set(test_df["company"].unique()))
    company_enc = LabelEncoder()
    company_enc.fit(all_companies)

    # Non-neural features (now includes hand features + esg_type + text_length)
    use_hand = not args.no_hand
    print(f"Building non-neural features (hand={use_hand})...")
    X_meta = build_features(train_df, company_enc, use_hand=use_hand)
    X_test_meta = build_features(test_df, company_enc, use_hand=use_hand)

    # Check which kfold_dirs actually exist
    kfold_dirs = [d for d in args.kfold_dirs if Path(d).exists()]
    if not kfold_dirs:
        print(f"ERROR: None of the kfold_dirs exist: {args.kfold_dirs}")
        return
    print(f"Using kfold_dirs: {kfold_dirs}")

    # Extract OOF probs (one per kfold_dir, each has its own seed)
    # C1 seed=42 uses kfold seed=42; s1 uses 123; s2 uses 456
    dir_seeds = {
        "runs/C1_roberta_dc_kfold_t3nc3":    42,
        "runs/C1_roberta_dc_kfold_t3nc3_s1": 123,
        "runs/C1_roberta_dc_kfold_t3nc3_s2": 456,
        "runs/G1_macbert_dc_kfold_t3nc3":    42,
        "runs/G1_macbert_dc_kfold_t3nc3_s1": 123,
        "runs/G1_macbert_dc_kfold_t3nc3_s2": 456,
        "runs/G2_lert_dc_kfold_t3nc3":       42,
        "runs/G2_lert_dc_kfold_t3nc3_s1":    123,
        "runs/G2_lert_dc_kfold_t3nc3_s2":    456,
        "runs/G3_roberta_dc_kfold_t3nc3_scl":    42,
        "runs/G3_roberta_dc_kfold_t3nc3_scl_s1": 123,
        "runs/G3_roberta_dc_kfold_t3nc3_scl_s2": 456,
        "runs/G5_roberta_dc_kfold_t3nc3_cemb64":    42,
        "runs/G5_roberta_dc_kfold_t3nc3_cemb64_s1": 123,
        "runs/G5_roberta_dc_kfold_t3nc3_cemb64_s2": 456,
    }
    em = _ensure_esg_main()
    print("\nExtracting OOF predictions (one by one for correct seed)...")
    all_oof_list = []
    for kd in kfold_dirs:
        seed = dir_seeds.get(kd, 42)
        print(f"  OOF from {kd} (seed={seed})")
        probs = em.extract_oof_teacher_probs(kd, train_df, n_splits=5, seed=seed)
        all_oof_list.append(probs)

    # Average OOF probs across seeds
    oof_probs: dict[str, list] = {}
    for task in LABELS:
        n = len(all_oof_list[0][task])
        oof_probs[task] = [
            [sum(ap[task][i][c] for ap in all_oof_list) / len(all_oof_list)
             for c in range(len(LABELS[task]))]
            for i in range(n)
        ]

    X_neural = build_neural_features(oof_probs)

    # Extra backbone OOF features (diversity signal from G1/G2/G3/G5 etc.)
    extra_neural_parts = []
    for group_str in args.extra_kfold_groups:
        dirs = [d.strip() for d in group_str.split(",") if d.strip()]
        # Determine seed per directory
        existing = [d for d in dirs if Path(d).exists()]
        if not existing:
            print(f"  [WARN] no dirs found in group: {group_str}")
            continue
        print(f"\n  Extra OOF from: {existing}")
        extra_oof_list = []
        for kd in existing:
            seed = dir_seeds.get(kd, 42)
            probs = em.extract_oof_teacher_probs(kd, train_df, n_splits=5, seed=seed)
            extra_oof_list.append(probs)
        # Average across seeds in this group
        avg_extra: dict[str, list] = {}
        for task in LABELS:
            n = len(extra_oof_list[0][task])
            avg_extra[task] = [
                [sum(ap[task][i][c] for ap in extra_oof_list) / len(extra_oof_list)
                 for c in range(len(LABELS[task]))]
                for i in range(n)
            ]
        extra_neural_parts.append(build_neural_features(avg_extra))

    if extra_neural_parts:
        X_neural = np.hstack([X_neural] + extra_neural_parts)
        print(f"  Neural features: {X_neural.shape[1]} dims (primary + {len(extra_neural_parts)} extra groups)")

    X_train = np.hstack([X_neural, X_meta])

    print(f"\nTrain shape: {X_train.shape}  (neural={X_neural.shape[1]}, meta={X_meta.shape[1]})")

    # Train per-task LGBM
    task_models = {}
    for task in ["t1", "t2", "t3", "t4"]:
        print(f"\nTraining LGBM for {task.upper()}...")
        y = train_df[task].map(LABEL2IDX[task]).values
        models, oof_f1 = train_task_lgbm(X_train, y, task)
        task_models[task] = models

    # OOF overall score (use same X_train which already has all features)
    print("\nComputing full OOF weighted-F1...")
    oof_result = {}
    for task in ["t1", "t2", "t3", "t4"]:
        y_true = train_df[task].map(LABEL2IDX[task]).values
        probs_all = ensemble_predict(task_models[task], X_train, len(LABELS[task]))
        # NC bias for t3
        if task == "t3" and args.ncbias != 0.0:
            nc_idx = LABEL2IDX["t3"]["Not Clear"]
            probs_all[:, nc_idx] *= np.exp(args.ncbias)
            probs_all /= probs_all.sum(axis=1, keepdims=True)
        y_pred = probs_all.argmax(axis=1)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"  {task.upper()}: {f1:.4f}")
        oof_result[f"{task}_true"] = y_true
        oof_result[f"{task}_pred"] = y_pred
        oof_result[f"{task}_f1"] = f1

    wf1 = sum(WEIGHTS[t] * oof_result[f"{t}_f1"] for t in WEIGHTS)
    print(f"\nOOF Weighted macro-F1: {wf1:.5f}")
    print("(Note: OOF F1 for LGBM is computed IN-SAMPLE for non-stacked features,")
    print(" so it may be optimistic. The real gain comes from company/industry features.)")

    # Save models
    Path(args.save_models).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_models, "wb") as f:
        pickle.dump({"models": task_models, "company_enc": company_enc,
                     "ncbias": args.ncbias}, f)
    print(f"\nModels saved → {args.save_models}")

    if args.gen_submission:
        print("\nGenerating test predictions...")
        print("Extracting test probs from primary kfold models...")
        test_probs = get_test_probs(kfold_dirs, test_df)
        X_test_neural = build_neural_features(test_probs)

        # Extra backbone test probs
        extra_test_parts = []
        for group_str in args.extra_kfold_groups:
            dirs = [d.strip() for d in group_str.split(",") if d.strip()]
            existing = [d for d in dirs if Path(d).exists()]
            if not existing:
                continue
            print(f"  Extra test probs from: {existing}")
            extra_test_probs = get_test_probs(existing, test_df)
            extra_test_parts.append(build_neural_features(extra_test_probs))

        if extra_test_parts:
            X_test_neural = np.hstack([X_test_neural] + extra_test_parts)

        X_test = np.hstack([X_test_neural, X_test_meta])

        preds = {}
        for task in ["t1", "t2", "t3", "t4"]:
            probs = ensemble_predict(task_models[task], X_test, len(LABELS[task]))
            if task == "t3" and args.ncbias != 0.0:
                nc_idx = LABEL2IDX["t3"]["Not Clear"]
                probs[:, nc_idx] *= np.exp(args.ncbias)
                probs /= probs.sum(axis=1, keepdims=True)
            preds[task] = [IDX2LABEL[task][i] for i in probs.argmax(axis=1)]

        # Apply NA cascade rules
        for i in range(len(test_df)):
            if preds["t1"][i] == "No":
                preds["t2"][i] = "N/A"
                preds["t3"][i] = "N/A"
                preds["t4"][i] = "N/A"
            elif preds["t2"][i] == "No":
                preds["t3"][i] = "N/A"

        # Build submission (use enumerate for positional indexing into preds lists)
        rows = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            label = [preds["t1"][i], preds["t2"][i], preds["t3"][i], preds["t4"][i]]
            rows.append({"id": row["id"], "label": str(label)})
        sub = pd.DataFrame(rows)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(args.out, index=False)
        print(f"Submission saved → {args.out}")

        if args.local_score:
            sol_path = Path(args.data_dir) / "solution_data.csv"
            if sol_path.exists():
                sol = pd.read_csv(sol_path)
                pub = sol[sol["Usage"] == "Public"].copy()
                pub["label"] = pub["label"].apply(ast.literal_eval)
                for i, t in enumerate(["T1", "T2", "T3", "T4"]):
                    pub[t] = pub["label"].apply(lambda x, i=i: x[i])
                sub2 = pd.read_csv(args.out)
                sub2["label"] = sub2["label"].apply(ast.literal_eval)
                for i, t in enumerate(["T1", "T2", "T3", "T4"]):
                    sub2[t] = sub2["label"].apply(lambda x, i=i: x[i])
                m = pub.merge(sub2[["id", "T1", "T2", "T3", "T4"]], on="id",
                              suffixes=("_true", "_pred"))
                total = sum(
                    WEIGHTS[t.lower()] *
                    f1_score(m[f"{t}_true"], m[f"{t}_pred"], average="macro")
                    for t in ["T1", "T2", "T3", "T4"]
                )
                print(f"\nPublic local score: {total:.5f}")
            else:
                print("(solution_data.csv not found — skip local scoring)")


if __name__ == "__main__":
    main()
