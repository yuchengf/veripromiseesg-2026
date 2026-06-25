"""Comprehensive data analysis:
  1. Real vs Aug label/text distribution
  2. Test set error analysis (using solution_data.csv + best submission)
  3. Per-class precision/recall breakdown (test + OOF)

Usage:
    conda activate AICUP
    python data_analysis.py
"""

import sys
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import esg_main as M

DATA_DIR   = Path("2026-esg-classification-challenge")
BEST_SUB   = "submissions/s153_v24_k10.csv"   # current best public
KFOLD_DIR  = "runs/A1_roberta_dc_kfold_llmaug"
N_SPLITS   = 5
SEED       = 42

TASK_COLS = {
    "t1": "promise_status",
    "t2": "evidence_status",
    "t3": "evidence_quality",
    "t4": "verification_timeline",
}
SEP = "=" * 65


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_label_col(series):
    """Parse label column like \"['Yes','Yes','Clear','already']\" → dict per task."""
    parsed = series.apply(ast.literal_eval)
    tasks = list(TASK_COLS.keys())
    return {t: parsed.apply(lambda x: x[i]).tolist() for i, t in enumerate(tasks)}


# ── Section 1: Real vs Aug distribution ──────────────────────────────────────

def section1():
    real    = pd.read_csv(DATA_DIR / "train_data.csv")
    aug_all = pd.read_csv(DATA_DIR / "train_data_augmented.csv")
    aug     = aug_all.iloc[len(real):].reset_index(drop=True)

    print(SEP)
    print("  SECTION 1: Real (800) vs LLM-aug (169) distribution")
    print(SEP)

    for t, col in TASK_COLS.items():
        labels = M.LABELS[t]
        real_cnt = real[col].value_counts().reindex(labels, fill_value=0)
        aug_cnt  = aug[col].value_counts().reindex(labels, fill_value=0)
        real_pct = real_cnt / len(real) * 100
        aug_pct  = aug_cnt  / len(aug)  * 100

        print(f"\n  [{t.upper()}] {col}")
        header = f"  {'label':<20} {'real_n':>6} {'real_%':>7} {'aug_n':>6} {'aug_%':>7} {'diff':>7}"
        print(header)
        print("  " + "-" * 57)
        for lbl in labels:
            diff = aug_pct[lbl] - real_pct[lbl]
            flag = "  ← !" if abs(diff) > 10 else ""
            print(f"  {lbl:<20} {real_cnt[lbl]:>6} {real_pct[lbl]:>6.1f}% "
                  f"{aug_cnt[lbl]:>6} {aug_pct[lbl]:>6.1f}% {diff:>+6.1f}%{flag}")

    # Text length
    real["text_len"] = real["data"].str.len()
    aug["text_len"]  = aug["data"].str.len()
    print(f"\n  Text length (chars):")
    print(f"    Real — mean:{real['text_len'].mean():.0f}  "
          f"median:{real['text_len'].median():.0f}  max:{real['text_len'].max()}")
    print(f"    Aug  — mean:{aug['text_len'].mean():.0f}  "
          f"median:{aug['text_len'].median():.0f}  max:{aug['text_len'].max()}")

    # esg_type
    print(f"\n  ESG type distribution:")
    for etype in sorted(real["esg_type"].dropna().unique()):
        r = (real["esg_type"] == etype).mean() * 100
        a = (aug["esg_type"]  == etype).mean() * 100 if etype in aug["esg_type"].values else 0.0
        print(f"    {etype}: real={r:.1f}%  aug={a:.1f}%  diff={a-r:+.1f}%")

    # T1=No samples in aug (these drive cascade errors)
    print(f"\n  T1=No samples (cascade trigger):")
    print(f"    Real: {(real['promise_status']=='No').sum()} / {len(real)}")
    print(f"    Aug : {(aug['promise_status']=='No').sum()} / {len(aug)}")


# ── Section 2: Test error analysis ───────────────────────────────────────────

def section2():
    sol      = pd.read_csv(DATA_DIR / "solution_data.csv")
    test_df  = pd.read_csv(DATA_DIR / "test_data.csv")
    sub      = pd.read_csv(BEST_SUB)

    # Merge everything on id
    sol  = sol.merge(test_df[["id", "company"]], on="id")
    sol  = sol.merge(sub.rename(columns={"label": "pred_label"}), on="id")

    gt   = parse_label_col(sol["label"])
    pred = parse_label_col(sol["pred_label"])

    # Add per-task correctness
    for t in TASK_COLS:
        sol[f"{t}_true"]  = gt[t]
        sol[f"{t}_pred"]  = pred[t]
        sol[f"{t}_wrong"] = [g != p for g, p in zip(gt[t], pred[t])]
    sol["any_wrong"] = sol[[f"{t}_wrong" for t in TASK_COLS]].any(axis=1)

    print(f"\n{SEP}")
    print(f"  SECTION 2: Test error analysis  [{Path(BEST_SUB).stem}]")
    print(SEP)

    for split in ["Public", "Private", None]:
        mask = sol["Usage"] == split if split else pd.Series([True] * len(sol))
        g = sol[mask]
        label = split if split else "All"
        n_err = g["any_wrong"].sum()
        print(f"\n  [{label}]  errors: {n_err}/{len(g)} ({n_err/len(g)*100:.1f}%)")
        for t in TASK_COLS:
            w = g[f"{t}_wrong"].sum()
            f1 = f1_score(g[f"{t}_true"], g[f"{t}_pred"],
                          labels=M.LABELS[t], average="macro", zero_division=0)
            print(f"    {t.upper()}: wrong={w:>3}  macro-F1={f1:.4f}")

    # Error by company
    print(f"\n  Error rate by company (Public+Private, min 3 samples):")
    co = sol.groupby("company")["any_wrong"].agg(["sum", "count"])
    co = co[co["count"] >= 3].copy()
    co["rate"] = co["sum"] / co["count"]
    co = co.sort_values("rate", ascending=False)
    for company, row in co.iterrows():
        bar = "█" * int(row["rate"] * 20)
        print(f"    {company:<20} {row['sum']:>2.0f}/{row['count']:>2.0f}  "
              f"({row['rate']*100:>4.1f}%)  {bar}")

    # Cross-task error pattern
    print(f"\n  Cross-task co-error counts (Public+Private):")
    for t1 in TASK_COLS:
        for t2 in TASK_COLS:
            if t1 >= t2:
                continue
            both = (sol[f"{t1}_wrong"] & sol[f"{t2}_wrong"]).sum()
            if both > 0:
                print(f"    {t1.upper()} & {t2.upper()} both wrong: {both}")

    # T3 errors broken down by T1 true label
    print(f"\n  T3 errors by T1 true label:")
    for t1_val, grp in sol.groupby("t1_true"):
        w = grp["t3_wrong"].sum()
        print(f"    T1={t1_val}: T3 wrong {w}/{len(grp)} ({w/len(grp)*100:.1f}%)")


# ── Section 3: Per-class breakdown on test ────────────────────────────────────

def section3():
    sol  = pd.read_csv(DATA_DIR / "solution_data.csv")
    sub  = pd.read_csv(BEST_SUB)
    sol  = sol.merge(sub.rename(columns={"label": "pred_label"}), on="id")

    gt   = parse_label_col(sol["label"])
    pred = parse_label_col(sol["pred_label"])

    print(f"\n{SEP}")
    print(f"  SECTION 3: Per-class breakdown on ALL test (Public+Private)")
    print(f"  Submission: {Path(BEST_SUB).stem}")
    print(SEP)

    for t, col in TASK_COLS.items():
        labels = M.LABELS[t]
        print(f"\n  [{t.upper()}] {col}")
        print(classification_report(
            gt[t], pred[t], labels=labels, target_names=labels,
            zero_division=0, digits=3))

        cm = confusion_matrix(gt[t], pred[t], labels=labels)
        cm_df = pd.DataFrame(cm,
                             index=[f"T:{l}" for l in labels],
                             columns=[f"P:{l}" for l in labels])
        print("  Confusion matrix (rows=true, cols=pred):")
        print(cm_df.to_string())


# ── Section 4: OOF per-class breakdown (training data) ───────────────────────

def section4():
    print(f"\n{SEP}")
    print("  SECTION 4: OOF per-class breakdown [rob_aug, training set]")
    print(SEP)

    if not Path(KFOLD_DIR).exists():
        print(f"  [SKIP] {KFOLD_DIR} not found")
        return

    import torch
    from torch.utils.data import DataLoader

    cfg = M.Config()
    train_full, _ = M.load_dataframes(cfg.data_dir, use_augmented=True)
    strat_col = train_full["promise_status"].tolist()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_true = {t: [] for t in TASK_COLS}
    all_pred = {t: [] for t in TASK_COLS}
    device = torch.device("cpu")
    print("  Running OOF inference on CPU ...")

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_full, strat_col)):
        ckpt_path = Path(KFOLD_DIR) / f"fold{fold_idx + 1}" / "best.pt"
        if not ckpt_path.exists():
            print(f"  fold{fold_idx+1}: MISSING, skipping")
            continue
        fold_val = train_full.iloc[val_idx].reset_index(drop=True)
        sys.modules["__main__"].Config = M.Config
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        saved_cfg: M.Config = ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        val_ds = M.ESGDataset(fold_val, tokenizer, saved_cfg.max_length, has_labels=True)
        loader = DataLoader(val_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
        model = M.ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
        model.load_state_dict(ckpt["model"])
        _, preds = M.evaluate(model, loader, device, apply_rule=True)
        del model
        for t, col in TASK_COLS.items():
            all_true[t].extend(fold_val[col].tolist())
            all_pred[t].extend(preds[t])
        print(f"  fold{fold_idx+1} done")

    print()
    for t, col in TASK_COLS.items():
        labels = M.LABELS[t]
        print(f"  [{t.upper()}] {col}")
        print(classification_report(
            all_true[t], all_pred[t], labels=labels, target_names=labels,
            zero_division=0, digits=3))
        cm = confusion_matrix(all_true[t], all_pred[t], labels=labels)
        cm_df = pd.DataFrame(cm,
                             index=[f"T:{l}" for l in labels],
                             columns=[f"P:{l}" for l in labels])
        print("  Confusion matrix (rows=true, cols=pred):")
        print(cm_df.to_string())
        print()


def main():
    section1()
    section2()
    section3()
    section4()


if __name__ == "__main__":
    main()
