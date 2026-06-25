"""G5: Per-task OOF F1 breakdown for rob_aug and lert_aug kfold models.

Usage:
    conda activate AICUP
    python g5_pertask_oof.py

Outputs a table comparing RoBERTa vs LERT per task (T1/T2/T3/T4 + weighted).
No GPU required if models are already trained (runs on CPU).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ── bootstrap: reuse esg_main internals ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import esg_main as M

TASK_WEIGHTS = M.TASK_WEIGHTS
LABELS       = M.LABELS
IDX2LABEL    = M.IDX2LABEL
LABEL2IDX    = M.LABEL2IDX

KFOLD_DIRS = {
    "rob_aug":  "runs/A1_roberta_dc_kfold_llmaug",
    "lert_aug": "runs/A1_lert_dc_kfold_llmaug",
}
N_SPLITS = 5
SEED     = 42


def oof_scores_for(name: str, kfold_dir: str) -> dict:
    """Run OOF inference across all folds and return per-task macro F1."""
    import torch
    from torch.utils.data import DataLoader

    cfg_dummy = M.Config()
    train_full, _ = M.load_dataframes(cfg_dummy.data_dir, use_augmented=True)

    strat_col = train_full["promise_status"].tolist()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_true = {t: [] for t in ["t1", "t2", "t3", "t4"]}
    all_pred = {t: [] for t in ["t1", "t2", "t3", "t4"]}

    device = torch.device("cpu")  # G5 is analysis-only, keep GPU free
    print(f"\n[{name}] running OOF on {device} ...")

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_full, strat_col)):
        ckpt_path = Path(kfold_dir) / f"fold{fold_idx + 1}" / "best.pt"
        if not ckpt_path.exists():
            print(f"  fold{fold_idx+1}: MISSING {ckpt_path}, skipping")
            continue

        fold_val = train_full.iloc[val_idx].reset_index(drop=True)

        import torch as _torch
        sys.modules["__main__"].Config = M.Config  # pickle fix
        ckpt = _torch.load(str(ckpt_path), map_location=device, weights_only=False)
        saved_cfg: M.Config = ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        val_ds = M.ESGDataset(fold_val, tokenizer, saved_cfg.max_length, has_labels=True)
        loader = DataLoader(val_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)

        model = M.ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
        model.load_state_dict(ckpt["model"])

        scores, preds = M.evaluate(model, loader, device, apply_rule=True)

        task_col = {"t1": "promise_status", "t2": "evidence_status",
                    "t3": "evidence_quality", "t4": "verification_timeline"}
        for t, col in task_col.items():
            all_true[t].extend(fold_val[col].tolist())
            all_pred[t].extend(preds[t])

        del model
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        f1w = scores.get("weighted", 0)
        print(f"  fold{fold_idx+1}: weighted={f1w:.4f}  "
              f"T1={scores.get('t1',0):.4f}  T2={scores.get('t2',0):.4f}  "
              f"T3={scores.get('t3',0):.4f}  T4={scores.get('t4',0):.4f}")

    # Full OOF scores
    result = {}
    for t in ["t1", "t2", "t3", "t4"]:
        result[t] = f1_score(all_true[t], all_pred[t],
                             labels=LABELS[t], average="macro", zero_division=0)
    result["weighted"] = sum(TASK_WEIGHTS[t] * result[t] for t in ["t1", "t2", "t3", "t4"])
    return result


def main():
    results = {}
    for name, kdir in KFOLD_DIRS.items():
        if not Path(kdir).exists():
            print(f"[SKIP] {name}: {kdir} not found")
            continue
        results[name] = oof_scores_for(name, kdir)

    if not results:
        print("No results — check KFOLD_DIRS paths.")
        return

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  G5: Per-task OOF F1 Breakdown")
    print("=" * 60)
    header = f"{'Model':<12} {'T1':>7} {'T2':>7} {'T3':>7} {'T4':>7} {'Weighted':>10}"
    print(header)
    print("-" * 60)
    for name, s in results.items():
        print(f"{name:<12} {s['t1']:>7.4f} {s['t2']:>7.4f} {s['t3']:>7.4f} "
              f"{s['t4']:>7.4f} {s['weighted']:>10.4f}")

    if len(results) == 2:
        names = list(results.keys())
        a, b = results[names[0]], results[names[1]]
        print("-" * 60)
        diff_label = f"{'diff':>12}"
        print(f"{diff_label} "
              f"{a['t1']-b['t1']:>+7.4f} {a['t2']-b['t2']:>+7.4f} "
              f"{a['t3']-b['t3']:>+7.4f} {a['t4']-b['t4']:>+7.4f} "
              f"{a['weighted']-b['weighted']:>+10.4f}")
        print(f"  (positive = {names[0]} better)")
    print("=" * 60)


if __name__ == "__main__":
    main()
