"""Generate kNN-enhanced AIdea submissions from retrain models."""
import sys, os
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]

from esg_main import (
    _predict_checkpoint, ensemble_preds_soft, apply_na_rule,
    load_dataframes, Config, extract_embeddings,
    compute_knn_ldl_probs, knn_fuse_probs, LABELS, IDX2LABEL,
)
import __main__
__main__.Config = Config

import pandas as pd, numpy as np, torch
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ── Step 1: Load all 60 fold predictions ──
MODELS = {
    "rt_fc1_s0":  "runs/RT_FC1_s0",   "rt_fc1_s1":  "runs/RT_FC1_s1",   "rt_fc1_s2":  "runs/RT_FC1_s2",
    "rt_fc1r_s0": "runs/RT_FC1R_s0",  "rt_fc1r_s1": "runs/RT_FC1R_s1",  "rt_fc1r_s2": "runs/RT_FC1R_s2",
    "rt_fc1s_s0": "runs/RT_FC1S_s0",  "rt_fc1s_s1": "runs/RT_FC1S_s1",  "rt_fc1s_s2": "runs/RT_FC1S_s2",
    "rt_fc1m_s0": "runs/RT_FC1M_s0",  "rt_fc1m_s1": "runs/RT_FC1M_s1",  "rt_fc1m_s2": "runs/RT_FC1M_s2",
}

model_fold_preds = {}
for key, run_dir in MODELS.items():
    fold_preds = []
    for fold in range(1, 6):
        ckpt = f"{run_dir}/fold{fold}/best.pt"
        if not Path(ckpt).exists():
            break
        result = _predict_checkpoint(ckpt, test_df)
        if result is not None:
            fold_preds.append(result[1])
    if len(fold_preds) == 5:
        model_fold_preds[key] = fold_preds
        print(f"  {key}: OK")
    else:
        print(f"  {key}: FAILED ({len(fold_preds)}/5)")

print(f"\nLoaded {len(model_fold_preds)}/12 models")

# ── Step 2: Compute Qwen3 kNN embeddings ──
print("\n[kNN] Computing Qwen3-Embedding for 2000 train + 2000 test...")
device = "cuda" if torch.cuda.is_available() else "cpu"
_BACKBONE = "Qwen/Qwen3-Embedding-0.6B"
_INSTRUCTION = "Retrieve similar ESG disclosure texts that share the same evidence quality level."
tr_embs = extract_embeddings(train_df["data"].tolist(), _BACKBONE, batch_size=16,
                              device=device, instruction=_INSTRUCTION).numpy()
te_embs = extract_embeddings(test_df["data"].tolist(), _BACKBONE, batch_size=16,
                              device=device, instruction=_INSTRUCTION).numpy()
print(f"  Embeddings: train={tr_embs.shape}, test={te_embs.shape}")

# Compute kNN probs
knn_probs = compute_knn_ldl_probs(tr_embs, te_embs, train_df, k=5, test_df=test_df)
print("  kNN probs computed")

# ── Step 3: Generate submissions ──
out_dir = Path("official_sub")
ALPHA = {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}
NC_BIAS = 0.1

def gen_submission(name, model_keys, weights=None):
    all_preds = []
    if weights is not None:
        expanded_w = []
        for w in weights:
            expanded_w.extend([w] * 5)
    else:
        expanded_w = None

    for k in model_keys:
        if k not in model_fold_preds:
            print(f"  [SKIP] {name}: missing {k}")
            return
        all_preds.extend(model_fold_preds[k])

    # Build base_probs as weighted average of raw probabilities
    ws = expanded_w or [1.0] * len(all_preds)
    total_w = sum(ws)
    n = len(all_preds[0]["t1"])
    from esg_main import NUM_LABELS
    base_probs = {}
    for task in LABELS:
        nc = NUM_LABELS[task]
        base_probs[task] = [
            [sum(ws[j] * all_preds[j][task][i][c]
                 for j in range(len(all_preds))) / total_w
             for c in range(nc)]
            for i in range(n)
        ]

    # Apply NC logit bias
    nc_idx = LABELS["t3"].index("Not Clear")
    import math
    new_t3 = []
    for p in base_probs["t3"]:
        p2 = list(p)
        p2[nc_idx] *= math.exp(NC_BIAS)
        total = sum(p2)
        new_t3.append([x / total for x in p2])
    base_probs_biased = dict(base_probs)
    base_probs_biased["t3"] = new_t3

    # Fuse with kNN
    preds = knn_fuse_probs(base_probs_biased, knn_probs, alpha=ALPHA)
    preds = apply_na_rule(preds)

    df_out = pd.DataFrame({
        "id": test_df["id"],
        "promise_status": preds["t1"],
        "verification_timeline": preds["t4"],
        "evidence_status": preds["t2"],
        "evidence_quality": preds["t3"],
    })
    path = out_dir / f"{name}.csv"
    df_out.to_csv(path, index=False)
    na = int((df_out == "N/A").sum().sum())
    print(f"  → {path} ({len(df_out)} rows, N/A={na})")

# 12-way + kNN (matches Kaggle best s5a3)
gen_submission("aidea_rt_all12_knn", list(MODELS.keys()))

# FC1R 3-seed + kNN (matches Kaggle s475)
gen_submission("aidea_rt_fc1r3_knn",
               ["rt_fc1r_s0", "rt_fc1r_s1", "rt_fc1r_s2"],
               [1.0, 2.0, 1.0])

print("\nDone!")
