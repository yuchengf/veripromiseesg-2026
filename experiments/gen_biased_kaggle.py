"""Apply Step 2 per-task biases to FC1 kfold models → Kaggle submissions."""
import sys, os, math
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["gen_biased_kaggle"]

from esg_main import (
    _predict_checkpoint, ensemble_preds_soft, apply_na_rule,
    load_dataframes, Config, extract_embeddings,
    compute_knn_ldl_probs, knn_fuse_probs, LABELS, IDX2LABEL, NUM_LABELS,
)
import __main__
__main__.Config = Config

import numpy as np, pandas as pd
from pathlib import Path

train_df, test_df = load_dataframes("final_data")
print(f"Train: {len(train_df)}, Test(valid): {len(test_df)}")

# ── Load all fold predictions ──
MODELS = {
    "fc1r_s0": "runs/FC1R_roberta_dc_kfold_t3nc3",
    "fc1r_s1": "runs/FC1R_roberta_dc_kfold_t3nc3_s1",
    "fc1r_s2": "runs/FC1R_roberta_dc_kfold_t3nc3_s2",
    "fc1s_s0": "runs/FC1S_roberta_dc_kfold_t3nc3",
    "fc1s_s1": "runs/FC1S_roberta_dc_kfold_t3nc3_s1",
    "fc1s_s2": "runs/FC1S_roberta_dc_kfold_t3nc3_s2",
    "fc1m_s0": "runs/FC1M_roberta_dc_kfold_t3nc3",
    "fc1m_s1": "runs/FC1M_roberta_dc_kfold_t3nc3_s1",
    "fc1m_s2": "runs/FC1M_roberta_dc_kfold_t3nc3_s2",
    "fc1_s0":  "runs/FC1_roberta_dc_kfold_t3nc3",
    "fc1_s1":  "runs/FC1_roberta_dc_kfold_t3nc3_s1",
    "fc1_s2":  "runs/FC1_roberta_dc_kfold_t3nc3_s2",
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

# ── Step 2 biases (from OOF tuning) ──
BIASES = {
    "t1": [-1.00, +0.30],           # Yes, No
    "t2": [-0.50, +0.00, +0.20],    # Yes, No, N/A
    "t3": [-0.10, +0.00, +1.10, +0.00],  # Clear, NC, Misleading, N/A
    "t4": [+0.20, +0.60, +0.00, +0.20, +1.30],  # already, within_2y, btw_2_5y, more_5y, N/A
}

def build_base_probs(model_keys, weights=None):
    """Build weighted average probabilities from fold predictions."""
    all_preds = []
    if weights:
        expanded_w = []
        for w in weights:
            expanded_w.extend([w] * 5)
    else:
        expanded_w = None
    for k in model_keys:
        all_preds.extend(model_fold_preds[k])

    ws = expanded_w or [1.0] * len(all_preds)
    total_w = sum(ws)
    n = len(all_preds[0]["t1"])
    base_probs = {}
    for task in LABELS:
        nc = NUM_LABELS[task]
        base_probs[task] = [
            [sum(ws[j] * all_preds[j][task][i][c]
                 for j in range(len(all_preds))) / total_w
             for c in range(nc)]
            for i in range(n)
        ]
    return base_probs

def apply_biases(base_probs, biases):
    """Apply per-task logit biases to probabilities."""
    result = {}
    for task in LABELS:
        nc = NUM_LABELS[task]
        new_probs = []
        for p in base_probs[task]:
            adj = [p[c] * math.exp(biases[task][c]) for c in range(nc)]
            total = sum(adj)
            new_probs.append([v / total for v in adj])
        result[task] = new_probs
    return result

def probs_to_preds(probs):
    """Argmax probabilities to label strings."""
    return {task: [IDX2LABEL[task][int(np.argmax(p))] for p in probs[task]]
            for task in LABELS}

def save_kaggle(name, preds, test_df):
    """Save Kaggle format submission."""
    preds = apply_na_rule(preds)
    df = pd.DataFrame({
        "id": test_df["id"],
        "promise_status": preds["t1"],
        "verification_timeline": preds["t4"],
        "evidence_status": preds["t2"],
        "evidence_quality": preds["t3"],
    })
    path = f"submissions/{name}.csv"
    df.replace("N/A", "-1").to_csv(path, index=False)
    aidea_path = f"submissions/{name}_aidea.csv"
    df.to_csv(aidea_path, index=False)
    mis_count = (df["evidence_quality"] == "Misleading").sum()
    print(f"  → {path} (Misleading={mis_count})")
    return path

# ── Generate submissions ──
all_keys = list(MODELS.keys())
out_dir = Path("submissions")

# 1. 12-way + biases only (no kNN)
print("\n=== 12-way + Step2 biases (no kNN) ===")
base = build_base_probs(all_keys)
biased = apply_biases(base, BIASES)
preds = probs_to_preds(biased)
save_kaggle("s5b0_all12_biased", preds, test_df)

# 2. 12-way + kNN + biases
print("\n=== Computing kNN ===")
device = "cuda"
_BACKBONE = "Qwen/Qwen3-Embedding-0.6B"
_INSTRUCTION = "Retrieve similar ESG disclosure texts that share the same evidence quality level."
tr_embs = extract_embeddings(train_df["data"].tolist(), _BACKBONE, batch_size=16,
                              device=device, instruction=_INSTRUCTION).numpy()
te_embs = extract_embeddings(test_df["data"].tolist(), _BACKBONE, batch_size=16,
                              device=device, instruction=_INSTRUCTION).numpy()
knn_probs = compute_knn_ldl_probs(tr_embs, te_embs, train_df, k=5, test_df=test_df)
print("  kNN probs computed")

# kNN configs to try
KNN_CFGS = [
    ("s5b1_all12_knn_biased", all_keys, None,
     {"alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}, True),
    ("s5b2_all12_knn_nobiased", all_keys, None,
     {"alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}, False),
    # T1 kNN alpha test
    ("s5b3_all12_knn_t1a05", all_keys, None,
     {"alpha": {"t1": 0.05, "t2": 0.0, "t3": 0.40, "t4": 0.0}}, True),
    ("s5b4_all12_knn_t1a10", all_keys, None,
     {"alpha": {"t1": 0.10, "t2": 0.0, "t3": 0.40, "t4": 0.0}}, True),
    # Biases + no kNN Misleading (safer for Kaggle: Misleading=+1.10 suppressed)
    ("s5b5_all12_knn_biased_nomis", all_keys, None,
     {"alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}, "nomis"),
]

for name, keys, weights, knn_cfg, use_biases in KNN_CFGS:
    print(f"\n=== {name} ===")
    base = build_base_probs(keys, weights)

    if use_biases == "nomis":
        # Biases but suppress Misleading
        b = {k: list(v) for k, v in BIASES.items()}
        b["t3"][2] = -10.0  # suppress Misleading
        biased = apply_biases(base, b)
    elif use_biases:
        biased = apply_biases(base, BIASES)
    else:
        biased = base

    # kNN fusion on T3
    alpha = knn_cfg["alpha"]
    fused = knn_fuse_probs(biased, knn_probs, alpha=alpha)
    save_kaggle(name, fused, test_df)

print("\nDone!")
