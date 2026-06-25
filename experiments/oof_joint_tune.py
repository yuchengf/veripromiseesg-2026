"""Step 1+2+3+4: OOF-based joint optimization for T3 improvement.

Usage: python oof_joint_tune.py
- Extracts OOF probs from FC1 kfold models (final_data, 1601 rows)
- Step 1: Joint logit adjustment for T3 (all 4 classes)
- Step 2: Cascade-aware joint threshold search (T1+T2+T3)
- Step 3: Misleading probability analysis
- Step 4: T1 kNN alpha tuning
"""
import sys, os
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["oof_joint_tune"]

import numpy as np
import pandas as pd
import math
from pathlib import Path
from sklearn.metrics import f1_score as sk_f1
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

# Import from esg_main
from esg_main import (
    LABELS, IDX2LABEL, LABEL2IDX, NUM_LABELS,
    load_dataframes, extract_oof_teacher_probs, apply_na_rule, Config,
)
import __main__
__main__.Config = Config

# ─── Load data ───
train_df, _ = load_dataframes("final_data")
print(f"Train: {len(train_df)} rows")

# Ground truth labels
GT = {}
GT["t1"] = train_df["promise_status"].tolist()
GT["t2"] = train_df["evidence_status"].tolist()
GT["t3"] = train_df["evidence_quality"].tolist()
GT["t4"] = train_df["verification_timeline"].tolist()

# ─── Extract OOF probs from multiple models ───
KFOLD_DIRS = {
    "FC1R_s0": ("runs/FC1R_roberta_dc_kfold_t3nc3", 42),
    "FC1R_s1": ("runs/FC1R_roberta_dc_kfold_t3nc3_s1", 123),
    "FC1R_s2": ("runs/FC1R_roberta_dc_kfold_t3nc3_s2", 456),
    "FC1S_s0": ("runs/FC1S_roberta_dc_kfold_t3nc3", 42),
    "FC1S_s1": ("runs/FC1S_roberta_dc_kfold_t3nc3_s1", 123),
    "FC1S_s2": ("runs/FC1S_roberta_dc_kfold_t3nc3_s2", 456),
    "FC1M_s0": ("runs/FC1M_roberta_dc_kfold_t3nc3", 42),
    "FC1_s0":  ("runs/FC1_roberta_dc_kfold_t3nc3", 42),
}

print("\n=== Extracting OOF probs ===")
all_oof = []
for name, (kdir, seed) in KFOLD_DIRS.items():
    if not Path(kdir).exists():
        print(f"  [SKIP] {name}: {kdir} not found")
        continue
    try:
        probs = extract_oof_teacher_probs(kdir, train_df, n_splits=5, seed=seed)
        all_oof.append(probs)
        print(f"  {name}: OK")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

print(f"\nLoaded {len(all_oof)} OOF sets")

# Average OOF probs across models
n = len(train_df)
avg_oof = {}
for task in LABELS:
    nc = NUM_LABELS[task]
    avg_oof[task] = []
    for i in range(n):
        avg = [0.0] * nc
        count = 0
        for oof in all_oof:
            if oof[task][i] is not None:
                for c in range(nc):
                    avg[c] += oof[task][i][c]
                count += 1
        if count > 0:
            avg = [v / count for v in avg]
        avg_oof[task].append(avg)

# ═══════════════════════════════════════════════════════════════
# STEP 1: Joint T3 Logit Adjustment
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1: Joint T3 Logit Adjustment")
print("=" * 60)

# Build arrays
t3_probs = np.array(avg_oof["t3"])  # (1601, 4)
t3_labels_str = GT["t3"]
t3_labels = np.array([LABEL2IDX["t3"].get(l, -1) for l in t3_labels_str])
valid_mask = t3_labels >= 0
t3_probs_v = t3_probs[valid_mask]
t3_labels_v = t3_labels[valid_mask]

# Class counts
for c, name in IDX2LABEL["t3"].items():
    cnt = (t3_labels_v == c).sum()
    print(f"  {name}: {cnt} samples")

# Baseline
baseline_preds = np.argmax(t3_probs_v, axis=1)
baseline_f1_4 = sk_f1(t3_labels_v, baseline_preds, average="macro")
# 3-class (exclude Misleading)
mis_idx = LABEL2IDX["t3"]["Misleading"]
mask_3c = t3_labels_v != mis_idx
baseline_f1_3 = sk_f1(t3_labels_v[mask_3c], baseline_preds[mask_3c], average="macro",
                       labels=[i for i in range(4) if i != mis_idx])
print(f"\n  Baseline T3 4-class Macro-F1: {baseline_f1_4:.4f}")
print(f"  Baseline T3 3-class Macro-F1: {baseline_f1_3:.4f}")

# Per-class F1
for c, name in IDX2LABEL["t3"].items():
    mask_c = t3_labels_v == c
    if mask_c.sum() > 0:
        tp = ((baseline_preds == c) & mask_c).sum()
        fp = ((baseline_preds == c) & ~mask_c).sum()
        fn = ((baseline_preds != c) & mask_c).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"    {name:15s}: P={p:.3f} R={r:.3f} F1={f:.3f} (n={mask_c.sum()})")

def apply_deltas(probs, deltas):
    """Apply per-class logit bias: adjusted[c] = probs[c] * exp(delta[c])"""
    adj = probs.copy()
    for c, d in enumerate(deltas):
        adj[:, c] *= math.exp(d)
    adj = adj / adj.sum(axis=1, keepdims=True)
    return adj

def neg_macro_f1_4class(params):
    """Objective: maximize 4-class macro F1."""
    deltas = [0.0, params[0], params[1], params[2]]  # Clear=0, NC, Misleading, N/A
    adj = apply_deltas(t3_probs_v, deltas)
    preds = np.argmax(adj, axis=1)
    return -sk_f1(t3_labels_v, preds, average="macro")

def neg_macro_f1_3class(params):
    """Objective: maximize 3-class macro F1 (exclude Misleading)."""
    deltas = [0.0, params[0], -100.0, params[1]]  # Misleading suppressed
    adj = apply_deltas(t3_probs_v, deltas)
    preds = np.argmax(adj, axis=1)
    return -sk_f1(t3_labels_v[mask_3c], preds[mask_3c], average="macro",
                   labels=[i for i in range(4) if i != mis_idx])

# Optimize 4-class
print("\n--- Optimizing 4-class F1 ---")
res4 = minimize(neg_macro_f1_4class, [0.0, 0.0, 0.0], method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 0.01, "fatol": 0.0001})
best_deltas_4 = [0.0, res4.x[0], res4.x[1], res4.x[2]]
best_f1_4 = -res4.fun
print(f"  Best deltas: Clear=0.00, NC={best_deltas_4[1]:+.3f}, Misleading={best_deltas_4[2]:+.3f}, NA={best_deltas_4[3]:+.3f}")
print(f"  Best 4-class F1: {best_f1_4:.4f} (gain: {best_f1_4 - baseline_f1_4:+.4f})")

# Per-class F1 at optimal
adj_opt = apply_deltas(t3_probs_v, best_deltas_4)
opt_preds = np.argmax(adj_opt, axis=1)
for c, name in IDX2LABEL["t3"].items():
    mask_c = t3_labels_v == c
    if mask_c.sum() > 0:
        tp = ((opt_preds == c) & mask_c).sum()
        fp = ((opt_preds == c) & ~mask_c).sum()
        fn = ((opt_preds != c) & mask_c).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"    {name:15s}: P={p:.3f} R={r:.3f} F1={f:.3f}")

# Optimize 3-class (for Kaggle)
print("\n--- Optimizing 3-class F1 (Kaggle) ---")
res3 = minimize(neg_macro_f1_3class, [0.0, 0.0], method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 0.01, "fatol": 0.0001})
best_deltas_3 = [0.0, res3.x[0], -100.0, res3.x[1]]
best_f1_3 = -res3.fun
print(f"  Best deltas: Clear=0.00, NC={best_deltas_3[1]:+.3f}, NA={best_deltas_3[3]:+.3f}")
print(f"  Best 3-class F1: {best_f1_3:.4f} (gain: {best_f1_3 - baseline_f1_3:+.4f})")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Cascade-Aware Joint Threshold Search
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Cascade-Aware Joint Threshold Search")
print("=" * 60)

WEIGHTS = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}

def eval_weighted_f1(preds_dict):
    """Compute weighted F1 with apply_na_rule."""
    preds_post = apply_na_rule(preds_dict)
    total = 0.0
    details = {}
    for task, w in WEIGHTS.items():
        gt_list = GT[task]
        pred_list = preds_post[task]
        # Only evaluate samples with valid GT
        pairs = [(g, p) for g, p in zip(gt_list, pred_list) if g in LABEL2IDX[task]]
        if not pairs:
            continue
        gt_arr = [g for g, p in pairs]
        pr_arr = [p for g, p in pairs]
        f1 = sk_f1(gt_arr, pr_arr, average="macro")
        total += w * f1
        details[task] = f1
    return total, details

# Baseline: argmax on avg_oof
base_preds = {}
for task in LABELS:
    probs = np.array(avg_oof[task])
    base_preds[task] = [IDX2LABEL[task][int(np.argmax(p))] for p in probs]

base_wf1, base_details = eval_weighted_f1(base_preds)
print(f"\n  Baseline weighted F1: {base_wf1:.4f}")
for t, f in base_details.items():
    print(f"    {t}: {f:.4f} (weight={WEIGHTS[t]})")

# Coordinate-ascent threshold search
# For each task, try per-class threshold overrides
def threshold_preds(probs_list, thresholds):
    """Convert probs to labels with per-class thresholds."""
    n = len(probs_list)
    preds = []
    nc = len(probs_list[0])
    for i in range(n):
        p = probs_list[i]
        # Check if any class exceeds its threshold
        best_c = int(np.argmax(p))
        for c in range(nc):
            if thresholds[c] > 0 and p[c] >= thresholds[c] and c != best_c:
                # Override if threshold met and prob is significant
                if p[c] > p[best_c] * 0.5:  # must be at least half of argmax
                    best_c = c
        preds.append(best_c)
    return preds

# Simple per-class bias sweep for each task
best_biases = {task: [0.0] * NUM_LABELS[task] for task in LABELS}

for round_num in range(3):
    improved = False
    for task in ["t1", "t2", "t3", "t4"]:
        probs = np.array(avg_oof[task])
        nc = NUM_LABELS[task]
        for c in range(nc):
            best_b = best_biases[task][c]
            best_score = -1
            for delta in np.linspace(-1.0, 2.0, 31):
                trial_biases = list(best_biases[task])
                trial_biases[c] = delta
                # Apply biases
                adj = probs.copy()
                for cc in range(nc):
                    adj[:, cc] *= math.exp(trial_biases[cc])
                adj = adj / adj.sum(axis=1, keepdims=True)
                # Predict
                trial_preds = dict(base_preds)
                trial_preds[task] = [IDX2LABEL[task][int(np.argmax(adj[i]))] for i in range(len(adj))]
                wf1, _ = eval_weighted_f1(trial_preds)
                if wf1 > best_score:
                    best_score = wf1
                    best_b = delta
            if best_b != best_biases[task][c]:
                improved = True
            best_biases[task][c] = best_b

        # Apply this task's best biases for subsequent tasks
        probs = np.array(avg_oof[task])
        nc = NUM_LABELS[task]
        adj = probs.copy()
        for cc in range(nc):
            adj[:, cc] *= math.exp(best_biases[task][cc])
        adj = adj / adj.sum(axis=1, keepdims=True)
        base_preds[task] = [IDX2LABEL[task][int(np.argmax(adj[i]))] for i in range(len(adj))]

    wf1, details = eval_weighted_f1(base_preds)
    print(f"\n  Round {round_num + 1}: weighted F1 = {wf1:.4f}")
    for t, f in details.items():
        print(f"    {t}: {f:.4f}")
    if not improved:
        break

print(f"\n  Best biases per task:")
for task in LABELS:
    names = LABELS[task]
    biases_str = ", ".join(f"{names[c]}={best_biases[task][c]:+.2f}" for c in range(len(names)))
    print(f"    {task}: {biases_str}")

final_wf1, final_details = eval_weighted_f1(base_preds)
print(f"\n  Final weighted F1: {final_wf1:.4f} (gain: {final_wf1 - base_wf1:+.4f})")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Misleading Probability Analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Misleading Probability Analysis")
print("=" * 60)

mis_probs = t3_probs[:, mis_idx]  # Misleading prob for all 1601 samples
mis_labels = np.array([1 if l == "Misleading" else 0 for l in t3_labels_str])

# Where are the 2 real Misleading samples?
real_mis_idx = np.where(mis_labels == 1)[0]
print(f"\n  Real Misleading samples: {len(real_mis_idx)}")
for idx in real_mis_idx:
    p = t3_probs[idx]
    rank = (mis_probs > mis_probs[idx]).sum() + 1
    print(f"    ID={train_df.iloc[idx]['id']}: P(Misleading)={mis_probs[idx]:.4f}, "
          f"rank={rank}/{len(mis_probs)}, argmax={IDX2LABEL['t3'][int(np.argmax(p))]}")
    print(f"      All probs: Clear={p[0]:.3f} NC={p[1]:.3f} Mis={p[2]:.3f} NA={p[3]:.3f}")

# Top-10 by Misleading prob
print(f"\n  Top-10 samples by P(Misleading):")
top_idx = np.argsort(-mis_probs)[:10]
for i, idx in enumerate(top_idx):
    p = t3_probs[idx]
    is_real = "★" if mis_labels[idx] == 1 else " "
    gt = t3_labels_str[idx]
    print(f"    {i+1}. {is_real} ID={train_df.iloc[idx]['id']}: P(Mis)={mis_probs[idx]:.4f}, "
          f"GT={gt}, argmax={IDX2LABEL['t3'][int(np.argmax(p))]}")

# Threshold sweep for Misleading
print(f"\n  Misleading threshold sweep (predict Misleading if P > tau):")
for tau in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    predicted_mis = (mis_probs >= tau).sum()
    tp = ((mis_probs >= tau) & (mis_labels == 1)).sum()
    fp = ((mis_probs >= tau) & (mis_labels == 0)).sum()
    print(f"    tau={tau:.2f}: predict {predicted_mis} Misleading, TP={tp}, FP={fp}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: T1 kNN Alpha Exploration (simplified, no actual kNN)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: T1 Analysis")
print("=" * 60)

t1_probs = np.array(avg_oof["t1"])  # (1601, 2)
t1_labels = np.array([LABEL2IDX["t1"].get(l, -1) for l in GT["t1"]])
t1_preds = np.argmax(t1_probs, axis=1)
t1_f1 = sk_f1(t1_labels, t1_preds, average="macro")
print(f"\n  T1 OOF Macro-F1: {t1_f1:.4f}")

# How many borderline T1 predictions?
t1_confidence = np.max(t1_probs, axis=1)
borderline = t1_confidence < 0.7
print(f"  Borderline T1 (confidence < 0.7): {borderline.sum()} samples")
print(f"  T1 errors: {(t1_preds != t1_labels).sum()} / {len(t1_labels)}")

# How would perfect T1 help?
perfect_t1_preds = dict(base_preds)
perfect_t1_preds["t1"] = GT["t1"]  # Use ground truth T1
wf1_perfect_t1, details_pt1 = eval_weighted_f1(perfect_t1_preds)
print(f"\n  Weighted F1 with PERFECT T1: {wf1_perfect_t1:.4f} (gain: {wf1_perfect_t1 - base_wf1:+.4f})")
for t, f in details_pt1.items():
    print(f"    {t}: {f:.4f}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
