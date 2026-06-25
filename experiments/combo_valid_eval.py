"""Compare uniform ensemble combos on valid 399, replicating the AIdea gen_rt_knn
fusion pipeline (NC bias 0.1, kNN k=5, T3 alpha=0.40). Pure numpy on cached probs.

Motivated by AIdea result: 12-way+kNN 0.6140 < FC1R3+kNN 0.6170 -> suspect FC1M.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ROOT = Path("/home/yucheng/Desktop/ESG")
LABELS = {
    "t1": ["Yes", "No"],
    "t2": ["Yes", "No", "N/A"],
    "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "t4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
}
TASK_W = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}
SOL_COL = {"t1": "promise_status", "t2": "evidence_status",
           "t3": "evidence_quality", "t4": "verification_timeline"}
NORM = {"longer_than_5_years": "more_than_5_years"}

sol = pd.read_csv(ROOT / "final_data/valid_solution_data.csv", keep_default_na=False)
train = pd.read_csv(ROOT / "final_data/train_data.csv", keep_default_na=False)
for df in (sol, train):
    df["verification_timeline"] = df["verification_timeline"].replace(NORM)
y = {t: sol[SOL_COL[t]].astype(str).tolist() for t in LABELS}
usage = sol["Usage"].values
N = len(sol)

PREFIX = {
    "FC1": "FC1_roberta_dc_kfold_t3nc3", "FC1R": "FC1R_roberta_dc_kfold_t3nc3",
    "FC1S": "FC1S_roberta_dc_kfold_t3nc3", "FC1M": "FC1M_roberta_dc_kfold_t3nc3",
}
CACHE = ROOT / "agent_cache/valid_probs"


def group_probs(g):
    out = []
    for suf in ["", "_s1", "_s2"]:
        for f in range(1, 6):
            d = np.load(CACHE / f"{PREFIX[g]}{suf}_f{f}.npz")
            out.append({t: d[t] for t in LABELS})
    return out


probs_cache = {g: group_probs(g) for g in PREFIX}

# kNN probs from cached Qwen3 embeddings (k=5, softmax weights), full train pool
d = np.load(ROOT / "agent_cache/qwen3_embs_final.npz")
tr, va = d["tr"], d["va"]
tr = tr / (np.linalg.norm(tr, axis=1, keepdims=True) + 1e-8)
va = va / (np.linalg.norm(va, axis=1, keepdims=True) + 1e-8)
sim = va @ tr.T
lab_idx = {t: np.array([LABELS[t].index(l) if l in LABELS[t] else -1
                        for l in train[SOL_COL[t]].astype(str)]) for t in LABELS}
knn = {t: np.zeros((N, len(LABELS[t])), dtype=np.float64) for t in LABELS}
for i in range(N):
    top = np.argsort(sim[i])[::-1][:5]
    w = np.exp(sim[i][top] - sim[i][top].max()); w /= w.sum()
    for t in LABELS:
        for rank, tidx in enumerate(top):
            c = lab_idx[t][tidx]
            if c >= 0:
                knn[t][i, c] += w[rank]
        s = knn[t][i].sum()
        knn[t][i] = knn[t][i] / s if s > 0 else 1.0 / len(LABELS[t])

NC_IDX = LABELS["t3"].index("Not Clear")


def evaluate(combo, weights=None, alpha_t3=0.40, nc_bias=0.1):
    fold_probs, ws = [], []
    for gi, g in enumerate(combo):
        wgt = weights[gi] if weights else 1.0
        for fp in probs_cache[g]:
            fold_probs.append(fp); ws.append(wgt)
    ws = np.array(ws); ws = ws / ws.sum()
    base = {t: np.tensordot(ws, np.stack([fp[t] for fp in fold_probs]), axes=1)
            for t in LABELS}
    t3 = base["t3"].copy()
    t3[:, NC_IDX] *= np.exp(nc_bias)
    base["t3"] = t3 / t3.sum(axis=1, keepdims=True)
    alpha = {"t1": 0.0, "t2": 0.0, "t3": alpha_t3, "t4": 0.0}
    fused = {t: (1 - alpha[t]) * base[t] + alpha[t] * knn[t] for t in LABELS}
    preds = {t: [LABELS[t][i] for i in np.argmax(fused[t], axis=1)] for t in LABELS}
    for i in range(N):
        if preds["t1"][i] == "No":
            preds["t2"][i] = preds["t3"][i] = preds["t4"][i] = "N/A"
        elif preds["t2"][i] == "No":
            preds["t3"][i] = "N/A"
    return preds


def score(preds, mask=None):
    idx = np.arange(N) if mask is None else np.where(mask)[0]
    s = {}
    for t in LABELS:
        yt = [y[t][i] for i in idx]
        pt = [preds[t][i] for i in idx]
        s[t] = f1_score(yt, pt, labels=sorted(set(yt)), average="macro", zero_division=0)
    s["total"] = sum(TASK_W[t] * s[t] for t in TASK_W)
    return s


COMBOS = [
    ("FC1R3 [1,2,1] (AIdea 0.6170)", ["FC1R"], [1.0]),  # weights apply per-group; seed weighting NA here
    ("FC1R3 uniform", ["FC1R"], None),
    ("6w FC1R+FC1S", ["FC1R", "FC1S"], None),
    ("9w no-FC1M", ["FC1", "FC1R", "FC1S"], None),
    ("12w all (AIdea 0.6140)", ["FC1", "FC1R", "FC1S", "FC1M"], None),
    ("9w no-FC1S", ["FC1", "FC1R", "FC1M"], None),
    ("6w FC1+FC1R", ["FC1", "FC1R"], None),
]

print(f"{'combo':32s} {'ALL':>8s} {'Public':>8s} {'Private':>8s}   T1/T2/T3/T4 (all)")
for name, combo, wts in COMBOS:
    p = evaluate(combo, wts)
    a, pu, pr = score(p), score(p, usage == "Public"), score(p, usage == "Private")
    print(f"{name:32s} {a['total']:8.5f} {pu['total']:8.5f} {pr['total']:8.5f}   "
          f"{a['t1']:.3f}/{a['t2']:.3f}/{a['t3']:.3f}/{a['t4']:.3f}")

# alpha sweep on the top candidates
print("\nT3 alpha sweep (NC bias 0.1):")
for name, combo, wts in COMBOS[1:5]:
    row = []
    for al in [0.0, 0.2, 0.3, 0.4, 0.5]:
        p = evaluate(combo, wts, alpha_t3=al)
        row.append(f"a{al}={score(p)['total']:.5f}")
    print(f"  {name:28s} " + "  ".join(row))
