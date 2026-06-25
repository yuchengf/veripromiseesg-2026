"""kNN fine sweep on valid 399 for the frozen 12-way combo.
Sweeps k, sim temperature, T3 alpha, NC bias, T2/T4 alpha.
CAVEAT: tunes on the Layer-2 set; prefer plateau + Public/Private consistency.
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

sol = pd.read_csv(ROOT / "final_data/valid_solution_data.csv", keep_default_na=False)
train = pd.read_csv(ROOT / "final_data/train_data.csv", keep_default_na=False)
for df in (sol, train):
    df["verification_timeline"] = df["verification_timeline"].replace(
        "longer_than_5_years", "more_than_5_years")
y = {t: sol[SOL_COL[t]].astype(str).tolist() for t in LABELS}
usage = sol["Usage"].values
N = len(sol)

PRE = {g: f"{g}_roberta_dc_kfold_t3nc3" for g in ["FC1", "FC1R", "FC1S", "FC1M"]}
CACHE = ROOT / "agent_cache/valid_probs"
base = {t: np.mean([np.load(CACHE / f"{p}{s}_f{f}.npz")[t]
                    for p in PRE.values() for s in ["", "_s1", "_s2"] for f in range(1, 6)],
                   axis=0) for t in LABELS}

d = np.load(ROOT / "agent_cache/qwen3_embs_final.npz")
tr, va = d["tr"], d["va"]
tr = tr / (np.linalg.norm(tr, axis=1, keepdims=True) + 1e-8)
va = va / (np.linalg.norm(va, axis=1, keepdims=True) + 1e-8)
sim = va @ tr.T
lab_idx = {t: np.array([LABELS[t].index(l) for l in train[SOL_COL[t]].astype(str)])
           for t in LABELS}


def knn_probs(k, temp):
    out = {t: np.zeros((N, len(LABELS[t]))) for t in LABELS}
    for i in range(N):
        top = np.argsort(sim[i])[::-1][:k]
        s = sim[i][top] / temp
        w = np.exp(s - s.max()); w /= w.sum()
        for t in LABELS:
            np.add.at(out[t][i], lab_idx[t][top], w)
            out[t][i] /= out[t][i].sum()
    return out


def evaluate(knn, a2, a3, a4, nc_bias):
    fused = dict(base)
    t3 = base["t3"].copy()
    t3[:, 1] *= np.exp(nc_bias)
    t3 /= t3.sum(1, keepdims=True)
    fused["t2"] = (1 - a2) * base["t2"] + a2 * knn["t2"]
    fused["t3"] = (1 - a3) * t3 + a3 * knn["t3"]
    fused["t4"] = (1 - a4) * base["t4"] + a4 * knn["t4"]
    preds = {t: [LABELS[t][i] for i in np.argmax(fused[t], axis=1)] for t in LABELS}
    for i in range(N):
        if preds["t1"][i] == "No":
            preds["t2"][i] = preds["t3"][i] = preds["t4"][i] = "N/A"
        elif preds["t2"][i] == "No":
            preds["t3"][i] = "N/A"
    return preds


def score(preds, mask=None):
    idx = np.arange(N) if mask is None else np.where(mask)[0]
    tot = 0.0
    for t in LABELS:
        yt = [y[t][i] for i in idx]; pt = [preds[t][i] for i in idx]
        tot += TASK_W[t] * f1_score(yt, pt, labels=sorted(set(yt)),
                                    average="macro", zero_division=0)
    return tot

results = []
for k in [5, 10, 20]:
    for temp in [0.2, 0.5, 1.0]:
        knn = knn_probs(k, temp)
        for a3 in [0.2, 0.3, 0.4, 0.5]:
            for nc in [0.0, 0.1, 0.2]:
                for a2 in [0.0, 0.1]:
                    for a4 in [0.0, 0.1]:
                        p = evaluate(knn, a2, a3, a4, nc)
                        results.append((score(p), score(p, usage == "Public"),
                                        score(p, usage == "Private"),
                                        dict(k=k, temp=temp, a2=a2, a3=a3, a4=a4, nc=nc)))

results.sort(key=lambda r: -r[0])
print(f"baseline (k=5,temp=1,a3=.4,nc=.1,a2=a4=0) in pipeline today")
print(f"{'ALL':>8s} {'Public':>8s} {'Private':>8s}  cfg")
for a, pu, pr, cfg in results[:20]:
    print(f"{a:8.5f} {pu:8.5f} {pr:8.5f}  {cfg}")
print("...")
for a, pu, pr, cfg in results:
    if cfg == dict(k=5, temp=1.0, a2=0.0, a3=0.4, a4=0.0, nc=0.1):
        print(f"current: {a:8.5f} {pu:8.5f} {pr:8.5f}  {cfg}")
