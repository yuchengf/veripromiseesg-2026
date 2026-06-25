"""Valid (final_data, 399) comparison of ALL combos @512 vs the @384 champion,
each with kNN(T3 a0.4, NCbias .1) and optional clarity overlay. Run after the
final_data 12-way@512 training finishes. Picks the best combo to take to AIdea."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

LAB = {"t1": ["Yes", "No"], "t2": ["Yes", "No", "N/A"],
       "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
       "t4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"]}
TW = {"t1": .2, "t2": .3, "t3": .35, "t4": .15}
SOL = {"t1": "promise_status", "t2": "evidence_status", "t3": "evidence_quality", "t4": "verification_timeline"}
C = Path("agent_cache/valid_probs")
tr = pd.read_csv("final_data/train_data.csv", keep_default_na=False)
sol = pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
sol["verification_timeline"] = sol["verification_timeline"].replace("longer_than_5_years", "more_than_5_years")
y = {t: sol[SOL[t]].astype(str).tolist() for t in LAB}
N = len(sol)

# kNN (T3) on valid
d = np.load("agent_cache/qwen3_embs_final.npz")
a = d["tr"] / (np.linalg.norm(d["tr"], axis=1, keepdims=True) + 1e-8)
b = d["va"] / (np.linalg.norm(d["va"], axis=1, keepdims=True) + 1e-8)
sim = b @ a.T
t3l = np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)])
knn = np.zeros((N, 4))
for i in range(N):
    tp = np.argsort(sim[i])[::-1][:5]
    w = np.exp(sim[i][tp] - sim[i][tp].max()); w /= w.sum()
    np.add.at(knn[i], t3l[tp], w); knn[i] /= knn[i].sum()
clp = np.load("agent_cache/clarity_valid_probs.npz")["probs"]
cca, ccf = clp.argmax(1), clp.max(1)
CL3 = ["Clear", "Not Clear", "Misleading"]


def avg(runs):
    return {t: np.mean([np.load(C / f"{r}_f{f}.npz")[t] for r in runs for f in range(1, 6)], axis=0) for t in LAB}


def score(base, clarity):
    t3 = base["t3"].copy(); t3[:, 1] *= np.exp(0.1); t3 /= t3.sum(1, keepdims=True)
    fused = 0.6 * t3 + 0.4 * knn
    pred = {t: [LAB[t][i] for i in (fused if t == "t3" else base[t]).argmax(1)] for t in LAB}
    if clarity:
        for i in range(N):
            if pred["t3"][i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading":
                pred["t3"][i] = CL3[cca[i]]
    for i in range(N):
        if pred["t1"][i] == "No":
            pred["t2"][i] = pred["t3"][i] = pred["t4"][i] = "N/A"
        elif pred["t2"][i] == "No":
            pred["t3"][i] = "N/A"
    tot = sum(TW[t] * f1_score(y[t], pred[t], labels=sorted(set(y[t])), average="macro", zero_division=0) for t in LAB)
    parts = {t: round(f1_score(y[t], pred[t], labels=sorted(set(y[t])), average="macro", zero_division=0), 4) for t in LAB}
    return tot, parts


def runs(prefix, recipes):  # recipes: list of ('', '_s1', '_s2') style handled below
    out = []
    for rc in recipes:
        out += [f"{rc}{prefix}_roberta_dc_kfold_t3nc3{sfx}" for sfx in ("", "_s1", "_s2")]
    return out


GROUPS = {
    "3way @384": runs("", ["FC1R"]),
    "3way @512": runs("512", ["FC1R"]),
    "6way @512": runs("512", ["FC1R", "FC1S"]),
    "9way @512": runs("512", ["FC1", "FC1R", "FC1S"]),
    "12way @512": runs("512", ["FC1", "FC1R", "FC1S", "FC1M"]),
}
best = None
for tag, rr in GROUPS.items():
    try:
        base = avg(rr)
    except FileNotFoundError:
        print(f"{tag:14s}: 缺檔(尚未訓練完)"); continue
    for cl in (False, True):
        tot, parts = score(base, cl)
        flag = "+clarity" if cl else "        "
        print(f"{tag:14s} {flag}: valid={tot:.5f}  {parts}")
        if best is None or tot > best[0]:
            best = (tot, f"{tag} {flag}")
if best:
    print(f"\n★ 本地 valid 最佳: {best[1]} = {best[0]:.5f}")
print("→ 取 valid 前 2-3 名 → 上 AIdea 驗證 transfer(ratio 不破才算數,不被 Kaggle 100%-public 數字干擾)")
