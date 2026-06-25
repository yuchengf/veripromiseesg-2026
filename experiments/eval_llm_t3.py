"""Validate the LLM-T3 override on valid: does using Qwen3-14B's Clear/NotClear
judgment improve T3 (esp Not Clear F1) and the overall weighted score? Tests several
override strategies x confidence thresholds, with bootstrap vs the base (champion).
Caveat: valid is a different (train-derived) distribution; AIdea is the real test.
But valid Clear/NotClear F1 is a sanity check (both classes present)."""
import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
LAB = {"t1": ["Yes", "No"], "t2": ["Yes", "No", "N/A"], "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
       "t4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"]}
TW = {"t1": .2, "t2": .3, "t3": .35, "t4": .15}
SOL = {"t1": "promise_status", "t2": "evidence_status", "t3": "evidence_quality", "t4": "verification_timeline"}
C = Path("agent_cache/valid_probs")
tr = pd.read_csv("final_data/train_data.csv", keep_default_na=False)
sol = pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
sol["verification_timeline"] = sol["verification_timeline"].replace("longer_than_5_years", "more_than_5_years")
vdata = pd.read_csv("final_data/valid_data.csv", keep_default_na=False)
ids = vdata["id"].astype(str).tolist()
y = {t: np.array(sol[SOL[t]].astype(str).tolist()) for t in LAB}
N = len(sol)
d = np.load("agent_cache/qwen3_embs_final.npz")
a = d["tr"] / (np.linalg.norm(d["tr"], axis=1, keepdims=True) + 1e-8)
bb = d["va"] / (np.linalg.norm(d["va"], axis=1, keepdims=True) + 1e-8)
sim = bb @ a.T
t3l = np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)])
knn = np.zeros((N, 4))
for i in range(N):
    tp = np.argsort(sim[i])[::-1][:5]; w = np.exp(sim[i][tp] - sim[i][tp].max()); w /= w.sum()
    np.add.at(knn[i], t3l[tp], w); knn[i] /= knn[i].sum()
clp = np.load("agent_cache/clarity_valid_probs.npz")["probs"]; cca, ccf = clp.argmax(1), clp.max(1)
CL3 = ["Clear", "Not Clear", "Misleading"]
def avg(rc): return {t: np.mean([np.load(C / f"{rc}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for s in ("", "_s1", "_s2") for f in range(1, 6)], axis=0) for t in LAB}
b = avg("FC1R")
t3 = b["t3"].copy(); t3[:, 1] *= np.exp(0.1); t3 /= t3.sum(1, keepdims=True); fused = 0.6 * t3 + 0.4 * knn
base = {t: np.array([LAB[t][i] for i in (fused if t == "t3" else b[t]).argmax(1)]) for t in LAB}
for i in range(N):  # clarity overlay
    if base["t3"][i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading": base["t3"][i] = CL3[cca[i]]
def cascade(p):
    p = {t: p[t].copy() for t in LAB}
    for i in range(N):
        if p["t1"][i] == "No": p["t2"][i] = p["t3"][i] = p["t4"][i] = "N/A"
        elif p["t2"][i] == "No": p["t3"][i] = "N/A"
    return p
def f1t(pred, t): return f1_score(y[t], pred[t], labels=sorted(set(y[t])), average="macro", zero_division=0)
def tot(pred, idx=None):
    idx = np.arange(N) if idx is None else idx
    return sum(TW[t] * f1_score(y[t][idx], pred[t][idx], labels=sorted(set(y[t])), average="macro", zero_division=0) for t in LAB)

llm = json.load(open("agent_cache/llm_t3_valid.json"))
cov = sum(1 for r in ids if r in llm)
print(f"LLM 覆蓋 {cov}/{N} valid 列")
base_c = cascade(base)
print(f"\nbase(champion): total={tot(base_c):.5f}  T3={f1t(base_c,'t3'):.4f} "
      f"[Clear {f1_score(y['t3'],base_c['t3'],labels=['Clear'],average='macro',zero_division=0):.3f} / "
      f"NotClear {f1_score(y['t3'],base_c['t3'],labels=['Not Clear'],average='macro',zero_division=0):.3f}]")

def apply_override(strategy, thr):
    t3new = base["t3"].copy()
    for i, rid in enumerate(ids):
        if base["t3"][i] not in ("Clear", "Not Clear"):  # only touch non-N/A T3
            continue
        j = llm.get(rid)
        if not j or j["label"] is None or j["conf"] < thr:
            continue
        if strategy == "replace":
            t3new[i] = j["label"]
        elif strategy == "flip_to_nc":   # only catch missed Not Clear
            if j["label"] == "Not Clear": t3new[i] = "Not Clear"
    p = {**{t: base[t] for t in LAB}, "t3": t3new}
    return cascade(p)

print("\n=== 策略 × 信心門檻(valid)===")
print(f"{'strategy':12s} {'thr':>3s}  total     ΔT3     T3      Clear   NotClear")
results = {}
for strat in ["flip_to_nc", "replace"]:
    for thr in [6, 7, 8, 9]:
        p = apply_override(strat, thr)
        ncf = f1_score(y['t3'], p['t3'], labels=['Not Clear'], average='macro', zero_division=0)
        clf = f1_score(y['t3'], p['t3'], labels=['Clear'], average='macro', zero_division=0)
        results[(strat, thr)] = p
        print(f"{strat:12s} {thr:>3d}  {tot(p):.5f}  {f1t(p,'t3')-f1t(base_c,'t3'):+.4f}  {f1t(p,'t3'):.4f}  {clf:.3f}   {ncf:.3f}")

# bootstrap best vs base
best = max(results, key=lambda k: tot(results[k]))
bp = results[best]
rng = np.random.RandomState(3); diffs = []
for _ in range(2000):
    idx = rng.randint(0, N, N); diffs.append(tot(bp, idx) - tot(base_c, idx))
diffs = np.array(diffs)
print(f"\n★ valid 最佳: {best} total={tot(bp):.5f} (base {tot(base_c):.5f}, Δ{tot(bp)-tot(base_c):+.5f})")
print(f"  bootstrap vs base: mean={diffs.mean():+.5f} CI=[{np.percentile(diffs,2.5):+.5f},{np.percentile(diffs,97.5):+.5f}] P(>base)={(diffs>0).mean():.3f}")
print("  注意:valid 是 train 分布,僅 sanity;真正看 AIdea。但若這裡就 ≤ base 或 CI 跨 0,基本不用上 AIdea。")
