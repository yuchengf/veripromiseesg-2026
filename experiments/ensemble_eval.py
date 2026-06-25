"""Test backbone-diversity ensemble on valid (no GPU — uses cached probs).
RoBERTa(FC1R) + {xlm-r, mmBERT, bge-m3} averaged per task, + kNN + clarity.
Does decorrelated backbone diversity beat RoBERTa-alone on valid? Bootstrap vs base.
Only the combos that convergently help here are worth training on retrain_data for AIdea."""
import numpy as np, pandas as pd
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

# backbone -> list of run names (its seeds)
BB = {
    "R(roberta)": ["FC1R_roberta_dc_kfold_t3nc3", "FC1R_roberta_dc_kfold_t3nc3_s1", "FC1R_roberta_dc_kfold_t3nc3_s2"],
    "X(xlmr)":    ["FX1R_xlmr_dc_kfold_t3nc3", "FX1R_xlmr_dc_kfold_t3nc3_s1", "FX1R_xlmr_dc_kfold_t3nc3_s2"],
    "M(mmbert)":  ["NB1R_mmbert_dc_kfold_t3nc3"],
    "G(bgem3)":   ["NG1R_bgem3_dc_kfold_t3nc3"],
}
def bbmean(runs):  # mean over seeds x folds for one backbone
    return {t: np.mean([np.load(C / f"{r}_f{f}.npz")[t] for r in runs for f in range(1, 6)], axis=0) for t in LAB}
P = {k: bbmean(v) for k, v in BB.items()}

def predict(base):
    t3 = base["t3"].copy(); t3[:, 1] *= np.exp(0.1); t3 /= t3.sum(1, keepdims=True); fused = 0.6 * t3 + 0.4 * knn
    pred = {t: np.array([LAB[t][i] for i in (fused if t == "t3" else base[t]).argmax(1)]) for t in LAB}
    for i in range(N):
        if pred["t3"][i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading": pred["t3"][i] = CL3[cca[i]]
    for i in range(N):
        if pred["t1"][i] == "No": pred["t2"][i] = pred["t3"][i] = pred["t4"][i] = "N/A"
        elif pred["t2"][i] == "No": pred["t3"][i] = "N/A"
    return pred
def ens(keys):  # equal-weight average of backbone-means, per task
    return {t: np.mean([P[k][t] for k in keys], axis=0) for t in LAB}
def f1t(pred, t): return f1_score(y[t], pred[t], labels=sorted(set(y[t])), average="macro", zero_division=0)
def tot(pred, idx=None):
    idx = np.arange(N) if idx is None else idx
    return sum(TW[t] * f1_score(y[t][idx], pred[t][idx], labels=sorted(set(y[t])), average="macro", zero_division=0) for t in LAB)

base = predict(ens(["R(roberta)"]))
print("=== 各 backbone 單獨 valid(+kNN+clarity)===")
for k in BB:
    p = predict(ens([k])); print(f"  {k:12s} total={tot(p):.5f}  T1={f1t(p,'t1'):.3f} T2={f1t(p,'t2'):.3f} T3={f1t(p,'t3'):.3f} T4={f1t(p,'t4'):.3f}")
print(f"\nbase = R alone = {tot(base):.5f}")
print("\n=== 集成(等權平均 backbone-means)vs R-alone ===")
combos = [["R(roberta)","X(xlmr)"], ["R(roberta)","M(mmbert)"], ["R(roberta)","G(bgem3)"],
          ["R(roberta)","X(xlmr)","G(bgem3)"], ["R(roberta)","X(xlmr)","M(mmbert)","G(bgem3)"]]
rng = np.random.RandomState(5)
for keys in combos:
    p = predict(ens(keys))
    diffs = np.array([tot(p, rng.randint(0, N, N)) - tot(base, rng.randint(0, N, N)) for _ in range(1500)])
    # paired
    rng2 = np.random.RandomState(11); pd_ = np.array([(lambda idx: tot(p, idx) - tot(base, idx))(rng2.randint(0, N, N)) for _ in range(1500)])
    name = "+".join(k[0] for k in keys)
    print(f"  {name:10s} total={tot(p):.5f} Δ{tot(p)-tot(base):+.5f}  bootstrap mean={pd_.mean():+.5f} CI=[{np.percentile(pd_,2.5):+.5f},{np.percentile(pd_,97.5):+.5f}] P(>R)={(pd_>0).mean():.3f}")
print("\n判定:只有「Δ>0 且 CI 不含 0(convergent-ish)」的組合,才值得花 GPU 在 retrain_data 上訓練做 AIdea。")
