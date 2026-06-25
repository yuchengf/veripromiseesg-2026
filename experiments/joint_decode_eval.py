"""Structured/JOINT decoding that exploits the subtask dependency in BOTH directions.

Current pipeline = greedy top-down: argmax T1; if No -> force T2/T3/T4=N/A; if T2=No -> T3=N/A.
This only uses the dependency FORWARD (gate decides downstream). It IGNORES the model's own
downstream N/A probabilities — so a barely-No T1 wrongly N/As strong downstream evidence.

JOINT decode: enumerate the 3 valid branches per row and pick the one maximizing the joint
(log-)prob across all four tasks, using the model's actual N/A probs. Downstream confidence
can now pull the gate (backward signal). Structural -> should transfer. Tested on valid."""
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
def avg(rc): return {t: np.mean([np.load(C / f"{rc}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for s in ("", "_s1", "_s2") for f in range(1, 6)], axis=0) for t in LAB}
b = avg("FC1R")
# T3 prob with NCbias + kNN fusion (same as champion), keep full 4-dim distribution
t3p = b["t3"].copy(); t3p[:, 1] *= np.exp(0.1); t3p /= t3p.sum(1, keepdims=True)
t3p = 0.6 * t3p + 0.4 * knn
p1, p2, p4 = b["t1"], b["t2"], b["t4"]
LG = lambda x: np.log(np.clip(x, 1e-9, 1))

def clarity(t3labels):
    t3labels = t3labels.copy()
    for i in range(N):
        if t3labels[i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading": t3labels[i] = CL3[cca[i]]
    return t3labels
def f1t(pred, t): return f1_score(y[t], pred[t], labels=sorted(set(y[t])), average="macro", zero_division=0)
def tot(pred, idx=None):
    idx = np.arange(N) if idx is None else idx
    return sum(TW[t] * f1_score(y[t][idx], pred[t][idx], labels=sorted(set(y[t])), average="macro", zero_division=0) for t in LAB)

# ---- GREEDY (current pipeline) ----
def greedy():
    pred = {"t1": np.array([LAB["t1"][i] for i in p1.argmax(1)]),
            "t2": np.array([LAB["t2"][i] for i in p2.argmax(1)]),
            "t3": np.array([LAB["t3"][i] for i in t3p.argmax(1)]),
            "t4": np.array([LAB["t4"][i] for i in p4.argmax(1)])}
    pred["t3"] = clarity(pred["t3"])
    for i in range(N):
        if pred["t1"][i] == "No": pred["t2"][i] = pred["t3"][i] = pred["t4"][i] = "N/A"
        elif pred["t2"][i] == "No": pred["t3"][i] = "N/A"
    return pred

# ---- JOINT decode (weight w on gate log-probs) ----
T4_TL = [0, 1, 2, 3]  # non-N/A timeline indices
T3_CONTENT = [0, 1, 2]  # Clear/NotClear/Misleading
def joint(wgate=1.0):
    t1 = np.empty(N, object); t2 = np.empty(N, object); t3 = np.empty(N, object); t4 = np.empty(N, object)
    for i in range(N):
        t4_best = T4_TL[np.argmax([p4[i, k] for k in T4_TL])]; t4_bs = LG(p4[i, t4_best])
        t3_best = T3_CONTENT[np.argmax([t3p[i, k] for k in T3_CONTENT])]; t3_bs = LG(t3p[i, t3_best])
        s_no = wgate * LG(p1[i, 1]) + wgate * LG(p2[i, 2]) + LG(t3p[i, 3]) + LG(p4[i, 4])
        s_yn = wgate * LG(p1[i, 0]) + wgate * LG(p2[i, 1]) + LG(t3p[i, 3]) + t4_bs
        s_yy = wgate * LG(p1[i, 0]) + wgate * LG(p2[i, 0]) + t3_bs + t4_bs
        bestb = np.argmax([s_no, s_yn, s_yy])
        if bestb == 0:
            t1[i], t2[i], t3[i], t4[i] = "No", "N/A", "N/A", "N/A"
        elif bestb == 1:
            t1[i], t2[i], t3[i], t4[i] = "Yes", "No", "N/A", LAB["t4"][t4_best]
        else:
            t1[i], t2[i], t3[i], t4[i] = "Yes", "Yes", LAB["t3"][t3_best], LAB["t4"][t4_best]
    pred = {"t1": t1, "t2": t2, "t3": clarity(t3), "t4": t4}
    return pred

g = greedy()
print(f"GREEDY (current): total={tot(g):.5f}  T1={f1t(g,'t1'):.4f} T2={f1t(g,'t2'):.4f} T3={f1t(g,'t3'):.4f} T4={f1t(g,'t4'):.4f}")
print("\n=== JOINT decode (gate weight sweep) vs greedy ===")
rng = np.random.RandomState(13)
best = (tot(g), "greedy", g)
for w in [0.5, 1.0, 1.5, 2.0, 3.0]:
    p = joint(w)
    pd_ = np.array([(lambda idx: tot(p, idx) - tot(g, idx))(rng.randint(0, N, N)) for _ in range(1500)])
    print(f"  wgate={w}: total={tot(p):.5f} Δ{tot(p)-tot(g):+.5f}  T1={f1t(p,'t1'):.4f} T2={f1t(p,'t2'):.4f} T3={f1t(p,'t3'):.4f} T4={f1t(p,'t4'):.4f}  P(>greedy)={(pd_>0).mean():.3f}")
    if tot(p) > best[0]: best = (tot(p), f"joint w={w}", p)
print(f"\n★ 最佳: {best[1]} = {best[0]:.5f} (greedy {tot(g):.5f}, Δ{best[0]-tot(g):+.5f})")
# how many rows differ from greedy
if best[1] != "greedy":
    diff = sum(1 for t in LAB for i in range(N) if best[2][t][i] != g[t][i])
    print(f"  改變的 (task,row) 數: {diff}")
print("判定:joint 用了下游 N/A 機率修正 gate。若 Δ>0 且 P 高 → 結構性槓桿(該轉移)→ 上 AIdea 驗證。")
