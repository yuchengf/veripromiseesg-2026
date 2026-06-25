"""ESG domain-feature re-ranker for T3 (Clear vs Not Clear) on valid (CPU, no GPU).
Rule-based version of the LLM idea: Clear = concrete (metrics/targets/verification/data);
Not Clear = vague aspirational language with no specifics. Override base champion T3 where
features strongly disagree; sweep thresholds; bootstrap vs base. If it fails on valid like
the 14B LLM did, T3 is label-limited (confirmed from another angle)."""
import re, numpy as np, pandas as pd
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
for i in range(N):
    if base["t3"][i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading": base["t3"][i] = CL3[cca[i]]

METRIC = re.compile(r'\d+\.?\d*\s*(%|％|噸|公噸|tco2e|kwh|mwh|gwh|度|億|萬|千瓦|減量|排放)', re.I)
TARGET = re.compile(r'(目標|減少\d|提升\d|降低\d|達成|基準年|淨零|碳中和|20\d\d\s*年)', re.I)
VERIFY = re.compile(r'(iso\s*\d|gri|sasb|tcfd|查證|確信|第三方|認證|盤查|簽證)', re.I)
NUM = re.compile(r'\d+\.?\d*\s*[%％]')
VAGUE = re.compile(r'(致力於|持續努力|核心原則|積極推動|期望|承諾將|努力|秉持|致力|期許|逐步)')
def clearscore(txt):
    txt = str(txt)
    return (2 * len(METRIC.findall(txt)) + 1.5 * len(TARGET.findall(txt)) + 2 * len(VERIFY.findall(txt))
            + len(NUM.findall(txt)) - 1.0 * len(VAGUE.findall(txt)))
cs = np.array([clearscore(t) for t in vdata["data"]])

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
base_c = cascade(base)
print(f"base(champion) total={tot(base_c):.5f} T3={f1t(base_c,'t3'):.4f}")
print(f"clearscore 分布: p10={np.percentile(cs,10):.1f} p50={np.percentile(cs,50):.1f} p90={np.percentile(cs,90):.1f}")

print("\n=== flip base-Clear→NotClear when clearscore <= lo  (catch vague 'Clear') ===")
rng = np.random.RandomState(9)
best = (tot(base_c), "base")
for lo in [-1, 0, 1, 2]:
    t3n = base["t3"].copy()
    for i in range(N):
        if base["t3"][i] == "Clear" and cs[i] <= lo: t3n[i] = "Not Clear"
    p = cascade({**{t: base[t] for t in LAB}, "t3": t3n})
    ncf = f1_score(y['t3'], p['t3'], labels=['Not Clear'], average='macro', zero_division=0)
    pd_ = np.array([(lambda idx: tot(p, idx) - tot(base_c, idx))(rng.randint(0, N, N)) for _ in range(1500)])
    print(f"  lo<={lo}: total={tot(p):.5f} Δ{tot(p)-tot(base_c):+.5f} NotClearF1={ncf:.3f} P(>base)={(pd_>0).mean():.3f}")
    if tot(p) > best[0]: best = (tot(p), f"lo<={lo}")
print(f"\n★ 最佳: {best[1]} = {best[0]:.5f} (base {tot(base_c):.5f})")
print("判定:若 ≤ base 或 CI/P 無顯著 → domain re-ranker 也無效 → T3 確認 label-limited(第 N 個角度)。")
