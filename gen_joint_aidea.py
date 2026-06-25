"""Apply JOINT/structured decoding to the AIdea (retrain) probs, on the mix's per-task
sources (T1/T3@384, T2/T4@512) + kNN + clarity. Backward dependency: downstream N/A
probs can correct the T1/T2 gate. Structural -> should transfer.
Usage: python gen_joint_aidea.py [wgate]   (default 2.0)"""
import sys, os, math
os.chdir(os.path.dirname(os.path.abspath(__file__)))
WGATE = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]
from esg_main import load_dataframes, Config, compute_knn_ldl_probs, knn_fuse_probs, LABELS
import __main__; __main__.Config = Config
import numpy as np, pandas as pd
from pathlib import Path
train_df, test_df = load_dataframes("retrain_data"); N = len(test_df)
def stack(cache, key, tasks):
    return {t: np.mean([np.load(Path(cache) / f"{key}_s{i}_f{f}.npz")[t] for i in range(3) for f in range(1, 6)], axis=0) for t in tasks}
p384 = stack("agent_cache/rt_test_probs", "rt_fc1r", ["t1", "t3"])       # T1, T3 @384
p512 = stack("agent_cache/rt512_test_probs", "rt_fc1r512", ["t2", "t4"])  # T2, T4 @512
p1, p2, p4 = p384["t1"], p512["t2"], p512["t4"]
# T3: NCbias + kNN
nc = LABELS["t3"].index("Not Clear")
t3p = p384["t3"].copy(); t3p[:, nc] *= math.exp(0.1); t3p /= t3p.sum(1, keepdims=True)
d = np.load("agent_cache/qwen3_embs_retrain.npz")
knnp = compute_knn_ldl_probs(d["tr"], d["te"], train_df, k=5, test_df=test_df)
knn_t3 = np.asarray(knnp["t3"])  # (N,4) over [Clear,NotClear,Misleading,N/A]
t3p = 0.6 * t3p + 0.4 * knn_t3
LG = lambda x: np.log(np.clip(x, 1e-9, 1))
CL3 = ["Clear", "Not Clear", "Misleading"]
clp = np.load("agent_cache/clarity_test_probs.npz")["probs"]; cca, ccf = clp.argmax(1), clp.max(1)
T4_TL = [0, 1, 2, 3]; T3_CONTENT = [0, 1, 2]
t1 = np.empty(N, object); t2 = np.empty(N, object); t3 = np.empty(N, object); t4 = np.empty(N, object)
for i in range(N):
    t4b = T4_TL[np.argmax([p4[i, k] for k in T4_TL])]
    t3b = T3_CONTENT[np.argmax([t3p[i, k] for k in T3_CONTENT])]
    s_no = WGATE * LG(p1[i, 1]) + WGATE * LG(p2[i, 2]) + LG(t3p[i, 3]) + LG(p4[i, 4])
    s_yn = WGATE * LG(p1[i, 0]) + WGATE * LG(p2[i, 1]) + LG(t3p[i, 3]) + LG(p4[i, t4b])
    s_yy = WGATE * LG(p1[i, 0]) + WGATE * LG(p2[i, 0]) + LG(t3p[i, t3b]) + LG(p4[i, t4b])
    bb = int(np.argmax([s_no, s_yn, s_yy]))
    if bb == 0: t1[i], t2[i], t3[i], t4[i] = "No", "N/A", "N/A", "N/A"
    elif bb == 1: t1[i], t2[i], t3[i], t4[i] = "Yes", "No", "N/A", LABELS["t4"][t4b]
    else: t1[i], t2[i], t3[i], t4[i] = "Yes", "Yes", LABELS["t3"][t3b], LABELS["t4"][t4b]
# clarity overlay on non-N/A T3
for i in range(N):
    if t3[i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading": t3[i] = CL3[cca[i]]
df = pd.DataFrame({"id": test_df["id"], "promise_status": t1, "verification_timeline": t4,
                   "evidence_status": t2, "evidence_quality": t3})
out = f"official_sub/aidea_joint_w{WGATE}.csv"; df.to_csv(out, index=False)
mix = pd.read_csv("official_sub/aidea_anchored_t2_t4512.csv", keep_default_na=False)
diff = {c: int((df[c].values != mix[c].values).sum()) for c in ["promise_status","evidence_status","evidence_quality","verification_timeline"]}
print(f"-> {out} (wgate={WGATE}) | rows={len(df)} N/A={int((df=='N/A').sum().sum())}")
print(f"   vs mix(0.6232) 差異欄: {diff}  (gate 修正會連動 N/A → 多欄變動正常)")
