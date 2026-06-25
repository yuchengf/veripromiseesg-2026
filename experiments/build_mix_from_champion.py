"""Build a clean per-task @384/@512 mix ANCHORED to the champion file, so @384 tasks
exactly match the AIdea-confirmed champion (0.6188) and only the @512-swapped tasks
change. Avoids the reconstruction drift in gen_mix (rt_fc1r cache != champion build).

Usage:  python3 build_mix_from_champion.py t3       # swap only T3 to @512  (PRIMARY)
        python3 build_mix_from_champion.py t2,t3    # swap T2+T3
        python3 build_mix_from_champion.py none      # = champion (sanity, 0 diff)

@384 tasks  -> champion's own labels (exact match).
@512 tasks  -> RT_FC1R512: T1/T2/T4 = model argmax; T3 = NC-bias + kNN(a0.4) + clarity.
Then re-apply N/A cascade (T1=No->T2/T3/T4=N/A; T2=No->T3=N/A)."""
import sys, os, math
os.chdir("/home/yucheng/Desktop/ESG")
spec = sys.argv[1] if len(sys.argv) > 1 else "t3"
S512 = set() if spec.lower() == "none" else {t.strip() for t in spec.split(",")}
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]
from esg_main import load_dataframes, Config, compute_knn_ldl_probs, knn_fuse_probs, LABELS
import __main__
__main__.Config = Config
import pandas as pd, numpy as np
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
N = len(test_df)
COL = {"t1": "promise_status", "t2": "evidence_status", "t3": "evidence_quality", "t4": "verification_timeline"}
ch = pd.read_csv("official_sub/aidea_clarity_on_fc1r3.csv", keep_default_na=False)


def stack512(tasks):
    C = Path("agent_cache/rt512_test_probs")
    return {t: np.mean([np.load(C / f"rt_fc1r512_s{i}_f{f}.npz")[t]
                        for i in range(3) for f in range(1, 6)], axis=0) for t in tasks}

# --- compute @512 labels (pre-cascade) for the swapped tasks ---
p512 = stack512(["t1", "t2", "t3", "t4"]) if S512 else {}
lab512 = {}
for t in ["t1", "t2", "t4"]:
    if t in S512:
        lab512[t] = np.array([LABELS[t][i] for i in p512[t].argmax(1)])
if "t3" in S512:
    t3 = p512["t3"].copy(); nc = LABELS["t3"].index("Not Clear")
    t3[:, nc] *= math.exp(0.1); t3 /= t3.sum(1, keepdims=True)
    d = np.load("agent_cache/qwen3_embs_retrain.npz")
    knn = compute_knn_ldl_probs(d["tr"], d["te"], train_df, k=5, test_df=test_df)
    fused = knn_fuse_probs({"t3": t3.tolist(), "t1": p512["t1"].tolist(), "t2": p512["t2"].tolist(),
                            "t4": p512["t4"].tolist()}, knn, alpha={"t1": 0, "t2": 0, "t3": 0.40, "t4": 0})
    # knn_fuse_probs returns labels already; take its t3 labels (pre-cascade)
    t3lab = np.array(fused["t3"])
    CL3 = ["Clear", "Not Clear", "Misleading"]
    clp = np.load("agent_cache/clarity_test_probs.npz")["probs"]; cca, ccf = clp.argmax(1), clp.max(1)
    for i in range(N):
        if t3lab[i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading":
            t3lab[i] = CL3[cca[i]]
    lab512["t3"] = t3lab

# --- assemble labels: @384 from champion, @512 from lab512 ---
lab = {}
for t in ["t1", "t2", "t3", "t4"]:
    lab[t] = lab512[t].copy() if t in S512 else ch[COL[t]].values.astype(object).copy()

# --- re-apply N/A cascade ---
for i in range(N):
    if lab["t1"][i] == "No":
        lab["t2"][i] = lab["t3"][i] = lab["t4"][i] = "N/A"
    elif lab["t2"][i] == "No":
        lab["t3"][i] = "N/A"

out_df = pd.DataFrame({"id": ch["id"], "promise_status": lab["t1"], "verification_timeline": lab["t4"],
                       "evidence_status": lab["t2"], "evidence_quality": lab["t3"]})
tag = "champ" if not S512 else "_".join(sorted(S512)) + "512"
path = f"official_sub/aidea_anchored_{tag}.csv"
out_df.to_csv(path, index=False)
diff = {COL[t]: int((out_df[COL[t]].values != ch[COL[t]].values).sum()) for t in ["t1", "t2", "t3", "t4"]}
print(f"-> {path}")
print(f"   @512 任務={sorted(S512) or '無'};vs champion 差異欄={diff}")
print(f"   (應只有 @512 任務 + 受級聯影響者改變;@384 任務鎖定 champion)")
