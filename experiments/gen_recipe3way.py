"""Build a coherent single-recipe 3-way AIdea submission (+kNN+NCbias+clarity),
same pipeline as the champion but a different recipe. For testing whether a
higher-CV recipe (FC1M/FC1) also wins on public = convergent, or is CV-only = overfit.
Usage: python3 gen_recipe3way.py fc1m   (or fc1, fc1s)"""
import sys, os, math
os.chdir("/home/yucheng/Desktop/ESG")
rc = sys.argv[1] if len(sys.argv) > 1 else "fc1m"
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]
from esg_main import apply_na_rule, load_dataframes, Config, compute_knn_ldl_probs, knn_fuse_probs, LABELS
import __main__
__main__.Config = Config
import pandas as pd, numpy as np
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
N = len(test_df)
C = Path("agent_cache/rt_test_probs")
base = {t: np.mean([np.load(C / f"rt_{rc}_s{i}_f{f}.npz")[t]
                    for i in range(3) for f in range(1, 6)], axis=0) for t in LABELS}
nc = LABELS["t3"].index("Not Clear")
t3 = base["t3"].copy(); t3[:, nc] *= math.exp(0.1); t3 /= t3.sum(1, keepdims=True); base["t3"] = t3
d = np.load("agent_cache/qwen3_embs_retrain.npz")
knn = compute_knn_ldl_probs(d["tr"], d["te"], train_df, k=5, test_df=test_df)
preds = apply_na_rule(knn_fuse_probs({t: base[t].tolist() for t in LABELS}, knn,
                                     alpha={"t1": 0, "t2": 0, "t3": 0.40, "t4": 0}))
df = pd.DataFrame({"id": test_df["id"], "promise_status": preds["t1"], "verification_timeline": preds["t4"],
                   "evidence_status": preds["t2"], "evidence_quality": preds["t3"]})
CL3 = ["Clear", "Not Clear", "Misleading"]
clp = np.load("agent_cache/clarity_test_probs.npz")["probs"]; assert len(clp) == N
cca, ccf = clp.argmax(1), clp.max(1)
eq = df["evidence_quality"].tolist()
for i in range(N):
    if eq[i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading":
        eq[i] = CL3[cca[i]]
df["evidence_quality"] = eq
out = f"official_sub/aidea_rt_{rc}3_knn_clarity.csv"
df.to_csv(out, index=False)
print(f"-> {out} ({len(df)} rows, N/A={int((df=='N/A').sum().sum())})")
