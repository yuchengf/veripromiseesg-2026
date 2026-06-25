"""Build an arbitrary per-task @384/@512 mix from RT_FC1R(@384)+RT_FC1R512(@512).
Usage:  python3 gen_mix.py t3          # only T3 from @512 (rest @384)
        python3 gen_mix.py t2,t3       # T2+T3 from @512
        python3 gen_mix.py t1,t2,t3,t4 # full @512 (reproduces aidea_rt512_3way)
        python3 gen_mix.py none        # champion (all @384)
After the pure-@512 AIdea probe reveals per-task @512 scores, pass the set of tasks
where @512 > @384 to build the AIdea-optimal mix. Shared kNN(T3 a0.4,NCbias.1)+clarity."""
import sys, os, math
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv_spec = sys.argv[1] if len(sys.argv) > 1 else "t3"
S512 = set() if sys.argv_spec.lower() == "none" else {t.strip() for t in sys.argv_spec.split(",")}
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]
from esg_main import (apply_na_rule, load_dataframes, Config,
                      compute_knn_ldl_probs, knn_fuse_probs, LABELS)
import __main__
__main__.Config = Config
import pandas as pd, numpy as np
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
N = len(test_df)


def stack(cache, key_fmt, tasks):
    out = {}
    for t in tasks:
        arrs = [np.load(cache / f"{key_fmt.format(i=i)}_f{f}.npz")[t]
                for i in range(3) for f in range(1, 6)]
        out[t] = np.mean(arrs, axis=0)
    return out

p384 = stack(Path("agent_cache/rt_test_probs"), "rt_fc1r_s{i}", ["t1", "t2", "t3", "t4"])
need512 = [t for t in ["t1", "t2", "t3", "t4"] if t in S512]
p512 = stack(Path("agent_cache/rt512_test_probs"), "rt_fc1r512_s{i}", need512) if need512 else {}
base = {t: (p512[t] if t in S512 else p384[t]) for t in LABELS}

NC_IDX = LABELS["t3"].index("Not Clear")
t3 = base["t3"].copy(); t3[:, NC_IDX] *= math.exp(0.1); t3 /= t3.sum(1, keepdims=True)
base["t3"] = t3
d = np.load("agent_cache/qwen3_embs_retrain.npz")
knn = compute_knn_ldl_probs(d["tr"], d["te"], train_df, k=5, test_df=test_df)
preds = apply_na_rule(knn_fuse_probs({t: base[t].tolist() for t in LABELS}, knn,
                                     alpha={"t1": 0, "t2": 0, "t3": 0.40, "t4": 0}))
df = pd.DataFrame({"id": test_df["id"], "promise_status": preds["t1"],
                   "verification_timeline": preds["t4"], "evidence_status": preds["t2"],
                   "evidence_quality": preds["t3"]})
CL3 = ["Clear", "Not Clear", "Misleading"]
clp = np.load("agent_cache/clarity_test_probs.npz")["probs"]; assert len(clp) == N
cca, ccf = clp.argmax(1), clp.max(1)
eq = df["evidence_quality"].tolist()
for i in range(N):
    if eq[i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading":
        eq[i] = CL3[cca[i]]
df["evidence_quality"] = eq
tag = "all384" if not S512 else "_".join(sorted(S512)) + "_512"
out = f"official_sub/aidea_mix_{tag}_knn_clarity.csv"
df.to_csv(out, index=False)
ch = pd.read_csv("official_sub/aidea_clarity_on_fc1r3.csv", keep_default_na=False)
diff = {c: int((df[c].values != ch[c].values).sum()) for c in df.columns if c != "id"}
print(f"-> {out}  (@512 任務={sorted(S512) or '無'}; vs champion 差異欄={diff})")
