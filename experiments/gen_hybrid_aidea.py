"""AIdea candidates that swap @512-trained tasks into the champion (@384 RT_FC1R).
Corrected analysis (2026-06-16): the ROBUST lever is T3-only @512 (bootstrap P=0.990,
CI excludes 0, structural: less evidence truncation). The full T2+T3 hybrid is less
robust (T2@512 noisy). So PRIMARY probe = swap-T3-only; hybrid kept as a control.

  swap-T3 : T1/T2/T4 from @384 RT_FC1R, T3 from @512 RT_FC1R512   <- PRIMARY
  hybrid  : T1/T4   from @384 RT_FC1R, T2/T3 from @512 RT_FC1R512  <- control

Run AFTER gen_rt512.py has cached rt512_test_probs."""
import sys, os, math
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]
from esg_main import (apply_na_rule, load_dataframes, Config,
                      compute_knn_ldl_probs, knn_fuse_probs, LABELS)
import __main__
__main__.Config = Config
import pandas as pd, numpy as np
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
N = len(test_df)
C384 = Path("agent_cache/rt_test_probs")      # @384 RT_FC1R
C512 = Path("agent_cache/rt512_test_probs")    # @512 RT_FC1R512


def stack(cache, key_fmt, tasks):
    out = {}
    for t in tasks:
        arrs = []
        for i in range(3):
            for f in range(1, 6):
                p = cache / f"{key_fmt.format(i=i)}_f{f}.npz"
                if not p.exists():
                    raise FileNotFoundError(f"缺 {p} — gen_rt512 跑完了嗎?")
                arrs.append(np.load(p)[t])
        out[t] = np.mean(arrs, axis=0)
    return out

p384 = stack(C384, "rt_fc1r_s{i}", ["t1", "t2", "t3", "t4"])
p512 = stack(C512, "rt_fc1r512_s{i}", ["t2", "t3"])

# shared kNN(T3) + clarity
d = np.load("agent_cache/qwen3_embs_retrain.npz")
knn = compute_knn_ldl_probs(d["tr"], d["te"], train_df, k=5, test_df=test_df)
NC_IDX = LABELS["t3"].index("Not Clear")
CL3 = ["Clear", "Not Clear", "Misleading"]
clp = np.load("agent_cache/clarity_test_probs.npz")["probs"]
assert len(clp) == N
cca, ccf = clp.argmax(1), clp.max(1)
out = Path("official_sub")


def build(base):
    t3 = base["t3"].copy(); t3[:, NC_IDX] *= math.exp(0.1); t3 /= t3.sum(1, keepdims=True)
    base = {**base, "t3": t3}
    base_l = {t: base[t].tolist() for t in LABELS}
    preds = apply_na_rule(knn_fuse_probs(base_l, knn, alpha={"t1": 0, "t2": 0, "t3": 0.40, "t4": 0}))
    return pd.DataFrame({"id": test_df["id"], "promise_status": preds["t1"],
                        "verification_timeline": preds["t4"], "evidence_status": preds["t2"],
                        "evidence_quality": preds["t3"]})


def add_clarity(df):
    df = df.copy(); eq = df["evidence_quality"].tolist()
    for i in range(N):
        if eq[i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading":
            eq[i] = CL3[cca[i]]
    df["evidence_quality"] = eq; return df


# PRIMARY: swap T3 only (T1/T2/T4 @384, T3 @512)
swapT3 = {"t1": p384["t1"], "t2": p384["t2"], "t3": p512["t3"], "t4": p384["t4"]}
df = add_clarity(build(swapT3))
df.to_csv(out / "aidea_swapT3_512_knn_clarity.csv", index=False)
print(f"★ PRIMARY -> official_sub/aidea_swapT3_512_knn_clarity.csv (valid 對應 0.67214, P=0.990)")

# control: full hybrid (T2/T3 @512)
hyb = {"t1": p384["t1"], "t2": p512["t2"], "t3": p512["t3"], "t4": p384["t4"]}
dfh = add_clarity(build(hyb))
dfh.to_csv(out / "aidea_hybrid_t14_384_t23_512_knn_clarity.csv", index=False)
print(f"  control -> official_sub/aidea_hybrid_t14_384_t23_512_knn_clarity.csv (valid 0.67286, P=0.897)")

# diff vs champion (sanity: should differ only in T3 for swapT3)
ch = pd.read_csv("official_sub/aidea_clarity_on_fc1r3.csv", keep_default_na=False)
dd = {c: int((df[c].values != ch[c].values).sum()) for c in df.columns if c != "id"}
print(f"  swapT3 vs champion 各欄差異: {dd}  (應只 evidence_quality 變)")
