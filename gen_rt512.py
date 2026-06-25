"""Inference + AIdea submission generation for the @512 RT models (trained on
retrain_data=2000). Mirrors gen_rt_more.py but for RT_*512 dirs, and emits every
combo (3/6/9/12-way) + kNN, plus a clarity-overlay variant for 3-way and 12-way.
Run after train_rt_512.sh finishes all 12 RT@512 models."""
import sys, os, math
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]

from esg_main import (
    _predict_checkpoint, apply_na_rule, load_dataframes, Config,
    compute_knn_ldl_probs, knn_fuse_probs, LABELS,
)
import __main__
__main__.Config = Config
import pandas as pd, numpy as np
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

# RT@512 models: seeds s0/s1/s2 = 42/123/456
MODELS = {f"rt_{r}512_s{i}": f"runs/RT_{R}512_s{i}"
          for r, R in [("fc1", "FC1"), ("fc1r", "FC1R"), ("fc1s", "FC1S"), ("fc1m", "FC1M")]
          for i in range(3)}
CACHE = Path("agent_cache/rt512_test_probs"); CACHE.mkdir(parents=True, exist_ok=True)

model_fold_preds = {}
for key, run_dir in MODELS.items():
    fold_preds = []
    for fold in range(1, 6):
        cache_p = CACHE / f"{key}_f{fold}.npz"
        if cache_p.exists():
            d = np.load(cache_p); fold_preds.append({t: d[t] for t in LABELS}); continue
        ckpt = f"{run_dir}/fold{fold}/best.pt"
        if not Path(ckpt).exists():
            break
        result = _predict_checkpoint(ckpt, test_df)
        if result is None:
            break
        probs = {t: np.asarray(result[1][t], dtype=np.float32) for t in LABELS}
        np.savez_compressed(cache_p, **probs); fold_preds.append(probs)
    if len(fold_preds) == 5:
        model_fold_preds[key] = fold_preds; print(f"  {key}: OK", flush=True)
    else:
        print(f"  {key}: MISSING ({len(fold_preds)}/5)", flush=True)

d = np.load("agent_cache/qwen3_embs_retrain.npz"); tr_embs, te_embs = d["tr"], d["te"]
knn_probs = compute_knn_ldl_probs(tr_embs, te_embs, train_df, k=5, test_df=test_df)
print("kNN probs computed", flush=True)

# clarity overlay (AIdea test): override non-N/A T3 with confident clarity head
CL3 = ["Clear", "Not Clear", "Misleading"]
clp = np.load("agent_cache/clarity_test_probs.npz")["probs"]
assert len(clp) == len(test_df), f"clarity_test_probs ({len(clp)}) != test ({len(test_df)}) — alignment broken"
cca, ccf = clp.argmax(1), clp.max(1)

out_dir = Path("official_sub")
ALPHA = {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}
NC_BIAS = 0.1
NC_IDX = LABELS["t3"].index("Not Clear")


def build(model_keys):
    avail = [k for k in model_keys if k in model_fold_preds]
    if not avail:
        return None
    stacks = {t: np.mean([fp[t] for k in avail for fp in model_fold_preds[k]], axis=0) for t in LABELS}
    base = {t: stacks[t].tolist() for t in LABELS}
    new_t3 = []
    for p in base["t3"]:
        p2 = list(p); p2[NC_IDX] *= math.exp(NC_BIAS); s = sum(p2); new_t3.append([x / s for x in p2])
    base["t3"] = new_t3
    preds = apply_na_rule(knn_fuse_probs(base, knn_probs, alpha=ALPHA))
    return pd.DataFrame({"id": test_df["id"], "promise_status": preds["t1"],
                         "verification_timeline": preds["t4"], "evidence_status": preds["t2"],
                         "evidence_quality": preds["t3"]})


def add_clarity(df):
    df = df.copy(); eq = df["evidence_quality"].tolist()
    for i in range(len(df)):
        if eq[i] in CL3 and ccf[i] >= 0.7 and CL3[cca[i]] != "Misleading":
            eq[i] = CL3[cca[i]]
    df["evidence_quality"] = eq; return df


def save(df, name):
    if df is None:
        print(f"  [skip {name}: models missing]", flush=True); return
    p = out_dir / f"{name}.csv"; df.to_csv(p, index=False)
    na = int((df == "N/A").sum().sum())
    print(f"  -> {p} ({len(df)} rows, N/A={na})", flush=True)


F = [f"rt_fc1512_s{i}" for i in range(3)]
R = [f"rt_fc1r512_s{i}" for i in range(3)]
S = [f"rt_fc1s512_s{i}" for i in range(3)]
M = [f"rt_fc1m512_s{i}" for i in range(3)]

combos = {"3way": R, "6way": R + S, "9way": F + R + S, "12way": F + R + S + M}
for name, keys in combos.items():
    df = build(keys)
    save(df, f"aidea_rt512_{name}_knn")
    if name in ("3way", "12way") and df is not None:
        save(add_clarity(df), f"aidea_rt512_{name}_knn_clarity")
print("Done! All @512 AIdea candidates in official_sub/", flush=True)
