"""Generate additional kNN-enhanced AIdea submissions (9-way no-FC1M, 6-way).
Same fusion pipeline as gen_rt_knn.py, but caches fold predictions + embeddings."""
import sys, os
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]

from esg_main import (
    _predict_checkpoint, apply_na_rule, load_dataframes, Config,
    extract_embeddings, compute_knn_ldl_probs, knn_fuse_probs,
    LABELS, NUM_LABELS,
)
import __main__
__main__.Config = Config

import math
import pandas as pd, numpy as np, torch
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

MODELS = {
    "rt_fc1_s0":  "runs/RT_FC1_s0",   "rt_fc1_s1":  "runs/RT_FC1_s1",   "rt_fc1_s2":  "runs/RT_FC1_s2",
    "rt_fc1r_s0": "runs/RT_FC1R_s0",  "rt_fc1r_s1": "runs/RT_FC1R_s1",  "rt_fc1r_s2": "runs/RT_FC1R_s2",
    "rt_fc1s_s0": "runs/RT_FC1S_s0",  "rt_fc1s_s1": "runs/RT_FC1S_s1",  "rt_fc1s_s2": "runs/RT_FC1S_s2",
    "rt_fc1m_s0": "runs/RT_FC1M_s0",  "rt_fc1m_s1": "runs/RT_FC1M_s1",  "rt_fc1m_s2": "runs/RT_FC1M_s2",
}
CACHE = Path("agent_cache/rt_test_probs"); CACHE.mkdir(parents=True, exist_ok=True)

model_fold_preds = {}
for key, run_dir in MODELS.items():
    fold_preds = []
    for fold in range(1, 6):
        cache_p = CACHE / f"{key}_f{fold}.npz"
        if cache_p.exists():
            d = np.load(cache_p)
            fold_preds.append({t: d[t] for t in LABELS})
            continue
        ckpt = f"{run_dir}/fold{fold}/best.pt"
        if not Path(ckpt).exists():
            break
        result = _predict_checkpoint(ckpt, test_df)
        if result is None:
            break
        probs = {t: np.asarray(result[1][t], dtype=np.float32) for t in LABELS}
        np.savez_compressed(cache_p, **probs)
        fold_preds.append(probs)
    if len(fold_preds) == 5:
        model_fold_preds[key] = fold_preds
        print(f"  {key}: OK", flush=True)
    else:
        print(f"  {key}: FAILED ({len(fold_preds)}/5)", flush=True)

emb_cache = Path("agent_cache/qwen3_embs_retrain.npz")
if emb_cache.exists():
    d = np.load(emb_cache); tr_embs, te_embs = d["tr"], d["te"]
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _INSTR = "Retrieve similar ESG disclosure texts that share the same evidence quality level."
    tr_embs = extract_embeddings(train_df["data"].tolist(), "Qwen/Qwen3-Embedding-0.6B",
                                 batch_size=16, device=device, instruction=_INSTR).numpy()
    te_embs = extract_embeddings(test_df["data"].tolist(), "Qwen/Qwen3-Embedding-0.6B",
                                 batch_size=16, device=device, instruction=_INSTR).numpy()
    np.savez_compressed(emb_cache, tr=tr_embs, te=te_embs)

knn_probs = compute_knn_ldl_probs(tr_embs, te_embs, train_df, k=5, test_df=test_df)
print("kNN probs computed", flush=True)

out_dir = Path("official_sub")
ALPHA = {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}
NC_BIAS = 0.1
NC_IDX = LABELS["t3"].index("Not Clear")


def gen_submission(name, model_keys):
    stacks = {t: np.mean([fp[t] for k in model_keys for fp in model_fold_preds[k]], axis=0)
              for t in LABELS}
    base_probs = {t: stacks[t].tolist() for t in LABELS}
    new_t3 = []
    for p in base_probs["t3"]:
        p2 = list(p); p2[NC_IDX] *= math.exp(NC_BIAS)
        s = sum(p2); new_t3.append([x / s for x in p2])
    base_probs["t3"] = new_t3
    preds = knn_fuse_probs(base_probs, knn_probs, alpha=ALPHA)
    preds = apply_na_rule(preds)
    df_out = pd.DataFrame({
        "id": test_df["id"],
        "promise_status": preds["t1"],
        "verification_timeline": preds["t4"],
        "evidence_status": preds["t2"],
        "evidence_quality": preds["t3"],
    })
    path = out_dir / f"{name}.csv"
    df_out.to_csv(path, index=False)
    na = int((df_out == "N/A").sum().sum())
    print(f"  -> {path} ({len(df_out)} rows, N/A={na})", flush=True)


R = ["rt_fc1r_s0", "rt_fc1r_s1", "rt_fc1r_s2"]
S = ["rt_fc1s_s0", "rt_fc1s_s1", "rt_fc1s_s2"]
F = ["rt_fc1_s0", "rt_fc1_s1", "rt_fc1_s2"]

gen_submission("aidea_rt_9way_knn", F + R + S)
gen_submission("aidea_rt_6way_knn", R + S)
print("Done!", flush=True)
