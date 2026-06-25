"""Evaluate a single 5-fold run on valid 399 (official-aligned scoring).
Usage: python eval_single_run.py RUN_NAME
Caches fold probs in agent_cache/valid_probs/ (same format as pertask_valid_eval).
"""
import os, sys
os.chdir("/home/yucheng/Desktop/ESG")
RUN = sys.argv[1]
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "final_data"]

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

from esg_main import _predict_checkpoint, load_dataframes, LABELS, Config
import __main__
__main__.Config = Config

TASK_W = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}
SOL_COL = {"t1": "promise_status", "t2": "evidence_status",
           "t3": "evidence_quality", "t4": "verification_timeline"}

train_df, valid_df = load_dataframes("final_data")
sol = pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
sol["verification_timeline"] = sol["verification_timeline"].replace(
    "longer_than_5_years", "more_than_5_years")
y = {t: sol[SOL_COL[t]].astype(str).tolist() for t in LABELS}
N = len(sol)

CACHE = Path("agent_cache/valid_probs"); CACHE.mkdir(parents=True, exist_ok=True)
fold_probs = []
for fold in range(1, 6):
    p = CACHE / f"{RUN}_f{fold}.npz"
    if p.exists():
        d = np.load(p)
        fold_probs.append({t: d[t] for t in LABELS})
        continue
    ckpt = f"runs/{RUN}/fold{fold}/best.pt"
    res = _predict_checkpoint(ckpt, valid_df)
    if res is None:
        sys.exit(f"missing/failed {ckpt}")
    probs = {t: np.asarray(res[1][t], dtype=np.float32) for t in LABELS}
    np.savez_compressed(p, **probs)
    fold_probs.append(probs)

avg = {t: np.mean([fp[t] for fp in fold_probs], axis=0) for t in LABELS}
preds = {t: [LABELS[t][i] for i in np.argmax(avg[t], axis=1)] for t in LABELS}
for i in range(N):
    if preds["t1"][i] == "No":
        preds["t2"][i] = preds["t3"][i] = preds["t4"][i] = "N/A"
    elif preds["t2"][i] == "No":
        preds["t3"][i] = "N/A"

total = 0.0
parts = []
for t in LABELS:
    f = f1_score(y[t], preds[t], labels=sorted(set(y[t])), average="macro", zero_division=0)
    total += TASK_W[t] * f
    parts.append(f"{t.upper()}={f:.4f}")
print(f"[eval_single] {RUN}: {' '.join(parts)} | total={total:.5f}")
print(f"[eval_single] reference FC1R seed42 single ~0.66 (3-seed group 0.66342)")
