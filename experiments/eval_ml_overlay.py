"""Evaluate the multilingual MX1R model as a Misleading-overlay onto the frozen
12-way + kNN pipeline, on the Chinese valid 399.

valid 399 has 0 true Misleading, so the overlay can only keep (0 overrides) or
HURT (false positives). We therefore measure, per Misleading-confidence
threshold tau:
  - #valid rows MX1R flags as Misleading  (false-positive count = precision proxy)
  - resulting valid weighted score vs frozen 0.67882
A high tau with ~0 valid false positives = safe to probe on AIdea (which DOES
contain Misleading). If even high tau fires on many valid rows -> low precision
-> reject before spending an AIdea upload.
"""
import os, sys
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "final_data"]
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from esg_main import _predict_checkpoint, load_dataframes, LABELS, Config
import __main__; __main__.Config = Config

MX1R = "MX1R_xlmr_ml_dc_kfold_t3nc3"
TASK_W = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}
SOL_COL = {"t1": "promise_status", "t2": "evidence_status",
           "t3": "evidence_quality", "t4": "verification_timeline"}
K, A3, NC_BIAS = 5, 0.40, 0.1
GROUPS = ["FC1", "FC1R", "FC1S", "FC1M"]; SEEDS = ["", "_s1", "_s2"]

train_df, valid_df = load_dataframes("final_data")
sol = pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
sol["verification_timeline"] = sol["verification_timeline"].replace(
    "longer_than_5_years", "more_than_5_years")
y = {t: sol[SOL_COL[t]].astype(str).tolist() for t in LABELS}
N = len(sol)
usage = sol["Usage"].values
mis_idx = LABELS["t3"].index("Misleading")

CACHE = Path("agent_cache/valid_probs"); CACHE.mkdir(parents=True, exist_ok=True)


def get_fold_probs(run):
    out = []
    for fold in range(1, 6):
        p = CACHE / f"{run}_f{fold}.npz"
        if p.exists():
            d = np.load(p); out.append({t: d[t] for t in LABELS}); continue
        ckpt = f"runs/{run}/fold{fold}/best.pt"
        res = _predict_checkpoint(ckpt, valid_df)
        if res is None: sys.exit(f"missing {ckpt}")
        probs = {t: np.asarray(res[1][t], dtype=np.float32) for t in LABELS}
        np.savez_compressed(p, **probs); out.append(probs)
    return out


# ---- MX1R multilingual model valid probs ----
mx = get_fold_probs(MX1R)
mx_avg = {t: np.mean([f[t] for f in mx], axis=0) for t in LABELS}
p_mis = mx_avg["t3"][:, mis_idx]                      # P(Misleading) per valid row
mx_pred = {t: [LABELS[t][i] for i in np.argmax(mx_avg[t], axis=1)] for t in LABELS}


def score(preds, mask=None):
    idx = np.arange(N) if mask is None else np.where(mask)[0]
    tot = 0.0
    for t in LABELS:
        yt = [y[t][i] for i in idx]; pt = [preds[t][i] for i in idx]
        tot += TASK_W[t] * f1_score(yt, pt, labels=sorted(set(yt)),
                                    average="macro", zero_division=0)
    return tot


# standalone MX1R score (cascade-applied)
mp = {t: list(mx_pred[t]) for t in LABELS}
for i in range(N):
    if mp["t1"][i] == "No": mp["t2"][i] = mp["t3"][i] = mp["t4"][i] = "N/A"
    elif mp["t2"][i] == "No": mp["t3"][i] = "N/A"
print(f"[MX1R standalone] valid total={score(mp):.5f}  "
      f"predicts Misleading on {int((np.array(mx_pred['t3'])=='Misleading').sum())}/{N} valid rows")
print(f"[MX1R] P(Misleading) stats: max={p_mis.max():.3f} "
      f"#>0.5={int((p_mis>0.5).sum())} #>0.7={int((p_mis>0.7).sum())} #>0.9={int((p_mis>0.9).sum())}")

# ---- frozen 12-way + kNN base prediction ----
base = {t: np.mean([np.load(CACHE / f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t]
                    for g in GROUPS for s in SEEDS for f in range(1, 6)], axis=0) for t in LABELS}
d = np.load("agent_cache/qwen3_embs_final.npz")
tr = d["tr"] / (np.linalg.norm(d["tr"], axis=1, keepdims=True) + 1e-8)
va = d["va"] / (np.linalg.norm(d["va"], axis=1, keepdims=True) + 1e-8)
sim = va @ tr.T
t3lab = np.array([LABELS["t3"].index(l) for l in train_df["evidence_quality"].astype(str)])
knn3 = np.zeros((N, 4))
for i in range(N):
    top = np.argsort(sim[i])[::-1][:K]; w = np.exp(sim[i][top]-sim[i][top].max()); w/=w.sum()
    np.add.at(knn3[i], t3lab[top], w); knn3[i] /= knn3[i].sum()
t3 = base["t3"].copy(); t3[:, 1] *= np.exp(NC_BIAS); t3 /= t3.sum(1, keepdims=True)
fused = dict(base); fused["t3"] = (1-A3)*t3 + A3*knn3
base_pred = {t: [LABELS[t][i] for i in np.argmax(fused[t], axis=1)] for t in LABELS}
for i in range(N):
    if base_pred["t1"][i] == "No": base_pred["t2"][i]=base_pred["t3"][i]=base_pred["t4"][i]="N/A"
    elif base_pred["t2"][i] == "No": base_pred["t3"][i]="N/A"
print(f"\n[12-way+kNN base] valid total={score(base_pred):.5f}")

# ---- overlay sweep: override T3->Misleading where MX1R confident & base predicts non-N/A ----
print("\n=== Misleading-overlay threshold sweep on valid (0 true Misleading -> FP probe) ===")
print(f"{'tau':>5} {'#overrides':>10} {'valid_total':>12} {'Public':>9} {'Private':>9}  note")
for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    ov = {t: list(base_pred[t]) for t in LABELS}
    nov = 0
    for i in range(N):
        # only override where promise/evidence allow a T3 (not forced N/A by cascade)
        if p_mis[i] > tau and ov["t3"][i] != "N/A":
            ov["t3"][i] = "Misleading"; nov += 1
    s = score(ov)
    note = "all FP (valid has 0 Mis)" if nov else "no change"
    print(f"{tau:5.2f} {nov:10d} {s:12.5f} {score(ov,usage=='Public'):9.5f} "
          f"{score(ov,usage=='Private'):9.5f}  {note}")
print("\nDecision: pick the LOWEST tau with #overrides==0 on valid (max recall at full "
      "valid-precision); that tau is the safe AIdea-probe setting. If even tau=0.9 fires "
      "on several valid rows, precision is too low -> reject.")
