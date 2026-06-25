"""Leave-one-out ablation of the 12-way ensemble on the golden valid 399.

Verifies each of the 4 architecture groups (FC1/FC1R/FC1S/FC1M) and each of the
12 individual seed-models contributes positively to the FROZEN submission
pipeline (base mean -> kNN T3 alpha=0.4 + NC bias 0.1 -> NA-rule cascade).

Pure numpy over cached valid_probs + Qwen3 embeddings. No GPU.

Verdict rule (3-layer protocol): a model/group is DEAD WEIGHT only if removing
it makes the score significantly BETTER (bootstrap 95% CI of
[without - with] all positive). CI overlapping 0 -> neutral -> KEEP (robustness).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ROOT = Path("/home/yucheng/Desktop/ESG")
LABELS = {
    "t1": ["Yes", "No"],
    "t2": ["Yes", "No", "N/A"],
    "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "t4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
}
TASK_W = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}
SOL_COL = {"t1": "promise_status", "t2": "evidence_status",
           "t3": "evidence_quality", "t4": "verification_timeline"}
GROUPS = ["FC1", "FC1R", "FC1S", "FC1M"]
SEEDS = ["", "_s1", "_s2"]
K, TEMP, A3, NC_BIAS = 5, 1.0, 0.40, 0.1

sol = pd.read_csv(ROOT / "final_data/valid_solution_data.csv", keep_default_na=False)
train = pd.read_csv(ROOT / "final_data/train_data.csv", keep_default_na=False)
for df in (sol, train):
    df["verification_timeline"] = df["verification_timeline"].replace(
        "longer_than_5_years", "more_than_5_years")
y = {t: sol[SOL_COL[t]].astype(str).tolist() for t in LABELS}
usage = sol["Usage"].values
N = len(sol)

CACHE = ROOT / "agent_cache/valid_probs"


def load_member(group, seed):
    pre = f"{group}_roberta_dc_kfold_t3nc3{seed}"
    return {t: np.mean([np.load(CACHE / f"{pre}_f{f}.npz")[t] for f in range(1, 6)], axis=0)
            for t in LABELS}


members = {(g, s): load_member(g, s) for g in GROUPS for s in SEEDS}

# kNN T3 distribution (fixed, independent of which base members are included)
d = np.load(ROOT / "agent_cache/qwen3_embs_final.npz")
tr = d["tr"] / (np.linalg.norm(d["tr"], axis=1, keepdims=True) + 1e-8)
va = d["va"] / (np.linalg.norm(d["va"], axis=1, keepdims=True) + 1e-8)
sim = va @ tr.T
t3_lab = np.array([LABELS["t3"].index(l) for l in train["evidence_quality"].astype(str)])
knn3 = np.zeros((N, 4))
for i in range(N):
    top = np.argsort(sim[i])[::-1][:K]
    w = np.exp(sim[i][top] / TEMP - (sim[i][top] / TEMP).max()); w /= w.sum()
    np.add.at(knn3[i], t3_lab[top], w)
    knn3[i] /= knn3[i].sum()


def preds_for(member_keys):
    base = {t: np.mean([members[k][t] for k in member_keys], axis=0) for t in LABELS}
    t3 = base["t3"].copy()
    t3[:, 1] *= np.exp(NC_BIAS)
    t3 /= t3.sum(1, keepdims=True)
    fused = dict(base)
    fused["t3"] = (1 - A3) * t3 + A3 * knn3
    p = {t: [LABELS[t][i] for i in np.argmax(fused[t], axis=1)] for t in LABELS}
    for i in range(N):
        if p["t1"][i] == "No":
            p["t2"][i] = p["t3"][i] = p["t4"][i] = "N/A"
        elif p["t2"][i] == "No":
            p["t3"][i] = "N/A"
    return p


def per_row_weighted(preds, idx):
    """Weighted-macro-F1 contribution is global, so for bootstrap we resample
    rows and recompute the weighted macro-F1 on the resampled set."""
    tot = 0.0
    for t in LABELS:
        yt = [y[t][i] for i in idx]
        pt = [preds[t][i] for i in idx]
        tot += TASK_W[t] * f1_score(yt, pt, labels=sorted(set(y[t])),
                                    average="macro", zero_division=0)
    return tot


def score(preds, mask=None):
    idx = np.arange(N) if mask is None else np.where(mask)[0]
    return per_row_weighted(preds, idx)


def boot_ci_delta(preds_without, preds_with, n_boot=2000):
    rng = np.random.RandomState(42)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, N, N)
        deltas[b] = per_row_weighted(preds_without, idx) - per_row_weighted(preds_with, idx)
    return np.percentile(deltas, 2.5), np.percentile(deltas, 97.5)


ALL = [(g, s) for g in GROUPS for s in SEEDS]
full = preds_for(ALL)
s_full = score(full)
print(f"=== FULL 12-way + kNN: ALL={s_full:.5f}  "
      f"Public={score(full, usage=='Public'):.5f}  Private={score(full, usage=='Private'):.5f}\n")


def verdict(lo, hi):
    if lo > 0:
        return "DEAD WEIGHT (removing helps, CI>0) -> consider dropping"
    if hi < 0:
        return "CONTRIBUTES (removing hurts, CI<0) -> keep"
    return "neutral (CI overlaps 0) -> keep for robustness"


print("=== Leave-one-GROUP-out (drop 3 seeds of one architecture -> 9-way) ===")
for g in GROUPS:
    keep = [k for k in ALL if k[0] != g]
    p = preds_for(keep)
    s = score(p)
    lo, hi = boot_ci_delta(p, full)
    print(f"  -{g:5s}: 9-way={s:.5f}  delta(without-full)={s - s_full:+.5f}  "
          f"CI=[{lo:+.5f},{hi:+.5f}]  {verdict(lo, hi)}")

print("\n=== Leave-one-SEED-out (drop one model -> 11-way) ===")
for k in ALL:
    keep = [m for m in ALL if m != k]
    p = preds_for(keep)
    s = score(p)
    lo, hi = boot_ci_delta(p, full)
    name = f"{k[0]}{k[1]}"
    print(f"  -{name:9s}: 11-way={s:.5f}  delta={s - s_full:+.5f}  "
          f"CI=[{lo:+.5f},{hi:+.5f}]  {verdict(lo, hi)}")

print("\n=== Seed-count curve (each group, cumulative seeds) ===")
for nseed in (1, 2, 3):
    keys = [(g, s) for g in GROUPS for s in SEEDS[:nseed]]
    p = preds_for(keys)
    print(f"  {nseed} seed/group ({len(keys)} models): ALL={score(p):.5f}  "
          f"Public={score(p, usage=='Public'):.5f}  Private={score(p, usage=='Private'):.5f}")
