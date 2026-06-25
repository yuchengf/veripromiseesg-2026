"""Three-layer anti-overfit protocol for per-task ensemble selection.

Layer 1 (SELECT):  greedy per-task group selection + kNN alpha sweep on OOF (train 1601)
Layer 2 (CONFIRM): chosen recipe evaluated ONCE on valid 399, bootstrap CI vs uniform refs
Layer 3 (Kaggle public) is manual, outside this script.

Motivated by SemEval-2025 Task 6 overview: uniform voting dilutes strong models;
per-task selective ensembles recommended. Selection is done on OOF so that valid
stays untouched as a confirmation set.

Usage: conda run -n AICUP python pertask_valid_eval.py
"""
import os, sys, json
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "final_data"]

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from esg_main import (
    _predict_checkpoint, extract_embeddings, load_dataframes,
    LABELS, NUM_LABELS, IDX2LABEL, LABEL2IDX, Config,
)
import __main__
__main__.Config = Config

TASK_W = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}
SOL_COL = {"t1": "promise_status", "t2": "evidence_status",
           "t3": "evidence_quality", "t4": "verification_timeline"}

# run name -> (group, kfold split seed)
RUNS = {}
for grp, prefix in [("FC1", "FC1_roberta_dc_kfold_t3nc3"),
                    ("FC1R", "FC1R_roberta_dc_kfold_t3nc3"),
                    ("FC1S", "FC1S_roberta_dc_kfold_t3nc3"),
                    ("FC1M", "FC1M_roberta_dc_kfold_t3nc3"),
                    ("FD1", "FD1_deberta_dc_kfold_t3nc3"),
                    ("FD1R", "FD1R_deberta_dc_kfold_t3nc3"),
                    ("FX1R", "FX1R_xlmr_dc_kfold_t3nc3")]:
    RUNS[prefix] = (grp, 42)
    RUNS[prefix + "_s1"] = (grp, 123)
    RUNS[prefix + "_s2"] = (grp, 456)

CACHE_V = Path("agent_cache/valid_probs"); CACHE_V.mkdir(parents=True, exist_ok=True)
CACHE_O = Path("agent_cache/oof_probs");   CACHE_O.mkdir(parents=True, exist_ok=True)

train_df, valid_df = load_dataframes("final_data")  # normalized labels, valid as test
sol = pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
assert (valid_df["id"].values == sol["id"].values).all()
y_valid = {t: sol[SOL_COL[t]].astype(str).tolist() for t in LABELS}
y_oof = {t: train_df[SOL_COL[t]].astype(str).tolist() for t in LABELS}
NV, NO = len(valid_df), len(train_df)
print(f"Train(OOF): {NO} rows, Valid: {NV} rows")


def valid_probs_for(run_name: str):
    """5 fold prob dicts {task: (NV, C)} on valid, or None."""
    out = []
    for fold in range(1, 6):
        p = CACHE_V / f"{run_name}_f{fold}.npz"
        if p.exists():
            d = np.load(p); out.append({t: d[t] for t in LABELS}); continue
        ckpt = f"runs/{run_name}/fold{fold}/best.pt"
        if not Path(ckpt).exists():
            return None
        r = _predict_checkpoint(ckpt, valid_df)
        if r is None:
            return None
        probs = {t: np.asarray(r[1][t], dtype=np.float32) for t in LABELS}
        np.savez_compressed(p, **probs)
        out.append(probs)
    return out


def oof_probs_for(run_name: str, split_seed: int):
    """Full-coverage OOF probs {task: (NO, C)} via per-fold val_idx prediction."""
    p = CACHE_O / f"{run_name}_oof.npz"
    if p.exists():
        d = np.load(p)
        return {t: d[t] for t in LABELS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    oof = {t: np.zeros((NO, NUM_LABELS[t]), dtype=np.float32) for t in LABELS}
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["promise_status"])):
        ckpt = f"runs/{run_name}/fold{fold_idx+1}/best.pt"
        if not Path(ckpt).exists():
            return None
        r = _predict_checkpoint(ckpt, train_df.iloc[val_idx].reset_index(drop=True))
        if r is None:
            return None
        for t in LABELS:
            oof[t][val_idx] = np.asarray(r[1][t], dtype=np.float32)
    np.savez_compressed(p, **oof)
    return oof


# ── gather probs ─────────────────────────────────────────────────────────────
groups_v: dict[str, list] = {}   # group -> list of 15 fold prob dicts (valid)
groups_o: dict[str, list] = {}   # group -> list of 3 OOF prob dicts (one per seed)
for rn, (grp, seed) in RUNS.items():
    vp = valid_probs_for(rn)
    op = oof_probs_for(rn, seed)
    if vp is None or op is None:
        print(f"[SKIP] {rn} incomplete")
        continue
    groups_v.setdefault(grp, []).extend(vp)
    groups_o.setdefault(grp, []).append(op)

avail = [g for g in ["FC1", "FC1R", "FC1S", "FC1M", "FD1", "FD1R", "FX1R"]
         if len(groups_v.get(g, [])) == 15 and len(groups_o.get(g, [])) == 3]
print(f"\nAvailable groups: {avail}")


def avg_v(gnames):
    return {t: np.mean([fp[t] for g in gnames for fp in groups_v[g]], axis=0) for t in LABELS}


def avg_o(gnames):
    return {t: np.mean([op[t] for g in gnames for op in groups_o[g]], axis=0) for t in LABELS}


def compose(task_probs, n):
    preds = {t: [IDX2LABEL[t][i] for i in np.argmax(task_probs[t], axis=1)] for t in LABELS}
    for i in range(n):
        if preds["t1"][i] == "No":
            preds["t2"][i] = preds["t3"][i] = preds["t4"][i] = "N/A"
        elif preds["t2"][i] == "No":
            preds["t3"][i] = "N/A"
    return preds


def score(preds, y_true):
    s = {}
    for t in LABELS:
        labels = sorted(set(y_true[t]))
        s[t] = f1_score(y_true[t], preds[t], labels=labels, average="macro", zero_division=0)
    s["total"] = sum(TASK_W[t] * s[t] for t in TASK_W)
    return s


def fmt(s):
    return f"T1={s['t1']:.4f} T2={s['t2']:.4f} T3={s['t3']:.4f} T4={s['t4']:.4f} | total={s['total']:.5f}"


# ── kNN-LDL probs (Qwen3), self-excluded for OOF ────────────────────────────
import torch
_INSTR = "Retrieve similar ESG disclosure texts that share the same evidence quality level."
emb_cache = Path("agent_cache/qwen3_embs_final.npz")
if emb_cache.exists():
    d = np.load(emb_cache); tr_embs, va_embs = d["tr"], d["va"]
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_embs = extract_embeddings(train_df["data"].tolist(), "Qwen/Qwen3-Embedding-0.6B",
                                 batch_size=16, device=device, instruction=_INSTR).numpy()
    va_embs = extract_embeddings(valid_df["data"].tolist(), "Qwen/Qwen3-Embedding-0.6B",
                                 batch_size=16, device=device, instruction=_INSTR).numpy()
    np.savez_compressed(emb_cache, tr=tr_embs, va=va_embs)


def knn_ldl(sim, k, exclude_diag=False):
    """Replicates esg_main.compute_knn_ldl_probs weighting (softmax over top-k sims)."""
    if exclude_diag:
        sim = sim.copy(); np.fill_diagonal(sim, -np.inf)
    out = {t: np.zeros((sim.shape[0], NUM_LABELS[t]), dtype=np.float32) for t in LABELS}
    task_lab_idx = {t: np.array([LABEL2IDX[t].get(l, -1) for l in y_oof[t]]) for t in LABELS}
    for i in range(sim.shape[0]):
        top = np.argsort(sim[i])[::-1][:k]
        w = np.exp(sim[i][top] - sim[i][top].max()); w /= w.sum()
        for t in LABELS:
            for rank, tidx in enumerate(top):
                c = task_lab_idx[t][tidx]
                if c >= 0:
                    out[t][i, c] += w[rank]
            tot = out[t][i].sum()
            out[t][i] = out[t][i] / tot if tot > 0 else 1.0 / NUM_LABELS[t]
    return out


tr_n = tr_embs / (np.linalg.norm(tr_embs, axis=1, keepdims=True) + 1e-8)
va_n = va_embs / (np.linalg.norm(va_embs, axis=1, keepdims=True) + 1e-8)
knn_o = knn_ldl(tr_n @ tr_n.T, k=5, exclude_diag=True)   # OOF: self-excluded
knn_v = knn_ldl(va_n @ tr_n.T, k=5)                       # valid: full train pool

# ═════════════════════════ LAYER 1: SELECT ON OOF ═══════════════════════════
print("\n========== LAYER 1: selection on OOF (train 1601) ==========")
print("\n=== Per-group OOF scores ===")
for g in avail:
    print(f"  {g:5s} {fmt(score(compose(avg_o([g]), NO), y_oof))}")

REFS = [["FC1R"], ["FC1R", "FC1S"], ["FC1", "FC1R", "FC1S", "FC1M"],
        ["FC1R", "FD1R"], ["FC1R", "FC1S", "FD1R"],
        ["FC1", "FC1R", "FC1S", "FC1M", "FD1", "FD1R"],
        ["FC1R", "FX1R"], ["FC1", "FC1R", "FC1S", "FC1M", "FX1R"]]
print("\n=== Reference combos (OOF) ===")
ref_oof = {}
for combo in REFS:
    if not all(g in avail for g in combo):
        continue
    s = score(compose(avg_o(combo), NO), y_oof)
    ref_oof["+".join(combo)] = s
    print(f"  {'+'.join(combo):28s} {fmt(s)}")

print("\n=== Greedy per-task selection (OOF) ===")
sel: dict[str, list[str]] = {}
fixed = avg_o(avail)  # placeholder for unselected tasks
for task in ["t1", "t2", "t3", "t4"]:
    remaining, chosen, best_f1 = list(avail), [], -1.0
    while remaining:
        cands = []
        for g in remaining:
            tp = dict(fixed); tp[task] = avg_o(chosen + [g])[task]
            cands.append((score(compose(tp, NO), y_oof)[task], g))
        cands.sort(reverse=True)
        f1, g = cands[0]
        if f1 > best_f1 + 1e-6:
            best_f1, chosen = f1, chosen + [g]; remaining.remove(g)
        else:
            break
    sel[task] = chosen
    fixed[task] = avg_o(chosen)[task]
    print(f"  {task}: {chosen}  OOF-F1={best_f1:.4f}")

s_sel_oof = score(compose(fixed, NO), y_oof)
print(f"\n  Per-task composed (OOF): {fmt(s_sel_oof)}")

print("\n=== kNN alpha sweep on OOF (plateau-preferred) ===")
grid = {}
A2, A3, A4 = [0.0, 0.1, 0.2], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 0.1, 0.2]
for a2 in A2:
    for a3 in A3:
        for a4 in A4:
            fused = {"t1": fixed["t1"],
                     "t2": (1 - a2) * fixed["t2"] + a2 * knn_o["t2"],
                     "t3": (1 - a3) * fixed["t3"] + a3 * knn_o["t3"],
                     "t4": (1 - a4) * fixed["t4"] + a4 * knn_o["t4"]}
            grid[(a2, a3, a4)] = score(compose(fused, NO), y_oof)["total"]


def plateau_score(key):
    """Average of own score and axis-neighbors -> prefers flat optima over sharp peaks."""
    a2, a3, a4 = key
    vals = [grid[key]]
    for d in (-0.1, 0.1):
        for nk in [(round(a2 + d, 1), a3, a4), (a2, round(a3 + d, 1), a4), (a2, a3, round(a4 + d, 1))]:
            if nk in grid:
                vals.append(grid[nk])
    return float(np.mean(vals))


ranked = sorted(grid, key=lambda k: (plateau_score(k), grid[k]), reverse=True)
for k in ranked[:6]:
    print(f"  a2={k[0]:.1f} a3={k[1]:.1f} a4={k[2]:.1f}  raw={grid[k]:.5f} plateau={plateau_score(k):.5f}")
best_alpha = ranked[0]
a2b, a3b, a4b = best_alpha
print(f"\n  Chosen alpha (plateau): a2={a2b} a3={a3b} a4={a4b}")

# ═════════════════════════ LAYER 2: CONFIRM ON VALID ════════════════════════
print("\n========== LAYER 2: one-shot confirmation on valid 399 ==========")
print("\n=== Reference combos (valid) ===")
best_ref_name, best_ref_total, best_ref_preds = None, -1, None
for combo in REFS:
    if not all(g in avail for g in combo):
        continue
    preds = compose(avg_v(combo), NV)
    s = score(preds, y_valid)
    print(f"  {'+'.join(combo):28s} {fmt(s)}")
    if s["total"] > best_ref_total:
        best_ref_name, best_ref_total, best_ref_preds = "+".join(combo), s["total"], preds

fixed_v = {t: avg_v(sel[t])[t] for t in LABELS}
fused_v = {"t1": fixed_v["t1"],
           "t2": (1 - a2b) * fixed_v["t2"] + a2b * knn_v["t2"],
           "t3": (1 - a3b) * fixed_v["t3"] + a3b * knn_v["t3"],
           "t4": (1 - a4b) * fixed_v["t4"] + a4b * knn_v["t4"]}
preds_recipe = compose(fused_v, NV)
s_recipe = score(preds_recipe, y_valid)
print(f"\n  RECIPE (per-task sel + kNN): {fmt(s_recipe)}")
print(f"  Best uniform ref ({best_ref_name}): total={best_ref_total:.5f}")
print(f"  Delta: {s_recipe['total'] - best_ref_total:+.5f}")

# Bootstrap 95% CI of (recipe - best_ref) on valid
rng = np.random.default_rng(42)
diffs = []
for _ in range(2000):
    idx = rng.integers(0, NV, NV)
    yt = {t: [y_valid[t][i] for i in idx] for t in LABELS}
    d_r = score({t: [preds_recipe[t][i] for i in idx] for t in LABELS}, yt)["total"]
    d_b = score({t: [best_ref_preds[t][i] for i in idx] for t in LABELS}, yt)["total"]
    diffs.append(d_r - d_b)
lo, hi = np.percentile(diffs, [2.5, 97.5])
verdict = ("SIGNIFICANTLY BETTER" if lo > 0
           else "SIGNIFICANTLY WORSE (reject recipe)" if hi < 0
           else "overlaps 0 (treat as tie, prefer simpler)")
print(f"  Bootstrap 95% CI of delta: [{lo:+.5f}, {hi:+.5f}]  -> {verdict}")

out = {
    "selection": sel,
    "alpha": {"t2": a2b, "t3": a3b, "t4": a4b},
    "oof_scores": {k: float(v) for k, v in s_sel_oof.items()},
    "valid_recipe": {k: float(v) for k, v in s_recipe.items()},
    "valid_best_ref": {"name": best_ref_name, "total": float(best_ref_total)},
    "delta_ci95": [float(lo), float(hi)],
}
with open("agent_cache/pertask_valid_results.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSaved -> agent_cache/pertask_valid_results.json")
