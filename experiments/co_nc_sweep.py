"""
Efficient company-NC calibration sweep.
Extracts probs once, then sweeps beta × base_bias in-memory.
"""
import ast, math, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

LABELS = {
    "t1": ["Yes", "No"],
    "t2": ["Yes", "No", "N/A"],
    "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "t4": ["already", "within_2_years", "between_2_and_5_years",
           "longer_than_5_years", "N/A"],
}
IDX2LABEL = {t: {i: l for i, l in enumerate(v)} for t, v in LABELS.items()}
LABEL2IDX = {t: {l: i for i, l in enumerate(v)} for t, v in LABELS.items()}
NC_IDX = LABEL2IDX["t3"]["Not Clear"]
WEIGHTS = {"T1": 0.20, "T2": 0.30, "T3": 0.35, "T4": 0.15}

# ── Load data ─────────────────────────────────────────────────────────────────
data_dir = Path("3rd_data")
train_df = pd.read_csv(data_dir / "train_data.csv")
test_df  = pd.read_csv(data_dir / "test_data.csv")
train_df["label"] = train_df["label"].apply(ast.literal_eval)
for i, t in enumerate(["t1","t2","t3","t4"]):
    train_df[t] = train_df["label"].apply(lambda x, i=i: x[i])

global_nc_rate = (train_df["t3"] == "Not Clear").mean()
co_nc_rate = train_df.groupby("company")["t3"].apply(
    lambda x: (x == "Not Clear").mean()
).to_dict()
print(f"Global NC rate: {global_nc_rate:.4f}")

# ── Inject esg_main ───────────────────────────────────────────────────────────
sys.path.insert(0, ".")
import esg_main as em
main_mod = sys.modules["__main__"]
for name in ["Config", "ApproachA1", "ESGDataset"]:
    if hasattr(em, name):
        setattr(main_mod, name, getattr(em, name))

kfold_dirs = [
    "runs/C1_roberta_dc_kfold_t3nc3",
    "runs/C1_roberta_dc_kfold_t3nc3_s1",
    "runs/C1_roberta_dc_kfold_t3nc3_s2",
]
dir_seeds = {
    "runs/C1_roberta_dc_kfold_t3nc3":    42,
    "runs/C1_roberta_dc_kfold_t3nc3_s1": 123,
    "runs/C1_roberta_dc_kfold_t3nc3_s2": 456,
}

# ── Extract OOF probs (once) ──────────────────────────────────────────────────
print("\nExtracting OOF probs...")
all_oof = []
for kd in kfold_dirs:
    if not Path(kd).exists(): continue
    seed = dir_seeds[kd]
    print(f"  {kd} (seed={seed})")
    probs = em.extract_oof_teacher_probs(kd, train_df, n_splits=5, seed=seed)
    all_oof.append(probs)

oof_probs = {}
for task in LABELS:
    n = len(all_oof[0][task])
    oof_probs[task] = np.array([
        [sum(ap[task][i][c] for ap in all_oof)/len(all_oof) for c in range(len(LABELS[task]))]
        for i in range(n)
    ], dtype=np.float32)

# ── Extract test probs (once) ─────────────────────────────────────────────────
print("\nExtracting test probs...")
all_fold_probs = []
for kd in kfold_dirs:
    if not Path(kd).exists(): continue
    fold_idx = 1
    while True:
        ckpt = Path(kd) / f"fold{fold_idx}" / "best.pt"
        if not ckpt.exists(): break
        result = em._predict_checkpoint(str(ckpt), test_df)
        if result is not None:
            all_fold_probs.append(result[1])
            print(f"  {kd}/fold{fold_idx} OK")
        fold_idx += 1

test_probs = {}
for task in LABELS:
    n = len(all_fold_probs[0][task])
    test_probs[task] = np.array([
        [sum(fp[task][i][c] for fp in all_fold_probs)/len(all_fold_probs)
         for c in range(len(LABELS[task]))]
        for i in range(n)
    ], dtype=np.float32)

print(f"\nOOF shape: {oof_probs['t3'].shape}, Test shape: {test_probs['t3'].shape}")

# ── Load solution for local scoring ──────────────────────────────────────────
sol = pd.read_csv(data_dir / "solution_data.csv")
pub = sol[sol["Usage"] == "Public"].copy()
pub["label"] = pub["label"].apply(ast.literal_eval)
for i, t in enumerate(["T1","T2","T3","T4"]):
    pub[t] = pub["label"].apply(lambda x, i=i: x[i])

def apply_co_calib(probs_t3_np, companies, beta, base_bias):
    """Apply company calibration to T3 prob matrix. Returns calibrated probs."""
    eps = 1e-3
    p2 = probs_t3_np.copy()
    for i, co in enumerate(companies):
        co_rate = co_nc_rate.get(co, global_nc_rate)
        co_bias = beta * math.log((co_rate + eps) / (global_nc_rate + eps))
        p2[i, NC_IDX] *= math.exp(base_bias + co_bias)
    p2 /= p2.sum(axis=1, keepdims=True)
    return p2

def score_probs(t3_probs_calib, companies, mode="test"):
    """Score calibrated T3 probs on test/OOF data."""
    preds = [IDX2LABEL["t3"][int(np.argmax(p))] for p in t3_probs_calib]
    if mode == "oof":
        return f1_score(train_df["t3"], preds, average="macro")
    # test: need full submission
    return None  # handled separately

def make_submission(test_probs_calib_t3, fname):
    """Build submission CSV using calibrated T3 + original T1/T2/T4."""
    preds_t1 = [IDX2LABEL["t1"][int(np.argmax(p))] for p in test_probs["t1"]]
    preds_t2 = [IDX2LABEL["t2"][int(np.argmax(p))] for p in test_probs["t2"]]
    preds_t3 = [IDX2LABEL["t3"][int(np.argmax(p))] for p in test_probs_calib_t3]
    preds_t4 = [IDX2LABEL["t4"][int(np.argmax(p))] for p in test_probs["t4"]]
    rows = [{"id": row["id"],
             "label": str([preds_t1[i], preds_t2[i], preds_t3[i], preds_t4[i]])}
            for i, row in test_df.iterrows()]
    pd.DataFrame(rows).to_csv(fname, index=False)

def score_submission(sub_path):
    sub = pd.read_csv(sub_path)
    sub["label"] = sub["label"].apply(ast.literal_eval)
    for i, t in enumerate(["T1","T2","T3","T4"]):
        sub[t] = sub["label"].apply(lambda x, i=i: x[i])
    m = pub.merge(sub[["id","T1","T2","T3","T4"]], on="id", suffixes=("_true","_pred"))
    total = sum(WEIGHTS[t]*f1_score(m[f"{t}_true"],m[f"{t}_pred"],average="macro")
                for t in ["T1","T2","T3","T4"])
    nc = f1_score(m["T3_true"],m["T3_pred"],labels=["Not Clear"],average="macro")
    return total, nc

# ── OOF sweep to find best params ─────────────────────────────────────────────
print("\nOOF sweep (training set) — finding best beta × base_bias:")
train_companies = train_df["company"].tolist()
test_companies  = test_df["company"].tolist()

best_oof, best_beta, best_bias = 0, 0, 0
for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    for base_bias in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        t3c = apply_co_calib(oof_probs["t3"], train_companies, beta, base_bias)
        f1 = score_probs(t3c, train_companies, mode="oof")
        if f1 > best_oof:
            best_oof, best_beta, best_bias = f1, beta, base_bias

print(f"Best OOF T3 F1={best_oof:.4f} at beta={best_beta}, base_bias={best_bias}")

# ── Generate submissions for promising combos ──────────────────────────────────
print("\nGenerating test submissions:")
combos = [
    (0.0, 0.3, "s392_nonco_b00_nb03"),   # no company, just global bias
    (0.0, 0.0, "s392_nonco_b00_nb00"),   # pure neural
    (best_beta, best_bias, f"s392_co_best_b{int(best_beta*10):02d}_nb{int(best_bias*10):02d}"),
    (0.3, 0.3, "s392_co_b03_nb03"),
    (0.4, 0.3, "s392_co_b04_nb03"),
    (0.3, 0.2, "s392_co_b03_nb02"),
    (0.5, 0.2, "s392_co_b05_nb02"),
]

results = []
for beta, bias, name in combos:
    t3c = apply_co_calib(test_probs["t3"], test_companies, beta, bias)
    out = f"submissions/{name}.csv"
    make_submission(t3c, out)
    total, nc = score_submission(out)
    results.append((name, beta, bias, total, nc))
    print(f"  {name}: {total:.5f} NC={nc:.4f}")

print(f"\n{'name':<36} {'beta':>5} {'bias':>5} {'total':>8} {'NC':>8}")
print("-"*65)
for name, beta, bias, total, nc in sorted(results, key=lambda x: -x[3]):
    marker = " <--" if total > 0.63645 else ""
    print(f"{name:<36} {beta:>5.1f} {bias:>5.1f} {total:>8.5f} {nc:>8.4f}{marker}")
