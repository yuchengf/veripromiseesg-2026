import sys, os
os.chdir("/home/yucheng/Desktop/ESG")
sys.argv = ["esg_main", "--mode", "gen_submissions", "--data_dir", "retrain_data"]

from esg_main import (_predict_checkpoint, ensemble_preds_soft,
                       apply_na_rule, load_dataframes, Config)
import __main__
__main__.Config = Config

import pandas as pd
from pathlib import Path

train_df, test_df = load_dataframes("retrain_data")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

MODELS = {
    "rt_fc1_s0":  "runs/RT_FC1_s0",
    "rt_fc1_s1":  "runs/RT_FC1_s1",
    "rt_fc1_s2":  "runs/RT_FC1_s2",
    "rt_fc1r_s0": "runs/RT_FC1R_s0",
    "rt_fc1r_s1": "runs/RT_FC1R_s1",
    "rt_fc1r_s2": "runs/RT_FC1R_s2",
    "rt_fc1s_s0": "runs/RT_FC1S_s0",
    "rt_fc1s_s1": "runs/RT_FC1S_s1",
    "rt_fc1s_s2": "runs/RT_FC1S_s2",
    "rt_fc1m_s0": "runs/RT_FC1M_s0",
    "rt_fc1m_s1": "runs/RT_FC1M_s1",
    "rt_fc1m_s2": "runs/RT_FC1M_s2",
}

# Collect raw fold predictions (logit dicts) per model
model_fold_preds = {}  # key -> list of 5 raw pred dicts
for key, run_dir in MODELS.items():
    fold_preds = []
    for fold in range(1, 6):
        ckpt = f"{run_dir}/fold{fold}/best.pt"
        if not Path(ckpt).exists():
            print(f"  [MISS] {ckpt}")
            break
        result = _predict_checkpoint(ckpt, test_df)
        if result is not None:
            fold_preds.append(result[1])
    if len(fold_preds) == 5:
        model_fold_preds[key] = fold_preds
        print(f"  {key}: OK (5 folds)")
    else:
        print(f"  {key}: FAILED ({len(fold_preds)}/5)")

print(f"\nLoaded {len(model_fold_preds)}/12 models")

out_dir = Path("official_sub")
out_dir.mkdir(exist_ok=True)

def save_submission(name, keys, weights=None):
    """Collect all fold preds from specified models and ensemble at once."""
    all_preds = []
    for k in keys:
        if k not in model_fold_preds:
            print(f"  [SKIP] {name}: missing {k}")
            return
        all_preds.extend(model_fold_preds[k])  # 5 fold preds per model

    # If weights specified, need to repeat for folds
    if weights is not None:
        expanded_weights = []
        for w in weights:
            expanded_weights.extend([w] * 5)
        final = ensemble_preds_soft(all_preds, weights=expanded_weights)
    else:
        final = ensemble_preds_soft(all_preds)

    final = apply_na_rule(final)
    df_out = pd.DataFrame({
        "id": test_df["id"],
        "promise_status": final["t1"],
        "verification_timeline": final["t4"],
        "evidence_status": final["t2"],
        "evidence_quality": final["t3"],
    })
    path = out_dir / f"{name}.csv"
    df_out.to_csv(path, index=False)
    na_count = int((df_out == "N/A").sum().sum())
    print(f"  → {path} ({len(df_out)} rows, N/A={na_count})")

# 12-way all models
save_submission("aidea_rt_all12", list(MODELS.keys()))

# FC1R 3-seed with s1 dominant
save_submission("aidea_rt_fc1r3",
                ["rt_fc1r_s0", "rt_fc1r_s1", "rt_fc1r_s2"],
                [1.0, 2.0, 1.0])

print("Done!")
