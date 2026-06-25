"""Post-processing: override T4 with company majority when company is concentrated.

Strategy: if a company has >= min_samples training examples AND the majority T4
label reaches >= threshold fraction, override the model prediction to that label.

This is NOT a soft blend — it is a hard override, because the model prediction
is one-hot and a soft blend at small beta never moves the argmax.

Usage:
    python company_t4_blend.py \
        --base_csv submissions/s153_v24_k10.csv \
        --output_dir submissions/company_blend \
        --threshold 0.6 0.65 0.7 0.75
"""
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("2026-esg-classification-challenge")
T4_LABELS = ["already", "within_2_years", "between_2_and_5_years",
             "longer_than_5_years", "N/A"]
T4_IDX = {l: i for i, l in enumerate(T4_LABELS)}


def build_company_t4_majority(train_csv: Path, min_samples: int = 3
                               ) -> dict[str, tuple[str, float] | None]:
    """Return (majority_label, fraction) per company, or None if < min_samples."""
    train = pd.read_csv(train_csv)
    result: dict[str, tuple[str, float] | None] = {}
    for company, grp in train.groupby("company"):
        if len(grp) < min_samples:
            result[company] = None
            continue
        counts = grp["verification_timeline"].value_counts()
        majority_label = counts.index[0]
        majority_frac = counts.iloc[0] / len(grp)
        result[company] = (majority_label, majority_frac)
    return result


def override_submission(base_csv: Path, test_csv: Path, train_csv: Path,
                        threshold: float, min_samples: int = 3) -> pd.DataFrame:
    base = pd.read_csv(base_csv)
    test = pd.read_csv(test_csv)[["id", "company"]]
    majority = build_company_t4_majority(train_csv, min_samples)

    id_to_company = dict(zip(test["id"], test["company"]))

    rows = []
    changed = 0
    for _, row in base.iterrows():
        sample_id = row["id"]
        labels = ast.literal_eval(row["label"])  # [T1, T2, T3, T4]
        t4_pred = labels[3]

        company = id_to_company.get(sample_id)
        info = majority.get(company) if company else None

        if info is not None:
            maj_label, maj_frac = info
            if maj_frac >= threshold and maj_label != t4_pred:
                labels[3] = maj_label
                changed += 1

        rows.append({"id": sample_id, "label": str(labels).replace('"', "'")})

    print(f"  threshold={threshold:.2f}: {changed}/200 T4 overridden")
    return pd.DataFrame(rows)


def show_company_stats(train_csv: Path, test_csv: Path, min_samples: int = 3) -> None:
    """Print per-company T4 distribution for inspection."""
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)
    test_companies = set(test["company"].unique())
    print(f"\n{'Company':<12} {'N_train':>7} {'Majority label':<25} {'Frac':>5}  In_test")
    print("-" * 65)
    for company, grp in sorted(train.groupby("company"),
                               key=lambda x: -len(x[1])):
        counts = grp["verification_timeline"].value_counts()
        maj = counts.index[0]
        frac = counts.iloc[0] / len(grp)
        in_test = "✓" if company in test_companies else ""
        print(f"  {company:<12} {len(grp):>7}  {maj:<25} {frac:>5.0%}  {in_test}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_csv", required=True)
    parser.add_argument("--output_dir", default="submissions/company_blend")
    parser.add_argument("--threshold", nargs="+", type=float,
                        default=[0.6, 0.65, 0.7, 0.75])
    parser.add_argument("--min_samples", type=int, default=3,
                        help="Minimum training samples to use company prior")
    parser.add_argument("--stats", action="store_true",
                        help="Print company T4 distribution and exit")
    args = parser.parse_args()

    train_csv = DATA_DIR / "train_data.csv"
    test_csv  = DATA_DIR / "test_data.csv"

    if args.stats:
        show_company_stats(train_csv, test_csv, args.min_samples)
        return

    base_csv = Path(args.base_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = base_csv.stem

    for thr in args.threshold:
        out_df = override_submission(base_csv, test_csv, train_csv,
                                     thr, args.min_samples)
        out_path = output_dir / f"{stem}_company_t4_thr{int(thr*100):02d}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
