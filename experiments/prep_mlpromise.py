"""Clean + normalize ML-Promise EN/FR/JA trainsets to our competition schema.

Output: external_data/mlpromise_enfrja.csv with columns matching final_data train
(id, data, promise_status, verification_timeline, evidence_status, evidence_quality).
KO/ZH skipped (no clean `data` text field / overlaps our data).

Label normalization:
  T3 evidence_quality: strip typos ("Clear]"->Clear), map "Potentially Misleading"->Misleading
  T4 verification_timeline -> our schema {already, within_2_years,
     between_2_and_5_years, more_than_5_years, N/A}; unmappable (corrupt JA labels)->N/A
Then enforce NA-rule (T1=No -> T2/T3/T4=N/A; T2=No -> T3=N/A).
"""
import json
import pandas as pd
from pathlib import Path

SRC = Path("external_data/mlpromise_examples")
OUT = Path("external_data/mlpromise_enfrja.csv")

T3_MAP = {
    "clear": "Clear", "not clear": "Not Clear",
    "misleading": "Misleading", "potentially misleading": "Misleading",
    "n/a": "N/A", "na": "N/A",
}
T4_MAP = {
    "already": "already", "n/a": "N/A", "na": "N/A",
    "less than 2 years": "within_2_years", "within_2_years": "within_2_years",
    "2 to 5 years": "between_2_and_5_years",
    "between_2_and_5_years": "between_2_and_5_years",
    "more than 5 years": "more_than_5_years",
    "more_than_5_years": "more_than_5_years",
    "longer than 5 years": "more_than_5_years",
}
YN_MAP = {"yes": "Yes", "no": "No", "n/a": "N/A", "na": "N/A"}


def clean(s):
    return str(s).strip().strip("]").strip().lower()


def norm_t3(v):
    return T3_MAP.get(clean(v), "N/A")


def norm_t4(v):
    return T4_MAP.get(clean(v), "N/A")   # unmappable (corrupt JA) -> N/A


def norm_yn(v):
    return YN_MAP.get(clean(v), "N/A")


rows_out = []
stats = {}
for lang in ["English", "French", "Japanese"]:
    data = json.load(open(SRC / f"Trainset_{lang}.json", encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = list(data.values())[0]
    n_skip = 0
    for i, r in enumerate(data):
        text = r.get("data")
        if not text or not str(text).strip():
            n_skip += 1
            continue
        t1 = norm_yn(r.get("promise_status"))
        t2 = norm_yn(r.get("evidence_status"))
        t3 = norm_t3(r.get("evidence_quality"))
        t4 = norm_t4(r.get("verification_timeline"))
        # enforce NA-rule
        if t1 == "No":
            t2 = t3 = t4 = "N/A"
        elif t2 == "No":
            t3 = "N/A"
        rows_out.append({
            "id": f"ML_{lang[:2].upper()}_{i}",
            "data": str(text).strip(),
            "promise_status": t1,
            "verification_timeline": t4,
            "evidence_status": t2,
            "evidence_quality": t3,
        })
    df_lang = pd.DataFrame([x for x in rows_out if x["id"].startswith(f"ML_{lang[:2].upper()}")])
    stats[lang] = {
        "kept": len(df_lang), "skipped_no_text": n_skip,
        "T3": df_lang["evidence_quality"].value_counts().to_dict(),
        "T4": df_lang["verification_timeline"].value_counts().to_dict(),
    }

df = pd.DataFrame(rows_out)
df.to_csv(OUT, index=False)
print(f"=== wrote {OUT}: {len(df)} rows ===")
for lang, s in stats.items():
    print(f"\n{lang}: kept={s['kept']} skipped(no text)={s['skipped_no_text']}")
    print(f"  T3: {s['T3']}")
    print(f"  T4: {s['T4']}")
print(f"\n=== MERGED totals ===")
print(f"  T3: {df['evidence_quality'].value_counts().to_dict()}")
print(f"  T4: {df['verification_timeline'].value_counts().to_dict()}")
print(f"  T1: {df['promise_status'].value_counts().to_dict()}")
print(f"  T2: {df['evidence_status'].value_counts().to_dict()}")
print(f"\n  *** Misleading examples gained: {int((df['evidence_quality']=='Misleading').sum())} ***")
