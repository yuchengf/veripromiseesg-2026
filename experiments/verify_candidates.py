"""Format-validate all 06-16 AIdea candidate files vs the champion. PASS/FAIL each."""
import pandas as pd
from pathlib import Path

CH = "official_sub/aidea_clarity_on_fc1r3.csv"
COLS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]
CANDS = [
    ("swap-T3 (PRIMARY)", "official_sub/aidea_swapT3_512_knn_clarity.csv", ["evidence_quality"]),
    ("hybrid (T2+T3)", "official_sub/aidea_hybrid_t14_384_t23_512_knn_clarity.csv", ["evidence_status", "evidence_quality"]),
    ("pure 3way@512", "official_sub/aidea_rt512_3way_knn_clarity.csv", None),
    ("12way@512 (low pri)", "official_sub/aidea_rt512_12way_knn_clarity.csv", None),
]
ch = pd.read_csv(CH, keep_default_na=False)
print(f"champion {CH}: {len(ch)} rows")
for name, path, expect_diff in CANDS:
    p = Path(path)
    if not p.exists():
        print(f"  [MISSING] {name}: {path}"); continue
    df = pd.read_csv(p, keep_default_na=False)
    ok = True; msgs = []
    if len(df) != 2000: ok = False; msgs.append(f"rows={len(df)}≠2000")
    if list(df.columns) != COLS: ok = False; msgs.append("欄位不符")
    na = int((df == "N/A").sum().sum())
    diff = {c: int((df[c].values != ch[c].values).sum()) for c in COLS if c != "id"}
    if expect_diff is not None:
        unexpected = [c for c, n in diff.items() if n > 0 and c not in expect_diff]
        if unexpected: ok = False; msgs.append(f"非預期差異欄 {unexpected}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: N/A={na} diff_vs_champ={diff} {';'.join(msgs)}")
print("→ 上傳順序:swap-T3 → hybrid → pure-@512(Round1);最後一次交 AIdea 確認最高分(地板 champion 0.6188)")
