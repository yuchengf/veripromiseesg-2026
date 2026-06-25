import pandas as pd, ast, os
from sklearn.metrics import f1_score, classification_report

sol = pd.read_csv('/home/yucheng/Desktop/ESG/2026-esg-classification-challenge/solution_data.csv')
sol_labels = sol['label'].apply(ast.literal_eval)

# Check Public vs Private split
print("=== Usage split ===")
print(sol['Usage'].value_counts())

for usage in ['Public', 'Private']:
    mask = sol['Usage'] == usage
    subset = sol[mask]
    labels = subset['label'].apply(ast.literal_eval)
    gt = {
        't1': labels.apply(lambda x: x[0]).tolist(),
        't2': labels.apply(lambda x: x[1]).tolist(),
        't3': labels.apply(lambda x: x[2]).tolist(),
        't4': labels.apply(lambda x: x[3]).tolist(),
    }
    print("\n=== %s (%d samples) ===" % (usage, len(subset)))
    print("T1:", dict(pd.Series(gt['t1']).value_counts()))
    print("T2:", dict(pd.Series(gt['t2']).value_counts()))
    print("T3:", dict(pd.Series(gt['t3']).value_counts()))
    print("T4:", dict(pd.Series(gt['t4']).value_counts()))

    # Score s4 on this subset
    sub = pd.read_csv('/home/yucheng/Desktop/ESG/submissions/s4_top3_a1.csv')
    sub = sub[sub['id'].isin(subset['id'])]
    pred_labels = sub['label'].apply(ast.literal_eval)
    pred = {
        't1': pred_labels.apply(lambda x: x[0]).tolist(),
        't2': pred_labels.apply(lambda x: x[1]).tolist(),
        't3': pred_labels.apply(lambda x: x[2]).tolist(),
        't4': pred_labels.apply(lambda x: x[3]).tolist(),
    }
    scores = {}
    for task in ['t1','t2','t3','t4']:
        all_labels = sorted(set(gt[task]) | set(pred[task]))
        scores[task] = f1_score(gt[task], pred[task], labels=all_labels, average='macro')
    weighted = 0.20*scores['t1'] + 0.30*scores['t2'] + 0.35*scores['t3'] + 0.15*scores['t4']
    print("s4 scores: T1=%.4f T2=%.4f T3=%.4f T4=%.4f Weighted=%.5f" %
          (scores['t1'], scores['t2'], scores['t3'], scores['t4'], weighted))

# Check: T1=Yes but T3=N/A cases in test
print("\n=== Distribution shift analysis ===")
all_gt = {
    't1': sol['label'].apply(ast.literal_eval).apply(lambda x: x[0]).tolist(),
    't2': sol['label'].apply(ast.literal_eval).apply(lambda x: x[1]).tolist(),
    't3': sol['label'].apply(ast.literal_eval).apply(lambda x: x[2]).tolist(),
    't4': sol['label'].apply(ast.literal_eval).apply(lambda x: x[3]).tolist(),
}
# T1=No → count
t1_no = sum(1 for x in all_gt['t1'] if x == 'No')
t3_na = sum(1 for x in all_gt['t3'] if x == 'N/A')
print("T1=No: %d, T3=N/A: %d" % (t1_no, t3_na))
# Cross-tab
for i in range(200):
    if all_gt['t1'][i] == 'Yes' and all_gt['t3'][i] == 'N/A':
        print("  ANOMALY: id=%s T1=Yes T2=%s T3=N/A T4=%s" %
              (sol.iloc[i]['id'], all_gt['t2'][i], all_gt['t4'][i]))

# Train cross-check
print("\n=== Train T2=No → T3 distribution ===")
train = pd.read_csv('/home/yucheng/Desktop/ESG/2026-esg-classification-challenge/train_data.csv')
t2_no = train[train['evidence_status'] == 'No']
print("T2=No samples:", len(t2_no))
print("  T3 distribution:", dict(t2_no['evidence_quality'].value_counts()))
print("  T1 distribution:", dict(t2_no['promise_status'].value_counts()))
