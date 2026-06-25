import pandas as pd, ast, os
from sklearn.metrics import f1_score, classification_report

sol = pd.read_csv('/home/yucheng/Desktop/ESG/2026-esg-classification-challenge/solution_data.csv')
sol_labels = sol['label'].apply(ast.literal_eval)
gt = {
    't1': sol_labels.apply(lambda x: x[0]).tolist(),
    't2': sol_labels.apply(lambda x: x[1]).tolist(),
    't3': sol_labels.apply(lambda x: x[2]).tolist(),
    't4': sol_labels.apply(lambda x: x[3]).tolist(),
}

sub_dir = '/home/yucheng/Desktop/ESG/submissions'
print("Submission                     T1      T2      T3      T4   Weighted   Kaggle")
print("-" * 85)

kaggle_scores = {
    's1_pertask_aug_single.csv': 0.56366,
    's2_ordinal_single.csv': 0.49216,
    's3_D_pertask_single.csv': 0.53134,
    's4_top3_a1.csv': 0.61646,
    's5_top3_mixed_D.csv': 0.60701,
    's6_all_a1_4way.csv': 0.57575,
    's7_all_5way.csv': 0.59297,
}

for f in sorted(os.listdir(sub_dir)):
    if not f.endswith('.csv'):
        continue
    df = pd.read_csv(os.path.join(sub_dir, f))
    pred_labels = df['label'].apply(ast.literal_eval)
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
    kg = kaggle_scores.get(f, 0)
    print("%-30s %7.4f %7.4f %7.4f %7.4f %9.5f %8.5f" % (f, scores['t1'], scores['t2'], scores['t3'], scores['t4'], weighted, kg))

print("\n=== s4_top3_a1 per-class breakdown ===")
df = pd.read_csv(os.path.join(sub_dir, 's4_top3_a1.csv'))
pred_labels = df['label'].apply(ast.literal_eval)
pred = {
    't1': pred_labels.apply(lambda x: x[0]).tolist(),
    't2': pred_labels.apply(lambda x: x[1]).tolist(),
    't3': pred_labels.apply(lambda x: x[2]).tolist(),
    't4': pred_labels.apply(lambda x: x[3]).tolist(),
}
task_names = {'t1': 'T1_promise', 't2': 'T2_evidence', 't3': 'T3_quality', 't4': 'T4_timeline'}
for task in ['t1','t2','t3','t4']:
    print("\n--- %s ---" % task_names[task])
    all_labels = sorted(set(gt[task]) | set(pred[task]))
    print(classification_report(gt[task], pred[task], labels=all_labels, digits=4, zero_division=0))
