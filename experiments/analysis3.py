"""Simulate impact of adding T2=No → T3=N/A post-processing rule."""
import pandas as pd, ast, os, copy
from sklearn.metrics import f1_score

sol = pd.read_csv('/home/yucheng/Desktop/ESG/2026-esg-classification-challenge/solution_data.csv')
sol_labels = sol['label'].apply(ast.literal_eval)
gt = {
    't1': sol_labels.apply(lambda x: x[0]).tolist(),
    't2': sol_labels.apply(lambda x: x[1]).tolist(),
    't3': sol_labels.apply(lambda x: x[2]).tolist(),
    't4': sol_labels.apply(lambda x: x[3]).tolist(),
}

# Score function
def score(pred):
    scores = {}
    for task in ['t1','t2','t3','t4']:
        all_labels = sorted(set(gt[task]) | set(pred[task]))
        scores[task] = f1_score(gt[task], pred[task], labels=all_labels, average='macro')
    scores['weighted'] = 0.20*scores['t1'] + 0.30*scores['t2'] + 0.35*scores['t3'] + 0.15*scores['t4']
    return scores

def score_public(pred):
    pub_mask = (sol['Usage'] == 'Public').tolist()
    gt_pub = {t: [v for v, m in zip(gt[t], pub_mask) if m] for t in gt}
    pred_pub = {t: [v for v, m in zip(pred[t], pub_mask) if m] for t in pred}
    scores = {}
    for task in ['t1','t2','t3','t4']:
        all_labels = sorted(set(gt_pub[task]) | set(pred_pub[task]))
        scores[task] = f1_score(gt_pub[task], pred_pub[task], labels=all_labels, average='macro')
    scores['weighted'] = 0.20*scores['t1'] + 0.30*scores['t2'] + 0.35*scores['t3'] + 0.15*scores['t4']
    return scores

sub_dir = '/home/yucheng/Desktop/ESG/submissions'
subs = ['s4_top3_a1.csv', 's1_pertask_aug_single.csv', 's5_top3_mixed_D.csv']

for fname in subs:
    df = pd.read_csv(os.path.join(sub_dir, fname))
    pred_labels = df['label'].apply(ast.literal_eval)
    orig = {
        't1': pred_labels.apply(lambda x: x[0]).tolist(),
        't2': pred_labels.apply(lambda x: x[1]).tolist(),
        't3': pred_labels.apply(lambda x: x[2]).tolist(),
        't4': pred_labels.apply(lambda x: x[3]).tolist(),
    }

    # Apply extended NA rule: T1=No → T2/T3/T4=N/A AND T2=No → T3=N/A
    fixed = copy.deepcopy(orig)
    changes = 0
    for i in range(len(fixed['t1'])):
        if fixed['t2'][i] == 'No' and fixed['t3'][i] != 'N/A':
            fixed['t3'][i] = 'N/A'
            changes += 1

    s_orig = score(orig)
    s_fixed = score(fixed)
    s_orig_pub = score_public(orig)
    s_fixed_pub = score_public(fixed)

    print("=== %s ===" % fname)
    print("  Changes: %d samples T2=No → T3=N/A" % changes)
    print("  ALL:    T3 %.4f→%.4f  Weighted %.5f→%.5f (Δ%.5f)" %
          (s_orig['t3'], s_fixed['t3'], s_orig['weighted'], s_fixed['weighted'],
           s_fixed['weighted'] - s_orig['weighted']))
    print("  PUBLIC: T3 %.4f→%.4f  Weighted %.5f→%.5f (Δ%.5f)" %
          (s_orig_pub['t3'], s_fixed_pub['t3'], s_orig_pub['weighted'], s_fixed_pub['weighted'],
           s_fixed_pub['weighted'] - s_orig_pub['weighted']))
    print()

# Also check: what if we add T2=N/A → T3=N/A too (in case of T2 N/A errors)
print("=== Additional rules analysis on s4 ===")
df = pd.read_csv(os.path.join(sub_dir, 's4_top3_a1.csv'))
pred_labels = df['label'].apply(ast.literal_eval)
orig = {
    't1': pred_labels.apply(lambda x: x[0]).tolist(),
    't2': pred_labels.apply(lambda x: x[1]).tolist(),
    't3': pred_labels.apply(lambda x: x[2]).tolist(),
    't4': pred_labels.apply(lambda x: x[3]).tolist(),
}

# Check how many T2=No predictions match T2=No ground truth
t2_no_pred = sum(1 for x in orig['t2'] if x == 'No')
t2_no_gt = sum(1 for x in gt['t2'] if x == 'No')
print("T2=No predictions: %d, Ground truth: %d" % (t2_no_pred, t2_no_gt))

# Check T2=No recall/precision
t2_no_correct = sum(1 for p, g in zip(orig['t2'], gt['t2']) if p == 'No' and g == 'No')
print("T2=No precision: %d/%d = %.3f" % (t2_no_correct, t2_no_pred, t2_no_correct/max(t2_no_pred,1)))
print("T2=No recall: %d/%d = %.3f" % (t2_no_correct, t2_no_gt, t2_no_correct/max(t2_no_gt,1)))

# Check: for the training data, verify T2=No → T3 label
print("\n=== Verify train: T2=No → T3 from label column ===")
train = pd.read_csv('/home/yucheng/Desktop/ESG/2026-esg-classification-challenge/train_data.csv')
for _, row in train[train['evidence_status'] == 'No'].head(5).iterrows():
    label = ast.literal_eval(row['label'])
    print("  id=%s T1=%s T2=%s T3=%s T4=%s (col_T3=%s)" %
          (row['id'], label[0], label[1], label[2], label[3], row['evidence_quality']))
