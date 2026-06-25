"""P1: training-free logit adjustment (prior-correction) on the frozen 12-way T3.
Tests whether principled rebalancing p_c / freq_c^tau lifts Not Clear recall on
valid 399, vs the current crude NC-bias-0.1. Instant (numpy on cached probs).
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
ROOT = Path("/home/yucheng/Desktop/ESG")
LABELS = {"t1":["Yes","No"],"t2":["Yes","No","N/A"],
          "t3":["Clear","Not Clear","Misleading","N/A"],
          "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TASK_W = {"t1":0.20,"t2":0.30,"t3":0.35,"t4":0.15}
SOL = {"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
G=["FC1","FC1R","FC1S","FC1M"]; S=["","_s1","_s2"]; C=ROOT/"agent_cache/valid_probs"
sol=pd.read_csv(ROOT/"final_data/valid_solution_data.csv",keep_default_na=False)
tr=pd.read_csv(ROOT/"final_data/train_data.csv",keep_default_na=False)
sol["verification_timeline"]=sol["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
y={t:sol[SOL[t]].astype(str).tolist() for t in LABELS}; N=len(sol); usage=sol["Usage"].values
base={t:np.mean([np.load(C/f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t]
        for g in G for s in S for f in range(1,6)],axis=0) for t in LABELS}
# train T3 class frequency prior
t3cnt=tr["evidence_quality"].value_counts().to_dict()
prior=np.array([t3cnt.get(c,1) for c in LABELS["t3"]],dtype=float); prior/=prior.sum()
print("T3 train prior:",{c:round(p,4) for c,p in zip(LABELS["t3"],prior)})
# kNN T3
d=np.load(ROOT/"agent_cache/qwen3_embs_final.npz")
a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8)
b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8); sim=b@a.T
t3l=np.array([LABELS["t3"].index(l) for l in tr["evidence_quality"].astype(str)])
knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5]; w=np.exp(sim[i][tp]-sim[i][tp].max()); w/=w.sum()
    np.add.at(knn[i],t3l[tp],w); knn[i]/=knn[i].sum()

def evaluate(t3_base):
    fused=dict(base); fused["t3"]=0.6*t3_base+0.4*knn
    pred={t:[LABELS[t][i] for i in np.argmax(fused[t],axis=1)] for t in LABELS}
    for i in range(N):
        if pred["t1"][i]=="No": pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No": pred["t3"][i]="N/A"
    return pred

def score(pred,mask=None):
    idx=np.arange(N) if mask is None else np.where(mask)[0]; tot=0.0
    for t in LABELS:
        yt=[y[t][i] for i in idx]; pt=[pred[t][i] for i in idx]
        tot+=TASK_W[t]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
    return tot

def nc_recall(pred):
    truth=np.array(y["t3"]); pr=np.array(pred["t3"]); m=truth=="Not Clear"
    return int(((pr=="Not Clear")&m).sum()), int(m.sum())

# baseline: current crude NC bias 0.1
t3_cur=base["t3"].copy(); t3_cur[:,1]*=np.exp(0.1); t3_cur/=t3_cur.sum(1,keepdims=True)
pc=evaluate(t3_cur); nr=nc_recall(pc)
print(f"\n[current NC-bias 0.1]  total={score(pc):.5f}  Pub={score(pc,usage=='Public'):.5f} "
      f"Priv={score(pc,usage=='Private'):.5f}  NotClear recall={nr[0]}/{nr[1]}")

print("\n=== prior-correction sweep: p_c / prior_c^tau (then renorm, then kNN) ===")
print(f"{'tau':>5} {'total':>9} {'Public':>9} {'Private':>9} {'NotClear_recall':>16}  T3-perclass-F1")
best=(score(pc),0.1,'ncbias')
for tau in [0.0,0.15,0.25,0.4,0.5,0.7,1.0]:
    t3a=base["t3"]/np.power(prior,tau)[None,:]; t3a/=t3a.sum(1,keepdims=True)
    p=evaluate(t3a); nr=nc_recall(p)
    f=f1_score(y["t3"],p["t3"],labels=sorted(set(y["t3"])),average=None,zero_division=0)
    perclass={l:round(v,3) for l,v in zip(sorted(set(y["t3"])),f)}
    tot=score(p)
    print(f"{tau:5.2f} {tot:9.5f} {score(p,usage=='Public'):9.5f} {score(p,usage=='Private'):9.5f} "
          f"{nr[0]:>7}/{nr[1]:<6} {perclass}")
    if tot>best[0]: best=(tot,tau,'prior')
print(f"\nBEST: total={best[0]:.5f} via {best[2]} tau={best[1]}  (frozen baseline 0.67882)")
print("Decision: if best prior-tau > 0.67882 AND NotClear recall up without hurting -> "
      "principled logit-adj is a free win; else Not Clear isn't liftable post-hoc -> needs P3 retrain")
