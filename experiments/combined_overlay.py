"""Stack the two confirmed conditional-head overlays: T2 evidence head + T3
clarity head, onto the frozen 12-way+kNN, on valid 399. Apply T2 first (it
cascades to T3), then clarity on surviving non-N/A T3 rows. Bootstrap CI vs base.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],
     "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
G=["FC1","FC1R","FC1S","FC1M"];S=["","_s1","_s2"];C=Path("agent_cache/valid_probs")
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
sol["verification_timeline"]=sol["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB};N=len(sol);usage=sol["Usage"].values
base={t:np.mean([np.load(C/f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for g in G for s in S for f in range(1,6)],axis=0) for t in LAB}
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True)
fused=dict(base);fused["t3"]=0.6*t3+0.4*knn
basep={t:[LAB[t][i] for i in np.argmax(fused[t],axis=1)] for t in LAB}
# heads
T2H=["Yes","No"]; t2p=np.load("agent_cache/t2_valid_probs.npz")["probs"]; t2a=t2p.argmax(1); t2c=t2p.max(1)
T3H=["Clear","Not Clear","Misleading"]; cp=np.load("agent_cache/clarity_valid_probs.npz")["probs"]; ca=cp.argmax(1); cc=cp.max(1)

def build(use_t2, use_t3, t2thr=0.8, t3thr=0.7):
    pred={t:list(basep[t]) for t in LAB}
    if use_t2:
        for i in range(N):
            if basep["t1"][i]=="Yes" and basep["t2"][i] in T2H and t2c[i]>=t2thr:
                pred["t2"][i]=T2H[t2a[i]]
    if use_t3:
        for i in range(N):
            if pred["t2"][i]=="Yes" and pred["t3"][i] in T3H and cc[i]>=t3thr and T3H[ca[i]]!="Misleading":
                pred["t3"][i]=T3H[ca[i]]
    for i in range(N):  # cascade
        if pred["t1"][i]=="No": pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No": pred["t3"][i]="N/A"
    return pred

def score(pred,idx):
    t=0.0
    for k in LAB:
        yt=[y[k][i] for i in idx];pt=[pred[k][i] for i in idx]
        t+=TW[k]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
    return t
ALL=np.arange(N);PUB=np.where(usage=="Public")[0]
variants={"base":build(False,False),"T2 only":build(True,False),
          "T3 only":build(False,True),"T2+T3 (stacked)":build(True,True)}
print(f"{'variant':>18} {'ALL':>9} {'Public':>9} {'Private':>9}")
for nm,p in variants.items():
    print(f"{nm:>18} {score(p,ALL):9.5f} {score(p,PUB):9.5f} {score(p,np.where(usage=='Private')[0]):9.5f}")
# bootstrap stacked vs base
pb=variants["base"];ps=variants["T2+T3 (stacked)"];rng=np.random.RandomState(42)
for nm,pool in [("ALL",ALL),("Public",PUB)]:
    ds=[]
    for _ in range(2000):
        bi=rng.choice(pool,len(pool),replace=True);ds.append(score(ps,bi)-score(pb,bi))
    ds=np.array(ds);lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"  [{nm}] stacked-base mean={ds.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] P(>0)={(ds>0).mean():.2f}")
print("frozen baseline 0.67882")
