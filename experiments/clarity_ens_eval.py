"""Evaluate 3-seed clarity-head ensemble overlay on valid 399 vs single-seed.
Averages clarity valid probs across seeds 42/123/456, hard-replace overlay onto
12-way+kNN, bootstrap CI vs base."""
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
CLS=["Clear","Not Clear","Misleading"]
files={"s42":"agent_cache/clarity_valid_probs.npz","s123":"agent_cache/clarity_valid_probs_s123.npz","s456":"agent_cache/clarity_valid_probs_s456.npz"}
probs={k:np.load(v)["probs"] for k,v in files.items()}
ens=np.mean([probs[k] for k in files],axis=0)

def overlay(cp,thr):
    ca=cp.argmax(1);cc=cp.max(1);t3o=list(basep["t3"])
    for i in range(N):
        if basep["t3"][i] in CLS and cc[i]>=thr and CLS[ca[i]]!="Misleading":
            t3o[i]=CLS[ca[i]]
    pred={t:list(basep[t]) for t in LAB};pred["t3"]=t3o
    for i in range(N):
        if pred["t1"][i]=="No":pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No":pred["t3"][i]="N/A"
    return pred
def sc(pred,idx):
    t=0.0
    for k in LAB:
        yt=[y[k][i] for i in idx];pt=[pred[k][i] for i in idx]
        t+=TW[k]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
    return t
ALL=np.arange(N);PUB=np.where(usage=="Public")[0]
bp={t:list(basep[t]) for t in LAB}
for i in range(N):
    if bp["t1"][i]=="No":bp["t2"][i]=bp["t3"][i]=bp["t4"][i]="N/A"
    elif bp["t2"][i]=="No":bp["t3"][i]="N/A"
print(f"[base] ALL={sc(bp,ALL):.5f} Pub={sc(bp,PUB):.5f}")
print(f"{'variant':>26} {'ALL':>9} {'Public':>9}")
for nm,cp in [("single s42",probs["s42"]),("3-seed ensemble",ens)]:
    for thr in [0.7,0.8]:
        p=overlay(cp,thr);print(f"{(nm+' c>='+str(thr)):>26} {sc(p,ALL):9.5f} {sc(p,PUB):9.5f}")
# bootstrap 3-seed c>=0.7 vs base
pe=overlay(ens,0.7);rng=np.random.RandomState(42)
for nm,pool in [("ALL",ALL),("Public",PUB)]:
    ds=[];[ds.append(sc(pe,(bi:=rng.choice(pool,len(pool),replace=True)))-sc(bp,bi)) for _ in range(2000)]
    ds=np.array(ds);lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"  [3seed c>=0.7 {nm}] mean={ds.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] P(>0)={(ds>0).mean():.2f}")
print("frozen baseline 0.67882; single-seed best 0.68059")
