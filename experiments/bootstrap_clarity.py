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
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)])
knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True)
fused=dict(base);fused["t3"]=0.6*t3+0.4*knn
bp={t:[LAB[t][i] for i in np.argmax(fused[t],axis=1)] for t in LAB}
cl=np.load("agent_cache/clarity_valid_probs.npz");cp=cl["probs"];CLS=["Clear","Not Clear","Misleading"];ca=cp.argmax(1)
def mk(overlay):
    p={t:list(bp[t]) for t in LAB}
    if overlay:
        for i in range(N):
            if p["t3"][i] in CLS and cp[i].max()>=0.7: p["t3"][i]=CLS[ca[i]]
    for i in range(N):
        if p["t1"][i]=="No":p["t2"][i]=p["t3"][i]=p["t4"][i]="N/A"
        elif p["t2"][i]=="No":p["t3"][i]="N/A"
    return p
P0,P1=mk(False),mk(True)
def sc(P,idx):
    t=0.0
    for k in LAB:
        yt=[y[k][i] for i in idx];pt=[P[k][i] for i in idx]
        t+=TW[k]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
    return t
rng=np.random.RandomState(42);allidx=np.arange(N);pubidx=np.where(usage=="Public")[0]
for name,pool in [("ALL",allidx),("Public",pubidx)]:
    ds=[]
    for _ in range(2000):
        bi=rng.choice(pool,len(pool),replace=True);ds.append(sc(P1,bi)-sc(P0,bi))
    ds=np.array(ds);lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"{name}: overlay-base delta mean={ds.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] P(>0)={(ds>0).mean():.2f}")
