"""Stack the organizer-endorsed COMPANY prior onto our confirmed best
(12-way + kNN + clarity T3 overlay). Blend company prior into T2 and T4 (probs),
keep clarity overlay on T3. Sweep beta, bootstrap best vs clarity-only best.
Company is a structural feature: all 50 companies in train+valid+test (stratified)
-> prior transfers (low overfit risk vs thin text signals)."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],
     "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
G=["FC1","FC1R","FC1S","FC1M"];S=["","_s1","_s2"];C=Path("agent_cache/valid_probs")
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
vdf=pd.read_csv("final_data/valid_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
for df in (tr,sol): df["verification_timeline"]=df["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB}; N=len(sol); vcomp=vdf["company"].tolist(); usage=sol["Usage"].values
base={t:np.mean([np.load(C/f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for g in G for s in S for f in range(1,6)],axis=0) for t in LAB}
# company prior
prior={}
for t in LAB:
    pc={};glob=tr[SOL[t]].astype(str).value_counts();garr=np.array([glob.get(l,0) for l in LAB[t]],float)+0.5;garr/=garr.sum()
    for comp,g in tr.groupby("company"):
        vc=g[SOL[t]].astype(str).value_counts();arr=np.array([vc.get(l,0) for l in LAB[t]],float)+0.5;arr/=arr.sum();pc[comp]=arr
    prior[t]=np.array([pc.get(c,garr) for c in vcomp])
# kNN T3 + NC bias + clarity overlay
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True);fused_t3=0.6*t3+0.4*knn
clp=np.load("agent_cache/clarity_valid_probs.npz")["probs"];cca=clp.argmax(1);ccf=clp.max(1);CL3=["Clear","Not Clear","Misleading"]
def build(b2,b4,use_clarity=True):
    t1p=base["t1"];t2p=(1-b2)*base["t2"]+b2*prior["t2"];t4p=(1-b4)*base["t4"]+b4*prior["t4"]
    pred={"t1":[LAB["t1"][i] for i in t1p.argmax(1)],"t2":[LAB["t2"][i] for i in t2p.argmax(1)],
          "t3":[LAB["t3"][i] for i in fused_t3.argmax(1)],"t4":[LAB["t4"][i] for i in t4p.argmax(1)]}
    if use_clarity:
        for i in range(N):
            if pred["t3"][i] in CL3 and ccf[i]>=0.7 and CL3[cca[i]]!="Misleading":
                pred["t3"][i]=CL3[cca[i]]
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
clarity_only=build(0,0); base_only=build(0,0,use_clarity=False)
print(f"[base 12w+kNN]           ALL={sc(base_only,ALL):.5f} Pub={sc(base_only,PUB):.5f}")
print(f"[+clarity T3]            ALL={sc(clarity_only,ALL):.5f} Pub={sc(clarity_only,PUB):.5f}")
print("\n=== +company blend T2/T4 on top of clarity ===")
best=(sc(clarity_only,ALL),0,0)
for b2 in [0,0.1,0.2,0.3]:
    for b4 in [0,0.2,0.4,0.5,0.6]:
        p=build(b2,b4);s=sc(p,ALL)
        if s>best[0]: best=(s,b2,b4)
print(f"best β2={best[1]} β4={best[2]} → ALL={best[0]:.5f} Pub={sc(build(best[1],best[2]),PUB):.5f}")
for (b2,b4,nm) in [(0.2,0.5,"β2=.2 β4=.5"),(best[1],best[2],"best")]:
    p=build(b2,b4);print(f"  {nm}: ALL={sc(p,ALL):.5f} Pub={sc(p,PUB):.5f}")
# bootstrap best vs clarity-only
pbest=build(best[1],best[2]);rng=np.random.RandomState(42)
for nm,pool in [("ALL",ALL),("Public",PUB)]:
    ds=[];[ds.append(sc(pbest,(bi:=rng.choice(pool,len(pool),replace=True)))-sc(clarity_only,bi)) for _ in range(2000)]
    ds=np.array(ds);lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"  [{nm}] best vs clarity-only mean={ds.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] P(>0)={(ds>0).mean():.2f}")
print("frozen baseline 0.67882")
