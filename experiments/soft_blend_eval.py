"""Anti-overfit overlay test on valid 399: compare HARD-replace (valid-tuned
threshold, overfit-prone) vs SOFT-blend (beta mix of clarity into base T3, no
threshold) using a given clarity-prob cache. Soft blend depends on no
valid-tuned cutoff -> less overfit. Bootstrap CI for the best variant.

Usage: python soft_blend_eval.py [clarity_probs.npz]
"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
PROB = sys.argv[1] if len(sys.argv) > 1 else "agent_cache/clarity_valid_probs.npz"
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
base_t3=0.6*t3+0.4*knn   # 4-way base T3 prob
bp={t:[LAB[t][i] for i in np.argmax(base["t1" if t=="t1" else t],axis=1)] for t in ["t1","t2","t4"]}
CLS=["Clear","Not Clear","Misleading"]
cl=np.load(PROB);cp=cl["probs"];ca=cp.argmax(1);conf=cp.max(1)
clar4=np.concatenate([cp,np.zeros((N,1))],axis=1)  # pad N/A=0

def score_from_t3prob(t3prob,idx=None):
    pred={"t1":list(bp["t1"]),"t2":list(bp["t2"]),"t4":list(bp["t4"]),
          "t3":[LAB["t3"][i] for i in np.argmax(t3prob,axis=1)]}
    for i in range(N):
        if pred["t1"][i]=="No":pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No":pred["t3"][i]="N/A"
    I=np.arange(N) if idx is None else idx;tot=0.0
    for t in LAB:
        yt=[y[t][i] for i in I];pt=[pred[t][i] for i in I]
        tot+=TW[t]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
    return tot,pred

def hard_t3(thr):
    pr=base_t3.copy();out=np.argmax(pr,1)
    t3p=np.zeros_like(pr)  # build a prob that argmaxes to chosen label
    res=base_t3.copy()
    # emulate hard replace by setting chosen class prob to 1 where applied
    lab=[LAB["t3"][i] for i in np.argmax(base_t3,1)]
    final=np.argmax(base_t3,1).copy()
    for i in range(N):
        if lab[i] in CLS and conf[i]>=thr and CLS[ca[i]]!="Misleading":
            final[i]=LAB["t3"].index(CLS[ca[i]])
    oh=np.eye(4)[final];return oh

pub=usage=="Public"
b_all,_=score_from_t3prob(base_t3);b_pub,_=score_from_t3prob(base_t3,np.where(pub)[0])
print(f"PROB={PROB}")
print(f"[base 12-way+kNN] ALL={b_all:.5f} Public={b_pub:.5f}\n")
print(f"{'variant':>22} {'ALL':>9} {'Public':>9}")
for thr in [0.7,0.8]:
    oh=hard_t3(thr);s,_=score_from_t3prob(oh);sp,_=score_from_t3prob(oh,np.where(pub)[0])
    print(f"{('hard replace c>='+str(thr)):>22} {s:9.5f} {sp:9.5f}")
best=(b_all,"base",base_t3)
for beta in [0.2,0.3,0.4,0.5]:
    bl=(1-beta)*base_t3+beta*clar4;bl/=bl.sum(1,keepdims=True)
    s,_=score_from_t3prob(bl);sp,_=score_from_t3prob(bl,np.where(pub)[0])
    print(f"{('soft blend b='+str(beta)):>22} {s:9.5f} {sp:9.5f}")
    if s>best[0]:best=(s,f"soft b={beta}",bl)
# bootstrap CI of best vs base
_,pbase=score_from_t3prob(base_t3);_,pbest=score_from_t3prob(best[2])
def wscore(P,idx):
    t=0.0
    for k in LAB:
        yt=[y[k][i] for i in idx];pt=[P[k][i] for i in idx]
        t+=TW[k]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
    return t
rng=np.random.RandomState(42)
for nm,pool in [("ALL",np.arange(N)),("Public",np.where(pub)[0])]:
    ds=np.array([wscore(pbest,rng.choice(pool,len(pool),replace=True))-wscore(pbase,rng.choice(pool,len(pool),replace=True)) for _ in range(1500)])
    # use paired resample
    ds=[]
    for _ in range(1500):
        bi=rng.choice(pool,len(pool),replace=True);ds.append(wscore(pbest,bi)-wscore(pbase,bi))
    ds=np.array(ds);lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"  [{nm}] best={best[1]} delta vs base mean={ds.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] P(>0)={(ds>0).mean():.2f}")
print(f"\nBEST variant: {best[1]} ALL={best[0]:.5f} (base {b_all:.5f}, frozen 0.67882)")
