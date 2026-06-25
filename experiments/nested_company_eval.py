"""Nested validation of company-stack: split valid 399 in 2 halves; tune beta on
one half, evaluate on the held-out half (both directions). Tests whether the
beta-tuning overfits valid. Company prior comes from TRAIN (unaffected)."""
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
tr["comp"]=tr["company"].str.strip().str.lower(); vcomp=vdf["company"].str.strip().str.lower().tolist()
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB}; N=len(sol)
base={t:np.mean([np.load(C/f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for g in G for s in S for f in range(1,6)],axis=0) for t in LAB}
def cprior(t):
    glob=tr[SOL[t]].astype(str).value_counts();garr=np.array([glob.get(l,0) for l in LAB[t]],float)+0.5;garr/=garr.sum();pc={}
    for k,g in tr.groupby("comp"):
        vc=g[SOL[t]].astype(str).value_counts();arr=np.array([vc.get(l,0) for l in LAB[t]],float)+0.5;arr/=arr.sum();pc[k]=arr
    return np.array([pc.get(c,garr) for c in vcomp])
P2=cprior("t2");P4=cprior("t4")
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True);fused_t3=0.6*t3+0.4*knn
clp=np.load("agent_cache/clarity_valid_probs.npz")["probs"];cca=clp.argmax(1);ccf=clp.max(1);CL3=["Clear","Not Clear","Misleading"]
def preds(b2,b4):
    t2p=(1-b2)*base["t2"]+b2*P2;t4p=(1-b4)*base["t4"]+b4*P4
    pr={"t1":[LAB["t1"][i] for i in base["t1"].argmax(1)],"t2":[LAB["t2"][i] for i in t2p.argmax(1)],
        "t3":[LAB["t3"][i] for i in fused_t3.argmax(1)],"t4":[LAB["t4"][i] for i in t4p.argmax(1)]}
    for i in range(N):
        if pr["t3"][i] in CL3 and ccf[i]>=0.7 and CL3[cca[i]]!="Misleading": pr["t3"][i]=CL3[cca[i]]
    for i in range(N):
        if pr["t1"][i]=="No":pr["t2"][i]=pr["t3"][i]=pr["t4"][i]="N/A"
        elif pr["t2"][i]=="No":pr["t3"][i]="N/A"
    return pr
def sc(pr,idx):
    return sum(TW[k]*f1_score([y[k][i] for i in idx],[pr[k][i] for i in idx],labels=sorted(set([y[k][i] for i in idx])),average="macro",zero_division=0) for k in LAB)
rng=np.random.RandomState(0); perm=rng.permutation(N); A=perm[:N//2]; B=perm[N//2:]
GRID=[(b2,b4) for b2 in [0,.1,.2,.3] for b4 in [0,.2,.4,.5,.6]]
def best_beta(idx):
    return max(GRID,key=lambda bb: sc(preds(*bb),idx))
for name,tune,test in [("A→B",A,B),("B→A",B,A)]:
    bb=best_beta(tune);
    base_test=sc(preds(0,0),test); stack_test=sc(preds(*bb),test)
    print(f"[{name}] β tuned on tune-half={bb} | held-out base={base_test:.5f} → company-stack={stack_test:.5f} ({stack_test-base_test:+.5f})")
print(f"\n[full valid] base={sc(preds(0,0),np.arange(N)):.5f} stack(β=.2,.4)={sc(preds(.2,.4),np.arange(N)):.5f}")
print("→ 若兩個方向 held-out 都 +,β 不過擬合、company-stack 泛化")
