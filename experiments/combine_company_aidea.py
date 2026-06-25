"""Build AIdea company-stack submission: 12-way RT base + kNN(T3) + NC bias +
clarity T3 overlay + company prior blend (T2 beta=0.2, T4 beta=0.4).
Mirrors company_stack_eval.py on the 2000 test using cached RT test probs."""
import numpy as np, pandas as pd
from pathlib import Path
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],
     "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
RT=Path("agent_cache/rt_test_probs"); KEYS=[f"rt_fc1{x}_s{s}" for x in ["","r","s","m"] for s in [0,1,2]]
tr=pd.read_csv("final_data/full_train_data.csv",keep_default_na=False)
tr["verification_timeline"]=tr["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
test=pd.read_csv("final_data/vpesg4k_test_2000.csv",keep_default_na=False)
N=len(test)
# 1. 12-way base test probs
base={t:np.mean([np.load(RT/f"{k}_f{f}.npz")[t] for k in KEYS for f in range(1,6)],axis=0) for t in LAB}
# 2. kNN T3
d=np.load("agent_cache/qwen3_embs_retrain.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["te"]/(np.linalg.norm(d["te"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True);fused_t3=0.6*t3+0.4*knn
# 3. company prior (from retrain) for t2,t4
def cprior(t):
    pc={};glob=tr[SOL[t]].astype(str).value_counts();garr=np.array([glob.get(l,0) for l in LAB[t]],float)+0.5;garr/=garr.sum()
    for comp,g in tr.groupby("company"):
        vc=g[SOL[t]].astype(str).value_counts();arr=np.array([vc.get(l,0) for l in LAB[t]],float)+0.5;arr/=arr.sum();pc[comp]=arr
    return np.array([pc.get(c,garr) for c in test["company"]])
p2=cprior("t2");p4=cprior("t4")
t2p=0.8*base["t2"]+0.2*p2; t4p=0.6*base["t4"]+0.4*p4
# 4. clarity overlay on T3
clp=np.load("agent_cache/clarity_test_probs.npz")["probs"];cca=clp.argmax(1);ccf=clp.max(1);CL3=["Clear","Not Clear","Misleading"]
pred={"t1":[LAB["t1"][i] for i in base["t1"].argmax(1)],"t2":[LAB["t2"][i] for i in t2p.argmax(1)],
      "t3":[LAB["t3"][i] for i in fused_t3.argmax(1)],"t4":[LAB["t4"][i] for i in t4p.argmax(1)]}
nt3=0
for i in range(N):
    if pred["t3"][i] in CL3 and ccf[i]>=0.7 and CL3[cca[i]]!="Misleading" and CL3[cca[i]]!=pred["t3"][i]:
        pred["t3"][i]=CL3[cca[i]];nt3+=1
# cascade
for i in range(N):
    if pred["t1"][i]=="No":pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
    elif pred["t2"][i]=="No":pred["t3"][i]="N/A"
out=pd.DataFrame({"id":test["id"],"promise_status":pred["t1"],"verification_timeline":pred["t4"],
                  "evidence_status":pred["t2"],"evidence_quality":pred["t3"]})
out.to_csv("official_sub/aidea_company_stack.csv",index=False)
from collections import Counter
# diff vs plain 12-way
b12=pd.read_csv("official_sub/aidea_rt_all12_knn.csv",keep_default_na=False)
dt2=int((out["evidence_status"]!=b12["evidence_status"]).sum());dt3=int((out["evidence_quality"]!=b12["evidence_quality"]).sum());dt4=int((out["verification_timeline"]!=b12["verification_timeline"]).sum())
print(f"[company_stack] clarity changed T3={nt3}; vs plain 12-way: T2 diff {dt2}, T3 diff {dt3}, T4 diff {dt4}")
print(f"  T2:{dict(Counter(out['evidence_status']))}  T4:{dict(Counter(out['verification_timeline']))}  T3:{dict(Counter(out['evidence_quality']))}")
print("-> official_sub/aidea_company_stack.csv")
