"""Evaluate the LLM gate-judge: (1) calibration vs gold (Yes-bias check), (2) blend the
LLM promise/evidence judgment into the T1/T2 gate, run joint decode (current mix sources,
wgate=2), compare valid. Only worth an AIdea probe if it beats the mix-source joint baseline
AND the LLM isn't just Yes-biased noise."""
import json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],"t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15};SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
C=Path("agent_cache/valid_probs")
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
sol["verification_timeline"]=sol["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
vdata=pd.read_csv("final_data/valid_data.csv",keep_default_na=False);ids=vdata["id"].astype(str).tolist()
y={t:np.array(sol[SOL[t]].astype(str).tolist()) for t in LAB};N=len(sol)
clp=np.load("agent_cache/clarity_valid_probs.npz")["probs"];cca,ccf=clp.argmax(1),clp.max(1);CL3=["Clear","Not Clear","Misleading"]
def avg(rc):return {t:np.mean([np.load(C/f"{rc}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for s in("","_s1","_s2") for f in range(1,6)],axis=0) for t in LAB}
F384=avg("FC1R");F512=avg("FC1R512")
p1,p2,p4=F384["t1"].copy(),F512["t2"].copy(),F512["t4"];t3raw=F384["t3"]
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)])
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
LG=lambda x:np.log(np.clip(x,1e-9,1))
def f1t(p,t):return f1_score(y[t],p[t],labels=sorted(set(y[t])),average="macro",zero_division=0)
def tot(p,idx=None):
    idx=np.arange(N) if idx is None else idx
    return sum(TW[t]*f1_score(y[t][idx],p[t][idx],labels=sorted(set(y[t])),average="macro",zero_division=0) for t in LAB)
def joint(p1,p2):
    t3p=t3raw.copy();t3p[:,1]*=np.exp(0.1);t3p/=t3p.sum(1,keepdims=True);t3p=0.6*t3p+0.4*knn
    t1=np.empty(N,object);t2=np.empty(N,object);t3=np.empty(N,object);t4=np.empty(N,object)
    for i in range(N):
        t4b=int(np.argmax(p4[i,:4]));t3b=int(np.argmax(t3p[i,:3]))
        s=[2*LG(p1[i,1])+2*LG(p2[i,2])+LG(t3p[i,3])+LG(p4[i,4]),
           2*LG(p1[i,0])+2*LG(p2[i,1])+LG(t3p[i,3])+LG(p4[i,t4b]),
           2*LG(p1[i,0])+2*LG(p2[i,0])+LG(t3p[i,t3b])+LG(p4[i,t4b])]
        bn=int(np.argmax(s))
        if bn==0:t1[i],t2[i],t3[i],t4[i]="No","N/A","N/A","N/A"
        elif bn==1:t1[i],t2[i],t3[i],t4[i]="Yes","No","N/A",LAB["t4"][t4b]
        else:t1[i],t2[i],t3[i],t4[i]="Yes","Yes",LAB["t3"][t3b],LAB["t4"][t4b]
    for i in range(N):
        if t3[i] in CL3 and ccf[i]>=0.7 and CL3[cca[i]]!="Misleading":t3[i]=CL3[cca[i]]
    return {"t1":t1,"t2":t2,"t3":t3,"t4":t4}
g=json.load(open("agent_cache/llm_gate_valid.json"))
# calibration
lp=np.array([1 if g.get(r,{}).get("promise")=="yes" else 0 for r in ids])
le=np.array([1 if g.get(r,{}).get("evidence")=="yes" else 0 for r in ids])
gold_p=(y["t1"]=="Yes").astype(int)
print(f"LLM promise=Yes {lp.sum()}/{N} | gold promise=Yes {gold_p.sum()} → LLM promise 準確率={(lp==gold_p).mean():.3f}")
print(f"  (若 LLM 幾乎全 Yes 或準確率≈gold base-rate → Yes-bias 噪音,無用)")
base=joint(p1,p2)
print(f"\nbaseline joint (mix sources, no LLM gate): valid={tot(base):.5f}")
rng=np.random.RandomState(31)
for beta in [0.2,0.35,0.5]:
    g1=p1.copy();g2=p2.copy()
    for i,r in enumerate(ids):
        j=g.get(r)
        if not j or j.get("conf",0)<6: continue
        if j["promise"]=="yes": g1[i]=(1-beta)*g1[i]+beta*np.array([1.0,0.0])
        elif j["promise"]=="no": g1[i]=(1-beta)*g1[i]+beta*np.array([0.0,1.0])
        if j["evidence"]=="yes": g2[i]=(1-beta)*g2[i]+beta*np.array([1.0,0.0,0.0])
        elif j["evidence"]=="no": g2[i]=(1-beta)*g2[i]+beta*np.array([0.0,1.0,0.0])
    g1/=g1.sum(1,keepdims=True);g2/=g2.sum(1,keepdims=True)
    p=joint(g1,g2)
    pd_=np.array([(lambda ix:tot(p,ix)-tot(base,ix))(rng.randint(0,N,N)) for _ in range(1500)])
    print(f"  +LLM gate β={beta}: valid={tot(p):.5f} Δ{tot(p)-tot(base):+.5f} T1={f1t(p,'t1'):.4f} T2={f1t(p,'t2'):.4f} P(>base)={(pd_>0).mean():.3f}")
print("\n判定:若 Δ>0 且 P 高 且 LLM 非 Yes-bias → 結構性 gate 訊號 → AIdea probe;否則否決(floor 0.6249 不動)。")
