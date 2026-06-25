"""Does the conditional-aux T3 (COND_FC1R) improve the joint-decode pipeline on valid?
Plug COND_FC1R for T1/T3@384 (vs current FC1R 3-seed), keep FC1R512 for T2/T4@512,
run joint decode (wgate=2) + kNN + clarity. Gate: T3 up & total up & bootstrap → worth
3 seeds + retrain_data + AIdea probe. Else: conditional aux rejected (label-limit holds)."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from pathlib import Path
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],"t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15};SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
C=Path("agent_cache/valid_probs")
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
sol["verification_timeline"]=sol["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
y={t:np.array(sol[SOL[t]].astype(str).tolist()) for t in LAB};N=len(sol)
clp=np.load("agent_cache/clarity_valid_probs.npz")["probs"];cca,ccf=clp.argmax(1),clp.max(1);CL3=["Clear","Not Clear","Misleading"]
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
def avg(runs):return {t:np.mean([np.load(C/f"{r}_f{f}.npz")[t] for r in runs for f in range(1,6)],axis=0) for t in LAB}
F512=avg(["FC1R512_roberta_dc_kfold_t3nc3","FC1R512_roberta_dc_kfold_t3nc3_s1","FC1R512_roberta_dc_kfold_t3nc3_s2"])
LG=lambda x:np.log(np.clip(x,1e-9,1))
def f1t(p,t):return f1_score(y[t],p[t],labels=sorted(set(y[t])),average="macro",zero_division=0)
def tot(p,idx=None):
    idx=np.arange(N) if idx is None else idx
    return sum(TW[t]*f1_score(y[t][idx],p[t][idx],labels=sorted(set(y[t])),average="macro",zero_division=0) for t in LAB)
def joint(src13):  # src13 provides t1,t3 ; T2/T4 from FC1R512
    p1,t3raw=src13["t1"],src13["t3"];p2,p4=F512["t2"],F512["t4"]
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
ref3=joint(avg(["FC1R_roberta_dc_kfold_t3nc3","FC1R_roberta_dc_kfold_t3nc3_s1","FC1R_roberta_dc_kfold_t3nc3_s2"]))
ref1=joint(avg(["FC1R_roberta_dc_kfold_t3nc3"]))   # seed s0 only — fair 1-vs-1
cond=joint(avg(["COND_FC1R_final"]))               # COND seed 42 (1 seed)
print(f"FC1R (3-seed) T1/T3 + joint: total={tot(ref3):.5f}  T3={f1t(ref3,'t3'):.4f}")
print(f"FC1R (1-seed s0)           : total={tot(ref1):.5f}  T3={f1t(ref1,'t3'):.4f}")
print(f"COND (1-seed)              : total={tot(cond):.5f}  T3={f1t(cond,'t3'):.4f}")
print("\n★ 公平比較 (1-vs-1,隔離 conditional aux 效果): COND vs FC1R_s0")
rng=np.random.RandomState(41);pd1=np.array([(lambda ix:tot(cond,ix)-tot(ref1,ix))(rng.randint(0,N,N)) for _ in range(1500)])
print(f"  Δtotal={tot(cond)-tot(ref1):+.5f}  ΔT3={f1t(cond,'t3')-f1t(ref1,'t3'):+.4f}  bootstrap P(cond>FC1R_s0)={(pd1>0).mean():.3f}")
t3_gain=f1t(cond,'t3')-f1t(ref1,'t3'); tot_ok=tot(cond)>=tot(ref1)-0.002
gate = (t3_gain>0.003) and tot_ok
verdict="PASS" if gate else "FAIL"
open("agent_cache/cond_gate.txt","w").write(verdict)
print(f"\n=== GATE: {verdict} ===  (條件: T3Δ>+0.003 且 total 不更差)")
print(f"  T3Δ(1v1)={t3_gain:+.4f}  total_ok={tot_ok}")
if gate:
    print("  → 下一步:訓 COND 2 個 final seed(123/456)做 3-vs-3 公平驗證 → 過則 RT_COND ×3(retrain)→ joint AIdea probe")
else:
    print("  → conditional aux 無效(confirm T3 label-limit 又一角度)→ 鎖定 0.6249,進報告")
