"""Compare 3-way (FC1R x3) @512 vs @384 on valid, with kNN(T3 a0.4,NCbias.1) +
optional clarity overlay. Tells us if max_length 512 helps the champion path."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],"t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15};SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
C=Path("agent_cache/valid_probs")
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
sol["verification_timeline"]=sol["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB};N=len(sol);usage=sol["Usage"].values
def avg(runs):
    return {t:np.mean([np.load(C/f"{r}_f{f}.npz")[t] for r in runs for f in range(1,6)],axis=0) for t in LAB}
RUNS384=["FC1R_roberta_dc_kfold_t3nc3","FC1R_roberta_dc_kfold_t3nc3_s1","FC1R_roberta_dc_kfold_t3nc3_s2"]
RUNS512=["FC1R512_roberta_dc_kfold_t3nc3","FC1R512_roberta_dc_kfold_t3nc3_s1","FC1R512_roberta_dc_kfold_t3nc3_s2"]
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
clp=np.load("agent_cache/clarity_valid_probs.npz")["probs"];cca=clp.argmax(1);ccf=clp.max(1);CL3=["Clear","Not Clear","Misleading"]
def score(base,clarity):
    t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True);fused=0.6*t3+0.4*knn
    pred={t:[LAB[t][i] for i in (fused if t=="t3" else base[t]).argmax(1)] for t in LAB}
    if clarity:
        for i in range(N):
            if pred["t3"][i] in CL3 and ccf[i]>=0.7 and CL3[cca[i]]!="Misleading": pred["t3"][i]=CL3[cca[i]]
    for i in range(N):
        if pred["t1"][i]=="No":pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No":pred["t3"][i]="N/A"
    tot=sum(TW[t]*f1_score(y[t],pred[t],labels=sorted(set(y[t])),average="macro",zero_division=0) for t in LAB)
    parts={t:round(f1_score(y[t],pred[t],labels=sorted(set(y[t])),average="macro",zero_division=0),4) for t in LAB}
    return tot,parts
for tag,runs in [("@384",RUNS384),("@512",RUNS512)]:
    try:
        base=avg(runs)
        for cl in [False,True]:
            tot,parts=score(base,cl)
            print(f"3-way {tag} {'+clarity' if cl else '       '}: valid={tot:.5f}  {parts}")
    except FileNotFoundError as e:
        print(f"{tag}: 缺檔(尚未訓練完?) {e}")
print("\n→ 若 @512 > @384(尤其 +clarity 版)且 T3/T4 升 → max_length 512 結構性有效 → 重訓 RT@512 上 AIdea")
