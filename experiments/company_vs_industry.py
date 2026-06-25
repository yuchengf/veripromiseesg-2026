"""Robustness: company-prior vs industry-prior vs hierarchical (company shrunk to
industry) for the T2+T4 blend, on valid 399, with bootstrap. Also normalizes
company case (Wistron/wistron = same firm)."""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],
     "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
G=["FC1","FC1R","FC1S","FC1M"];S=["","_s1","_s2"];C=Path("agent_cache/valid_probs")
INDUSTRY={  # 50 Taiwan firms -> sector
 'financial':['cathay','ctbc','fubon','hnfhc','ffhc','kgi','mega','taishin','esfh','yuanta','tcfhc','scsb','chailease'],
 'telecom':['cht','twm','fet'],
 'shipping':['emc','ymtc','wanhai'],
 'materials':['tcc','fpc','npc','csc','yfy','fpcc'],
 'retailfood':['unipresident','pcsc','hotaimotor'],
 'semiconductor':['tsmc','umc','mediatek','alchip','novatek','aseh','yageo','largan'],
 'electronics':['accton','acl','delta','honhai','wistron','pegatron','avc','rt','qci','emc2','wiwynn','pec','ltc'],
}
comp2ind={c:k for k,v in INDUSTRY.items() for c in v}
def norm(c): return str(c).strip().lower()
def ind(c): return comp2ind.get(norm(c),'other')
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
vdf=pd.read_csv("final_data/valid_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
for df in (tr,sol): df["verification_timeline"]=df["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
tr["comp"]=tr["company"].map(norm); tr["ind"]=tr["company"].map(ind)
vcomp=vdf["company"].map(norm).tolist(); vind=vdf["company"].map(ind).tolist()
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB}; N=len(sol); usage=sol["Usage"].values
print("產業分布(train):",tr["ind"].value_counts().to_dict())
print("未映射到產業的公司:",sorted(set(tr[tr['ind']=='other']['comp'].unique())))
base={t:np.mean([np.load(C/f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for g in G for s in S for f in range(1,6)],axis=0) for t in LAB}
def prior_by(keycol_train, keys_valid, t, k_smooth=0.5):
    glob=tr[SOL[t]].astype(str).value_counts();garr=np.array([glob.get(l,0) for l in LAB[t]],float)+0.5;garr/=garr.sum()
    pc={}
    for key,g in tr.groupby(keycol_train):
        vc=g[SOL[t]].astype(str).value_counts();arr=np.array([vc.get(l,0) for l in LAB[t]],float)+k_smooth;arr/=arr.sum();pc[key]=arr
    return np.array([pc.get(k,garr) for k in keys_valid])
def hier_prior(t):  # company shrunk toward industry (k=10 pseudo-counts of industry)
    ip={}
    for key,g in tr.groupby("ind"):
        vc=g[SOL[t]].astype(str).value_counts();arr=np.array([vc.get(l,0) for l in LAB[t]],float);ip[key]=arr/arr.sum()
    out=[]
    cp_counts={}
    for key,g in tr.groupby("comp"):
        vc=g[SOL[t]].astype(str).value_counts();cp_counts[key]=(np.array([vc.get(l,0) for l in LAB[t]],float), comp2ind.get(key,'other'))
    glob=tr[SOL[t]].astype(str).value_counts();garr=np.array([glob.get(l,0) for l in LAB[t]],float);garr/=garr.sum()
    for c in vcomp:
        if c in cp_counts:
            cnt,iname=cp_counts[c]; ipv=ip.get(iname,garr)
            arr=(cnt+10*ipv)/(cnt.sum()+10); out.append(arr)
        else: out.append(garr)
    return np.array(out)
priors={
 "company":{t:prior_by("comp",vcomp,t) for t in ["t2","t4"]},
 "industry":{t:prior_by("ind",vind,t) for t in ["t2","t4"]},
 "hierarchical":{t:hier_prior(t) for t in ["t2","t4"]},
}
# clarity T3 + kNN for full pipeline
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((N,4))
for i in range(N):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True);fused_t3=0.6*t3+0.4*knn
clp=np.load("agent_cache/clarity_valid_probs.npz")["probs"];cca=clp.argmax(1);ccf=clp.max(1);CL3=["Clear","Not Clear","Misleading"]
def build(P,b2=0.2,b4=0.4):
    t2p=(1-b2)*base["t2"]+b2*P["t2"];t4p=(1-b4)*base["t4"]+b4*P["t4"]
    pred={"t1":[LAB["t1"][i] for i in base["t1"].argmax(1)],"t2":[LAB["t2"][i] for i in t2p.argmax(1)],
          "t3":[LAB["t3"][i] for i in fused_t3.argmax(1)],"t4":[LAB["t4"][i] for i in t4p.argmax(1)]}
    for i in range(N):
        if pred["t3"][i] in CL3 and ccf[i]>=0.7 and CL3[cca[i]]!="Misleading": pred["t3"][i]=CL3[cca[i]]
    for i in range(N):
        if pred["t1"][i]=="No":pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No":pred["t3"][i]="N/A"
    return pred
def sc(pred,idx):
    return sum(TW[k]*f1_score([y[k][i] for i in idx],[pred[k][i] for i in idx],labels=sorted(set(y[k])),average="macro",zero_division=0) for k in LAB)
ALL=np.arange(N);PUB=np.where(usage=="Public")[0]
print(f"\n{'prior':>14} {'ALL':>9} {'Public':>9} {'Private':>9}")
preds={}
for nm,P in priors.items():
    p=build(P);preds[nm]=p
    print(f"{nm:>14} {sc(p,ALL):9.5f} {sc(p,PUB):9.5f} {sc(p,np.where(usage=='Private')[0]):9.5f}")
# bootstrap each vs base(no prior)
basep=build({"t2":base["t2"],"t4":base["t4"]},0,0)
print(f"{'(no prior)':>14} {sc(basep,ALL):9.5f} {sc(basep,PUB):9.5f}")
rng=np.random.RandomState(42)
for nm in priors:
    ds=[];[ds.append(sc(preds[nm],(bi:=rng.choice(ALL,N,replace=True)))-sc(basep,bi)) for _ in range(2000)]
    ds=np.array(ds);lo,hi=np.percentile(ds,[2.5,97.5])
    print(f"  {nm} vs base: mean={ds.mean():+.5f} CI[{lo:+.5f},{hi:+.5f}] P(>0)={(ds>0).mean():.2f}")
