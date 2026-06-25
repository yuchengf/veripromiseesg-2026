"""Conditional EVIDENCE head for T2 (the next bottleneck: 30% weight, No-class
F1=0.523). Same recipe that won on T3: train a 2-way Yes/No head ONLY on T1=Yes
rows (P(evidence|promise=Yes)), balanced softmax + label smoothing, overlay onto
the 12-way T2 where the 12-way predicts T1=Yes and base T2 in {Yes,No}.
Usage: python t2_head.py [--smoke]
"""
import os, sys
os.chdir("/home/yucheng/Desktop/ESG")
SMOKE = "--smoke" in sys.argv
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
torch.manual_seed(42); np.random.seed(42)
BACKBONE="hfl/chinese-roberta-wwm-ext-large"; MAXLEN,BS,EPOCHS,LR,LS=384,8,(1 if SMOKE else 5),1e-5,0.1
CLS=["Yes","No"]; K=2
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],
     "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
vdf=pd.read_csv("final_data/valid_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
sol["verification_timeline"]=sol["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
cl=tr[(tr["promise_status"]=="Yes") & (tr["evidence_status"].isin(CLS))].reset_index(drop=True)
if SMOKE: cl=cl.sample(120,random_state=0).reset_index(drop=True)
texts=cl["data"].astype(str).tolist(); ylab=np.array([CLS.index(x) for x in cl["evidence_status"]])
prior=np.bincount(ylab,minlength=K).astype(float); prior/=prior.sum()
logprior=torch.tensor(np.log(prior+1e-9),dtype=torch.float32).cuda()
print(f"[t2head] train(T1=Yes)={len(cl)} dist={np.bincount(ylab,minlength=K).tolist()} {CLS} SMOKE={SMOKE}",flush=True)
vtexts=vdf["data"].astype(str).tolist(); Nv=len(vtexts)
tok=AutoTokenizer.from_pretrained(BACKBONE)
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=AutoModel.from_pretrained(BACKBONE)
        self.drop=nn.Dropout(0.1); self.head=nn.Linear(self.enc.config.hidden_size,K)
    def forward(self,ids,m): return self.head(self.drop(self.enc(input_ids=ids,attention_mask=m).last_hidden_state[:,0]))
def enc_batch(bt):
    e=tok(bt,padding=True,truncation=True,max_length=MAXLEN,return_tensors="pt"); return e["input_ids"].cuda(),e["attention_mask"].cuda()
@torch.no_grad()
def predict(model,bt):
    model.eval(); out=[]
    for i in range(0,len(bt),16):
        ids,m=enc_batch(bt[i:i+16])
        with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(ids,m)
        out.append(F.softmax(lo.float(),-1).cpu().numpy())
    return np.concatenate(out)
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); val_probs=np.zeros((Nv,K)); oof=[]
for fi,(tri,vi) in enumerate(skf.split(texts,ylab),1):
    model=Net().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    steps=max(1,-(-len(tri)//BS))*EPOCHS; sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.1)
    for ep in range(EPOCHS):
        model.train(); perm=np.random.permutation(tri)
        for j in range(0,len(perm),BS):
            bi=perm[j:j+BS]
            if len(bi)==0: continue
            ids,m=enc_batch([texts[k] for k in bi])
            with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(ids,m)
            loss=F.cross_entropy(lo.float()+logprior,torch.tensor(ylab[bi]).cuda(),label_smoothing=LS)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad()
    hp=predict(model,[texts[k] for k in vi]).argmax(1); oof.append((hp==ylab[vi]).mean())
    val_probs+=predict(model,vtexts); print(f"  fold{fi} heldout_acc={oof[-1]:.3f}",flush=True)
    del model; torch.cuda.empty_cache()
    if SMOKE: break
val_probs/=(1 if SMOKE else 5)
np.savez_compressed("agent_cache/t2_valid_probs.npz",probs=val_probs,classes=np.array(CLS))
print(f"[t2head] mean heldout acc={np.mean(oof):.3f}",flush=True)
if SMOKE: print("[SMOKE] OK"); sys.exit(0)
# overlay onto 12-way
G=["FC1","FC1R","FC1S","FC1M"];S=["","_s1","_s2"];C=Path("agent_cache/valid_probs")
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB}; usage=sol["Usage"].values
base={t:np.mean([np.load(C/f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t] for g in G for s in S for f in range(1,6)],axis=0) for t in LAB}
d=np.load("agent_cache/qwen3_embs_final.npz");a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8);b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8);sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)]);knn=np.zeros((Nv,4))
for i in range(Nv):
    tp=np.argsort(sim[i])[::-1][:5];w=np.exp(sim[i][tp]-sim[i][tp].max());w/=w.sum();np.add.at(knn[i],t3l[tp],w);knn[i]/=knn[i].sum()
t3=base["t3"].copy();t3[:,1]*=np.exp(0.1);t3/=t3.sum(1,keepdims=True)
fused=dict(base);fused["t3"]=0.6*t3+0.4*knn
basep={t:[LAB[t][i] for i in np.argmax(fused[t],axis=1)] for t in LAB}
ca=val_probs.argmax(1);conf=val_probs.max(1)
def score(t2pred):
    pred={t:list(basep[t]) for t in LAB}; pred["t2"]=list(t2pred)
    for i in range(Nv):
        if pred["t1"][i]=="No": pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No": pred["t3"][i]="N/A"
    tot=0.0;parts={}
    for t in LAB:
        f=f1_score(y[t],pred[t],labels=sorted(set(y[t])),average="macro",zero_division=0);tot+=TW[t]*f;parts[t]=f
    pub=sum(TW[t]*f1_score([y[t][i] for i in np.where(usage=='Public')[0]],[pred[t][i] for i in np.where(usage=='Public')[0]],labels=sorted(set(y[t])),average="macro",zero_division=0) for t in LAB)
    return tot,parts,pub
bt,bp,bpub=score(basep["t2"])
print(f"\n[12-way base] total={bt:.5f} Pub={bpub:.5f} T2={bp['t2']:.4f}",flush=True)
print("=== conditional T2 head overlay (replace where T1=Yes & base T2 in Yes/No & conf>=thr) ===")
for thr in [0.5,0.6,0.7,0.8,0.9]:
    t2o=list(basep["t2"]);ch=0
    for i in range(Nv):
        if basep["t1"][i]=="Yes" and basep["t2"][i] in CLS and conf[i]>=thr and CLS[ca[i]]!=basep["t2"][i]:
            t2o[i]=CLS[ca[i]];ch+=1
    tot,parts,pub=score(t2o)
    print(f"  conf>={thr}: changed {ch:3d}  total={tot:.5f} Pub={pub:.5f} T2={parts['t2']:.4f} T3={parts['t3']:.4f}",flush=True)
print("frozen baseline 0.67882",flush=True)
