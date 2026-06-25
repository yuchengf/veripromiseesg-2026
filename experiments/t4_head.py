"""T4-specific timeline head with SOFT-ORDINAL label encoding (research-backed
for ordinal + rare class). Train a 4-way head (already<within_2y<2-5y<>5y) ONLY
on T1=Yes rows, soft-ordinal targets soft[i]=exp(-|i-k|/tau) + mild class weights,
overlay onto 12-way T4 where T1=Yes. Validatable (all T4 classes in valid).
Usage: python t4_head.py [--smoke]
"""
import os, sys
os.chdir("/home/yucheng/Desktop/ESG")
SMOKE="--smoke" in sys.argv
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
torch.manual_seed(42); np.random.seed(42)
BACKBONE="hfl/chinese-roberta-wwm-ext-large"; MAXLEN,BS,EPOCHS,LR,TAU=384,8,(1 if SMOKE else 5),1e-5,1.0
CLS=["already","within_2_years","between_2_and_5_years","more_than_5_years"]; K=4  # ordinal order
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],
     "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
vdf=pd.read_csv("final_data/valid_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
for df in (tr,sol): df["verification_timeline"]=df["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
cl=tr[(tr["promise_status"]=="Yes") & (tr["verification_timeline"].isin(CLS))].reset_index(drop=True)
if SMOKE: cl=cl.sample(120,random_state=0).reset_index(drop=True)
texts=cl["data"].astype(str).tolist(); ylab=np.array([CLS.index(x) for x in cl["verification_timeline"]])
cnt=np.bincount(ylab,minlength=K).astype(float)
cw=np.power(cnt.sum()/(cnt+1e-9),0.5); cw=cw/cw.mean()   # mild inverse-freq class weights
cw_t=torch.tensor(cw,dtype=torch.float32).cuda()
# soft-ordinal target matrix: soft[k] = softmax over i of -|i-k|/tau
idx=np.arange(K); SOFT=np.stack([np.exp(-np.abs(idx-k)/TAU) for k in range(K)]); SOFT/=SOFT.sum(1,keepdims=True)
SOFT_t=torch.tensor(SOFT,dtype=torch.float32).cuda()
print(f"[t4head] train(T1=Yes)={len(cl)} dist={cnt.astype(int).tolist()} {CLS} cw={np.round(cw,2).tolist()} SMOKE={SMOKE}",flush=True)
vtexts=vdf["data"].astype(str).tolist(); Nv=len(vtexts)
tok=AutoTokenizer.from_pretrained(BACKBONE)
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=AutoModel.from_pretrained(BACKBONE); self.drop=nn.Dropout(0.1); self.head=nn.Linear(self.enc.config.hidden_size,K)
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
def soft_ord_loss(logits,targets):
    logp=F.log_softmax(logits,dim=-1); soft=SOFT_t[targets]; w=cw_t[targets]
    return -((soft*logp).sum(1)*w).mean()
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
            loss=soft_ord_loss(lo.float(),torch.tensor(ylab[bi]).cuda())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad()
    hp=predict(model,[texts[k] for k in vi]).argmax(1); oof.append((hp==ylab[vi]).mean())
    val_probs+=predict(model,vtexts); print(f"  fold{fi} heldout_acc={oof[-1]:.3f}",flush=True)
    del model; torch.cuda.empty_cache()
    if SMOKE: break
val_probs/=(1 if SMOKE else 5)
np.savez_compressed("agent_cache/t4_valid_probs.npz",probs=val_probs,classes=np.array(CLS))
print(f"[t4head] mean heldout acc={np.mean(oof):.3f}",flush=True)
if SMOKE: print("[SMOKE] OK"); sys.exit(0)
# overlay
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
def score(t4pred):
    pred={t:list(basep[t]) for t in LAB}; pred["t4"]=list(t4pred)
    for i in range(Nv):
        if pred["t1"][i]=="No": pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No": pred["t3"][i]="N/A"
    tot=0.0;parts={}
    for t in LAB:
        f=f1_score(y[t],pred[t],labels=sorted(set(y[t])),average="macro",zero_division=0);tot+=TW[t]*f;parts[t]=f
    pub=sum(TW[t]*f1_score([y[t][i] for i in np.where(usage=='Public')[0]],[pred[t][i] for i in np.where(usage=='Public')[0]],labels=sorted(set(y[t])),average="macro",zero_division=0) for t in LAB)
    w2=int(((np.array(pred["t4"])=="within_2_years")&(np.array(y["t4"])=="within_2_years")).sum())
    return tot,parts,pub,w2
bt,bp,bpub,bw2=score(basep["t4"])
print(f"\n[12-way base] total={bt:.5f} Pub={bpub:.5f} T4={bp['t4']:.4f} within2y_recall={bw2}/5",flush=True)
print("=== T4 soft-ordinal head overlay (T1=Yes & base T4 non-N/A & conf>=thr) ===")
for thr in [0.5,0.6,0.7,0.8]:
    t4o=list(basep["t4"]);ch=0
    for i in range(Nv):
        if basep["t1"][i]=="Yes" and basep["t4"][i] in CLS and conf[i]>=thr and CLS[ca[i]]!=basep["t4"][i]:
            t4o[i]=CLS[ca[i]];ch+=1
    tot,parts,pub,w2=score(t4o)
    print(f"  conf>={thr}: changed {ch:3d} total={tot:.5f} Pub={pub:.5f} T4={parts['t4']:.4f} within2y={w2}/5",flush=True)
print("frozen baseline 0.67882",flush=True)
