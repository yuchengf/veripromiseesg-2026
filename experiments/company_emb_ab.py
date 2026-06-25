"""A/B test: simple multi-task RoBERTa ± learned company embedding, same recipe,
final_data 1601 -> valid 399. If +company-emb raises valid (esp T4), learned
company×text interaction is worth a full 12-way retrain. (50 companies all in
train+test/stratified -> learning company-specific patterns transfers.)
Usage: python company_emb_ab.py [--company] [--smoke]
"""
import os, sys
os.chdir("/home/yucheng/Desktop/ESG")
USE_COMP="--company" in sys.argv; SMOKE="--smoke" in sys.argv
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
torch.manual_seed(42); np.random.seed(42)
BACKBONE="hfl/chinese-roberta-wwm-ext-large"; MAXLEN,BS,EPOCHS,LR=384,8,(1 if SMOKE else 4),1e-5
LAB={"t1":["Yes","No"],"t2":["Yes","No","N/A"],"t3":["Clear","Not Clear","Misleading","N/A"],
     "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TW={"t1":.2,"t2":.3,"t3":.35,"t4":.15}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}
tr=pd.read_csv("final_data/train_data.csv",keep_default_na=False)
vdf=pd.read_csv("final_data/valid_data.csv",keep_default_na=False)
sol=pd.read_csv("final_data/valid_solution_data.csv",keep_default_na=False)
for df in (tr,sol): df["verification_timeline"]=df["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
comps=sorted(tr["company"].str.strip().str.lower().unique()); cid={c:i for i,c in enumerate(comps)}; NC=len(comps)
def cids(df): return df["company"].str.strip().str.lower().map(lambda c:cid.get(c,0)).values
tr_c=cids(tr); va_c=cids(vdf)
texts=tr["data"].astype(str).tolist(); vtexts=vdf["data"].astype(str).tolist(); Nv=len(vtexts)
ylab={t:np.array([LAB[t].index(x) for x in tr[SOL[t]].astype(str)]) for t in LAB}
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB}; usage=sol["Usage"].values
if SMOKE:
    idx=np.random.RandomState(0).choice(len(texts),150,False)
    texts=[texts[i] for i in idx]; tr_c=tr_c[idx]; ylab={t:ylab[t][idx] for t in LAB}
tok=AutoTokenizer.from_pretrained(BACKBONE)
NCW=3.0
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=AutoModel.from_pretrained(BACKBONE); h=self.enc.config.hidden_size
        self.drop=nn.Dropout(0.1)
        self.cemb=nn.Embedding(NC,64) if USE_COMP else None
        d=h+(64 if USE_COMP else 0)
        self.heads=nn.ModuleDict({t:nn.Linear(d,len(LAB[t])) for t in LAB})
    def forward(self,ids,m,comp):
        x=self.drop(self.enc(input_ids=ids,attention_mask=m).last_hidden_state[:,0])
        if self.cemb is not None: x=torch.cat([x,self.cemb(comp)],-1)
        return {t:self.heads[t](x) for t in LAB}
def enc_batch(bt):
    e=tok(bt,padding=True,truncation=True,max_length=MAXLEN,return_tensors="pt"); return e["input_ids"].cuda(),e["attention_mask"].cuda()
@torch.no_grad()
def predict(model,bt,cc):
    model.eval(); out={t:[] for t in LAB}
    for i in range(0,len(bt),16):
        ids,m=enc_batch(bt[i:i+16]); cp=torch.tensor(cc[i:i+16]).cuda()
        with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(ids,m,cp)
        for t in LAB: out[t].append(F.softmax(lo[t].float(),-1).cpu().numpy())
    return {t:np.concatenate(out[t]) for t in LAB}
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); vp={t:np.zeros((Nv,len(LAB[t]))) for t in LAB}
for fi,(tri,vi) in enumerate(skf.split(texts,ylab["t1"]),1):
    model=Net().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    steps=max(1,-(-len(tri)//BS))*EPOCHS; sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.1)
    for ep in range(EPOCHS):
        model.train(); perm=np.random.permutation(tri)
        for j in range(0,len(perm),BS):
            bi=perm[j:j+BS]
            if len(bi)==0: continue
            ids,m=enc_batch([texts[k] for k in bi]); cp=torch.tensor(tr_c[bi]).cuda()
            with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(ids,m,cp)
            loss=0
            for t in LAB:
                w=torch.ones(len(LAB[t])).cuda()
                if t=="t3": w[1]=NCW
                loss=loss+TW[t]*F.cross_entropy(lo[t].float(),torch.tensor(ylab[t][bi]).cuda(),weight=w)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); opt.zero_grad()
    pv=predict(model,vtexts,va_c)
    for t in LAB: vp[t]+=pv[t]
    print(f"  fold{fi} done",flush=True); del model; torch.cuda.empty_cache()
    if SMOKE: break
for t in LAB: vp[t]/=(1 if SMOKE else 5)
tag="withCOMP" if USE_COMP else "noCOMP"
np.savez_compressed(f"agent_cache/cemb_ab_{tag}.npz",**vp)
if SMOKE: print("[SMOKE] OK"); sys.exit(0)
pred={t:[LAB[t][i] for i in vp[t].argmax(1)] for t in LAB}
for i in range(Nv):
    if pred["t1"][i]=="No":pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
    elif pred["t2"][i]=="No":pred["t3"][i]="N/A"
tot=0;parts={}
for t in LAB:
    parts[t]=f1_score(y[t],pred[t],labels=sorted(set(y[t])),average="macro",zero_division=0);tot+=TW[t]*parts[t]
print(f"[{tag}] valid total={tot:.5f}  per-task={ {t:round(parts[t],4) for t in LAB} }",flush=True)
