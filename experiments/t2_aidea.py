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
tr=pd.read_csv("final_data/full_train_data.csv",keep_default_na=False)
vdf=pd.read_csv("final_data/vpesg4k_test_2000.csv",keep_default_na=False)
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
np.savez_compressed("agent_cache/t2_test_probs.npz",probs=val_probs,classes=np.array(CLS))
print(f"[t2head] mean heldout acc={np.mean(oof):.3f}",flush=True)
if SMOKE: print("[SMOKE] OK"); sys.exit(0)
print("[t2_aidea] saved test probs", flush=True)
