"""Build the AIdea clarity-overlay submission probe.
Train the non-aug 3-way clarity head (balanced softmax) on retrain_data
(full_train_data 2000, non-N/A rows) -> predict the 2000 AIdea test -> overlay
onto aidea_rt_all12_knn.csv where the 12-way T3 is non-N/A and clarity conf>=thr.
Outputs candidate submission(s) + a report of what changed (esp. Misleading
injections, which are high-risk: only 2 training Misleading).
"""
import os, sys
os.chdir("/home/yucheng/Desktop/ESG")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
torch.manual_seed(42); np.random.seed(42)

BACKBONE="hfl/chinese-roberta-wwm-ext-large"; MAXLEN,BS,EPOCHS,LR=384,8,5,1e-5
CLS=["Clear","Not Clear","Misleading"]
tr=pd.read_csv("final_data/full_train_data.csv",keep_default_na=False)
test=pd.read_csv("final_data/vpesg4k_test_2000.csv",keep_default_na=False)
base=pd.read_csv("official_sub/aidea_rt_all12_knn.csv",keep_default_na=False)
cl=tr[tr["evidence_quality"].isin(CLS)].reset_index(drop=True)
texts=cl["data"].astype(str).tolist(); ylab=np.array([CLS.index(x) for x in cl["evidence_quality"]])
prior=np.bincount(ylab,minlength=3).astype(float); prior/=prior.sum()
logprior=torch.tensor(np.log(prior+1e-9),dtype=torch.float32).cuda()
print(f"[clarity-aidea] train(non-N/A)={len(cl)} dist={np.bincount(ylab,minlength=3).tolist()} {CLS}",flush=True)
ttexts=test["data"].astype(str).tolist(); Nt=len(ttexts)
tok=AutoTokenizer.from_pretrained(BACKBONE)

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.enc=AutoModel.from_pretrained(BACKBONE)
        self.drop=nn.Dropout(0.1); self.head=nn.Linear(self.enc.config.hidden_size,3)
    def forward(self,ids,m):
        return self.head(self.drop(self.enc(input_ids=ids,attention_mask=m).last_hidden_state[:,0]))

def enc_batch(bt):
    e=tok(bt,padding=True,truncation=True,max_length=MAXLEN,return_tensors="pt")
    return e["input_ids"].cuda(),e["attention_mask"].cuda()

@torch.no_grad()
def predict(model,bt):
    model.eval(); out=[]
    for i in range(0,len(bt),16):
        ids,m=enc_batch(bt[i:i+16])
        with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(ids,m)
        out.append(F.softmax(lo.float(),-1).cpu().numpy())
    return np.concatenate(out)

skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
test_probs=np.zeros((Nt,3))
for fi,(tri,vi) in enumerate(skf.split(texts,ylab),1):
    model=Net().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=0.01)
    steps=max(1,-(-len(tri)//BS))*EPOCHS
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=LR,total_steps=steps,pct_start=0.1)
    for ep in range(EPOCHS):
        model.train(); perm=np.random.permutation(tri)
        for j in range(0,len(perm),BS):
            bi=perm[j:j+BS]
            if len(bi)==0: continue
            ids,m=enc_batch([texts[k] for k in bi])
            with torch.autocast("cuda",dtype=torch.bfloat16): lo=model(ids,m)
            loss=F.cross_entropy(lo.float()+logprior,torch.tensor(ylab[bi]).cuda())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); opt.zero_grad()
    test_probs+=predict(model,ttexts)
    print(f"  fold{fi} done",flush=True); del model; torch.cuda.empty_cache()
test_probs/=5
np.savez_compressed("agent_cache/clarity_test_probs.npz",probs=test_probs,classes=np.array(CLS))
ca=test_probs.argmax(1); conf=test_probs.max(1)

# overlay onto 12-way AIdea file
assert (base["id"].tolist()==test["id"].tolist()), "id mismatch"
b3=base["evidence_quality"].tolist()
def build(thr, allow_misleading):
    out=list(b3); ch=0; mis=0
    for i in range(Nt):
        if b3[i] in CLS and conf[i]>=thr:
            newc=CLS[ca[i]]
            if newc=="Misleading" and not allow_misleading: continue
            if newc!=b3[i]:
                out[i]=newc; ch+=1
                if newc=="Misleading": mis+=1
    return out,ch,mis

print(f"\n[clarity test preds] argmax dist={np.bincount(ca,minlength=3).tolist()} {CLS}",flush=True)
print(f"  P(Misleading)>0.7 on test: {int(((ca==2)&(conf>=0.7)).sum())} rows; >0.5: {int((test_probs[:,2]>=0.5).sum())}",flush=True)
for thr in [0.7,0.8]:
    for allowm in [False,True]:
        o,ch,mis=build(thr,allowm)
        from collections import Counter
        tag=f"thr{thr}_{'withMis' if allowm else 'noMis'}"
        fn=f"official_sub/aidea_clarity_{tag}.csv"
        sub=base.copy(); sub["evidence_quality"]=o; sub.to_csv(fn,index=False)
        print(f"  {tag}: changed {ch} rows (Misleading injected={mis}) T3={dict(Counter(o))} -> {fn}",flush=True)
print("\nbase 12-way T3:",dict(pd.Series(b3).value_counts()),flush=True)
print("PICK: thr0.7_noMis = safest(只改 Clear<->NotClear); withMis = 賭 Misleading(高風險,2 訓練例)",flush=True)
