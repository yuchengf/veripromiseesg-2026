"""P2/P3: standalone 3-way clarity head (Clear/Not Clear/Misleading) trained ONLY
on T3-non-N/A rows (where evidence exists), with balanced-softmax loss. Then
overlay onto the frozen 12-way: where the 12-way predicts a non-N/A T3, replace
with this head's prediction. Measures Not Clear recall + total on valid 399.

Hypothesis (from P1): Not Clear is representationally confusable, not a calibration
issue. A clarity head not diluted by 525 N/A rows may separate Clear vs Not Clear
better. Balanced softmax (Ren 2020) = train with CE(logits + log prior).

Usage: python clarity_head.py [--smoke]
"""
import os, sys
os.chdir("/home/yucheng/Desktop/ESG")
SMOKE = "--smoke" in sys.argv
AUG = "--aug" in sys.argv
LS = 0.1 if "--ls" in sys.argv else 0.0
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
SEED = int(sys.argv[sys.argv.index("--seed")+1]) if "--seed" in sys.argv else 42
ENS = "--ens" in sys.argv   # ensemble mode: save seed-suffixed probs, skip overlay eval
torch.manual_seed(SEED); np.random.seed(SEED)

BACKBONE = "hfl/chinese-roberta-wwm-ext-large"
MAXLEN, BS, EPOCHS, LR = 384, 8, (1 if SMOKE else 5), 1e-5
TWOWAY = "--twoway" in sys.argv
CLS = ["Clear", "Not Clear"] if TWOWAY else ["Clear", "Not Clear", "Misleading"]
K = len(CLS)
LAB = {"t1":["Yes","No"],"t2":["Yes","No","N/A"],
       "t3":["Clear","Not Clear","Misleading","N/A"],
       "t4":["already","within_2_years","between_2_and_5_years","more_than_5_years","N/A"]}
TASK_W={"t1":0.20,"t2":0.30,"t3":0.35,"t4":0.15}
SOL={"t1":"promise_status","t2":"evidence_status","t3":"evidence_quality","t4":"verification_timeline"}

tr = pd.read_csv("final_data/train_data.csv", keep_default_na=False)
vdf = pd.read_csv("final_data/valid_data.csv", keep_default_na=False)
sol = pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
sol["verification_timeline"]=sol["verification_timeline"].replace("longer_than_5_years","more_than_5_years")
# training rows: T3 in {Clear,NotClear,Misleading} (non-N/A)
cl = tr[tr["evidence_quality"].isin(CLS)].reset_index(drop=True)
if AUG:
    augf = "external_data/clarity_aug_zh.csv"
    aug = pd.read_csv(augf, keep_default_na=False)
    aug = aug[aug["evidence_quality"].isin(CLS)]
    cl = pd.concat([cl[["data","evidence_quality"]], aug[["data","evidence_quality"]]],
                   ignore_index=True)
    print(f"[clarity] +AUG {len(aug)} translated rows -> {len(cl)} total", flush=True)
if SMOKE: cl = cl.sample(120, random_state=0).reset_index(drop=True)
texts = cl["data"].astype(str).tolist()
ylab = np.array([CLS.index(x) for x in cl["evidence_quality"]])
prior = np.bincount(ylab, minlength=K).astype(float); prior/=prior.sum()
logprior = torch.tensor(np.log(prior+1e-9), dtype=torch.float32).cuda()
print(f"[clarity] train rows={len(cl)} dist={np.bincount(ylab,minlength=K).tolist()} (order {CLS}) SMOKE={SMOKE}", flush=True)
vtexts = vdf["data"].astype(str).tolist(); Nv=len(vtexts)

tok = AutoTokenizer.from_pretrained(BACKBONE)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(BACKBONE)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.enc.config.hidden_size, K)
    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:,0]
        return self.head(self.drop(h))

def enc_batch(bt):
    e = tok(bt, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
    return e["input_ids"].cuda(), e["attention_mask"].cuda()

@torch.no_grad()
def predict(model, btexts):
    model.eval(); out=[]
    for i in range(0,len(btexts),16):
        ids,m = enc_batch(btexts[i:i+16])
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lo = model(ids,m)
        out.append(F.softmax(lo.float(),dim=-1).cpu().numpy())
    return np.concatenate(out)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
val_probs = np.zeros((Nv,K)); oof_acc=[]
for fi,(tri,vi) in enumerate(skf.split(texts, ylab),1):
    model = Net().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = max(1, -(-len(tri)//BS))*EPOCHS   # ceil(len/BS) batches per epoch
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    for ep in range(EPOCHS):
        model.train(); perm=np.random.permutation(tri); losses=[]
        for j in range(0,len(perm),BS):
            bi=perm[j:j+BS]
            if len(bi)==0: continue
            ids,m=enc_batch([texts[k] for k in bi])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lo=model(ids,m)
            loss=F.cross_entropy(lo.float()+logprior, torch.tensor(ylab[bi]).cuda(), label_smoothing=LS)  # balanced softmax + label smoothing
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); sch.step(); opt.zero_grad(); losses.append(loss.item())
    # held-out clarity acc (sanity)
    hp = predict(model, [texts[k] for k in vi]).argmax(1)
    acc=(hp==ylab[vi]).mean(); oof_acc.append(acc)
    val_probs += predict(model, vtexts)
    print(f"  fold{fi} loss={np.mean(losses):.3f} heldout_clarity_acc={acc:.3f}", flush=True)
    del model; torch.cuda.empty_cache()
    if SMOKE: break
val_probs /= (1 if SMOKE else 5)
_probpath = (f"agent_cache/clarity_valid_probs_s{SEED}.npz" if ENS
             else "agent_cache/clarity_valid_probs_aug.npz" if AUG
             else "agent_cache/clarity_valid_probs_ls.npz" if LS
             else "agent_cache/clarity_valid_probs.npz")
np.savez_compressed(_probpath, probs=val_probs, classes=np.array(CLS))
print(f"[clarity] mean heldout clarity acc={np.mean(oof_acc):.3f}", flush=True)
if ENS:
    print(f"[ENS] saved seed {SEED} probs -> {_probpath}"); sys.exit(0)
if SMOKE:
    print("[SMOKE] OK — pipeline runs. Re-run without --smoke for full 5-fold + overlay."); sys.exit(0)

# ---------- overlay eval on valid 399 ----------
G=["FC1","FC1R","FC1S","FC1M"]; S=["","_s1","_s2"]; C=Path("agent_cache/valid_probs")
y={t:sol[SOL[t]].astype(str).tolist() for t in LAB}; usage=sol["Usage"].values
base={t:np.mean([np.load(C/f"{g}_roberta_dc_kfold_t3nc3{s}_f{f}.npz")[t]
        for g in G for s in S for f in range(1,6)],axis=0) for t in LAB}
d=np.load("agent_cache/qwen3_embs_final.npz")
a=d["tr"]/(np.linalg.norm(d["tr"],axis=1,keepdims=True)+1e-8)
b=d["va"]/(np.linalg.norm(d["va"],axis=1,keepdims=True)+1e-8); sim=b@a.T
t3l=np.array([LAB["t3"].index(l) for l in tr["evidence_quality"].astype(str)])
knn=np.zeros((Nv,4))
for i in range(Nv):
    tp=np.argsort(sim[i])[::-1][:5]; w=np.exp(sim[i][tp]-sim[i][tp].max()); w/=w.sum()
    np.add.at(knn[i],t3l[tp],w); knn[i]/=knn[i].sum()
t3=base["t3"].copy(); t3[:,1]*=np.exp(0.1); t3/=t3.sum(1,keepdims=True)
fused=dict(base); fused["t3"]=0.6*t3+0.4*knn
basepred={t:[LAB[t][i] for i in np.argmax(fused[t],axis=1)] for t in LAB}

def cascade_score(t3pred):
    pred={t:list(basepred[t]) for t in LAB}; pred["t3"]=list(t3pred)
    for i in range(Nv):
        if pred["t1"][i]=="No": pred["t2"][i]=pred["t3"][i]=pred["t4"][i]="N/A"
        elif pred["t2"][i]=="No": pred["t3"][i]="N/A"
    tot=0.0
    for t in LAB:
        yt=y[t]; pt=pred[t]
        tot+=TASK_W[t]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
    truth=np.array(y["t3"]); pr=np.array(pred["t3"]); mnc=truth=="Not Clear"
    ncrec=int(((pr=="Not Clear")&mnc).sum())
    pub=0.0;priv=0.0
    for tag,msk in [("pub",usage=="Public"),("priv",usage=="Private")]:
        idx=np.where(msk)[0]; s=0.0
        for t in LAB:
            yt=[y[t][i] for i in idx]; pt=[pred[t][i] for i in idx]
            s+=TASK_W[t]*f1_score(yt,pt,labels=sorted(set(yt)),average="macro",zero_division=0)
        if tag=="pub":pub=s
        else:priv=s
    return tot,pub,priv,ncrec

base_t3 = list(basepred["t3"])
btot,bpub,bpriv,bnc = cascade_score(base_t3)
print(f"\n[12-way base] total={btot:.5f} Pub={bpub:.5f} Priv={bpriv:.5f} NotClear={bnc}/45")
print("\n=== overlay clarity head where 12-way T3 is non-N/A ===")
print(f"{'mode':>22} {'total':>9} {'Public':>9} {'Private':>9} {'NotClear':>9}")
ch_arg = val_probs.argmax(1)  # 0=Clear 1=NotClear 2=Misleading
for thr in [0.0,0.5,0.6,0.7,0.8]:
    t3o=list(base_t3)
    for i in range(Nv):
        if base_t3[i] in CLS and val_probs[i].max()>=thr:
            t3o[i]=CLS[ch_arg[i]]
    tot,pub,priv,nc=cascade_score(t3o)
    tag=f"replace(conf>={thr})" if thr>0 else "replace(all non-N/A)"
    print(f"{tag:>22} {tot:9.5f} {pub:9.5f} {priv:9.5f} {nc:>6}/45")
print(f"\nfrozen baseline 0.67882. Clarity head valid P(NotClear) fired on "
      f"{int((ch_arg==1).sum())}/{Nv} rows.")
