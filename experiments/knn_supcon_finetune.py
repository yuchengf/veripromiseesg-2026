"""SupCon (supervised contrastive) LoRA fine-tune of Qwen3-Embedding-0.6B
for the T3 kNN component.

Protocol (anti-overfit):
  Phase A (Layer 1): 5-fold OOF — fine-tune on 4 folds, embed held-out fold,
    kNN (k=5, softmax sim weights) T3 macro-F1 vs same-structure baseline
    using the original (un-finetuned) cached embeddings.
  Phase B (Layer 2): only if Phase A improves — fine-tune on full train 1601,
    embed train+valid, run the FROZEN 12-way fusion (a3=0.4, nc_bias=0.1,
    k=5, temp=1) once on valid 399 vs baseline 0.67882.

Outputs:
  agent_cache/qwen3_supcon_embs_final.npz  (tr, va) if Phase B runs
  runs/knn_supcon/adapter/                  final LoRA adapter
  stdout log: per-fold OOF F1, fused valid score.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

os.chdir("/home/yucheng/Desktop/ESG")
torch.manual_seed(42)
np.random.seed(42)

BACKBONE = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCTION = "Retrieve similar ESG disclosure texts that share the same evidence quality level."
MAX_LEN = 384
BATCH = 8
EPOCHS = 3
LR = 5e-5
TAU = 0.07
K = 5

LABELS = {
    "t1": ["Yes", "No"],
    "t2": ["Yes", "No", "N/A"],
    "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "t4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
}
TASK_W = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}
SOL_COL = {"t1": "promise_status", "t2": "evidence_status",
           "t3": "evidence_quality", "t4": "verification_timeline"}

train = pd.read_csv("final_data/train_data.csv", keep_default_na=False)
sol = pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
for df in (train, sol):
    df["verification_timeline"] = df["verification_timeline"].replace(
        "longer_than_5_years", "more_than_5_years")
texts = [INSTRUCTION + t for t in train["data"].astype(str)]
t3_lab = np.array([LABELS["t3"].index(l) for l in train["evidence_quality"].astype(str)])
print(f"[data] train={len(texts)} T3 dist={np.bincount(t3_lab, minlength=4).tolist()} "
      f"(order {LABELS['t3']})", flush=True)


def build_model():
    tok = AutoTokenizer.from_pretrained(BACKBONE, padding_side="right")
    base = AutoModel.from_pretrained(BACKBONE, dtype=torch.float32,
                                     attn_implementation="sdpa").cuda()
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base, cfg)
    return tok, model


def embed_batch(model, tok, batch_texts):
    enc = tok(batch_texts, padding=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt").to("cuda")
    out = model(**enc).last_hidden_state
    idx = enc["attention_mask"].sum(1) - 1
    z = out[torch.arange(out.size(0)), idx]
    return F.normalize(z.float(), dim=-1)


@torch.no_grad()
def embed_all(model, tok, all_texts, bs=16):
    model.eval()
    chunks = []
    for i in range(0, len(all_texts), bs):
        chunks.append(embed_batch(model, tok, all_texts[i:i + bs]).cpu())
    return torch.cat(chunks).numpy()


def supcon_loss(z, labels):
    sim = z @ z.T / TAU
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    has_pos = pos.any(1)
    if not has_pos.any():
        return None
    sim = sim.masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # zero out non-positive entries (incl. the -inf diagonal) BEFORE summing,
    # so the masked -inf never hits a 0 multiply (-inf * 0 = NaN).
    logp = logp.masked_fill(~pos, 0.0)
    loss = -logp.sum(1)[has_pos] / pos.sum(1)[has_pos]
    return loss.mean()


def finetune(idx_train):
    tok, model = build_model()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps = (len(idx_train) // BATCH) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(steps, 1),
                                                pct_start=0.1)
    model.train()
    for ep in range(EPOCHS):
        perm = np.random.permutation(idx_train)
        losses = []
        for i in range(0, len(perm) - BATCH + 1, BATCH):
            b = perm[i:i + BATCH]
            z = embed_batch(model, tok, [texts[j] for j in b])
            loss = supcon_loss(z, torch.tensor(t3_lab[b], device="cuda"))
            if loss is None or not torch.isfinite(loss):
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item())
        print(f"  ep{ep + 1} supcon={np.mean(losses):.4f}", flush=True)
    return tok, model


def knn_t3_preds(q_emb, k_emb, k_lab):
    sim = q_emb @ k_emb.T
    preds = np.zeros(len(q_emb), dtype=int)
    for i in range(len(q_emb)):
        top = np.argsort(sim[i])[::-1][:K]
        s = sim[i][top]
        w = np.exp(s - s.max()); w /= w.sum()
        p = np.zeros(4)
        np.add.at(p, k_lab[top], w)
        preds[i] = p.argmax()
    return preds


def oof_f1(emb_by_fold):
    """emb_by_fold: list of (heldout_idx, q_emb, train_idx, k_emb)."""
    yt, yp = [], []
    for ho, q, tr_idx, k in emb_by_fold:
        preds = knn_t3_preds(q, k, t3_lab[tr_idx])
        yt.extend(t3_lab[ho]); yp.extend(preds)
    return f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0)


# ── Phase A: 5-fold OOF ──
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(texts)), t3_lab))

base_emb = np.load("agent_cache/qwen3_embs_final.npz")["tr"]
base_emb = base_emb / (np.linalg.norm(base_emb, axis=1, keepdims=True) + 1e-8)
base_folds = [(ho, base_emb[ho], tr_idx, base_emb[tr_idx]) for tr_idx, ho in folds]
base_oof = oof_f1(base_folds)
print(f"[Phase A] baseline OOF kNN T3 macro-F1 = {base_oof:.4f}", flush=True)

ft_folds = []
for fi, (tr_idx, ho) in enumerate(folds, 1):
    print(f"[Phase A] fold {fi}/5 finetune on {len(tr_idx)}", flush=True)
    tok, model = finetune(tr_idx)
    q = embed_all(model, tok, [texts[j] for j in ho])
    k = embed_all(model, tok, [texts[j] for j in tr_idx])
    ft_folds.append((ho, q, tr_idx, k))
    del model; torch.cuda.empty_cache()
ft_oof = oof_f1(ft_folds)
print(f"[Phase A] finetuned OOF kNN T3 macro-F1 = {ft_oof:.4f} "
      f"(delta {ft_oof - base_oof:+.4f})", flush=True)

if ft_oof <= base_oof:
    print("[Phase A] no OOF gain -> STOP (reject contrastive finetune)", flush=True)
    raise SystemExit(0)

# ── Phase B: full-train finetune + frozen fusion on valid 399 ──
print("[Phase B] finetune on full train 1601", flush=True)
tok, model = finetune(np.arange(len(texts)))
val_texts = [INSTRUCTION + t for t in
             pd.read_csv("final_data/valid_data.csv", keep_default_na=False)["data"].astype(str)]
tr_new = embed_all(model, tok, texts)
va_new = embed_all(model, tok, val_texts)
np.savez_compressed("agent_cache/qwen3_supcon_embs_final.npz", tr=tr_new, va=va_new)
Path("runs/knn_supcon").mkdir(parents=True, exist_ok=True)
model.save_pretrained("runs/knn_supcon/adapter")
del model; torch.cuda.empty_cache()

# frozen 12-way fusion (same as knn_sweep current config)
y = {t: sol[SOL_COL[t]].astype(str).tolist() for t in LABELS}
N = len(sol)
PRE = [f"{g}_roberta_dc_kfold_t3nc3" for g in ["FC1", "FC1R", "FC1S", "FC1M"]]
CACHE = Path("agent_cache/valid_probs")
base = {t: np.mean([np.load(CACHE / f"{p}{s}_f{f}.npz")[t]
                    for p in PRE for s in ["", "_s1", "_s2"] for f in range(1, 6)],
                   axis=0) for t in LABELS}
lab_idx = {t: np.array([LABELS[t].index(l) for l in train[SOL_COL[t]].astype(str)])
           for t in LABELS}


def fused_score(tr_e, va_e):
    sim = va_e @ tr_e.T
    knn3 = np.zeros((N, 4))
    for i in range(N):
        top = np.argsort(sim[i])[::-1][:K]
        s = sim[i][top]
        w = np.exp(s - s.max()); w /= w.sum()
        np.add.at(knn3[i], lab_idx["t3"][top], w)
        knn3[i] /= knn3[i].sum()
    t3 = base["t3"].copy()
    t3[:, 1] *= np.exp(0.1)
    t3 /= t3.sum(1, keepdims=True)
    fused = dict(base)
    fused["t3"] = 0.6 * t3 + 0.4 * knn3
    preds = {t: [LABELS[t][i] for i in np.argmax(fused[t], axis=1)] for t in LABELS}
    for i in range(N):
        if preds["t1"][i] == "No":
            preds["t2"][i] = preds["t3"][i] = preds["t4"][i] = "N/A"
        elif preds["t2"][i] == "No":
            preds["t3"][i] = "N/A"
    tot = 0.0
    for t in LABELS:
        tot += TASK_W[t] * f1_score(y[t], preds[t], labels=sorted(set(y[t])),
                                    average="macro", zero_division=0)
    return tot


d = np.load("agent_cache/qwen3_embs_final.npz")
tr0 = d["tr"] / (np.linalg.norm(d["tr"], axis=1, keepdims=True) + 1e-8)
va0 = d["va"] / (np.linalg.norm(d["va"], axis=1, keepdims=True) + 1e-8)
s_base = fused_score(tr0, va0)
s_new = fused_score(tr_new, va_new)
print(f"[Phase B] valid 399 fused 12-way+kNN: baseline={s_base:.5f} "
      f"supcon={s_new:.5f} (delta {s_new - s_base:+.5f}, frozen ref 0.67882)", flush=True)
