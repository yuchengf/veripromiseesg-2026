#!/usr/bin/env python3
"""
LLM Test-Time Augmentation (BA-TTA style).

For each test sample, generate N paraphrases via Qwen3-8B, run all (original + paraphrases)
through the BERT kfold pipeline, average softmax probs. This provides semantic diversity
at inference time without retraining.

Also generates LLM zero-shot predictions for hybrid ensemble (LLM + BERT).

Usage:
    python llm_tta.py --n_aug 4
    python llm_tta.py --n_aug 4 --mode hybrid  # also generate LLM zero-shot preds
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from esg_main import (
    LABELS, LABEL2IDX, IDX2LABEL, NUM_LABELS, TASK_WEIGHTS,
    Config, load_dataframes, ApproachA1, ESGDataset, predict_probs,
    compute_knn_ldl_probs, knn_fuse_probs,
)
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Paraphrase prompt (preserve meaning, change surface form)
# ─────────────────────────────────────────────────────────────────────────────
PARA_PROMPT = """\
你是一位ESG報告改寫專家。請將以下ESG報告段落用不同的詞彙和句式改寫，但必須嚴格保留原文的所有事實內容、語氣和含義。

規則：
1. 保留所有數字、百分比、年份、公司行動、承諾內容
2. 改變詞彙選擇和句子結構
3. 保持繁體中文
4. 長度與原文相近
5. 不要添加或刪除任何資訊

原文：
{text}

改寫："""

# ─────────────────────────────────────────────────────────────────────────────
# LLM zero-shot classification prompt
# ─────────────────────────────────────────────────────────────────────────────
CLASSIFY_PROMPT = """\
你是ESG永續報告分析專家。請分析以下段落並回答四個問題。

判斷標準：
1. promise_status: 該段落是否包含明確的企業永續承諾？(Yes/No)
2. evidence_status: 承諾是否有具體的行動計畫或佐證？(Yes/No/N/A=無承諾)
3. evidence_quality: 陳述的語意清晰程度？(Clear=清晰/Not Clear=模糊/Misleading=有誤導/N/A=無承諾)
4. verification_timeline: 預計完成時程？(already/within_2_years/between_2_and_5_years/longer_than_5_years/N/A=無承諾)

重要：若promise_status=No，則其餘三項必須為N/A。

段落：
{text}

請只輸出JSON格式：{{"promise_status": "...", "evidence_status": "...", "evidence_quality": "...", "verification_timeline": "..."}}"""


def llm_generate(model, tokenizer, device, prompt: str, max_tokens: int = 400,
                 temp: float = 0.7) -> str:
    msgs = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tokenizer(chat, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=mask, max_new_tokens=max_tokens,
            temperature=temp, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def generate_paraphrases(model, tokenizer, device, text: str, n: int = 4) -> list[str]:
    results = []
    for _ in range(n):
        prompt = PARA_PROMPT.format(text=text[:500])
        para = llm_generate(model, tokenizer, device, prompt, max_tokens=400, temp=0.8)
        if len(para) >= 30:
            results.append(para)
    return results


def classify_zero_shot(model, tokenizer, device, text: str) -> dict[str, str] | None:
    prompt = CLASSIFY_PROMPT.format(text=text[:500])
    output = llm_generate(model, tokenizer, device, prompt, max_tokens=200, temp=0.1)
    # Parse JSON
    try:
        match = re.search(r'\{[^}]+\}', output)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


def predict_test_df(test_df: pd.DataFrame, kfold_dirs: list[str],
                    device: torch.device) -> dict[str, list[list[float]]]:
    """Run kfold inference on a test dataframe, return averaged probs."""
    all_fold_probs = []
    for kdir in kfold_dirs:
        fold_idx = 1
        while True:
            ckpt_path = Path(kdir) / f"fold{fold_idx}" / "best.pt"
            if not ckpt_path.exists():
                break
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            saved_cfg = ckpt["cfg"]
            dc = getattr(saved_cfg, "deep_cascade", False)
            tok = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
            ds = ESGDataset(test_df, tok, saved_cfg.max_length, has_labels=False)
            loader = DataLoader(ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
            mdl = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
            mdl.load_state_dict(ckpt["model"])
            probs = predict_probs(mdl, loader, device)
            all_fold_probs.append(probs)
            del mdl
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fold_idx += 1

    # Average all fold probs
    n = len(test_df)
    avg = {}
    for t in LABELS:
        nc = NUM_LABELS[t]
        avg[t] = [
            [sum(fp[t][i][c] for fp in all_fold_probs) / len(all_fold_probs)
             for c in range(nc)]
            for i in range(n)
        ]
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="Qwen/Qwen3-14B")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load LLM in 4-bit (for 32B+ models on 32GB VRAM)")
    parser.add_argument("--n_aug", type=int, default=4, help="Paraphrases per test sample")
    parser.add_argument("--mode", choices=["tta", "hybrid", "both"], default="both")
    parser.add_argument("--kfold_dirs", nargs="+", default=[
        "runs/A1_roberta_dc_kfold_t3nc3",
        "runs/A1_roberta_dc_kfold_t3nc3_s1",
        "runs/A1_roberta_dc_kfold_t3nc3_s2",
    ])
    parser.add_argument("--knn_encoder_dir", default="runs/A1_roberta_dc_kfold_t3nc3")
    parser.add_argument("--weights", nargs="+", type=float, default=[1.0, 2.0, 1.0])
    parser.add_argument("--t3_alpha", type=float, default=0.42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    train_df, test_df = load_dataframes(
        cfg.data_dir, use_augmented=True, aug_filename="train_data_augmented.csv")
    n_test = len(test_df)
    print(f"Test samples: {n_test}")

    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)

    # ── Load LLM ──────────────────────────────────────────────────────────────
    print(f"Loading {args.backbone} (4bit={args.load_in_4bit}) ...")
    llm_tok = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if llm_tok.pad_token is None:
        llm_tok.pad_token = llm_tok.eos_token
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        llm = AutoModelForCausalLM.from_pretrained(
            args.backbone, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.backbone, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(device)
    llm.eval()
    print(f"LLM loaded ({sum(p.numel() for p in llm.parameters())/1e9:.1f}B params)")

    # ── Generate paraphrases ──────────────────────────────────────────────────
    if args.mode in ("tta", "both"):
        print(f"\n=== Generating {args.n_aug} paraphrases per test sample ===")
        all_paras: list[list[str]] = []
        for i, row in test_df.iterrows():
            paras = generate_paraphrases(llm, llm_tok, device, str(row["data"]), args.n_aug)
            all_paras.append(paras)
            print(f"  [{i+1}/{n_test}] id={row['id']} → {len(paras)} paraphrases")

        # Save paraphrases for reuse
        para_path = ROOT / "test_paraphrases.json"
        with open(para_path, "w") as f:
            json.dump(all_paras, f, ensure_ascii=False)
        print(f"Paraphrases saved to {para_path}")

    # ── LLM zero-shot classification ──────────────────────────────────────────
    if args.mode in ("hybrid", "both"):
        print(f"\n=== LLM zero-shot classification ===")
        llm_preds: list[dict] = []
        for i, row in test_df.iterrows():
            result = classify_zero_shot(llm, llm_tok, device, str(row["data"]))
            if result is None:
                result = {"promise_status": "Yes", "evidence_status": "Yes",
                          "evidence_quality": "Clear", "verification_timeline": "already"}
            llm_preds.append(result)
            print(f"  [{i+1}/{n_test}] id={row['id']} → T3={result.get('evidence_quality', '?')}")

        # Convert to soft probs (one-hot from LLM predictions)
        llm_probs: dict[str, list[list[float]]] = {t: [] for t in LABELS}
        task_col = {"t1": "promise_status", "t2": "evidence_status",
                    "t3": "evidence_quality", "t4": "verification_timeline"}
        for pred in llm_preds:
            for task, col in task_col.items():
                nc = NUM_LABELS[task]
                p = [0.0] * nc
                val = pred.get(col, "")
                idx = LABEL2IDX[task].get(val, -1)
                if idx >= 0:
                    p[idx] = 1.0
                else:
                    p = [1.0 / nc] * nc
                llm_probs[task].append(p)

        # Save LLM predictions
        llm_path = ROOT / "llm_zero_shot_probs.json"
        with open(llm_path, "w") as f:
            json.dump(llm_preds, f, ensure_ascii=False)
        print(f"LLM predictions saved to {llm_path}")

    # ── Free LLM VRAM ────────────────────────────────────────────────────────
    del llm
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("LLM freed from VRAM")

    # ── BERT pipeline: original test ──────────────────────────────────────────
    print(f"\n=== BERT ensemble on original test ===")
    orig_probs = predict_test_df(test_df, args.kfold_dirs, device)

    # Weight the seeds
    ws = args.weights
    total_w = sum(ws)
    # Actually predict_test_df already averages all folds across all dirs...
    # We need per-dir probs for weighted ensemble
    print(f"\n=== BERT per-seed predictions (weighted ensemble) ===")
    seed_probs_list = []
    for kdir in args.kfold_dirs:
        sp = predict_test_df(test_df, [kdir], device)
        seed_probs_list.append(sp)
        print(f"  {kdir}: done")

    # Weighted average
    weighted_probs: dict[str, list[list[float]]] = {}
    for t in LABELS:
        nc = NUM_LABELS[t]
        weighted_probs[t] = [
            [sum(ws[j] * seed_probs_list[j][t][i][c] for j in range(len(ws))) / total_w
             for c in range(nc)]
            for i in range(n_test)
        ]

    # ── kNN fusion ────────────────────────────────────────────────────────────
    print(f"\n=== kNN fusion ===")
    # Get train embeddings from knn_encoder
    knn_enc_dir = args.knn_encoder_dir
    # Compute kNN (same as gen_submissions)
    from esg_main import _predict_checkpoint
    # Need train/test embeddings — extract from first fold
    ckpt_path = Path(knn_enc_dir) / "fold1" / "best.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    saved_cfg = ckpt["cfg"]
    tok = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
    dc = getattr(saved_cfg, "deep_cascade", False)
    mdl = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
    mdl.load_state_dict(ckpt["model"])

    # Get embeddings
    from esg_main import ESGDataset as _DS
    # Train embeddings (real 800 only)
    train_orig = pd.read_csv(ROOT / "2026-esg-classification-challenge" / "train_data.csv")
    tr_ds = _DS(train_orig, tok, saved_cfg.max_length, has_labels=False)
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=False, num_workers=0)
    te_ds = _DS(test_df, tok, saved_cfg.max_length, has_labels=False)
    te_loader = DataLoader(te_ds, batch_size=32, shuffle=False, num_workers=0)

    mdl.eval()
    tr_embs, te_embs = [], []
    with torch.no_grad():
        for batch in tr_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            kwargs = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            out = mdl.encoder(**kwargs)
            tr_embs.append(out.last_hidden_state[:, 0].cpu().numpy())
        for batch in te_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            kwargs = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)
            out = mdl.encoder(**kwargs)
            te_embs.append(out.last_hidden_state[:, 0].cpu().numpy())
    tr_embs = np.concatenate(tr_embs, axis=0)
    te_embs = np.concatenate(te_embs, axis=0)
    del mdl; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    knn_probs = compute_knn_ldl_probs(tr_embs, te_embs, train_orig, k=10)
    alpha = {"t1": 0.0, "t2": 0.1, "t3": args.t3_alpha, "t4": 0.0}

    # ── Save: baseline weighted ensemble + kNN (s296 equivalent) ──────────────
    fused_base = knn_fuse_probs(weighted_probs, knn_probs, alpha=alpha)
    _save_submission(out_dir / "s303_v59_baseline_check.csv", fused_base, test_df)

    # ── TTA: average original + paraphrase probs ─────────────────────────────
    if args.mode in ("tta", "both"):
        print(f"\n=== BERT on paraphrased test samples ===")
        # For each test sample, create augmented test_df rows
        aug_test_rows = []
        aug_map = []  # (orig_idx, aug_idx) mapping
        for i in range(n_test):
            for para in all_paras[i]:
                row = test_df.iloc[i].copy()
                row["data"] = para
                aug_test_rows.append(row)
                aug_map.append(i)

        if aug_test_rows:
            aug_test_df = pd.DataFrame(aug_test_rows).reset_index(drop=True)
            print(f"  Augmented test size: {len(aug_test_df)}")

            # Run BERT per-seed on augmented
            aug_seed_probs = []
            for kdir in args.kfold_dirs:
                sp = predict_test_df(aug_test_df, [kdir], device)
                aug_seed_probs.append(sp)

            # Weighted average for augmented
            aug_weighted: dict[str, list[list[float]]] = {}
            n_aug_total = len(aug_test_df)
            for t in LABELS:
                nc = NUM_LABELS[t]
                aug_weighted[t] = [
                    [sum(ws[j] * aug_seed_probs[j][t][i][c] for j in range(len(ws))) / total_w
                     for c in range(nc)]
                    for i in range(n_aug_total)
                ]

            # Average: original + all paraphrases per test sample
            tta_probs: dict[str, list[list[float]]] = {t: [] for t in LABELS}
            for i in range(n_test):
                para_indices = [j for j, orig in enumerate(aug_map) if orig == i]
                n_versions = 1 + len(para_indices)  # original + paraphrases
                for t in LABELS:
                    nc = NUM_LABELS[t]
                    avg_p = list(weighted_probs[t][i])  # start with original
                    for j in para_indices:
                        for c in range(nc):
                            avg_p[c] += aug_weighted[t][j][c]
                    avg_p = [v / n_versions for v in avg_p]
                    tta_probs[t].append(avg_p)

            # kNN fusion on TTA probs
            fused_tta = knn_fuse_probs(tta_probs, knn_probs, alpha=alpha)
            _save_submission(out_dir / "s304_v59_tta.csv", fused_tta, test_df)
            print("Saved s304_v59_tta.csv")

    # ── Hybrid: BERT + LLM as extra ensemble member ──────────────────────────
    if args.mode in ("hybrid", "both"):
        print(f"\n=== Hybrid: BERT + LLM ensemble ===")
        # Mix: (1-β)*BERT + β*LLM
        for beta, sname in [(0.1, "s305"), (0.2, "s306"), (0.3, "s307")]:
            hybrid: dict[str, list[list[float]]] = {}
            for t in LABELS:
                nc = NUM_LABELS[t]
                hybrid[t] = [
                    [(1 - beta) * weighted_probs[t][i][c] + beta * llm_probs[t][i][c]
                     for c in range(nc)]
                    for i in range(n_test)
                ]
            fused_hybrid = knn_fuse_probs(hybrid, knn_probs, alpha=alpha)
            fname = f"{sname}_v60_hybrid_b{int(beta*100):02d}.csv"
            _save_submission(out_dir / fname, fused_hybrid, test_df)
            print(f"Saved {fname}")

    print("\nDone!")


def _save_submission(path: Path, preds: dict[str, list[str]], test_df: pd.DataFrame):
    rows = []
    n = len(test_df)
    for i in range(n):
        label = [preds[t][i] for t in ["t1", "t2", "t3", "t4"]]
        rows.append({"id": test_df.iloc[i]["id"], "label": str(label)})
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
