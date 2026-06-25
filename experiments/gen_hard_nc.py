#!/usr/bin/env python3
"""
LLM2LLM-style hard NC augmentation using local Qwen3-8B.

Strategy (LLM2LLM ACL 2024 + condition_yx ImbLLM 2025):
  1. Extract OOF predictions from nc3 5-fold model
  2. Find hard NC cases: true=NC AND (pred≠NC OR P(NC) < nc_thr)
  3. For each hard case generate 1 augmented NC sample (condition on NC feature type)
  4. Apply rule verification + character-bigram diversity filter
  5. Save to train_data_aug_hardnc.csv = base aug + new hard NC rows

Usage:
    python gen_hard_nc.py
    python gen_hard_nc.py --kfold_dir runs/A1_roberta_dc_kfold_t3nc3 --n_iters 2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from esg_main import (  # noqa: E402
    Config, IDX2LABEL, LABEL2IDX, extract_oof_probs, load_dataframes,
)

# ─────────────────────────────────────────────────────────────────────────────
# 5 NC feature types for condition_yx generation (rotate through them)
# ─────────────────────────────────────────────────────────────────────────────
NC_FEATURES = [
    (
        "fuzzy_timeline",
        "時間模糊：使用「持續」「陸續」「逐步」等模糊時間詞，完全不給出具體年份或截止日期",
        "在段落中提出承諾但迴避具體完成日期，改用「未來」「持續進行」「將逐步推動」等語氣"
    ),
    (
        "fuzzy_quantity",
        "量化模糊：使用「大幅」「顯著提升」「積極改善」等副詞，無具體百分比、金額或數字目標",
        "提到改善目標但不給任何數字，例如「大幅降低碳排放」而非「降低30%」"
    ),
    (
        "vague_promise",
        "空泛承諾：使用「致力於」「積極推動」「持續關注」等宣示性語言，缺乏具體行動計畫或執行機制",
        "以正面宣示性語言描述目標，但不說明任何具體步驟、負責人或可衡量的里程碑"
    ),
    (
        "no_kpi",
        "缺乏KPI：有提及行動或措施，但缺乏可量化的KPI或第三方驗證機制，讀者無法確認真實成效",
        "提到有在做，但不提供可驗證的數據或認證，讓讀者難以評估實際進展"
    ),
    (
        "conditional",
        "條件模糊：以「視市場情況」「如條件許可」「配合法規調整」等條件性語言，使承諾難以驗證",
        "在承諾中加入大量但書與條件，讓目標看起來有彈性空間，難以判斷是否真正履行"
    ),
]

ESG_TYPE_DESC = {
    "E": "環境（Environment）類：碳排放、能源使用、氣候變遷、環境保護相關",
    "S": "社會（Social）類：員工權益、供應鏈管理、社會責任、人權保障相關",
    "G": "治理（Governance）類：企業治理、法規遵循、董事會運作、風險管理相關",
}

HARDNC_PROMPT = """\
你是一位ESG永續報告寫作專家，同時精通「Not Clear（語意模糊）」的辨識。

請生成一段【全新的、不同於參考文字的】繁體中文ESG報告片段，需符合以下規格：

【ESG類型】{esg_type_desc}
【目標標籤】
- promise_status: Yes
- evidence_status: Yes
- evidence_quality: Not Clear（語意模糊，佐證不充分）
- verification_timeline: {timeline}

【Not Clear特徵類型】{nc_feature_name}：{nc_feature_desc}

【寫作技巧】{nc_writing_tip}

【參考語氣風格】（請勿複製原文，僅參考句式風格）
{ref_text}

【生成規則】
1. 長度：80~220字
2. 繁體中文，模擬真實台灣上市公司ESG報告語氣
3. 不要包含公司名稱
4. 必須展現「{nc_feature_name}」特徵，讓文本明確屬於Not Clear而非Clear
5. 不得包含具體百分比數字、具體完成年份、ISO認證號碼或第三方機構驗證聲明

請直接輸出報告片段，不要任何說明文字。"""

# ─────────────────────────────────────────────────────────────────────────────
# Rule-based verification
# ─────────────────────────────────────────────────────────────────────────────
_FUZZY_WORDS = {
    "持續推進", "努力改善", "積極探索", "逐步推動", "致力推動",
    "持續關注", "積極推動", "持續努力", "逐步落實", "積極改善",
    "持續進行", "將逐步", "陸續推動", "致力於", "持續致力",
    "積極落實", "持續推動", "逐步實現", "持續優化", "積極規劃",
}
_SPECIFIC_PAT = re.compile(
    r'\d+\s*%|\d+\s*億|\d+\s*萬|ISO\s*\d+|第三方驗證|第三方認證|已達成|已完成|已建置|已導入'
)


def rule_verify_nc(text: str) -> bool:
    has_fuzzy = any(w in text for w in _FUZZY_WORDS)
    has_specific = bool(_SPECIFIC_PAT.search(text))
    return has_fuzzy and not has_specific and len(text) >= 50


# ─────────────────────────────────────────────────────────────────────────────
# Diversity filter: character bigram Jaccard similarity
# ─────────────────────────────────────────────────────────────────────────────
def _bigrams(text: str) -> set[str]:
    chars = [c for c in text if not c.isspace()]
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def is_diverse(new_text: str, existing_texts: list[str], threshold: float = 0.85) -> bool:
    ng = _bigrams(new_text)
    if not ng:
        return False
    for old in existing_texts:
        og = _bigrams(old)
        if not og:
            continue
        sim = len(ng & og) / max(len(ng | og), 1)
        if sim > threshold:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Local LLM generation (Qwen3-8B, same as gen_llm_augdata)
# ─────────────────────────────────────────────────────────────────────────────
def _llm_generate(model, tokenizer, device: str, prompt_text: str,
                  max_tokens: int = 350, temp: float = 0.85) -> str:
    msgs = [{"role": "user", "content": prompt_text}]
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


def generate_nc_sample(model, tokenizer, device: str,
                       row: pd.Series, feature_idx: int) -> str | None:
    fname, fdesc, ftip = NC_FEATURES[feature_idx % len(NC_FEATURES)]
    esg_type = str(row.get("esg_type", "E"))
    timeline = str(row.get("verification_timeline", "N/A"))
    prompt = HARDNC_PROMPT.format(
        esg_type_desc=ESG_TYPE_DESC.get(esg_type, ESG_TYPE_DESC["E"]),
        nc_feature_name=fname,
        nc_feature_desc=fdesc,
        nc_writing_tip=ftip,
        ref_text=str(row["data"])[:250],
        timeline=timeline,
    )
    try:
        return _llm_generate(model, tokenizer, device, prompt, max_tokens=350, temp=0.85)
    except Exception as e:
        print(f"    [LLM error] {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="LLM2LLM hard NC augmentation")
    parser.add_argument("--kfold_dir", default="runs/A1_roberta_dc_kfold_t3nc3")
    parser.add_argument("--backbone", default="Qwen/Qwen3-8B",
                        help="Local generative LLM for augmentation")
    parser.add_argument("--nc_thr", type=float, default=0.65,
                        help="P(NC) < nc_thr → hard case (default 0.65)")
    parser.add_argument("--n_iters", type=int, default=1,
                        help="Augmentation iterations (default 1)")
    parser.add_argument("--diversity_thr", type=float, default=0.85)
    parser.add_argument("--out", default="2026-esg-classification-challenge/train_data_aug_hardnc.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load training data ────────────────────────────────────────────────────
    cfg = Config()
    train_df, _ = load_dataframes(
        cfg.data_dir, use_augmented=True, aug_filename="train_data_augmented.csv"
    )
    print(f"Loaded {len(train_df)} training samples")

    # ── Extract OOF probs ─────────────────────────────────────────────────────
    print(f"Extracting OOF probs from {args.kfold_dir} ...")
    oof_matrix, _ = extract_oof_probs(args.kfold_dir, train_df)

    # T3 probs: oof_matrix[:, 3:7] = [Clear, Not Clear, Misleading, N/A]
    t3_probs = oof_matrix[:, 3:]
    nc_prob = t3_probs[:, 1]
    pred_t3_idx = np.argmax(t3_probs, axis=1)
    true_t3 = np.array([LABEL2IDX["t3"].get(str(v), -1) for v in train_df["evidence_quality"]])
    nc_idx = LABEL2IDX["t3"]["Not Clear"]  # = 1

    # ── Hard NC: original rows (id < 100000) where model struggles ────────────
    orig_mask = train_df["id"].to_numpy() < 100000
    hard_mask = (
        orig_mask
        & (true_t3 == nc_idx)
        & ((pred_t3_idx != nc_idx) | (nc_prob < args.nc_thr))
    )
    hard_indices = np.where(hard_mask)[0]

    print(f"Original rows: {orig_mask.sum()}  |  NC in original: {(orig_mask & (true_t3 == nc_idx)).sum()}")
    print(f"Hard NC cases (pred≠NC or P(NC)<{args.nc_thr}): {len(hard_indices)}")
    if len(hard_indices) == 0:
        print("No hard NC cases found. Exiting.")
        return

    misclf = int(((orig_mask & (true_t3 == nc_idx)) & (pred_t3_idx != nc_idx)).sum())
    low_conf = int(((orig_mask & (true_t3 == nc_idx)) & (nc_prob < args.nc_thr)).sum())
    print(f"  Misclassified (pred≠NC): {misclf}  |  Low P(NC)<{args.nc_thr}: {low_conf}")

    # ── Load local LLM ────────────────────────────────────────────────────────
    print(f"\nLoading {args.backbone} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # ── Diversity pool ────────────────────────────────────────────────────────
    nc_texts = train_df[train_df["evidence_quality"] == "Not Clear"]["data"].tolist()
    print(f"Diversity pool: {len(nc_texts)} existing NC samples\n")

    augmented_rows: list[dict] = []
    next_id = 200001

    # ── Augmentation iterations ───────────────────────────────────────────────
    for iteration in range(args.n_iters):
        print(f"=== Iteration {iteration + 1}/{args.n_iters} ===")
        iter_generated = 0
        iter_skipped = 0

        for enum_i, orig_i in enumerate(hard_indices):
            row = train_df.iloc[orig_i]
            feature_idx = (iteration * len(NC_FEATURES) + enum_i) % len(NC_FEATURES)
            fname = NC_FEATURES[feature_idx][0]

            print(
                f"  [{enum_i+1:3d}/{len(hard_indices)}] id={row['id']} "
                f"P(NC)={nc_prob[orig_i]:.3f} "
                f"pred={IDX2LABEL['t3'][pred_t3_idx[orig_i]]:9s} "
                f"feat={fname}",
                end=" ",
            )

            text = generate_nc_sample(model, tokenizer, device, row, feature_idx)
            if text is None:
                print("→ LLM error")
                iter_skipped += 1
                continue

            if not rule_verify_nc(text):
                print(f"→ rule fail [{text[:35]!r}]")
                iter_skipped += 1
                continue

            all_existing = nc_texts + [r["data"] for r in augmented_rows]
            if not is_diverse(text, all_existing, args.diversity_thr):
                print("→ too similar")
                iter_skipped += 1
                continue

            new_row = row.to_dict()
            new_row["data"] = text
            new_row["id"] = next_id
            new_row["evidence_quality"] = "Not Clear"
            next_id += 1
            augmented_rows.append(new_row)
            nc_texts.append(text)
            iter_generated += 1
            print(f"→ OK ({len(text)} chars)")

        print(f"  Iter {iteration + 1}: generated={iter_generated} skipped={iter_skipped}\n")

    print(f"Total new hard NC samples: {len(augmented_rows)}")

    if not augmented_rows:
        print("Nothing generated. Exiting without writing file.")
        return

    # ── Save ──────────────────────────────────────────────────────────────────
    base_path = ROOT / "2026-esg-classification-challenge" / "train_data_augmented.csv"
    base_df = pd.read_csv(base_path)
    new_df = pd.DataFrame(augmented_rows)
    combined = pd.concat([base_df, new_df], ignore_index=True)
    out_path = ROOT / args.out
    combined.to_csv(out_path, index=False)
    nc_count = (combined["evidence_quality"] == "Not Clear").sum()
    print(f"Saved {len(combined)} rows → {out_path}")
    print(f"  ({len(base_df)} base  +  {len(new_df)} hard NC)")
    print(f"  NC count: {nc_count}/{len(combined)} ({nc_count/len(combined)*100:.1f}%)")


if __name__ == "__main__":
    main()
