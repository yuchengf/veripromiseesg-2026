#!/usr/bin/env python3
"""
llm_lora.py — LLM LoRA fine-tuning + RAG inference for ESG JSON output

Approach:
  - Instruction-tune Chinese LLMs via LoRA to output JSON labels
  - TF-IDF character n-gram retrieval (RAG) for dynamic few-shot examples
  - 5-fold CV matching esg_main.py split for fair comparison
  - Constrained JSON parsing + cascade rule post-processing

Model priority (all fit in 32 GB VRAM, bf16):
  ① Qwen/Qwen3-8B        ~16 GB  — ALREADY DOWNLOADED, best Chinese NLU,
                                    thinking mode available for inference
  ② Qwen/Qwen3-4B        ~8  GB  — newer than Qwen2.5-3B, better instruct follow
  ③ Qwen/Qwen2.5-7B-Instruct ~14GB — fallback if Qwen3-4B not available
  ④ Qwen/Qwen2.5-3B-Instruct ~6GB  — last resort, less capable

Why Qwen3 > Qwen2.5 for this task:
  - Stronger Chinese ESG domain understanding
  - Better structured output / instruction following
  - Thinking mode: model reasons step-by-step before producing JSON
    → helps with ambiguous Clear vs Not Clear boundaries
  Previous Qwen3-8B failures (Approach D/E) were methodology issues:
    D = mean-pooling head (not generative)  →  overcfitting
    E = zero-shot, no LoRA fine-tuning     →  poor generalization
  THIS approach: LoRA r=8, attn-only targets (~7M trainable params) + RAG

Thinking mode notes:
  --thinking=False (default): /no_think → JSON-only output, faster training
  --thinking=True  (inference): /think → model reasons before answering,
    useful for hard boundary cases (Clear vs Not Clear)

Usage:
  # ① Best: Qwen3-8B LoRA (already downloaded, no HF download needed)
  conda run -n AICUP python llm_lora.py --mode kfold \\
    --backbone Qwen/Qwen3-8B --rag_k 3 \\
    --run_dir runs/LLM_qwen3_8B_r8_rag3

  # RAG-only baseline, no training, with thinking enabled
  conda run -n AICUP python llm_lora.py --mode infer_rag \\
    --backbone Qwen/Qwen3-8B --rag_k 5 --thinking \\
    --run_dir runs/LLM_qwen3_8B_ragonly_think

  # ② Qwen3-4B if 8B overfits
  conda run -n AICUP python llm_lora.py --mode kfold \\
    --backbone Qwen/Qwen3-4B --rag_k 3 \\
    --run_dir runs/LLM_qwen3_4B_r8_rag3

  # After training, build submission CSVs
  conda run -n AICUP python llm_lora.py --mode gen_submissions
"""

import argparse
import ast
import json
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED = {
    "promise_status":       ["Yes", "No"],
    "evidence_status":      ["Yes", "No", "N/A"],
    "evidence_quality":     ["Clear", "Not Clear", "Misleading", "N/A"],
    "verification_timeline": ["already", "within_2_years", "between_2_and_5_years",
                               "longer_than_5_years", "N/A"],
}
TASK_KEYS = ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]
TASK_WEIGHTS = [0.20, 0.30, 0.35, 0.15]

# Scoring weights from competition
SCORE_W = dict(zip(TASK_KEYS, TASK_WEIGHTS))

# ── Shared task definition (used in both system prompts) ──
_TASK_DEF = """
## 評估指標

**T1 - promise_status（承諾狀態）**：文本是否包含ESG承諾、目標或計畫
- "Yes"：包含明確的ESG承諾/目標/行動計畫（含未來期望）
- "No"：純粹的事實描述或說明，不含任何ESG承諾

**T2 - evidence_status（執行證據）**：是否有具體執行行動或量化成果的證據
- "Yes"：有具體行動、數據、成果或第三方核實作為佐證
- "No"：承諾存在但僅為意向聲明，無具體執行證明
- "N/A"：T1="No" 時此項必須為 "N/A"

**T3 - evidence_quality（證據品質）**：現有執行證據的可信度與清晰度
- "Clear"：證據具體明確、可量化、可核實（含數字、日期、第三方）
- "Not Clear"：證據存在但模糊、難以量化或核實
- "Misleading"：證據存在但具誤導性、自相矛盾或刻意迴避
- "N/A"：T2="No" 或 T2="N/A" 時此項必須為 "N/A"

**T4 - verification_timeline（目標時間線）**：承諾或計畫的預計完成時間
- "already"：目標已達成或正在持續執行中（現在進行式）
- "within_2_years"：承諾預計在2年內完成
- "between_2_and_5_years"：承諾預計在2至5年內完成
- "longer_than_5_years"：承諾需5年以上才能完成
- "N/A"：T1="No" 時此項必須為 "N/A"

## 語言信號指引

**模糊字眼（漂綠高風險）**
- 常見詞彙：「持續推進」「努力改善」「積極探索」「逐步推動」「將致力於」「致力推動」「持續關注」
- 判斷規則：
  - 文中有執行聲稱，但全用模糊字眼、無具體數字/日期/第三方 → evidence_status="Yes", evidence_quality="Not Clear"
  - 文中完全無執行聲稱，只有意向聲明 → evidence_status="No", evidence_quality="N/A"
  - 注意：evidence_quality="Not Clear" 只在 evidence_status="Yes" 時才有效

**具體佐證信號 → 通常對應 Clear**
- 明確數字（如「降低15%」「達成25%」「投入3億」）
- 具體年份（如「2023年已完成」「截至本年度」）
- 第三方認證（如「ISO 14064」「GRI準則」「外部稽核」）
- 已完成的具體行動（如「已建置」「已導入」「已達成」）

**時間線信號**
- "already"：「已」「目前」「截至」「持續執行中」「本年度達成」
- "within_2_years"：「明年」「2年內」「近期」「短期」
- "between_2_and_5_years"：「2030年」「3至5年」「中期目標」
- "longer_than_5_years"：「2035年後」「長期願景」「下一個十年」

## 強制邏輯規則（違反即為錯誤）
1. promise_status="No" → evidence_status="N/A", evidence_quality="N/A", verification_timeline="N/A"
2. evidence_status="No" 或 "N/A" → evidence_quality="N/A"
"""

# System prompt: PromptCast format with CoT extraction
SYSTEM_PROMPT_PROMPTCAST = """你是一位專業的ESG（環境、社會、治理）報告核實分析師。
分析給定的ESG報告片段，先識別關鍵文字片段，再依序評估四個指標。
""" + _TASK_DEF + """
## 輸出格式（先提取片段，再PromptCast評估）
**步驟1**：用「」引號標出原文中的關鍵片段：
- 承諾片段：含ESG承諾/目標的關鍵句（若無ESG承諾則寫「無」）
- 佐證片段：含具體執行/量化數據的關鍵句（若無佐證則寫「無」）

**步驟2**：用一個連貫的中文段落，在每個判斷點後用全形括號（）標注答案值。
四個值必須依序出現：promise_status → evidence_status → evidence_quality → verification_timeline。

範例（Yes/Yes/Clear/already）：
承諾片段：「自2030年起減少50%碳排放」。佐證片段：「2023年已完成3座太陽能電站建設，減碳15%」。
此ESG報告片段包含明確的永續目標（Yes），有具體建設成果與量化數據作為佐證（Yes），佐證數據清晰可核實（Clear），部分目標已在持續執行中（already）。

範例（No/N/A/N/A/N/A）：
承諾片段：無。佐證片段：無。
此段文字為純粹的事實描述，不含ESG承諾或目標（No），故執行證據（N/A）、證據品質（N/A）及驗證時間線（N/A）均不適用。"""

# System prompt: JSON format with CoT extraction
SYSTEM_PROMPT_JSON = """你是一位專業的ESG（環境、社會、治理）報告核實分析師。
分析給定的ESG報告片段，先識別關鍵文字片段，再評估四個指標，以JSON格式輸出。
""" + _TASK_DEF + """
## 輸出格式（JSON with CoT）
輸出純JSON，先提取原文片段作為推理依據，再給出四個分類結果：
{
  "promise_string": "含ESG承諾/目標的原文關鍵句，若無ESG承諾則為null",
  "evidence_string": "含具體執行/量化數據的原文關鍵句，若無佐證則為null",
  "promise_status": "Yes|No",
  "evidence_status": "Yes|No|N/A",
  "evidence_quality": "Clear|Not Clear|Misleading|N/A",
  "verification_timeline": "already|within_2_years|between_2_and_5_years|longer_than_5_years|N/A"
}

範例（Yes/Yes/Clear/already）：
{"promise_string": "自2030年起減少50%碳排放", "evidence_string": "2023年已完成3座太陽能電站，減碳15%", "promise_status": "Yes", "evidence_status": "Yes", "evidence_quality": "Clear", "verification_timeline": "already"}

範例（No/N/A/N/A/N/A）：
{"promise_string": null, "evidence_string": null, "promise_status": "No", "evidence_status": "N/A", "evidence_quality": "N/A", "verification_timeline": "N/A"}"""

# Backward compat alias (used nowhere now but kept for safety)
SYSTEM_PROMPT = SYSTEM_PROMPT_PROMPTCAST


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    backbone: str = "Qwen/Qwen3-8B"           # already downloaded
    run_dir: str = "runs/LLM_qwen3_8B_r8_rag3"
    data_path: str = "2026-esg-classification-challenge/train_data.csv"
    test_path: str = "2026-esg-classification-challenge/test_data.csv"
    mode: str = "kfold"

    # LoRA — attention-only to keep trainable params ~7M (safe for 640 samples)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    attn_only: bool = True       # True = q/k/v/o only; False = also FFN (gate/up/down)

    # Training
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4          # effective batch = batch_size * grad_accum = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    max_length: int = 1024       # input+output combined

    # RAG
    rag_k: int = 3               # 0 = no RAG

    # KFold
    kfold: int = 5
    seed: int = 42

    # Qwen3 thinking mode
    # False (default for training): adds /no_think → clean JSON output
    # True  (optional for inference): adds /think  → chain-of-thought before JSON
    use_thinking: bool = False

    # Output format: "promptcast" (NL sentence + labels in （）) or "json" (JSON with CoT fields)
    output_format: str = "promptcast"

    # Misc
    dtype: str = "bf16"          # bf16 or fp16 or fp32


# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF RETRIEVER  (BM25-like with character n-grams, no jieba needed)
# ──────────────────────────────────────────────────────────────────────────────

class TFIDFRetriever:
    """Character n-gram TF-IDF retriever for Chinese text (no jieba needed)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.ids = df["id"].tolist()
        # char 2-4 grams work well for Chinese without segmentation
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=30000)
        self.matrix = self.vec.fit_transform(df["data"].tolist())

    def retrieve(self, query: str, k: int, exclude_id: Optional[int] = None) -> list[dict]:
        if k == 0:
            return []
        q_vec = self.vec.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        if exclude_id is not None and exclude_id in self.ids:
            sims[self.ids.index(exclude_id)] = -1.0
        top_idx = sims.argsort()[::-1][:k]
        return [self.df.iloc[i].to_dict() for i in top_idx]


# ──────────────────────────────────────────────────────────────────────────────
# MESSAGE BUILDING
# ──────────────────────────────────────────────────────────────────────────────

def _label_to_dict(label_raw) -> dict:
    """Convert label string/list to dict."""
    if isinstance(label_raw, str):
        label_raw = ast.literal_eval(label_raw)
    return dict(zip(TASK_KEYS, label_raw))


def _safe_str(val) -> Optional[str]:
    """Return None if val is NaN/None, else str."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val) if val else None


def _format_json_cot_response(row_dict: dict) -> str:
    """JSON output with CoT: includes promise_string/evidence_string from training annotations."""
    d = _label_to_dict(row_dict["label"])
    ps = _safe_str(row_dict.get("promise_string"))
    es = _safe_str(row_dict.get("evidence_string"))
    return json.dumps({
        "promise_string": ps,
        "evidence_string": es,
        "promise_status": d["promise_status"],
        "evidence_status": d["evidence_status"],
        "evidence_quality": d["evidence_quality"],
        "verification_timeline": d["verification_timeline"],
    }, ensure_ascii=False)


def _format_response(row_dict: dict, output_format: str) -> str:
    """Dispatch to the correct response formatter based on output_format."""
    if output_format == "json":
        return _format_json_cot_response(row_dict)
    else:  # promptcast
        ps = _safe_str(row_dict.get("promise_string"))
        es = _safe_str(row_dict.get("evidence_string"))
        prefix = f"承諾片段：「{ps}」。" if ps else "承諾片段：無。"
        prefix += f"佐證片段：「{es}」。" if es else "佐證片段：無。"
        return prefix + "\n" + _format_promptcast_response(row_dict["label"])


def build_messages(row: dict, few_shots: list[dict],
                   include_response: bool = False,
                   use_thinking: bool = False,
                   output_format: str = "promptcast") -> list[dict]:
    """Build ChatML messages for one example, with optional RAG few-shots.

    use_thinking=False → prepend /no_think (Qwen3: skip CoT, output JSON directly)
    use_thinking=True  → prepend /think  (Qwen3: reason first, then output JSON)
    output_format: "promptcast" (default) or "json"
    """
    # Select system prompt based on output format
    system_content = SYSTEM_PROMPT_JSON if output_format == "json" else SYSTEM_PROMPT_PROMPTCAST

    user_parts = []

    # Qwen3 thinking mode control (safe no-op for other model families)
    thinking_prefix = "/think\n" if use_thinking else "/no_think\n"
    user_parts.append(thinking_prefix)

    # Inject few-shot examples (in-context demonstrations)
    if few_shots:
        user_parts.append("以下是一些已標注的參考案例，供你理解標注標準：\n")
        for i, ex in enumerate(few_shots, 1):
            # Build context tag (company + esg_type if available)
            ctx_parts = []
            company = _safe_str(ex.get("company"))
            esg_type = _safe_str(ex.get("esg_type"))
            if company:
                ctx_parts.append(f"公司：{company}")
            if esg_type:
                ctx_parts.append(f"ESG類別：{esg_type}")
            ctx_str = f"[{', '.join(ctx_parts)}] " if ctx_parts else ""
            resp = _format_response(ex, output_format)
            user_parts.append(f"[參考案例 {i}]\n{ctx_str}文本：{ex['data']}\n分析：{resp}")
        user_parts.append("")  # blank line separator

    # Main query context (company available in both train & test)
    ctx_parts = []
    company = _safe_str(row.get("company"))
    esg_type = _safe_str(row.get("esg_type"))
    if company:
        ctx_parts.append(f"公司：{company}")
    if esg_type:
        ctx_parts.append(f"ESG類別：{esg_type}")
    if ctx_parts:
        user_parts.append(f"[{', '.join(ctx_parts)}]")
    user_parts.append(f"請分析以下ESG報告片段：\n\n{row['data']}")

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": "\n".join(user_parts)},
    ]

    if include_response:
        messages.append({"role": "assistant", "content": _format_response(row, output_format)})

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTCAST RESPONSE FORMAT
# Labels are embedded inside （） in a coherent Chinese sentence.
# Extraction uses regex to find valid labels in order of appearance.
# ──────────────────────────────────────────────────────────────────────────────

# Per-task sentence fragments; label value is appended inside （）
_PC_PHRASES = {
    "promise_status": {
        "Yes": "此ESG報告片段包含明確的永續承諾、目標或行動計畫",
        "No":  "此ESG報告片段為純粹的事實描述，不含ESG承諾或目標",
    },
    "evidence_status": {
        "Yes": "並有具體的執行行動、量化數據或可核實成果作為佐證",
        "No":  "然而承諾僅為意向聲明，缺乏具體執行行動的佐證",
        "N/A": "故執行證據狀態",
    },
    "evidence_quality": {
        "Clear":      "執行證據清晰、可量化且可獨立核實",
        "Not Clear":  "執行證據存在但模糊，難以量化或獨立核實",
        "Misleading": "執行證據存在但具誤導性或與承諾自相矛盾",
        "N/A":        "證據品質",
    },
    "verification_timeline": {
        "already":                "相關目標已達成或正持續執行中",
        "within_2_years":         "承諾預計在2年內完成",
        "between_2_and_5_years":  "承諾預計在2至5年內完成",
        "longer_than_5_years":    "承諾需5年以上方能達成",
        "N/A":                    "驗證時間線",
    },
}

def _format_promptcast_response(label_raw) -> str:
    """Generate a PromptCast-style sentence with labels embedded in （）.

    Output example:
      此ESG報告片段包含明確的永續承諾、目標或行動計畫（Yes），
      並有具體的執行行動、量化數據或可核實成果作為佐證（Yes），
      執行證據清晰、可量化且可獨立核實（Clear），
      相關目標已達成或正持續執行中（already）。
    """
    d = _label_to_dict(label_raw)
    parts = []
    for key in TASK_KEYS:
        val = d[key]
        phrase = _PC_PHRASES[key][val]
        parts.append(f"{phrase}（{val}）")
    return "，".join(parts) + "。"


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT PARSING + POST-PROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def parse_output(text: str) -> Optional[dict]:
    """Parse model output with PromptCast extraction as primary, JSON as fallback.

    PromptCast: find valid label values inside （） in order of appearance.
    JSON fallback: find first {...} object and extract fields.
    Qwen3 thinking tokens (<think>...</think>) are stripped first.
    """
    # Strip Qwen3 thinking block
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # ── Primary: PromptCast （label） extraction ──
    # All possible label values, longest first to avoid "N/A" matching inside longer values
    _ALL_VALUES = (
        "between_2_and_5_years|longer_than_5_years|within_2_years|already"
        "|Not Clear|Misleading|Clear|Yes|No|N/A"
    )
    pc_pattern = rf'（({_ALL_VALUES})）'
    matches = re.findall(pc_pattern, text)
    if len(matches) >= 4:
        labels = matches[:4]
        # Validate: each value must be in its task's allowed set
        valid = all(labels[i] in list(ALLOWED.values())[i] for i in range(4))
        if valid:
            return dict(zip(TASK_KEYS, labels))

    # ── Fallback: JSON extraction ──
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            result = {}
            for key, allowed in ALLOWED.items():
                val = obj.get(key, "").strip()
                if val not in allowed:
                    val = _fuzzy_match(val, allowed)
                result[key] = val
            return result
        except json.JSONDecodeError:
            pass

    return None


# Keep old name as alias for any remaining call sites
parse_json_output = parse_output


def _fuzzy_match(val: str, allowed: list[str]) -> str:
    """Return the allowed value with the most character overlap."""
    val_lower = val.lower()
    best, best_score = allowed[0], -1
    for a in allowed:
        score = sum(c in val_lower for c in a.lower())
        if score > best_score:
            best, best_score = a, score
    return best


def apply_na_rule(pred: dict) -> dict:
    """Enforce cascade rules regardless of model output."""
    p = dict(pred)
    if p["promise_status"] == "No":
        p["evidence_status"]      = "N/A"
        p["evidence_quality"]     = "N/A"
        p["verification_timeline"] = "N/A"
    if p["evidence_status"] in ("No", "N/A"):
        p["evidence_quality"] = "N/A"
    return p


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_weighted_f1(preds: list[dict], golds: list[dict]) -> tuple[float, dict]:
    """Return weighted macro-F1 matching competition scoring."""
    per_task = {}
    for key in TASK_KEYS:
        y_true = [g[key] for g in golds]
        y_pred = [p[key] for p in preds]
        per_task[key] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = sum(SCORE_W[k] * per_task[k] for k in TASK_KEYS)
    return weighted, per_task


# ──────────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────────

class ESGSFTDataset(Dataset):
    """SFT dataset that pre-tokenizes prompt+response with response-only labels."""

    def __init__(self, df: pd.DataFrame, tokenizer, retriever: TFIDFRetriever,
                 rag_k: int = 3, max_length: int = 1024, training: bool = True,
                 output_format: str = "promptcast"):
        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            few_shots = retriever.retrieve(row["data"], k=rag_k,
                                           exclude_id=row["id"]) if rag_k > 0 else []
            # Build full conversation (including assistant response)
            # use_thinking=False during training: model outputs labels directly, no <think> overhead
            messages = build_messages(row.to_dict(), few_shots, include_response=True,
                                      use_thinking=False, output_format=output_format)
            # Build prompt-only part to determine label start position
            prompt_messages = messages[:-1]

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True)

            full_ids = tokenizer(full_text, add_special_tokens=False,
                                 max_length=max_length, truncation=True)["input_ids"]
            # Must truncate prompt_ids to same max_length; without this,
            # prompt_len can exceed len(full_ids) → all labels=-100 → loss=nan
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False,
                                   max_length=max_length, truncation=True)["input_ids"]
            prompt_len = len(prompt_ids)

            # Skip samples where response was entirely truncated away
            if prompt_len >= len(full_ids):
                skipped += 1
                continue

            labels = [-100] * prompt_len + full_ids[prompt_len:]

            self.samples.append({
                "input_ids": full_ids,
                "labels": labels,
            })
        if skipped:
            print(f"  [ESGSFTDataset] WARNING: skipped {skipped}/{len(df)} samples (response truncated away)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v, dtype=torch.long) for k, v in self.samples[idx].items()}


def collate_fn(batch: list[dict], pad_id: int) -> dict:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100,   dtype=torch.long)
    attn_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        L = x["input_ids"].shape[0]
        input_ids[i, :L] = x["input_ids"]
        labels[i, :L]    = x["labels"]
        attn_mask[i, :L] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}


# ──────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
    """Load base model and tokenizer, then apply LoRA."""
    from peft import LoraConfig, TaskType, get_peft_model

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[cfg.dtype]

    print(f"  Loading tokenizer: {cfg.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT: pad right

    print(f"  Loading model: {cfg.backbone} (dtype={cfg.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.backbone,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Detect LoRA target modules from model architecture
    target_modules = _detect_target_modules(model, attn_only=cfg.attn_only)
    print(f"  LoRA target_modules: {target_modules}  (attn_only={cfg.attn_only})")

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def _detect_target_modules(model, attn_only: bool = True) -> list[str]:
    """Auto-detect projection layer names from the model.

    attn_only=True  → q/k/v/o only (~7M params for Qwen3-8B r=8), safer for 640 samples
    attn_only=False → also include gate/up/down FFN (~20M params), more expressive
    """
    attn_candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
    ffn_candidates  = ["gate_proj", "up_proj", "down_proj"]
    candidates = attn_candidates if attn_only else attn_candidates + ffn_candidates

    found = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                found.add(c)
    # Fallback for other architectures (e.g. Bloom, Falcon)
    if not found:
        for name, _ in model.named_modules():
            if any(x in name for x in ["query", "key", "value", "dense"]):
                found.add(name.split(".")[-1])
    return sorted(found) if found else ["q_proj", "v_proj"]


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING (single fold)
# ──────────────────────────────────────────────────────────────────────────────

def train_fold(cfg: Config, train_df: pd.DataFrame, val_df: pd.DataFrame,
               retriever: TFIDFRetriever, fold_dir: Path) -> dict:
    """Train one LoRA fold, return best val metrics."""
    model, tokenizer = load_model_and_tokenizer(cfg)
    # Gradient checkpointing: trade compute for memory (~30% slower, ~40% less VRAM)
    # enable_input_require_grads() is required for PEFT/LoRA compatibility
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    device = next(model.parameters()).device

    train_ds = ESGSFTDataset(train_df, tokenizer, retriever, cfg.rag_k, cfg.max_length,
                             output_format=cfg.output_format)
    collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = (len(loader) // cfg.grad_accum) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = _get_cosine_schedule(optimizer, warmup_steps, total_steps)

    best_score, best_state = 0.0, None
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            loss = out.loss / cfg.grad_accum
            loss.backward()
            epoch_loss += loss.item() * cfg.grad_accum

            if step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(loader)
        val_score, val_per_task = evaluate_fold(model, tokenizer, val_df, retriever, cfg)
        print(f"  [ep{epoch}/{cfg.epochs}] loss={avg_loss:.4f}  "
              f"val_weighted={val_score:.4f}  "
              f"T1={val_per_task['promise_status']:.3f} "
              f"T2={val_per_task['evidence_status']:.3f} "
              f"T3={val_per_task['evidence_quality']:.3f} "
              f"T4={val_per_task['verification_timeline']:.3f}")

        if val_score > best_score:
            best_score = val_score
            # Save LoRA adapter only (much smaller than full model)
            model.save_pretrained(fold_dir / "adapter")
            tokenizer.save_pretrained(fold_dir / "adapter")
            print(f"    ↑ best saved ({val_score:.4f})")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    print(f"  [VRAM] freed → {torch.cuda.mem_get_info()[0]/1e9:.1f} GB free")
    return {"val_score": best_score, "per_task": val_per_task}


def _get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay scheduler."""
    from torch.optim.lr_scheduler import LambdaLR
    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_fn)


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE (single fold or test set)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_fold(model, tokenizer, df: pd.DataFrame,
                  retriever: TFIDFRetriever, cfg: Config) -> tuple[float, dict]:
    """Run inference on df, return (weighted_f1, per_task_f1)."""
    model.eval()
    preds, golds = [], []
    device = next(model.parameters()).device
    for _, row in df.iterrows():
        pred = _infer_one(model, tokenizer, row.to_dict(), retriever, cfg, device)
        preds.append(pred)
        golds.append(_label_to_dict(row["label"]))
    return compute_weighted_f1(preds, golds)


@torch.no_grad()
def _infer_one(model, tokenizer, row: dict, retriever: TFIDFRetriever,
               cfg: Config, device) -> dict:
    """Infer labels for a single row, with fallback on parse failure."""
    few_shots = retriever.retrieve(row["data"], k=cfg.rag_k,
                                   exclude_id=row.get("id")) if cfg.rag_k > 0 else []
    messages = build_messages(row, few_shots, include_response=False,
                              use_thinking=cfg.use_thinking,
                              output_format=cfg.output_format)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=cfg.max_length,
                       truncation=True).to(device)

    # JSON CoT needs ~200 tokens; PromptCast CoT needs ~250 (extraction prefix + labels)
    # thinking mode needs more for <think> block
    max_new = 512 if cfg.use_thinking else 300
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    parsed = parse_output(generated)
    if parsed is None:
        # Fallback: most-common labels
        parsed = {"promise_status": "Yes", "evidence_status": "Yes",
                  "evidence_quality": "Clear", "verification_timeline": "already"}
    return apply_na_rule(parsed)


def infer_dataset(model, tokenizer, df: pd.DataFrame,
                  retriever: TFIDFRetriever, cfg: Config) -> list[dict]:
    """Infer all rows in df (no labels required)."""
    model.eval()
    device = next(model.parameters()).device
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        pred = _infer_one(model, tokenizer, row.to_dict(), retriever, cfg, device)
        results.append(pred)
        if (i + 1) % 20 == 0:
            print(f"    infer {i+1}/{len(df)}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5-FOLD CV  (matching esg_main.py split exactly)
# ──────────────────────────────────────────────────────────────────────────────

def kfold_train(cfg: Config):
    train_df = pd.read_csv(cfg.data_path)
    test_df  = pd.read_csv(cfg.test_path)
    run_dir  = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=cfg.kfold, shuffle=True, random_state=cfg.seed)
    y = train_df["promise_status"].values  # same stratification as esg_main.py

    # Build retriever on full training set (for inference-time RAG)
    full_retriever = TFIDFRetriever(train_df)

    fold_scores, oof_preds = [], {}
    test_softmaxes = []   # collect fold predictions on test set

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y), 1):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold}/{cfg.kfold}")
        print(f"{'='*60}")
        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True)

        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        # Retriever built only from training portion (no leakage)
        fold_retriever = TFIDFRetriever(tr_df)

        result = train_fold(cfg, tr_df, va_df, fold_retriever, fold_dir)
        fold_scores.append(result["val_score"])
        print(f"\n  Fold {fold} best val: {result['val_score']:.4f}")

        # Re-load best adapter for OOF and test inference
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.backbone,
            torch_dtype=torch.bfloat16 if cfg.dtype == "bf16" else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(fold_dir / "adapter", trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, fold_dir / "adapter")
        model.eval()

        # OOF predictions
        for (_, row), pred in zip(va_df.iterrows(), infer_dataset(model, tokenizer, va_df, fold_retriever, cfg)):
            oof_preds[row["id"]] = pred

        # Test set predictions (use full retriever)
        test_preds = infer_dataset(model, tokenizer, test_df, full_retriever, cfg)
        test_softmaxes.append(test_preds)

        del model, base_model
        torch.cuda.empty_cache()

    mean_score = np.mean(fold_scores)
    std_score  = np.std(fold_scores)
    print(f"\n{'='*60}")
    print(f"5-Fold CV: {mean_score:.4f} ± {std_score:.4f}")
    for i, s in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {s:.4f}")

    # ── OOF evaluation ──
    oof_df = train_df.copy()
    oof_df["pred"] = oof_df["id"].map(lambda x: oof_preds.get(x, {}))
    oof_preds_list = [oof_preds.get(i, {}) for i in oof_df["id"]]
    oof_golds = [_label_to_dict(lb) for lb in oof_df["label"]]
    oof_score, oof_per = compute_weighted_f1(oof_preds_list, oof_golds)
    print(f"\nOOF weighted F1: {oof_score:.4f}  {oof_per}")

    # ── Ensemble test predictions (majority vote per label key) ──
    test_ensemble = _majority_vote(test_softmaxes)
    submission = _build_submission(test_df, test_ensemble)
    sub_path = run_dir / "submission_kfold.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved → {sub_path}")

    # Save fold scores summary
    summary = {"backbone": cfg.backbone, "run_dir": cfg.run_dir,
               "fold_scores": fold_scores, "mean": mean_score, "std": std_score,
               "oof_score": oof_score}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _majority_vote(fold_preds: list[list[dict]]) -> list[dict]:
    """Majority vote across fold predictions for each test sample."""
    from collections import Counter
    n_test = len(fold_preds[0])
    result = []
    for i in range(n_test):
        row_preds = [fold_preds[f][i] for f in range(len(fold_preds))]
        voted = {}
        for key in TASK_KEYS:
            counts = Counter(p[key] for p in row_preds)
            voted[key] = counts.most_common(1)[0][0]
        result.append(apply_na_rule(voted))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE RUN  (80/20 split, one training run — fast format comparison)
# ──────────────────────────────────────────────────────────────────────────────

def single_train(cfg: Config):
    """Train once on 80% of data, evaluate on 20%, generate test submission.
    Used for quick format comparison (json vs promptcast) without full kfold overhead.
    """
    from sklearn.model_selection import train_test_split

    train_df = pd.read_csv(cfg.data_path)
    test_df  = pd.read_csv(cfg.test_path)
    run_dir  = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    tr_df, va_df = train_test_split(
        train_df, test_size=0.2, random_state=cfg.seed,
        stratify=train_df["promise_status"])
    tr_df = tr_df.reset_index(drop=True)
    va_df = va_df.reset_index(drop=True)

    retriever = TFIDFRetriever(tr_df)
    result = train_fold(cfg, tr_df, va_df, retriever, run_dir)
    print(f"\nVal weighted F1: {result['val_score']:.4f}")

    # Re-load best adapter for test inference
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.backbone,
        torch_dtype=torch.bfloat16 if cfg.dtype == "bf16" else torch.float16,
        device_map="auto", trust_remote_code=True,
    )
    full_retriever = TFIDFRetriever(train_df)
    tokenizer = AutoTokenizer.from_pretrained(run_dir / "adapter", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, run_dir / "adapter")
    model.eval()

    test_preds = infer_dataset(model, tokenizer, test_df, full_retriever, cfg)
    submission = _build_submission(test_df, test_preds)
    sub_path = run_dir / "submission_single.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved → {sub_path}")

    summary = {"output_format": cfg.output_format, "val_score": result["val_score"]}
    with open(run_dir / "single_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    del model, base_model
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# RAG-ONLY INFERENCE  (no training, zero-shot + few-shot baseline)
# ──────────────────────────────────────────────────────────────────────────────

def infer_rag_only(cfg: Config, eval_sample: int = 100):
    """Pure RAG few-shot inference (no LoRA training).

    eval_sample: number of training samples for quick proxy evaluation (LOO).
                 Set to 0 to skip train evaluation entirely.
                 Full 800-sample LOO with thinking=True takes ~2 hours.
    """
    train_df = pd.read_csv(cfg.data_path)
    test_df  = pd.read_csv(cfg.test_path)
    run_dir  = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    print(f"Loading {cfg.backbone} (dtype={cfg.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.backbone, torch_dtype=dtype_map[cfg.dtype],
        device_map="auto", trust_remote_code=True)
    model.eval()

    retriever = TFIDFRetriever(train_df)

    # Quick proxy evaluation on a stratified sample (avoid 2-hour full LOO)
    if eval_sample > 0:
        sample_df = train_df.groupby("promise_status", group_keys=False).apply(
            lambda g: g.sample(min(len(g), eval_sample // 2), random_state=42)
        ).head(eval_sample).reset_index(drop=True)
        print(f"Quick eval on {len(sample_df)} sampled training rows ...")
        preds, golds = [], []
        for _, row in sample_df.iterrows():
            pred = _infer_one(model, tokenizer, row.to_dict(), retriever, cfg,
                              next(model.parameters()).device)
            preds.append(pred)
            golds.append(_label_to_dict(row["label"]))
        score, per_task = compute_weighted_f1(preds, golds)
        print(f"Sample Train weighted F1: {score:.4f}")
        print(f"  T1={per_task['promise_status']:.3f} T2={per_task['evidence_status']:.3f} "
              f"T3={per_task['evidence_quality']:.3f} T4={per_task['verification_timeline']:.3f}")

    # Inference on test set
    print("Inferring test set ...")
    test_preds = infer_dataset(model, tokenizer, test_df, retriever, cfg)
    submission = _build_submission(test_df, test_preds)
    sub_path = run_dir / "submission_ragonly.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved → {sub_path}")

    summary = {"backbone": cfg.backbone, "rag_k": cfg.rag_k,
               "loo_train_score": score if eval_sample > 0 else None,
               "per_task": per_task if eval_sample > 0 else {}}
    with open(run_dir / "ragonly_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    del model
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# SUBMISSION GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def _build_submission(test_df: pd.DataFrame, preds: list[dict]) -> pd.DataFrame:
    rows = []
    for (_, row), pred in zip(test_df.iterrows(), preds):
        label_str = str([pred["promise_status"], pred["evidence_status"],
                          pred["evidence_quality"], pred["verification_timeline"]])
        rows.append({"id": row["id"], "label": label_str})
    return pd.DataFrame(rows)


# Run dirs that have kfold/ragonly submissions to auto-discover
_LLM_KFOLD_DIRS = {
    # name                           run_dir
    "v5_qwen3_8B_r8_rag3":          "runs/LLM_qwen3_8B_r8_rag3",
    "v5_qwen3_8B_ragonly_think":     "runs/LLM_qwen3_8B_ragonly_think",
    "v5_qwen3_4B_r8_rag3":          "runs/LLM_qwen3_4B_r8_rag3",
    "v5_qwen25_7B_r8_rag3":         "runs/LLM_qwen25_7B_r8_rag3",
}

def gen_submissions():
    """Generate submission CSVs for all completed LLM runs."""
    sub_dir = Path("submissions")
    sub_dir.mkdir(exist_ok=True)

    # Find highest existing sNN index
    existing = [int(re.match(r's(\d+)_', p.name).group(1))
                for p in sub_dir.glob("s*_*.csv")
                if re.match(r's(\d+)_', p.name)]
    next_id = max(existing, default=75) + 1

    for name, run_dir in _LLM_KFOLD_DIRS.items():
        run_path = Path(run_dir)
        # Check for kfold submission
        for sub_fname in ["submission_kfold.csv", "submission_ragonly.csv"]:
            src = run_path / sub_fname
            if src.exists():
                tag = "kfold" if "kfold" in sub_fname else "ragonly"
                dst = sub_dir / f"s{next_id}_{name}_{tag}.csv"
                if not dst.exists():
                    import shutil
                    shutil.copy(src, dst)
                    print(f"  s{next_id} ← {src} → {dst}")
                    next_id += 1


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM LoRA + RAG for ESG classification")
    parser.add_argument("--mode", choices=["kfold", "single", "infer_rag", "gen_submissions"],
                        default="kfold")
    parser.add_argument("--backbone",   default="Qwen/Qwen3-8B")
    parser.add_argument("--run_dir",    default="runs/LLM_qwen3_8B_r8_rag3")
    parser.add_argument("--lora_r",     type=int,   default=8)
    parser.add_argument("--lora_alpha", type=int,   default=16)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--grad_accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--rag_k",      type=int,   default=3,
                        help="Number of RAG few-shot examples (0=disable)")
    parser.add_argument("--kfold",      type=int,   default=5)
    parser.add_argument("--max_length", type=int,   default=1024)
    parser.add_argument("--dtype",      choices=["bf16","fp16","fp32"], default="bf16")
    parser.add_argument("--thinking",   action="store_true",
                        help="Enable Qwen3 thinking mode (adds /think prefix; use for inference)")
    parser.add_argument("--no_attn_only", action="store_true", default=False,
                        help="Also add LoRA to FFN modules (gate/up/down), not just attention")
    parser.add_argument("--output_format", choices=["promptcast", "json"], default="promptcast",
                        help="Output format: 'promptcast' (NL+labels) or 'json' (JSON with CoT)")
    parser.add_argument("--data_path",  default="2026-esg-classification-challenge/train_data.csv")
    parser.add_argument("--test_path",  default="2026-esg-classification-challenge/test_data.csv")
    args = parser.parse_args()

    cfg = Config(
        backbone=args.backbone,
        run_dir=args.run_dir,
        data_path=args.data_path,
        test_path=args.test_path,
        mode=args.mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        rag_k=args.rag_k,
        kfold=args.kfold,
        max_length=args.max_length,
        dtype=args.dtype,
        use_thinking=args.thinking,
        attn_only=not args.no_attn_only,
        output_format=args.output_format,
    )

    print(f"{'='*60}")
    print(f"  Mode: {cfg.mode}")
    print(f"  Backbone: {cfg.backbone}")
    print(f"  LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}, epochs={cfg.epochs}, attn_only={cfg.attn_only}")
    print(f"  RAG k={cfg.rag_k}, batch={cfg.batch_size}×{cfg.grad_accum}={cfg.batch_size*cfg.grad_accum}")
    print(f"  Output format: {cfg.output_format}")
    print(f"  Thinking mode: {cfg.use_thinking}")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"{'='*60}\n")

    if args.mode == "kfold":
        kfold_train(cfg)
    elif args.mode == "single":
        single_train(cfg)
    elif args.mode == "infer_rag":
        infer_rag_only(cfg)
    elif args.mode == "gen_submissions":
        gen_submissions()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
