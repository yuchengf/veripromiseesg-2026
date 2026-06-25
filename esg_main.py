"""
AI CUP 2026 — VeriPromise ESG Classification
Single-file implementation: Dataset / Models / Loss / Train / Evaluate / Predict / Tests

Modes:
  python esg_main.py --mode test
  python esg_main.py --mode train   --approach A   --backbone hfl/chinese-macbert-large
  python esg_main.py --mode train   --approach A1  --backbone hfl/chinese-macbert-large
  python esg_main.py --mode train   --approach C   --backbone BAAI/bge-m3
  python esg_main.py --mode train   --approach B   --backbone Qwen/Qwen3-8B --use_lora
  python esg_main.py --mode geneval --backbone Qwen/Qwen3-8B   (Approach E)
  python esg_main.py --mode run_all                                      (all experiments)
  python esg_main.py --mode predict --checkpoint runs/expA/best.pt

Approaches:
  A        : BERT-family encoder → CLS → 4 independent heads
  A1       : BERT encoder → CLS → Cascade head (Task1 logits fed into Task2/3/4)
  C        : Sentence-embedding model (BGE-M3 / Qwen3-Embedding) → frozen → MLP heads
  C_contrastive : Same backbone, supervised contrastive fine-tuning first, then classify
  B        : Causal LLM last-token hidden state → MLP heads (frozen backbone)
  B_lora   : Same with LoRA fine-tuning
  E        : Qwen3 generative with thinking mode → parse JSON output

Tasks:
  T1 promise_status        Yes / No                                           weight 0.20
  T2 evidence_status       Yes / No / N/A                                     weight 0.30
  T3 evidence_quality      Clear / Not Clear / Misleading / N/A               weight 0.35
  T4 verification_timeline already/within_2_years/between_2_and_5_years/
                           longer_than_5_years / N/A                          weight 0.15

Data insight:
  800 train / 200 test | Misleading=1 | within_2_years=11 | Task1=No(149) → T2/3/4=N/A
"""

from __future__ import annotations

import argparse
import ast
import json
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONSTANTS & LABEL MAPS
# ─────────────────────────────────────────────────────────────────────────────

TASK_WEIGHTS = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}

# Backbones that do not accept token_type_ids
_NO_TOKEN_TYPE_IDS: frozenset[str] = frozenset({"modernbert", "deberta-v2", "electra"})

# ── Hand-crafted linguistic features (13-dim) ────────────────────────────────
# Derived from domain rules in the LLM system prompt.
# Index layout:
#   0 has_fuzzy_words      1 has_specific_number  2 has_third_party
#   3 has_completed_action 4 has_contradiction     5 fuzzy_no_specific
#   6 has_already_signal   7 has_within2y_signal   8 has_2to5y_signal
#   9 has_5yplus_signal    10 has_promise_verb     11 text_length_norm
#   12 number_density
HAND_FEATURE_DIM = 13

_HF_FUZZY_WORDS = frozenset({
    "持續推進", "努力改善", "積極探索", "逐步推動", "將致力於", "致力推動",
    "持續關注", "各項環保", "各項措施", "相關措施", "多元管道",
    "積極推動", "持續努力", "逐步落實", "積極改善",
})
_HF_SPECIFIC_PAT   = re.compile(r'\d+\s*%|\d+\s*億|\d+\s*萬|\d+\s*元|\d+\s*公噸|\d+\s*度')
_HF_THIRD_PAT      = re.compile(r'ISO\s*\d+|GRI|第三方|外部稽核|認證機構|驗證單位')
_HF_COMPLETED_PAT  = re.compile(r'已建置|已導入|已達成|已完成|已實施|已通過')
_HF_CONTRADICT_PAT = re.compile(r'但(?!書)|然而|雖然[\s\S]{0,10}?卻|實際上[\s\S]{0,10}?未|尚未達標|未能如期')
_HF_ALREADY_PAT    = re.compile(r'目前|截至|本年度|持續執行中|已於\d{4}')
# Year ranges anchored to base year 2024 (competition data vintage)
# within_2_years: by end of 2026 (2024+2)
# between_2_and_5_years: 2027-2029 (2024+3 to 2024+5)
# longer_than_5_years: 2030+ (beyond 2024+5)
_HF_WITHIN2Y_PAT   = re.compile(r'明年|今年底|2024年?|2025年?|2026年?|2年內|兩年內|近期內|短期內|年底前完成')
_HF_2TO5Y_PAT      = re.compile(r'2027|2028|2029|3至5年|三至五年|中期目標|中長期')
_HF_5YPLUS_PAT     = re.compile(r'2030|2031|2032|2033|2034|2035|長期願景|下一個十年|長遠目標')
_HF_PROMISE_PAT    = re.compile(r'承諾|目標|計畫|將(?!來)|預計|期望達成|致力')


def extract_hand_features(text: str) -> torch.Tensor:
    """Return 13-dim rule-based feature vector for a Chinese ESG text."""
    has_fuzzy     = float(any(w in text for w in _HF_FUZZY_WORDS))
    has_specific  = float(bool(_HF_SPECIFIC_PAT.search(text)))
    has_third     = float(bool(_HF_THIRD_PAT.search(text)))
    has_completed = float(bool(_HF_COMPLETED_PAT.search(text)))
    has_contra    = float(bool(_HF_CONTRADICT_PAT.search(text)))
    fuzzy_no_spec = float(has_fuzzy == 1.0 and has_specific == 0.0)
    has_already   = float(bool(_HF_ALREADY_PAT.search(text)))
    has_within2y  = float(bool(_HF_WITHIN2Y_PAT.search(text)))
    has_2to5y     = float(bool(_HF_2TO5Y_PAT.search(text)))
    has_5yplus    = float(bool(_HF_5YPLUS_PAT.search(text)))
    has_promise   = float(bool(_HF_PROMISE_PAT.search(text)))
    length_norm   = min(len(text) / 500.0, 2.0)
    num_density   = sum(c.isdigit() for c in text) / max(len(text), 1) * 100
    return torch.tensor([
        has_fuzzy, has_specific, has_third, has_completed, has_contra,
        fuzzy_no_spec, has_already, has_within2y, has_2to5y, has_5yplus,
        has_promise, length_norm, num_density,
    ], dtype=torch.float32)


LABELS = {
    "t1": ["Yes", "No"],
    "t2": ["Yes", "No", "N/A"],
    "t3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "t4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"],
}

IDX2LABEL = {task: {i: l for i, l in enumerate(labels)} for task, labels in LABELS.items()}
LABEL2IDX = {task: {l: i for i, l in enumerate(labels)} for task, labels in LABELS.items()}
NUM_LABELS = {task: len(labels) for task, labels in LABELS.items()}

DATA_DIR = Path(__file__).parent / "2026-esg-classification-challenge"

# Prompt template for Approach E (generative)
GEN_SYSTEM_PROMPT = """你是一位ESG永續報告分析專家。請仔細閱讀以下公司ESG報告段落，依序回答四個問題。

判斷標準：
1. promise_status: 該段落是否包含明確的企業永續承諾？(Yes=有承諾/No=無承諾)
2. evidence_status: 承諾是否有具體的行動計畫或佐證？(Yes=有/No=無/N/A=無承諾)
3. evidence_quality: 陳述的語意清晰程度？(Clear=清晰/Not Clear=模糊/Misleading=有誤導嫌疑/N/A=無承諾)
4. verification_timeline: 承諾預計完成的時程？(already=已完成/within_2_years=2年內/between_2_and_5_years=2-5年/longer_than_5_years=5年以上/N/A=無承諾)

重要規則：若 promise_status=No，則其餘三項必須為 N/A。"""

GEN_USER_TEMPLATE = """報告段落：
{text}

請先逐步推理，再輸出如下JSON（必須嚴格符合格式）：
{{"promise_status": "...", "evidence_status": "...", "evidence_quality": "...", "verification_timeline": "..."}}"""


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # data
    data_dir: str = str(DATA_DIR)
    use_augmented: bool = False   # if True, loads train_data_augmented.csv instead of train_data.csv
    aug_filename: str = "train_data_augmented.csv"  # which aug file to use when use_augmented=True
    max_length: int = 512

    # model
    approach: str = "A"
    backbone: str = "hfl/chinese-macbert-large"
    hidden_dim: int = 256
    dropout: float = 0.1

    # LoRA
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # training
    epochs: int = 10
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    seed: int = 42

    # loss
    loss_type: str = "focal"     # ce | focal | label_smooth | ordinal | db
    focal_gamma: float = 2.0
    label_smooth: float = 0.1
    use_class_weight: bool = True
    # DB loss hyperparameters (Wu et al. ECCV 2021)
    db_alpha: float = 0.1        # base weight lift
    db_beta: float = 10.0        # smoothing steepness
    db_mu: float = 0.5           # smoothing centre
    db_kappa: float = 0.05       # NTR scale factor

    # per-task loss: T1=CE, T2=Focal, T3=DB, T4=Ordinal
    per_task_loss: bool = False

    # extended cascade: feed T2 prob into T3 head (T1→T2, [T1,T2]→T3, [T1]→T4)
    deep_cascade: bool = False

    # 2-layer projector intermediate dim (Approach B / D)
    proj_hidden_dim: int = 1024

    # early stopping
    early_stopping_patience: int = 3   # 0 = disabled

    # data augmentation for rare classes (Misleading, within_2_years)
    augment_rare: bool = False
    augment_target_n: int = 15         # target count per rare class after augmentation

    # k-fold cross-validation (1 = single split)
    kfold: int = 1

    # contrastive (Approach C_contrastive)
    contrastive_epochs: int = 3
    contrastive_lr: float = 1e-5
    contrastive_temp: float = 0.07

    # generative (Approach E)
    gen_max_new_tokens: int = 512
    gen_temperature: float = 0.1
    enable_thinking: bool = True

    # FGM adversarial training (0.0 = disabled)
    fgm_epsilon: float = 0.0

    # LogSigma automatic task weighting (Kendall et al. 2018)
    # Learns log(σ²_t) per task; loss = Σ_t [ 0.5*exp(-s_t)*L_t + 0.5*s_t ]
    use_logsigma: bool = False

    # Adaptive task-weighting (HCAL-style)
    # Each epoch: w_t ∝ (1 - f1_t), blended with fixed TASK_WEIGHTS
    # alpha=0 → fixed weights, alpha=1 → fully adaptive
    adaptive_task_weight: bool = False
    adaptive_alpha: float = 0.5   # blend ratio

    # hand-crafted linguistic features (13-dim rule-based signals)
    use_hand_features: bool = False

    # post-processing
    apply_na_rule: bool = True

    # evidence span auxiliary loss (uses evidence_string from training data)
    use_span: bool = False
    span_weight: float = 0.1       # weight for span BCE loss

    # supervised contrastive loss on T3 embeddings (in-batch, dropout augmentation)
    scl_weight: float = 0.0        # 0 = disabled; try 0.05–0.15
    scl_temp: float = 0.07         # temperature for SCL

    # Class-Aware Prototype Contrastive Loss for T3 (arXiv 2410.22197)
    # Contrasts each sample against in-batch class centroids (better for imbalanced T3)
    proto_weight: float = 0.0      # 0 = disabled; try 0.05–0.20
    proto_temp: float = 0.07       # temperature for prototype contrastive loss

    # SharpReCL: prototype-guided contrastive rebalancing (arxiv 2405.11524)
    sharp_recl_weight: float = 0.0  # 0 = disabled; try 0.05–0.15
    sharp_recl_mixup: float = 0.3   # Beta param for hard-neg Mixup

    # MR2: adaptive per-class margin regularization (arxiv 2602.00205)
    mr2_weight: float = 0.0         # 0 = disabled; try 0.03–0.10
    mr2_margin_scale: float = 0.3   # margin = scale * class_spread

    # inference: dynamic per-sample T3 kNN alpha based on model confidence
    dynamic_alpha: bool = False

    # masked gradient: zero T2/T3/T4 loss for samples where T1=No
    # (T1=No → T2/T3/T4 are definitionally N/A; training on them may add noise to T3 head)
    mask_t1_no: bool = False

    # T3 Not Clear boost: override T3 loss to CE + class weights, then multiply Not Clear weight
    # Addresses Clear >> Not Clear imbalance (model underpredicts Not Clear on test)
    t3_nc_weight: float = 1.0   # >1.0 → boost Not Clear class weight in T3 loss
    t3_loss_type: str = "ce"    # "ce" or "focal" — loss used when t3_nc_weight > 1.0
    t3_cond_weight: float = 0.0  # >0 → aux conditional chain-rule loss: T3 content CE on T2=Yes rows only
    ce_label_smoothing: float = 0.0  # PyTorch CE label_smoothing (0.0=off, 0.1=typical)

    # R-Drop regularization (Wu et al., NeurIPS 2021)
    # Same input forward twice with different dropout, add KL(p1, p2) to loss
    rdrop_alpha: float = 0.0       # 0 = disabled; try 0.5–5.0

    # SWA: average last K checkpoints' weights for flatter minima
    swa_start_epoch: int = 0       # 0 = disabled; e.g. 7 means SWA from epoch 7

    # Self-distillation: use ensemble OOF soft labels as training targets
    distill_alpha: float = 0.0     # 0 = disabled; 0.3–0.7 typical
    distill_temperature: float = 3.0
    distill_oof_dir: str = ""      # path to kfold dir whose OOF probs are teacher

    # TMix: embedding-level mixup (Chen et al., ACL 2020)
    # Mixes token embeddings of two training samples; blends labels proportionally.
    # Beta(α,α) controls mixing strength: α=0.2 → mostly one sample; α=1.0 → uniform.
    tmix_alpha: float = 0.0    # 0 = disabled; try 0.2, 0.4, 1.0

    # Hard-Mixup: cross-class NC↔Clear mixing for T3 decision boundary (WWW 2026)
    # Unlike TMix (random pairs), specifically mixes Not Clear with Clear samples.
    # λ ~ Beta(α,α), clamped to ≥0.5 so NC sample always dominates the mixed example.
    # Added on top of regular loss; compatible with FGM/R-Drop but not TMix.
    hard_mixup_alpha: float = 0.0  # 0 = disabled; try 0.4, 1.0

    # Company embedding: learnable per-company bias injected into CLS (0 = disabled)
    # Teacher's LGBM showed +0.03 with company_id + industry; inject directly into neural model
    company_emb_dim: int = 0   # try 32, 64
    pooling: str = "cls"       # "cls", "attn", "mean" (SemEval-2025: attn best)
    prepend_linguistic: bool = False  # prepend linguistic markers to input text

    # Remap Misleading → Not Clear at inference (for models trained on mislead_as_nc data)
    remap_misleading: bool = False

    # few-shot demonstration: prepend same-company retrieved examples to input
    use_fewshot_demo: bool = False
    fewshot_ref_ckpt: str = ""  # path to existing checkpoint used to compute retrieval embeddings

    # output
    run_dir: str = "runs/exp"
    checkpoint: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 2.  UTILS
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_label(raw: str) -> tuple[str, str, str, str]:
    parsed = ast.literal_eval(raw)
    return tuple(parsed)


def apply_na_rule(preds: dict[str, list[str]],
                  remap_misleading: bool = False) -> dict[str, list[str]]:
    """T1=No → T2/T3/T4=N/A.  T2=No → T3=N/A (100% rule in training data).
    remap_misleading: Misleading → Not Clear (for models trained without Misleading label).
    """
    result = {k: list(v) for k, v in preds.items()}
    for i, t1 in enumerate(result["t1"]):
        if t1 == "No":
            result["t2"][i] = "N/A"
            result["t3"][i] = "N/A"
            result["t4"][i] = "N/A"
        elif result["t2"][i] == "No":
            result["t3"][i] = "N/A"
        if remap_misleading and result["t3"][i] == "Misleading":
            result["t3"][i] = "Not Clear"
    return result


def preds_to_label_strings(preds: dict[str, list[str]]) -> list[str]:
    n = len(preds["t1"])
    return [
        str([preds["t1"][i], preds["t2"][i], preds["t3"][i], preds["t4"][i]])
        for i in range(n)
    ]


def compute_weighted_f1(
    y_true: dict[str, list[str]],
    y_pred: dict[str, list[str]],
) -> dict[str, float]:
    scores = {}
    for task in ["t1", "t2", "t3", "t4"]:
        labels = LABELS[task]
        f1 = f1_score(y_true[task], y_pred[task], labels=labels, average="macro", zero_division=0)
        scores[task] = f1
    scores["weighted"] = sum(TASK_WEIGHTS[t] * scores[t] for t in ["t1", "t2", "t3", "t4"])
    return scores


def compute_class_weights(labels_list: list[str], task: str) -> torch.Tensor:
    from collections import Counter
    counts = Counter(labels_list)
    total = len(labels_list)
    num_cls = NUM_LABELS[task]
    weights = torch.ones(num_cls)
    for label, idx in LABEL2IDX[task].items():
        if label in counts:
            weights[idx] = total / (num_cls * counts[label])
    return weights


def parse_gen_output(text: str) -> dict[str, str]:
    """Extract JSON from generative model output. Returns default on failure."""
    default = {"promise_status": "Yes", "evidence_status": "Yes",
               "evidence_quality": "Clear", "verification_timeline": "already"}
    # Strip thinking block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find JSON block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not match:
        return default
    try:
        obj = json.loads(match.group())
        # Validate each key
        valid = {
            "promise_status": LABELS["t1"],
            "evidence_status": LABELS["t2"],
            "evidence_quality": LABELS["t3"],
            "verification_timeline": LABELS["t4"],
        }
        for k, allowed in valid.items():
            if obj.get(k) not in allowed:
                obj[k] = default[k]
        return obj
    except (json.JSONDecodeError, KeyError):
        return default


def ensemble_preds(
    pred_list: list[dict[str, list[str]]],
    weights: Optional[list[float]] = None,
) -> dict[str, list[str]]:
    """Majority vote ensemble across multiple prediction dicts."""
    if weights is None:
        weights = [1.0] * len(pred_list)
    n = len(pred_list[0]["t1"])
    result = {t: [] for t in LABELS}
    for i in range(n):
        for task in LABELS:
            votes: dict[str, float] = {}
            for pred, w in zip(pred_list, weights):
                label = pred[task][i]
                votes[label] = votes.get(label, 0.0) + w
            result[task].append(max(votes, key=votes.__getitem__))
    return result


@torch.no_grad()
def extract_cls_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract [CLS] token embeddings from the encoder (no head). Returns (N, hidden) array."""
    model.eval()
    all_embs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and model.encoder.config.model_type not in _NO_TOKEN_TYPE_IDS:
            kwargs["token_type_ids"] = token_type_ids
        out = model.encoder(**kwargs)
        cls = out.last_hidden_state[:, 0].cpu().float().numpy()
        all_embs.append(cls)
    return np.concatenate(all_embs, axis=0)


def compute_span_mask(
    text: str,
    evidence_string: str | None,
    tokenizer,
    max_length: int,
) -> list[float]:
    """Return a binary float mask [max_length] marking evidence_string token positions.

    Uses offset_mapping from HuggingFace fast tokenizers (RoBERTa, LERT).
    Falls back to all-zeros if evidence_string is not found or offset_mapping unavailable.
    """
    if not evidence_string or not isinstance(evidence_string, str):
        return [0.0] * max_length
    start_char = text.find(evidence_string)
    if start_char < 0:
        return [0.0] * max_length
    end_char = start_char + len(evidence_string)
    try:
        enc = tokenizer(
            text, max_length=max_length, truncation=True,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
    except Exception:
        print(f"  [WARN] compute_span_mask: fallback to zeros (fast tokenizer unavailable?)")
        return [0.0] * max_length
    mask = []
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:            # special / padding token
            mask.append(0.0)
        elif tok_start < end_char and tok_end > start_char:   # overlaps span
            mask.append(1.0)
        else:
            mask.append(0.0)
    # Pad to max_length
    mask = mask[:max_length]
    mask += [0.0] * (max_length - len(mask))
    return mask


def compute_knn_ldl_probs(
    train_embs: np.ndarray,
    test_embs: np.ndarray,
    train_df: pd.DataFrame,
    k: int = 5,
    sim_temp: float = 1.0,
    test_df: pd.DataFrame | None = None,
    company_knn: bool = False,
) -> dict[str, list[list[float]]]:
    """Compute kNN-LDL pseudo-probabilities for each task.

    LDL (Label Distribution Learning): neighbor contribution is weighted by
    that neighbor's model confidence (max softmax), not uniform count.
    Since we don't store train confidences here, we use cosine similarity
    as a proxy for confidence (closer neighbor → higher weight).

    sim_temp: softmax temperature over similarities.
        < 1.0 → sharper (more weight on nearest neighbor)
        > 1.0 → flatter (closer to uniform vote)
        1.0   → default behaviour

    company_knn: if True and test_df provided, restrict kNN pool to same company.
        Falls back to global kNN when same-company pool has fewer than k samples.

    Returns: {task: [[p0, p1, ...], ...]} for all test samples.
    """
    # Normalize for cosine similarity
    train_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    test_norm  = test_embs  / (np.linalg.norm(test_embs,  axis=1, keepdims=True) + 1e-8)
    sim = test_norm @ train_norm.T  # (n_test, n_train)

    assert len(test_embs) == sim.shape[0] and len(train_embs) == sim.shape[1]
    if test_df is not None:
        assert len(test_df) == len(test_embs), \
            f"test_df rows ({len(test_df)}) != te_embs rows ({len(test_embs)})"

    # Pre-build company index for company_knn mode
    use_company_knn = (company_knn and test_df is not None
                       and 'company' in train_df.columns and 'company' in test_df.columns)
    if use_company_knn:
        train_companies = train_df['company'].values
        test_companies  = test_df['company'].values

    task_col = {"t1": "promise_status", "t2": "evidence_status",
                "t3": "evidence_quality", "t4": "verification_timeline"}

    result: dict[str, list[list[float]]] = {t: [] for t in LABELS}
    for i in range(len(test_embs)):
        if use_company_knn:
            company = test_companies[i]
            company_mask = (train_companies == company)
            n_company = int(company_mask.sum())
            if n_company >= k:
                company_idx = np.where(company_mask)[0]
                top_local = np.argsort(sim[i][company_idx])[::-1][:k]
                top_k_idx = company_idx[top_local]
            else:
                top_k_idx = np.argsort(sim[i])[::-1][:k]
        else:
            top_k_idx = np.argsort(sim[i])[::-1][:k]
        top_k_sim = sim[i][top_k_idx].astype(float)
        # Softmax over similarities as weights (temperature-scaled)
        exp_sim = np.exp((top_k_sim - top_k_sim.max()) / sim_temp)
        weights = exp_sim / exp_sim.sum()

        for task, col in task_col.items():
            nc = NUM_LABELS[task]
            pseudo = [0.0] * nc
            for rank, tidx in enumerate(top_k_idx):
                label = train_df.iloc[tidx][col]
                cidx = LABEL2IDX[task].get(label, -1)
                if cidx >= 0:
                    pseudo[cidx] += float(weights[rank])
            # Normalize (should already sum to ~1, but guard for missing labels)
            total = sum(pseudo)
            if total > 0:
                pseudo = [v / total for v in pseudo]
            else:
                pseudo = [1.0 / nc] * nc
            result[task].append(pseudo)

    return result


def knn_fuse_probs(
    model_probs: dict[str, list[list[float]]],
    knn_probs: dict[str, list[list[float]]],
    alpha: dict[str, float],
    t3_nc_threshold: float = 0.0,
    dynamic_alpha: bool = False,
    t3_la_tau: float = 0.0,
    t3_class_freq: list[float] | None = None,
) -> dict[str, list[str]]:
    """Fuse model softmax with kNN-LDL pseudo-probs using task-specific alpha.

    final_prob[task][i] = (1 - alpha_i[task]) * model_prob + alpha_i[task] * knn_prob

    dynamic_alpha: if True, T3 alpha is per-sample and inversely proportional to model
    confidence (max softmax). Low-confidence samples lean more on kNN.
        alpha_t3(i) = base_lo + (base_hi - base_lo) * (1 - max_softmax_t3(i))
    where base_lo=0.3 (high confidence) and base_hi=0.8 (low confidence).

    t3_nc_threshold: if > 0, predict "Not Clear" for T3 when fused p(Not Clear) >= tau.

    t3_la_tau: if > 0, apply post-hoc logit adjustment to T3 fused probs.
        adjusted_logit[c] = log(fused[c]) - tau * log(max(class_freq[c], min_freq=0.05))
        Boosts rare classes (Not Clear, Misleading) relative to frequent ones (Clear, N/A).
        Misleading is capped at min_freq=0.05 to prevent explosion (only 2 train samples).
    """
    import math
    not_clear_idx = LABEL2IDX["t3"]["Not Clear"]
    result = {t: [] for t in LABELS}
    n = len(model_probs["t1"])
    _LA_MIN_FREQ = 0.05  # floor for Misleading (0.1% train) to prevent explosion
    _LA_EPS = 1e-8
    for i in range(n):
        for task in LABELS:
            if task == "t3" and dynamic_alpha:
                confidence = max(model_probs["t3"][i])
                a = 0.3 + 0.5 * (1.0 - confidence)   # [0.3, 0.8]
            else:
                a = alpha.get(task, 0.0)
            mp = model_probs[task][i]
            kp = knn_probs[task][i]
            fused = [(1 - a) * mp[c] + a * kp[c] for c in range(NUM_LABELS[task])]
            # Post-hoc logit adjustment (GALA / Logit Adjustment, CVPR 2024)
            if task == "t3" and t3_la_tau > 0.0 and t3_class_freq is not None:
                log_f = [math.log(max(t3_class_freq[c], _LA_MIN_FREQ))
                         for c in range(len(fused))]
                adj = [math.log(fused[c] + _LA_EPS) - t3_la_tau * log_f[c]
                       for c in range(len(fused))]
                max_adj = max(adj)
                exp_adj = [math.exp(v - max_adj) for v in adj]
                s = sum(exp_adj)
                fused = [e / s for e in exp_adj]
            if task == "t3" and t3_nc_threshold > 0:
                if fused[not_clear_idx] >= t3_nc_threshold:
                    result[task].append(IDX2LABEL["t3"][not_clear_idx])
                    continue
            result[task].append(IDX2LABEL[task][int(np.argmax(fused))])
    return result


def extract_oof_probs(
    kfold_dir: str,
    train_full: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-create the exact kfold splits and run inference on each fold's val set.

    Returns:
        oof_probs: (N, sum_of_nc_for_t2_and_t3) float array — T2(3) + T3(4) = 7 dims
        oof_order: (N,) int array — original row indices in train_full
    """
    from sklearn.model_selection import StratifiedKFold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat_col = train_full["promise_status"]

    all_probs: list[np.ndarray] = [None] * len(train_full)  # type: ignore[list-item]
    all_order: list[int] = []

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_full, strat_col)):
        ckpt_path = Path(kfold_dir) / f"fold{fold_idx + 1}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fold checkpoint not found: {ckpt_path}")

        fold_val = train_full.iloc[val_idx].reset_index(drop=True)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        saved_cfg: Config = ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        val_ds = ESGDataset(fold_val, tokenizer, saved_cfg.max_length, has_labels=False)
        loader = DataLoader(val_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
        model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
        model.load_state_dict(ckpt["model"])

        probs = predict_probs(model, loader, device)  # {task: [[p0,p1,...], ...]}
        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Store T2 + T3 probs for each val sample at its original position
        for local_i, orig_i in enumerate(val_idx):
            feat = probs["t2"][local_i] + probs["t3"][local_i]  # 3 + 4 = 7 floats
            all_probs[orig_i] = np.array(feat, dtype=np.float32)

        print(f"  [OOF] fold{fold_idx + 1}/{n_splits} done ({len(val_idx)} samples)")

    oof_matrix = np.stack(all_probs, axis=0)   # (800, 7)
    return oof_matrix, np.arange(len(train_full))


def extract_oof_teacher_probs(
    kfold_dir: str,
    train_full: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, list[list[float]]]:
    """Extract OOF teacher soft labels for all 4 tasks from a kfold directory.

    Returns: {task: [[p0, p1, ...], ...]} — per-sample probability distributions.
    Each sample's probs come from the fold where it was in the validation set.
    """
    from sklearn.model_selection import StratifiedKFold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat_col = train_full["promise_status"]

    all_probs: dict[str, list] = {t: [None] * len(train_full) for t in LABELS}

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_full, strat_col)):
        ckpt_path = Path(kfold_dir) / f"fold{fold_idx + 1}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fold checkpoint not found: {ckpt_path}")

        fold_val = train_full.iloc[val_idx].reset_index(drop=True)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        saved_cfg: Config = ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        _ced = getattr(saved_cfg, "company_emb_dim", 0)
        _pool = getattr(saved_cfg, "pooling", "cls")
        _uhf = getattr(saved_cfg, "use_hand_features", False) and saved_cfg.approach == "A1"
        _usp = getattr(saved_cfg, "use_span", False) and saved_cfg.approach == "A1"
        # Build company2idx if needed
        _n_co = 51
        _c2i = None
        if _ced > 0 and "company" in train_full.columns:
            companies = sorted(train_full["company"].dropna().unique().tolist())
            _c2i = {c: i for i, c in enumerate(companies)}
            _n_co = len(_c2i)
        _prepend = getattr(saved_cfg, "prepend_linguistic", False)
        val_ds = ESGDataset(fold_val, tokenizer, saved_cfg.max_length, has_labels=False,
                            use_hand_features=_uhf, company2idx=_c2i,
                            prepend_linguistic=_prepend)
        loader = DataLoader(val_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
        model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc,
                           use_hand_features=_uhf, use_span=_usp,
                           company_emb_dim=_ced, n_companies=_n_co,
                           pooling=_pool).to(device)
        model.load_state_dict(ckpt["model"])

        probs = predict_probs(model, loader, device)
        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for local_i, orig_i in enumerate(val_idx):
            for t in LABELS:
                all_probs[t][orig_i] = probs[t][local_i]

        print(f"  [teacher] fold{fold_idx + 1}/{n_splits} done")

    return all_probs


def search_oof_thresholds(
    kfold_dir: str,
    train_full: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    tasks: tuple[str, ...] = ("t3",),
    steps: int = 20,
) -> dict[str, list[float]]:
    """Grid-search per-class thresholds on OOF predictions to maximize weighted F1.

    Strategy: for each task, independently sweep each class threshold (one-vs-rest):
    predict class c if prob[c] > thr[c], else fall back to argmax of remaining.
    Optimizes the competition weighted F1 on OOF val set.

    Returns: {task: [thr_class0, thr_class1, ...]}
    """
    from sklearn.metrics import f1_score as sk_f1
    from sklearn.model_selection import StratifiedKFold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat_col = train_full["promise_status"]

    TASK_WEIGHTS = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}

    # Collect OOF probs + true labels for each requested task
    all_task_probs: dict[str, list] = {t: [None] * len(train_full) for t in tasks}
    all_task_labels: dict[str, list] = {t: [None] * len(train_full) for t in tasks}

    label_cols = {"t1": "promise_status", "t2": "evidence_status",
                  "t3": "evidence_quality", "t4": "verification_timeline"}

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_full, strat_col)):
        ckpt_path = Path(kfold_dir) / f"fold{fold_idx + 1}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fold checkpoint not found: {ckpt_path}")

        fold_val = train_full.iloc[val_idx].reset_index(drop=True)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        saved_cfg: Config = ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        val_ds = ESGDataset(fold_val, tokenizer, saved_cfg.max_length, has_labels=False)
        loader = DataLoader(val_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
        model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
        model.load_state_dict(ckpt["model"])
        probs = predict_probs(model, loader, device)
        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for t in tasks:
            for local_i, orig_i in enumerate(val_idx):
                all_task_probs[t][orig_i] = probs[t][local_i]
                all_task_labels[t][orig_i] = LABEL2IDX[t][train_full.iloc[orig_i][label_cols[t]]]
        print(f"  [OOF thr] fold{fold_idx + 1}/{n_splits} done")

    best_thresholds: dict[str, list[float]] = {}
    for t in tasks:
        probs_arr = np.array(all_task_probs[t])   # (N, n_classes)
        labels_arr = np.array(all_task_labels[t])  # (N,)
        n_classes = NUM_LABELS[t]
        # Baseline: argmax
        baseline_preds = np.argmax(probs_arr, axis=1)
        baseline_f1 = sk_f1(labels_arr, baseline_preds, average="macro")
        print(f"  [OOF thr] {t} baseline macro-F1 = {baseline_f1:.4f}")

        best_thrs = [0.0] * n_classes  # 0 = use argmax (no override)
        best_f1 = baseline_f1

        # Sweep each class threshold independently (one-at-a-time greedy)
        for cls_idx in range(n_classes):
            cls_best_thr = best_thrs[cls_idx]
            cls_best_f1 = best_f1
            for thr in np.linspace(0.1, 0.9, steps):
                thrs = list(best_thrs)
                thrs[cls_idx] = float(thr)
                # Apply thresholds: if any class prob > thr, predict it (highest prob wins among ties)
                preds = []
                for row in probs_arr:
                    triggered = [(row[c], c) for c, thr_c in enumerate(thrs) if thr_c > 0 and row[c] >= thr_c]
                    if triggered:
                        preds.append(max(triggered)[1])
                    else:
                        preds.append(int(np.argmax(row)))
                f1 = sk_f1(labels_arr, preds, average="macro")
                if f1 > cls_best_f1:
                    cls_best_f1 = f1
                    cls_best_thr = float(thr)
            best_thrs[cls_idx] = cls_best_thr
            best_f1 = cls_best_f1
            label_name = LABELS[t][cls_idx]
            print(f"    class {cls_idx} ({label_name}): thr={cls_best_thr:.2f} → F1={cls_best_f1:.4f}")

        best_thresholds[t] = best_thrs
        print(f"  [OOF thr] {t} final thresholds: {best_thrs} → macro-F1={best_f1:.4f}")

    return best_thresholds


def apply_thresholds(
    probs: dict[str, list[list[float]]],
    thresholds: dict[str, list[float]],
) -> dict[str, list[str]]:
    """Convert probs to label strings using per-class thresholds."""
    result: dict[str, list[str]] = {}
    for task in LABELS:
        thrs = thresholds.get(task, [0.0] * NUM_LABELS[task])
        preds = []
        for row in probs[task]:
            triggered = [(row[c], c) for c, thr_c in enumerate(thrs) if thr_c > 0 and row[c] >= thr_c]
            if triggered:
                preds.append(IDX2LABEL[task][max(triggered)[1]])
            else:
                preds.append(IDX2LABEL[task][int(np.argmax(row))])
        result[task] = preds
    return result


def tune_t3_nc_bias(
    kfold_dir: str,
    train_full: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    delta_range: tuple[float, float] = (-0.5, 2.0),
    steps: int = 25,
) -> float:
    """Grid-search T3 NC logit bias δ on OOF predictions.

    Applies exp(δ) scaling to the Not Clear probability before renormalization,
    equivalent to adding δ to the NC logit. Returns the best δ.
    """
    import math
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score as sk_f1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    strat_col = train_full["promise_status"]
    label_col = "evidence_quality"

    all_t3_probs: list = [None] * len(train_full)
    all_t3_labels: list = [None] * len(train_full)

    for fold_idx, (_, val_idx) in enumerate(skf.split(train_full, strat_col)):
        ckpt_path = Path(kfold_dir) / f"fold{fold_idx + 1}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fold checkpoint not found: {ckpt_path}")

        fold_val = train_full.iloc[val_idx].reset_index(drop=True)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        saved_cfg: Config = ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        val_ds = ESGDataset(fold_val, tokenizer, saved_cfg.max_length, has_labels=False)
        loader = DataLoader(val_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
        model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
        model.load_state_dict(ckpt["model"])
        probs = predict_probs(model, loader, device)
        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for local_i, orig_i in enumerate(val_idx):
            all_t3_probs[orig_i] = probs["t3"][local_i]
            lbl = train_full.iloc[orig_i][label_col]
            all_t3_labels[orig_i] = LABEL2IDX["t3"].get(lbl, -1)
        print(f"  [bias] fold{fold_idx + 1}/{n_splits} done")

    valid = [(p, l) for p, l in zip(all_t3_probs, all_t3_labels)
             if p is not None and l >= 0]
    probs_arr = np.array([p for p, _ in valid])
    labels_arr = np.array([l for _, l in valid])

    nc_idx = LABEL2IDX["t3"]["Not Clear"]
    baseline_f1 = sk_f1(labels_arr, np.argmax(probs_arr, axis=1), average="macro")
    print(f"\n  Baseline T3 macro-F1 (δ=0): {baseline_f1:.4f}")

    best_delta = 0.0
    best_f1 = baseline_f1
    for delta in np.linspace(delta_range[0], delta_range[1], steps):
        adj = probs_arr.copy()
        adj[:, nc_idx] *= math.exp(delta)
        adj = adj / adj.sum(axis=1, keepdims=True)
        f1 = sk_f1(labels_arr, np.argmax(adj, axis=1), average="macro")
        marker = " ←" if f1 > best_f1 else ""
        print(f"    δ={delta:+.2f} → F1={f1:.4f}{marker}")
        if f1 > best_f1:
            best_f1 = f1
            best_delta = round(float(delta), 2)

    print(f"\n  Best δ = {best_delta:+.2f} → T3 macro-F1 = {best_f1:.4f} "
          f"(gain = {best_f1 - baseline_f1:+.4f})")
    return best_delta


def build_stacking_lr(
    kfold_dirs: dict[str, str],
    train_full: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> object:
    """Train a Logistic Regression meta-learner per task using OOF probs.

    Features: T2+T3 OOF probs from each model (7 dims each).
    Returns a dict of fitted LR per task: {"t2": lr2, "t3": lr3}.
    """
    from sklearn.linear_model import LogisticRegression

    # Collect OOF from each backbone
    oof_feats_list = []
    for name, kdir in kfold_dirs.items():
        if not Path(kdir).exists():
            continue
        print(f"  [Stacking] extracting OOF from {name} ...")
        oof, _ = extract_oof_probs(kdir, train_full, n_splits, seed)
        oof_feats_list.append(oof)

    if not oof_feats_list:
        raise RuntimeError("No kfold dirs found for stacking")

    X = np.concatenate(oof_feats_list, axis=1)  # (800, 7 * n_models)

    task_col = {"t2": "evidence_status", "t3": "evidence_quality"}
    meta_models = {}
    for task, col in task_col.items():
        y = np.array([LABEL2IDX[task][v] for v in train_full[col]])
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        lr.fit(X, y)
        train_acc = lr.score(X, y)
        print(f"  [Stacking] {task} LR train acc={train_acc:.3f} (classes={lr.classes_})")
        meta_models[task] = lr

    return meta_models


def stacking_predict(
    meta_models: dict,
    model_probs_list: list[dict[str, list[list[float]]]],
    base_preds: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Apply stacking meta-learner to override T2 and T3 predictions.

    base_preds: already-computed ensemble predictions (T1/T4 kept as-is).
    model_probs_list: list of probs dicts from each backbone (same order as OOF).
    """
    # mp["t2"] is list of lists → shape (N, nc)
    n = len(model_probs_list[0]["t2"])
    X_parts = []
    for mp in model_probs_list:
        arr = np.array(mp["t2"], dtype=np.float32)   # (N, 3)
        arr2 = np.array(mp["t3"], dtype=np.float32)  # (N, 4)
        X_parts.append(np.concatenate([arr, arr2], axis=1))  # (N, 7)
    X_test = np.concatenate(X_parts, axis=1)  # (N, 7*n_models)

    result = {t: list(base_preds[t]) for t in LABELS}
    for task, lr in meta_models.items():
        pred_idx = lr.predict(X_test)
        result[task] = [IDX2LABEL[task][int(i)] for i in pred_idx]

    return result


def ensemble_preds_soft(
    prob_list: list[dict[str, list[list[float]]]],
    weights: Optional[list[float]] = None,
) -> dict[str, list[str]]:
    """Soft voting: average per-class probabilities, then argmax.
    Almost always outperforms hard majority vote.

    Args:
        prob_list: list of {task: [[p0,p1,...], ...]} from predict_probs()
        weights:   optional per-model scalar weights (default: uniform)
    """
    if weights is None:
        weights = [1.0] * len(prob_list)
    total_w = sum(weights)
    n = len(prob_list[0]["t1"])
    result = {t: [] for t in LABELS}
    for i in range(n):
        for task in LABELS:
            num_classes = NUM_LABELS[task]
            avg = [0.0] * num_classes
            for probs, w in zip(prob_list, weights):
                for c in range(num_classes):
                    avg[c] += w * probs[task][i][c]
            avg = [v / total_w for v in avg]
            result[task].append(IDX2LABEL[task][int(np.argmax(avg))])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ESGDataset(Dataset):
    """For encoder-based approaches (A, A1, B)."""

    @staticmethod
    def _prepend_markers(text: str) -> str:
        """Prepend linguistic markers to text (SemEval-2025 Model 2 approach)."""
        feats = extract_hand_features(text)
        markers = []
        if feats[0] > 0: markers.append("含模糊詞")
        if feats[1] > 0: markers.append("含具體數字")
        if feats[2] > 0: markers.append("含第三方驗證")
        if feats[3] > 0: markers.append("含完成動作")
        if feats[4] > 0: markers.append("含轉折語氣")
        prefix = "；".join(markers) if markers else "無特殊標記"
        return f"[{prefix}]{text}"

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512,
                 has_labels: bool = True, use_hand_features: bool = False,
                 use_span: bool = False,
                 demo_prefixes: list | None = None,
                 company2idx: dict | None = None,
                 prepend_linguistic: bool = False) -> None:
        self.texts = df["data"].tolist()
        if prepend_linguistic:
            self.texts = [self._prepend_markers(t) for t in self.texts]
        self.demo_prefixes = demo_prefixes
        self.ids = df["id"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels
        self.use_hand_features = use_hand_features
        if use_hand_features:
            self.hand_features = [extract_hand_features(t) for t in self.texts]
        if has_labels:
            self.t1 = [LABEL2IDX["t1"][v] for v in df["promise_status"]]
            self.t2 = [LABEL2IDX["t2"][v] for v in df["evidence_status"]]
            self.t3 = [LABEL2IDX["t3"][v] for v in df["evidence_quality"]]
            self.t4 = [LABEL2IDX["t4"][v] for v in df["verification_timeline"]]
        # Company embedding IDs (None = disabled)
        self.company_ids: list[int] | None = None
        if company2idx is not None and "company" in df.columns:
            unk_idx = len(company2idx)  # unknown companies → padding_idx
            self.company_ids = [company2idx.get(c, unk_idx) for c in df["company"].tolist()]
        # Teacher soft labels for self-distillation (set externally)
        self.teacher_probs: dict[str, list[list[float]]] | None = None
        # Evidence span masks — only available in training data
        self.span_masks: list[list[float]] | None = None
        if use_span and has_labels and "evidence_string" in df.columns:
            ev_strs = df["evidence_string"].tolist()
            self.span_masks = [
                compute_span_mask(text, ev, tokenizer, max_length)
                for text, ev in zip(self.texts, ev_strs)
            ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        if self.demo_prefixes is not None and self.demo_prefixes[idx]:
            text = text + self.demo_prefixes[idx]  # suffix: query first, demos after
        enc = self.tokenizer(
            text, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sample_id": self.ids[idx],
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        if self.use_hand_features:
            item["hand_features"] = self.hand_features[idx]
        if self.company_ids is not None:
            item["company_id"] = torch.tensor(self.company_ids[idx], dtype=torch.long)
        if self.has_labels:
            item["t1"] = torch.tensor(self.t1[idx], dtype=torch.long)
            item["t2"] = torch.tensor(self.t2[idx], dtype=torch.long)
            item["t3"] = torch.tensor(self.t3[idx], dtype=torch.long)
            item["t4"] = torch.tensor(self.t4[idx], dtype=torch.long)
        if self.span_masks is not None:
            item["span_mask"] = torch.tensor(self.span_masks[idx], dtype=torch.float)
        if self.teacher_probs is not None:
            for t in ["t1", "t2", "t3", "t4"]:
                item[f"teacher_{t}"] = torch.tensor(self.teacher_probs[t][idx], dtype=torch.float)
        return item


class DemoRetriever:
    """Retrieve top-k same-company training examples as labeled few-shot demonstrations.

    For each query sample, finds the most similar same-company examples from the
    reference pool using cosine similarity on pre-computed CLS embeddings.
    """

    def __init__(self, ref_embs: np.ndarray, ref_df: pd.DataFrame, n_demos: int = 2):
        self.embs = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)
        self.df = ref_df.reset_index(drop=True)
        self.companies = self.df["company"].values
        self.ids = self.df["id"].values if "id" in self.df.columns else None
        self.n_demos = n_demos

    def build_prefixes(self, query_df: pd.DataFrame, exclude_self: bool = True) -> list[str]:
        """Build demo prefix strings for each row of query_df.

        exclude_self: if True, exclude each query sample from its own pool (matched by id).
        Returns one prefix string per row; empty string when no same-company pool exists.
        """
        prefixes = []
        for _, row in query_df.iterrows():
            company = row.get("company", "")
            qid = row.get("id", None)

            same_co = self.companies == company
            if exclude_self and self.ids is not None and qid is not None:
                same_co = same_co & (self.ids != qid)

            cand_idx = np.where(same_co)[0]
            if len(cand_idx) == 0:
                prefixes.append("")
                continue

            # Query embedding: use ref embedding if sample is in ref_df, else pool mean
            if self.ids is not None and qid is not None:
                match = np.where(self.ids == qid)[0]
                q_emb = self.embs[match[0]] if len(match) > 0 else self.embs[cand_idx].mean(0)
            else:
                q_emb = self.embs[cand_idx].mean(0)
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

            sims = self.embs[cand_idx] @ q_emb
            top_k = min(self.n_demos, len(cand_idx))
            top_idx = cand_idx[np.argsort(sims)[::-1][:top_k]]

            parts = []
            for tidx in top_idx:
                r = self.df.iloc[tidx]
                label = (f"{r['promise_status']} {r['evidence_status']} "
                         f"{r['evidence_quality']} {r['verification_timeline']}")
                demo_text = r['data'][:80]  # cap to ~80 Chinese tokens to stay within budget
                parts.append(f"[參考] {demo_text} [標注] {label}")
            # Suffix format: query goes FIRST so truncation cuts demos, not query
            prefixes.append("\n" + "\n".join(parts))

        return prefixes


def _load_retrieval_embs(ref_ckpt: str, df: pd.DataFrame,
                         device: torch.device) -> np.ndarray:
    """Load a saved checkpoint and extract CLS embeddings for df rows."""
    ckpt = torch.load(ref_ckpt, map_location=device, weights_only=False)
    saved_cfg: Config = ckpt["cfg"]
    dc = getattr(saved_cfg, "deep_cascade", False)
    tok = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ds = ESGDataset(df, tok, saved_cfg.max_length, has_labels=False)
    loader = DataLoader(ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
    model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
    model.load_state_dict(ckpt["model"])
    embs = extract_cls_embeddings(model, loader, device)
    del model
    _free_vram()
    return embs


class EmbeddingDataset(Dataset):
    """Pre-computed embeddings + labels for Approach C."""

    def __init__(self, embeddings: torch.Tensor, df: pd.DataFrame,
                 has_labels: bool = True) -> None:
        self.embeddings = embeddings
        self.ids = df["id"].tolist()
        self.has_labels = has_labels
        if has_labels:
            self.t1 = [LABEL2IDX["t1"][v] for v in df["promise_status"]]
            self.t2 = [LABEL2IDX["t2"][v] for v in df["evidence_status"]]
            self.t3 = [LABEL2IDX["t3"][v] for v in df["evidence_quality"]]
            self.t4 = [LABEL2IDX["t4"][v] for v in df["verification_timeline"]]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict:
        item = {"embedding": self.embeddings[idx], "sample_id": self.ids[idx]}
        if self.has_labels:
            item["t1"] = torch.tensor(self.t1[idx], dtype=torch.long)
            item["t2"] = torch.tensor(self.t2[idx], dtype=torch.long)
            item["t3"] = torch.tensor(self.t3[idx], dtype=torch.long)
            item["t4"] = torch.tensor(self.t4[idx], dtype=torch.long)
        return item


class ContrastivePairDataset(Dataset):
    """Same-task3-label pairs for supervised contrastive pre-training."""

    def __init__(self, embeddings: torch.Tensor, df: pd.DataFrame) -> None:
        self.embeddings = embeddings
        self.t3_labels = [LABEL2IDX["t3"][v] for v in df["evidence_quality"]]
        # Build per-class index lists
        self.class_indices: dict[int, list[int]] = {}
        for i, lbl in enumerate(self.t3_labels):
            self.class_indices.setdefault(lbl, []).append(i)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict:
        label = self.t3_labels[idx]
        # Sample a positive (same class, different index)
        pos_pool = [j for j in self.class_indices[label] if j != idx]
        pos_idx = random.choice(pos_pool) if pos_pool else idx
        return {
            "anchor": self.embeddings[idx],
            "positive": self.embeddings[pos_idx],
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_dataframes(data_dir: str, use_augmented: bool = False,
                    aug_filename: str = "train_data_augmented.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_filename = aug_filename if use_augmented else "train_data.csv"
    train = pd.read_csv(data_dir / train_filename)
    # final_data uses valid_data.csv as the eval target (old test_data renamed)
    if (data_dir / "valid_data.csv").exists() and not (data_dir / "test_data.csv").exists():
        test = pd.read_csv(data_dir / "valid_data.csv")
    else:
        test = pd.read_csv(data_dir / "test_data.csv")
    # CSV stores missing downstream labels as NaN → "N/A"
    for col in ["evidence_status", "evidence_quality", "verification_timeline"]:
        if col in train.columns:
            train[col] = train[col].fillna("N/A")
    # Normalize T4 label: longer_than_5_years → more_than_5_years (new standard)
    if "verification_timeline" in train.columns:
        train["verification_timeline"] = train["verification_timeline"].replace(
            "longer_than_5_years", "more_than_5_years")
    # Enforce dependency rule in training labels
    if "promise_status" in train.columns:
        na_mask = train["promise_status"] == "No"
        train.loc[na_mask, "evidence_status"] = "N/A"
        train.loc[na_mask, "evidence_quality"] = "N/A"
        train.loc[na_mask, "verification_timeline"] = "N/A"
    return train, test


def split_train_val(df: pd.DataFrame, val_ratio: float = 0.1,
                    seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=df["promise_status"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def augment_rare_classes(
    df: pd.DataFrame,
    target_n: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Augment rare-class samples via simple Chinese character-level perturbation.
    Targets Misleading (T3, typically 1 sample) and within_2_years (T4, ~11 samples).
    Augmented rows are appended; original rows are kept unchanged.
    """
    rng = random.Random(seed)

    # Common single-char substitution pairs (semantically similar in formal Chinese)
    SUBS: dict[str, list[str]] = {
        '的': ['地', '之'],
        '是': ['為', '係'],
        '將': ['會', '預計'],
        '已': ['已經'],
        '以': ['藉由', '透過'],
        '並': ['且', '同時'],
        '其': ['該', '此'],
        '有': ['具有', '設有'],
    }

    def _perturb(text: str) -> str:
        chars = list(text)
        punct = set('，。、；：！？「」『』【】（）…—\n')
        # Random character dropout ~8%
        chars = [c for c in chars if c in punct or rng.random() > 0.08]
        # Random character substitution ~20% of eligible chars
        result: list[str] = []
        for c in chars:
            if c in SUBS and rng.random() < 0.20:
                result.append(rng.choice(SUBS[c]))
            else:
                result.append(c)
        return ''.join(result)

    rare_conditions = [
        df["evidence_quality"] == "Misleading",
        df["verification_timeline"] == "within_2_years",
    ]
    # Use integer IDs beyond the existing max to avoid DataLoader collate errors
    next_id = int(df["id"].max()) + 1
    augmented_rows: list[pd.Series] = []
    augmented_source_ids: set = set()   # track which original rows have been augmented
    for mask in rare_conditions:
        rare = df[mask]
        if len(rare) == 0:
            continue
        n_needed = max(0, target_n - len(rare))
        for i in range(n_needed):
            src_row = rare.iloc[i % len(rare)]
            row = src_row.copy()
            row["data"] = _perturb(str(src_row["data"]))
            row["id"] = next_id
            next_id += 1
            augmented_source_ids.add(src_row["id"])
            augmented_rows.append(row)

    if not augmented_rows:
        return df
    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)


def gen_pseudo_labels(cfg: Config, confidence_thr: float = 0.80) -> None:
    """Generate pseudo-labels for high-confidence test predictions.

    Runs the best ensemble (rob+lert aug kfold + kNN-LDL) on test data,
    keeps samples where the weighted confidence across all tasks exceeds
    confidence_thr, and saves them as a CSV that can be merged with training data.

    Output: data/pseudo_labeled_test.csv (same columns as train_data.csv)
    """
    TASK_WEIGHTS = {"t1": 0.20, "t2": 0.30, "t3": 0.35, "t4": 0.15}
    KFOLD_DIRS_PL = {
        "rob": "runs/A1_roberta_dc_kfold_llmaug",
        "lert": "runs/A1_lert_dc_kfold_llmaug",
    }
    KNN_ENCODER_DIR = "runs/A1_roberta_dc_kfold_llmaug"
    KNN_K = 10
    KNN_ALPHA = {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}

    train_df, test_df = load_dataframes(cfg.data_dir, use_augmented=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Collect soft probs from each kfold model ──────────────────────────────
    all_probs_list: list[dict[str, list[list[float]]]] = []
    for model_name, kfold_dir in KFOLD_DIRS_PL.items():
        fold_probs_list: list[dict[str, list[list[float]]]] = []
        fold_idx = 1
        while True:
            ckpt_path = Path(kfold_dir) / f"fold{fold_idx}" / "best.pt"
            if not ckpt_path.exists():
                break
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            saved_cfg: Config = ckpt["cfg"]
            dc = getattr(saved_cfg, "deep_cascade", False)
            tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
            te_ds = ESGDataset(test_df, tokenizer, saved_cfg.max_length, has_labels=False)
            te_loader = DataLoader(te_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
            model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc).to(device)
            model.load_state_dict(ckpt["model"])
            fold_probs_list.append(predict_probs(model, te_loader, device))
            del model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [{model_name}] fold{fold_idx} done")
            fold_idx += 1
        # Average across folds for this model
        avg = {t: [[sum(fold_probs_list[f][t][i][c] for f in range(len(fold_probs_list))) / len(fold_probs_list)
                    for c in range(NUM_LABELS[t])]
                   for i in range(len(test_df))]
               for t in LABELS}
        all_probs_list.append(avg)

    # ── Ensemble rob + lert (1:1) ─────────────────────────────────────────────
    n = len(test_df)
    ensemble_probs: dict[str, list[list[float]]] = {
        t: [[sum(all_probs_list[m][t][i][c] for m in range(len(all_probs_list))) / len(all_probs_list)
             for c in range(NUM_LABELS[t])]
            for i in range(n)]
        for t in LABELS
    }

    # ── kNN-LDL fusion ────────────────────────────────────────────────────────
    knn_enc_ckpt = torch.load(
        str(Path(KNN_ENCODER_DIR) / "fold1" / "best.pt"), map_location=device, weights_only=False)
    knn_saved_cfg: Config = knn_enc_ckpt["cfg"]
    knn_dc = getattr(knn_saved_cfg, "deep_cascade", False)
    knn_tokenizer = AutoTokenizer.from_pretrained(knn_saved_cfg.backbone, trust_remote_code=True)
    knn_model = ApproachA1(knn_saved_cfg.backbone, knn_saved_cfg.dropout, deep_cascade=knn_dc).to(device)
    knn_model.load_state_dict(knn_enc_ckpt["model"])

    knn_train_df = train_df.head(800)  # real samples only
    tr_ds = ESGDataset(knn_train_df, knn_tokenizer, knn_saved_cfg.max_length, has_labels=False)
    tr_loader = DataLoader(tr_ds, batch_size=knn_saved_cfg.batch_size, shuffle=False, num_workers=0)
    tr_embs = extract_cls_embeddings(knn_model, tr_loader, device)

    te_ds2 = ESGDataset(test_df, knn_tokenizer, knn_saved_cfg.max_length, has_labels=False)
    te_loader2 = DataLoader(te_ds2, batch_size=knn_saved_cfg.batch_size, shuffle=False, num_workers=0)
    te_embs = extract_cls_embeddings(knn_model, te_loader2, device)
    del knn_model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    knn_probs = compute_knn_ldl_probs(tr_embs, te_embs, knn_train_df, k=KNN_K)
    fused_labels = knn_fuse_probs(ensemble_probs, knn_probs, alpha=KNN_ALPHA)
    # Also need fused probs for confidence computation (recompute inline)
    fused_probs: dict[str, list[list[float]]] = {t: [] for t in LABELS}
    for i in range(n):
        for task in LABELS:
            a = KNN_ALPHA.get(task, 0.0)
            mp = ensemble_probs[task][i]
            kp = knn_probs[task][i]
            fused_probs[task].append([(1 - a) * mp[c] + a * kp[c] for c in range(NUM_LABELS[task])])

    # ── Filter by weighted confidence ─────────────────────────────────────────
    kept, skipped = 0, 0
    pseudo_rows = []
    for i in range(n):
        weighted_conf = sum(
            TASK_WEIGHTS[t] * max(fused_probs[t][i]) for t in LABELS)
        if weighted_conf < confidence_thr:
            skipped += 1
            continue
        row = test_df.iloc[i].to_dict()
        row["promise_status"]        = fused_labels["t1"][i]
        row["evidence_status"]       = fused_labels["t2"][i]
        row["evidence_quality"]      = fused_labels["t3"][i]
        row["verification_timeline"] = fused_labels["t4"][i]
        row["promise_string"]        = ""
        row["evidence_string"]       = ""
        row["esg_type"]              = ""
        pseudo_rows.append(row)
        kept += 1

    print(f"\n[Pseudo-label] kept={kept}, skipped={skipped} (thr={confidence_thr})")

    if not pseudo_rows:
        print("  No samples kept. Try lowering confidence_thr.")
        return

    out_path = Path(cfg.data_dir) / "pseudo_labeled_test.csv"
    pd.DataFrame(pseudo_rows).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    # Print label distribution of pseudo-labeled samples
    pl_df = pd.DataFrame(pseudo_rows)
    for col in ["promise_status", "evidence_status", "evidence_quality", "verification_timeline"]:
        print(f"  {col}: {pl_df[col].value_counts().to_dict()}")


def gen_llm_augdata(cfg: Config) -> None:
    """Use Qwen3-8B to generate synthetic training samples for rare classes.

    Targets:
      - Not Clear (99 samples → augment to ~200)
      - Misleading (1 sample → augment to ~30)
      - within_2_years (11 samples → augment to ~50)

    Generates paraphrases by prompting the LLM to rewrite existing examples
    in different words while preserving the ESG classification label.
    Output saved to: data/llm_augmented.csv (appended to train_data.csv).
    """
    train_df, _ = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)

    AUG_TARGETS = {
        # (column, value, target_total, label_dict)
        "Misleading":     ("evidence_quality",      "Misleading",      50,
                           {"promise_status": "Yes", "evidence_status": "Yes",
                            "evidence_quality": "Misleading", "verification_timeline": None}),
        "within_2_years": ("verification_timeline", "within_2_years",  80,
                           {"promise_status": "Yes", "evidence_status": "Yes",
                            "evidence_quality": None, "verification_timeline": "within_2_years"}),
        "Not_Clear":      ("evidence_quality",      "Not Clear",       350,
                           {"promise_status": "Yes", "evidence_status": "Yes",
                            "evidence_quality": "Not Clear", "verification_timeline": None}),
    }

    ESG_TYPE_DESC = {
        "E": "環境（Environment）類：碳排放、能源使用、氣候變遷、環境保護相關",
        "S": "社會（Social）類：員工權益、供應鏈管理、社會責任、人權保障相關",
        "G": "治理（Governance）類：企業治理、法規遵循、董事會運作、風險管理相關",
    }
    QUALITY_HINT = {
        "Not Clear":   "佐證存在但使用模糊字眼（如「持續推進」「努力改善」「積極探索」「逐步推動」），"
                       "避免具體數字、完成日期或第三方認證",
        "Clear":       "佐證具體明確，包含可量化數字（如百分比、金額）、完成年份或第三方認證",
        "Misleading":  "聲稱有執行成果，但論述自相矛盾、刻意迴避核心指標或誇大效益",
    }
    TIMELINE_HINT = {
        "already":               "目標已達成或持續執行中（使用「已」「目前」「本年度達成」）",
        "within_2_years":        "承諾預計2年內完成（使用「明年」「2026年」「近期」「短期內」）",
        "between_2_and_5_years": "承諾預計2至5年內完成（使用「2028年」「2030年」「中期目標」）",
        "longer_than_5_years":   "承諾需5年以上才能完成（使用「2035年」「長期願景」「下一個十年」）",
        "N/A": "不適用",
    }

    GEN_PROMPT = (
        "你是一位ESG永續報告寫作專家，同時精通「漂綠風險」評估。\n\n"
        "請生成一段【全新的、不同於參考文字的】繁體中文ESG報告片段，需符合以下規格：\n\n"
        "【ESG類型】{esg_type_desc}\n"
        "【目標標籤】\n"
        "- promise_status: {promise_status}\n"
        "- evidence_status: {evidence_status}\n"
        "- evidence_quality: {evidence_quality}（{quality_hint}）\n"
        "- verification_timeline: {verification_timeline}（{timeline_hint}）\n\n"
        "【參考語氣風格】（請勿複製原文，僅參考句式）\n{ref_text}\n\n"
        "【生成規則】\n"
        "1. 長度：100~250字\n"
        "2. 繁體中文，模擬真實台灣上市公司ESG報告語氣\n"
        "3. 不要包含公司名稱\n\n"
        "請直接輸出報告片段，不要任何說明文字。"
    )

    # Rule-based verifiers (replaces LLM self-verification which self-sabotages)
    _FUZZY_WORDS = {"持續推進", "努力改善", "積極探索", "逐步推動", "致力推動",
                    "持續關注", "各項環保", "各項措施", "相關措施", "多元管道",
                    "將致力", "積極推動", "持續努力", "逐步落實", "積極改善"}
    _SPECIFIC_PAT = re.compile(r'\d+\s*%|\d+\s*億|\d+\s*萬|ISO\s*\d+|第三方|已達成|已完成|已建置|已導入')
    _MISLEAD_PAT  = re.compile(r'(但|然而|雖然.*卻|實際上|卻未|尚未達標|未能如期)')
    _WITHIN2_PAT  = re.compile(r'明年|2026|2027|2年內|兩年內|近期內|短期內|年底前完成')

    def _rule_verify(text: str, target: dict) -> bool:
        eq = target.get("evidence_quality")
        tl = target.get("verification_timeline")
        if eq == "Not Clear":
            has_fuzzy    = any(w in text for w in _FUZZY_WORDS)
            has_specific = bool(_SPECIFIC_PAT.search(text))
            return has_fuzzy and not has_specific
        if eq == "Misleading":
            return bool(_MISLEAD_PAT.search(text))
        if tl == "within_2_years":
            return bool(_WITHIN2_PAT.search(text))
        # For other classes just accept (length already checked by caller)
        return True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {cfg.backbone} for LLM augmentation ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.backbone, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    import json as _json

    def _llm_generate(m, tok, dev, prompt_text: str, max_tokens: int = 300, temp: float = 0.85) -> str:
        msgs = [{"role": "user", "content": prompt_text}]
        chat = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(chat, return_tensors="pt")
        ids = enc["input_ids"].to(dev)
        mask = enc["attention_mask"].to(dev)
        with torch.no_grad():
            out = m.generate(
                ids, attention_mask=mask, max_new_tokens=max_tokens,
                temperature=temp, do_sample=True, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def _verify_labels(text: str, target: dict) -> bool:
        return _rule_verify(text, target)

    augmented_rows: list[dict] = []
    next_id = int(train_df["id"].max()) + 100000  # far from real IDs

    for aug_name, (col, val, target, labels_override) in AUG_TARGETS.items():
        rare = train_df[train_df[col] == val]
        if len(rare) == 0:
            print(f"  [SKIP] {aug_name}: 0 samples found")
            continue
        n_needed = max(0, target - len(rare))
        print(f"\n  [{aug_name}] {len(rare)} existing → generating {n_needed} synthetic samples")

        generated = 0
        attempt = 0
        seen_texts: set = set(rare["data"].tolist())  # dedup against real + prior synthetic
        while generated < n_needed and attempt < n_needed * 5:
            src_row = rare.iloc[attempt % len(rare)]
            attempt += 1
            esg_type = str(src_row.get("esg_type", "E"))

            def _resolve(key: str) -> str:
                v = labels_override.get(key)
                return v if v is not None else str(src_row[key])

            gen_prompt = GEN_PROMPT.format(
                esg_type_desc=ESG_TYPE_DESC.get(esg_type, ESG_TYPE_DESC["E"]),
                promise_status=_resolve("promise_status"),
                evidence_status=_resolve("evidence_status"),
                evidence_quality=_resolve("evidence_quality"),
                quality_hint=QUALITY_HINT.get(_resolve("evidence_quality"), ""),
                verification_timeline=_resolve("verification_timeline"),
                timeline_hint=TIMELINE_HINT.get(_resolve("verification_timeline"), ""),
                ref_text=src_row["data"][:200],
            )
            generated_text = _llm_generate(model, tokenizer, device, gen_prompt, max_tokens=350, temp=0.85)

            if len(generated_text) < 30 or generated_text in seen_texts:
                continue

            # Self-verification: re-classify with LLM, discard mismatches
            target_labels = {k: v for k, v in labels_override.items() if v is not None}
            if not _verify_labels(generated_text, target_labels):
                continue

            seen_texts.add(generated_text)
            new_row = src_row.to_dict()
            new_row["data"] = generated_text
            new_row["id"] = next_id
            next_id += 1
            for k, v in labels_override.items():
                if v is not None:
                    new_row[k] = v
            augmented_rows.append(new_row)
            generated += 1
            if generated % 10 == 0:
                print(f"    {generated}/{n_needed} done")

        if generated < n_needed:
            print(f"  [WARN] {aug_name}: only generated {generated}/{n_needed} (retry budget exhausted)")

    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not augmented_rows:
        print("No augmented data generated.")
        return

    aug_df = pd.DataFrame(augmented_rows)
    out_path = Path(cfg.data_dir) / "llm_augmented.csv"
    aug_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(aug_df)} augmented samples → {out_path}")

    # Also merge with train and save combined
    combined = pd.concat([train_df, aug_df], ignore_index=True)
    combined_path = Path(cfg.data_dir) / "train_data_augmented.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Combined dataset: {len(combined)} samples → {combined_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class OrdinalLoss(nn.Module):
    """Ordinal penalty: larger mis-class distance → larger loss. N/A index excluded from penalty."""
    def __init__(self, num_classes: int = 5, weight: Optional[torch.Tensor] = None,
                 na_idx: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.na_idx = na_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        na_mask = targets == self.na_idx
        pred_cls = logits.argmax(dim=-1)
        dist = torch.abs(pred_cls.float() - targets.float())
        scale = torch.where(na_mask, torch.ones_like(dist), 1.0 + 0.5 * dist.clamp(max=3))
        return (ce * scale).mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        smooth_val = self.smoothing / (self.num_classes - 1)
        one_hot = torch.full_like(log_prob, smooth_val)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(one_hot * log_prob).sum(dim=-1).mean()


class DBLoss(nn.Module):
    """
    Distribution-Balanced Loss for long-tailed single-label multi-class classification.
    Paper: Wu et al., ECCV 2021 — adapted from multi-label BCE to single-label CE.

    Two components from the paper:

    1. Re-balanced Weighting (Eq. 3–5):
       r_i = max_n / n_i                       (freq ratio: rare → higher)
       r̂_i = α + sigmoid(β * (r_i - μ))       (Eq.4 smoothing into [α, α+1])
       Loss weight for sample with label y = r̂_y

    2. Negative-Tolerant Regularization — NTR (Eq. 9, 10):
       p_i = n_i / N                           (class prior)
       b̂_i = logit(p_i) = log(p_i / (1-p_i)) (optimal bias at random init)
       ν_i  = -κ * b̂_i                        (Eq.9: rare classes → ν>0 → logit lifted)
       shifted logits: z_i' = z_i - ν_i        (lower decision threshold for rare classes)

    Final: L = mean( r̂[y] · CE(z - ν, y) )

    Adaptation note:
      The paper uses sigmoid-BCE per label (multi-label).
      For single-label multi-class we use softmax-CE on shifted logits.
      Re-balanced weight r̂_i is applied per-sample on the true class,
      which is equivalent to the paper's Eq.5 restricted to the positive label.
    """

    def __init__(
        self,
        class_counts: list[int],
        total_samples: int,
        alpha: float = 0.1,    # base weight lift (paper default 0.1)
        beta: float = 10.0,    # smoothing steepness (paper default 10)
        mu: float = 0.5,       # smoothing centre (paper default 0.3–0.5)
        kappa: float = 0.05,   # NTR scale factor (paper default 0.05)
    ) -> None:
        super().__init__()
        n = torch.tensor(class_counts, dtype=torch.float32)   # [C]
        N = float(total_samples)

        # ── Re-balanced weights ───────────────────────────────────────────
        # r_i: ratio between class-level and instance-level sampling frequency
        # For single-label this simplifies to max_n / n_i (inverse freq ratio)
        r = n.max() / n.clamp(min=1.0)                         # [C]
        r_hat = alpha + torch.sigmoid(beta * (r - mu))         # [C] Eq.4

        # ── NTR logit shifts ─────────────────────────────────────────────
        p = (n / N).clamp(1e-6, 1.0 - 1e-6)                   # [C]
        b_hat = torch.log(p / (1.0 - p))                       # logit(p_i)  Eq.9
        nu = -kappa * b_hat                                     # [C]

        self.register_buffer("r_hat", r_hat)
        self.register_buffer("nu", nu)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply NTR shift: lower threshold for rare classes
        shifted = logits - self.nu.unsqueeze(0)                # [B, C]
        # Per-sample re-balance weight (weight of the true class)
        r_weights = self.r_hat[targets]                        # [B]
        ce = F.cross_entropy(shifted, targets, reduction="none")  # [B]
        return (r_weights * ce).mean()


class OLLLoss(nn.Module):
    """
    Ordinal Log-Loss (COLING 2022, https://aclanthology.org/2022.coling-1.407.pdf).

    L = -log(p_y) + alpha * sum_{k != y} D[y, k] * p_k

    where D[y, k] = |rank(y) - rank(k)| is the ordinal distance.
    N/A class (na_idx) has distance 0 to all classes and from all classes,
    so it contributes no ordinal penalty either as target or as comparison.
    When target is N/A, loss reduces to pure CE (no ordinal structure applies).
    """

    def __init__(self, num_classes: int, alpha: float = 1.5, na_idx: int = -1) -> None:
        super().__init__()
        self.alpha = alpha
        self.na_idx = na_idx
        # D[i, j] = |i - j|; rows/cols for na_idx are zeroed out
        D = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                if i != na_idx and j != na_idx:
                    D[i, j] = abs(i - j)
        self.register_buffer("D", D)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)          # [B, C]
        ce = F.cross_entropy(logits, targets, reduction="none")  # [B]
        d = self.D[targets]                        # [B, C] ordinal distances from each target
        ordinal_penalty = self.alpha * (d * probs).sum(dim=-1)   # [B]
        return (ce + ordinal_penalty).mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    NTXent-style supervised contrastive loss for embedding fine-tuning.
    Pulls same-class pairs together, pushes different-class apart.
    """
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        # Normalise
        z1 = F.normalize(anchors, dim=-1)      # [B, D]
        z2 = F.normalize(positives, dim=-1)    # [B, D]
        z = torch.cat([z1, z2], dim=0)         # [2B, D]
        labels_2x = torch.cat([labels, labels], dim=0)  # [2B]

        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]
        # Mask out self-similarity
        mask_self = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        sim.masked_fill_(mask_self, float("-inf"))

        # Positive mask: same label, not self
        pos_mask = (labels_2x.unsqueeze(0) == labels_2x.unsqueeze(1)) & ~mask_self

        # Log-sum-exp over all non-self pairs as denominator
        log_denom = torch.logsumexp(sim, dim=1)    # [2B]

        # Mean log-prob of positive pairs
        pos_sim = sim.masked_fill(~pos_mask, float("-inf"))
        log_pos = torch.logsumexp(pos_sim, dim=1)  # [2B]

        # Only compute loss for rows with at least one positive
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, requires_grad=True, device=sim.device)
        loss = -(log_pos - log_denom)[has_pos].mean()
        return loss


class ClassAwarePrototypeLoss(nn.Module):
    """
    Class-Aware Prototype Contrastive Loss for T3 (arXiv 2410.22197).
    Computes per-class centroids from in-batch embeddings, then for each sample
    maximises similarity to its class prototype and minimises similarity to others.
    Handles class imbalance better than pairwise SCL: even 1 NC sample per batch
    forms a valid NC prototype that is contrasted against the Clear prototype.
    """
    def __init__(self, temperature: float = 0.07, weight: float = 0.1,
                 num_classes: int = 3) -> None:
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.num_classes = num_classes  # Clear=0, NC=1, Misleading=2 (N/A=3 excluded)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Filter out N/A (index ≥ num_classes)
        valid = labels < self.num_classes
        if valid.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)
        emb = F.normalize(embeddings[valid], dim=-1)   # [B', D]
        lbl = labels[valid]
        unique_cls = lbl.unique()
        if len(unique_cls) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        # Build per-class prototypes (mean of in-batch instances)
        protos, proto_lbl = [], []
        for c in unique_cls:
            mask = lbl == c
            protos.append(F.normalize(emb[mask].mean(dim=0), dim=-1))
            proto_lbl.append(c)
        protos = torch.stack(protos, dim=0)       # [K, D]
        proto_lbl = torch.stack(proto_lbl)        # [K]
        # Similarity of each sample to each prototype
        sim = torch.mm(emb, protos.T) / self.temperature   # [B', K]
        # Target: which prototype column corresponds to each sample's class
        target = torch.zeros(lbl.size(0), dtype=torch.long, device=lbl.device)
        for i, c in enumerate(proto_lbl):
            target[lbl == c] = i
        return self.weight * F.cross_entropy(sim, target)


class SharpReCL(nn.Module):
    """Prototype-guided contrastive rebalancing (arxiv 2405.11524).

    Unlike standard SCL which uses random in-batch pairs, SharpReCL:
    1. Maintains running per-class prototypes (EMA updated).
    2. For each sample, generates a hard negative via Mixup with the
       nearest *different-class* prototype embedding.
    3. Contrastive loss: pull sample toward its own prototype, push away
       from the hard negative.
    This directly addresses minority-class boundary confusion (NC vs Clear).
    """

    def __init__(self, hidden_size: int, num_classes: int = 3,
                 temperature: float = 0.07, weight: float = 0.10,
                 mixup_alpha: float = 0.3, momentum: float = 0.99) -> None:
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.mixup_alpha = mixup_alpha
        self.momentum = momentum
        self.num_classes = num_classes
        # Running prototypes (EMA)
        self.register_buffer("prototypes",
                             torch.randn(num_classes, hidden_size))
        self.register_buffer("proto_init",
                             torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update_prototypes(self, embeddings: torch.Tensor,
                           labels: torch.Tensor) -> None:
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            centroid = F.normalize(embeddings[mask].mean(dim=0), dim=-1)
            if not self.proto_init[c]:
                self.prototypes[c] = centroid
                self.proto_init[c] = True
            else:
                self.prototypes[c] = (self.momentum * self.prototypes[c]
                                      + (1 - self.momentum) * centroid)
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=-1)

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        valid = labels < self.num_classes
        if valid.sum() < 2:
            return torch.tensor(0.0, device=embeddings.device)
        emb = F.normalize(embeddings[valid], dim=-1)
        lbl = labels[valid]
        if lbl.unique().numel() < 2:
            return torch.tensor(0.0, device=embeddings.device)

        self._update_prototypes(emb, lbl)

        # For each sample, find nearest different-class prototype
        protos = F.normalize(self.prototypes[:self.num_classes], dim=-1)
        sim_to_proto = torch.mm(emb, protos.T)  # [B', K]
        # Mask own-class prototype with -inf
        own_mask = F.one_hot(lbl, self.num_classes).bool()
        sim_other = sim_to_proto.masked_fill(own_mask, -1e9)
        nearest_other = sim_other.argmax(dim=-1)  # [B']

        # Generate hard negatives via Mixup: bias toward other-class prototype
        # Small lam → more weight on prototype → harder negative
        sampled = float(np.random.beta(self.mixup_alpha + 0.1,
                                       self.mixup_alpha + 0.1))
        lam = min(1.0 - self.mixup_alpha, sampled)  # cap at 0.7 → proto gets ≥0.3
        hard_neg = (lam * emb
                    + (1 - lam) * protos[nearest_other])
        hard_neg = F.normalize(hard_neg, dim=-1)

        # Contrastive: positive = own prototype, negative = hard_neg
        pos_sim = (emb * protos[lbl]).sum(dim=-1) / self.temperature
        neg_sim = (emb * hard_neg).sum(dim=-1) / self.temperature
        # InfoNCE-style
        logits = torch.stack([pos_sim, neg_sim], dim=-1)  # [B', 2]
        targets = torch.zeros(logits.size(0), dtype=torch.long,
                              device=logits.device)
        return self.weight * F.cross_entropy(logits, targets)


class MR2Loss(nn.Module):
    """Margin Regularization for macro-F1 (arxiv 2602.00205, Feb 2026).

    Adds adaptive per-class logit margins proportional to intra-class
    feature spread.  Classes with larger spread (harder to classify, e.g.
    'Not Clear') automatically get larger margins.
    N/A (index >= num_classes) is excluded — only real quality classes.
    """

    def __init__(self, num_classes: int = 3,
                 margin_scale: float = 0.3, weight: float = 0.05) -> None:
        super().__init__()
        self.num_classes = num_classes  # 3: Clear/NC/Misleading (exclude N/A)
        self.margin_scale = margin_scale
        self.weight = weight
        self.register_buffer("class_spread",
                             torch.ones(num_classes))
        self.register_buffer("spread_init",
                             torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update_spread(self, embeddings: torch.Tensor,
                       labels: torch.Tensor) -> None:
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() < 2:
                continue
            emb_c = embeddings[mask]
            spread = emb_c.std(dim=0).mean()
            if not self.spread_init[c]:
                self.class_spread[c] = spread
                self.spread_init[c] = True
            else:
                self.class_spread[c] = 0.9 * self.class_spread[c] + 0.1 * spread

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                embeddings: torch.Tensor) -> torch.Tensor:
        # Exclude N/A samples (label >= num_classes)
        valid = labels < self.num_classes
        if valid.sum() < 2:
            return torch.tensor(0.0, device=logits.device)
        self._update_spread(embeddings[valid].detach(), labels[valid])
        # Out-of-place margin subtraction to avoid autograd in-place errors
        margins = self.margin_scale * self.class_spread[:self.num_classes]
        one_hot = F.one_hot(labels[valid], logits.size(-1)).float()[:, :self.num_classes]
        # Pad margins to match full logit width (including N/A column)
        full_margins = torch.zeros(logits.size(-1), device=logits.device)
        full_margins[:self.num_classes] = margins
        margin_matrix = one_hot @ torch.diag(margins)
        # Pad to full logit width
        margin_full = torch.zeros_like(logits[valid])
        margin_full[:, :self.num_classes] = margin_matrix
        margin_adjusted = logits[valid] - margin_full
        return self.weight * F.cross_entropy(margin_adjusted, labels[valid])


def build_loss_fn(
    loss_type: str,
    task: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smooth: float = 0.1,
    class_counts: Optional[list[int]] = None,
    total_samples: int = 800,
    db_alpha: float = 0.1,
    db_beta: float = 10.0,
    db_mu: float = 0.5,
    db_kappa: float = 0.05,
    ce_label_smoothing: float = 0.0,
) -> nn.Module:
    num_cls = NUM_LABELS[task]
    w = class_weights
    if loss_type == "focal":
        return FocalLoss(gamma=focal_gamma, weight=w)
    if loss_type == "ordinal" and task == "t4":
        return OrdinalLoss(num_classes=num_cls, weight=w, na_idx=4)
    if loss_type == "ordinal" and task == "t3":
        # T3: Clear(0) > Not Clear(1) > Misleading(2); N/A(3) excluded from ordinal penalty
        return OrdinalLoss(num_classes=num_cls, weight=w, na_idx=3)
    if loss_type == "label_smooth":
        return LabelSmoothingLoss(num_classes=num_cls, smoothing=label_smooth)
    if loss_type == "db":
        counts = class_counts or [total_samples // num_cls] * num_cls
        return DBLoss(counts, total_samples,
                      alpha=db_alpha, beta=db_beta, mu=db_mu, kappa=db_kappa)
    if loss_type == "oll" and task == "t3":
        # T3: Clear(0) > Not Clear(1) > Misleading(2); N/A(3) excluded from ordinal distance
        return OLLLoss(num_classes=num_cls, alpha=1.5, na_idx=3)
    return nn.CrossEntropyLoss(weight=w, label_smoothing=ce_label_smoothing)


def build_loss_fns(cfg: Config, train_df: pd.DataFrame,
                   device: torch.device) -> dict[str, nn.Module]:
    task_col = {"t1": "promise_status", "t2": "evidence_status",
                "t3": "evidence_quality", "t4": "verification_timeline"}

    # Per-task optimal loss: T1=CE(weighted), T2=CE, T3=DB, T4=Ordinal
    # NOTE: T2=Focal was proven harmful in V2 (caused No→Yes prediction bias); reverted to CE
    # NOTE: T3 Ordinal (V15) tested → worse; OLL (V19, COLING 2022) tested → worse
    # NOTE: DB's NTR logit shift handles extreme long-tail better than ordinal penalty losses
    PERTASK: dict[str, str] = {"t1": "ce", "t2": "ce", "t3": "db", "t4": "ordinal"}

    from collections import Counter
    t3_nc_weight = cfg.t3_nc_weight
    loss_fns = {}
    for task, col in task_col.items():
        loss_type = PERTASK[task] if cfg.per_task_loss else cfg.loss_type
        # T3 Not Clear boost: override DB → CE/focal with boosted Not Clear class weight
        if task == "t3" and t3_nc_weight > 1.0:
            loss_type = cfg.t3_loss_type
            w = compute_class_weights(train_df[col].tolist(), task).to(device)
            nc_idx = LABEL2IDX["t3"]["Not Clear"]
            w[nc_idx] = w[nc_idx] * t3_nc_weight
        else:
            # DB handles its own weighting internally; skip class_weight for it
            use_cw = cfg.use_class_weight and loss_type not in ("db",)
            w = compute_class_weights(train_df[col].tolist(), task).to(device) if use_cw else None
        counts_map = Counter(train_df[col].tolist())
        counts = [counts_map.get(lbl, 0) for lbl in LABELS[task]]
        loss_fns[task] = build_loss_fn(
            loss_type, task,
            class_weights=w,
            focal_gamma=cfg.focal_gamma,
            label_smooth=cfg.label_smooth,
            class_counts=counts,
            total_samples=len(train_df),
            db_alpha=cfg.db_alpha,
            db_beta=cfg.db_beta,
            db_mu=cfg.db_mu,
            db_kappa=cfg.db_kappa,
            ce_label_smoothing=getattr(cfg, 'ce_label_smoothing', 0.0),
        ).to(device)
    # conditional chain-rule aux weight (read by _compute_loss; ignored by task loop)
    loss_fns["_t3_cond_w"] = float(getattr(cfg, "t3_cond_weight", 0.0))
    return loss_fns


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODELS
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, NUM_LABELS[task]) for task in LABELS
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.dropout(x)
        return {task: head(x) for task, head in self.heads.items()}


class CascadeHead(nn.Module):
    """Task1 logits (softmax) concatenated into Task2/3/4 inputs."""
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.t1 = nn.Linear(hidden_size, NUM_LABELS["t1"])
        ext = hidden_size + NUM_LABELS["t1"]
        self.t2 = nn.Linear(ext, NUM_LABELS["t2"])
        self.t3 = nn.Linear(ext, NUM_LABELS["t3"])
        self.t4 = nn.Linear(ext, NUM_LABELS["t4"])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.drop(x)
        t1_logits = self.t1(x)
        t1_prob = torch.softmax(t1_logits.detach(), -1)
        ext = torch.cat([x, t1_prob], dim=-1)
        return {"t1": t1_logits, "t2": self.t2(ext), "t3": self.t3(ext), "t4": self.t4(ext)}


class CascadeHeadV2(nn.Module):
    """Extended cascade: T1→T2, [T1+T2]→T3, [T1]→T4.
    Captures the T2=No → T3=N/A dependency directly in the architecture.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.t1 = nn.Linear(hidden_size, NUM_LABELS["t1"])
        ext1 = hidden_size + NUM_LABELS["t1"]
        self.t2 = nn.Linear(ext1, NUM_LABELS["t2"])
        ext2 = hidden_size + NUM_LABELS["t1"] + NUM_LABELS["t2"]
        self.t3 = nn.Linear(ext2, NUM_LABELS["t3"])  # sees both T1 and T2 probs
        self.t4 = nn.Linear(ext1, NUM_LABELS["t4"])  # only T1, not T2

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.drop(x)
        t1_logits = self.t1(x)
        t1_prob = torch.softmax(t1_logits.detach(), -1)
        ext1 = torch.cat([x, t1_prob], dim=-1)
        t2_logits = self.t2(ext1)
        t2_prob = torch.softmax(t2_logits.detach(), -1)
        ext2 = torch.cat([x, t1_prob, t2_prob], dim=-1)
        return {
            "t1": t1_logits,
            "t2": t2_logits,
            "t3": self.t3(ext2),
            "t4": self.t4(ext1),
        }


# ── Approach A / A1 ──────────────────────────────────────────────────────────

class ApproachA(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.head = MultiTaskHead(self.encoder.config.hidden_size, dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> dict[str, torch.Tensor]:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None and self.encoder.config.model_type not in _NO_TOKEN_TYPE_IDS:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        return self.head(out.last_hidden_state[:, 0])


class AttentionPool(nn.Module):
    """Learnable attention-weighted pooling (SemEval-2025 Task 6 best practice)."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(hidden_states).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~attention_mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1)  # (B, L)
        return (weights.unsqueeze(-1) * hidden_states).sum(dim=1)  # (B, H)


class ApproachA1(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.1,
                 deep_cascade: bool = False,
                 use_hand_features: bool = False,
                 use_span: bool = False,
                 company_emb_dim: int = 0,
                 n_companies: int = 51,
                 pooling: str = "cls") -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        h = self.encoder.config.hidden_size
        self.use_hand_features = use_hand_features
        self.use_span = use_span
        self.pooling = pooling
        if pooling == "attn":
            self.attn_pool = AttentionPool(h)
        if use_hand_features:
            # Project hand features to hidden dim
            self.hf_proj = nn.Sequential(
                nn.Linear(HAND_FEATURE_DIM, h),
                nn.Tanh(),
            )
            # Gate conditioned on both CLS and projected hand features
            self.hf_gate = nn.Linear(h * 2, h)
        # Company embedding: learnable per-company bias on CLS representation
        # n_companies+1 entries: 0..n_companies-1 = known companies, n_companies = unknown/padding
        self.company_emb_dim = company_emb_dim
        if company_emb_dim > 0:
            self.company_emb = nn.Embedding(n_companies + 1, company_emb_dim,
                                            padding_idx=n_companies)
            self.company_proj = nn.Linear(company_emb_dim, h)
        self.head = CascadeHeadV2(h, dropout) if deep_cascade else CascadeHead(h, dropout)
        if use_span:
            # Token-level binary head: is this token part of the evidence_string?
            self.span_head = nn.Linear(h, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                hand_features=None,
                company_id=None,
                inputs_embeds=None) -> dict[str, torch.Tensor]:
        kwargs = dict(attention_mask=attention_mask)
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        else:
            kwargs["input_ids"] = input_ids
            if token_type_ids is not None and self.encoder.config.model_type not in _NO_TOKEN_TYPE_IDS:
                kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        if self.pooling == "attn":
            cls = self.attn_pool(out.last_hidden_state, attention_mask)
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            cls = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            cls = out.last_hidden_state[:, 0]
        if self.use_hand_features and hand_features is not None:
            hf = self.hf_proj(hand_features)                          # (B, h)
            gate = torch.sigmoid(self.hf_gate(torch.cat([cls, hf], dim=-1)))  # (B, h)
            cls = cls + gate * hf                                      # gated residual
        if self.company_emb_dim > 0 and company_id is not None:
            cls = cls + self.company_proj(self.company_emb(company_id))  # additive injection
        logits = self.head(cls)
        logits["cls"] = cls          # expose CLS for supervised contrastive loss
        if self.use_span:
            # span_logits: (B, L) — probability each token is evidence
            logits["span_logits"] = self.span_head(out.last_hidden_state).squeeze(-1)
        return logits


# ── Approach B (LLM hidden state) ─────────────────────────────────────────────

class ApproachB(nn.Module):
    def __init__(self, backbone: str, hidden_dim: int = 256, dropout: float = 0.1,
                 freeze_backbone: bool = True, use_lora: bool = False,
                 lora_r: int = 8, lora_alpha: int = 16,
                 lora_target_modules: Optional[list[str]] = None,
                 proj_hidden_dim: int = 1024) -> None:
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            backbone, torch_dtype=torch.bfloat16, device_map="auto")
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=lora_r, lora_alpha=lora_alpha,
                target_modules=lora_target_modules or ["q_proj", "v_proj"],
                lora_dropout=dropout, bias="none",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
            self.llm.print_trainable_parameters()
        elif freeze_backbone:
            for p in self.llm.parameters():
                p.requires_grad = False
        llm_hidden = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(llm_hidden, proj_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(hidden_dim, dropout=0.0)
        # LLM is loaded with device_map="auto" (→ GPU); move proj/head to same device
        # so projector(last_tok) doesn't hit a cuda/cpu mismatch.
        _dev = next(self.llm.parameters()).device
        self.projector.to(_dev)
        self.head.to(_dev)

    def forward(self, input_ids, attention_mask, **_) -> dict[str, torch.Tensor]:
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        seq_lens = attention_mask.sum(dim=1) - 1
        last_tok = last_hidden[torch.arange(last_hidden.size(0)), seq_lens].float()
        return self.head(self.projector(last_tok))


# ── Approach D (LLM2Vec-style: mean pool + LoRA) ──────────────────────────────

class ApproachD(nn.Module):
    """
    Bidirectional-style LLM encoder via mean pooling + LoRA fine-tuning.
    Inspired by LLM2Vec (McGill-NLP, 2024).

    Key difference vs Approach B (last-token):
      B: last token hidden state — causal, sees only left context at each position
      D: mean pool over ALL token hidden states — captures full sequence symmetrically

    Why mean pool approximates bidirectionality:
      - Each token's hidden state is shaped by its left context (causal attention)
      - Mean pool aggregates ALL positions → effectively sees full sequence
      - Combined with LoRA fine-tuning the attention weights adapt to classify

    True bidirectional (LLM2Vec step 1) requires patching causal mask → all-ones.
    We use mean pool as a practical approximation that avoids model surgery.
    """

    def __init__(
        self,
        backbone: str,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: Optional[list[str]] = None,
        proj_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model, TaskType

        self.base = AutoModel.from_pretrained(
            backbone,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or ["q_proj", "v_proj", "o_proj"],
            lora_dropout=dropout,
            bias="none",
        )
        self.base = get_peft_model(self.base, lora_cfg)
        self.base.print_trainable_parameters()

        hidden = self.base.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden, proj_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(hidden_dim, dropout=0.0)

    def forward(self, input_ids, attention_mask, **_) -> dict[str, torch.Tensor]:
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state.float()               # [B, T, H]
        # Masked mean pool: ignore padding tokens
        mask = attention_mask.unsqueeze(-1).float()          # [B, T, 1]
        mean_repr = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # [B, H]
        return self.head(self.projector(mean_repr))


# ── Approach C (Sentence Embedding + MLP) ────────────────────────────────────

class ApproachC(nn.Module):
    """
    Frozen sentence-embedding model as feature extractor.
    Input: pre-computed embeddings [B, D] from EmbeddingDataset.
    """
    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = MultiTaskHead(hidden_dim, dropout=0.0)

    def forward(self, embedding: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        return self.head(self.mlp(embedding))


def extract_embeddings(texts: list[str], backbone: str,
                       batch_size: int = 32, device: str = "cuda",
                       instruction: str = "") -> torch.Tensor:
    """
    Extract sentence embeddings using sentence-transformers or HuggingFace model.
    Supports BGE-M3, Qwen3-Embedding, multilingual-e5-large, etc.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(backbone, trust_remote_code=True, device=device)
        if instruction:
            texts = [f"{instruction}{t}" for t in texts]
        embeddings = model.encode(texts, batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=True)
        return torch.tensor(embeddings, dtype=torch.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to extract embeddings with {backbone}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperatures = nn.ParameterDict({
            task: nn.Parameter(torch.ones(1)) for task in LABELS
        })

    def forward(self, logits: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {task: logits[task] / self.temperatures[task] for task in logits}

    def fit(self, logits: dict[str, torch.Tensor], labels: dict[str, torch.Tensor],
            lr: float = 0.01, max_iter: int = 50) -> dict[str, float]:
        self.train()
        opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = sum(F.cross_entropy(logits[t] / self.temperatures[t], labels[t]) for t in LABELS)
            loss.backward()
            return loss

        opt.step(closure)
        self.eval()
        return {t: self.temperatures[t].item() for t in LABELS}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING  (Approach A / A1 / C / B)
# ─────────────────────────────────────────────────────────────────────────────

class FGM:
    """Fast Gradient Method adversarial training on the embedding layer."""
    def __init__(self, model: nn.Module, epsilon: float = 0.5):
        self.model = model
        self.epsilon = epsilon
        self._backup: dict = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name and param.grad is not None:
                self._backup[name] = param.data.clone()
                norm = param.grad.norm()
                if norm > 1e-8:
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup.clear()


def _compute_loss(logits, batch, loss_fns, device, log_sigma=None,
                  dyn_weights: dict | None = None,
                  scl_fn=None, span_weight: float = 0.0,
                  mask_t1_no: bool = False,
                  distill_alpha: float = 0.0,
                  distill_temperature: float = 3.0,
                  proto_fn=None,
                  sharp_recl_fn=None,
                  mr2_fn=None) -> torch.Tensor:
    """Compute total loss with optional LogSigma, SCL, span, or distillation losses.

    Standard mode:  L = Σ_t TASK_WEIGHTS[t] * L_t
    LogSigma mode:  L = Σ_t [ 0.5 * exp(-s_t) * L_t + 0.5 * s_t ]
    + span loss:    L += span_weight * BCE(span_logits, span_mask)
    + SCL loss:     L += scl_weight * SCL(dropout(cls), dropout(cls), t3_labels)
    + distill:      L = (1-α)*hard_loss + α*T²*KL(student/T, teacher/T)
    mask_t1_no:     zero T2/T3/T4 loss for samples where T1=No (definitionally N/A)
    """
    weights = dyn_weights if dyn_weights is not None else TASK_WEIGHTS

    if mask_t1_no and log_sigma is not None:
        import warnings
        warnings.warn("mask_t1_no=True is incompatible with log_sigma mode; masking is disabled.")
    if mask_t1_no and "t1" in batch and log_sigma is None:
        t1_labels = batch["t1"].to(device)
        t1_yes = t1_labels == 0  # LABEL2IDX["t1"]["Yes"] = 0
        task_loss = weights["t1"] * loss_fns["t1"](logits["t1"], t1_labels)
        for t in ["t2", "t3", "t4"]:
            if t1_yes.any():
                task_loss = task_loss + weights[t] * loss_fns[t](
                    logits[t][t1_yes], batch[t].to(device)[t1_yes])
    elif log_sigma is not None:
        task_loss = sum(
            0.5 * torch.exp(-log_sigma[t]) * loss_fns[t](logits[t], batch[t].to(device))
            + 0.5 * log_sigma[t]
            for t in ["t1", "t2", "t3", "t4"]
        )
    else:
        task_loss = sum(weights[t] * loss_fns[t](logits[t], batch[t].to(device))
                       for t in ["t1", "t2", "t3", "t4"])

    # Evidence span auxiliary loss
    if span_weight > 0 and "span_logits" in logits and "span_mask" in batch:
        span_loss = F.binary_cross_entropy_with_logits(
            logits["span_logits"], batch["span_mask"].to(device),
        )
        task_loss = task_loss + span_weight * span_loss

    # Supervised contrastive loss on T3 (in-batch, dropout as augmentation)
    if scl_fn is not None and "cls" in logits and "t3" in batch:
        cls = logits["cls"]
        t3_labels = batch["t3"].to(device)
        cls1 = F.dropout(cls, p=0.1, training=True)
        cls2 = F.dropout(cls, p=0.1, training=True)
        scl_loss = scl_fn(cls1, cls2, t3_labels)
        task_loss = task_loss + scl_loss

    # Class-Aware Prototype Contrastive Loss on T3 (arXiv 2410.22197)
    if proto_fn is not None and "cls" in logits and "t3" in batch:
        proto_loss = proto_fn(logits["cls"], batch["t3"].to(device))
        task_loss = task_loss + proto_loss

    # SharpReCL: prototype-guided contrastive rebalancing for T3
    if sharp_recl_fn is not None and "cls" in logits and "t3" in batch:
        sharp_loss = sharp_recl_fn(logits["cls"], batch["t3"].to(device))
        task_loss = task_loss + sharp_loss

    # MR2: adaptive margin regularization for T3
    if mr2_fn is not None and "cls" in logits and "t3" in batch:
        mr2_loss = mr2_fn(logits["t3"], batch["t3"].to(device), logits["cls"])
        task_loss = task_loss + mr2_loss

    # Self-distillation: KL(student/T, teacher/T) for all tasks
    if distill_alpha > 0:
        T = distill_temperature
        kd_loss = torch.tensor(0.0, device=device)
        for t in ["t1", "t2", "t3", "t4"]:
            teacher_key = f"teacher_{t}"
            if teacher_key in batch:
                teacher_probs = batch[teacher_key].to(device)  # (B, n_classes)
                student_log = F.log_softmax(logits[t] / T, dim=-1)
                teacher_soft = (teacher_probs + 1e-8)  # already probabilities
                teacher_soft = teacher_soft / teacher_soft.sum(dim=-1, keepdim=True)
                kd_loss = kd_loss + F.kl_div(student_log, teacher_soft, reduction="batchmean")
        task_loss = (1 - distill_alpha) * task_loss + distill_alpha * (T * T) * kd_loss

    # Conditional chain-rule aux (arXiv 2410.01305): focus the T3 *content* classifier
    # (Clear/NotClear/Misleading) on rows where evidence actually exists (T1=Yes & T2=Yes),
    # instead of diluting it with N/A rows. Keeps the 4-way head intact (downstream unchanged).
    t3cw = loss_fns.get("_t3_cond_w", 0.0) if isinstance(loss_fns, dict) else 0.0
    if t3cw and all(k in batch for k in ("t1", "t2", "t3")):
        t1 = batch["t1"].to(device); t2 = batch["t2"].to(device); t3 = batch["t3"].to(device)
        active = (t1 == 0) & (t2 == 0) & (t3 < 3)   # T1=Yes, T2=Yes, content label (not N/A)
        if active.any():
            task_loss = task_loss + t3cw * F.cross_entropy(
                logits["t3"][active][:, :3], t3[active])

    return task_loss


def _rdrop_kl(logits1: dict, logits2: dict) -> torch.Tensor:
    """Symmetric KL divergence between two forward passes (R-Drop)."""
    kl_loss = torch.tensor(0.0, device=next(iter(logits1.values())).device)
    for t in LABELS:
        p = F.log_softmax(logits1[t], dim=-1)
        q = F.log_softmax(logits2[t], dim=-1)
        kl_loss = kl_loss + 0.5 * (
            F.kl_div(p, q.exp(), reduction="batchmean") +
            F.kl_div(q, p.exp(), reduction="batchmean"))
    return kl_loss


def train_epoch(model, loader, optimizer, scheduler, loss_fns, device, grad_clip,
                fgm: FGM | None = None, log_sigma=None, dyn_weights=None,
                scl_fn=None, span_weight: float = 0.0,
                mask_t1_no: bool = False,
                rdrop_alpha: float = 0.0,
                distill_alpha: float = 0.0,
                distill_temperature: float = 3.0,
                tmix_alpha: float = 0.0,
                hard_mixup_alpha: float = 0.0,
                proto_fn=None,
                sharp_recl_fn=None,
                mr2_fn=None) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            hand_features = batch.get("hand_features")
            if hand_features is not None:
                hand_features = hand_features.to(device)
            company_id = batch.get("company_id")
            if company_id is not None:
                company_id = company_id.to(device)

            # TMix: embedding-level mixup
            if tmix_alpha > 0 and input_ids.size(0) > 1:
                lam = float(np.random.beta(tmix_alpha, tmix_alpha))
                lam = max(lam, 1.0 - lam)   # λ ≥ 0.5: main sample always dominates
                # perm on CPU: used for batch label indexing (CPU tensors)
                perm = torch.randperm(input_ids.size(0))
                perm_dev = perm.to(device)  # device copy: used for GPU tensor indexing
                with torch.no_grad():
                    # Get full token embeddings (word + position + token_type)
                    enc = model.encoder if hasattr(model, 'encoder') else model
                    embeds = enc.embeddings(input_ids=input_ids,
                                            token_type_ids=token_type_ids)
                mixed_embeds = lam * embeds + (1.0 - lam) * embeds[perm_dev].detach()
                mixed_mask = torch.max(attention_mask, attention_mask[perm_dev])
                _fwd = lambda: model(inputs_embeds=mixed_embeds, attention_mask=mixed_mask)
                logits = _fwd()
                loss_a = _compute_loss(logits, batch, loss_fns, device, log_sigma, dyn_weights,
                                       scl_fn=scl_fn, span_weight=span_weight, mask_t1_no=mask_t1_no)
                # batch tensors are on CPU here — use CPU perm for indexing
                batch_b = {k: (v[perm] if isinstance(v, torch.Tensor) else v)
                           for k, v in batch.items()}
                loss_b = _compute_loss(logits, batch_b, loss_fns, device, log_sigma, dyn_weights,
                                       scl_fn=scl_fn, span_weight=span_weight, mask_t1_no=mask_t1_no)
                loss = lam * loss_a + (1.0 - lam) * loss_b
            else:
                _fwd = lambda: model(input_ids, attention_mask, token_type_ids=token_type_ids,
                                     hand_features=hand_features, company_id=company_id)
                logits = _fwd()
                loss = _compute_loss(logits, batch, loss_fns, device, log_sigma, dyn_weights,
                                     scl_fn=scl_fn, span_weight=span_weight, mask_t1_no=mask_t1_no,
                                     distill_alpha=distill_alpha, distill_temperature=distill_temperature,
                                     proto_fn=proto_fn,
                                     sharp_recl_fn=sharp_recl_fn, mr2_fn=mr2_fn)
        else:
            _fwd = lambda: model(embedding=batch["embedding"].to(device))
            logits = _fwd()
            loss = _compute_loss(logits, batch, loss_fns, device, log_sigma, dyn_weights,
                                 scl_fn=scl_fn, span_weight=span_weight, mask_t1_no=mask_t1_no,
                                 distill_alpha=distill_alpha, distill_temperature=distill_temperature,
                                 proto_fn=proto_fn,
                                 sharp_recl_fn=sharp_recl_fn, mr2_fn=mr2_fn)

        # Hard-Mixup: NC↔Clear cross-class boundary mixing (WWW 2026)
        # Targets T3 Not Clear/Clear boundary; only active for input_ids models, not TMix mode.
        if hard_mixup_alpha > 0 and tmix_alpha == 0 and "input_ids" in batch:
            t3_cpu = batch.get("t3")
            if t3_cpu is not None and input_ids.size(0) >= 2:
                NC_IDX, CLEAR_IDX = 1, 0
                nc_pos = (t3_cpu == NC_IDX).nonzero(as_tuple=True)[0]   # batch indices
                cl_pos = (t3_cpu == CLEAR_IDX).nonzero(as_tuple=True)[0]
                if len(nc_pos) > 0 and len(cl_pos) > 0:
                    lam_hm = float(np.random.beta(hard_mixup_alpha, hard_mixup_alpha))
                    lam_hm = max(lam_hm, 1.0 - lam_hm)  # NC dominates (λ ≥ 0.5)
                    n_pairs = min(len(nc_pos), len(cl_pos))
                    sel_nc = nc_pos[torch.randperm(len(nc_pos))[:n_pairs]]  # batch indices
                    sel_cl = cl_pos[torch.randperm(len(cl_pos))[:n_pairs]]
                    with torch.no_grad():
                        enc = model.encoder if hasattr(model, 'encoder') else model
                        embeds_all = enc.embeddings(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids)
                    idx_nc_d = sel_nc.to(device)
                    idx_cl_d = sel_cl.to(device)
                    mixed_embeds = lam_hm * embeds_all[idx_nc_d] + (1.0 - lam_hm) * embeds_all[idx_cl_d]
                    mixed_mask = torch.max(attention_mask[idx_nc_d], attention_mask[idx_cl_d])
                    hm_logits = model(inputs_embeds=mixed_embeds, attention_mask=mixed_mask)
                    batch_a = {k: (v[sel_nc] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                    batch_b = {k: (v[sel_cl] if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                    hm_loss = (lam_hm * _compute_loss(hm_logits, batch_a, loss_fns, device,
                                                      log_sigma, dyn_weights, mask_t1_no=mask_t1_no)
                               + (1.0 - lam_hm) * _compute_loss(hm_logits, batch_b, loss_fns, device,
                                                                 log_sigma, dyn_weights, mask_t1_no=mask_t1_no))
                    loss = loss + hm_loss

        # R-Drop: second forward with different dropout mask, add symmetric KL
        if rdrop_alpha > 0:
            logits2 = _fwd()
            loss2 = _compute_loss(logits2, batch, loss_fns, device, log_sigma, dyn_weights,
                                  scl_fn=scl_fn, span_weight=span_weight, mask_t1_no=mask_t1_no)
            kl = _rdrop_kl(logits, logits2)
            loss = 0.5 * (loss + loss2) + rdrop_alpha * kl

        optimizer.zero_grad()
        loss.backward()

        if fgm is not None:
            fgm.attack()
            adv_logits = _fwd()
            adv_loss = _compute_loss(adv_logits, batch, loss_fns, device, log_sigma, dyn_weights,
                                     scl_fn=scl_fn, span_weight=span_weight, mask_t1_no=mask_t1_no)
            adv_loss.backward()
            fgm.restore()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, apply_rule: bool = True) -> tuple[dict, dict]:
    model.eval()
    all_preds = {t: [] for t in LABELS}
    all_true = {t: [] for t in LABELS}

    for batch in loader:
        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            hand_features = batch.get("hand_features")
            if hand_features is not None:
                hand_features = hand_features.to(device)
            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids,
                           hand_features=hand_features)
        else:
            logits = model(embedding=batch["embedding"].to(device))

        for t in LABELS:
            pred_idx = logits[t].argmax(dim=-1).cpu().tolist()
            all_preds[t].extend([IDX2LABEL[t][i] for i in pred_idx])
            if t in batch:
                all_true[t].extend([IDX2LABEL[t][i] for i in batch[t].tolist()])

    if apply_rule:
        all_preds = apply_na_rule(all_preds)
    scores = compute_weighted_f1(all_true, all_preds) if all_true["t1"] else {}
    return scores, all_preds


@torch.no_grad()
def predict_probs(model, loader, device) -> dict[str, list[list[float]]]:
    """Return per-class softmax probabilities for all samples (for soft voting)."""
    model.eval()
    all_probs: dict[str, list[list[float]]] = {t: [] for t in LABELS}
    for batch in loader:
        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            hand_features = batch.get("hand_features")
            if hand_features is not None:
                hand_features = hand_features.to(device)
            company_id = batch.get("company_id")
            if company_id is not None:
                company_id = company_id.to(device)
            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids,
                           hand_features=hand_features, company_id=company_id)
        else:
            logits = model(embedding=batch["embedding"].to(device))
        for t in LABELS:
            probs = torch.softmax(logits[t], dim=-1).cpu().tolist()
            all_probs[t].extend(probs)
    return all_probs


@torch.no_grad()
def predict_probs_tta(model, loader, device, n: int = 8) -> dict[str, list[list[float]]]:
    """Test-Time Augmentation: run n forward passes with dropout active, average softmax.

    Dropout introduces stochasticity per pass. Averaging n predictions reduces variance
    and improves probability calibration, especially for borderline samples.
    n=8 balances quality vs inference time (~8× slower than single pass).
    """
    # Accumulate sum of softmax probs across n passes
    accum: dict[str, list[list[float]]] | None = None
    was_training = model.training
    try:
        for _ in range(n):
            model.train()  # enable dropout
            pass_probs: dict[str, list[list[float]]] = {t: [] for t in LABELS}
            for batch in loader:
                if "input_ids" in batch:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device)
                    hand_features = batch.get("hand_features")
                    if hand_features is not None:
                        hand_features = hand_features.to(device)
                    company_id = batch.get("company_id")
                    if company_id is not None:
                        company_id = company_id.to(device)
                    with torch.no_grad():
                        logits = model(input_ids, attention_mask, token_type_ids=token_type_ids,
                                       hand_features=hand_features, company_id=company_id)
                else:
                    with torch.no_grad():
                        logits = model(embedding=batch["embedding"].to(device))
                for t in LABELS:
                    probs = torch.softmax(logits[t], dim=-1).cpu().tolist()
                    pass_probs[t].extend(probs)
            if accum is None:
                accum = pass_probs
            else:
                for t in LABELS:
                    for i in range(len(accum[t])):
                        for c in range(NUM_LABELS[t]):
                            accum[t][i][c] += pass_probs[t][i][c]
    finally:
        model.train(was_training)
    # Average
    result: dict[str, list[list[float]]] = {}
    for t in LABELS:
        result[t] = [[v / n for v in row] for row in accum[t]]  # type: ignore[index]
    return result


def train(cfg: Config,
          _train_df: Optional[pd.DataFrame] = None,
          _val_df: Optional[pd.DataFrame] = None,
          _teacher_probs: Optional[dict[str, list[list[float]]]] = None) -> dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_full, _ = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)
    if _train_df is not None and _val_df is not None:
        train_df, val_df = _train_df.reset_index(drop=True), _val_df.reset_index(drop=True)
    else:
        train_df, val_df = split_train_val(train_full, cfg.val_ratio, cfg.seed)

    if cfg.augment_rare:
        orig_len = len(train_df)
        train_df = augment_rare_classes(train_df, cfg.augment_target_n, cfg.seed)
        print(f"  [augment] {orig_len} → {len(train_df)} rows "
              f"(+{len(train_df)-orig_len} rare-class samples)")

    # ── Approach C: pre-compute embeddings ────────────────────────────────
    if cfg.approach in ("C", "C_contrastive"):
        instruction = "為以下ESG報告段落生成用於永續承諾分析的語意嵌入：" \
                      if "Qwen3-Embedding" in cfg.backbone else ""
        print(f"[C] Extracting embeddings with {cfg.backbone} ...")
        train_embs = extract_embeddings(
            train_df["data"].tolist(), cfg.backbone, device=str(device), instruction=instruction)
        val_embs = extract_embeddings(
            val_df["data"].tolist(), cfg.backbone, device=str(device), instruction=instruction)

        embed_dim = train_embs.shape[1]

        if cfg.approach == "C_contrastive":
            train_embs = _contrastive_finetune(
                cfg, train_embs, train_df, device, run_dir)

        train_ds = EmbeddingDataset(train_embs, train_df)
        val_ds = EmbeddingDataset(val_embs, val_df)
        model = ApproachC(embed_dim, cfg.hidden_dim, cfg.dropout).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _use_hf = cfg.use_hand_features and cfg.approach == "A1"
        _use_span = cfg.use_span and cfg.approach == "A1"

        # Few-shot demo: pre-compute retrieval embeddings, then build prefixes
        train_demo_pfx = val_demo_pfx = None
        if getattr(cfg, "use_fewshot_demo", False) and getattr(cfg, "fewshot_ref_ckpt", ""):
            if Path(cfg.fewshot_ref_ckpt).exists():
                print("  [fewshot] computing retrieval embeddings...")
                real_train_df, _ = load_dataframes(cfg.data_dir, use_augmented=False)
                ref_embs = _load_retrieval_embs(cfg.fewshot_ref_ckpt, real_train_df, device)
                retriever = DemoRetriever(ref_embs, real_train_df)
                train_demo_pfx = retriever.build_prefixes(train_df, exclude_self=True)
                val_demo_pfx   = retriever.build_prefixes(val_df,   exclude_self=True)
                del ref_embs, retriever
            else:
                print(f"  [fewshot] WARNING: ref ckpt not found: {cfg.fewshot_ref_ckpt}")

        # Build company2idx from full training data when company_emb_dim > 0
        company2idx: dict | None = None
        if cfg.company_emb_dim > 0 and "company" in train_full.columns:
            companies = sorted(train_full["company"].dropna().unique().tolist())
            company2idx = {c: i for i, c in enumerate(companies)}
            print(f"  [company_emb] {len(company2idx)} companies, dim={cfg.company_emb_dim}")

        _prepend = getattr(cfg, "prepend_linguistic", False)
        train_ds = ESGDataset(train_df, tokenizer, cfg.max_length,
                              use_hand_features=_use_hf, use_span=_use_span,
                              demo_prefixes=train_demo_pfx,
                              company2idx=company2idx,
                              prepend_linguistic=_prepend)
        val_ds = ESGDataset(val_df, tokenizer, cfg.max_length,
                            use_hand_features=_use_hf,
                            demo_prefixes=val_demo_pfx,
                            company2idx=company2idx,
                            prepend_linguistic=_prepend)

        if cfg.approach == "A":
            model = ApproachA(cfg.backbone, cfg.dropout).to(device)
        elif cfg.approach == "A1":
            n_companies = len(company2idx) if company2idx is not None else 51
            model = ApproachA1(cfg.backbone, cfg.dropout,
                               deep_cascade=cfg.deep_cascade,
                               use_hand_features=_use_hf,
                               use_span=_use_span,
                               company_emb_dim=cfg.company_emb_dim,
                               n_companies=n_companies,
                               pooling=cfg.pooling).to(device)
        elif cfg.approach in ("B", "B_lora"):
            model = ApproachB(cfg.backbone, cfg.hidden_dim, cfg.dropout,
                              freeze_backbone=(cfg.approach == "B"),
                              use_lora=cfg.use_lora, lora_r=cfg.lora_r,
                              lora_alpha=cfg.lora_alpha,
                              lora_target_modules=cfg.lora_target_modules,
                              proj_hidden_dim=cfg.proj_hidden_dim)
        elif cfg.approach == "D":
            model = ApproachD(cfg.backbone, cfg.hidden_dim, cfg.dropout,
                              lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                              lora_target_modules=cfg.lora_target_modules,
                              proj_hidden_dim=cfg.proj_hidden_dim)
            model = model.to(device)
        else:
            raise ValueError(f"Unknown approach: {cfg.approach}")

    # Attach teacher soft labels for self-distillation
    if _teacher_probs is not None and cfg.distill_alpha > 0:
        train_ds.teacher_probs = _teacher_probs
        print(f"  [distill] teacher soft labels attached (α={cfg.distill_alpha}, T={cfg.distill_temperature})")

    # num_workers=0 when called from k-fold to avoid /dev/shm exhaustion across folds
    _nw = 0 if _train_df is not None else 2
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=_nw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=_nw)
    loss_fns = build_loss_fns(cfg, train_df, device)

    # ── LogSigma: learnable per-task noise parameters ─────────────────────────
    if cfg.use_logsigma:
        log_sigma = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1, device=device)) for t in ["t1", "t2", "t3", "t4"]
        })
        params_to_opt = (list(filter(lambda p: p.requires_grad, model.parameters()))
                         + list(log_sigma.parameters()))
    else:
        log_sigma = None
        params_to_opt = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.AdamW(params_to_opt, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)

    fgm = FGM(model, epsilon=cfg.fgm_epsilon) if cfg.fgm_epsilon > 0 else None

    # Supervised contrastive loss for T3 (weighted by scl_weight)
    _scl_fn = None
    if cfg.scl_weight > 0 and cfg.approach == "A1":
        _base_scl = SupervisedContrastiveLoss(temperature=cfg.scl_temp).to(device)
        _scl_weight = cfg.scl_weight
        class _WeightedSCL(nn.Module):
            def forward(self, a, p, l):
                return _scl_weight * _base_scl(a, p, l)
        _scl_fn = _WeightedSCL().to(device)

    # Class-Aware Prototype Contrastive Loss for T3
    _proto_fn = None
    if getattr(cfg, 'proto_weight', 0.0) > 0 and cfg.approach == "A1":
        _proto_fn = ClassAwarePrototypeLoss(
            temperature=getattr(cfg, 'proto_temp', 0.07),
            weight=cfg.proto_weight,
        ).to(device)

    # SharpReCL: prototype-guided contrastive rebalancing for T3
    _sharp_recl_fn = None
    if getattr(cfg, 'sharp_recl_weight', 0.0) > 0 and cfg.approach == "A1":
        _hidden = model.encoder.config.hidden_size if hasattr(model, 'encoder') else 1024
        _sharp_recl_fn = SharpReCL(
            hidden_size=_hidden, num_classes=3,
            weight=cfg.sharp_recl_weight,
            mixup_alpha=getattr(cfg, 'sharp_recl_mixup', 0.3),
        ).to(device)

    # MR2: adaptive margin regularization for T3
    _mr2_fn = None
    if getattr(cfg, 'mr2_weight', 0.0) > 0 and cfg.approach == "A1":
        _mr2_fn = MR2Loss(
            num_classes=3,  # Clear/NC/Misleading; N/A excluded
            margin_scale=getattr(cfg, 'mr2_margin_scale', 0.3),
            weight=cfg.mr2_weight,
        ).to(device)

    _span_weight = cfg.span_weight if cfg.use_span and cfg.approach == "A1" else 0.0

    best_score, best_scores = -1.0, {}
    history = []
    no_improve = 0
    dyn_weights = None  # updated each epoch when adaptive_task_weight=True

    # SWA: collect state_dicts from epochs >= swa_start_epoch for weight averaging
    _swa_states: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fns, device,
                              cfg.grad_clip, fgm=fgm, log_sigma=log_sigma,
                              dyn_weights=dyn_weights,
                              scl_fn=_scl_fn, span_weight=_span_weight,
                              mask_t1_no=getattr(cfg, 'mask_t1_no', False),
                              rdrop_alpha=cfg.rdrop_alpha,
                              distill_alpha=cfg.distill_alpha,
                              distill_temperature=cfg.distill_temperature,
                              tmix_alpha=getattr(cfg, 'tmix_alpha', 0.0),
                              hard_mixup_alpha=getattr(cfg, 'hard_mixup_alpha', 0.0),
                              proto_fn=_proto_fn,
                              sharp_recl_fn=_sharp_recl_fn,
                              mr2_fn=_mr2_fn)
        val_scores, _ = evaluate(model, val_loader, device, cfg.apply_na_rule)

        # Update adaptive weights for next epoch based on current val F1
        if cfg.adaptive_task_weight:
            gap = {t: max(0.0, 1.0 - val_scores.get(t, 0.0)) for t in ["t1", "t2", "t3", "t4"]}
            total_gap = sum(gap.values()) or 1.0
            w_adaptive = {t: gap[t] / total_gap for t in gap}
            dyn_weights = {t: cfg.adaptive_alpha * w_adaptive[t]
                           + (1 - cfg.adaptive_alpha) * TASK_WEIGHTS[t]
                           for t in ["t1", "t2", "t3", "t4"]}
        elapsed = time.time() - t0

        sigma_info = ""
        if log_sigma is not None:
            eff = {t: (0.5 * torch.exp(-log_sigma[t])).item() for t in ["t1", "t2", "t3", "t4"]}
            sigma_info = (f" eff_w=({eff['t1']:.3f},{eff['t2']:.3f},"
                          f"{eff['t3']:.3f},{eff['t4']:.3f})")
        elif dyn_weights is not None:
            sigma_info = (f" dyn_w=({dyn_weights['t1']:.3f},{dyn_weights['t2']:.3f},"
                          f"{dyn_weights['t3']:.3f},{dyn_weights['t4']:.3f})")

        row = {"epoch": epoch, "train_loss": tr_loss, "time_s": elapsed, **val_scores}
        history.append(row)
        print(f"[{epoch:02d}/{cfg.epochs}] loss={tr_loss:.4f} "
              f"T1={val_scores.get('t1',0):.3f} T2={val_scores.get('t2',0):.3f} "
              f"T3={val_scores.get('t3',0):.3f} T4={val_scores.get('t4',0):.3f} "
              f"weighted={val_scores.get('weighted',0):.4f} ({elapsed:.1f}s){sigma_info}")

        if val_scores.get("weighted", 0) > best_score:
            best_score = val_scores["weighted"]
            best_scores = val_scores
            no_improve = 0
            log_sigma_state = (
                {t: log_sigma[t].item() for t in log_sigma}
                if log_sigma is not None else None
            )
            torch.save({"model": model.state_dict(), "cfg": cfg, "scores": val_scores,
                        "embed_dim": train_embs.shape[1] if cfg.approach.startswith("C") else None,
                        "log_sigma": log_sigma_state},
                       run_dir / "best.pt")
            print(f"  ↑ best saved (weighted={best_score:.4f})")
        else:
            no_improve += 1
            if cfg.early_stopping_patience > 0 and no_improve >= cfg.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.early_stopping_patience} epochs)")
                break

        # SWA: collect state from late epochs
        if cfg.swa_start_epoch > 0 and epoch >= cfg.swa_start_epoch:
            _swa_states.append({k: v.clone().cpu() for k, v in model.state_dict().items()})

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    # SWA: average collected states, evaluate, save if better
    if _swa_states and len(_swa_states) >= 2:
        print(f"  [SWA] averaging {len(_swa_states)} checkpoints (epochs {cfg.swa_start_epoch}+)...")
        avg_state = {}
        for key in _swa_states[0]:
            avg_state[key] = torch.stack([s[key].float() for s in _swa_states]).mean(dim=0)
            if _swa_states[0][key].dtype != torch.float32:
                avg_state[key] = avg_state[key].to(_swa_states[0][key].dtype)
        model.load_state_dict({k: v.to(device) for k, v in avg_state.items()})
        swa_scores, _ = evaluate(model, val_loader, device, cfg.apply_na_rule)
        swa_weighted = swa_scores.get("weighted", 0)
        print(f"  [SWA] weighted={swa_weighted:.4f} (vs best={best_score:.4f})")
        if swa_weighted > best_score:
            best_score = swa_weighted
            best_scores = swa_scores
            torch.save({"model": model.state_dict(), "cfg": cfg, "scores": swa_scores,
                        "embed_dim": train_embs.shape[1] if cfg.approach.startswith("C") else None,
                        "log_sigma": None},
                       run_dir / "best.pt")
            print(f"  [SWA] ↑ new best saved (weighted={best_score:.4f})")
        del _swa_states, avg_state

    print(f"\n[{cfg.approach}] Best weighted Macro-F1: {best_score:.4f}\n")

    # Explicitly release model to free VRAM for next experiment
    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_scores


def _predict_checkpoint(
    ckpt_path: str,
    test_df: pd.DataFrame,
    tta_n: int = 0,
) -> Optional[tuple[dict[str, list[str]], dict[str, list[list[float]]]]]:
    """Load checkpoint; return (label_preds, prob_preds) or None.

    label_preds: {task: [label, ...]}  — raw argmax labels, no NA rule applied
    prob_preds:  {task: [[p0,p1,...], ...]}  — softmax probabilities for soft voting
    """
    if not Path(ckpt_path).exists():
        print(f"  [WARN] checkpoint not found, skipping: {ckpt_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg: Config = ckpt["cfg"]
    dc = getattr(saved_cfg, "deep_cascade", False)

    if saved_cfg.approach.startswith("C"):
        instruction = "為以下ESG報告段落生成用於永續承諾分析的語意嵌入：" \
                      if "Qwen3-Embedding" in saved_cfg.backbone else ""
        test_embs = extract_embeddings(
            test_df["data"].tolist(), saved_cfg.backbone,
            device=str(device), instruction=instruction)
        test_ds = EmbeddingDataset(test_embs, test_df, has_labels=False)
        model = ApproachC(ckpt["embed_dim"], saved_cfg.hidden_dim, saved_cfg.dropout).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        uhf = getattr(saved_cfg, "use_hand_features", False) and saved_cfg.approach == "A1"
        usp = getattr(saved_cfg, "use_span", False) and saved_cfg.approach == "A1"

        # Few-shot demo: build test prefixes before loading main model (to save VRAM)
        test_demo_pfx = None
        if getattr(saved_cfg, "use_fewshot_demo", False):
            ref_ckpt = getattr(saved_cfg, "fewshot_ref_ckpt", "")
            if ref_ckpt and Path(ref_ckpt).exists():
                real_train_df, _ = load_dataframes(saved_cfg.data_dir, use_augmented=False)
                pool_embs = _load_retrieval_embs(ref_ckpt, real_train_df, device)
                retriever = DemoRetriever(pool_embs, real_train_df)
                test_demo_pfx = retriever.build_prefixes(test_df, exclude_self=False)
                del pool_embs, retriever
                _free_vram()

        # Rebuild company2idx from training data when company_emb_dim > 0
        ckpt_company2idx: dict | None = None
        saved_company_emb_dim = getattr(saved_cfg, "company_emb_dim", 0)
        if saved_company_emb_dim > 0:
            real_train_df, _ = load_dataframes(saved_cfg.data_dir, use_augmented=False)
            if "company" in real_train_df.columns:
                companies = sorted(real_train_df["company"].dropna().unique().tolist())
                ckpt_company2idx = {c: i for i, c in enumerate(companies)}

        # test_ds never uses span masks (no evidence_string at test time)
        _prepend_ckpt = getattr(saved_cfg, "prepend_linguistic", False)
        test_ds = ESGDataset(test_df, tokenizer, saved_cfg.max_length, has_labels=False,
                             use_hand_features=uhf, demo_prefixes=test_demo_pfx,
                             company2idx=ckpt_company2idx,
                             prepend_linguistic=_prepend_ckpt)

        if saved_cfg.approach == "A":
            model = ApproachA(saved_cfg.backbone, saved_cfg.dropout).to(device)
        elif saved_cfg.approach == "A1":
            n_companies = len(ckpt_company2idx) if ckpt_company2idx is not None else 51
            model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc,
                               use_hand_features=uhf, use_span=usp,
                               company_emb_dim=saved_company_emb_dim,
                               n_companies=n_companies,
                               pooling=getattr(saved_cfg, "pooling", "cls")).to(device)
        elif saved_cfg.approach in ("B", "B_lora"):
            model = ApproachB(saved_cfg.backbone, saved_cfg.hidden_dim, saved_cfg.dropout,
                              freeze_backbone=True, proj_hidden_dim=saved_cfg.proj_hidden_dim)
            model.projector = model.projector.to(device)
            model.head = model.head.to(device)
        elif saved_cfg.approach == "D":
            model = ApproachD(saved_cfg.backbone, saved_cfg.hidden_dim, saved_cfg.dropout,
                              lora_r=saved_cfg.lora_r, lora_alpha=saved_cfg.lora_alpha,
                              lora_target_modules=saved_cfg.lora_target_modules,
                              proj_hidden_dim=saved_cfg.proj_hidden_dim)
            model = model.to(device)
        else:
            raise ValueError(f"_predict_checkpoint: unsupported approach '{saved_cfg.approach}'")

    model.load_state_dict(ckpt["model"])
    loader = DataLoader(test_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)

    if tta_n > 1:
        probs = predict_probs_tta(model, loader, device, n=tta_n)
    else:
        probs = predict_probs(model, loader, device)
    labels = {t: [IDX2LABEL[t][int(np.argmax(p))] for p in probs[t]] for t in LABELS}

    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return labels, probs


def train_kfold(cfg: Config) -> None:
    """K-fold cross-validation: trains K models, ensembles test predictions."""
    from sklearn.model_selection import StratifiedKFold
    from dataclasses import fields as dc_fields

    train_full, test_df = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)

    skf = StratifiedKFold(n_splits=cfg.kfold, shuffle=True, random_state=cfg.seed)
    strat_col = train_full["promise_status"]

    # Pre-compute teacher OOF probs for self-distillation
    _full_teacher_probs: dict[str, list] | None = None
    if cfg.distill_alpha > 0 and cfg.distill_oof_dir:
        teacher_seed = 42  # teacher was trained with default seed
        print(f"\n  [distill] extracting teacher OOF probs from {cfg.distill_oof_dir}...")
        _full_teacher_probs = extract_oof_teacher_probs(
            cfg.distill_oof_dir, train_full, n_splits=5, seed=teacher_seed)

    fold_scores: list[dict] = []
    fold_test_preds: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(train_full, strat_col)):
        print(f"\n{'='*60}\n  FOLD {fold_idx+1}/{cfg.kfold}\n{'='*60}")

        fold_cfg = Config(**{f.name: getattr(cfg, f.name) for f in dc_fields(cfg)})
        fold_cfg.run_dir = f"{cfg.run_dir}/fold{fold_idx+1}"
        fold_cfg.seed = cfg.seed + fold_idx
        fold_cfg.kfold = 1  # prevent recursion

        # Skip if checkpoint already exists
        ckpt_path = f"{fold_cfg.run_dir}/best.pt"
        if Path(ckpt_path).exists():
            print(f"  [SKIP] fold{fold_idx+1} already trained ({ckpt_path})")
            result = _predict_checkpoint(ckpt_path, test_df)
            if result is not None:
                fold_test_preds.append(result[1])
            continue

        fold_train = train_full.iloc[tr_idx].reset_index(drop=True)
        fold_val = train_full.iloc[val_idx].reset_index(drop=True)

        # Slice teacher probs for this fold's training indices
        fold_teacher = None
        if _full_teacher_probs is not None:
            fold_teacher = {t: [_full_teacher_probs[t][i] for i in tr_idx] for t in LABELS}

        scores = train(fold_cfg, _train_df=fold_train, _val_df=fold_val,
                        _teacher_probs=fold_teacher)
        fold_scores.append(scores)

        ckpt_path = f"{fold_cfg.run_dir}/best.pt"
        result = _predict_checkpoint(ckpt_path, test_df)
        if result is not None:
            fold_test_preds.append(result[1])  # probs for soft voting
        _free_vram()

    # Soft-vote ensemble across folds, then apply NA rule
    final_preds = ensemble_preds_soft(fold_test_preds)
    if cfg.apply_na_rule:
        final_preds = apply_na_rule(final_preds,
                                    remap_misleading=getattr(cfg, 'remap_misleading', False))

    out_path = Path(cfg.run_dir) / "submission_kfold.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_df["id"],
                  "label": preds_to_label_strings(final_preds)}).to_csv(out_path, index=False)

    weights = [s.get("weighted", 0) for s in fold_scores]
    mean_w, std_w = float(np.mean(weights)), float(np.std(weights))
    print(f"\n{cfg.kfold}-Fold CV: {mean_w:.4f} ± {std_w:.4f}")
    for i, s in enumerate(fold_scores):
        print(f"  Fold {i+1}: {s.get('weighted',0):.4f}")
    print(f"Submission saved → {out_path}")


def _contrastive_finetune(cfg: Config, train_embs: torch.Tensor,
                           train_df: pd.DataFrame, device: torch.device,
                           run_dir: Path) -> torch.Tensor:
    """Fine-tune a linear projection on top of frozen embeddings with supervised contrastive loss."""
    print("[C_contrastive] Supervised contrastive fine-tuning ...")
    embed_dim = train_embs.shape[1]
    proj = nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.ReLU(),
        nn.Linear(embed_dim, embed_dim),
    ).to(device)
    loss_fn = SupervisedContrastiveLoss(temperature=cfg.contrastive_temp)
    opt = torch.optim.AdamW(proj.parameters(), lr=cfg.contrastive_lr)

    dataset = ContrastivePairDataset(train_embs, train_df)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    proj.train()
    for epoch in range(1, cfg.contrastive_epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            anchors = proj(batch["anchor"].to(device))
            positives = proj(batch["positive"].to(device))
            labels = batch["label"].to(device)
            loss = loss_fn(anchors, positives, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"  contrastive epoch {epoch}/{cfg.contrastive_epochs} loss={epoch_loss/len(loader):.4f}")

    # Apply learned projection to embeddings
    proj.eval()
    with torch.no_grad():
        new_embs = proj(train_embs.to(device)).cpu()
    return new_embs


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GENERATIVE APPROACH E  (Qwen3 with thinking mode)
# ─────────────────────────────────────────────────────────────────────────────

def run_generative(cfg: Config) -> dict[str, float]:
    """
    Approach E: Qwen3-8B-Instruct with thinking mode.
    Generates JSON output for each test sample, returns val scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E] Loading {cfg.backbone} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.backbone, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    train_full, test_df = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)
    _, val_df = split_train_val(train_full, cfg.val_ratio, cfg.seed)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    def predict_one(text: str) -> dict[str, str]:
        messages = [
            {"role": "system", "content": GEN_SYSTEM_PROMPT},
            {"role": "user", "content": GEN_USER_TEMPLATE.format(text=text)},
        ]
        # Qwen3 thinking mode: pass enable_thinking to apply_chat_template (tokenize=False),
        # then tokenize separately to get a plain tensor (avoids BatchEncoding keys leaking
        # into generate() as model_kwargs).
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
        input_ids = tokenizer(chat_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=cfg.gen_max_new_tokens,
                temperature=cfg.gen_temperature,
                do_sample=cfg.gen_temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
            out_ids = model.generate(input_ids, **gen_kwargs)

        new_tokens = out_ids[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_gen_output(output_text)

    # ── Evaluate on validation set ────────────────────────────────────────
    print(f"[E] Running inference on {len(val_df)} validation samples ...")
    val_preds = {t: [] for t in LABELS}
    val_true = {t: [] for t in LABELS}

    for _, row in val_df.iterrows():
        pred = predict_one(row["data"])
        val_preds["t1"].append(pred["promise_status"])
        val_preds["t2"].append(pred["evidence_status"])
        val_preds["t3"].append(pred["evidence_quality"])
        val_preds["t4"].append(pred["verification_timeline"])
        val_true["t1"].append(row["promise_status"])
        val_true["t2"].append(row["evidence_status"])
        val_true["t3"].append(row["evidence_quality"])
        val_true["t4"].append(row["verification_timeline"])

    if cfg.apply_na_rule:
        val_preds = apply_na_rule(val_preds)

    val_scores = compute_weighted_f1(val_true, val_preds)
    print(f"[E] Val weighted F1: {val_scores['weighted']:.4f}")

    # ── Generate test submission ──────────────────────────────────────────
    print(f"[E] Running inference on {len(test_df)} test samples ...")
    test_preds = {t: [] for t in LABELS}
    for _, row in test_df.iterrows():
        pred = predict_one(row["data"])
        test_preds["t1"].append(pred["promise_status"])
        test_preds["t2"].append(pred["evidence_status"])
        test_preds["t3"].append(pred["evidence_quality"])
        test_preds["t4"].append(pred["verification_timeline"])

    if cfg.apply_na_rule:
        test_preds = apply_na_rule(test_preds)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": preds_to_label_strings(test_preds),
    })
    out_path = run_dir / "submission_E.csv"
    submission.to_csv(out_path, index=False)
    print(f"[E] Submission saved → {out_path}")
    return val_scores


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PREDICTION / SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    saved_cfg: Config = ckpt["cfg"]

    _, test_df = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)

    if saved_cfg.approach.startswith("C"):
        instruction = "為以下ESG報告段落生成用於永續承諾分析的語意嵌入：" \
                      if "Qwen3-Embedding" in saved_cfg.backbone else ""
        test_embs = extract_embeddings(
            test_df["data"].tolist(), saved_cfg.backbone,
            device=str(device), instruction=instruction)
        test_ds = EmbeddingDataset(test_embs, test_df, has_labels=False)
        model = ApproachC(ckpt["embed_dim"], saved_cfg.hidden_dim, saved_cfg.dropout).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        test_ds = ESGDataset(test_df, tokenizer, saved_cfg.max_length, has_labels=False)
        if saved_cfg.approach == "A":
            model = ApproachA(saved_cfg.backbone, saved_cfg.dropout).to(device)
        elif saved_cfg.approach == "A1":
            model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout).to(device)
        else:
            model = ApproachB(saved_cfg.backbone, saved_cfg.hidden_dim, saved_cfg.dropout,
                              proj_hidden_dim=getattr(saved_cfg, "proj_hidden_dim", 1024))
            model.projector = model.projector.to(device)
            model.head = model.head.to(device)

    model.load_state_dict(ckpt["model"])
    test_loader = DataLoader(test_ds, batch_size=saved_cfg.batch_size, shuffle=False)
    _, preds = evaluate(model, test_loader, device, cfg.apply_na_rule)

    out_path = cfg.checkpoint.replace(".pt", "_submission.csv")
    pd.DataFrame({"id": test_df["id"], "label": preds_to_label_strings(preds)}) \
      .to_csv(out_path, index=False)
    print(f"Submission saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  RUN ALL EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    # ── Baseline (keep for comparison) ──────────────────────────────────────
    ("A1_macbert_ordinal",      "A1", "hfl/chinese-macbert-large",
     {"loss_type": "ordinal"}),                                        # prev best 0.676

    # ── New best config: per-task loss + augmentation + early stopping ──────
    ("A1_macbert_pertask_aug",  "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True}),

    # ── Ablations ───────────────────────────────────────────────────────────
    ("A1_macbert_pertask",      "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True}),                                         # pertask only, no aug

    ("A1_macbert_aug_ordinal",  "A1", "hfl/chinese-macbert-large",
     {"loss_type": "ordinal", "augment_rare": True}),                  # ordinal + aug

    # ── LLM mean-pool + LoRA + per-task loss (Approach D) ───────────────────
    ("D_qwen3_8b_pertask_aug",  "D",  "Qwen/Qwen3-8B",
     {"lora_r": 16, "lora_alpha": 32, "batch_size": 4, "epochs": 8,
      "per_task_loss": True, "augment_rare": True}),

    ("D_qwen3_8b_pertask",      "D",  "Qwen/Qwen3-8B",
     {"lora_r": 16, "lora_alpha": 32, "batch_size": 4, "epochs": 8,
      "per_task_loss": True}),

    # ── V4: Extended cascade (T1→T2, [T1,T2]→T3, [T1]→T4) ─────────────────
    ("A1_macbert_dc_pertask",     "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "deep_cascade": True}),

    ("A1_macbert_dc_pertask_aug", "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True}),

    # ── V4: RoBERTa-large + extended cascade ─────────────────────────────────
    ("A1_roberta_dc_pertask",     "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "deep_cascade": True}),

    ("A1_roberta_dc_pertask_aug", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True}),
]


def _free_vram() -> None:
    """Force-release all PyTorch VRAM cache between experiments."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free = torch.cuda.mem_get_info()[0] / 1e9
        print(f"  [VRAM] freed cache → {free:.1f} GB free")


def run_all(base_cfg: Config) -> None:
    results = []
    for name, approach, backbone, extra in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {name}")
        print(f"{'='*60}")
        # Base config from CLI, then experiment-specific overrides
        cfg = Config(
            approach=approach,
            backbone=backbone,
            run_dir=f"runs/{name}",
            epochs=base_cfg.epochs,
            batch_size=base_cfg.batch_size,
            loss_type=base_cfg.loss_type,
            seed=base_cfg.seed,
        )
        for k, v in extra.items():
            setattr(cfg, k, v)
        try:
            scores = train(cfg)
            results.append({"experiment": name, "approach": approach,
                            "backbone": backbone, **scores})
        except Exception as e:
            print(f"  [SKIP] {name} failed: {e}")
            results.append({"experiment": name, "approach": approach,
                            "backbone": backbone, "weighted": -1, "error": str(e)})
        finally:
            _free_vram()  # always release between experiments

    # Approach E (generative) — separate flow
    if torch.cuda.is_available():
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: E_qwen3_generative")
        print(f"{'='*60}")
        gen_cfg = Config(backbone="Qwen/Qwen3-8B", run_dir="runs/E_qwen3", seed=base_cfg.seed)
        try:
            scores = run_generative(gen_cfg)
            results.append({"experiment": "E_qwen3_generative", "approach": "E",
                            "backbone": "Qwen/Qwen3-8B", **scores})
        except Exception as e:
            print(f"  [SKIP] E_qwen3_generative failed: {e}")

    # ── Summary table ────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("weighted", ascending=False)
    summary_path = Path("runs/summary.csv")
    summary_path.parent.mkdir(exist_ok=True)
    df_results.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    cols = ["experiment", "t1", "t2", "t3", "t4", "weighted"]
    print(df_results[[c for c in cols if c in df_results.columns]].to_string(index=False))
    print(f"\nSaved → {summary_path}")

    # ── Best-model ensemble ───────────────────────────────────────────────
    top3 = df_results[df_results["weighted"] > 0].head(3)["experiment"].tolist()
    if len(top3) >= 2:
        print(f"\n[Ensemble] Top-{len(top3)} models: {top3}")
        print("  (run 'python esg_main.py --mode predict' per checkpoint, then ensemble manually)")


V4_EXPERIMENTS = [
    ("A1_macbert_dc_pertask",     "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "deep_cascade": True}),
    ("A1_macbert_dc_pertask_aug", "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True}),
    ("A1_roberta_dc_pertask",     "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "deep_cascade": True}),
    ("A1_roberta_dc_pertask_aug", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True}),
    # LERT-large: same size as macBERT-large (1024/24L) but with linguistic pre-training
    ("A1_lert_dc_pertask",        "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "deep_cascade": True}),
    ("A1_lert_dc_pertask_aug",    "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True}),
    # DeBERTa-v2-97M: better architecture (disentangled attn), already cached
    ("A1_deberta97m_dc_pertask",  "A1", "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese",
     {"per_task_loss": True, "deep_cascade": True}),
    ("A1_deberta97m_dc_aug",      "A1", "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True}),
]

V4_KFOLD_EXPERIMENTS = [
    # Run 5-fold CV for the best V4 configs
    ("A1_macbert_dc_kfold",  "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5}),
    ("A1_roberta_dc_kfold",  "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5}),
    ("A1_lert_dc_kfold",     "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5}),
    # V31: span auxiliary loss + supervised contrastive loss for T3
    # Uses aug data (969 samples with evidence_string labels); span mask from evidence_string
    ("A1_roberta_dc_kfold_span_scl", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "use_span": True, "span_weight": 0.1,
      "scl_weight": 0.05, "scl_temp": 0.07}),
    ("A1_lert_dc_kfold_span_scl",    "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "use_span": True, "span_weight": 0.1,
      "scl_weight": 0.05, "scl_temp": 0.07}),
    # V33: 10-fold CV on best aug config (RoBERTa + LERT, 969 samples)
    # Each fold trains on ~872 samples vs ~727 in 5-fold → more data per fold
    ("A1_roberta_dc_kfold10_llmaug", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 10,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv"}),
    ("A1_lert_dc_kfold10_llmaug",    "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 10,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv"}),
    # V34: MacBERT-large + LLM aug, 5-fold (same recipe as best RoBERTa/LERT)
    ("A1_macbert_dc_kfold_llmaug",   "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv"}),
    # V36: Pseudo-labeling — aug + high-confidence test predictions (thr=0.80)
    # Run gen_pseudo_labels first, then merge: cat train_data_augmented.csv pseudo_labeled_test.csv
    ("A1_roberta_dc_kfold_pseudo",   "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_pseudo.csv"}),
    ("A1_lert_dc_kfold_pseudo",      "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_pseudo.csv"}),
    # V38: Few-shot demonstration — same-company top-2 similar examples prepended to input
    # Reference embeddings from RoBERTa aug fold1; training data excludes Misleading aug
    ("A1_roberta_dc_kfold_fewshot",  "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_nomislead.csv",
      "use_fewshot_demo": True,
      "fewshot_ref_ckpt": "runs/A1_roberta_dc_kfold_llmaug/fold1/best.pt"}),
    ("A1_lert_dc_kfold_fewshot",     "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_nomislead.csv",
      "use_fewshot_demo": True,
      "fewshot_ref_ckpt": "runs/A1_roberta_dc_kfold_llmaug/fold1/best.pt"}),
    # V39: Masked gradient — zero T2/T3/T4 loss for T1=No samples (N/A by definition)
    # Theory: 149 T1=No samples (19%) add noise to T3 head; masking gives cleaner T3 signal
    ("A1_roberta_dc_kfold_masked",   "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "mask_t1_no": True}),
    ("A1_lert_dc_kfold_masked",      "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "mask_t1_no": True}),
    # V40: T3 Not Clear boost via CE with 2x Not Clear class weight
    # Clear >> Not Clear imbalance (test: 14.2% NC vs train: 18.3% NC); boost to fix underprediction
    ("A1_roberta_dc_kfold_t3nc2",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 2.0}),
    ("A1_lert_dc_kfold_t3nc2",       "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 2.0}),
    ("A1_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0}),
    ("A1_lert_dc_kfold_t3nc3",       "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0}),
    # V47: seed ensemble — same nc3 config, different kfold splits (seed=42 already trained)
    ("A1_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 123}),
    ("A1_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 456}),
    # V46: T3 focal loss + NC class weight boost (focal down-weights easy Clear, NC weight targets imbalance)
    ("A1_roberta_dc_kfold_t3nc3_focal1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "t3_loss_type": "focal", "focal_gamma": 1.0}),
    ("A1_roberta_dc_kfold_t3nc3_focal2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "t3_loss_type": "focal", "focal_gamma": 2.0}),
    ("A1_lert_dc_kfold_t3nc3_focal2",   "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "t3_loss_type": "focal", "focal_gamma": 2.0}),
    # V48: More seeds (5-seed ensemble), R-Drop, SWA, lert seeds, self-distill
    ("A1_roberta_dc_kfold_t3nc3_s3", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 789}),
    ("A1_roberta_dc_kfold_t3nc3_s4", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 999}),
    # V48b: LERT nc3 seeds for cross-backbone ensemble
    ("A1_lert_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 123}),
    ("A1_lert_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 456}),
    # V48c: R-Drop regularization (α=1.0) on nc3 roberta
    ("A1_roberta_dc_kfold_t3nc3_rdrop", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "rdrop_alpha": 1.0}),
    # V48d: SWA (average checkpoints from epoch 7+) on nc3 roberta
    ("A1_roberta_dc_kfold_t3nc3_swa", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "swa_start_epoch": 7}),
    # V48e: Self-distillation from nc3 teacher (seed=42)
    ("A1_roberta_dc_kfold_t3nc3_distill", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "distill_alpha": 0.5, "distill_temperature": 3.0,
      "distill_oof_dir": "runs/A1_roberta_dc_kfold_t3nc3"}),
    # V50: TMix — embedding-level mixup on nc3 roberta
    # α=0.4: Beta(0.4,0.4) → mostly one sample; α=1.0: uniform mix (stronger augmentation)
    ("A1_roberta_dc_kfold_t3nc3_tmix04", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "tmix_alpha": 0.4}),
    ("A1_roberta_dc_kfold_t3nc3_tmix10", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "tmix_alpha": 1.0}),
    # V51: TAPT backbone — uses TAPT-adapted roberta as backbone for nc3 training
    # Step 1: python esg_main.py --mode tapt --backbone hfl/chinese-roberta-wwm-ext-large
    #          --run_dir runs/tapt_roberta --epochs 3 --lr 5e-5 --batch_size 16
    # Step 2: run_v4 trains this entry using the adapted backbone
    ("A1_roberta_dc_kfold_t3nc3_tapt", "A1", "runs/tapt_roberta",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0}),
    # V41: DeBERTa-v2-710M-Chinese (Erlangshen, 710M params — largest Chinese DeBERTa, cached)
    # batch_size=2: disentangled attention OOMs at bs=8 on 32GB VRAM
    ("A1_deberta710m_dc_kfold_llmaug", "A1", "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "batch_size": 2}),
    # V42: DeBERTa-v2-320M-Chinese (Erlangshen, 320M — larger than 97M we tried, cached)
    # batch_size=4: 320M > 97M, reduce to avoid OOM
    ("A1_deberta320m_dc_kfold_llmaug", "A1", "IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "batch_size": 4}),
    # V42b: PERT-large (Permuted Language Model Training, different pre-training from MLM, cached)
    ("A1_pert_dc_kfold_llmaug",        "A1", "hfl/chinese-pert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv"}),
    # V44: T3 nc_weight=4.0/5.0 — push further from nc3 new best (s201=0.69308)
    ("A1_roberta_dc_kfold_t3nc4",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 4.0}),
    ("A1_lert_dc_kfold_t3nc4",       "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 4.0}),
    ("A1_roberta_dc_kfold_t3nc5",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 5.0}),
    # V43: Misleading → Not Clear (remap in training data + inference post-process)
    # Not Clear: 200+30=230 aug; model only sees 3 T3 classes effectively
    ("A1_roberta_dc_kfold_mislead_nc", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_mislead_as_nc.csv",
      "remap_misleading": True}),
    ("A1_lert_dc_kfold_mislead_nc",    "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_mislead_as_nc.csv",
      "remap_misleading": True}),
    # V53: Hard NC augmentation (LLM2LLM-style error-driven generation)
    # Step 1: python gen_hard_nc.py [--n_iters 1]  → generates train_data_aug_hardnc.csv
    # Step 2: run_v4 trains these entries on the hard-NC-augmented dataset
    ("A1_roberta_dc_kfold_t3nc3_hardnc", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_hardnc.csv",
      "t3_nc_weight": 3.0}),
    ("A1_lert_dc_kfold_t3nc3_hardnc",   "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_hardnc.csv",
      "t3_nc_weight": 3.0}),
    # V53b: Hard NC data + nc_weight=1.0 (data already rebalances, no extra NC boost)
    ("A1_roberta_dc_kfold_hardnc_nw1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_hardnc.csv",
      "t3_nc_weight": 1.0}),
    ("A1_lert_dc_kfold_hardnc_nw1",   "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_hardnc.csv",
      "t3_nc_weight": 1.0}),
    # V54: nomislead (939 rows, no synthetic Misleading) + nc_weight=3.0
    # V26 tested nomislead WITHOUT nc3 weight → 0.676. Never tested with nc3.
    ("A1_roberta_dc_kfold_nomislead_nc3", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_aug_nomislead.csv",
      "t3_nc_weight": 3.0}),
    # V57: Label smoothing 0.1 on CE loss (all tasks except T4=ordinal)
    ("A1_roberta_dc_kfold_t3nc3_ls01", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "ce_label_smoothing": 0.1}),
    # ── 2nd Stage: train on official 1602-row dataset (2nd_data/) ──────────────
    # Use --data_dir 2nd_data when running. No LLM aug needed (data is 2x larger).
    # NC ratio: 180/1602=11.2%, Misleading=2 → nc3 loss still necessary.
    # Seed ensemble: s0=42, s1=123, s2=456 (same as 1st stage, proven effective)
    ("B1_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 42}),
    ("B1_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 123}),
    ("B1_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 456}),
    # ── 3rd Stage: train on reshuffled 1601-row dataset (3rd_data/) ──────────────
    # T4 class name reverted to longer_than_5_years. 171 new test samples.
    ("C1_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 42}),
    ("C1_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 123}),
    ("C1_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 456}),
    # ── D1: LLM-augmented training (3rd_data + llm_augmented.csv) ────────────
    ("D1_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 42}),
    ("D1_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 123}),
    ("D1_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 3.0, "seed": 456}),
    # ── D1 nc5: LLM-aug + stronger NC class weight ──────────────────────────
    ("D1_roberta_dc_kfold_t3nc5",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 5.0, "seed": 42}),
    ("D1_roberta_dc_kfold_t3nc5_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 5.0, "seed": 123}),
    ("D1_roberta_dc_kfold_t3nc5_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": True, "aug_filename": "train_data_augmented.csv",
      "t3_nc_weight": 5.0, "seed": 456}),

    # E2: Hard-Mixup (WWW 2026) — NC↔Clear cross-class boundary mixing, alpha=1.0
    # Base: C1 config (dc, nc3) + hard_mixup_alpha=1.0; 3 seeds × 5-fold
    ("E2_roberta_dc_kfold_t3nc3_hm",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "t3_nc_weight": 3.0, "hard_mixup_alpha": 1.0, "seed": 42}),
    ("E2_roberta_dc_kfold_t3nc3_hm_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "t3_nc_weight": 3.0, "hard_mixup_alpha": 1.0, "seed": 123}),
    ("E2_roberta_dc_kfold_t3nc3_hm_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "t3_nc_weight": 3.0, "hard_mixup_alpha": 1.0, "seed": 456}),

    # F3: Class-Aware Prototype Contrastive Loss for T3 (arXiv 2410.22197)
    # Base: C1 config (dc, nc3) + proto_weight=0.10; 3 seeds × 5-fold
    ("F3_roberta_dc_kfold_t3nc3_proto",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "t3_nc_weight": 3.0, "proto_weight": 0.10, "seed": 42}),
    ("F3_roberta_dc_kfold_t3nc3_proto_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "t3_nc_weight": 3.0, "proto_weight": 0.10, "seed": 123}),
    ("F3_roberta_dc_kfold_t3nc3_proto_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "t3_nc_weight": 3.0, "proto_weight": 0.10, "seed": 456}),

    # ── G1: MacBERT at C1 stage — same settings as C1_roberta, new backbone ──
    # MacBERT-large addresses BERT's [MASK]-mismatch issue; strong on CLUE benchmark
    ("G1_macbert_dc_kfold_t3nc3",    "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 42}),
    ("G1_macbert_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 123}),
    ("G1_macbert_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 456}),

    # ── G2: LERT at C1 stage — linguistically-enhanced, strong on Chinese NLU ──
    ("G2_lert_dc_kfold_t3nc3",    "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 42}),
    ("G2_lert_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 123}),
    ("G2_lert_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-lert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "seed": 456}),

    # ── G3: RoBERTa + SupCon T3 (SCL) at C1 stage ───────────────────────────
    # SCL pulls same-class T3 embeddings together, improving NC/Clear boundary
    ("G3_roberta_dc_kfold_t3nc3_scl",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "scl_weight": 0.05, "seed": 42}),
    ("G3_roberta_dc_kfold_t3nc3_scl_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "scl_weight": 0.05, "seed": 123}),
    ("G3_roberta_dc_kfold_t3nc3_scl_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "scl_weight": 0.05, "seed": 456}),

    # ── G4: MacBERT + SupCon T3 at C1 stage ─────────────────────────────────
    ("G4_macbert_dc_kfold_t3nc3_scl",    "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "scl_weight": 0.05, "seed": 42}),
    ("G4_macbert_dc_kfold_t3nc3_scl_s1", "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "scl_weight": 0.05, "seed": 123}),
    ("G4_macbert_dc_kfold_t3nc3_scl_s2", "A1", "hfl/chinese-macbert-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "scl_weight": 0.05, "seed": 456}),

    # ── G5: RoBERTa + company_emb_dim=64 at C1 stage ─────────────────────────
    ("G5_roberta_dc_kfold_t3nc3_cemb64",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "company_emb_dim": 64, "seed": 42}),
    ("G5_roberta_dc_kfold_t3nc3_cemb64_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "company_emb_dim": 64, "seed": 123}),
    ("G5_roberta_dc_kfold_t3nc3_cemb64_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "company_emb_dim": 64, "seed": 456}),

    # ── H1: RoBERTa + Attention Pooling (SemEval-2025 best practice) ──────────
    ("H1_roberta_dc_kfold_t3nc3_attn",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "pooling": "attn", "seed": 42}),
    ("H1_roberta_dc_kfold_t3nc3_attn_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "pooling": "attn", "seed": 123}),
    ("H1_roberta_dc_kfold_t3nc3_attn_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "pooling": "attn", "seed": 456}),

    # ── H2: RoBERTa + Feature Prepending (linguistic markers) ─────────────────
    ("H2_roberta_dc_kfold_t3nc3_prep",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "prepend_linguistic": True, "seed": 42}),
    ("H2_roberta_dc_kfold_t3nc3_prep_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "prepend_linguistic": True, "seed": 123}),
    ("H2_roberta_dc_kfold_t3nc3_prep_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "prepend_linguistic": True, "seed": 456}),

    # ── H3: RoBERTa + Attention Pooling + Feature Prepending (combined) ───────
    ("H3_roberta_dc_kfold_t3nc3_attn_prep",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "pooling": "attn", "prepend_linguistic": True, "seed": 42}),
    ("H3_roberta_dc_kfold_t3nc3_attn_prep_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "pooling": "attn", "prepend_linguistic": True, "seed": 123}),
    ("H3_roberta_dc_kfold_t3nc3_attn_prep_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "pooling": "attn", "prepend_linguistic": True, "seed": 456}),

    # ── C1R: C1 + R-Drop + SWA (YNU-HPCC SemEval-2025, NeurIPS 2021) ────────
    ("C1R_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "rdrop_alpha": 0.5,
      "swa_start_epoch": 7, "seed": 42}),
    ("C1R_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "rdrop_alpha": 0.5,
      "swa_start_epoch": 7, "seed": 123}),
    ("C1R_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "rdrop_alpha": 0.5,
      "swa_start_epoch": 7, "seed": 456}),

    # ── C1S: C1 + SharpReCL prototype-guided contrastive (arxiv 2405.11524) ──
    ("C1S_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "sharp_recl_weight": 0.10,
      "seed": 42}),
    ("C1S_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "sharp_recl_weight": 0.10,
      "seed": 123}),
    ("C1S_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "sharp_recl_weight": 0.10,
      "seed": 456}),

    # ── C1M: C1 + MR2 adaptive margin (arxiv 2602.00205) ────────────────────
    ("C1M_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "mr2_weight": 0.05,
      "seed": 42}),
    ("C1M_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "mr2_weight": 0.05,
      "seed": 123}),
    ("C1M_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "mr2_weight": 0.05,
      "seed": 456}),

    # ── C1RS: C1 + R-Drop + SharpReCL combined ──────────────────────────────
    ("C1RS_roberta_dc_kfold_t3nc3",    "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "rdrop_alpha": 0.5,
      "swa_start_epoch": 7, "sharp_recl_weight": 0.10, "seed": 42}),
    ("C1RS_roberta_dc_kfold_t3nc3_s1", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "rdrop_alpha": 0.5,
      "swa_start_epoch": 7, "sharp_recl_weight": 0.10, "seed": 123}),
    ("C1RS_roberta_dc_kfold_t3nc3_s2", "A1", "hfl/chinese-roberta-wwm-ext-large",
     {"per_task_loss": True, "augment_rare": True, "deep_cascade": True, "kfold": 5,
      "use_augmented": False, "t3_nc_weight": 3.0, "rdrop_alpha": 0.5,
      "swa_start_epoch": 7, "sharp_recl_weight": 0.10, "seed": 456}),
]


def run_v4(base_cfg: Config) -> None:
    """Run V4 experiments: extended cascade + RoBERTa, then k-fold for best configs."""
    from dataclasses import fields as dc_fields

    results = []
    for name, approach, backbone, extra in V4_EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  V4 EXPERIMENT: {name}")
        print(f"{'='*60}")
        # Skip if already trained
        if Path(f"runs/{name}/best.pt").exists():
            print(f"  [SKIP] already trained (runs/{name}/best.pt exists)")
            continue
        cfg = Config(approach=approach, backbone=backbone, run_dir=f"runs/{name}",
                     epochs=base_cfg.epochs, batch_size=base_cfg.batch_size,
                     seed=base_cfg.seed, data_dir=base_cfg.data_dir)
        for k, v in extra.items():
            setattr(cfg, k, v)
        try:
            scores = train(cfg)
            results.append({"experiment": name, **scores})
        except Exception as e:
            print(f"  [SKIP] {name} failed: {e}")
            results.append({"experiment": name, "weighted": -1, "error": str(e)})
        finally:
            _free_vram()

    # Print summary
    if results:
        df = pd.DataFrame(results).sort_values("weighted", ascending=False)
        print(f"\n{'='*60}")
        print("  V4 RESULTS SUMMARY")
        print(f"{'='*60}")
        cols = ["experiment", "t1", "t2", "t3", "t4", "weighted"]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        df.to_csv("runs/v4_summary.csv", index=False)

    # Run kfold for V4 configs
    print(f"\n{'='*60}")
    print("  V4 K-FOLD EXPERIMENTS")
    print(f"{'='*60}")
    for name, approach, backbone, extra in V4_KFOLD_EXPERIMENTS:
        # Skip entirely if all folds already done
        n_folds_expected = extra.get("kfold", 5)
        all_done = all(Path(f"runs/{name}/fold{i}/best.pt").exists()
                       for i in range(1, n_folds_expected + 1))
        if all_done:
            print(f"\n  [SKIP] {name}: all {n_folds_expected} folds already trained")
            continue
        print(f"\n{'='*60}")
        print(f"  V4 K-FOLD: {name}")
        print(f"{'='*60}")
        cfg = Config(approach=approach, backbone=backbone, run_dir=f"runs/{name}",
                     epochs=base_cfg.epochs, batch_size=base_cfg.batch_size,
                     seed=base_cfg.seed, data_dir=base_cfg.data_dir)
        for k, v in extra.items():
            setattr(cfg, k, v)
        try:
            train_kfold(cfg)
        except Exception as e:
            print(f"  [SKIP] {name} kfold failed: {e}")
        finally:
            _free_vram()


# ─────────────────────────────────────────────────────────────────────────────
# 11.  TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    import traceback
    passed = failed = 0

    def ok(name):
        nonlocal passed; passed += 1; print(f"  ✓ {name}")

    def fail(name, e):
        nonlocal failed; failed += 1
        print(f"  ✗ {name}: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)

    # ── utils ──
    try:
        t1, t2, t3, t4 = parse_label("['Yes', 'Yes', 'Clear', 'already']")
        assert t1 == "Yes" and t4 == "already"
        ok("parse_label: valid")
    except Exception as e: fail("parse_label: valid", e)

    try:
        parse_label("bad")
        fail("parse_label: invalid should raise", Exception())
    except (ValueError, SyntaxError): ok("parse_label: invalid raises")
    except Exception as e: fail("parse_label: invalid raises", e)

    try:
        p = {"t1": ["No", "Yes", "No"], "t2": ["Yes","Yes","No"],
             "t3": ["Clear","Clear","Clear"], "t4": ["already","already","already"]}
        r = apply_na_rule(p)
        assert r["t2"][0] == "N/A" and r["t2"][1] == "Yes" and r["t2"][2] == "N/A"
        ok("apply_na_rule: T1=No cascade")
    except Exception as e: fail("apply_na_rule: T1=No cascade", e)

    try:
        # T2=No → T3=N/A rule (new in V4)
        p = {"t1": ["Yes", "Yes"], "t2": ["No", "Yes"],
             "t3": ["Not Clear", "Clear"], "t4": ["already", "already"]}
        r = apply_na_rule(p)
        assert r["t3"][0] == "N/A", "T2=No must force T3=N/A"
        assert r["t3"][1] == "Clear", "T2=Yes must not change T3"
        assert r["t4"][0] == "already", "T2=No must NOT change T4"
        ok("apply_na_rule: T2=No → T3=N/A")
    except Exception as e: fail("apply_na_rule: T2=No → T3=N/A", e)

    try:
        p = {"t1": ["Yes"], "t2": ["No"], "t3": ["Not Clear"], "t4": ["within_2_years"]}
        assert preds_to_label_strings(p)[0] == "['Yes', 'No', 'Not Clear', 'within_2_years']"
        ok("preds_to_label_strings: format")
    except Exception as e: fail("preds_to_label_strings", e)

    try:
        yt = {"t1":["Yes","No"], "t2":["Yes","N/A"], "t3":["Clear","N/A"], "t4":["already","N/A"]}
        yp = dict(yt)
        s = compute_weighted_f1(yt, yp)
        assert abs(s["weighted"] - 1.0) < 1e-6
        ok("compute_weighted_f1: perfect=1.0")
    except Exception as e: fail("compute_weighted_f1: perfect", e)

    try:
        yt = {"t1":["Yes","Yes"], "t2":["Yes","Yes"], "t3":["Clear","Clear"], "t4":["already","already"]}
        yp = {"t1":["No","No"], "t2":["No","No"], "t3":["Not Clear","Not Clear"], "t4":["within_2_years","within_2_years"]}
        s = compute_weighted_f1(yt, yp)
        assert s["weighted"] < 0.5
        ok("compute_weighted_f1: wrong<0.5")
    except Exception as e: fail("compute_weighted_f1: wrong", e)

    # ── losses ──
    try:
        fl = FocalLoss(gamma=2.0)
        lg = torch.randn(4, 3, requires_grad=True)
        tg = torch.randint(0, 3, (4,))
        loss = fl(lg, tg)
        assert loss >= 0 and not torch.isnan(loss)
        loss.backward()
        assert lg.grad is not None and not torch.isnan(lg.grad).any()
        ok("FocalLoss: forward + gradient")
    except Exception as e: fail("FocalLoss", e)

    try:
        ol = OrdinalLoss(num_classes=5)
        lg = torch.randn(6, 5, requires_grad=True)
        tg = torch.tensor([0, 1, 2, 3, 4, 0])
        loss = ol(lg, tg)
        assert loss >= 0 and not torch.isnan(loss)
        loss.backward()
        ok("OrdinalLoss: forward + gradient")
    except Exception as e: fail("OrdinalLoss", e)

    try:
        ls = LabelSmoothingLoss(num_classes=4)
        lg = torch.randn(4, 4, requires_grad=True)
        loss = ls(lg, torch.randint(0, 4, (4,)))
        assert loss >= 0 and not torch.isnan(loss)
        loss.backward()
        ok("LabelSmoothingLoss: forward + gradient")
    except Exception as e: fail("LabelSmoothingLoss", e)

    try:
        scl = SupervisedContrastiveLoss(temperature=0.07)
        anchors = torch.randn(8, 64, requires_grad=True)
        positives = torch.randn(8, 64)
        labels = torch.randint(0, 4, (8,))
        loss = scl(anchors, positives, labels)
        assert not torch.isnan(loss)
        loss.backward()
        assert anchors.grad is not None
        ok("SupervisedContrastiveLoss: forward + gradient")
    except Exception as e: fail("SupervisedContrastiveLoss", e)

    try:
        for lt in ["ce", "focal", "ordinal", "label_smooth"]:
            fn = build_loss_fn(lt, "t4")
            lg = torch.randn(4, NUM_LABELS["t4"])
            tg = torch.randint(0, NUM_LABELS["t4"], (4,))
            assert not torch.isnan(fn(lg, tg))
        ok("build_loss_fn: all types valid")
    except Exception as e: fail("build_loss_fn", e)

    # ── DBLoss ──
    try:
        counts = [651, 149]  # Task1 class counts
        db = DBLoss(counts, total_samples=800)
        lg = torch.randn(8, 2, requires_grad=True)
        tg = torch.randint(0, 2, (8,))
        loss = db(lg, tg)
        assert not torch.isnan(loss) and loss >= 0
        loss.backward()
        assert lg.grad is not None and not torch.isnan(lg.grad).any()
        ok("DBLoss: forward + gradient")
    except Exception as e: fail("DBLoss: forward + gradient", e)

    try:
        # Rare class (idx=1, count=1) should have higher r̂ than common class (idx=0, count=799)
        counts = [799, 1]
        db = DBLoss(counts, total_samples=800)
        assert db.r_hat[1] > db.r_hat[0], "rare class must have higher weight"
        ok("DBLoss: rare class gets higher re-balance weight")
    except Exception as e: fail("DBLoss: rare class weight", e)

    try:
        # Rare class ν should be positive (lifts its logit threshold down)
        counts = [799, 1]
        db = DBLoss(counts, total_samples=800)
        assert db.nu[1] > 0, "rare class ν should be positive (kappa>0, logit(p_rare)<0)"
        assert db.nu[0] < 0, "common class ν should be negative"
        ok("DBLoss: NTR ν signs correct")
    except Exception as e: fail("DBLoss: NTR ν signs", e)

    try:
        # DB loss via build_loss_fn
        fn = build_loss_fn("db", "t3",
                           class_counts=[441, 99, 1, 149], total_samples=690)
        lg = torch.randn(4, 4)
        tg = torch.randint(0, 4, (4,))
        loss = fn(lg, tg)
        assert not torch.isnan(loss)
        ok("build_loss_fn: db type via factory")
    except Exception as e: fail("build_loss_fn: db type", e)

    # ── ApproachD (mean pool) — shape test without LLM ──
    try:
        # Simulate ApproachD's mean pool logic independently
        B, T, H, hidden_dim = 4, 32, 128, 64
        hidden = torch.randn(B, T, H)
        mask = torch.ones(B, T)
        mask[0, 20:] = 0  # first sample has padding
        mask_3d = mask.unsqueeze(-1)
        mean_repr = (hidden * mask_3d).sum(1) / mask_3d.sum(1).clamp(min=1e-9)
        assert mean_repr.shape == (B, H)
        # Verify padding is excluded: mean over first 20 tokens ≠ mean over all 32
        manual_mean = hidden[0, :20].mean(0)
        assert not torch.allclose(mean_repr[0], hidden[0].mean(0), atol=1e-4), \
            "mean pool should exclude padding"
        ok("ApproachD: masked mean pool logic correct")
    except Exception as e: fail("ApproachD: mean pool", e)

    # ── models ──
    try:
        head = MultiTaskHead(hidden_size=64, dropout=0.0)
        x = torch.randn(4, 64)
        out = head(x)
        for t, n in NUM_LABELS.items():
            assert out[t].shape == (4, n), f"wrong shape for {t}"
        ok("MultiTaskHead: shapes")
    except Exception as e: fail("MultiTaskHead", e)

    try:
        head = CascadeHead(hidden_size=64, dropout=0.0)
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        assert out["t1"].shape == (4, 2) and out["t3"].shape == (4, 4)
        sum(o.sum() for o in out.values()).backward()
        assert x.grad is not None
        ok("CascadeHead: shapes + gradient")
    except Exception as e: fail("CascadeHead", e)

    try:
        head = CascadeHeadV2(hidden_size=64, dropout=0.0)
        x = torch.randn(4, 64, requires_grad=True)
        out = head(x)
        assert out["t1"].shape == (4, 2)
        assert out["t2"].shape == (4, 3)
        assert out["t3"].shape == (4, 4)   # sees T1(2)+T2(3) probs
        assert out["t4"].shape == (4, 5)   # only T1 probs
        sum(o.sum() for o in out.values()).backward()
        assert x.grad is not None
        ok("CascadeHeadV2: shapes + gradient")
    except Exception as e: fail("CascadeHeadV2", e)

    try:
        # ensemble_preds_soft: average uniform probs → same result as argmax
        probs1 = {"t1": [[0.8, 0.2]], "t2": [[0.7, 0.1, 0.2]],
                  "t3": [[0.6, 0.1, 0.1, 0.2]], "t4": [[0.5, 0.1, 0.2, 0.1, 0.1]]}
        probs2 = {"t1": [[0.6, 0.4]], "t2": [[0.5, 0.3, 0.2]],
                  "t3": [[0.4, 0.3, 0.1, 0.2]], "t4": [[0.4, 0.2, 0.2, 0.1, 0.1]]}
        r = ensemble_preds_soft([probs1, probs2])
        assert r["t1"][0] == "Yes"   # both favor Yes
        assert r["t2"][0] == "Yes"   # both favor Yes
        assert r["t3"][0] == "Clear" # avg [0.5, 0.2, 0.1, 0.2] → Clear
        ok("ensemble_preds_soft: correctness")
    except Exception as e: fail("ensemble_preds_soft", e)

    try:
        model = ApproachC(embed_dim=128, hidden_dim=64, dropout=0.0)
        embs = torch.randn(4, 128)
        out = model(embedding=embs)
        for t, n in NUM_LABELS.items():
            assert out[t].shape == (4, n)
        ok("ApproachC: forward shape")
    except Exception as e: fail("ApproachC", e)

    # ── temperature scaler ──
    try:
        scaler = TemperatureScaler()
        logits = {t: torch.randn(10, NUM_LABELS[t]) for t in LABELS}
        labels = {t: torch.randint(0, NUM_LABELS[t], (10,)) for t in LABELS}
        temps = scaler.fit(logits, labels)
        assert all(v > 0 for v in temps.values())
        scaled = scaler(logits)
        assert all(scaled[t].shape == logits[t].shape for t in LABELS)
        ok("TemperatureScaler: fit + forward")
    except Exception as e: fail("TemperatureScaler", e)

    # ── data ──
    try:
        train_df, test_df = load_dataframes(str(DATA_DIR))
        assert len(train_df) == 800 and len(test_df) == 200
        na_mask = train_df["promise_status"] == "No"
        assert (train_df.loc[na_mask, "evidence_quality"] == "N/A").all()
        # No NaN should remain
        for col in ["evidence_status", "evidence_quality", "verification_timeline"]:
            assert train_df[col].isna().sum() == 0
        ok("load_dataframes: shape + NA rule + no NaN")
    except Exception as e: fail("load_dataframes", e)

    try:
        train_df, _ = load_dataframes(str(DATA_DIR))
        tr, va = split_train_val(train_df, val_ratio=0.1, seed=42)
        assert len(tr) + len(va) == len(train_df) and len(va) == 80
        ok("split_train_val: sizes")
    except Exception as e: fail("split_train_val", e)

    try:
        from transformers import AutoTokenizer as AT
        tok = AT.from_pretrained("hfl/chinese-macbert-base", trust_remote_code=True)
        train_df, _ = load_dataframes(str(DATA_DIR))
        ds = ESGDataset(train_df.head(4), tok, max_length=128)
        item = ds[0]
        assert item["input_ids"].shape == (128,)
        assert item["t1"].dtype == torch.long
        ok("ESGDataset: tokenisation + labels")
    except Exception as e: fail("ESGDataset", e)

    try:
        train_df, _ = load_dataframes(str(DATA_DIR))
        embs = torch.randn(10, 64)
        ds = EmbeddingDataset(embs, train_df.head(10))
        item = ds[0]
        assert item["embedding"].shape == (64,)
        assert item["t3"].dtype == torch.long
        ok("EmbeddingDataset: shapes + labels")
    except Exception as e: fail("EmbeddingDataset", e)

    try:
        train_df, _ = load_dataframes(str(DATA_DIR))
        embs = torch.randn(10, 64)
        ds = ContrastivePairDataset(embs, train_df.head(10))
        item = ds[0]
        assert item["anchor"].shape == (64,) and item["positive"].shape == (64,)
        ok("ContrastivePairDataset: shapes")
    except Exception as e: fail("ContrastivePairDataset", e)

    # ── class weights ──
    try:
        train_df, _ = load_dataframes(str(DATA_DIR))
        w = compute_class_weights(train_df["promise_status"].tolist(), "t1")
        assert w.shape == (2,) and (w > 0).all()
        ok("compute_class_weights: shape + positive")
    except Exception as e: fail("compute_class_weights", e)

    # ── generative parser ──
    try:
        raw = '<think>some reasoning</think>\n{"promise_status": "Yes", "evidence_status": "No", "evidence_quality": "Clear", "verification_timeline": "already"}'
        out = parse_gen_output(raw)
        assert out["promise_status"] == "Yes" and out["evidence_quality"] == "Clear"
        ok("parse_gen_output: with thinking block")
    except Exception as e: fail("parse_gen_output: thinking block", e)

    try:
        out = parse_gen_output("completely broken output %%%")
        assert out["promise_status"] in LABELS["t1"]
        ok("parse_gen_output: fallback on bad output")
    except Exception as e: fail("parse_gen_output: fallback", e)

    try:
        out = parse_gen_output('{"promise_status": "INVALID", "evidence_status": "Yes", "evidence_quality": "Clear", "verification_timeline": "already"}')
        assert out["promise_status"] in LABELS["t1"]
        ok("parse_gen_output: invalid label → default")
    except Exception as e: fail("parse_gen_output: invalid label", e)

    # ── ensemble ──
    try:
        p1 = {"t1":["Yes","No"], "t2":["Yes","N/A"], "t3":["Clear","N/A"], "t4":["already","N/A"]}
        p2 = {"t1":["Yes","Yes"], "t2":["No","Yes"], "t3":["Clear","Clear"], "t4":["already","within_2_years"]}
        result = ensemble_preds([p1, p2])
        assert result["t1"][0] == "Yes"
        ok("ensemble_preds: majority vote")
    except Exception as e: fail("ensemble_preds", e)

    # ── augment_rare_classes ──
    try:
        _pd = pd
        mislead_row = {
            "id": 9999, "data": "公司承諾將達成減碳目標，實際上排放量持續增加，屬於誤導性陳述。",
            "promise_status": "Yes", "evidence_status": "Yes",
            "evidence_quality": "Misleading", "verification_timeline": "within_2_years",
        }
        dummy_df = _pd.DataFrame([mislead_row] * 3)  # 3 Misleading + 3 within_2_years
        augmented = augment_rare_classes(dummy_df, target_n=10, seed=0)
        mislead_count = (augmented["evidence_quality"] == "Misleading").sum()
        w2y_count = (augmented["verification_timeline"] == "within_2_years").sum()
        assert mislead_count >= 10, f"Expected ≥10 Misleading, got {mislead_count}"
        assert w2y_count >= 10, f"Expected ≥10 within_2_years, got {w2y_count}"
        # All IDs must be integers (no strings — DataLoader collate requires uniform type)
        assert all(isinstance(i, (int, np.integer)) for i in augmented["id"]), \
            "Augmented IDs must all be integers"
        ok("augment_rare_classes: rare classes reach target_n")
    except Exception as e: fail("augment_rare_classes", e)

    try:
        _pd = pd
        # No rare classes → DataFrame unchanged
        normal_row = {
            "id": 1, "data": "正常陳述", "promise_status": "Yes",
            "evidence_status": "Yes", "evidence_quality": "Clear",
            "verification_timeline": "already",
        }
        df_no_rare = _pd.DataFrame([normal_row] * 5)
        result = augment_rare_classes(df_no_rare, target_n=10, seed=0)
        assert len(result) == 5, "No rare classes → no augmentation"
        ok("augment_rare_classes: no rare classes → unchanged")
    except Exception as e: fail("augment_rare_classes: no rare", e)

    # ── LogSigma _compute_loss ──
    try:
        logits_mock = {t: torch.randn(4, NUM_LABELS[t]) for t in LABELS}
        batch_mock  = {t: torch.randint(0, NUM_LABELS[t], (4,)) for t in LABELS}
        loss_fns_mock = {t: nn.CrossEntropyLoss() for t in LABELS}
        device_cpu = torch.device("cpu")

        # Standard mode (log_sigma=None) should equal TASK_WEIGHTS sum
        std_loss = _compute_loss(logits_mock, batch_mock, loss_fns_mock, device_cpu, log_sigma=None)
        assert not torch.isnan(std_loss), "standard loss is NaN"

        # LogSigma mode with zero init → effective weight = 0.5 for all tasks
        ls = nn.ParameterDict({t: nn.Parameter(torch.zeros(1)) for t in LABELS})
        ls_loss = _compute_loss(logits_mock, batch_mock, loss_fns_mock, device_cpu, log_sigma=ls)
        assert not torch.isnan(ls_loss), "logsigma loss is NaN"

        # Gradient must flow through log_sigma params
        ls_loss.backward()
        for t in LABELS:
            assert ls[t].grad is not None, f"log_sigma[{t}].grad is None"
        ok("_compute_loss: standard + logsigma gradient flow")
    except Exception as e: fail("_compute_loss: logsigma", e)

    # ── per_task_loss build_loss_fns ──
    try:
        _pd = pd
        rows = []
        for _ in range(10):
            rows.append({"promise_status": "Yes", "evidence_status": "Yes",
                         "evidence_quality": "Clear", "verification_timeline": "already"})
        rows[0]["evidence_quality"] = "Misleading"
        rows[1]["verification_timeline"] = "within_2_years"
        df_mock = _pd.DataFrame(rows)
        cfg_pt = Config(per_task_loss=True)
        fns = build_loss_fns(cfg_pt, df_mock, torch.device("cpu"))
        assert isinstance(fns["t3"], DBLoss), f"T3 should be DBLoss, got {type(fns['t3'])}"
        assert isinstance(fns["t4"], OrdinalLoss), f"T4 should be OrdinalLoss, got {type(fns['t4'])}"
        # T2 uses CE (Focal was proven harmful in V2: caused No→Yes prediction bias)
        assert isinstance(fns["t2"], nn.CrossEntropyLoss), f"T2 should be CE, got {type(fns['t2'])}"
        # Verify all loss functions are callable
        for task in ["t1", "t2", "t3", "t4"]:
            lg = torch.randn(4, NUM_LABELS[task])
            tg = torch.randint(0, NUM_LABELS[task], (4,))
            loss = fns[task](lg, tg)
            assert not torch.isnan(loss), f"NaN in {task} per_task loss"
        ok("build_loss_fns: per_task_loss correct types + callable")
    except Exception as e: fail("build_loss_fns: per_task_loss", e)

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        raise SystemExit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="ESG Classification — VeriPromise AI CUP 2026")
    p.add_argument("--mode", choices=["train","predict","test","geneval","run_all","kfold",
                                      "gen_submissions","run_v4","gen_llm_augdata",
                                      "search_thresholds","gen_pseudo_labels",
                                      "tune_t3_nc_bias","tapt"], default="test")
    p.add_argument("--skip_existing", action="store_true", default=False,
                   help="gen_submissions: skip combos whose CSV already exists in submissions/")
    p.add_argument("--approach", choices=["A","A1","B","B_lora","C","C_contrastive","D"], default="A")
    p.add_argument("--backbone", default="hfl/chinese-macbert-large")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--loss_type", choices=["ce","focal","ordinal","label_smooth","db"], default="focal")
    p.add_argument("--db_alpha", type=float, default=0.1)
    p.add_argument("--db_beta",  type=float, default=10.0)
    p.add_argument("--db_mu",    type=float, default=0.5)
    p.add_argument("--db_kappa", type=float, default=0.05)
    p.add_argument("--use_class_weight", action="store_true", default=True)
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--apply_na_rule", action="store_true", default=True)
    p.add_argument("--run_dir", default="runs/exp")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--kfold_dir", default="", help="tune_t3_nc_bias: override kfold dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default=str(DATA_DIR))
    p.add_argument("--use_augmented", action="store_true", default=False)
    p.add_argument("--aug_filename", type=str, default="train_data_augmented.csv")
    p.add_argument("--enable_thinking", action="store_true", default=True)
    p.add_argument("--per_task_loss", action="store_true", default=False)
    p.add_argument("--deep_cascade", action="store_true", default=False)
    p.add_argument("--augment_rare", action="store_true", default=False)
    p.add_argument("--augment_target_n", type=int, default=15)
    p.add_argument("--fgm_epsilon", type=float, default=0.0)
    p.add_argument("--use_logsigma", action="store_true", default=False)
    p.add_argument("--adaptive_task_weight", action="store_true", default=False)
    p.add_argument("--adaptive_alpha", type=float, default=0.5)
    p.add_argument("--use_hand_features", action="store_true", default=False)
    p.add_argument("--pooling", type=str, default="cls", choices=["cls", "attn", "mean"])
    p.add_argument("--prepend_linguistic", action="store_true", default=False)
    p.add_argument("--t3_nc_weight", type=float, default=1.0)
    p.add_argument("--t3_cond_weight", type=float, default=0.0)
    p.add_argument("--company_emb_dim", type=int, default=0)
    p.add_argument("--rdrop_alpha", type=float, default=0.0)
    p.add_argument("--swa_start_epoch", type=int, default=0)
    p.add_argument("--sharp_recl_weight", type=float, default=0.0)
    p.add_argument("--mr2_weight", type=float, default=0.0)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--kfold", type=int, default=1)
    p.add_argument("--proj_hidden_dim", type=int, default=1024)
    args = p.parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


def run_tapt(cfg: Config) -> None:
    """Task-Adaptive Pre-Training: continue MLM on train + test texts for domain adaptation.

    Saves the adapted backbone (encoder only, no MLM head) to cfg.run_dir,
    which can then be used as --backbone for fine-tuning experiments.
    """
    from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  TAPT: {cfg.backbone} → {cfg.run_dir}")
    print(f"  epochs={cfg.epochs}, lr={cfg.lr}, batch_size={cfg.batch_size}")
    print(f"{'='*60}")

    train_df, test_df = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)
    texts = list(train_df["data"]) + list(test_df["data"])
    print(f"  Texts: {len(texts)} (train={len(train_df)}, test={len(test_df)})")

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(cfg.backbone, trust_remote_code=True).to(device)

    # Tokenize all texts (no padding here — collator handles dynamic padding + MLM masking)
    encodings = tokenizer(texts, truncation=True, max_length=cfg.max_length,
                          return_attention_mask=True)

    class _TAPTDataset(Dataset):
        def __len__(self):
            return len(texts)
        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in encodings.items()
                    if k in ("input_ids", "attention_mask")}

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    loader = DataLoader(_TAPTDataset(), batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    total_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch}/{cfg.epochs}: MLM loss={avg:.4f}")

    # Save backbone encoder (not MLM head) — detect model architecture
    out_dir = Path(cfg.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # RoBERTa-family: model.roberta; BERT/LERT/MacBERT: model.bert
    for attr in ("roberta", "bert", "megatron_bert"):
        if hasattr(model, attr):
            getattr(model, attr).save_pretrained(str(out_dir))
            break
    else:
        # Fallback: save full model, user loads with AutoModel
        model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"  TAPT backbone saved to {out_dir}")
    print(f"  Usage: --backbone {out_dir} for fine-tuning")


def gen_submissions(cfg: Config) -> None:
    """Generate multiple submission CSVs from different checkpoint combinations.

    Uses soft voting (probability averaging) for multi-model ensembles.
    """
    train_df, test_df = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)
    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)

    # ── T3 class frequencies (for logit adjustment) ───────────────────────────
    t3_counts = [
        (train_df["evidence_quality"].fillna("N/A") == lbl).sum()
        for lbl in LABELS["t3"]
    ]
    t3_total = sum(t3_counts)
    T3_CLASS_FREQ = [c / t3_total for c in t3_counts]  # [Clear, NC, Misleading, N/A]

    # ── Single-run checkpoints (name → path) ─────────────────────────────────
    CKPTS = {
        "pertask_aug":          "runs/A1_macbert_pertask_aug/best.pt",
        "pertask":              "runs/A1_macbert_pertask/best.pt",
        "aug_ordinal":          "runs/A1_macbert_aug_ordinal/best.pt",
        "ordinal":              "runs/A1_macbert_ordinal/best.pt",
        "D_pertask":            "runs/D_qwen3_8b_pertask/best.pt",
        "D_pertask_aug":        "runs/D_qwen3_8b_pertask_aug/best.pt",
        # V4 cascade experiments
        "v4_dc_pertask":        "runs/A1_macbert_dc_pertask/best.pt",
        "v4_dc_pertask_aug":    "runs/A1_macbert_dc_pertask_aug/best.pt",
        "v4_roberta_dc":        "runs/A1_roberta_dc_pertask/best.pt",
        "v4_roberta_dc_aug":    "runs/A1_roberta_dc_pertask_aug/best.pt",
        # V4 LERT-large experiments
        "v4_lert_dc":           "runs/A1_lert_dc_pertask/best.pt",
        "v4_lert_dc_aug":       "runs/A1_lert_dc_pertask_aug/best.pt",
        # V4 DeBERTa-v2-97M experiments
        "v4_deberta97m_dc":     "runs/A1_deberta97m_dc_pertask/best.pt",
        "v4_deberta97m_dc_aug": "runs/A1_deberta97m_dc_aug/best.pt",
    }

    # ── K-fold directories (each fold's best.pt is soft-voted into one entry) ─
    KFOLD_DIRS = {
        "v4_dc_kfold":            "runs/A1_macbert_dc_kfold",
        "v4_roberta_kfold":       "runs/A1_roberta_dc_kfold",
        "v4_lert_kfold":          "runs/A1_lert_dc_kfold",
        "v4_modernbert_kfold":    "runs/A1_modernbert_dc_kfold",
        "v4_roberta_fgm_kfold":        "runs/A1_roberta_dc_kfold_fgm",
        "v4_lert_fgm_kfold":           "runs/A1_lert_dc_kfold_fgm",
        "v4_roberta_logsigma_kfold":   "runs/A1_roberta_dc_kfold_logsigma_fgm",
        "v4_lert_logsigma_kfold":      "runs/A1_lert_dc_kfold_logsigma_fgm",
        # V11: LLM-augmented kfold
        "v4_roberta_llmaug_kfold":     "runs/A1_roberta_dc_kfold_llmaug",
        "v4_lert_llmaug_kfold":        "runs/A1_lert_dc_kfold_llmaug",
        # V13: new backbone LLM-augmented kfold
        "v4_deberta_llmaug_kfold":     "runs/A1_deberta_dc_kfold_llmaug",
        "v4_electra_llmaug_kfold":     "runs/A1_electra_dc_kfold_llmaug",
        "v4_ernie_llmaug_kfold":       "runs/A1_ernie_dc_kfold_llmaug",
        # V14: hand features + LLM-augmented kfold
        "v4_roberta_llmaug_hf_kfold":  "runs/A1_roberta_dc_kfold_llmaug_hf",
        "v4_lert_llmaug_hf_kfold":     "runs/A1_lert_dc_kfold_llmaug_hf",
        # V15: T3 ordinal loss + LLM-augmented kfold
        "v4_roberta_t3ordinal_kfold":  "runs/A1_roberta_dc_kfold_t3ordinal",
        "v4_lert_t3ordinal_kfold":     "runs/A1_lert_dc_kfold_t3ordinal",
        # V17: LogSigma + FGM(ε=0.1) + aug kfold
        "v4_roberta_ls_fgm_kfold":     "runs/A1_roberta_dc_kfold_ls_fgm",
        "v4_lert_ls_fgm_kfold":        "runs/A1_lert_dc_kfold_ls_fgm",
        # V19: OLL loss (Ordinal Log-Loss, COLING 2022) + LLM-augmented kfold
        "v4_roberta_t3oll_kfold":      "runs/A1_roberta_dc_kfold_t3oll",
        "v4_lert_t3oll_kfold":         "runs/A1_lert_dc_kfold_t3oll",
        # V26: aug without Misleading samples (940 samples)
        "v4_roberta_nomislead_kfold":  "runs/A1_roberta_dc_kfold_nomislead",
        "v4_lert_nomislead_kfold":     "runs/A1_lert_dc_kfold_nomislead",
        # V31: span auxiliary loss + supervised contrastive loss (T3 focused)
        "v5_roberta_span_scl_kfold":   "runs/A1_roberta_dc_kfold_span_scl",
        "v5_lert_span_scl_kfold":      "runs/A1_lert_dc_kfold_span_scl",
        # V33: 10-fold aug models
        "v5_roberta_kfold10_llmaug":   "runs/A1_roberta_dc_kfold10_llmaug",
        "v5_lert_kfold10_llmaug":      "runs/A1_lert_dc_kfold10_llmaug",
        # V34: MacBERT + LLM aug, 5-fold
        "v4_macbert_llmaug_kfold":     "runs/A1_macbert_dc_kfold_llmaug",
        # V36: Pseudo-labeled (aug + high-conf test)
        "v4_roberta_pseudo_kfold":     "runs/A1_roberta_dc_kfold_pseudo",
        "v4_lert_pseudo_kfold":        "runs/A1_lert_dc_kfold_pseudo",
        # V38: Few-shot demonstration (same-company context)
        "v4_roberta_fewshot_kfold":    "runs/A1_roberta_dc_kfold_fewshot",
        "v4_lert_fewshot_kfold":       "runs/A1_lert_dc_kfold_fewshot",
        # V39: Masked gradient (zero T2/T3/T4 loss for T1=No samples)
        "v4_roberta_masked_kfold":     "runs/A1_roberta_dc_kfold_masked",
        "v4_lert_masked_kfold":        "runs/A1_lert_dc_kfold_masked",
        # V40: T3 Not Clear boost (CE loss with 2x/3x Not Clear class weight)
        "v4_roberta_t3nc2_kfold":      "runs/A1_roberta_dc_kfold_t3nc2",
        "v4_lert_t3nc2_kfold":         "runs/A1_lert_dc_kfold_t3nc2",
        "v4_roberta_t3nc3_kfold":       "runs/A1_roberta_dc_kfold_t3nc3",
        "v4_lert_t3nc3_kfold":          "runs/A1_lert_dc_kfold_t3nc3",
        # V47: seed ensemble
        "v4_roberta_t3nc3_s1_kfold":     "runs/A1_roberta_dc_kfold_t3nc3_s1",
        "v4_roberta_t3nc3_s2_kfold":     "runs/A1_roberta_dc_kfold_t3nc3_s2",
        # V48: more seeds + lert seeds + R-Drop + SWA + distill
        "v4_roberta_t3nc3_s3_kfold":     "runs/A1_roberta_dc_kfold_t3nc3_s3",
        "v4_roberta_t3nc3_s4_kfold":     "runs/A1_roberta_dc_kfold_t3nc3_s4",
        "v4_lert_t3nc3_s1_kfold":        "runs/A1_lert_dc_kfold_t3nc3_s1",
        "v4_lert_t3nc3_s2_kfold":        "runs/A1_lert_dc_kfold_t3nc3_s2",
        "v4_roberta_t3nc3_rdrop_kfold":  "runs/A1_roberta_dc_kfold_t3nc3_rdrop",
        "v4_roberta_t3nc3_swa_kfold":    "runs/A1_roberta_dc_kfold_t3nc3_swa",
        "v4_roberta_t3nc3_distill_kfold":"runs/A1_roberta_dc_kfold_t3nc3_distill",
        # V50: TMix embedding mixup
        "v4_roberta_t3nc3_tmix04_kfold": "runs/A1_roberta_dc_kfold_t3nc3_tmix04",
        "v4_roberta_t3nc3_tmix10_kfold": "runs/A1_roberta_dc_kfold_t3nc3_tmix10",
        # V51: TAPT backbone
        "v4_roberta_tapt_t3nc3_kfold":  "runs/A1_roberta_dc_kfold_t3nc3_tapt",
        # V53: hard NC augmentation
        "v4_roberta_t3nc3_hardnc_kfold": "runs/A1_roberta_dc_kfold_t3nc3_hardnc",
        "v4_lert_t3nc3_hardnc_kfold":    "runs/A1_lert_dc_kfold_t3nc3_hardnc",
        # V53b: hard NC + nc_weight=1.0
        "v4_roberta_hardnc_nw1_kfold":   "runs/A1_roberta_dc_kfold_hardnc_nw1",
        "v4_lert_hardnc_nw1_kfold":      "runs/A1_lert_dc_kfold_hardnc_nw1",
        # V54: nomislead + nc3
        "v4_roberta_nomislead_nc3_kfold":"runs/A1_roberta_dc_kfold_nomislead_nc3",
        # V57: label smoothing
        "v4_roberta_t3nc3_ls01_kfold":  "runs/A1_roberta_dc_kfold_t3nc3_ls01",
        # V46: focal loss + NC weight
        "v4_roberta_t3nc3_focal1_kfold": "runs/A1_roberta_dc_kfold_t3nc3_focal1",
        "v4_roberta_t3nc3_focal2_kfold": "runs/A1_roberta_dc_kfold_t3nc3_focal2",
        "v4_lert_t3nc3_focal2_kfold":    "runs/A1_lert_dc_kfold_t3nc3_focal2",
        # V44: T3 nc_weight=4.0/5.0
        "v4_roberta_t3nc4_kfold":      "runs/A1_roberta_dc_kfold_t3nc4",
        "v4_lert_t3nc4_kfold":         "runs/A1_lert_dc_kfold_t3nc4",
        "v4_roberta_t3nc5_kfold":      "runs/A1_roberta_dc_kfold_t3nc5",
        # V41: DeBERTa-v2-710M-Chinese
        "v4_deberta710m_llmaug_kfold": "runs/A1_deberta710m_dc_kfold_llmaug",
        # V42: DeBERTa-v2-320M-Chinese + PERT-large
        "v4_deberta320m_llmaug_kfold": "runs/A1_deberta320m_dc_kfold_llmaug",
        "v4_pert_llmaug_kfold":        "runs/A1_pert_dc_kfold_llmaug",
        # V43: Misleading → Not Clear (training + inference remap)
        "v4_roberta_mislead_nc_kfold": "runs/A1_roberta_dc_kfold_mislead_nc",
        "v4_lert_mislead_nc_kfold":    "runs/A1_lert_dc_kfold_mislead_nc",
        # ── 2nd Stage (B1): trained on 1602-row official dataset ──
        "b1_roberta_t3nc3_kfold":      "runs/B1_roberta_dc_kfold_t3nc3",
        "b1_roberta_t3nc3_s1_kfold":   "runs/B1_roberta_dc_kfold_t3nc3_s1",
        "b1_roberta_t3nc3_s2_kfold":   "runs/B1_roberta_dc_kfold_t3nc3_s2",
        # ── 3rd Stage (C1): trained on reshuffled 1601-row dataset ──
        "c1_roberta_t3nc3_kfold":      "runs/C1_roberta_dc_kfold_t3nc3",
        "c1_roberta_t3nc3_s1_kfold":   "runs/C1_roberta_dc_kfold_t3nc3_s1",
        "c1_roberta_t3nc3_s2_kfold":   "runs/C1_roberta_dc_kfold_t3nc3_s2",
        # ── D1: LLM-augmented training (3rd_data + llm_augmented.csv) ──
        "d1_roberta_t3nc3_kfold":      "runs/D1_roberta_dc_kfold_t3nc3",
        "d1_roberta_t3nc3_s1_kfold":   "runs/D1_roberta_dc_kfold_t3nc3_s1",
        "d1_roberta_t3nc3_s2_kfold":   "runs/D1_roberta_dc_kfold_t3nc3_s2",
        "d1_roberta_t3nc5_kfold":      "runs/D1_roberta_dc_kfold_t3nc5",
        "d1_roberta_t3nc5_s1_kfold":   "runs/D1_roberta_dc_kfold_t3nc5_s1",
        "d1_roberta_t3nc5_s2_kfold":   "runs/D1_roberta_dc_kfold_t3nc5_s2",
        # ── E2: Hard-Mixup (WWW 2026), NC↔Clear cross-class boundary mixing ──
        "e2_roberta_hm_kfold":         "runs/E2_roberta_dc_kfold_t3nc3_hm",
        "e2_roberta_hm_s1_kfold":      "runs/E2_roberta_dc_kfold_t3nc3_hm_s1",
        "e2_roberta_hm_s2_kfold":      "runs/E2_roberta_dc_kfold_t3nc3_hm_s2",
        # ── F3: Class-Aware Prototype Contrastive Loss for T3 ──
        "f3_roberta_proto_kfold":      "runs/F3_roberta_dc_kfold_t3nc3_proto",
        "f3_roberta_proto_s1_kfold":   "runs/F3_roberta_dc_kfold_t3nc3_proto_s1",
        "f3_roberta_proto_s2_kfold":   "runs/F3_roberta_dc_kfold_t3nc3_proto_s2",
        # ── G1: MacBERT at C1 stage ──
        "g1_macbert_t3nc3_kfold":      "runs/G1_macbert_dc_kfold_t3nc3",
        "g1_macbert_t3nc3_s1_kfold":   "runs/G1_macbert_dc_kfold_t3nc3_s1",
        "g1_macbert_t3nc3_s2_kfold":   "runs/G1_macbert_dc_kfold_t3nc3_s2",
        # ── G2: LERT at C1 stage ──
        "g2_lert_t3nc3_kfold":         "runs/G2_lert_dc_kfold_t3nc3",
        "g2_lert_t3nc3_s1_kfold":      "runs/G2_lert_dc_kfold_t3nc3_s1",
        "g2_lert_t3nc3_s2_kfold":      "runs/G2_lert_dc_kfold_t3nc3_s2",
        # ── G3: RoBERTa + SCL at C1 stage ──
        "g3_roberta_scl_kfold":        "runs/G3_roberta_dc_kfold_t3nc3_scl",
        "g3_roberta_scl_s1_kfold":     "runs/G3_roberta_dc_kfold_t3nc3_scl_s1",
        "g3_roberta_scl_s2_kfold":     "runs/G3_roberta_dc_kfold_t3nc3_scl_s2",
        # ── G4: MacBERT + SCL at C1 stage ──
        "g4_macbert_scl_kfold":        "runs/G4_macbert_dc_kfold_t3nc3_scl",
        "g4_macbert_scl_s1_kfold":     "runs/G4_macbert_dc_kfold_t3nc3_scl_s1",
        "g4_macbert_scl_s2_kfold":     "runs/G4_macbert_dc_kfold_t3nc3_scl_s2",
        # ── G5: RoBERTa + company_emb at C1 stage ──
        "g5_roberta_cemb_kfold":       "runs/G5_roberta_dc_kfold_t3nc3_cemb64",
        "g5_roberta_cemb_s1_kfold":    "runs/G5_roberta_dc_kfold_t3nc3_cemb64_s1",
        "g5_roberta_cemb_s2_kfold":    "runs/G5_roberta_dc_kfold_t3nc3_cemb64_s2",
        # ── H1: RoBERTa + Attention Pooling ──
        "h1_roberta_attn_kfold":       "runs/H1_roberta_dc_kfold_t3nc3_attn",
        "h1_roberta_attn_s1_kfold":    "runs/H1_roberta_dc_kfold_t3nc3_attn_s1",
        "h1_roberta_attn_s2_kfold":    "runs/H1_roberta_dc_kfold_t3nc3_attn_s2",
        # ── H2: RoBERTa + Feature Prepending ──
        "h2_roberta_prep_kfold":       "runs/H2_roberta_dc_kfold_t3nc3_prep",
        "h2_roberta_prep_s1_kfold":    "runs/H2_roberta_dc_kfold_t3nc3_prep_s1",
        "h2_roberta_prep_s2_kfold":    "runs/H2_roberta_dc_kfold_t3nc3_prep_s2",
        # ── H3: RoBERTa + AttnPool + FeaturePrepend ──
        "h3_roberta_attn_prep_kfold":    "runs/H3_roberta_dc_kfold_t3nc3_attn_prep",
        "h3_roberta_attn_prep_s1_kfold": "runs/H3_roberta_dc_kfold_t3nc3_attn_prep_s1",
        "h3_roberta_attn_prep_s2_kfold": "runs/H3_roberta_dc_kfold_t3nc3_attn_prep_s2",
        # ── C1R: R-Drop + SWA ──
        "c1r_roberta_t3nc3_kfold":     "runs/C1R_roberta_dc_kfold_t3nc3",
        "c1r_roberta_t3nc3_s1_kfold":  "runs/C1R_roberta_dc_kfold_t3nc3_s1",
        "c1r_roberta_t3nc3_s2_kfold":  "runs/C1R_roberta_dc_kfold_t3nc3_s2",
        # ── C1S: SharpReCL ──
        "c1s_roberta_t3nc3_kfold":     "runs/C1S_roberta_dc_kfold_t3nc3",
        "c1s_roberta_t3nc3_s1_kfold":  "runs/C1S_roberta_dc_kfold_t3nc3_s1",
        "c1s_roberta_t3nc3_s2_kfold":  "runs/C1S_roberta_dc_kfold_t3nc3_s2",
        # ── C1M: MR2 Margin ──
        "c1m_roberta_t3nc3_kfold":     "runs/C1M_roberta_dc_kfold_t3nc3",
        "c1m_roberta_t3nc3_s1_kfold":  "runs/C1M_roberta_dc_kfold_t3nc3_s1",
        "c1m_roberta_t3nc3_s2_kfold":  "runs/C1M_roberta_dc_kfold_t3nc3_s2",
        # ── C1RS: R-Drop + SharpReCL ──
        "c1rs_roberta_t3nc3_kfold":    "runs/C1RS_roberta_dc_kfold_t3nc3",
        "c1rs_roberta_t3nc3_s1_kfold": "runs/C1RS_roberta_dc_kfold_t3nc3_s1",
        "c1rs_roberta_t3nc3_s2_kfold": "runs/C1RS_roberta_dc_kfold_t3nc3_s2",
        # ═══ Final stage (FC1/FC1R/FC1S/FC1M): trained on final_data ═══
        "fc1_roberta_t3nc3_kfold":     "runs/FC1_roberta_dc_kfold_t3nc3",
        "fc1_roberta_t3nc3_s1_kfold":  "runs/FC1_roberta_dc_kfold_t3nc3_s1",
        "fc1_roberta_t3nc3_s2_kfold":  "runs/FC1_roberta_dc_kfold_t3nc3_s2",
        "fc1r_roberta_t3nc3_kfold":    "runs/FC1R_roberta_dc_kfold_t3nc3",
        "fc1r_roberta_t3nc3_s1_kfold": "runs/FC1R_roberta_dc_kfold_t3nc3_s1",
        "fc1r_roberta_t3nc3_s2_kfold": "runs/FC1R_roberta_dc_kfold_t3nc3_s2",
        "fc1s_roberta_t3nc3_kfold":    "runs/FC1S_roberta_dc_kfold_t3nc3",
        "fc1s_roberta_t3nc3_s1_kfold": "runs/FC1S_roberta_dc_kfold_t3nc3_s1",
        "fc1s_roberta_t3nc3_s2_kfold": "runs/FC1S_roberta_dc_kfold_t3nc3_s2",
        "fc1m_roberta_t3nc3_kfold":    "runs/FC1M_roberta_dc_kfold_t3nc3",
        "fc1m_roberta_t3nc3_s1_kfold": "runs/FC1M_roberta_dc_kfold_t3nc3_s1",
        "fc1m_roberta_t3nc3_s2_kfold": "runs/FC1M_roberta_dc_kfold_t3nc3_s2",
        # Retrain models (full_train_data 2000 rows → predict test_2000)
        "rt_fc1_s0_kfold":  "runs/RT_FC1_s0",
        "rt_fc1_s1_kfold":  "runs/RT_FC1_s1",
        "rt_fc1_s2_kfold":  "runs/RT_FC1_s2",
        "rt_fc1r_s0_kfold": "runs/RT_FC1R_s0",
        "rt_fc1r_s1_kfold": "runs/RT_FC1R_s1",
        "rt_fc1r_s2_kfold": "runs/RT_FC1R_s2",
        "rt_fc1s_s0_kfold": "runs/RT_FC1S_s0",
        "rt_fc1s_s1_kfold": "runs/RT_FC1S_s1",
        "rt_fc1s_s2_kfold": "runs/RT_FC1S_s2",
        "rt_fc1m_s0_kfold": "runs/RT_FC1M_s0",
        "rt_fc1m_s1_kfold": "runs/RT_FC1M_s1",
        "rt_fc1m_s2_kfold": "runs/RT_FC1M_s2",
    }

    def save(name: str, final_preds: dict, remap_misleading: bool = False) -> None:
        if cfg.apply_na_rule:
            final_preds = apply_na_rule(final_preds, remap_misleading=remap_misleading)
        path = out_dir / f"{name}.csv"
        if cfg.data_dir in ("final_data", "retrain_data"):
            # Final stage: 4-column format
            df_out = pd.DataFrame({
                "id": test_df["id"],
                "promise_status": final_preds["t1"],
                "verification_timeline": final_preds["t4"],
                "evidence_status": final_preds["t2"],
                "evidence_quality": final_preds["t3"],
            })
            df_out.to_csv(path, index=False)                          # AIdea: N/A as-is
            kaggle_path = out_dir / f"{name}_kaggle.csv"
            df_out.replace("N/A", "-1").to_csv(kaggle_path, index=False)  # Kaggle: -1
            print(f"  → {path} + {kaggle_path}")
        else:
            pd.DataFrame({"id": test_df["id"],
                          "label": preds_to_label_strings(final_preds)}).to_csv(path, index=False)
            print(f"  → {path}")

    # ── kNN-LDL setup (lazy: only computed when a kNN combo is encountered) ──
    train_df, _ = load_dataframes(cfg.data_dir, cfg.use_augmented, cfg.aug_filename)
    # kNN retrieval pool: always use original 800 real samples (not synthetic aug)
    # Synthetic samples have different embedding distribution → pollutes kNN retrieval
    knn_train_df, _ = load_dataframes(cfg.data_dir, use_augmented=False)
    _knn_cache: dict[str, object] = {}  # "train_embs", "test_embs"

    # ── Stacking setup (lazy: built when first stacking combo is encountered) ─
    _stacking_meta: dict | None = None

    def _get_stacking_meta() -> dict:
        nonlocal _stacking_meta
        if _stacking_meta is not None:
            return _stacking_meta
        print("  [Stacking] building meta-learner ...")
        stacking_dirs = {
            "v4_roberta_kfold": KFOLD_DIRS["v4_roberta_kfold"],
            "v4_lert_kfold":    KFOLD_DIRS["v4_lert_kfold"],
        }
        _stacking_meta = build_stacking_lr(stacking_dirs, train_df, n_splits=5, seed=42)
        return _stacking_meta

    def _get_qwen3_embs(key: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings using Qwen3-Embedding (instruction-aware, frozen)."""
        if key in _knn_cache:
            return _knn_cache[key]  # type: ignore[return-value]
        _BACKBONE = "Qwen/Qwen3-Embedding-0.6B"
        _INSTRUCTION = "Retrieve similar ESG disclosure texts that share the same evidence quality level."
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tr_texts = knn_train_df["data"].tolist()
        te_texts = test_df["data"].tolist()
        print(f"  [kNN] Qwen3-Embedding: {len(tr_texts)} train, {len(te_texts)} test ...", end="", flush=True)
        tr_embs = extract_embeddings(tr_texts, _BACKBONE, batch_size=16, device=device, instruction=_INSTRUCTION).numpy()
        te_embs = extract_embeddings(te_texts, _BACKBONE, batch_size=16, device=device, instruction=_INSTRUCTION).numpy()
        print(" OK")
        _knn_cache[key] = (tr_embs, te_embs)
        return tr_embs, te_embs

    def _get_knn_embs(ckpt_key: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract train+test CLS embeddings from the first fold of a kfold run."""
        if ckpt_key.startswith("qwen3_emb"):
            return _get_qwen3_embs(ckpt_key)
        if ckpt_key in _knn_cache:
            return _knn_cache[ckpt_key]  # type: ignore[return-value]
        # Find checkpoint: prefer kfold fold1, fallback to CKPTS
        if ckpt_key in KFOLD_DIRS:
            ckpt_path = str(Path(KFOLD_DIRS[ckpt_key]) / "fold1" / "best.pt")
        elif ckpt_key in CKPTS:
            ckpt_path = CKPTS[ckpt_key]
        else:
            raise ValueError(f"kNN encoder source not found: {ckpt_key}")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"kNN encoder checkpoint not found: {ckpt_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        saved_cfg: Config = ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)
        train_ds = ESGDataset(knn_train_df, tokenizer, saved_cfg.max_length, has_labels=True)
        test_ds  = ESGDataset(test_df,      tokenizer, saved_cfg.max_length, has_labels=False)
        tr_loader = DataLoader(train_ds, batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
        te_loader = DataLoader(test_ds,  batch_size=saved_cfg.batch_size, shuffle=False, num_workers=0)
        _usp_k = getattr(saved_cfg, "use_span", False)
        model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc,
                           use_span=_usp_k).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"  [kNN] extracting embeddings from {ckpt_key} ...", end="", flush=True)
        tr_embs = extract_cls_embeddings(model, tr_loader, device)
        te_embs = extract_cls_embeddings(model, te_loader, device)
        del model
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(" OK")
        _knn_cache[ckpt_key] = (tr_embs, te_embs)
        return tr_embs, te_embs

    def _get_knn_embs_oof(ckpt_key: str) -> tuple[np.ndarray, np.ndarray]:
        """OOF CLS embeddings: each real train sample is embedded by the fold that did NOT train on it.
        Test embeddings are averaged across all folds.
        Falls back to _get_knn_embs for non-kfold sources."""
        cache_key = f"{ckpt_key}__oof"
        if cache_key in _knn_cache:
            return _knn_cache[cache_key]  # type: ignore[return-value]

        if ckpt_key not in KFOLD_DIRS:
            return _get_knn_embs(ckpt_key)

        from sklearn.model_selection import StratifiedKFold as _SKF

        kfold_dir = Path(KFOLD_DIRS[ckpt_key])
        n_folds = sum(1 for i in range(1, 20) if (kfold_dir / f"fold{i}" / "best.pt").exists())
        if n_folds == 0:
            raise FileNotFoundError(f"No fold checkpoints found in {kfold_dir}")

        # Load fold1 config for backbone / tokenizer info
        fold1_ckpt_path = kfold_dir / "fold1" / "best.pt"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold1_ckpt = torch.load(str(fold1_ckpt_path), map_location=device, weights_only=False)
        saved_cfg: Config = fold1_ckpt["cfg"]
        dc = getattr(saved_cfg, "deep_cascade", False)
        tokenizer = AutoTokenizer.from_pretrained(saved_cfg.backbone, trust_remote_code=True)

        # Reconstruct the exact splits used during training
        aug_fn  = getattr(saved_cfg, "aug_filename",  "train_data_augmented.csv")
        use_aug = getattr(saved_cfg, "use_augmented", True)
        train_full_split, _ = load_dataframes(saved_cfg.data_dir, use_aug, aug_fn)
        strat_col = train_full_split["promise_status"]
        skf = _SKF(n_splits=n_folds, shuffle=True, random_state=saved_cfg.seed)

        n_real: int = len(knn_train_df)        # 800 real-only samples
        oof_embs: np.ndarray | None = None     # shape (n_real, dim), filled per fold
        test_embs_acc: np.ndarray | None = None

        print(f"  [kNN OOF] {ckpt_key} ({n_folds} folds) ...")
        for fold_idx, (_, val_idx) in enumerate(skf.split(train_full_split, strat_col)):
            fold_ckpt_path = kfold_dir / f"fold{fold_idx + 1}" / "best.pt"
            if not fold_ckpt_path.exists():
                print(f"    fold{fold_idx+1}: MISSING, skipping")
                continue

            # Only the real samples (index < n_real) that fall in this fold's val set
            real_val_idx = val_idx[val_idx < n_real]
            if len(real_val_idx) == 0:
                continue

            ckpt = torch.load(str(fold_ckpt_path), map_location=device, weights_only=False)
            _usp_knn = getattr(saved_cfg, "use_span", False)
            model = ApproachA1(saved_cfg.backbone, saved_cfg.dropout, deep_cascade=dc,
                               use_span=_usp_knn).to(device)
            model.load_state_dict(ckpt["model"])

            # Embed only this fold's real val samples (preserve index order)
            fold_val_df = knn_train_df.iloc[real_val_idx].reset_index(drop=True)
            val_ds = ESGDataset(fold_val_df, tokenizer, saved_cfg.max_length, has_labels=True)
            val_loader = DataLoader(val_ds, batch_size=saved_cfg.batch_size,
                                    shuffle=False, num_workers=0)
            fold_tr_embs = extract_cls_embeddings(model, val_loader, device)

            if oof_embs is None:
                oof_embs = np.zeros((n_real, fold_tr_embs.shape[1]), dtype=np.float32)
            oof_embs[real_val_idx] = fold_tr_embs

            # Test embeddings (accumulate for averaging)
            te_ds = ESGDataset(test_df, tokenizer, saved_cfg.max_length, has_labels=False)
            te_loader = DataLoader(te_ds, batch_size=saved_cfg.batch_size,
                                   shuffle=False, num_workers=0)
            fold_te_embs = extract_cls_embeddings(model, te_loader, device)
            test_embs_acc = fold_te_embs if test_embs_acc is None else test_embs_acc + fold_te_embs

            del model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"    fold{fold_idx+1}: {len(real_val_idx)} real samples")

        if oof_embs is None or test_embs_acc is None:
            raise RuntimeError(f"No folds processed for {ckpt_key}")

        n_folds_accumulated = sum(1 for i in range(n_folds) if (kfold_dir / f"fold{i + 1}" / "best.pt").exists())
        te_embs_avg = test_embs_acc / n_folds_accumulated
        _knn_cache[cache_key] = (oof_embs, te_embs_avg)
        return oof_embs, te_embs_avg

    print("\nGenerating submissions (soft voting):")

    COMBOS = [
        # (filename,                    [ckpt keys to soft-vote ensemble])
        # ── V3 baselines (now using soft vote) ──────────────────────────────
        ("s1_pertask_aug_single",       ["pertask_aug"]),
        ("s4_top3_a1_soft",             ["pertask_aug", "pertask", "aug_ordinal"]),
        # ── V4 single cascade models ─────────────────────────────────────────
        ("s8_v4_dc_pertask_single",     ["v4_dc_pertask"]),
        ("s9_v4_dc_pertask_aug_single", ["v4_dc_pertask_aug"]),
        ("s10_v4_roberta_single",       ["v4_roberta_dc"]),
        # ── V4 ensembles ─────────────────────────────────────────────────────
        ("s11_v4_dc_2way",              ["v4_dc_pertask", "v4_dc_pertask_aug"]),
        ("s12_v4_dc_roberta_3way",      ["v4_dc_pertask", "v4_dc_pertask_aug", "v4_roberta_dc"]),
        ("s13_v4_dc_v3_5way",           ["pertask_aug", "pertask", "aug_ordinal",
                                         "v4_dc_pertask", "v4_dc_pertask_aug"]),
        ("s14_v4_all_6way",             ["pertask_aug", "pertask", "aug_ordinal",
                                         "v4_dc_pertask", "v4_dc_pertask_aug", "v4_roberta_dc"]),
        # ── V4 LERT/DeBERTa single models ────────────────────────────────────
        ("s19_v4_lert_single",          ["v4_lert_dc"]),
        ("s20_v4_lert_aug_single",      ["v4_lert_dc_aug"]),
        ("s21_v4_deberta97m_single",    ["v4_deberta97m_dc"]),
        # ── V4 multi-backbone ensembles (macBERT + RoBERTa + LERT) ───────────
        ("s22_v4_3backbone_6way",       ["v4_dc_pertask", "v4_dc_pertask_aug",
                                         "v4_roberta_dc", "v4_roberta_dc_aug",
                                         "v4_lert_dc",    "v4_lert_dc_aug"]),
        ("s23_v4_4backbone_8way",       ["v4_dc_pertask", "v4_dc_pertask_aug",
                                         "v4_roberta_dc", "v4_roberta_dc_aug",
                                         "v4_lert_dc",    "v4_lert_dc_aug",
                                         "v4_deberta97m_dc", "v4_deberta97m_dc_aug"]),
        # ── V4 kfold submissions ──────────────────────────────────────────────
        ("s15_v4_dc_kfold_single",      ["v4_dc_kfold"]),
        ("s16_v4_roberta_kfold_single", ["v4_roberta_kfold"]),
        ("s17_v4_kfold_2way",           ["v4_dc_kfold", "v4_roberta_kfold"]),
        ("s18_v4_kfold_v3_5way",        ["pertask_aug", "pertask", "aug_ordinal",
                                         "v4_dc_kfold", "v4_roberta_kfold"]),
        ("s24_v4_lert_kfold_single",    ["v4_lert_kfold"]),
        ("s25_v4_3kfold_3way",          ["v4_dc_kfold", "v4_roberta_kfold", "v4_lert_kfold"]),
        # ── V5 kfold + single hybrid ensembles ───────────────────────────────
        # 3kfold + roberta aug single (best test single was roberta-based)
        ("s26_v5_3kfold_robaug_4way",   ["v4_dc_kfold", "v4_roberta_kfold", "v4_lert_kfold",
                                          "v4_roberta_dc_aug"]),
        # 3kfold + all 3 backbone singles aug (more coverage)
        ("s27_v5_3kfold_3aug_6way",     ["v4_dc_kfold", "v4_roberta_kfold", "v4_lert_kfold",
                                          "v4_dc_pertask_aug", "v4_roberta_dc_aug", "v4_lert_dc_aug"]),
        # 3kfold + V3 pertask_aug (cross-version diversity)
        ("s28_v5_3kfold_v3_4way",       ["v4_dc_kfold", "v4_roberta_kfold", "v4_lert_kfold",
                                          "pertask_aug"]),
        # roberta kfold + roberta single (best kfold + same backbone for consistency)
        ("s29_v5_roberta_kfold_aug_2way", ["v4_roberta_kfold", "v4_roberta_dc_aug"]),
        # 2kfold (mac+rob) + 3 backbone aug singles
        ("s30_v5_2kfold_3aug_5way",     ["v4_dc_kfold", "v4_roberta_kfold",
                                          "v4_dc_pertask_aug", "v4_roberta_dc_aug", "v4_lert_dc_aug"]),
        # ── V5 weighted kfold ensembles (weights ∝ individual test scores) ──
        # Individual kfold scores: roberta=0.624, lert=0.605, macbert=0.571
        # Weight proportional to score: roberta:lert:macbert ≈ 2:1.5:1
        ("s31_v5_3kfold_weighted",      ["v4_dc_kfold", "v4_roberta_kfold", "v4_lert_kfold"],
         [1.0, 2.0, 1.5]),
        # roberta heavy: 3:1:1
        ("s32_v5_3kfold_robheavy",      ["v4_dc_kfold", "v4_roberta_kfold", "v4_lert_kfold"],
         [1.0, 3.0, 1.5]),
        # roberta + lert only (drop weakest macbert) ← NEW BEST 0.63993
        ("s33_v5_roberta_lert_kfold",   ["v4_roberta_kfold", "v4_lert_kfold"]),
        # ── V5b: weighted rob+lert variants ──────────────────────────────────
        # rob=0.624, lert=0.605 → try rob:lert = 2:1, 3:2, 4:3
        ("s34_v5b_roblert_2to1",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [2.0, 1.0]),
        ("s35_v5b_roblert_3to2",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [3.0, 2.0]),
        # rob+lert kfold + roberta aug single (no macBERT)
        ("s36_v5b_roblert_robaug",      ["v4_roberta_kfold", "v4_lert_kfold",
                                          "v4_roberta_dc_aug"]),
        # rob+lert kfold + lert aug single
        ("s37_v5b_roblert_lertaug",     ["v4_roberta_kfold", "v4_lert_kfold",
                                          "v4_lert_dc_aug"]),
        # rob+lert kfold + both aug singles
        ("s38_v5b_roblert_2aug",        ["v4_roberta_kfold", "v4_lert_kfold",
                                          "v4_roberta_dc_aug", "v4_lert_dc_aug"]),
        # ── V5c: fine-tune weight ratio around 3:2 (=1.5) ───────────────────
        # 3:2=1.50 best so far; 2:1=2.00 too much; trying 1.25~1.67 range
        ("s39_v5c_roblert_5to4",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [5.0, 4.0]),   # 1.25:1
        ("s40_v5c_roblert_4to3",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [4.0, 3.0]),   # 1.33:1
        ("s41_v5c_roblert_7to5",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [7.0, 5.0]),   # 1.40:1
        ("s42_v5c_roblert_5to3",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [5.0, 3.0]),   # 1.67:1
        # ── V5d: fine-tune below 1.25 ────────────────────────────────────────
        ("s43_v5d_roblert_9to8",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [9.0, 8.0]),   # 1.125:1
        ("s44_v5d_roblert_6to5",        ["v4_roberta_kfold", "v4_lert_kfold"],
         [6.0, 5.0]),   # 1.20:1
        ("s45_v5d_roblert_11to9",       ["v4_roberta_kfold", "v4_lert_kfold"],
         [11.0, 9.0]),  # 1.22:1
        ("s46_v5d_roblert_13to11",      ["v4_roberta_kfold", "v4_lert_kfold"],
         [13.0, 11.0]), # 1.18:1
        ("s47_v5d_roblert_11to10",      ["v4_roberta_kfold", "v4_lert_kfold"],
         [11.0, 10.0]), # 1.10:1
        ("s48_v5d_roblert_17to16",      ["v4_roberta_kfold", "v4_lert_kfold"],
         [17.0, 16.0]), # 1.0625:1
        ("s49_v5d_roblert_19to16",      ["v4_roberta_kfold", "v4_lert_kfold"],
         [19.0, 16.0]), # 1.1875:1
        # ── V6: ModernBERT kfold submissions ─────────────────────────────────
        ("s50_v6_modernbert_kfold",     ["v4_modernbert_kfold"]),
        ("s51_v6_rob_modern_2way",      ["v4_roberta_kfold", "v4_modernbert_kfold"]),
        ("s52_v6_lert_modern_2way",     ["v4_lert_kfold",   "v4_modernbert_kfold"]),
        ("s53_v6_3kfold",               ["v4_roberta_kfold", "v4_lert_kfold", "v4_modernbert_kfold"]),
        ("s54_v6_rob_modern_weighted",  ["v4_roberta_kfold", "v4_modernbert_kfold"],
         [1.0, 1.0]),  # placeholder, adjust after seeing s50 score
        # ── V7: FGM kfold submissions ─────────────────────────────────────────
        ("s55_v7_roberta_fgm_single",   ["v4_roberta_fgm_kfold"]),
        ("s56_v7_lert_fgm_single",      ["v4_lert_fgm_kfold"]),
        ("s57_v7_fgm_2way",             ["v4_roberta_fgm_kfold", "v4_lert_fgm_kfold"]),
        ("s58_v7_fgm_roblert_9to8",     ["v4_roberta_fgm_kfold", "v4_lert_fgm_kfold"],
         [9.0, 8.0]),  # best ratio from non-FGM search
        ("s59_v7_mix_rob_fgm_lert",     ["v4_roberta_fgm_kfold", "v4_lert_kfold"]),
        ("s60_v7_mix_rob_lert_fgm",     ["v4_roberta_kfold",     "v4_lert_fgm_kfold"]),
        ("s61_v7_mix_both_fgm_orig",    ["v4_roberta_fgm_kfold", "v4_lert_fgm_kfold",
                                          "v4_roberta_kfold",     "v4_lert_kfold"]),
        # ── V8: kNN-LDL retrieval augmented (4-tuple: name, keys, weights, knn_cfg) ──
        # knn_cfg keys: "knn_encoder" (ckpt key), "k" (neighbors), "alpha" (per-task dict)
        ("s62_v8_knn_k5_a03",  ["v4_roberta_kfold", "v4_lert_kfold"], [9.0, 8.0],
         {"knn_encoder": "v4_roberta_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.3, "t4": 0.1}}),
        ("s63_v8_knn_k5_a04",  ["v4_roberta_kfold", "v4_lert_kfold"], [9.0, 8.0],
         {"knn_encoder": "v4_roberta_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.4, "t4": 0.1}}),
        ("s64_v8_knn_k10_a03", ["v4_roberta_kfold", "v4_lert_kfold"], [9.0, 8.0],
         {"knn_encoder": "v4_roberta_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.3, "t4": 0.1}}),
        ("s65_v8_knn_k10_a04", ["v4_roberta_kfold", "v4_lert_kfold"], [9.0, 8.0],
         {"knn_encoder": "v4_roberta_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.4, "t4": 0.1}}),
        ("s66_v8_knn_k5_a05",  ["v4_roberta_kfold", "v4_lert_kfold"], [9.0, 8.0],
         {"knn_encoder": "v4_roberta_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.2, "t3": 0.5, "t4": 0.1}}),
        # ── V9: Stacking (LR meta-learner on T2+T3 OOF probs) ────────────────
        # 5-tuple: (name, keys, weights, None, {"stacking": True})
        ("s67_v9_stacking",    ["v4_roberta_kfold", "v4_lert_kfold"], [9.0, 8.0],
         None, {"stacking": True}),
        # ── V10: LogSigma + FGM(ε=0.1) retrained kfold ───────────────────────
        # Single backbone baselines
        ("s68_v10_roberta_logsigma",        ["v4_roberta_logsigma_kfold"]),
        ("s69_v10_lert_logsigma",           ["v4_lert_logsigma_kfold"]),
        # 2-way combinations (mirroring best from original weight search)
        ("s70_v10_logsigma_2way",           ["v4_roberta_logsigma_kfold",
                                              "v4_lert_logsigma_kfold"]),
        ("s71_v10_logsigma_5to4",           ["v4_roberta_logsigma_kfold",
                                              "v4_lert_logsigma_kfold"], [5.0, 4.0]),
        ("s72_v10_logsigma_9to8",           ["v4_roberta_logsigma_kfold",
                                              "v4_lert_logsigma_kfold"], [9.0, 8.0]),
        # Cross: LogSigma rob + original lert (diagnose per-backbone benefit)
        ("s73_v10_ls_rob_orig_lert",        ["v4_roberta_logsigma_kfold",
                                              "v4_lert_kfold"], [9.0, 8.0]),
        # Cross: original rob + LogSigma lert
        ("s74_v10_orig_rob_ls_lert",        ["v4_roberta_kfold",
                                              "v4_lert_logsigma_kfold"], [9.0, 8.0]),
        # 4-way: both LogSigma + both original (maximum diversity)
        ("s75_v10_logsigma_orig_4way",      ["v4_roberta_logsigma_kfold",
                                              "v4_lert_logsigma_kfold",
                                              "v4_roberta_kfold",
                                              "v4_lert_kfold"]),
        # ── V11: LLM-augmented kfold ─────────────────────────────────────────
        ("s76_v11_roberta_llmaug_single",   ["v4_roberta_llmaug_kfold"]),
        ("s77_v11_lert_llmaug_single",      ["v4_lert_llmaug_kfold"]),
        ("s78_v11_aug_2way",                ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"]),
        ("s79_v11_aug_5to4",                ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"],
         [5.0, 4.0]),
        ("s80_v11_aug_9to8",                ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"],
         [9.0, 8.0]),
        # Cross: aug rob + orig lert (diagnose per-backbone benefit)
        ("s81_v11_aug_rob_orig_lert",       ["v4_roberta_llmaug_kfold", "v4_lert_kfold"],
         [5.0, 4.0]),
        # Cross: orig rob + aug lert
        ("s82_v11_orig_rob_aug_lert",       ["v4_roberta_kfold", "v4_lert_llmaug_kfold"],
         [5.0, 4.0]),
        # 4-way: both aug + both orig (maximum diversity)
        ("s83_v11_aug_orig_4way",           ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                              "v4_roberta_kfold",        "v4_lert_kfold"]),
        # ── V13: new backbone single submissions ─────────────────────────────
        ("s89_v13_deberta_single",     ["v4_deberta_llmaug_kfold"]),
        ("s90_v13_electra_single",     ["v4_electra_llmaug_kfold"]),
        ("s91_v13_ernie_single",       ["v4_ernie_llmaug_kfold"]),
        # 2-way: best aug pair + each new backbone
        ("s92_v13_rob_lert_deberta",   ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                         "v4_deberta_llmaug_kfold"]),
        ("s93_v13_rob_lert_electra",   ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                         "v4_electra_llmaug_kfold"]),
        ("s94_v13_rob_lert_ernie",     ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                         "v4_ernie_llmaug_kfold"]),
        # 4-way: rob + lert + electra + ernie (skip deberta, weakest val)
        ("s95_v13_4way_no_deberta",    ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                         "v4_electra_llmaug_kfold", "v4_ernie_llmaug_kfold"]),
        # 5-way: all aug backbones
        ("s96_v13_5way_all",           ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                         "v4_deberta_llmaug_kfold", "v4_electra_llmaug_kfold",
                                         "v4_ernie_llmaug_kfold"]),
        # ── V14: hand features submissions ───────────────────────────────────
        ("s97_v14_rob_hf_single",      ["v4_roberta_llmaug_hf_kfold"]),
        ("s98_v14_lert_hf_single",     ["v4_lert_llmaug_hf_kfold"]),
        ("s99_v14_hf_2way",            ["v4_roberta_llmaug_hf_kfold", "v4_lert_llmaug_hf_kfold"]),
        ("s100_v14_hf_5to4",           ["v4_roberta_llmaug_hf_kfold", "v4_lert_llmaug_hf_kfold"],
         [5.0, 4.0]),
        ("s101_v14_hf_9to8",           ["v4_roberta_llmaug_hf_kfold", "v4_lert_llmaug_hf_kfold"],
         [9.0, 8.0]),
        # Cross: hf models + non-hf aug models (diversity)
        ("s102_v14_rob_hf_lert_aug",   ["v4_roberta_llmaug_hf_kfold", "v4_lert_llmaug_kfold"],
         [5.0, 4.0]),
        ("s103_v14_rob_aug_lert_hf",   ["v4_roberta_llmaug_kfold",    "v4_lert_llmaug_hf_kfold"],
         [5.0, 4.0]),
        # 4-way: both hf + both non-hf
        ("s104_v14_hf_aug_4way",       ["v4_roberta_llmaug_hf_kfold", "v4_lert_llmaug_hf_kfold",
                                         "v4_roberta_llmaug_kfold",    "v4_lert_llmaug_kfold"]),
        # ── V16: kNN-LDL on best aug ensemble (rob_aug + lert_aug) ──────────────
        # Base: rob_aug + lert_aug 1:1 (best ensemble 0.65994)
        # Encoder: rob_aug fold1 (trained on aug data, best single backbone)
        ("s110_v16_knn_k5_a03",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.3, "t4": 0.0}}),
        ("s111_v16_knn_k5_a04",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.4, "t4": 0.0}}),
        ("s112_v16_knn_k10_a03", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.3, "t4": 0.0}}),
        ("s113_v16_knn_k10_a04", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.4, "t4": 0.0}}),
        # V16b: fine-tune alpha around best (k=5, α=0.4)
        ("s114_v16_knn_k5_a045", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.45, "t4": 0.0}}),
        ("s115_v16_knn_k5_a05",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.5, "t4": 0.0}}),
        # V16c: add T2 alpha (V8 used T2=0.1, test if helpful on aug ensemble)
        ("s116_v16_knn_k5_t2t3", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.4, "t4": 0.0}}),
        # ── V17: LogSigma + FGM(ε=0.1) submissions ───────────────────────────────
        ("s126_v17_rob_ls_fgm_single", ["v4_roberta_ls_fgm_kfold"]),
        ("s127_v17_lert_ls_fgm_single",["v4_lert_ls_fgm_kfold"]),
        ("s128_v17_ls_fgm_2way",       ["v4_roberta_ls_fgm_kfold", "v4_lert_ls_fgm_kfold"]),
        # Cross: new ls_fgm + best aug (diversity test)
        ("s129_v17_rob_lsfgm_lert_aug",["v4_roberta_ls_fgm_kfold", "v4_lert_llmaug_kfold"]),
        ("s130_v17_rob_aug_lert_lsfgm",["v4_roberta_llmaug_kfold", "v4_lert_ls_fgm_kfold"]),
        # kNN-LDL on best ls_fgm ensemble (best params: k=5, T3=0.55, T2=0.1)
        ("s131_v17_ls_fgm_knn",        ["v4_roberta_ls_fgm_kfold", "v4_lert_ls_fgm_kfold"], None,
         {"knn_encoder": "v4_roberta_ls_fgm_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V18: best kNN-LDL + Not Clear threshold (tau=0.09 from OOF search) ──
        ("s132_v18_knn_tau009", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_threshold": 0.09}),
        # V16g: lert encoder for kNN (compare to rob encoder at same best params)
        ("s125_v16_lert_enc_k5_a055_t2", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_lert_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Average of rob+lert encoder kNN probs — requires custom handling, skip for now
        # V16f: continue pushing T3 alpha
        ("s122_v16_knn_k5_a06_t2",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.60, "t4": 0.0}}),
        ("s123_v16_knn_k5_a065_t2", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.65, "t4": 0.0}}),
        ("s124_v16_knn_k5_a07_t2",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.70, "t4": 0.0}}),
        # V16e: push T3 further + tune T2
        ("s119_v16_knn_k5_a055_t2", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s120_v16_knn_k5_a05_t2_15", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.15, "t3": 0.5, "t4": 0.0}}),
        ("s121_v16_knn_k3_a05_t2",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 3,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.5, "t4": 0.0}}),
        # V16d: T3=0.45/0.5 + T2=0.1
        ("s117_v16_knn_k5_a045_t2", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.45, "t4": 0.0}}),
        ("s118_v16_knn_k5_a05_t2",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.5, "t4": 0.0}}),
        # ── V19: OLL Loss (Ordinal Log-Loss, COLING 2022) ─────────────────────
        ("s133_v19_rob_t3oll_single",   ["v4_roberta_t3oll_kfold"]),
        ("s134_v19_lert_t3oll_single",  ["v4_lert_t3oll_kfold"]),
        ("s135_v19_t3oll_2way",         ["v4_roberta_t3oll_kfold", "v4_lert_t3oll_kfold"]),
        # Cross: OLL models + best aug models (diversity)
        ("s136_v19_rob_oll_lert_aug",   ["v4_roberta_t3oll_kfold",  "v4_lert_llmaug_kfold"]),
        ("s137_v19_rob_aug_lert_oll",   ["v4_roberta_llmaug_kfold", "v4_lert_t3oll_kfold"]),
        # kNN-LDL on OLL 2-way ensemble (best params: k=5, T3α=0.55, T2α=0.1)
        ("s138_v19_oll_knn",            ["v4_roberta_t3oll_kfold",  "v4_lert_t3oll_kfold"], None,
         {"knn_encoder": "v4_roberta_t3oll_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # kNN-LDL on OLL 2-way + aug 4-way (most diversity)
        ("s139_v19_oll_aug_4way_knn",   ["v4_roberta_t3oll_kfold", "v4_lert_t3oll_kfold",
                                          "v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_t3oll_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V20: kNN on real-only 800 samples (G4 fix: exclude synthetic aug from retrieval pool) ──
        # Baseline comparison: s119 used aug 969 pool, s140 uses real 800 only
        ("s140_v20_knn_real800", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V21: kNN softmax temperature search (sharper vs flatter weighting) ──
        # Base: same as s119/s140; only sim_temp changes
        ("s141_v21_knn_temp05",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}, "sim_temp": 0.5}),
        ("s142_v21_knn_temp20",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}, "sim_temp": 2.0}),
        # ── V22: T4 alpha search (previously always 0.0, never tested non-zero) ──
        ("s143_v22_t4a01",       ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.1}}),
        ("s144_v22_t4a02",       ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.2}}),
        ("s145_v22_t4a03",       ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.3}}),
        # ── V23: ensemble weight search (G5: rob stronger on T1/T2/T3, try rob-heavy) ──
        # G5 OOF: rob 0.7444 vs lert 0.7293 (+0.015); current best s119 is 1:1
        ("s146_v23_rob6_lert5_knn", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], [6.0, 5.0],
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s147_v23_rob5_lert4_knn", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], [5.0, 4.0],
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V15: T3 ordinal loss submissions ──────────────────────────────────
        ("s105_v15_rob_t3ord_single",  ["v4_roberta_t3ordinal_kfold"]),
        ("s106_v15_lert_t3ord_single", ["v4_lert_t3ordinal_kfold"]),
        ("s107_v15_t3ord_2way",        ["v4_roberta_t3ordinal_kfold", "v4_lert_t3ordinal_kfold"]),
        # Cross: new T3ord models + best aug models (diversity)
        ("s108_v15_rob_t3ord_lert_aug", ["v4_roberta_t3ordinal_kfold", "v4_lert_llmaug_kfold"]),
        ("s109_v15_rob_aug_lert_t3ord", ["v4_roberta_llmaug_kfold",    "v4_lert_t3ordinal_kfold"]),
        # ── V12: extreme weight search on aug ensemble ────────────────────────
        ("s84_v12_aug_2to1",                ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"],
         [2.0, 1.0]),
        ("s85_v12_aug_3to1",                ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"],
         [3.0, 1.0]),
        # aug rob + original lert (9:8 = best ratio from aug+aug search)
        ("s86_v12_aug_rob_orig_lert_9to8",  ["v4_roberta_llmaug_kfold", "v4_lert_kfold"],
         [9.0, 8.0]),
        # ── V12: 3-way aug + orig_rob ─────────────────────────────────────────
        ("s87_v12_aug2_orig_rob_3way",      ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                              "v4_roberta_kfold"]),
        ("s88_v12_aug2_orig_rob_weighted",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold",
                                              "v4_roberta_kfold"],
         [2.0, 2.0, 1.0]),
        # ── V24: kNN parameter search (T3α, T2α, k) ──────────────────────────
        # A: T3 alpha fine-tune (baseline T3α=0.55)
        ("s148_v24_t3a60", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.60, "t4": 0.0}}),
        ("s149_v24_t3a65", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.65, "t4": 0.0}}),
        # B: T2 alpha search (baseline T2α=0.1)
        ("s150_v24_t2a20", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.2, "t3": 0.55, "t4": 0.0}}),
        ("s151_v24_t2a30", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.3, "t3": 0.55, "t4": 0.0}}),
        # C: k value search (baseline k=5)
        ("s152_v24_k3", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 3,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s153_v24_k10", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V26: no-Misleading aug (940 samples) ─────────────────────────────
        ("s160_v26_nomislead_ensemble", ["v4_roberta_nomislead_kfold", "v4_lert_nomislead_kfold"]),
        ("s161_v26_nomislead_k10", ["v4_roberta_nomislead_kfold", "v4_lert_nomislead_kfold"], None,
         {"knn_encoder": "v4_roberta_nomislead_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V27: OOF embeddings for kNN (G1) ─────────────────────────────────
        ("s162_v27_oof_k10", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "use_oof_embs": True}),
        ("s163_v27_oof_k5", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "use_oof_embs": True}),
        # ── V28: LERT as kNN encoder ──────────────────────────────────────────
        ("s164_v28_lert_knn_k10", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_lert_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s165_v28_lert_knn_k5", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_lert_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V25: k continuation + T3α re-tune at k=10 ────────────────────────
        # k search continuation
        ("s154_v25_k15", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 15,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s155_v25_k20", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 20,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # T3α re-tune at k=10
        ("s156_v25_k10_t3a50", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.50, "t4": 0.0}}),
        ("s157_v25_k10_t3a60", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.60, "t4": 0.0}}),
        ("s158_v25_k10_t3a65", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.65, "t4": 0.0}}),
        ("s159_v25_k10_t3a70", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.70, "t4": 0.0}}),
        # ── V29: Not Clear threshold tuning at best config (k=10, T3α=0.55) ──────
        ("s166_v29_tau015", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_threshold": 0.15}),
        ("s167_v29_tau020", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_threshold": 0.20}),
        ("s168_v29_tau025", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_threshold": 0.25}),
        ("s169_v29_tau030", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_threshold": 0.30}),
        # ── V30: T4 alpha search at k=10 (V22 was k=5; best is now k=10) ────────
        ("s170_v30_k10_t4a005", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.05}}),
        ("s171_v30_k10_t4a010", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.10}}),
        ("s172_v30_k10_t4a015", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.15}}),
        ("s173_v30_k10_t4a020", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.20}}),
        # ── V30b: dynamic alpha on existing s153 models (zero cost, no retraining) ──
        ("s174_v30b_dyn_alpha", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "dynamic_alpha": True}),
        # ── V32: TTA (Test-Time Augmentation) on best s153 models ────────────────
        # TTA: n forward passes with dropout active, average softmax probs, then kNN
        ("s178_v32_tta5",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "tta_n": 5}),
        ("s179_v32_tta10", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "tta_n": 10}),
        # ── V33: 10-fold aug models ───────────────────────────────────────────────────
        ("s180_v33_rob10fold",  ["v5_roberta_kfold10_llmaug"], None,
         {"knn_encoder": "v5_roberta_kfold10_llmaug", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s181_v33_lert10fold", ["v5_lert_kfold10_llmaug"], None,
         {"knn_encoder": "v5_lert_kfold10_llmaug", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s182_v33_rob_lert_10fold", ["v5_roberta_kfold10_llmaug", "v5_lert_kfold10_llmaug"], None,
         {"knn_encoder": "v5_roberta_kfold10_llmaug", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Cross-ensemble: 10-fold + 5-fold (ensemble diversity)
        ("s183_v33_cross_4way", ["v5_roberta_kfold10_llmaug", "v5_lert_kfold10_llmaug",
                                  "v4_roberta_llmaug_kfold",   "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v5_roberta_kfold10_llmaug", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V36: Pseudo-labeling (aug + high-conf test, thr=0.80) ───────────────────────────
        ("s187_v36_rob_pseudo",  ["v4_roberta_pseudo_kfold"], None,
         {"knn_encoder": "v4_roberta_pseudo_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s188_v36_lert_pseudo", ["v4_lert_pseudo_kfold"], None,
         {"knn_encoder": "v4_roberta_pseudo_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s189_v36_rob_lert_pseudo", ["v4_roberta_pseudo_kfold", "v4_lert_pseudo_kfold"], None,
         {"knn_encoder": "v4_roberta_pseudo_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V37: Company-only kNN-LDL (pool restricted to same company) ──────────────────
        # Base: rob_aug + lert_aug 1:1 (best ensemble), encoder: RoBERTa aug kfold
        # company_knn=True → for each test sample, kNN pool = same-company train samples only
        # Fallback to global kNN when company pool < k
        # k=3 and k=5 only — k=10 from pool≈16 has no selectivity
        ("s190_v37_company_k5",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "company_knn": True}),
        ("s191_v37_company_k3",  ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 3,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "company_knn": True}),
        # ── V38: Few-shot demonstration (same-company labeled context in input) ─────────────
        # kNN encoder: use original roberta (fewshot model's embedding space is different)
        ("s192_v38_rob_fewshot",       ["v4_roberta_fewshot_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s193_v38_lert_fewshot",      ["v4_lert_fewshot_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s194_v38_rob_lert_fewshot",  ["v4_roberta_fewshot_kfold", "v4_lert_fewshot_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V39: Masked gradient (zero T2/T3/T4 loss for T1=No) ───────────────────────────
        ("s195_v39_rob_masked",        ["v4_roberta_masked_kfold"], None,
         {"knn_encoder": "v4_roberta_masked_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s196_v39_lert_masked",       ["v4_lert_masked_kfold"], None,
         {"knn_encoder": "v4_roberta_masked_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s197_v39_rob_lert_masked",   ["v4_roberta_masked_kfold", "v4_lert_masked_kfold"], None,
         {"knn_encoder": "v4_roberta_masked_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V40: T3 Not Clear boost ─────────────────────────────────────────────────────────
        ("s198_v40_rob_t3nc2",         ["v4_roberta_t3nc2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s199_v40_lert_t3nc2",        ["v4_lert_t3nc2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s200_v40_rob_lert_t3nc2",    ["v4_roberta_t3nc2_kfold", "v4_lert_t3nc2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s201_v40_rob_t3nc3",         ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s202_v40_lert_t3nc3",        ["v4_lert_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s203_v40_rob_lert_t3nc3",    ["v4_roberta_t3nc3_kfold", "v4_lert_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Cross: t3nc2/nc3 + original aug (diversity)
        ("s204_v40_nc2_rob_aug_lert",  ["v4_roberta_t3nc2_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s205_v40_nc2_rob_lert_aug",  ["v4_roberta_llmaug_kfold", "v4_lert_t3nc2_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V41: DeBERTa-710M ──────────────────────────────────────────────────────────────
        ("s206_v41_deberta710m",       ["v4_deberta710m_llmaug_kfold"], None,
         {"knn_encoder": "v4_deberta710m_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s207_v41_710m_rob_lert",     ["v4_deberta710m_llmaug_kfold",
                                         "v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V42: DeBERTa-320M + PERT-large ────────────────────────────────────────────────
        ("s208_v42_deberta320m",       ["v4_deberta320m_llmaug_kfold"], None,
         {"knn_encoder": "v4_deberta320m_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s209_v42_pert",              ["v4_pert_llmaug_kfold"], None,
         {"knn_encoder": "v4_pert_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s210_v42_320m_rob_lert",     ["v4_deberta320m_llmaug_kfold",
                                         "v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s211_v42_pert_rob_lert",     ["v4_pert_llmaug_kfold",
                                         "v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V43: Misleading → Not Clear ────────────────────────────────────────────────────
        ("s212_v43_rob_mislead_nc",    ["v4_roberta_mislead_nc_kfold"], None,
         {"knn_encoder": "v4_roberta_mislead_nc_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "remap_misleading": True}),
        ("s213_v43_lert_mislead_nc",   ["v4_lert_mislead_nc_kfold"], None,
         {"knn_encoder": "v4_roberta_mislead_nc_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "remap_misleading": True}),
        ("s214_v43_rob_lert_mislead_nc", ["v4_roberta_mislead_nc_kfold",
                                           "v4_lert_mislead_nc_kfold"], None,
         {"knn_encoder": "v4_roberta_mislead_nc_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "remap_misleading": True}),
        # Cross: mislead_nc + original aug (diverse T3 signal)
        ("s215_v43_mislead_nc_rob_aug_lert", ["v4_roberta_mislead_nc_kfold",
                                               "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_mislead_nc_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "remap_misleading": True}),
        ("s216_v43_rob_aug_lert_mislead_nc", ["v4_roberta_llmaug_kfold",
                                               "v4_lert_mislead_nc_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "remap_misleading": True}),
        # ── V44b: nc4/nc5 weight submissions ─────────────────────────────────────────────────
        ("s225_v44b_rob_nc4",    ["v4_roberta_t3nc4_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc4_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s226_v44b_lert_nc4",   ["v4_lert_t3nc4_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc4_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s227_v44b_rob_lert_nc4", ["v4_roberta_t3nc4_kfold", "v4_lert_t3nc4_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc4_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s228_v44b_rob_nc5",    ["v4_roberta_t3nc5_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc5_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V45: post-hoc T3 NC logit bias on best nc3 model ─────────────────────────────────
        # OOF tuning on nc3 says δ=-0.5 is best (nc3 over-predicts NC on OOF).
        # Test PUBLIC suggests more NC is needed → trying negative range to find balance.
        ("s229_v45_nc3_biasN05", ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_logit_bias": -0.5}),
        ("s230_v45_nc3_biasN03", ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_logit_bias": -0.3}),
        # ── V47: seed ensemble — same nc3, different kfold splits ────────────────────────────
        # seed=42 already trained (v4_roberta_t3nc3_kfold). Add s1 (123) and s2 (456).
        ("s235_v47_nc3_s1",      ["v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_s1_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s236_v47_nc3_s2",      ["v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_s2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s237_v47_nc3_3seed",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V48: extended seed ensemble + cross-backbone + R-Drop + SWA + distill ────────────
        # Phase 1: cross-backbone (lert_nc3 already trained, zero cost)
        ("s238_v48_rob_lert_nc3",     ["v4_roberta_t3nc3_kfold", "v4_lert_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s239_v48_3seed_lert",       ["v4_roberta_t3nc3_kfold",
                                       "v4_roberta_t3nc3_s1_kfold",
                                       "v4_roberta_t3nc3_s2_kfold",
                                       "v4_lert_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Phase 2: 5-seed roberta
        ("s240_v48_nc3_5seed",        ["v4_roberta_t3nc3_kfold",
                                       "v4_roberta_t3nc3_s1_kfold",
                                       "v4_roberta_t3nc3_s2_kfold",
                                       "v4_roberta_t3nc3_s3_kfold",
                                       "v4_roberta_t3nc3_s4_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Phase 2b: 5-seed rob + lert
        ("s241_v48_5seed_lert",       ["v4_roberta_t3nc3_kfold",
                                       "v4_roberta_t3nc3_s1_kfold",
                                       "v4_roberta_t3nc3_s2_kfold",
                                       "v4_roberta_t3nc3_s3_kfold",
                                       "v4_roberta_t3nc3_s4_kfold",
                                       "v4_lert_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Phase 2c: cross-backbone 3-seed (rob*3 + lert*3)
        ("s242_v48_rob3_lert3",       ["v4_roberta_t3nc3_kfold",
                                       "v4_roberta_t3nc3_s1_kfold",
                                       "v4_roberta_t3nc3_s2_kfold",
                                       "v4_lert_t3nc3_kfold",
                                       "v4_lert_t3nc3_s1_kfold",
                                       "v4_lert_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Phase 3: R-Drop, SWA, distill singles
        ("s243_v48_rdrop",            ["v4_roberta_t3nc3_rdrop_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_rdrop_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s244_v48_swa",              ["v4_roberta_t3nc3_swa_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_swa_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s245_v48_distill",          ["v4_roberta_t3nc3_distill_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_distill_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Phase 3b: best new method + 3-seed ensemble
        ("s246_v48_rdrop_3seed",      ["v4_roberta_t3nc3_rdrop_kfold",
                                       "v4_roberta_t3nc3_kfold",
                                       "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s247_v48_swa_3seed",        ["v4_roberta_t3nc3_swa_kfold",
                                       "v4_roberta_t3nc3_kfold",
                                       "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V49: kNN alpha re-tuning for 3-seed nc3 ensemble ──────────────────────────────────
        # Motivation: kNN α=0.55 was tuned for DB-loss model.  nc3 model is better calibrated
        # for Not Clear (nc_weight=3.0 already boosts NC), so kNN may be counterproductive.
        # Finding: kNN α=0.55 changes 40.5% of T3 predictions, nearly all NC→Clear.
        # Hypothesis: nc3 3-seed needs lower α (or α=0.0).  Sweep: 0.0, 0.10, 0.20, 0.30, 0.40, 0.45
        ("s248_v49_3seed_a0",    ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0}}),
        ("s249_v49_3seed_a10",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.10, "t4": 0.0}}),
        ("s250_v49_3seed_a20",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.20, "t4": 0.0}}),
        ("s251_v49_3seed_a30",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.30, "t4": 0.0}}),
        ("s252_v49_3seed_a40",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.40, "t4": 0.0}}),
        ("s253_v49_3seed_a45",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.45, "t4": 0.0}}),
        # k-sweep: smaller k=5 might give better-calibrated kNN label dist
        ("s254_v49_3seed_k5",    ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s255_v49_3seed_k3",    ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 3,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # T2 alpha sweep on 3-seed (T2α=0.0 vs 0.1 vs 0.2)
        ("s256_v49_3seed_t2a0",  ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.55, "t4": 0.0}}),
        # ── V49: fine-grained alpha sweep 0.40–0.55 to find exact peak ───────────────────────
        ("s257_v49_3seed_a42",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s258_v49_3seed_a44",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.44, "t4": 0.0}}),
        ("s259_v49_3seed_a47",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.47, "t4": 0.0}}),
        ("s260_v49_3seed_a50",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.50, "t4": 0.0}}),
        # ── V49: best alpha=0.42 applied to 3-seed+lert ensemble (s239 model set) ────────────
        ("s261_v49_3lert_a42",   ["v4_roberta_t3nc3_kfold",
                                   "v4_roberta_t3nc3_s1_kfold",
                                   "v4_roberta_t3nc3_s2_kfold",
                                   "v4_lert_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V49: T3α=0.42 on single nc3 model ────────────────────────────────────────────────
        ("s262_v49_nc3_single_a42", ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V50: TMix (embedding-level mixup) — needs training ────────────────────────────────
        # Train: python esg_main.py --mode run_v4 --skip_existing
        # α=0.4: moderate mix, preserves sample identity; α=1.0: aggressive uniform mix
        ("s263_v50_tmix04",       ["v4_roberta_t3nc3_tmix04_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_tmix04_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s264_v50_tmix10",       ["v4_roberta_t3nc3_tmix10_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_tmix10_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # TMix04 + 3-seed ensemble (diversity from tmix + 3 deterministic seeds)
        ("s265_v50_tmix04_3seed", ["v4_roberta_t3nc3_tmix04_kfold",
                                    "v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # TMix04 4-way: tmix + all 3 seeds for maximum diversity
        ("s266_v50_tmix04_4way",  ["v4_roberta_t3nc3_tmix04_kfold",
                                    "v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold",
                                    "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V52: 4/5-seed nc3 with optimal T3α=0.42 (zero cost, models already trained) ─────
        # V48 trained s3(seed=789) and s4(seed=999). They were submitted with α=0.55.
        # Re-test with α=0.42 which is now confirmed optimal for the 3-seed ensemble.
        ("s270_v52_4seed_a42",    ["v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold",
                                    "v4_roberta_t3nc3_s2_kfold",
                                    "v4_roberta_t3nc3_s3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s271_v52_5seed_a42",    ["v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold",
                                    "v4_roberta_t3nc3_s2_kfold",
                                    "v4_roberta_t3nc3_s3_kfold",
                                    "v4_roberta_t3nc3_s4_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V51: TAPT (domain-adapted backbone) — needs tapt mode first ──────────────────────
        # Single TAPT nc3
        ("s267_v51_tapt_nc3",     ["v4_roberta_tapt_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_tapt_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # TAPT + 3-seed ensemble (TAPT diverse from original seeds)
        ("s268_v51_tapt_3seed",   ["v4_roberta_tapt_t3nc3_kfold",
                                    "v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # TAPT 4-way (TAPT + all 3 original seeds)
        ("s269_v51_tapt_4way",    ["v4_roberta_tapt_t3nc3_kfold",
                                    "v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold",
                                    "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V53: Hard NC augmentation (LLM2LLM error-driven, condition_yx) ─────────────────────
        # Run gen_hard_nc.py first to produce train_data_aug_hardnc.csv, then run_v4 trains these.
        ("s272_v53_hardnc_rob",   ["v4_roberta_t3nc3_hardnc_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_hardnc_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s273_v53_hardnc_lert",  ["v4_lert_t3nc3_hardnc_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_hardnc_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # 3-seed: hardnc_rob + original s0 + s1 (best current ensemble = 0.70177)
        ("s274_v53_hardnc_3seed", ["v4_roberta_t3nc3_hardnc_kfold",
                                    "v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # 4-way: hardnc_rob + lert + original s0 + s1
        ("s275_v53_hardnc_4way",  ["v4_roberta_t3nc3_hardnc_kfold",
                                    "v4_lert_t3nc3_hardnc_kfold",
                                    "v4_roberta_t3nc3_kfold",
                                    "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V53b: Hard NC + nc_weight=1.0 (data rebalances, no extra NC boost) ─────────────────
        ("s276_v53b_hardnc_nw1_rob",  ["v4_roberta_hardnc_nw1_kfold"], None,
         {"knn_encoder": "v4_roberta_hardnc_nw1_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s277_v53b_hardnc_nw1_lert", ["v4_lert_hardnc_nw1_kfold"], None,
         {"knn_encoder": "v4_roberta_hardnc_nw1_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # 3-seed: nw1 + original s0 + s1
        ("s278_v53b_nw1_3seed",       ["v4_roberta_hardnc_nw1_kfold",
                                        "v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V53c: Diversity seed — add hardnc_rob (nc3) as 4th model in best 3-seed ──────────
        # hardnc_rob trained on different data distribution → diversity via data, not seed
        ("s279_v53c_div4seed",        ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_lert_t3nc3_kfold",
                                        "v4_roberta_t3nc3_hardnc_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # nw1 as diversity seed instead
        ("s280_v53c_div4seed_nw1",    ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_lert_t3nc3_kfold",
                                        "v4_roberta_hardnc_nw1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V58: Confidence-weighted ensemble (OOF F1 as weights, zero training cost) ───────────
        # OOF T3_F1: s0=0.761, s1=0.944, s2=0.929.  Public single: s0=0.693, s1=0.669, s2=0.646
        # NOTE: OOF and Public are poorly correlated — try both OOF-based and Public-based weights
        # OOF-weighted: upweight s1/s2
        ("s293_v58_oof_weight",       ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [0.76, 0.94, 0.93],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # Public-weighted: upweight s0
        ("s294_v58_pub_weight",       ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [0.69, 0.67, 0.65],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # s0-dominant (2:1:1)
        ("s295_v58_s0_dominant",      ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [2.0, 1.0, 1.0],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # s1-dominant (1:2:1) — s1 has highest OOF
        ("s296_v58_s1_dominant",      ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [1.0, 2.0, 1.0],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # s1 even more dominant
        ("s297_v58_s1_heavy",         ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [1.0, 3.0, 1.0],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # s1+s2 only (drop s0 which has worst OOF)
        ("s298_v58_s1s2_only",        ["v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # s1+s2 with s1 dominant
        ("s299_v58_s1s2_s1dom",       ["v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [2.0, 1.0],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # 1:2:1 with different T3α (maybe s1-dominant wants different kNN blend)
        ("s300_v58_s1dom_a35",        ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [1.0, 2.0, 1.0],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.35, "t4": 0.0}}),
        ("s301_v58_s1dom_a50",        ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [1.0, 2.0, 1.0],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.50, "t4": 0.0}}),
        ("s302_v58_s1dom_a30",        ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"],
                                       [1.0, 2.0, 1.0],
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.30, "t4": 0.0}}),
        # ── V57: Label smoothing 0.1 on nc3 (better calibration on small dataset) ──────────────
        ("s291_v57_ls01_single",      ["v4_roberta_t3nc3_ls01_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_ls01_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s292_v57_ls01_3seed",       ["v4_roberta_t3nc3_ls01_kfold",
                                        "v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V55: kNN temperature on best nc3 3-seed (sim_temp never tested with nc3+α=0.42) ──
        ("s284_v55_temp05",           ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}, "sim_temp": 0.5}),
        ("s285_v55_temp02",           ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}, "sim_temp": 0.2}),
        ("s286_v55_temp20",           ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}, "sim_temp": 2.0}),
        # ── V56: T2 alpha fine sweep on best nc3 3-seed (T2=30% of score, never fine-tuned) ──
        ("s287_v56_t2a00",            ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.42, "t4": 0.0}}),
        ("s288_v56_t2a05",            ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.05, "t3": 0.42, "t4": 0.0}}),
        ("s289_v56_t2a20",            ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.20, "t3": 0.42, "t4": 0.0}}),
        ("s290_v56_t2a30",            ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.30, "t3": 0.42, "t4": 0.0}}),
        # ── V54: nomislead (no synthetic Misleading) + nc3 weight ─────────────────────────────
        # 939 rows = aug - 30 Misleading. Model never predicts Misleading → remove training noise.
        ("s281_v54_nomislead_nc3",     ["v4_roberta_nomislead_nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_nomislead_nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # nomislead_nc3 as 3-seed with original s0+s1
        ("s282_v54_nomislead_3seed",  ["v4_roberta_nomislead_nc3_kfold",
                                        "v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # nomislead_nc3 as diversity 4th seed
        ("s283_v54_nomislead_div4",   ["v4_roberta_t3nc3_kfold",
                                        "v4_roberta_t3nc3_s1_kfold",
                                        "v4_lert_t3nc3_kfold",
                                        "v4_roberta_nomislead_nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── V46: T3 focal loss + NC weight=3 (focal down-weights easy Clear samples) ─────────
        ("s231_v46_rob_nc3_focal1", ["v4_roberta_t3nc3_focal1_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_focal1_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s232_v46_rob_nc3_focal2", ["v4_roberta_t3nc3_focal2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_focal2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s233_v46_lert_nc3_focal2", ["v4_lert_t3nc3_focal2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_focal2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s234_v46_rob_lert_nc3_focal2", ["v4_roberta_t3nc3_focal2_kfold", "v4_lert_t3nc3_focal2_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_focal2_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V44: kNN parameter search on t3nc3 models (new best s201=0.69308) ──────────────
        # Base: rob_t3nc3 single (best so far). Tune k and T3α.
        ("s217_v44_nc3_k5",    ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s218_v44_nc3_k15",   ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 15,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s219_v44_nc3_t3a45", ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.45, "t4": 0.0}}),
        ("s220_v44_nc3_t3a65", ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.65, "t4": 0.0}}),
        ("s221_v44_nc3_t3a70", ["v4_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.70, "t4": 0.0}}),
        # Cross: nc3 rob + original aug lert (diversity)
        ("s222_v44_nc3_rob_aug_lert", ["v4_roberta_t3nc3_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        ("s223_v44_nc3_aug_rob_lert",  ["v4_roberta_llmaug_kfold", "v4_lert_t3nc3_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # 4-way: nc3 rob+lert + aug rob+lert
        ("s224_v44_nc3_aug_4way", ["v4_roberta_t3nc3_kfold", "v4_lert_t3nc3_kfold",
                                    "v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V35: OOF threshold search → T3 Not Clear thr=0.44 on best base (s153 config) ──
        # OOF macro-F1: 0.6873 → 0.6882 (+0.0009); threshold only on Not Clear
        ("s186_v35_nc_thr44", ["v4_roberta_llmaug_kfold", "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "t3_nc_threshold": 0.44}),
        # ── V34: MacBERT + LLM aug 5-fold ────────────────────────────────────────────────
        ("s184_v34_macbert_llmaug", ["v4_macbert_llmaug_kfold"], None,
         {"knn_encoder": "v4_macbert_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Tri-model ensemble: MacBERT + RoBERTa + LERT (all aug, 5-fold)
        ("s185_v34_mac_rob_lert", ["v4_macbert_llmaug_kfold",
                                    "v4_roberta_llmaug_kfold",
                                    "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v4_roberta_llmaug_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── V31: Span + SCL retrained models (requires GPU training with new flags) ────
        # Train command:
        #   python esg_main.py --mode run_v4  (trains v5_roberta/lert_span_scl_kfold)
        # Ensemble of rob+lert span_scl, kNN at k=10, T3α=0.55 T2α=0.1
        ("s175_v31_rob_span_scl", ["v5_roberta_span_scl_kfold", "v5_lert_span_scl_kfold"], None,
         {"knn_encoder": "v5_roberta_span_scl_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # Dynamic alpha on top of V31 models
        ("s176_v31_span_scl_dyn", ["v5_roberta_span_scl_kfold", "v5_lert_span_scl_kfold"], None,
         {"knn_encoder": "v5_roberta_span_scl_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0},
          "dynamic_alpha": True}),
        # Cross-ensemble: V31 span_scl + best V4 aug (diverse representations)
        ("s177_v31_cross_ensemble", ["v5_roberta_span_scl_kfold", "v5_lert_span_scl_kfold",
                                      "v4_roberta_llmaug_kfold",   "v4_lert_llmaug_kfold"], None,
         {"knn_encoder": "v5_roberta_span_scl_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.55, "t4": 0.0}}),
        # ── 2nd Stage (B1): 1602-row official dataset, same nc3+kNN architecture ──────────
        # Train: python esg_main.py --mode run_v4 --data_dir 2nd_data --skip_existing
        # After training, gen_submissions also needs --data_dir 2nd_data --skip_existing
        # Start with T3α=0.42 (best from 1st stage), sweep later if needed
        ("s308_b1_single",    ["b1_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "b1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s309_b1_3seed",     ["b1_roberta_t3nc3_kfold",
                                "b1_roberta_t3nc3_s1_kfold",
                                "b1_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "b1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s310_b1_3seed_s1dom", ["b1_roberta_t3nc3_kfold",
                                  "b1_roberta_t3nc3_s1_kfold",
                                  "b1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "b1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── 3rd Stage (C1): reshuffled 1601-row dataset ──
        ("s311_c1_single",    ["c1_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s312_c1_3seed",     ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"], None,
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        ("s313_c1_3seed_s1dom", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.42, "t4": 0.0}}),
        # ── C1 alpha sweep ──
        ("s314_c1_s1dom_a30", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.30, "t4": 0.0}}),
        ("s315_c1_s1dom_a50", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.50, "t4": 0.0}}),
        ("s316_c1_s1dom_a60", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.60, "t4": 0.0}}),
        ("s317_c1_s1dom_a70", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.70, "t4": 0.0}}),
        ("s318_c1_s1dom_a80", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.80, "t4": 0.0}}),
        ("s319_c1_s1dom_a10", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.10, "t4": 0.0}}),
        ("s320_c1_s1dom_a20", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.1, "t3": 0.20, "t4": 0.0}}),
        ("s321_c1_s1dom_a00", ["c1_roberta_t3nc3_kfold",
                                "c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0}}),
        # ── C1 Logit Adjustment (GALA/Dynamic LA, CVPR 2024) ──
        # t3_la_tau: post-hoc logit adj on T3 fused probs to boost NC/Misleading
        # min_freq floor=0.05 prevents Misleading (0.1% train) from exploding
        ("s322_c1_la_tau03", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 0.3}),
        ("s323_c1_la_tau05", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 0.5}),
        ("s324_c1_la_tau07", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 0.7}),
        ("s325_c1_la_tau10", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 1.0}),
        ("s326_c1_la_tau15", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 1.5}),
        ("s327_c1_la_tau06", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 0.6}),
        ("s328_c1_la_tau08", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 0.8}),
        ("s329_c1_la_tau09", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "c1_roberta_t3nc3_kfold", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.00, "t4": 0.0},
          "t3_la_tau": 0.9}),

        # ── Qwen3-Embedding kNN: alpha sweep (no LA) ──────────────────────────
        ("s330_c1_qwen3_a10", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.10, "t4": 0.0}}),
        ("s331_c1_qwen3_a20", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.20, "t4": 0.0}}),
        ("s332_c1_qwen3_a30", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0}}),
        ("s333_c1_qwen3_a40", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),
        ("s334_c1_qwen3_a50", ["c1_roberta_t3nc3_kfold",
                               "c1_roberta_t3nc3_s1_kfold",
                               "c1_roberta_t3nc3_s2_kfold"],
                              [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.50, "t4": 0.0}}),

        # ── Qwen3-Embedding kNN + LA tau=0.7 ─────────────────────────────────
        ("s335_c1_qwen3_a20_la07", ["c1_roberta_t3nc3_kfold",
                                    "c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.20, "t4": 0.0},
          "t3_la_tau": 0.7}),
        ("s336_c1_qwen3_a30_la07", ["c1_roberta_t3nc3_kfold",
                                    "c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0},
          "t3_la_tau": 0.7}),
        ("s337_c1_qwen3_a40_la07", ["c1_roberta_t3nc3_kfold",
                                    "c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_la_tau": 0.7}),

        # ── Qwen3 kNN k sweep at best alpha ──────────────────────────────────
        ("s338_c1_qwen3_k5_a30", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0}}),

        # ── Qwen3 k=5 finer alpha sweep ───────────────────────────────────────
        ("s339_c1_qwen3_k5_a15", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.15, "t4": 0.0}}),
        ("s340_c1_qwen3_k5_a25", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.25, "t4": 0.0}}),
        ("s341_c1_qwen3_k5_a35", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.35, "t4": 0.0}}),
        ("s342_c1_qwen3_k5_a40", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),

        # ── Qwen3 k=5 best alpha + LA tau=0.7 ────────────────────────────────
        ("s343_c1_qwen3_k5_a30_la07", ["c1_roberta_t3nc3_kfold",
                                       "c1_roberta_t3nc3_s1_kfold",
                                       "c1_roberta_t3nc3_s2_kfold"],
                                      [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0},
          "t3_la_tau": 0.5}),
        ("s344_c1_qwen3_k5_a25_la07", ["c1_roberta_t3nc3_kfold",
                                       "c1_roberta_t3nc3_s1_kfold",
                                       "c1_roberta_t3nc3_s2_kfold"],
                                      [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.25, "t4": 0.0},
          "t3_la_tau": 0.5}),

        # ── C1 MC Dropout TTA ─────────────────────────────────────────────────
        ("s355_c1_tta5", ["c1_roberta_t3nc3_kfold",
                          "c1_roberta_t3nc3_s1_kfold",
                          "c1_roberta_t3nc3_s2_kfold"],
                         [1.0, 2.0, 1.0],
         {"tta_n": 5}),
        ("s356_c1_tta10", ["c1_roberta_t3nc3_kfold",
                           "c1_roberta_t3nc3_s1_kfold",
                           "c1_roberta_t3nc3_s2_kfold"],
                          [1.0, 2.0, 1.0],
         {"tta_n": 10}),
        # ── C1 TTA + Qwen3 kNN ───────────────────────────────────────────────
        ("s357_c1_tta5_qwen3_a40", ["c1_roberta_t3nc3_kfold",
                                    "c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"tta_n": 5, "knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),

        # ── Qwen3 k=5 alpha push higher ───────────────────────────────────────
        ("s345_c1_qwen3_k5_a45", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.45, "t4": 0.0}}),
        ("s346_c1_qwen3_k5_a50", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.50, "t4": 0.0}}),
        ("s347_c1_qwen3_k5_a55", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.55, "t4": 0.0}}),
        ("s348_c1_qwen3_k5_a60", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.60, "t4": 0.0}}),
        # ── Qwen3 k=5 alpha=0.40 + small LA tau sweep ────────────────────────
        ("s351_c1_qwen3_k5_a40_la02", ["c1_roberta_t3nc3_kfold",
                                       "c1_roberta_t3nc3_s1_kfold",
                                       "c1_roberta_t3nc3_s2_kfold"],
                                      [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_la_tau": 0.2}),
        ("s352_c1_qwen3_k5_a40_la03", ["c1_roberta_t3nc3_kfold",
                                       "c1_roberta_t3nc3_s1_kfold",
                                       "c1_roberta_t3nc3_s2_kfold"],
                                      [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_la_tau": 0.3}),
        ("s353_c1_qwen3_k5_a40_la04", ["c1_roberta_t3nc3_kfold",
                                       "c1_roberta_t3nc3_s1_kfold",
                                       "c1_roberta_t3nc3_s2_kfold"],
                                      [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_la_tau": 0.4}),
        ("s354_c1_qwen3_k5_a40_la05", ["c1_roberta_t3nc3_kfold",
                                       "c1_roberta_t3nc3_s1_kfold",
                                       "c1_roberta_t3nc3_s2_kfold"],
                                      [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_la_tau": 0.5}),

        # ── Qwen3 k=3 at best alpha ───────────────────────────────────────────
        ("s349_c1_qwen3_k3_a40", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 3,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),
        ("s350_c1_qwen3_k7_a40", ["c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 7,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),

        # ── D1 nc3: LLM-aug ensemble baseline ────────────────────────────────
        ("s360_d1nc3_3seed",     ["d1_roberta_t3nc3_kfold",
                                  "d1_roberta_t3nc3_s1_kfold",
                                  "d1_roberta_t3nc3_s2_kfold"], None, None),
        ("s361_d1nc3_s1dom",     ["d1_roberta_t3nc3_kfold",
                                  "d1_roberta_t3nc3_s1_kfold",
                                  "d1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0], None),
        # ── D1 nc3 + Qwen3 kNN ───────────────────────────────────────────────
        ("s362_d1nc3_qwen3_a40", ["d1_roberta_t3nc3_kfold",
                                  "d1_roberta_t3nc3_s1_kfold",
                                  "d1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),
        ("s363_d1nc3_qwen3_a30", ["d1_roberta_t3nc3_kfold",
                                  "d1_roberta_t3nc3_s1_kfold",
                                  "d1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0}}),
        # ── D1 nc5: stronger NC weight ensemble ───────────────────────────────
        ("s364_d1nc5_s1dom",     ["d1_roberta_t3nc5_kfold",
                                  "d1_roberta_t3nc5_s1_kfold",
                                  "d1_roberta_t3nc5_s2_kfold"],
                                 [1.0, 2.0, 1.0], None),
        ("s365_d1nc5_qwen3_a40", ["d1_roberta_t3nc5_kfold",
                                  "d1_roberta_t3nc5_s1_kfold",
                                  "d1_roberta_t3nc5_s2_kfold"],
                                 [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),
        # ── D1 + C1 cross-model ensemble ─────────────────────────────────────
        ("s366_d1c1_cross",      ["d1_roberta_t3nc3_kfold",
                                  "d1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 1.0, 1.0, 2.0, 1.0], None),
        ("s367_d1c1_cross_q40",  ["d1_roberta_t3nc3_kfold",
                                  "d1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_kfold",
                                  "c1_roberta_t3nc3_s1_kfold",
                                  "c1_roberta_t3nc3_s2_kfold"],
                                 [1.0, 1.0, 1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),

        # ── E2: Hard-Mixup models (3-seed, nc3) ──────────────────────────────
        # s370: E2 3-seed ensemble (no kNN)
        ("s370_e2hm_3seed",       ["e2_roberta_hm_kfold",
                                   "e2_roberta_hm_s1_kfold",
                                   "e2_roberta_hm_s2_kfold"],
                                  [1.0, 2.0, 1.0], None),
        # s371: E2 + Qwen3 kNN (alpha=0.40 on T3)
        ("s371_e2hm_qwen3_a40",   ["e2_roberta_hm_kfold",
                                   "e2_roberta_hm_s1_kfold",
                                   "e2_roberta_hm_s2_kfold"],
                                  [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),
        # s372: E2 + C1 cross-ensemble (no kNN)
        ("s372_e2c1_cross",       ["e2_roberta_hm_kfold",
                                   "e2_roberta_hm_s1_kfold",
                                   "e2_roberta_hm_s2_kfold",
                                   "c1_roberta_t3nc3_kfold",
                                   "c1_roberta_t3nc3_s1_kfold",
                                   "c1_roberta_t3nc3_s2_kfold"],
                                  [1.0, 2.0, 1.0, 1.0, 2.0, 1.0], None),
        # s373: E2 + C1 cross-ensemble + Qwen3 kNN
        ("s373_e2c1_cross_q40",   ["e2_roberta_hm_kfold",
                                   "e2_roberta_hm_s1_kfold",
                                   "e2_roberta_hm_s2_kfold",
                                   "c1_roberta_t3nc3_kfold",
                                   "c1_roberta_t3nc3_s1_kfold",
                                   "c1_roberta_t3nc3_s2_kfold"],
                                  [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),

        # ── F3: Prototype Contrastive Loss models ─────────────────────────────
        # s374: F3 3-seed ensemble (no kNN)
        ("s374_f3proto_3seed",    ["f3_roberta_proto_kfold",
                                   "f3_roberta_proto_s1_kfold",
                                   "f3_roberta_proto_s2_kfold"],
                                  [1.0, 2.0, 1.0], None),
        # s375: F3 + Qwen3 kNN (alpha=0.40 on T3)
        ("s375_f3proto_qwen3_a40", ["f3_roberta_proto_kfold",
                                    "f3_roberta_proto_s1_kfold",
                                    "f3_roberta_proto_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),
        # s376: F3 + C1 cross-ensemble (no kNN)
        ("s376_f3c1_cross",       ["f3_roberta_proto_kfold",
                                   "f3_roberta_proto_s1_kfold",
                                   "f3_roberta_proto_s2_kfold",
                                   "c1_roberta_t3nc3_kfold",
                                   "c1_roberta_t3nc3_s1_kfold",
                                   "c1_roberta_t3nc3_s2_kfold"],
                                  [1.0, 2.0, 1.0, 1.0, 2.0, 1.0], None),
        # s377: F3 + C1 cross-ensemble + Qwen3 kNN
        ("s377_f3c1_cross_q40",   ["f3_roberta_proto_kfold",
                                   "f3_roberta_proto_s1_kfold",
                                   "f3_roberta_proto_s2_kfold",
                                   "c1_roberta_t3nc3_kfold",
                                   "c1_roberta_t3nc3_s1_kfold",
                                   "c1_roberta_t3nc3_s2_kfold"],
                                  [1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}}),

        # ── NC logit bias sweep (C1 baseline) ─────────────────────────────────
        # s378-s383: C1 3-seed + Qwen3 k5 α=0.40 with varying NC logit boost
        ("s378_c1_q40_ncb03", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        ("s379_c1_q40_ncb05", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.5}),
        ("s380_c1_q40_ncb07", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.7}),
        ("s381_c1_q40_ncb10", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 1.0}),
        ("s382_c1_q40_ncb15", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 1.5}),
        ("s383_c1_q40_ncb20", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 2.0}),

        # ── Fine-grained NC bias sweep around 0.3 (best so far) ──────────────
        ("s384_c1_q40_ncb01", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.1}),
        ("s385_c1_q40_ncb02", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.2}),
        ("s386_c1_q40_ncb04", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                "c1_roberta_t3nc3_s2_kfold"],
                               [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.4}),

        # ── Company NC calibration + kNN (beta applied before kNN fusion) ────
        # s393: global bias=0.3 + company beta=0.3
        ("s393_c1_q40_nb03_cb03", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3, "t3_company_nc_beta": 0.3}),
        # s394: global bias=0.3 + company beta=0.5
        ("s394_c1_q40_nb03_cb05", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3, "t3_company_nc_beta": 0.5}),
        # s395: global bias=0.3 + company beta=0.8
        ("s395_c1_q40_nb03_cb08", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3, "t3_company_nc_beta": 0.8}),
        # s396: global bias=0.0 + company beta=0.5 (pure company prior)
        ("s396_c1_q40_nb00_cb05", ["c1_roberta_t3nc3_kfold","c1_roberta_t3nc3_s1_kfold",
                                    "c1_roberta_t3nc3_s2_kfold"],
                                   [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_company_nc_beta": 0.5}),

        # ── G1: MacBERT at C1 stage ───────────────────────────────────────────
        # s400: G1 MacBERT 3-seed (s1-dominant) + Qwen3 kNN k5 α0.40 NC-bias=0.3
        ("s400_g1_mac_3s_q40_ncb03", ["g1_macbert_t3nc3_kfold","g1_macbert_t3nc3_s1_kfold",
                                       "g1_macbert_t3nc3_s2_kfold"],
                                      [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s401: G1 MacBERT single seed (quick probe)
        ("s401_g1_mac_single_q40_ncb03", ["g1_macbert_t3nc3_kfold"], None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── G2: LERT at C1 stage ─────────────────────────────────────────────
        # s402: G2 LERT 3-seed (s1-dominant) + Qwen3 kNN k5 α0.40 NC-bias=0.3
        ("s402_g2_lert_3s_q40_ncb03", ["g2_lert_t3nc3_kfold","g2_lert_t3nc3_s1_kfold",
                                        "g2_lert_t3nc3_s2_kfold"],
                                       [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── G1+G2+C1 Cross-backbone ensemble ─────────────────────────────────
        # s403: C1(RoBERTa) + G1(MacBERT) + G2(LERT) = 9-model ensemble, uniform weight
        ("s403_c1g1g2_9way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",    "c1_roberta_t3nc3_s1_kfold",    "c1_roberta_t3nc3_s2_kfold",
          "g1_macbert_t3nc3_kfold",    "g1_macbert_t3nc3_s1_kfold",    "g1_macbert_t3nc3_s2_kfold",
          "g2_lert_t3nc3_kfold",       "g2_lert_t3nc3_s1_kfold",       "g2_lert_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s404: C1+G1+G2 9-way with s1-dominant per backbone
        ("s404_c1g1g2_9way_wtd_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",    "c1_roberta_t3nc3_s1_kfold",    "c1_roberta_t3nc3_s2_kfold",
          "g1_macbert_t3nc3_kfold",    "g1_macbert_t3nc3_s1_kfold",    "g1_macbert_t3nc3_s2_kfold",
          "g2_lert_t3nc3_kfold",       "g2_lert_t3nc3_s1_kfold",       "g2_lert_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s405: C1(RoBERTa) + G1(MacBERT) 6-way (if LERT doesn't add value)
        ("s405_c1g1_6way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",    "c1_roberta_t3nc3_s1_kfold",    "c1_roberta_t3nc3_s2_kfold",
          "g1_macbert_t3nc3_kfold",    "g1_macbert_t3nc3_s1_kfold",    "g1_macbert_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── G3: RoBERTa + SCL ────────────────────────────────────────────────
        # s406: G3 RoBERTa+SCL 3-seed + Qwen3 kNN
        ("s406_g3_rob_scl_3s_q40_ncb03", ["g3_roberta_scl_kfold","g3_roberta_scl_s1_kfold",
                                           "g3_roberta_scl_s2_kfold"],
                                          [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s407: G3(SCL) + C1 ensemble
        ("s407_g3scl_c1_6way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",    "c1_roberta_t3nc3_s1_kfold",    "c1_roberta_t3nc3_s2_kfold",
          "g3_roberta_scl_kfold",      "g3_roberta_scl_s1_kfold",      "g3_roberta_scl_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── G4: MacBERT + SCL ────────────────────────────────────────────────
        # s408: G4 MacBERT+SCL 3-seed + Qwen3 kNN
        ("s408_g4_mac_scl_3s_q40_ncb03", ["g4_macbert_scl_kfold","g4_macbert_scl_s1_kfold",
                                           "g4_macbert_scl_s2_kfold"],
                                          [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s409: G4(MacBERT+SCL) + G1(MacBERT) 6-way (same backbone, diversity via SCL)
        ("s409_g4scl_g1_6way_q40_ncb03",
         ["g1_macbert_t3nc3_kfold",    "g1_macbert_t3nc3_s1_kfold",    "g1_macbert_t3nc3_s2_kfold",
          "g4_macbert_scl_kfold",      "g4_macbert_scl_s1_kfold",      "g4_macbert_scl_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s410: Full 12-model ensemble: C1(Rob) + G1(Mac) + G2(LERT) + G3(Rob+SCL)
        ("s410_c1g1g2g3_12way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",    "c1_roberta_t3nc3_s1_kfold",    "c1_roberta_t3nc3_s2_kfold",
          "g1_macbert_t3nc3_kfold",    "g1_macbert_t3nc3_s1_kfold",    "g1_macbert_t3nc3_s2_kfold",
          "g2_lert_t3nc3_kfold",       "g2_lert_t3nc3_s1_kfold",       "g2_lert_t3nc3_s2_kfold",
          "g3_roberta_scl_kfold",      "g3_roberta_scl_s1_kfold",      "g3_roberta_scl_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0,  1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── G5: RoBERTa + company_emb_dim=64 ─────────────────────────────────
        # s411: G5 company_emb 3-seed + Qwen3 kNN k5 α0.40 NC-bias=0.3
        ("s411_g5_rob_cemb_3s_q40_ncb03",
         ["g5_roberta_cemb_kfold", "g5_roberta_cemb_s1_kfold", "g5_roberta_cemb_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s412: G5 + C1 6-way (compare company_emb vs plain RoBERTa)
        ("s412_g5cemb_c1_6way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",    "c1_roberta_t3nc3_s1_kfold",    "c1_roberta_t3nc3_s2_kfold",
          "g5_roberta_cemb_kfold",     "g5_roberta_cemb_s1_kfold",     "g5_roberta_cemb_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s413: G5 + G1(Mac) + G2(LERT) 9-way (diverse backbones + company_emb)
        ("s413_g5cemb_g1g2_9way_q40_ncb03",
         ["g5_roberta_cemb_kfold",  "g5_roberta_cemb_s1_kfold",  "g5_roberta_cemb_s2_kfold",
          "g1_macbert_t3nc3_kfold", "g1_macbert_t3nc3_s1_kfold", "g1_macbert_t3nc3_s2_kfold",
          "g2_lert_t3nc3_kfold",    "g2_lert_t3nc3_s1_kfold",    "g2_lert_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── H1: RoBERTa + Attention Pooling ──────────────────────────────────
        # s430: H1 attn pool 3-seed
        ("s430_h1_attn_3s_q40_ncb03",
         ["h1_roberta_attn_kfold", "h1_roberta_attn_s1_kfold", "h1_roberta_attn_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s431: H1 + C1 6-way
        ("s431_h1attn_c1_6way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",  "c1_roberta_t3nc3_s1_kfold",  "c1_roberta_t3nc3_s2_kfold",
          "h1_roberta_attn_kfold",   "h1_roberta_attn_s1_kfold",   "h1_roberta_attn_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── H2: RoBERTa + Feature Prepending ─────────────────────────────────
        # s432: H2 feature prepend 3-seed
        ("s432_h2_prep_3s_q40_ncb03",
         ["h2_roberta_prep_kfold", "h2_roberta_prep_s1_kfold", "h2_roberta_prep_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s433: H2 + C1 6-way
        ("s433_h2prep_c1_6way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",  "c1_roberta_t3nc3_s1_kfold",  "c1_roberta_t3nc3_s2_kfold",
          "h2_roberta_prep_kfold",   "h2_roberta_prep_s1_kfold",   "h2_roberta_prep_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── H3: RoBERTa + AttnPool + FeaturePrepend (combined) ───────────────
        # s434: H3 combined 3-seed
        ("s434_h3_attn_prep_3s_q40_ncb03",
         ["h3_roberta_attn_prep_kfold", "h3_roberta_attn_prep_s1_kfold", "h3_roberta_attn_prep_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),
        # s435: H3 + C1 6-way (best of both worlds)
        ("s435_h3_c1_6way_q40_ncb03",
         ["c1_roberta_t3nc3_kfold",        "c1_roberta_t3nc3_s1_kfold",        "c1_roberta_t3nc3_s2_kfold",
          "h3_roberta_attn_prep_kfold",    "h3_roberta_attn_prep_s1_kfold",    "h3_roberta_attn_prep_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ═══════ v4.1: R-Drop + SWA + SharpReCL + MR2 experiments ═══════

        # ── C1R: R-Drop + SWA ──
        ("s440_c1r_3seed",
         ["c1r_roberta_t3nc3_kfold", "c1r_roberta_t3nc3_s1_kfold", "c1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── C1S: SharpReCL ──
        ("s441_c1s_3seed",
         ["c1s_roberta_t3nc3_kfold", "c1s_roberta_t3nc3_s1_kfold", "c1s_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── C1M: MR2 ──
        ("s442_c1m_3seed",
         ["c1m_roberta_t3nc3_kfold", "c1m_roberta_t3nc3_s1_kfold", "c1m_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── C1RS: R-Drop + SharpReCL combined ──
        ("s443_c1rs_3seed",
         ["c1rs_roberta_t3nc3_kfold", "c1rs_roberta_t3nc3_s1_kfold", "c1rs_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── C1R + C1 cross ensemble (6-way) ──
        ("s444_c1r_c1_6way",
         ["c1_roberta_t3nc3_kfold",  "c1_roberta_t3nc3_s1_kfold",  "c1_roberta_t3nc3_s2_kfold",
          "c1r_roberta_t3nc3_kfold", "c1r_roberta_t3nc3_s1_kfold", "c1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── C1S + C1 cross ensemble (6-way) ──
        ("s445_c1s_c1_6way",
         ["c1_roberta_t3nc3_kfold",  "c1_roberta_t3nc3_s1_kfold",  "c1_roberta_t3nc3_s2_kfold",
          "c1s_roberta_t3nc3_kfold", "c1s_roberta_t3nc3_s1_kfold", "c1s_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── C1RS + C1 cross ensemble (6-way) ──
        ("s446_c1rs_c1_6way",
         ["c1_roberta_t3nc3_kfold",   "c1_roberta_t3nc3_s1_kfold",   "c1_roberta_t3nc3_s2_kfold",
          "c1rs_roberta_t3nc3_kfold", "c1rs_roberta_t3nc3_s1_kfold", "c1rs_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── 9-way: C1 + C1R + C1S ──
        ("s447_c1_c1r_c1s_9way",
         ["c1_roberta_t3nc3_kfold",  "c1_roberta_t3nc3_s1_kfold",  "c1_roberta_t3nc3_s2_kfold",
          "c1r_roberta_t3nc3_kfold", "c1r_roberta_t3nc3_s1_kfold", "c1r_roberta_t3nc3_s2_kfold",
          "c1s_roberta_t3nc3_kfold", "c1s_roberta_t3nc3_s1_kfold", "c1s_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0,  1.0, 2.0, 1.0,  1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── C1R no kNN (pure neural) ──
        ("s448_c1r_3seed_noknn",
         ["c1r_roberta_t3nc3_kfold", "c1r_roberta_t3nc3_s1_kfold", "c1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0], None),

        # ── C1S no kNN (pure neural) ──
        ("s449_c1s_3seed_noknn",
         ["c1s_roberta_t3nc3_kfold", "c1s_roberta_t3nc3_s1_kfold", "c1s_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0], None),

        # ═══════ Final Stage (FC1*): trained on final_data ═══════

        # ── FC1R singles ──
        ("s450_fc1r_s1", ["fc1r_roberta_t3nc3_s1_kfold"], None, None),
        ("s451_fc1r_s2", ["fc1r_roberta_t3nc3_s2_kfold"], None, None),

        # ── FC1R 2-seed (s1+s2) ──
        ("s452_fc1r_2seed",
         ["fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"], None, None),

        # ── FC1R 3-seed ──
        ("s453_fc1r_3seed",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0], None),

        # ── FC1 baseline ──
        ("s454_fc1_single", ["fc1_roberta_t3nc3_kfold"], None, None),

        # ── FC1S 3-seed ──
        ("s458_fc1s_3seed",
         ["fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold"],
         None, None),

        # ── FC1M single ──
        ("s459_fc1m_single", ["fc1m_roberta_t3nc3_kfold"], None, None),

        # ── Cross-type: FC1R(2) + FC1S ──
        ("s461_fc1r2_fc1s_3way",
         ["fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold"], None, None),

        # ── Cross-type: FC1R(2) + FC1M ──
        ("s462_fc1r2_fc1m_3way",
         ["fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1m_roberta_t3nc3_kfold"], None, None),

        # ── FC1R(3) + FC1S(3) 6-way ──
        ("s463_fc1r3_fc1s3_6way",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold"],
         None, None),

        # ── FC1R(3) + FC1M 4-way ──
        ("s464_fc1r3_fc1m_4way",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1m_roberta_t3nc3_kfold"], None, None),

        # ── FC1R 3-seed + kNN ──
        ("s466_fc1r3_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ── FC1R(3) + FC1S(3) 6-way + kNN ──
        ("s467_fc1_6way_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0},
          "t3_nc_logit_bias": 0.3}),

        # ═══ kNN alpha sweep on FC1R 3-seed ═══
        ("s470_fc1r3_knn_a20",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.20, "t4": 0.0}, "t3_nc_logit_bias": 0.3}),
        ("s471_fc1r3_knn_a30",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0}, "t3_nc_logit_bias": 0.3}),
        ("s472_fc1r3_knn_a50",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.50, "t4": 0.0}, "t3_nc_logit_bias": 0.3}),
        ("s473_fc1r3_knn_a60",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.60, "t4": 0.0}, "t3_nc_logit_bias": 0.3}),

        # ═══ NC bias sweep on FC1R 3-seed + kNN a40 ═══
        ("s474_fc1r3_knn_b00",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.0}),
        ("s475_fc1r3_knn_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s476_fc1r3_knn_b05",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.5}),

        # ═══ 6-way + kNN alpha sweep ═══
        ("s477_6way_knn_a30",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0}, "t3_nc_logit_bias": 0.3}),
        ("s478_6way_knn_a50",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.50, "t4": 0.0}, "t3_nc_logit_bias": 0.3}),

        # ═══ FC1R 3-seed + FC1 baseline cross ensemble ═══
        ("s479_fc1r3_fc1_4way",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1_roberta_t3nc3_kfold"], None, None),
        ("s480_fc1r3_fc1_4way_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.3}),

        # ═══ Fine sweep: bias=0.1 + alpha variations (s475 was best: a40 b01) ═══
        ("s481_fc1r3_a30_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.30, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s482_fc1r3_a50_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.50, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s483_fc1r3_a35_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.35, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s484_fc1r3_a45_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.45, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ 6-way + bias=0.1 ═══
        ("s485_6way_a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s486_6way_a50_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.50, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ FC1R3 + FC1 4-way + bias=0.1 ═══
        ("s487_fc1r3_fc1_a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1_roberta_t3nc3_kfold"], None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ bias=0.15, 0.20 fine tune ═══
        ("s488_fc1r3_a40_b015",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.15}),
        ("s489_fc1r3_a40_b02",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.2}),

        # ═══ kNN k-value sweep (best config: α=0.40, bias=0.1) ═══
        ("s490_fc1r3_k3_a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 3,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s491_fc1r3_k7_a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 7,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s492_fc1r3_k10_a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 10,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ T2 also gets kNN alpha (T3 best config + T2 kNN) ═══
        ("s493_fc1r3_t2a10_t3a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.10, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        ("s494_fc1r3_t2a20_t3a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.20, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ FC1 baseline 3-seed + kNN (now all 3 seeds complete) ═══
        ("s496_fc1_3seed",
         ["fc1_roberta_t3nc3_kfold", "fc1_roberta_t3nc3_s1_kfold", "fc1_roberta_t3nc3_s2_kfold"],
         None, None),
        ("s497_fc1_3seed_knn",
         ["fc1_roberta_t3nc3_kfold", "fc1_roberta_t3nc3_s1_kfold", "fc1_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ FC1R(3) + FC1(3) 6-way + kNN best config ═══
        ("s498_fc1r3_fc13_6way_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1_roberta_t3nc3_kfold", "fc1_roberta_t3nc3_s1_kfold", "fc1_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ FC1R(3) + FC1S(3) + FC1(3) 9-way + kNN ═══
        ("s499_fc1r3_fc1s3_fc13_9way_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold",
          "fc1_roberta_t3nc3_kfold", "fc1_roberta_t3nc3_s1_kfold", "fc1_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ FC1M 3-seed (all complete now) ═══
        ("s5a0_fc1m_3seed",
         ["fc1m_roberta_t3nc3_kfold", "fc1m_roberta_t3nc3_s1_kfold", "fc1m_roberta_t3nc3_s2_kfold"],
         None, None),
        ("s5a1_fc1m_3seed_knn",
         ["fc1m_roberta_t3nc3_kfold", "fc1m_roberta_t3nc3_s1_kfold", "fc1m_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ Full 12-way: FC1R(3)+FC1S(3)+FC1M(3)+FC1(3) ═══
        ("s5a2_all12_noknn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold",
          "fc1m_roberta_t3nc3_kfold", "fc1m_roberta_t3nc3_s1_kfold", "fc1m_roberta_t3nc3_s2_kfold",
          "fc1_roberta_t3nc3_kfold", "fc1_roberta_t3nc3_s1_kfold", "fc1_roberta_t3nc3_s2_kfold"],
         None, None),
        ("s5a3_all12_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold",
          "fc1m_roberta_t3nc3_kfold", "fc1m_roberta_t3nc3_s1_kfold", "fc1m_roberta_t3nc3_s2_kfold",
          "fc1_roberta_t3nc3_kfold", "fc1_roberta_t3nc3_s1_kfold", "fc1_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ FC1R(3)+FC1M(3) 6-way + kNN ═══
        ("s5a4_fc1r3_fc1m3_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1m_roberta_t3nc3_kfold", "fc1m_roberta_t3nc3_s1_kfold", "fc1m_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ FC1R(3)+FC1S(3)+FC1M(3) 9-way + kNN ═══
        ("s5a5_fc1r3_fc1s3_fc1m3_9way_knn",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold",
          "fc1s_roberta_t3nc3_kfold", "fc1s_roberta_t3nc3_s1_kfold", "fc1s_roberta_t3nc3_s2_kfold",
          "fc1m_roberta_t3nc3_kfold", "fc1m_roberta_t3nc3_s1_kfold", "fc1m_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ Equal weight 3-seed (no s1 dominance) ═══
        ("s495_fc1r3_equal_a40_b01",
         ["fc1r_roberta_t3nc3_kfold", "fc1r_roberta_t3nc3_s1_kfold", "fc1r_roberta_t3nc3_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),

        # ═══ Retrain: full_train_data (2000) → predict test_2000 ═══
        # 12-way no kNN
        ("rt_all12_noknn",
         ["rt_fc1r_s0_kfold", "rt_fc1r_s1_kfold", "rt_fc1r_s2_kfold",
          "rt_fc1s_s0_kfold", "rt_fc1s_s1_kfold", "rt_fc1s_s2_kfold",
          "rt_fc1m_s0_kfold", "rt_fc1m_s1_kfold", "rt_fc1m_s2_kfold",
          "rt_fc1_s0_kfold", "rt_fc1_s1_kfold", "rt_fc1_s2_kfold"],
         None, None),
        # 12-way + kNN (best config: α=0.40, bias=0.1)
        ("rt_all12_knn",
         ["rt_fc1r_s0_kfold", "rt_fc1r_s1_kfold", "rt_fc1r_s2_kfold",
          "rt_fc1s_s0_kfold", "rt_fc1s_s1_kfold", "rt_fc1s_s2_kfold",
          "rt_fc1m_s0_kfold", "rt_fc1m_s1_kfold", "rt_fc1m_s2_kfold",
          "rt_fc1_s0_kfold", "rt_fc1_s1_kfold", "rt_fc1_s2_kfold"],
         None,
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
        # FC1R 3-seed + kNN (runner-up config)
        ("rt_fc1r3_knn",
         ["rt_fc1r_s0_kfold", "rt_fc1r_s1_kfold", "rt_fc1r_s2_kfold"],
         [1.0, 2.0, 1.0],
         {"knn_encoder": "qwen3_emb", "k": 5,
          "alpha": {"t1": 0.0, "t2": 0.0, "t3": 0.40, "t4": 0.0}, "t3_nc_logit_bias": 0.1}),
    ]

    # ── Determine which model keys are actually needed ────────────────────────
    if getattr(cfg, "skip_existing", False):
        needed_keys: set[str] = set()
        for combo in COMBOS:
            if (out_dir / f"{combo[0]}.csv").exists():
                continue
            needed_keys.update(combo[1])
            if len(combo) >= 4 and combo[3] is not None:
                knn_enc = combo[3].get("knn_encoder")
                if knn_enc:
                    needed_keys.add(knn_enc)
        ckpts_to_load   = {k: v for k, v in CKPTS.items()      if k in needed_keys}
        kfolds_to_load  = {k: v for k, v in KFOLD_DIRS.items() if k in needed_keys}
        n_skipped = len(CKPTS) + len(KFOLD_DIRS) - len(ckpts_to_load) - len(kfolds_to_load)
        if n_skipped:
            print(f"  (skip_existing: skipping {n_skipped} unused model(s))")
    else:
        ckpts_to_load  = CKPTS
        kfolds_to_load = KFOLD_DIRS

    # ── Load predictions ──────────────────────────────────────────────────────
    print("Loading predictions from checkpoints...")
    labels_map: dict[str, dict] = {}
    probs_map: dict[str, dict] = {}
    for name, path in ckpts_to_load.items():
        if not Path(path).exists():
            print(f"  [SKIP] {name}: checkpoint not found")
            continue
        print(f"  {name} ...", end="", flush=True)
        result = _predict_checkpoint(path, test_df)
        if result is not None:
            labels_map[name], probs_map[name] = result
            print(" OK")
        else:
            print(" FAILED")

    # Load k-fold runs: soft-vote all folds in each kfold dir into one entry
    # tta_n_map[name] is set below when a COMBO for this key requests TTA
    tta_n_map: dict[str, int] = {}
    for combo in COMBOS:
        if getattr(cfg, "skip_existing", False) and (out_dir / f"{combo[0]}.csv").exists():
            continue
        if len(combo) >= 4 and combo[3] is not None:
            _tta = combo[3].get("tta_n", 0)
            if _tta > 1:
                for k in combo[1]:
                    tta_n_map[k] = max(tta_n_map.get(k, 0), _tta)

    for name, kfold_dir in kfolds_to_load.items():
        fold_probs = []
        fold_idx = 1
        _tta = tta_n_map.get(name, 0)
        while True:
            fold_ckpt = Path(kfold_dir) / f"fold{fold_idx}" / "best.pt"
            if not fold_ckpt.exists():
                break
            print(f"  {name}/fold{fold_idx} ...", end="", flush=True)
            result = _predict_checkpoint(str(fold_ckpt), test_df, tta_n=_tta)
            if result is not None:
                fold_probs.append(result[1])
                print(f" OK{' (TTA×'+str(_tta)+')' if _tta>1 else ''}")
            else:
                print(" FAILED")
            fold_idx += 1
        if fold_probs:
            avg_probs: dict[str, list[list[float]]] = {}
            for t in LABELS:
                n = len(fold_probs[0][t])
                avg_probs[t] = [
                    [sum(fp[t][i][c] for fp in fold_probs) / len(fold_probs)
                     for c in range(NUM_LABELS[t])]
                    for i in range(n)
                ]
            probs_map[name] = avg_probs
            labels_map[name] = {t: [IDX2LABEL[t][int(np.argmax(p))]
                                     for p in avg_probs[t]] for t in LABELS}
            print(f"  {name}: {len(fold_probs)} folds loaded")

    # COMBOS entries:
    #   2-tuple: (name, keys)
    #   3-tuple: (name, keys, weights)
    #   4-tuple: (name, keys, weights, knn_cfg)   ← kNN-LDL
    #   5-tuple: (name, keys, weights, None, stacking_cfg)  ← Stacking
    for combo in COMBOS:
        fname, keys    = combo[0], combo[1]
        weights        = combo[2] if len(combo) >= 3 else None
        knn_cfg        = combo[3] if len(combo) >= 4 else None
        stacking_cfg   = combo[4] if len(combo) == 5 else None

        if getattr(cfg, "skip_existing", False) and (out_dir / f"{fname}.csv").exists():
            print(f"  [SKIP] {fname}: already exists")
            continue

        available = [k for k in keys if k in probs_map]
        avail_weights = None if weights is None else [
            weights[i] for i, k in enumerate(keys) if k in probs_map
        ]
        if not available:
            print(f"  [SKIP] {fname}: no predictions available")
            continue

        # Build base model probs (weighted average across ensemble keys)
        if len(available) == 1:
            base_probs = probs_map[available[0]]
        else:
            ws = avail_weights or [1.0] * len(available)
            total_w = sum(ws)
            n = len(probs_map[available[0]]["t1"])
            base_probs = {}
            for task in LABELS:
                nc = NUM_LABELS[task]
                base_probs[task] = [
                    [sum(ws[j] * probs_map[available[j]][task][i][c]
                         for j in range(len(available))) / total_w
                     for c in range(nc)]
                    for i in range(n)
                ]

        # Apply T3 NC logit bias if specified (global + optional per-company)
        t3_nc_bias = (knn_cfg or {}).get("t3_nc_logit_bias", 0.0)
        t3_co_beta = (knn_cfg or {}).get("t3_company_nc_beta", 0.0)
        if t3_nc_bias != 0.0 or t3_co_beta != 0.0:
            import math as _math
            nc_idx = LABEL2IDX["t3"]["Not Clear"]
            # Per-company NC rates (computed from train once)
            _co_nc_rate: dict[str, float] = {}
            _global_nc_rate = 0.0
            if t3_co_beta != 0.0:
                # evidence_quality is the raw T3 label column in train data
                _t3_col = ("t3" if "t3" in knn_train_df.columns
                           else "evidence_quality")
                _nc_counts = knn_train_df[_t3_col].value_counts()
                _total = len(knn_train_df)
                _global_nc_rate = (_nc_counts.get("Not Clear", 0) / _total) if _total else 0.1
                for _co, _grp in knn_train_df.groupby("company"):
                    _co_nc_rate[_co] = (_grp[_t3_col] == "Not Clear").mean()
            _test_companies = test_df["company"].tolist() if t3_co_beta != 0.0 else []
            new_t3 = []
            for i, p in enumerate(base_probs["t3"]):
                p2 = list(p)
                bias = t3_nc_bias
                if t3_co_beta != 0.0:
                    co = _test_companies[i]
                    co_rate = _co_nc_rate.get(co, _global_nc_rate)
                    _eps = 1e-3
                    bias += t3_co_beta * _math.log(
                        (co_rate + _eps) / (_global_nc_rate + _eps)
                    )
                p2[nc_idx] *= _math.exp(bias)
                total = sum(p2)
                new_t3.append([x / total for x in p2])
            base_probs = dict(base_probs)
            base_probs["t3"] = new_t3

        if stacking_cfg is not None and stacking_cfg.get("stacking"):
            # Stacking path: LR meta-learner overrides T2 + T3
            try:
                meta = _get_stacking_meta()
            except Exception as e:
                print(f"  [SKIP] {fname}: stacking failed — {e}")
                continue
            base_preds = {t: [IDX2LABEL[t][int(np.argmax(p))] for p in base_probs[t]]
                          for t in LABELS}
            # Provide per-model probs in the same order as OOF (rob first, lert second)
            model_probs_list = [probs_map[k] for k in available if k in probs_map]
            preds = stacking_predict(meta, model_probs_list, base_preds)
            save(fname, preds)
        elif knn_cfg is not None and "knn_encoder" in knn_cfg:
            # kNN-LDL path
            enc_key = knn_cfg["knn_encoder"]
            try:
                if knn_cfg.get("use_oof_embs", False):
                    tr_embs, te_embs = _get_knn_embs_oof(enc_key)
                else:
                    tr_embs, te_embs = _get_knn_embs(enc_key)
            except (FileNotFoundError, ValueError) as e:
                print(f"  [SKIP] {fname}: {e}")
                continue
            knn_probs = compute_knn_ldl_probs(
                tr_embs, te_embs, knn_train_df, k=knn_cfg["k"],
                sim_temp=knn_cfg.get("sim_temp", 1.0),
                test_df=test_df,
                company_knn=knn_cfg.get("company_knn", False))
            t3_tau = knn_cfg.get("t3_nc_threshold", 0.0)
            dyn_a = knn_cfg.get("dynamic_alpha", False)
            t3_la_tau = knn_cfg.get("t3_la_tau", 0.0)
            preds = knn_fuse_probs(base_probs, knn_probs, alpha=knn_cfg["alpha"],
                                   t3_nc_threshold=t3_tau, dynamic_alpha=dyn_a,
                                   t3_la_tau=t3_la_tau, t3_class_freq=T3_CLASS_FREQ)
            save(fname, preds, remap_misleading=knn_cfg.get("remap_misleading", False))
        else:
            # Standard path: argmax on base_probs
            preds = {t: [IDX2LABEL[t][int(np.argmax(p))] for p in base_probs[t]]
                     for t in LABELS}
            save(fname, preds)

    print(f"\nDone. {len(list(out_dir.glob('*.csv')))} files in {out_dir}/")


def main() -> None:
    cfg = parse_args()
    if cfg.mode == "test":
        _run_tests()
    elif cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "predict":
        if not cfg.checkpoint:
            raise ValueError("--checkpoint required")
        predict(cfg)
    elif cfg.mode == "geneval":
        run_generative(cfg)
    elif cfg.mode == "run_all":
        run_all(cfg)
    elif cfg.mode == "kfold":
        if cfg.kfold < 2:
            raise ValueError("--kfold must be ≥ 2")
        train_kfold(cfg)
    elif cfg.mode == "gen_submissions":
        gen_submissions(cfg)
    elif cfg.mode == "run_v4":
        run_v4(cfg)
    elif cfg.mode == "gen_llm_augdata":
        gen_llm_augdata(cfg)
    elif cfg.mode == "gen_pseudo_labels":
        gen_pseudo_labels(cfg, confidence_thr=0.80)
    elif cfg.mode == "search_thresholds":
        train_df, _ = load_dataframes(cfg.data_dir, use_augmented=False)
        kfold_dir = "runs/A1_roberta_dc_kfold_llmaug"
        print(f"\nSearching OOF thresholds on: {kfold_dir}")
        thrs = search_oof_thresholds(kfold_dir, train_df, n_splits=5, seed=cfg.seed,
                                     tasks=("t3",), steps=20)
        import json
        print("\n=== Best thresholds ===")
        print(json.dumps(thrs, indent=2))
    elif cfg.mode == "tune_t3_nc_bias":
        train_df, _ = load_dataframes(cfg.data_dir, use_augmented=False)
        kfold_dir = getattr(cfg, "kfold_dir", None) or "runs/A1_roberta_dc_kfold_t3nc3"
        print(f"\nTuning T3 NC logit bias on OOF: {kfold_dir}")
        best_delta = tune_t3_nc_bias(kfold_dir, train_df, n_splits=5, seed=cfg.seed)
        print(f"\nAdd to COMBO knn_cfg: \"t3_nc_logit_bias\": {best_delta}")
    elif cfg.mode == "tapt":
        run_tapt(cfg)


if __name__ == "__main__":
    main()
