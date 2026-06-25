# AI CUP 2026 — VeriPromiseESG（ESG 永續承諾驗證競賽）

**Team:** TEAM_10505 (學生組)
**Private LB:** **0.6432854 — Rank 12**　|　Public LB: 0.6249259 — Rank 15

This repository contains the reproducible code for our submission to the AI CUP 2026
VeriPromiseESG competition (the Chinese counterpart of SemEval-2025 Task 6, *PromiseEval*).
The task is to predict, for each ESG-report passage, four dependent sub-tasks:

| Sub-task | Classes | Weight |
|---|---|---|
| **T1** promise_status | Yes / No | 0.20 |
| **T2** evidence_status | Yes / No / N/A | 0.30 |
| **T3** evidence_quality | Clear / Not Clear / Misleading / N/A | 0.35 |
| **T4** verification_timeline | already / within_2y / 2–5y / >5y / N/A | 0.15 |

Metric: weighted macro-F1 `0.20·T1 + 0.30·T2 + 0.35·T3 + 0.15·T4` (macro over present classes).

## Method (one-paragraph)

A Chinese **RoBERTa-large** encoder with a **cascade multi-task head** (`CascadeHeadV2`:
T1→T2, [T1,T2]→T3, T1→T4), 3 seeds × 5 folds. T3 (the weakest, highest-weight task) is
strengthened with **kNN label-distribution learning** (Qwen3-Embedding-0.6B) and an
independent **clarity head**. Two contributions drove the gains: (1) a **per-task
`max_length` harvest** (T1/T3 @384, T2/T4 @512, chosen via the per-subtask LB read-out),
and (2) **joint/structured decoding** — picking the valid joint label configuration that
maximizes the joint log-prob, letting downstream confidence correct the upstream gate.
All models are open-source and run **locally (no API at inference)**. Full write-up in
[`report/REPORT.md`](report/REPORT.md).

## Repository structure

```
esg_main.py            # model, CascadeHeadV2, per-task losses, k-fold training, inference
gen_rt_more.py         # @384 RT models → test probs + kNN-LDL fusion
gen_rt512.py           # @512 RT models → test probs
clarity_head.py        # train the independent T3 clarity head
clarity_aidea.py       # clarity head → test clarity probs (overlay)
gen_joint_aidea.py     # FINAL: joint/structured decode → submission CSV
eval_single_run.py     # local valid (399) evaluation + prob caching
overnight_retrain.sh   # train RT_FC1/FC1R/FC1S @384 on retrain_data
train_rt_512.sh        # train RT_*512 @512 on retrain_data
report/                # competition report (.docx) + build_report.py + REPORT.md
docs/                  # CHAMPION.md, FINAL_PLAN.md, DAY1_AUDIT.md, SUBMISSIONS.md (journey logs)
experiments/           # all exploratory / ablated / rejected approaches (the journey)
official_sub/aidea_joint_w2.0.csv   # the final submitted file (reference)
```

> **Data is NOT included** (competition licensing). Place the official files under
> `final_data/` (train_data.csv, valid_data.csv, valid_solution_data.csv) and
> `retrain_data/` (train_data.csv, test_data.csv) — both are `.gitignore`d.
> `runs/` (weights) and `agent_cache/` (caches) are also ignored.

## Environment

- Ubuntu Linux, **Python 3.13**, single **NVIDIA RTX 5090 (32 GB)**.
- `pip install -r requirements.txt` (install PyTorch matching your CUDA — tested cu128).

## Reproduce the final submission

```bash
# 0) place competition data in final_data/ and retrain_data/ (see note above)

# 1) train the RoBERTa cascade ensemble on retrain_data (2000)
bash overnight_retrain.sh      # RT_FC1R etc. @ max_length 384
bash train_rt_512.sh           # RT_FC1R512 etc. @ max_length 512

# 2) train the T3 clarity head and cache its test probabilities
python clarity_head.py
python clarity_aidea.py        # → agent_cache/clarity_test_probs.npz

# 3) cache per-task test probabilities (+ kNN-LDL via Qwen3-Embedding-0.6B)
python gen_rt_more.py          # @384 → agent_cache/rt_test_probs/
python gen_rt512.py            # @512 → agent_cache/rt512_test_probs/

# 4) FINAL submission via joint/structured decoding (wgate=2.0)
python gen_joint_aidea.py 2.0  # → official_sub/aidea_joint_w2.0.csv  (Private 0.6432854)
```

## Generative-AI & external-resource disclosure

Per the competition rules: development was assisted by **Anthropic Claude (Claude Code,
model Claude Opus 4.x → 4.8)** for code implementation, experiment design, literature
research, and analysis; **all strategy, methodology discipline, and final decisions were
made by the team**. The submission pipeline uses only open-source models
(`hfl/chinese-roberta-wwm-ext-large`, `Qwen/Qwen3-Embedding-0.6B`) run locally — **no
external API at inference**. `experiments/` documents methods that were tried and
*rejected* (company prior, 12-way ensemble, cross-backbone, LLM judges, SupCon, ordinal,
soft-F1, SSL/pseudo-label …) — see `docs/FINAL_PLAN.md`. Details in `report/REPORT.md` §8.
