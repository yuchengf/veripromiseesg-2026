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

## Usage

All scripts `os.chdir` to their own location on startup, so you can run them from
anywhere after placing the data — no path editing needed.

### 1. Training entry point — `esg_main.py`

Every model is trained through one CLI. The flags below are exactly the recipe used
for the final ensemble:

```bash
python esg_main.py \
  --mode kfold --approach A1 \
  --backbone hfl/chinese-roberta-wwm-ext-large \
  --data_dir retrain_data --kfold 5 \
  --per_task_loss --augment_rare --deep_cascade --t3_nc_weight 3.0 \
  --max_length 384 \
  --rdrop_alpha 0.5 --swa_start_epoch 7 \
  --seed 42 --run_dir runs/RT_FC1R_s0 --batch_size 8
```

Key arguments:

| Flag | Meaning |
|---|---|
| `--mode` | `kfold` (K-fold train + cache OOF probs — used here) · `train` (single split) · `predict --checkpoint <pt>` (inference only) · `test` (self-tests) · `geneval` (LLM) |
| `--approach A1` | BERT encoder → CLS → **CascadeHeadV2** (T1→T2, [T1+T2]→T3, [T1]→T4). Our architecture. (`A`=4 independent heads, `B/B_lora`=causal LLM, `C`=frozen sentence-embedding + MLP.) |
| `--backbone` | HF model id (we use `hfl/chinese-roberta-wwm-ext-large`). |
| `--data_dir` | `final_data` (1601 train, for local-valid model selection) or `retrain_data` (2000 train, for the final submission). |
| `--kfold 5` | 5-fold CV; writes `runs/<run_dir>/fold{1..5}/best.pt` + cached OOF predictions. |
| `--max_length` | `384` or `512` (per-task harvest — T1/T3 favor 384, T2/T4 favor 512; see report §伍). |
| `--per_task_loss` | T1/T2 = CE, T3 = Distribution-Balanced (+`--t3_nc_weight 3.0`), T4 = ordinal. |
| `--deep_cascade` `--augment_rare` | deeper cascade coupling · rare-class oversampling. |
| `--seed` `--run_dir` `--batch_size` | reproducibility · output dir · batch size (8 for R-Drop, 16 otherwise). |

**Recipe variants** (the only difference is the trailing flags):

| Recipe | Extra flags | batch |
|---|---|---|
| `FC1` (base) | *(none)* | 16 |
| **`FC1R`** ← used in final | `--rdrop_alpha 0.5 --swa_start_epoch 7` | 8 |
| `FC1S` | `--sharp_recl_weight 0.10` | 16 |
| `FC1M` | `--mr2_weight 0.05` | 16 |

### 2. Train the whole ensemble

Each script trains its recipe set across **3 seeds (42/123/456) × 5 folds**:

```bash
bash overnight_retrain.sh   # RT_FC1 / RT_FC1R / RT_FC1S @384  → runs/RT_*
bash train_rt_512.sh        # RT_*512 @512                     → runs/RT_*512
```

The final submission only needs `RT_FC1R*` (gives T1/T3) and `RT_FC1R512*` (gives T2/T4).

### 3. Quick start — train one model and score it locally

```bash
python esg_main.py --mode kfold --approach A1 --backbone hfl/chinese-roberta-wwm-ext-large \
  --data_dir final_data --kfold 5 --per_task_loss --augment_rare --deep_cascade \
  --t3_nc_weight 3.0 --max_length 384 --rdrop_alpha 0.5 --swa_start_epoch 7 \
  --seed 42 --run_dir runs/FC1R_s0 --batch_size 8

python eval_single_run.py FC1R_s0   # weighted macro-F1 on valid 399 (official-aligned) + caches probs
```

### 4. Inference — cache per-task test probabilities

Reads the trained checkpoints, predicts the 2000-row test set, fuses **kNN-LDL**
(`Qwen/Qwen3-Embedding-0.6B`, k=5) and caches everything:

```bash
python gen_rt_more.py   # @384 models → agent_cache/rt_test_probs/   (+ kNN, + combo CSVs)
python gen_rt512.py     # @512 models → agent_cache/rt512_test_probs/
```

### 5. T3 clarity head (Clear / Not Clear / Misleading overlay)

```bash
python clarity_head.py   # train on non-N/A rows + report Not-Clear recall on valid
python clarity_aidea.py  # predict the test set → agent_cache/clarity_test_probs.npz
```

### 6. Final submission — joint / structured decoding

Combines the per-task sources (T1/T3 @384, T2/T4 @512) + kNN + clarity, then **jointly
decodes** the cascade so a confident downstream N/A can correct the upstream gate:

```bash
python gen_joint_aidea.py 2.0   # wgate=2.0 → official_sub/aidea_joint_w2.0.csv  (Private LB 0.6432854)
```

### End-to-end

Run **2 → 4 → 5 → 6** in order (step 3 is per-model diagnostics only).
`gen_joint_aidea.py 2.0` reproduces the exact submitted file.

### Output layout (all git-ignored)

- `runs/<name>/fold{1..5}/best.pt` — model checkpoints.
- `agent_cache/{valid_probs,rt_test_probs,rt512_test_probs}/*.npz` — cached probabilities.
- `agent_cache/clarity_test_probs.npz`, `agent_cache/qwen3_embs_retrain.npz` — clarity probs, kNN embeddings.
- `official_sub/*.csv` — submission files (only the final `aidea_joint_w2.0.csv` is tracked).

## Generative-AI & external-resource disclosure

Per the competition rules: development was assisted by **Anthropic Claude (Claude Code,
model Claude Opus 4.x → 4.8)** for code implementation, experiment design, literature
research, and analysis; **all strategy, methodology discipline, and final decisions were
made by the team**. The submission pipeline uses only open-source models
(`hfl/chinese-roberta-wwm-ext-large`, `Qwen/Qwen3-Embedding-0.6B`) run locally — **no
external API at inference**. `experiments/` documents methods that were tried and
*rejected* (company prior, 12-way ensemble, cross-backbone, LLM judges, SupCon, ordinal,
soft-F1, SSL/pseudo-label …) — see `docs/FINAL_PLAN.md`. Details in `report/REPORT.md` §8.
