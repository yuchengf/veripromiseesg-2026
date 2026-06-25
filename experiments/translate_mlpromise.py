"""#3 step A: translate ML-Promise EN/FR/JA Not-Clear + Misleading rows into
Traditional Chinese (Qwen3-14B), to augment the clarity-head training with
near-distribution Chinese examples of the two hard classes (removes the
language gap that broke cross-lingual TRANSFER).

Resume-safe: appends to agent_cache/mlpromise_zh.jsonl; skips done ids.
Final: external_data/clarity_aug_zh.csv (same schema as clarity training).
"""
import os, json, sys
os.chdir("/home/yucheng/Desktop/ESG")
import pandas as pd, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-14B"
SRC = "external_data/mlpromise_enfrja.csv"
JSONL = Path("agent_cache/mlpromise_zh.jsonl")
OUT = "external_data/clarity_aug_zh.csv"
MAX_IN_CHARS = 1500
MAX_NEW = 1024

df = pd.read_csv(SRC, keep_default_na=False)
sub = df[df["evidence_quality"].isin(["Not Clear", "Misleading"])].reset_index(drop=True)
print(f"[translate] {len(sub)} rows (NotClear+Misleading)", flush=True)

done = {}
if JSONL.exists():
    for line in open(JSONL):
        try:
            r = json.loads(line); done[r["id"]] = r["zh"]
        except Exception:
            pass
print(f"[translate] {len(done)} already done", flush=True)

todo = sub[~sub["id"].isin(done.keys())]
if len(todo):
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()

    def translate(text):
        prompt = ("將以下 ESG 永續報告文字翻譯成**繁體中文**，僅輸出翻譯結果、"
                  "不要任何解釋或前後綴，保持原意與專業語氣：\n\n" + text[:MAX_IN_CHARS])
        msgs = [{"role": "user", "content": prompt}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        enc = tok(chat, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    with open(JSONL, "a") as fout:
        for n, (_, row) in enumerate(todo.iterrows(), 1):
            zh = translate(str(row["data"]))
            done[row["id"]] = zh
            fout.write(json.dumps({"id": row["id"], "zh": zh}, ensure_ascii=False) + "\n")
            fout.flush()
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} id={row['id']} len={len(zh)} :: {zh[:50]}", flush=True)
    del model; torch.cuda.empty_cache()

# build augmentation CSV
rows = []
for _, row in sub.iterrows():
    zh = done.get(row["id"], "").strip()
    if len(zh) < 10:
        continue
    rows.append({"id": f"ZH_{row['id']}", "data": zh,
                 "promise_status": row["promise_status"],
                 "verification_timeline": row["verification_timeline"],
                 "evidence_status": row["evidence_status"],
                 "evidence_quality": row["evidence_quality"]})
aug = pd.DataFrame(rows)
aug.to_csv(OUT, index=False)
print(f"\n[translate] wrote {OUT}: {len(aug)} rows  "
      f"T3={aug['evidence_quality'].value_counts().to_dict()}", flush=True)
