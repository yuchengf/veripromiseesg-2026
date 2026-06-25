"""LLM re-classification of T3 evidence_quality (Clear vs Not Clear) with Qwen3-14B.
External, decorrelated signal that does NOT overfit our train distribution -> may
transfer to AIdea better than the fine-tuned model + clarity head, and targets the
real bottleneck (Not Clear F1=0.326; model under-predicts Not Clear).

Usage: python llm_t3.py valid    # validate on valid first (~270 non-N/A rows)
       python llm_t3.py test     # full retrain test (2000)
Caches agent_cache/llm_t3_<target>.json = {id: {"label","conf","raw"}}.
Misleading is NOT judged (unpredictable) — only the Clear<->Not Clear boundary."""
import os, sys, re, json
os.chdir("/home/yucheng/Desktop/ESG")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "valid"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 = all
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(42)
MODEL = "Qwen/Qwen3-14B"; MAX_IN = 900; MAX_NEW = 200

tr = pd.read_csv("final_data/train_data.csv", keep_default_na=False)
# few-shot: short, representative exemplars of each class (deterministic)
def pick(label, n):
    sub = tr[tr.evidence_quality == label].copy()
    sub["L"] = sub["data"].str.len()
    return sub[(sub.L > 60) & (sub.L < 240)].sort_values("L").head(n)
fs = pd.concat([pick("Not Clear", 3), pick("Clear", 3)]).sample(frac=1, random_state=7)
fewshot = "\n".join(f"範例【{r.evidence_quality}】：{str(r['data'])[:220]}" for _, r in fs.iterrows())

PROMPT = """你是 ESG 永續報告書分析專家。以下是一段企業永續報告書文字，描述某項永續承諾及其佐證。請判斷其「佐證證據品質」：

- **Clear（明確）**：證據具體、可查證——有明確的方案/計畫/行動名稱、量化數據（數字、百分比、目標值、基準年）、實際成果、第三方查證、具體案例。
- **Not Clear（不明確）**：只有空泛、願景式的宣示——如「致力於」「持續努力」「核心原則」「推動」「期望」等抽象語言，缺乏具體行動、數據或可查證佐證。

判斷重點：具體案例或數據＝Clear；空泛原則或願景宣示＝Not Clear。

{fewshot}

待判斷段落：
{text}

只輸出一行，格式：`T3: Clear 信心數字` 或 `T3: Not Clear 信心數字`（信心 0-10）。例如 `T3: Not Clear 8`。"""

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True).eval()

def judge(text):
    msgs = [{"role": "user", "content": PROMPT.format(fewshot=fewshot, text=str(text)[:MAX_IN])}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(chat, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    low = txt.lower()
    # "Not Clear" must be checked before "Clear" (substring)
    m = re.search(r't3:\s*(not\s*clear|clear)', low)
    if m:
        label = "Not Clear" if "not" in m.group(1) else "Clear"
    elif "not clear" in low:
        label = "Not Clear"
    elif "clear" in low:
        label = "Clear"
    else:
        return None, 5, txt[:120]
    cm = re.search(r'(?:not\s*clear|clear)\D*(\d+)', low)
    conf = min(int(cm.group(1)), 10) if cm else 5
    return label, conf, txt[:120]

# load target rows (id, data)
if TARGET == "valid":
    df = pd.read_csv("final_data/valid_data.csv", keep_default_na=False) if os.path.exists("final_data/valid_data.csv") \
         else pd.read_csv("final_data/valid_solution_data.csv", keep_default_na=False)
else:
    df = pd.read_csv("retrain_data/test_data.csv", keep_default_na=False)
if LIMIT:
    df = df.head(LIMIT)
print(f"[llm_t3:{TARGET}] {len(df)} rows; fewshot={len(fs)} exemplars", flush=True)

cache_p = f"agent_cache/llm_t3_{TARGET}.json"
cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
for i, r in df.iterrows():
    rid = str(r["id"])
    if rid in cache:
        continue
    label, conf, raw = judge(r["data"])
    cache[rid] = {"label": label, "conf": conf, "raw": raw}
    if i % 50 == 0:
        json.dump(cache, open(cache_p, "w"), ensure_ascii=False)
        print(f"  [{i}/{len(df)}] id{rid}: {label} ({conf})", flush=True)
json.dump(cache, open(cache_p, "w"), ensure_ascii=False)
nc = sum(1 for v in cache.values() if v["label"] == "Not Clear")
print(f"[done] cached {len(cache)} → {cache_p}; LLM 判 Not Clear={nc} Clear={len(cache)-nc}", flush=True)
