"""LLM gate-judge: Qwen3-14B judges T1 (is there a promise?) and T2 (is there concrete
evidence?) — the GATE tasks the joint decode wins on. T1/T2 are more content-derivable
than T3 quality (where the LLM failed). A decorrelated external gate signal could feed
the joint decode for further gate correction. Validated on valid before any AIdea probe.
Usage: python llm_gate.py valid|test [limit]"""
import os, sys, re, json
os.chdir("/home/yucheng/Desktop/ESG")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "valid"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(42)
MODEL = "Qwen/Qwen3-14B"; MAX_IN = 900; MAX_NEW = 220
tr = pd.read_csv("final_data/train_data.csv", keep_default_na=False)
def pick(col, val, n):
    sub = tr[tr[col] == val].copy(); sub["L"] = sub["data"].str.len()
    return sub[(sub.L > 60) & (sub.L < 240)].sort_values("L").head(n)
# few-shot: promise Yes/No and evidence Yes/No
fs = pd.concat([pick("promise_status", "Yes", 2), pick("promise_status", "No", 2)]).drop_duplicates("id").sample(frac=1, random_state=7)
fewshot = "\n".join(
    f"範例：promise={r.promise_status}, evidence={r.evidence_status} ← {str(r['data'])[:200]}"
    for _, r in fs.iterrows())
PROMPT = """你是 ESG 永續報告書分析專家。讀以下企業永續報告書文字,判斷兩件事:
1. promise(承諾):這段是否包含一項永續「承諾/目標/計畫意圖」?(Yes/No)
2. evidence(佐證):若有承諾,文中是否提供「具體佐證」(行動/數據/成果/查證)?(Yes/No;若無承諾則 evidence=No)

{fewshot}

待判斷:
{text}

只輸出一行,格式:`PROMISE: Yes/No | EVIDENCE: Yes/No | CONF: 0-10`"""
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
def judge(text):
    msgs = [{"role": "user", "content": PROMPT.format(fewshot=fewshot, text=str(text)[:MAX_IN])}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(chat, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True); low = txt.lower()
    pm = re.search(r'promise:\s*(yes|no)', low); ev = re.search(r'evidence:\s*(yes|no)', low); cf = re.search(r'conf:\s*(\d+)', low)
    return (pm.group(1) if pm else None), (ev.group(1) if ev else None), (min(int(cf.group(1)), 10) if cf else 5), txt[:100]
df = pd.read_csv(f"final_data/valid_data.csv" if TARGET == "valid" else "retrain_data/test_data.csv", keep_default_na=False)
if LIMIT: df = df.head(LIMIT)
print(f"[llm_gate:{TARGET}] {len(df)} rows; fewshot={len(fs)}", flush=True)
cp = f"agent_cache/llm_gate_{TARGET}.json"; cache = json.load(open(cp)) if os.path.exists(cp) else {}
for i, r in df.iterrows():
    rid = str(r["id"])
    if rid in cache: continue
    pm, ev, cf, raw = judge(r["data"]); cache[rid] = {"promise": pm, "evidence": ev, "conf": cf, "raw": raw}
    if i % 50 == 0: json.dump(cache, open(cp, "w"), ensure_ascii=False); print(f"  [{i}/{len(df)}] id{rid}: P={pm} E={ev} ({cf})", flush=True)
json.dump(cache, open(cp, "w"), ensure_ascii=False)
yn = sum(1 for v in cache.values() if v["promise"] == "yes")
print(f"[done] {len(cache)} → {cp}; LLM promise=Yes {yn} No {len(cache)-yn}", flush=True)
