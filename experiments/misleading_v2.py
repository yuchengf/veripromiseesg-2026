"""Improved Misleading detector v2 — greenwashing-specific Chain-of-Thought +
self-consistency, on Qwen3-14B (local; no frontier API). Tests on a labeled
panel whether it discriminates the 2 true Misleading from Not Clear / Clear.
If it ranks true Misleading above non-Misleading -> signal -> worth a full overlay.

VRAM (RTX 5090 32GB): Qwen3-14B bf16 ~28GB; batch=1 sequential; input capped 800
chars; max_new_tokens 320; self-consistency 3 samples. Fits with margin.
Usage: python misleading_v2.py [--smoke]
"""
import os, sys, re, json
os.chdir("/home/yucheng/Desktop/ESG")
SMOKE = "--smoke" in sys.argv
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(42); np.random.seed(42)
MODEL = "Qwen/Qwen3-14B"; N_SAMPLES = 1 if SMOKE else 3; MAX_IN = 800; MAX_NEW = 700

tr = pd.read_csv("final_data/full_train_data.csv", keep_default_na=False)
# panel: focus on the hard content classes (T2=Yes rows): 2 Misleading + NotClear + Clear
mis = tr[tr["evidence_quality"]=="Misleading"]
nc  = tr[tr["evidence_quality"]=="Not Clear"].sample(20, random_state=42)
cl  = tr[tr["evidence_quality"]=="Clear"].sample(10, random_state=42)
panel = pd.concat([mis, nc, cl]).reset_index(drop=True)
if SMOKE: panel = pd.concat([mis, nc.head(1), cl.head(1)]).reset_index(drop=True)
print(f"[panel] {len(panel)} rows: Misleading={len(mis)} NotClear={len(nc) if not SMOKE else 1} Clear={len(cl) if not SMOKE else 1}", flush=True)

PROMPT = """你是 ESG 永續報告的漂綠(greenwashing)稽核專家。判斷以下 ESG 報告段落的「證據品質」是否為 **Misleading(誤導)**。

官方定義:Misleading = 證據看起來蓄意欺騙,或誤述承諾的實際履行程度。相對地,「Not Clear」只是證據模糊/不足(非蓄意誤導),「Clear」是證據具體充分。

請依序逐步推理(漂綠稽核準則):
1. 這段在宣稱/承諾什麼?
2. 提供了什麼證據?是具體可查證(數據、指標、第三方查證、明確時程)還是空泛?
3. 漂綠紅旗檢查:是否把「法律本來就要求的事」當成就吹噓?是否用模糊承諾掩蓋無實質作為?是否誇大/挑選有利數據而迴避不利面?證據與宣稱是否有落差(言過其實)?
4. 綜合:這是「蓄意誤導(Misleading)」、還是只是「模糊(Not Clear)」、還是「具體(Clear)」?

最後輸出一行:`MISLEADING_SCORE: X`(X 為 0-10 整數,0=完全不誤導,10=明顯漂綠誤導)。

ESG 段落:
{text}"""

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True).eval()

def score_once(text, temp):
    msgs = [{"role":"user","content":PROMPT.format(text=str(text)[:MAX_IN])}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(chat, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_new_tokens=MAX_NEW, do_sample=(temp>0), temperature=max(temp,0.01),
                             pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.findall(r'MISLEADING_SCORE\D*(\d+)', txt)          # handles "**MISLEADING_SCORE: 6**"
    if m: return min(int(m[-1]), 10)
    tail = re.findall(r'(\d+)\s*/\s*10|\b([0-9]|10)\b', txt[-40:])  # fallback: last number / X/10
    nums = [int(a or b) for a, b in tail if (a or b)]
    return min(nums[-1], 10) if nums else None

rows=[]
for i,r in panel.iterrows():
    scs=[]
    for s in range(N_SAMPLES):
        v=score_once(r["data"], temp=(0.0 if s==0 else 0.7))
        if v is not None: scs.append(min(v,10))
    avg=float(np.mean(scs)) if scs else 0.0
    rows.append({"id":r["id"],"true":r["evidence_quality"],"score":avg,"raw_scores":scs})
    print(f"  {r['evidence_quality']:>10} id{r['id']} score={avg:.1f} {scs}", flush=True)
res=pd.DataFrame(rows)
json.dump(rows, open("agent_cache/misleading_v2_panel.json","w"), ensure_ascii=False, indent=1)
if SMOKE: print("[SMOKE] OK"); sys.exit(0)
print("\n=== 鑑別力:各類平均分 ===")
print(res.groupby("true")["score"].agg(["mean","min","max","count"]).to_string())
print("\n=== 排名:2 個真 Misleading 在全 panel 的分數排名(越高越好)===")
res_sorted=res.sort_values("score",ascending=False).reset_index(drop=True)
for _,r in res_sorted[res_sorted["true"]=="Misleading"].iterrows():
    rank=res_sorted[res_sorted["id"]==r["id"]].index[0]+1
    print(f"  Mis id{r['id']}: score={r['score']:.1f}, 排名 {rank}/{len(res_sorted)}")
# threshold sweep
mis_ids=set(mis["id"])
print("\n=== 門檻掃描 ===")
for thr in [4,5,6,7,8]:
    fl=res[res["score"]>=thr]; tp=fl["true"].eq("Misleading").sum()
    print(f"  >={thr}: flagged={len(fl)} tp={tp} fp={len(fl)-tp} prec={tp/max(len(fl),1):.2f} rec={tp/len(mis):.2f}")
print("\n判定:若 2 個真 Misleading 分數明顯高於 NotClear/Clear(排名在前)→ 有鑑別力 → 值得全 overlay")
