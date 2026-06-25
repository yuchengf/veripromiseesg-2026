"""Misleading detector: Qwen3-14B few-shot judge.

Phase 1 (panel): validate prompt on train — 2 real Misleading (leave-one-out
prompting) + 60 sampled non-Misleading. Saves score distributions.
Phase 2 (scan): score all 2000 AIdea test rows. Resume-safe JSONL.

Output: agent_cache/misleading_panel.json, agent_cache/misleading_scan.jsonl
Thresholding/overlay decided later from panel stats.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/yucheng/Desktop/ESG")
MODEL = "Qwen/Qwen3-14B"

train = pd.read_csv(ROOT / "final_data/train_data.csv", keep_default_na=False)
test = pd.read_csv(ROOT / "retrain_data/test_data.csv", keep_default_na=False)
mis = train[train["evidence_quality"] == "Misleading"]
assert len(mis) == 2, f"expected 2 Misleading, got {len(mis)}"

DEFINITION = """\
你是ESG永續報告審查專家，任務是偵測「誤導性陳述（Misleading）」。

定義：段落包含企業承諾且看似有佐證，但佐證有下列問題之一：
- 佐證與承諾不相符或無關（拿無關資訊偽裝成佐證）
- 空泛的官樣文章包裝成具體行動（只有口號，無可驗證的作為、數字、機制）
- 誇大或漂綠（greenwashing）：宣稱效果超出佐證能支持的範圍
- 佐證內部矛盾或避重就輕

注意：「Not Clear（模糊）」不算 Misleading——模糊是資訊不足，誤導是資訊帶有欺騙性。
只有當段落刻意營造「有具體佐證」的印象但實質不符時，才是 Misleading。"""


def build_prompt(text: str, exclude_id: int | None = None) -> str:
    shots = []
    for _, r in mis.iterrows():
        if exclude_id is not None and r["id"] == exclude_id:
            continue
        shots.append(f"【誤導範例】\n{r['data'][:450]}")
    examples = "\n\n".join(shots)
    return f"""{DEFINITION}

{examples}

請評估以下段落的誤導程度，給 0-10 分（0=完全不誤導，10=明確誤導），並給出判定。
只輸出 JSON：{{"misleading_score": <0-10>, "verdict": "Yes/No"}}

段落：
{text[:600]}"""


def judge(model, tok, device, text: str, exclude_id=None):
    prompt = build_prompt(text, exclude_id)
    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(chat, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    score, verdict = None, None
    m = re.search(r'"misleading_score"\s*:\s*(\d+(?:\.\d+)?)', raw)
    if m:
        score = float(m.group(1))
    m = re.search(r'"verdict"\s*:\s*"(Yes|No)"', raw, re.IGNORECASE)
    if m:
        verdict = m.group(1).capitalize()
    return score, verdict, raw


def main():
    device = "cuda"
    print(f"Loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                 device_map=device)
    model.eval()

    # ── Phase 1: panel ──
    panel_path = ROOT / "agent_cache/misleading_panel.json"
    if not panel_path.exists():
        rng = np.random.RandomState(42)
        t2yes = train[(train["promise_status"] == "Yes") & (train["evidence_status"] == "Yes")]
        neg_clear = t2yes[t2yes["evidence_quality"] == "Clear"].sample(30, random_state=rng)
        neg_nc = t2yes[t2yes["evidence_quality"] == "Not Clear"].sample(20, random_state=rng)
        neg_other = train[train["evidence_status"] == "No"].sample(10, random_state=rng)
        panel = []
        for _, r in mis.iterrows():
            s, v, raw = judge(model, tok, device, r["data"], exclude_id=r["id"])
            panel.append({"id": int(r["id"]), "true": "Misleading", "score": s,
                          "verdict": v, "raw": raw})
            print(f"  [panel] id={r['id']} true=Misleading score={s} verdict={v}", flush=True)
        for df, lab in [(neg_clear, "Clear"), (neg_nc, "Not Clear"), (neg_other, "EvidNo")]:
            for _, r in df.iterrows():
                s, v, _ = judge(model, tok, device, r["data"])
                panel.append({"id": int(r["id"]), "true": lab, "score": s, "verdict": v})
        panel_path.write_text(json.dumps(panel, ensure_ascii=False, indent=1))
        scores = {}
        for p in panel:
            scores.setdefault(p["true"], []).append(p["score"] if p["score"] is not None else -1)
        for lab, ss in scores.items():
            print(f"  [panel] {lab:10s} n={len(ss)} mean={np.mean(ss):.2f} "
                  f"max={max(ss)} yes_rate={np.mean([s >= 7 for s in ss]):.2f}", flush=True)
    else:
        print("panel exists, skip", flush=True)

    # ── Phase 2: scan 2000 test rows (resume-safe) ──
    scan_path = ROOT / "agent_cache/misleading_scan.jsonl"
    done = set()
    if scan_path.exists():
        for line in scan_path.open():
            done.add(json.loads(line)["id"])
    print(f"Scanning test: {len(test)} rows, {len(done)} already done", flush=True)
    with scan_path.open("a") as f:
        for n, (_, r) in enumerate(test.iterrows()):
            if int(r["id"]) in done:
                continue
            s, v, raw = judge(model, tok, device, r["data"])
            f.write(json.dumps({"id": int(r["id"]), "score": s, "verdict": v},
                               ensure_ascii=False) + "\n")
            f.flush()
            if n % 100 == 0:
                print(f"  scan {n}/{len(test)} id={r['id']} score={s}", flush=True)
    print("Scan done.", flush=True)


if __name__ == "__main__":
    main()
