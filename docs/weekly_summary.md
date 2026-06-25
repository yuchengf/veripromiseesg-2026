# ESG 競賽週報（Sub ~50 之後）

## 當前最佳：0.65994（rob_aug + lert_aug ensemble）

---

## 一、LLM LoRA 格式實驗（4 組）

在 sub ~50 之前已有 RoBERTa kfold（0.624）和 rob:lert 5:4（0.644）。
本週繼續做以下實驗：

| 組合 | Public F1 | 備注 |
|------|-----------|------|
| pc_rag3 | 0.55433 | LLM 系列最佳 |
| json_rag3 | 0.54719 | |
| pc_norag | 0.49413 | |
| json_norag | 0.48714 | |

**結論**：LLM 天花板 ~0.55，比 BERT 低 0.10，後續放棄 LLM 方向。

---

## 二、System Prompt 設計（完整版）

### 架構

兩個 System Prompt 共用同一個 `_TASK_DEF` 核心，再各自加上輸出格式說明。

---

### _TASK_DEF（共用核心，v2 版本）

```
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
```

---

### SYSTEM_PROMPT_PROMPTCAST（PromptCast 格式）

```
你是一位專業的ESG（環境、社會、治理）報告核實分析師。
分析給定的ESG報告片段，先識別關鍵文字片段，再依序評估四個指標。

[_TASK_DEF 插入]

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
此段文字為純粹的事實描述，不含ESG承諾或目標（No），故執行證據（N/A）、證據品質（N/A）及驗證時間線（N/A）均不適用。
```

---

### SYSTEM_PROMPT_JSON（JSON CoT 格式）

```
你是一位專業的ESG（環境、社會、治理）報告核實分析師。
分析給定的ESG報告片段，先識別關鍵文字片段，再評估四個指標，以JSON格式輸出。

[_TASK_DEF 插入]

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
{"promise_string": null, "evidence_string": null, "promise_status": "No", "evidence_status": "N/A", "evidence_quality": "N/A", "verification_timeline": "N/A"}
```

---

### 改進歷程

- **v1**：只有 T1-T4 定義，無語言指引
- **v2（當前）**：加入「語言信號指引」section：
  - 模糊字眼 list → Not Clear 判斷規則
  - 具體佐證信號 → Clear 判斷規則
  - 時間線關鍵詞 → T4 判斷規則
  - 強制邏輯規則（cascade 依賴明確化）

**結果**：pc_rag3_v2 val 0.5110（比 v1 的 0.5430 更差），未送。
推測原因：prompt 變複雜後模型更容易 overthink，val 排名本就不可靠（val ≠ test）。

---

## 三、ModernBERT kfold

- Public F1：**0.54292**（比 macBERT 還弱）
- 結論：❌ 不加入 ensemble

---

## 四、LLM 資料增強（gen_llm_augdata）

目標：補充少數類（Not Clear 99→200, Misleading 1→30, within_2y 11→50）

**兩個關鍵修正**：
1. LLM self-verification 自相矛盾問題（同一個 LLM 生成模糊文字，卻分類成 Clear）→ 改用規則型驗證
2. `VERIFY_PROMPT` 中 JSON `{...}` 被 Python format string 解析為佔位符 → 改為 `{{...}}`

最終生成：169 筆（101 Not Clear + 39 within_2y + 29 Misleading）

---

## 五、BERT 實驗失敗三連

| 實驗 | Val CV | Public F1 | 結論 |
|------|--------|-----------|------|
| RoBERTa + LLM aug | — | 0.62215 | ↓ 低於 baseline 0.624 |
| 10-fold CV | 0.6046 ± 0.0453 | 0.58838 | ↓ 嚴重下降 |
| Adaptive task-weighting（HCAL-style） | 0.5882 ± 0.0303 | 未送 | ↓ 遠差於 baseline |

---

## 六、LERT + LLM aug kfold → 突破

LERT + augmented data：Val CV **0.7296**（非常高）

Ensemble 結果：

| 組合 | Public F1 |
|------|-----------|
| lert_aug 單獨 | 0.58887（比 LERT baseline 0.605 還差）|
| rob_aug + lert_aug 1:1 | **0.65994** ← 新最佳 |
| rob_aug + lert_aug 5:4 | **0.65994**（同） |
| rob_aug + lert_aug 9:8 | **0.65994**（同） |
| aug_rob + orig_lert 5:4 | 0.65282 |
| orig_rob + aug_lert 5:4 | 0.63432 |
| aug + orig 4way | 0.64958 |

**關鍵發現**：
- aug ensemble 任何比例結果相同 → 兩個 model 預測高度一致，加權不影響 argmax
- rob_aug 貢獻 >> lert_aug（s81 > s82）
- 加回 orig model 反而拖分

---

## 七、架構探索討論

討論 CascadeHead 改進方向：
- 現在：T1 softmax concat 到後續任務 input（線性、無參數）
- 提案：FiLM/Gate conditioning（用 T1 預測調整整個 hidden state）
  - Option A：proj 2→1024，residual add
  - Option B：Gate `x * sigmoid(W * t1_prob)`（推薦，輕量）
  - Option C：FiLM `gamma * x + beta`
- Survey 方向：FiLM conditioning MTL、hierarchical MTL gating、classifier chain + neural

---

## 八、V12 weight search 結果 ✅

| 組合 | Public F1 |
|------|-----------|
| s84 rob_aug:lert_aug = 2:1 | 0.63820 |
| s85 rob_aug:lert_aug = 3:1 | 0.62232 |
| s86 aug_rob + orig_lert 9:8 | 0.63148 |
| s87 aug2 + orig_rob 3way 等權 | 0.62951 |
| s88 aug2 + orig_rob 2:2:1 | 0.65809 |

**結論**：全部低於 0.65994。加重 rob_aug 或加回 orig 都無法突破。**0.65994 是當前 ensemble 天花板。**

---

## 九、下一步

### FiLM/Gate CascadeHeadV3（唯一未試的架構改進方向）
- Survey：FiLM conditioning MTL、hierarchical MTL gating、classifier chain + neural
- 實作 Gate 版本（Option B，最輕量）：
  ```python
  self.t1_gate = nn.Linear(2, hidden_size)  # 2 → 1024
  # forward: gate = sigmoid(t1_gate(t1_prob)); x2 = x * gate
  ```
- 用 aug data 重跑 rob + lert kfold 比較有無 gate 的差異

---

## 十、Encoder-Decoder 架構可能性

**問題**：當前 BERT encoder-only + linear head，是否值得改成 Encoder-Decoder（T5/mT5 style）？

**分析**：

| 面向 | Encoder-Only（現在） | Encoder-Decoder |
|------|---------------------|-----------------|
| 適合任務 | 分類 | 生成/seq2seq |
| 輸出方式 | 每個 task 獨立 linear | 自回歸生成 label 字串 |
| 資料量需求 | 低（800 筆足夠） | 高（生成模型更容易過擬合）|
| 推論速度 | 快 | 慢（autoregressive） |
| Label dependency | 靠 cascade head 硬編 | 可透過 generation 順序自然建模 |
| 已知結果 | BERT 0.66，LLM 0.55 | 類似 LLM，預期 ~0.55-0.60 |

**結論**：LLM 實驗已測試生成式方向（Qwen3-8B），天花板 0.55，遠低於 BERT 0.66。
Encoder-Decoder 屬於同一類（生成式），預期不會超越。
800 筆資料對 Decoder 部分過擬合風險高。

**除非**：用小型 T5（base/small），且輸出固定 template（非自由文字），當作更結構化的 BERT 替代品——但實作複雜度高，CP 值低。**暫不推薦。**

---

## 十一、已確認有害的方向（不要做）

- ❌ T2 用 Focal loss
- ❌ macBERT / ModernBERT 加入 ensemble
- ❌ single model 加進 kfold ensemble
- ❌ 生成式 LLM 做分類（800 樣本過擬合）
- ❌ 靠 val F1 選 submission（val variance 太大）
- ❌ LLM self-verification
- ❌ 10-fold（val 太小，不穩定）
- ❌ Adaptive task-weighting（HCAL）（比 baseline 差很多）
- ❌ aug ensemble 加重 rob_aug 或加回 orig（全部低於 0.65994）
