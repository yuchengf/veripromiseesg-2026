# AI CUP 2026 VeriPromiseESG 競賽報告

**隊伍代號:** TEAM_10505　**組別:** 學生組
**最終成績:** Public LB **0.6249259**(第 15 名)→ **Private LB 0.6432854(第 12 名)**

> 隊員 / 學號:〔請填寫〕
> 指導教授 / 業師:〔姓名、學校或公司、科系或職稱、信箱 — 若無可刪除本欄〕
> 可重現程式碼(GitHub):〔https://github.com/&lt;your-account&gt;/veripromiseesg-2026〕

---

## 1. 摘要

本競賽要求對企業 ESG 永續報告書段落,完成四個相依的子任務:**T1 承諾辨識(promise)、T2 證據佐證(evidence)、T3 證據品質(Clear / Not Clear / Misleading)、T4 驗證時程(timeline)**,以加權 macro-F1(`0.20·T1 + 0.30·T2 + 0.35·T3 + 0.15·T4`)評分。

我們的系統以 **中文 RoBERTa-large + 級聯多任務頭(cascade head)** 為骨幹,輔以 **kNN 標籤分布學習(kNN-LDL)** 與 **獨立 clarity 頭** 強化最弱的 T3,並以兩項方法學貢獻收尾:**(a) 逐子任務的 max_length「收割」(per-task harvest)**,**(b) 結構化聯合解碼(joint/structured decoding)**。

最關鍵的不是單一模型技巧,而是一套**抗過擬合的嚴謹方法論**:把交叉驗證(CV)、Public LB、Private LB 視為三個有偏的噪音估計,只採用能被**獨立證據共同確認(convergent evidence)**、且具**結構性(會轉移)**的改動;對任何只在單一榜單變好的「幻覺增益」一律拒絕。最終的證據是:我們的提交在 Private 重排中**從 Public 第 15 名上升至第 12 名(分數 0.6249 → 0.6432,+0.018)**,而許多在截止前大量刷 Public 的隊伍則退步——這正是上述紀律的直接驗證。

---

## 2. 任務定義與評分

| 子任務 | 類別 | 權重 | 依賴關係 |
|---|---|---|---|
| T1 promise_status | Yes / No | 0.20 | 最上層 gate |
| T2 evidence_status | Yes / No / N/A | 0.30 | T1=No → N/A |
| T3 evidence_quality | Clear / Not Clear / Misleading / N/A | **0.35** | T1=No 或 T2=No → N/A |
| T4 verification_timeline | already / within_2_years / between_2_and_5_years / more_than_5_years / N/A | 0.15 | T1=No → N/A |

- 評分為 **macro-F1**,且**僅計入解答中實際出現的類別**。
- 四任務間存在**邏輯硬性相依**(N/A 由上游決定),此結構是本題的核心特性,也是我們方法的著力點。

---

## 3. 系統架構與程式邏輯

整體推論流程(對應原始碼 `esg_main.py`):

```
段落文字
  └─ RoBERTa-large 編碼器(CLS pooling)
       └─ CascadeHeadV2 級聯頭
            T1 = Linear(h)
            T2 = Linear([h, softmax(T1)])
            T3 = Linear([h, softmax(T1), softmax(T2)])   # 看到 T1、T2
            T4 = Linear([h, softmax(T1)])                # 只看 T1
       └─ 3 seeds × 5 folds 機率平均(ensemble)
       └─ T3:NC-bias → kNN-LDL 融合 → clarity 頭覆蓋
       └─ 聯合結構化解碼(joint decode)+ N/A 級聯規則
  └─ 提交 CSV
```

### 3.1 骨幹與級聯頭
- **Backbone:** `hfl/chinese-roberta-wwm-ext-large`(在所有測試的 backbone 中轉移最佳;見 §7)。
- **CascadeHeadV2:** 把上游任務的 softmax 機率串接進下游頭的輸入,**在架構層面**直接編碼「T1→T2→T3、T1→T4」的依賴。

### 3.2 損失與訓練(per-task loss)
- T1 / T2:加權交叉熵(CE)。
- T3:**Distribution-Balanced(NTR)損失** + Not Clear 類別加權(`t3_nc_weight=3.0`),對抗 Clear ≫ Not Clear 的不平衡。
- T4:**ordinal 損失**(時程具自然順序)。
- **路徑損失(path loss):** T1=No 的列不計算下游 T2/T3/T4 損失(`mask_t1_no`),使下游頭只在「有意義」的列學習。
- **正則化:** R-Drop + SWA(主力配方 FC1R)、`augment_rare`(稀有類過採樣)。
- **訓練設定:** 5-fold 交叉驗證 × 3 個隨機種子(42 / 123 / 456)做機率平均。種子曲線(1→2→3 seed:0.658→0.673→0.679)顯示報酬遞減,3 seed 為最佳折衷(5 seed 在噪音內,不採用)。
- **兩套資料:** `final_data`(1601 筆,用於本地 valid 選型)與 `retrain_data`(2000 筆,用於產生最終提交)。

### 3.3 T3 強化模組(本題最弱、權重最高的任務)
- **NC-bias:** 推論時將 Not Clear 機率乘 `exp(0.1)` 再正規化。
- **kNN-LDL(標籤分布學習):** 以 `Qwen/Qwen3-Embedding-0.6B` 對訓練/測試文字做嵌入,取 k=5 最近鄰的 T3 標籤分布,與模型機率以 `0.6·model + 0.4·kNN` 融合(對應 `gen_rt_more.py`)。
- **Clarity 頭:** 一個獨立的 balanced-softmax T3 分類器(`clarity_head.py`),在高信心(≥0.7)且非 Misleading 時硬覆蓋模型的 T3 預測。

---

## 4. 兩項關鍵方法學貢獻

### 4.1 逐子任務 max_length「收割」(per-task harvest)
RoBERTa 上限為 512 token;我們發現:
- **資料分布:** p95 = 381 token,僅 4.8% 超過 384、1.4% 超過 512。
- 我們訓練了 @384 與 @512 兩套模型,並利用 **AIdea 每次提交都會回報四個子任務分數**的特性,用**一次**全 @512 的提交,一次讀出四個子任務在 @512 的分數,再與已知的 @384 對照,**逐任務**挑選較佳來源。
- **重大發現(valid 與 test 反轉):** 在本地 valid 上,@512 看似對 T3 較好、對 T4 較差;但在 AIdea(真實測試集)上**完全相反**——T4@512 大幅勝出(+0.0216)、T2@512 小勝,而 T3 應留在 @384。
- 最終 mix 來源:**T1/T3 @384,T2/T4 @512**(AIdea 0.6232)。此發現直接體現「本地 CV 與測試分布不一致時,應以**同分布**的訊號為準」。

### 4.2 結構化聯合解碼(joint / structured decoding)— 核心貢獻
原本的解碼是**貪婪、單向**的:先 argmax 上游 gate,若 T1=No 就強制把下游清成 N/A。這只用到依賴關係的**前向**。

我們改為**聯合解碼**:對每一列,列舉三個合法的聯合標籤分支(`No` / `Yes-No` / `Yes-Yes`),選擇使**四任務聯合對數機率最大**者,並使用模型**自己對下游 N/A 的機率**。如此,**下游的高信心可以反過來修正上游 gate**(後向):當模型很確定「有明確證據、有具體時程」時,即使 T1 的 gate 勉強判 No,聯合解碼也會把它拉回 Yes。

- 性質:**推論期、不需重訓、結構性**(用真實依賴 + 模型自身機率,而非調參捷徑)。
- 效果:**同一批機率,只改解碼方式**,即在 AIdea 上**四個子任務同時提升**(T1 0.7967→0.8008、T2 0.7057→0.7072、T3 0.4522→0.453、T4 0.6261→0.6271),總分 0.6232 → **0.6249**。
- **穩健性驗證:** 對解碼權重 `wgate ∈ [1.0, 2.5]` 擾動時,提交僅改變 0.1–0.2% 的儲存格(非刀鋒解);且最終檔有 98.2% 與已確認的 mix 相同——是「已確認基底 + 微小結構性修正」,而非脆弱的新建構。

---

## 5. 方法論與紀律(本報告的核心經驗)

我們認為本隊能在 Private 重排中上升,關鍵在方法論而非單一技巧:

1. **三個訊號皆為有偏噪音估計。** CV(本地 399)、Public LB、Private 是三個不同分布/不同樣本的估計。最昂貴的錯誤就是把其中一個當作真相去最佳化。
2. **Convergent evidence(共同確認)規則。** 一個改動只有在**兩個獨立訊號**(CV 與 LB,或 CV 與結構性機制)同時支持時,才採用為最終;只在單一訊號變好者,多半是該訊號的噪音。
3. **結構性改動會轉移,valid-tuned 不會。** 我們以此鐵律分類所有候選:joint decode、per-task @512、clarity overlay 屬結構性(已在 AIdea 確認並最終在 Private 轉移);而 per-class threshold、12-way 集成、company prior 等屬調參/捷徑,被拒絕。
4. **天花板診斷:T3 是 label-limited。** 我們從 6 個獨立角度(14B LLM 判別、領域特徵、跨 backbone、kNN、post-hoc logit 調整、train-time conditional-softmax)以及 SemEval-2025 Task 6 / LeWiDi 文獻,確認 **T3 的 Clear↔Not Clear↔Misleading 邊界無法由文字內容判定**(Misleading 是主觀標註,訊號不在輸入中)。因此我們**停止在 T3 上投入**,把資源集中在可轉移的結構性改動。
5. **把排行榜讀成噪音帶。** 以 bootstrap 估計指標抽樣標準差(本地 ≈ 0.02),與排行榜密集區(第 3–20 名僅 0.011 之差)相當——代表整個 pack 在噪音內,Public 名次不等於 Private 名次。我們**不追逐密集的 Public 榜**,而選擇結構性、未過度擬合的提交。
6. **最終提交的穩健性驗證。** 在鎖定前檢查:格式/級聯一致性、超參擾動穩定性、與已確認基底的組成、是否為截止前刷榜。

**結果驗證:** Public 0.6249(第 15)→ Private 0.6432854(**第 12**)。在 Private 重排中上升 3 名、+0.018 分;大量截止前刷 Public 的隊伍則退步。這是上述紀律的直接實證。

---

## 6. 結果

| | 名次 | 加權分數 | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|
| Public LB | 15 | 0.6249259 | 0.8008 | 0.7072 | 0.453 | 0.6271 |
| **Private LB** | **12** | **0.6432854** | — | — | — | —(官方未提供分項)|

- Private 相對 Public **+0.018**、名次上升,顯示提交對分布變動穩健(非過擬合 Public)。

---

## 7. 嘗試但**否決**的方法(誠實的負面結果)

下列方法皆經公平測試(本地 bootstrap + 在可行時以 AIdea 確認),因**過擬合**或**碰到 T3 的 label-limit** 而否決,構成「為何最終如此簡潔」的依據:

| 方法 | 否決原因 |
|---|---|
| company / ticker prior 融合 | 本地 valid 顯著正,AIdea 過擬合退步(身分捷徑不轉移) |
| 12-way 跨配方集成 | valid 0.681(最高)卻 AIdea 0.614(最差)= valid→test 反轉鐵證 |
| 跨 backbone 集成(XLM-R / mmBERT / BGE-M3 / DeBERTa) | 單獨皆弱於 RoBERTa;集成贏不過 |
| Misleading / T3 偵測(Qwen3-14B few-shot judge、LLM gate judge) | 對主觀標籤零鑑別力;LLM 過度預測負類 |
| 領域特徵 re-ranker、SupCon 對比微調、ordinal / OLL、conditional-softmax 條件鏈式 aux | 對 T3 分離度零幫助(representation 限制,非校準) |
| post-hoc logit-adjustment / per-class threshold、soft-macro-F1 | 只在固定決策面滑動,無法增加分離度;且過擬合 valid |
| 半監督 / pseudo-label | 確認偏誤;對 label-limited 類別注入錯誤標籤 |

**統一結論:** 上述多為「重塑輸出/損失」的方法;當瓶頸是**輸入訊號/標籤本身的限制**時,沒有輸出端方法能補回不存在的訊號。

---

## 8. 外部資料與生成式 AI 使用聲明(依簡章誠實揭露)

### 8.1 預訓練模型(開源)
- **`hfl/chinese-roberta-wwm-ext-large`**(HFL):最終提交之骨幹編碼器。
- **`Qwen/Qwen3-Embedding-0.6B`**(Alibaba Qwen):kNN-LDL 之文字嵌入。
- 僅用於**實驗、最終未採用**的模型:`Qwen/Qwen3-14B`、`Qwen/Qwen3-8B`(Misleading/gate 判別與條件嘗試)、`FacebookAI/xlm-roberta-large`、`jhu-clsp/mmBERT-base`、`BAAI/bge-m3`、`hfl/chinese-macbert-large`(跨 backbone 集成,皆否決)。

### 8.2 外部資料
- 下載並清理 **ML-Promise(SemEval-2025 Task 6)EN/FR/JA** 多語資料用於 T3 Misleading 增補實驗;**最終未採用**(跨語遷移失敗)。
- 競賽主辦單位已於官方討論串明確說明:**任何訓練階段皆不限制使用開源或閉源 LLM 及外部資料。**

### 8.3 生成式 AI 輔助(誠實貢獻聲明)
本隊在整個競賽過程中,使用 **Claude(Anthropic;透過 Claude Code,模型 Claude Opus 4.x)** 作為**程式開發、實驗設計、文獻研究與統計分析的 AI 助手**。具體用途包括:
- 撰寫訓練 / 推論 / 評估的 Python 程式碼;
- 設計與編排消融實驗、bootstrap 顯著性檢定;
- 搜尋與整理相關研究文獻(long-tailed / ordinal / structured prediction / 學習於標註分歧等);
- 提出並推理 per-task harvest 與 joint decoding 的方法。

**貢獻比例(誠實區分):**
- **隊伍主導:** 競賽策略與方向、方法論紀律的堅持(抗過擬合、convergent-evidence 規則、最終決策)、運算資源與環境、所有最終提交的選擇與把關。
- **AI 輔助:** 在隊伍指示下完成程式實作、實驗執行與數據分析、文獻研究。
- 致勝的核心構想(per-task harvest、joint decode 與整體紀律)源自**人機協作**:由隊伍提出問題與紀律約束,AI 協助快速實作、驗證與分析。

> 〔請隊伍依實際情況校準本節用字與比例;以上為誠實草稿。〕

---

## 9. 可重現程式碼

- **GitHub Repository:** 〔https://github.com/&lt;your-account&gt;/veripromiseesg-2026〕(公開,可直接 clone 執行)
- **主要檔案:**
  - `esg_main.py` — 模型、級聯頭、損失、訓練 / k-fold / 推論主程式。
  - `gen_rt_more.py` — 以 retrain 模型產生提交、kNN-LDL 融合。
  - `gen_joint_aidea.py` — **聯合結構化解碼**,產生最終提交。
  - `clarity_head.py` — 獨立 clarity 頭訓練與覆蓋。
  - `eval_single_run.py` — 本地 valid 評估與機率快取。
- **環境:** Python 3.13、PyTorch(CUDA)、transformers;單張 GPU(RTX 5090, 32GB)。
- **重現最終提交:** 詳見 repo `README.md`(資料路徑、訓練指令、`python gen_joint_aidea.py 2.0` 產生 `aidea_joint_w2.0.csv`)。

---

## 10. 結論

在一個四子任務、嚴重類別不平衡、且最關鍵任務(T3)本質上 **label-limited** 的題目上,我們以中文 RoBERTa 級聯架構為基礎,透過 **per-task max_length 收割** 與 **結構化聯合解碼** 取得可轉移的增益,並以**抗過擬合的方法論紀律**作為最重要的決策準則。最終在 Private LB 取得**第 12 名(0.6432854)**,且相對 Public **名次與分數雙雙上升**——這正是「只押結構性、共同確認的改動,不追逐單一榜單」的價值所在。
