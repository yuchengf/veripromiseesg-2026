# Final Stage 作戰計畫（2026-06-13 ~ 06-17）

> 比賽：AI CUP 2026 VeriPromiseESG（= SemEval-2025 Task 6 中文延伸版，ML-Promise 4K 資料集）
> 現況：AIdea **0.6170501，4/93**（aidea_rt_fc1r3_knn.csv）；Kaggle final-stage best 0.62039（6-way+kNN）
> 目標：穩住獎金圈（前 3 = 3~8 萬），衝 Private 第一

## 0. 規則紅線（已從官網確認）

- **06/17 預測上傳截止**；AIdea 每日上限 3 次，**系統保留最後一次**（非最佳）
- 最終排名 = **Private LB + 報告審查**；Public 僅參考
- **06/24–06/30 必繳報告書 + 原始程式碼 + 自構訓練資料**，漏交 = 除名、無補交
- 預測必須程式自動生成，嚴禁人工修正

## 1. 文獻洞見（SemEval-2025 Task 6 overview, 2025.semeval-1.321）

1. **均勻投票 ensemble 會被弱模型拖累** → 作者明確建議 per-task weighted/selective ensemble（所有隊伍都沒做好）
2. 中文賽道冠軍 SemanticEval 0.561（無論文）— 我們已超越此水準
3. DeBERTa + 目標式增強對 evidence/timeline 有效（Oath 隊）
4. R-Drop 有效（YNU-HPCC）→ FC1R 已驗證 ✅
5. LLM 增強對中文有效但過量有害 → 與 D1 overfit 觀察一致 ✅
6. XLM-R-large 是 top 系統常用 backbone，**我們唯一沒試過的主流 backbone**
7. LLM 結構化 prompt + RAG + CoT 可行（CYUT 法文冠軍）— 成本高，備選

## 2. 模型資產

| 代號 | 訓練資料 | Backbone | 變體 | 用途 |
|------|---------|----------|------|------|
| FC1 / FC1R / FC1S / FC1M ×3 seeds | final_data train 1601 | RoBERTa-large | base / R-Drop+SWA / SharpReCL / MR2 | **valid 399 本地驗證** |
| FD1 / FD1R ×3 seeds | final_data train 1601 | DeBERTa-320M | base / R-Drop+SWA | 跨 backbone 多樣性（06-13 凌晨完成） |
| RT_FC1* ×12 | retrain_data 2000（train+valid 全量） | RoBERTa-large | 同上 4 種 | **產生正式 submission** |

工作流：**FC1*/FD1* 在 valid 399 上選配方 → 把配方套用到 RT_* 模型 → Kaggle public 驗證 → AIdea**
（RT 缺 FD1 對應版 → 若 DeBERTa 配方有效，需補訓 RT_FD1* ×3 на retrain_data）

## 3. 實驗排程

### Day 0（06-13 凌晨，自動執行中）
- [x] FD1_s2 訓練完成（~01:00）
- [ ] `pertask_valid_eval.py` 自動跑完（~01:45）：
  - 18 模型 × 5 fold 預測 valid 399（cache: `agent_cache/valid_probs/`）
  - 每組 per-task F1 表 + 參考組合（FC1R3 / 6-way / 12-way / 跨 backbone）
  - 貪婪 per-task 群組選擇（T1→T2→T3→T4，NA-rule aware）
  - Qwen3 kNN alpha sweep（a2∈{0,.1,.2} × a3∈{0,...,.5} × a4∈{0,.1,.2}）
  - 結果: `agent_cache/pertask_valid_results.json` + `agent_cache/pertask_eval.log`

### Day 1（06-13 白天）
- [x] 分析 valid 結果，決策門檻：**per-task 組合 ≥ uniform best + 0.005 才算有效** → **未過：Layer 2 否決（見決策紀錄）**
- [x] ~~有效 → 用 RT_* 模型按配方生成 submission~~ → 改為維持 uniform 12-way 路線
- [x] ~~DeBERTa 有貢獻 → 啟動 RT_FD1/FD1R retrain~~ → **取消**（DeBERTa 拖分）
- [ ] （並行）XLM-R-large FX1R ×3 seeds 在 final_data 訓練（驗證第三 backbone 價值）
- [x] AIdea 12-way+kNN 已於 06-12 19:04 繳交：**0.6140110**（T3=0.4396）< FC1R3+kNN 0.6170501 →差距 0.003 在 public 雜訊範圍
- [x] **評估 bug 發現並修復**：solution 的 "N/A" 被 pandas 讀成 NaN → N/A 類 F1 恆 0。修正後（keep_default_na=False）重跑全部評估
- [x] 修正後結論：①per-task recipe 仍被否決（CI [−0.0234, −0.0032] 全負）②FC1M 平反：12w=0.67882 > 9w=0.67481 > 6w=0.67334 > FC1R3=0.66206 ③α=0.4 確認最佳 ④**valid-Public 12w 0.62881 ≈ Kaggle 0.62880：本地評估與官方計分完全對齊**
- [x] **T3 重大發現**：valid T3（3類）=0.616，AIdea T3=0.4396 ≈ 0.616×3/4 → **AIdea 測試集含 Misleading 類**，我們從不預測它（train 僅 2 例）→ F1=0 拖累 T3 macro。高精度 Misleading 偵測 = +0.02~0.035 總分潛力
- [ ] Misleading hunter：Qwen3-8B few-shot 判別（llm_tta.py 基礎設施現成）+ 高信心 overlay 到 12w 預測
- [ ] AIdea 額度（06-13 的 3 次）：留給 Misleading overlay probe；**注意現在 standing = all12_knn 0.614（06-12 最後上傳），收盤前須換回最佳檔**

### Day 2（06-14）
- [ ] RT_FD1* / XLM-R 結果整合，重跑 per-task 選擇
- [ ] kNN 細掃（k∈{5,10}, sim_temp, t3_la_tau）於最終配方上
- [ ] AIdea 探測：最佳新配方 vs 現有 0.617 比較

### Day 3（06-15 ~ 06-16）
- [ ] 凍結配方。風險評估：valid 399 過擬合檢查（Public/Private 分半比對、bootstrap CI）
- [ ] 生成最終 submission + 備援（保守版 = 已驗證 0.617 配方）

### Day 4（06-17 截止日）
- [ ] **最後一筆上傳 = 最有信心的檔案**（保留規則！）
- [ ] 若新配方 Public 未明顯勝出 → 交保守版

### 06-18 之後
- [ ] 整理可重現程式碼（esg_main.py + 訓練腳本 + LLM 增強資料）
- [ ] 報告書：方法、消融實驗（SUBMISSIONS.md 有完整紀錄）、外部資源討論（規則要求）
- [ ] 06/24–06/30 繳交，**勿拖到最後一天**

## 4. 防 overfit 紀律：三層隔離驗證協議

```
Layer 1（選擇）  OOF on train 1601 — 貪婪 per-task 選擇 + kNN alpha 掃描只在這裡做
                 （kNN 用 self-excluded 近鄰；alpha 選 plateau 中心，不選尖峰）
Layer 2（確認）  valid 399 — 選好的配方只測一次 + bootstrap 95% CI vs 最佳 uniform ref
Layer 3（泛化）  Kaggle public — 再測一次
三層都贏 → 才上 AIdea；任何一層失敗 → 退回保守配方（已驗證 0.617）
```

- 選擇粒度限制在「模型群組」層級（6 組），不做 per-seed/per-class threshold（V18/V29/V35 已證明不泛化）
- CI 重疊 0 視為平手 → 選簡單配方
- 失敗方向清單見 memory `project_esg.md` 第五節 — 不要重試
- 無法保證雙榜第一（private ~1000 筆、T3 稀有類波動 ±0.01），策略 = 最大化期望分數並最小化變異：
  ensemble 降變異 + 跨 stage 重現過的方法優先（nc3、kNN α≈0.4、R-Drop 都是多階段驗證過的）

## 5. 決策紀錄

| 日期 | 決策 | 依據 |
|------|------|------|
| 06-13 | 主攻 per-task selective ensemble | SemEval overview 建議 + 零訓練成本 |
| 06-13 | DeBERTa-320M 補進 ensemble 池 | Oath 隊證據 + 跨 backbone 多樣性 |
| 06-13 | XLM-R-large 列為候選 | 唯一未試的主流 backbone |
| 06-13 | LLM RAG/CoT 暫緩 | 成本高、剩 4 天、現有路線未榨乾 |
| 06-13 | **per-task selective ensemble 否決** | Layer 1 OOF +0.005 過門檻，但 Layer 2 valid 0.49673 vs uniform 0.50849，CI [−0.0215, −0.0036] 全負 → 顯著更差，依協議退回 |
| 06-13 | **DeBERTa-320M 移出 ensemble 池；取消 RT_FD1 補訓** | FD1/FD1R OOF 0.585/0.592 遠低於 RoBERTa ~0.62，所有含 DeBERTa 組合在 OOF 與 valid 都更差 |
| 06-13 | 本地最佳配方 = uniform 12-way RoBERTa（FC1+FC1R+FC1S+FC1M）+ kNN α3=0.4 + NC bias 0.1 | 修正 N/A bug 後 valid=0.67882 全組合最高；valid-Public 0.62881 重現 Kaggle 0.62880 |
| 06-13 | **主攻 T3 Misleading**：AIdea T3 gap 幾乎全由 Misleading F1=0 解釋 | AIdea 0.4396 ≈ valid 3-class 0.616×3/4；期望 +0.02~0.035 總分 |
| 06-13 PM | **Misleading overlay 放棄（DEAD）** | Qwen3-14B few-shot judge 對 panel 零鑑別力：2 個真 Misleading 得分 0/2，所有門檻 precision=recall=0（全是 Clear/NC 假陽性）。overlay 只會注入假陽性傷 T3。Misleading 標籤太主觀 + train 僅 2 例 |
| 06-13 PM | **FX1R (XLM-R-large ×3) 否決** | valid 上加入反而變差：5-group FC1+FC1R+FC1S+FC1M+FX1R=0.66601 < 4-group 0.67310。跨 backbone 戰績 0 勝 9 敗 |
| 06-13 PM | **per-task selection + kNN 再次否決** | valid 0.65974 < uniform 0.67310，bootstrap CI [−0.0282,+0.0010] 跨 0 → 依協議取簡單者（uniform 12-way） |
| 06-13 PM | 修 watcher 死鎖（pgrep -f 自我匹配 harness wrapper cmdline）→ relaunch_gpu.sh 串接 mmBERT/BGE-M3 smoke + supcon | train_newbb/queue_supcon 因 pgrep 匹配到含 overnight.sh 字串的常駐 wrapper 永遠看不到 quiet |
| 06-13 PM | **最終押 12-way（robust 優先）** | 使用者決定：valid 0.017 優勢在已對齊集合上、且 12 模型低 variance 於 Private 更穩；AIdea public 3-way 領先 0.003 視為雜訊 |
| 06-13 PM | **LOO 消融驗證 12-way 每個成員都站得住（loo_ablation.py）** | 逐組剔除 4 架構 delta 全負（無拖油瓶）；逐 seed 剔除 12 個無一 CI 全正（無死重），6/12 顯著正貢獻；seed 曲線單調 0.658→0.673→0.679（3 seed>2>1，報酬遞減合理）。報告 ensemble justification 證據 |
| 06-13 PM | mmBERT / BGE-M3 smoke 否決 | single 0.63461 / 0.65504 < FC1R 0.66。跨 backbone 0-11 |
| 06-13 PM | **SupCon kNN 對比微調否決（公平測試）** | 修掉 OOM + `-inf*0` NaN loss bug 後重跑：Phase A OOF kNN T3 0.3102→0.3635（+0.053 真進步）通過 gate，但 Phase B valid 融合 0.67882→0.66276（−0.016）。孤立 kNN 進步但融進 ensemble 變差（過擬合鄰居結構 + 互補性下降）。三層協議擋下 |
| 06-13 PM | **收官：所有 lever 用盡，最終 = 凍結 12-way + kNN（valid 0.67882）** | backbone/per-task/Misleading/SupCon 全否決。唯一剩餘動作 = 06-17 收盤前上傳 aidea_rt_all12_knn.csv |
| 06-13 晚 | 外部資料**明確允許**（簡章 L71-74 逐字確認，存 external_data/official_drive/）；下載 ML-Promise EN/FR/JA 清理合併 final_data_ml（2801 筆，Misleading 2→20） | 規則綠燈，攻 T3 Misleading 唯一上限路線 |
| 06-13 晚 | **多語 Misleading overlay 否決**：XLM-R 在 ZH+EN/FR/JA 訓練（MX1R），但在中文 valid 對 Misleading 的 max P=0.021、預測 0 個 → 跨語遷移失敗，overlay 在 AIdea 必為 no-op、零增益。standalone valid 0.638<0.679 | 18 個跨語 Misleading 訊號太薄、佔比 0.7%，標準 CE 下模型永不 fire。valid 0 Misleading 無法本地驗證精度 → 強迫 fire 變盲賭 |

---

## 研究綜合 + Next Steps（2026-06-14，四向 web survey）

> 註:subagent 因組織月花費上限不可用,以下全為直接 web 查證。survey 多綁 2024-2025,另補一輪 2026 專掃。

### 四向發現(附來源)
- **方向1 SemEval Task6 + 漂綠**:中文第一名 SemanticEval 未發論文;**GPT-4o 資料增強對中文有效但過量傷害**;官方:Clarity/Timing 最難(主觀、標註者一致性低),Misleading 各語極少;官方建議**跨語「翻譯」增強**(非遷移)。漂綠核心信號=**specificity vs vagueness**(ClimateBERT-Specificity, Bingler 2024)。Misleading 官方定義=證據蓄意欺騙/誤述履行=**承諾↔證據不一致**。
- **方向2 cascade 依賴**:**Conditional softmax**(arXiv 2410.01305)P(y|x)=∏P(z|x,parent),兄弟內歸一,loss 只算真實路徑;**深度+不平衡並存時勝過獨立頭+事後約束**(=我們情境)。logit-adjusted 版 s[y]+τ·log ν(y|π) 提升稀有葉。HCAL(2508.13452)adaptive loss balancing λ=softmax(L_i/γ)。
- **方向3 稀有類**:**Text Grafting**(2406.11115)=挖真語料模板→LLM 生可變內容→嫁接→majority 分類器過濾(修正 D1 過擬合死因);**「LLM 當生成器>分類器」**(2601.16278)~50-100 合成樣本即夠、>150 停滯;NLI 零樣本對中文+抽象標籤先驗差;Snorkel LF 弱監督。
- **方向4 長尾**:logit adjustment / balanced softmax / LDAM / **decoupling-cRT(只重訓分類頭+平衡採樣)**。
- **2026 專掃**:DeepGreen(2504.07733,**中文** LLM 漂綠);A3CG 資料集(2502.15821,抗漂綠 aspect-action);robust greenwashing(2601.21722);EmeraldMind KG(2512.11506);⚠️ 2605.07201 **兩階段階層分類 underperform、泛化落差大**(對提案#1的黃旗);RSG 潛空間稀有類合成(2509.15859)。

### Next Steps(排序:期望值×可驗證×趕得上 06-17)
1. **[最優] Conditional-softmax cascade + logit-adjusted「純 clarity 頭」**:只在 T2=Yes 列(~1076)學 Clear/NotClear/Misleading 三分類,兄弟內歸一,稀有類加 τ·log-prior bias。攻 Not Clear recall 0.40(可本地驗證 45 個)。⚠️監控泛化落差(2605.07201 警訊)。
2. **[廉價先導] cRT**:凍結現有 backbone,只重訓 T3 頭(平衡採樣/logit adj)→ 先驗證 Not Clear 是否可救,再決定要不要做 #1 全訓。
3. **[唯一有據的 Misleading 槓桿] Text-Grafting 合成**:挖中文 ESG 模板+LLM 生 Misleading/NotClear(~100,過濾)+ ML-Promise 翻譯增強。Misleading 不可本地驗證(valid 0 個)、NotClear 可。
4. 中文 specificity/含糊度特徵(詞表或模型)當信號/rerank。
5. 前沿 LLM/NLI Misleading overlay(需 AIdea probe,中文+抽象封頂)。

### 已否決(不重複):per-task ensemble、跨 backbone(0-11)、SupCon、Qwen3 LLM-judge、跨語**遷移**(非翻譯)、Focal/OLL/Ordinal、nc≥4、pseudo-label、5-seed。

---

## P-series 實作結果（2026-06-14 03:05，停電前）

- **P1 logit-adjustment（訓練-free）否決**:post-hoc prior-correction 救不動 Not Clear——τ 升則 NotClear recall 18-19/45 但總分掉(犧牲 Clear/N/A)。現有 NC-bias 0.1 已是 calibration 最佳點。→ **證明 Not Clear 瓶頸是表徵非校準**。(p1_logit_adjust.py)
- **P3 conditional clarity 頭（balanced-softmax,只在 T2=Yes ~1076 列學 3-way）→ overlay = 第一個贏過 0.67882**:
  - baseline 12-way: 0.67882 (Pub 0.62881 / Priv 0.72582, NotClear 17/45)
  - **overlay conf>=0.7: 0.68059** (Pub 0.63516 **+0.0064** / Priv 0.72324, NotClear 19/45)
  - conf>=0.8: 0.68034。clarity 頭 mean heldout acc 0.828,在 110/399 valid 預測 NotClear。
  - ⚠️ 增益小(+0.0018)、門檻在 valid 挑的(需 bootstrap CI)、Public↑Private↓(疑雜訊)。**第一個正訊號,未確認**。
  - 產物:`clarity_head.py`、`agent_cache/clarity_valid_probs.npz`(valid 3-way 機率快取)。

### RESUME HERE（電來後續跑,按序）
1. **bootstrap CI** 驗證 clarity overlay(conf>=0.7)的 +0.0018 是否顯著(vs 0.67882),Public/Private 分開看。
2. **#3 翻譯增強**:Qwen3-14B 把 ML-Promise EN/FR/JA 的 18 Mis+230 NotClear 翻成中文 → 加進 clarity 頭訓練(→894 Clear/410 NC/20 Mis)→ 重訓 clarity_head.py → 重評 overlay(NotClear recall + Misleading 假陽性率 + 總分)。
3. 若確認有效 → 把 clarity 頭套到 **RT_* (retrain 2000) 模型**生成 AIdea 檔 → **花 1 個額度 probe**(看 AIdea T3 是否真升)。
4. 並行:bootstrap、specificity 特徵(#4)為後備。
- 凍結保底檔不變:06-17 收盤前最後一筆 = aidea_rt_all12_knn.csv(或確認後的 clarity-overlay 版)。
  - bootstrap CI(2026-06-14 03:08):ALL delta +0.0016 CI[-0.0046,+0.0093] P(>0)=0.64;Public delta +0.0063 CI[-0.0054,+0.0211] P(>0)=0.81 → **跨 0 不顯著,偏正(Public 較強)**。視為「有戲未證實」,需 #3 強化訊號 + AIdea probe。(bootstrap_clarity.py)

---

## #3 翻譯增強結果（2026-06-14 13:xx）—— 否決

- 翻譯 248 筆(230 NotClear+18 Misleading)EN/FR/JA→繁中(Qwen3-14B),加進 clarity 頭(→894/410/20)重訓。
- overlay 全面變差:conf>=0.8 **0.67527** < baseline 0.67882 < 非增強 P3 0.68059;NotClear recall 17→16。
- 結論:翻譯腔/外語分布傷原生中文 valid(印證 overview「過量合成增強有害」)。**翻譯增強否決**。
- → **最佳 overlay 仍是非增強 clarity 頭 conf>=0.7 = 0.68059(未顯著,Public lean +0.0064)**。產物 clarity_valid_probs.npz(非增強)。

---

## ★ AIdea probe 確認:clarity overlay 有效（2026-06-14 14:06）

- `aidea_clarity_thr0.7_noMis.csv` → **AIdea 0.6158504 (11/105)**, T3 **0.4449**(vs 12-way 0.4396, +0.0053), T1/T2/T4 不變。
- **第一個 AIdea 確認正增益**:NotClear 精修(19 列 Clear→NotClear)有效。但 0.6159 < 3-way 0.6170(底較弱)。
- Misleading 仍測不出(clarity head 在 2000 測試 P(Mis)>0.7 = 0 列,第 4 次確認)。
- → 下一步:clarity overlay 套到 3-way(fc1r3, AIdea 最佳底 0.6170),預期更高。clarity_test_probs.npz 已存,免重訓。

## 抗過擬合微迭代結果（2026-06-14 14:3x）—— 兩個都不採用
- **軟混合(soft blend β·clarity+base)否決**:全輸 base 0.67882(0.671~0.675)。全域混合把對的 Clear/NA 也拉偏;hard-replace 的「只 override 高信心列」選擇性才對。(soft_blend_eval.py)
- **Label smoothing(ε=0.1)壞掉**:與 balanced-softmax 不相容——Misleading 先驗 0.0015→logprior −6.5,LS 的均勻 floor 逼模型把 Misleading logit 對全部樣本灌爆→推論退化成全 Misleading,heldout acc 0.000、overlay 崩到 0.537。**非「LS 無用」,是設定衝突**;若要測須改成 plain CE+LS 或排除 Misleading 先驗。
- **結論**:clarity 頭維持原版(balanced-softmax,無 smoothing、hard-replace c≥0.7),已是確認最佳。**clarity-on-12way AIdea 0.6159 = 抗過擬合的最終候選**。

- **2-way+LS clarity 頭(丟 Misleading、label smoothing 修復)**:valid 平手原 3-way(c>=0.8 = 0.68059, Pub 0.63678 略高於原 0.63516)。更乾淨/抗過擬合但無明確勝出(雜訊內)。→ clarity overlay 槓桿已徹底探索,確認小贏(+0.0019 AIdea)。

## ★★ Stacked conditional heads（2026-06-14 14:51）— 目前最強
- **T2 evidence head（cond on T1=Yes, balanced-softmax+LS）conf>=0.8**: valid 0.68082, T2 0.6816→0.6891（改 3 列）。
- **T2+T3 stacked overlay**: valid **0.68274**（Pub **0.64086**, +0.0039 總/+0.0120 Public）。bootstrap：ALL +0.0037 P(>0)=0.72；**Public +0.0118 P(>0)=0.93**。兩個 conditional head 加性疊加,robust 12-way 底,非追 3-way。**最強候選 → AIdea probe**。
- 機制:conditional-head 做法在 T3 與 T2 都贏(各 ~+0.002),不同任務疊加。
- 待辦:t2_aidea.py（T2 head on retrain→test）→ combine 兩 overlay → aidea_stacked.csv → 上傳 slot 2。

## ✗ Stacked T2+T3 AIdea probe = T2 head 過擬合（2026-06-14 15:03）
- aidea_stacked → **AIdea 0.6146 < clarity-only 0.6159**。T2 0.6982→0.6972（AIdea 降）、T3 0.4449→0.4421（T2→No cascade 吃掉 clarity 增益）。
- **T2 head 過擬合 valid**：valid +0.0075（僅改 3 列、bootstrap CI 跨 0 P=0.72）→ AIdea −0.001。三層協議抓到：過 Layer2 勉強、死 Layer3。
- **對照**：T3 clarity overlay valid+AIdea 都正（+0.0019 確認轉移）。差別=T3 改 19 列真實 recall 改善 vs T2 改 3 列雜訊。
- **教訓**：thin valid 訊號（少數列改動、CI 跨 0）即使 Public lean 也別信，需 Layer3 確認。
- **最佳確認候選不變 = clarity-on-12way（T3-only）AIdea 0.6159**。T2 head 丟棄。standing 現為 stacked 0.6146（只 06-17 最後一筆重要）。

## ★★★ 公司先驗 = 第一個統計顯著突破（2026-06-14，主辦方來信啟發）
- 主辦方來信:test 提供 company/esg_type/page_number;切分=按公司+四任務選項平均抽樣(50 家全在 train+test);Misleading 人工標註本就極少(train 1 個→test 也極少);**老師 LGBM 加公司id+產業 0.58→0.61,尤其 T4(金融→already)**。
- 驗證:公司先驗「軟 blend」助 T2(β0.2 +0.0035)、T4(β0.4,F1 0.612→0.634)、T3(+0.0015);T1 無。
- **完整 stack 12w+kNN+clarity(T3)+公司blend(T2,T4) = valid 0.68721**(+0.0084 vs 0.67882)。bootstrap vs clarity-only:**ALL +0.0065 CI[+0.0015,+0.0139] P=0.99(排除0,首次顯著!)**;Public +0.0036 P=0.96。
- 公司=結構性特徵(全公司在 train+test,stratified)→ 應轉移,非薄訊號過擬合。組織者獨立佐證。
- 待辦:用 retrain 公司先驗建 AIdea company-stack 檔 → probe。

## 穩健性:company vs industry prior（2026-06-14）
- company 0.68680(P=0.99顯著) > hierarchical 0.68510(P=0.91) > industry 0.68185(P=0.60不顯著) > no-prior 0.68059。
- **用 company 先驗**:50 家全在 train+test(stratified)→ 結構性、可轉移、非過擬合;pooling 成 industry 丟失公司特異訊號反而變弱。修正 Wistron/wistron 大小寫。
- Kaggle final-stage test = valid 399(submission_sample 399 行)→ Kaggle 僅能確認本地計分無 bug,泛化只能靠 AIdea 2000。

## PDF 頁面 context 對齊 + 測試（2026-06-14）— 負面,關閉
- 用 pdftotext 重抽 49 份本地 3rd_data/esg-pdf(pdf_url 已失效)→ agent_cache/pdf_pages_3rd_norm.json,(company,page)±1 對齊 ~70%(cache 原只 51%,且缺 ctbc/yfy;本地 PDF 較全)。tsmc/tcc 等版本不符 ~42-47%。
- 測 T4 page-context 年份信號:真值未來類樣本上 implied 命中僅 41%(~隨機)。根因:ESG 報告到處是年份(2030 SDG/2050 淨零),頁面年份雜訊大,不及聚焦片段。
- **結論:PDF 頁面 context 無乾淨正交訊號 → 關閉。** company metadata 仍是唯一正交顯著槓桿(已捕捉)。

## company-stack 驗證收口（2026-06-14）
- nested split(valid 切兩半,A調β→B測,B調β→A測):兩方向 held-out 都正(+0.0016, +0.0054),β 穩定(T4=0.4)→ **β 不過擬合、泛化**。
- 三重驗證:bootstrap P=0.99 + nested 雙向正 + company 先驗來自 train(結構性)→ 高信心轉移。
- Kaggle 無憑證無法自動傳;且 Kaggle=valid 399 非獨立。nested 才是正解,已過。
- 最終 transfer 唯一測試 = 明天 AIdea probe aidea_company_stack.csv。

## Kaggle 官方 scorer 確認 company-stack（2026-06-14）
- 提交 company_stack(N/A→-1)→ **Kaggle public 0.63877** = 本地 valid-Public 0.63878(吻合小數第4位)→ pipeline 計分零 bug。
- vs 之前最佳 12-way(s5a3_all12_knn)Kaggle public 0.62880 → **company_stack +0.00997**(官方 scorer 確認改善真實)。
- ⚠️ **格式**:Kaggle N/A 編碼=`-1`(AIdea=`"N/A"`)。第一次提交因 "N/A" 被 scorer ERROR;轉 -1 後 COMPLETE。AIdea 檔用 "N/A" 正確不改。kaggle 提交檔=official_sub/kaggle_company_stack_v2.csv,slug=final-stage-2026-esg-classification-challenge,認證已設(kaggle auth login)。
- Kaggle=valid-Public(調β同份)→ 確認計分+改善,非獨立泛化(nested 已證泛化)。真實 transfer 仍待 AIdea 2000。

## company 利用方式定論（2026-06-14 deep research + A/B）
- **marginal prior 晚期融合(現用)= 最佳**:+0.0084 valid 顯著、Kaggle public +0.01 確認。
- EB-shrinkage(λ=n/(n+m)):無增益(公司樣本數相近,固定 +0.5 平滑已最優)。
- **learned company embedding(端到端,A/B 對照):更差** noCOMP 0.65539 → withCOMP 0.64978(-0.0056,T4 也降)→ 50 家×小資料聯合學過擬合。
- 鐵律重申:小資料(1601)簡單統計/晚期融合 > 可學複雜度(同 SupCon/meta/翻譯增強)。**不做 company-emb 重訓。** company 訊號已用 marginal prior 榨到最佳。

## ✗✗ company_stack AIdea probe 失敗（2026-06-15 01:36）— 重大教訓
- aidea_company_stack → **AIdea 0.6126(23/111)< clarity-12w 0.6159 < 12-way 0.6140**。逐任務:T3 0.4396→0.446(clarity 轉移✓)、T2 0.6982→0.6968(-0.0014)、**T4 0.6238→0.6023(-0.0215,company T4 崩)**。
- **company T4 valid +0.022 → AIdea -0.0215 正負翻轉**:company 先驗不轉移到 AIdea(train 的公司→T4 關聯在 AIdea 不成立)。
- **教訓:valid bootstrap P=0.99 + nested split + Kaggle 全沒抓到**(皆同 valid/train 分布);AIdea 2000 是不同分布。**valid 訊號打折、AIdea LB 加權。**
- **clarity(T3)轉移、company 不轉移。** 最佳 AIdea 仍是 3-way 0.6170;clarity-on-3way 待測(aidea_clarity_on_fc1r3.csv)。standing 現為 company_stack 0.6126,需換回最佳。

## 架構審查（2026-06-15,跨領域 ESG+ML 專家）
- 🔴 **CRITICAL#1 Misleading 策略+工具都錯**:macro-F1 稀有類結構獎勵「中等召回」(test ~2 Mis、其他類上百→抓 1-2 真 Mis 即使 8-10 假陽性,Mis-F1 大漲、大類幾乎不損)。Mis-F1 0.2→總分 +0.015、0.33→+0.029(進 top-3)。我們做反了:**保守高精度(0 fire)+ 只試 Qwen3-14B 簡單 judge**。唯一進獎金圈的槓桿,嚴重低度開發。前提:偵測器需對真 Mis 非隨機(Qwen3 簡單版=隨機)。
- 🔴 **CRITICAL#2 驗證無法偵測分布偏移**:valid=train 同分布 reshuffle,company 因此騙過 bootstrap/nested/Kaggle。應建 company/report-holdout 驗證。
- 🟡 max_len 384 截斷 5%(→512 便宜);cascade 事後硬覆寫傳播誤差;未試領域/更大 backbone;T3 頭混structural-N/A。
- ESG 視角:Misleading=漂綠有已知標記(模糊無量化/法定義務當成就/無第三方/cherry-pick)→ 應建漂綠標記感知偵測器。
- **行動 #1**:前沿級 Misleading 偵測器(漂綠 CoT+ICL+self-consistency+中等召回校準)。第一步=panel 鑑別力測試。無前沿 API→Qwen3-14B 強化版(VRAM:14B bf16 ~28GB/32GB,batch1 序列,prompt 上限,max_new 256)。

## Misleading 徹底關閉（2026-06-15,5 方法 + 前沿模型確認）
- Qwen3 漂綠 CoT+self-consistency panel:Mis 均分 4.0 < NotClear 5.93 < Clear 4.8;2 個真 Mis 排 27-28/32(墊底=反向鑑別力)。
- Claude 親手盲判 11 段:無法分離 Mis 與 NotClear;id11836 內容看似實質卻標 Mis=鐵證(標籤非內容可導)。
- 5 方法全敗→ Misleading 無法從片段推導,連前沿 LLM 都不行。#1 槓桿正式關閉。

## ★ max_length 384→512 = 結構性槓桿（2026-06-15）
- RoBERTa max_position_embeddings=512=架構硬上限(超過需換 backbone,而長文 backbone 我們已證較弱)。所以 512=max。
- **FC1R512 seed42 單模型 valid 0.65853 vs FC1R@384 0.63957 = +0.019**(T1 .774→.788, T2 .671→.674, T3 .566→.595 +0.028, T4 .568→.605 +0.037)。長文本結尾的證據/時間軸被救回。
- 結構性改動→應轉移(不像 company)。進行中:FC1R512 s1/s2 訓練 → 3-seed valid 比較(compare_512.py)→ 若 @512>@384 則重訓 RT_FC1R×3@512 → 3-way@512 AIdea probe(對抗 champion 0.6170)。
- VRAM:RoBERTa-large @512 bs4 ≈12GB,輕鬆。

## ★ 過夜全 12-way @512 流水線(2026-06-15 ~13:00 啟動)
全部一次跑完:1600(final_data,valid)+ 2000(retrain_data,AIdea)的全 12-way @512 + 推論 + 生成提交檔。
連鎖(單 GPU 連續跑):
1. **FC1R512 ×3**(final_data)PID 397382 → ~14:10 → monitor bu7lmdh1f 跑 compare_512(3-way@512 vs @384 首訊號)
2. **FC1/FC1S/FC1M ×3 @512**(final_data)PID 408854,等①→ ~21:30 → monitor 420073 跑 **compare_all_512.py**(全 combo valid 比較,選最佳)
3. **RT 全 12-way @512**(retrain_data=2000)PID 418836,等②→ ~06-16 中午 → 自動跑 **gen_rt512.py** 生成所有 AIdea 候選:
   - official_sub/aidea_rt512_{3,6,9,12}way_knn.csv + {3way,12way}_knn_clarity.csv
配方:FC1=base / FC1R=rdrop0.5+swa7 / FC1S=sharp_recl0.10 / FC1M=mr2 0.05;seeds 42/123/456;bs8(OOM 退4)。
判讀:valid(本地399)選候選 → AIdea(2000)確認 transfer;不被 Kaggle 100%-public(~0.68)數字干擾。對抗 champion 3-way@384 AIdea 0.6170。

## ★ 收官 6-submission 計畫(2026-06-16,2天6次含最後一次)
AIdea = 唯一仲裁;最後一次上傳 = Private 最終。廣測 @512 方向,順序守紀律。
Round 1(06-16,3 probes):
  1. aidea_swapT3_512_knn_clarity.csv      (champion+T3@512;valid0.67214,bootstrap P=0.990,結構性)← 最高 EV
  2. aidea_hybrid_t14_384_t23_512_knn_clarity.csv (T2+T3@512;valid0.67286,P=0.897)
  3. aidea_rt512_3way_knn_clarity.csv       (全部@512;valid0.66512)← 「全換512」對照
Round 2(06-17,3 含最後):
  4. adaptive(依 R1 AIdea 結果:贏家方向再優化,如 6way@512)
  5. buffer / 微調
  6. 最後一次 = AIdea 已確認最高分檔(地板 = champion aidea_clarity_on_fc1r3.csv 0.6188)
低優先(雙重指向更差,不佔核心槽):aidea_rt512_{9,12}way_knn_clarity.csv
判讀:beat 0.6188 才採;不被 Kaggle 100%-public(~0.68)干擾;valid 只當 prior,AIdea 定生死。
檔案全部 ~05:30 由 gen_rt512 + gen_hybrid_aidea 自動生成。

## ★ 修正計畫(2026-06-16):用 AIdea 子任務分數,1 次 harvest(隊友提點)
AIdea 每次提交顯示四子任務 F1。故:
1. **先傳 pure @512(aidea_rt512_3way_knn_clarity.csv)→ 一次拿到 T1/T2/T3/T4 的 @512 AIdea 分數**。
2. 對照 champion 已知 @384 子任務:T1 0.7967 / T2 0.701 / T3 0.4522 / T4 0.6058。
3. 逐任務取較高來源 → `python3 gen_mix.py <用512的任務,如 t3 或 t2,t3>` 組最佳 mix → 上傳確認總分。
4. 最後一次交最佳(地板 champion 0.6188)。
caveat:子任務 F1 在各自 N/A 級聯下算(T1 遮罩 T2/T3/T4);T1@384/@512 遮罩高度相似 → 二階效應,故 mix 後「確認一次」。
工具:gen_mix.py(任意逐任務 384/512 mix,共用 NC bias+kNN α0.4+clarity)。pure@512 由 gen_rt512 產;swap-T3/hybrid 由 gen_hybrid_aidea 產(備用)。
