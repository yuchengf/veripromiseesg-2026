# CHAMPION (frozen fallback) — Final Stage AIdea

## Scoring rule (§1.1, re-confirmed)
- AIdea keeps **LAST** upload; final rank = **Private** (public ~ approximates).
- Deadline 06-17. We are rank ~23/111, NOT in contention for top-3 (cutoff 0.6242).

## Current CHAMPION (frozen, never overwrite)
- **3-way: `official_sub/aidea_rt_fc1r3_knn.csv` — AIdea 0.6170501** (highest confirmed)
- = FC1R ×3 seed + Qwen3 kNN (α3=0.4, NC bias 0.1). Best val→LB transfer ratio (0.932).

## val→LB ratio table (§2.8 diagnostic; valid399 → AIdea2000)
| submission | valid | AIdea | ratio | verdict |
|---|---|---|---|---|
| 3-way fc1r3 | 0.66206 | 0.6170 | 0.932 | ↑ transfers best (champion) |
| 9-way+kNN | 0.67481 | 0.6140 | 0.910 | stable |
| 12-way+kNN | 0.67882 | 0.6140 | 0.905 | stable (reference) |
| clarity-on-12w | 0.68059 | 0.6159 | 0.905 | structural, transfers ✓ |
| company_stack | 0.68721 | 0.6126 | 0.891 | ↓ BROKEN = overfit ✗ DROP |
| 6-way+kNN | 0.67334 | 0.6081 | 0.903 | — |

## Structural vs val-tuned (§2.8) — which levers transfer
- **clarity T3 overlay = STRUCTURAL** (dedicated NotClear classifier) → AIdea-confirmed +0.0053 T3 → KEEP.
- **company prior (T2/T4) = VAL-TUNED per-entity** → broke ratio, AIdea −0.0215 T4 → DROP.

## Endgame plan (§4)
1. Frozen fallback = 3-way 0.6170 (above). Never overwrite.
2. Calculated bold swing (ONE lever, §4.4): **clarity-on-3way** = `official_sub/aidea_clarity_on_fc1r3.csv` (clarity T3 overlay on 3-way base; changes only 23 T3 rows). Expected ~0.619.
3. Not in contention → §4.3 take the swing WITH fallback.
4. 06-17 LAST upload = whichever of {clarity-on-3way, 3-way} scores higher on AIdea.
5. DO NOT submit company anything. DO NOT bundle changes (§2.2).

## ⚠️ 共用帳號協調危機（2026-06-15）
- 隊友上傳的更差模型會蓋掉 standing(LAST upload counts):
  - 06-14 LGBM companyTE 0.5919 / 06-15 02:58 R-Drop+BGE public_top **0.5713(77名)**。
  - 都遠低於我們 champion 3-way 0.6170(差 0.04-0.05)。BGE backbone 我們已否決(< RoBERTa);public_top=public 過擬合。
- **鐵律(§1.1/§4.2)**:收盤 06-17 23:59 前,帳號最後一筆 = 我們的 champion(3-way 0.6170 或確認後 clarity-on-3way)。
- **行動**:① 立即重傳 aidea_rt_fc1r3_knn.csv 救 standing;② 與隊友協調額度,別再覆蓋;③ 團隊收斂到我們的 RoBERTa champion,勿併入 BGE 弱模型(跨 backbone 0-11、弱模型拖累投票)。

## 修正:standing 紀律（2026-06-15,使用者澄清）
- **只有 06-17 收盤前「最後一筆」重要**;中途 standing 是什麼都無所謂(隊友的 0.5713 不用急著蓋)。
- 含意:06-15/16 **自由用額度 probe 候選**(clarity-on-3way 等),確定最強檔。
- **唯一硬性要求**:06-17 23:59 前,帳號**最後一個動作** = 上傳我們確認的最佳檔,且**之後隊友不可再傳**(只需協調收盤那一刻)。

## ⚠️ Kaggle 改 100% public（2026-06-15）— 避免誤判
- Kaggle final-stage 改成 100% public = 全 valid 399(不再是 Public 199 子集)。
- 含意:Kaggle 數字會從 ~0.63(舊 Public-199)跳到 ~0.68(全 399,因 Private-200 子集分數高)。**這是計分範圍變,不是模型進步!**
- 舊對應「Kaggle 0.628≈valid-Public」作廢。新:Kaggle(100%)= 本地全 valid 399(我們有標籤→仍只是 bug-check,無新泛化資訊)。
- **真實 transfer 唯一看 AIdea 2000(不受影響)**;本地 valid 399 為選擇主指標。比較永遠 like-for-like。

## ★★ 新 CHAMPION(2026-06-15 12:46)= clarity-on-3way @384
- **`official_sub/aidea_clarity_on_fc1r3.csv` = AIdea 0.6188030,rank 7/113**(舊 3-way 0.6170 → +0.0018)。
- 分項:T1 0.7967 / T2 0.701 / **T3 0.4522**(clarity 再升)/ T4 0.6058。
- = champion 3-way + clarity overlay(僅改 23 列 T3)。**clarity-on-3way 確認轉移!**
- **凍結 fallback 升級為 0.6188**(若 @512 明天打不過就交這個)。
- **明天 @512 目標:beat 0.6188**;@512 版要套 clarity(gen_rt512 已自動產 _clarity)。距 prize zone(top-3)很近(目前 7)。

## @512 全 combo valid 結論(2026-06-15 18:38)+ HYBRID 槓桿
- 純 combo:3way@384+clarity=0.66532 ≈ 3way@512+clarity=0.66512;**12way@512+clarity=0.66057 更差**(額外 recipe@512 拉低 T3)。→ 12-way@512 不如 3-way,已驗證。
- **★ HYBRID(T1/T4@384 + T2/T3@512)+clarity = valid 0.67286**,贏所有純 combo +0.0075。結構理由:@512 減截斷強 T3/T2,@384 對短訊號 T1/T4 已足。
- AIdea 計畫:RT@512 完成後 gen_hybrid_aidea.py 產 `aidea_hybrid_t14_384_t23_512_knn_clarity.csv` → 06-16 probe(beat 0.6188;若 0.93 ratio 轉移→~0.6258 可能 top-3)。注意:hybrid 是 valid-selected,須 AIdea 確認 transfer(company 教訓)。

## ⚠️ HYBRID robustness 檢驗 = FAIL(2026-06-16 00:30)
對「T1/T4@384 + T2/T3@512 +clarity valid 0.67286」做穩健性檢驗,結論:**不 robust,不採為最終**。
- CHECK1 per-seed:T1 @384 只贏 1/3 seed(@512 贏 2/3!)→ hybrid 選 @384 是 seed2 離群撐的,機制故事對 T1 被打臉。T4 @384 贏 2/3 但邊際 0.001。
- CHECK3 bootstrap(同一 valid):diff +0.0077,95%CI [−0.0046,+0.0199] **跨 0**,P=0.888 < company-stack 當時 0.99(且 company 還是破了)。
- 疊加 selection bias(per-task argmax 挑在同一 valid 上 → 0.67286 灌水)。
- **裁決**:@512 重訓無 robust 改進。12way@512 更差、hybrid 噪音、純 3way@512≈champion。**最終鎖 champion 0.6188**。06-16 可低期望 probe hybrid 純實驗,AIdea 贏才採用。

## ✅ 修正:swap-T3-only @512 才是 robust 槓桿(2026-06-16 00:38)
**上一條「HYBRID FAIL」的分析有誤——測錯任務了。** champion 與 hybrid 在 T1/T4 都用 @384,差異只在 T2/T3;我卻拿 T1/T4 的 seed 不一致去殺 hybrid(紅鯡魚)。正確拆解:
- **swap-T3-only(champion + T3 換 @512):valid 0.67214,bootstrap mean +0.0069,CI [+0.0009,+0.0139] 不含 0,P=0.990** ✓ robust。
- swap-T2-only:+0.0018,P=0.661,噪音(seed2 反轉 −0.036)。
- 全 hybrid(T2+T3):0.67286 但 P=0.897 CI 跨 0(被噪音 T2 拉低)。
- **結構理由**:T3=evidence_quality 直接吃被 384 截掉的證據文字 → @512 少截斷 → 該轉移;且正打 champion 最弱最高權的 T3(AIdea 0.4522,w=0.35)。不同於 company(valid-specific 捷徑)。
- **決策**:06-16 主推 probe `aidea_swapT3_512_knn_clarity.csv`(只比 champion 多換 T3 一欄);全 hybrid 當對照。AIdea 贏 0.6188 才採為 06-17 最終;否則回 champion。

## ⚠️ 重建漂移 bug + 修正(2026-06-16 11:45)
- verify 抓到:gen_hybrid/gen_mix 從 rt_fc1r 快取重建的「全@384」≠ champion 檔(全四欄都差:T1:6 T2:17 T3:18 T4:36)。即 rt_fc1r 快取 ≠ 當初建 champion 的模型/pipeline。
- 後果:那批 aidea_swapT3_*/aidea_hybrid_*(非錨定)其實偷改了 T1/T2/T4 → 不是乾淨的單變數對照。**不要上傳那批。**
- 修正:`build_mix_from_champion.py` — @384 任務鎖定 champion 標籤,只覆蓋 @512 任務 + 重跑級聯。驗證:none=0差異✓;t3=只差 evidence_quality 26 列✓。
- 乾淨候選:`aidea_anchored_t3512.csv`(=champion+只換 T3@512,26 列)。pure@512 `aidea_rt512_3way_knn_clarity.csv` 仍可上傳(它是自洽全@512,用來 harvest 子任務分數)。

## ★★★ AIdea harvest 顛覆 valid(2026-06-16 12:00)— pure@512 已是新最佳
pure @512(aidea_rt512_3way_knn_clarity.csv)= **AIdea 0.6221564,rank 10/122**(> champion 0.6188,+0.0034)。
逐任務 @512 vs @384(AIdea 實測):
- T1 0.796 vs 0.7967(平手)| T2 **0.7095** vs 0.701(@512+0.0085)| T3 0.4457 vs **0.4522**(@384 勝!)| T4 **0.6274** vs 0.6058(@512 +0.0216!)
- **valid 完全反轉**:valid 說 T3@512 最該換、T4@512 最不該換;AIdea 實測相反(T3 留 384、T4 換 512)。「T3@512 結構性轉移」假設被 AIdea 否決。
- **新 AIdea 地板 = pure@512 0.6222**(取代 0.6188)。
- 下一步:AIdea-最佳 mix = T1@384 + T2@512 + T3@384 + T4@512(aidea_anchored_t2_t4512.csv),naive 估 ~0.6246,待上傳確認。
教訓:per-task 該換哪個 max_length,只有 AIdea harvest 看得出,valid/結構推理都會錯。

## ★ 決定性紀律(2026-06-16):AIdea 有 private 分割 → 拒絕 public 最大化
使用者確認:AIdea 顯示分數=public,最終排名用看不到的 private 子集。故 per-task 在 public 挑贏家 = overfit private 的風險。
防線=只採用「獨立樣本(valid 跨配方)+ public 都重現」的 convergent swap:
- **T2@512:convergent(public +0.0085 + valid 3/4 配方)→ 真實,納入。**
- T4@512:public +0.0216 但 FC1R valid −0.0211、跨配方僅 2/4 → divergent=public-only=不納入(pure@512 的進步大半來自此,private 恐回吐)。
- T3@512:public −0.0065、valid +0.0195 也 divergent → 留 champion @384(已 AIdea 確認 0.4522)。
**最終決定 = swap-T2-only `aidea_anchored_t2512.csv`**(champion + 唯一 convergent 的 T2@512,public-fit 自由度=0)。私榜期望 ~0.6214 > champion。地板 = champion 0.6188。
鐵則升級:AIdea public 分數高 ≠ private 好;final 只押 convergent 證據,不押 public 最大值。pure@512(public 0.6222)不可當 final(含可疑 T4)。

## ★★★ 最終決定(2026-06-17)= mix `aidea_anchored_t2_t4512.csv`
public 0.6232408(T1 0.7967 / T2 0.7057 / T3 0.4522 / T4 0.6261)。
- T2@512:convergent(valid 3/4 + public)→ 真實。
- T4@512:public(=test 分布)量到 +0.0203 → 應帶到 private(同分布);valid 的 −0.0211 是 train 分布、已證不可靠 → 不該否決。
- 修正前一手過度保守的 swap-T2(0.6202,把 T4@512 的好處丟掉了)。
- 行動:用最後 slot 把 `aidea_anchored_t2_t4512.csv` 補成最後一次上傳 = private 最終,然後停手(別 panic-chase 虛胖 public 榜)。
教訓已寫入 ml-competition skill §2.9(root cause #2:representativeness 先於 convergent veto)。

## ★★★★ 新最佳(2026-06-17 22:05)= joint decode `aidea_joint_w2.0.csv` 0.6249259
- 結構化/聯合解碼(用子任務相依關係的「後向」:下游 N/A 機率修正 T1/T2 gate),wgate=2.0。
- 同一批模型機率(mix 的 T1/T3@384+T2/T4@512+kNN+clarity),**只改解碼方式**(貪婪前向→聯合)。
- AIdea: T1 0.8008 / T2 0.7072 / T3 0.453 / T4 0.6271 = 0.6249(> mix 0.6232 +0.0017,四任務全升)。
- valid +0.0114(P=0.971)→ AIdea +0.0017:結構性槓桿轉移成功(第一個過 valid 又轉移的新法)。
- **新 floor = 0.6249**。floor 升級。產生器 gen_joint_aidea.py(wgate 可調)。
- 唯一調參風險=wgate(有 class-count 平衡的原則依據,且 w∈[1,3] 都正)。

## LLM gate-judge (T1/T2) 否決(2026-06-17 22:30)
Qwen3-14B 判 promise/evidence:promise 準確率 0.767 < 全-Yes 基線 0.812(over-flag No,與 LLM-T3 同失敗模式)。blend 進 gate→joint:β0.2 Δ−0.00002、β0.5 Δ−0.017,全部 ≤base,P≤0.51。模型 gate 已優於 14B LLM。→ 否決。floor 0.6249 不動。
資料血緣已驗:data 06-11/12 穩定,所有 cache/model 在其後,無 stale;SupCon 在新設計重驗仍 reject。

## 條件鏈式 aux (conditional-softmax cascade, arXiv 2410.01305) 否決(2026-06-18 00:23)
1-vs-1 公平比較(隔離 aux):COND_s0 T3=0.5935 vs FC1R_s0 T3=0.5932 → **ΔT3=+0.0002(噪音),P=0.522**。GATE FAIL。
train-time chain-rule(T3 content 只在 T2=Yes 訓練)對 T3 分離度零幫助 → **T3 label-limited 第 6 個獨立角度確認**(LLM-14B/domain/backbone/kNN/logit-adj/conditional-aux 全失敗)。
**最終鎖定 = joint decode `aidea_joint_w2.0.csv` 0.6249**(結構性、AIdea 確認、四任務全升)。已是最後上傳。剩餘時間轉報告。
