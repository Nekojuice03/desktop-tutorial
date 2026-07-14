# 實驗章節大綱(圖表清單 × 主張 × 寫作指引)

> 依據 2026-07 和平東路 SUMO 正式結果整理。每小節列出:用哪張圖/表、
> 圖說怎麼寫、這張圖支撐的「主張(claim)」、以及要避免的過度宣稱。

---

## 5.1 實驗設定

**內容**:場景(和平東路二段/新生南路,876×867m,3 號誌路口,8 RSU:3路口×2角+路肩2,覆蓋92%)、
真實車流(台北交控 VD 路段級 TotalVol × 小客車代理比例 0.75,僅小型客車)、
參數表(引 PROGRESS.md §3)、對照組(Local/RSU/Cloud-only、Random、Greedy)、
指標(成功率、vision-only、延遲、能耗、成本、移動性事件)、
訓練協定(replay 快照依時間 70/15/15 切分，train 訓練、validation 選模、test
只做最終報告；150 iter、LR 線性衰減、固定 EVAL_SEEDS、n=3 seeds)。
三個號誌採固定時制，號誌控制不是本研究的學習變數。

**圖**:`rsu_layout.png`(RSU 佈點)、README 架構圖。
**必寫的假設聲明**(§8 of PROGRESS):無線無競爭(SNR 非 SINR)、任務原子、
轉向均分、確定性通道、對向鏡射假設、抵達補送之「結果仍具價值」假設。

## 5.2 主結果:與基準方法對照

**圖**:`fig_latency.png`、`fig_energy.png`、`fig_cost.png`、`fig_success.png` + 彙總表。

**Claims**:
1. 卸載必要性:Local-only 延遲 3865ms,重任務弱車不可行。
2. **主主張:MAPPO 與最強基準(Greedy)成功率/延遲相當(731 vs 686ms),
   能耗 −34%(3.02 vs 4.57J)、使用成本 −93%(0.012 vs 0.171)。**
3. Greedy 需讀取完整成本模型真值(現實不可得),MAPPO 僅需局部觀測 → 可部署性論證。
4. 反雲三機制有效:Cloud-only 能耗/成本最高,學到的策略 0% 上雲。

**避免**:不要宣稱延遲/成功率「贏過」Greedy(沒有);賣點是同水準下的資源效率。

## 5.3 收斂分析

**圖**:`fig_convergence_metrics.png`(2×3 多指標面板)。

**Claims**:~27 iter 收斂;**vision-only 51→74%(+23pp)= RL 學會重任務分流的直接證據**;
延遲 1030→710ms;能耗 2.66→3.02J 上升為「花能耗買 deadline」的**刻意權衡**(主動解釋);
成本 0.054→0.01 再微調 = 先避費再平衡強車競爭。

## 5.4 移動性機制消融(本文核心貢獻)

**圖**:`fig_ablation.png`(8 組合:預判{linear,kalman,route}×恢復{fail,v2i,v2i+arr})。

**Claims**(全部鎖定右圖事件層):
1. **發現:都市場景移動性失敗以「任務發起者抵達離場」為主(6/8=75%)**,
   而非傳統假設的通訊距離斷裂 → 挑戰 sojourn-only 建模的文獻慣例。
2. 意圖分享預判(route)事前迴避 13 件、斷線 8→6、離場 6→4;
   kalman/linear 擋不到離場(無運動學前兆)—— 誠實極限,kalman 價值見 5.5。
3. 純 V2I 遷移對離場無效(救回 0)→ 動機化「抵達補送」。
4. **抵達補送救回 50% 離場案例(3/6),上限受路側覆蓋率(92%)制約** →
   連回 RSU 佈點設計。

**避免**:左圖成功率八組同水準(誤差棒重疊)—— 不得在成功率上宣稱消融差異。

## 5.5 數位孿生同步延遲(DT 的量化貢獻)

**圖**:`fig_dt_delay.png` + 建議補一張 pred_reject 對數座標圖(數據在
`dt_delay_results.json`)。

**Claims**:
1. 孿生時效性影響決策品質:τ↑ → linear 預判劣化(過度保守,誤殺↑、成功率↓最快)。
2. **kalman(EKF-CTRV + dead-reckoning 前推)全 τ 優於 linear** —— 「DT 提供
   未來狀態」的量化證據,回答「DT 與普通模擬之差」。
3. 反直覺點先講:linear τ=8 斷線反降是保守化假象(犧牲卸載機會)。

**Caveat**:若此圖為場景錯配期所跑,須重跑後再定稿。

## 5.6 統計嚴謹性與局部 critic 消融

**表**:多 seed(n=3):成功率 88.61±0.05%、延遲 707.7±1.7ms、能耗 3.007±0.009J、
成本 0.0101±0.0018;歷史 LocalCriticPPO 88.72±0.09%（重構後須重跑）。

**Claims**:
1. 高度可重現(std 千分位);主結果非幸運抽樣。
2. **LocalCriticPPO≈MAPPO(不顯著)**:以 Lyu et al.(JAIR 76, 2023)理論解釋 ——
   協調資訊(佇列/回程)局部可觀測 → 中央 critic 無額外資訊;
   對照 VNC 2025(arXiv:2505.03558)耦合強時 MAPPO 佔優 → CTDE 價值場景相依。
   引用組合:Yu 2022(實證)+ Lyu 2023(理論)+ Amato 2024(專論)+ VNC 2025(領域)。
   注意：本實作是共享 actor + tick-level 團隊 advantage 的局部 critic 資訊消融，
   不是逐代理 reward/value/GAE 的標準 IPPO，不得以 IPPO 名義做強結論。
3. 策略族一致:六次訓練皆收斂至 strong 70–86%+RSU 其餘,策略穩定;
   strong↔RSU 間存在近等值平坦谷(成本隨 RSU 佔比 20→30% 由 0.009→0.013)。

## 5.7 決策空間分析(動機/討論用,可放 5.2 之前)

**圖**:`fig_crossover_distance.png`、`fig_crossover_load.png`、`fig_tradeoff_vision.png`
(oracle 單任務分析,mock,圖說標明 mechanism illustration)。

**Claims**:V2V 強車勝出至 ~150m;RSU 排 1 件即被雲反超;重任務「最快≠最划算」
(雲最快但能耗/成本壓垮綜合分)→ 智慧分流的必要性。

## 5.8 可視化(口試/demo)

**圖**:`dt_snapshot.png`(SUMO 版,場景對齊後重產)+ `dt_visualization.gif`。
孿生殘影(twin ghost)疊加層 = 同步誤差的直觀展示,與 5.5 呼應。
mock 圖一律標註 synthetic corridor。

---

## 收尾檢查清單
- [ ] 確認 8 組合消融是在「場景對齊後」跑的(osm.sumocfg=hepingeast2);否則重跑
- [ ] `run_dt_delay.py --sumo --plot` 對齊後重跑一次定稿
- [ ] `visualize_dt.py --sumo --tau 2 --ticks 200` 產正式版 snapshot/GIF
- [ ] pred_reject 對數圖(從 dt_delay_results.json 補畫)
- [ ] 所有 mock 來源圖標註 synthetic;SUMO 圖標註場景與資料日期
