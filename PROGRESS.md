# 專案進度與架構總覽（PROGRESS.md）

> 新生南路數位孿生 — 車聯網運算卸載（MARL / MAPPO），SUMO + 真實 VD 流量。
> 分支：`claude/sumo-twin-offloading-debug-17iut1`
> 最後更新：見 git log。本檔為專案「單一事實來源」，換 session/換電腦先讀這份。

---

## 1. 專案在做什麼

車輛產生運算任務（感測 / 導航 / 影像），需決定卸載到哪一層執行，
目標：在 **延遲、能耗、使用成本、deadline、移動性** 多重限制下最大化任務成功率。

- **三層架構**：車輛(本地/V2V) → 邊緣(RSU) → 雲端
- **方法**：MAPPO（CTDE 多智能體 PPO，共享 actor + 中央 critic）
- **對照組**：Local-only / RSU-only / Cloud-only / Random / Greedy(就近最快)
- **動作(MA, 5 個)**：`local / v2v_strong / v2v_near / rsu / cloud`
- **真實性來源**：SUMO 真實路網 + 台北 VD 偵測器流量 + calibrator 校正

---

## 2. 模型摘要（公式詳見 COMM_MODEL.md）

### 通訊（comm_model.py）— C-V2X / 3GPP 鏈路預算
```
路徑損耗 PL(d) = FSPL(d0) + 10·n·log10(d/d0)      (5.9GHz，確定性)
接收功率 Prx[dBm] = Ptx − PL ；雜訊 N = −174 + 10log10(B) + NF
速率 rate = B · log2(1 + 10^((Prx−N)/10))         (Shannon)
```
| 鏈路 | 標準 | 頻寬 | 覆蓋 | 路徑損耗指數 |
|---|---|---|---|---|
| V2V | C-V2X PC5 | 10 MHz | 150 m | 3.0 |
| V2I | C-V2X Uu | 20 MHz | 200 m | 2.7 |
| 回程 RSU↔雲 | 有線骨幹 | 100 Mbps 共享(會壅塞) | — | — |

發射 23 dBm、雜訊 −174 dBm/Hz、NF 9 dB。

### 延遲與能耗（nodes.py estimate）
```
總延遲 = 上行傳輸 + 上行存取 + (雲)骨幹排隊+傳輸 + 排隊 + 運算 + (雲)骨幹 + 下行傳輸
能耗  = 運算(執行節點係數;強車κf²較高) + 無線傳輸(含指派跳) + (雲)骨幹傳輸
成本  = 運算量 × 執行節點單價(雲 > 邊 > 本地/V2V=0)
```
- **佇列**：車輛/RSU 為 FIFO(busy_until)，雲端無佇列；回程為 100Mbps 共享 FIFO
- **移動性(兩層)**：
  - 預判層三級階梯：`predictor="linear"|"kalman"|"route"`
    - linear=等速外推(naive)；**kalman=EKF-CTRV(預設,現實可部署,只用 BSM
      位置/速度/航向估轉彎率 ω → 預判轉彎,kalman_tracker.py)**；
      route=路線分歧+離場時間(V2X 意圖分享,oracle 上界)
    - 預測斷線 → 拒卸載(`pred_reject`)
  - 事件驅動結算：V2V 到「完成時刻」用真實位置驗證 → 預判失準產生真實 `link_break`
  - 恢復層：`recovery="v2i"|"fail"`(v2i=結果經 RSU 遷移接續,含離場車的
    優雅退場交接,`break_recovered`；持有車離場=`consumer_left`)
  - 消融 3×2=6 組合(run_ablation.py；詳見 COMM_MODEL.md 第四節)

### 演算法（mappo.py）
GAE(γ=0.95, λ=0.95) + PPO clip(0.2) + entropy(0.02)；團隊獎勵=該 tick 各 agent 平均。

---

## 3. 關鍵參數（infra_config.py / vec_env.py）

| 參數 | 值 | 意義 |
|---|---|---|
| 算力(cycles/s) | 弱車 1e9 / 強車 12e9(多核聚合) / RSU 8e9 / 雲 50e9 | — |
| 能耗(J/cycle) | 弱車 1e-9 / 強車 2.25e-9(κf²) / RSU 2e-9 / 雲 4e-9 | — |
| 存取延遲 | V2V 5ms / V2I 20ms | MAC/協議開銷 |
| `CLOUD_EXTRA_LATENCY` | 40ms 單向 | 雲骨幹核網/傳播(可調) |
| `BACKHAUL_CAPACITY_BPS` | 100e6 | 回程容量(越小越不鼓勵用雲) |
| 使用成本(/cycle) | 雲 2e-10 / RSU 5e-11 / 本地·V2V 0 | pay-per-use |
| `STRONG_RATIO` | 0.35 | server 中強車比例 |
| 獎勵權重 | ENERGY_W 1.0 / ENERGY_NORM 5.0 / COST_W 1.0 | |
| 罰則 | PENALTY_MISS 2.0 / PENALTY_FAIL 5.0 | |
| 觀測正規化 | DATA_NORM 10e6 / CPU_NORM 6e9 / DEAD_NORM 3.0 | |
| **觀測維度** | 單代理 12 / 多代理 18；全域狀態 = RSU數 + 5 | |

**避免過度用雲的旋鈕**：`BACKHAUL_CAPACITY_BPS`↓、`COST_W`↑、`CLOUD_EXTRA_LATENCY`↑。

---

## 4. 進度（依階段；完整 commit 見 git log）

| 階段 | 內容 | 關鍵 commit |
|---|---|---|
| 落地 | 完整專案進 repo 根目錄 | `272ce82` |
| 通訊升級 | C-V2X/3GPP 鏈路預算、三層獨立協議、存取延遲 | `29f917e` |
| 抑制上雲 | 有限回程壅塞 + pay-per-use 成本 | `ea3e4e5` |
| 審查修正 | link_break、回程可觀測、指派跳能耗、κf²、CPU_NORM | `79e5587` |
| 分析工具 | oracle 決策分析 + 交叉點/Pareto 圖 | `e9f8d48` `9274f92` |
| 多指標訓練 | vision-only 成功率、成本欄、LR 衰減、成本對照圖 | `35faf07` |
| **移動性閉環** | 事件驅動結算 + route 預判 + V2I 遷移恢復 + 離場處理(優雅退場/consumer_left) | `43c67d1` `d20b31c` |
| **卡曼預判** | EKF-CTRV(僅 BSM 觀測)三級預判階梯 linear/kalman/route | `09c5c72` |
| 驗證/消融 | verify_invariants 25 項、run_ablation 3×2 | `79e7ac5` `bba119a` |
| 對稱化 | RSU/雲也事件驅動(換手 rsu_handover)、sweep_params 修復 | `bba119a` |
| **和平東路場景** | 路網(hepingeast2)、junction RSU 佈點(3路口×2角+路肩=8)、build_net/build_scenario | `61c4d9f` `e217daa` `f1c97a0` |
| **真實車流管線** | make_real_flow：台北 VD(路段/設備雙模式)→僅小客車 flow、自動對應+校對+去重 | `395a9b9`…`e2a9960` |
| 文件 | README(架構圖+流程圖)、COMM_MODEL、本檔 | — |

---

## 4.5 和平東路 SUMO 正式結果(2026-07，真實 VD 車流)

- **主對照**：MAPPO vs Greedy —— 延遲 731 vs 686ms(+6.6%)、
  **能耗 3.02 vs 4.57J(−34%)**、**成本 0.012 vs 0.171(−93%)**；
  Local-only 3865ms(卸載必要性成立)。
- **多 seed(n=3)**：成功率 88.61±0.05%、延遲 707.7±1.7ms、
  能耗 3.007±0.009J、成本 0.0101±0.0018 —— 高度可重現。
- **IPPO 消融**：IPPO 88.72±0.09% ≈ MAPPO(差距 < 2σ，不顯著)→
  本場景協調問題大多可分解(佇列狀態局部可觀測)；與 MAPPO 原論文
  「IPPO 常與 MAPPO 相當」一致。論文以誠實負結果呈現，主賣點放
  移動性閉環與 DT，不主打中央 critic。
- **DT 延遲掃描**：kalman 全 τ 壓過 linear(τ=8: 88.5 vs 88.1%)；
  linear τ=8 斷線下降屬過度保守假象(pred_reject 換來的)。
- **收斂**：~27 iter 達平台；vision-only 51→74%(+23pp)。
- **待解**：消融 v2i 救回=0，疑 consumer_left(車主離場)主導 →
  以新版消融圖(紫條)驗證；考慮「抵達補送」語意升級。

## 5. ⚠️ 重要狀態

- **既有 `mappo_vec.pt` / `dqn_vec.zip` 已作廢**：觀測維度改變(11→12、17→18)，
  載入會 shape 報錯 → **必須重新訓練**。
- 完整模擬需在**裝好 SUMO + traci + stable-baselines3 + gymnasium** 的環境執行。
- 純邏輯模組(comm_model / nodes / vec_env*)自我測試已驗證通過。

---

## 6. 下一步（建議順序）

1. **和平東路真實車流跑通**：`make_real_flow.py --remap` → sumo-gui 校對
   VD 方向(vd_debug 粉紅點) → 確認和平東路兩向覆蓋(缺向可鏡射假設)。
2. **正式重訓與評估**（和平東路 + 真實小客車流）：
   ```
   python verify_invariants.py            # 25 項全 PASS
   python train_mappo.py --sumo
   python compare_and_plot.py --sumo
   python run_ablation.py --sumo --plot   # 3 預判 × 2 恢復消融
   ```
   驗收：kalman 的 link_break 介於 linear 與 route 之間、v2i 救回率高、
   rsu_handover 在長路上出現、MAPPO 能耗/成本顯著低於 Greedy。
3. **論文補強(工具已備，SUMO 上正式跑)**：
   - IPPO 消融：`train_mappo.py --ippo` 或 `run_seeds.py --ippo`
   - 多 seed 統計：`run_seeds.py --sumo --seeds 3`
   - DT 同步延遲掃描：`run_dt_delay.py --sumo --plot`
     (mock 已證：τ=8s 時 kalman(dead-reckoning 補償) 比 linear 成功率+4pp、誤殺-58%)
   - 敏感度掃描：sweep_params.py
4. （可選）RSU 無線電資源競爭(可重用 link 佇列機制)、任務拆分。

---

## 7. 換路網管線（edge ID 全變，這些要重做）

| 檔案 | 動作 |
|---|---|
| `setup_rsu.py` | 重佈 RSU → 產生新的 `rsu_positions.json`（✅ 已自適應：RSU 數依覆蓋率自動決定、半徑同步 RSU_RANGE_M；`python setup_rsu.py` 即可，或 `--num/--target/--radius` 覆寫） |
| `vd_sumo_mapping.csv` | 重抓新區 VD(TDX/data.taipei)對應到新 edge |
| `turning_ratio.csv` / `calibrators.add.xml` | 依新 edge 重建轉向比與校正器 |
| `osm.sumocfg` | 指向新 net |
| MAPPO/DQN 模型 | RSU 數變 → 全域狀態維度變 → **重新訓練** |

**取得路網(已腳本化，見 README 流程)**：
Geofabrik `taiwan-latest.osm.pbf` → `osmconvert -b="西,南,東,北"` 裁切 →
`build_net.bat xxx.osm`(等效 wizard 轉檔，不依賴 Overpass)。

**目前主場景**：`hepingeast2.net.xml`（和平東路二段/復興南路，876×867m，
45 junction、3 號誌路口）；RSU=3 路口×2 角 + 路肩 2 = 8 個(覆蓋 92%)。
真實車流：`make_real_flow.py`(路段模式：TotalVol × 全市小客車比例 ≈0.75)。
舊場景(新生南路)檔案保留於 repo 供回溯。

---

## 8. 已知待辦 / 未建模（論文「假設」一節可聲明）

- RSU 無線電資源競爭（多車同傳同一 RSU 各自拿滿頻寬；目前 SNR 非 SINR、無干擾）
- 任務拆分 / partial offloading（目前任務原子、不可分割）
- 服務遷移 / handover 成本
- 重傳 / 封包錯誤率（確定性通道為已選定設計）
- 多 seed 訓練取統計（目前單一 run）
