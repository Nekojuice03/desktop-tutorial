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
- **移動性約束**：總延遲 > 連線可維持時間(contact_time) → `link_break` 失敗

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

## 4. 進度（commit 歷史）

| Commit | 內容 |
|---|---|
| `e48759d` | (前 session) 修能耗模型、資源分配偏差、訓練曲線 |
| `272ce82` | 完整專案落地 repo 根目錄 |
| `29f917e` | 通訊升級為 C-V2X/3GPP 鏈路預算，三層獨立協議 |
| `ea3e4e5` | 避免過度用雲：①有限回程頻寬壅塞 ②雲端 pay-per-use 成本 |
| `79e5587` | 審查修正：移動性失敗判定、回程可觀測、指派跳能耗、κf² 強車能耗、CPU_NORM、Random 種子 |

---

## 5. ⚠️ 重要狀態

- **既有 `mappo_vec.pt` / `dqn_vec.zip` 已作廢**：觀測維度改變(11→12、17→18)，
  載入會 shape 報錯 → **必須重新訓練**。
- 完整模擬需在**裝好 SUMO + traci + stable-baselines3 + gymnasium** 的環境執行。
- 純邏輯模組(comm_model / nodes / vec_env*)自我測試已驗證通過。

---

## 6. 下一步（建議順序）

1. **先在現有小路網重訓並驗收**（找問題最快）：
   ```
   python train_mappo.py          # mock 快測會學習
   python train_mappo.py --sumo   # 真實 SUMO
   python compare_and_plot.py --sumo
   ```
   驗收：RSU 能耗 < Cloud、雲端佔比下降、出現少量 `link_break`、有 `avg_cost` 欄位。
2. **換複雜路網**（大安格網 / 公館）+ 換網管線（見下節）。
3. （可選）RSU 無線電競爭、成本對照圖、多 seed 訓練取 mean±std。

---

## 7. 換路網管線（edge ID 全變，這些要重做）

| 檔案 | 動作 |
|---|---|
| `setup_rsu.py` | 重佈 RSU → 產生新的 `rsu_positions.json` |
| `vd_sumo_mapping.csv` | 重抓新區 VD(TDX/data.taipei)對應到新 edge |
| `turning_ratio.csv` / `calibrators.add.xml` | 依新 edge 重建轉向比與校正器 |
| `osm.sumocfg` | 指向新 net |
| MAPPO/DQN 模型 | RSU 數變 → 全域狀態維度變 → **重新訓練** |

**取得路網兩條路**：
- osmWebWizard（目前卡在 Overpass 下載 `Download failed`，疑似 IP 限流/網路阻擋，
  待換手機熱點或等限流解除再試）
- 手動：openstreetmap.org Export / bbbike 抓 `.osm` → `netconvert`（不依賴 Overpass，最穩）

**目前網路範圍**：lon 121.5307–121.5353, lat 25.0402–25.0444（新生南路一段，約 510m²，17 junction）。

---

## 8. 已知待辦 / 未建模（論文「假設」一節可聲明）

- RSU 無線電資源競爭（多車同傳同一 RSU 各自拿滿頻寬；目前 SNR 非 SINR、無干擾）
- 任務拆分 / partial offloading（目前任務原子、不可分割）
- 服務遷移 / handover 成本
- 重傳 / 封包錯誤率（確定性通道為已選定設計）
- 多 seed 訓練取統計（目前單一 run）
