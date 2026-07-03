# 三層通訊模型（V2V / V2I / 回程）說明

本專案的車—邊—雲三層卸載,通訊不再用「任意 REF_SNR」近似,改採車聯網文獻
主流的 **C-V2X / 3GPP 鏈路預算**,並區分三層各自的通訊方式。所有公式為
**確定性**(無隨機陰影/快衰落),以利 RL 訓練收斂與固定種子評估之可重現性。

## 一、三層通訊架構

| 層 | 通訊方式 | 標準依據 | 程式 |
|----|----------|----------|------|
| V2V(車↔協助車) | 直連 sidelink | C-V2X **PC5**（亦相容 DSRC/802.11p） | `V2V_LINK` |
| V2I(車↔RSU 邊緣) | 蜂巢/路側上行 | C-V2X **Uu** / PC5 | `V2I_LINK` |
| 回程(RSU↔雲) | 有線光纖骨幹(FiWi) | 固定傳播延遲 + **有限頻寬(會壅塞)** | `CLOUD_EXTRA_LATENCY` / `BACKHAUL_CAPACITY_BPS` |

雲端為純邏輯節點,經「服務 RSU」以有線骨幹轉送。回程 = 固定核網/傳播延遲
(反映雲端較遠) + **有限頻寬的共享傳輸**(所有上雲任務共用,會排隊壅塞;見第五節)。

## 二、無線鏈路預算（V2V / V2I）

於 `comm_model.py`:

**1. 路徑損耗(log-distance,確定性)** `path_loss_db()`
```
PL(d) = FSPL(d0) + 10·n·log10(d / d0)            [dB]
FSPL(d0) = 20·log10(4π·d0·fc / c)                 (自由空間,參考距離 d0)
```
- `fc = 5.9 GHz`(ITS 頻段)、`d0 = 1 m`
- 路徑損耗指數 `n`:V2V `PL_EXP_V2V = 3.0`(車間多遮蔽)、V2I `PL_EXP_V2I = 2.7`(基站架高、視距較好)

**2. 接收 SNR → Shannon 速率** `data_rate()`
```
Prx[dBm] = Ptx[dBm] − PL(d)[dB]
N[dBm]   = −174 + 10·log10(B) + NF
SNR      = 10^((Prx − N)/10)            (線性)
rate(d)  = B · log2(1 + SNR)            [bits/s]
```

**3. 傳輸延遲** `transmission_delay()`
```
delay = data_bits / rate(d)             ；d > 覆蓋半徑 → ∞(卸載失敗)
```

### 參數（`infra_config.py`,取自文獻常用值）
| 參數 | 值 | 出處/說明 |
|------|----|-----------|
| 載波 `CARRIER_FREQ_HZ` | 5.9 GHz | DSRC/C-V2X ITS 頻段 |
| 車輛發射功率 `VEH_TX_POWER_DBM` | 23 dBm | 3GPP C-V2X 規範(≈0.2W) |
| 雜訊密度 `NOISE_PSD_DBM_HZ` | −174 dBm/Hz | 熱雜訊 |
| 雜訊指數 `NOISE_FIGURE_DB` | 9 dB | 收發機典型值 |
| V2V 頻寬 | 10 MHz | PC5 sidelink |
| V2I 頻寬 | 20 MHz | Uu/路側 |
| V2V/V2I 覆蓋 | 150 / 200 m | 都市直連/路側典型 |

實測速率(本模型):V2I 10m≈267 Mbps、150m≈60 Mbps、200m≈41 Mbps,
落在 C-V2X 實際區間。

## 三、協議存取延遲與雲端回程

純 Shannon 只算「資料推送時間」,省略 MAC 競爭/排程/握手,故補上固定存取延遲
(於 `nodes.estimate()` 實際計入上行):
- V2V `V2V_EXTRA_LATENCY = 5 ms`(PC5 分散式排程,快)
- V2I `RSU_EXTRA_LATENCY = 20 ms`(Uu 含基站排程;走雲端也先付此 V2I 存取)
- 雲端骨幹 `CLOUD_EXTRA_LATENCY = 40 ms` **單向**(≈80ms RTT),區域雲資料中心實際區間(20–100ms)。
  此為**最影響三層取捨的旋鈕**:值越大,近處邊緣/V2V 越易勝過雲端。

## 四、延遲與能耗總式（`nodes.estimate()`）

```
總延遲 = 上行傳輸 + 上行存取 + (雲)骨幹排隊+傳輸 + 排隊 + 運算 + (雲)骨幹 + 下行傳輸
能耗  = 運算能耗(依執行節點係數;強車依 κf² 較高)
       + 車輛無線傳輸能耗 (Ptx_W × 無線傳輸時間，含 client→server 指派跳)
       + (僅雲)有線骨幹能耗 (Pbackhaul_W × 骨幹傳輸時間)
```
排隊為 FIFO(車輛/RSU 有佇列,雲端無),見 `nodes.ComputeNodes`。

**移動性:兩層機制(預判 proactive + 恢復 reactive)** —— V2V 中途斷線的完整處理:

*預判層(admission control)*:決策當下以預測的連線壽命 `contact_s` 檢查
`總延遲 ≤ contact_s`,不足即拒絕(`pred_reject`)。預測器**三級階梯**
(naive → 可部署 → oracle;消融用,`VECMultiEnv(predictor=...)`):
- `"linear"`:等速直線外推(`comm_model.contact_time` 解析解)——路口轉彎會**高估**。
- `"kalman"`(**預設,現實可部署**):EKF-CTRV 卡曼濾波(`kalman_tracker.py`)。
  **只用車輛本就廣播的 BSM/CAM 可觀測量**(位置/速度/航向,無需路線),
  狀態 [x,y,v,θ,ω] 中的轉彎率 ω 由航向歷史濾波估出 → 前推兩車軌跡求
  「距離首超通訊半徑」時刻 = 預測連線壽命(PLL)。**能預判對方正在轉彎或
  維持同向**;ω 以半衰期 10s 衰減(轉彎為暫態)。回應「模擬知道路線、
  現實怎麼辦」:數位孿生本就持續收 BSM/CAM,以 KF 追蹤是標準做法。
  文獻:Kalman 預測 link residual lifetime(IEEE VTC 2016)、
  VANET 位置預測(WPC 2014)、EKF 之 PLL 指標。
- `"route"`(oracle 上界):再取 `min(直線外推, 離場時間, 路線分歧時間)`。由
  `TraciWorld.route_divergence_time()` 估計:
  (i) **離場時間**——任一車走完剩餘路線抵達終點就會離開模擬(SUMO 消融實測的
  斷線**主因**,而非開出範圍);紅燈停等以車道限速三成為速度下限。
  (ii) **分歧時間**——同路兩車的共同前綴長度/較快車速(不同路無法對齊時仍保留離場上限)。
  物理上對應 **V2X 意圖分享**(SAE J2735 BSM Path Prediction、ETSI CAM path
  history:車輛本就廣播預測路徑與目的地,數位孿生為其匯集點),非作弊。
  文獻:link lifetime prediction 可將任務丟失/重做率降 70–80%(Sensors 2022)。

*事件驅動結算(V2V 與 RSU/雲對稱)*:所有卸載任務(本地除外)不在決策當下拍板,
而是掛到「預計完成時刻」,屆時用 SUMO **真實位置**驗證「結果送得回持有車嗎」:
- V2V:兩車仍在範圍 → 照預測;否則真實斷線(`link_break`) → 進恢復層。
- RSU/雲:持有車仍在**原服務基站**覆蓋 → 照預測;駛入**另一 RSU** 覆蓋 →
  基礎設施**換手**(`rsu_handover`,蜂巢常態,付換手存取+重送下行代價);
  完全無覆蓋 → 失敗。對稱設計避免「只有 V2V 承受執行期風險」的偏袒。
RL 先收預測獎勵,結算差額補進後續 tick 的團隊獎勵(GAE 回傳信用)。
kalman 預判層同樣對稱:V2V 用兩車 CTRV 前推、V2I 用單車對靜止 RSU 前推
(`predict_contact_static`)。

*恢復層(service migration)*:真實斷線時(`VECMultiEnv(recovery=...)`):
- `"v2i"`(預設):結果經基礎設施遷移 執行車→RSU→(有線)→RSU→持有車,
  遷移延遲 = V2I 存取 + 兩段 V2I 傳輸,可能因此超時;救回計 `break_recovered`。
  **執行車已抵達離場**時用其離場前最後位置做「優雅退場交接」(車輛抵達不是
  瞬間蒸發,離場前可把結果交給路側)——修正先前對象離場即無法恢復的問題。
- `"fail"`:直接失敗(`break_failed`);兩端任一側無 RSU 覆蓋時亦同。
- **持有車離場**(結果無人接收)為任務作廢,獨立計 `consumer_left`
  (break_failed 子類),論文可分開報告。
文獻:migration-enabled task offloading / service continuity(CCF TPCI 2024 等)。

rsu/cloud 目標為靜止基站,仍用等速外推的 sojourn 檢查;本地不檢查。

**異質車輛能耗(DVFS κf²)**:強車為多核聚合 12 GHz(如 8核×1.5GHz),
依 E/cycle ∝ κf²(以單核頻率計)其每 cycle 能耗 = (1.5/1.0)² = 2.25× 弱車
(`STRONG_VEHICLE_ENERGY_PER_CYCLE`)。強車「更快但更耗能」,移除免費午餐。

## 五、避免「過度用雲」的機制（文獻做法,非硬調延遲）

只把 `CLOUD_EXTRA_LATENCY` 調大屬「硬調旋鈕」;本專案改採文獻主流的兩個有原理機制,
讓用雲在模型裡**自然產生代價**:

**① 有限回程頻寬 + 壅塞**(`nodes.estimate()` cloud 分支 + `ComputeNodes` 鏈路佇列)
RSU↔雲為一條**被所有上雲任務共享的有限容量鏈路** `BACKHAUL_CAPACITY_BPS`(預設 100 Mbps),
用 FIFO 佇列建模。上雲任務越多,回程越塞,延遲**隨負載自己上升** → 自我調節。
```
回程傳輸 = (上行資料 + 下行結果) / 骨幹容量      (排隊 = max(0, busy_until − 抵達時刻))
```
> 來源:VEC 綜述指 offloading 到 edge 可減輕 backhaul 壓力;拓樸感知負載平衡將
> backhaul 壅塞納入路由決策(arXiv:2502.06963)。

**② 使用成本 pay-per-use**(`nodes.estimate()` 回傳 `cost`,於獎勵以 `COST_W` 加權)
反映真實計費:雲端按用量最貴、邊緣較便宜、本地/V2V 免費。
```
cost = 運算量(cycles) × 每 cycle 單價     (cloud 2e-10 > rsu 5e-11 > local/v2v = 0)
獎勵 = −(延遲 + ENERGY_W·能耗/ENERGY_NORM + COST_W·cost)
```
> 來源:能耗-延遲-成本三目標權衡(arXiv:1805.02006);定價驅動卸載(arXiv:2011.02154)。

**可調**:`COST_W=0` 關閉成本項;`BACKHAUL_CAPACITY_BPS` 越小越早壅塞、越不鼓勵用雲。
兩者皆比硬調 `CLOUD_EXTRA_LATENCY` 更有論文說服力。

## 六、設計選擇與可調項
- **確定性通道**:不含隨機陰影(σ≈3dB)與快衰落,換取可重現性。若論文需隨機性,
  可在 `path_loss_db()` 加 log-normal 陰影項。
- **覆蓋判斷**:以距離門檻(覆蓋半徑)代替 SINR 解碼門檻,為常見簡化。
- **回程**:固定傳播延遲 `CLOUD_EXTRA_LATENCY` + 有限頻寬 `BACKHAUL_CAPACITY_BPS`
  (共享 FIFO,會壅塞);若要改為「無限快」可把容量設極大、成本權重 `COST_W=0`。

## 參考文獻
- 3GPP TR 37.885, *Study on evaluation methodology of new V2X use cases for LTE and NR*（V2X 通道/路徑損耗模型）。
- "Evaluating Link Lifetime Prediction to Support Computational Offloading Decision in VANETs," *Sensors* 22(16), 2022（連線壽命預測支援卸載決策；ML 預測降任務丟失 70–80%）。
- "Prediction of link residual lifetime using Kalman filter in vehicular ad hoc networks," *IEEE VTC*, 2016（卡曼濾波預測連線殘餘壽命——kalman 預判器的直接依據）。
- "Location Prediction of Vehicles in VANETs Using A Kalman Filter," *Wireless Personal Communications*, 2014（KF 車輛位置預測）。
- "A bandwidth-fair migration-enabled task offloading for vehicular edge computing," *CCF TPCI*, 2024（斷線後任務遷移/服務連續性）。
- "Vehicle Motion Prediction at Intersections Based on the Turning Intention and Prior Trajectories Model," *IEEE/CAA JAS*, 2021（路口轉向意圖辨識）。
- SAE J2735（BSM Path Prediction）、ETSI EN 302 637-2（CAM path history）——V2X 意圖分享標準，route-aware 預判器的物理依據。
- Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games," NeurIPS 2022（MAPPO）。
- H. Ye et al., "Intelligent Task Offloading for Heterogeneous V2X Communications," arXiv:2006.15855（DSRC/C-V2X/mmWave 異質 V2X 卸載）。
- X. Huang et al., "Joint Task Offloading and Resource Allocation for Vehicular Edge Computing Based on V2I and V2V Modes," *IEEE T-ITS*, 2022.
- Vehicular Edge Computing 綜述, arXiv:1908.06849（車—邊—雲三層架構、RSU↔雲光纖回程）。
