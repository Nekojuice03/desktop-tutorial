# 三層通訊模型（V2V / V2I / 回程）說明

本專案的車—邊—雲三層卸載,通訊不再用「任意 REF_SNR」近似,改採車聯網文獻
主流的 **C-V2X / 3GPP 鏈路預算**,並區分三層各自的通訊方式。所有公式為
**確定性**(無隨機陰影/快衰落),以利 RL 訓練收斂與固定種子評估之可重現性。

## 一、三層通訊架構

| 層 | 通訊方式 | 標準依據 | 程式 |
|----|----------|----------|------|
| V2V(車↔協助車) | 直連 sidelink | C-V2X **PC5**（亦相容 DSRC/802.11p） | `V2V_LINK` |
| V2I(車↔RSU 邊緣) | 蜂巢/路側上行 | C-V2X **Uu** / PC5 | `V2I_LINK` |
| 回程(RSU↔雲) | 有線光纖骨幹(FiWi) | 固定核網/傳播延遲,無 rate 延遲 | `CLOUD_EXTRA_LATENCY` |

雲端為純邏輯節點,經「服務 RSU」以有線骨幹轉送;骨幹頻寬視為極大,故
**資料傳輸(rate)延遲≈0**,僅保留核網繞送 + 傳播的固定延遲(反映雲端較遠)。

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
總延遲 = 上行傳輸 + 上行存取 + (雲)骨幹 + 排隊 + 運算 + (雲)骨幹 + 下行傳輸
能耗  = 運算能耗(依執行節點係數)
       + 車輛無線傳輸能耗 (Ptx_W × 無線傳輸時間)
       + (僅雲)有線骨幹能耗 (Pbackhaul_W × 骨幹延遲)
```
排隊為 FIFO(車輛/RSU 有佇列,雲端無),見 `nodes.ComputeNodes`。

## 五、設計選擇與可調項
- **確定性通道**:不含隨機陰影(σ≈3dB)與快衰落,換取可重現性。若論文需隨機性,
  可在 `path_loss_db()` 加 log-normal 陰影項。
- **覆蓋判斷**:以距離門檻(覆蓋半徑)代替 SINR 解碼門檻,為常見簡化。
- **回程**:依設計為有線無 rate 延遲;若要改為有限骨幹頻寬,於 cloud 分支加一段
  `data_bits / R_backhaul` 即可。

## 參考文獻
- 3GPP TR 37.885, *Study on evaluation methodology of new V2X use cases for LTE and NR*（V2X 通道/路徑損耗模型）。
- H. Ye et al., "Intelligent Task Offloading for Heterogeneous V2X Communications," arXiv:2006.15855（DSRC/C-V2X/mmWave 異質 V2X 卸載）。
- X. Huang et al., "Joint Task Offloading and Resource Allocation for Vehicular Edge Computing Based on V2I and V2V Modes," *IEEE T-ITS*, 2022.
- Vehicular Edge Computing 綜述, arXiv:1908.06849（車—邊—雲三層架構、RSU↔雲光纖回程）。
