"""
基站(RSU)與雲端的「邏輯規格」——這就是 SUMO 不負責、要你自己定義的部分。

- 基站「位置」由 setup_rsu.py 從路網讀出，存在 rsu_positions.json，這裡載入它
- 再補上算力、覆蓋半徑、延遲（這些都是通訊/運算邏輯，與 SUMO 無關）
- 雲端是純邏輯節點，沒有位置

數字都是「合理起始值」，之後依你的論文設定調整即可。
"""
import json
import os

# 此檔案所在資料夾（讓 load_rsus 不論從哪裡被 import 都找得到 json）
_HERE = os.path.dirname(os.path.abspath(__file__))

# ========== 通訊 / 物理參數（簡化解析模型）==========
RSU_RANGE_M      = 200        # 基站覆蓋半徑(公尺)。★務必與 setup_rsu.py 的 COVER_RADIUS 一致
V2V_RANGE_M      = 150        # 車對車通訊半徑(公尺)
RSU_BANDWIDTH_HZ = 20e6       # 基站頻寬
V2V_BANDWIDTH_HZ = 10e6       # V2V 頻寬

# ========== 無線鏈路預算（C-V2X / 3GPP，確定性路徑損耗）==========
# 取代舊的「REF_SNR 任意常數」模型，改用真實鏈路預算：
#   發射功率(dBm) − 路徑損耗(dB) = 接收功率 → 對雜訊算 SINR → Shannon 速率。
# 參數依車聯網文獻常用值(C-V2X / 3GPP TR 37.885，5.9GHz ITS 頻段)：
CARRIER_FREQ_HZ  = 5.9e9      # 載波頻率(5.9 GHz，DSRC/C-V2X 共用之 ITS 頻段)
VEH_TX_POWER_DBM = 23.0       # 車輛發射功率(3GPP C-V2X 規範 23 dBm ≈ 0.2W)，V2V/V2I 上行皆用此
NOISE_PSD_DBM_HZ = -174.0     # 熱雜訊功率譜密度(dBm/Hz)
NOISE_FIGURE_DB  = 9.0        # 接收端雜訊指數(車載/路側收發機典型值)
PL_REF_DIST_M    = 1.0        # 路徑損耗參考距離 d0(公尺)，FSPL 由此起算
PL_EXP_V2V       = 3.0        # V2V 都市路徑損耗指數(車間多遮蔽，衰減快)
PL_EXP_V2I       = 2.7        # V2I 都市(基站架高、視距較好 → 衰減略慢)

# 鏈路設定檔：把每條無線鏈路的頻寬/覆蓋/發射功率/路徑損耗指數打包，
# 供 comm_model 計算 SINR→速率→傳輸延遲。三層各自參數不同 → 速率自然分層。
V2V_LINK = {"bandwidth": V2V_BANDWIDTH_HZ, "range": V2V_RANGE_M,
            "tx_dbm": VEH_TX_POWER_DBM, "pl_exp": PL_EXP_V2V}
V2I_LINK = {"bandwidth": RSU_BANDWIDTH_HZ, "range": RSU_RANGE_M,
            "tx_dbm": VEH_TX_POWER_DBM, "pl_exp": PL_EXP_V2I}

# ========== 三層節點算力 (CPU 週期/秒，越大越快) ==========
# 對應你的優先級：車最弱 → 基站中等 → 雲最強
# 車輛再分強弱：弱車(多數)算不動重任務，強車(少數)能當 V2V 幫手
# ★算力一律解讀為「多核聚合 cycles/s」(單核 12GHz 物理上不存在)：
#   弱車 ≈ 1核×1.0GHz；強車 ≈ 8核×1.5GHz 聚合 12GHz(車載高效能平台)。
VEHICLE_CPU      = 1.0e9     # 弱 server / client 的算力（最弱）
STRONG_VEHICLE_CPU = 12.0e9  # 強 server 的算力（多核聚合；V2V 幫手，故意高於基站）
STRONG_RATIO     = 0.35      # server 中強車比例(近未來車聯網成熟場景，三至四成車有運算餘裕)
RSU_CPU          = 8.0e9     # 基站「每個並行服務槽」的算力（故意略低於強車，讓近距離 V2V 有優勢）
# ★RSU 並行服務槽數(2026-08 檢視新增)。先前 RSU 是「單一 FIFO」：一個 vision 任務
#   就佔住 ~560ms，第二個直接等 560ms(deadline 只有 1500ms) → 邊緣層幾乎不被選用
#   (實測 greedy 只有 1% 的任務選 RSU)。真實 MEC 伺服器是多核並行的。
#   RSU 總容量 = RSU_CORES × RSU_CPU。
#   ⚠ 預設 1 = 維持既有行為(舊結果可比)。要探討邊緣層的合理配置，
#     用 sweep_params.py 掃 RSU_CORES，並在論文說明所採用的值與理由。
RSU_CORES        = 1
CLOUD_CPU        = 50.0e9    # 雲端（最強，但延遲高）

# ========== 各層「協議存取」固定延遲 (秒) ==========
# 純 Shannon 速率只算「資料推送時間」，省略了 MAC 競爭/排程/協議握手的固定開銷。
# 這裡補上各無線鏈路的「存取延遲」，由 nodes.estimate() 在上行時實際計入(更貼近實際)：
LOCAL_EXTRA_LATENCY = 0.000   # 本地執行，無傳輸、無存取
V2V_EXTRA_LATENCY   = 0.005   # 5 ms（C-V2X PC5 直連，分散式排程，存取快）
RSU_EXTRA_LATENCY   = 0.020   # 20 ms（V2I/Uu，含基站排程；走雲端也先經此 V2I 存取）

# ========== 雲端回程（RSU↔雲，有線骨幹）==========
# 回程 = 固定傳播延遲 + 「有限頻寬」傳輸（會壅塞）。兩段分工：
#   (a) 固定核網/傳播延遲(單向)：反映雲端物理上較遠，來回各計一次。
CLOUD_EXTRA_LATENCY  = 0.040   # 40 ms 單向(≈80ms RTT)，區域型雲資料中心實際區間(20~100ms)
#   (b) ★方法①：有限回程頻寬 → 共享 FIFO 佇列。上雲任務越多，回程越塞、延遲自己上升，
#       形成「過度用雲會自我懲罰」的物理機制(取代硬調延遲)。值越小越快塞、越不鼓勵用雲。
BACKHAUL_CAPACITY_BPS = 100e6  # 100 Mbps 共享骨幹(RSU↔雲)；所有上雲任務共用此容量

# ========== 能耗係數 ==========
# 各層「運算能耗」係數 (焦耳/cycle)：誰執行任務，就用誰的係數計算運算耗能。
# 反映「邊緣比雲端節能」的層級差異：雲端每 cycle 耗能最高(資料中心散熱/PUE 開銷)，
# 邊緣居中，車輛最低。★這同時修正了「邊緣與雲端能耗一樣」的問題：
#   先前只算傳輸能耗，而 RSU 與 Cloud 的傳輸路徑相同 → 能耗必然相等。
#   現在三層各有獨立的運算能耗係數，邊緣與雲端自然不同。
VEHICLE_ENERGY_PER_CYCLE = 1e-9   # 弱車(本地/V2V)每 cycle 耗能(焦耳/cycle)，1核×1.0GHz
# ★強車依 DVFS 能耗定律 E/cycle ∝ κf²(以「單核頻率」計)：強車為 8核×1.5GHz，
#   per-cycle 能耗 = (1.5/1.0)² × 弱車 = 2.25e-9。修正先前「強車更快又不更耗能」
#   的免費午餐(違反物理)，讓 v2v_strong 在能耗面付出合理代價。
STRONG_VEHICLE_ENERGY_PER_CYCLE = 2.25e-9
RSU_ENERGY_PER_CYCLE     = 2e-9   # 邊緣基站(中等)
CLOUD_ENERGY_PER_CYCLE   = 4e-9   # 雲端(最高，含資料中心散熱/PUE 開銷)

# 車輛無線電的「消耗」功率(瓦)。注意這與鏈路預算的 23 dBm 不是同一件事：
#   23 dBm = 0.2 W 是天線「輻射」功率(算 SINR 用)；
#   0.5 W  是收發機「消耗」功率(算能耗用)，兩者比值 2.5× 隱含功率放大器效率約 40%，
#   為車載 PA 的合理區間。★兩個數字並非筆誤，論文請一併說明。
TX_POWER_W               = 0.5
# RSU 下行的等效消耗功率(瓦)。先前下行一律用車輛的 TX_POWER_W 計能耗，
# 但 RSU→車 這段是 RSU 在發射，路側設備功率高於車載。能耗為「系統總能耗」，
# 故仍計入總帳，只是改用發射端自己的功率。
RSU_TX_POWER_W           = 2.0
# 雲端回程(RSU↔雲，有線骨幹)的等效傳輸功率(瓦)：只有走雲端這段才會用到，
# 讓「雲端」在傳輸面也比「邊緣」多一份回程能耗 → 進一步拉開邊緣 vs 雲端。
CLOUD_BACKHAUL_POWER_W   = 1.0

# ========== 使用成本（pay-per-use；★方法②：避免過度用雲）==========
# 反映真實計費：雲端按用量收費最貴、邊緣較便宜、本地/V2V 用自有資源免費。
# 由 nodes.estimate() 依「執行節點」算成本(成本 = 運算量 × 每 cycle 單價)，
# 再以 COST_W 權重(見 vec_env.py)併入獎勵。單位為任意貨幣，與能耗/延遲一起權衡。
CLOUD_COST_PER_CYCLE = 2.0e-10   # 雲端每 cycle 計費(最貴)
RSU_COST_PER_CYCLE   = 5.0e-11   # 邊緣較便宜(可設 0 表示自營免費)
# 本地 / V2V = 0（自有車輛資源，不計費）


def load_rsus(path=None):
    """載入基站位置，並補上算力/覆蓋等執行期屬性。"""
    if path is None:
        path = os.path.join(_HERE, "rsu_positions.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rsus = {}
    for rid, info in raw.items():
        rsus[rid] = {
            "x": info["x"], "y": info["y"],
            "cpu":   RSU_CPU,
            "range": RSU_RANGE_M,
            "load":  0.0,         # 當前負載，執行期動態更新
        }
    return rsus


# 雲端：純邏輯節點，沒有座標，視為隨時可達（透過基站回傳）
CLOUD = {
    "cpu":           CLOUD_CPU,
    "extra_latency": CLOUD_EXTRA_LATENCY,
    "load":          0.0,
}


if __name__ == "__main__":
    # 小測試：印出載入的基站（需先跑過 setup_rsu.py 產生 json）
    try:
        rsus = load_rsus()
        print(f"載入 {len(rsus)} 個基站：")
        for rid, r in rsus.items():
            print(f"  {rid}: 座標({r['x']:.1f},{r['y']:.1f}) "
                  f"算力{r['cpu']/1e9:.0f}GHz 覆蓋{r['range']}m")
        print(f"雲端：算力{CLOUD['cpu']/1e9:.0f}GHz 延遲{CLOUD['extra_latency']*1000:.0f}ms")
    except FileNotFoundError:
        print("尚未找到 rsu_positions.json，請先在有路網的電腦上執行 setup_rsu.py")
