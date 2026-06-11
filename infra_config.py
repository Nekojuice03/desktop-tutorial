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

# ========== 通訊通道模型（資料速率隨距離衰減）==========
REF_DISTANCE_M    = 10.0      # 參考距離(公尺)：此距離內視為最佳訊號
REF_SNR           = 100.0     # 參考距離處的訊雜比(線性值，約 20 dB)
PATHLOSS_EXPONENT = 3.0       # 路徑損耗指數(都市約 2.7~3.5，越大衰減越快)

# ========== 三層節點算力 (CPU 週期/秒，越大越快) ==========
# 對應你的優先級：車最弱 → 基站中等 → 雲最強
# 車輛再分強弱：弱車(多數)算不動重任務，強車(少數)能當 V2V 幫手
VEHICLE_CPU      = 1.0e9     # 弱 server / client 的算力（最弱）
STRONG_VEHICLE_CPU = 12.0e9  # 強 server 的算力（V2V 幫手，故意高於基站）
STRONG_RATIO     = 0.35      # server 中強車比例(近未來車聯網成熟場景，三至四成車有運算餘裕)
RSU_CPU          = 8.0e9     # 基站（故意略低於強車，讓近距離 V2V 有優勢）
CLOUD_CPU        = 50.0e9    # 雲端（最強，但延遲高）

# ========== 各層「額外」固定延遲 (秒) ==========
# 反映卸載優先級 車 < 基站 < 雲：越往上層傳輸延遲越高
LOCAL_EXTRA_LATENCY = 0.000   # 本地執行，無傳輸
V2V_EXTRA_LATENCY   = 0.005   # 5 ms（車對車，近）
RSU_EXTRA_LATENCY   = 0.020   # 20 ms（車→基站）
CLOUD_EXTRA_LATENCY = 0.300   # 300 ms（基站→雲，回程慢；調高讓近處強車V2V有機會贏過雲）

# ========== 能耗係數 ==========
# 各層「運算能耗」係數 (焦耳/cycle)：誰執行任務，就用誰的係數計算運算耗能。
# 反映「邊緣比雲端節能」的層級差異：雲端每 cycle 耗能最高(資料中心散熱/PUE 開銷)，
# 邊緣居中，車輛最低。★這同時修正了「邊緣與雲端能耗一樣」的問題：
#   先前只算傳輸能耗，而 RSU 與 Cloud 的傳輸路徑相同 → 能耗必然相等。
#   現在三層各有獨立的運算能耗係數，邊緣與雲端自然不同。
VEHICLE_ENERGY_PER_CYCLE = 1e-9   # 車輛(本地/V2V 幫手)每個 CPU 週期的耗能(焦耳/cycle)
RSU_ENERGY_PER_CYCLE     = 2e-9   # 邊緣基站(中等)
CLOUD_ENERGY_PER_CYCLE   = 4e-9   # 雲端(最高，含資料中心散熱/PUE 開銷)

TX_POWER_W               = 0.5    # 車輛無線傳輸功率(瓦)：上/下行卸載耗能 = 功率 × 傳輸時間
# 雲端回程(RSU↔雲，有線骨幹)的等效傳輸功率(瓦)：只有走雲端這段才會用到，
# 讓「雲端」在傳輸面也比「邊緣」多一份回程能耗 → 進一步拉開邊緣 vs 雲端。
CLOUD_BACKHAUL_POWER_W   = 1.0


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
