"""
通訊模型（comm_model.py）
================================
這是「SUMO 不負責的另一個物理層」——無線通訊。
SUMO 只給你車在哪、跑多快；這支程式用那些座標/速度算出：

  1. can_communicate() ：兩點距離夠近、連得到嗎
  2. data_rate()       ：這個距離下的傳輸速率(bits/s)，距離越遠越慢
  3. transmission_delay()：傳一筆資料要花多少秒(超出範圍 → 無限大)
  4. contact_time()    ：兩個會動的東西還能保持連線多久 ← 斷線接手的關鍵
  5. neighbors_in_range() / rsus_in_range()：找出範圍內的鄰車 / 基站

設計原則：純函式，不碰 SUMO。輸入都是普通數字(位置、速度)，
所以可以單獨測試（直接執行此檔會跑自我測試）。
之後的環境程式再用 TraCI 取得真實位置/速度，餵進這些函式。
"""
import math

from infra_config import (
    RSU_RANGE_M, V2V_RANGE_M,
    CARRIER_FREQ_HZ, NOISE_PSD_DBM_HZ, NOISE_FIGURE_DB, PL_REF_DIST_M,
    V2V_LINK, V2I_LINK,
)

_LIGHT_SPEED = 299_792_458.0   # 光速 (m/s)，算自由空間路徑損耗用


# ---------- 基本幾何 ----------
def distance(a, b):
    """兩點 (x,y) 的直線距離(公尺)。"""
    return math.dist(a, b)


def velocity_from_speed_angle(speed, angle_deg):
    """
    把 SUMO 的『速率(scalar) + 角度』轉成速度向量 (vx, vy)。
    SUMO 角度：0 度朝北，順時針遞增。
    靜止物件(基站)就直接傳 (0.0, 0.0)。
    """
    rad = math.radians(angle_deg)
    return (speed * math.sin(rad), speed * math.cos(rad))


# ---------- 連線判斷 ----------
def can_communicate(pos_a, pos_b, comm_range):
    """距離在 comm_range 內就視為連得到。"""
    return distance(pos_a, pos_b) <= comm_range


# ---------- 路徑損耗（3GPP log-distance，確定性）----------
def free_space_pl_db(d0, fc_hz=CARRIER_FREQ_HZ):
    """參考距離 d0(公尺)的自由空間路徑損耗(dB)：FSPL = 20·log10(4π·d0·fc / c)。"""
    return 20.0 * math.log10(4.0 * math.pi * d0 * fc_hz / _LIGHT_SPEED)


def path_loss_db(d, pl_exp, fc_hz=CARRIER_FREQ_HZ, d0=PL_REF_DIST_M):
    """
    log-distance 路徑損耗(dB)：PL(d) = FSPL(d0) + 10·n·log10(d/d0)。
    n = 路徑損耗指數(V2V/V2I 各異)。確定性(無隨機陰影/快衰落) → 結果可重現。
    """
    d = max(d, d0)
    return free_space_pl_db(d0, fc_hz) + 10.0 * pl_exp * math.log10(d / d0)


# ---------- 資料速率（真實鏈路預算 → Shannon）----------
def data_rate(d, link):
    """
    距離 d(公尺)在某鏈路(link 設定檔)上的可達速率(bits/s)。
    用實際鏈路預算取代舊的任意 REF_SNR：
      接收功率 Prx[dBm] = 發射功率[dBm] − 路徑損耗[dB]
      雜訊功率 N[dBm]   = -174 + 10·log10(頻寬) + 雜訊指數
      SNR[dB] = Prx − N → 轉線性 → 速率 = 頻寬 × log2(1 + SNR)   (Shannon 容量)
    """
    pl_db = path_loss_db(d, link["pl_exp"])
    prx_dbm = link["tx_dbm"] - pl_db
    noise_dbm = NOISE_PSD_DBM_HZ + 10.0 * math.log10(link["bandwidth"]) + NOISE_FIGURE_DB
    snr_linear = 10.0 ** ((prx_dbm - noise_dbm) / 10.0)
    return link["bandwidth"] * math.log2(1.0 + snr_linear)


# ---------- 傳輸延遲 ----------
def transmission_delay(data_bits, d, link):
    """
    在 link 上傳 data_bits 位元要花幾秒(= 資料量 / 速率)。
    超出該鏈路覆蓋範圍 → 回傳無限大(代表傳不到、卸載失敗)。
    """
    if d > link["range"]:
        return float("inf")
    rate = data_rate(d, link)
    return data_bits / rate if rate > 0 else float("inf")


# 方便用的兩個包裝：直接指定是 V2V 還是車對基站(V2I)
def v2v_delay(data_bits, pos_a, pos_b):
    return transmission_delay(data_bits, distance(pos_a, pos_b), V2V_LINK)


def v2i_delay(data_bits, veh_pos, rsu_pos):
    return transmission_delay(data_bits, distance(veh_pos, rsu_pos), V2I_LINK)


# ---------- 連線可續持多久（residence / contact time）----------
def contact_time(pos_a, vel_a, pos_b, vel_b, comm_range, horizon=60.0):
    """
    在目前的相對運動下，兩個東西還能保持在 comm_range 內多久(秒)。
    解 |相對位置 + 相對速度 × t| = comm_range 的正根(離開範圍的時刻)。

    - 現在就不在範圍內 → 回傳 0
    - 相對靜止且在範圍內(例如車停著、或對基站但速度抵消) → 回傳上限 horizon
    - 否則回傳「離開範圍的時間」，並以 horizon 封頂
    這個值讓決策能避免「把任務丟給快要離開的對象」。
    """
    px, py = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
    vx, vy = vel_b[0] - vel_a[0], vel_b[1] - vel_a[1]

    C = px * px + py * py - comm_range * comm_range
    if C > 0:
        return 0.0                      # 現在就超出範圍

    A = vx * vx + vy * vy
    if A == 0:
        return horizon                  # 相對靜止且在範圍內 → 視為持續

    B = 2 * (px * vx + py * vy)
    disc = B * B - 4 * A * C
    if disc < 0:
        return horizon
    sq = math.sqrt(disc)
    t_exit = max((-B - sq) / (2 * A), (-B + sq) / (2 * A))   # 較大正根 = 離開時刻
    if t_exit <= 0:
        return 0.0
    return min(t_exit, horizon)


# ---------- 找範圍內的鄰居 / 基站 ----------
def neighbors_in_range(ego_pos, others, comm_range=V2V_RANGE_M):
    """
    others: dict { id -> (x,y) }。回傳範圍內的 id 清單(依距離由近到遠)。
    """
    found = [(oid, distance(ego_pos, opos))
             for oid, opos in others.items()
             if distance(ego_pos, opos) <= comm_range]
    found.sort(key=lambda p: p[1])
    return [oid for oid, _ in found]


def rsus_in_range(veh_pos, rsus, comm_range=RSU_RANGE_M):
    """
    rsus: dict { rsu_id -> {'x':..,'y':..} }(就是 infra_config.load_rsus 的格式)。
    回傳範圍內的 rsu_id 清單(依距離由近到遠)。
    """
    found = []
    for rid, info in rsus.items():
        d = distance(veh_pos, (info["x"], info["y"]))
        if d <= comm_range:
            found.append((rid, d))
    found.sort(key=lambda p: p[1])
    return [rid for rid, _ in found]


# ==================================================================
# 自我測試：直接執行 `python comm_model.py` 就會跑，不需要開 SUMO
# ==================================================================
if __name__ == "__main__":
    print("=== 通訊模型自我測試 ===\n")

    TASK_BITS = 5_000_000   # 一筆 5 Mbit(約 0.6 MB)的任務

    print("[1] 速率與延遲隨距離變化（V2I 車對基站，鏈路預算 23dBm/20MHz/5.9GHz）：")
    for d in (10, 50, 100, 150, 200, 250):
        r = data_rate(d, V2I_LINK)
        delay = transmission_delay(TASK_BITS, d, V2I_LINK)
        delay_str = f"{delay*1000:.1f} ms" if delay != float("inf") else "超出範圍(∞)"
        print(f"    距離 {d:>3} m → 速率 {r/1e6:6.2f} Mbps，傳 5Mbit 約 {delay_str}")

    print("\n[2] 連線判斷：")
    print(f"    車(0,0) 與 車(100,0) 能 V2V 嗎？ {can_communicate((0,0),(100,0),V2V_RANGE_M)}（範圍150m內，應為True）")
    print(f"    車(0,0) 與 車(200,0) 能 V2V 嗎？ {can_communicate((0,0),(200,0),V2V_RANGE_M)}（超過150m，應為False）")

    print("\n[3] 連線可續持多久(contact_time)：")
    pa, va = (0, 0), velocity_from_speed_angle(15, 90)   # 朝東 15 m/s
    pb, vb = (50, 0), velocity_from_speed_angle(15, 90)  # 同向同速
    print(f"    兩車同向同速、相距50m → {contact_time(pa,va,pb,vb,V2V_RANGE_M):.1f} s（應為上限60）")
    pb2, vb2 = (50, 0), velocity_from_speed_angle(15, 270)  # 朝西(相向)
    print(f"    兩車相向而行、相距50m → {contact_time(pa,va,pb2,vb2,V2V_RANGE_M):.1f} s（應為較短時間）")
    rsu_pos, rsu_vel = (100, 0), (0.0, 0.0)
    print(f"    車朝東駛離基站、距基站100m → {contact_time(pa,va,rsu_pos,rsu_vel,RSU_RANGE_M):.1f} s（車遠離，會在某刻離開200m範圍）")

    print("\n[4] 找範圍內鄰車：")
    others = {"v1": (30, 0), "v2": (120, 0), "v3": (300, 0)}
    print(f"    ego(0,0) 的 V2V 鄰車(<150m，由近到遠)：{neighbors_in_range((0,0), others)}（應為 v1, v2）")

    print("\n[5] 邊界：超出範圍傳輸延遲應為無限大：")
    print(f"    距離 500m 傳 5Mbit：{transmission_delay(TASK_BITS, 500, V2I_LINK)}（應為 inf）")

    print("\n=== 測試結束（數字合理即代表模組正常）===")
