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
    RSU_BANDWIDTH_HZ, V2V_BANDWIDTH_HZ,
    REF_DISTANCE_M, REF_SNR, PATHLOSS_EXPONENT,
)


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


# ---------- 資料速率（隨距離衰減）----------
def data_rate(d, bandwidth):
    """
    距離 d 公尺時的傳輸速率(bits/s)。
    用簡化 Shannon 公式：速率 = 頻寬 × log2(1 + 訊雜比)，
    訊雜比隨距離以 d^(-路徑損耗指數) 衰減 → 距離越遠越慢。
    """
    snr = REF_SNR * (REF_DISTANCE_M / max(d, REF_DISTANCE_M)) ** PATHLOSS_EXPONENT
    return bandwidth * math.log2(1 + snr)


# ---------- 傳輸延遲 ----------
def transmission_delay(data_bits, d, bandwidth, comm_range):
    """
    傳 data_bits 位元要花幾秒。
    超出通訊範圍 → 回傳無限大(代表傳不到)。
    """
    if d > comm_range:
        return float("inf")
    rate = data_rate(d, bandwidth)
    return data_bits / rate if rate > 0 else float("inf")


# 方便用的兩個包裝：直接指定是 V2V 還是車對基站
def v2v_delay(data_bits, pos_a, pos_b):
    return transmission_delay(data_bits, distance(pos_a, pos_b),
                              V2V_BANDWIDTH_HZ, V2V_RANGE_M)


def v2i_delay(data_bits, veh_pos, rsu_pos):
    return transmission_delay(data_bits, distance(veh_pos, rsu_pos),
                              RSU_BANDWIDTH_HZ, RSU_RANGE_M)


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

    print("[1] 速率與延遲隨距離變化（車對基站，頻寬 20MHz）：")
    for d in (10, 50, 100, 150, 200, 250):
        r = data_rate(d, RSU_BANDWIDTH_HZ)
        delay = transmission_delay(TASK_BITS, d, RSU_BANDWIDTH_HZ, RSU_RANGE_M)
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
    print(f"    距離 500m 傳 5Mbit：{transmission_delay(TASK_BITS, 500, RSU_BANDWIDTH_HZ, RSU_RANGE_M)}（應為 inf）")

    print("\n=== 測試結束（數字合理即代表模組正常）===")
