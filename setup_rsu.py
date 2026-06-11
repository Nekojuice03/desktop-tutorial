"""
自動放置基站(RSU) —— 貪婪最大覆蓋 + 路邊偏移版。
在「你的電腦」上跑（需要有路網檔，且已安裝 sumo）。

策略：
  1. 沿所有道路取樣
  2. 用「貪婪最大覆蓋」選位置：第一個基站會落在最密集處(通常主路口)，
     後續去補最長的道路
  3. 把選到的點沿道路垂直方向推到「路邊」(半路寬 + margin)，
     符合真實 RSU 裝在路肩/桿上，而非車道中央

輸出：
  - rsu.add.xml        → 給 SUMO 看（地圖上藍點，純標記）
  - rsu_positions.json → 給 Python 程式讀座標

執行：python setup_rsu.py
"""
import os
import glob
import json
import sys
import math
import sumolib

# ===== 自動切換到此腳本所在資料夾 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"工作目錄：{SCRIPT_DIR}\n")

# ===== 可調參數 =====
OUT_ADD      = "rsu.add.xml"
OUT_JSON     = "rsu_positions.json"
NUM_RSU      = 3      # 想放幾個基站
COVER_RADIUS = 200    # 評估覆蓋用半徑(公尺)。★建議與 infra_config.py 的 RSU_RANGE_M 一致
SAMPLE_STEP  = 40     # 沿道路取樣間距(公尺)
ROADSIDE_MARGIN = 5   # 推到路邊時，超出車道外緣的距離(公尺)


def find_net_file():
    for name in ("osm.net.xml.gz", "osm.net.xml"):
        if os.path.exists(name):
            return name
    hits = glob.glob("*.net.xml*")
    return hits[0] if hits else None


def sample_edge(edge, step, margin):
    """
    沿一條道路每隔 step 公尺取一點。
    每點回傳 (x, y, px, py, off)：
      x,y    = 道路中心線上的點(算覆蓋用)
      px,py  = 垂直道路的單位向量(往路邊推的方向)
      off    = 要推出去的距離 = 半路寬 + margin
    """
    shape = edge.getShape()
    if len(shape) < 2:
        return []
    lanes = edge.getLaneNumber()
    lane_w = edge.getLanes()[0].getWidth() if edge.getLanes() else 3.2
    off = lanes * lane_w / 2.0 + margin
    pts = []
    leftover = 0.0
    for i in range(len(shape) - 1):
        ax, ay = shape[i]
        bx, by = shape[i + 1]
        seg = math.dist((ax, ay), (bx, by))
        if seg == 0:
            continue
        tx, ty = (bx - ax) / seg, (by - ay) / seg   # 單位切向
        px, py = -ty, tx                            # 單位法向(垂直道路)
        pos = leftover
        while pos <= seg:
            t = pos / seg
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            pts.append((x, y, px, py, off))
            pos += step
        leftover = pos - seg
    return pts


# ===== 讀路網 =====
NET_FILE = find_net_file()
if NET_FILE is None:
    print("找不到路網檔（*.net.xml 或 *.net.xml.gz）。此資料夾內容：")
    for fn in sorted(os.listdir(".")):
        print("   ", fn)
    sys.exit(1)

print(f"找到路網檔：{NET_FILE}")
net = sumolib.net.readNet(NET_FILE)

edges = [e for e in net.getEdges()
         if not e.getID().startswith(":") and e.allows("passenger")]
if not edges:
    print("找不到可放置的道路邊，請檢查路網。")
    sys.exit(1)

samples = []
for e in edges:
    samples.extend(sample_edge(e, SAMPLE_STEP, ROADSIDE_MARGIN))
coords = [(s[0], s[1]) for s in samples]    # 中心線座標(算覆蓋)
print(f"沿道路取樣 {len(samples)} 個點，開始放置基站...")

# ===== 貪婪最大覆蓋（用中心線座標評估）=====
covered = [False] * len(coords)
placed = []
for _ in range(NUM_RSU):
    best_gain, best_i = -1, -1
    for ci, (cx, cy) in enumerate(coords):
        gain = sum(
            1 for j, (sx, sy) in enumerate(coords)
            if not covered[j] and math.dist((cx, cy), (sx, sy)) <= COVER_RADIUS
        )
        if gain > best_gain:
            best_gain, best_i = gain, ci
    if best_i < 0 or best_gain <= 0:
        break
    cx, cy = coords[best_i]
    for j, (sx, sy) in enumerate(coords):
        if not covered[j] and math.dist((cx, cy), (sx, sy)) <= COVER_RADIUS:
            covered[j] = True
    # 把選到的中心線點推到路邊
    sx, sy, px, py, off = samples[best_i]
    placed.append((sx + px * off, sy + py * off))

# ===== 輸出 =====
rsus = {}
with open(OUT_ADD, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<additional>\n')
    for i, (x, y) in enumerate(placed):
        rid = f"rsu_{i}"
        rsus[rid] = {"x": x, "y": y}
        f.write(f'    <poi id="{rid}" x="{x:.2f}" y="{y:.2f}" '
                f'type="RSU" color="0,0,255" layer="200"/>\n')
    f.write('</additional>\n')

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(rsus, f, ensure_ascii=False, indent=2)

coverage = sum(covered) / len(coords) * 100 if coords else 0
print(f"\n已放置 {len(placed)} 個基站（覆蓋半徑 {COVER_RADIUS}m，已推到路邊）：")
for i, (x, y) in enumerate(placed):
    print(f"  rsu_{i} @ 座標({x:.1f}, {y:.1f})")
print(f"道路覆蓋率：約 {coverage:.0f}%")
if coverage < 60:
    print("（覆蓋率偏低：可調大 COVER_RADIUS 或增加 NUM_RSU）")
print(f"\n輸出：{OUT_ADD}（給SUMO看）、{OUT_JSON}（給Python讀）")