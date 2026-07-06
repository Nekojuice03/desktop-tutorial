"""
自動放置基站(RSU) —— 貪婪最大覆蓋 + 路邊偏移版（自適應任意大小路網）。
在「你的電腦」上跑（需要有路網檔，且已安裝 sumo）。

策略：
  1. 沿所有道路取樣
  2. 用「貪婪最大覆蓋」選位置：第一個基站落在最密集處(通常主路口)，後續去補最長的道路
  3. 把選到的點沿道路垂直方向推到「路邊」(半路寬 + margin)，符合真實 RSU 裝在路肩/桿上

★自適應(換路網不會炸)：
  - RSU 數量「不寫死」：預設依覆蓋率自動決定 —— 一直放到道路覆蓋率達 TARGET_COVERAGE，
    或新基站的邊際貢獻過低為止（有 MAX_RSU 安全上限）。小路網放幾個、大路網自動放更多。
  - 覆蓋半徑「自動同步」infra_config.RSU_RANGE_M，避免兩邊參數飄移。
  - 取樣間距「自動加粗」：大路網用較大 SAMPLE_STEP，避免貪婪 O(R·N²) 太慢。

執行：
  python setup_rsu.py                 # 自適應(依覆蓋率自動決定 RSU 數)
  python setup_rsu.py --num 8         # 強制剛好放 8 個
  python setup_rsu.py --target 0.95   # 自適應，覆蓋率目標改 95%
  python setup_rsu.py --radius 250 --step 30   # 覆寫覆蓋半徑 / 取樣間距
"""
import os
import glob
import json
import sys
import math
import argparse
import sumolib

# ===== 自動切換到此腳本所在資料夾 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"工作目錄：{SCRIPT_DIR}\n")

# 覆蓋半徑預設值：與通訊模型同步（單一事實來源），找不到就退回 200
try:
    from infra_config import RSU_RANGE_M as _DEFAULT_RADIUS
except Exception:
    _DEFAULT_RADIUS = 200

# ===== 預設參數（可被 CLI 覆寫）=====
OUT_ADD          = "rsu.add.xml"
OUT_JSON         = "rsu_positions.json"
TARGET_COVERAGE  = 0.90    # 自適應停止條件：道路覆蓋率達此比例即停
MIN_GAIN_FRAC    = 0.01    # 自適應停止條件：新基站覆蓋 < 全部取樣點的此比例就停(邊際太低)
MAX_RSU          = 80      # 安全上限，避免極端大圖無限放
ROADSIDE_MARGIN  = 5       # 推到路邊時，超出車道外緣的距離(公尺)


def parse_args():
    p = argparse.ArgumentParser(description="自適應 RSU 佈點")
    p.add_argument("--net", type=str, default=None,
                   help="指定路網檔(如 new.net.xml)；不給則自動搜尋")
    p.add_argument("--mode", choices=["greedy", "junction"], default="greedy",
                   help="greedy=純貪婪最大覆蓋；junction=先佈號誌路口四角、"
                        "再對長路路肩補洞(貼近真實 RSU 部署)")
    p.add_argument("--corners", type=int, default=2,
                   help="junction 模式：每個主路口佈幾個角(預設 2，最多=臂數)")
    p.add_argument("--junctions", type=str, default=None,
                   help="junction 模式：逗號分隔的路口 id；不給=所有號誌路口")
    p.add_argument("--plot", action="store_true",
                   help="輸出佈點圖 rsu_layout.png(路網+RSU+覆蓋圈)")
    p.add_argument("--num", type=int, default=None,
                   help="強制剛好放 N 個(給定時關閉自適應；僅 greedy 模式)")
    p.add_argument("--target", type=float, default=TARGET_COVERAGE,
                   help=f"自適應覆蓋率目標(預設 {TARGET_COVERAGE})")
    p.add_argument("--radius", type=float, default=_DEFAULT_RADIUS,
                   help=f"覆蓋半徑(公尺，預設取 infra_config.RSU_RANGE_M={_DEFAULT_RADIUS})")
    p.add_argument("--step", type=float, default=None,
                   help="沿道路取樣間距(公尺)；不給則依路網大小自動決定")
    p.add_argument("--max-rsu", type=int, default=MAX_RSU, help="RSU 數量安全上限")
    return p.parse_args()


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


def auto_step(total_len, default=40.0):
    """
    依總道路長度自動決定取樣間距：大路網用較粗的取樣，
    把貪婪覆蓋的成本(約 O(R·N²))壓在可接受範圍。
    目標：取樣點數約 ≤ 1500。
    """
    if total_len <= 0:
        return default
    step = max(default, total_len / 1500.0)
    return round(step)


def _wrapa(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def place_junction_corners(net, n_corners, junction_ids, margin):
    """
    在「主路口」的路邊四角佈 RSU(貼近真實部署：RSU 裝在路口號誌桿/路角)。
    - 主路口 = 號誌路口(或 --junctions 指定)。
    - 對每個路口：由進出邊的方位角找出各「臂」，相鄰兩臂的角平分線即「路角」
      方向；沿該方向推出 路口半徑+margin 得到路角座標。
    - 每路口挑 n_corners 個彼此角度最分散的角(2 個≈對角佈署)。
    回傳 [(x, y, 路口id), ...]
    """
    if junction_ids:
        nodes = [net.getNode(j.strip()) for j in junction_ids.split(",")]
    else:
        nodes = [n for n in net.getNodes() if "traffic_light" in n.getType()]
    placed = []
    for nd in nodes:
        cx, cy = nd.getCoord()
        shape = nd.getShape() or []
        rad = max((math.dist((cx, cy), p) for p in shape), default=10.0)
        off = rad + margin
        # 各臂方位角(從路口指出去)
        angs = []
        for e in nd.getOutgoing():
            sh = e.getShape()
            if len(sh) >= 2:
                angs.append(math.atan2(sh[1][1] - sh[0][1], sh[1][0] - sh[0][0]))
        for e in nd.getIncoming():
            sh = e.getShape()
            if len(sh) >= 2:
                angs.append(math.atan2(sh[-2][1] - sh[-1][1], sh[-2][0] - sh[-1][0]))
        arms = []
        for a in sorted(angs):   # 15 度內視為同一臂
            if not arms or min(abs(_wrapa(a - b)) for b in arms) > math.radians(15):
                arms.append(a)
        if len(arms) < 2:
            continue
        arms.sort()
        corners = []
        for i in range(len(arms)):   # 相鄰臂的角平分線 = 路角
            a1, a2 = arms[i], arms[(i + 1) % len(arms)]
            gap = (a2 - a1) % (2 * math.pi)
            corners.append(_wrapa(a1 + gap / 2))
        chosen = [corners[0]]        # 貪婪挑彼此最分散的 n 個角(2個≈對角)
        while len(chosen) < min(n_corners, len(corners)):
            rest = [c for c in corners if c not in chosen]
            chosen.append(max(rest, key=lambda c: min(abs(_wrapa(c - x)) for x in chosen)))
        for c in chosen:
            placed.append((cx + off * math.cos(c), cy + off * math.sin(c), nd.getID()))
    return placed


def mark_covered(covered, coords, x, y, r2):
    for j, (sx, sy) in enumerate(coords):
        if not covered[j]:
            dx, dy = x - sx, y - sy
            if dx * dx + dy * dy <= r2:
                covered[j] = True


def greedy_cover(coords, samples, radius, target, min_gain_frac, max_rsu, force_num,
                 covered=None):
    """
    貪婪最大覆蓋。用平方距離(免開根號)加速。
    force_num 指定時剛好放該數量；否則放到覆蓋率達 target 或邊際貢獻過低。
    covered 可傳入「預佈點已蓋掉」的初始狀態(junction 模式的路肩補洞用)。
    回傳 (placed[(x,y)...], covered[bool...])。
    """
    n = len(coords)
    r2 = radius * radius
    covered = [False] * n if covered is None else covered
    placed = []
    min_gain = max(1, int(min_gain_frac * n))   # 自適應的「邊際太低」門檻

    limit = force_num if force_num else max_rsu
    for _ in range(limit):
        if not force_num and n and sum(covered) / n >= target:
            print(f"  → 覆蓋率已達標({target*100:.0f}%)，停止。")
            break
        best_gain, best_i = 0, -1
        for ci in range(n):
            cx, cy = coords[ci]
            gain = 0
            for j in range(n):
                if covered[j]:
                    continue
                dx = cx - coords[j][0]
                dy = cy - coords[j][1]
                if dx * dx + dy * dy <= r2:
                    gain += 1
            if gain > best_gain:
                best_gain, best_i = gain, ci
        if best_i < 0 or best_gain <= 0:
            break
        cx, cy = coords[best_i]
        for j in range(n):
            if not covered[j]:
                dx = cx - coords[j][0]
                dy = cy - coords[j][1]
                if dx * dx + dy * dy <= r2:
                    covered[j] = True
        sx, sy, px, py, off = samples[best_i]   # 推到路邊
        placed.append((sx + px * off, sy + py * off))

        cov = sum(covered) / n
        print(f"  放第 {len(placed)} 個：新增覆蓋 {best_gain} 點，"
              f"累積覆蓋率 {cov*100:.0f}%")
        if not force_num:   # 自適應停止條件
            if cov >= target:
                print(f"  → 覆蓋率達標({target*100:.0f}%)，停止。")
                break
            if best_gain < min_gain:
                print(f"  → 邊際貢獻過低(<{min_gain}點)，停止。")
                break
    return placed, covered


def main():
    args = parse_args()

    NET_FILE = args.net or find_net_file()
    if NET_FILE and not os.path.isfile(NET_FILE):
        print(f"指定的路網檔不是有效檔案：{NET_FILE}（注意舊專案的 osm.net.xml 是資料夾）")
        sys.exit(1)
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

    total_len = sum(e.getLength() for e in edges)
    step = args.step if args.step else auto_step(total_len)
    radius = args.radius

    # 路網規模摘要 + 不給 --num 時的 RSU 數量先驗估計(僅供參考，實際由覆蓋率決定)
    xmin, ymin, xmax, ymax = net.getBoundary()
    area_km2 = max(0.0, (xmax - xmin) * (ymax - ymin)) / 1e6
    print(f"路網規模：{len(edges)} 條 edge，總長約 {total_len/1000:.1f} km，"
          f"範圍約 {area_km2:.2f} km²")
    print(f"覆蓋半徑 {radius:.0f} m（同步 infra_config）、取樣間距 {step:.0f} m\n")

    samples = []
    for e in edges:
        samples.extend(sample_edge(e, step, ROADSIDE_MARGIN))
    coords = [(s[0], s[1]) for s in samples]
    if not coords:
        print("取樣點為 0，請檢查路網或調小 --step。")
        sys.exit(1)

    placed, tags = [], []
    covered = [False] * len(coords)
    if args.mode == "junction":
        # 第一階段：主路口(號誌)的路邊四角，貼近真實 RSU 部署位置
        cor = place_junction_corners(net, args.corners, args.junctions,
                                     ROADSIDE_MARGIN + 5.0)
        r2 = radius * radius
        print(f"[路口四角] 佈 {len(cor)} 個(每路口 {args.corners} 角)：")
        for (x, y, jid) in cor:
            placed.append((x, y))
            tags.append("corner")
            mark_covered(covered, coords, x, y, r2)
            print(f"  角落 @ ({x:6.1f},{y:6.1f})  路口 {jid}")
        cov0 = sum(covered) / len(coords)
        print(f"  → 路口四角後覆蓋率 {cov0*100:.0f}%，開始長路路肩補洞...")

    mode_txt = (f"junction(路口{args.corners}角+路肩補洞)" if args.mode == "junction"
                else (f"強制 {args.num} 個" if args.num
                      else f"greedy 自適應(目標 {args.target*100:.0f}%)"))
    print(f"沿道路取樣 {len(coords)} 個點，放置模式：{mode_txt}")

    fill, covered = greedy_cover(
        coords, samples, radius, args.target, MIN_GAIN_FRAC, args.max_rsu,
        args.num if args.mode == "greedy" else None, covered=covered)
    placed += fill
    tags += ["shoulder"] * len(fill)

    # ===== 輸出 =====
    if not tags:
        tags = ["shoulder"] * len(placed)
    rsus = {}
    with open(OUT_ADD, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<additional>\n')
        for i, (x, y) in enumerate(placed):
            rid = f"rsu_{i}"
            rsus[rid] = {"x": x, "y": y, "tag": tags[i]}
            color = "255,140,0" if tags[i] == "corner" else "0,0,255"
            f.write(f'    <poi id="{rid}" x="{x:.2f}" y="{y:.2f}" '
                    f'type="RSU_{tags[i]}" color="{color}" layer="200"/>\n')
        f.write('</additional>\n')

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rsus, f, ensure_ascii=False, indent=2)

    coverage = sum(covered) / len(coords) * 100 if coords else 0
    print(f"\n已放置 {len(placed)} 個基站（覆蓋半徑 {radius:.0f}m）：")
    for i, ((x, y), tg) in enumerate(zip(placed, tags)):
        print(f"  rsu_{i} [{'路口角' if tg=='corner' else '路肩'}] @ ({x:.1f}, {y:.1f})")
    print(f"道路覆蓋率：約 {coverage:.0f}%")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 9))
        for e in edges:   # 路網
            sh = e.getShape()
            ax.plot([p[0] for p in sh], [p[1] for p in sh],
                    color="#b0bec5", lw=max(1, e.getLaneNumber()), zorder=1)
        for (x, y), tg in zip(placed, tags):   # 覆蓋圈 + RSU
            ax.add_patch(plt.Circle((x, y), radius, color="#42a5f5", alpha=0.10, zorder=2))
        for (x, y), tg in zip(placed, tags):
            ax.scatter(x, y, s=90, zorder=3, edgecolor="black",
                       color=("#ff8f00" if tg == "corner" else "#1565c0"),
                       marker=("s" if tg == "corner" else "o"))
        ax.scatter([], [], s=90, color="#ff8f00", marker="s", edgecolor="black",
                   label="RSU (intersection corner)")
        ax.scatter([], [], s=90, color="#1565c0", marker="o", edgecolor="black",
                   label="RSU (road shoulder)")
        ax.set_aspect("equal")
        ax.set_title(f"RSU deployment — {len(placed)} units, "
                     f"r={radius:.0f}m, coverage {coverage:.0f}%")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("rsu_layout.png", dpi=150)
        print("佈點圖已存 rsu_layout.png")
    if coverage < args.target * 100 - 1:
        print("（未達目標覆蓋率：可能受 MAX_RSU 上限限制，或調大 --radius / --max-rsu）")
    print(f"\n輸出：{OUT_ADD}（給SUMO看）、{OUT_JSON}（給Python讀）")
    print("提醒：換路網後 RSU 數會改變 → 全域狀態維度改變 → MAPPO/DQN 必須重新訓練。")


if __name__ == "__main__":
    main()
