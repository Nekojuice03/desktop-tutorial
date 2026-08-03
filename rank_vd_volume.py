"""
VD 車流量排序(rank_vd_volume.py)—— 自動找出尖峰/離峰快照
============================================================
你的 VD 快照都是白天、無法靠時間分尖峰/離峰 → 改用『實測車流量』來分。

掃過 traffic_data/ 內所有 VD 檔，對每個檔計算會注入路網的小型客車量
(只累加『有對應到 SUMO 路網』的 VD 設備 Svolume，與 make_real_flow 的
實際注入一致)，排序後：
  - 量最少的 K 個 → 低車流(離峰代表)
  - 量最多的 K 個 → 高車流(尖峰代表)
並直接印出可貼上執行的 make_real_flow 與 run_density_compare 指令。

用法：
  python rank_vd_volume.py                       # 掃 traffic_data/，K=3
  python rank_vd_volume.py --dir traffic_data --k 3
  python rank_vd_volume.py --all-devices         # 不限路網，累加全部 VD(備援)
"""
import os
import csv
import glob
import argparse
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_CSV = "vd_sumo_mapping.csv"


def parse_live_svolume(text):
    """只取小型客車 Svolume（與 make_real_flow 同邏輯，但不依賴 sumolib）。
    回傳 {DeviceID: {LaneNO: Svolume}}。"""
    if not text.lstrip().startswith("<?xml"):
        text = '<?xml version="1.0" encoding="utf-8"?>\n<root>\n' + text + "\n</root>"
    root = ET.fromstring(text)
    sv = {}
    for devel in root.iter():
        if not devel.tag.endswith("VDDevice"):
            continue
        dev = (devel.findtext("DeviceID") or "").strip()
        if not dev:
            continue
        for lane in devel.findall("LaneData"):
            ln = int(float(lane.findtext("LaneNO") or 0))
            s = float(lane.findtext("Svolume") or 0)   # ★小型客車
            sv.setdefault(dev, {})[ln] = s
    return sv


def mapped_device_ids():
    """讀 vd_sumo_mapping.csv 內、實際被對應進路網的 DeviceID 集合。"""
    path = os.path.join(SCRIPT_DIR, MAPPING_CSV)
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = (row.get("DeviceID") or "").strip()
            if d:
                ids.add(d)
    return ids


def file_volume(path, keep_ids):
    """回傳 (注入路網小客車量, 命中設備數, 全部設備小客車量)。"""
    text = open(path, encoding="utf-8").read()
    sv = parse_live_svolume(text)
    total_all = sum(sum(lanes.values()) for lanes in sv.values())
    if keep_ids:
        hit = {d: lanes for d, lanes in sv.items() if d in keep_ids}
        total_mapped = sum(sum(lanes.values()) for lanes in hit.values())
        return total_mapped, len(hit), total_all
    return total_all, len(sv), total_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="traffic_data", help="VD xml 所在資料夾")
    ap.add_argument("--k", type=int, default=3, help="低/高各取幾個檔")
    ap.add_argument("--all-devices", action="store_true",
                    help="不限路網，累加全部 VD 設備(路網對應失效時的備援)")
    args = ap.parse_args()

    keep = set() if args.all_devices else mapped_device_ids()
    if keep:
        print(f"依 {MAPPING_CSV} 只計入 {len(keep)} 個路網內 VD 設備的小客車量。")
    else:
        print("未套用路網過濾(累加全部 VD 設備)。")

    files = sorted(glob.glob(os.path.join(args.dir, "*.xml")))
    if not files:
        print(f"在 {args.dir}/ 找不到任何 .xml，請確認路徑。")
        return

    rows = []
    for fp in files:
        try:
            vol, hit, all_vol = file_volume(fp, keep)
        except Exception as e:
            print(f"  ⚠ 略過 {os.path.basename(fp)}({e})")
            continue
        rows.append((os.path.basename(fp), vol, hit, all_vol, fp))

    if keep and all(r[2] == 0 for r in rows):
        print("⚠ 沒有任何檔命中路網內 VD 設備 —— VD 檔的 DeviceID 與 mapping 不同"
              "ID 系統。改用 --all-devices 重跑，或檢查 vd_sumo_mapping.csv。")
        return

    # 依注入量排序
    rows.sort(key=lambda r: r[1])
    print(f"\n共 {len(rows)} 檔，依『注入路網小客車量』由少到多:")
    print(f"  {'#':>3}  {'檔名':<28}{'注入量':>8}{'命中站':>7}{'全站量':>9}")
    for i, (name, vol, hit, all_vol, _) in enumerate(rows):
        print(f"  {i+1:>3}  {name:<28}{vol:>8.0f}{hit:>7}{all_vol:>9.0f}")

    k = min(args.k, len(rows) // 2) or 1
    low = rows[:k]
    high = rows[-k:]
    lo_v = sum(r[1] for r in low) / k
    hi_v = sum(r[1] for r in high) / k
    print(f"\n低車流(最少 {k} 檔，平均注入 {lo_v:.0f}):"
          + "  ".join(r[0] for r in low))
    print(f"高車流(最多 {k} 檔，平均注入 {hi_v:.0f}):"
          + "  ".join(r[0] for r in high))
    ratio = hi_v / lo_v if lo_v else float("inf")
    print(f"高/低 車流量比約 {ratio:.1f}× "
          + ("(差距明顯，適合做密度對比 ✔)" if ratio >= 1.5
             else "(差距偏小，對比效果可能不強 —— 可加大 K 或補收尖離峰更懸殊的資料)"))

    # 產生可貼上的指令
    print("\n" + "=" * 64)
    print("① 產生低/高車流路由檔(複製貼上):")
    lows, highs = [], []
    for i, (name, _v, _h, _a, fp) in enumerate(low, 1):
        out = f"real_traffic_low_d{i}.rou.xml"
        lows.append(out)
        print(f'python make_real_flow.py --net hepingeast2.net.xml '
              f'--xml {fp} --out {out}')
    for i, (name, _v, _h, _a, fp) in enumerate(high, 1):
        out = f"real_traffic_high_d{i}.rou.xml"
        highs.append(out)
        print(f'python make_real_flow.py --net hepingeast2.net.xml '
              f'--xml {fp} --out {out}')
    print("\n② 一鍵密度對比:")
    print(f'python run_density_compare.py --sumo --low {",".join(lows)} '
          f'--high {",".join(highs)} --seeds 3 --plot')


if __name__ == "__main__":
    main()
