"""
真實車流引進(make_real_flow.py)—— 把台北市 VD 車流接上任意新路網
====================================================================
把舊專案「手工對應」的管線自動化，三步一次完成：

  1. 抓「VD 靜態清單」(含經緯度)，自動篩出落在路網範圍內的偵測器
  2. 每個 VD 自動對應到最近的 SUMO edge(含車道) → 產出 vd_sumo_mapping.csv
     (同時產出 vd_debug.add.xml：在 sumo-gui 顯示 VD 位置，方便人工校對)
  3. 讀「VD 即時/歷史資料」，★只取 Svolume(小型客車)★ 轉成 SUMO flow
     → real_traffic_hep.rou.xml(vType=passenger)

用法(在你電腦、可連台北開放資料的網路)：
  python make_real_flow.py --net hepingeast2.net.xml                # 全自動(現抓即時資料)
  python make_real_flow.py --net hepingeast2.net.xml --xml traffic_data/VD_20260505_131323.xml
                                                                    # 用歷史檔(離線可跑)
  python make_real_flow.py --net hepingeast2.net.xml --static my_vd.xml   # 靜態清單用本地檔
  python make_real_flow.py --net hepingeast2.net.xml --map-only    # 只做對應，先人工校對

產出：
  vd_sumo_mapping.csv     DeviceID,LaneNO,SumoEdgeID,departLane(可手動修正後重跑)
  vd_debug.add.xml        VD 位置 POI(sumo-gui 校對用)
  real_traffic_hep.rou.xml  小型客車 flow(vehsPerHour = Svolume×12，5分鐘→每小時)

注意：
  - VD 靜態清單無「方向」欄位 → 最近邊對應可能選到對向邊。請務必用
    sumo-gui 開 vd_debug.add.xml 校對一次，錯的直接改 CSV 再重跑本腳本
    (已有 CSV 時自動沿用，不會覆蓋你的手工修正；--remap 可強制重對應)。
  - 出口分配：路網範圍太小、無真實轉向資料時，把每條進流平均分配到
    「可達的邊界出口」。之後有轉向比資料可換 turning_ratio 機制。
"""
import argparse
import csv
import gzip
import math
import os
import sys
import xml.etree.ElementTree as ET

import requests
import sumolib
try:
    import pyproj  # noqa: F401  經緯度→路網座標需要(pip install pyproj)
except ImportError:
    print("[缺套件] 請先安裝：pip install pyproj  (經緯度→UTM 轉換用)")
    sys.exit(1)

STATIC_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
LIVE_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVDDATA.xml.gz"
UA = {"User-Agent": "NTUT_Research_Traffic_Collector/1.0"}
MAPPING_CSV = "vd_sumo_mapping.csv"
DEBUG_ADD = "vd_debug.add.xml"
OUT_ROUTE = "real_traffic_hep.rou.xml"


def fetch_gz(url, timeout=30):
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    data = r.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="ignore")


# ---------- 1. 靜態清單：找出範圍內的 VD ----------
def parse_static_vds(text):
    """
    解析 VD 位置清單。優先支援台北交控中心 GetVD.xml 的真實 schema：
      <vd:SectionData><vd:SectionId>ZVCGQ40</vd:SectionId>
        <vd:StartWgsX>121.50</..><vd:StartWgsY>25.12</..>
        <vd:EndWgsX>…</..><vd:EndWgsY>…</..></vd:SectionData>
    位置取路段起訖點的「中點」。解析不到再退回泛用啟發式。
    """
    if not text.lstrip().startswith("<?xml"):
        text = '<?xml version="1.0" encoding="utf-8"?>\n<root>\n' + text + "\n</root>"
    root = ET.fromstring(text)

    def fnum(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    # ---- 格式A：SectionData + Start/End WGS84(實測台北 GetVD.xml)----
    vds = {}
    for sec in root.iter():
        if not sec.tag.split("}")[-1] == "SectionData":
            continue
        sid = sx = sy = ex = ey = None
        for child in sec:
            tag = child.tag.split("}")[-1]
            val = (child.text or "").strip()
            if tag == "SectionId":
                sid = val
            elif tag == "StartWgsX":
                sx = fnum(val)
            elif tag == "StartWgsY":
                sy = fnum(val)
            elif tag == "EndWgsX":
                ex = fnum(val)
            elif tag == "EndWgsY":
                ey = fnum(val)
        if sid and sx and sy:
            lon = (sx + ex) / 2 if ex else sx     # 路段中點
            lat = (sy + ey) / 2 if ey else sy
            if 118.0 < lon < 124.0 and 21.0 < lat < 26.5:
                vds[sid] = (lon, lat)
    if vds:
        return vds

    # ---- 格式B：泛用啟發式(DeviceID/VDID + 台灣範圍內的經緯度欄位)----
    for elem in root.iter():
        dev = None
        lon = lat = None
        for child in elem:
            tag = child.tag.split("}")[-1].lower()
            val = (child.text or "").strip()
            if tag in ("deviceid", "vdid", "id") and val and dev is None:
                dev = val
            else:
                x = fnum(val)
                if x is None:
                    continue
                if 118.0 < x < 124.0:
                    lon = x
                elif 21.0 < x < 26.5:
                    lat = x
        if dev and lon is not None and lat is not None:
            vds[dev] = (lon, lat)
    return vds


def vds_in_net(vds, net, margin=30.0):
    """篩出落在路網邊界(外擴 margin 公尺)內的 VD，回傳 {dev:(x,y,lon,lat)}(net座標)。"""
    xmin, ymin, xmax, ymax = net.getBoundary()
    keep = {}
    for dev, (lon, lat) in vds.items():
        x, y = net.convertLonLat2XY(lon, lat)
        if xmin - margin <= x <= xmax + margin and ymin - margin <= y <= ymax + margin:
            keep[dev] = (x, y, lon, lat)
    return keep


# ---------- 2. VD → edge 對應 ----------
def map_vd_to_edges(net, vd_xy, live_lanes, search_r=60.0):
    """
    每台 VD 對到最近的可通行 edge；每個實際回報的 LaneNO 給一列
    (departLane 夾在該 edge 車道數內)。回傳 list of dict(CSV 列)。
    """
    rows = []
    for dev, (x, y, lon, lat) in sorted(vd_xy.items()):
        cands = net.getNeighboringEdges(x, y, search_r)
        cands = [(e, d) for e, d in cands
                 if e.allows("passenger") and not e.getID().startswith(":")]
        if not cands:
            print(f"  [略過] {dev}：{search_r}m 內無可通行 edge")
            continue
        cands.sort(key=lambda p: p[1])
        edge = cands[0][0]
        n_lane = edge.getLaneNumber()
        lanes = sorted(live_lanes.get(dev, {0}))
        for ln in lanes:
            rows.append({"DeviceID": dev, "LaneNO": ln,
                         "SumoEdgeID": edge.getID(),
                         "departLane": min(int(ln), n_lane - 1)})
        print(f"  {dev} @({x:.0f},{y:.0f}) → edge {edge.getID()} "
              f"(距離 {cands[0][1]:.0f}m，{n_lane} 車道，VD 車道 {lanes})")
    return rows


# ---------- 3. 即時/歷史資料：只取小型客車 Svolume ----------
def parse_live_svolume(text):
    """回傳 {DeviceID: {LaneNO: Svolume}} 與 {DeviceID: TimeInterval分}。"""
    if not text.lstrip().startswith("<?xml"):
        text = '<?xml version="1.0" encoding="utf-8"?>\n<root>\n' + text + "\n</root>"
    root = ET.fromstring(text)
    sv, ti = {}, {}
    for devel in root.iter():
        if not devel.tag.endswith("VDDevice"):
            continue
        dev = (devel.findtext("DeviceID") or "").strip()
        if not dev:
            continue
        ti[dev] = float(devel.findtext("TimeInterval") or 5.0)
        for lane in devel.findall("LaneData"):
            ln = int(float(lane.findtext("LaneNO") or 0))
            s = float(lane.findtext("Svolume") or 0)     # ★小型客車
            sv.setdefault(dev, {})[ln] = s
    return sv, ti


# ---------- 4. 產生 flow(出口=可達的邊界出口，均分) ----------
def fringe_exits(net):
    return [e for e in net.getEdges()
            if e.allows("passenger") and not e.getID().startswith(":")
            and len(e.getOutgoing()) == 0]


def write_routes(net, mapping_rows, sv, ti, out=OUT_ROUTE, end=3600):
    exits = fringe_exits(net)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
             'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
             '',
             '    <!-- 小型客車(VD Svolume)；server/client 角色由 Python 端指派 -->',
             '    <vType id="passenger" vClass="passenger"/>', '']
    fid = 0
    skipped = 0
    for row in mapping_rows:
        dev, ln = row["DeviceID"], int(row["LaneNO"])
        if not net.hasEdge(row.get("SumoEdgeID", "")):
            skipped += 1        # 對應表殘留別路網的 edge → 跳過不 crash
            continue
        s = sv.get(dev, {}).get(ln)
        if s is None or s <= 0:
            skipped += 1
            continue
        vph = s * (60.0 / ti.get(dev, 5.0))     # 5分鐘量 → 每小時
        src = net.getEdge(row["SumoEdgeID"])
        dests = [x for x in exits
                 if x.getID() != src.getID()
                 and net.getShortestPath(src, x)[0] is not None]
        if not dests:
            skipped += 1
            continue
        share = vph / len(dests)                # 無轉向資料 → 均分可達出口
        for d in dests:
            if share < 1.0:
                continue
            lines.append(
                f'    <flow id="f{fid}" type="passenger" from="{src.getID()}" '
                f'to="{d.getID()}" begin="0" end="{end}" vehsPerHour="{share:.1f}" '
                f'departLane="{row["departLane"]}" departSpeed="max"/>')
            fid += 1
    lines += ['', '</routes>', '']
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n已產生 {out}：{fid} 條 flow(僅小型客車)"
          f"{f'，{skipped} 筆無量/不可達已略過' if skipped else ''}")
    return fid


def write_debug_pois(vd_xy, path=DEBUG_ADD):
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
        for dev, (x, y, lon, lat) in vd_xy.items():
            f.write(f'    <poi id="VD_{dev}" x="{x:.2f}" y="{y:.2f}" '
                    f'type="VD" color="255,0,255" layer="210"/>\n')
        f.write('</additional>\n')
    print(f"VD 位置圖已存 {path}(sumo-gui 加此檔可校對粉紅點)")


def main():
    p = argparse.ArgumentParser(description="把台北 VD 真實車流(僅小型客車)接上新路網")
    p.add_argument("--net", default="hepingeast2.net.xml")
    p.add_argument("--static", default=None, help="VD 靜態清單本地檔(不給則線上抓)")
    p.add_argument("--xml", default=None, help="VD 即時資料本地檔(不給則線上抓)")
    p.add_argument("--map-only", action="store_true", help="只做 VD→edge 對應，不產車流")
    p.add_argument("--remap", action="store_true", help="忽略既有 CSV 強制重新對應")
    p.add_argument("--end", type=int, default=3600, help="flow 結束秒數")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    net = sumolib.net.readNet(args.net)
    print(f"路網 {args.net}：邊界 {tuple(round(v) for v in net.getBoundary())}")

    # 即時/歷史車流(也用於得知每台 VD 實際有哪些車道)
    live_text = (open(args.xml, encoding="utf-8").read() if args.xml
                 else fetch_gz(LIVE_VD_URL))
    sv, ti = parse_live_svolume(live_text)
    print(f"車流資料：{len(sv)} 台 VD(全市)")

    # 對應表：已有「且屬於本路網」就沿用(保住手工修正)，否則自動建立。
    # ★關鍵防呆：舊專案的 CSV 是別的路網(如新生南路)做的，其 edge 不在
    #   本路網 → 自動視為需要重建，避免 KeyError。
    rows = None
    if os.path.exists(MAPPING_CSV) and not args.remap and not args.map_only:
        with open(MAPPING_CSV, encoding="utf-8-sig") as f:
            rows = [r for r in csv.DictReader(f)]
        n_valid = sum(1 for r in rows if net.hasEdge(r.get("SumoEdgeID", "")))
        if n_valid == 0:
            print(f"既有 {MAPPING_CSV}({len(rows)} 列)的 edge 全部不在本路網"
                  f"(應為舊路網的對應表) → 自動重新對應")
            rows = None
        else:
            if n_valid < len(rows):
                print(f"[警告] {MAPPING_CSV} 有 {len(rows)-n_valid} 列 edge 不在本路網，將略過")
            print(f"沿用既有 {MAPPING_CSV}({len(rows)} 列，{n_valid} 列有效)；"
                  f"要重建請加 --remap")
    if rows is None:
        static_text = (open(args.static, encoding="utf-8").read() if args.static
                       else fetch_gz(STATIC_VD_URL))
        vds = parse_static_vds(static_text)
        print(f"靜態清單：{len(vds)} 台 VD(全市)")
        if not vds:
            # 自我診斷：格式與預期不符 → 存檔並印出開頭，方便對照真實 schema
            with open("vd_static_dump.xml", "w", encoding="utf-8") as f:
                f.write(static_text)
            print("[診斷] 解析不到任何含座標的 VD。原始內容已存 vd_static_dump.xml，"
                  "開頭如下：")
            print(static_text[:800])
            sys.exit(1)
        vd_xy = vds_in_net(vds, net)
        print(f"落在本路網範圍內：{len(vd_xy)} 台 → 開始對應 edge")
        if not vd_xy:
            print("範圍內找不到 VD——請確認靜態清單有座標欄位，或路網範圍。")
            sys.exit(1)
        # 該站在即時資料查不到車道時，先給單一車道 0(之後診斷會顯示對不上的站)
        lanes_of = {d: (set(sv[d].keys()) if d in sv else {0}) for d in vd_xy}
        rows = map_vd_to_edges(net, vd_xy, lanes_of)
        with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["DeviceID", "LaneNO",
                                              "SumoEdgeID", "departLane"])
            w.writeheader()
            w.writerows(rows)
        print(f"對應表已存 {MAPPING_CSV}(共 {len(rows)} 列)")
        write_debug_pois(vd_xy)
        print("★請開 sumo-gui 校對：sumo-gui -n", args.net,
              "-a vd_debug.add.xml,rsu.add.xml")

    # ★關鍵診斷：靜態清單(SectionId)與即時車道資料(DeviceID)是兩套來源，
    #   確認範圍內的站有幾台真的查得到 Svolume(對不上=兩套 ID 需另接橋)。
    devs = {r["DeviceID"] for r in rows}
    have = sorted(d for d in devs if d in sv)
    print(f"對應表 {len(devs)} 站中，查得到即時車道資料(Svolume)者 {len(have)} 站"
          + (f"：{have}" if have else ""))
    if not have:
        print("[診斷] 範圍內各站 ID 在即時資料(GetVDDATA)中都查不到——"
              "靜態 SectionId 與即時 DeviceID 可能是兩套編號。請把本輸出貼回。")

    if args.map_only:
        return
    n = write_routes(net, rows, sv, ti, end=args.end)
    if n:
        print("下一步：把 osm.sumocfg 的 route-files 改成 real_traffic_hep.rou.xml，"
              "或執行 sumo-gui -n {} -r {} -a rsu.add.xml".format(args.net, OUT_ROUTE))


if __name__ == "__main__":
    main()
