"""
孿生保真度驗證(validate_twin_fidelity.py)—— 回答「你的孿生跟實體有多像」
==========================================================================
DT 論文的必考題:你宣稱「真實資料數位孿生」,證據是什麼?
本腳本用交通模擬校正的業界標準指標,量化 VD 實測流量與 SUMO 模擬流量的差距:

  GEH = sqrt( 2(M-C)^2 / (M+C) )      M=模擬小時流量, C=實測小時流量
  驗收慣例:GEH<5 的 link 佔比 >= 85% 視為校正合格(Highways England DMRB)
  另報 MAPE / RMSE / 需求實現度(rou.xml 指定量 vs 實際跑出量)

★ 兩個層級要分清楚(論文寫作關鍵):
  L1 需求實現度(demand realization):rou.xml 的 vehsPerHour vs 模擬通過量。
     落差來自插入失敗/壅塞/路徑選擇 —— 這是「模擬器有沒有照做」。
  L2 孿生保真度(twin fidelity):VD 實測量 vs 模擬通過量 → GEH。
     這是論文要的那張表。若驗證用的 VD 快照 = 產生車流的那份快照,
     這是「校正契合度(calibration fit)」而非獨立驗證;要做獨立驗證請用
     --vd-xml 指定「另一個時刻」的快照(見 --holdout 提示)。

★ 只有「實測站」列入主指標。mapping 中標註「鏡射」「代理」的列是建模假設
  (對向鏡射、以鄰站代理),拿它們算 GEH 是循環論證 —— 本腳本分開報告。

用法:
  # 用歷史快照離線驗證(建議:可重現)
  python validate_twin_fidelity.py --vd-xml GetVD_snapshot.xml
  # 現抓即時資料
  python validate_twin_fidelity.py --fetch
  # 已有 edgeData 輸出,只重算指標/重繪
  python validate_twin_fidelity.py --vd-xml snap.xml --no-run --plot

產出:twin_fidelity.csv / twin_fidelity.json / fig_twin_fidelity.png(--plot)
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVD.xml.gz"
LIVE_VD_URL = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVDDATA.xml.gz"
UA = {"User-Agent": "NTUT_Research_Traffic_Collector/1.0"}

ADD_FILE = "twin_validation.add.xml"
EDGEDATA_FILE = "edgeData_twinval.xml"
STATS_FILE = "twin_validation_stats.xml"
OUT_CSV = "twin_fidelity.csv"
OUT_JSON = "twin_fidelity.json"
OUT_FIG = "fig_twin_fidelity.png"

# mapping 的 SectionName 關鍵字 → 這一列是實測還是建模假設
KIND_KEYWORDS = [("鏡射", "mirrored"), ("代理", "proxy"), ("實測", "measured")]
KIND_LABEL = {"measured": "實測", "mirrored": "鏡射(假設)", "proxy": "代理(假設)"}


# ── 資料抓取 / 解析 ────────────────────────────────────────────────
def fetch_gz(url, timeout=30):
    import requests
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    data = r.content
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="ignore")


def archive_snapshot(text, prefix):
    """把現抓的 VD 快照存進 traffic_data/,論文才可重現。回傳存檔路徑。"""
    from datetime import datetime
    d = os.path.join(SCRIPT_DIR, "traffic_data")
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.xml")
    with open(fp, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[存檔] 快照已保存 {os.path.relpath(fp, SCRIPT_DIR)}"
          f"(請與論文一起歸檔,否則保真度數字無法重現)")
    return fp


def read_maybe_gz(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return data.decode("utf-8", errors="ignore")


def open_xml(path):
    """回傳可餵給 ElementTree 的檔案物件;SUMO 的 net/rou 可能是 .gz。"""
    with open(path, "rb") as f:
        head = f.read(2)
    return gzip.open(path, "rb") if head == b"\x1f\x8b" else open(path, "rb")


def _fnum(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def parse_section_totalvol(text):
    """GetVD.xml(路段級)→ {SectionId: (TotalVol 輛/5分, SectionName)}。

    與 make_real_flow.parse_static_vds 讀同一組欄位;此處只要量,不要座標。
    """
    root = ET.fromstring(text)
    out = {}
    for sec in root.iter():
        if sec.tag.split("}")[-1] != "SectionData":
            continue
        sid = name = None
        vol = None
        for child in sec:
            tag = child.tag.split("}")[-1]
            val = (child.text or "").strip()
            if tag == "SectionId":
                sid = val
            elif tag == "SectionName":
                name = " ".join(val.split())
            elif tag == "TotalVol":
                vol = _fnum(val)
        if sid and vol is not None:
            out[sid] = (vol, name or "")
    return out


def parse_device_svolume(text):
    """GetVDDATA.xml(設備級)→ ({DeviceID: {lane: Svolume}}, {DeviceID: 分鐘},
    全市小客車比例 s_ratio)。s_ratio 定義與 make_real_flow.parse_live_svolume 一致。
    """
    root = ET.fromstring(text)
    sv, ti = {}, {}
    tot_s = tot_all = 0.0
    for dev in root.iter():
        if dev.tag.split("}")[-1] != "VDDevice":
            continue
        did = None
        interval = 5.0
        lanes = {}
        for child in dev:
            tag = child.tag.split("}")[-1]
            if tag == "DeviceID":
                did = (child.text or "").strip()
            elif tag == "TimeInterval":
                interval = _fnum((child.text or "").strip()) or 5.0
            elif tag == "LaneData":
                ln = s = m = l = None
                for gc in child:
                    gtag = gc.tag.split("}")[-1]
                    val = _fnum((gc.text or "").strip())
                    if gtag == "LaneNO":
                        ln = int(val) if val is not None else None
                    elif gtag == "Svolume":
                        s = val
                    elif gtag == "Mvolume":
                        m = val
                    elif gtag == "Lvolume":
                        l = val
                if ln is not None and s is not None:
                    lanes[ln] = s
                    tot_s += s
                    tot_all += s + (m or 0.0) + (l or 0.0)
        if did and lanes:
            sv[did] = lanes
            ti[did] = interval
    s_ratio = (tot_s / tot_all) if tot_all > 0 else 0.8
    return sv, ti, s_ratio


def classify_row(section_name):
    for kw, kind in KIND_KEYWORDS:
        if kw in (section_name or ""):
            return kind
    return "measured"      # 無標註 → 當實測(保守:會被列入主指標受檢驗)


# ── 場景解析 ──────────────────────────────────────────────────────
def parse_sumocfg(path):
    """讀 sumocfg 的 net-file / route-files / additional-files。"""
    with open_xml(path) as fh:
        root = ET.parse(fh).getroot()
    got = {}
    for key in ("net-file", "route-files", "additional-files"):
        el = root.find(".//" + key)
        if el is not None and el.get("value"):
            got[key] = el.get("value")
    return got


def net_edge_ids(net_path):
    """不依賴 sumolib,直接從 .net.xml 取 edge id(略過內部 edge)。"""
    ids, _ = net_edges_and_sinks(net_path)
    return ids


def net_edges_and_sinks(net_path):
    """回傳 (所有 edge id, 邊界出口 edge id)。邊界出口 = 沒有任何出向 connection,
    車輛在該 edge 結束行程 → edgeData 記為 arrived 而非 left。"""
    ids, has_out = set(), set()
    with open_xml(net_path) as fh:
        for _, el in ET.iterparse(fh, events=("end",)):
            if el.tag == "edge" and el.get("function") != "internal":
                ids.add(el.get("id"))
                el.clear()
            elif el.tag == "connection":
                has_out.add(el.get("from"))
                el.clear()
    return ids, {e for e in ids if e not in has_out}


def route_flow_targets(route_path):
    """rou.xml → {edge_id: 指定的 vehsPerHour 總和}, 以及 flow 的最大 end 時刻。"""
    targets, end_max = {}, 0.0
    with open_xml(route_path) as fh:
        root = ET.parse(fh).getroot()
    for fl in root.iter("flow"):
        src = fl.get("from")
        vph = _fnum(fl.get("vehsPerHour"))
        if src and vph:
            targets[src] = targets.get(src, 0.0) + vph
        e = _fnum(fl.get("end"))
        if e:
            end_max = max(end_max, e)
    return targets, end_max


# ── SUMO 執行 / edgeData 解析 ─────────────────────────────────────
def write_edgedata_add(path, begin, end):
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!-- 由 validate_twin_fidelity.py 產生:量測窗內的逐 edge 流量 -->\n'
                '<additional>\n'
                f'    <edgeData id="twinval" file="{EDGEDATA_FILE}" '
                f'begin="{begin:.0f}" end="{end:.0f}" period="{end - begin:.0f}" '
                f'excludeEmpty="false"/>\n'
                '</additional>\n')


def run_sumo(net, routes, additionals, sim_end, seed, quiet=True):
    binary = os.environ.get("SUMO_BINARY", "sumo")
    cmd = [binary, "-n", net, "-r", routes,
           "-a", ",".join(additionals),
           "--begin", "0", "--end", str(int(sim_end)),
           "--seed", str(seed),
           "--statistic-output", STATS_FILE,
           "--no-step-log", "true", "--duration-log.disable", "true"]
    if quiet:
        cmd += ["--no-warnings", "true"]
    print("  $ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"SUMO 執行失敗(returncode={res.returncode})")
    return res


def parse_edgedata(path, metric):
    """edgeData 輸出 → {edge_id: 計數}, 以及實際量測窗長度(秒)。

    metric:
      left+arrived    ★預設。駛離下游端 + 在此 edge 結束行程者。
                      邊界出口 edge 沒有下游,車輛是 arrived 而非 left,
                      只看 left 會恆為 0 → 必須把 arrived 一起算才是通過量。
      left            僅駛離下游端(中段 edge 用;邊界出口會低估為 0)
      entered         由上游進入(≈路段起點的 VD)
      departed        在此 edge 上被插入模擬
      entered+departed  進入 + 插入(該 edge 的總流入)
    """
    root = ET.parse(path).getroot()
    counts, window = {}, 0.0
    for iv in root.iter("interval"):
        b, e = _fnum(iv.get("begin")) or 0.0, _fnum(iv.get("end")) or 0.0
        window += (e - b)
        for edge in iv.iter("edge"):
            eid = edge.get("id")
            if "+" in metric:
                v = sum(_fnum(edge.get(k)) or 0.0 for k in metric.split("+"))
            else:
                v = _fnum(edge.get(metric)) or 0.0
            counts[eid] = counts.get(eid, 0.0) + v
    return counts, window


def parse_teleports(path):
    try:
        root = ET.parse(path).getroot()
        tp = root.find(".//teleports")
        veh = root.find(".//vehicles")
        return (int(tp.get("total")) if tp is not None and tp.get("total") else 0,
                int(veh.get("inserted")) if veh is not None and veh.get("inserted") else 0,
                int(veh.get("running")) if veh is not None and veh.get("running") else 0)
    except Exception:
        return (0, 0, 0)


# ── 指標 ──────────────────────────────────────────────────────────
def geh(model, count):
    """GEH statistic。兩者皆 0 時定義為 0(無流量、無誤差)。"""
    s = model + count
    if s <= 0:
        return 0.0
    return math.sqrt(2.0 * (model - count) ** 2 / s)


def geh_envelope(count, threshold=5.0):
    """給定實測量 C,回傳 GEH<threshold 的模擬量上下界(畫圖用)。

    解 2(M-C)^2 = t^2 (M+C) → 2M^2 - (4C+t^2)M + (2C^2 - t^2 C) = 0
    """
    t2 = threshold ** 2
    b = 4.0 * count + t2
    disc = b * b - 8.0 * (2.0 * count * count - t2 * count)
    if disc < 0:
        return (count, count)
    r = math.sqrt(disc)
    lo, hi = (b - r) / 4.0, (b + r) / 4.0
    return (max(0.0, lo), hi)


def summarize(rows):
    """對一組列(通常是實測站)算彙總指標。"""
    if not rows:
        return {"n": 0}
    gs = [r["geh"] for r in rows]
    obs = [r["vd_vph"] for r in rows]
    sim = [r["sim_vph"] for r in rows]
    n = len(rows)
    ape = [abs(m - c) / c for m, c in zip(sim, obs) if c > 0]
    return {
        "n": n,
        "geh_mean": sum(gs) / n,
        "geh_max": max(gs),
        "geh_lt5_n": sum(1 for g in gs if g < 5),
        "geh_lt5_pct": 100.0 * sum(1 for g in gs if g < 5) / n,
        "geh_lt10_pct": 100.0 * sum(1 for g in gs if g < 10) / n,
        "mape_pct": 100.0 * sum(ape) / len(ape) if ape else float("nan"),
        "rmse_vph": math.sqrt(sum((m - c) ** 2 for m, c in zip(sim, obs)) / n),
        "obs_total_vph": sum(obs),
        "sim_total_vph": sum(sim),
    }


# ── 出圖 ──────────────────────────────────────────────────────────
def plot_fidelity(rows, summ, tag, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meas = [r for r in rows if r["kind"] == "measured"]
    other = [r for r in rows if r["kind"] != "measured"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 左:實測 vs 模擬散佈 + GEH<5 信封
    ax = axes[0]
    hi_c = max([r["vd_vph"] for r in rows] + [r["sim_vph"] for r in rows] + [1.0]) * 1.15
    cs = [hi_c * i / 100.0 for i in range(101)]
    los = [geh_envelope(c, 5.0)[0] for c in cs]
    his = [geh_envelope(c, 5.0)[1] for c in cs]
    ax.fill_between(cs, los, his, color="#66bb6a", alpha=0.18,
                    label="GEH < 5 (accepted calibration band)")
    ax.plot([0, hi_c], [0, hi_c], "--", color="#546e7a", lw=1, label="y = x")
    if meas:
        ax.scatter([r["vd_vph"] for r in meas], [r["sim_vph"] for r in meas],
                   s=90, color="#1565c0", zorder=5, label="Measured VD station (in headline metric)")
    if other:
        ax.scatter([r["vd_vph"] for r in other], [r["sim_vph"] for r in other],
                   s=70, facecolors="none", edgecolors="#ef6c00", zorder=5,
                   label="Mirrored / proxy link (modelling assumption)")
    for i, r in enumerate(sorted(rows, key=lambda z: (z["vd_vph"], z["sim_vph"]))):
        dy = 5 + 9 * (i % 3)          # 同量的點會疊在一起 → 依序錯開標註
        ax.annotate(r["edge"], (r["vd_vph"], r["sim_vph"]), fontsize=7,
                    xytext=(6, dy), textcoords="offset points",
                    color="#263238" if r["kind"] == "measured" else "#ef6c00")
    ax.set_xlim(0, hi_c); ax.set_ylim(0, hi_c)
    ax.set_xlabel("VD observed passenger-car flow (veh/h)")
    ax.set_ylabel("SUMO simulated flow (veh/h)")
    ax.set_title(f"Twin fidelity: observed vs simulated ({tag})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 右:逐 link 的 GEH 長條 + 門檻線
    ax = axes[1]
    labels = [f"{r['edge']}\n{r['device']}" for r in rows]
    vals = [r["geh"] for r in rows]
    colors = ["#1565c0" if r["kind"] == "measured" else "#ef6c00" for r in rows]
    ax.bar(range(len(rows)), vals, color=colors)
    ax.axhline(5, color="#2e7d32", ls="--", lw=1.2, label="GEH = 5 (acceptance threshold)")
    ax.axhline(10, color="#c62828", ls=":", lw=1.2, label="GEH = 10")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("GEH")
    n = summ.get("n", 0)
    sub = (f"measured n={n}, GEH<5: {summ.get('geh_lt5_pct', 0):.0f}%, "
           f"MAPE {summ.get('mape_pct', float('nan')):.1f}%") if n else "no measured station"
    ax.set_title(f"Per-link GEH — {sub}")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"已產生 {out_path}")


# ── 主流程 ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="孿生保真度驗證(GEH/MAPE)")
    p.add_argument("--sumocfg", default="osm.sumocfg", help="場景設定(取 net/rou/add)")
    p.add_argument("--net", help="覆寫 net-file")
    p.add_argument("--routes", help="覆寫 route-files")
    p.add_argument("--mapping", default="vd_sumo_mapping.csv")
    p.add_argument("--vd-xml", help="GetVD.xml 快照(路段級 TotalVol);離線驗證用")
    p.add_argument("--live-xml", help="GetVDDATA.xml 快照(設備級,算全市小客車比例)")
    p.add_argument("--fetch", action="store_true", help="現抓台北開放資料(需連線)")
    p.add_argument("--metric", default="left+arrived",
                   choices=["left+arrived", "left", "entered", "departed",
                            "entered+departed"],
                   help="模擬流量取哪個 edgeData 欄位(預設 left+arrived:"
                        "邊界出口 edge 的車輛是 arrived 而非 left)")
    p.add_argument("--warmup", type=float, default=300.0, help="暖機秒數(不計入量測)")
    p.add_argument("--duration", type=float, default=3600.0, help="量測窗秒數")
    p.add_argument("--seeds", type=int, default=1, help="重複跑幾個 seed 取平均")
    p.add_argument("--no-run", action="store_true", help="不跑 SUMO,沿用既有 edgeData")
    p.add_argument("--all-rows", action="store_true",
                   help="把鏡射/代理站也列入主指標(預設不列,避免循環論證)")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    os.chdir(SCRIPT_DIR)

    # 1) 場景 --------------------------------------------------------
    cfg = parse_sumocfg(args.sumocfg) if os.path.exists(args.sumocfg) else {}
    net = args.net or cfg.get("net-file")
    routes = args.routes or cfg.get("route-files")
    adds = [a for a in (cfg.get("additional-files") or "").split(",") if a.strip()]
    if not net or not routes:
        sys.exit(f"[錯誤] 無法從 {args.sumocfg} 取得 net-file/route-files,"
                 f"請用 --net/--routes 指定")
    for f in (net, routes):
        if not os.path.exists(f):
            sys.exit(f"[錯誤] 找不到 {f}(場景檔案不齊)")
    print(f"場景:net={net}  routes={routes}  additional={adds or '(無)'}")

    edges_in_net, sink_edges = net_edges_and_sinks(net)
    targets, flow_end = route_flow_targets(routes)
    print(f"路網 {len(edges_in_net)} 條 edge;車流檔 {len(targets)} 條起始 edge、"
          f"flow 結束於 t={flow_end:.0f}s")

    # 2) 量測窗 ------------------------------------------------------
    begin = args.warmup
    end = begin + args.duration
    if flow_end and end > flow_end:
        print(f"[調整] 量測窗超過 flow 結束時刻 → 截到 t={flow_end:.0f}s"
              f"(否則尾段無車會低估流量)")
        end = flow_end
    if end <= begin:
        sys.exit("[錯誤] 量測窗長度 <= 0,請調小 --warmup 或加長車流 end")
    print(f"量測窗:[{begin:.0f}s, {end:.0f}s] = {end - begin:.0f}s")

    # 3) mapping + 實測地真 ------------------------------------------
    if not os.path.exists(args.mapping):
        sys.exit(f"[錯誤] 找不到 {args.mapping}")
    with open(args.mapping, encoding="utf-8-sig") as f:
        map_rows = list(csv.DictReader(f))
    print(f"對應表 {len(map_rows)} 列")

    # 全市小客車比例(路段模式換算用)
    live_text = None
    live_src = args.live_xml
    if args.live_xml:
        live_text = read_maybe_gz(args.live_xml)
    elif args.fetch:
        live_text = fetch_gz(LIVE_VD_URL)
        live_src = archive_snapshot(live_text, "VD")
    else:
        cands = sorted(glob.glob(os.path.join(SCRIPT_DIR, "traffic_data", "VD_*.xml")))
        if cands:
            live_text = read_maybe_gz(cands[-1])
            live_src = cands[-1]
            print(f"[預設] 全市小客車比例取自最新歷史快照 {os.path.basename(cands[-1])}")
    if live_text is None:
        sys.exit("[錯誤] 需要設備級快照才能算全市小客車比例;"
                 "請用 --live-xml 指定,或 --fetch 現抓")
    sv, ti, s_ratio = parse_device_svolume(live_text)
    print(f"設備級快照:{len(sv)} 台 VD;全市小客車比例 s_ratio={s_ratio:.3f}")

    devs = {r["DeviceID"] for r in map_rows}
    device_mode = bool(devs & set(sv))
    sec_vol = {}
    static_src = None
    if not device_mode:
        static_text = None
        static_src = args.vd_xml
        if args.vd_xml:
            static_text = read_maybe_gz(args.vd_xml)
        elif args.fetch:
            static_text = fetch_gz(STATIC_VD_URL)
            static_src = archive_snapshot(static_text, "GetVD")
        if static_text is None:
            sys.exit(
                "[錯誤] 本對應表為『路段模式』(SectionId),需要路段級 GetVD.xml 快照\n"
                "       才有 TotalVol 地真。traffic_data/ 現有的 VD_*.xml 是設備級\n"
                "       (GetVDDATA,V 開頭 DeviceID),不含 SectionId → 不能當地真。\n"
                "\n"
                "       解法一(建議):現抓並自動存檔\n"
                "         python validate_twin_fidelity.py --sumocfg hepingeast2.sumocfg "
                "--fetch --plot\n"
                "       解法二:已有快照檔時指定路徑(注意 PowerShell 不要留角括號)\n"
                "         python validate_twin_fidelity.py --sumocfg hepingeast2.sumocfg "
                "--vd-xml traffic_data\\GetVD_20260817_1200.xml --plot")
        sec_vol = parse_section_totalvol(static_text)
        print(f"[路段模式] 快照含 {len(sec_vol)} 個路段的 TotalVol;"
              f"地真 = TotalVol × 12 × {s_ratio:.3f}")
    else:
        print("[設備模式] 地真 = 逐車道 Svolume × (60/TimeInterval)")

    # 逐 edge 聚合地真(edgeData 的比較單位是 edge)
    per_edge = {}
    missing = []
    for r in map_rows:
        eid = r.get("SumoEdgeID", "")
        if eid not in edges_in_net:
            missing.append((r["DeviceID"], eid))
            continue
        if device_mode:
            lanes = sv.get(r["DeviceID"], {})
            s = lanes.get(int(r["LaneNO"]))
            vph = None if s is None else s * (60.0 / ti.get(r["DeviceID"], 5.0))
        else:
            v = sec_vol.get(r["DeviceID"])
            vph = None if v is None else v[0] * 12.0 * s_ratio
        if vph is None:
            missing.append((r["DeviceID"], eid + "(快照無此站的量)"))
            continue
        kind = classify_row(r.get("SectionName", ""))
        e = per_edge.setdefault(eid, {"edge": eid, "devices": [], "kinds": set(),
                                      "vd_vph": 0.0, "name": r.get("SectionName", "")})
        e["devices"].append(r["DeviceID"])
        e["kinds"].add(kind)
        e["vd_vph"] += vph
    if missing:
        print(f"[略過] {len(missing)} 列(edge 不在路網或快照無資料):"
              + ", ".join(f"{d}/{e}" for d, e in missing[:6])
              + (" …" if len(missing) > 6 else ""))
    if not per_edge:
        sample = sorted({r.get("SumoEdgeID", "") for r in map_rows})[:4]
        sys.exit(
            "[錯誤] 對應表裡沒有任何 edge 存在於這個路網 → 場景錯配。\n"
            f"       路網 {net} 的 edge 例:{sorted(edges_in_net)[:4]}\n"
            f"       對應表要求的 edge 例:{sample}\n"
            "       多半是 sumocfg 仍指向舊路網。和平東路場景的三件套應為:\n"
            "         <net-file value=\"hepingeast2.net.xml\"/>\n"
            "         <route-files value=\"real_traffic_hep.rou.xml\"/>\n"
            "         <additional-files value=\"rsu.add.xml\"/>\n"
            "       或直接用 --sumocfg hepingeast2.sumocfg")

    # 4) 跑 SUMO -----------------------------------------------------
    if args.no_run:
        if not os.path.exists(EDGEDATA_FILE):
            sys.exit(f"[錯誤] --no-run 但找不到 {EDGEDATA_FILE}")
        print(f"[--no-run] 沿用既有 {EDGEDATA_FILE}")
        sim_counts, window = parse_edgedata(EDGEDATA_FILE, args.metric)
        n_seeds, teleports = 1, None
    else:
        write_edgedata_add(ADD_FILE, begin, end)
        acc, window = {}, 0.0
        teleports = 0
        for k in range(args.seeds):
            print(f"\n── 執行 SUMO(seed {k}) ──")
            run_sumo(net, routes, adds + [ADD_FILE], end, seed=k)
            c, w = parse_edgedata(EDGEDATA_FILE, args.metric)
            tp, ins, run = parse_teleports(STATS_FILE)
            teleports += tp
            print(f"  插入 {ins} 車、結束時在跑 {run} 車、teleport {tp} 次")
            for eid, v in c.items():
                acc[eid] = acc.get(eid, 0.0) + v
            window = w
        sim_counts = {eid: v / args.seeds for eid, v in acc.items()}
        n_seeds = args.seeds
    if window <= 0:
        sys.exit("[錯誤] edgeData 量測窗長度為 0,無法換算小時流量")

    scale = 3600.0 / window

    # 5) 指標 --------------------------------------------------------
    rows = []
    for eid, e in sorted(per_edge.items()):
        kinds = e["kinds"]
        kind = kinds.pop() if len(kinds) == 1 else "mixed"
        sim_vph = sim_counts.get(eid, 0.0) * scale
        is_sink = eid in sink_edges
        rows.append({
            "edge": eid,
            "device": "+".join(sorted(set(e["devices"]))),
            "section_name": e["name"],
            "kind": kind,
            "vd_vph": e["vd_vph"],
            "target_vph": targets.get(eid, 0.0),
            "sim_vph": sim_vph,
            "geh": geh(sim_vph, e["vd_vph"]),
            "err_pct": (100.0 * (sim_vph - e["vd_vph"]) / e["vd_vph"]
                        if e["vd_vph"] > 0 else float("nan")),
            "demand_realized_pct": (100.0 * sim_vph / targets[eid]
                                    if targets.get(eid) else float("nan")),
            "is_sink_edge": is_sink,
        })

    head = rows if args.all_rows else [r for r in rows if r["kind"] == "measured"]
    summ = summarize(head)

    # 6) 輸出 --------------------------------------------------------
    print(f"\n=== 孿生保真度({args.metric},{n_seeds} seed 平均) ===")
    print(f"{'edge':<20}{'站別':<12}{'VD(veh/h)':>11}{'目標':>9}{'模擬':>9}"
          f"{'GEH':>7}{'誤差%':>8}{'需求實現%':>10}")
    for r in rows:
        print(f"{r['edge']:<20}{KIND_LABEL.get(r['kind'], r['kind']):<12}"
              f"{r['vd_vph']:>11.0f}{r['target_vph']:>9.0f}{r['sim_vph']:>9.0f}"
              f"{r['geh']:>7.2f}{r['err_pct']:>8.1f}{r['demand_realized_pct']:>10.1f}")

    if summ["n"]:
        print(f"\n--- 主指標(僅{'全部' if args.all_rows else '實測'}站,n={summ['n']}) ---")
        print(f"  GEH<5   : {summ['geh_lt5_n']}/{summ['n']} = {summ['geh_lt5_pct']:.0f}%"
              f"   (業界驗收慣例 >= 85%)")
        print(f"  GEH<10  : {summ['geh_lt10_pct']:.0f}%")
        print(f"  平均GEH : {summ['geh_mean']:.2f}  (最大 {summ['geh_max']:.2f})")
        print(f"  MAPE    : {summ['mape_pct']:.1f}%")
        print(f"  RMSE    : {summ['rmse_vph']:.0f} veh/h")
        print(f"  總量    : 實測 {summ['obs_total_vph']:.0f} vs "
              f"模擬 {summ['sim_total_vph']:.0f} veh/h")
        verdict = ("✅ 達校正合格慣例(GEH<5 佔比 >= 85%)"
                   if summ["geh_lt5_pct"] >= 85 else
                   "⚠ 未達 85% 慣例 —— 論文須據實報告並解釋(見下方診斷)")
        print(f"  判定    : {verdict}")
        if summ["n"] < 10:
            print(f"  ⚠ 樣本數僅 {summ['n']} 個 link,GEH 佔比統計意義有限;"
                  f"論文請直接報逐 link 的 GEH,並聲明 n。")
    _sinks = [r["edge"] for r in rows if r["is_sink_edge"]]
    if _sinks:
        print(f"\n  註:{len(_sinks)} 條為『邊界出口 edge』(無下游,車輛在此結束行程):"
              f"{', '.join(_sinks)}")
        print(f"     通過量必須含 arrived(本次 metric='{args.metric}');"
              f"改用 --metric left 會恆為 0。")
        print("     它們也無法被指派 flow(無可達出口)→ 目標量為 0,"
              "流量全部來自上游穿越。")
    if teleports:
        print(f"\n  ⚠ 模擬過程發生 {teleports} 次 teleport(壅塞到車輛被瞬移),"
              f"表示部分路段需求超過通行能力 → 會壓低該 edge 的模擬流量。")

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    payload = {
        "scenario": {"net": net, "routes": routes, "additional": adds},
        "measure_window_s": [begin, end], "metric": args.metric,
        "seeds": n_seeds, "s_ratio": s_ratio,
        "ground_truth": "device" if device_mode else "section",
        "vd_snapshot": (os.path.relpath(static_src, SCRIPT_DIR)
                        if not device_mode and static_src else None),
        "device_snapshot": (os.path.relpath(live_src, SCRIPT_DIR)
                            if live_src else None),
        "teleports": teleports,
        "headline_scope": "all" if args.all_rows else "measured_only",
        "summary": summ, "links": rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已存 {OUT_CSV} / {OUT_JSON}")

    if args.plot:
        plot_fidelity(rows, summ, os.path.basename(net), OUT_FIG)


if __name__ == "__main__":
    main()
