"""
車流校正(calibrate_flow.py)—— 用 VD 計數當「約束」而非「注入量」
====================================================================
make_real_flow.py 的作法是把每個 VD 的路段流量直接當成該 edge 的產生量注入。
但 VD 量到的是**通過量**:上游 VD 量到的車開到下游會再被下游 VD 量一次,
現實中是同一批車,模型裡卻變成兩批 → 走廊被灌爆。
實測(和平東路,2026-08):205066812#6 的需求實現度 185%,GEH 14.5。

本腳本改用交通模擬校正的標準作法(SUMO 自帶 routeSampler.py):
  1. 產生一組涵蓋路網的候選路徑(randomTrips + duarouter)
  2. 把 VD 計數寫成約束檔
  3. routeSampler 從候選路徑中挑出一組**同時滿足所有計數**的組合
→ 同一批車只會被算一次,空間分布才會對。

★ 不動 make_real_flow.py。兩種車流可以並存,各跑一次保真度驗證再決定論文用哪個。

用法(需要 SUMO_HOME):
  python calibrate_flow.py --net hepingeast2.net.xml \
         --vd-xml traffic_data/GetVD_20260817_140317.xml
  # 比較校正前後
  python validate_twin_fidelity.py --sumocfg hepingeast2.sumocfg \
         --routes real_traffic_hep_calibrated.rou.xml \
         --vd-xml traffic_data/GetVD_20260817_140317.xml --plot

產出:real_traffic_hep_calibrated.rou.xml、flow_counts.xml(約束檔)
"""
import argparse
import os
import re
import subprocess
import sys

from validate_twin_fidelity import (
    SCRIPT_DIR, KIND_LABEL, load_vd_ground_truth, read_mapping,
    net_edges_and_sinks,
)

OUT_ROUTE = "real_traffic_hep_calibrated.rou.xml"
COUNTS_FILE = "flow_counts.xml"
CAND_TRIPS = "_cand_trips.xml"
CAND_ROUTES = "_cand_routes.rou.xml"


# ── SUMO 工具定位 ─────────────────────────────────────────────────
def sumo_tools_dir():
    home = os.environ.get("SUMO_HOME")
    if not home:
        sys.exit("[錯誤] 找不到環境變數 SUMO_HOME。\n"
                 "       Windows 一般是 C:\\Program Files (x86)\\Eclipse\\Sumo\n"
                 "       PowerShell 暫時設定:$env:SUMO_HOME='C:\\Program Files (x86)\\Eclipse\\Sumo'")
    tools = os.path.join(home, "tools")
    if not os.path.isdir(tools):
        sys.exit(f"[錯誤] {tools} 不存在,SUMO_HOME 可能指錯位置")
    return tools


def need_tool(tools, name):
    fp = os.path.join(tools, name)
    if not os.path.exists(fp):
        sys.exit(f"[錯誤] 找不到 {fp}(SUMO 版本過舊?routeSampler 需 SUMO >= 1.9)")
    return fp


def supported_flags(script_path):
    """跑 --help 取出該版本支援的長旗標。

    routeSampler / randomTrips 的參數在各版本間有增減,先問過再用,
    免得因為一個不存在的旗標整個管線掛掉。
    """
    try:
        r = subprocess.run([sys.executable, script_path, "--help"],
                           capture_output=True, text=True, timeout=120)
        return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9\-]*", r.stdout + r.stderr))
    except Exception as e:
        print(f"  ⚠ 無法取得 {os.path.basename(script_path)} 的參數清單({e}),"
              f"改用最保守的參數組合")
        return set()


def run(cmd, what):
    print(f"\n── {what} ──")
    print("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if out.strip():
        print("\n".join("  " + ln for ln in out.strip().splitlines()[-40:]))
    if r.returncode != 0:
        sys.exit(f"[錯誤] {what} 失敗(returncode={r.returncode})")
    return out


# ── 約束檔 ────────────────────────────────────────────────────────
def write_counts(path, rows, begin, end):
    """VD 地真(veh/h)→ routeSampler 的計數約束檔。

    同時寫 count 與 entered 兩個屬性:不同 SUMO 版本的 routeSampler
    預設讀的屬性名不同,兩個都放最保險。
    """
    dur_h = (end - begin) / 3600.0
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<!-- 由 calibrate_flow.py 產生:VD 實測小客車量(輛/量測窗) -->\n'
                '<data>\n'
                f'    <interval id="vd" begin="{begin:.0f}" end="{end:.0f}">\n')
        for r in rows:
            n = int(round(r["vd_vph"] * dur_h))
            f.write(f'        <edge id="{r["edge"]}" count="{n}" entered="{n}"/>'
                    f'   <!-- {r["kind"]}: {r["devices"]} {r["vd_vph"]:.0f} veh/h -->\n')
        f.write('    </interval>\n</data>\n')
    print(f"已產生 {path}:{len(rows)} 條約束")


def main():
    p = argparse.ArgumentParser(description="以 VD 計數為約束校正車流(routeSampler)")
    p.add_argument("--net", default="hepingeast2.net.xml")
    p.add_argument("--mapping", default="vd_sumo_mapping.csv")
    p.add_argument("--vd-xml", help="GetVD.xml 路段級快照(建議與驗證用同一份)")
    p.add_argument("--live-xml", help="GetVDDATA.xml 設備級快照(算全市小客車比例)")
    p.add_argument("--fetch", action="store_true", help="現抓台北開放資料並存檔")
    p.add_argument("--counts", default="measured", choices=["measured", "all"],
                   help="哪些 link 當約束:measured=只用實測站(預設,避免把"
                        "鏡射/代理的建模假設當成量測);all=全部列入")
    p.add_argument("--begin", type=float, default=0.0)
    p.add_argument("--end", type=float, default=3600.0)
    p.add_argument("--candidates", type=int, default=5000,
                   help="候選路徑數(routeSampler 從中挑選,多一點比較挑得到解)")
    p.add_argument("--fringe-factor", type=float, default=100.0,
                   help="randomTrips 偏好邊界起訖的程度(穿越車流用高值)")
    p.add_argument("--optimize", default="full",
                   help="routeSampler 的 --optimize(full 或整數;none=不最佳化)")
    p.add_argument("--total-count", type=int,
                   help="指定總車輛數(計數不足以定案時可用來釘住總量)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=OUT_ROUTE)
    p.add_argument("--keep-temp", action="store_true", help="保留候選路徑暫存檔")
    args = p.parse_args()

    os.chdir(SCRIPT_DIR)
    tools = sumo_tools_dir()
    random_trips = need_tool(tools, "randomTrips.py")
    route_sampler = need_tool(tools, "routeSampler.py")
    print(f"SUMO tools:{tools}")

    if not os.path.exists(args.net):
        sys.exit(f"[錯誤] 找不到路網 {args.net}")
    edges_in_net, sink_edges = net_edges_and_sinks(args.net)
    print(f"路網 {args.net}:{len(edges_in_net)} 條 edge、{len(sink_edges)} 個邊界出口")

    # 1) VD 地真(與保真度驗證共用同一份換算定義)
    map_rows = read_mapping(args.mapping)
    per_edge, _missing, meta = load_vd_ground_truth(
        map_rows, edges_in_net, vd_xml=args.vd_xml, live_xml=args.live_xml,
        fetch=args.fetch, caller="calibrate_flow.py")
    if not per_edge:
        sys.exit("[錯誤] 沒有任何可用的 VD 計數(對應表與路網是同一場景嗎?)")

    rows = []
    for eid, e in sorted(per_edge.items()):
        kinds = e["kinds"]
        rows.append({"edge": eid, "kind": kinds.copy().pop() if len(kinds) == 1 else "mixed",
                     "devices": "+".join(sorted(set(e["devices"]))),
                     "vd_vph": e["vd_vph"]})

    chosen = [r for r in rows if r["kind"] == "measured"] if args.counts == "measured" else rows
    print(f"\n=== 計數約束({args.counts})===")
    for r in rows:
        mark = "✓ 約束" if r in chosen else "  (不列入)"
        print(f"  {mark}  {r['edge']:<18}{KIND_LABEL.get(r['kind'], r['kind']):<12}"
              f"{r['vd_vph']:>8.0f} veh/h   {r['devices']}")
    if not chosen:
        sys.exit("[錯誤] 沒有任何可用的約束。若實測站都缺資料,可改用 --counts all,"
                 "或換個時段重抓快照。")
    if len(chosen) < 4:
        print(f"\n  ⚠ 只有 {len(chosen)} 條約束 —— 路網的需求是**欠定**的:"
              f"滿足這些計數的流量組合不只一種。\n"
              f"     routeSampler 會在滿足約束的前提下盡量少放車;總量若需釘住,"
              f"用 --total-count。\n"
              f"     論文必須聲明「以 n={len(chosen)} 個偵測器校正」,"
              f"不要宣稱整個走廊都被量測約束。")

    write_counts(COUNTS_FILE, chosen, args.begin, args.end)

    # 2) 候選路徑(偏好邊界進、邊界出 = 穿越車流)
    period = max((args.end - args.begin) / max(args.candidates, 1), 0.01)
    rt_flags = supported_flags(random_trips)
    cmd = [sys.executable, random_trips, "-n", args.net,
           "-o", CAND_TRIPS, "-r", CAND_ROUTES,
           "-b", str(args.begin), "-e", str(args.end), "-p", f"{period:.4f}",
           "--fringe-factor", str(args.fringe_factor), "--seed", str(args.seed)]
    for flag, val in (("--validate", None), ("--vehicle-class", "passenger")):
        if flag in rt_flags:
            cmd.append(flag)
            if val:
                cmd.append(val)
        else:
            print(f"  ⚠ 這個 SUMO 版本沒有 {flag},略過")
    run(cmd, f"產生 {args.candidates} 條候選路徑")
    if not os.path.exists(CAND_ROUTES):
        sys.exit(f"[錯誤] 候選路徑 {CAND_ROUTES} 沒產出來")

    # 3) routeSampler:挑出滿足所有計數的路徑組合
    rs_flags = supported_flags(route_sampler)
    cmd = [sys.executable, route_sampler, "-r", CAND_ROUTES,
           "--edgedata-files", COUNTS_FILE, "-o", args.out,
           "--seed", str(args.seed)]
    if "--edgedata-attribute" in rs_flags:
        cmd += ["--edgedata-attribute", "count"]
    if args.optimize and args.optimize != "none" and "--optimize" in rs_flags:
        cmd += ["--optimize", args.optimize]
    if args.total_count and "--total-count" in rs_flags:
        cmd += ["--total-count", str(args.total_count)]
    if "--verbose" in rs_flags:
        cmd.append("--verbose")
    out = run(cmd, "routeSampler:挑選滿足計數的路徑組合")

    # routeSampler 自己會報 GEH,把關鍵行抓出來再講一次
    hits = [ln.strip() for ln in out.splitlines()
            if "GEH" in ln or "deviation" in ln.lower() or "wrote" in ln.lower()]
    if hits:
        print("\n=== routeSampler 自報的契合度 ===")
        for ln in hits[-8:]:
            print("  " + ln)

    if not args.keep_temp:
        for f in (CAND_TRIPS, CAND_ROUTES):
            if os.path.exists(f):
                os.remove(f)

    snap = meta["static_src"] or "<你的快照>"
    print(f"\n已產生 {args.out}")
    print("\n下一步 —— 用同一份快照驗證校正前後的保真度:")
    print(f"  python validate_twin_fidelity.py --sumocfg hepingeast2.sumocfg \\\n"
          f"         --routes {args.out} --vd-xml {snap} --plot")
    print("\n  ⚠ 若採用校正後車流,車流密度會改變 → V2V 機會與 RSU 負載跟著變,")
    print("     §5.2~5.6 的結果需重跑,MAPPO 需重訓(觀測維度不變,但策略要重學)。")


if __name__ == "__main__":
    main()
