"""
realtime_calibrator.py
用 SUMO Calibrator 機制做即時數位孿生。
參考：Kušić et al. (2023) Advanced Engineering Informatics

★★★ LEGACY —— 未接入任何實驗管線,且仍綁「舊場景」★★★
    本檔的 VD_TO_EDGE / EDGE_TO_CAL / EDGE_FLOWS 全部是舊的忠孝東路小十字
    路網 edge ID(506322846#0 等),與論文主場景「和平東路二段」
    (hepingeast2.net.xml、8 個 RSU)完全不同。直接執行會對不上路網。

    定位:這是本專案唯一達到 Digital Shadow(實體→數位自動流)的元件,
    但正式實驗走的是離線校正管線(make_real_flow.py 產靜態 rou.xml)。
    論文請把它寫成 future work,不要在口試展示,否則會與主場景自相矛盾。
    詳見 DT_DEFINITION.md §2 與 §7。

    要復用到和平東路場景,需重做:
      1. VD_TO_EDGE   → 依 vd_sumo_mapping.csv 的 Z 開頭 SectionId 重建
      2. EDGE_TO_CAL / EDGE_FLOWS → 依 hepingeast2 的 edge 與可達出口重建
      3. calibrators.add.xml      → 依新 edge 重新產生
      4. SUMO_CFG     → 改為 hepingeast2.sumocfg

架構：
  1. 啟動 SUMO（含 calibrators.add.xml）
  2. 每 UPDATE_INTERVAL 秒抓一次北市 VD XML
  3. 把 volume/5min 換算成 vehsPerHour
  4. 透過 TraCI calibrator.setFlow() 更新每條 edge 的流量
  5. SUMO 自動在該 edge 產生/移除車輛，不需手動注入

執行：
  python realtime_calibrator.py
  python realtime_calibrator.py --gui        # 開視覺化視窗
  python realtime_calibrator.py --dry-run    # 不抓真實資料，用假流量測試
"""
import sys, os, time, gzip, argparse, requests, xml.etree.ElementTree as ET

# 確保使用 SUMO 附帶的 traci
SUMO_TOOLS = r"C:\Program Files (x86)\Eclipse\Sumo\tools"
if os.path.exists(SUMO_TOOLS) and SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

import traci

# ── 設定 ──────────────────────────────────────────────────────────────
SUMO_CFG        = "osm.sumocfg"   # ← LEGACY:舊場景;和平東路應為 hepingeast2.sumocfg
VD_URL          = "https://tcgbusfs.blob.core.windows.net/blobtisv/GetVDDATA.xml.gz"
UPDATE_INTERVAL = 60     # 秒

# VD DeviceID → edge 對應（依實際量測方向）
# 格式：DeviceID → (edge_id, 哪些車道屬於這個方向)
VD_TO_EDGE = {
    'VJQJI20': [
        ('238526531#0', [0, 1, 2]),   # 忠孝東路二段往西
        ('506322846#0', [3, 4, 5]),   # 忠孝東路二段往東
    ],
    'VJWJ700': [
        ('1458697048#0', [0, 1, 2, 3, 4, 5]),  # 新生南路往北
    ],
    'VJWJ740': [
        ('182334379#0', [0, 1, 2, 3, 4, 5]),   # 新生南路往南
    ],
}

# edge → calibrator ID
EDGE_TO_CAL = {
    '506322854#0': 'cal_506322854_0',
    '506322846#0': 'cal_506322846_0',
    '238526531#0': 'cal_238526531_0',
    '1458697048#0': 'cal_1458697048_0',
    '182334379#0': 'cal_182334379_0',
}

# edge → (route_id, 轉向比例) 列表
EDGE_FLOWS = {
    '506322854#0': [
        ('route_506322854_0__to__238526531_0', 0.60),
        ('route_506322854_0__to__288135626_0', 0.25),
        ('route_506322854_0__to__288135628_0', 0.15),
    ],
    '506322846#0': [
        ('route_506322846_0__to__506322851_0', 0.70),
        ('route_506322846_0__to__288135626_0', 0.30),
    ],
    '238526531#0': [
        ('route_238526531_0__to__506322846_0', 1.00),
    ],
    '1458697048#0': [
        ('route_1458697048_0__to__288135626_0', 0.60),
        ('route_1458697048_0__to__506322851_0', 0.25),
        ('route_1458697048_0__to__238526531_0', 0.15),
    ],
    '182334379#0': [
        ('route_182334379_0__to__288135628_0', 0.60),
        ('route_182334379_0__to__238526531_0', 0.25),
        ('route_182334379_0__to__506322851_0', 0.15),
    ],
}

# ── VD 資料擷取 ────────────────────────────────────────────────────────
def fetch_vd_speeds(dry_run=False) -> dict[str, float]:
    """回傳 {edge_id: vehsPerHour}"""
    if dry_run:
        import random
        return {
            '238526531#0':  random.uniform(600, 1000),
            '506322846#0':  random.uniform(600, 1000),
            '1458697048#0': random.uniform(200,  400),
            '182334379#0':  random.uniform(200,  400),
            '506322854#0':  random.uniform(600, 1000),
        }

    try:
        r = requests.get(VD_URL, timeout=15,
                         headers={'User-Agent': 'Research_DT/1.0'})
        data = r.content
        if data[:2] == b'\x1f\x8b':
            data = gzip.decompress(data)
        root = ET.fromstring(data.decode('utf-8', errors='ignore'))
    except Exception as e:
        print(f"[VD] 擷取失敗：{e}")
        return {}

    # 每個 DeviceID 的各車道 volume 加總
    device_vol: dict[str, float] = {}
    for dev in root.iter('VDDevice'):
        did = dev.findtext('DeviceID', '').strip()
        if did not in VD_TO_EDGE:
            continue
        for lane in dev.findall('LaneData'):
            lane_no = int(float(lane.findtext('LaneNO', '0')))
            vol     = float(lane.findtext('Volume', '0'))   # 輛/5min
            device_vol.setdefault(did, {})
            device_vol[did][lane_no] = vol

    # 換算成各 edge 的 vehsPerHour
    edge_vph: dict[str, float] = {}
    for did, lane_map in device_vol.items():
        for edge_id, lane_nos in VD_TO_EDGE.get(did, []):
            total_5min = sum(lane_map.get(ln, 0) for ln in lane_nos)
            vph = total_5min * 12   # 5min → 1hr
            edge_vph[edge_id] = edge_vph.get(edge_id, 0) + vph

    return edge_vph

# ── Calibrator 更新 ────────────────────────────────────────────────────
def update_calibrators(edge_vph: dict[str, float], sim_time: float):
    """把新的流量設定推進 SUMO Calibrator"""
    updated = 0
    for edge_id, total_vph in edge_vph.items():
        cal_id = EDGE_TO_CAL.get(edge_id)
        flows  = EDGE_FLOWS.get(edge_id, [])
        if not cal_id or not flows:
            continue

        begin = sim_time
        end   = sim_time + UPDATE_INTERVAL + 10   # 多 10 秒緩衝

        for route_id, ratio in flows:
            vph = total_vph * ratio
            try:
                traci.calibrator.setFlow(
                    cal_id,
                    begin,
                    end,
                    vph,          # vehsPerHour
                    -1,           # speed（-1 = 不限制）
                    'passenger',  # vType
                    route_id,
                )
                updated += 1
            except Exception as e:
                print(f"  [CAL] {cal_id} 更新失敗：{e}")

    return updated

# ── 主程式 ─────────────────────────────────────────────────────────────
def run(args):
    port = args.port
    proc = None

    if args.connect_only:
        # 只連線，SUMO 已在外部啟動
        print(f"連線到已執行的 SUMO（port={port}）...")
        time.sleep(1)
    else:
        binary = 'sumo-gui' if args.gui else 'sumo'
        cmd = [binary, f'--remote-port', str(port), '-c', SUMO_CFG]
        print(f"啟動 SUMO：{' '.join(cmd)}")
        import subprocess
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        time.sleep(6 if args.gui else 3)

    traci.connect(port=port)
    print("已連線到 TraCI")
    print(f"更新間隔：{UPDATE_INTERVAL} 秒  |  Ctrl+C 停止\n")

    next_update = 0.0
    try:
        while True:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            if sim_time >= next_update:
                print(f"[t={sim_time:.0f}s] 抓取 VD 資料...")
                edge_vph = fetch_vd_speeds(dry_run=args.dry_run)

                if edge_vph:
                    n = update_calibrators(edge_vph, sim_time)
                    print(f"[t={sim_time:.0f}s] 更新 {n} 個 calibrator flow")
                    for eid, vph in sorted(edge_vph.items()):
                        NAMES = {
                            '506322851#0':'忠孝東路三段→東',
                            '506322854#0':'忠孝東路三段→西',
                            '506322846#0':'忠孝東路二段→東',
                            '238526531#0':'忠孝東路二段→西',
                            '1458697048#0':'新生南路一段→北',
                            '182334379#0':'新生南路一段→南',
                        }
                        print(f"  {NAMES.get(eid,eid):<20}  {vph:.0f} 輛/hr")
                else:
                    print(f"[t={sim_time:.0f}s] 無資料，保持現有流量")

                next_update = sim_time + UPDATE_INTERVAL

    except KeyboardInterrupt:
        print("\n停止。")
    except Exception as e:
        print(f"錯誤：{e}")
        import traceback; traceback.print_exc()
    finally:
        try: traci.close()
        except: pass
        if proc:
            try: proc.terminate()
            except: pass

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gui',          action='store_true', help='開 SUMO GUI')
    p.add_argument('--dry-run',      action='store_true', help='用假資料測試')
    p.add_argument('--port',         type=int, default=8813, help='TraCI port')
    p.add_argument('--connect-only', action='store_true',
                   help='不自啟動 SUMO，連線到已執行的 SUMO')
    run(p.parse_args())
