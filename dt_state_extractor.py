#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dt_state_extractor.py
數位孿生層 — 狀態萃取器

從 SUMO（TraCI）讀取車輛狀態，計算通道與計算資源，
輸出多車 state vector 給 multi-agent DQN 使用。

State vector（每輛車 10 維）:
  [0] veh_speed        車速 (m/s)，正規化 /30
  [1] veh_pos          在 edge 上的位置比例 [0,1]
  [2] nearest_rsu_id   最近 RSU 編號 /3
  [3] rsu_distance     到最近 RSU 的距離 (m)，正規化 /300
  [4] channel_bw       估算頻寬 (Mbps)，正規化 /20
  [5] channel_delay    估算延遲 (ms)，正規化 /50
  [6] rsu_cpu_load     最近 RSU 的 CPU 負載 [0,1]
  [7] cloud_latency    雲端延遲代理 (ms)，正規化 /200
  [8] task_size        任務大小 (MB)，正規化 /10（固定）
  [9] task_deadline    截止時限 (s)，正規化 /2（固定）

執行方式：
  # 視窗一：啟動 SUMO
  sumo-gui --remote-port 8813 -c osm.sumocfg

  # 視窗二：啟動狀態萃取器
  python dt_state_extractor.py

  # 或不開 GUI：
  python dt_state_extractor.py --no-gui
"""

import sys
import os
import time
import math
import argparse
import subprocess
import json
from datetime import datetime
from collections import defaultdict

# 確保使用 SUMO 附帶的 traci
SUMO_TOOLS = r"C:\Program Files (x86)\Eclipse\Sumo\tools"
if os.path.exists(SUMO_TOOLS) and SUMO_TOOLS not in sys.path:
    sys.path.insert(0, SUMO_TOOLS)

import traci

# ── RSU 定義 ──────────────────────────────────────────────────────
# 忠孝東路三段 × 新生南路一段路口四角
# GPS 座標轉換為 SUMO 內部座標（由路網決定，此處用近似值）
RSU_LIST = [
    {"id": 0, "name": "RSU-0-NW", "lon": 121.5308, "lat": 25.0418,
     "primary_edges": ["506322846#0", "1458697048#0"]},
    {"id": 1, "name": "RSU-1-NE", "lon": 121.5328, "lat": 25.0418,
     "primary_edges": ["506322854#0", "288135626#0"]},
    {"id": 2, "name": "RSU-2-SW", "lon": 121.5308, "lat": 25.0408,
     "primary_edges": ["182334379#0", "238526531#0"]},
    {"id": 3, "name": "RSU-3-SE", "lon": 121.5328, "lat": 25.0408,
     "primary_edges": ["288135628#0", "506322851#0"]},
]

RSU_COVERAGE_M = 150.0   # 覆蓋半徑 (m)
MAX_BW_MBPS    = 20.0    # 最大頻寬 (Mbps)
MIN_BW_MBPS    = 0.5     # 最小頻寬 (Mbps)
BASE_DELAY_MS  = 5.0     # 最小延遲 (ms)
RSU_CPU_CORES  = 4       # RSU CPU 核心數

# 固定任務參數（可依需求調整）
TASK_SIZE_MB   = 5.0     # 任務大小 (MB)
TASK_DEADLINE  = 1.5     # 截止時限 (s)

# ── 通道模型（Free-Space Path Loss 簡化版）─────────────────────────
def estimate_channel(dist_m: float, num_nearby_vehs: int) -> dict:
    """根據距離估算 BW 和 delay"""
    if dist_m < 1:
        dist_m = 1.0

    # 路徑損耗：距離越遠 BW 越小
    bw = MAX_BW_MBPS * (RSU_COVERAGE_M / (dist_m + RSU_COVERAGE_M))
    # 競爭懲罰：同時接入車輛越多 BW 越低
    bw = bw / max(1, 1 + 0.1 * num_nearby_vehs)
    bw = max(MIN_BW_MBPS, bw)

    # 延遲：基礎 + 距離比例 + 競爭
    delay = BASE_DELAY_MS + (dist_m / RSU_COVERAGE_M) * 20.0
    delay += num_nearby_vehs * 0.5
    delay = min(delay, 50.0)

    return {"bw_mbps": bw, "delay_ms": delay}


# ── RSU 狀態管理 ──────────────────────────────────────────────────
class RSUManager:
    def __init__(self):
        self.cpu_load = {rsu["id"]: 0.0 for rsu in RSU_LIST}
        self.connected_vehs = {rsu["id"]: [] for rsu in RSU_LIST}

    def update(self, veh_rsu_map: dict):
        """根據每輛車的最近 RSU 更新 CPU 負載"""
        counts = defaultdict(int)
        connections = defaultdict(list)
        for veh_id, rsu_id in veh_rsu_map.items():
            counts[rsu_id] += 1
            connections[rsu_id].append(veh_id)

        for rsu in RSU_LIST:
            rid = rsu["id"]
            n = counts.get(rid, 0)
            # CPU 負載：每輛車消耗 1 核，上限 1.0
            self.cpu_load[rid] = min(1.0, n / RSU_CPU_CORES)
            self.connected_vehs[rid] = connections.get(rid, [])

    def get_load(self, rsu_id: int) -> float:
        return self.cpu_load.get(rsu_id, 0.0)

    def get_nearby_count(self, rsu_id: int) -> int:
        return len(self.connected_vehs.get(rsu_id, []))


# ── 座標距離計算 ───────────────────────────────────────────────────
def haversine_m(lon1, lat1, lon2, lat2) -> float:
    """GPS 兩點距離（公尺）"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_rsu_xy(rsu: dict) -> tuple:
    """用 TraCI 把 GPS 轉成 SUMO 內部 XY"""
    try:
        x, y = traci.simulation.convertGeo(rsu["lon"], rsu["lat"], fromGeo=True)
        return x, y
    except Exception:
        return None, None


# ── 主萃取器 ──────────────────────────────────────────────────────
class DTStateExtractor:
    def __init__(self, output_file="dt_states.jsonl", verbose=True):
        self.output_file = output_file
        self.verbose = verbose
        self.rsu_manager = RSUManager()
        self.rsu_xy = {}   # {rsu_id: (x, y)}
        self.step = 0

    def init_rsu_positions(self):
        """把 RSU GPS 座標轉成 SUMO XY"""
        for rsu in RSU_LIST:
            x, y = get_rsu_xy(rsu)
            if x is not None:
                self.rsu_xy[rsu["id"]] = (x, y)
                print(f"  {rsu['name']}: SUMO XY = ({x:.1f}, {y:.1f})")
            else:
                # fallback: 路口中心附近固定偏移
                cx, cy = 310.0, 260.0  # 路口中心估計值
                offset = {0: (-50, 50), 1: (50, 50), 2: (-50, -50), 3: (50, -50)}
                dx, dy = offset[rsu["id"]]
                self.rsu_xy[rsu["id"]] = (cx + dx, cy + dy)
                print(f"  {rsu['name']}: 使用 fallback XY = ({cx+dx:.1f}, {cy+dy:.1f})")

    def get_nearest_rsu(self, veh_x: float, veh_y: float) -> tuple:
        """回傳 (rsu_id, distance_m)"""
        best_id, best_dist = 0, float("inf")
        for rsu_id, (rx, ry) in self.rsu_xy.items():
            d = math.sqrt((veh_x - rx)**2 + (veh_y - ry)**2)
            if d < best_dist:
                best_dist, best_id = d, rsu_id
        return best_id, best_dist

    def extract(self) -> dict:
        """萃取當前所有車輛的 state vector，回傳 {veh_id: state_10d}"""
        veh_ids = traci.vehicle.getIDList()
        if not veh_ids:
            return {}

        # 第一輪：取得每輛車的最近 RSU（更新 RSU 負載前）
        veh_rsu_map = {}
        veh_raw = {}
        edge_lengths = {}

        for veh_id in veh_ids:
            try:
                x, y       = traci.vehicle.getPosition(veh_id)
                speed      = traci.vehicle.getSpeed(veh_id)
                edge_id    = traci.vehicle.getRoadID(veh_id)
                lane_pos   = traci.vehicle.getLanePosition(veh_id)

                # Edge 長度（快取）
                if edge_id not in edge_lengths:
                    try:
                        edge_lengths[edge_id] = traci.lane.getLength(edge_id + "_0")
                    except Exception:
                        edge_lengths[edge_id] = 100.0

                rsu_id, dist = self.get_nearest_rsu(x, y)
                veh_rsu_map[veh_id] = rsu_id
                veh_raw[veh_id] = {
                    "x": x, "y": y, "speed": speed,
                    "edge": edge_id, "lane_pos": lane_pos,
                    "edge_len": edge_lengths[edge_id],
                    "rsu_id": rsu_id, "rsu_dist": dist,
                }
            except Exception:
                continue

        # 更新 RSU 負載
        self.rsu_manager.update(veh_rsu_map)

        # 第二輪：計算完整 state vector
        states = {}
        cloud_latency_ms = 80.0  # 固定雲端延遲代理值

        for veh_id, raw in veh_raw.items():
            rsu_id   = raw["rsu_id"]
            dist     = raw["rsu_dist"]
            n_nearby = self.rsu_manager.get_nearby_count(rsu_id)
            ch       = estimate_channel(dist, n_nearby)
            cpu_load = self.rsu_manager.get_load(rsu_id)
            edge_len = max(raw["edge_len"], 1.0)
            pos_ratio = min(1.0, raw["lane_pos"] / edge_len)

            state = [
                min(1.0, raw["speed"] / 30.0),          # [0] veh_speed
                pos_ratio,                                # [1] veh_pos
                rsu_id / 3.0,                             # [2] nearest_rsu_id
                min(1.0, dist / 300.0),                   # [3] rsu_distance
                ch["bw_mbps"] / MAX_BW_MBPS,             # [4] channel_bw
                min(1.0, ch["delay_ms"] / 50.0),         # [5] channel_delay
                cpu_load,                                 # [6] rsu_cpu_load
                min(1.0, cloud_latency_ms / 200.0),      # [7] cloud_latency
                TASK_SIZE_MB / 10.0,                      # [8] task_size
                TASK_DEADLINE / 2.0,                      # [9] task_deadline
            ]
            states[veh_id] = state

        self.step += 1
        return states

    def run(self, step_interval=1):
        """持續萃取並寫入 JSONL 檔，每 step_interval 步輸出一次"""
        print(f"\n[DT] 初始化 RSU 座標...")
        self.init_rsu_positions()
        print(f"[DT] 開始萃取，輸出至 {self.output_file}")
        print(f"[DT] 每 {step_interval} 步輸出一次。按 Ctrl+C 停止。\n")

        with open(self.output_file, "w", encoding="utf-8") as f:
            while True:
                try:
                    traci.simulationStep()
                    sim_time = traci.simulation.getTime()

                    if int(sim_time) % step_interval == 0:
                        states = self.extract()
                        if states:
                            record = {
                                "sim_time": sim_time,
                                "step": self.step,
                                "num_vehicles": len(states),
                                "states": states,
                                "rsu_loads": {
                                    str(r["id"]): self.rsu_manager.get_load(r["id"])
                                    for r in RSU_LIST
                                }
                            }
                            f.write(json.dumps(record) + "\n")
                            f.flush()

                            if self.verbose:
                                print(f"[t={sim_time:.0f}s] {len(states)} 輛車 | "
                                      f"RSU loads: "
                                      + " ".join(f"R{r['id']}={self.rsu_manager.get_load(r['id']):.2f}"
                                                 for r in RSU_LIST))

                except traci.exceptions.FatalTraCIError:
                    print("[DT] SUMO 模擬結束。")
                    break
                except KeyboardInterrupt:
                    print("\n[DT] 手動停止。")
                    break


# ── 直接讀取最新狀態（供演算法呼叫）────────────────────────────────
def get_current_states(extractor: DTStateExtractor) -> dict:
    """
    供 DQN 環境呼叫：
    回傳 {veh_id: np.array(10,)} 的字典
    """
    try:
        import numpy as np
        states = extractor.extract()
        return {vid: np.array(s, dtype=np.float32) for vid, s in states.items()}
    except Exception as e:
        print(f"[DT] 狀態萃取失敗：{e}")
        return {}


# ── 主程式 ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DT State Extractor")
    parser.add_argument("--port",      type=int, default=8813)
    parser.add_argument("--sumo-cfg",  default="osm.sumocfg")
    parser.add_argument("--no-gui",    action="store_true")
    parser.add_argument("--output",    default="dt_states.jsonl")
    parser.add_argument("--interval",  type=int, default=1, help="每幾步輸出一次")
    parser.add_argument("--connect-only", action="store_true", help="連線到已啟動的 SUMO")
    args = parser.parse_args()

    proc = None
    if not args.connect_only:
        binary = "sumo" if args.no_gui else "sumo-gui"
        cmd = [binary, f"--remote-port", str(args.port), "-c", args.sumo_cfg]
        print(f"啟動 SUMO：{' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        time.sleep(4 if not args.no_gui else 2)

    traci.connect(port=args.port)
    print(f"已連線 TraCI (port={args.port})")

    extractor = DTStateExtractor(output_file=args.output)
    try:
        extractor.run(step_interval=args.interval)
    finally:
        try:
            traci.close()
        except Exception:
            pass
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
