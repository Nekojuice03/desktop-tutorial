"""
SUMO 整合 + 啟發式基準（run_baseline.py）
==========================================
把四個模組(infra_config / comm_model / task_model / nodes)接進 SUMO，
用 TraCI 取得真實車輛位置與速度，跑一個簡單的「就近最快」啟發式策略，
產生 baseline 數字(成功率、平均延遲、卸載分布)。
之後 RL 的目標就是贏過這個 baseline。

v1 簡化模型(已標註，之後可加強)：
  - 車輛分 server(決策者) / client(產任務)，全部都是可運算節點
  - client 產生任務 → 指派給範圍內最近的 server(計一跳 V2V 傳輸)
    沒有 server 在範圍 → client 自己處理
  - 持有者(server 或 client)用啟發式選：本地 / 最近基站 / 雲端 中最快的可行者
    (V2V 之間的再卸載留給 RL；baseline 先不做)

執行(在你電腦，需安裝 SUMO)：
  python run_baseline.py        # 無視窗，跑得快
  python run_baseline.py --gui  # 開 sumo-gui 看畫面
"""
import os
import sys
import zlib
from collections import Counter

from comm_model import (distance, transmission_delay,
                        neighbors_in_range, rsus_in_range)
from nodes import build_nodes, estimate, INF
from task_model import TaskGenerator
from infra_config import (VEHICLE_CPU, V2V_BANDWIDTH_HZ, V2V_RANGE_M,
                          RSU_RANGE_M, load_rsus)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== 可調參數 ======
NUM_STEPS = 1000      # 跑幾個模擬步(1 步約 1 秒)
SERVER_RATIO = 0.3    # 多少比例的車當 server
ARRIVAL_RATE = 0.1    # 每台 client 每秒平均產生幾個任務
SEED = 0
REPORT_EVERY = 200    # 每幾步印一次進度


# ---------- 角色(穩定的雜湊分配，跨次執行一致) ----------
def is_server(veh_id, server_ratio=SERVER_RATIO):
    return (zlib.crc32(veh_id.encode()) % 1000) < server_ratio * 1000


def is_strong(veh_id, strong_ratio=0.2):
    """在 server 之中，判斷是否為強車(高算力)。用不同 salt 與 is_server 獨立。"""
    return (zlib.crc32(("strong_" + veh_id).encode()) % 1000) < strong_ratio * 1000


# ---------- 啟發式決策：本地 / 最近基站 / 雲端 中挑最快的可行者 ----------
def heuristic_decide(task, holder_id, holder_pos, nodes, rsus, now):
    cands = [("local", holder_id, None)]
    near = rsus_in_range(holder_pos, rsus, RSU_RANGE_M)
    if near:
        rid = near[0]
        rpos = (rsus[rid]["x"], rsus[rid]["y"])
        cands.append(("rsu", rid, rpos))
        cands.append(("cloud", "cloud", rpos))   # 經最近基站轉送雲端
    best = None
    for kind, tid, tpos in cands:
        r = estimate(task, kind, now, holder_pos, nodes,
                     target_id=tid, target_pos=tpos, commit=False)
        if r["feasible"] and (best is None or r["latency"] < best[3]["latency"]):
            best = (kind, tid, tpos, r)
    if best is None:
        return "none", None, {"feasible": False, "latency": INF,
                              "energy": INF, "breakdown": {}}
    kind, tid, tpos, r = best
    estimate(task, kind, now, holder_pos, nodes,
             target_id=tid, target_pos=tpos, commit=True)   # 真的佔用選中的節點
    return kind, tid, r


# ---------- 統計容器 ----------
def new_stats():
    return {"generated": 0, "assigned_to_server": 0, "self_served": 0,
            "success": 0, "fail": 0, "deadline_miss": 0,
            "latency_sum": 0.0, "latency_n": 0, "by_target": Counter()}


# ---------- 一個時間步的核心處理(純邏輯，可用假資料測試) ----------
def process_step(now, dt, veh_states, roles, nodes, gen, rsus, stats,
                 server_ratio=SERVER_RATIO):
    # 1. 新車：註冊為運算節點 + 決定角色
    for vid in veh_states:
        if not nodes.has(vid):
            nodes.register(vid, VEHICLE_CPU, "vehicle")
        if vid not in roles:
            roles[vid] = "server" if is_server(vid, server_ratio) else "client"

    servers = {vid: s["pos"] for vid, s in veh_states.items()
               if roles.get(vid) == "server"}
    clients = [vid for vid in veh_states if roles.get(vid) == "client"]

    # 2. client 產生任務
    new_tasks = gen.step(clients, now, dt)
    stats["generated"] += len(new_tasks)

    # 3. 每個任務：指派 + 決策 + 計延遲
    for task in new_tasks:
        cpos = veh_states[task.source]["pos"]
        hop = 0.0
        holder_id, holder_pos = task.source, cpos

        # 找範圍內最近的(別的)server
        others = {sid: p for sid, p in servers.items() if sid != task.source}
        near_servers = neighbors_in_range(cpos, others, V2V_RANGE_M)
        if near_servers:
            sid = near_servers[0]
            holder_id, holder_pos = sid, servers[sid]
            hop = transmission_delay(task.data_bits, distance(cpos, holder_pos),
                                     V2V_BANDWIDTH_HZ, V2V_RANGE_M)
            stats["assigned_to_server"] += 1
        else:
            stats["self_served"] += 1

        if hop == INF:   # 指派的 server 其實連不到
            stats["fail"] += 1
            stats["by_target"]["infeasible"] += 1
            continue

        kind, tid, r = heuristic_decide(task, holder_id, holder_pos,
                                        nodes, rsus, now + hop)
        if not r["feasible"]:
            stats["fail"] += 1
            stats["by_target"]["infeasible"] += 1
            continue

        total = hop + r["latency"]
        stats["latency_sum"] += total
        stats["latency_n"] += 1
        stats["by_target"][kind] += 1
        if total <= task.deadline_s:
            stats["success"] += 1
        else:
            stats["fail"] += 1
            stats["deadline_miss"] += 1


# ---------- 進度與總結列印 ----------
def print_progress(step, now, nveh, s):
    done = s["success"] + s["fail"]
    avg = s["latency_sum"] / s["latency_n"] * 1000 if s["latency_n"] else 0
    succ = s["success"] / done * 100 if done else 0
    print(f"  step {step:>4} (t={now:5.0f}s) 車{nveh:>3} ｜ "
          f"任務{s['generated']:>5} 成功率{succ:5.1f}% 平均延遲{avg:6.1f}ms")


def print_summary(s):
    done = s["success"] + s["fail"]
    avg = s["latency_sum"] / s["latency_n"] * 1000 if s["latency_n"] else 0
    print("\n===== 基準(啟發式：就近最快) 結果 =====")
    print(f" 產生任務      : {s['generated']}")
    print(f" 指派給 server : {s['assigned_to_server']}，client 自理：{s['self_served']}")
    if done:
        print(f" 成功(達 deadline): {s['success']}（{s['success']/done*100:.1f}%）")
        print(f" 失敗          : {s['fail']}（其中超時 {s['deadline_miss']}）")
    print(f" 平均延遲      : {avg:.1f} ms")
    print(f" 卸載分布      : {dict(s['by_target'])}")
    print(" （這是 baseline；之後 RL 要贏過這個成功率與延遲）")


# ---------- 真正連 SUMO 跑(在你電腦執行) ----------
def run(num_steps=NUM_STEPS, gui=False, seed=SEED):
    os.chdir(SCRIPT_DIR)   # 確保 SUMO 找得到相對路徑的網路檔
    # 找 sumocfg
    cfg = None
    for name in ("osm.sumocfg",):
        if os.path.exists(name):
            cfg = name
    if cfg is None:
        import glob
        hits = glob.glob("*.sumocfg")
        cfg = hits[0] if hits else None
    if cfg is None:
        print("找不到 .sumocfg，請確認此腳本與設定檔在同一資料夾。")
        return

    try:
        import traci
        from sumolib import checkBinary
    except ImportError:
        print("找不到 traci / sumolib，請確認已安裝 SUMO 並設定好 PYTHONPATH（SUMO_HOME/tools）。")
        return

    rsus_full = load_rsus()   # {rid:{x,y,cpu,range,load}}
    rsus = {rid: {"x": v["x"], "y": v["y"]} for rid, v in rsus_full.items()}
    nodes = build_nodes(rsus)
    gen = TaskGenerator(arrival_rate=ARRIVAL_RATE, seed=seed)
    roles = {}
    stats = new_stats()

    sumo_bin = checkBinary("sumo-gui" if gui else "sumo")
    print(f"啟動 SUMO（{'GUI' if gui else '無視窗'}）：{cfg}，基站 {len(rsus)} 個")
    traci.start([sumo_bin, "-c", cfg, "--start", "--quit-on-end"])
    try:
        dt = traci.simulation.getDeltaT()
        for step in range(num_steps):
            traci.simulationStep()
            now = traci.simulation.getTime()
            ids = traci.vehicle.getIDList()
            veh_states = {
                vid: {"pos": traci.vehicle.getPosition(vid),
                      "speed": traci.vehicle.getSpeed(vid),
                      "angle": traci.vehicle.getAngle(vid)}
                for vid in ids
            }
            process_step(now, dt, veh_states, roles, nodes, gen, rsus, stats)
            if (step + 1) % REPORT_EVERY == 0:
                print_progress(step + 1, now, len(ids), stats)
            if traci.simulation.getMinExpectedNumber() <= 0:
                print("（路網已無車輛，提前結束）")
                break
    finally:
        traci.close()

    print_summary(stats)


if __name__ == "__main__":
    run(gui=("--gui" in sys.argv))
