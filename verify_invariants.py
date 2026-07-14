"""
方法驗證套件（verify_invariants.py）
====================================
把各機制「理論上必然成立的性質(不變量)」做成自動檢查——改參數/改程式後跑一次，
全部 PASS 代表模型自洽；任何 FAIL 都指出被破壞的物理假設。不需 SUMO。

執行：python verify_invariants.py
"""
import numpy as np

from digital_twin import DigitalTwinStateStore
from comm_model import data_rate, transmission_delay, contact_time
from infra_config import (V2V_LINK, V2I_LINK, V2V_RANGE_M, RSU_RANGE_M,
                          VEHICLE_CPU, STRONG_VEHICLE_CPU,
                          STRONG_VEHICLE_ENERGY_PER_CYCLE)
from nodes import build_nodes, estimate
from task_model import Task, TaskGenerator

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


def make_task(data=3e6, cyc=0.8e9, ddl=1.5):
    return Task("t", "s", "nav", data_bits=data, cpu_cycles=cyc,
                deadline_s=ddl, created_at=0.0, result_bits=data * 0.1)


def world():
    nodes = build_nodes({"rsu_0": {"x": 0.0, "y": 0.0}})
    nodes.register("self", VEHICLE_CPU, "vehicle")
    nodes.register("nb", VEHICLE_CPU, "vehicle")
    nodes.register("sg", STRONG_VEHICLE_CPU, "vehicle",
                   energy_per_cycle=STRONG_VEHICLE_ENERGY_PER_CYCLE)
    return nodes


print("=== [1] 通訊模型(鏈路預算) ===")
r10, r100, r200 = (data_rate(d, V2I_LINK) for d in (10, 100, 200))
check("速率隨距離單調遞減", r10 > r100 > r200,
      f"{r10/1e6:.0f} > {r100/1e6:.0f} > {r200/1e6:.0f} Mbps")
check("V2I 近距速率在 C-V2X 合理區間(50~500Mbps)", 50e6 < r10 < 500e6)
check("超出覆蓋 → 延遲無限大",
      transmission_delay(1e6, RSU_RANGE_M + 1, V2I_LINK) == float("inf"))
check("同距離下 V2I(20MHz) 速率 > V2V(10MHz)",
      data_rate(50, V2I_LINK) > data_rate(50, V2V_LINK))

print("\n=== [2] 延遲分解與佇列 ===")
nodes = world()
t = make_task()
res = {k: estimate(t, k, 0.0, (50, 0), nodes,
                   target_id=i, target_pos=p, commit=False)
       for k, i, p in (("local", "self", None), ("v2v", "nb", (80, 0)),
                       ("rsu", "rsu_0", (0, 0)), ("cloud", "cloud", (0, 0)))}
check("本地無傳輸延遲", res["local"]["breakdown"]["uplink"] == 0)
check("雲端上行含 V2I存取+骨幹傳播(>60ms)",
      res["cloud"]["breakdown"]["uplink"] > 0.06)
waits = []
for i in range(4):
    r = estimate(make_task(cyc=2e9), "rsu", 0.0, (50, 0), nodes,
                 target_id="rsu_0", target_pos=(0, 0), commit=True)
    waits.append(r["breakdown"]["wait"])
check("RSU FIFO 佇列等待單調遞增", all(waits[i] < waits[i+1] for i in range(3)),
      "→".join(f"{w*1000:.0f}ms" for w in waits))
bw = []
for i in range(4):
    r = estimate(make_task(data=8e6, cyc=2e9), "cloud", 0.0, (50, 0), nodes,
                 target_id="cloud", target_pos=(0, 0), commit=True)
    bw.append(r["latency"])
check("回程壅塞：連續上雲延遲單調遞增", all(bw[i] < bw[i+1] for i in range(3)),
      "→".join(f"{w*1000:.0f}ms" for w in bw))

print("\n=== [3] 能耗與成本階層 ===")
nodes = world()
t = make_task(cyc=4e9)
e = {k: estimate(t, k, 0.0, (50, 0), nodes, target_id=i, target_pos=p, commit=False)
     for k, i, p in (("local", "self", None), ("v2v", "nb", (80, 0)),
                     ("rsu", "rsu_0", (0, 0)), ("cloud", "cloud", (0, 0)))}
check("能耗階層：本地 < RSU < 雲端",
      e["local"]["energy"] < e["rsu"]["energy"] < e["cloud"]["energy"],
      f"{e['local']['energy']:.2f} < {e['rsu']['energy']:.2f} < {e['cloud']['energy']:.2f} J")
check("成本階層：本地=V2V=0 < RSU < 雲端",
      e["local"]["cost"] == e["v2v"]["cost"] == 0 and
      0 < e["rsu"]["cost"] < e["cloud"]["cost"])
rs = estimate(t, "v2v", 0.0, (50, 0), nodes, target_id="sg", target_pos=(80, 0))
rw = e["v2v"]
check("κf²：強車比弱車 快、且較耗能",
      rs["latency"] < rw["latency"] and rs["energy"] > rw["energy"],
      f"快{rw['latency']/rs['latency']:.1f}x 耗能{rs['energy']/rw['energy']:.2f}x")

print("\n=== [4] 移動性 ===")
same = contact_time((0, 0), (15, 0), (50, 0), (15, 0), V2V_RANGE_M)
opp = contact_time((0, 0), (15, 0), (50, 0), (-15, 0), V2V_RANGE_M)
check("同向同速 contact=上限、相向 contact 短", same == 60.0 and opp < 8.0,
      f"{same:.0f}s vs {opp:.1f}s")
nodes = world()
t = make_task()
r_ok = estimate(t, "v2v", 0.0, (50, 0), nodes, target_id="nb", target_pos=(80, 0),
                contact_s=10.0)
r_brk = estimate(t, "v2v", 0.0, (50, 0), nodes, target_id="nb", target_pos=(80, 0),
                 contact_s=0.3)
check("sojourn 約束：contact 足夠→可行；不足→link_break",
      r_ok["feasible"] and (not r_brk["feasible"]) and r_brk["link_break"])

print("\n=== [5] 任務模型 ===")
gen = TaskGenerator(arrival_rate=0.2, seed=1)
tasks = []
for step in range(500):
    tasks += gen.step(["c1", "c2"], now=float(step))
n_exp = 500 * 2 * 0.2
check("到達率符合期望(±15%)", abs(len(tasks) - n_exp) / n_exp < 0.15,
      f"{len(tasks)} vs 期望 {n_exp:.0f}")
kinds = {k: sum(1 for x in tasks if x.kind == k) for k in ("sensor", "nav", "vision")}
check("類型權重：sensor 最多", kinds["sensor"] > kinds["nav"] and
      kinds["sensor"] > kinds["vision"], str(kinds))

print("\n=== [6] 卡曼預判器(EKF-CTRV，僅 BSM 觀測量) ===")
import math as _m
from kalman_tracker import EKFCTRV, predict_contact, _ctrv_step
tr = EKFCTRV()
for k in range(10):
    tr.step(10.0 * k, 0.0, 10.0, 0.0, 1.0)
px, py = tr.forward_position(5.0)
check("直行車 5s 位置預測誤差 < 5m", _m.dist((px, py), (140, 0)) < 5,
      f"({px:.1f},{py:.1f}) vs (140,0)")
om_true, v, th = 0.2, 10.0, 0.0
x = y = 0.0
tr2 = EKFCTRV()
for k in range(15):
    tr2.step(x, y, v, th, 1.0)
    x, y, v, th, _ = _ctrv_step(x, y, v, th, om_true, 1.0)
check("轉彎率 ω 估計收斂(|ω̂-0.2|<0.05)", abs(tr2.omega - om_true) < 0.05,
      f"ω̂={tr2.omega:.3f}")


def _feed_pair(turn_last_k):
    trA, trB = EKFCTRV(), EKFCTRV()
    xB, yB, thB = 0.0, 30.0, 0.0
    for k in range(12):
        trA.step(10.0 * k, 0.0, 10.0, 0.0, 1.0)
        trB.step(xB, yB, 10.0, thB, 1.0)
        om = 0.25 if k >= 12 - turn_last_k else 0.0
        xB, yB, _, thB, _ = _ctrv_step(xB, yB, 10.0, thB, om, 1.0)
    return predict_contact(trA, trB, 150.0)


c_str, c_turn = _feed_pair(0), _feed_pair(4)
check("轉彎預判：對方開始轉彎 → 預測 contact 明顯縮短",
      c_turn < c_str * 0.7, f"直行 {c_str:.0f}s vs 轉彎 {c_turn:.0f}s")

print("\n=== [7] 環境層(帳務與觀測) ===")
from vec_env_ma import VECMultiEnv
env = VECMultiEnv(mock=True, arrival_rate=0.5, mock_vehicles=24, server_ratio=0.45,
                  episode_ticks=200, task_cpu_scale=1.0)
obs, state = env.reset(seed=42)
check("觀測維度 18、值域[0,1]", obs.shape[1] == 18 and
      float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0)
check("全域狀態維度 = RSU數+5", state.shape[0] == len(env.rsu_ids) + 5)
masks = env.current_action_masks()
check("action mask 維度正確且 local 永遠可用",
      masks.shape == (obs.shape[0], env.n_actions) and bool(masks[:, 0].all()))
sampled = env.sample_valid_actions(np.random.default_rng(123))
check("random baseline 只抽樣孿生視角中的可行動作",
      all(masks[i, action] for i, action in enumerate(sampled)))
rng = np.random.default_rng(42)
while True:
    rewards, obs, state, done, info = env.step(
        rng.integers(0, env.n_actions, size=obs.shape[0]))
    if done:
        break
s, st = info["episode_stats"], env.stats
check("帳務：latency_n = success + deadline_miss",
      st["latency_n"] == st["success"] + st["deadline_miss"])
check("帳務：fail = miss + infeasible + pred_reject + break_failed",
      st["fail"] == st["deadline_miss"] + st["infeasible"]
      + st["pred_reject"] + st["break_failed"])
check("帳務：link_break = recovered + failed",
      s["link_break"] == s["break_recovered"] + s["break_failed"])
check("帳務：consumer_left ≤ link_break(車主離場為斷線事件子類)",
      s["consumer_left"] <= s["link_break"])
check("加權成功率存在且介於 0~1",
      0.0 <= s.get("weighted_success_rate", -1) <= 1.0)
check("每筆成功/失敗恰好進入加權統計一次",
      s.get("weighted_outcome_n", -1) == st["success"] + st["fail"])

print("\n=== [8] 數位孿生時間邊界與 masked policy ===")
store = DigitalTwinStateStore(delay_ticks=2)
physical = {"v": {"pos": (0.0, 0.0), "speed": 1.0, "angle": 0.0}}
store.update(0.0, physical)
physical["v"]["pos"] = (99.0, 0.0)  # 呼叫端後續突變不得污染已存快照
store.update(1.0, {"v": {"pos": (1.0, 0.0), "speed": 1.0, "angle": 0.0}})
snap = store.update(2.0, {"v": {"pos": (2.0, 0.0), "speed": 1.0, "angle": 0.0}})
check("τ=2 時決策讀 t-2 快照，且不被物理 dict 突變污染",
      snap.states["v"]["pos"] == (0.0, 0.0) and snap.age_s == 2.0)

# 終點強制結算發生在世界推進期間；驗證該 correction 會回到當前 transition，
# 不會像舊實作一樣留在 _settle_delta 後直接結束回合。
env_term = VECMultiEnv(mock=True, arrival_rate=0.2, mock_vehicles=8,
                       episode_ticks=5)
env_term.reset(seed=77)
env_term._active = [("dummy", {"now": 0.0})]
env_term._resolve_one = lambda ctx, action: 0.0
def _finish_with_delta():
    env_term._settle_delta = 3.0
    return False
env_term._advance_to_active = _finish_with_delta
terminal_rewards, _, _, terminal_done, terminal_info = env_term.step(np.array([0]))
check("terminal settlement reward 不遺失",
      terminal_done and np.isclose(terminal_rewards[0], 3.0)
      and np.isclose(terminal_info.get("settlement_delta", 0.0), 3.0))
env_term.close()

# 任務優先權：profile 抽樣範圍 + priority_aware 觀測維度
from task_model import TASK_PROFILES as _TP
gen2 = TaskGenerator(arrival_rate=1.0, seed=5)
_ts = []
for step in range(80):
    _ts += gen2.step(["c1"], now=float(step))
ok_rng = all(_TP[t.kind]["priority"][0] <= t.priority <= _TP[t.kind]["priority"][1]
             for t in _ts)
check("任務優先權落在各 profile 範圍(sensor/nav/vision)", ok_rng)
env_p = VECMultiEnv(mock=True, arrival_rate=0.5, mock_vehicles=24,
                    server_ratio=0.45, episode_ticks=30, priority_aware=True)
obs_p, _ = env_p.reset(seed=9)
check("priority_aware：觀測 19 維且值域[0,1]",
      env_p.n_features == 19 and obs_p.shape[1] == 19
      and float(obs_p.min()) >= 0.0 and float(obs_p.max()) <= 1.0)
env_p.close()

env_tq = VECMultiEnv(mock=True, arrival_rate=0.5, mock_vehicles=24,
                     server_ratio=0.45, episode_ticks=30,
                     twin_quality_aware=True)
obs_tq, state_tq = env_tq.reset(seed=10)
check("twin_quality：觀測 +2、state +2，且值域[0,1]",
      env_tq.n_features == 20 and obs_tq.shape[1] == 20
      and state_tq.shape[0] == len(env_tq.rsu_ids) + 7
      and float(obs_tq.min()) >= 0.0 and float(obs_tq.max()) <= 1.0)
env_tq.close()

# 抵達補送(arrival_delivery)白箱測試：mock 車輛不會離場，直接餵一筆
# 「車主已離場」的在途任務給結算器，驗證補送路徑與計數正確。
from task_model import Task as _Task
env_ad = VECMultiEnv(mock=True, arrival_rate=0.5, mock_vehicles=24,
                     server_ratio=0.45, episode_ticks=10, arrival_delivery=True)
env_ad.reset(seed=3)
_t = _Task("tad", "cX", "nav", data_bits=2e6, cpu_cycles=0.5e9,
           deadline_s=5.0, created_at=0.0, result_bits=2e5)
env_ad._last_seen["ghost_owner"] = (100.0, 0.0)   # 停靠處在 rsu_0(0,0) 覆蓋內
_before = {k: env_ad.stats[k] for k in
           ("arrival_delivered", "break_recovered", "consumer_left",
            "break_failed", "success")}
env_ad._settle_one({"mode": "infra", "task": _t, "helper": "rsu_0",
                    "holder": "ghost_owner", "serving_rsu": "rsu_0",
                    "t_done": 0.0, "total": 0.5, "energy": 1.0,
                    "cost": 0.0, "pred_reward": -1.0})
sa = env_ad.stats
_d = {k: sa[k] - _before[k] for k in _before}   # 差分：不受 reset 期間 fallback 影響
check("抵達補送：車主離場經 RSU 補送成功且計數正確",
      _d["arrival_delivered"] == 1 and _d["break_recovered"] == 1
      and _d["consumer_left"] == 1 and _d["break_failed"] == 0
      and _d["success"] == 1)
env_ad.close()
check("回合有實際處理任務(>200)", st["latency_n"] > 200, f"{st['latency_n']} 筆")
env.close()

print("\n=== [9] 和平東路場景與 VD 邊界流量 ===")
import os as _os
import xml.etree.ElementTree as _ET
from scenario_config import (resolve_scenario as _resolve_scenario,
                             model_suffix as _model_suffix,
                             scenario_cli as _scenario_cli,
                             vd_split_cli as _vd_split_cli)
from vd_provider import (load_mapping as _load_mapping,
                         rates_from_snapshot as _rates_from_snapshot,
                         _read_net_routes as _read_net_routes,
                         select_replay_files as _select_replay_files)
from vec_env import TraciWorld as _TraciWorld
_sc = _resolve_scenario("heping")
_rows = _load_mapping(_sc.vd_mapping)
_net_root = _ET.parse(_sc.net_file).getroot()
_edge_ids = {e.get("id") for e in _net_root.findall("edge")
             if e.get("id") and not e.get("function")}
check("和平東路場景資產存在且 8/8 VD edge 有效",
      _os.path.isfile(_sc.sumocfg) and _os.path.isfile(_sc.rsu_positions)
      and len(_rows) == 8
      and all(r["SumoEdgeID"] in _edge_ids for r in _rows))
_section_xml = "<root>" + "".join(
    f"<SectionData><SectionId>{did}</SectionId><TotalVol>{vol}</TotalVol>"
    "<DataCollectTimeInterval>5</DataCollectTimeInterval></SectionData>"
    for did, vol in (("ZFZK620", 100), ("ZF9KB40", 50),
                     ("ZFYKD00", 40), ("ZFYKD40", 30))) + "</root>"
_rates, _ = _rates_from_snapshot(_rows, _section_xml)
check("Section VD 流量依 FlowShare 分配且不重複計數",
      len(_rates) == 8 and np.isclose(sum(_rates), 1980.0))
_device_xml = """<root><VDDevice><DeviceID>ZFZK620</DeviceID>
<DataCollectTimeInterval>5</DataCollectTimeInterval><LaneData><LaneNO>0</LaneNO>
<Svolume>100</Svolume></LaneData></VDDevice></root>"""
_lane_rates, _ = _rates_from_snapshot(_rows, _device_xml)
check("設備級 lane 流量也依 FlowShare 分配而不重複",
      np.isclose(sum(_lane_rates), 1200.0))
_z_rows = _load_mapping(_resolve_scenario("zhongxiao").vd_mapping)
_z_device_xml = "<root><VDDevice><DeviceID>VJQJI20</DeviceID>" \
    "<DataCollectTimeInterval>5</DataCollectTimeInterval>" + "".join(
        f"<LaneData><LaneNO>{lane}</LaneNO><Svolume>100</Svolume></LaneData>"
        for lane in range(6)) + "</VDDevice></root>"
_z_lane_rates, _ = _rates_from_snapshot(_z_rows, _z_device_xml)
check("設備級不同 lane 各自保留量測，不被 device 列數二次均分",
      np.isclose(sum(_z_lane_rates), 7200.0))
_routes = _read_net_routes(_sc.net_file, {r["SumoEdgeID"] for r in _rows})
check("每個和平東路 VD 邊界 edge 都有可注入 SUMO route", len(_routes) == 8)
check("場景/VD 模式納入 checkpoint 名稱，避免模型混用",
      _model_suffix(scenario=_sc, vd_mode="replay", twin_quality=True)
      == "_heping_replay_tq")
_cli_sc, _cli_mode = _scenario_cli(
    ["train_mappo.py", "--sumo", "--scenario", "heping", "--vd-mode", "replay"],
    True)
check("scenario CLI 同時接受 '--key value' 寫法",
      _cli_sc.name == "heping" and _cli_mode == "replay")
_files = [f"VD_{i:02}.xml" for i in range(10)]
_tr = _select_replay_files(_files, "train")
_va = _select_replay_files(_files, "validation")
_te = _select_replay_files(_files, "test")
check("VD 依時間切成 70/15/15 且三組互斥",
      len(_tr) == 7 and len(_va) == 1 and len(_te) == 2
      and not (set(_tr) & set(_va) or set(_tr) & set(_te) or set(_va) & set(_te)))
check("--vd-split 可指定未見 test partition",
      _vd_split_cli(["eval.py", "--vd-split", "test"], True, "replay") == "test")
_tw1 = _TraciWorld(_sc.sumocfg)
_tw2 = _TraciWorld(_sc.sumocfg)
check("訓練與評估配置不同 TraCI label，避免全域連線互換",
      _tw1.label != _tw2.label)

print("\n" + "=" * 60)
fails = [n for n, ok, _ in RESULTS if not ok]
print(f"共 {len(RESULTS)} 項檢查：{len(RESULTS)-len(fails)} PASS / {len(fails)} FAIL")
if fails:
    print("FAIL 項目：", fails)
    raise SystemExit(1)
print("全部不變量成立 —— 模型自洽 ✓")
