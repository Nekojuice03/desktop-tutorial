"""
方法驗證套件（verify_invariants.py）
====================================
把各機制「理論上必然成立的性質(不變量)」做成自動檢查——改參數/改程式後跑一次，
全部 PASS 代表模型自洽；任何 FAIL 都指出被破壞的物理假設。不需 SUMO。

執行：python verify_invariants.py
"""
import numpy as np

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
check("帳務：consumer_left ⊆ break_failed",
      s["consumer_left"] <= s["break_failed"])
check("回合有實際處理任務(>200)", st["latency_n"] > 200, f"{st['latency_n']} 筆")
env.close()

print("\n" + "=" * 60)
fails = [n for n, ok, _ in RESULTS if not ok]
print(f"共 {len(RESULTS)} 項檢查：{len(RESULTS)-len(fails)} PASS / {len(fails)} FAIL")
if fails:
    print("FAIL 項目：", fails)
    raise SystemExit(1)
print("全部不變量成立 —— 模型自洽 ✓")
