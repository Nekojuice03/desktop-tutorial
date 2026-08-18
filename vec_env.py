"""
車載卸載 RL 環境（vec_env.py）
================================
把前面所有模組 + SUMO 包成 Gymnasium 環境，讓策略學習卸載決策。
重用 run_baseline 的整合方式，只是把「就近最快」的固定規則，
換成「agent 自己學的策略」——底層完全相同，所以可信。

決策形式（單一共享策略 = 參數共享版的多智能體）：
  每一步有「一個待決策的任務」，agent 從 4 個動作選一個：
    0=本地 1=V2V(最近鄰居server) 2=基站(最近) 3=雲端(經最近基站)
  獎勵 = -(總延遲 + 能耗權重×正規化能耗) - 超時/不可達懲罰
  目標：最小化延遲與能耗、避免超時與不可達。

兩種模式：
  mock=True ：用內建假車流，不需 SUMO、不需 torch，純驗證環境邏輯
  mock=False ：用 TraCI 連真實 SUMO（你電腦上跑）

直接執行 `python vec_env.py` 會跑 mock 模式 + 隨機動作的自我測試。
"""
import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from comm_model import (distance, transmission_delay, velocity_from_speed_angle,
                        neighbors_in_range, rsus_in_range, contact_time)
from nodes import build_nodes, estimate, INF
from task_model import TaskGenerator
from infra_config import (VEHICLE_CPU, V2V_LINK, V2V_RANGE_M,
                          RSU_RANGE_M, V2V_EXTRA_LATENCY, TX_POWER_W, load_rsus)
from run_baseline import is_server   # 重用角色分配

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 觀測特徵正規化常數（把數值縮到大約 [0,1]，DQN 比較好學）----
# ★CPU_NORM 取任務運算量上限(vision 6e9)：先前 2e9 會讓所有重任務削頂成 1.0，
#   agent 分不出 3G 和 6G cycle 的任務 → 失去「該上雲還是上邊緣」的關鍵資訊。
DATA_NORM, CPU_NORM, DEAD_NORM = 10e6, 6e9, 3.0
WAIT_NORM, CONTACT_NORM = 2.0, 60.0
NBR_NORM = 10.0      # 範圍內 server 數的正規化上限(競爭程度特徵)
N_FEATURES = 12   # +1：回程(backhaul)佇列等待 —— agent 須看得到雲端壅塞才公平

# ---- 動作 ----
ACTIONS = ["local", "v2v", "rsu", "cloud"]

# ---- 獎勵參數（可調）----
ENERGY_W = 1.0       # 能耗在獎勵中的權重
ENERGY_NORM = 5.0    # 能耗正規化常數(焦耳)：把能耗縮到與延遲(秒)相近的尺度，
                     #   避免「加入節點運算能耗」後能耗項把延遲項淹沒 → 同時穩定訓練曲線
COST_W = 1.0         # ★方法②：使用成本(pay-per-use)在獎勵中的權重。
                     #   設 0 = 關閉成本項；調高 = 更不鼓勵用雲(雲端 cost 最高)
PENALTY_MISS = 2.0   # 超過 deadline 的額外懲罰
PENALTY_FAIL = 5.0   # 動作不可達(例如選V2V卻無鄰居)的懲罰


def _clip01(v):
    return float(min(1.0, max(0.0, v)))


# ==================================================================
# 世界來源：真實 SUMO / 假車流，介面相同(reset / step / close)
# ==================================================================
class TraciWorld:
    """用 TraCI 連真實 SUMO。"""
    def __init__(self, cfg, gui=False):
        self.cfg = cfg
        self.gui = gui
        self.dt = 1.0
        self.done = False
        self._t = None

    def reset(self):
        import traci
        from sumolib import checkBinary
        self._t = traci
        if traci.isLoaded():
            traci.close()
        sumo_bin = checkBinary("sumo-gui" if self.gui else "sumo")
        traci.start([sumo_bin, "-c", self.cfg, "--start", "--quit-on-end"])
        self.dt = traci.simulation.getDeltaT()
        self.done = False

    def step(self):
        t = self._t
        t.simulationStep()
        now = t.simulation.getTime()
        ids = t.vehicle.getIDList()
        states = {vid: {"pos": t.vehicle.getPosition(vid),
                        "speed": t.vehicle.getSpeed(vid),
                        "angle": t.vehicle.getAngle(vid)} for vid in ids}
        self.done = t.simulation.getMinExpectedNumber() <= 0
        return now, states

    def _route_exit_time(self, vid):
        """
        車輛把「剩餘路線」走完、抵達終點駛離模擬還需幾秒。
        ★SUMO 消融實測的斷線主因：對象車抵達終點 despawn，而非開出通訊範圍——
          所以預判必須含「離場時間」。紅燈停等會讓當前速度≈0 → 以車道限速三成
          為速度下限(等紅燈的車終究會走)，避免除以近零而永不預警。
        """
        t = self._t
        route = t.vehicle.getRoute(vid)
        idx = max(t.vehicle.getRouteIndex(vid), 0)
        remain = 0.0
        for e in route[idx:]:
            remain += t.lane.getLength(e + "_0")
        if t.vehicle.getRoadID(vid) == route[idx]:
            remain -= t.vehicle.getLanePosition(vid)
        v_floor = 0.3 * t.lane.getMaxSpeed(route[idx] + "_0")
        v = max(t.vehicle.getSpeed(vid), v_floor, 1.0)
        return max(remain, 0.0) / v

    def route_exit_time(self, vid):
        """單車離場時間的公開介面(V2I 移動性上限用)；查詢失敗回 None。"""
        try:
            return self._route_exit_time(vid)
        except Exception:
            return None

    def route_divergence_time(self, a, b):
        """
        兩車依「已知路線」還能保持連線多久(秒)；資訊不足回 None(退回等速外推)。
        物理意涵：模擬 V2X 意圖分享(SAE J2735 BSM Path Prediction / ETSI CAM)——
        數位孿生知道各車路線與「終點」，可預判：
          (1) 離場：任一車走完剩餘路線就會消失 → min(兩車離場時間)   ←斷線主因
          (2) 分歧：同路兩車在前方岔口分道 → 共同前綴長度 / 較快車速
        取各項下限。兩車不同路無法對齊時仍回傳離場上限(先前此情況整個放棄)。
        """
        t = self._t
        try:
            caps = [self._route_exit_time(a), self._route_exit_time(b)]

            ra = t.vehicle.getRoute(a); ia = max(t.vehicle.getRouteIndex(a), 0)
            rb = t.vehicle.getRoute(b); ib = max(t.vehicle.getRouteIndex(b), 0)
            fa, fb = list(ra[ia:]), list(rb[ib:])
            # 對齊：允許一車稍前(其當前邊出現在另一車最近的未來三條邊)
            aligned = bool(fa and fb)
            if aligned and fa[0] != fb[0]:
                if fb[0] in fa[:3]:
                    fa = fa[fa.index(fb[0]):]
                elif fa[0] in fb[:3]:
                    fb = fb[fb.index(fa[0]):]
                else:
                    aligned = False   # 不同路：分歧項不可得，僅用離場上限
            if aligned:
                shared = 0.0
                for i in range(min(len(fa), len(fb))):
                    if fa[i] != fb[i]:
                        break
                    length = t.lane.getLength(fa[i] + "_0")
                    if i == 0:
                        pos = 0.0
                        for vid in (a, b):
                            if t.vehicle.getRoadID(vid) == fa[0]:
                                pos = max(pos, t.vehicle.getLanePosition(vid))
                        length = max(0.0, length - pos)
                    shared += length
                v = max(t.vehicle.getSpeed(a), t.vehicle.getSpeed(b), 0.1)
                caps.append(shared / v)   # 較快者先走完共同路段 → 保守
            return min(caps)
        except Exception:
            return None

    def close(self):
        try:
            if self._t is not None and self._t.isLoaded():
                self._t.close()
        except Exception:
            pass


class MockWorld:
    """不需 SUMO 的假車流：幾台車在一條東西向走廊上來回移動。"""
    def __init__(self, n=8, seed=0):
        self.n = n
        self.seed = seed
        self.dt = 1.0
        self.done = False
        self.area_x = (-60.0, 300.0)
        self.area_y = (-40.0, 40.0)

    def reset(self):
        self.rng = random.Random(self.seed)
        self.t = 0
        self.pos = []
        self.vx = []
        for _ in range(self.n):
            x = self.rng.uniform(*self.area_x)
            y = self.rng.uniform(*self.area_y)
            sp = self.rng.uniform(5, 15) * self.rng.choice([1, -1])
            self.pos.append([x, y])
            self.vx.append(sp)
        self.done = False

    def step(self):
        self.t += 1
        states = {}
        for i in range(self.n):
            self.pos[i][0] += self.vx[i] * self.dt
            if self.pos[i][0] < self.area_x[0]:
                self.pos[i][0] = self.area_x[0]; self.vx[i] *= -1
            if self.pos[i][0] > self.area_x[1]:
                self.pos[i][0] = self.area_x[1]; self.vx[i] *= -1
            angle = 90 if self.vx[i] >= 0 else 270
            states[f"veh{i}"] = {"pos": (self.pos[i][0], self.pos[i][1]),
                                 "speed": abs(self.vx[i]), "angle": angle}
        self.done = False
        return float(self.t), states

    def route_divergence_time(self, a, b):
        """Mock 車流沒有路線資訊 → None(呼叫端退回等速外推)。"""
        return None

    def route_exit_time(self, vid):
        """Mock 車流沒有路線資訊 → None。"""
        return None

    def close(self):
        pass


# ==================================================================
# RL 環境
# ==================================================================
class VECEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, mock=True, gui=False, cfg="osm.sumocfg",
                 arrival_rate=0.1, server_ratio=0.3, seed=0,
                 episode_decisions=500, rsus=None, mock_vehicles=8):
        super().__init__()
        self.mock = mock
        self.gui = gui
        self.cfg = cfg
        self.arrival_rate = arrival_rate
        self.server_ratio = server_ratio
        self.episode_decisions = episode_decisions
        self.base_seed = seed
        self.mock_vehicles = mock_vehicles

        if rsus is not None:
            self.rsus = rsus
        elif mock:
            self.rsus = {"rsu_0": {"x": 0.0, "y": 0.0},
                         "rsu_1": {"x": 200.0, "y": 0.0}}
        else:
            os.chdir(SCRIPT_DIR)
            self.rsus = {rid: {"x": v["x"], "y": v["y"]}
                         for rid, v in load_rsus().items()}

        self.world = None
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_FEATURES,),
                                            dtype=np.float32)
        self._ep = 0

    # ---------- 內部：建立一個待決策任務的「情境」 ----------
    def _build_context(self, task, holder_id, holder_pos, now, hop, hop_energy=0.0):
        near = rsus_in_range(holder_pos, self.rsus, RSU_RANGE_M)
        rsu_id = near[0] if near else None
        rsu_pos = (self.rsus[rsu_id]["x"], self.rsus[rsu_id]["y"]) if rsu_id else None
        others = {sid: self.veh_states[sid]["pos"] for sid in self.servers
                  if sid != holder_id and sid in self.veh_states}
        nbrs = neighbors_in_range(holder_pos, others, V2V_RANGE_M)
        nb_id = nbrs[0] if nbrs else None
        nb_pos = self.veh_states[nb_id]["pos"] if nb_id else None
        return {"task": task, "holder_id": holder_id, "holder_pos": holder_pos,
                "now": now, "hop": hop, "hop_energy": hop_energy,
                "rsu_id": rsu_id, "rsu_pos": rsu_pos,
                "nb_id": nb_id, "nb_pos": nb_pos, "n_nbrs": len(nbrs)}

    def _vel_of(self, vid):
        """車輛的速度向量；不在場(或為靜止基站)回傳 (0,0)。"""
        if vid in self.veh_states:
            s = self.veh_states[vid]
            return velocity_from_speed_angle(s["speed"], s["angle"])
        return (0.0, 0.0)

    def _make_obs(self, ctx):
        task = ctx["task"]
        pos = ctx["holder_pos"]
        now = ctx["now"]

        local_wait = self.nodes.wait_time(ctx["holder_id"], now)

        if ctx["rsu_id"] is not None:
            rsu_in = 1.0
            rsu_dist = distance(pos, ctx["rsu_pos"])
            rsu_wait = self.nodes.wait_time(ctx["rsu_id"], now)
        else:
            rsu_in, rsu_dist, rsu_wait = 0.0, RSU_RANGE_M, 0.0

        if ctx["nb_id"] is not None:
            nb_in = 1.0
            nb_dist = distance(pos, ctx["nb_pos"])
            hv = velocity_from_speed_angle(
                self.veh_states[ctx["holder_id"]]["speed"],
                self.veh_states[ctx["holder_id"]]["angle"]) \
                if ctx["holder_id"] in self.veh_states else (0.0, 0.0)
            nv = velocity_from_speed_angle(self.veh_states[ctx["nb_id"]]["speed"],
                                           self.veh_states[ctx["nb_id"]]["angle"])
            nb_contact = contact_time(pos, hv, ctx["nb_pos"], nv, V2V_RANGE_M)
        else:
            nb_in, nb_dist, nb_contact = 0.0, V2V_RANGE_M, 0.0

        obs = np.array([
            _clip01(task.data_bits / DATA_NORM),
            _clip01(task.cpu_cycles / CPU_NORM),
            _clip01(task.deadline_s / DEAD_NORM),
            _clip01(task.remaining(now) / DEAD_NORM),
            _clip01(local_wait / WAIT_NORM),
            rsu_in,
            _clip01(rsu_dist / RSU_RANGE_M),
            _clip01(rsu_wait / WAIT_NORM),
            _clip01(min(ctx["n_nbrs"], 5) / 5.0),
            _clip01(nb_dist / V2V_RANGE_M),
            _clip01(nb_contact / CONTACT_NORM),
            _clip01(self.nodes.wait_time("backhaul", now) / WAIT_NORM),  # 回程壅塞
        ], dtype=np.float32)
        return obs

    # ---------- 內部：推進世界一步，把新任務排進 pending ----------
    def _world_step(self):
        now, veh_states = self.world.step()
        self.now = now
        self.veh_states = veh_states
        for vid in veh_states:
            if not self.nodes.has(vid):
                self.nodes.register(vid, VEHICLE_CPU, "vehicle")
            if vid not in self.roles:
                self.roles[vid] = "server" if is_server(vid, self.server_ratio) else "client"
        self.servers = {vid: s["pos"] for vid, s in veh_states.items()
                        if self.roles.get(vid) == "server"}
        clients = [vid for vid in veh_states if self.roles.get(vid) == "client"]

        new_tasks = self.gen.step(clients, now, self.world.dt)
        self.stats["generated"] += len(new_tasks)
        for task in new_tasks:
            cpos = veh_states[task.source]["pos"]
            hop = hop_energy = 0.0
            holder_id, holder_pos = task.source, cpos
            others = {sid: p for sid, p in self.servers.items() if sid != task.source}
            nbrs = neighbors_in_range(cpos, others, V2V_RANGE_M)
            if nbrs:
                sid = nbrs[0]
                holder_id, holder_pos = sid, self.servers[sid]
                hop_tx = transmission_delay(task.data_bits, distance(cpos, holder_pos),
                                            V2V_LINK)
                if hop_tx == INF:
                    continue
                # 指派跳(client→server)：補上 PC5 存取開銷與傳輸能耗(先前漏算)
                hop = V2V_EXTRA_LATENCY + hop_tx
                hop_energy = TX_POWER_W * hop_tx
            self.pending.append(
                self._build_context(task, holder_id, holder_pos, now + hop, hop,
                                    hop_energy))
        self.sim_steps += 1

    def _advance_to_next_decision(self):
        guard = 0
        while not self.pending:
            if self.world.done or guard > 5000:
                return False
            self._world_step()
            guard += 1
        self.current = self.pending.pop(0)
        return True

    # ---------- 套用動作 → 算獎勵 ----------
    def _apply(self, ctx, action):
        task = ctx["task"]
        now = ctx["now"]
        pos = ctx["holder_pos"]
        kind = ACTIONS[action]

        # 解析動作目標
        if kind == "local":
            tid, tpos = ctx["holder_id"], None
            reachable = True
        elif kind == "v2v":
            tid, tpos = ctx["nb_id"], ctx["nb_pos"]
            reachable = ctx["nb_id"] is not None
        elif kind == "rsu":
            tid, tpos = ctx["rsu_id"], ctx["rsu_pos"]
            reachable = ctx["rsu_id"] is not None
        else:  # cloud（經最近基站）
            tid, tpos = "cloud", ctx["rsu_pos"]
            reachable = ctx["rsu_id"] is not None

        info = {"kind": kind}
        if not reachable:
            self.stats["fail"] += 1
            self.stats["infeasible"] += 1
            info["result"] = "infeasible"
            return -PENALTY_FAIL, info

        # ---- 移動性約束：算「持有者與目標」的連線可維持時間，交給 estimate 檢查 ----
        #   v2v：兩台車的相對運動；rsu/cloud：車對(靜止)服務基站；local：不需檢查。
        contact_s = None
        if kind == "v2v":
            contact_s = contact_time(pos, self._vel_of(ctx["holder_id"]),
                                     tpos, self._vel_of(tid), V2V_RANGE_M)
        elif kind in ("rsu", "cloud"):
            contact_s = contact_time(pos, self._vel_of(ctx["holder_id"]),
                                     tpos, (0.0, 0.0), RSU_RANGE_M)

        r = estimate(task, kind, now, pos, self.nodes,
                     target_id=tid, target_pos=tpos, commit=True,
                     contact_s=contact_s)
        if not r["feasible"]:
            self.stats["fail"] += 1
            if r.get("link_break"):
                self.stats["link_break"] += 1     # 完成前駛離範圍 → 連線中斷
                info["result"] = "link_break"
            else:
                self.stats["infeasible"] += 1
                info["result"] = "infeasible"
            return -PENALTY_FAIL, info

        total = ctx["hop"] + r["latency"]
        energy = r["energy"] + ctx["hop_energy"]   # 含指派跳(client→server)的傳輸能耗
        self.stats["latency_sum"] += total
        self.stats["latency_n"] += 1
        self.stats["energy_sum"] += energy
        self.stats["energy_n"] += 1
        self.stats["cost_sum"] += r["cost"]
        self.stats["by_target"][kind] = self.stats["by_target"].get(kind, 0) + 1

        # 獎勵 = -(延遲 + 能耗權重×正規化能耗 + 成本權重×使用成本)。
        #   能耗正規化讓不同層的能耗差異被公平比較、尺度與延遲相近 → 訓練更穩；
        #   成本項(方法②)讓「過度用雲」多付一份代價。
        reward = -(total + ENERGY_W * energy / ENERGY_NORM + COST_W * r["cost"])
        if total <= task.deadline_s:
            self.stats["success"] += 1
            info["result"] = "success"
        else:
            self.stats["fail"] += 1
            self.stats["deadline_miss"] += 1
            reward -= PENALTY_MISS
            info["result"] = "deadline_miss"
        return reward, info

    # ---------- Gym 介面 ----------
    def _reset_stats(self):
        self.stats = {"generated": 0, "success": 0, "fail": 0,
                      "deadline_miss": 0, "infeasible": 0, "link_break": 0,
                      "latency_sum": 0.0, "latency_n": 0,
                      "energy_sum": 0.0, "energy_n": 0,
                      "cost_sum": 0.0, "by_target": {}}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.world is not None:
            self.world.close()
        if self.mock:
            self.world = MockWorld(n=self.mock_vehicles, seed=self.base_seed + self._ep)
        else:
            self.world = TraciWorld(self.cfg, gui=self.gui)
        self.world.reset()

        self.nodes = build_nodes(self.rsus)
        self.gen = TaskGenerator(arrival_rate=self.arrival_rate,
                                 seed=self.base_seed + self._ep)
        self.roles = {}
        self.pending = []
        self.veh_states = {}
        self.servers = {}
        self.decisions = 0
        self.sim_steps = 0
        self._ep += 1
        self._reset_stats()

        ok = self._advance_to_next_decision()
        obs = self._make_obs(self.current) if ok else np.zeros(N_FEATURES, np.float32)
        return obs, {}

    def step(self, action):
        reward, info = self._apply(self.current, int(action))
        self.decisions += 1

        terminated = False
        truncated = False
        if self.decisions >= self.episode_decisions:
            truncated = True
        else:
            if not self._advance_to_next_decision():
                terminated = True

        if terminated or truncated:
            obs = np.zeros(N_FEATURES, np.float32)
            info["episode_stats"] = self.episode_summary()
        else:
            obs = self._make_obs(self.current)
        return obs, reward, terminated, truncated, info

    def episode_summary(self):
        s = self.stats
        done = s["success"] + s["fail"]
        return {
            "generated": s["generated"],
            "success_rate": (s["success"] / done) if done else 0.0,
            "avg_latency_ms": (s["latency_sum"] / s["latency_n"] * 1000)
                              if s["latency_n"] else 0.0,
            "avg_energy_j": (s["energy_sum"] / s["energy_n"])
                            if s["energy_n"] else 0.0,
            "avg_cost": (s["cost_sum"] / s["energy_n"])
                        if s["energy_n"] else 0.0,
            "deadline_miss": s["deadline_miss"],
            "infeasible": s["infeasible"],
            "link_break": s["link_break"],
            "by_target": dict(s["by_target"]),
        }

    def close(self):
        if self.world is not None:
            self.world.close()


# ==================================================================
# 自我測試：mock 模式 + 隨機動作，不需 SUMO、不需 torch
# ==================================================================
if __name__ == "__main__":
    print("=== VECEnv 自我測試（mock 模式 + 隨機動作）===\n")
    env = VECEnv(mock=True, arrival_rate=0.5, episode_decisions=300, seed=0)

    # 檢查觀測/動作空間
    print(f"觀測空間：{env.observation_space}")
    print(f"動作空間：{env.action_space}（{ACTIONS}）\n")

    obs, _ = env.reset()
    print(f"首個觀測(長度{len(obs)})：{np.round(obs, 2)}")
    assert env.observation_space.contains(obs), "觀測超出空間範圍！"
    print("觀測在空間範圍內 ✓\n")

    # 跑 3 回合隨機動作
    for ep in range(3):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        last_info = {}
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            last_info = info
            done = term or trunc
        s = last_info["episode_stats"]
        print(f"回合 {ep+1}: 總獎勵 {ep_reward:8.1f} ｜ 任務{s['generated']:>4} "
              f"成功率 {s['success_rate']*100:5.1f}% 平均延遲 {s['avg_latency_ms']:6.1f}ms "
              f"能耗 {s['avg_energy_j']:.3f}J ｜ 分布 {s['by_target']}")

    env.close()
    print("\n=== 測試結束：環境可正常 reset/step、回合能結束、統計正常 ===")
    print("（隨機動作的成功率會偏低，這正常——之後 DQN 要學到比隨機高很多）")
