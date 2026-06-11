"""
多智能體車載卸載環境（vec_env_ma.py）—— Stage B，V2V 為主版
================================================================
車輛算力異質化：少數「強車」(高算力) + 多數「弱車」。
弱 server 收到重任務自己算不動 → 找附近的強車幫忙(V2V) 變得有價值。
強車稀缺 → 多台弱車搶同一台強車 → 需要協調(中央 critic)。

V2V 動作拆兩種，讓 agent 自己學「強但遠」vs「近但弱」的取捨：
  動作：0=本地 1=V2V丟強車 2=V2V丟近車 3=基站 4=雲端

精簡批次介面(給自寫 MAPPO)：
  reset() -> (obs_batch [k,F], state [S])
  step(actions [k]) -> (rewards [k], next_obs [k2,F], next_state [S], done, info)

直接執行 `python vec_env_ma.py` 跑 mock 自我測試。
"""
import os
import numpy as np

from comm_model import (distance, transmission_delay, velocity_from_speed_angle,
                        neighbors_in_range, rsus_in_range, contact_time)
from nodes import build_nodes, estimate, INF
from task_model import TaskGenerator
from infra_config import (VEHICLE_CPU, STRONG_VEHICLE_CPU, STRONG_RATIO,
                          STRONG_VEHICLE_ENERGY_PER_CYCLE,
                          V2V_LINK, V2V_RANGE_M, RSU_RANGE_M,
                          V2V_EXTRA_LATENCY, TX_POWER_W, load_rsus)
from run_baseline import is_server, is_strong
# 重用 Stage A 的世界來源、正規化常數、獎勵參數
from vec_env import (MockWorld, TraciWorld, _clip01,
                     DATA_NORM, CPU_NORM, DEAD_NORM, WAIT_NORM, CONTACT_NORM,
                     ENERGY_W, ENERGY_NORM, COST_W, PENALTY_MISS, PENALTY_FAIL,
                     SCRIPT_DIR)

# 多智能體專用的動作集(5 個，V2V 拆強/近)
MA_ACTIONS = ["local", "v2v_strong", "v2v_near", "rsu", "cloud"]
MA_N_FEATURES = 18   # +1：回程(backhaul)佇列等待 —— agent 須看得到雲端壅塞
MAX_AGENTS_NORM = 20.0


class VECMultiEnv:
    def __init__(self, mock=True, gui=False, cfg="osm.sumocfg",
                 arrival_rate=0.3, server_ratio=0.4, seed=0,
                 episode_ticks=200, rsus=None, mock_vehicles=24,
                 task_cpu_scale=1.0, task_deadline_scale=1.0):
        self.mock = mock
        self.gui = gui
        self.cfg = cfg
        self.arrival_rate = arrival_rate
        self.server_ratio = server_ratio
        self.episode_ticks = episode_ticks
        self.base_seed = seed
        self.mock_vehicles = mock_vehicles
        self.task_cpu_scale = task_cpu_scale
        self.task_deadline_scale = task_deadline_scale

        if rsus is not None:
            self.rsus = rsus
        elif mock:
            self.rsus = {"rsu_0": {"x": 0.0, "y": 0.0},
                         "rsu_1": {"x": 200.0, "y": 0.0}}
        else:
            os.chdir(SCRIPT_DIR)
            self.rsus = {rid: {"x": v["x"], "y": v["y"]}
                         for rid, v in load_rsus().items()}

        self.n_features = MA_N_FEATURES
        self.n_actions = len(MA_ACTIONS)
        self.rsu_ids = sorted(self.rsus.keys())
        # 全域狀態：各基站負載 + 回程壅塞 + 活躍數 + 平均本地負載 + 車數 + 強車數
        self.state_dim = len(self.rsu_ids) + 5

        self.world = None
        self._ep = 0

    # ---------- 連線維持時間小工具 ----------
    def _contact(self, holder_id, holder_pos, nb_id, nb_pos):
        hv = velocity_from_speed_angle(self.veh_states[holder_id]["speed"],
                                       self.veh_states[holder_id]["angle"]) \
            if holder_id in self.veh_states else (0.0, 0.0)
        nv = velocity_from_speed_angle(self.veh_states[nb_id]["speed"],
                                       self.veh_states[nb_id]["angle"])
        return contact_time(holder_pos, hv, nb_pos, nv, V2V_RANGE_M)

    def _contact_for(self, ctx, kind, tid, tpos):
        """
        此動作的「連線可維持時間」(交給 estimate 做移動性檢查)：
          v2v：持有車與目標車的相對運動；rsu/cloud：持有車對(靜止)服務基站；
          local：不需連線 → None(不檢查)。
        """
        if kind in ("v2v_strong", "v2v_near"):
            return self._contact(ctx["holder_id"], ctx["holder_pos"], tid, tpos)
        if kind in ("rsu", "cloud"):
            hv = velocity_from_speed_angle(
                self.veh_states[ctx["holder_id"]]["speed"],
                self.veh_states[ctx["holder_id"]]["angle"]) \
                if ctx["holder_id"] in self.veh_states else (0.0, 0.0)
            return contact_time(ctx["holder_pos"], hv, tpos, (0.0, 0.0), RSU_RANGE_M)
        return None  # local

    # ---------- 單一任務情境 ----------
    def _build_context(self, task, holder_id, holder_pos, now, hop, hop_energy=0.0):
        # 最近基站
        near_rsu = rsus_in_range(holder_pos, self.rsus, RSU_RANGE_M)
        rsu_id = near_rsu[0] if near_rsu else None
        rsu_pos = (self.rsus[rsu_id]["x"], self.rsus[rsu_id]["y"]) if rsu_id else None
        # 範圍內的其他 server(依距離排序)
        others = {sid: self.veh_states[sid]["pos"] for sid in self.servers
                  if sid != holder_id and sid in self.veh_states}
        nbrs = neighbors_in_range(holder_pos, others, V2V_RANGE_M)
        near_id = nbrs[0] if nbrs else None
        near_pos = self.veh_states[near_id]["pos"] if near_id else None
        # 範圍內最近的『強車』
        strong_nbrs = [s for s in nbrs if s in self.strong]
        strong_id = strong_nbrs[0] if strong_nbrs else None
        strong_pos = self.veh_states[strong_id]["pos"] if strong_id else None
        return {"task": task, "holder_id": holder_id, "holder_pos": holder_pos,
                "now": now, "hop": hop, "hop_energy": hop_energy,
                "holder_strong": holder_id in self.strong,
                "rsu_id": rsu_id, "rsu_pos": rsu_pos,
                "near_id": near_id, "near_pos": near_pos,
                "near_strong": near_id in self.strong if near_id else False,
                "strong_id": strong_id, "strong_pos": strong_pos,
                "n_nbrs": len(nbrs)}

    def _obs_of(self, ctx):
        task, pos, now = ctx["task"], ctx["holder_pos"], ctx["now"]
        local_wait = self.nodes.wait_time(ctx["holder_id"], now)

        if ctx["rsu_id"] is not None:
            rsu_in, rsu_dist = 1.0, distance(pos, ctx["rsu_pos"])
            rsu_wait = self.nodes.wait_time(ctx["rsu_id"], now)
        else:
            rsu_in, rsu_dist, rsu_wait = 0.0, RSU_RANGE_M, 0.0

        if ctx["strong_id"] is not None:
            sg_in = 1.0
            sg_dist = distance(pos, ctx["strong_pos"])
            sg_wait = self.nodes.wait_time(ctx["strong_id"], now)
            sg_contact = self._contact(ctx["holder_id"], pos, ctx["strong_id"], ctx["strong_pos"])
        else:
            sg_in, sg_dist, sg_wait, sg_contact = 0.0, V2V_RANGE_M, 0.0, 0.0

        if ctx["near_id"] is not None:
            nr_in = 1.0
            nr_dist = distance(pos, ctx["near_pos"])
            nr_strong = 1.0 if ctx["near_strong"] else 0.0
            nr_contact = self._contact(ctx["holder_id"], pos, ctx["near_id"], ctx["near_pos"])
        else:
            nr_in, nr_dist, nr_strong, nr_contact = 0.0, V2V_RANGE_M, 0.0, 0.0

        return np.array([
            _clip01(task.data_bits / DATA_NORM),
            _clip01(task.cpu_cycles / CPU_NORM),
            _clip01(task.deadline_s / DEAD_NORM),
            _clip01(task.remaining(now) / DEAD_NORM),
            1.0 if ctx["holder_strong"] else 0.0,   # 自己是不是強車
            _clip01(local_wait / WAIT_NORM),
            rsu_in,
            _clip01(rsu_dist / RSU_RANGE_M),
            _clip01(rsu_wait / WAIT_NORM),
            sg_in,                                    # 附近有強車嗎
            _clip01(sg_dist / V2V_RANGE_M),           # 強車多遠
            _clip01(sg_wait / WAIT_NORM),             # 強車多忙
            _clip01(sg_contact / CONTACT_NORM),       # 強車還能連多久
            nr_in,                                    # 附近有鄰車嗎
            _clip01(nr_dist / V2V_RANGE_M),           # 最近鄰車多遠
            nr_strong,                                # 最近鄰車本身是不是強車
            _clip01(nr_contact / CONTACT_NORM),
            _clip01(self.nodes.wait_time("backhaul", now) / WAIT_NORM),  # 回程壅塞
        ], dtype=np.float32)

    def _global_state(self, now, n_active):
        parts = [_clip01(self.nodes.wait_time(rid, now) / WAIT_NORM)
                 for rid in self.rsu_ids]
        parts.append(_clip01(self.nodes.wait_time("backhaul", now) / WAIT_NORM))  # 回程壅塞
        parts.append(_clip01(n_active / MAX_AGENTS_NORM))
        local_waits = [self.nodes.wait_time(c["holder_id"], now)
                       for _, c in self._active] if self._active else [0.0]
        parts.append(_clip01(np.mean(local_waits) / WAIT_NORM))
        parts.append(_clip01(len(self.veh_states) / 50.0))
        n_strong = sum(1 for v in self.veh_states if v in self.strong)
        parts.append(_clip01(n_strong / 10.0))   # 場上強車數(稀缺資源)
        return np.array(parts, dtype=np.float32)

    # ---------- 推進世界一個 tick、指派任務 ----------
    def _tick_and_route(self):
        now, veh_states = self.world.step()
        self.now = now
        self.veh_states = veh_states
        for vid in veh_states:
            if vid not in self.roles:
                if is_server(vid, self.server_ratio):
                    self.roles[vid] = "server"
                    if is_strong(vid, STRONG_RATIO):
                        self.strong.add(vid)
                else:
                    self.roles[vid] = "client"
            if not self.nodes.has(vid):
                if vid in self.strong:   # 強車：算力高但依 κf² 每 cycle 較耗能
                    self.nodes.register(vid, STRONG_VEHICLE_CPU, "vehicle",
                                        energy_per_cycle=STRONG_VEHICLE_ENERGY_PER_CYCLE)
                else:
                    self.nodes.register(vid, VEHICLE_CPU, "vehicle")
        self.servers = {vid: s["pos"] for vid, s in veh_states.items()
                        if self.roles.get(vid) == "server"}
        clients = [vid for vid in veh_states if self.roles.get(vid) == "client"]

        new_tasks = self.gen.step(clients, now, self.world.dt)
        self.stats["generated"] += len(new_tasks)

        assign = {}
        for task in new_tasks:
            cpos = veh_states[task.source]["pos"]
            others = {sid: p for sid, p in self.servers.items() if sid != task.source}
            nbrs = neighbors_in_range(cpos, others, V2V_RANGE_M)
            if nbrs and nbrs[0] not in assign:
                sid = nbrs[0]
                hop_tx = transmission_delay(task.data_bits,
                                            distance(cpos, self.servers[sid]), V2V_LINK)
                if hop_tx == INF:
                    self._fallback_local(task, cpos, now)
                    continue
                # 指派跳(client→server)：補上 PC5 存取開銷與傳輸能耗(先前漏算)
                hop = V2V_EXTRA_LATENCY + hop_tx
                assign[sid] = self._build_context(task, sid, self.servers[sid],
                                                  now + hop, hop,
                                                  hop_energy=TX_POWER_W * hop_tx)
            else:
                self._fallback_local(task, cpos, now)
        return now, assign

    def _fallback_local(self, task, pos, now):
        if not self.nodes.has(task.source):
            self.nodes.register(task.source, VEHICLE_CPU, "vehicle")
        r = estimate(task, "local", now, pos, self.nodes,
                     target_id=task.source, commit=True)
        self.stats["fallback"] += 1
        if r["feasible"]:
            total = r["latency"]
            self.stats["latency_sum"] += total
            self.stats["latency_n"] += 1
            self.stats["energy_sum"] += r["energy"]
            self.stats["energy_n"] += 1
            self.stats["cost_sum"] += r["cost"]   # 本地 fallback：成本為 0
            if total <= task.deadline_s:
                self.stats["success"] += 1
            else:
                self.stats["fail"] += 1
                self.stats["deadline_miss"] += 1

    def _advance_to_active(self):
        guard = 0
        while True:
            if self.tick_count >= self.episode_ticks or self.world.done or guard > 5000:
                self._active = []
                return False
            now, assign = self._tick_and_route()
            self.tick_count += 1
            guard += 1
            active = [(sid, ctx) for sid, ctx in assign.items()
                      if sid in self.veh_states]
            if active:
                self._active = active
                return True

    # ---------- 結算單一決策 → 獎勵 ----------
    def _resolve_one(self, ctx, action):
        task, now, pos = ctx["task"], ctx["now"], ctx["holder_pos"]
        kind = MA_ACTIONS[action]

        if kind == "local":
            tid, tpos, reach = ctx["holder_id"], None, True
        elif kind == "v2v_strong":
            tid, tpos, reach = ctx["strong_id"], ctx["strong_pos"], ctx["strong_id"] is not None
        elif kind == "v2v_near":
            tid, tpos, reach = ctx["near_id"], ctx["near_pos"], ctx["near_id"] is not None
        elif kind == "rsu":
            tid, tpos, reach = ctx["rsu_id"], ctx["rsu_pos"], ctx["rsu_id"] is not None
        else:  # cloud
            tid, tpos, reach = "cloud", ctx["rsu_pos"], ctx["rsu_id"] is not None

        self.stats["by_target"][kind] = self.stats["by_target"].get(kind, 0) + 1
        if not reach:
            self.stats["fail"] += 1
            self.stats["infeasible"] += 1
            return -PENALTY_FAIL

        # V2V 兩種動作在成本模型裡都是 "v2v"
        est_kind = "v2v" if kind in ("v2v_strong", "v2v_near") else kind
        # 移動性約束：完成前駛離範圍 → link_break 失敗(讓 contact 觀測真正影響結果)
        contact_s = self._contact_for(ctx, kind, tid, tpos)
        r = estimate(task, est_kind, now, pos, self.nodes,
                     target_id=tid, target_pos=tpos, commit=True,
                     contact_s=contact_s)
        if not r["feasible"]:
            self.stats["fail"] += 1
            if r.get("link_break"):
                self.stats["link_break"] += 1
            else:
                self.stats["infeasible"] += 1
            return -PENALTY_FAIL

        total = ctx["hop"] + r["latency"]
        energy = r["energy"] + ctx["hop_energy"]   # 含指派跳(client→server)的傳輸能耗
        self.stats["latency_sum"] += total
        self.stats["latency_n"] += 1
        self.stats["energy_sum"] += energy
        self.stats["energy_n"] += 1
        self.stats["cost_sum"] += r["cost"]
        # 能耗正規化(同 vec_env) + 使用成本項(方法②)：讓本地/邊緣/雲的能耗與成本
        # 差異被公平比較，且「過度用雲」多付一份代價。
        reward = -(total + ENERGY_W * energy / ENERGY_NORM + COST_W * r["cost"])
        if total <= task.deadline_s:
            self.stats["success"] += 1
        else:
            self.stats["fail"] += 1
            self.stats["deadline_miss"] += 1
            reward -= PENALTY_MISS
        return reward

    # ---------- 介面 ----------
    def greedy_actions(self):
        """
        就近最快啟發式基準：每個 agent 選『預估總延遲最低』的可行動作。
        這是公平對照組(在同一環境、同參數下與 MAPPO 比)。回傳 actions 陣列。
        """
        acts = []
        for sid, ctx in self._active:
            best_a, best_lat = 0, float("inf")
            for a, kind in enumerate(MA_ACTIONS):
                if kind == "local":
                    tid, tpos, reach = ctx["holder_id"], None, True
                elif kind == "v2v_strong":
                    tid, tpos, reach = ctx["strong_id"], ctx["strong_pos"], ctx["strong_id"] is not None
                elif kind == "v2v_near":
                    tid, tpos, reach = ctx["near_id"], ctx["near_pos"], ctx["near_id"] is not None
                elif kind == "rsu":
                    tid, tpos, reach = ctx["rsu_id"], ctx["rsu_pos"], ctx["rsu_id"] is not None
                else:
                    tid, tpos, reach = "cloud", ctx["rsu_pos"], ctx["rsu_id"] is not None
                if not reach:
                    continue
                est_kind = "v2v" if kind in ("v2v_strong", "v2v_near") else kind
                # 與 RL 同樣受移動性約束(公平)：會斷線的選項視為不可行
                r = estimate(ctx["task"], est_kind, ctx["now"], ctx["holder_pos"],
                             self.nodes, target_id=tid, target_pos=tpos, commit=False,
                             contact_s=self._contact_for(ctx, kind, tid, tpos))
                if r["feasible"]:
                    lat = ctx["hop"] + r["latency"]
                    if lat < best_lat:
                        best_lat, best_a = lat, a
            acts.append(best_a)
        return np.array(acts, dtype=np.int64)

    def _reset_stats(self):
        self.stats = {"generated": 0, "success": 0, "fail": 0, "deadline_miss": 0,
                      "infeasible": 0, "link_break": 0, "fallback": 0,
                      "latency_sum": 0.0, "latency_n": 0,
                      "energy_sum": 0.0, "energy_n": 0,
                      "cost_sum": 0.0, "by_target": {}}

    def reset(self, seed=None):
        if self.world is not None:
            self.world.close()
        s = self.base_seed + self._ep if seed is None else seed
        self.world = MockWorld(n=self.mock_vehicles, seed=s) if self.mock \
            else TraciWorld(self.cfg, gui=self.gui)
        self.world.reset()
        self.nodes = build_nodes(self.rsus)
        self.gen = TaskGenerator(arrival_rate=self.arrival_rate, seed=s,
                                 cpu_scale=self.task_cpu_scale,
                                 deadline_scale=self.task_deadline_scale)
        self.roles = {}
        self.strong = set()
        self.veh_states = {}
        self.servers = {}
        self.tick_count = 0
        self._active = []
        self._ep += 1
        self._reset_stats()

        ok = self._advance_to_active()
        if not ok:
            return np.zeros((0, self.n_features), np.float32), \
                np.zeros(self.state_dim, np.float32)
        obs = np.stack([self._obs_of(c) for _, c in self._active])
        state = self._global_state(self.now, len(self._active))
        return obs, state

    def step(self, actions):
        items = list(enumerate(self._active))
        order = sorted(range(len(items)),
                       key=lambda k: (items[k][1][1]["now"], k))
        rewards = np.zeros(len(items), dtype=np.float32)
        for k in order:
            i, (sid, ctx) = items[k]
            rewards[i] = self._resolve_one(ctx, int(actions[i]))

        ok = self._advance_to_active()
        done = not ok
        if done:
            next_obs = np.zeros((0, self.n_features), np.float32)
            next_state = np.zeros(self.state_dim, np.float32)
            info = {"episode_stats": self.episode_summary()}
        else:
            next_obs = np.stack([self._obs_of(c) for _, c in self._active])
            next_state = self._global_state(self.now, len(self._active))
            info = {}
        return rewards, next_obs, next_state, done, info

    def episode_summary(self):
        s = self.stats
        done = s["success"] + s["fail"]
        return {"generated": s["generated"],
                "success_rate": (s["success"] / done) if done else 0.0,
                "avg_latency_ms": (s["latency_sum"] / s["latency_n"] * 1000)
                                  if s["latency_n"] else 0.0,
                "avg_energy_j": (s["energy_sum"] / s["energy_n"])
                                if s["energy_n"] else 0.0,
                "avg_cost": (s["cost_sum"] / s["energy_n"])
                            if s["energy_n"] else 0.0,
                "deadline_miss": s["deadline_miss"], "infeasible": s["infeasible"],
                "link_break": s["link_break"], "fallback": s["fallback"],
                "by_target": dict(s["by_target"])}

    def close(self):
        if self.world is not None:
            self.world.close()


# ==================================================================
# 自我測試：mock，異質算力，看 V2V 是否被用到
# ==================================================================
def _run_episode(env, policy, seed):
    obs, state = env.reset(seed=seed)
    info = {}
    while True:
        k = obs.shape[0]
        if policy == "random":
            actions = np.random.randint(0, env.n_actions, size=k)
        else:
            actions = np.full(k, MA_ACTIONS.index(policy))
        rewards, obs, state, done, info = env.step(actions)
        if done:
            break
    return info["episode_stats"]


if __name__ == "__main__":
    print("=== VECMultiEnv 自我測試（mock，異質算力 V2V 版）===\n")
    cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24, server_ratio=0.45,
               episode_ticks=150, task_cpu_scale=2.5)

    env = VECMultiEnv(**cfg)
    print(f"觀測維度 {env.n_features}，動作 {env.n_actions}（{MA_ACTIONS}），"
          f"全域狀態維度 {env.state_dim}")
    obs, state = env.reset(seed=0)
    n_strong = len(env.strong)
    print(f"場上 server 中強車數：{n_strong}（稀缺資源）")
    print(f"首個 tick 同時決策 agent 數：{obs.shape[0]}\n")

    print("[A] 隨機動作（看分布是否含 V2V）：")
    for ep in range(3):
        s = _run_episode(VECMultiEnv(**cfg), "random", seed=ep)
        print(f"  成功率{s['success_rate']*100:5.1f}% 延遲{s['avg_latency_ms']:6.0f}ms "
              f"分布{s['by_target']}")

    print("\n[B] 全部本地（弱車算不動重任務 → 應該很慘）：")
    for ep in range(2):
        s = _run_episode(VECMultiEnv(**cfg), "local", seed=ep)
        print(f"  成功率{s['success_rate']*100:5.1f}% 延遲{s['avg_latency_ms']:6.0f}ms")

    print("\n[C] 全部丟強車（v2v_strong；沒強車在範圍時會失敗）：")
    for ep in range(2):
        s = _run_episode(VECMultiEnv(**cfg), "v2v_strong", seed=ep)
        print(f"  成功率{s['success_rate']*100:5.1f}% 延遲{s['avg_latency_ms']:6.0f}ms "
              f"(infeasible {s['infeasible']})")

    print("\n=== 解讀 ===")
    print("重任務下純本地很慘；V2V 丟強車在有強車時能救援但強車稀缺(會有 infeasible)。")
    print("這代表決策變有意義：何時找強車、何時退而求其次上基站/雲 → 留給 MAPPO 學。")
