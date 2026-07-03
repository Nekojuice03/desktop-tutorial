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
import math
import numpy as np

from kalman_tracker import EKFCTRV, predict_contact, predict_contact_static

from comm_model import (distance, transmission_delay, velocity_from_speed_angle,
                        neighbors_in_range, rsus_in_range, contact_time)
from nodes import build_nodes, estimate, INF
from task_model import TaskGenerator
from infra_config import (VEHICLE_CPU, STRONG_VEHICLE_CPU, STRONG_RATIO,
                          STRONG_VEHICLE_ENERGY_PER_CYCLE,
                          V2V_LINK, V2I_LINK, V2V_RANGE_M, RSU_RANGE_M,
                          V2V_EXTRA_LATENCY, RSU_EXTRA_LATENCY,
                          TX_POWER_W, load_rsus)
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
                 task_cpu_scale=1.0, task_deadline_scale=1.0,
                 predictor="kalman", recovery="v2i"):
        """
        predictor：V2V 連線壽命預判器(消融階梯：naive → 可部署 → oracle)
          "linear" = 等速直線外推(baseline，路口轉彎會高估)
          "kalman" = 預設。EKF-CTRV 卡曼濾波：只用車輛本就廣播的 BSM/CAM
                     可觀測量(位置/速度/航向)估計「轉彎率 ω」→ 預判對方
                     會轉彎或維持同向(現實可部署；文獻: Kalman link
                     residual lifetime, IEEE VTC 2016)
          "route"  = 疊加「路線分歧+離場時間」(需 V2X 意圖分享=已知路線，
                     oracle 上界；SUMO 模式才有 route，mock 退回 linear)
        recovery：V2V 任務在完成時刻「實際已斷線」時的恢復策略(消融用)
          "fail" = 直接失敗(最保守)
          "v2i"  = 結果經 V2I 遷移接續(執行車→RSU→持有車)，付遷移延遲/能耗
        """
        self.mock = mock
        self.predictor = predictor
        self.recovery = recovery
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

    def _pair_contact(self, a_id, a_pos, b_id, b_pos):
        """
        兩車的「預測連線壽命」＝ min(等速外推, 路線分歧時間)。
        predictor="route" 且世界能提供路線(SUMO)時，用意圖分享修正路口轉彎的高估；
        否則退回等速外推(linear baseline / mock)。觀測、決策、greedy 都走這裡，
        所以升級預判器時 agent 的 contact 特徵會同步變準。
        """
        if self.predictor == "kalman":
            # 可部署層：EKF-CTRV 前推(含轉彎率) → 預測連線壽命。
            # 追蹤器未熱身(<3 筆量測)時退回等速外推。
            ta, tb = self._trackers.get(a_id), self._trackers.get(b_id)
            if ta is not None and tb is not None and ta.ready and tb.ready:
                return predict_contact(ta, tb, V2V_RANGE_M)
        c = self._contact(a_id, a_pos, b_id, b_pos)
        if self.predictor == "route":
            t_div = self.world.route_divergence_time(a_id, b_id)
            if t_div is not None:
                c = min(c, t_div)
        return c

    def _contact_for(self, ctx, kind, tid, tpos):
        """
        此動作的「連線可維持時間」(交給 estimate 做移動性檢查)：
          v2v：持有車與目標車的相對運動；rsu/cloud：持有車對(靜止)服務基站；
          local：不需連線 → None(不檢查)。
        """
        if kind in ("v2v_strong", "v2v_near"):
            return self._pair_contact(ctx["holder_id"], ctx["holder_pos"], tid, tpos)
        if kind in ("rsu", "cloud"):
            # V2I 與 V2V 同一套預判階梯(對稱，避免偏袒基礎設施)：
            if self.predictor == "kalman":
                tr = self._trackers.get(ctx["holder_id"])
                if tr is not None and tr.ready:
                    return predict_contact_static(tr, tpos, RSU_RANGE_M)
            hv = velocity_from_speed_angle(
                self.veh_states[ctx["holder_id"]]["speed"],
                self.veh_states[ctx["holder_id"]]["angle"]) \
                if ctx["holder_id"] in self.veh_states else (0.0, 0.0)
            c = contact_time(ctx["holder_pos"], hv, tpos, (0.0, 0.0), RSU_RANGE_M)
            if self.predictor == "route":
                t_exit = self.world.route_exit_time(ctx["holder_id"])
                if t_exit is not None:
                    c = min(c, t_exit)   # 持有車離場前必須收到結果
            return c
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
            sg_contact = self._pair_contact(ctx["holder_id"], pos, ctx["strong_id"], ctx["strong_pos"])
        else:
            sg_in, sg_dist, sg_wait, sg_contact = 0.0, V2V_RANGE_M, 0.0, 0.0

        if ctx["near_id"] is not None:
            nr_in = 1.0
            nr_dist = distance(pos, ctx["near_pos"])
            nr_strong = 1.0 if ctx["near_strong"] else 0.0
            nr_contact = self._pair_contact(ctx["holder_id"], pos, ctx["near_id"], ctx["near_pos"])
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
        # 記住每台車最後出現的位置：車輛抵達終點會從模擬消失(despawn)，
        # 結算時若執行車已離場，用它離場前的位置做「優雅退場交接」(見 _settle_one)
        for vid, s in veh_states.items():
            self._last_seen[vid] = s["pos"]
        # 卡曼追蹤：把每 tick 的 BSM 量測(位置/速度/航向)餵進各車的 EKF-CTRV
        if self.predictor == "kalman":
            dt = getattr(self.world, "dt", 1.0)
            for vid, s in veh_states.items():
                vx, vy = velocity_from_speed_angle(s["speed"], s["angle"])
                theta = math.atan2(vy, vx)
                tr = self._trackers.get(vid)
                if tr is None:
                    tr = self._trackers[vid] = EKFCTRV()
                tr.step(s["pos"][0], s["pos"][1], s["speed"], theta, dt)
            for vid in [v for v in self._trackers if v not in veh_states]:
                del self._trackers[vid]   # 離場車清掉
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
                self._record_kind(task.kind, True)
            else:
                self.stats["fail"] += 1
                self.stats["deadline_miss"] += 1
                self._record_kind(task.kind, False)

    def _advance_to_active(self):
        guard = 0
        while True:
            if self.tick_count >= self.episode_ticks or self.world.done or guard > 5000:
                self._settle_inflight(force=True)   # 回合結束：所有在途 V2V 一次結清
                self._active = []
                return False
            now, assign = self._tick_and_route()
            self.tick_count += 1
            guard += 1
            self._settle_inflight()   # 每 tick：結算已到「完成時刻」的在途 V2V
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
            self._record_kind(task.kind, False)
            return -PENALTY_FAIL

        # V2V 兩種動作在成本模型裡都是 "v2v"
        est_kind = "v2v" if kind in ("v2v_strong", "v2v_near") else kind
        # ★預判層(proactive)：預測「完成前就會斷線」→ 拒絕此卸載(admission control)。
        #   predictor="route" 時用路線分歧修正路口高估(文獻: link lifetime prediction)。
        contact_s = self._contact_for(ctx, kind, tid, tpos)
        r = estimate(task, est_kind, now, pos, self.nodes,
                     target_id=tid, target_pos=tpos, commit=True,
                     contact_s=contact_s)
        if not r["feasible"]:
            self.stats["fail"] += 1
            if r.get("link_break"):
                self.stats["pred_reject"] += 1   # 被「預判」擋下(非執行期真實斷線)
            else:
                self.stats["infeasible"] += 1
            self._record_kind(task.kind, False)
            return -PENALTY_FAIL

        total = ctx["hop"] + r["latency"]
        energy = r["energy"] + ctx["hop_energy"]   # 含指派跳(client→server)的傳輸能耗
        # 能耗正規化(同 vec_env) + 使用成本項(方法②)：讓本地/邊緣/雲的能耗與成本
        # 差異被公平比較，且「過度用雲」多付一份代價。
        reward = -(total + ENERGY_W * energy / ENERGY_NORM + COST_W * r["cost"])
        if total > task.deadline_s:
            reward -= PENALTY_MISS

        if est_kind != "local":
            # ★事件驅動結算(V2V 與 RSU/雲一視同仁，避免只有 V2V 承受執行期風險)：
            #   成敗不在決策當下拍板，而是掛到「預計完成時刻」，屆時用世界的
            #   真實位置驗證「結果送得回持有車嗎」。此刻先回「預測獎勵」給 RL；
            #   結算與預測不符的差額(delta)補進之後 tick 的團隊獎勵(GAE 回傳信用)。
            self._inflight.append({
                "mode": "v2v" if est_kind == "v2v" else "infra",
                "task": task, "helper": tid, "holder": ctx["holder_id"],
                "serving_rsu": ctx["rsu_id"],   # infra：執行/轉送用的服務基站
                "t_done": now + r["latency"],   # now 已含指派跳時刻
                "total": total, "energy": energy, "cost": r["cost"],
                "pred_reward": reward,
            })
            return reward

        # 本地：不經無線傳輸 → 決策當下即結算
        self._finalize(total, energy, r["cost"], task, ok=(total <= task.deadline_s))
        return reward

    def _finalize(self, total, energy, cost, task, ok):
        """把一筆任務的最終結果記進統計(成功/超時、延遲/能耗/成本、任務類型)。"""
        self.stats["latency_sum"] += total
        self.stats["latency_n"] += 1
        self.stats["energy_sum"] += energy
        self.stats["energy_n"] += 1
        self.stats["cost_sum"] += cost
        if ok:
            self.stats["success"] += 1
            self._record_kind(task.kind, True)
        else:
            self.stats["fail"] += 1
            self.stats["deadline_miss"] += 1
            self._record_kind(task.kind, False)

    # ---------- 事件驅動結算：V2V 任務到「完成時刻」用真實位置驗證 ----------
    def _settle_inflight(self, force=False):
        """
        結算已到期(t_done ≤ now)的 V2V 任務；force=True(回合結束)全部結算。
        - 兩車仍在 V2V 範圍 → 照預測結果入帳(delta=0)。
        - 已超出範圍(真實斷線，通常因轉彎/岔路使預判失準)→ 恢復層(reactive)：
            recovery="v2i"：結果經基礎設施遷移 執行車→RSU→持有車(文獻: service
              migration)。遷移延遲 = V2I 存取 + 兩段 V2I 傳輸；可能因此超時。
            recovery="fail" 或兩端無 RSU 覆蓋：任務失敗(break_failed)。
          與「預測獎勵」的差額累入 _settle_delta，於下一步併入團隊獎勵。
        """
        remain = []
        for f in self._inflight:
            if not force and f["t_done"] > self.now:
                remain.append(f)
                continue
            self._settle_one(f)
        self._inflight = remain

    def _fail_settle(self, f, consumer_gone=False):
        """結算為不可恢復失敗：計數 + 統計 + 獎勵差額。"""
        task = f["task"]
        self.stats["link_break"] += 1
        if consumer_gone:
            self.stats["consumer_left"] += 1
        self.stats["break_failed"] += 1
        self.stats["fail"] += 1
        self._record_kind(task.kind, False)
        self._settle_delta += (-PENALTY_FAIL) - f["pred_reward"]

    def _settle_one(self, f):
        task = f["task"]
        hp = self.veh_states.get(f["holder"])

        if f["mode"] == "infra":
            # ---- RSU/雲：結果由基礎設施下行送回持有車 ----
            if hp is None:
                self._fail_settle(f, consumer_gone=True)   # 持有車離場，結果無人接收
                return
            srv = f["serving_rsu"]
            srv_pos = (self.rsus[srv]["x"], self.rsus[srv]["y"])
            if distance(hp["pos"], srv_pos) <= RSU_RANGE_M:
                self._finalize(f["total"], f["energy"], f["cost"], task,
                               ok=(f["total"] <= task.deadline_s))
                return
            # 駛出原服務基站覆蓋 → 基礎設施互聯，換手到目前最近的 RSU 重送結果
            # (蜂巢式常態操作，不算斷線；但要付換手存取 + 重送下行的代價)
            near = rsus_in_range(hp["pos"], self.rsus, RSU_RANGE_M)
            if near:
                p2 = (self.rsus[near[0]]["x"], self.rsus[near[0]]["y"])
                t_dn = transmission_delay(task.result_bits,
                                          distance(hp["pos"], p2), V2I_LINK)
                if t_dn != INF:
                    extra = RSU_EXTRA_LATENCY + t_dn
                    new_total = f["total"] + extra
                    new_energy = f["energy"] + TX_POWER_W * t_dn
                    ok = new_total <= task.deadline_s
                    self.stats["rsu_handover"] += 1
                    self._finalize(new_total, new_energy, f["cost"], task, ok=ok)
                    new_reward = -(new_total + ENERGY_W * new_energy / ENERGY_NORM
                                   + COST_W * f["cost"]) - (0.0 if ok else PENALTY_MISS)
                    self._settle_delta += new_reward - f["pred_reward"]
                    return
            self._fail_settle(f)   # 覆蓋空洞：無任何 RSU 可送達
            return

        # ---- V2V：驗證持有車與執行車是否仍在範圍 ----
        ep = self.veh_states.get(f["helper"])
        in_range = (hp is not None and ep is not None and
                    distance(hp["pos"], ep["pos"]) <= V2V_RANGE_M)
        if in_range:
            # 預判正確：照預測入帳
            self._finalize(f["total"], f["energy"], f["cost"], task,
                           ok=(f["total"] <= task.deadline_s))
            return

        # ---- 真實斷線(runtime link break) ----
        # SUMO 實測主因不是「開出範圍」而是「車輛抵達終點離開模擬(despawn)」：
        #   持有車離場 → 結果無人接收，性質上是任務作廢(consumer_left)，不可恢復。
        #   執行車離場 → 「優雅退場交接」：車輛抵達不是瞬間蒸發，離場前可把結果
        #     交給其最後位置附近的 RSU(用 _last_seen 快取) → 仍可走 V2I 遷移。
        self.stats["link_break"] += 1
        if hp is None:
            self.stats["consumer_left"] += 1   # break_failed 的子類(帳務仍計 break_failed)
        elif self.recovery == "v2i":
            helper_pos = ep["pos"] if ep is not None else self._last_seen.get(f["helper"])
            if helper_pos is not None:
                r1 = rsus_in_range(helper_pos, self.rsus, RSU_RANGE_M)  # 執行車側 RSU
                r2 = rsus_in_range(hp["pos"], self.rsus, RSU_RANGE_M)   # 持有車側 RSU
                if r1 and r2:
                    p1 = (self.rsus[r1[0]]["x"], self.rsus[r1[0]]["y"])
                    p2 = (self.rsus[r2[0]]["x"], self.rsus[r2[0]]["y"])
                    t_up = transmission_delay(task.result_bits,
                                              distance(helper_pos, p1), V2I_LINK)
                    t_dn = transmission_delay(task.result_bits,
                                              distance(hp["pos"], p2), V2I_LINK)
                    if t_up != INF and t_dn != INF:
                        extra = RSU_EXTRA_LATENCY + t_up + t_dn  # RSU 間走有線骨幹，忽略
                        new_total = f["total"] + extra
                        new_energy = f["energy"] + TX_POWER_W * (t_up + t_dn)
                        ok = new_total <= task.deadline_s
                        self.stats["break_recovered"] += 1
                        self._finalize(new_total, new_energy, f["cost"], task, ok=ok)
                        new_reward = -(new_total + ENERGY_W * new_energy / ENERGY_NORM
                                       + COST_W * f["cost"]) - (0.0 if ok else PENALTY_MISS)
                        self._settle_delta += new_reward - f["pred_reward"]
                        return
        # 無法恢復(持有車離場、策略=fail、或任一側無 RSU 覆蓋)
        self.stats["break_failed"] += 1
        self.stats["fail"] += 1
        self._record_kind(task.kind, False)
        self._settle_delta += (-PENALTY_FAIL) - f["pred_reward"]

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
                      "infeasible": 0, "fallback": 0,
                      "pred_reject": 0,      # 預判層擋下的 V2V(admission control)
                      "link_break": 0,       # 執行期真實斷線事件(= recovered+failed)
                      "break_recovered": 0,  # 斷線但經 V2I 遷移救回(可能仍超時)
                      "break_failed": 0,     # 斷線且無法恢復 → 任務失敗
                      "consumer_left": 0,    # 其中：持有車已離場(結果無人接收，break_failed 子類)
                      "rsu_handover": 0,     # RSU/雲結果經「換手」由新服務基站送達(非斷線)
                      "latency_sum": 0.0, "latency_n": 0,
                      "energy_sum": 0.0, "energy_n": 0,
                      "cost_sum": 0.0, "by_target": {},
                      "kind_total": {}, "kind_success": {}}   # 各任務類型成功率

    def _record_kind(self, kind, ok):
        """記錄某任務類型的成敗（給 vision-only 等分類型成功率用）。"""
        self.stats["kind_total"][kind] = self.stats["kind_total"].get(kind, 0) + 1
        if ok:
            self.stats["kind_success"][kind] = self.stats["kind_success"].get(kind, 0) + 1

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
        self._inflight = []       # 在途 V2V 任務(等完成時刻結算)
        self._settle_delta = 0.0  # 結算 vs 預測獎勵的差額(補進下一步團隊獎勵)
        self._last_seen = {}      # vid -> 最後出現位置(離場車的優雅退場交接用)
        self._trackers = {}       # vid -> EKF-CTRV 追蹤器(predictor="kalman")
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

        # 把「上一輪推進期間」結算出的 V2V 差額(斷線懲罰/遷移代價)攤進本步團隊獎勵。
        # 延後一步是刻意的：GAE 會把信用沿 tick 序列往回傳給當初的決策。
        if self._settle_delta and len(rewards):
            rewards = rewards + np.float32(self._settle_delta / len(rewards))
            self._settle_delta = 0.0

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
        kt, ks = s["kind_total"], s["kind_success"]
        kind_sr = {k: (ks.get(k, 0) / kt[k]) for k in kt if kt[k]}
        return {"generated": s["generated"],
                "success_rate": (s["success"] / done) if done else 0.0,
                "vision_success_rate": kind_sr.get("vision", 0.0),  # 重任務成功率(動態範圍大)
                "kind_success_rate": kind_sr,                       # 各類型成功率
                "avg_latency_ms": (s["latency_sum"] / s["latency_n"] * 1000)
                                  if s["latency_n"] else 0.0,
                "avg_energy_j": (s["energy_sum"] / s["energy_n"])
                                if s["energy_n"] else 0.0,
                "avg_cost": (s["cost_sum"] / s["energy_n"])
                            if s["energy_n"] else 0.0,
                "deadline_miss": s["deadline_miss"], "infeasible": s["infeasible"],
                "pred_reject": s["pred_reject"], "link_break": s["link_break"],
                "break_recovered": s["break_recovered"],
                "break_failed": s["break_failed"],
                "consumer_left": s["consumer_left"],
                "rsu_handover": s["rsu_handover"],
                "fallback": s["fallback"],
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
