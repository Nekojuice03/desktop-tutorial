"""
執行 / 成本模型（nodes.py）
===========================
把任務、通訊、節點算力接起來，算出「一個卸載決策的總代價」：
   總延遲 = 上傳傳輸 + 在節點排隊 + 運算 + 回傳傳輸
   能耗   = 本地→運算耗能；卸載→傳輸耗能

這就是獎勵函數的材料：reward = -(延遲 + 能耗 + 超時懲罰)。

兩個重點設計：
  1. 佇列(FIFO)：每個節點用 busy_until(它何時會空閒)記錄忙碌程度。
     任務抵達時若節點還在忙，就得排隊。→ 讓「基站塞住該不該往雲送」成為真實決策。
  2. 雲端無佇列：雲端算力彈性無限，永遠不排隊。→ 壅塞時雲端反而划算。

純邏輯，可單獨測試（直接執行會跑自我測試）。
"""
from comm_model import distance, transmission_delay
from infra_config import (
    VEHICLE_CPU, RSU_CPU, CLOUD_CPU,
    V2V_LINK, V2I_LINK,
    V2V_EXTRA_LATENCY, RSU_EXTRA_LATENCY, CLOUD_EXTRA_LATENCY,
    BACKHAUL_CAPACITY_BPS,
    VEHICLE_ENERGY_PER_CYCLE, RSU_ENERGY_PER_CYCLE, CLOUD_ENERGY_PER_CYCLE,
    TX_POWER_W, CLOUD_BACKHAUL_POWER_W,
    CLOUD_COST_PER_CYCLE, RSU_COST_PER_CYCLE,
)

INF = float("inf")


class ComputeNodes:
    """
    管理所有可運算節點(車輛/基站/雲)的算力與忙碌狀態。
    用 busy_until(節點何時空閒)來模擬 FIFO 佇列，時間戳制，不需手動衰減。
    """
    def __init__(self):
        self._cpu = {}        # node_id -> 算力(cycle/s)
        self._busy = {}       # node_id -> busy_until(秒)
        self._kind = {}       # node_id -> 'vehicle' / 'rsu' / 'cloud' / 'link'
        self._no_queue = set()  # 無佇列的節點(雲端運算)
        self._rate = {}       # link_id -> 傳輸容量(bits/s)，給共享鏈路(回程)排隊用
        self._eppc = {}       # node_id -> 每 cycle 耗能(J/cycle)，未設定者用該層預設值

    def register(self, node_id, cpu, kind, energy_per_cycle=None):
        self._cpu[node_id] = cpu
        self._busy.setdefault(node_id, 0.0)
        self._kind[node_id] = kind
        if energy_per_cycle is not None:    # 異質能耗(如強車依 κf² 較耗能)
            self._eppc[node_id] = energy_per_cycle
        if kind == "cloud":
            self._no_queue.add(node_id)

    def energy_per_cycle(self, node_id, default):
        """節點的每 cycle 耗能；未個別設定時回傳該層預設值。"""
        return self._eppc.get(node_id, default)

    def register_link(self, link_id, rate_bps):
        """註冊一條有頻寬上限的共享鏈路(如 RSU↔雲回程)，用 FIFO 佇列建模壅塞。"""
        self._rate[link_id] = rate_bps
        self._busy.setdefault(link_id, 0.0)
        self._kind[link_id] = "link"

    def has(self, node_id):
        return node_id in self._cpu

    def service_time(self, node_id, task):
        """純運算時間 = 運算量 / 算力。"""
        return task.cpu_cycles / self._cpu[node_id]

    def link_service_time(self, link_id, data_bits):
        """共享鏈路上傳這些位元的傳輸時間 = 資料量 / 容量。"""
        return data_bits / self._rate[link_id]

    def wait_time(self, node_id, arrival):
        """任務在 arrival 時刻抵達，需要排隊多久才輪到它。"""
        if node_id in self._no_queue:
            return 0.0
        return max(0.0, self._busy.get(node_id, 0.0) - arrival)

    def commit(self, node_id, arrival, task):
        """真的把任務排進去，更新節點 busy_until。回傳排隊時間。"""
        if node_id in self._no_queue:
            return 0.0
        start = max(self._busy.get(node_id, 0.0), arrival)
        self._busy[node_id] = start + self.service_time(node_id, task)
        return start - arrival

    def commit_link(self, link_id, arrival, data_bits):
        """把一筆傳輸排進共享鏈路，更新 busy_until。回傳排隊時間。"""
        start = max(self._busy.get(link_id, 0.0), arrival)
        self._busy[link_id] = start + self.link_service_time(link_id, data_bits)
        return start - arrival


def build_nodes(rsus):
    """
    用基站清單(infra_config.load_rsus 的格式)建立節點池，並加入雲端。
    車輛之後由環境程式在它們進入 SUMO 時動態註冊(VEHICLE_CPU)。
    """
    nodes = ComputeNodes()
    for rid in rsus:
        nodes.register(rid, RSU_CPU, "rsu")
    nodes.register("cloud", CLOUD_CPU, "cloud")
    nodes.register_link("backhaul", BACKHAUL_CAPACITY_BPS)   # RSU↔雲共享回程(會壅塞)
    return nodes


def estimate(task, target_kind, now, src_pos, nodes,
             target_id=None, target_pos=None, commit=False, contact_s=None):
    """
    估算把 task 放到某目標執行的總延遲與能耗。
    target_kind: 'local' | 'v2v' | 'rsu' | 'cloud'
      local : 在來源車自己算(target_id = 來源車 id)，無傳輸
      v2v   : 卸載到鄰居 server 車(target_id/target_pos = 鄰居)
      rsu   : 卸載到基站(target_id/target_pos = 基站)
      cloud : 經由服務基站轉送到雲端(target_pos = 該服務基站位置)
    commit=True 時會真的佔用該節點(更新佇列)；估算多選項時用 False。
    contact_s：來源與目標的「連線可維持時間」(由呼叫端依雙方位置/速度算出)。
      ★移動性約束(sojourn time constraint，文獻標準)：若總延遲 > contact_s，
        代表任務完成前連線就斷(車駛離範圍) → 卸載失敗(link_break)。
        None = 不檢查(如本地執行)。

    回傳 dict：feasible, link_break, latency, energy, cost,
              breakdown{uplink,wait,compute,downlink,backhaul}
    """
    up_tx = down_tx = 0.0        # 車輛無線電實際資料傳輸時間(= 資料量/速率，算能耗用)
    up_extra = down_extra = 0.0  # 固定延遲：協議存取(上行) + 雲端骨幹核網/傳播(來回)

    if target_kind == "local":
        node_id = target_id                       # 本地：無傳輸、無存取延遲

    elif target_kind == "v2v":
        d = distance(src_pos, target_pos)
        up_tx   = transmission_delay(task.data_bits,   d, V2V_LINK)
        down_tx = transmission_delay(task.result_bits, d, V2V_LINK)
        up_extra = V2V_EXTRA_LATENCY              # C-V2X PC5 存取/排程開銷
        node_id = target_id

    elif target_kind == "rsu":
        d = distance(src_pos, target_pos)
        up_tx   = transmission_delay(task.data_bits,   d, V2I_LINK)
        down_tx = transmission_delay(task.result_bits, d, V2I_LINK)
        up_extra = RSU_EXTRA_LATENCY              # V2I/Uu 存取/排程開銷
        node_id = target_id

    elif target_kind == "cloud":
        d = distance(src_pos, target_pos)         # 車先經 V2I 無線傳到服務基站
        up_tx   = transmission_delay(task.data_bits,   d, V2I_LINK)
        down_tx = transmission_delay(task.result_bits, d, V2I_LINK)
        # 上行 = V2I 存取 + 骨幹核網/傳播；下行 = 骨幹核網/傳播。
        # 骨幹的「有限頻寬傳輸 + 排隊」另計於下方(方法①)，這裡只放固定延遲。
        up_extra   = RSU_EXTRA_LATENCY + CLOUD_EXTRA_LATENCY
        down_extra = CLOUD_EXTRA_LATENCY
        node_id = "cloud"

    else:
        raise ValueError(f"未知的 target_kind: {target_kind}")

    # 任一段傳輸超出範圍 → 不可達(視為任務失敗)
    if up_tx == INF or down_tx == INF:
        return {"feasible": False, "link_break": False, "latency": INF,
                "energy": INF, "cost": INF, "breakdown": {}}

    # ★方法①：雲端先過「有限頻寬」的共享回程 → FIFO 排隊。上雲任務越多越塞，
    #   每筆回程傳輸時間 = (上行資料 + 下行結果) / 骨幹容量；佇列等待隨負載自己上升。
    bh_wait = bh_tx = 0.0
    if target_kind == "cloud":
        arrival_bh = now + up_tx + up_extra          # 抵達骨幹入口(已含 V2I 存取 + 上行傳播)
        bh_bits = task.data_bits + task.result_bits
        bh_wait = nodes.wait_time("backhaul", arrival_bh)
        bh_tx = nodes.link_service_time("backhaul", bh_bits)

    arrival = now + up_tx + up_extra + bh_wait + bh_tx   # 任務抵達運算節點的時刻
    wait = nodes.wait_time(node_id, arrival)
    compute = nodes.service_time(node_id, task)
    latency = up_tx + up_extra + bh_wait + bh_tx + wait + compute + down_extra + down_tx

    # ---- ★移動性約束(方法②之後新增)：任務完成前連線就斷 → 卸載失敗 ----
    # 取保守的 sojourn time constraint：總延遲 ≤ 連線可維持時間。
    # 這讓觀測中的 contact 特徵真正影響結果，agent 才有誘因學「別丟給快離開的對象」。
    if contact_s is not None and latency > contact_s:
        return {"feasible": False, "link_break": True, "latency": INF,
                "energy": INF, "cost": INF, "breakdown": {}}

    # ---- 能耗 = 運算能耗 + 車輛無線傳輸能耗 + (僅雲端)回程能耗 ----
    # 依「實際執行任務的節點」選運算能耗係數，讓本地/邊緣/雲端各自不同：
    #   local / v2v → 由車輛(弱或強車)執行 → 車輛係數
    #   rsu         → 邊緣基站執行
    #   cloud       → 雲端執行(係數最高)
    # ★這修正了兩個問題：
    #   (1) 邊緣與雲端能耗一樣：先前只算傳輸，路徑相同故相等；現在運算係數不同 → 不再相等。
    #   (2) 資源全擠雲/邊：先前卸載幾乎零能耗，使本地永遠吃虧；現在卸載要計運算能耗，
    #       輕任務在本地反而更省 → 卸載決策回到合理的分層分工。
    if target_kind in ("local", "v2v"):
        # 依執行車輛查個別係數(強車依 κf² 較耗能)，未設定者用弱車預設值
        e_per_cycle = nodes.energy_per_cycle(node_id, VEHICLE_ENERGY_PER_CYCLE)
    elif target_kind == "rsu":
        e_per_cycle = RSU_ENERGY_PER_CYCLE
    else:  # cloud
        e_per_cycle = CLOUD_ENERGY_PER_CYCLE
    compute_energy = task.cpu_cycles * e_per_cycle               # 運算耗能(執行節點)
    tx_energy = TX_POWER_W * (up_tx + down_tx)                   # 車輛無線傳輸耗能(本地為0)
    backhaul_energy = CLOUD_BACKHAUL_POWER_W * bh_tx             # 僅雲端的有線骨幹傳輸耗能
    energy = compute_energy + tx_energy + backhaul_energy

    # ---- 使用成本(方法②：pay-per-use) = 運算量 × 執行節點單價 ----
    #   雲端最貴、邊緣較便宜、本地/V2V 免費 → 讓「過度用雲」在獎勵裡有額外代價。
    if target_kind == "cloud":
        cost = task.cpu_cycles * CLOUD_COST_PER_CYCLE
    elif target_kind == "rsu":
        cost = task.cpu_cycles * RSU_COST_PER_CYCLE
    else:  # local / v2v：自有車輛資源，不計費
        cost = 0.0

    if commit:
        if target_kind == "cloud":
            nodes.commit_link("backhaul", now + up_tx + up_extra,
                              task.data_bits + task.result_bits)
        nodes.commit(node_id, arrival, task)

    return {
        "feasible": True,
        "link_break": False,
        "latency": latency,
        "energy": energy,
        "cost": cost,
        "breakdown": {
            "uplink":   up_tx + up_extra,
            "wait":     wait + bh_wait,
            "compute":  compute,
            "downlink": down_tx + down_extra,
            "backhaul": bh_tx,
        },
    }


# ==================================================================
# 自我測試：直接執行 `python nodes.py`，不需要開 SUMO
# ==================================================================
if __name__ == "__main__":
    from task_model import Task

    print("=== 執行/成本模型自我測試 ===\n")

    rsus = {"rsu_0": {"x": 0, "y": 0}}
    nodes = build_nodes(rsus)
    nodes.register("server1", VEHICLE_CPU, "vehicle")   # 做決策的 server 車
    nodes.register("server2", VEHICLE_CPU, "vehicle")   # 鄰居 server 車

    src = (50, 0)        # server1 距 rsu_0 50m
    nb = (80, 0)         # 鄰居 server2，距 server1 30m
    rsu_pos = (0, 0)

    task = Task("t1", "server1", "nav", data_bits=3e6, cpu_cycles=0.8e9,
                deadline_s=1.5, created_at=0.0, result_bits=0.3e6)

    print(f"[1] 任務 {task.task_id}：資料{task.data_bits/1e6:.0f}Mbit "
          f"運算{task.cpu_cycles/1e9:.1f}Gcycle deadline{task.deadline_s}s")
    print("    四種選項的延遲(節點都空閒時)：")
    opts = [
        ("本地 local", dict(target_kind="local", target_id="server1")),
        ("V2V 鄰居",   dict(target_kind="v2v",   target_id="server2", target_pos=nb)),
        ("基站 RSU",   dict(target_kind="rsu",   target_id="rsu_0",   target_pos=rsu_pos)),
        ("雲端 cloud", dict(target_kind="cloud", target_id="cloud",   target_pos=rsu_pos)),
    ]
    for name, kw in opts:
        r = estimate(task, now=0.0, src_pos=src, nodes=nodes, **kw)
        if r["feasible"]:
            b = r["breakdown"]
            print(f"      {name:10}: 總延遲 {r['latency']*1000:6.1f} ms "
                  f"｜傳輸{(b['uplink']+b['downlink'])*1000:5.1f} 排隊{b['wait']*1000:4.0f} 運算{b['compute']*1000:5.1f} "
                  f"｜能耗 {r['energy']:.3f} J")
        else:
            print(f"      {name:10}: 不可達")

    print("\n[2] 基站佇列效應：連續送 5 個 vision 任務到 RSU，排隊時間會累積：")
    for i in range(5):
        vt = Task(f"v{i}", "s", "vision", data_bits=8e6, cpu_cycles=2.0e9,
                  deadline_s=3.0, created_at=0.0, result_bits=0.8e6)
        r = estimate(vt, now=0.0, src_pos=src, nodes=nodes,
                     target_kind="rsu", target_id="rsu_0", target_pos=rsu_pos, commit=True)
        print(f"      第{i+1}個到 RSU：排隊 {r['breakdown']['wait']*1000:6.0f} ms，總延遲 {r['latency']*1000:6.0f} ms")

    vt = Task("vc", "s", "vision", data_bits=8e6, cpu_cycles=2.0e9,
              deadline_s=3.0, created_at=0.0, result_bits=0.8e6)
    rc = estimate(vt, now=0.0, src_pos=src, nodes=nodes,
                  target_kind="cloud", target_id="cloud", target_pos=rsu_pos)
    print(f"      → 此時同樣任務改送雲端：排隊 {rc['breakdown']['wait']*1000:.0f} ms(雲端無佇列)，"
          f"總延遲 {rc['latency']*1000:.0f} ms")
    print("      （RSU 塞住時往雲端送反而更快 —— 這正是 agent 要學的取捨）")

    print("\n[3] 超出範圍 → 不可達：")
    far = (500, 0)
    r = estimate(task, now=0.0, src_pos=far, nodes=nodes,
                 target_kind="rsu", target_id="rsu_0", target_pos=rsu_pos)
    print(f"      server1 距 RSU 500m 想卸載到 RSU：feasible={r['feasible']}（應為 False）")

    print("\n=== 測試結束（數字合理即代表模組正常）===")
