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
    V2V_BANDWIDTH_HZ, V2V_RANGE_M,
    RSU_BANDWIDTH_HZ, RSU_RANGE_M,
    CLOUD_EXTRA_LATENCY,
    VEHICLE_ENERGY_PER_CYCLE, RSU_ENERGY_PER_CYCLE, CLOUD_ENERGY_PER_CYCLE,
    TX_POWER_W, CLOUD_BACKHAUL_POWER_W,
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
        self._kind = {}       # node_id -> 'vehicle' / 'rsu' / 'cloud'
        self._no_queue = set()  # 無佇列的節點(雲端)

    def register(self, node_id, cpu, kind):
        self._cpu[node_id] = cpu
        self._busy.setdefault(node_id, 0.0)
        self._kind[node_id] = kind
        if kind == "cloud":
            self._no_queue.add(node_id)

    def has(self, node_id):
        return node_id in self._cpu

    def service_time(self, node_id, task):
        """純運算時間 = 運算量 / 算力。"""
        return task.cpu_cycles / self._cpu[node_id]

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


def build_nodes(rsus):
    """
    用基站清單(infra_config.load_rsus 的格式)建立節點池，並加入雲端。
    車輛之後由環境程式在它們進入 SUMO 時動態註冊(VEHICLE_CPU)。
    """
    nodes = ComputeNodes()
    for rid in rsus:
        nodes.register(rid, RSU_CPU, "rsu")
    nodes.register("cloud", CLOUD_CPU, "cloud")
    return nodes


def estimate(task, target_kind, now, src_pos, nodes,
             target_id=None, target_pos=None, commit=False):
    """
    估算把 task 放到某目標執行的總延遲與能耗。
    target_kind: 'local' | 'v2v' | 'rsu' | 'cloud'
      local : 在來源車自己算(target_id = 來源車 id)，無傳輸
      v2v   : 卸載到鄰居 server 車(target_id/target_pos = 鄰居)
      rsu   : 卸載到基站(target_id/target_pos = 基站)
      cloud : 經由服務基站轉送到雲端(target_pos = 該服務基站位置)
    commit=True 時會真的佔用該節點(更新佇列)；估算多選項時用 False。

    回傳 dict：feasible, latency, energy, breakdown{uplink,wait,compute,downlink}
    """
    up_tx = down_tx = 0.0       # 車輛無線電實際傳輸時間(算能耗用)
    up_extra = down_extra = 0.0  # 額外固定延遲(如雲端回程，非車輛傳輸)

    if target_kind == "local":
        node_id = target_id

    elif target_kind == "v2v":
        d = distance(src_pos, target_pos)
        up_tx = transmission_delay(task.data_bits,   d, V2V_BANDWIDTH_HZ, V2V_RANGE_M)
        down_tx = transmission_delay(task.result_bits, d, V2V_BANDWIDTH_HZ, V2V_RANGE_M)
        node_id = target_id

    elif target_kind == "rsu":
        d = distance(src_pos, target_pos)
        up_tx = transmission_delay(task.data_bits,   d, RSU_BANDWIDTH_HZ, RSU_RANGE_M)
        down_tx = transmission_delay(task.result_bits, d, RSU_BANDWIDTH_HZ, RSU_RANGE_M)
        node_id = target_id

    elif target_kind == "cloud":
        d = distance(src_pos, target_pos)   # 先傳到服務基站
        up_tx = transmission_delay(task.data_bits,   d, RSU_BANDWIDTH_HZ, RSU_RANGE_M)
        down_tx = transmission_delay(task.result_bits, d, RSU_BANDWIDTH_HZ, RSU_RANGE_M)
        up_extra = down_extra = CLOUD_EXTRA_LATENCY   # 基站↔雲的回程延遲
        node_id = "cloud"

    else:
        raise ValueError(f"未知的 target_kind: {target_kind}")

    # 任一段傳輸超出範圍 → 不可達(視為任務失敗)
    if up_tx == INF or down_tx == INF:
        return {"feasible": False, "latency": INF, "energy": INF, "breakdown": {}}

    arrival = now + up_tx + up_extra                 # 任務抵達運算節點的時刻
    wait = nodes.wait_time(node_id, arrival)
    compute = nodes.service_time(node_id, task)
    latency = up_tx + up_extra + wait + compute + down_extra + down_tx

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
        e_per_cycle = VEHICLE_ENERGY_PER_CYCLE
    elif target_kind == "rsu":
        e_per_cycle = RSU_ENERGY_PER_CYCLE
    else:  # cloud
        e_per_cycle = CLOUD_ENERGY_PER_CYCLE
    compute_energy = task.cpu_cycles * e_per_cycle               # 運算耗能(執行節點)
    tx_energy = TX_POWER_W * (up_tx + down_tx)                   # 車輛無線傳輸耗能(本地為0)
    backhaul_energy = CLOUD_BACKHAUL_POWER_W * (up_extra + down_extra)  # 僅雲端的回程耗能
    energy = compute_energy + tx_energy + backhaul_energy

    if commit:
        nodes.commit(node_id, arrival, task)

    return {
        "feasible": True,
        "latency": latency,
        "energy": energy,
        "breakdown": {
            "uplink":   up_tx + up_extra,
            "wait":     wait,
            "compute":  compute,
            "downlink": down_tx + down_extra,
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
