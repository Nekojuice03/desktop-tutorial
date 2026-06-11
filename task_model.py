"""
任務模型（task_model.py）
=========================
定義「計算任務」長什麼樣，以及 client 車怎麼產生任務。

每個任務有三個關鍵屬性：
  - data_bits  ：若要卸載，需要上傳的資料量(bit)。配 comm_model 算傳輸延遲。
  - cpu_cycles ：需要的運算量(cycle)。除以節點算力(cycle/s)就是運算時間。
  - deadline_s ：從產生起算，最多容許幾秒內完成，超過就算失敗。

卸載決策 = 替每個任務選「本地/V2V/基站/雲」，使
   總延遲(傳輸 + 運算) 最小、且不超過 deadline、能耗也低。

純邏輯，可單獨測試（直接執行會跑自我測試）。
"""
import random
from dataclasses import dataclass

# 任務類型範本：不同應用有不同的資料量/運算量/deadline
# (數值為範圍，產生時在範圍內隨機取；weight = 出現的相對頻率)
TASK_PROFILES = {
    "sensor": {   # 感測資料聚合：很輕、本地就能快速做完
        "data_bits":  (0.3e6, 1.0e6),
        "cpu_cycles": (0.05e9, 0.2e9),
        "deadline":   (0.3, 0.6),
        "weight": 4,
    },
    "nav": {      # 導航/路徑規劃：中等
        "data_bits":  (1.0e6, 3.0e6),
        "cpu_cycles": (0.3e9, 0.8e9),
        "deadline":   (1.0, 2.0),
        "weight": 3,
    },
    "vision": {   # 影像/物件辨識：很重，本地算不動 → 必須卸載(強車/基站/雲)
        "data_bits":  (3.0e6, 8.0e6),
        "cpu_cycles": (3.0e9, 6.0e9),
        "deadline":   (1.0, 2.0),
        "weight": 3,
    },
}

RESULT_RATIO = 0.1   # 回傳結果的資料量 = 輸入的 10%（結果通常比輸入小）


@dataclass
class Task:
    task_id: str
    source: str          # 產生此任務的 client 車 ID
    kind: str            # 任務類型(sensor/nav/vision)
    data_bits: float     # 要上傳的資料量(bit)
    cpu_cycles: float    # 需要的運算量(cycle)
    deadline_s: float    # 從產生起算，最多容許幾秒完成
    created_at: float    # 產生時間(模擬秒)
    result_bits: float   # 回傳結果的資料量(bit)

    def age(self, now):
        """任務已存在多久(秒)。"""
        return now - self.created_at

    def remaining(self, now):
        """距離 deadline 還剩幾秒(負值代表已超時)。"""
        return self.deadline_s - self.age(now)

    def is_expired(self, now):
        return self.remaining(now) < 0


class TaskGenerator:
    """
    依抵達率讓每台 client 車隨機產生任務。
    arrival_rate：每台 client 每秒平均產生幾個任務
                 (例如 0.1 ≈ 平均每 10 秒一個)。
    seed：固定亂數種子，讓實驗可重現。
    """
    def __init__(self, arrival_rate=0.1, seed=0, cpu_scale=1.0, deadline_scale=1.0):
        self.arrival_rate = arrival_rate
        self.rng = random.Random(seed)
        self.cpu_scale = cpu_scale            # 運算量倍率(>1 = 任務更重)
        self.deadline_scale = deadline_scale  # deadline 倍率(<1 = 更緊)
        self._counter = 0
        self._kinds = list(TASK_PROFILES.keys())
        self._weights = [TASK_PROFILES[k]["weight"] for k in self._kinds]

    def _make_one(self, source, now):
        kind = self.rng.choices(self._kinds, weights=self._weights)[0]
        p = TASK_PROFILES[kind]
        data = self.rng.uniform(*p["data_bits"])
        self._counter += 1
        return Task(
            task_id=f"t{self._counter}",
            source=source,
            kind=kind,
            data_bits=data,
            cpu_cycles=self.rng.uniform(*p["cpu_cycles"]) * self.cpu_scale,
            deadline_s=self.rng.uniform(*p["deadline"]) * self.deadline_scale,
            created_at=now,
            result_bits=data * RESULT_RATIO,
        )

    def step(self, client_ids, now, dt=1.0):
        """
        推進一個時間步：每台 client 以機率 arrival_rate*dt 產生一個任務。
        回傳這一步新產生的任務清單。
        """
        new_tasks = []
        prob = self.arrival_rate * dt
        for cid in client_ids:
            if self.rng.random() < prob:
                new_tasks.append(self._make_one(cid, now))
        return new_tasks


# ==================================================================
# 自我測試：直接執行 `python task_model.py`，不需要開 SUMO
# ==================================================================
if __name__ == "__main__":
    from collections import Counter
    from infra_config import VEHICLE_CPU, RSU_CPU, CLOUD_CPU

    print("=== 任務模型自我測試 ===\n")

    gen = TaskGenerator(arrival_rate=0.1, seed=42)
    clients = ["c1", "c2", "c3"]
    STEPS = 200
    all_tasks = []
    for t in range(STEPS):
        all_tasks += gen.step(clients, now=float(t), dt=1.0)

    print(f"[1] {len(clients)} 台 client、{STEPS} 秒、抵達率 0.1/秒：")
    print(f"    共產生 {len(all_tasks)} 個任務"
          f"（理論約 {len(clients) * STEPS * 0.1:.0f} 個，接近即正常）")

    print("\n[2] 任務類型分布（sensor 應最多，vision 最少）：")
    cnt = Counter(t.kind for t in all_tasks)
    for k in TASK_PROFILES:
        print(f"    {k:7}: {cnt.get(k, 0)} 個")

    print("\n[3] 三個範例任務：")
    for tk in all_tasks[:3]:
        print(f"    {tk.task_id} 來自{tk.source} 類型{tk.kind:6} "
              f"資料{tk.data_bits/1e6:5.2f}Mbit 運算{tk.cpu_cycles/1e9:4.2f}Gcycle "
              f"deadline{tk.deadline_s:.1f}s")

    print("\n[4] 同一任務在三層的『運算時間』（只算運算，不含傳輸）：")
    tk = all_tasks[0]
    print(f"    任務 {tk.task_id}（{tk.cpu_cycles/1e9:.2f} Gcycle）：")
    for name, cpu in (("本地車", VEHICLE_CPU), ("基站", RSU_CPU), ("雲端", CLOUD_CPU)):
        print(f"      {name:4}: {tk.cpu_cycles/cpu*1000:6.1f} ms")
    print("    → 雲端運算最快，但加上傳輸延遂(comm_model)後不一定划算，三層要一起權衡")

    print("\n[5] deadline 邏輯：")
    tk = all_tasks[0]
    base = tk.created_at
    print(f"    任務於 t={base:.0f} 產生，deadline {tk.deadline_s:.1f}s")
    print(f"    到 t={base+0.5:.1f}：還剩 {tk.remaining(base+0.5):.2f}s，超時? {tk.is_expired(base+0.5)}")
    print(f"    到 t={base+5:.1f}：還剩 {tk.remaining(base+5):.2f}s，超時? {tk.is_expired(base+5)}")

    print("\n=== 測試結束（數字合理即代表模組正常）===")
