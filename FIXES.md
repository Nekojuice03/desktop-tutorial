# 數位孿生車輛任務卸載 — 問題診斷與修正紀錄

本次針對三個回報問題進行檢視與修正。以下為根因分析與對應修改。

---

## 問題 3：邊緣(RSU)與雲端(Cloud)能耗一樣 ← 最根本

**根因**：`nodes.py` 的 `estimate()` 原本能耗只有兩條公式：

```python
if target_kind == "local":
    energy = task.cpu_cycles * VEHICLE_ENERGY_PER_CYCLE   # 本地：運算能耗
else:
    energy = TX_POWER_W * (up_tx + down_tx)               # 卸載：只算傳輸能耗
```

- 卸載到 RSU 與 Cloud 的上/下行**都走「車→最近基站」這條無線鏈路**（相同頻寬、相同距離），
  cloud 只是多了 `extra_latency`（回程延遲，不影響傳輸量）→ **兩者傳輸能耗必然相等**。
- 而且**節點端的「運算能耗」完全沒被計入**，`infra_config.py` 也只有車輛的能耗係數。

**修正**：
- `infra_config.py` 新增三層各自的運算能耗係數，反映「邊緣比雲端節能」：
  - `VEHICLE_ENERGY_PER_CYCLE = 1e-9`（車輛）
  - `RSU_ENERGY_PER_CYCLE = 2e-9`（邊緣，中等）
  - `CLOUD_ENERGY_PER_CYCLE = 4e-9`（雲端，含資料中心散熱/PUE 開銷，最高）
  - `CLOUD_BACKHAUL_POWER_W = 1.0`（雲端 RSU↔雲 回程的等效傳輸功率）
- `nodes.py` 的能耗改為：**運算能耗（依執行節點）＋ 車輛無線傳輸能耗 ＋（僅雲端）回程能耗**。

**驗證**（`python nodes.py`，0.8 Gcycle 任務）：

| 目標 | 能耗 |
|------|------|
| 本地 local | 0.800 J |
| V2V 鄰居 | 0.874 J |
| 邊緣 RSU | **1.697 J** |
| 雲端 cloud | **3.897 J** |

邊緣 ≠ 雲端，且 `本地 < V2V < 邊緣 < 雲` 排序合理。

---

## 問題 2：資源分配都跑去雲端和邊緣（本地/V2V 幾乎沒人選）

**根因**：這是問題 3 的直接後果。獎勵為 `-(延遲 + ENERGY_W × 能耗)`，而原本：
- 本地運算能耗 ≈ 數焦耳（`cpu_cycles × 1e-9`）
- 卸載能耗 ≈ 0.01 焦耳（因為節點運算能耗沒被計）

→ 只要卸載，能耗項幾乎歸零，本地永遠在能耗上吃大虧；加上弱車算力低、延遲也輸，
agent 自然學成「什麼都往 RSU/Cloud 丟」。而 RSU 與 Cloud 能耗又相同（問題 3），
只能靠延遲區分 → 兩者都被大量使用，本地/V2V 被餓死。

**修正**：問題 3 的能耗模型修正後，卸載要計入節點運算能耗：
- 輕任務（如 sensor）在本地運算能耗很低、又能在 deadline 內完成 → 回到本地更划算。
- 重任務（如 vision）本地算不動，會優先選邊緣/強車（能耗較雲低），而非無腦上雲。
- 動作分布自然回到合理的分層分工。

> 註：能耗在獎勵中以 `ENERGY_NORM`（見 `vec_env.py`）正規化，讓能耗與延遲尺度相近，
> 各層能耗差異才能被公平比較。

---

## 問題 1：訓練曲線奇怪

**根因**：
1. **評估種子每次都在變**：`train_mappo.py` 的 `evaluate()` 與訓練**共用同一個 env**，
   而 env 每次 `reset()` 用 `base_seed + self._ep`（遞增）→ 每個評估點都是**不同的隨機情境**，
   收斂曲線因此劇烈抖動、不可重現。
2. **獎勵尺度被能耗尖峰主導**：本地能耗（數焦耳）vs 卸載（≈0）落差極大、未正規化 →
   獎勵變異大 → advantage 不穩 → 曲線亂。

**修正**：
- `train_mappo.py`：改用**獨立的評估環境 `eval_env` + 固定種子集 `EVAL_SEEDS`**，
  每次評估都面對相同情境 → 收斂曲線穩定、可重現、可比較。
- 能耗正規化（`ENERGY_NORM`）使獎勵尺度穩定，降低訓練變異。
- 收斂紀錄 `mappo_train_log.csv` 增加 `avg_energy_j` 欄位，方便畫能耗收斂圖。

---

## 修改檔案清單

| 檔案 | 修改 |
|------|------|
| `infra_config.py` | 新增 RSU/Cloud 運算能耗係數、雲端回程功率 |
| `nodes.py` | `estimate()` 能耗 = 運算 + 傳輸 + 回程，依執行節點分層 |
| `vec_env.py` | 新增 `ENERGY_NORM`、獎勵能耗正規化、統計加入能耗 |
| `vec_env_ma.py` | 同上（MAPPO 多智能體環境）|
| `train_mappo.py` | 評估改用獨立固定種子環境，log 加入能耗欄位 |

## 可調參數（依你的論文設定微調）

- 三層能耗係數比例（`*_ENERGY_PER_CYCLE`）：控制「邊緣 vs 雲」的節能差距。
- `ENERGY_W`、`ENERGY_NORM`（`vec_env.py`）：延遲與能耗在獎勵中的權衡。
- `EVAL_SEEDS`（`train_mappo.py`）：評估情境組（數量越多曲線越平滑穩定）。

## 備註

- 重建說明：本次因 Drive 下載授權在連線重整後失效，部分 `.py` 由完整內容重建並逐一
  `py_compile` 驗證；`infra_config.py`、`nodes.py`、`task_model.py` 為原檔。
- SUMO 設定/資料檔（`*.sumocfg`、`*.rou.xml`、`*.add.xml`、`rsu_positions.json` 等）
  尚未納入此 repo，跑 SUMO 模式時請補上；mock 模式不需要這些檔即可驗證上述修正。
