# 卸載模型與決策架構（OFFLOADING_MODEL.md）

> 本檔說明**卸載這一側**:任務怎麼產生、決策鏈長什麼樣、能耗/成本怎麼算、
> PPO 看到哪些狀態。與 `DT_DEFINITION.md`(孿生層)、`COMM_MODEL.md`(通訊公式)
> 互補,是論文「系統模型」與「假設聲明」兩節的底稿。
>
> 最後一次全面檢視:2026-08。該次發現的問題與修正見 §7。

---

## 1. 決策鏈

```
client 車產生任務(sensor / nav / vision,Poisson 到達)
   ↓ 指派跳:V2V 傳給範圍內「第一個本 tick 尚未被指派」的 server 車
             (付 PC5 存取 5ms + 傳輸延遲 + 傳輸能耗)
server 車 = RL agent(持有車 holder)
   ↓ 五選一
   local        持有車自己算(弱車 1 GHz / 強車 12 GHz)
   v2v_strong   最近的強車(12 GHz,κf² 較耗能,會移動、會離場)
   v2v_near     最近的鄰車(算力低但近)
   rsu          最近的邊緣基站(8 GHz/槽,FIFO,計費)
   cloud        雲端(50 GHz,無佇列,經 V2I + 有限回程,計費最貴)
   ↓ 事件驅動結算
   到「預計完成時刻」用**物理真實位置**驗證結果送不送得回持有車;
   送不回 → link_break → 恢復層(V2I 遷移 / 抵達補送 / 失敗)
```

**每個 server 每 tick 只當一個任務的 agent**(RL 一 agent 一觀測)。範圍內所有
server 都已被指派時,任務退回 client 本地執行(`fallback`),並記錄成因:

| 統計 | 意義 |
|---|---|
| `fb_no_server` | 範圍內沒有任何 server → **佈署密度**不足 |
| `fb_saturated` | 有 server 但本 tick 全被佔用 → **每 tick 一 agent** 的結構上限 |

論文談 fallback 時要分開講,兩者的改善方向完全不同。

---

## 2. ⚠ 能耗的定義(最容易被誤讀的一點)

**本專案的「能耗」是系統總能耗,不是卸載車的省電量。**

```
energy = 執行節點的運算能耗(cycles × 該層 J/cycle)
       + 無線傳輸能耗(上行用車輛功率、下行用發射端功率)
       + (僅雲端)有線回程傳輸能耗
```

實測一個 vision 任務(4.5 Gcycle,佇列全空):

| 選項 | 延遲 | 能耗 |
|---|---|---|
| local(弱車 1 GHz) | 4.50 s | **4.50 J ← 最低** |
| v2v_strong(12 GHz) | 0.46 s | 10.16 J |
| rsu(8 GHz) | 0.60 s | 9.01 J |
| cloud(50 GHz) | 0.27 s | 18.07 J |

**在這個模型裡,本地執行最省能;卸載是拿能耗換 deadline。**
這與 MEC 領域的預設直覺(卸載是為了幫裝置省電)相反,因為弱車的
J/cycle 最低(1e-9)且不需傳輸。

→ 「MAPPO 能耗比 Greedy 低 34%」的正確解讀是:**MAPPO 更常把任務留在本地
或避開雲端,因而系統總能耗較低**,不是「幫車輛省電」。

**論文必寫的一句話:**

> Energy is accounted **system-wide** (compute at the executing node + radio +
> backhaul), not as the offloading vehicle's battery saving. Under this
> definition, offloading trades energy for deadline compliance.

---

## 3. 兩個發射功率不是筆誤

| 常數 | 值 | 用途 |
|---|---|---|
| `VEH_TX_POWER_DBM` | 23 dBm = **0.2 W** | 天線**輻射**功率 → 鏈路預算 → SINR → Shannon |
| `TX_POWER_W` | **0.5 W** | 收發機**消耗**功率 → 能耗 |
| `RSU_TX_POWER_W` | **2.0 W** | RSU 下行的消耗功率 |

0.5 / 0.2 = 2.5× 隱含功率放大器效率約 40%,為車載 PA 的合理區間。
**論文要一併說明**,否則看起來像前後矛盾。

`RSU_TX_POWER_W` 是 2026-08 新增:先前下行一律用車輛的 0.5 W,但
RSU→車 這段是路側設備在發射。能耗是系統總帳,所以仍計入,只是改用
發射端自己的功率。

---

## 4. 成本(pay-per-use)

```
cost = cpu_cycles × 執行節點單價
  雲端 2.0e-10 /cycle · 邊緣 5.0e-11 /cycle · 本地/V2V 0
```

一個 vision 任務在雲端要 0.90、在 RSU 要 0.23、在車上 0。
在獎勵裡的量級與延遲項(秒)相當,**不是可忽略的小項**。

---

## 5. PPO 的狀態與獎勵

### 觀測(19 維,`priority_aware` 時 20 維)

| # | 內容 | 正規化 |
|---|---|---|
| 0–3 | 任務:資料量、運算量、deadline、剩餘時間 | 10 Mb / 6 G / 3 s / 3 s |
| 4–5 | 自身:是否強車、本地佇列等待 | — / 2 s |
| 6–8 | RSU:在覆蓋內?、距離、佇列等待 | — / 200 m / 2 s |
| 9–12 | 最近強車:在?、距離、佇列、**連線可維持時間** | — / 150 m / 2 s / 60 s |
| 13–16 | 最近鄰車:在?、距離、是否強車、連線時間 | — / 150 m / — / 60 s |
| 17 | 回程(骨幹)佇列等待 | 2 s |
| 18 | **範圍內 server 數(競爭程度)** | 10 |
| (19) | 任務優先權(`--priority` 時) | 2.5 |

**★ 決策側的所有動態量都取自孿生視角(`twin_states`,τ 秒前)**,物理側只用於
實際傳輸距離與結算。詳見 `DT_DEFINITION.md` §3。

### 全域狀態(中央 critic 專用)

各 RSU 佇列 + 回程佇列 + 活躍 agent 數 + 平均本地負載 + 車數 + 強車數
= `RSU數 + 5` 維。**RSU 數改變 → 維度改變 → 必須重訓。**

### 獎勵

```
reward = -( w·總延遲 + ENERGY_W·能耗/ENERGY_NORM + COST_W·成本 )
         - w·PENALTY_MISS   (超過 deadline)
w = 任務優先權(priority_aware=False 時恆為 1)
```

- 團隊獎勵 = 該 tick 各 agent 個別獎勵的**平均**;GAE 在 tick 序列上算。
- 延遲結算與預測的差額(`settle_delta`)攤進之後某個 tick 的團隊獎勵。
- **`--individual-reward`(2026-08 新增)**:agent 的 advantage 額外加上
  「自己的獎勵 − 該 tick 團隊平均」(差分獎勵)。時間結構仍走團隊獎勵,
  但個別功過不再被平均掉。用途見 §7。

### 動作合法性

無效動作(選 V2V 卻沒鄰車、選 RSU 卻不在覆蓋內)以 **invalid action masking**
處理(`env.action_masks()` → logit 設 −inf),不再只靠 `-PENALTY_FAIL` 去學。
`local` 永遠合法,保證至少有一個動作可選。

---

## 6. 邊緣層的使用率高度依賴 RSU 配置

佇列全空時,一個 vision 任務的總延遲:

```
雲(經 RSU 50m)  271 ms   ← 最快
強車 30m         457 ms
RSU 50m          597 ms   ← 光運算就 562 ms
```

**RSU 在佇列全空時就已經輸**,所以提高並行度(`RSU_CORES`)救不了它 ——
實測 cores=1→8,greedy 選 RSU 的比例固定在 1.2%。真正的槓桿是每槽算力
(`RSU_CPU`,預設 8 GHz,`infra_config.py` 註明「故意略低於強車」)。

`sweep_rsu_config.py --cpu 8 16 24 32`(mock,greedy):

| 單槽 GHz | 選 RSU % | 選雲 % | 能耗 J | 成本 |
|---|---|---|---|---|
| 8 | 1.2 | 27.7 | 5.50 | 0.237 |
| 16 | 11.8 | 26.2 | 5.39 | 0.233 |
| 24 | **41.8** | 1.5 | 3.24 | 0.078 |
| 32 | 47.8 | 0.0 | 3.08 | 0.067 |

**跨過約 24 GHz,邊緣層取代雲端成為主要卸載對象,能耗降 41%、成本降 72%。**

→ 論文的「反雲三機制」結論**對 `RSU_CPU` 高度敏感**。必須在假設一節說明
採用的值與理由,並附這張敏感度表,否則會被質疑「反雲只是把邊緣層設弱設出來的」。

另一個可寫的觀察:純延遲導向的 **greedy 幾乎不用 RSU,而 MAPPO 會用 RSU 取代
雲端** —— 貪婪法看不見邊緣層在能耗/成本上的價值,這正是學習型策略的優勢。

---

## 7. 2026-08 檢視:發現的問題與修正

### 🔴 已修:39% 的任務繞過策略

`_tick_and_route()` 原本只試**最近**那台 server,被佔用就直接退回 client 本地:

```python
if nbrs and nbrs[0] not in assign:      # 舊
free = [n for n in nbrs if n not in assign]   # 新
```

歸因統計顯示 **582 筆 fallback 中 100% 是「最近那台被佔用」,沒有一筆是
附近沒有 server**。這些任務全在 1 GHz 弱車上跑 → vision fallback 成功率 **0%**。

影響(mock,greedy,3 seeds):

| | 修正前 | 修正後 |
|---|---|---|
| 整體成功率 | 78.2% | **95.7%** |
| vision 成功率 | 60.7% | **90.9%** |
| 平均延遲 | 1290 ms | **400 ms** |
| fallback 筆數 | 583 | 135 |

**更嚴重的是**:策略真正決策的那部分,greedy 成功率是 **100%**。也就是說
先前論文的「任務成功率」量到的幾乎全是那 39% 策略沒碰過的固定池 ——
六個方法共享同一組結果,方法間差距被稀釋。這正是所有方法都擠在 78~89%、
而「MAPPO ≈ Greedy」的主因之一。

**⚠ 修正前的成功率與延遲數字全部失效,必須重跑。**

### 🔴 須聲明:能耗語意(見 §2)

### 🟠 已修:觀測缺「競爭程度」

`n_nbrs`(範圍內 server 數)先前算了但沒放進觀測 → agent 無法感知資源競爭。
已補為第 18 維。**觀測維度 18 → 19,舊模型作廢,必須重訓。**

### 🟠 已修:下行能耗記錯發射端(見 §3)

### 🟡 已修:無效動作改用 masking(見 §5)

### 🟡 已加工具:差分獎勵消融

`--individual-reward`。動機:任務之間幾乎獨立(耦合只來自佇列與回程),
團隊獎勵平均會把個別功過抹平,同一 tick 做對與做錯的 agent 拿到相同 advantage。
**這可能才是「IPPO ≈ MAPPO」的真正原因** —— 不只是「協調資訊局部可觀測」,
而是團隊獎勵本身就消掉了中央 critic 能提供的信用分配優勢。

建議跑:`train_mappo.py --individual-reward` 與 `--ippo --individual-reward`,
若差分獎勵下 MAPPO 明顯優於 IPPO,論文的解釋就要改寫(比現在只引 Lyu 2023
更貼近自己的實作)。

### 🟠 未修(設計問題,需你決定)

**原始 client 在指派後就從模型消失了。** `task.source` 只在指派那一刻用到;
之後「持有車」都是 server,`consumer_left` 指的也是 server 離場。
意思是**結果算完停在 server 車上,從未送回產生任務的 client**,少了一段
回程(延遲 + 能耗),且 client 中途離場完全不影響結果。

兩種處理:
1. 論文明說「server 才是任務擁有者,client 只是感測資料來源」——
   那就不需要回程,但要解釋為什麼還付了指派跳的代價。
2. 補上 server → client 的回程與結算,較貼近「誰要結果誰是消費者」。

**未修(可選改進)**

- 動作只能指向**最近**的強車/鄰車/RSU;最近那台忙碌時沒有第二選擇。
  要改需擴充動作空間(研究設計決策,不宜默默更動)。
- `v2v_strong` 與 `v2v_near` 在最近鄰車本身就是強車時指向同一台(冗餘動作)。
- RSU 無線電資源競爭(SNR 非 SINR)、任務不可拆分、無重傳 —— 既有假設。

---

## 8. 重跑檢查清單

觀測維度與指派邏輯都變了,**既有的 `mappo_vec.pt` 一律作廢**。建議把車流校正
(見 `DT_DEFINITION.md` §4)與本檔的修正**一次改完再重訓**,只做一輪:

- [ ] `python verify_invariants.py` → 40/40 PASS
- [ ] `python train_mappo.py --sumo`(觀測 19 維,需重訓)
- [ ] `python compare_and_plot.py --sumo`
- [ ] `python run_ablation.py --sumo --plot`
- [ ] `python run_seeds.py --sumo --seeds 3`
- [ ] `python run_dt_delay.py --sumo --plot`
- [ ] `python sweep_rsu_config.py --cpu 8 16 24 32 --mappo --plot` → 敏感度表進假設一節
- [ ] `python train_mappo.py --individual-reward` + `--ippo --individual-reward`
      → 重新檢驗 IPPO ≈ MAPPO 的解釋
- [ ] 論文加上:能耗語意(§2)、兩個發射功率(§3)、RSU 配置敏感度(§6)、
      fallback 成因拆解(§1)
