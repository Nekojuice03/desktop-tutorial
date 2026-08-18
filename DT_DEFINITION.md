# 數位孿生定位與實驗方法（DT_DEFINITION.md）

> 本檔回答三個會被審稿人/口委直接問的問題:
> **(1) 你說的「數位孿生」是哪一種?(2) 你的孿生跟實體有多像?(3) 這跟普通的
> SUMO 模擬差在哪?**
> 這是論文 §5.1 與「假設聲明」一節的底稿。技術細節見 `COMM_MODEL.md`,
> 進度與參數見 `PROGRESS.md`,圖表主張見 `EXPERIMENTS_OUTLINE.md`。

---

## 1. 定義:判準是資料流,不是「像不像」

最常被引用的分類是 Kritzinger 等人的三階梯。分類依據**只有一件事**:
實體與數位之間,兩個方向的資料流各自是自動還是手動。

| 層級 | 實體→數位 | 數位→實體 | 白話 |
|---|---|---|---|
| Digital Model(數位模型) | 手動 | 手動 | 照著實體建的模型;實體變了模型不會自己變 |
| Digital Shadow(數位陰影) | **自動** | 手動 | 模型自動跟著實體更新,但結論不會自動回去改實體 |
| Digital Twin(數位孿生) | **自動** | **自動** | 雙向閉環,孿生的決策自動作用回實體 |

兩點補充,對本題目關鍵:

- Grieves 的原始定義另外要求孿生能**預測實體的未來狀態**,而不只是鏡射現況。
  這正是本專案 EKF-CTRV 預判器的立足點(見 §5)。
- 交通領域的代表作(Kušić 等,*Advanced Engineering Informatics*, 2023,
  即 `realtime_calibrator.py` 檔頭所引)做到 Digital Shadow:即時偵測器串流
  經 calibrator 餵進 SUMO。**該文有報同步/校正誤差** —— 這是本專案先前缺的一塊,
  已由 `validate_twin_fidelity.py` 補上。

> 參考文獻的卷期頁碼請在投稿前自行核對:
> Kritzinger et al., *IFAC-PapersOnLine* 51(11), 2018;
> Grieves & Vickers, 2017;Kušić et al., *Adv. Eng. Informatics*, 2023;
> 另可參考 ISO 23247 系列(製造業 DT 框架)作為「定義層級」的標準依據。

---

## 2. 本專案逐元件定位(誠實版)

| 元件 | 實體→數位 | 層級 |
|---|---|---|
| 路網 OSM → `hepingeast2.net.xml` | 一次性離線 | Digital Model |
| VD 車流 → `make_real_flow.py` → `real_traffic_hep.rou.xml` | **一次性離線**(抓一次快照,產靜態 rou) | Digital Model(有真實資料校正,但非自動流) |
| `realtime_calibrator.py`(每 60s 抓 VD → `calibrator.setFlow()`) | 自動、持續 | **Digital Shadow** |
| 卸載決策 → 回到真實車輛/RSU | 無 | 未達 Digital Twin |

**結論:論文正式實驗跑的那條管線是 Digital Model 等級(資料驅動、離線校正)。**
`realtime_calibrator.py` 是本專案唯一達到 Shadow 的元件,但它**未接入任何實驗腳本**,
且仍綁在舊場景 —— 見 §7。

**所以不要在論文裡寫 "real-time digital twin" 或 "closed-loop digital twin"。**

---

## 3. 關鍵聲明:SUMO 的雙重角色

本專案沒有實體測試場,因此 **SUMO 同時扮演兩個角色,論文必須把它們拆成兩個物件**:

```
物理世界(ground truth) = SUMO 的 veh_states   ← 實際傳輸距離、完成時刻結算
孿生(twin)            = twin_states(τ 秒前)  ← 鄰居發現、觀測、預判、KF 量測
```

程式上的分離點在 `vec_env_ma.py`:`self._hist`(長度 τ+1 的緩衝)與
`self.twin_states`;`_build_context()` 同時產出孿生位置與 `*_pos_phys` 物理位置。

**這件事為什麼非講不可:** 如果孿生與物理是同一個物件(τ=0),那「數位孿生」
四個字就只是包裝,「這不就是普通的 SUMO 模擬?」就答不出來。
本專案的答案是:孿生是一個**有同步誤差、且能對誤差做補償**的獨立物件,
補償能力可直接換算成任務成功率(§5)。

論文用語建議:*SUMO is used as a **physical-world surrogate** providing the
ground truth against which the twin's (delayed) view is settled.*

---

## 4. 證據一:孿生保真度(`validate_twin_fidelity.py`)

宣稱「真實資料數位孿生」就必須量化孿生與實體有多像。用交通模擬校正的業界標準:

```
GEH = sqrt( 2(M-C)^2 / (M+C) )     M=模擬小時流量, C=VD 實測小時流量
驗收慣例:GEH<5 的 link 佔比 >= 85%
```

執行:

```bash
# 建議:先存一份 VD 快照,論文才可重現
python validate_twin_fidelity.py --sumocfg hepingeast2.sumocfg \
       --vd-xml traffic_data/GetVD_<時間>.xml --plot
```

輸出 `twin_fidelity.csv` / `twin_fidelity.json` / `fig_twin_fidelity.png`,
包含 GEH、GEH<5 佔比、MAPE、RMSE,以及「需求實現度」(rou.xml 指定量 vs 實際跑出量)。

**兩個層級要在論文裡分清楚:**

- **L1 需求實現度**:rou.xml 指定的 vehsPerHour vs 模擬通過量。落差來自插入失敗、
  壅塞、路徑選擇 —— 這是「模擬器有沒有照做」,是診斷用的。
- **L2 孿生保真度**:VD 實測量 vs 模擬通過量 → GEH。這是論文要的那張表。

**⚠ 循環論證的兩個陷阱,腳本已處理,但論文必須寫出來:**

1. **同一份快照**:若驗證用的 VD 快照就是產生車流的那份,GEH 衡量的是
   **校正契合度(calibration fit)**,不是獨立驗證。要做獨立驗證,請用
   **另一個時刻**的快照當地真(`--vd-xml` 指到不同檔),或保留一站不參與建流當
   hold-out。論文請據實說明是哪一種。
2. **鏡射/代理站**:`vd_sumo_mapping.csv` 中標註「鏡射」「代理」的列是建模假設
   (對向鏡射、以鄰站代理),拿它們算 GEH 是拿自己的假設驗證自己。
   腳本只把「實測站」列入主指標,其餘分開報告。**對應表中實測站只有 4 條 link,
   且 2026-08-17 快照中僅 2 站有 TotalVol**(`ZF9KB40`、`ZFYKD00` 缺值)→ 實際 n=2。
   佔比型統計意義有限 → 論文請直接報逐 link 的 GEH 並聲明當次的 n。
   台北 VD 偶爾缺站,換時段重抓可能補回,值得多抓幾份快照挑 n 最大的一份。

**⚠ 第三個陷阱(2026-08 實測發現):把 link counts 當成 OD 需求**

`make_real_flow.py` 把每個 VD 的路段流量直接當成該 edge 的**產生量**注入。
但 VD 量到的是**通過量**:上游 VD 量到的車開到下游會再被下游 VD 量一次,
現實中是同一批車,模型裡卻變成兩批。加上一個站的量被鏡射/代理注入到多條 edge,
整條走廊會被灌爆。

和平東路實測(2026-08-17 快照,n=2 實測站):

| edge | VD 實測 | 模擬 | 需求實現度 | GEH |
|---|---|---|---|---|
| 205066812#6 | 410 | 761 | **185.5%** | **14.50** |
| 317526886#0 | 271 | 268 | 99.2% | 0.14 |

拓撲檢查確認機制:`58976063#4`、`1466668326#0`、`1068183466#0`、`317526886#0`
四條 mapping edge 的車流都會經過 `205066812#6`。注入總量約 1772 veh/h、
SUMO 實際插入 1643 車 —— **總量級對,但空間分布錯**。

標準解法是把計數當**約束**而非注入量,用 OD/route 估計反推一致的路徑集合。
本專案以 `calibrate_flow.py`(包裝 SUMO 自帶的 `routeSampler.py`)實作:

```bash
python calibrate_flow.py --net hepingeast2.net.xml --vd-xml traffic_data/GetVD_<時間>.xml
python validate_twin_fidelity.py --sumocfg hepingeast2.sumocfg \
       --routes real_traffic_hep_calibrated.rou.xml --vd-xml <同一份快照> --plot
```

預設只用**實測站**當約束(鏡射/代理是建模假設,不該當量測)。
⚠ 實測站僅 2–4 個 → 走廊需求是**欠定**的:滿足這些計數的流量組合不只一種。
論文必須寫「以 n=N 個偵測器校正」,不可宣稱整條走廊都被量測約束。

⚠ 採用校正後車流會改變車流密度 → V2V 機會與 RSU 負載改變 →
§5.2~5.6 全部需重跑、MAPPO 需重訓。

### 4.1 三組車流的實測比較與選擇(2026-08-17 快照)

同一份 VD 快照(`traffic_data/GetVD_20260817_140317.xml`)、同一路網,
三種產生車流的方式,以**實測站**(n=2)為主指標:

| | 車流來源 | 插入車輛 | 205066812#6 GEH | 317526886#0 GEH | MAPE | 判定 |
|---|---|---|---|---|---|---|
| (a) | `make_real_flow.py`(計數當注入量) | 1643 | **14.50** | 0.14 | 43.2% | ✗ |
| (b) | `calibrate_flow.py --counts measured`(n=2 約束) | 613 | **0.08** | 0.63 | 2.1% | ✓ |
| (c) | `calibrate_flow.py --counts all`(n=6 約束) | 1269 | **1.31** | 0.36 | 4.3% | ✓ |

**(b) 的實測站數字最漂亮,但不能用。**它的其他 link 幾乎沒有車:

```
1466668326#0   21 vs 405  (−94.9%)
260789408#0    46 vs 405  (−88.7%)
1068183460#0   44 vs 267  (−83.6%)
58976063#4    212 vs 405  (−47.7%)
```

只有 2 個約束時,routeSampler 只把車放在那兩條 link 的路徑上,**整條走廊的
對向幾乎是空的**。對 V2V 卸載研究是致命的:鄰居發現、強車選擇、移動性斷線
全部失去動態範圍。

**採用 (c)。** 理由:
1. 實測站(真實量測、無循環論證)的 GEH 為 1.31 / 0.36,遠低於合格門檻 5;
2. 保留了合理的車輛密度(1269 vs (a) 的 1643),V2V 情境仍有意義;
3. 其他 link 除 `1466668326#0` 外皆在 −12% 以內。

**(c) 的已知缺陷**:`1466668326#0` 需求實現度僅 44.2%,且模擬有 13.4% 的車
插不進路網。原因是**該 edge 只有 1 條車道**,而對向 `260789408#0` 是 3 車道 ——
和平東路東段西向在現實中不可能是單線道,這是路網幾何缺陷(可能就是當初 netedit
補路時用了預設車道數)。修正後應重跑 (c)。

**論文寫法**:(c) 為主結果;(b) 放附錄當 sensitivity check,並說明
「僅用獨立量測約束時 GEH 更低,但車輛數不足以支撐 V2V 情境」——
這反而是「為何需要對向鏡射假設」的正面論證。

> Traffic demand was estimated with SUMO's `routeSampler`, using link counts
> from n=2 independently measured VD stations extended to the opposing and
> adjacent links under the stated mirroring assumption (6 constraints in total).
> Fidelity on the two independently measured links: GEH 1.31 and 0.36
> (MAPE 4.3%), against the industry acceptance threshold of GEH < 5.
> A variant constrained only by the 2 measured links reaches GEH 0.08/0.63 but
> leaves the opposing direction nearly empty (−48% to −95% on four links) and
> was therefore not used.

---

## 5. 證據二:同步延遲 τ(`run_dt_delay.py`)—— 「DT vs 普通模擬」的答案

孿生收到的 BSM 比物理世界舊 τ 秒。**決策側全面吃舊資料**(鄰居發現與排序、
最近基站判定、觀測距離特徵、contact 預判、KF 量測),物理側不受影響。
孿生過期因此有三種可歸因的後果:

| 統計欄位 | 機制 |
|---|---|
| `stale_miss` | 選錯目標 —— 孿生看得到、物理已出範圍,傳輸根本送不到 |
| `link_break` | 預判失準 —— 完成時刻才發現連線已斷 |
| `pred_reject` | 過度保守 —— 舊資料讓預判器誤殺本來可行的卸載 |

**mock 實測(greedy):**

| τ(s) | linear 成功率 | linear stale_miss | kalman 成功率 | kalman stale_miss |
|---|---|---|---|---|
| 0 | 80.7% | 0 | 80.7% | 0 |
| 2 | 80.1% | 6 | 80.2% | 0 |
| 4 | 79.5% | 47 | 79.9% | 0 |
| 8 | 72.3% | 373 | 75.0% | 1 |

**機制(論文可直接寫):** greedy/MAPPO 都在**孿生視角**上規劃,但在**物理世界**上執行。
linear 預判器被舊位置誤導,持續選中「孿生以為在旁邊、其實已經開走」的車;
EKF-CTRV 以 `lead=τ` 做 dead-reckoning 前推,把量測補償回當下,因而幾乎完全避開這類目標。

→ **這就是「DT 提供未來狀態,而非僅鏡射現況」的量化證據**,也是 KF 預判器
不是包裝而是實質貢獻的直接證明。

**⚠ 既有數字需重跑:** 此語意修正前,τ 只影響預判器,舊版 τ 對結果幾乎沒有作用
(mock,3 seeds:τ=0/2/4/8 成功率皆 78.2%、link_break 恆為 0)。
因此 **§5.5 的 DT 延遲掃描必須重跑**(`python run_dt_delay.py --sumo --plot`)。
τ=0 時孿生與物理為同一份、行為逐位元不變,故**主對照、收斂、多 seed、IPPO、
移動性消融的既有數字全部不受影響**。

---

## 6. 可宣稱 vs 不可宣稱

**✅ 可以說**

- "a **VD-calibrated** digital twin of the Heping E. Rd. corridor, with an
  explicit twin-to-physical **synchronization delay** model"
- "we quantify how **twin staleness τ** degrades offloading decision quality,
  decomposed into stale-target selection, prediction failure, and over-conservatism"
- "EKF-CTRV dead-reckoning **compensates** twin staleness"(DT 的實質貢獻)
- "SUMO serves as the **physical-world surrogate** providing settlement ground truth"
- 保真度數字必須連同 n、快照時間、是否 hold-out 一起報

**❌ 不要說**

- "real-time digital twin"(正式實驗吃的是靜態 rou.xml)
- "closed-loop digital twin"(沒有數位→實體的回饋)
- "high-fidelity"(除非 GEH 表撐得住)
- 不要用鏡射/代理 link 的 GEH 充當驗證證據

**論文 §5.1 可直接用的一段:**

> We build a **digital-shadow-level** twin of a Taipei arterial corridor
> (OpenStreetMap network calibrated with traffic-control-center VD flows).
> As a physical testbed is unavailable, SUMO serves as the **physical-world
> surrogate**, and the twin is modeled as a **separate, delayed view (τ)** of
> that surrogate. This separation lets us quantify the impact of twin
> synchronization delay on offloading decisions — a question a conventional
> simulation study cannot express. Twin fidelity against VD counts is reported
> in Table X (GEH, MAPE); the twin→physical control loop is left as future work.

---

## 7. 已知邊界與待辦(論文「假設」一節)

**孿生層**

- 校正為一次性離線,非持續同步 → Digital Model 等級;
  `realtime_calibrator.py` 已具 Shadow 能力但未接入實驗管線,**且仍綁舊場景**
  (忠孝東路 edge ID / VD 對應),`dt_state_extractor.py` 同樣是舊場景(4 RSU、
  10 維 state,與現行 18 維多代理觀測無關)。兩支目前為 legacy,勿在口試展示,
  否則會與和平東路 8 RSU 場景自相矛盾。
- 無數位→實體的回饋(未達 Kritzinger 的 Digital Twin)。
- 孿生「完全看不到某台車」(而非位置過期)未建模:候選僅取孿生已看過的車,
  離場車的效應由結算層的 `consumer_left` 處理。
- 轉向分配:無真實轉向資料,均分可達邊界出口。
- 對向鏡射假設:單向 VD 以鏡射代理對向。

**通訊/運算層**(既有,見 `PROGRESS.md` §8)

- RSU 無線電資源競爭(目前 SNR 非 SINR、無干擾)、任務不可拆分、
  無重傳/封包錯誤、抵達補送受路側覆蓋率(92%)制約。

**場景一致性**

- 倉庫中的 `osm.sumocfg` 仍指向**舊的忠孝東路小十字路網**
  (`osm.net.xml.gz` + `real_traffic.rou.xml`)。正式實驗請一律使用
  **`hepingeast2.sumocfg`**(net/rou/rsu 三件套已對齊)。
  `validate_twin_fidelity.py` 會偵測並印出場景錯配診斷。

---

## 8. 重現性檢查清單(投稿前)

- [ ] 用 `hepingeast2.sumocfg` 跑所有正式實驗(勿用 `osm.sumocfg`)
- [ ] **把用來產生車流的 VD 快照存進 `traffic_data/` 並入庫**
      (目前只有設備級 `GetVDDATA` 快照,缺路段級 `GetVD` 快照 → 保真度無法離線重現)
- [ ] 把定稿用的 `real_traffic_hep.rou.xml` 與 `twin_fidelity.json` 隨論文歸檔
      (前者目前在 `.gitignore` 中)
- [ ] `python validate_twin_fidelity.py --sumocfg hepingeast2.sumocfg --vd-xml <快照> --plot`
      → 保真度表進 §5.1
- [ ] 若保真度未過關:`python calibrate_flow.py` 產校正車流,兩種車流各驗一次再擇一
      (採用校正版則主結果全部需重跑)
- [ ] `python run_dt_delay.py --sumo --plot` **重跑**(τ 語意已修正)→ §5.5
- [ ] `python verify_invariants.py` → 35/35 PASS
- [ ] 所有 mock 來源圖標註 synthetic;SUMO 圖標註場景與資料日期
