# 接續交接單（NEXT_SESSION.md）

> 給下一個 session 的「立刻接手」重點。完整脈絡見 `PROGRESS.md`；
> 論文寫作底稿見 `EXPERIMENTS_OUTLINE.md`；模型公式見 `COMM_MODEL.md`。
> 分支：`claude/sumo-twin-offloading-debug-17iut1`（全部已 push，工作區乾淨）。

## 一句話現況
程式側**全部完成並驗證**（26 項不變量 PASS）；和平東路 SUMO 主結果已跑出且亮眼
（MAPPO vs Greedy：能耗 −34%、成本 −93%，成功率相當）。剩下的是**使用者端跑指令 + 寫論文**。

## 🔴 目前正在處理的事（唯一 in-progress）
使用者發現 `hepingeast2.net.xml` 抓地圖時**缺了一段路**，正用 **netedit** 手動補：
```
& "$env:SUMO_HOME\bin\netedit.exe" hepingeast2.net.xml
```
（E 畫 edge → I 設車道數/限速同鄰段 → C 檢查連接 → T 檢查號誌相位 → Ctrl+S）

**補完路後的檢查鏈（重要，順序別跳）：**
1. `sumo-gui -n hepingeast2.net.xml` 目檢
2. `python make_real_flow.py --net hepingeast2.net.xml`（**不加 --remap**，沿用手工校正的
   `vd_sumo_mapping.csv`；會自動驗證 edge 存在性）
3. `python setup_rsu.py --net hepingeast2.net.xml --mode junction --corners 2 --plot`
   → 看 `rsu_layout.png`。⚠️ **若 RSU 數從 8 變動 → 全域狀態維度變 → MAPPO 必須重訓**
4. `python verify_invariants.py`（應 26/26 PASS）
5. 拓撲改變會改車輛路徑選擇 → **對照/消融/DT 掃描的既有數字與新網不再嚴格可比**：
   若缺的是幹道 → 全部重跑；若是小巷 → 聲明後沿用。

## ⚠️ 必須先釐清的疑點（影響數據有效性）
先前某批實驗可能在**場景錯配**下跑的：本機 `osm.sumocfg` 的 net-file 一度指向另一個
小十字路網，但 `rsu_positions.json` 仍是和平東路的 8 個 RSU（可視化 GIF 因此混了兩個世界）。
**要確認**：8 組合消融、DT 延遲掃描這兩批是在「osm.sumocfg = hepingeast2 三件套對齊後」跑的嗎？
- 是 → 有效，直接用
- 否 → 對齊後重跑（`visualize_dt.py --sumo` 現在會自動檢查 RSU 是否落在路網內並警告）

`osm.sumocfg` 正確內容應為：
```xml
<net-file value="hepingeast2.net.xml"/>
<route-files value="real_traffic_hep.rou.xml"/>
<additional-files value="rsu.add.xml"/>
```

## ✅ 已完成且有效的結果（PROGRESS.md §4.5 有數字）
- 主對照六方法（fig_latency/energy/cost/success）：MAPPO 能耗 3.02J（Greedy 4.57）、成本 0.012（Greedy 0.171）
- 多指標收斂：vision-only 51→74%
- 多 seed n=3：88.61±0.05%，高度可重現
- IPPO 消融：≈MAPPO（不顯著，有 JAIR 2023 理論支撐，論文寫成負結果）
- 8 組合移動性消融：**證實斷線主因是車主離場(consumer_left)**；抵達補送(v2i+arr)救回 50%

## 📋 使用者端剩餘指令（對齊場景後）
```powershell
python run_dt_delay.py --replot --sumo               # 新版三面板（含 pred_reject 對數圖）
python visualize_dt.py --sumo --tau 2 --ticks 200    # 正式版可視化（自動檢查場景一致性）
```

## 檔案地圖（13 支程式）
- 模型：`infra_config / comm_model / task_model / nodes / kalman_tracker`
- 環境+RL：`vec_env / vec_env_ma / mappo / train_mappo`
- 場景：`setup_rsu(--mode junction) / make_real_flow / build_net.bat / build_scenario.bat`
- 實驗：`compare_and_plot / run_ablation(8組合) / run_seeds / run_dt_delay / analyze_offloading / verify_invariants(26項) / visualize_dt`
- 文件：`README(架構圖) / COMM_MODEL / PROGRESS / EXPERIMENTS_OUTLINE / 本檔`

## 已知未做（論文假設聲明 or future work）
RSU 無線電競爭(SNR 非 SINR)、任務拆分、重傳/封包錯誤；補送成功率受路側覆蓋(92%)制約。
