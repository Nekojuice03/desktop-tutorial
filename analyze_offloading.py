"""
卸載決策分析器（analyze_offloading.py）
========================================
不需 SUMO / torch，純用 nodes.estimate() 的成本模型，回答：
  「到底走 V2V(強車/近車) 好，還是 V2I(基站) / 雲 / 本地好？在什麼條件下翻盤？」

做四個研究：
  [1] 各任務類型在典型情境下，五種卸載選擇的延遲/能耗/成本對照與贏家
  [2] 距離掃描：強車距離拉遠，何時 V2I(基站) 反超 V2V
  [3] 基站負載掃描：RSU 佇列變長，何時雲端反超 RSU
  [4] 強車 vs 弱車 V2V：強車快但依 κf² 較耗能的取捨

執行：python analyze_offloading.py
"""
from nodes import build_nodes, estimate, ComputeNodes
from task_model import Task
from infra_config import (VEHICLE_CPU, STRONG_VEHICLE_CPU,
                          STRONG_VEHICLE_ENERGY_PER_CYCLE)

# 獎勵權重：與 vec_env.py 一致（reward = -(延遲 + W_E·能耗/NORM + W_C·成本)）
ENERGY_W, ENERGY_NORM, COST_W = 1.0, 5.0, 1.0

# 代表性任務（取各 profile 範圍中點）
TASKS = {
    "sensor 感測(輕)": dict(data_bits=0.65e6, cpu_cycles=0.125e9, deadline=0.45),
    "nav 導航(中)":   dict(data_bits=2.0e6,  cpu_cycles=0.55e9,  deadline=1.5),
    "vision 影像(重)": dict(data_bits=5.5e6,  cpu_cycles=4.5e9,   deadline=1.5),
}

# 五種卸載選擇（顯示名 → estimate 的 target_kind）
OPTIONS = [
    ("本地 local",   "local"),
    ("V2V 強車",     "v2v"),    # 目標=強車
    ("V2V 近車",     "v2v"),    # 目標=近的弱車
    ("基站 RSU",     "rsu"),
    ("雲端 cloud",   "cloud"),
]


def make_task(name, prof, now=0.0):
    return Task(task_id="t", source="self", kind=name.split()[0],
                data_bits=prof["data_bits"], cpu_cycles=prof["cpu_cycles"],
                deadline_s=prof["deadline"], created_at=now,
                result_bits=prof["data_bits"] * 0.1)


def fresh_world(strong_d=60.0, near_d=40.0, holder_to_rsu=70.0):
    """
    建立一個分析用世界：
      RSU 在原點；持有任務的弱車在 (holder_to_rsu, 0)；
      強車鄰居在持有車左側 strong_d 公尺；近的弱車鄰居在 near_d 公尺。
    回傳 (nodes, 幾何位置 dict)。
    """
    nodes = build_nodes({"rsu_0": {"x": 0.0, "y": 0.0}})
    holder = (holder_to_rsu, 0.0)
    nodes.register("self", VEHICLE_CPU, "vehicle")
    nodes.register("strong", STRONG_VEHICLE_CPU, "vehicle",
                   energy_per_cycle=STRONG_VEHICLE_ENERGY_PER_CYCLE)
    nodes.register("near", VEHICLE_CPU, "vehicle")
    geo = {"holder": holder, "rsu": (0.0, 0.0),
           "strong": (holder_to_rsu - strong_d, 0.0),
           "near":   (holder_to_rsu - near_d, 0.0)}
    return nodes, geo


def eval_option(nodes, geo, task, disp, kind):
    """算單一卸載選擇的 latency/energy/cost/reward；不可行回 None。"""
    if disp == "本地 local":
        tid, tpos = "self", None
    elif disp == "V2V 強車":
        tid, tpos = "strong", geo["strong"]
    elif disp == "V2V 近車":
        tid, tpos = "near", geo["near"]
    elif disp == "基站 RSU":
        tid, tpos = "rsu_0", geo["rsu"]
    else:
        tid, tpos = "cloud", geo["rsu"]
    r = estimate(task, kind, 0.0, geo["holder"], nodes,
                 target_id=tid, target_pos=tpos, commit=False)
    if not r["feasible"]:
        return None
    reward = -(r["latency"] + ENERGY_W * r["energy"] / ENERGY_NORM + COST_W * r["cost"])
    return {"latency": r["latency"], "energy": r["energy"],
            "cost": r["cost"], "reward": reward}


def study1():
    print("=" * 78)
    print("[研究1] 典型情境下，各任務類型的五種卸載選擇對照")
    print("  情境：持有車距基站70m；強車鄰居60m、近弱車鄰居40m；佇列皆空")
    print("=" * 78)
    for name, prof in TASKS.items():
        nodes, geo = fresh_world()
        task = make_task(name, prof)
        print(f"\n● {name}  (資料{prof['data_bits']/1e6:.1f}Mb, "
              f"運算{prof['cpu_cycles']/1e9:.2f}Gc, deadline{prof['deadline']}s)")
        print(f"  {'選擇':10}{'延遲ms':>9}{'能耗J':>9}{'成本':>8}{'獎勵':>9}  達標?")
        rows = []
        for disp, kind in OPTIONS:
            res = eval_option(nodes, geo, task, disp, kind)
            rows.append((disp, res))
            if res is None:
                print(f"  {disp:10}{'不可行':>9}")
                continue
            ok = "✓" if res["latency"] <= prof["deadline"] else "✗超時"
            print(f"  {disp:10}{res['latency']*1000:9.0f}{res['energy']:9.3f}"
                  f"{res['cost']:8.3f}{res['reward']:9.2f}   {ok}")
        # 贏家（延遲最低 / 獎勵最高，僅看達標者）
        feas = [(d, r) for d, r in rows if r and r["latency"] <= prof["deadline"]]
        if feas:
            best_lat = min(feas, key=lambda x: x[1]["latency"])[0]
            best_rew = max(feas, key=lambda x: x[1]["reward"])[0]
            print(f"  → 最快：{best_lat}；綜合獎勵最佳：{best_rew}")
        else:
            print("  → 無任何選擇能在 deadline 內完成（此情境過苛）")


def study2():
    print("\n" + "=" * 78)
    print("[研究2] 距離掃描：強車鄰居距離拉遠 → 何時『基站 RSU』反超『V2V 強車』")
    print("  任務：vision(重)；基站固定距持有車70m；強車距離由近到遠")
    print("=" * 78)
    prof = TASKS["vision 影像(重)"]
    print(f"  {'強車距離m':>9}{'V2V強車延遲ms':>15}{'RSU延遲ms':>11}   贏家(延遲)")
    crossover = None
    for sd in (10, 20, 30, 50, 70, 90, 110, 130, 150):
        nodes, geo = fresh_world(strong_d=sd)
        task = make_task("vision", prof)
        v2v = eval_option(nodes, geo, task, "V2V 強車", "v2v")
        rsu = eval_option(nodes, geo, task, "基站 RSU", "rsu")
        if v2v is None:
            vtxt, win = "超出範圍", "RSU"
            vlat = float("inf")
        else:
            vlat = v2v["latency"]
            vtxt = f"{vlat*1000:.0f}"
            win = "V2V強車" if vlat < rsu["latency"] else "RSU"
        if crossover is None and win == "RSU":
            crossover = sd
        print(f"  {sd:9}{vtxt:>15}{rsu['latency']*1000:11.0f}   {win}")
    if crossover:
        print(f"  → 交叉點：強車距離 ≳ {crossover}m 後，改用基站 RSU 較快")
    else:
        print("  → 在可連線範圍內，V2V 強車始終較快（強車算力 12G > RSU 8G）")


def study3():
    print("\n" + "=" * 78)
    print("[研究3] 基站負載掃描：RSU 佇列變長 → 何時『雲端』反超『RSU』")
    print("  任務：nav(中)；持有車距基站70m；對 RSU 預先灌入 N 個排隊任務")
    print("  (用中任務：vision 太重→雲端運算永遠最快無交叉；nav 才看得到翻盤)")
    print("=" * 78)
    prof = TASKS["nav 導航(中)"]
    print(f"  {'RSU前置任務數':>12}{'RSU延遲ms':>11}{'雲端延遲ms':>12}   贏家(延遲)")
    crossover = None
    for nload in range(0, 7):
        nodes, geo = fresh_world()
        # 預先在 RSU 灌入 nload 個 nav 任務（用真實 commit 累積佇列）
        for i in range(nload):
            dummy = make_task("nav", prof)
            estimate(dummy, "rsu", 0.0, geo["holder"], nodes,
                     target_id="rsu_0", target_pos=geo["rsu"], commit=True)
        task = make_task("nav", prof)
        rsu = eval_option(nodes, geo, task, "基站 RSU", "rsu")
        cloud = eval_option(nodes, geo, task, "雲端 cloud", "cloud")
        win = "RSU" if rsu["latency"] < cloud["latency"] else "雲端"
        if crossover is None and win == "雲端":
            crossover = nload
        print(f"  {nload:12}{rsu['latency']*1000:11.0f}{cloud['latency']*1000:12.0f}"
              f"   {win}")
    if crossover is not None:
        print(f"  → 交叉點：RSU 前面排 ≳ {crossover} 個任務後，改送雲端較快"
              f"（雲端無佇列，但有骨幹延遲/壅塞）")
    else:
        print("  → 此範圍內 RSU 始終較快")


def study4():
    print("\n" + "=" * 78)
    print("[研究4] 強車 vs 弱車 V2V：強車快但依 κf² 較耗能")
    print("  任務：vision(重)；兩鄰居同距持有車40m，一強一弱")
    print("=" * 78)
    prof = TASKS["vision 影像(重)"]
    nodes, geo = fresh_world(strong_d=40.0, near_d=40.0)
    task = make_task("vision", prof)
    strong = eval_option(nodes, geo, task, "V2V 強車", "v2v")
    weak = eval_option(nodes, geo, task, "V2V 近車", "v2v")
    print(f"  {'選擇':12}{'延遲ms':>9}{'能耗J':>9}{'達標?':>8}")
    for disp, res in (("V2V 強車(12G)", strong), ("V2V 弱車(1G)", weak)):
        if res is None:
            print(f"  {disp:12}{'不可行':>9}")
            continue
        ok = "✓" if res["latency"] <= prof["deadline"] else "✗超時"
        print(f"  {disp:12}{res['latency']*1000:9.0f}{res['energy']:9.3f}{ok:>8}")
    print("  → 重任務在弱車上算不動(易超時)，強車能救援但能耗較高；")
    print("    輕任務則相反，丟強車是浪費 → agent 要學會分流。")


def make_figures():
    """產生三張論文用圖：距離交叉、負載交叉、延遲-能耗權衡(Pareto)。標籤用英文。"""
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    here = os.path.dirname(os.path.abspath(__file__))
    saved = []

    # --- 圖A：距離交叉 (V2V strong vs RSU)，vision ---
    prof = TASKS["vision 影像(重)"]
    ds = list(range(10, 151, 5))
    v2v_lat, rsu_lat = [], []
    for sd in ds:
        nodes, geo = fresh_world(strong_d=sd)
        t = make_task("vision", prof)
        v = eval_option(nodes, geo, t, "V2V 強車", "v2v")
        r = eval_option(nodes, geo, t, "基站 RSU", "rsu")
        v2v_lat.append(v["latency"] * 1000 if v else np.nan)
        rsu_lat.append(r["latency"] * 1000)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ds, v2v_lat, "-o", ms=3, color="#1565c0", label="V2V (strong helper, 12 GHz)")
    ax.plot(ds, rsu_lat, "--", color="#ef6c00", label="RSU (edge, 8 GHz)")
    ax.set_xlabel("Distance to strong helper vehicle (m)")
    ax.set_ylabel("Task latency (ms)")
    ax.set_title("V2V vs V2I crossover with helper distance (heavy task)")
    ax.grid(alpha=0.3); ax.legend()
    fA = os.path.join(here, "fig_crossover_distance.png")
    plt.tight_layout(); plt.savefig(fA, dpi=150); plt.close(); saved.append(fA)

    # --- 圖B：負載交叉 (RSU vs Cloud)，nav ---
    prof = TASKS["nav 導航(中)"]
    loads = list(range(0, 9))
    rsu_l, cloud_l = [], []
    for nload in loads:
        nodes, geo = fresh_world()
        for _ in range(nload):
            d = make_task("nav", prof)
            estimate(d, "rsu", 0.0, geo["holder"], nodes,
                     target_id="rsu_0", target_pos=geo["rsu"], commit=True)
        t = make_task("nav", prof)
        rsu_l.append(eval_option(nodes, geo, t, "基站 RSU", "rsu")["latency"] * 1000)
        cloud_l.append(eval_option(nodes, geo, t, "雲端 cloud", "cloud")["latency"] * 1000)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(loads, rsu_l, "-o", ms=4, color="#ef6c00", label="RSU (edge, FIFO queue)")
    ax.plot(loads, cloud_l, "--", color="#c62828", label="Cloud (no queue, +backhaul)")
    ax.set_xlabel("Tasks already queued at RSU")
    ax.set_ylabel("Task latency (ms)")
    ax.set_title("RSU vs Cloud crossover with edge congestion (medium task)")
    ax.grid(alpha=0.3); ax.legend()
    fB = os.path.join(here, "fig_crossover_load.png")
    plt.tight_layout(); plt.savefig(fB, dpi=150); plt.close(); saved.append(fB)

    # --- 圖C：延遲-能耗權衡 (vision，五選擇 Pareto) ---
    prof = TASKS["vision 影像(重)"]
    nodes, geo = fresh_world()
    t = make_task("vision", prof)
    labels = {"本地 local": "Local", "V2V 強車": "V2V-strong",
              "V2V 近車": "V2V-weak", "基站 RSU": "RSU", "雲端 cloud": "Cloud"}
    fig, ax = plt.subplots(figsize=(7, 5))
    pts = []
    for disp, kind in OPTIONS:
        res = eval_option(nodes, geo, t, disp, kind)
        if res is None:
            continue
        miss = res["latency"] > prof["deadline"]
        ax.scatter(res["latency"] * 1000, res["energy"], s=90,
                   color=("#b0b0b0" if miss else "#1565c0"),
                   edgecolor="black", zorder=3)
        pts.append((res["latency"] * 1000, res["energy"],
                    labels[disp] + (" (miss)" if miss else "")))
    # 標註錯開:超時的選項(本地/弱車)延遲相近,標籤會疊在一起 → 依 x 排序上下交錯
    for i, (x, y, lab) in enumerate(sorted(pts)):
        ax.annotate(lab, (x, y), textcoords="offset points",
                    xytext=(8, 5 + 13 * (i % 2)), fontsize=9)
    ax.axvline(prof["deadline"] * 1000, color="red", ls=":", alpha=0.7,
               label=f"deadline {prof['deadline']*1000:.0f} ms")
    ax.set_xlabel("Task latency (ms)  — lower is better")
    ax.set_ylabel("Energy per task (J)  — lower is better")
    ax.set_title("Latency–Energy tradeoff (heavy task): lower-left is best")
    ax.grid(alpha=0.3); ax.legend()
    fC = os.path.join(here, "fig_tradeoff_vision.png")
    plt.tight_layout(); plt.savefig(fC, dpi=150); plt.close(); saved.append(fC)

    print("\n已產生圖檔：")
    for s in saved:
        print("  " + s)
    return saved


if __name__ == "__main__":
    import sys
    print("卸載決策分析（純成本模型，不需 SUMO）\n")
    study1()
    study2()
    study3()
    study4()
    if "--plot" in sys.argv:
        make_figures()
    else:
        print("\n(加 --plot 可另存三張論文用圖：距離交叉、負載交叉、延遲-能耗權衡)")
    print("\n" + "=" * 78)
    print("解讀：本分析是『單筆任務、給定情境』的最佳解 oracle。")
    print("MAPPO 要在『多車競爭 + 佇列動態 + 移動性』下逼近這種分流，")
    print("並在全域層面協調(別讓大家同時擠同一個強車/基站)。")
    print("=" * 78)
