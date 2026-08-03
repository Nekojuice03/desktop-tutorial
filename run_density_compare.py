"""
車流密度對比(run_density_compare.py)—— 車少 vs 車多 對卸載策略的影響
========================================================================
與 run_load_sweep 不同軸：
  - run_load_sweep 變的是「每台車發任務的頻率」(任務負載)。
  - 本腳本變的是「路上車輛密度」(用真實 VD 尖峰/離峰車流)——直接影響
    V2V 幫手供給(強車/鄰車在不在範圍)與 RSU/回程爭用。

用同一個訓練好的策略(不重訓)在低/高車流兩組場景評估，量測並對比：
  • 密度指標  ：平均在場車數、平均強車數
  • 各層供給  ：決策當下『強車/鄰車/RSU 在範圍內』的比例(取自觀測，與模型無關)
  • 使用結果  ：動作分布、成功率、延遲、能耗、成本、不可行率

實驗設計建議(你已收集 2~3 天資料)：
  低車流 = 各天『離峰』時段快照(如 03:00~05:00)
  高車流 = 各天『尖峰』時段快照(如 08:00 或 18:00)
  每個密度等級放多天的路由檔 → 天與天之間的變異即為自然重複(誤差棒)。

前置：先用 make_real_flow.py --xml <該時段VD快照> --out <檔名> 產生各路由檔。

用法：
  python run_density_compare.py --sumo \
      --low  real_traffic_low_d1.rou.xml,real_traffic_low_d2.rou.xml \
      --high real_traffic_high_d1.rou.xml,real_traffic_high_d2.rou.xml \
      --seeds 3 --plot
  # 也可直接給自製 .sumocfg(偵測副檔名)：--low scen_low.sumocfg
產出：density_compare.json、fig_density_compare.png(--plot)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import sys
import json
import argparse

import numpy as np

from vec_env_ma import VECMultiEnv, MA_ACTIONS, SCRIPT_DIR

ACTION_ORDER = ["local", "v2v_strong", "v2v_near", "rsu", "cloud"]
BASE_CFG = "osm.sumocfg"
# 觀測欄位索引(見 vec_env_ma._build_obs)：供給訊號來自觀測本身
OBS_RSU_IN, OBS_SG_IN, OBS_NR_IN = 6, 9, 13


def make_cfg_for_route(route_file, label, idx):
    """依 BASE_CFG 產一份只換 route-files 的臨時 sumocfg(寫在專案目錄，
    確保 net/additional 相對路徑仍有效)。回傳 (cfg_path, is_temp)。"""
    if route_file.endswith(".sumocfg"):
        return route_file, False        # 使用者自備場景，直接用
    base_path = os.path.join(SCRIPT_DIR, BASE_CFG)
    with open(base_path, encoding="utf-8") as f:
        text = f.read()
    new_text, n = re.subn(r'(<route-files\s+value=")[^"]*(")',
                          rf'\g<1>{route_file}\g<2>', text)
    if n == 0:
        raise RuntimeError(f"{BASE_CFG} 找不到 <route-files> 可替換")
    tmp = os.path.join(SCRIPT_DIR, f"_dens_{label}_{idx}.sumocfg")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new_text)
    return tmp, True


def eval_scenario(cfg_path, algo, seeds, arrival_rate, episode_ticks):
    """單一場景(單一路由檔)跑 len(seeds) 回合，回傳聚合指標。"""
    env = VECMultiEnv(mock=False, gui=False, cfg=os.path.basename(cfg_path),
                      arrival_rate=arrival_rate, episode_ticks=episode_ticks,
                      task_cpu_scale=1.0)
    veh_sum = strong_sum = tick_n = 0.0
    sg_in = nr_in = rsu_in = dec_n = 0.0
    succ, lat, ener, cost, infe = [], [], [], [], []
    by_target = {a: 0 for a in ACTION_ORDER}
    for sd in seeds:
        obs, state = env.reset(seed=sd)
        info = {}
        while True:
            # 密度取樣：目前在場車數 / 強車數(reset 後、每次 step 後皆更新)
            veh_sum += len(env.veh_states)
            strong_sum += sum(1 for v in env.veh_states if v in env.strong)
            tick_n += 1
            if obs.shape[0] == 0:
                _, obs, state, done, info = env.step(np.zeros(0, dtype=np.int64))
                if done:
                    break
                continue
            # 各層供給：決策當下強車/鄰車/RSU 在範圍內的比例(觀測欄位)
            sg_in += float(obs[:, OBS_SG_IN].sum())
            nr_in += float(obs[:, OBS_NR_IN].sum())
            rsu_in += float(obs[:, OBS_RSU_IN].sum())
            dec_n += obs.shape[0]
            a = algo.act_greedy(obs)
            _, obs, state, done, info = env.step(a)
            if done:
                break
        s = info["episode_stats"]
        succ.append(s["success_rate"] * 100)
        lat.append(s["avg_latency_ms"])
        ener.append(s["avg_energy_j"])
        cost.append(s.get("avg_cost", 0.0))
        gen = max(s["generated"], 1)
        infe.append(100.0 * s["infeasible"] / gen)
        for a_, c in s["by_target"].items():
            by_target[a_] = by_target.get(a_, 0) + c
    env.close()
    tot = sum(by_target.values()) or 1
    dd = max(dec_n, 1)
    return {
        "avg_vehicles": veh_sum / max(tick_n, 1),
        "avg_strong": strong_sum / max(tick_n, 1),
        "supply_strong_pct": 100.0 * sg_in / dd,
        "supply_near_pct": 100.0 * nr_in / dd,
        "supply_rsu_pct": 100.0 * rsu_in / dd,
        "success": float(np.mean(succ)),
        "latency": float(np.mean(lat)),
        "energy": float(np.mean(ener)),
        "cost": float(np.mean(cost)),
        "infeasible_pct": float(np.mean(infe)),
        "share": {a: 100.0 * by_target.get(a, 0) / tot for a in ACTION_ORDER},
    }


def aggregate(per_file):
    """把同一密度等級的多個路由檔(多天)聚合成 mean±std。"""
    keys_scalar = ["avg_vehicles", "avg_strong", "supply_strong_pct",
                   "supply_near_pct", "supply_rsu_pct", "success", "latency",
                   "energy", "cost", "infeasible_pct"]
    out = {}
    for k in keys_scalar:
        vals = [r[k] for r in per_file]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    out["share"] = {a: float(np.mean([r["share"][a] for r in per_file]))
                    for a in ACTION_ORDER}
    out["n_files"] = len(per_file)
    return out


def run_level(files, label, algo, n_seeds, arrival_rate, episode_ticks):
    seeds = [4000 + i for i in range(n_seeds)]
    per_file, temps = [], []
    for i, rf in enumerate(files):
        cfg, is_temp = make_cfg_for_route(rf, label, i)
        if is_temp:
            temps.append(cfg)
        print(f"  [{label}] {rf} …", flush=True)
        try:
            per_file.append(eval_scenario(cfg, algo, seeds,
                                          arrival_rate, episode_ticks))
        finally:
            if is_temp and os.path.exists(cfg):
                os.remove(cfg)
    return aggregate(per_file)


def interpret(low, high):
    print("\n判讀(密度效應):")
    d_sg = high["supply_strong_pct"]["mean"] - low["supply_strong_pct"]["mean"]
    d_v2v = ((high["share"]["v2v_strong"] + high["share"]["v2v_near"]) -
             (low["share"]["v2v_strong"] + low["share"]["v2v_near"]))
    d_lat = high["latency"]["mean"] - low["latency"]["mean"]
    d_inf = high["infeasible_pct"]["mean"] - low["infeasible_pct"]["mean"]
    d_cloud = high["share"]["cloud"] - low["share"]["cloud"]
    print(f"  • 強車供給 {low['supply_strong_pct']['mean']:.1f}%→"
          f"{high['supply_strong_pct']['mean']:.1f}% ({d_sg:+.1f}) → "
          + ("車多時 V2V 幫手變多" if d_sg > 2 else "供給差異不大"))
    print(f"  • V2V 使用率 {(low['share']['v2v_strong']+low['share']['v2v_near']):.1f}%→"
          f"{(high['share']['v2v_strong']+high['share']['v2v_near']):.1f}% ({d_v2v:+.1f}) → "
          + ("車多→更靠 V2V 分流(協同運算價值浮現)" if d_v2v > 2
             else "車多→更靠 RSU/雲(V2V 未被更多利用)" if d_v2v < -2
             else "使用結構穩定"))
    print(f"  • 延遲 {low['latency']['mean']:.0f}→{high['latency']['mean']:.0f}ms "
          f"({d_lat:+.0f})、不可行率 {low['infeasible_pct']['mean']:.1f}%→"
          f"{high['infeasible_pct']['mean']:.1f}% ({d_inf:+.1f}) → "
          + ("車多→RSU/回程爭用惡化" if d_lat > 50 or d_inf > 2
             else "車多下策略仍穩健(密度強健性)"))
    if d_cloud > 2:
        print(f"  • 雲端 {low['share']['cloud']:.1f}%→{high['share']['cloud']:.1f}% "
              f"({d_cloud:+.1f}) → 高密度壅塞時雲端作為壓力閥被啟用")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumo", action="store_true")
    ap.add_argument("--low", default="", help="低車流路由檔(逗號分隔多天)")
    ap.add_argument("--high", default="", help="高車流路由檔(逗號分隔多天)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arrival", type=float, default=0.3, help="固定任務到達率(隔離密度變因)")
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--model", default="mappo_vec.pt")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if not args.sumo:
        print("密度對比需要真實 VD 車流 → 請加 --sumo，並用 --low/--high 指定路由檔。")
        print("(mock 無真實車輛密度概念，此實驗只在 SUMO 有意義)")
        return
    low_files = [x for x in args.low.split(",") if x]
    high_files = [x for x in args.high.split(",") if x]
    if not low_files or not high_files:
        print("請提供 --low 與 --high 路由檔，例如：")
        print("  --low real_traffic_low_d1.rou.xml,real_traffic_low_d2.rou.xml \\")
        print("  --high real_traffic_high_d1.rou.xml,real_traffic_high_d2.rou.xml")
        return

    from mappo import MAPPO
    probe = VECMultiEnv(mock=False, gui=False, cfg=BASE_CFG,
                        arrival_rate=args.arrival, episode_ticks=5)
    algo = MAPPO(obs_dim=probe.n_features, state_dim=probe.state_dim,
                 n_actions=probe.n_actions)
    probe.close()
    mp = os.path.join(SCRIPT_DIR, args.model)
    if not os.path.exists(mp):
        print(f"找不到模型 {args.model}，請先訓練(python train_mappo.py --sumo)。")
        return
    algo.load(mp)
    print(f"=== 車流密度對比(策略={args.model}，每檔 {args.seeds} 種子，"
          f"任務到達率固定 {args.arrival}) ===")

    print("低車流:")
    low = run_level(low_files, "low", algo, args.seeds, args.arrival, args.ticks)
    print("高車流:")
    high = run_level(high_files, "high", algo, args.seeds, args.arrival, args.ticks)

    # 表格
    print("\n" + "=" * 70)
    print(f"{'指標':<20}{'低車流':>16}{'高車流':>16}")
    print("=" * 70)
    def row(name, k, unit="", scale=1.0):
        lo, hi = low[k], high[k]
        print(f"{name:<20}{lo['mean']*scale:>11.1f}±{lo['std']*scale:<4.1f}"
              f"{hi['mean']*scale:>11.1f}±{hi['std']*scale:<4.1f}  {unit}")
    row("平均在場車數", "avg_vehicles")
    row("平均強車數", "avg_strong")
    row("強車供給%", "supply_strong_pct")
    row("鄰車供給%", "supply_near_pct")
    row("RSU供給%", "supply_rsu_pct")
    row("成功率%", "success")
    row("延遲ms", "latency")
    row("能耗J", "energy")
    row("成本", "cost")
    row("不可行率%", "infeasible_pct")
    print("\n動作分布:")
    print(f"  低車流: " + "  ".join(f"{a}={low['share'][a]:.1f}%" for a in ACTION_ORDER))
    print(f"  高車流: " + "  ".join(f"{a}={high['share'][a]:.1f}%" for a in ACTION_ORDER))
    interpret(low, high)

    out = os.path.join(SCRIPT_DIR, "density_compare.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"low": low, "high": high,
                   "config": {"low_files": low_files, "high_files": high_files,
                              "seeds": args.seeds, "arrival": args.arrival}},
                  f, ensure_ascii=False, indent=2)
    print(f"\n數據已存 {out}")

    if args.plot:
        make_plot(low, high)


def make_plot(low, high):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # 左：動作分布(低 vs 高，疊圖)
    ax = axes[0]
    colors = {"local": "#66bb6a", "v2v_strong": "#1565c0", "v2v_near": "#00acc1",
              "rsu": "#ffa726", "cloud": "#ef5350"}
    labels = ["low\n(車少)", "high\n(車多)"]
    bottoms = np.zeros(2)
    for a in ACTION_ORDER:
        vals = np.array([low["share"][a], high["share"][a]])
        ax.bar(labels, vals, bottom=bottoms, color=colors[a],
               edgecolor="black", linewidth=0.5, label=a)
        bottoms += vals
    ax.set_ylabel("Share of decisions (%)")
    ax.set_title("Offloading tier usage: low vs high traffic")
    ax.legend(fontsize=8)
    # 右：各層供給 + 密度(分組長條，含誤差棒)
    ax2 = axes[1]
    metrics = [("strong\nsupply", "supply_strong_pct"),
               ("near\nsupply", "supply_near_pct"),
               ("RSU\nsupply", "supply_rsu_pct")]
    x = np.arange(len(metrics))
    w = 0.38
    ax2.bar(x - w/2, [low[k]["mean"] for _, k in metrics], w,
            yerr=[low[k]["std"] for _, k in metrics], capsize=4,
            label="low", color="#90caf9", edgecolor="black")
    ax2.bar(x + w/2, [high[k]["mean"] for _, k in metrics], w,
            yerr=[high[k]["std"] for _, k in metrics], capsize=4,
            label="high", color="#1565c0", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m for m, _ in metrics])
    ax2.set_ylabel("In-range availability at decision (%)")
    ax2.set_title("V2V/RSU supply vs traffic density")
    ax2.legend()
    plt.tight_layout()
    fp = os.path.join(SCRIPT_DIR, "fig_density_compare.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"已產生 {fp}")


if __name__ == "__main__":
    main()
