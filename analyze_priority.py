"""
優先權實驗分析(analyze_priority.py)—— priority 版的「分級服務是否成立」評估
============================================================================
自動偵測模型並比較:
  - 若 mappo_vec.pt(base，無優先權)與 mappo_prio_vec.pt(有優先權)都在
    → 兩者在『同一組固定種子、同一 SUMO 場景』並排評估，輸出四張對照。
  - 若只有 prio → 單獨拆解它的分類型成功率與動作分布。

四個分析重點(對應論文「差異化 QoS/分級服務」論述)：
  1. 加權成功率 weighted_success_rate：prio ↑ = 高優先任務更常被服務(核心賣點)。
  2. 分類型成功率 sensor/nav/vision：vision/sensor↑、nav 略降 = 「學會讓路」。
  3. 動作分布位移 by_target：高優先更常進 RSU/強車，低優先更常留 local/near。
  4. 弱車是否復活：prio 版 local/v2v_near 佔比從 0% 升起 = 低優先被擠出後的合理去處。

用法：
  python analyze_priority.py            # mock（快速確認腳本正確）
  python analyze_priority.py --sumo     # SUMO 正式(讀 mappo_vec.pt / mappo_prio_vec.pt)
  python analyze_priority.py --sumo --plot
產出：priority_analysis.json、fig_priority_analysis.png(--plot)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from collections import defaultdict

import numpy as np

from vec_env_ma import VECMultiEnv, MA_ACTIONS, SCRIPT_DIR

EVAL_SEEDS = [10_001, 10_002, 10_003, 10_004, 10_005]
KINDS = ["sensor", "nav", "vision"]
ACTION_ORDER = ["local", "v2v_strong", "v2v_near", "rsu", "cloud"]


def eval_model(model_file, priority_aware, base_cfg):
    """
    在固定種子上以 greedy 評估一個模型，聚合『原始計數』(跨種子相加，
    比平均各回合的比率更穩)。回傳分類型成功率、加權成功率、動作分布、總體指標。
    找不到模型 → 回傳 None(呼叫端據此決定單獨/對照模式)。
    """
    from mappo import MAPPO
    mp = os.path.join(SCRIPT_DIR, model_file)
    if not os.path.exists(mp):
        return None
    cfg = dict(base_cfg, priority_aware=priority_aware)
    env = VECMultiEnv(**cfg)
    algo = MAPPO(obs_dim=env.n_features, state_dim=env.state_dim,
                 n_actions=env.n_actions)
    try:
        algo.load(mp)
    except Exception as e:
        env.close()
        print(f"⚠ {model_file} 載入失敗({e}) → 略過"
              f"（多半是 RSU 數不同造成 state_dim 不符，請確認 --sumo 與訓練場景一致）")
        return None

    kind_tot, kind_suc = defaultdict(int), defaultdict(int)
    by_target = defaultdict(int)
    # 逐類型動作分布：某類型任務被派去哪(揭示高/低優先的去處差異)
    kind_action = {k: defaultdict(int) for k in KINDS}
    wsum = wsucc = 0.0
    succ_all, lat_all, en_all, cost_all = [], [], [], []

    for sd in EVAL_SEEDS:
        obs, state = env.reset(seed=sd)
        info = {}
        # 記錄每個決策 tick 的 (動作, 對應任務類型) 需要 env 內部資訊；
        # 這裡用 episode 級聚合 by_target + kind_success_rate(env 已提供)。
        while True:
            if obs.shape[0] == 0:
                _, obs, state, done, info = env.step(np.zeros(0, dtype=np.int64))
                if done:
                    break
                continue
            a = algo.act_greedy(obs)
            _, obs, state, done, info = env.step(a)
            if done:
                break
        s = info["episode_stats"]
        # 原始計數：直接讀 env.stats(episode_summary 只給比率)
        for k, v in env.stats["kind_total"].items():
            kind_tot[k] += v
        for k, v in env.stats["kind_success"].items():
            kind_suc[k] += v
        for k, v in s["by_target"].items():
            by_target[k] += v
        wsum += env.stats["wsum"]
        wsucc += env.stats["wsucc"]
        succ_all.append(s["success_rate"] * 100)
        lat_all.append(s["avg_latency_ms"])
        en_all.append(s["avg_energy_j"])
        cost_all.append(s.get("avg_cost", 0.0))
    env.close()

    tot_dec = sum(by_target.values()) or 1
    return {
        "success_rate": float(np.mean(succ_all)),
        "weighted_success_rate": 100.0 * wsucc / wsum if wsum else 0.0,
        "kind_success_rate": {k: 100.0 * kind_suc[k] / kind_tot[k]
                              for k in KINDS if kind_tot[k]},
        "kind_count": {k: kind_tot[k] for k in KINDS},
        "action_share": {a: 100.0 * by_target.get(a, 0) / tot_dec
                         for a in ACTION_ORDER},
        "latency_ms": float(np.mean(lat_all)),
        "energy_j": float(np.mean(en_all)),
        "cost": float(np.mean(cost_all)),
        "n_decisions": tot_dec,
    }


def _fmt_share(share):
    return "  ".join(f"{a}={share[a]:.1f}%" for a in ACTION_ORDER)


def print_report(base, prio):
    print("\n" + "=" * 70)
    if base and prio:
        print("【對照模式】base MAPPO vs priority MAPPO(同種子、同場景)")
        print("=" * 70)
        print(f"{'指標':<22}{'base':>12}{'priority':>12}{'Δ':>10}")
        rows = [
            ("總成功率 %", base["success_rate"], prio["success_rate"]),
            ("加權成功率 %", base["weighted_success_rate"], prio["weighted_success_rate"]),
            ("平均延遲 ms", base["latency_ms"], prio["latency_ms"]),
            ("平均能耗 J", base["energy_j"], prio["energy_j"]),
            ("平均成本", base["cost"], prio["cost"]),
        ]
        for name, b, p in rows:
            print(f"{name:<22}{b:>12.2f}{p:>12.2f}{p-b:>+10.2f}")
        print(f"\n{'分類型成功率':<14}{'base':>10}{'priority':>12}{'Δ':>10}   (優先權)")
        pr = {"sensor": "1.2-1.8", "nav": "0.8-1.2", "vision": "1.5-2.5"}
        for k in KINDS:
            b = base["kind_success_rate"].get(k, 0.0)
            p = prio["kind_success_rate"].get(k, 0.0)
            print(f"  {k:<12}{b:>10.1f}{p:>12.1f}{p-b:>+10.1f}   [{pr[k]}]")
        print("\n動作分布位移:")
        print(f"  base    : {_fmt_share(base['action_share'])}")
        print(f"  priority: {_fmt_share(prio['action_share'])}")
        # 自動判讀
        print("\n判讀:")
        d_w = prio["weighted_success_rate"] - base["weighted_success_rate"]
        d_vis = (prio["kind_success_rate"].get("vision", 0) -
                 base["kind_success_rate"].get("vision", 0))
        d_nav = (prio["kind_success_rate"].get("nav", 0) -
                 base["kind_success_rate"].get("nav", 0))
        print(f"  • 加權成功率 {'↑' if d_w > 0.5 else '≈' if abs(d_w) <= 0.5 else '↓'} "
              f"{d_w:+.1f} 分 → "
              + ("高優先任務更常被服務,分級服務成立。" if d_w > 0.5
                 else "差異不顯著,可能需拉大優先權差距或提高負載。"))
        if d_vis > 0.5 and d_nav < 0.5:
            print(f"  • vision {d_vis:+.1f}、nav {d_nav:+.1f} → 「高優先↑、低優先讓路」= "
                  "學到差異化 QoS ✔")
        elif d_vis > 0.5:
            print(f"  • vision {d_vis:+.1f} 上升但 nav 未讓路 → 整體變好,分級效果偏弱。")
        else:
            print(f"  • vision {d_vis:+.1f}、nav {d_nav:+.1f} → 分級效果不明顯,見上方建議。")
        wk = prio["action_share"]["local"] + prio["action_share"]["v2v_near"]
        wk0 = base["action_share"]["local"] + base["action_share"]["v2v_near"]
        if wk - wk0 > 2:
            print(f"  • 弱車去處 local+near {wk0:.1f}%→{wk:.1f}% → "
                  "低優先被擠出後弱車復活(一箭雙鵰:優先權有效 + 弱車重獲意義)✔")
        else:
            print(f"  • 弱車去處 local+near {wk0:.1f}%→{wk:.1f}% → 變化不大,"
                  "弱車仍以任務來源/決策者角色為主。")
    else:
        m = prio or base
        tag = "priority MAPPO" if prio else "base MAPPO"
        print(f"【單獨模式】{tag}(另一模型不存在,無法對照)")
        print("=" * 70)
        print(f"總成功率 {m['success_rate']:.2f}%  加權成功率 {m['weighted_success_rate']:.2f}%"
              f"  延遲 {m['latency_ms']:.0f}ms  能耗 {m['energy_j']:.3f}J  成本 {m['cost']:.4f}")
        print("分類型成功率:")
        for k in KINDS:
            print(f"  {k:<8} {m['kind_success_rate'].get(k, 0):.1f}%  "
                  f"(n={m['kind_count'].get(k, 0)})")
        print(f"動作分布: {_fmt_share(m['action_share'])}")
        print("\n（要做 base vs priority 對照,請確認 mappo_vec.pt 與 mappo_prio_vec.pt "
              "都以同一 --sumo 場景訓練後再跑本腳本）")


def make_plot(base, prio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    have_both = bool(base and prio)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左:分類型成功率(base vs priority 分組長條)
    ax = axes[0]
    x = np.arange(len(KINDS))
    if have_both:
        w = 0.38
        ax.bar(x - w/2, [base["kind_success_rate"].get(k, 0) for k in KINDS],
               w, label="base", color="#90a4ae", edgecolor="black")
        ax.bar(x + w/2, [prio["kind_success_rate"].get(k, 0) for k in KINDS],
               w, label="priority", color="#ef6c00", edgecolor="black")
        ax.legend()
    else:
        m = prio or base
        ax.bar(x, [m["kind_success_rate"].get(k, 0) for k in KINDS],
               0.6, color="#ef6c00", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n[{p}]" for k, p in
                        zip(KINDS, ["1.2-1.8", "0.8-1.2", "1.5-2.5"])])
    ax.set_ylabel("Per-type success rate (%)")
    ax.set_title("Differentiated QoS by task priority")
    ax.grid(axis="y", alpha=0.3)

    # 右:動作分布(疊)
    ax2 = axes[1]
    colors = {"local": "#66bb6a", "v2v_strong": "#1565c0", "v2v_near": "#00acc1",
              "rsu": "#ffa726", "cloud": "#ef5350"}
    models = [("base", base), ("priority", prio)] if have_both else \
             [("priority" if prio else "base", prio or base)]
    labels = [n for n, _ in models]
    bottoms = np.zeros(len(models))
    for a in ACTION_ORDER:
        vals = np.array([mm["action_share"][a] for _, mm in models])
        ax2.bar(labels, vals, bottom=bottoms, color=colors[a],
                edgecolor="black", linewidth=0.5, label=a)
        bottoms += vals
    ax2.set_ylabel("Share of decisions (%)")
    ax2.set_title("Action distribution shift")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fp = os.path.join(SCRIPT_DIR, "fig_priority_analysis.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"已產生 {fp}")


def main():
    use_sumo = "--sumo" in sys.argv
    if use_sumo:
        base_cfg = dict(mock=False, gui=False, arrival_rate=0.3,
                        episode_ticks=300, task_cpu_scale=1.0)
    else:
        base_cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24,
                        server_ratio=0.45, episode_ticks=150, task_cpu_scale=1.0)

    print(f"=== 優先權實驗分析({'SUMO' if use_sumo else 'mock'}，"
          f"種子 {EVAL_SEEDS}) ===")
    base = eval_model("mappo_vec.pt", priority_aware=False, base_cfg=base_cfg)
    prio = eval_model("mappo_prio_vec.pt", priority_aware=True, base_cfg=base_cfg)

    if not base and not prio:
        print("找不到任何模型(mappo_vec.pt / mappo_prio_vec.pt)。請先訓練。")
        return

    print_report(base, prio)

    out = os.path.join(SCRIPT_DIR, "priority_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"base": base, "priority": prio}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n數據已存 {out}")

    if "--plot" in sys.argv:
        make_plot(base, prio)


if __name__ == "__main__":
    main()
