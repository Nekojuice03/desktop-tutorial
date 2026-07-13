"""
負載掃描(run_load_sweep.py)—— 回答「雲端/弱車動作是死權重還是備援」
======================================================================
拿「已訓練好的策略」(不重訓)在不同任務到達率下評估，觀察：
  - 動作分布如何隨負載移動(RSU 塞了之後雲端會不會被啟用？
    強車搶不到之後 local/v2v_near 會不會出現？)
  - 成功率/延遲/能耗/成本隨負載的劣化曲線(策略的負載強健性)

論文用途：
  - 若 cloud/near 在高負載被啟用 → 它們是「壓力閥/備援」而非死權重，
    低負載時 0% 是合理的按需行為。
  - 若始終為 0 → 誠實寫入 limitation(動作被支配/觀測缺 nr_wait)。

用法：
  python run_load_sweep.py                # mock
  python run_load_sweep.py --sumo --plot  # SUMO 正式(用 mappo_vec.pt)
產出：load_sweep_results.json、fig_load_sweep.png(--plot)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from collections import Counter

import numpy as np

from vec_env_ma import VECMultiEnv, MA_ACTIONS, SCRIPT_DIR

RATES = [0.15, 0.3, 0.5, 0.8, 1.2]   # 任務到達率(每 client 每秒)
EPISODES = 4
ACTION_ORDER = ["local", "v2v_strong", "v2v_near", "rsu", "cloud"]


def run_at_rate(env, mode, algo, episodes, seed0=3000):
    succ, lat, ener, cost = [], [], [], []
    dist = Counter()
    for ep in range(episodes):
        rng = np.random.default_rng(seed0 + ep)
        obs, state = env.reset(seed=seed0 + ep)
        info = {}
        while True:
            k = obs.shape[0]
            if k == 0:
                _, obs, state, done, info = env.step(np.zeros(0, dtype=np.int64))
                if done:
                    break
                continue
            if mode == "greedy":
                actions = env.greedy_actions()
            else:
                actions = algo.act_greedy(obs, env.current_action_masks())
            _, obs, state, done, info = env.step(actions)
            if done:
                break
        s = info["episode_stats"]
        succ.append(s["success_rate"] * 100)
        lat.append(s["avg_latency_ms"])
        ener.append(s["avg_energy_j"])
        cost.append(s.get("avg_cost", 0.0))
        for a, c in s["by_target"].items():
            dist[a] += c
    total = sum(dist.values()) or 1
    share = {a: 100.0 * dist.get(a, 0) / total for a in ACTION_ORDER}
    return {"success": float(np.mean(succ)), "latency": float(np.mean(lat)),
            "energy": float(np.mean(ener)), "cost": float(np.mean(cost)),
            "share": share, "n_decisions": total}


def main():
    use_sumo = "--sumo" in sys.argv
    priority = "--priority" in sys.argv
    twin_quality = "--twin-quality" in sys.argv
    if use_sumo:
        base = dict(mock=False, episode_ticks=300, task_cpu_scale=1.0)
    else:
        base = dict(mock=True, mock_vehicles=24, server_ratio=0.45,
                    episode_ticks=300, task_cpu_scale=1.0)
    if priority:
        base["priority_aware"] = True
    if twin_quality:
        base["twin_quality_aware"] = True
    artifact_suffix = ("_prio" if priority else "") + ("_tq" if twin_quality else "")

    mode = "greedy"
    algo = None
    mp = os.path.join(SCRIPT_DIR, f"mappo{artifact_suffix}_vec.pt")
    if os.path.exists(mp):
        from mappo import MAPPO
        probe = VECMultiEnv(**base, arrival_rate=0.3)
        algo = MAPPO(obs_dim=probe.n_features, state_dim=probe.state_dim,
                     n_actions=probe.n_actions)
        probe.close()
        try:
            algo.load(mp)
            mode = "mappo"
            print("使用已訓練 MAPPO(不重訓，純評估其負載強健性)")
        except Exception as e:
            print(f"⚠ 模型載入失敗({e}) → 改用 greedy")
            algo = None
    tag = "SUMO" if use_sumo else "mock"
    print(f"=== 負載掃描({tag}，策略={mode}，arrival={RATES}，每點 {EPISODES} 回合) ===")
    print(f"  {'rate':>5}{'成功率%':>8}{'延遲ms':>8}{'能耗J':>7}{'成本':>7}  "
          + "".join(f"{a:>11}" for a in ACTION_ORDER))

    results = {}
    for rate in RATES:
        env = VECMultiEnv(**base, arrival_rate=rate)
        r = run_at_rate(env, mode, algo, EPISODES)
        env.close()
        results[str(rate)] = r
        print(f"  {rate:5.2f}{r['success']:8.1f}{r['latency']:8.0f}"
              f"{r['energy']:7.2f}{r['cost']:7.3f}  "
              + "".join(f"{r['share'][a]:10.1f}%" for a in ACTION_ORDER))

    out = os.path.join(SCRIPT_DIR, "load_sweep_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n數據已存 {out}")
    # 一句話判讀
    hi = results[str(RATES[-1])]["share"]
    lo = results[str(RATES[0])]["share"]
    for a in ("cloud", "v2v_near", "local"):
        trend = hi[a] - lo[a]
        if trend > 2:
            print(f"  → {a}: 低負載 {lo[a]:.1f}% → 高負載 {hi[a]:.1f}% —— 壓力閥/備援，非死權重")
        elif hi[a] < 1 and lo[a] < 1:
            print(f"  → {a}: 全負載皆 ~0% —— 死動作(被支配或觀測不足)，寫入 limitation")

    if "--plot" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = {"local": "#66bb6a", "v2v_strong": "#1565c0",
                  "v2v_near": "#00acc1", "rsu": "#ffa726", "cloud": "#ef5350"}
        bottoms = np.zeros(len(RATES))
        for a in ACTION_ORDER:
            vals = [results[str(r)]["share"][a] for r in RATES]
            axes[0].bar([str(r) for r in RATES], vals, bottom=bottoms,
                        color=colors[a], edgecolor="black", linewidth=0.5, label=a)
            bottoms += np.array(vals)
        axes[0].set_xlabel("Task arrival rate (tasks/s per client)")
        axes[0].set_ylabel("Share of decisions (%)")
        axes[0].set_title(f"Action distribution vs load ({mode}, {tag})")
        axes[0].legend(fontsize=8)
        ax2 = axes[1]
        ax2.plot([str(r) for r in RATES],
                 [results[str(r)]["success"] for r in RATES],
                 "-o", color="#1565c0", label="success rate (%)")
        ax2.set_ylabel("Success rate (%)", color="#1565c0")
        ax2.set_xlabel("Task arrival rate (tasks/s per client)")
        ax3 = ax2.twinx()
        ax3.plot([str(r) for r in RATES],
                 [results[str(r)]["latency"] for r in RATES],
                 "-s", color="#ef5350", label="latency (ms)")
        ax3.set_ylabel("Avg latency (ms)", color="#ef5350")
        ax2.set_title("Robustness under load")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        fp = os.path.join(SCRIPT_DIR, "fig_load_sweep.png")
        plt.savefig(fp, dpi=150); plt.close()
        print(f"已產生 {fp}")


if __name__ == "__main__":
    main()
