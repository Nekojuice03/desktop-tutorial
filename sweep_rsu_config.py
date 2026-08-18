"""
RSU 配置敏感度掃描(sweep_rsu_config.py)—— 邊緣層是不是被建模參數架空了?
============================================================================
2026-08 檢視發現:greedy 只有約 1% 的任務選 RSU —— 一篇談三層卸載的論文,
中間那層等於沒被用到。實測釐清了原因(先前誤以為是佇列建模):

  佇列全空時,一個 vision 任務(4.5 Gcycle)的總延遲
    雲(經 RSU 50m) 271 ms   ← 最快
    強車 30m       457 ms
    RSU 50m        597 ms   ← 光運算就 562 ms
  RSU 在**佇列全空**時就已經輸,所以提高並行度救不了它。
  真正的原因是 RSU_CPU(8 GHz)刻意設得低於強車(12 GHz)
  —— 見 infra_config.py 的註解「故意略低於強車,讓近距離 V2V 有優勢」。

因此本腳本掃兩個軸,分開回答兩件事:

  --cores  並行服務槽數(總容量 = RSU_CORES × RSU_CPU):佇列是不是瓶頸?
  --cpu    每槽算力(GHz):邊緣層要多快才會被策略採用?

判讀:純延遲導向的 greedy 幾乎不用 RSU;而 MAPPO 因為同時看能耗與成本,
會用 RSU 取代雲端。這個對比本身就是論文可寫的結果(貪婪法看不見邊緣層的價值)。

用法:
  python sweep_rsu_config.py --cores 1 2 4 8       # 掃並行度(佇列是不是瓶頸)
  python sweep_rsu_config.py --cpu 8 16 24 32      # 掃單槽算力(GHz)
  python sweep_rsu_config.py --cpu 8 16 24 32 --plot --mappo
  python sweep_rsu_config.py --sumo                # SUMO 場景

產出:rsu_config_results.json、fig_rsu_config.png(--plot)

實測(mock,greedy,每組 3 回合):
  單槽GHz   選RSU%   選雲%   能耗J    成本
      8       1.2    27.7    5.50   0.237
     16      11.8    26.2    5.39   0.233
     24      41.8     1.5    3.24   0.078
     32      47.8     0.0    3.08   0.067
→ RSU 算力跨過約 24 GHz 後,邊緣層取代雲端成為主要卸載對象,
  能耗降 41%、成本降 72%。**「反雲」這個結論對 RSU_CPU 高度敏感。**

⚠ 論文寫法:RSU_CPU / RSU_CORES 是你選定的建模參數,不是量測值。
   請在假設一節說明採用的值與理由,並附上本掃描作為敏感度分析 ——
   否則審稿人會質疑「反雲」只是把邊緣層設弱設出來的。
"""
import argparse
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

import infra_config
import nodes as nodes_mod
from vec_env_ma import VECMultiEnv, MA_ACTIONS, SCRIPT_DIR

OUT_JSON = "rsu_config_results.json"
OUT_FIG = "fig_rsu_config.png"


def set_rsu_cores(n):
    """覆寫 RSU 並行度。

    nodes.py 是 `from infra_config import RSU_CORES` 進來的,改 infra_config
    不會影響已載入的 nodes —— 兩邊都要設,否則掃描等於沒作用。
    """
    infra_config.RSU_CORES = n
    nodes_mod.RSU_CORES = n


def run_episodes(env, algo, episodes, seed0=2000):
    acc = {"success": [], "vision": [], "latency": [], "energy": [], "cost": []}
    by_target = {}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        info = {}
        while True:
            if obs.shape[0] == 0:
                _, obs, _, done, info = env.step(np.zeros(0, dtype=np.int64))
            else:
                actions = (algo.act_greedy(obs, env.action_masks()) if algo
                           else env.greedy_actions())
                _, obs, _, done, info = env.step(actions)
            if done:
                break
        s = info["episode_stats"]
        acc["success"].append(s["success_rate"] * 100)
        acc["vision"].append(s.get("vision_success_rate", 0.0) * 100)
        acc["latency"].append(s["avg_latency_ms"])
        acc["energy"].append(s["avg_energy_j"])
        acc["cost"].append(s.get("avg_cost", 0.0))
        for k, v in s["by_target"].items():
            by_target[k] = by_target.get(k, 0) + v
    tot = sum(by_target.values()) or 1
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["by_target_pct"] = {k: 100.0 * v / tot for k, v in by_target.items()}
    out["decided"] = tot
    return out


def set_rsu_cpu(cpu):
    """覆寫 RSU 每槽算力(同 set_rsu_cores,兩個模組都要設)。"""
    infra_config.RSU_CPU = cpu
    nodes_mod.RSU_CPU = cpu


def plot_results(results, axis_keys, tag, axis_label="cores"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pols = sorted(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"greedy": "#90a4ae", "mappo": "#1565c0"}

    for pol in pols:
        rs = [results[pol][k] for k in axis_keys]
        xs = [float(k) for k in axis_keys]
        axes[0].plot(xs, [r["by_target_pct"].get("rsu", 0.0) for r in rs],
                     "-o", color=colors.get(pol, "#455a64"), label=pol)
        axes[1].plot(xs, [r["success"] for r in rs],
                     "-o", color=colors.get(pol, "#455a64"), label=pol)
    axes[0].set_ylabel("Share of tasks offloaded to RSU (%)")
    axes[0].set_title(f"Is the edge tier usable? ({tag})")
    axes[1].set_ylabel("Task success rate (%)")
    axes[1].set_title(f"Success vs RSU {axis_label}")
    for ax in axes[:2]:
        ax.set_xlabel(f"RSU {axis_label}")
        ax.grid(alpha=0.3); ax.legend()

    # 右:最後一個策略的動作分布堆疊圖(看三層如何重新分配)
    pol = pols[-1]
    kinds = ["local", "v2v_strong", "v2v_near", "rsu", "cloud"]
    bottom = np.zeros(len(axis_keys))
    palette = {"local": "#b0bec5", "v2v_strong": "#2e7d32", "v2v_near": "#00acc1",
               "rsu": "#8e24aa", "cloud": "#c62828"}
    for k in kinds:
        vals = np.array([results[pol][ak]["by_target_pct"].get(k, 0.0)
                         for ak in axis_keys])
        axes[2].bar(axis_keys, vals, bottom=bottom,
                    color=palette[k], label=k)
        bottom += vals
    axes[2].set_xlabel(f"RSU {axis_label}")
    axes[2].set_ylabel("Action distribution (%)")
    axes[2].set_title(f"Where tasks go ({pol})")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fp = os.path.join(SCRIPT_DIR, OUT_FIG)
    plt.savefig(fp, dpi=150); plt.close()
    print(f"已產生 {fp}")


def main():
    p = argparse.ArgumentParser(description="RSU 配置敏感度掃描(並行度 / 單槽算力)")
    p.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="並行服務槽數(總容量 = cores × 單槽算力)")
    p.add_argument("--cpu", type=float, nargs="+",
                   help="改掃每槽算力(GHz),例如 --cpu 4 8 16 24 32;"
                        "指定時 cores 固定為 --cores 的第一個值")
    p.add_argument("--sumo", action="store_true")
    p.add_argument("--mappo", action="store_true", help="另外評估 mappo_vec.pt")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    os.chdir(SCRIPT_DIR)
    if args.sumo:
        cfg = dict(mock=False, arrival_rate=0.3, episode_ticks=300)
        episodes = args.episodes or 5
    else:
        cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24,
                   server_ratio=0.45, episode_ticks=200)
        episodes = args.episodes or 5
    tag = "SUMO" if args.sumo else "mock"

    policies = {"greedy": None}
    if args.mappo:
        mp = os.path.join(SCRIPT_DIR, "mappo_vec.pt")
        if not os.path.exists(mp):
            print("⚠ 找不到 mappo_vec.pt,只跑 greedy")
        else:
            from mappo import MAPPO
            probe = VECMultiEnv(**cfg)
            algo = MAPPO(obs_dim=probe.n_features, state_dim=probe.state_dim,
                         n_actions=probe.n_actions)
            probe.close()
            try:
                algo.load(mp)
                policies["mappo"] = algo
            except Exception as e:
                print(f"⚠ 模型載入失敗({e}) —— 觀測維度改過的話需要重訓。只跑 greedy")

    cpu0 = infra_config.RSU_CPU
    sweep_cpu = args.cpu is not None
    axis = [g * 1e9 for g in args.cpu] if sweep_cpu else args.cores
    fixed_cores = args.cores[0]
    label = "單槽GHz" if sweep_cpu else "cores"
    label_en = "per-slot CPU (GHz)" if sweep_cpu else "parallel service slots"
    print(f"=== RSU 配置敏感度({tag},掃 {label}={args.cpu or args.cores},"
          f"每組 {episodes} 回合) ===")
    if sweep_cpu:
        print(f"  並行槽數固定 {fixed_cores};每槽算力依序覆寫")
    else:
        print(f"  每槽算力固定 {cpu0/1e9:.0f} GHz;總容量 = cores × 單槽算力")

    results = {pol: {} for pol in policies}
    for pol, algo in policies.items():
        print(f"\n── 策略 {pol} ──")
        print(f"  {label:>9}{'總容量GHz':>10}{'成功率%':>9}{'vision%':>9}"
              f"{'延遲ms':>9}{'能耗J':>8}{'成本':>8}{'選RSU%':>9}{'選雲%':>8}")
        for v in axis:
            if sweep_cpu:
                set_rsu_cpu(v); set_rsu_cores(fixed_cores)
                key, shown, cap = f"{v/1e9:.0f}", v / 1e9, fixed_cores * v / 1e9
            else:
                set_rsu_cpu(cpu0); set_rsu_cores(v)
                key, shown, cap = str(v), v, v * cpu0 / 1e9
            env = VECMultiEnv(**cfg)
            r = run_episodes(env, algo, episodes)
            env.close()
            results[pol][key] = r
            print(f"  {shown:>9.0f}{cap:>10.0f}{r['success']:>9.1f}"
                  f"{r['vision']:>9.1f}{r['latency']:>9.0f}{r['energy']:>8.2f}"
                  f"{r['cost']:>8.3f}{r['by_target_pct'].get('rsu',0.0):>9.1f}"
                  f"{r['by_target_pct'].get('cloud',0.0):>8.1f}")
    set_rsu_cores(1); set_rsu_cpu(cpu0)   # 還原,避免污染同一 session 的其他實驗

    axis_keys = [f"{v/1e9:.0f}" for v in axis] if sweep_cpu else [str(v) for v in axis]
    payload = {"scenario": tag, "axis": label, "values": axis_keys,
               "episodes": episodes, "fixed_cores": fixed_cores,
               "rsu_cpu_per_slot": cpu0, "results": results}
    with open(os.path.join(SCRIPT_DIR, OUT_JSON), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n數據已存 {OUT_JSON}")

    base = results[list(policies)[0]]
    rsu_lo = base[axis_keys[0]]["by_target_pct"].get("rsu", 0.0)
    rsu_hi = base[axis_keys[-1]]["by_target_pct"].get("rsu", 0.0)
    print(f"\n判讀:{label}={axis_keys[0]} 時 RSU 佔 {rsu_lo:.1f}%,"
          f"{label}={axis_keys[-1]} 時佔 {rsu_hi:.1f}%")
    if rsu_hi - rsu_lo > 5:
        print(f"  → 邊緣層的使用率對「{label}」敏感 → 這是建模參數在決定結果,")
        print("    論文必須說明所採用的值與理由,並附本掃描作為敏感度分析。")
    else:
        print(f"  → 改變「{label}」對 RSU 佔比幾乎沒有影響 →")
        print("    邊緣層的劣勢不在這個維度,結論對此參數穩健。")

    if args.plot:
        plot_results(results, axis_keys, tag, label_en)


if __name__ == "__main__":
    main()
