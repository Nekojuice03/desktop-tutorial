"""
compare_and_plot.py —— 公平對照所有方法 + 產生報告圖表
========================================================
在「同一個環境、同一組參數」下評估所有方法，確保對照公平：
  全本地 / 純基站 / 純雲端 / 隨機 / 就近最快(貪婪) / MAPPO(本方法)

產出：
  1. 終端機印出對照表
  2. compare_results.json（數據）
  3. fig_success.png      成功率長條圖
  4. fig_latency.png      平均延遲長條圖
  5. fig_action_dist.png  MAPPO 學到的動作分布
  6. fig_convergence.png  訓練收斂曲線(需先跑過 train_mappo 產生 csv)

用法：
  python compare_and_plot.py            # mock 情境(快速，驗證流程)
  python compare_and_plot.py --sumo     # 真實 SUMO(論文數字，需先訓練好 mappo_vec.pt)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vec_env_ma import VECMultiEnv, MA_ACTIONS, SCRIPT_DIR
from mappo import MAPPO

# 評估方法清單：(顯示名稱(英文,給圖表用), mode)
METHODS = [
    ("Local-only", "local"),
    ("RSU-only", "rsu"),
    ("Cloud-only", "cloud"),
    ("Random", "random"),
    ("Greedy", "greedy"),
    ("MAPPO (ours)", "mappo"),
]
# 動作的英文標籤(給圖表)
ACTION_LABELS = {"local": "Local", "v2v_strong": "V2V-strong",
                 "v2v_near": "V2V-near", "rsu": "RSU", "cloud": "Cloud"}


def run_episodes(env, mode, algo=None, episodes=8, seed0=1000):
    succ, lat, ener = [], [], []
    dist = Counter()
    for ep in range(episodes):
        obs, state = env.reset(seed=seed0 + ep)
        info = {"episode_stats": {"success_rate": 0.0, "avg_latency_ms": 0.0,
                                  "avg_energy_j": 0.0, "by_target": {}}}
        guard = 0
        while True:
            guard += 1
            if guard > 100000:
                break
            k = obs.shape[0]
            if k == 0:
                rewards, obs, state, done, info = env.step(np.zeros(0, dtype=np.int64))
                if done:
                    break
                continue
            if mode == "greedy":
                actions = env.greedy_actions()
            elif mode == "random":
                actions = np.random.randint(0, env.n_actions, size=k)
            elif mode == "mappo":
                actions = algo.act_greedy(obs)
            else:
                actions = np.full(k, MA_ACTIONS.index(mode), dtype=np.int64)
            rewards, obs, state, done, info = env.step(actions)
            if done:
                break
        s = info["episode_stats"]
        succ.append(s["success_rate"] * 100)
        lat.append(s["avg_latency_ms"])
        ener.append(s.get("avg_energy_j", 0.0))
        for a, c in s["by_target"].items():
            dist[a] += c
    return {"success_mean": float(np.mean(succ)),
            "success_std": float(np.std(succ)),
            "latency_mean": float(np.mean(lat)),
            "latency_std": float(np.std(lat)),
            "energy_mean": float(np.mean(ener)),
            "energy_std": float(np.std(ener)),
            "dist": dict(dist)}


def main():
    use_sumo = "--sumo" in sys.argv
    episodes = 5 if use_sumo else 10

    if use_sumo:
        cfg = dict(mock=False, arrival_rate=0.3, episode_ticks=300, task_cpu_scale=1.0)
    else:
        cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24, server_ratio=0.45,
                   episode_ticks=150, task_cpu_scale=1.0)

    print(f"=== 公平對照（{'SUMO 真實車流' if use_sumo else 'mock 情境'}，"
          f"每法 {episodes} 回合）===\n")

    # 載入訓練好的 MAPPO
    env0 = VECMultiEnv(**cfg)
    algo = MAPPO(obs_dim=env0.n_features, state_dim=env0.state_dim,
                 n_actions=env0.n_actions)
    model_path = os.path.join(SCRIPT_DIR, "mappo_vec.pt")
    has_model = os.path.exists(model_path)
    if has_model:
        algo.load(model_path)
        print(f"已載入訓練模型 {model_path}\n")
    else:
        print("⚠ 找不到 mappo_vec.pt，MAPPO 將用未訓練權重(請先跑 train_mappo.py)\n")
    env0.close()

    results = {}
    for disp, mode in METHODS:
        if mode == "mappo" and not has_model:
            continue
        env = VECMultiEnv(**cfg)
        r = run_episodes(env, mode, algo=algo, episodes=episodes)
        env.close()
        results[disp] = r
        print(f"  {disp:16s} 成功率 {r['success_mean']:5.1f}% (±{r['success_std']:.1f}) "
              f"延遲 {r['latency_mean']:6.0f}ms  能耗 {r['energy_mean']:.3f}J")

    # 存 JSON
    out_json = os.path.join(SCRIPT_DIR, "compare_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n數據已存 {out_json}")

    names = list(results.keys())
    colors = ["#bdbdbd", "#bdbdbd", "#bdbdbd", "#bdbdbd", "#90caf9", "#1565c0"][:len(names)]
    tag = "SUMO" if use_sumo else "mock"

    # 圖1：成功率
    fig, ax = plt.subplots(figsize=(8, 5))
    sm = [results[n]["success_mean"] for n in names]
    ss = [results[n]["success_std"] for n in names]
    bars = ax.bar(names, sm, yerr=ss, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title(f"Task Success Rate by Method ({tag})")
    ax.set_ylim(0, 100)
    for b, v in zip(bars, sm):
        ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    f1 = os.path.join(SCRIPT_DIR, "fig_success.png")
    plt.savefig(f1, dpi=150); plt.close()

    # 圖2：平均延遲
    fig, ax = plt.subplots(figsize=(8, 5))
    lm = [results[n]["latency_mean"] for n in names]
    ls = [results[n]["latency_std"] for n in names]
    bars = ax.bar(names, lm, yerr=ls, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title(f"Average Task Latency by Method ({tag})  — lower is better")
    for b, v in zip(bars, lm):
        ax.text(b.get_x() + b.get_width()/2, v + max(lm)*0.01, f"{v:.0f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    f2 = os.path.join(SCRIPT_DIR, "fig_latency.png")
    plt.savefig(f2, dpi=150); plt.close()

    saved = [f1, f2]

    # 圖2b：平均能耗
    fig, ax = plt.subplots(figsize=(8, 5))
    em = [results[n]["energy_mean"] for n in names]
    es = [results[n]["energy_std"] for n in names]
    bars = ax.bar(names, em, yerr=es, capsize=4, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Average Energy per Task (J)")
    ax.set_title(f"Average Energy Consumption by Method ({tag})  — lower is better")
    for b, v in zip(bars, em):
        ax.text(b.get_x() + b.get_width()/2, v + max(em)*0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    f2b = os.path.join(SCRIPT_DIR, "fig_energy.png")
    plt.savefig(f2b, dpi=150); plt.close()
    saved.append(f2b)

    # 圖3：MAPPO 動作分布
    if "MAPPO (ours)" in results:
        d = results["MAPPO (ours)"]["dist"]
        order = ["local", "v2v_strong", "v2v_near", "rsu", "cloud"]
        total = sum(d.values()) or 1
        vals = [100 * d.get(k, 0) / total for k in order]
        labels = [ACTION_LABELS[k] for k in order]
        acol = ["#66bb6a", "#1565c0", "#90caf9", "#ffa726", "#ef5350"]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, vals, color=acol, edgecolor="black", linewidth=0.6)
        ax.set_ylabel("Share of Decisions (%)")
        ax.set_title(f"MAPPO Learned Offloading Decision Distribution ({tag})")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
        plt.tight_layout()
        f3 = os.path.join(SCRIPT_DIR, "fig_action_dist.png")
        plt.savefig(f3, dpi=150); plt.close()
        saved.append(f3)

    # 圖4：收斂曲線(若有 csv)：原始線(淡) + 滑動平均(粗)
    csv_path = os.path.join(SCRIPT_DIR, "mappo_train_log.csv")
    if os.path.exists(csv_path):
        its, srs = [], []
        with open(csv_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 2:
                    its.append(int(p[0])); srs.append(float(p[1]))
        if len(its) >= 2:
            its = np.array(its); srs = np.array(srs)
            # 滑動平均(視窗大小依點數自動調整)
            w = max(3, len(srs) // 8)
            if w % 2 == 0:
                w += 1
            if len(srs) >= w:
                kernel = np.ones(w) / w
                smooth = np.convolve(srs, kernel, mode="same")
                # 修正首尾邊界(用累積平均避免邊緣下垂)
                for i in range(w // 2):
                    smooth[i] = srs[:i + w // 2 + 1].mean()
                    smooth[-(i + 1)] = srs[-(i + w // 2 + 1):].mean()
            else:
                smooth = srs
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(its, srs, color="#90caf9", linewidth=1.2, alpha=0.7,
                    label="Raw (per eval)")
            ax.plot(its, smooth, color="#1565c0", linewidth=2.5,
                    label=f"Smoothed (moving avg, w={w})")
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Task Success Rate (%)")
            ax.set_title(f"MAPPO Training Convergence ({tag})")
            ax.grid(alpha=0.3)
            ax.legend(loc="lower right")
            plt.tight_layout()
            f4 = os.path.join(SCRIPT_DIR, "fig_convergence.png")
            plt.savefig(f4, dpi=150); plt.close()
            saved.append(f4)

    print("已產生圖檔：")
    for s in saved:
        print("  " + s)


if __name__ == "__main__":
    main()
