"""
移動性機制消融實驗（run_ablation.py）
======================================
四組合：predictor ∈ {linear, route} × recovery ∈ {fail, v2i}
在同一組種子/情境下，對每組合分別以 Greedy / Random /（若有模型）MAPPO 評估，
量測：成功率、vision 成功率、延遲、能耗、成本、pred_reject / link_break /
break_recovered / break_failed。

預期敘事(論文)：
  - route 預判把「執行期真實斷線(link_break)」轉成「事前拒絕(pred_reject)」
    → 不浪費已傳輸/已運算的資源(mock 無路線資訊,route 退化為 linear,兩者相同
      —— 這本身是設計驗證；差異要在 SUMO 十字路口才會顯現)
  - v2i 恢復把殘餘斷線救回(break_recovered)，成功率高於 fail

用法：
  python run_ablation.py            # mock(驗證流程/機制方向)
  python run_ablation.py --sumo     # 真實 SUMO(論文數字)
  python run_ablation.py --plot     # 另存 fig_ablation.png
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import numpy as np

from vec_env_ma import VECMultiEnv, SCRIPT_DIR

COMBOS = [("linear", "fail"), ("linear", "v2i"),
          ("route", "fail"), ("route", "v2i")]
EVENT_KEYS = ("pred_reject", "link_break", "break_recovered", "break_failed",
              "consumer_left")


def run_episodes(env, mode, algo=None, episodes=6, seed0=2000):
    succ, vis, lat, ener, cost = [], [], [], [], []
    events = {k: 0 for k in EVENT_KEYS}
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
            elif mode == "random":
                actions = rng.integers(0, env.n_actions, size=k)
            else:  # mappo
                actions = algo.act_greedy(obs)
            _, obs, state, done, info = env.step(actions)
            if done:
                break
        s = info["episode_stats"]
        succ.append(s["success_rate"] * 100)
        vis.append(s.get("vision_success_rate", 0.0) * 100)
        lat.append(s["avg_latency_ms"])
        ener.append(s["avg_energy_j"])
        cost.append(s.get("avg_cost", 0.0))
        for k in EVENT_KEYS:
            events[k] += s.get(k, 0)
    return {"success": float(np.mean(succ)), "success_std": float(np.std(succ)),
            "vision": float(np.mean(vis)), "latency": float(np.mean(lat)),
            "energy": float(np.mean(ener)), "cost": float(np.mean(cost)),
            **{k: events[k] for k in EVENT_KEYS}}


def main():
    use_sumo = "--sumo" in sys.argv
    if use_sumo:
        base_cfg = dict(mock=False, arrival_rate=0.3, episode_ticks=300,
                        task_cpu_scale=1.0)
        episodes = 5
    else:
        base_cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24,
                        server_ratio=0.45, episode_ticks=300, task_cpu_scale=1.0)
        episodes = 6

    # 有訓練好的模型就加入 MAPPO 策略
    policies = ["greedy", "random"]
    algo = None
    model_path = os.path.join(SCRIPT_DIR, "mappo_vec.pt")
    if os.path.exists(model_path):
        from mappo import MAPPO
        probe = VECMultiEnv(**base_cfg)
        algo = MAPPO(obs_dim=probe.n_features, state_dim=probe.state_dim,
                     n_actions=probe.n_actions)
        probe.close()
        try:
            algo.load(model_path)
            policies.append("mappo")
            print(f"已載入 {model_path} → MAPPO 一併評估")
        except Exception as e:
            print(f"⚠ 模型載入失敗({e})，僅評估 greedy/random")
            algo = None

    tag = "SUMO" if use_sumo else "mock"
    print(f"=== 移動性機制消融（{tag}，每組 {episodes} 回合，"
          f"策略：{policies}）===")
    if not use_sumo:
        print("（mock 無路線資訊：route 預判退化為 linear，兩者應相同 —— 屬設計驗證；"
              "預判軸差異需在 SUMO 路口情境）")

    results = {}
    for pol in policies:
        print(f"\n── 策略：{pol} ──")
        print(f"  {'predictor':9}{'recovery':9}{'成功率%':>8}{'vision%':>8}"
              f"{'延遲ms':>8}{'能耗J':>7}{'成本':>7}"
              f"{'預判拒':>7}{'斷線':>5}{'救回':>5}{'損失':>5}{'客離場':>6}")
        for pred, rec in COMBOS:
            env = VECMultiEnv(**base_cfg, predictor=pred, recovery=rec)
            r = run_episodes(env, pol, algo=algo, episodes=episodes)
            env.close()
            results[f"{pol}/{pred}/{rec}"] = r
            print(f"  {pred:9}{rec:9}{r['success']:8.1f}{r['vision']:8.1f}"
                  f"{r['latency']:8.0f}{r['energy']:7.2f}{r['cost']:7.3f}"
                  f"{r['pred_reject']:7d}{r['link_break']:5d}"
                  f"{r['break_recovered']:5d}{r['break_failed']:5d}"
                  f"{r['consumer_left']:6d}")

    out = os.path.join(SCRIPT_DIR, "ablation_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n數據已存 {out}")

    if "--plot" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pol = "mappo" if "mappo" in policies else "greedy"
        labels = [f"{p}\n{r}" for p, r in COMBOS]
        keys = [f"{pol}/{p}/{r}" for p, r in COMBOS]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sr = [results[k]["success"] for k in keys]
        sd = [results[k]["success_std"] for k in keys]
        axes[0].bar(labels, sr, yerr=sd, capsize=4,
                    color=["#b0bec5", "#90caf9", "#64b5f6", "#1565c0"],
                    edgecolor="black", linewidth=0.6)
        axes[0].set_ylabel("Task Success Rate (%)")
        axes[0].set_title(f"Ablation: predictor × recovery ({pol}, {tag})")
        x = np.arange(len(keys)); w = 0.27
        for off, key, lbl, col in ((-w, "pred_reject", "Predicted-reject", "#ffa726"),
                                   (0, "break_recovered", "Break-recovered", "#66bb6a"),
                                   (w, "break_failed", "Break-failed", "#ef5350")):
            axes[1].bar(x + off, [results[k][key] for k in keys], w,
                        label=lbl, color=col, edgecolor="black", linewidth=0.6)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Event count (all episodes)")
        axes[1].set_title("Mobility events by mechanism")
        axes[1].legend()
        plt.tight_layout()
        fp = os.path.join(SCRIPT_DIR, "fig_ablation.png")
        plt.savefig(fp, dpi=150); plt.close()
        print(f"已產生 {fp}")


if __name__ == "__main__":
    main()
