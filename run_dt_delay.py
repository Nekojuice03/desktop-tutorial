"""
數位孿生同步延遲掃描(run_dt_delay.py)—— 回答「你的 DT 和普通模擬差在哪」
==========================================================================
DT 文獻的核心議題是「孿生與物理世界的同步性」。本實驗把孿生收到的車輛
動態(BSM)延遲 τ 秒(VECMultiEnv(obs_delay=τ))：
  - 預判器與 contact 觀測特徵吃 τ 秒前的舊資料
  - 物理結算(真實位置驗證)不受影響 → 舊資料造成的預判失準會變成真實斷線

預期敘事(論文)：
  τ↑ → linear 預判劣化最快(直接外推舊狀態)；
  kalman(EKF-CTRV)可從舊量測「前推」補償部分延遲 → 劣化較慢；
  → 量化 DT 同步性需求 + 證明 KF 預判是 DT 的實質貢獻(非包裝)。

用法：
  python run_dt_delay.py                 # mock(greedy + 已訓練 MAPPO)
  python run_dt_delay.py --sumo --plot   # SUMO 正式 + 出圖
產出：dt_delay_results.json、fig_dt_delay.png(--plot)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json

from vec_env_ma import VECMultiEnv, SCRIPT_DIR
from run_ablation import run_episodes, EVENT_KEYS

TAUS = [0, 1, 2, 4, 8]                 # 孿生延遲(秒)
PREDICTORS = ["linear", "kalman"]      # route 為 oracle(不受 BSM 延遲影響)，不掃


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

    policies = ["greedy"]
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
        except Exception as e:
            print(f"⚠ 模型載入失敗({e})，僅評估 greedy")
            algo = None

    tag = "SUMO" if use_sumo else "mock"
    print(f"=== 數位孿生同步延遲掃描({tag}，τ={TAUS}s，每組 {episodes} 回合) ===")

    results = {}
    for pol in policies:
        for pred in PREDICTORS:
            print(f"\n── 策略 {pol} × 預判 {pred} ──")
            print(f"  {'τ(s)':>5}{'成功率%':>8}{'vision%':>8}{'延遲ms':>8}"
                  f"{'預判拒':>7}{'斷線':>5}{'救回':>5}{'損失':>5}")
            for tau in TAUS:
                env = VECMultiEnv(**base_cfg, predictor=pred, obs_delay=tau)
                r = run_episodes(env, pol, algo=algo, episodes=episodes)
                env.close()
                results[f"{pol}/{pred}/tau{tau}"] = r
                print(f"  {tau:5d}{r['success']:8.1f}{r['vision']:8.1f}"
                      f"{r['latency']:8.0f}{r['pred_reject']:7d}{r['link_break']:5d}"
                      f"{r['break_recovered']:5d}{r['break_failed']:5d}")

    out = os.path.join(SCRIPT_DIR, "dt_delay_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n數據已存 {out}")

    if "--plot" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pol = "mappo" if "mappo" in policies else "greedy"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = {"linear": "#90a4ae", "kalman": "#1565c0"}
        for pred in PREDICTORS:
            sr = [results[f"{pol}/{pred}/tau{t}"]["success"] for t in TAUS]
            lb = [results[f"{pol}/{pred}/tau{t}"]["link_break"] for t in TAUS]
            axes[0].plot(TAUS, sr, "-o", color=colors[pred],
                         label=f"{pred} predictor")
            axes[1].plot(TAUS, lb, "-o", color=colors[pred],
                         label=f"{pred} predictor")
        axes[0].set_xlabel("Digital-twin sync delay τ (s)")
        axes[0].set_ylabel("Task Success Rate (%)")
        axes[0].set_title(f"Success vs twin staleness ({pol}, {tag})")
        axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].set_xlabel("Digital-twin sync delay τ (s)")
        axes[1].set_ylabel("Runtime link breaks (all episodes)")
        axes[1].set_title("Mobility failures vs twin staleness")
        axes[1].grid(alpha=0.3); axes[1].legend()
        plt.tight_layout()
        fp = os.path.join(SCRIPT_DIR, "fig_dt_delay.png")
        plt.savefig(fp, dpi=150); plt.close()
        print(f"已產生 {fp}")


if __name__ == "__main__":
    main()
