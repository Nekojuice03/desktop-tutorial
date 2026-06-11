"""
DQN 訓練（train_dqn.py）
========================
用 Stable-Baselines3 的 DQN 訓練卸載策略。

先用 mock 模式確認「學得起來」(成功率隨訓練上升)，
再換 SUMO 模式在真實車流上訓練。

執行：
  python train_dqn.py                 # mock 模式(快、不需SUMO)，先確認會學習
  python train_dqn.py --sumo          # 用真實 SUMO 訓練(慢，需SUMO)
  python train_dqn.py --sumo --gui    # 同上並開視窗

需要：pip install stable-baselines3
"""
import sys
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from vec_env import VECEnv, ACTIONS


def evaluate(model, env, episodes=5):
    """跑幾回合，回傳平均成功率與延遲（不訓練）。"""
    succ, lat, n = [], [], 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        s = info["episode_stats"]
        succ.append(s["success_rate"])
        lat.append(s["avg_latency_ms"])
    return float(np.mean(succ)), float(np.mean(lat))


def main():
    use_sumo = "--sumo" in sys.argv
    gui = "--gui" in sys.argv

    # 訓練環境
    def make_env():
        return Monitor(VECEnv(
            mock=not use_sumo, gui=gui,
            arrival_rate=0.3,          # 製造一點壓力，讓策略有得學
            episode_decisions=300,
            seed=0,
        ))

    env = make_env()
    print(f"訓練環境：{'SUMO' if use_sumo else 'mock'} 模式，動作 {ACTIONS}\n")

    # 訓練前先評估隨機/未訓練策略當對照
    model = DQN(
        "MlpPolicy", env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.95,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        target_update_interval=500,
        verbose=0,
    )

    print("訓練前(未學習)評估：")
    sr0, lat0 = evaluate(model, env, episodes=5)
    print(f"  成功率 {sr0*100:.1f}%，平均延遲 {lat0:.1f}ms\n")

    total = 30_000 if not use_sumo else 50_000
    print(f"開始訓練 {total} 步...（mock 約一兩分鐘）")
    model.learn(total_timesteps=total, progress_bar=False)
    print("訓練完成。\n")

    print("訓練後評估：")
    sr1, lat1 = evaluate(model, env, episodes=5)
    print(f"  成功率 {sr1*100:.1f}%，平均延遲 {lat1:.1f}ms")
    print(f"\n改善：成功率 {sr0*100:.1f}% → {sr1*100:.1f}%，"
          f"延遲 {lat0:.1f} → {lat1:.1f}ms")

    model.save("dqn_vec")
    print("\n模型已存成 dqn_vec.zip")
    env.close()


if __name__ == "__main__":
    main()
