"""
MAPPO 訓練（train_mappo.py）—— Stage B-2
========================================
用自寫的 MAPPO 訓練多智能體卸載策略(團隊獎勵 + CTDE)。

先用 mock 壓力情境(單一基站 + 高密度)確認：
  - 成功率隨訓練上升
  - 「全擠基站」的比例下降(代表 agent 學會錯開 → 協調)
再換 SUMO 在真實車流上訓練。

執行：
  python train_mappo.py          # mock，快，先確認會學習
  python train_mappo.py --sumo   # 真實 SUMO
  python train_mappo.py --sumo --gui
"""
import os
import sys
from collections import Counter
import numpy as np

from vec_env_ma import VECMultiEnv, MA_ACTIONS as ACTIONS, SCRIPT_DIR
from mappo import MAPPO

ROLLOUT_TICKS = 512   # 每次更新前收集幾個 tick
ITERATIONS = 200      # 更新幾次(輪數多→後期在高點附近飽和，呈上升→平台趨勢)
EVAL_EVERY = 3        # 每幾次更新評估一次(越密集→收斂圖的點越多→曲線越平滑)
# 固定的評估種子集：每次評估都用同一組情境，收斂曲線才穩定可重現(不再每次抽不同隨機局)
EVAL_SEEDS = [10_001, 10_002, 10_003, 10_004, 10_005]


def collect(env, algo, n_ticks, carry):
    """收集一段 rollout（跨回合邊界，done 的 tick 會標記；含空值防護）。"""
    obs, state = carry
    ticks = []
    steps = 0
    while steps < n_ticks:
        if obs.shape[0] == 0:   # 退化(空)觀測 → 重置再來
            obs, state = env.reset()
            continue
        actions, logps = algo.act(obs)
        rewards, next_obs, next_state, done, info = env.step(actions)
        team_r = float(rewards.mean()) if rewards.size else 0.0
        ticks.append({"obs": obs, "actions": actions, "logprobs": logps,
                      "state": state,
                      "reward": team_r,   # 團隊獎勵=個別平均(agent數可變時較穩)
                      "done": done})
        steps += 1
        if done:
            obs, state = env.reset()
        else:
            obs, state = next_obs, next_state
    last_value = 0.0 if ticks[-1]["done"] else algo.value(state)
    return ticks, (obs, state), last_value


def evaluate(eval_env, algo, seeds=EVAL_SEEDS):
    """
    用 greedy 策略在『固定種子』的獨立環境上評估：成功率、延遲、能耗、動作分布。
    用獨立 eval_env + 固定 seeds → 每次評估面對相同情境，收斂曲線不再亂跳(修正訓練曲線異常)。
    """
    succ, lat, ener = [], [], []
    act_count = Counter()
    for sd in seeds:
        obs, state = eval_env.reset(seed=sd)
        info = {}
        while True:
            if obs.shape[0] == 0:
                # 該情境已無可決策任務 → 直接取結算
                _, obs, state, done, info = eval_env.step(np.zeros(0, dtype=np.int64))
                if done:
                    break
                continue
            a = algo.act_greedy(obs)
            for x in a:
                act_count[ACTIONS[int(x)]] += 1
            rewards, obs, state, done, info = eval_env.step(a)
            if done:
                break
        s = info["episode_stats"]
        succ.append(s["success_rate"])
        lat.append(s["avg_latency_ms"])
        ener.append(s["avg_energy_j"])
    total = sum(act_count.values())
    dist = {k: round(act_count[k] / total * 100) for k in ACTIONS} if total else {}
    return (float(np.mean(succ)) * 100, float(np.mean(lat)),
            float(np.mean(ener)), dist)


def main():
    use_sumo = "--sumo" in sys.argv
    gui = "--gui" in sys.argv

    if use_sumo:
        cfg = dict(mock=False, gui=gui, arrival_rate=0.3, episode_ticks=300,
                   task_cpu_scale=1.0)
    else:
        # mock 異質算力情境：任務已在 TASK_PROFILES 裡分輕(sensor)/重(vision)，
        # 高密度 + 強車 35% → 逼出 V2V(找強車) 與多層分工
        cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24, server_ratio=0.45,
                   episode_ticks=150, task_cpu_scale=1.0)

    env = VECMultiEnv(**cfg)
    # 獨立的評估環境(同設定，但種子由 EVAL_SEEDS 固定，與訓練環境互不干擾)
    eval_env = VECMultiEnv(**cfg)
    algo = MAPPO(obs_dim=env.n_features, state_dim=env.state_dim,
                 n_actions=env.n_actions)
    print(f"訓練環境：{'SUMO' if use_sumo else 'mock 壓力情境'}，"
          f"觀測{env.n_features} 全域狀態{env.state_dim} 動作{ACTIONS}\n")

    print("訓練前(未學習)評估：")
    sr0, lat0, en0, d0 = evaluate(eval_env, algo)
    print(f"  成功率 {sr0:.1f}%，延遲 {lat0:.0f}ms，能耗 {en0:.3f}J，動作分布 {d0}\n")

    # 收斂曲線記錄(供 compare_and_plot 畫圖)
    log_path = os.path.join(SCRIPT_DIR, "mappo_train_log.csv")
    log_rows = [(0, sr0, lat0, en0)]   # iter 0 = 訓練前

    carry = env.reset()
    iters = ITERATIONS if not use_sumo else 150   # SUMO 慢但輪數足夠才會收斂到平台
    for it in range(1, iters + 1):
        ticks, carry, last_v = collect(env, algo, ROLLOUT_TICKS, carry)
        info = algo.update(ticks, last_value=last_v)
        if it % EVAL_EVERY == 0 or it == iters:
            sr, lat, en, dist = evaluate(eval_env, algo)
            log_rows.append((it, sr, lat, en))
            print(f"  iter {it:>3} | 成功率 {sr:5.1f}% 延遲 {lat:6.0f}ms 能耗 {en:.3f}J "
                  f"分布 {dist} | actor_loss {info['actor_loss']:.3f} "
                  f"entropy {info['entropy']:.2f}")

    print("\n訓練後評估：")
    sr1, lat1, en1, d1 = evaluate(eval_env, algo)
    print(f"  成功率 {sr1:.1f}%，延遲 {lat1:.0f}ms，能耗 {en1:.3f}J，動作分布 {d1}")
    print(f"\n改善：成功率 {sr0:.1f}% → {sr1:.1f}%，延遲 {lat0:.0f} → {lat1:.0f}ms")
    print(f"動作分布變化：{d0} → {d1}")
    print("（若基站/雲比例從偏高下降、本地/V2V 上升且成功率上升，代表學到分層分工）")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("iter,success_rate,avg_latency_ms,avg_energy_j\n")
        for it, sr, lat, en in log_rows:
            f.write(f"{it},{sr:.2f},{lat:.1f},{en:.4f}\n")
    print(f"收斂紀錄已存成 mappo_train_log.csv")

    algo.save("mappo_vec.pt")
    print("\n模型已存成 mappo_vec.pt")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
