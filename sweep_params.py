"""
參數掃描（sweep_params.py）
============================
目的：找一組參數，讓「卸載確實必要、三層都用得到、沒有單一選項獨大」，
這樣後面的對照實驗與 RL 才有意義。

對每組參數，回報：
  - 純本地/純基站/純雲端的成功率(看是否有單一選項獨大、本地是否就夠)
  - 短 MAPPO 訓練後的成功率 + 動作分布(看 RL 學到什麼、是否用到多層)

判讀好組合的訊號：
  1. 純本地成功率「不要太高」(否則卸載沒必要)
  2. MAPPO 動作分布「分散」(用到 ≥3 種，不是全擠一個)
  3. MAPPO 成功率「贏過所有純策略」

掃描用 mock(快)，找到方向後再套到 SUMO 上微調。
執行：python sweep_params.py
"""
import numpy as np

import infra_config
import nodes
from vec_env_ma import VECMultiEnv
from vec_env import ACTIONS
from mappo import MAPPO
from train_mappo import collect, evaluate

# ---- 掃描網格(可自行增減)----
CLOUD_LATS  = [0.10, 0.30, 0.50]   # 雲端額外延遲(秒)：調大讓雲端不再是免費後援
TASK_SCALES = [1.0, 2.5]           # 任務運算量倍率：調大讓本地不足以應付
MAPPO_ITERS = 10                   # 每組短訓練的更新次數(掃描用，正式再加長)


def set_cloud_latency(x):
    """動態覆寫雲端延遲(同時改 infra_config 與已載入的 nodes 模組)。"""
    infra_config.CLOUD_EXTRA_LATENCY = x
    nodes.CLOUD_EXTRA_LATENCY = x


def make_env(cpu_scale):
    return VECMultiEnv(
        mock=True, arrival_rate=0.4, mock_vehicles=20, server_ratio=0.35,
        rsus={"rsu_0": {"x": 0.0, "y": 0.0}, "rsu_1": {"x": 200.0, "y": 0.0}},
        episode_ticks=150, task_cpu_scale=cpu_scale)


def fixed_success(cpu_scale, action_name, seed=0):
    env = make_env(cpu_scale)
    obs, _ = env.reset(seed=seed)
    info = {}
    aidx = ACTIONS.index(action_name)
    while True:
        _, obs, _, done, info = env.step(np.full(obs.shape[0], aidx))
        if done:
            break
    env.close()
    return info["episode_stats"]["success_rate"] * 100


def train_eval(cpu_scale, iters=MAPPO_ITERS):
    env = make_env(cpu_scale)
    algo = MAPPO(obs_dim=env.n_features, state_dim=env.state_dim,
                 n_actions=env.n_actions)
    carry = env.reset()
    for _ in range(iters):
        ticks, carry, lv = collect(env, algo, 512, carry)
        algo.update(ticks, last_value=lv)
    sr, lat, dist = evaluate(env, algo, episodes=3)
    env.close()
    return sr, lat, dist


def spread_score(dist):
    """用到幾種動作(比例 >= 10% 才算數)，越多代表越分散。"""
    return sum(1 for k in ACTIONS if dist.get(k, 0) >= 10)


def main():
    print("=== 參數掃描：尋找三層分工且 RL 有價值的組合 ===\n")
    rows = []
    for cl in CLOUD_LATS:
        set_cloud_latency(cl)
        for sc in TASK_SCALES:
            loc = fixed_success(sc, "local")
            rsu = fixed_success(sc, "rsu")
            cld = fixed_success(sc, "cloud")
            sr, lat, dist = train_eval(sc)
            best_fixed = max(loc, rsu, cld)
            rows.append({
                "cloud_lat": cl, "task_scale": sc,
                "local": loc, "rsu": rsu, "cloud": cld,
                "mappo": sr, "mappo_lat": lat, "dist": dist,
                "spread": spread_score(dist),
                "beats_fixed": sr > best_fixed + 1,
            })
            print(f"  cloud_lat={cl:.2f} task×{sc:.1f} | "
                  f"純本地{loc:5.1f}% 純基站{rsu:5.1f}% 純雲{cld:5.1f}% | "
                  f"MAPPO {sr:5.1f}%({lat:.0f}ms) 用到{spread_score(dist)}層 {dist}")

    print("\n=== 判讀 ===")
    print(f"{'參數':<22}{'本地夠嗎':<10}{'分散度':<8}{'贏純策略':<8}{'評語'}")
    for r in rows:
        local_ok = "本地不足✓" if r["local"] < 85 else "本地太夠✗"
        spread = f"{r['spread']}層"
        beat = "✓" if r["beats_fixed"] else "✗"
        # 好組合：本地不足 + 用到>=3層 + 贏純策略
        good = (r["local"] < 85) and (r["spread"] >= 3) and r["beats_fixed"]
        verdict = "★推薦" if good else ("可考慮" if r["spread"] >= 2 else "不佳(易獨大)")
        tag = f"cloud{r['cloud_lat']:.1f}/task×{r['task_scale']:.1f}"
        print(f"{tag:<22}{local_ok:<11}{spread:<9}{beat:<9}{verdict}")

    print("\n挑『★推薦』(或分散度最高)的那組，把它的 cloud_lat 寫進 infra_config.py 的")
    print("CLOUD_EXTRA_LATENCY、task 倍率用在訓練/baseline，再到 SUMO 上跑正式實驗。")
    print("(mock 與 SUMO 會有差異，此掃描給方向；最終數字以 SUMO 為準)")


if __name__ == "__main__":
    main()
