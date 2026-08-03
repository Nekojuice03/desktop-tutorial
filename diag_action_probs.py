"""
動作機率診斷(diag_action_probs.py)—— A：argmax 分布 vs 真實 softmax 機率
============================================================================
問題：train_mappo / analyze_priority 畫的動作分布是用 argmax 統計的。
      若兩動作 logits 接近(如 55%/45%)，argmax 會報成 100%/0%，
      把「差一點被壓成第二名」的動作誤判成「死動作」。

本腳本對每個決策輸出完整 softmax 機率，逐動作比較:
  - argmax_share：被選為第一名的比例(= 分布圖的長條)
  - mean_prob   ：策略實際指派給它的平均機率(真實傾向)
  - max_prob    ：單一決策中它拿到的最高機率(有沒有『差點被選中』)
  - top2_share  ：它排進前二名的比例(是否常是第二選擇)

判讀(對 argmax≈0% 的動作):
  mean_prob 也 ~0、max_prob 也小  → 「真死動作」(策略確實幾乎不考慮它)
  mean_prob 有份量 / max_prob 高  → 「argmax 假象」(其實常是接近的第二名)
→ 直接關係到「弱車 v2v_near 是否真的無意義」的結論可信度。

用法:
  python diag_action_probs.py               # mock
  python diag_action_probs.py --sumo        # SUMO(讀 mappo_vec.pt / mappo_prio_vec.pt)
  python diag_action_probs.py --sumo --plot
產出:action_prob_diag.json、fig_action_prob_diag.png(--plot)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json

import numpy as np

from vec_env_ma import VECMultiEnv, MA_ACTIONS, SCRIPT_DIR

EVAL_SEEDS = [10_001, 10_002, 10_003, 10_004, 10_005]
N = len(MA_ACTIONS)


def diagnose(model_file, priority_aware, base_cfg):
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
        print(f"⚠ {model_file} 載入失敗({e}) → 略過")
        return None

    prob_sum = np.zeros(N)          # 各動作機率總和(→平均)
    prob_max = np.zeros(N)          # 各動作單次最高機率
    argmax_cnt = np.zeros(N)        # 各動作被選為第一名次數
    top2_cnt = np.zeros(N)          # 各動作排進前二名次數
    ent_sum = 0.0                   # 策略熵總和(越低越確定)
    n_dec = 0                       # 總決策數

    for sd in EVAL_SEEDS:
        obs, state = env.reset(seed=sd)
        info = {}
        while True:
            if obs.shape[0] == 0:
                _, obs, state, done, info = env.step(np.zeros(0, dtype=np.int64))
                if done:
                    break
                continue
            probs = algo.act_probs(obs)          # [k, N]
            am = probs.argmax(axis=1)
            # 前二名
            order = np.argsort(-probs, axis=1)
            for i in range(probs.shape[0]):
                prob_sum += probs[i]
                prob_max = np.maximum(prob_max, probs[i])
                argmax_cnt[am[i]] += 1
                top2_cnt[order[i, 0]] += 1
                top2_cnt[order[i, 1]] += 1
                p = probs[i]
                ent_sum += float(-(p * np.log(p + 1e-12)).sum())
                n_dec += 1
            _, obs, state, done, info = env.step(am)   # 走 greedy 軌跡
            if done:
                break
    env.close()
    if n_dec == 0:
        return None
    return {
        "n_decisions": int(n_dec),
        "mean_entropy": ent_sum / n_dec,          # nats；上限 ln(5)=1.609
        "max_entropy": float(np.log(N)),
        "per_action": {
            MA_ACTIONS[a]: {
                "argmax_share": 100.0 * argmax_cnt[a] / n_dec,
                "mean_prob": 100.0 * prob_sum[a] / n_dec,
                "max_prob": 100.0 * prob_max[a],
                "top2_share": 100.0 * top2_cnt[a] / n_dec,
            } for a in range(N)
        },
    }


def verdict(pa):
    """對 argmax≈0 的動作判定『真死』或『argmax 假象』。"""
    lines = []
    for act, d in pa.items():
        if d["argmax_share"] < 1.0:   # 分布圖上看起來是死的
            if d["mean_prob"] < 3.0 and d["max_prob"] < 15.0:
                lines.append(f"  • {act:11}: argmax {d['argmax_share']:.1f}%，"
                             f"平均機率 {d['mean_prob']:.1f}%，最高 {d['max_prob']:.1f}% "
                             f"→ 真死動作(策略確實幾乎不考慮)")
            else:
                lines.append(f"  • {act:11}: argmax {d['argmax_share']:.1f}% 但"
                             f"平均機率 {d['mean_prob']:.1f}%、最高 {d['max_prob']:.1f}%、"
                             f"前二名 {d['top2_share']:.1f}% "
                             f"→ ⚠ argmax 假象(其實常是接近的第二名，不能稱死動作)")
    return lines


def report(name, res):
    print("\n" + "=" * 72)
    print(f"【{name}】決策數 {res['n_decisions']}，"
          f"平均策略熵 {res['mean_entropy']:.3f} / {res['max_entropy']:.3f} nats "
          f"({'接近確定性' if res['mean_entropy'] < 0.5 else '仍有隨機性'})")
    print("=" * 72)
    print(f"{'動作':<12}{'argmax%':>9}{'平均機率%':>11}{'最高機率%':>11}{'前二名%':>10}")
    for act in MA_ACTIONS:
        d = res["per_action"][act]
        print(f"{act:<12}{d['argmax_share']:>9.1f}{d['mean_prob']:>11.1f}"
              f"{d['max_prob']:>11.1f}{d['top2_share']:>10.1f}")
    vs = verdict(res["per_action"])
    if vs:
        print("判讀:")
        print("\n".join(vs))
    else:
        print("判讀:所有動作 argmax 皆有份量，無『疑似死動作』需釐清。")


def make_plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), squeeze=False)
    x = np.arange(N)
    w = 0.38
    for j, (name, res) in enumerate(results):
        ax = axes[0][j]
        am = [res["per_action"][a]["argmax_share"] for a in MA_ACTIONS]
        mp = [res["per_action"][a]["mean_prob"] for a in MA_ACTIONS]
        ax.bar(x - w/2, am, w, label="argmax share", color="#90a4ae",
               edgecolor="black")
        ax.bar(x + w/2, mp, w, label="mean softmax prob", color="#ef6c00",
               edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(MA_ACTIONS, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("%")
        ax.set_title(f"{name}\nargmax vs true probability")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(SCRIPT_DIR, "fig_action_prob_diag.png")
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"\n已產生 {fp}")


def main():
    use_sumo = "--sumo" in sys.argv
    if use_sumo:
        base_cfg = dict(mock=False, gui=False, arrival_rate=0.3,
                        episode_ticks=300, task_cpu_scale=1.0)
    else:
        base_cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24,
                        server_ratio=0.45, episode_ticks=150, task_cpu_scale=1.0)

    print(f"=== 動作機率診斷({'SUMO' if use_sumo else 'mock'}，種子 {EVAL_SEEDS}) ===")
    results = []
    for fn, pa, label in [("mappo_vec.pt", False, "base MAPPO"),
                          ("mappo_prio_vec.pt", True, "priority MAPPO")]:
        res = diagnose(fn, pa, base_cfg)
        if res:
            report(label, res)
            results.append((label, res))
    if not results:
        print("找不到任何模型。請先訓練。")
        return
    out = os.path.join(SCRIPT_DIR, "action_prob_diag.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({name: res for name, res in results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n數據已存 {out}")
    if "--plot" in sys.argv:
        make_plot(results)


if __name__ == "__main__":
    main()
