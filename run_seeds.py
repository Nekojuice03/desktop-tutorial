"""
多種子訓練統計(run_seeds.py)—— 論文統計嚴謹性
=================================================
同一設定用 N 個不同種子完整訓練，回報各指標的 mean±std。
也用於 LocalCriticPPO vs MAPPO 消融(同種子集各訓一次，比較中央 critic 的價值)。

用法：
  python run_seeds.py                     # MAPPO × 3 seeds(mock)
  python run_seeds.py --local-critic      # 局部 critic 消融 × 3 seeds
  python run_seeds.py --sumo --seeds 3    # SUMO 正式(慢)
  python run_seeds.py --iters 50          # 縮短訓練(快速驗證)
產出：seeds_results.json(含每 seed 明細與彙總)
"""
import argparse
import json
import os
import numpy as np

from vec_env_ma import VECMultiEnv
from mappo import MAPPO
from train_mappo import (collect, evaluate, set_global_seed,
                         ROLLOUT_TICKS, ITERATIONS)
from scenario_config import resolve_scenario, env_scenario_kwargs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS = ("sr", "vis", "lat", "en", "cost")
LABELS = {"sr": "成功率%", "vis": "vision%", "lat": "延遲ms",
          "en": "能耗J", "cost": "成本"}


def train_one(cfg, seed, iters, local_critic):
    set_global_seed(seed)
    env = VECMultiEnv(**cfg, seed=1000 * seed)   # 各 seed 訓練情境不同
    eval_env = VECMultiEnv(**cfg)                # 評估用固定 EVAL_SEEDS → 跨 seed 公平
    algo = MAPPO(obs_dim=env.n_features, state_dim=env.state_dim,
                 n_actions=env.n_actions, central_critic=not local_critic)
    carry = env.reset()
    for it in range(1, iters + 1):
        algo.set_lr(1.0 - (it - 1) / iters)
        ticks, carry, lv = collect(env, algo, ROLLOUT_TICKS, carry)
        algo.update(ticks, last_value=lv)
    m = evaluate(eval_env, algo)
    env.close()
    eval_env.close()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--local-critic", action="store_true",
                   help="shared actor + local critic information ablation")
    p.add_argument("--ippo", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--sumo", action="store_true")
    p.add_argument("--priority", action="store_true")
    p.add_argument("--twin-quality", action="store_true")
    p.add_argument("--scenario", default="heping")
    p.add_argument("--cfg", default=None)
    p.add_argument("--vd-mode", choices=("static", "replay", "live"),
                   default="replay")
    args = p.parse_args()

    if args.sumo:
        cfg = dict(mock=False, arrival_rate=0.3, episode_ticks=300, task_cpu_scale=1.0)
        scenario = resolve_scenario(args.scenario, args.cfg)
        cfg.update(env_scenario_kwargs(scenario, args.vd_mode))
        iters = args.iters or 150
    else:
        scenario = None
        cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24, server_ratio=0.45,
                   episode_ticks=150, task_cpu_scale=1.0)
        iters = args.iters or ITERATIONS
    if args.priority:
        cfg["priority_aware"] = True
    if args.twin_quality:
        cfg["twin_quality_aware"] = True

    local_critic = args.local_critic or args.ippo
    name = "LocalCriticPPO" if local_critic else "MAPPO"
    name += "+priority" if args.priority else ""
    name += "+twin-quality" if args.twin_quality else ""
    print(f"=== 多種子訓練：{name} × {args.seeds} seeds × {iters} iters "
          f"({'SUMO' if args.sumo else 'mock'}) ===")
    runs = []
    for s in range(args.seeds):
        m = train_one(cfg, s, iters, local_critic)
        runs.append(m)
        print(f"  seed {s} | " + " ".join(
            f"{LABELS[k]} {m[k]:.2f}" for k in METRICS) + f" | 分布 {m['dist']}")

    print(f"\n=== {name} 彙總(mean ± std, n={args.seeds}) ===")
    summary = {}
    for k in METRICS:
        vals = [r[k] for r in runs]
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {LABELS[k]:8}: {np.mean(vals):8.2f} ± {np.std(vals):.2f}")

    out = os.path.join(SCRIPT_DIR, "seeds_results.json")
    payload = {"algo": name, "seeds": args.seeds, "iters": iters,
               "sumo": args.sumo, "priority_aware": args.priority,
               "twin_quality_aware": args.twin_quality,
               "scenario": scenario.name if scenario else "mock",
               "vd_mode": args.vd_mode if args.sumo else "off",
               "runs": runs, "summary": summary}
    # 若已有另一演算法的結果，合併保存方便對照
    if os.path.exists(out):
        try:
            old = json.load(open(out, encoding="utf-8"))
            old = old if isinstance(old, list) else [old]
        except Exception:
            old = []
        identity = (name, payload["scenario"], payload["vd_mode"],
                    payload["priority_aware"], payload["twin_quality_aware"])
        old = [o for o in old if (
            o.get("algo"), o.get("scenario", "mock"), o.get("vd_mode", "off"),
            o.get("priority_aware", False), o.get("twin_quality_aware", False)
        ) != identity]
        old.append(payload)
        payload = old
    else:
        payload = [payload]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已存 {out}")
    if isinstance(payload, list) and len(payload) == 2:
        a, b = payload
        print(f"\n=== {a['algo']} vs {b['algo']}(中央 critic 消融) ===")
        for k in METRICS:
            print(f"  {LABELS[k]:8}: {a['summary'][k]['mean']:.2f}±{a['summary'][k]['std']:.2f}"
                  f"  vs  {b['summary'][k]['mean']:.2f}±{b['summary'][k]['std']:.2f}")


if __name__ == "__main__":
    main()
