"""
數位孿生可視化(visualize_dt.py)—— 把孿生狀態渲染成動畫
==========================================================
回答「這個研究的 DT 可做可視化嗎」：可以。本腳本重放一個 episode，
把孿生的完整狀態畫成動畫 GIF(論文 demo / 口試展示用)：

  - 車輛：client(灰點)、弱 server(藍點)、強 server(橘星)
  - RSU：覆蓋圈，顏色隨佇列負載變深；路口角=方塊、路肩=圓
  - 卸載決策：決策瞬間畫「持有車→目標」的連線(藍=V2V、橘=RSU、紅=雲)
  - 在途 V2V 任務:虛線(等完成時刻結算)
  - ★孿生延遲殘影(obs_delay=τ>0 時)：空心「殘影」=孿生看到的舊位置，
    實心=物理真實位置 —— 一張圖看懂「孿生 vs 物理世界」的差距
  - HUD：tick / 成功率 / 斷線與救回統計

用法：
  python visualize_dt.py                    # mock、τ=2(展示殘影)、120 tick
  python visualize_dt.py --tau 0            # 完美同步
  python visualize_dt.py --sumo --ticks 200 # SUMO 場景(畫真實路網底圖)
產出：dt_visualization.gif、dt_snapshot.png
"""
import argparse
import io
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vec_env_ma import VECMultiEnv, MA_ACTIONS, SCRIPT_DIR
from infra_config import RSU_RANGE_M

KIND_COLOR = {"v2v_strong": "#1565c0", "v2v_near": "#64b5f6",
              "rsu": "#ef6c00", "cloud": "#c62828"}


def record_episode(env, algo, ticks_limit):
    """跑一個 episode，逐 tick 錄下孿生狀態快照。"""
    frames = []
    obs, state = env.reset(seed=0)
    while True:
        if obs.shape[0] == 0:
            _, obs, state, done, info = env.step(np.zeros(0, dtype=np.int64))
            if done or env.tick_count >= ticks_limit:
                break
            continue
        # 決策(有模型用模型，否則 greedy)
        actions = algo.act_greedy(obs) if algo else env.greedy_actions()
        # 決策連線：持有車 → 目標
        decisions = []
        for (sid, ctx), a in zip(env._active, actions):
            kind = MA_ACTIONS[int(a)]
            tpos = {"v2v_strong": ctx["strong_pos"], "v2v_near": ctx["near_pos"],
                    "rsu": ctx["rsu_pos"], "cloud": ctx["rsu_pos"]}.get(kind)
            if tpos is not None:
                decisions.append((ctx["holder_pos"], tpos, kind))
        _, obs, state, done, info = env.step(actions)

        s = env.stats
        done_n = s["success"] + s["fail"]
        frames.append({
            "tick": env.tick_count,
            "veh": {v: st["pos"] for v, st in env.veh_states.items()},
            "twin": {v: st["pos"] for v, st in env.twin_states.items()},
            "roles": dict(env.roles), "strong": set(env.strong),
            "rsu_wait": {r: env.nodes.wait_time(r, env.now) for r in env.rsu_ids},
            "decisions": decisions,
            "inflight": [(f["holder"], f["helper"]) for f in env._inflight],
            "hud": (f"tick {env.tick_count}   success "
                    f"{(s['success']/done_n*100 if done_n else 0):.1f}%   "
                    f"breaks {s['link_break']} (recovered {s['break_recovered']})   "
                    f"pred-reject {s['pred_reject']}"),
        })
        if done or env.tick_count >= ticks_limit:
            break
    return frames


def draw_frame(ax, fr, env, net_shapes, tau):
    ax.clear()
    for sh in net_shapes:   # 路網底圖(SUMO 模式)
        ax.plot([p[0] for p in sh], [p[1] for p in sh], color="#e0e0e0",
                lw=2, zorder=0)
    # RSU 覆蓋圈(負載越高越深)
    for rid in env.rsu_ids:
        r = env.rsus[rid]
        load = min(fr["rsu_wait"].get(rid, 0.0) / 2.0, 1.0)
        ax.add_patch(plt.Circle((r["x"], r["y"]), RSU_RANGE_M,
                                color="#42a5f5", alpha=0.06 + 0.15 * load, zorder=1))
        mark = "s" if r.get("tag") == "corner" else "o"
        ax.scatter(r["x"], r["y"], s=110, marker=mark, color="#0d47a1",
                   edgecolor="black", zorder=5)
    # 孿生殘影(舊位置)
    if tau > 0:
        gx = [p[0] for p in fr["twin"].values()]
        gy = [p[1] for p in fr["twin"].values()]
        ax.scatter(gx, gy, s=26, facecolors="none", edgecolors="#9e9e9e",
                   linewidths=1.0, zorder=2, label=f"twin view (τ={tau}s stale)")
    # 車輛(物理真實位置)
    for grp, mark, col, size, lbl in (
            ("client", "o", "#bdbdbd", 18, "client"),
            ("weak", "o", "#1e88e5", 34, "server (weak)"),
            ("strong", "*", "#ff8f00", 120, "server (strong)")):
        xs, ys = [], []
        for vid, pos in fr["veh"].items():
            role = fr["roles"].get(vid)
            g = ("strong" if vid in fr["strong"] else "weak") \
                if role == "server" else "client"
            if g == grp:
                xs.append(pos[0]); ys.append(pos[1])
        ax.scatter(xs, ys, s=size, marker=mark, color=col,
                   edgecolor="black", linewidths=0.4, zorder=4, label=lbl)
    # 在途 V2V(虛線)
    for h, e in fr["inflight"]:
        if h in fr["veh"] and e in fr["veh"]:
            ax.plot([fr["veh"][h][0], fr["veh"][e][0]],
                    [fr["veh"][h][1], fr["veh"][e][1]],
                    ls="--", lw=0.8, color="#7e57c2", zorder=3)
    # 本 tick 的卸載決策連線
    for hp, tp, kind in fr["decisions"]:
        ax.plot([hp[0], tp[0]], [hp[1], tp[1]], lw=1.8,
                color=KIND_COLOR[kind], alpha=0.9, zorder=6)
    ax.set_title("Digital-Twin View — offloading decisions in real time")
    ax.text(0.01, 0.01, fr["hud"], transform=ax.transAxes, fontsize=9,
            family="monospace", va="bottom",
            bbox=dict(fc="white", alpha=0.8, ec="none"))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sumo", action="store_true")
    p.add_argument("--tau", type=int, default=2, help="孿生同步延遲(秒)，畫殘影用")
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    if args.sumo:
        cfg = dict(mock=False, arrival_rate=0.3, episode_ticks=args.ticks + 5)
    else:
        cfg = dict(mock=True, arrival_rate=0.5, mock_vehicles=24, server_ratio=0.45,
                   episode_ticks=args.ticks + 5, task_cpu_scale=1.0)
    env = VECMultiEnv(**cfg, obs_delay=args.tau)

    algo = None
    mp = os.path.join(SCRIPT_DIR, "mappo_vec.pt")
    if os.path.exists(mp):
        from mappo import MAPPO
        algo = MAPPO(obs_dim=env.n_features, state_dim=env.state_dim,
                     n_actions=env.n_actions)
        try:
            algo.load(mp)
            print("使用已訓練 MAPPO 策略")
        except Exception:
            algo = None
    if algo is None:
        print("無可用模型 → 使用 greedy 策略展示")

    # SUMO 模式畫路網底圖：★從 osm.sumocfg 讀 net-file，
    # 避免資料夾裡有多個 *.net.xml 時抓錯底圖(車輛與底圖不同路網)
    net_shapes = []
    if args.sumo:
        try:
            import sumolib
            import xml.etree.ElementTree as ET
            net_file = None
            cfg_path = os.path.join(SCRIPT_DIR, "osm.sumocfg")
            if os.path.exists(cfg_path):
                nf = ET.parse(cfg_path).getroot().find(".//net-file")
                if nf is not None:
                    net_file = os.path.join(SCRIPT_DIR, nf.get("value"))
            if net_file is None or not os.path.isfile(net_file):
                import glob
                hits = [f for f in glob.glob(os.path.join(SCRIPT_DIR, "*.net.xml"))
                        if os.path.isfile(f)]
                net_file = hits[0] if hits else None
            if net_file:
                print(f"底圖路網：{os.path.basename(net_file)}")
                net = sumolib.net.readNet(net_file)
                net_shapes = [e.getShape() for e in net.getEdges()
                              if not e.getID().startswith(":")]
        except Exception:
            pass

    print(f"錄製 episode({'SUMO' if args.sumo else 'mock'}，τ={args.tau}s，"
          f"{args.ticks} ticks)...")
    frames = record_episode(env, algo, args.ticks)
    env2 = env   # draw 需要 rsus/rsu_ids
    print(f"共 {len(frames)} 個 frame，渲染 GIF...")

    fig, ax = plt.subplots(figsize=(8, 7))
    images = []
    for fr in frames:
        draw_frame(ax, fr, env2, net_shapes, args.tau)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        buf.seek(0)
        images.append(Image.open(buf).convert("P"))
    plt.close(fig)

    gif = os.path.join(SCRIPT_DIR, "dt_visualization.gif")
    images[0].save(gif, save_all=True, append_images=images[1:],
                   duration=int(1000 / args.fps), loop=0)
    # 另存一張中段快照(論文靜態圖)
    snap = os.path.join(SCRIPT_DIR, "dt_snapshot.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_frame(ax, frames[len(frames) // 2], env2, net_shapes, args.tau)
    fig.savefig(snap, dpi=150, bbox_inches="tight")
    plt.close(fig)
    env.close()
    print(f"已產生 {gif}(動畫)與 {snap}(快照)")


if __name__ == "__main__":
    main()
