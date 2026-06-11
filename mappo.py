"""
MAPPO 演算法（mappo.py）—— Stage B-2
=====================================
CTDE + 團隊獎勵 的多智能體 PPO。

三個核心：
  1. 共享 actor ：局部觀測 → 動作機率。所有 server 車共用同一個(分散執行)。
  2. 中央 critic ：全域狀態 → 團隊價值 V。只在訓練時用(CTDE 的 C)。
  3. PPO 更新 ：用團隊獎勵序列 + critic 算 GAE advantage，clip 更新。

資料流(配合 vec_env_ma 的批次介面)：
  每個 tick 收集：k 個 agent 的 (obs, action, logprob) + 全域 state + 團隊獎勵。
  團隊獎勵 = 該 tick 所有 agent 個別獎勵的總和。
  GAE 在「tick 序列」上算 → 同一 tick 的所有 agent 共享該 tick 的 advantage。

純演算法，可單獨測試(直接執行會用假資料測前向+更新)。
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Actor(nn.Module):
    """局部觀測 → 動作 logits（分散執行的策略）。"""
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)

    def dist(self, obs):
        return Categorical(logits=self.forward(obs))


class Critic(nn.Module):
    """全域狀態 → 團隊價值 V（CTDE 的中央 critic）。"""
    def __init__(self, state_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


class MAPPO:
    def __init__(self, obs_dim, state_dim, n_actions,
                 lr=3e-4, gamma=0.95, lam=0.95, clip=0.2,
                 epochs=6, ent_coef=0.02, vf_coef=0.5, device=DEVICE):
        self.actor = Actor(obs_dim, n_actions).to(device)
        self.critic = Critic(state_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.ent_coef, self.vf_coef = epochs, ent_coef, vf_coef
        self.device = device

    # ---------- 與環境互動（rollout 時用）----------
    @torch.no_grad()
    def act(self, obs_batch):
        """一批觀測 → 取樣動作 + log機率（探索用）。"""
        if len(obs_batch) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        dist = self.actor.dist(obs)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy()

    @torch.no_grad()
    def act_greedy(self, obs_batch):
        """一批觀測 → 最佳動作（評估用，不探索）。"""
        if len(obs_batch) == 0:
            return np.array([], dtype=np.int64)
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        return self.actor.forward(obs).argmax(-1).cpu().numpy()

    @torch.no_grad()
    def value(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(s).item())

    # ---------- PPO 更新 ----------
    def update(self, ticks, last_value=0.0):
        """
        ticks: list of dict，每個 tick 含
          obs[k,F], actions[k], logprobs[k], state[S], reward(團隊), done(bool)
        last_value: rollout 最後一個 state 的 bootstrap 價值(未結束時)。
        """
        T = len(ticks)
        if T == 0:
            return {}

        states_np = np.stack([t["state"] for t in ticks])
        rewards = np.array([t["reward"] for t in ticks], dtype=np.float32)
        dones = np.array([1.0 if t["done"] else 0.0 for t in ticks], dtype=np.float32)

        # critic 對各 tick 全域狀態的價值(算 GAE 用，不需梯度)
        with torch.no_grad():
            states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
            values_np = self.critic(states_t).cpu().numpy()
        values = np.append(values_np, np.float32(last_value))

        # GAE：在 tick 序列上算 advantage 與 return
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterm - values[t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t] = gae
        ret = adv + values[:T]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 標準化，訓練更穩

        # 攤平所有 agent 樣本；同 tick 的 agent 共享該 tick 的 advantage
        obs_all, act_all, logp_all, adv_all = [], [], [], []
        for t, tk in enumerate(ticks):
            k = len(tk["actions"])
            if k == 0:
                continue
            obs_all.append(tk["obs"])
            act_all.append(tk["actions"])
            logp_all.append(tk["logprobs"])
            adv_all.append(np.full(k, adv[t], dtype=np.float32))

        obs_t = torch.as_tensor(np.concatenate(obs_all), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.concatenate(act_all), dtype=torch.long, device=self.device)
        oldlogp_t = torch.as_tensor(np.concatenate(logp_all), dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(np.concatenate(adv_all), dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        last = {}
        for _ in range(self.epochs):
            # actor：PPO clip 目標
            dist = self.actor.dist(obs_t)
            newlogp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(newlogp - oldlogp_t)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            actor_loss = -torch.min(s1, s2).mean()

            # critic：對團隊 return 做回歸
            v = self.critic(states_t)
            critic_loss = ((v - ret_t) ** 2).mean()

            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.opt.step()
            last = {"actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy": entropy.item()}
        return last

    def save(self, path="mappo_vec.pt"):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path="mappo_vec.pt"):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])


# ==================================================================
# 自我測試：用假資料測網路前向 + 一次更新跑得動
# ==================================================================
if __name__ == "__main__":
    print("=== MAPPO 自我測試（假資料）===\n")
    OBS_DIM, STATE_DIM, N_ACT = 11, 6, 4
    algo = MAPPO(OBS_DIM, STATE_DIM, N_ACT)
    print(f"裝置：{DEVICE}")
    print(f"actor 參數量 {sum(p.numel() for p in algo.actor.parameters())}，"
          f"critic 參數量 {sum(p.numel() for p in algo.critic.parameters())}\n")

    # 測 act：一批 5 個 agent 的觀測
    obs_batch = np.random.rand(5, OBS_DIM).astype(np.float32)
    actions, logps = algo.act(obs_batch)
    print(f"[1] act(): 5 個 agent → 動作 {actions}，log機率 {np.round(logps,2)}")
    print(f"    greedy 動作：{algo.act_greedy(obs_batch)}")
    print(f"    空 batch 不報錯：{algo.act(np.zeros((0,OBS_DIM),np.float32))[0].shape}")

    # 造一段假 rollout（每 tick 隨機數量 agent）
    rng = np.random.default_rng(0)
    ticks = []
    for t in range(40):
        k = rng.integers(1, 6)
        obs = rng.random((k, OBS_DIM)).astype(np.float32)
        a, lp = algo.act(obs)
        ticks.append({"obs": obs, "actions": a, "logprobs": lp,
                      "state": rng.random(STATE_DIM).astype(np.float32),
                      "reward": float(rng.normal(-1, 0.5)),   # 假團隊獎勵
                      "done": (t == 39)})

    print(f"\n[2] 假 rollout：{len(ticks)} 個 tick，"
          f"共 {sum(len(t['actions']) for t in ticks)} 個 agent 樣本")
    before = algo.value(ticks[0]["state"])
    info = algo.update(ticks, last_value=0.0)
    after = algo.value(ticks[0]["state"])
    print(f"[3] 一次 update(): {info}")
    print(f"    critic 對同一狀態的估值 {before:.3f} → {after:.3f}（有變化代表有在學）")

    assert np.isfinite(info["actor_loss"]) and np.isfinite(info["critic_loss"]), "loss 非有限！"
    print("\n=== 測試結束：網路前向、取樣、GAE、PPO 更新全部正常 ===")
