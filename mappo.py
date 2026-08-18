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
  團隊獎勵 = 該 tick 所有 agent 個別獎勵的「平均」(agent 數可變時尺度較穩，
  與 train_mappo.collect 的實作一致)。
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

    def dist(self, obs, masks=None):
        """masks: bool tensor [.., n_actions]，False 的動作被遮成 -inf(不可選)。"""
        logits = self.forward(obs)
        if masks is not None:
            logits = logits.masked_fill(~masks, float("-inf"))
        return Categorical(logits=logits)


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
                 epochs=6, ent_coef=0.02, vf_coef=0.5, device=DEVICE,
                 central_critic=True):
        """
        central_critic：CTDE 消融開關。
          True  = MAPPO：critic 吃「全域狀態」(各RSU佇列/回程/車數/強車數)。
          False = IPPO ：critic 只吃各 agent 的「局部觀測」——沒有任何全域資訊。
        兩者 actor 完全相同 → 效能差距即為「中央 critic(全域信用分配)」的價值。
        """
        self.central = central_critic
        self.actor = Actor(obs_dim, n_actions).to(device)
        self.critic = Critic(state_dim if central_critic else obs_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.ent_coef, self.vf_coef = epochs, ent_coef, vf_coef
        self.device = device
        self.lr0 = lr      # 初始學習率(供線性衰減用)

    def set_lr(self, frac):
        """線性衰減學習率：lr = lr0 × frac(frac 由 1→0)。穩定後期、減少策略震盪(那個 dip)。"""
        lr = self.lr0 * max(0.0, frac)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def _masks_t(self, masks):
        if masks is None:
            return None
        return torch.as_tensor(np.asarray(masks, dtype=bool), device=self.device)

    # ---------- 與環境互動（rollout 時用）----------
    @torch.no_grad()
    def act(self, obs_batch, masks=None):
        """一批觀測 → 取樣動作 + log機率（探索用）。masks=合法動作遮罩。"""
        if len(obs_batch) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        dist = self.actor.dist(obs, self._masks_t(masks))
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy()

    @torch.no_grad()
    def act_greedy(self, obs_batch, masks=None):
        """一批觀測 → 最佳動作（評估用，不探索）。masks=合法動作遮罩。"""
        if len(obs_batch) == 0:
            return np.array([], dtype=np.int64)
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        logits = self.actor.forward(obs)
        mt = self._masks_t(masks)
        if mt is not None:
            logits = logits.masked_fill(~mt, float("-inf"))
        return logits.argmax(-1).cpu().numpy()

    @torch.no_grad()
    def act_probs(self, obs_batch, masks=None):
        """一批觀測 → 完整 softmax 機率矩陣 [k, n_actions]（診斷用）。
        揭示 argmax 分布圖背後的真實策略機率：某動作 argmax=0% 未必機率=0%，
        可能只是永遠差一點被壓成第二名。"""
        if len(obs_batch) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        logits = self.actor.forward(obs)
        mt = self._masks_t(masks)
        if mt is not None:
            logits = logits.masked_fill(~mt, float("-inf"))
        return torch.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def value(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(s).item())

    @torch.no_grad()
    def value_from(self, obs_batch, state):
        """bootstrap 價值：MAPPO 用全域狀態；IPPO 用該 tick 各 agent 局部觀測的平均。"""
        if self.central:
            return self.value(state)
        if obs_batch is None or len(obs_batch) == 0:
            return 0.0
        o = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        return float(self.critic(o).mean().item())

    # ---------- PPO 更新 ----------
    def update(self, ticks, last_value=0.0, reward_mode="team"):
        """
        ticks: list of dict，每個 tick 含
          obs[k,F], actions[k], logprobs[k], state[S], reward(團隊), done(bool)
          選用：masks[k,A](合法動作遮罩)、rewards_i[k](各 agent 個別獎勵)
        last_value: rollout 最後一個 state 的 bootstrap 價值(未結束時)。
        reward_mode:
          "team"       = 既有行為。同一 tick 的所有 agent 共享該 tick 的 advantage。
          "individual" = ★差分獎勵(difference reward)：agent 的 advantage 再加上
                         「自己的獎勵與該 tick 團隊平均的差」。時間結構(GAE)仍走
                         團隊獎勵，但個別功過不再被平均掉。
                         用途:檢驗「IPPO≈MAPPO 是否肇因於團隊獎勵稀釋信用」。
        """
        T = len(ticks)
        if T == 0:
            return {}

        states_np = np.stack([t["state"] for t in ticks])
        rewards = np.array([t["reward"] for t in ticks], dtype=np.float32)
        dones = np.array([1.0 if t["done"] else 0.0 for t in ticks], dtype=np.float32)

        # critic 對各 tick 的價值(算 GAE 用，不需梯度)：
        #   MAPPO=全域狀態；IPPO=該 tick 各 agent 局部觀測 V 的平均(無全域資訊)
        states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.central:
                values_np = self.critic(states_t).cpu().numpy()
            else:
                values_np = np.array([
                    float(self.critic(torch.as_tensor(
                        t["obs"], dtype=torch.float32, device=self.device)).mean())
                    if len(t["obs"]) else 0.0
                    for t in ticks], dtype=np.float32)
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
        if reward_mode == "team":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 標準化，訓練更穩

        # 攤平所有 agent 樣本；team 模式下同 tick 的 agent 共享該 tick 的 advantage
        obs_all, act_all, logp_all, adv_all, ret_all, mask_all = [], [], [], [], [], []
        has_masks = all(tk.get("masks") is not None for tk in ticks if len(tk["actions"]))
        for t, tk in enumerate(ticks):
            k = len(tk["actions"])
            if k == 0:
                continue
            obs_all.append(tk["obs"])
            act_all.append(tk["actions"])
            logp_all.append(tk["logprobs"])
            if reward_mode == "individual" and tk.get("rewards_i") is not None:
                ri = np.asarray(tk["rewards_i"], dtype=np.float32)
                adv_all.append(adv[t] + (ri - ri.mean()))   # 差分獎勵
            else:
                adv_all.append(np.full(k, adv[t], dtype=np.float32))
            ret_all.append(np.full(k, ret[t], dtype=np.float32))
            if has_masks:
                mask_all.append(np.asarray(tk["masks"], dtype=bool))

        obs_t = torch.as_tensor(np.concatenate(obs_all), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.concatenate(act_all), dtype=torch.long, device=self.device)
        oldlogp_t = torch.as_tensor(np.concatenate(logp_all), dtype=torch.float32, device=self.device)
        adv_np = np.concatenate(adv_all)
        if reward_mode != "team":
            # 個別模式：advantage 在「agent 樣本」層級標準化(團隊模式維持原本的
            # tick 層級標準化，行為逐位元不變)
            adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
        adv_t = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        masks_t = (torch.as_tensor(np.concatenate(mask_all), device=self.device)
                   if mask_all else None)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        ret_agent_t = torch.as_tensor(np.concatenate(ret_all), dtype=torch.float32,
                                      device=self.device)

        last = {}
        for _ in range(self.epochs):
            # actor：PPO clip 目標
            dist = self.actor.dist(obs_t, masks_t)
            newlogp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()
            ratio = torch.exp(newlogp - oldlogp_t)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t
            actor_loss = -torch.min(s1, s2).mean()

            # critic：對團隊 return 做回歸
            #   MAPPO：V(全域狀態)；IPPO：V(局部觀測)——同 tick 的 agent 共用該 tick 的 return
            if self.central:
                v = self.critic(states_t)
                critic_loss = ((v - ret_t) ** 2).mean()
            else:
                v = self.critic(obs_t)
                critic_loss = ((v - ret_agent_t) ** 2).mean()

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
