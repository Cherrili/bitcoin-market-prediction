"""
RL Component — Deep Q-Network (DQN) for Bitcoin market-state prediction.

The agent maps on-chain feature vectors to trading positions:
    Action 0 → Bear  (short / exit)
    Action 1 → Sideways (hold / flat)
    Action 2 → Bull  (long / enter)

Reward = position × realized_return  (PnL-shaped, not label-matching)

This is fundamentally different from the supervised classifiers: the
objective is maximising cumulative portfolio return rather than matching
a discrete label, so the agent can learn asymmetric risk preferences.

Public API
----------
train_dqn(X_train, returns_train, *, n_episodes, verbose) -> DQNAgent
predict_dqn(agent, X_test)                                -> np.ndarray ({0,1,2})
evaluate_dqn(agent, X_test, y_test_enc, returns_test, output_dir) -> dict
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Environment ───────────────────────────────────────────────────────────────

class BitcoinRLEnv:
    """
    Lightweight MDP over a pre-computed feature matrix.

    Each step presents the current on-chain observation; the agent picks
    a position; the step reward is position × actual 30-day return.
    An episode runs the full time-series once (no randomisation — the
    data are already time-ordered, and we must not look ahead).
    """

    N_ACTIONS = 3   # 0=bear, 1=sideways, 2=bull

    def __init__(self, X: np.ndarray, returns: np.ndarray) -> None:
        assert len(X) == len(returns), "X and returns must have equal length"
        self.X       = X.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.n       = len(X)
        self._step   = 0

    # ── gym-like interface ────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self._step = 0
        return self.X[0]

    def step(self, action: int):
        position = action - 1          # {-1, 0, +1}
        reward   = float(position * self.returns[self._step])

        self._step += 1
        done       = self._step >= self.n - 1
        next_obs   = self.X[self._step] if not done else self.X[-1]
        return next_obs, reward, done

    @property
    def obs_dim(self) -> int:
        return self.X.shape[1]


# ── Neural network ─────────────────────────────────────────────────────────────

class _DQNet(nn.Module):
    """3-layer MLP Q-network with layer normalisation."""

    def __init__(self, state_dim: int, action_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class _ReplayBuffer:
    def __init__(self, capacity: int = 20_000) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done) -> None:
        self._buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        return random.sample(self._buf, n)

    def __len__(self) -> int:
        return len(self._buf)


# ── Agent ─────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Double-DQN agent with experience replay and a soft target network.

    Parameters
    ----------
    state_dim       : dimensionality of the observation vector
    action_dim      : number of discrete actions (3)
    lr              : Adam learning rate
    gamma           : discount factor
    eps_start/end   : epsilon-greedy schedule bounds
    eps_decay       : multiplicative decay per training step
    batch_size      : mini-batch size for gradient updates
    target_update   : steps between target-network hard updates
    buffer_capacity : maximum transitions stored
    """

    def __init__(
        self,
        state_dim:        int,
        action_dim:       int   = 3,
        lr:               float = 3e-4,
        gamma:            float = 0.95,
        eps_start:        float = 1.0,
        eps_end:          float = 0.05,
        eps_decay:        float = 0.997,
        batch_size:       int   = 64,
        target_update:    int   = 200,
        buffer_capacity:  int   = 20_000,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the DQN agent. "
                              "Install it with: pip install torch")

        self.action_dim    = action_dim
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update

        self.q_net      = _DQNet(state_dim, action_dim)
        self.target_net = _DQNet(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr,
                                    weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=500, gamma=0.5
        )
        self.memory    = _ReplayBuffer(buffer_capacity)

        self.epsilon       = eps_start
        self._eps_end      = eps_end
        self._eps_decay    = eps_decay
        self._global_steps = 0

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(torch.FloatTensor(state).unsqueeze(0))
        self.q_net.train()
        return int(q.argmax(dim=1).item())

    # ── one gradient step ─────────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        batch   = self.memory.sample(self.batch_size)
        s, a, r, s2, done = zip(*batch)

        states      = torch.FloatTensor(np.array(s))
        actions     = torch.LongTensor(a).unsqueeze(1)
        rewards     = torch.FloatTensor(r)
        next_states = torch.FloatTensor(np.array(s2))
        dones       = torch.FloatTensor([float(d) for d in done])

        # Double-DQN: online net selects action, target net scores it
        q_vals   = self.q_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1, keepdim=True)
            q_next       = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target       = rewards + self.gamma * q_next * (1 - dones)

        loss = F.smooth_l1_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self._global_steps += 1

        # Hard target-network update
        if self._global_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay
        self.epsilon = max(self._eps_end, self.epsilon * self._eps_decay)
        self.scheduler.step()

        return loss.item()


# ── Training loop ─────────────────────────────────────────────────────────────

def train_dqn(
    X_train:        np.ndarray,
    returns_train:  np.ndarray,
    *,
    n_episodes:     int   = 60,
    verbose:        bool  = True,
) -> tuple[DQNAgent, list]:
    """
    Train a DQN agent on the training split.

    Parameters
    ----------
    X_train       : (n, d) feature matrix — same Top-40 features as the
                    supervised models, already StandardScaler-transformed.
    returns_train : (n,)  actual 30-day forward returns (floats, not labels).
    n_episodes    : number of full-sequence episodes to train for.

    Returns
    -------
    agent          : trained DQNAgent
    rewards_hist   : list of total episode rewards (for plotting)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")

    env   = BitcoinRLEnv(X_train, returns_train)
    agent = DQNAgent(state_dim=env.obs_dim)

    rewards_hist: list[float] = []

    for ep in range(1, n_episodes + 1):
        obs          = env.reset()
        total_reward = 0.0
        losses: list[float] = []

        while True:
            action          = agent.select_action(obs)
            next_obs, rew, done = env.step(action)
            agent.memory.push(obs, action, rew, next_obs, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            total_reward += rew
            obs = next_obs
            if done:
                break

        rewards_hist.append(total_reward)
        if verbose and (ep % 10 == 0 or ep == 1):
            avg_loss = np.mean(losses) if losses else float("nan")
            print(f"   DQN ep {ep:3d}/{n_episodes}  "
                  f"reward={total_reward:+.4f}  "
                  f"ε={agent.epsilon:.3f}  "
                  f"loss={avg_loss:.5f}")

    return agent, rewards_hist


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_dqn(agent: DQNAgent, X_test: np.ndarray) -> np.ndarray:
    """
    Return encoded predictions {0=bear, 1=sideways, 2=bull} for every row.
    """
    agent.q_net.eval()
    preds: list[int] = []
    with torch.no_grad():
        for state in X_test.astype(np.float32):
            q = agent.q_net(torch.FloatTensor(state).unsqueeze(0))
            preds.append(int(q.argmax(dim=1).item()))
    return np.array(preds, dtype=int)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_dqn(
    agent:        DQNAgent,
    X_test:       np.ndarray,
    y_test_enc:   np.ndarray,   # {0,1,2}
    returns_test: np.ndarray,
    rewards_hist: list,
    output_dir:   str,
) -> dict:
    """
    Compute classification metrics and cumulative-return curves; save plots.

    Returns a dict with keys: accuracy, f1_macro, cumulative_return_dqn,
    cumulative_return_bh (buy-and-hold baseline).
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    preds = predict_dqn(agent, X_test)

    acc = accuracy_score(y_test_enc, preds)
    f1  = f1_score(y_test_enc, preds, average="macro", zero_division=0)
    print(f"   DQN Agent   Acc={acc:.4f}  F1={f1:.4f}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    CLASS_NAMES = ["Bear(0)", "Sideways(1)", "Bull(2)"]
    cm = confusion_matrix(y_test_enc, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_title("DQN Agent\nConfusion Matrix (encoded labels)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/DQN_confusion_matrix.png")
    plt.close(fig)

    # ── training reward curve ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards_hist, color="#F7931A", lw=1.5, label="Episode reward")
    window = max(1, len(rewards_hist) // 10)
    smoothed = np.convolve(rewards_hist, np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(rewards_hist)), smoothed,
            color="#1f77b4", lw=2, label=f"Moving avg (w={window})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cumulative reward (PnL)")
    ax.set_title("DQN Training Reward Curve")
    ax.legend(); plt.tight_layout()
    fig.savefig(f"{output_dir}/DQN_training_rewards.png")
    plt.close(fig)

    # ── cumulative return comparison ──────────────────────────────────────────
    positions     = preds - 1                       # {-1, 0, +1}
    dqn_returns   = positions * returns_test
    bh_returns    = returns_test                    # buy-and-hold is always long

    cum_dqn = np.cumprod(1 + np.clip(dqn_returns, -0.5, 0.5)) - 1
    cum_bh  = np.cumprod(1 + np.clip(bh_returns,  -0.5, 0.5)) - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum_dqn, color="#F7931A", lw=2, label="DQN Agent")
    ax.plot(cum_bh,  color="#1f77b4", lw=1.5, ls="--", label="Buy-and-Hold")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Test step (days)"); ax.set_ylabel("Cumulative return")
    ax.set_title("DQN Agent vs Buy-and-Hold (test period)")
    ax.legend(); plt.tight_layout()
    fig.savefig(f"{output_dir}/DQN_cumulative_return.png")
    plt.close(fig)

    return {
        "accuracy":                acc,
        "f1_macro":                f1,
        "cumulative_return_dqn":   float(cum_dqn[-1]),
        "cumulative_return_bh":    float(cum_bh[-1]),
    }
