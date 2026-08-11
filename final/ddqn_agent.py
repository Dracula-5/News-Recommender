"""
Dueling Double DQN agent for news category selection.

Changes vs. original:
  * State dimension expanded from 5 → 10 (richer context)
  * Deeper network: 256 → 128 → (value / advantage heads)
  * LayerNorm after first layer for training stability
  * HybridPolicy (Thompson Sampling + UCB) replaces plain ε-greedy
  * Hybrid policy state saved/loaded alongside model weights
  * Graceful architecture-mismatch recovery (old 5-D → new 10-D)
"""
from __future__ import annotations

import json
import logging
import queue
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from database import DB_PATH, db_connect, fetch_categories, init_db
from rl_policies import HybridPolicy

logger = logging.getLogger(__name__)

STATE_DIM = 10          # see build_state_from_row in recommender.py
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Shared, bounded pool for background checkpoint saves — every DDQNAgent
# used to spin up its own dedicated daemon thread that blocked forever on
# an internal queue.get(), which meant one permanently-alive OS thread per
# distinct user_id a running process ever served (recommender.py caches
# one DDQNAgent per user for the process lifetime). A long-running server
# — or just this project's own evaluation harness, which mints a fresh
# synthetic user_id per run — would leak threads without bound. A shared
# pool caps the thread count regardless of how many users are served.
_SAVE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ddqn-save")


# ──────────────────────────────────────────────────────────
#  Network
# ──────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Dueling network with LayerNorm for stable learning on the 10-D state.

    Architecture:
      input(10) → Linear(256) → LayerNorm(256) → ReLU
                → Linear(128) → ReLU
                ┌→ value_head:     Linear(64) → ReLU → Linear(1)
                └→ advantage_head: Linear(64) → ReLU → Linear(n_actions)
      Q = V(s) + A(s,a) − mean(A(s,·))
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


# ──────────────────────────────────────────────────────────
#  Replay buffer
# ──────────────────────────────────────────────────────────

@dataclass
class Transition:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


class ReplayBuffer:
    def __init__(self, capacity: int = 1000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, n: int) -> list[Transition]:
        return random.sample(self.buffer, n)

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────────────────
#  Agent
# ──────────────────────────────────────────────────────────

class DDQNAgent:
    def __init__(
        self,
        categories: Optional[Sequence[str]] = None,
        user_id: str = "default",
        gamma: float = 0.9,
        lr: float = 1e-3,
        batch_size: int = 16,
        target_update: int = 20,
        save_interval: int = 20,
        db_path: Path = DB_PATH,
    ) -> None:
        init_db(db_path)
        self.db_path    = db_path
        self.user_id    = user_id
        self.categories = list(categories or fetch_categories(db_path)) or ["general"]
        self.action_dim = len(self.categories)
        self.gamma      = gamma
        self.batch_size = batch_size
        self.target_update  = target_update
        self.save_interval  = save_interval
        self.device = torch.device("cpu")

        self.model        = DuelingDQN(STATE_DIM, self.action_dim).to(self.device)
        self.target_model = DuelingDQN(STATE_DIM, self.action_dim).to(self.device)
        self.optimizer    = Adam(self.model.parameters(), lr=lr)
        self.replay       = ReplayBuffer(1000)
        self.step         = 0
        self.epsilon      = 0.15

        # Hybrid bandit policy (TS + UCB + Q-value blend)
        self.hybrid = HybridPolicy(self.action_dim)

        # Async save: coalescing queue (only the latest pending checkpoint
        # is kept per agent) drained by the shared _SAVE_EXECUTOR pool —
        # see the module-level note on why this isn't a dedicated thread.
        self._save_queue: queue.Queue = queue.Queue(maxsize=1)

        self._load_or_init()

    # ── model file path ───────────────────────────────────

    @property
    def model_path(self) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.user_id)
        return MODEL_DIR / f"ddqn_{safe}.pt"

    # ── load / init ───────────────────────────────────────

    def _load_or_init(self) -> None:
        with db_connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT step, epsilon, category_order FROM agent_state WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT OR REPLACE INTO agent_state(user_id, category_order) VALUES (?, ?)",
                    (self.user_id, json.dumps(self.categories)),
                )
            else:
                self.step    = int(row["step"])
                self.epsilon = float(row["epsilon"])

        if not self.model_path.exists():
            self.target_model.load_state_dict(self.model.state_dict())
            return

        try:
            payload = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if list(payload.get("categories", [])) == self.categories:
                try:
                    self.model.load_state_dict(payload["model"])
                    self.target_model.load_state_dict(payload["target_model"])
                    self.optimizer.load_state_dict(payload["optimizer"])
                    self.step    = int(payload.get("step", self.step))
                    self.epsilon = float(payload.get("epsilon", self.epsilon))
                    # Restore hybrid policy state
                    if "hybrid" in payload:
                        self.hybrid.load_state(payload["hybrid"])
                except RuntimeError:
                    logger.info(
                        "Model architecture mismatch for user %s — reinitialising.",
                        self.user_id,
                    )
                    self.target_model.load_state_dict(self.model.state_dict())
            else:
                self.target_model.load_state_dict(self.model.state_dict())
        except Exception as exc:
            logger.warning("Could not load model for %s: %s", self.user_id, exc)
            self.target_model.load_state_dict(self.model.state_dict())

    # ── async save ────────────────────────────────────────

    def _do_save(self, payload: dict) -> None:
        torch.save(
            {
                "model":        payload["model"],
                "target_model": payload["target_model"],
                "optimizer":    payload["optimizer"],
                "categories":   payload["categories"],
                "step":         payload["step"],
                "epsilon":      payload["epsilon"],
                "hybrid":       payload["hybrid"],
            },
            self.model_path,
        )
        with db_connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_state
                (user_id, step, epsilon, category_order, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (self.user_id, payload["step"], payload["epsilon"],
                 json.dumps(payload["categories"])),
            )

    def _enqueue_save(self, payload: dict) -> None:
        try:
            if self._save_queue.full():
                self._save_queue.get_nowait()  # coalesce: drop the stale pending save
            self._save_queue.put_nowait(payload)
        except queue.Full:
            return
        _SAVE_EXECUTOR.submit(self._drain_and_save)

    def _drain_and_save(self) -> None:
        try:
            payload = self._save_queue.get_nowait()
        except queue.Empty:
            return  # already drained by another submitted task — nothing to do
        try:
            self._do_save(payload)
        except Exception:
            logger.exception("DDQNAgent: checkpoint save failed for user %s", self.user_id)
        finally:
            self._save_queue.task_done()

    def save(self) -> None:
        self._enqueue_save({
            "model":        self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "categories":   self.categories,
            "step":         self.step,
            "epsilon":      self.epsilon,
            "hybrid":       self.hybrid.get_state(),
        })

    # ── inference ─────────────────────────────────────────

    def q_values(self, state: Sequence[float]) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor([state], dtype=torch.float32, device=self.device)
            return self.model(t).cpu().numpy()[0]

    def q_values_batch(self, states: Sequence[Sequence[float]]) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(np.array(states, dtype=np.float32), device=self.device)
            return self.model(t).cpu().numpy()

    # ── action selection ──────────────────────────────────

    def choose_action(
        self,
        state: Sequence[float],
        blocked: Optional[set[str]] = None,
    ) -> int:
        blocked = blocked or set()
        allowed = [i for i, c in enumerate(self.categories) if c not in blocked]
        if not allowed:
            allowed = list(range(self.action_dim))

        mask = np.zeros(self.action_dim, dtype=bool)
        mask[allowed] = True

        q = self.q_values(state)
        return self.hybrid.select(q, mask=mask, epsilon=self.epsilon)

    # ── training ──────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        batch  = self.replay.sample(self.batch_size)
        states = torch.tensor(np.array([t.state for t in batch]),      dtype=torch.float32)
        acts   = torch.tensor([t.action for t in batch],               dtype=torch.long).unsqueeze(1)
        rews   = torch.tensor([t.reward for t in batch],               dtype=torch.float32).unsqueeze(1)
        nexts  = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32)
        dones  = torch.tensor([t.done for t in batch],                 dtype=torch.float32).unsqueeze(1)

        q_curr   = self.model(states).gather(1, acts)
        with torch.no_grad():
            next_acts  = self.model(nexts).argmax(dim=1, keepdim=True)
            next_q     = self.target_model(nexts).gather(1, next_acts)
            targets    = rews + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_curr, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.step    += 1
        self.epsilon  = max(0.03, self.epsilon * 0.995)
        if self.step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return float(loss.item())

    # ── observe (called after every interaction) ──────────

    def observe(
        self,
        state:      Sequence[float],
        category:   str,
        reward:     float,
        next_state: Sequence[float],
        done:       bool = False,
        liked:      bool = False,
    ) -> Optional[float]:
        if category not in self.categories:
            return None
        action = self.categories.index(category)

        self.replay.push(Transition(
            np.array(state,      dtype=np.float32),
            action,
            float(reward),
            np.array(next_state, dtype=np.float32),
            done,
        ))

        # Update bandit policies
        self.hybrid.update(action, reward, liked)

        loss = self.train_step()
        if self.step % self.save_interval == 0:
            self.save()
        return loss

    # ── metrics (for the dashboard) ───────────────────────

    def get_metrics(self) -> dict:
        return {
            "user_id":       self.user_id,
            "step":          self.step,
            "epsilon":       round(self.epsilon, 4),
            "replay_size":   len(self.replay),
            "ddqn_weight":   round(self.hybrid.ddqn_weight(), 3),
            "hybrid_step":   self.hybrid.step,
            "ts_expected":   [round(float(x), 4) for x in self.hybrid.ts.get_expected_reward()],
            "categories":    self.categories,
        }
