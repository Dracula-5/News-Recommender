"""
Bandit exploration policies for news category selection.

Three complementary strategies, combined into a HybridPolicy that blends
them with a DDQN's Q-values:

  ThompsonSamplingPolicy — Beta distribution per action; good cold-start
  UCBPolicy              — UCB1; balances visits vs. expected reward
  HybridPolicy           — weighted blend of TS + DDQN Q-values;
                           shifts from TS towards DDQN as step count grows

All state is serialisable (get_state / load_state) for per-user persistence
alongside the DDQN checkpoint.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────
#  Thompson Sampling  (Beta distribution, binary rewards)
# ──────────────────────────────────────────────────────────

class ThompsonSamplingPolicy:
    """
    Thompson Sampling with Beta(alpha, beta) per action.

    * Update: liked → alpha += 1;  not liked → beta += 1
    * Select: sample from each Beta distribution, pick argmax
    * Expected reward: alpha / (alpha + beta)

    Provides 95% credible intervals for the dashboard UI.

    Discounted (non-stationary): plain Thompson Sampling assumes a fixed
    reward distribution per action, but a user's taste for a category
    drifts over weeks/months. Without decay, alpha/beta accumulate forever
    — after a few hundred interactions a single new one barely moves the
    posterior at all, so the bandit stops being able to notice a user
    going off a category they used to like. `decay` (<1) shrinks the whole
    posterior a little on every update, so recent evidence keeps
    proportionally more weight — standard "discounted Thompson Sampling."
    """

    def __init__(self, n_actions: int, decay: float = 0.995) -> None:
        self.n_actions = n_actions
        self.decay = decay
        # Both start at 1 → uniform prior
        self.alpha = np.ones(n_actions, dtype=np.float64)
        self.beta  = np.ones(n_actions, dtype=np.float64)

    # ── select / update ───────────────────────────────────

    def select(self, mask: Optional[np.ndarray] = None) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        if mask is not None:
            samples = np.where(mask, samples, -np.inf)
        return int(np.argmax(samples))

    def update(self, action: int, liked: bool) -> None:
        # Discount every arm's posterior a little each round, floored at
        # the uniform prior (1, 1) so the Beta distribution never
        # degenerates as it decays.
        self.alpha = np.maximum(self.alpha * self.decay, 1.0)
        self.beta  = np.maximum(self.beta  * self.decay, 1.0)
        if liked:
            self.alpha[action] += 1.0
        else:
            self.beta[action]  += 1.0

    # ── queries ───────────────────────────────────────────

    def get_expected_reward(self) -> np.ndarray:
        """Mean of Beta(alpha, beta) = alpha / (alpha+beta).  Range [0,1]."""
        return self.alpha / (self.alpha + self.beta)

    def get_confidence_intervals(self) -> List[Dict]:
        """
        95 % credible interval for each action.
        Returns list of dicts suitable for JSON serialisation.
        """
        from scipy.stats import beta as _beta  # only imported when called

        out = []
        for i in range(self.n_actions):
            lo, hi = _beta.ppf([0.025, 0.975], self.alpha[i], self.beta[i])
            out.append({
                "action": i,
                "mean":   round(float(self.get_expected_reward()[i]), 4),
                "lo":     round(float(lo), 4),
                "hi":     round(float(hi), 4),
                "alpha":  round(float(self.alpha[i]), 2),
                "beta":   round(float(self.beta[i]),  2),
            })
        return out

    def get_confidence_intervals_fast(self) -> List[Dict]:
        """
        Approximate 95% CI using normal approximation (no scipy needed).
        Used in the dashboard when scipy is unavailable.
        """
        mu = self.get_expected_reward()
        n  = self.alpha + self.beta
        se = np.sqrt(mu * (1 - mu) / np.maximum(n, 1))
        z  = 1.96
        out = []
        for i in range(self.n_actions):
            out.append({
                "action": i,
                "mean":   round(float(mu[i]),                      4),
                "lo":     round(float(max(0.0, mu[i] - z * se[i])), 4),
                "hi":     round(float(min(1.0, mu[i] + z * se[i])), 4),
                "alpha":  round(float(self.alpha[i]), 2),
                "beta":   round(float(self.beta[i]),  2),
            })
        return out

    # ── serialisation ─────────────────────────────────────

    def get_state(self) -> Dict:
        return {
            "alpha": self.alpha.tolist(),
            "beta":  self.beta.tolist(),
        }

    def load_state(self, state: Dict) -> None:
        self.alpha = np.array(state["alpha"], dtype=np.float64)
        self.beta  = np.array(state["beta"],  dtype=np.float64)
        self.n_actions = len(self.alpha)


# ──────────────────────────────────────────────────────────
#  UCB1 policy
# ──────────────────────────────────────────────────────────

class UCBPolicy:
    """
    UCB1: selects  argmax[ value + c * sqrt(ln(t) / (count + ε)) ]

    c = sqrt(2) is the theoretically optimal constant for bounded [0,1] rewards.

    Discounted, same reasoning as ThompsonSamplingPolicy above: `counts`
    decays a little on every update, so (a) the running-average `values`
    update keeps weighting recent rewards more heavily instead of an
    ever-growing raw count diluting new evidence toward nothing, and (b)
    the explore bonus naturally grows for an action that hasn't been
    picked in a while even without decaying `values` directly, since the
    global round counter `t` keeps climbing while that action's decayed
    `count` shrinks.
    """

    def __init__(self, n_actions: int, c: float = 1.41, decay: float = 0.995) -> None:
        self.n_actions = n_actions
        self.c = c
        self.decay = decay
        self.counts = np.zeros(n_actions, dtype=np.float64)
        self.values = np.zeros(n_actions, dtype=np.float64)
        self.t = 0

    def select(self, mask: Optional[np.ndarray] = None) -> int:
        self.t += 1
        # Unvisited actions have infinite UCB score
        unvisited = np.where(self.counts == 0)[0]
        if mask is not None:
            unvisited = unvisited[mask[unvisited]]
        if len(unvisited):
            return int(np.random.choice(unvisited))

        bonus = self.c * np.sqrt(np.log(self.t) / (self.counts + 1e-9))
        scores = self.values + bonus
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        return int(np.argmax(scores))

    def update(self, action: int, reward: float) -> None:
        self.counts *= self.decay
        self.counts[action] += 1
        n = self.counts[action]
        # Incremental running average over the *decayed* count, so a
        # streak of recent rewards moves this faster than it would if `n`
        # were the raw, ever-growing number of times this action was picked.
        self.values[action] += (reward - self.values[action]) / n

    def get_state(self) -> Dict:
        return {
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
            "t": int(self.t),
        }

    def load_state(self, state: Dict) -> None:
        self.counts = np.array(state["counts"], dtype=np.float64)
        self.values = np.array(state["values"], dtype=np.float64)
        self.t = int(state["t"])
        self.n_actions = len(self.counts)


# ──────────────────────────────────────────────────────────
#  Hybrid policy (DDQN Q-values  +  Thompson Sampling)
# ──────────────────────────────────────────────────────────

class HybridPolicy:
    """
    Blends Dueling DDQN Q-values with Thompson Sampling expected rewards.

    Cold start (step ≈ 0):
        almost all weight on TS (pure exploration / exploitation via Bayes)

    Warm      (step ≥ 200):
        DDQN weight dominates (70 %), TS keeps 30 % for robustness

    Transition: linear interpolation between the two regimes.

    The `blend_scores` method returns a combined score array that the
    DDQNAgent uses for action selection instead of raw epsilon-greedy.
    """

    WARM_STEP = 200           # step at which DDQN weight plateaus
    MAX_DDQN_WEIGHT = 0.70    # final DDQN blend weight
    MIN_DDQN_WEIGHT = 0.20    # initial DDQN blend weight (cold start)

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.ts  = ThompsonSamplingPolicy(n_actions)
        self.ucb = UCBPolicy(n_actions)
        self.step = 0

    # ── core blend ────────────────────────────────────────

    def ddqn_weight(self) -> float:
        """DDQN weight as a function of step count."""
        ratio = min(1.0, self.step / max(1, self.WARM_STEP))
        return self.MIN_DDQN_WEIGHT + ratio * (self.MAX_DDQN_WEIGHT - self.MIN_DDQN_WEIGHT)

    def blend_scores(
        self,
        ddqn_q: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Combine DDQN Q-values and TS expected rewards into a single score array.
        Higher score → preferred action.
        """
        # Normalise Q-values to [0,1]
        q = ddqn_q.copy().astype(np.float64)
        q_range = q.max() - q.min()
        if q_range > 1e-8:
            q = (q - q.min()) / q_range
        else:
            q = np.full_like(q, 0.5)

        ts_r = self.ts.get_expected_reward()
        dw   = self.ddqn_weight()
        blended = dw * q + (1.0 - dw) * ts_r

        if mask is not None:
            blended = np.where(mask, blended, -np.inf)
        return blended

    def select(
        self,
        ddqn_q: np.ndarray,
        mask: Optional[np.ndarray] = None,
        epsilon: float = 0.0,
    ) -> int:
        """Select action using blended policy (with optional ε-greedy fallback)."""
        import random
        allowed = (
            np.where(mask)[0].tolist()
            if mask is not None
            else list(range(self.n_actions))
        )
        if not allowed:
            allowed = list(range(self.n_actions))

        if random.random() < epsilon:
            return random.choice(allowed)

        blended = self.blend_scores(ddqn_q, mask)
        return int(np.argmax(blended))

    # ── update ────────────────────────────────────────────

    def update(self, action: int, reward: float, liked: bool) -> None:
        # Normalise reward to [0,1] for TS/UCB (they expect [0,1])
        norm_reward = 1.0 / (1.0 + np.exp(-reward * 0.3))
        self.ts.update(action, liked)
        self.ucb.update(action, norm_reward)
        self.step += 1

    # ── queries ───────────────────────────────────────────

    def get_ts_confidence(self, categories: Sequence[str]) -> List[Dict]:
        """
        Return TS confidence intervals for each category.
        Used by the dashboard endpoint.
        """
        cis = self.ts.get_confidence_intervals_fast()
        for ci, cat in zip(cis, categories):
            ci["category"] = cat
        return cis

    def get_stats(self) -> Dict:
        return {
            "step":         self.step,
            "ddqn_weight":  round(self.ddqn_weight(), 3),
            "ts_expected":  [round(float(x), 4) for x in self.ts.get_expected_reward()],
            "ucb_values":   [round(float(x), 4) for x in self.ucb.values],
            "ucb_counts":   [int(x) for x in self.ucb.counts],
        }

    # ── serialisation ─────────────────────────────────────

    def get_state(self) -> Dict:
        return {
            "step": self.step,
            "ts":   self.ts.get_state(),
            "ucb":  self.ucb.get_state(),
        }

    def load_state(self, state: Dict) -> None:
        self.step = int(state.get("step", 0))
        if "ts" in state:
            self.ts.load_state(state["ts"])
        if "ucb" in state:
            self.ucb.load_state(state["ucb"])
