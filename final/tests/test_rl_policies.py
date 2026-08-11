"""
Regression tests for the non-stationarity fix in rl_policies.py: plain
Thompson Sampling / UCB1 accumulate evidence forever, so after enough
history a single new interaction (or even a long streak of new ones)
barely moves the estimate — the bandit stops being able to notice a user's
taste actually changing. Both now decay a little on every update
(discount factor < 1); these tests compare decayed vs. undiscounted
(decay=1.0, i.e. the old behavior) head-to-head to prove the fix actually
changes the outcome, not just that the code runs.
"""
from __future__ import annotations

from rl_policies import ThompsonSamplingPolicy, UCBPolicy


def test_thompson_sampling_posterior_never_degenerates_below_prior():
    ts = ThompsonSamplingPolicy(n_actions=3, decay=0.9)
    for _ in range(200):
        ts.update(action=0, liked=False)
    assert (ts.alpha >= 1.0).all()
    assert (ts.beta >= 1.0).all()


def test_thompson_sampling_with_decay_adapts_faster_to_a_preference_reversal():
    # Long history of "liked" on action 0, then a reversal to "not liked".
    # Decayed TS should end up noticeably less confident in action 0 than
    # undiscounted TS, which barely budges after 300 prior likes.
    decayed = ThompsonSamplingPolicy(n_actions=2, decay=0.99)
    stale   = ThompsonSamplingPolicy(n_actions=2, decay=1.0)

    for policy in (decayed, stale):
        for _ in range(300):
            policy.update(action=0, liked=True)
        for _ in range(30):
            policy.update(action=0, liked=False)

    decayed_belief = decayed.get_expected_reward()[0]
    stale_belief   = stale.get_expected_reward()[0]

    assert decayed_belief < stale_belief


def test_ucb_values_never_used_with_zero_count_division():
    ucb = UCBPolicy(n_actions=3, decay=0.9)
    for _ in range(500):
        ucb.update(action=1, reward=1.0)
    assert (ucb.counts > 0).any()
    assert not any(v != v for v in ucb.values)  # no NaNs


def test_ucb_with_decay_adapts_faster_to_a_reward_reversal():
    decayed = UCBPolicy(n_actions=2, decay=0.99)
    stale   = UCBPolicy(n_actions=2, decay=1.0)

    for policy in (decayed, stale):
        for _ in range(300):
            policy.update(action=0, reward=1.0)
        for _ in range(30):
            policy.update(action=0, reward=0.0)

    assert decayed.values[0] < stale.values[0]
