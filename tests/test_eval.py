"""Tests for evaluation metrics and runner."""

from rl2048.eval.metrics import distribution_summary, wilson_interval
from rl2048.eval.runner import EpisodeResult, evaluate_policy
from rl2048.eval.seeds import dev_seeds, generate_seeds
from rl2048.policies.random_policy import RandomPolicy


def test_wilson_interval_bounds():
    ci = wilson_interval(20, 100)
    assert 0.0 <= ci.lower <= ci.rate <= ci.upper <= 1.0
    assert ci.successes == 20 and ci.n == 100


def test_dev_seeds_reproducible():
    assert dev_seeds(10) == dev_seeds(10)
    assert len(generate_seeds(99, 50)) == 50


def test_evaluate_random_policy_smoke():
    summary = evaluate_policy(
        RandomPolicy(),
        policy_key="random",
        policy_label="random",
        seeds=dev_seeds(5),
        max_episode_steps=100,
    )
    assert summary.episodes == 5
    assert 0.0 <= summary.p_reach_2048 <= 1.0
    assert summary.score_stats["mean"] >= 0.0
    assert len(summary.raw_episodes) == 5
    assert isinstance(summary.raw_episodes[0], EpisodeResult)
