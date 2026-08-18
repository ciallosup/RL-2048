from __future__ import annotations

import os

from rl2048.policies.base import Policy
from rl2048.policies.fixed_priority import FixedPriorityPolicy
from rl2048.policies.greedy import GreedyMergePolicy
from rl2048.policies.heuristic import HeuristicPolicy
from rl2048.policies.manual import ManualPolicy
from rl2048.policies.random_policy import RandomPolicy

# Policies shown in visualizer (manual first).
POLICY_ORDER = ("manual", "random", "heuristic", "greedy", "fixed")

# Non-neural baselines for evaluation (roadmap section 4).
BASELINE_POLICIES = ("random", "fixed", "heuristic", "greedy")

POLICY_REGISTRY: dict[str, type] = {
    ManualPolicy.key: ManualPolicy,
    RandomPolicy.key: RandomPolicy,
    HeuristicPolicy.key: HeuristicPolicy,
    GreedyMergePolicy.key: GreedyMergePolicy,
    FixedPriorityPolicy.key: FixedPriorityPolicy,
}


def list_policies() -> list[tuple[str, str]]:
    items = [(key, POLICY_REGISTRY[key].label) for key in POLICY_ORDER]
    items.append(("dqn", "RL (DQN checkpoint)"))
    return items


def list_baselines() -> list[tuple[str, str]]:
    return [(key, POLICY_REGISTRY[key].label) for key in BASELINE_POLICIES]


def get_policy_class(key: str) -> type:
    if key not in POLICY_REGISTRY:
        raise KeyError(f"Unknown policy: {key}")
    return POLICY_REGISTRY[key]


def get_policy(key: str) -> Policy:
    if key == "dqn":
        from rl2048.policies.dqn_policy import DQNPolicy

        return DQNPolicy.from_checkpoint(os.environ["RL2048_CHECKPOINT"])
    return get_policy_class(key)()
