"""
training/env_integration/envs.py
=================================
Vectorizes WorkforceWorker into the batched reset()/step() shape that
verl-agent's EnvironmentManagerBase expects: reset() returns
(text_obs: List[str], infos: List[dict]), step(actions) returns
(text_obs, rewards: np.ndarray, dones: np.ndarray, infos: List[dict]).

Task assignment is deliberately NOT decided in here. This file's only job
is plumbing -- fan calls out to N workers, gather results back. WHICH task
each worker gets is a training-design decision (how many easy/medium/hard/
crisis episodes per batch, how GRPO/GiGPO's group_n replicas are laid out)
that belongs in the training config, not hardcoded into the vectorizer.
Keeping those concerns separate means the task-mix policy can change later
without ever touching this file.
"""

from __future__ import annotations

from typing import List

import numpy as np
import ray

from env_integration.worker import WorkforceWorker


class MeridianVecEnv:
    """Fans reset/step out across `num_envs` parallel WorkforceWorker actors."""

    def __init__(self, num_envs: int, resources_per_worker: dict | None = None):
        self.num_envs = num_envs
        opts = {}
        if resources_per_worker:
            opts["num_cpus"] = resources_per_worker.get("num_cpus", 0.1)
        actor_cls = WorkforceWorker.options(**opts) if opts else WorkforceWorker
        self.workers = [actor_cls.remote(i) for i in range(num_envs)]

    def reset(self, task_names: List[str]):
        assert len(task_names) == self.num_envs, (
            f"Expected {self.num_envs} task names, got {len(task_names)}"
        )
        raw = ray.get([
            w.reset.remote(t) for w, t in zip(self.workers, task_names)
        ])
        text_obs = [_format_observation(r, first=True) for r in raw]
        infos = [{"raw_observation": r} for r in raw]
        return text_obs, infos

    def step(self, actions: List[tuple[str, str]]):
        """
        actions: list of (action_type, target) tuples, one per worker,
        already parsed by the projection function -- this layer only
        forwards them, it doesn't parse raw model text.
        """
        assert len(actions) == self.num_envs, (
            f"Expected {self.num_envs} actions, got {len(actions)}"
        )
        raw = ray.get([
            w.step.remote(a_type, target)
            for w, (a_type, target) in zip(self.workers, actions)
        ])
        text_obs = [_format_observation(r["observation"]) for r in raw]
        rewards = np.array([r["reward"] for r in raw], dtype=np.float32)
        dones = np.array([r["done"] for r in raw], dtype=bool)
        infos = [{"raw_observation": r["observation"], **r["info"]} for r in raw]
        return text_obs, rewards, dones, infos


def _format_observation(obs: dict, first: bool = False) -> str:
    """
    Turn one worker's observation dict into the plain-text description the
    prompt template will embed. This is intentionally minimal -- just the
    facts, no instructions -- because the *instructions* (how to respond,
    what format to use) belong in the prompt template, not repeated on
    every single step.
    """
    lines = []
    if obs.get("last_action_error"):
        lines.append(f"Last action FAILED: {obs['last_action_error']}")
    elif not first:
        lines.append(f"Last action result: {obs.get('last_action_result', '')}")

    if obs.get("current_blockers"):
        lines.append("Blockers preventing finalization: " + ", ".join(obs["current_blockers"]))

    lines.append("Available actions: " + ", ".join(obs.get("available_actions", [])))
    return "\n".join(lines)


def build_meridian_envs(
    seed: int,
    env_num: int,
    group_n: int,
    task_pool: List[str],
    is_train: bool = True,
    resources_per_worker: dict | None = None,
):
    """
    Build a MeridianVecEnv sized for env_num distinct episodes x group_n
    replicas each (the GRPO/GiGPO group size) -- same convention as
    build_sokoban_envs / build_alfworld_envs elsewhere in verl-agent.

    task_pool: the task names to cycle through across the env_num distinct
    episodes (e.g. ["easy", "medium", "hard", "crisis"]). Each is repeated
    group_n times so every task gets group_n independent attempts, which is
    what both GRPO and GiGPO need to compute a within-group advantage.
    """
    import random
    rng = random.Random(seed)
    total = env_num * group_n

    base_tasks = [task_pool[i % len(task_pool)] for i in range(env_num)]
    if is_train:
        rng.shuffle(base_tasks)
    task_names = [t for t in base_tasks for _ in range(group_n)]

    assert len(task_names) == total
    return MeridianVecEnv(num_envs=total, resources_per_worker=resources_per_worker), task_names