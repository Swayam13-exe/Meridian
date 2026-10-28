"""
training/env_integration/envs.py
=================================
Vectorizes WorkforceEnv into the batched reset()/step() shape that
verl-agent's EnvironmentManagerBase expects: reset() returns
(text_obs: List[str], infos: List[dict]), step(actions) returns
(text_obs, rewards: np.ndarray, dones: np.ndarray, infos: List[dict]).

Why the environments run in-process instead of one Ray actor each:
verl-agent's other environments use a Ray actor per instance because those
environments are heavy (servers, large datasets, search indexes). WorkforceEnv
is a pure in-memory state machine whose step() costs microseconds -- nothing
next to LLM generation -- while every Ray worker process costs ~0.3 GB of host
RAM (it re-imports the torch stack through the worker_process_setup_hook).
That is a poor trade at any scale and a fatal one at realistic batch sizes
(16 prompts x group of 8 = 128 processes = ~40 GB). So the instances live in
the driver process and are stepped in a loop.

The price of sharing a process is that isolation between episodes is no longer
enforced by a process boundary. Each reset() starts from a deep copy of the
task template (see WorkforceEnv.reset), and test_isolation.py verifies that
episodes do not leak into each other and that the shared module-level rule
tables are never mutated.

Task assignment is deliberately NOT decided in here. This file's only job
is plumbing -- fan calls out to N environments, gather results back. WHICH task
each environment gets is a training-design decision (how many easy/medium/hard/
crisis episodes per batch, how GRPO/GiGPO's group_n replicas are laid out)
that belongs in the training config, not hardcoded into the vectorizer.
Keeping those concerns separate means the task-mix policy can change later
without ever touching this file.
"""

from __future__ import annotations

import copy
from typing import List

import numpy as np

from env.environment import WorkforceEnv
from env.models import Action
from training.env_integration.projection import format_action


class MeridianVecEnv:
    """Steps `num_envs` independent WorkforceEnv instances in lockstep."""

    def __init__(self, num_envs: int, resources_per_worker: dict | None = None):
        # resources_per_worker only mattered when each environment was a Ray
        # actor. It is accepted and ignored so existing callers and configs
        # (env.resources_per_worker.*) keep working unchanged.
        self.num_envs = num_envs
        self.envs = [WorkforceEnv() for _ in range(num_envs)]

    def reset(self, task_names: List[str]):
        assert len(task_names) == self.num_envs, (
            f"Expected {self.num_envs} task names, got {len(task_names)}"
        )
        raw = [
            _observation_to_dict(env.reset(t))
            for env, t in zip(self.envs, task_names)
        ]
        text_obs = [_format_observation(r, first=True) for r in raw]
        infos = [{"raw_observation": r} for r in raw]
        return text_obs, infos

    def step(self, actions: List[tuple[str, str]]):
        """
        actions: list of (action_type, target) tuples, one per environment,
        already parsed by the projection function -- this layer only
        forwards them, it doesn't parse raw model text.
        """
        assert len(actions) == self.num_envs, (
            f"Expected {self.num_envs} actions, got {len(actions)}"
        )
        raw = []
        for env, (a_type, target) in zip(self.envs, actions):
            result = env.step(Action(action_type=a_type, target=target))
            raw.append({
                "observation": _observation_to_dict(result.observation),
                "reward": result.reward,
                "done": result.done,
                # Copied so a caller that edits an info dict can never reach
                # back into the environment's own state.
                "info": copy.deepcopy(result.info),
            })
        text_obs = [_format_observation(r["observation"]) for r in raw]
        rewards = np.array([r["reward"] for r in raw], dtype=np.float32)
        dones = np.array([r["done"] for r in raw], dtype=bool)
        infos = [{"raw_observation": r["observation"], **r["info"]} for r in raw]
        return text_obs, rewards, dones, infos

    def close(self) -> None:
        """Release the environment instances. verl-agent's environment
        managers expose close() and delegate it to the vectorized env."""
        self.envs = []


def _observation_to_dict(obs) -> dict:
    """
    Flatten an Observation pydantic model into a plain dict. Keeping this
    as one small shared helper (instead of repeating .model_dump() calls
    all over the codebase) means if the Observation schema changes later,
    there's exactly one place to update. Lists are copied so the returned
    dict never aliases the environment's internal containers.
    """
    return {
        "state": obs.state.model_dump(),
        "available_actions": list(obs.available_actions),
        "current_blockers": list(obs.current_blockers),
        "last_action_result": obs.last_action_result,
        "last_action_error": obs.last_action_error,
        "steps_taken": obs.steps_taken,
        "done": obs.done,
    }


def _format_observation(obs: dict, first: bool = False) -> str:
    """
    Turn one environment's observation dict into the plain-text description the
    prompt template will embed. This is intentionally minimal -- just the
    facts, no instructions -- because the *instructions* (how to respond,
    what format to use) belong in the prompt template, not repeated on
    every single step.

    The one place the format does show up is the list of available moves: they
    are written as the JSON objects the model has to reply with, not as the
    environment's own 'action_type:target' shorthand. In the generation-only
    pre-flight the model copied the shorthand it was shown: Qwen3-0.6B wrote it
    instead of JSON on 60-67% of its steps, and Qwen3-1.7B put it inside
    "action_type", so the environment rejected about 89% of its moves. Listing
    the moves as JSON fixed both (setting nothink-json-512 in preflight_generation.py).
    """
    lines = []
    if obs.get("last_action_error"):
        lines.append(f"Last action FAILED: {obs['last_action_error']}")
    elif not first:
        lines.append(f"Last action result: {obs.get('last_action_result', '')}")

    if obs.get("current_blockers"):
        lines.append("Blockers preventing finalization: " + ", ".join(obs["current_blockers"]))

    lines.append("Available actions: " + ", ".join(_action_as_json(a) for a in obs.get("available_actions", [])))
    return "\n".join(lines)


def _action_as_json(shorthand: str) -> str:
    """'request_document:passport' -> {"action_type": "request_document", "target": "passport"};
    'approve_hr' (a move with no target) -> {"action_type": "approve_hr", "target": ""}.
    The environment names every move this way (see WorkforceEnv._get_available_actions);
    splitting at the FIRST colon is safe because an action_type never contains one."""
    action_type, _, target = shorthand.partition(":")
    return format_action(action_type, target)


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