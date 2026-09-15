"""
training/env_integration/worker.py
===================================
A Ray actor holding exactly one WorkforceEnv instance.

Why Ray, and why one actor per environment instance:
verl-agent trains on *batches* of parallel episodes per training step (this
is what env.rollout.n / data.train_batch_size control in their config). To
produce a batch, something has to run many independent episodes of
WorkforceEnv at once. Ray actors are how every environment in verl-agent
does this -- each actor is a separate worker process holding its own
environment instance, so N actors can each be mid-episode simultaneously
without stepping on each other's state.

WorkforceEnv itself needs none of the heavier machinery AppWorld's workers
use (no ports, no subprocess servers, no network calls) -- it's pure,
fast, in-memory Python. So this worker is intentionally the simplest
version of this pattern in the whole framework: construct one WorkforceEnv,
expose reset/step as remote methods, done.
"""

from __future__ import annotations

import ray

from env.environment import WorkforceEnv
from env.models import Action


@ray.remote
class WorkforceWorker:
    """One WorkforceEnv, running in its own Ray worker process."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.env = WorkforceEnv()

    def reset(self, task_name: str):
        """
        Start a new episode. Returns a plain dict (not the Observation
        pydantic object) -- Ray serializes return values to send them back
        to the main process, and plain dicts are the simplest, most
        predictable thing to serialize reliably across that boundary.
        """
        obs = self.env.reset(task_name)
        return _observation_to_dict(obs)

    def step(self, action_type: str, target: str):
        action = Action(action_type=action_type, target=target)
        result = self.env.step(action)
        return {
            "observation": _observation_to_dict(result.observation),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }


def _observation_to_dict(obs) -> dict:
    """
    Flatten an Observation pydantic model into a plain dict. Keeping this
    as one small shared helper (instead of repeating .model_dump() calls
    all over the codebase) means if the Observation schema changes later,
    there's exactly one place to update.
    """
    return {
        "state": obs.state.model_dump(),
        "available_actions": obs.available_actions,
        "current_blockers": obs.current_blockers,
        "last_action_result": obs.last_action_result,
        "last_action_error": obs.last_action_error,
        "steps_taken": obs.steps_taken,
        "done": obs.done,
    }