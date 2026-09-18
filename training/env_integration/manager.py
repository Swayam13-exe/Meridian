"""
training/env_integration/manager.py
=====================================
MeridianEnvironmentManager -- ties the vectorized environment, the
projection function, and the prompt templates into the shape
verl-agent's trainer actually consumes.

Anchor design (this is the piece that matters most for our GiGPO
comparison, so it's worth stating explicitly): the anchor is a canonical
snapshot of DECISION-RELEVANT state only -- which documents are verified,
which departments have approved, which compliance items are set, any
unresolved conflicts, and whether a regulatory event has fired/been
acknowledged. It deliberately excludes: the case ID (just a label), task
metadata like employee/countries (static all episode), previous_actions
(that's what the dialogue history is for), and deadline_days (basically a
proxy for step count). Two trajectories that reached the same real-world
situation by different paths should produce the same anchor, which is
exactly what lets GiGPO group them for step-level credit assignment.
"""

from __future__ import annotations

import json
from typing import List

from training.env_integration.prompts import build_step_prompt

try:
    from agent_system.environments.base import EnvironmentManagerBase
except ImportError:
    # verl-agent isn't installed in every environment this code runs in
    # (e.g. a CPU-only sandbox used to test this file's own logic). The
    # real dependency is required to actually train -- this fallback only
    # exists so the manager's logic can be unit-tested without it.
    class EnvironmentManagerBase:
        def __init__(self, envs, projection_f, config):
            self.envs = envs
            self.projection_f = projection_f
            self.config = config


ANCHOR_FIELDS = [
    "documents", "departments", "compliance", "conflicts",
    "status", "regulatory_event_fired", "regulatory_event_acknowledged",
]


class MeridianEnvironmentManager(EnvironmentManagerBase):

    def __init__(self, envs, projection_f, config, task_pool: List[str] | None = None):
        super().__init__(envs, projection_f, config)
        self.task_pool = task_pool or ["easy", "medium", "hard", "crisis"]
        self.history_length = getattr(getattr(config, "env", object()), "history_length", 5)
        self._history: List[List[dict]] = []
        self._task_states: List[dict] = []
        self._step_counts: List[int] = []

    def reset(self, kwargs=None):
        task_names = kwargs.get("task_names") if kwargs else None
        if task_names is None:
            task_names = [self.task_pool[i % len(self.task_pool)] for i in range(self.envs.num_envs)]

        text_obs, infos = self.envs.reset(task_names)

        self._task_states = [info["raw_observation"]["state"] for info in infos]
        self._history = [[] for _ in infos]
        self._step_counts = [0 for _ in infos]

        full_text = [
            build_step_prompt(self._task_states[i], text_obs[i], [], self.history_length, 0)
            for i in range(len(infos))
        ]
        anchors = [_make_anchor(s) for s in self._task_states]

        return {"text": full_text, "image": None, "anchor": anchors}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        text_obs, rewards, dones, infos = self.envs.step(actions)

        for i in range(len(infos)):
            a_type, target = actions[i]
            self._history[i].append({
                "action": f"{a_type}:{target}" if target else a_type,
                "result": text_obs[i].splitlines()[0] if text_obs[i] else "",
            })
            self._step_counts[i] += 1
            state = infos[i]["raw_observation"]["state"]
            self._task_states[i] = state
            infos[i]["is_action_valid"] = valids[i]
            infos[i]["won"] = (state["status"] == "success")

        full_text = [
            build_step_prompt(
                self._task_states[i], text_obs[i], self._history[i],
                self.history_length, self._step_counts[i],
            )
            for i in range(len(infos))
        ]
        anchors = [_make_anchor(self._task_states[i]) for i in range(len(infos))]

        return {"text": full_text, "image": None, "anchor": anchors}, rewards, dones, infos


def _make_anchor(state: dict) -> str:
    canonical = {k: state.get(k) for k in ANCHOR_FIELDS}
    return json.dumps(canonical, sort_keys=True)