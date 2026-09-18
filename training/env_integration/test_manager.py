"""
training/env_integration/test_manager.py
==========================================
Verifies the projection function and MeridianEnvironmentManager: response
parsing (well-formed, malformed, and edge cases), the reset()/step()
observation shape, valid/invalid action flagging, the 'won' flag used by
verl-agent's success_evaluator, and -- most importantly for our actual
research question -- that the GiGPO anchor genuinely reflects state:
identical for two fresh episodes of the same task, different once their
states diverge.

Run with:
    python training/env_integration/test_manager.py
"""

from __future__ import annotations

import sys
import os
import json

_repo_root = os.getcwd()
_training_dir = os.path.join(_repo_root, "training")
sys.path.insert(0, _repo_root)
sys.path.insert(0, _training_dir)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_repo_root, _training_dir, os.environ.get("PYTHONPATH", "")]
)

import ray

from env_integration.projection import meridian_projection
from env_integration.envs import build_meridian_envs
from env_integration.manager import MeridianEnvironmentManager


class _FakeEnvConfig:
    history_length = 5

class _FakeConfig:
    env = _FakeEnvConfig()


def _make_response(action_type: str, target: str) -> str:
    return f'<think>reasoning</think><action>{{"action_type": "{action_type}", "target": "{target}"}}</action>'


# ---------------------------------------------------------------------------
# Test 1 -- projection handles well-formed, malformed, and edge cases
# ---------------------------------------------------------------------------

def test_projection_parsing():
    texts = [
        _make_response("request_document", "passport"),
        '<action>{"action_type": "approve_hr"}</action>',   # no target key
        "no tags here at all",                                # missing tag entirely
        '<action>{"action_type": "x" "target": "y"}</action>', # malformed json
    ]
    actions, valids = meridian_projection(texts)
    assert valids == [1, 1, 0, 0], f"Unexpected validity: {valids}"
    assert actions[0] == ("request_document", "passport")
    assert actions[1] == ("approve_hr", "")


# ---------------------------------------------------------------------------
# Test 2 -- reset() returns the right shape, anchors match for identical
# fresh states
# ---------------------------------------------------------------------------

def test_reset_shape_and_anchor_equality():
    vec_env, _ = build_meridian_envs(seed=0, env_num=2, group_n=1, task_pool=["easy"], is_train=False)
    mgr = MeridianEnvironmentManager(vec_env, meridian_projection, _FakeConfig(), task_pool=["easy"])

    obs, infos = mgr.reset({"task_names": ["easy", "easy"]})
    assert set(obs.keys()) == {"text", "image", "anchor"}
    assert obs["image"] is None
    assert len(obs["text"]) == 2 and len(obs["anchor"]) == 2
    json.loads(obs["anchor"][0])  # must be valid JSON

    assert obs["anchor"][0] == obs["anchor"][1], "Two fresh episodes of the same task must anchor identically"


# ---------------------------------------------------------------------------
# Test 3 -- step() flags valid/invalid actions correctly, and anchors
# diverge once states actually differ
# ---------------------------------------------------------------------------

def test_step_validity_and_anchor_divergence():
    vec_env, _ = build_meridian_envs(seed=0, env_num=2, group_n=1, task_pool=["easy"], is_train=False)
    mgr = MeridianEnvironmentManager(vec_env, meridian_projection, _FakeConfig(), task_pool=["easy"])
    mgr.reset({"task_names": ["easy", "easy"]})

    obs, rewards, dones, infos = mgr.step([
        _make_response("request_document", "passport"),
        "not a valid response",
    ])
    assert infos[0]["is_action_valid"] == 1
    assert infos[1]["is_action_valid"] == 0
    assert rewards[0] > 0
    assert rewards[1] < 0
    assert obs["anchor"][0] != obs["anchor"][1], "Diverging actions must diverge the anchor"


# ---------------------------------------------------------------------------
# Test 4 -- a full episode through the manager reaches won=True
# ---------------------------------------------------------------------------

def test_full_episode_via_manager():
    vec_env, _ = build_meridian_envs(seed=0, env_num=1, group_n=1, task_pool=["easy"], is_train=False)
    mgr = MeridianEnvironmentManager(vec_env, meridian_projection, _FakeConfig(), task_pool=["easy"])
    mgr.reset({"task_names": ["easy"]})

    scripted = [
        ("request_document", "passport"), ("request_document", "visa"),
        ("request_document", "employment_letter"), ("request_document", "work_permit"),
        ("verify_document", "passport"), ("verify_document", "visa"),
        ("verify_document", "employment_letter"), ("verify_document", "work_permit"),
        ("approve_hr", ""), ("approve_legal", ""),
        ("set_tax_id", "Germany"), ("set_payroll", "Germany"),
        ("finalize_case", ""),
    ]

    done = False
    infos = None
    for a_type, target in scripted:
        if done:
            break
        _, _, dones, infos = mgr.step([_make_response(a_type, target)])
        done = bool(dones[0])

    assert done, "Episode should reach a terminal state"
    assert infos[0]["won"] is True, "won flag should be True on a successful case"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    tests = [
        test_projection_parsing,
        test_reset_shape_and_anchor_equality,
        test_step_validity_and_anchor_divergence,
        test_full_episode_via_manager,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  OK:  {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {test_fn.__name__} -- {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__} -- {type(e).__name__}: {e}")
            failed += 1

    ray.shutdown()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)
    if failed > 0:
        sys.exit(1)