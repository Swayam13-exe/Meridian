"""
training/env_integration/test_wrapper.py
=========================================
Verifies the Ray-based vectorized wrapper around WorkforceEnv: parallel
episodes stay isolated, batched step() returns correctly-shaped arrays,
and a full scripted episode actually reaches completion through the
wrapper (not just a single step).

Run with:
    python training/env_integration/test_wrapper.py
"""

from __future__ import annotations

import sys
import os

# Ray runs each worker as a SEPARATE process. A parent process's sys.path
# edits don't cross that boundary -- only environment variables do. So the
# import path has to go through PYTHONPATH, set before ray.init(), not
# just sys.path.insert() (which is enough for this driver script itself,
# but not for the worker processes Ray spawns to run WorkforceWorker).
_repo_root = os.getcwd()
_training_dir = os.path.join(_repo_root, "training")
sys.path.insert(0, _repo_root)
sys.path.insert(0, _training_dir)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_repo_root, _training_dir, os.environ.get("PYTHONPATH", "")]
)

import ray
import numpy as np

from env_integration.envs import build_meridian_envs


# ---------------------------------------------------------------------------
# Test 1 — parallel workers stay isolated
# ---------------------------------------------------------------------------

def test_parallel_isolation():
    vec_env, task_names = build_meridian_envs(
        seed=0, env_num=2, group_n=2, task_pool=["easy", "crisis"], is_train=False
    )
    assert task_names == ["easy", "easy", "crisis", "crisis"], (
        f"Expected easy/easy/crisis/crisis, got {task_names}"
    )

    text_obs, infos = vec_env.reset(task_names)
    assert len(text_obs) == 4 and len(infos) == 4

    reported = [info["raw_observation"]["state"]["task_name"] for info in infos]
    assert reported == task_names, f"Worker task mismatch: {reported} != {task_names}"


# ---------------------------------------------------------------------------
# Test 2 — batched step returns correctly-shaped arrays
# ---------------------------------------------------------------------------

def test_batched_step_shapes():
    vec_env, task_names = build_meridian_envs(
        seed=1, env_num=4, group_n=1, task_pool=["easy"], is_train=False
    )
    vec_env.reset(task_names)

    actions = [("request_document", "passport")] * 4
    text_obs, rewards, dones, infos = vec_env.step(actions)

    assert isinstance(rewards, np.ndarray) and rewards.shape == (4,)
    assert isinstance(dones, np.ndarray) and dones.shape == (4,)
    assert len(text_obs) == 4 and len(infos) == 4
    assert (rewards > 0).all(), f"Expected all positive rewards for a valid first action, got {rewards}"


# ---------------------------------------------------------------------------
# Test 3 — a full episode actually completes through the wrapper
# ---------------------------------------------------------------------------

def test_full_episode_completes():
    vec_env, task_names = build_meridian_envs(
        seed=2, env_num=1, group_n=1, task_pool=["easy"], is_train=False
    )
    vec_env.reset(task_names)

    scripted_actions = [
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
    for a_type, target in scripted_actions:
        if done:
            break
        _, rewards, dones, infos = vec_env.step([(a_type, target)])
        done = bool(dones[0])

    assert done, "Episode should have reached a terminal state"
    status = infos[0]["raw_observation"]["state"]["status"]
    assert status == "success", f"Expected status=success, got {status}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    tests = [
        test_parallel_isolation,
        test_batched_step_shapes,
        test_full_episode_completes,
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