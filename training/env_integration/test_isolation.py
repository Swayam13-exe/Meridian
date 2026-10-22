"""
training/env_integration/test_isolation.py
===========================================
MeridianVecEnv steps all of its environments inside ONE process. Ray used
to give every environment its own process, so isolation between episodes
was enforced by the operating system. Now it has to be true by construction,
and these tests prove it:

  1. Episodes do not leak into each other -- each environment behaves exactly
     like a fresh, standalone environment given the same actions, no matter
     what its neighbours are doing.
  2. The shared module-level rule tables are never mutated, even under a
     noisy, many-environment stress run that fires the crisis task's
     regulatory event.
  3. close() works and the builder still accepts its old resource arguments.

Run with:
    python training/env_integration/test_isolation.py
"""

from __future__ import annotations

import copy
import os
import random
import sys
import time

_repo_root = os.getcwd()
_training_dir = os.path.join(_repo_root, "training")
sys.path.insert(0, _repo_root)
sys.path.insert(0, _training_dir)

from env_integration.envs import MeridianVecEnv, build_meridian_envs
from env.environment import WorkforceEnv, VALID_ACTION_TYPES
from env.models import Action

# A productive action script: enough to walk any task most of the way
# through, including acknowledging the crisis task's regulatory change.
SCRIPT = [
    ("request_document", "passport"), ("request_document", "visa"),
    ("request_document", "employment_letter"), ("request_document", "work_permit"),
    ("verify_document", "passport"), ("verify_document", "visa"),
    ("verify_document", "employment_letter"), ("verify_document", "work_permit"),
    ("acknowledge_regulatory_change", ""), ("approve_hr", ""),
    ("approve_legal", ""), ("approve_finance", ""),
    ("set_tax_id", "Germany"), ("set_payroll", "Germany"),
    ("resolve_conflict", ""), ("finalize_case", ""),
]
NOISE_TARGETS = ["passport", "visa", "employment_letter", "work_permit",
                 "Germany", "Singapore", "UAE", "HR", "Legal", "Finance", "", "nonsense"]


def _solo_trajectory(task: str, actions: list[tuple[str, str]]) -> list[dict]:
    """The reference: one fresh environment, run entirely by itself."""
    env = WorkforceEnv()
    env.reset(task)
    states = []
    for a_type, target in actions:
        result = env.step(Action(action_type=a_type, target=target))
        states.append(result.observation.state.model_dump())
    return states


# ---------------------------------------------------------------------------
# Test 1 — episodes do not leak into each other
# ---------------------------------------------------------------------------

def test_episodes_do_not_leak_into_each_other():
    """
    Two 'crisis' environments in the same process: one works productively
    through the script, the other flails with failing actions. Every state
    each one reports must equal that environment's own solo run.
    """
    n_steps = 14
    productive = SCRIPT[:n_steps]
    flailing = [("approve_finance", ""), ("finalize_case", ""), ("set_tax_id", "UAE")] * 5
    flailing = flailing[:n_steps]

    vec = MeridianVecEnv(num_envs=2)
    vec.reset(["crisis", "crisis"])
    shared_states = []
    for t in range(n_steps):
        _, _, _, infos = vec.step([productive[t], flailing[t]])
        shared_states.append([i["raw_observation"]["state"] for i in infos])

    for idx, actions in enumerate([productive, flailing]):
        solo = _solo_trajectory("crisis", actions)
        for t in range(n_steps):
            assert shared_states[t][idx] == solo[t], (
                f"env {idx} diverged from its solo run at step {t}: "
                f"another environment's activity leaked in"
            )

    final_a, final_b = shared_states[-1]
    assert final_a != final_b, "the two environments should have ended in different states"


# ---------------------------------------------------------------------------
# Test 2 — shared rule tables are never mutated
# ---------------------------------------------------------------------------

def _shared_tables() -> dict:
    import env.tasks as tasks
    import env.rules_engine as rules
    import env.reward as reward
    import env.validators as validators

    tables = {
        "tasks.TASKS": tasks.TASKS,
        "rules.COUNTRY_RULES": rules.COUNTRY_RULES,
        "rules.DEPARTMENT_DEPENDENCIES": rules.DEPARTMENT_DEPENDENCIES,
        "rules.REQUIRED_DOCUMENTS": rules.REQUIRED_DOCUMENTS,
        "rules.REQUIRED_COMPLIANCE": rules.REQUIRED_COMPLIANCE,
        "rules._load_country_rules()": rules._load_country_rules(),
        "rules._load_visa_types()": rules._load_visa_types(),
        "rules._load_tax_treaties()": rules._load_tax_treaties(),
        "reward.REWARDS": reward.REWARDS,
        "validators._COUNTRY_RULES": validators._COUNTRY_RULES,
    }
    # A renamed table must fail loudly, never make this test check nothing.
    for name, obj in tables.items():
        assert obj, f"shared table {name} is empty or missing"
    return tables


def test_shared_rule_tables_are_never_mutated():
    before = {k: copy.deepcopy(v) for k, v in _shared_tables().items()}

    rng = random.Random(2024)
    task_names = ["easy", "medium", "hard", "crisis"] * 2
    vec = MeridianVecEnv(num_envs=len(task_names))
    vec.reset(task_names)
    action_types = sorted(VALID_ACTION_TYPES) + ["malformed_response", "bogus_action"]

    for t in range(60):
        acts = []
        for i in range(len(task_names)):
            if rng.random() < 0.6:
                acts.append(SCRIPT[(t + i) % len(SCRIPT)])
            else:
                acts.append((rng.choice(action_types), rng.choice(NOISE_TARGETS)))
        vec.step(acts)

    after = _shared_tables()
    for name, snapshot in before.items():
        assert after[name] == snapshot, (
            f"shared table {name} was mutated during episodes -- "
            f"environments are no longer independent"
        )


# ---------------------------------------------------------------------------
# Test 3 — close() works, and the builder keeps its old arguments
# ---------------------------------------------------------------------------

def test_close_and_builder_compat():
    vec, names = build_meridian_envs(
        seed=0, env_num=2, group_n=2, task_pool=["easy", "crisis"], is_train=False,
        resources_per_worker={"num_cpus": 0.1, "num_gpus": 0},  # accepted, ignored
    )
    assert vec.num_envs == 4 and len(names) == 4
    vec.reset(names)
    vec.close()
    assert vec.envs == [], "close() should release the environment instances"


# ---------------------------------------------------------------------------
# Test 4 — realistic batch size is cheap (this is why Ray actors had to go)
# ---------------------------------------------------------------------------

def test_realistic_batch_size_is_cheap():
    # 16 distinct episodes x group of 8 = 128 environments. As Ray actors this
    # was 128 processes (~0.3 GB each); in-process it is just 128 small objects.
    vec, names = build_meridian_envs(
        seed=0, env_num=16, group_n=8,
        task_pool=["easy", "medium", "hard", "crisis"], is_train=True,
    )
    assert vec.num_envs == 128
    start = time.perf_counter()
    vec.reset(names)
    for t in range(5):
        text_obs, rewards, dones, infos = vec.step([SCRIPT[t]] * 128)
    elapsed = time.perf_counter() - start
    assert len(text_obs) == 128 and rewards.shape == (128,) and dones.shape == (128,)
    print(f"        (128 envs: reset + 5 batched steps in {elapsed * 1000:.0f} ms)")


if __name__ == "__main__":
    tests = [
        test_episodes_do_not_leak_into_each_other,
        test_shared_rule_tables_are_never_mutated,
        test_close_and_builder_compat,
        test_realistic_batch_size_is_cheap,
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

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)