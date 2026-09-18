"""
training/test_patch_registration.py
=====================================
Verifies the monkey-patch itself: that "meridian" configs get routed to
our MeridianEnvironmentManager, and everything else still falls through
to verl-agent's original make_envs.

This test needs verl-agent actually installed (torch and all) to import
agent_system.environments in the first place -- that's exactly the piece
that couldn't be verified in the sandbox this was built in (no torch
there; the construction logic underneath was already verified separately
against a real OmegaConf config). Run this once, after installing
verl-agent, before trusting the patch in a real training run:

    python training/test_patch_registration.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "training"))

from omegaconf import OmegaConf

import ray

import training.patch_env_registry as patch_mod
import agent_system.environments as env_pkg


def test_meridian_config_routes_to_our_manager():
    from training.env_integration.manager import MeridianEnvironmentManager

    config = OmegaConf.create({
        "env": {
            "env_name": "meridian/WorkforceEnv",
            "rollout": {"n": 2},
            "seed": 0,
            "meridian": {"task_pool": ["easy", "crisis"]},
            "history_length": 5,
        },
        "data": {"train_batch_size": 2, "val_batch_size": 2},
    })

    envs, val_envs = env_pkg.make_envs(config)
    assert isinstance(envs, MeridianEnvironmentManager)
    assert isinstance(val_envs, MeridianEnvironmentManager)

    # Actually exercise the built actors, not just check their type -- this
    # is a stronger test, and it also means every worker gets used at least
    # once rather than sitting requested-but-idle, which is what caused the
    # noisy "actor creation cancelled" messages on exit.
    obs, infos = envs.reset({"task_names": ["easy", "easy", "crisis", "crisis"]})
    assert len(obs["text"]) == 4
    val_obs, val_infos = val_envs.reset({"task_names": ["easy", "crisis"]})
    assert len(val_obs["text"]) == 2


def test_non_meridian_config_falls_through_to_original():
    """
    Doesn't need alfworld/sokoban's own dependencies installed -- swaps in
    a tracking stub for "the original make_envs" and confirms THAT gets
    called for a non-meridian name, rather than our meridian branch.
    """
    calls = []

    def stub_original(config):
        calls.append(config.env.env_name)
        return "stub_train_envs", "stub_val_envs"

    real_original = patch_mod._original_make_envs
    patch_mod._original_make_envs = stub_original
    patch_mod.apply_patch()  # re-apply so the patched closure picks up the stub
    try:
        config = OmegaConf.create({
            "env": {"env_name": "sokoban", "rollout": {"n": 1}, "seed": 0},
            "data": {"train_batch_size": 1, "val_batch_size": 1},
        })
        result = env_pkg.make_envs(config)
        assert result == ("stub_train_envs", "stub_val_envs")
        assert calls == ["sokoban"], f"Expected the original to be called with 'sokoban', got {calls}"
    finally:
        patch_mod._original_make_envs = real_original
        patch_mod.apply_patch()


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    tests = [test_meridian_config_routes_to_our_manager, test_non_meridian_config_falls_through_to_original]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK:  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {t.__name__} -- {type(e).__name__}: {e}")
            failed += 1

    ray.shutdown()

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)
    if failed:
        sys.exit(1)