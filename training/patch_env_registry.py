"""
training/patch_env_registry.py
================================
Registers "meridian" as a recognized environment for verl-agent's trainer,
without modifying verl-agent's own source.

Why this file exists at all: verl-agent's make_envs() is a hardcoded
if/elif chain over six environment names (see agent_system/environments/
env_manager.py) -- there's no plugin hook to add a seventh. Forking
verl-agent to add one would mean Meridian's training logic living inside
a fork of someone else's repo. Instead, this module wraps the ORIGINAL
make_envs in a new function that checks for "meridian" first and falls
through to the original for every other environment name, then swaps
that wrapped version in wherever verl's trainer actually looks it up.

Where it actually looks it up matters here, and isn't obvious: verl's
main_ppo.py does `from agent_system.environments import make_envs` --
that's a re-export in agent_system/environments/__init__.py, a SEPARATE
bound name from agent_system.environments.env_manager.make_envs. Patching
only the env_manager copy would silently do nothing, because the
re-exported name would still point at the original function. Both get
patched here to be safe, but the __init__-level one is the one that
actually matters, since it's what main_ppo.py imports at call time.

Usage -- must run BEFORE verl's training entrypoint is invoked:
    import patch_env_registry  # noqa: F401  (side-effecting import)
    # ... then launch training as normal (verl.trainer.main_ppo, etc.)
"""

from __future__ import annotations

from functools import partial

import agent_system.environments as _env_pkg
import agent_system.environments.env_manager as _env_manager_module
from agent_system.environments.env_manager import EnvironmentManagerBase

from training.env_integration.envs import build_meridian_envs
from training.env_integration.manager import MeridianEnvironmentManager
from training.env_integration.projection import meridian_projection

_original_make_envs = _env_manager_module.make_envs


def _make_meridian_envs(config):
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1

    task_pool = list(config.env.meridian.task_pool)

    _envs, _ = build_meridian_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        task_pool=task_pool,
        is_train=True,
    )
    _val_envs, _ = build_meridian_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        task_pool=task_pool,
        is_train=False,
    )

    projection_f = partial(meridian_projection)
    envs = MeridianEnvironmentManager(_envs, projection_f, config, task_pool=task_pool)
    val_envs = MeridianEnvironmentManager(_val_envs, projection_f, config, task_pool=task_pool)
    return envs, val_envs


def _patched_make_envs(config):
    if "meridian" in config.env.env_name.lower():
        return _make_meridian_envs(config)
    return _original_make_envs(config)


def apply_patch():
    """Idempotent: safe to call more than once."""
    _env_manager_module.make_envs = _patched_make_envs
    _env_pkg.make_envs = _patched_make_envs  # the name main_ppo.py actually imports


apply_patch()