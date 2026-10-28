"""
training/test_smoke_scripts.py
===============================
Checks on the two smoke-test scripts. Text only: nothing in them is executed.

  1. run_gigpo_smoke_test.sh promises to be identical to run_grpo_smoke_test.sh
     except for the lines that are really about the algorithm. If any other line
     differs, a GRPO-vs-GiGPO difference could come from something else, so this
     is checked rather than trusted.
  2. Both scripts switch Qwen3 thinking off the way the pre-flight measured it
     (+data.apply_chat_template_kwargs.enable_thinking=False).
  3. The settings the pre-flight measures with (prompt length limit, response cap,
     history length, rollout temperature) are the ones the scripts run with, so what
     the pre-flight reports is what training would see.

Run with (from the repo root):
    python training/test_smoke_scripts.py
"""

from __future__ import annotations

import os
import sys

_repo_root = os.getcwd()
sys.path.insert(0, _repo_root)

from training import preflight_generation as pf

GRPO = os.path.join(_repo_root, "training", "run_grpo_smoke_test.sh")
GIGPO = os.path.join(_repo_root, "training", "run_gigpo_smoke_test.sh")

# The only lines allowed to differ between the two scripts.
ALGORITHM_LINES = ("algorithm.adv_estimator=", "algorithm.gamma=", "algorithm.gigpo.", "gigpo_mode=", "trainer.experiment_name=")


def command_lines(path):
    """The script's lines without comments, blank lines, indentation or the trailing line-continuation backslash."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f.read().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line[:-1].rstrip() if line.endswith("\\") else line)
    return lines


def value_of(lines, key):
    found = [ln.split("=", 1)[1] for ln in lines if ln.startswith(key + "=")]
    assert len(found) == 1, f"{key} should be set exactly once, found {found}"
    return found[0]


def test_scripts_differ_only_in_the_algorithm_lines():
    grpo, gigpo = command_lines(GRPO), command_lines(GIGPO)
    split = lambda lines: ([ln for ln in lines if ln.startswith(ALGORITHM_LINES)],
                           [ln for ln in lines if not ln.startswith(ALGORITHM_LINES)])
    grpo_algo, grpo_rest = split(grpo)
    gigpo_algo, gigpo_rest = split(gigpo)
    assert grpo_rest == gigpo_rest, "a line outside the algorithm settings differs: " + \
        str([ln for ln in grpo_rest + gigpo_rest if (ln in grpo_rest) != (ln in gigpo_rest)])
    assert value_of(grpo_algo, "algorithm.adv_estimator") == "grpo"
    assert value_of(gigpo_algo, "algorithm.adv_estimator") == "gigpo"
    assert len(grpo_rest) > 40, "suspiciously few settings were compared"


def test_both_scripts_switch_thinking_off():
    flag = "+data.apply_chat_template_kwargs.enable_thinking=False"
    for path in (GRPO, GIGPO):
        assert command_lines(path).count(flag) == 1, f"{os.path.basename(path)} should contain {flag} exactly once"
    # and that is what the pre-flight setting it was chosen with does
    nothink = {s.name: s for s in pf.DEFAULT_SETTINGS}["nothink-512"]
    assert nothink.chat_template_kwargs == {"enable_thinking": False}


def test_preflight_measures_what_the_scripts_run():
    nothink = {s.name: s for s in pf.DEFAULT_SETTINGS}["nothink-512"]
    for path in (GRPO, GIGPO):
        lines = command_lines(path)
        name = os.path.basename(path)
        assert int(value_of(lines, "data.max_prompt_length")) == pf.TRAIN_MAX_PROMPT_LENGTH, name
        assert int(value_of(lines, "data.max_response_length")) == nothink.max_tokens, name
        assert int(value_of(lines, "env.history_length")) == 5, name           # the pre-flight's default --history-length
        assert value_of(lines, "data.truncation") == "'error'", name           # why max_prompt is checked at all
        # the pre-flight samples at temperature 1.0; the scripts must not override verl's default of 1.0
        assert not any(ln.startswith("actor_rollout_ref.rollout.temperature=") for ln in lines), name
        assert not any(ln.startswith(("actor_rollout_ref.rollout.top_p=", "actor_rollout_ref.rollout.top_k=")) for ln in lines), name


if __name__ == "__main__":
    tests = [
        test_scripts_differ_only_in_the_algorithm_lines,
        test_both_scripts_switch_thinking_off,
        test_preflight_measures_what_the_scripts_run,
    ]

    passed = failed = 0
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