"""
training/env_integration/test_rendering.py
===========================================
What the model is shown. The generation-only pre-flight found that when the
prompt listed moves as "action_type:target" (the environment's own shorthand),
the model copied the shorthand into its answer instead of the JSON it is asked
for. The fix is to write the available moves, and the history of past moves, as
the same JSON object the model has to answer with. These tests pin that down:

  1. format_action() is the exact inverse of the parser: whatever it writes,
     the projection function reads back unchanged, on one line, and in the
     shape the prompt's own RESPONSE FORMAT example shows.
  2. Every move the environment lists, in every task and at every step of
     several full episodes, appears as JSON, matches the environment's own
     list one for one, and works as an answer when copied verbatim.
  3. The history shows each move as the model wrote it (JSON, one line per
     move), including a reply that could not be parsed, a move glued into one
     field, and text that would break a line if it were not escaped.
  4. JSON braces in the observation and history do not break the prompt
     templates (str.format treats braces specially).

No GPU, Ray or verl-agent needed (the manager falls back to a stub base class).

Run with (from the repo root):
    python training/env_integration/test_rendering.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import types

_repo_root = os.getcwd()
sys.path.insert(0, _repo_root)

from env.environment import VALID_ACTION_TYPES
from training.env_integration.envs import build_meridian_envs
from training.env_integration.manager import MeridianEnvironmentManager
from training.env_integration.projection import format_action, meridian_projection
from training.env_integration.prompts import build_initial_prompt, build_step_prompt

TASKS = ["easy", "medium", "hard", "crisis"]
ACTIONS_PREFIX = "Available actions: "


def parse_json_objects(text):
    """'{...}, {...}' -> [dict, dict]. Raises if anything between the objects is not ', '."""
    decoder, found, i = json.JSONDecoder(), [], 0
    while i < len(text):
        obj, i = decoder.raw_decode(text, i)
        found.append(obj)
        if text[i:i + 2] == ", ":
            i += 2
        else:
            assert i == len(text), f"unexpected text after a JSON object: {text[i:i + 30]!r}"
    return found


def make_manager(tasks, history_length=5):
    envs, names = build_meridian_envs(seed=0, env_num=len(tasks), group_n=1, task_pool=tasks, is_train=False)
    config = types.SimpleNamespace(env=types.SimpleNamespace(history_length=history_length))
    manager = MeridianEnvironmentManager(envs, meridian_projection, config, task_pool=tasks)
    obs, infos = manager.reset(kwargs={"task_names": names})
    return manager, obs, infos


def listed_moves(prompt):
    """The moves the prompt lists, as dicts. Exactly one 'Available actions:' line is expected, and
    it must be the last line of the observation (just before the closing question)."""
    lines = [ln for ln in prompt.split("\n") if ln.startswith(ACTIONS_PREFIX)]
    assert len(lines) == 1, f"expected exactly one Available actions line, found {len(lines)}"
    return parse_json_objects(lines[0][len(ACTIONS_PREFIX):])


def history_entries(prompt):
    """[(step number, dict, the line that follows it)] for every 'Step N: {...}' line of the history."""
    lines = prompt.split("\n")
    found = []
    for i, line in enumerate(lines):
        m = re.fullmatch(r"Step (\d+): (\{.*\})", line)
        if m:
            found.append((int(m.group(1)), json.loads(m.group(2)), lines[i + 1] if i + 1 < len(lines) else ""))
    return found


# ---------------------------------------------------------------------------
# 1. format_action is the inverse of the parser
# ---------------------------------------------------------------------------

def test_format_action_is_the_inverse_of_the_parser():
    # The wire format, pinned exactly: this is the spacing the pre-flight measured and the prompt shows.
    assert format_action("request_document", "passport") == '{"action_type": "request_document", "target": "passport"}'
    assert format_action("approve_hr") == '{"action_type": "approve_hr", "target": ""}'
    assert format_action("approve_hr", "") == format_action("approve_hr")

    awkward_targets = [
        "", "passport", "Germany", "a, b", "colon:inside", "{braces}", 'quote"inside', "back\\slash",
        "two\nlines", "tab\there", "caf" + chr(0xE9), "emoji " + chr(0x1F600),
    ]
    n = 0
    for action_type in sorted(VALID_ACTION_TYPES) + ["dance", "request_document:visa"]:
        for target in awkward_targets:
            text = format_action(action_type, target)
            assert "\n" not in text and "\r" not in text, f"not a single line: {text!r}"
            reply = f"<action>{text}</action>"
            assert meridian_projection([reply]) == ([(action_type, target)], [1]), (action_type, target)
            n += 1
    assert n == (len(VALID_ACTION_TYPES) + 2) * len(awkward_targets) and n > 100


def test_the_prompt_asks_for_the_same_shape_it_shows():
    # The RESPONSE FORMAT example in the rules and the lists of moves must be the same kind of object.
    _, _, infos = make_manager(["easy"])
    rules = build_initial_prompt(infos[0]["raw_observation"]["state"])
    assert f"<action>{format_action('...', '...')}</action>" in rules, "the response-format example changed shape"
    assert "{{" not in rules and "}}" not in rules, "template braces were not unescaped"


# ---------------------------------------------------------------------------
# 2. The available moves
# ---------------------------------------------------------------------------

def play_listed_moves(policy):
    """Play every task for up to 35 steps, each time answering with the move at position `policy(n)` of the
    list the model was shown (copied verbatim). Yields (task, prompt, environment's own list) at every step."""
    manager, obs, infos = make_manager(TASKS)
    names = [info["raw_observation"]["state"]["task_name"] for info in infos]
    done = [False] * len(TASKS)
    for _ in range(35):
        replies = []
        for i in range(len(TASKS)):
            if done[i]:
                replies.append("")
                continue
            native = infos[i]["raw_observation"]["available_actions"]
            yield names[i], obs["text"][i], native
            moves = listed_moves(obs["text"][i])
            replies.append("<action>" + json.dumps(moves[policy(len(moves))]) + "</action>" if moves else "")
        obs, _, dones, infos = manager.step(replies)
        done = [d or bool(x) for d, x in zip(done, dones)]
        if all(done):
            break


def test_every_listed_move_is_json_and_answerable_verbatim():
    seen_tasks, n_prompts, n_moves, n_untargeted = set(), 0, 0, 0
    for policy in (lambda n: 0, lambda n: n - 1, lambda n: n // 2):
        for task, prompt, native in play_listed_moves(policy):
            seen_tasks.add(task)
            n_prompts += 1
            moves = listed_moves(prompt)
            # one for one, in order, with the environment's own list as the reference
            expected = [tuple(item.strip().partition(":")[::2]) for item in native]
            assert [(m["action_type"], m["target"]) for m in moves] == expected, (task, native, moves)
            for m in moves:
                assert set(m) == {"action_type", "target"} and all(isinstance(v, str) for v in m.values()), m
                # copied verbatim as an answer, it is read back as exactly that move
                answer = "<action>" + json.dumps(m) + "</action>"
                assert meridian_projection([answer]) == ([(m["action_type"], m["target"])], [1]), m
                n_moves += 1
                n_untargeted += m["target"] == ""
            # nothing is left in the shorthand form anywhere in the action list
            line = [ln for ln in prompt.split("\n") if ln.startswith(ACTIONS_PREFIX)][0]
            assert not re.search(r"\b[a-z_]+:[A-Za-z_]+\b", line[len(ACTIONS_PREFIX):]), line
            # and the list is still the last line of the observation, right above the closing question
            tail = prompt.rstrip().split("\n")
            assert tail[-3] == line and tail[-2] == "", "the action list is no longer right above the closing question"
    assert seen_tasks == set(TASKS) and n_prompts > 100 and n_moves > 300 and n_untargeted > 20, \
        (seen_tasks, n_prompts, n_moves, n_untargeted)


# ---------------------------------------------------------------------------
# 3. The history
# ---------------------------------------------------------------------------

def reply(action_type, target=None):
    body = {"action_type": action_type} if target is None else {"action_type": action_type, "target": target}
    return "<action>" + json.dumps(body, separators=(",", ":")) + "</action>"


def test_history_shows_each_move_as_the_model_wrote_it():
    tricky_target = 'pass\nport "x"'
    script = [
        (reply("request_document", "passport"), ("request_document", "passport")),
        ("I think we should request the visa.",   ("malformed_response", "")),                 # unparseable reply
        (reply("approve_hr"),                     ("approve_hr", "")),                         # no target key at all
        (reply("dance", ""),                      ("dance", "")),                              # well-formed nonsense
        (reply("request_document:visa", ""),      ("request_document:visa", "")),              # move glued into one field
        (reply("request_document", tricky_target), ("request_document", tricky_target)),       # would break a line unescaped
        (reply("set_tax_id", 5),                  ("set_tax_id", "5")),                        # non-string target
    ]
    # Two environments run side by side, each playing the script from a different starting point, so a
    # history that was mixed up between environments of the same batch would show.
    offsets = [0, 3]
    manager, obs, _ = make_manager(["easy", "easy"], history_length=len(script))
    assert all(history_entries(p) == [] and "(none yet)" in p for p in obs["text"]), "no history at step 0"

    for step in range(1, len(script) + 1):
        moves = [script[(step - 1 + off) % len(script)] for off in offsets]
        obs, _, _, _ = manager.step([text for text, _ in moves])
        for env, off in enumerate(offsets):
            shown = history_entries(obs["text"][env])
            assert [n for n, _, _ in shown] == list(range(step)), (env, step, [n for n, _, _ in shown])
            # every history line is exactly format_action of what the projection parsed, and the result follows it.
            # (history_entries only matches a whole 'Step N: {...}' line, so a move that spilled onto a second
            # line would not match and the step numbers above would be wrong.)
            for (n, entry, following), k in zip(shown, range(step)):
                expected = script[(k + off) % len(script)][1]
                assert entry == json.loads(format_action(*expected)), (env, n, entry, expected)
                assert following.startswith("Result: "), (env, n, following)

    # the failed parse reads as the placeholder move, and the glued move shows exactly what was written
    last = {n: e for n, e, _ in history_entries(obs["text"][0])}      # environment 0 played the script in order
    assert last[1] == {"action_type": "malformed_response", "target": ""}
    assert last[4] == {"action_type": "request_document:visa", "target": ""}
    assert last[5] == {"action_type": "request_document", "target": tricky_target}
    assert 'pass\\nport \\"x\\"' in obs["text"][0], "the target is shown JSON-escaped, on one line"


def test_history_window_keeps_only_the_most_recent_moves():
    manager, obs, _ = make_manager(["easy"], history_length=3)
    for text in [reply("approve_hr"), reply("request_document", "passport"), reply("request_document", "visa"),
                 reply("request_document", "work_permit"), reply("verify_document", "passport")]:
        obs, _, _, _ = manager.step([text])
    shown = history_entries(obs["text"][0])
    # steps are numbered from 0, and the window ends at the move just made (step 4)
    assert [(n, e["action_type"], e["target"]) for n, e, _ in shown] == [
        (2, "request_document", "visa"), (3, "request_document", "work_permit"), (4, "verify_document", "passport"),
    ], shown


# ---------------------------------------------------------------------------
# 4. JSON braces and the templates
# ---------------------------------------------------------------------------

def test_json_braces_do_not_break_the_templates():
    _, _, infos = make_manager(["easy"])
    state = infos[0]["raw_observation"]["state"]
    move = format_action("request_document", "passport")
    observation = "Last action result: success\nAvailable actions: " + move + ", " + format_action("approve_hr")
    history = [{"action": format_action("approve_hr"), "result": "Last action result: success"},
               {"action": format_action("dance"), "result": "Last action FAILED: Unknown action_type 'dance'. Valid: ['a', 'b']"}]
    prompt = build_step_prompt(state, observation, history, 5, 2)         # must not raise (KeyError / ValueError from str.format)
    assert observation in prompt
    assert f"Step 0: {format_action('approve_hr')}\nResult: Last action result: success" in prompt
    assert f"Step 1: {format_action('dance')}\nResult: Last action FAILED" in prompt
    assert "{{" not in prompt and "}}" not in prompt, "doubled braces leaked into the prompt"
    assert "{rules_header}" not in prompt and "{current_observation}" not in prompt


if __name__ == "__main__":
    tests = [
        test_format_action_is_the_inverse_of_the_parser,
        test_the_prompt_asks_for_the_same_shape_it_shows,
        test_every_listed_move_is_json_and_answerable_verbatim,
        test_history_shows_each_move_as_the_model_wrote_it,
        test_history_window_keeps_only_the_most_recent_moves,
        test_json_braces_do_not_break_the_templates,
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