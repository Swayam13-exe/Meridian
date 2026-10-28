"""
training/test_preflight.py
===========================
Tests for training/preflight_generation.py. The point of that script is to be
trusted BEFORE spending GPU hours, so these tests check that its measurements
are right, not merely that it runs:

  1. Settings reach the model as claimed (thinking switch, suffix, observation
     rewrite, token cap, temperature), and a suffix changes ONLY the end of the prompt.
  2. The format reminder's example is a real action the environment accepts, and the
     JSON rewrite turns the old action_type:target lists into exactly what the training
     code now writes, touches no other line, and leaves JSON that is already there alone.
  3. "name@temperature" parses, rejects nonsense, and never edits the shared defaults.
  4. Every response lands in exactly ONE outcome (valid, cut_off, empty, ...), the
     labels follow the documented precedence, and "valid" can never disagree with
     the real projection function that training uses.
  5. Metrics are counted exactly (format-valid, truncated, tokens), a perfect
     policy registers as a success, and a policy that never emits an action
     registers as zero signal.
  6. summarize() does its arithmetic correctly on hand-built records.
  7. Prompt lengths are tracked, and ONLY prompts strictly longer than the limit
     are flagged (training raises when length > max_prompt_length).
  8. Examples are capped and de-duplicated; every raw response is kept.
  9. The plain-language warnings fire when they should and stay quiet when they
     should not, and name the biggest cause of failure.
 10. The report has every section and its columns stay aligned (even for the
     longest setting name).
 11. The vLLM adapter passes the right settings to the engine, lets a setting's own
     temperature (including 0) win over the global one, and converts its output
     correctly (checked against a fake engine: no GPU needed).
 12. The GPU probe parser and the "no GPU" path behave.
 13. The CLI (--dry-run) runs end to end, writes valid JSON, a text report and a
     raw-responses file, creates a missing output folder up front, and keeps the
     results of finished settings if a later one crashes.

No GPU, vLLM, transformers or downloads needed.

Run with (from the repo root):
    python training/test_preflight.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from collections import Counter

_repo_root = os.getcwd()
sys.path.insert(0, _repo_root)

from training import preflight_generation as pf

EASY_WIN = [
    ("request_document", "passport"), ("request_document", "visa"),
    ("request_document", "employment_letter"), ("request_document", "work_permit"),
    ("verify_document", "passport"), ("verify_document", "visa"),
    ("verify_document", "employment_letter"), ("verify_document", "work_permit"),
    ("approve_hr", ""), ("approve_legal", ""),
    ("set_tax_id", "Germany"), ("set_payroll", "Germany"),
    ("finalize_case", ""),
]

# Characters that are awkward to write in a source file are built from their code points instead.
STRAY_CLOSER = chr(0x27E9)      # a closing bracket the real model once wrote in place of </action>
LONE_SURROGATE = chr(0xD83D)    # half of an emoji: not encodable as UTF-8 as it stands
WEIRD_TEXT = "caf" + chr(0xE9) + " " + STRAY_CLOSER + " " + chr(0x1F600) + " and a lone surrogate " + LONE_SURROGATE + " here"

def action_text(action_type: str, target: str) -> str:
    return f'<think>ok</think><action>{{"action_type": "{action_type}", "target": "{target}"}}</action>'

class RecordingTokenizer:
    """Mimics HF's apply_chat_template and exposes what it was asked to do."""
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat, add_generation_prompt=True, tokenize=False, **kwargs):
        self.calls.append({"chat": chat, "add_generation_prompt": add_generation_prompt,
                           "tokenize": tokenize, "kwargs": dict(kwargs)})
        think_off = kwargs.get("enable_thinking", True) is False
        return f"USER[{chat[0]['content']}]ASSISTANT{'<EMPTY_THINK>' if think_off else ''}"

    def user_messages(self):
        return [c["chat"][0]["content"] for c in self.calls]

def raw_first_observations(task, replicas, seed=0, history_length=5):
    """The step-0 observation text, built the way the training path builds it. No pre-flight code is
    involved in producing it, so it can serve as the reference for 'the prompt was left untouched'."""
    envs, task_names = pf.build_meridian_envs(seed=seed, env_num=1, group_n=replicas, task_pool=[task], is_train=False)
    config = types.SimpleNamespace(env=types.SimpleNamespace(history_length=history_length))
    manager = pf.MeridianEnvironmentManager(envs, pf.meridian_projection, config, task_pool=[task])
    obs, _ = manager.reset(kwargs={"task_names": task_names})
    return list(obs["text"])

def step_of(prompt: str) -> int:
    return int(re.search(r"CURRENT STEP: (\d+)", prompt).group(1))

def run(setting, generate_fn, tasks, replicas=2, max_steps=6, tokenizer=None, **kwargs):
    return pf.run_setting(setting, generate_fn, tokenizer or RecordingTokenizer(), tasks=tasks,
                          replicas=replicas, max_steps=max_steps, show_progress=False, **kwargs)

def rambling(prompts, setting):
    return [pf.Generation("hmm let me think about it " * 30, "length", setting.max_tokens) for _ in prompts]

def expect_exit(fn, needle):
    """fn() must stop with SystemExit whose message contains `needle`."""
    try:
        fn()
    except SystemExit as e:
        assert needle in str(e), f"expected {needle!r} in the exit message, got {str(e)!r}"
    else:
        raise AssertionError(f"expected SystemExit mentioning {needle!r}")

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# Every outcome once, plus the edge cases around each boundary: (text, truncated, expected label).
TAXONOMY_CASES = [
    ('<action>{"action_type": "request_document", "target": "passport"}</action>', False, "valid"),
    ('<think>so</think>\n<action>\n {"action_type": "approve_hr"} \n</action>', False, "valid"),
    ('<action>"first"</action> then <action>{"action_type": "approve_hr"}</action>', False, "valid"),
    # a complete action before the cap is still an action: 'valid' beats 'cut_off'
    ('<action>{"action_type": "approve_hr"}</action> and then it rambles on and on', True, "valid"),
    ("Okay so let me think about the passport " * 10, True, "cut_off"),
    ('<action>{"action_type": "request_document", "tar', True, "cut_off"),
    ("", True, "cut_off"),                       # hitting the cap outranks every other failure
    ("", False, "empty"),
    ("  \n\t ", False, "empty"),
    ("I would request the passport first.", False, "no_action_tag"),
    ('<Action>{"action_type": "approve_hr"}</Action>', False, "no_action_tag"),    # tags are case-sensitive
    ('<action {"action_type": "approve_hr"}', False, "no_action_tag"),
    ('<action>{"action_type": "request_document", "target": "visa"}' + STRAY_CLOSER, False, "unclosed_action"),
    ('<action>{"action_type": "approve_hr"}', False, "unclosed_action"),
    ('<action>{"action_type": "approve_hr"} </action', False, "unclosed_action"),
    ('<action>"request_document:visa"</action>', False, "not_an_object"),
    ("<action>request_document</action>", False, "not_an_object"),
    ("<action>[1, 2]</action>", False, "not_an_object"),
    ("<action></action>", False, "not_an_object"),
    ('<action>{"action_type": "approve_hr"} extra</action>', False, "not_an_object"),
    ("<action>{'action_type': 'approve_hr'}</action>", False, "bad_json"),
    ('<action>{"action_type": "approve_hr",}</action>', False, "bad_json"),
    ("<action>{action_type: approve_hr}</action>", False, "bad_json"),
    ('<action>{"target": "passport"}</action>', False, "no_action_type"),
    ('<action>{"action_type": ""}</action>', False, "no_action_type"),
    ('<action>{"action_type": 5}</action>', False, "no_action_type"),
    ('<action>{"action_type": null}</action>', False, "no_action_type"),
    ("<action>{}</action>", False, "no_action_type"),
]

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def test_settings_reach_the_generator():
    seen = {}

    def spy(prompts, setting):
        seen["prompts"], seen["setting"] = prompts, setting
        return [pf.Generation("no action", "stop", 3) for _ in prompts]

    by_name = {s.name: s for s in pf.DEFAULT_SETTINGS}
    expectations = {
        # name:             (token cap, thinking switched off?, text that must end the user message)
        "think-512":        (512,  False, ""),
        "think-1024":       (1024, False, ""),
        "nothink-512":      (512,  True,  ""),
        "softnothink-512":  (512,  False, "\n/no_think"),
        "nothink-fmt-512":  (512,  True,  pf.FORMAT_REMINDER),
        "nothink-ex-512":   (512,  True,  pf.FORMAT_EXAMPLE),
        "nothink-json-512": (512,  True,  ""),
    }
    rewrites = {"nothink-json-512": pf.actions_as_json}      # settings that rewrite the observation itself
    assert set(by_name) == set(expectations), "default settings changed; update this test"
    assert pf.FORMAT_REMINDER != pf.FORMAT_EXAMPLE
    assert pf.FORMAT_REMINDER not in pf.FORMAT_EXAMPLE and pf.FORMAT_EXAMPLE not in pf.FORMAT_REMINDER

    baseline = raw_first_observations("easy", replicas=2)     # the observation exactly as training builds it
    assert len(baseline) == 2 and all(b.strip() for b in baseline)
    assert baseline == raw_first_observations("easy", replicas=2), "the reference observation must be reproducible"

    for name, (tokens, think_off, suffix) in expectations.items():
        tok = RecordingTokenizer()
        run(by_name[name], spy, ["easy"], replicas=2, max_steps=1, tokenizer=tok)
        assert seen["setting"] is by_name[name], f"{name}: the generator was not given its own Setting"
        assert seen["setting"].max_tokens == tokens, f"{name}: wrong token cap {seen['setting'].max_tokens}"
        assert seen["setting"].temperature is None, f"{name}: defaults must leave the temperature to --temperature"
        assert all(("<EMPTY_THINK>" in p) == think_off for p in seen["prompts"]), f"{name}: thinking switch wrong"
        # A suffix changes ONLY the end of the user message; with no suffix the observation is untouched. The one
        # rewrite there is leaves today's observations alone, because the training code already writes JSON.
        rewritten = [rewrites[name](b) for b in baseline] if name in rewrites else baseline
        assert rewritten == baseline, f"{name}: the rewrite must not change an observation that is already JSON"
        expected = [b.rstrip() + suffix for b in rewritten] if suffix else rewritten
        assert tok.user_messages() == expected, f"{name}: user message differs from the observation + its suffix"
        # Same call shape as verl-agent's rollout_loop: one user message, generation prompt on, no tokenizing.
        assert all(c["add_generation_prompt"] is True and c["tokenize"] is False for c in tok.calls)
        assert all(len(c["chat"]) == 1 and c["chat"][0]["role"] == "user" for c in tok.calls)
        assert all(c["kwargs"] == by_name[name].chat_template_kwargs for c in tok.calls), name

    # Because that rewrite is a no-op on JSON, show it is really wired in: an old-style prompt (moves as
    # action_type:target) reaches the model as JSON under nothink-json-512 and stays as it was under nothink-512.
    old_style = to_shorthand(baseline[0])
    assert old_style != baseline[0] and "request_document:passport" in old_style
    plain_prompt = pf.build_chat_prompt(RecordingTokenizer(), baseline[0], by_name["nothink-512"])
    assert pf.build_chat_prompt(RecordingTokenizer(), old_style, by_name["nothink-json-512"]) == plain_prompt
    assert pf.build_chat_prompt(RecordingTokenizer(), old_style, by_name["nothink-512"]) != plain_prompt

def test_reminder_example_is_a_real_accepted_action():
    # The example we show the model must itself be something training would accept.
    match = re.search(r"<action>.*</action>", pf.FORMAT_EXAMPLE, re.S)
    assert match, "FORMAT_EXAMPLE contains no <action>...</action> example"
    example = match.group(0)
    assert pf.meridian_projection([example])[1][0], f"the example is not a valid action: {example}"
    assert pf.classify_response(example) == "valid"

    # ...and the environment accepts it as the opening move of every task, so the example never misleads.
    def play_example(prompts, setting):
        return [pf.Generation(example, "stop", 20) for _ in prompts]

    o = pf.summarize(run(pf.Setting("t", 64), play_example, pf.ALL_TASKS, replicas=2, max_steps=1))["overall"]
    assert o["format_valid"] == 1.0 and o["env_ok_given_valid"] == 1.0, o

    # The blank template is parseable too (the model copying it verbatim gets a valid-looking action).
    template = re.search(r"<action>.*</action>", pf.FORMAT_REMINDER, re.S).group(0)
    assert pf.meridian_projection([template])[1][0]

def parse_json_objects(text):
    """'{...}, {...}' -> [dict, dict]"""
    decoder, found, i = json.JSONDecoder(), [], 0
    while i < len(text):
        obj, i = decoder.raw_decode(text, i)
        found.append(obj)
        if text[i:i + 2] == ", ":
            i += 2
    return found

def short_form(d):
    """{"action_type": "approve_hr", "target": ""} -> approve_hr;  {..."request_document", "target": "visa"} -> request_document:visa"""
    return f"{d['action_type']}:{d['target']}" if d["target"] else d["action_type"]

def to_shorthand(prompt):
    """The prompt as the environment used to write it: every move as action_type:target (just action_type
    when it has no target), in the action list and in the history. A test-local inverse of what the training
    code writes now, used to build old-style prompts."""
    lines = []
    for line in prompt.split("\n"):
        history = re.fullmatch(r"(Step \d+): (\{.*\})", line)
        if line.startswith("Available actions: "):
            line = "Available actions: " + ", ".join(short_form(d) for d in parse_json_objects(line[len("Available actions: "):]))
        elif history:
            line = f"{history.group(1)}: {short_form(json.loads(history.group(2)))}"
        lines.append(line)
    return "\n".join(lines)

def test_actions_as_json_turns_the_old_shorthand_into_what_training_writes():
    # Real prompts from every task, several steps deep (so there is history), driven by a mix of good moves,
    # a no-target move, an unparseable reply and a well-formed nonsense move.
    cycle = [action_text("request_document", "passport"), "no action here", action_text("approve_hr", ""),
             action_text("set_tax_id", "Germany"), action_text("dance", "")]

    def mixed_moves(prompts, setting):
        return [pf.Generation(cycle[step_of(p) % len(cycle)], "stop", 9) for p in prompts]

    tok = RecordingTokenizer()
    run(pf.Setting("raw", 64), mixed_moves, pf.ALL_TASKS, replicas=1, max_steps=7, tokenizer=tok)
    prompts = tok.user_messages()
    assert len(prompts) == 4 * 7

    n_history = n_untargeted = n_targeted = 0
    for prompt in prompts:
        old = to_shorthand(prompt)                        # what the environment used to write
        assert old != prompt, "the old-style twin must differ from the real prompt, or this test proves nothing"
        for a, b in zip(prompt.split("\n"), old.split("\n")):     # the twin differs in the two kinds of line only
            assert a == b or a.startswith("Available actions: ") or re.match(r"Step \d+: ", a), (a, b)
        # old style in -> exactly the prompt training writes now, and nothing else touched; JSON in -> JSON out
        assert pf.actions_as_json(old) == prompt
        assert pf.actions_as_json(prompt) == prompt
        for line in prompt.split("\n"):
            if line.startswith("Available actions: "):
                body = line[len("Available actions: "):]
                assert not re.search(r"\b[a-z_]+:[A-Za-z_]+\b", body), f"shorthand in the action list: {body}"
                for d in parse_json_objects(body):        # every listed move is something the model can answer with, verbatim
                    answer = f"<action>{json.dumps(d)}</action>"
                    assert pf.meridian_projection([answer]) == ([(d["action_type"], d["target"])], [1]), d
                    n_untargeted += d["target"] == ""
                    n_targeted += d["target"] != ""
            elif re.match(r"Step \d+: ", line):
                n_history += 1
    assert n_history > 10 and n_untargeted > 0 and n_targeted > 0, (n_history, n_untargeted, n_targeted)
    assert any("malformed_response" in p for p in prompts), "the battery should include a failed step in the history"

    # text without those lines passes through unchanged; an empty action list stays empty
    assert pf.actions_as_json("nothing to rewrite here\nCURRENT STEP: 3") == "nothing to rewrite here\nCURRENT STEP: 3"
    assert pf.actions_as_json("Available actions: ") == "Available actions: "
    assert pf.actions_as_json("Available actions: approve_hr") == 'Available actions: {"action_type": "approve_hr", "target": ""}'
    # JSON is never split again: not at the commas inside a target, and not at the colon of a move glued into one field
    already = ('Available actions: {"action_type": "request_document", "target": "a, b"}, {"action_type": "approve_hr", "target": ""}\n'
               'Step 4: {"action_type": "request_document:visa", "target": ""}')
    assert pf.actions_as_json(already) == already

def test_select_settings():
    names = lambda spec: [s.name for s in pf.select_settings(spec)]
    assert names("all") == [s.name for s in pf.DEFAULT_SETTINGS]
    assert names("nothink-512, think-512") == ["nothink-512", "think-512"]

    # name@temperature: the same base at several temperatures is fine; names are normalised
    assert names("nothink-512,nothink-512@0.7,nothink-512@0.5") == ["nothink-512", "nothink-512@0.7", "nothink-512@0.5"]
    assert names("nothink-512@1") == ["nothink-512@1"] and names("nothink-512@0.50") == ["nothink-512@0.5"]
    plain, hot = pf.select_settings("nothink-512,nothink-512@0.7")
    base = {s.name: s for s in pf.DEFAULT_SETTINGS}["nothink-512"]
    assert plain.temperature is None and hot.temperature == 0.7
    assert (hot.max_tokens, hot.chat_template_kwargs, hot.user_suffix) == (base.max_tokens, base.chat_template_kwargs, base.user_suffix)
    assert pf.select_settings("nothink-json-512@0.7")[0].obs_transform is pf.actions_as_json, "@T must keep the rewrite"
    assert base.temperature is None and base.name == "nothink-512", "select_settings must not edit the shared defaults"
    greedy = pf.select_settings("nothink-512@0")[0]
    assert greedy.temperature == 0.0 and greedy.temperature is not None and greedy.name == "nothink-512@0"

    for bad, needle in [
        ("think-999", "Unknown setting"), ("think-999@0.7", "Unknown setting"),
        ("nothink-512@abc", "not a number"), ("nothink-512@", "not a number"),
        ("nothink-512@0.7@0.5", "not a number"),
        ("nothink-512@-0.5", "0 or more"), ("nothink-512@nan", "finite"), ("nothink-512@inf", "finite"),
        ("nothink-512,nothink-512", "more than once"), ("nothink-512@0.7,nothink-512@0.70", "more than once"),
    ]:
        expect_exit(lambda bad=bad: pf.select_settings(bad), needle)

# ---------------------------------------------------------------------------
# Why did a step fail?
# ---------------------------------------------------------------------------

def test_taxonomy_battery():
    assert {expected for _, _, expected in TAXONOMY_CASES} == set(pf.OUTCOMES), "the battery must hit every outcome"
    for text, truncated, expected in TAXONOMY_CASES:
        got = pf.classify_response(text, truncated)
        assert got == expected, f"{text!r} (truncated={truncated}): expected {expected}, got {got}"

def test_valid_label_agrees_with_the_real_projection():
    # 'valid' must be exactly what training's projection says, whatever the text, so the table can never
    # claim a rate that differs from episode/valid_action_ratio. Fuzz around a good action to hunt for gaps.
    good = '<action>{"action_type": "request_document", "target": "passport"}</action>'
    corpus = [text for text, _, _ in TAXONOMY_CASES] + [good]
    for i in range(len(good)):
        corpus.append(good[:i] + good[i + 1:])          # one character deleted
        corpus.append(good[:i] + "~" + good[i + 1:])    # one character replaced
    for text in corpus:
        for truncated in (False, True):
            label = pf.classify_response(text, truncated)
            assert label in pf.OUTCOMES, (text, truncated, label)
            projected = bool(pf.meridian_projection([text])[1][0])
            assert (label == "valid") == projected, f"{text!r} (truncated={truncated}): label {label}, projection {projected}"

def test_failures_are_counted_once_and_examples_are_capped():
    responses = [
        (action_text("request_document", "passport"), "stop"),              # valid
        ("hmm " * 50, "length"),                                            # cut_off
        ("", "stop"),                                                       # empty
        ("I would ask for the passport.", "stop"),                          # no_action_tag
        ("Maybe the visa.", "stop"),                                        # no_action_tag (a different text)
        ('<action>{"action_type": "approve_hr"}' + STRAY_CLOSER, "stop"),   # unclosed_action
        ('<action>"request_document:visa"</action>', "stop"),               # not_an_object
        ("<action>{'action_type': 'approve_hr'}</action>", "stop"),         # bad_json
        ('<action>{"target": "x"}</action>', "stop"),                       # no_action_type
    ]

    def scripted(prompts, setting):
        return [pf.Generation(text, reason, 1) for (text, reason), _ in zip(responses, prompts)]

    result = run(pf.Setting("t", 512), scripted, ["easy"], replicas=len(responses), max_steps=1)
    expected_order = ["valid", "cut_off", "empty", "no_action_tag", "no_action_tag", "unclosed_action",
                      "not_an_object", "bad_json", "no_action_type"]
    assert [s["category"] for s in result["steps"]] == expected_order
    assert [r["category"] for r in result["responses"]] == expected_order
    assert [r["text"] for r in result["responses"]] == [t for t, _ in responses], "every raw response is kept, in order"
    assert all(r["setting"] == "t" and r["task"] == "easy" and r["t"] == 0 for r in result["responses"])

    o = pf.summarize(result)["overall"]
    n = len(responses)
    assert abs(sum(o["outcomes"].values()) - 1.0) < 1e-9, "every step must be counted exactly once"
    for kind in pf.OUTCOMES:
        want = expected_order.count(kind) / n
        assert abs(o["outcomes"][kind] - want) < 1e-9, (kind, o["outcomes"][kind], want)
    kind, share = o["top_failure"]
    assert kind == "no_action_tag" and abs(share - 2 / n) < 1e-9, o["top_failure"]
    assert {k: len(v) for k, v in result["samples"].items()} == {
        "valid": 1, "cut_off": 1, "empty": 1, "no_action_tag": 2, "unclosed_action": 1,
        "not_an_object": 1, "bad_json": 1, "no_action_type": 1}

    # The cap: only the first MAX_EXAMPLES distinct examples are kept, but every response still is.
    def distinct(prompts, setting):
        return [pf.Generation(f"I would pick option number {i}.", "stop", 6) for i, _ in enumerate(prompts)]

    result = run(pf.Setting("t", 512), distinct, ["easy"], replicas=6, max_steps=1)
    assert pf.MAX_EXAMPLES == 3
    assert [x["text"] for x in result["samples"]["no_action_tag"]] == [f"I would pick option number {i}." for i in range(3)]
    assert len(result["responses"]) == 6

    # Duplicates are not shown twice (at low temperature the replicas of a task often write the same thing).
    def same(prompts, setting):
        return [pf.Generation("same words", "stop", 2) for _ in prompts]

    result = run(pf.Setting("t", 512), same, ["easy"], replicas=5, max_steps=2)
    assert len(result["samples"]["no_action_tag"]) == 1 and len(result["responses"]) == 10

def test_dry_run_generator_exercises_every_common_failure():
    gen = pf.make_dry_run_generate_fn(0)
    seen = Counter(pf.classify_response(g.text, g.finish_reason == "length")
                   for g in gen(["p"] * 400, pf.Setting("s", 512)))
    assert {"valid", "cut_off", "no_action_tag", "unclosed_action", "not_an_object"} <= set(seen), seen

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_metrics_are_counted_exactly():
    # Step parity decides the response: even steps = valid action, odd steps = rambling cut off at the cap.
    def alternating(prompts, setting):
        out = []
        for p in prompts:
            if step_of(p) % 2 == 0:
                out.append(pf.Generation(action_text("request_document", "passport"), "stop", 10))
            else:
                out.append(pf.Generation("Okay so let me think", "length", 100))
        return out

    tasks = ["easy", "medium", "hard", "crisis"]
    summary = pf.summarize(run(pf.Setting("t", 512), alternating, tasks, replicas=2, max_steps=4))
    o = summary["overall"]
    assert o["episodes"] == 8 and o["steps"] == 32, (o["episodes"], o["steps"])
    assert o["format_valid"] == 0.5, o["format_valid"]
    assert o["truncated"] == 0.5, o["truncated"]
    assert o["outcomes"]["valid"] == 0.5 and o["outcomes"]["cut_off"] == 0.5, o["outcomes"]
    assert o["top_failure"] == ("cut_off", 0.5), o["top_failure"]
    assert o["mean_tokens"] == 55.0, o["mean_tokens"]            # (10 + 100) / 2
    assert set(summary["per_task"]) == set(tasks)
    assert o["top_actions"] == [("request_document", 16)], o["top_actions"]

def test_perfect_policy_registers_as_success():
    def perfect(prompts, setting):
        return [pf.Generation(action_text(*EASY_WIN[step_of(p)]), "stop", 12) for p in prompts]

    # max_steps=20 > 13: episodes must stop counting once the environment ends them.
    run_result = run(pf.Setting("t", 512), perfect, ["easy"], replicas=3, max_steps=20)
    o = pf.summarize(run_result)["overall"]
    assert o["success"] == 1.0, o["success"]
    assert o["format_valid"] == 1.0 and o["truncated"] == 0.0
    assert o["outcomes"]["valid"] == 1.0 and o["top_failure"] is None, (o["outcomes"], o["top_failure"])
    assert o["env_ok_given_valid"] == 1.0, o["env_ok_given_valid"]
    assert all(e["steps"] == len(EASY_WIN) for e in run_result["episodes"]), [e["steps"] for e in run_result["episodes"]]
    assert o["steps"] == 3 * len(EASY_WIN)
    assert o["progress"] > 0 and o["mean_episode_reward"] > 0
    assert all(e["won"] and e["status"] == "success" for e in run_result["episodes"])

def test_no_action_policy_means_zero_signal():
    result = run(pf.Setting("t", 512), rambling, ["easy", "hard"], replicas=3, max_steps=5)
    s = pf.summarize(result)
    o = s["overall"]
    assert o["format_valid"] == 0.0 and o["truncated"] == 1.0
    assert o["outcomes"]["cut_off"] == 1.0 and o["top_failure"] == ("cut_off", 1.0), o["top_failure"]
    assert o["success"] == 0.0 and o["progress"] == 0.0
    assert o["env_ok_given_valid"] is None            # no valid actions to judge
    assert o["reward_spread"] == 0.0, o["reward_spread"]   # identical failures: nothing for GRPO to compare
    assert o["top_actions"] == []

def test_summarize_arithmetic_on_known_records():
    def step(task, t, valid, env_error, reward, truncated, n_tokens, action, n_prompt, over, category):
        return {"task": task, "t": t, "format_valid": valid, "env_error": env_error, "reward": reward,
                "truncated": truncated, "n_tokens": n_tokens, "action_type": action,
                "n_prompt_tokens": n_prompt, "over_limit": over, "category": category}

    steps = [
        step("easy", 0, True,  False,  0.2, False, 10, "a", 100, False, "valid"),
        step("easy", 1, True,  True,  -0.3, False, 20, "b", 300, False, "valid"),
        step("easy", 0, False, True,  -0.3, True,  30, "(unparseable)", 500, False, "cut_off"),
        step("hard", 0, False, True,  -0.3, True,  40, "(unparseable)", 2500, True, "cut_off"),
    ]
    episodes = [
        {"task": "easy", "steps": 2, "reward": -0.1, "won": True,  "status": "success", "progress_steps": 1},
        {"task": "easy", "steps": 1, "reward": -0.3, "won": False, "status": "in_progress", "progress_steps": 0},
        {"task": "hard", "steps": 1, "reward": -0.3, "won": False, "status": "in_progress", "progress_steps": 0},
    ]
    s = pf.summarize({"setting": "x", "max_tokens": 1, "seconds": 60.0, "steps": steps, "episodes": episodes, "samples": {}})
    o = s["overall"]
    assert o["format_valid"] == 0.5 and o["truncated"] == 0.5 and o["progress"] == 0.25
    assert o["env_ok_given_valid"] == 0.5            # 1 of the 2 valid steps was accepted
    assert o["mean_tokens"] == 25.0
    assert abs(o["success"] - 1 / 3) < 1e-9
    assert o["tasks_with_mixed_outcomes"] == 1       # easy had one win and one loss
    # spread: easy rewards [-0.1, -0.3] -> pstdev 0.1 ; hard has one episode -> 0 ; mean 0.05
    assert abs(o["reward_spread"] - 0.05) < 1e-9, o["reward_spread"]
    assert abs(s["per_task"]["easy"]["success"] - 0.5) < 1e-9
    assert o["mean_prompt_tokens"] == 850.0, o["mean_prompt_tokens"]     # (100 + 300 + 500 + 2500) / 4
    assert o["max_prompt_tokens"] == 2500 and o["over_limit_steps"] == 1
    assert s["per_task"]["easy"]["max_prompt_tokens"] == 500 and s["per_task"]["easy"]["over_limit_steps"] == 0
    assert s["per_task"]["hard"]["over_limit_steps"] == 1
    # outcomes: overall 2 valid + 2 cut_off of 4; easy 2 of 3 valid; hard all cut_off
    assert o["outcomes"]["valid"] == 0.5 and o["outcomes"]["cut_off"] == 0.5
    assert o["outcomes"]["empty"] == 0.0 and o["top_failure"] == ("cut_off", 0.5)
    assert abs(s["per_task"]["easy"]["outcomes"]["valid"] - 2 / 3) < 1e-9
    assert s["per_task"]["hard"]["outcomes"]["cut_off"] == 1.0

def test_prompt_length_is_tracked_and_flagged():
    # Prompt length by step: 100, 250, 251, 400. With a limit of 250 only the last two are over it:
    # training raises when length > max_prompt_length, so a prompt of exactly 250 is fine.
    lengths = [100, 250, 251, 400]

    def growing(prompts, setting):
        return [pf.Generation("no action", "stop", 3, n_prompt_tokens=lengths[step_of(p)]) for p in prompts]

    result = run(pf.Setting("t", 512), growing, ["easy"], replicas=2, max_steps=4, max_prompt_length=250)
    s = pf.summarize(result)
    o = s["overall"]
    assert o["steps"] == 8
    assert o["max_prompt_tokens"] == 400, o["max_prompt_tokens"]
    assert o["over_limit_steps"] == 4, o["over_limit_steps"]          # steps 2 and 3, for both episodes
    assert s["max_prompt_length"] == 250
    crash = [n for n in pf.diagnose(s) if n.startswith("CRASH RISK")]
    assert len(crash) == 1 and "250" in crash[0] and "400" in crash[0], crash

    ok = pf.summarize(run(pf.Setting("t", 512), growing, ["easy"], replicas=2, max_steps=2, max_prompt_length=250))
    assert ok["overall"]["over_limit_steps"] == 0 and ok["overall"]["max_prompt_tokens"] == 250
    assert not any(n.startswith("CRASH RISK") for n in pf.diagnose(ok))

def test_rewards_are_saved_without_float32_noise():
    # The environment hands back float32; saved numbers should read -0.3, not -0.30000001192092896.
    def nonsense(prompts, setting):
        return [pf.Generation(action_text("dance", ""), "stop", 9) for _ in prompts]

    result = run(pf.Setting("t", 512), nonsense, ["easy"], replicas=2, max_steps=3)
    assert {s["reward"] for s in result["steps"]} == {-0.3}, {s["reward"] for s in result["steps"]}
    assert {r["reward"] for r in result["responses"]} == {-0.3}

# ---------------------------------------------------------------------------
# Warnings and the report
# ---------------------------------------------------------------------------

def test_diagnose_flags_the_right_things():
    # 1. Rambling model: no parseable action, everything cut off, identical returns.
    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), rambling, ["easy", "hard"], replicas=3, max_steps=5)))
    joined = " | ".join(notes)
    assert "parseable" in joined and "cut off" in joined and "Reward spread" in joined, notes
    assert "Biggest cause: cut_off (100% of steps)." in notes[0], notes
    assert len(notes) == 3, notes

    # 2. A model that finishes its answer but always wraps a bare string: the cause is named, nothing is cut off.
    def bare_string(prompts, setting):
        return [pf.Generation('<action>"approve_hr"</action>', "stop", 6) for _ in prompts]

    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), bare_string, ["easy"], replicas=2, max_steps=3)))
    assert "Biggest cause: not_an_object (100% of steps)." in notes[0], notes
    assert not any("cut off" in n for n in notes), notes

    # 3. Perfect model: nothing wrong except that every attempt is identical (a solved task teaches nothing new).
    def perfect(prompts, setting):
        return [pf.Generation(action_text(*EASY_WIN[step_of(p)]), "stop", 12) for p in prompts]

    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), perfect, ["easy"], replicas=3, max_steps=20)))
    assert len(notes) == 1 and "Reward spread" in notes[0], notes

    # 4. Well-formed but meaningless moves: the environment rejects all of them.
    def nonsense(prompts, setting):
        return [pf.Generation(action_text("dance", ""), "stop", 9) for _ in prompts]

    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), nonsense, ["easy"], replicas=2, max_steps=4)))
    assert any("rejects 100%" in n for n in notes), notes
    assert not any("parseable" in n for n in notes), notes

    # 5. A healthy mix (half the attempts play well, half play nonsense): no false alarms.
    def half_and_half(prompts, setting):
        out = []
        for i, p in enumerate(prompts):       # nobody finishes within 6 steps, so position == episode
            out.append(pf.Generation(action_text(*EASY_WIN[step_of(p)]) if i % 2 == 0 else action_text("dance", ""),
                                     "stop", 12))
        return out

    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), half_and_half, ["easy"], replicas=2, max_steps=6)))
    assert notes == [], notes

def test_noise_margin():
    assert pf.noise_margin(0) is None
    assert abs(pf.noise_margin(100) - 9.8) < 1e-9       # 1.96 * 0.5 / sqrt(100), in points
    assert abs(pf.noise_margin(400) - 4.9) < 1e-9
    assert pf.noise_margin(560) < pf.noise_margin(96)

def test_report_has_every_section_and_aligned_columns():
    long_name = "softnothink-512@0.123457"           # the longest realistic name
    assert len(long_name) == pf.NAME_WIDTH
    assert pf.select_settings(long_name)[0].name == long_name

    def mixed(prompts, setting):
        out = []
        for i, p in enumerate(prompts):
            if i % 3 == 0:
                out.append(pf.Generation(action_text("request_document", "passport"), "stop", 14))
            elif i % 3 == 1:
                out.append(pf.Generation('<action>"approve_hr"</action>', "stop", 8))
            else:
                out.append(pf.Generation("thinking, thinking", "length", setting.max_tokens))
        return out

    result = run(pf.Setting(long_name, 64), mixed, ["easy", "crisis"], replicas=2, max_steps=3)
    s = pf.summarize(result)
    assert len(pf.summary_line(s)) == len(pf.SUMMARY_HEADER), (pf.summary_line(s), pf.SUMMARY_HEADER)
    assert len(pf.outcome_line(s)) == len(pf.OUTCOME_HEADER), (pf.outcome_line(s), pf.OUTCOME_HEADER)

    # a second, larger setting: the noise line must quote the SMALLEST one (the least certain number)
    bigger = run(pf.Setting("other", 64), mixed, ["easy", "crisis"], replicas=2, max_steps=5)
    s_bigger = pf.summarize(bigger)
    assert s_bigger["overall"]["steps"] == 20 and s["overall"]["steps"] == 12

    info = {"model": "m", "tasks": "easy,crisis", "replicas": 2, "max_steps": 3, "history_length": 5,
            "temperature": 1.0, "gpu": "(none)", "dtype": "x", "max_prompt_length": 2048}
    report = pf.format_report([s, s_bigger], [result, bigger], info)
    lines = report.splitlines()
    task_rows = [ln for ln in lines if ln.startswith(long_name + "  ") and ("easy" in ln or "crisis" in ln)]
    assert len(task_rows) == 2 and all(len(r) == len(pf.TASK_HEADER) for r in task_rows), task_rows
    assert pf.SUMMARY_HEADER in lines and pf.OUTCOME_HEADER in lines and pf.TASK_HEADER in lines
    for needle in ("PRE-FLIGHT", "WHY STEPS FAILED", "WARNINGS", "PER TASK", "MOST COMMON VALID ACTIONS",
                   "EXAMPLES", "WHAT THE MODEL SEES", "HOW TO READ THIS", "Calibration",
                   "a setting named ...@T overrides it", "no_type = valid JSON"):
        assert needle in report, needle
    # examples show what the model really wrote, JSON-escaped
    assert '"<action>\\"approve_hr\\"</action>"' in report and "[" + long_name + "] not_an_object" in report
    # the prompt tail shows what the model was given: the end of the user message and the thinking switch
    assert result["prompt_tail"] and result["prompt_tail"].strip().splitlines()[-1].strip() in report
    # the sampling-noise line is computed from the smallest setting (12 steps -> +/-28 points)
    assert "with 12 steps in the smallest setting" in report and "+/-28" in report, report

def test_prompt_tail_shows_what_the_model_sees():
    by_name = {s.name: s for s in pf.DEFAULT_SETTINGS}
    tails = {}
    for name in ("think-512", "nothink-512", "softnothink-512", "nothink-fmt-512", "nothink-ex-512", "nothink-json-512"):
        result = run(by_name[name], rambling, ["easy"], replicas=1, max_steps=1)
        tails[name] = result["prompt_tail"]
        assert 0 < len(result["prompt_tail"]) <= pf.PROMPT_TAIL_CHARS, name
    assert tails["think-512"].endswith("]ASSISTANT") and pf.FORMAT_REMINDER not in tails["think-512"]
    assert tails["nothink-512"].endswith("]ASSISTANT<EMPTY_THINK>") and pf.FORMAT_REMINDER not in tails["nothink-512"]
    assert "/no_think]" in tails["softnothink-512"]
    assert pf.FORMAT_REMINDER in tails["nothink-fmt-512"] and tails["nothink-fmt-512"].endswith("<EMPTY_THINK>")
    assert pf.FORMAT_EXAMPLE in tails["nothink-ex-512"]
    # every setting shows the actions the way the model must write them (JSON), not as action_type:target
    for name, tail in tails.items():
        assert '{"action_type": "request_document", "target": "passport"}' in tail, name
        assert "request_document:passport" not in tail, name

# ---------------------------------------------------------------------------
# GPU and vLLM plumbing
# ---------------------------------------------------------------------------

def test_gpu_probe_parsing_and_failure_modes():
    good = '{"name": "Tesla T4", "major": 7, "minor": 5, "total_gb": 15.0}'
    assert pf.parse_gpu_probe("a library warning\n" + good + "\n") == json.loads(good)
    assert pf.parse_gpu_probe("null\n") is None                      # torch imported, but no CUDA
    assert pf.parse_gpu_probe("") is None
    assert pf.parse_gpu_probe("Traceback...\nImportError: no torch\n") is None
    assert pf.parse_gpu_probe('{"name": "x"}') is None               # an incomplete record is not trusted

    real_run = subprocess.run
    try:
        subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=good + "\n", stderr="", returncode=0)
        gpu, detail = pf.probe_gpu()
        assert gpu["major"] == 7 and detail == "", (gpu, detail)

        subprocess.run = lambda *a, **k: types.SimpleNamespace(
            stdout="", stderr="Traceback (most recent call last):\nModuleNotFoundError: No module named 'torch'\n", returncode=1)
        gpu, detail = pf.probe_gpu()
        assert gpu is None and "torch" in detail, (gpu, detail)

        def hang(*a, **k):
            raise subprocess.TimeoutExpired("probe", 1)
        subprocess.run = hang
        gpu, detail = pf.probe_gpu()
        assert gpu is None and "Timeout" in detail, (gpu, detail)
    finally:
        subprocess.run = real_run

def test_no_gpu_stops_before_loading_anything():
    real_probe = pf.probe_gpu
    err = io.StringIO()
    try:
        pf.probe_gpu = lambda: (None, "ModuleNotFoundError: No module named 'torch'")
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err):
            code = pf.main(["--settings", "think-512", "--out", os.path.join(d, "r.json")])
            assert os.listdir(d) == [], "a run that never started must not leave files behind"
    finally:
        pf.probe_gpu = real_probe
    assert code == 2, code
    assert "No CUDA GPU" in err.getvalue() and "No module named 'torch'" in err.getvalue()

def test_vllm_adapter_with_fake_engine():
    seen = {}

    class FakeCompletion:
        def __init__(self, text, token_ids, finish_reason):
            self.text, self.token_ids, self.finish_reason = text, token_ids, finish_reason

    class FakeRequestOutput:
        def __init__(self, prompt_token_ids, completion):
            self.prompt_token_ids, self.outputs = prompt_token_ids, [completion]

    class FakeSamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    class FakeLLM:
        def __init__(self, **kw):
            seen["llm"] = kw

        def generate(self, prompts, params, use_tqdm=True):
            seen["params"], seen["use_tqdm"], seen["prompts"] = params.kw, use_tqdm, list(prompts)
            return [
                FakeRequestOutput([1, 2, 3],    FakeCompletion("resp0", [7] * 5, "stop")),
                FakeRequestOutput([1, 2, 3, 4], FakeCompletion("resp1", [7] * 9, "length")),
                FakeRequestOutput(None,         FakeCompletion("resp2", [7] * 2, None)),   # fields vLLM may leave empty
            ]

    t4 = {"name": "Tesla T4", "major": 7, "minor": 5, "total_gb": 15.0}
    a100 = {"name": "A100", "major": 8, "minor": 0, "total_gb": 40.0}
    env_keys = ["VLLM_ATTENTION_BACKEND", "VLLM_ENABLE_V1_MULTIPROCESSING"]
    saved_env = {k: os.environ.get(k) for k in env_keys}
    saved_module = sys.modules.get("vllm")
    fake = types.ModuleType("vllm")
    fake.LLM, fake.SamplingParams = FakeLLM, FakeSamplingParams
    sys.modules["vllm"] = fake

    def build(gpu, **kw):
        for k in env_keys:
            os.environ.pop(k, None)
        return pf.make_vllm_generate_fn("some/model", max_model_len=8192, max_num_seqs=16,
                                        gpu_memory_utilization=0.6, dtype=kw.pop("dtype", None),
                                        seed=7, gpu=gpu, **kw)
    try:
        # A T4: fp16 (no bf16 on compute 7.5), Triton attention, eager mode, settings passed straight through.
        gen = build(t4)
        kw = seen["llm"]
        assert kw["model"] == "some/model" and kw["dtype"] == "float16" and gen.dtype == "float16", kw
        assert kw["enforce_eager"] is True and kw["max_model_len"] == 8192 and kw["max_num_seqs"] == 16
        assert kw["gpu_memory_utilization"] == 0.6 and kw["seed"] == 7
        assert os.environ.get("VLLM_ATTENTION_BACKEND") == "TRITON_ATTN"
        assert "VLLM_ENABLE_V1_MULTIPROCESSING" not in os.environ

        # Output conversion: order kept, a missing finish_reason means the model stopped on its own,
        # a missing prompt id list counts as 0 instead of crashing.
        gens = gen(["p0", "p1", "p2"], pf.Setting("s", 512))
        assert seen["prompts"] == ["p0", "p1", "p2"] and seen["use_tqdm"] is False
        assert seen["params"] == {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512}, seen["params"]
        assert [g.text for g in gens] == ["resp0", "resp1", "resp2"]
        assert [g.finish_reason for g in gens] == ["stop", "length", "stop"]
        assert [g.n_tokens for g in gens] == [5, 9, 2]
        assert [g.n_prompt_tokens for g in gens] == [3, 4, 0]

        # An Ampere GPU: bf16, and no forced attention backend.
        gen = build(a100)
        assert seen["llm"]["dtype"] == "bfloat16" and gen.dtype == "bfloat16"
        assert "VLLM_ATTENTION_BACKEND" not in os.environ

        # An explicit dtype wins; --in-process turns multiprocessing off.
        gen = build(t4, dtype="float32", in_process=True, temperature=0.7)
        assert seen["llm"]["dtype"] == "float32"
        assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"

        # Temperature: the global value applies to a setting with none of its own; a setting's own value wins,
        # and 0 (greedy) is a real value, not "unset". The token cap always comes from the setting.
        gen(["p"], pf.Setting("s", 64))
        assert seen["params"]["temperature"] == 0.7 and seen["params"]["max_tokens"] == 64
        gen(["p"], pf.Setting("s", 96, temperature=0.3))
        assert seen["params"]["temperature"] == 0.3 and seen["params"]["max_tokens"] == 96
        gen(["p"], pf.Setting("s", 64, temperature=0.0))
        assert seen["params"]["temperature"] == 0.0, "temperature 0 must not fall back to the global value"
        assert seen["params"]["top_p"] == 1.0
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if saved_module is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved_module

# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------

def test_unknown_task_is_rejected():
    expect_exit(lambda: pf.main(["--dry-run", "--tasks", "easy,bogus"]), "bogus")

def test_empty_settings_are_rejected():
    expect_exit(lambda: pf.main(["--dry-run", "--settings", " , "]), "No settings")

def test_output_paths_never_collide():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.txt")      # a .txt name for the JSON would let the report overwrite it
        with contextlib.redirect_stdout(io.StringIO()):
            code = pf.main(["--dry-run", "--replicas", "1", "--max-steps", "2", "--settings", "think-512",
                            "--no-progress", "--out", out])
        assert code == 0
        assert "summaries" in json.load(open(out))
        assert "PRE-FLIGHT" in open(out + ".report.txt", encoding="utf-8").read()
        assert read_jsonl(os.path.join(d, "r_responses.jsonl")), "the raw responses go to their own file"

def test_output_folder_is_created_before_anything_slow():
    with tempfile.TemporaryDirectory() as d:
        expect_exit(lambda: pf.main(["--dry-run", "--out", d]), "file name")       # a folder is not a result file

        out = os.path.join(d, "a", "b", "r.json")            # neither folder exists yet
        with contextlib.redirect_stdout(io.StringIO()):
            code = pf.main(["--dry-run", "--replicas", "1", "--max-steps", "2", "--settings", "think-512",
                            "--no-progress", "--out", out])
        assert code == 0 and os.path.exists(out)
        assert os.path.exists(os.path.join(d, "a", "b", "r_responses.jsonl"))

        # A folder that cannot be made (a plain file is in the way) stops the run before the GPU is even probed.
        blocker = os.path.join(d, "blocker")
        open(blocker, "w").close()
        probed = []
        real_probe = pf.probe_gpu
        try:
            pf.probe_gpu = lambda: probed.append(1) or (None, "")
            expect_exit(lambda: pf.main(["--settings", "think-512", "--out", os.path.join(blocker, "x", "r.json")]),
                        "Cannot create the output folder")
        finally:
            pf.probe_gpu = real_probe
        assert probed == [], "the output location must be checked before the slow startup"

def test_odd_characters_in_model_output_never_break_the_files():
    real_make = pf.make_dry_run_generate_fn
    weird = WEIRD_TEXT

    def make_weird(seed):
        def gen(prompts, setting):
            return [pf.Generation(weird, "stop", 5) for _ in prompts]
        gen.dtype = "(dry run)"
        return gen

    pf.make_dry_run_generate_fn = make_weird
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                code = pf.main(["--dry-run", "--replicas", "1", "--max-steps", "2", "--no-progress",
                                "--settings", "think-512", "--out", out])
            assert code == 0
            lines = read_jsonl(os.path.join(d, "r_responses.jsonl"))
            assert lines and all(ln["text"] == WEIRD_TEXT for ln in lines)
            report = open(os.path.join(d, "r.txt"), encoding="utf-8").read()
            assert "caf" + chr(0xE9) in report and chr(92) + "ud83d" in report, "a lone surrogate must appear as an escape"
            report.encode("utf-8")                      # fine for a UTF-8 terminal or file
            printed.getvalue().encode("utf-8")
    finally:
        pf.make_dry_run_generate_fn = real_make

    # result files are opened as UTF-8 that never raises, whatever the machine's default encoding
    with tempfile.TemporaryDirectory() as d:
        with pf._open_utf8(os.path.join(d, "x.txt"), "w") as f:
            assert f.encoding.lower().replace("-", "") == "utf8" and f.errors == "backslashreplace", (f.encoding, f.errors)

def test_results_so_far_survive_a_crash_midway():
    real_make = pf.make_dry_run_generate_fn

    def make_flaky(seed):
        inner = real_make(seed)

        def flaky(prompts, setting):
            if setting.name == "think-1024":
                raise RuntimeError("the GPU fell over")
            return inner(prompts, setting)

        flaky.dtype = "(dry run)"
        return flaky

    pf.make_dry_run_generate_fn = make_flaky
    try:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "r.json")
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pf.main(["--dry-run", "--replicas", "1", "--max-steps", "3", "--no-progress",
                             "--settings", "think-512,think-1024", "--out", out])
            except RuntimeError as e:
                assert "fell over" in str(e)
            else:
                raise AssertionError("the crash should propagate")
            data = json.load(open(out))
            assert [s["setting"] for s in data["summaries"]] == ["think-512"]
            lines = read_jsonl(os.path.join(d, "r_responses.jsonl"))
            assert len(lines) == len(data["steps"]["think-512"]) > 0
            assert {ln["setting"] for ln in lines} == {"think-512"}
    finally:
        pf.make_dry_run_generate_fn = real_make

def test_cli_dry_run_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        responses_path = os.path.join(d, "r_responses.jsonl")
        argv = ["--dry-run", "--replicas", "2", "--max-steps", "6", "--no-progress", "--out", out]
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = pf.main(argv)
        assert code == 0
        names = [s.name for s in pf.DEFAULT_SETTINGS]
        data = json.load(open(out))
        assert [s["setting"] for s in data["summaries"]] == names
        assert (set(data["samples"]) == set(data["episodes"]) == set(data["steps"]) == set(data["warnings"])
                == set(data["prompt_tails"]) == set(names))
        for s in data["summaries"]:
            assert s["overall"]["episodes"] == 8
            assert 0.0 <= s["overall"]["format_valid"] <= 1.0
            assert abs(sum(s["overall"]["outcomes"].values()) - 1.0) < 1e-9     # one outcome per step
        assert all(len(v) > 0 for v in data["steps"].values())
        assert all("text" not in step for steps in data["steps"].values() for step in steps), "raw text lives in the .jsonl"
        assert data["args"]["max_prompt_length"] == 2048 and data["args"]["temperature"] == 1.0

        # every raw response, one per line, labelled
        total_steps = sum(len(v) for v in data["steps"].values())
        lines = read_jsonl(responses_path)
        assert len(lines) == total_steps, (len(lines), total_steps)
        assert {ln["setting"] for ln in lines} == set(names)
        assert all(ln["category"] in pf.OUTCOMES and "text" in ln for ln in lines)

        report = open(os.path.join(d, "r.txt"), encoding="utf-8").read()
        for needle in ("PRE-FLIGHT", "WHY STEPS FAILED", "WARNINGS", "PER TASK", "EXAMPLES",
                       "WHAT THE MODEL SEES", "HOW TO READ THIS"):
            assert needle in report, needle
        # every setting's row was printed live, then the report repeated at the end
        assert all(printed.getvalue().count(n) >= 2 for n in names)
        assert responses_path in printed.getvalue(), "the raw-responses file should be announced"

        # running again starts the .jsonl afresh instead of appending to the old one
        with contextlib.redirect_stdout(io.StringIO()):
            assert pf.main(argv) == 0
        again = json.load(open(out))
        assert len(read_jsonl(responses_path)) == sum(len(v) for v in again["steps"].values())

if __name__ == "__main__":
    tests = [
        test_settings_reach_the_generator,
        test_reminder_example_is_a_real_accepted_action,
        test_actions_as_json_turns_the_old_shorthand_into_what_training_writes,
        test_select_settings,
        test_taxonomy_battery,
        test_valid_label_agrees_with_the_real_projection,
        test_failures_are_counted_once_and_examples_are_capped,
        test_dry_run_generator_exercises_every_common_failure,
        test_metrics_are_counted_exactly,
        test_perfect_policy_registers_as_success,
        test_no_action_policy_means_zero_signal,
        test_summarize_arithmetic_on_known_records,
        test_prompt_length_is_tracked_and_flagged,
        test_rewards_are_saved_without_float32_noise,
        test_diagnose_flags_the_right_things,
        test_noise_margin,
        test_report_has_every_section_and_aligned_columns,
        test_prompt_tail_shows_what_the_model_sees,
        test_gpu_probe_parsing_and_failure_modes,
        test_no_gpu_stops_before_loading_anything,
        test_vllm_adapter_with_fake_engine,
        test_unknown_task_is_rejected,
        test_empty_settings_are_rejected,
        test_output_paths_never_collide,
        test_output_folder_is_created_before_anything_slow,
        test_odd_characters_in_model_output_never_break_the_files,
        test_results_so_far_survive_a_crash_midway,
        test_cli_dry_run_end_to_end,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  OK:  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {fn.__name__} -- {e}")
            failed += 1
        except (Exception, SystemExit) as e:
            print(f"  ERROR: {fn.__name__} -- {type(e).__name__}: {e}")
            failed += 1
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)
    sys.exit(1 if failed else 0)