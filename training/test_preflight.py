"""
training/test_preflight.py
===========================
Tests for training/preflight_generation.py. The point of that script is to be
trusted BEFORE spending GPU hours, so these tests check that its measurements
are right, not merely that it runs:

  1. Settings reach the model as claimed (thinking switch, suffix, token cap).
  2. Metrics are counted exactly (format-valid, truncated, tokens).
  3. A perfect policy registers as a success, and episodes stop counting
     steps once they end.
  4. A policy that never emits an action registers as zero signal.
  5. summarize() does its arithmetic correctly on hand-built records.
  6. Prompt lengths are tracked, and ONLY prompts strictly longer than the
     limit are flagged (training raises when length > max_prompt_length).
  7. Responses are sorted into valid / cut-off / bad-format samples.
  8. The plain-language warnings fire when they should and stay quiet when
     they should not.
  9. The vLLM adapter passes the right settings to the engine and converts its
     output correctly (checked against a fake engine: no GPU needed).
 10. The GPU probe parser and the "no GPU" path behave.
 11. The CLI (--dry-run) runs end to end and writes valid JSON and a text report.

No GPU, vLLM, transformers or downloads needed.

Run with:
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

def step_of(prompt: str) -> int:
    return int(re.search(r"CURRENT STEP: (\d+)", prompt).group(1))

def run(setting, generate_fn, tasks, replicas=2, max_steps=6, tokenizer=None, **kwargs):
    return pf.run_setting(setting, generate_fn, tokenizer or RecordingTokenizer(), tasks=tasks,
                          replicas=replicas, max_steps=max_steps, show_progress=False, **kwargs)

def rambling(prompts, max_tokens):
    return [pf.Generation("hmm let me think about it " * 30, "length", max_tokens) for _ in prompts]

# ---------------------------------------------------------------------------

def test_settings_reach_the_generator():
    seen = {}

    def spy(prompts, max_tokens):
        seen["prompts"], seen["max_tokens"] = prompts, max_tokens
        return [pf.Generation("no action", "stop", 3) for _ in prompts]

    by_name = {s.name: s for s in pf.DEFAULT_SETTINGS}
    expectations = {
        "think-512":       dict(tok=512,  empty_think=False, suffix=False),
        "think-1024":      dict(tok=1024, empty_think=False, suffix=False),
        "nothink-512":     dict(tok=512,  empty_think=True,  suffix=False),
        "softnothink-512": dict(tok=512,  empty_think=False, suffix=True),
    }
    assert set(by_name) == set(expectations), "default settings changed; update this test"
    for name, exp in expectations.items():
        tok = RecordingTokenizer()
        run(by_name[name], spy, ["easy"], replicas=2, max_steps=1, tokenizer=tok)
        assert seen["max_tokens"] == exp["tok"], f"{name}: wrong token cap {seen['max_tokens']}"
        assert all(("<EMPTY_THINK>" in p) == exp["empty_think"] for p in seen["prompts"]), f"{name}: thinking switch wrong"
        assert all(p.replace("ASSISTANT<EMPTY_THINK>", "").replace("ASSISTANT", "").endswith("/no_think]") == exp["suffix"]
                   for p in seen["prompts"]), f"{name}: /no_think suffix wrong"
        # Same call shape as verl-agent's rollout_loop: one user message, generation prompt on, no tokenizing.
        assert all(c["add_generation_prompt"] is True and c["tokenize"] is False for c in tok.calls)
        assert all(len(c["chat"]) == 1 and c["chat"][0]["role"] == "user" for c in tok.calls)

def test_metrics_are_counted_exactly():
    # Step parity decides the response: even steps = valid action, odd steps = rambling cut off at the cap.
    def alternating(prompts, max_tokens):
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
    assert o["mean_tokens"] == 55.0, o["mean_tokens"]            # (10 + 100) / 2
    assert set(summary["per_task"]) == set(tasks)
    assert o["top_actions"] == [("request_document", 16)], o["top_actions"]

def test_perfect_policy_registers_as_success():
    def perfect(prompts, max_tokens):
        return [pf.Generation(action_text(*EASY_WIN[step_of(p)]), "stop", 12) for p in prompts]

    # max_steps=20 > 13: episodes must stop counting once the environment ends them.
    run_result = run(pf.Setting("t", 512), perfect, ["easy"], replicas=3, max_steps=20)
    o = pf.summarize(run_result)["overall"]
    assert o["success"] == 1.0, o["success"]
    assert o["format_valid"] == 1.0 and o["truncated"] == 0.0
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
    assert o["success"] == 0.0 and o["progress"] == 0.0
    assert o["env_ok_given_valid"] is None            # no valid actions to judge
    assert o["reward_spread"] == 0.0, o["reward_spread"]   # identical failures: nothing for GRPO to compare
    assert o["top_actions"] == []

def test_summarize_arithmetic_on_known_records():
    def step(task, t, valid, env_error, reward, truncated, n_tokens, action, n_prompt, over):
        return {"task": task, "t": t, "format_valid": valid, "env_error": env_error, "reward": reward,
                "truncated": truncated, "n_tokens": n_tokens, "action_type": action,
                "n_prompt_tokens": n_prompt, "over_limit": over}

    steps = [
        step("easy", 0, True,  False,  0.2, False, 10, "a", 100, False),
        step("easy", 1, True,  True,  -0.3, False, 20, "b", 300, False),
        step("easy", 0, False, True,  -0.3, True,  30, "(unparseable)", 500, False),
        step("hard", 0, False, True,  -0.3, True,  40, "(unparseable)", 2500, True),
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

def test_prompt_length_is_tracked_and_flagged():
    # Prompt length by step: 100, 250, 251, 400. With a limit of 250 only the last two are over it:
    # training raises when length > max_prompt_length, so a prompt of exactly 250 is fine.
    lengths = [100, 250, 251, 400]

    def growing(prompts, max_tokens):
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

def test_sample_kinds_are_separated():
    def mixed(prompts, max_tokens):
        out = []
        for i, _ in enumerate(prompts):
            if i == 0:
                out.append(pf.Generation(action_text("request_document", "passport"), "stop", 10))
            elif i == 1:
                out.append(pf.Generation("still thinking about the passport", "length", max_tokens))
            else:
                out.append(pf.Generation("I would request the passport first.", "stop", 8))
        return out

    samples = run(pf.Setting("t", 256), mixed, ["easy"], replicas=3, max_steps=1)["samples"]
    assert set(samples) == {"valid", "cut_off", "bad_format"}
    assert samples["valid"]["finish_reason"] == "stop" and "request_document" in samples["valid"]["text"]
    assert samples["cut_off"]["finish_reason"] == "length" and samples["cut_off"]["n_tokens"] == 256
    assert samples["bad_format"]["finish_reason"] == "stop" and "<action>" not in samples["bad_format"]["text"]

def test_diagnose_flags_the_right_things():
    # 1. Rambling model: no parseable action, everything cut off, identical returns.
    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), rambling, ["easy", "hard"], replicas=3, max_steps=5)))
    joined = " | ".join(notes)
    assert "parseable" in joined and "cut off" in joined and "Reward spread" in joined, notes
    assert len(notes) == 3, notes

    # 2. Perfect model: nothing wrong except that every attempt is identical (a solved task teaches nothing new).
    def perfect(prompts, max_tokens):
        return [pf.Generation(action_text(*EASY_WIN[step_of(p)]), "stop", 12) for p in prompts]

    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), perfect, ["easy"], replicas=3, max_steps=20)))
    assert len(notes) == 1 and "Reward spread" in notes[0], notes

    # 3. Well-formed but meaningless moves: the environment rejects all of them.
    def nonsense(prompts, max_tokens):
        return [pf.Generation(action_text("dance", ""), "stop", 9) for _ in prompts]

    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), nonsense, ["easy"], replicas=2, max_steps=4)))
    assert any("rejects 100%" in n for n in notes), notes
    assert not any("parseable" in n for n in notes), notes

    # 4. A healthy mix (half the attempts play well, half play nonsense): no false alarms.
    def half_and_half(prompts, max_tokens):
        out = []
        for i, p in enumerate(prompts):       # nobody finishes within 6 steps, so position == episode
            out.append(pf.Generation(action_text(*EASY_WIN[step_of(p)]) if i % 2 == 0 else action_text("dance", ""),
                                     "stop", 12))
        return out

    notes = pf.diagnose(pf.summarize(run(pf.Setting("t", 512), half_and_half, ["easy"], replicas=2, max_steps=6)))
    assert notes == [], notes

def test_report_columns_stay_aligned():
    s = pf.summarize(run(pf.Setting("t", 512), rambling, ["easy", "crisis"], replicas=2, max_steps=3))
    assert len(pf.summary_line(s)) == len(pf.SUMMARY_HEADER), (pf.summary_line(s), pf.SUMMARY_HEADER)
    report = pf.format_report([s], [{"setting": "t", "samples": {"valid": None, "cut_off": None, "bad_format": None}}],
                              {"model": "m", "tasks": "easy,crisis", "replicas": 2, "max_steps": 3,
                               "history_length": 5, "temperature": 1.0, "gpu": "(none)", "dtype": "x",
                               "max_prompt_length": 2048})
    task_rows = [ln for ln in report.splitlines() if ln.startswith("t ") and ("easy" in ln or "crisis" in ln)]
    assert len(task_rows) == 2 and all(len(r) == len(pf.TASK_HEADER) for r in task_rows), task_rows

def test_select_settings():
    assert [s.name for s in pf.select_settings("all")] == [s.name for s in pf.DEFAULT_SETTINGS]
    assert [s.name for s in pf.select_settings("nothink-512, think-512")] == ["nothink-512", "think-512"]
    try:
        pf.select_settings("think-999")
    except SystemExit as e:
        assert "Unknown setting" in str(e)
    else:
        raise AssertionError("an unknown setting name must be rejected")

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
        with contextlib.redirect_stderr(err):
            code = pf.main(["--settings", "think-512"])
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
        gens = gen(["p0", "p1", "p2"], 512)
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

        # An explicit dtype wins; --in-process turns multiprocessing off; temperature is passed through.
        gen = build(t4, dtype="float32", in_process=True, temperature=0.7)
        assert seen["llm"]["dtype"] == "float32"
        assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
        gen(["p"], 64)
        assert seen["params"]["temperature"] == 0.7 and seen["params"]["max_tokens"] == 64
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

def test_unknown_task_is_rejected():
    try:
        pf.main(["--dry-run", "--tasks", "easy,bogus"])
    except SystemExit as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("an unknown task name must be rejected")

def test_empty_settings_are_rejected():
    try:
        pf.main(["--dry-run", "--settings", " , "])
    except SystemExit as e:
        assert "No settings" in str(e)
    else:
        raise AssertionError("an empty settings list must be rejected")

def test_output_paths_never_collide():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.txt")      # a .txt name for the JSON would let the report overwrite it
        with contextlib.redirect_stdout(io.StringIO()):
            code = pf.main(["--dry-run", "--replicas", "1", "--max-steps", "2", "--settings", "think-512",
                            "--no-progress", "--out", out])
        assert code == 0
        assert "summaries" in json.load(open(out))
        assert "PRE-FLIGHT" in open(out + ".report.txt").read()

def test_cli_dry_run_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.json")
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = pf.main(["--dry-run", "--replicas", "2", "--max-steps", "6", "--no-progress", "--out", out])
        assert code == 0
        names = [s.name for s in pf.DEFAULT_SETTINGS]
        data = json.load(open(out))
        assert [s["setting"] for s in data["summaries"]] == names
        assert set(data["samples"]) == set(data["episodes"]) == set(data["steps"]) == set(data["warnings"]) == set(names)
        for s in data["summaries"]:
            assert s["overall"]["episodes"] == 8
            assert 0.0 <= s["overall"]["format_valid"] <= 1.0
        assert all(len(v) > 0 for v in data["steps"].values())
        assert data["args"]["max_prompt_length"] == 2048 and data["args"]["temperature"] == 1.0

        report = open(os.path.join(d, "r.txt")).read()
        for needle in ("PRE-FLIGHT", "WARNINGS", "PER TASK", "SAMPLE RESPONSES", "HOW TO READ THIS"):
            assert needle in report, needle
        # every setting's row was printed live, then the report repeated at the end
        assert all(printed.getvalue().count(n) >= 2 for n in names)

if __name__ == "__main__":
    tests = [
        test_settings_reach_the_generator,
        test_metrics_are_counted_exactly,
        test_perfect_policy_registers_as_success,
        test_no_action_policy_means_zero_signal,
        test_summarize_arithmetic_on_known_records,
        test_prompt_length_is_tracked_and_flagged,
        test_sample_kinds_are_separated,
        test_diagnose_flags_the_right_things,
        test_report_columns_stay_aligned,
        test_select_settings,
        test_gpu_probe_parsing_and_failure_modes,
        test_no_gpu_stops_before_loading_anything,
        test_vllm_adapter_with_fake_engine,
        test_unknown_task_is_rejected,
        test_empty_settings_are_rejected,
        test_output_paths_never_collide,
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
        except Exception as e:
            print(f"  ERROR: {fn.__name__} -- {type(e).__name__}: {e}")
            failed += 1
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)
    sys.exit(1 if failed else 0)