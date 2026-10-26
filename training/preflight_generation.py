"""
training/preflight_generation.py
=================================
Generation-only pre-flight check. Run it BEFORE any long training run.

The question it answers: "if I train this model on this task, will the very
first steps contain anything to learn from, and will the run survive?" The
smoke tests showed the plumbing works, but they also showed an untrained
Qwen3-0.6B emits a parseable action only ~12% of the time, and ~56-60% of its
responses are cut off at the token cap while still inside <think>. A multi-hour
run in that state would run fine and teach almost nothing. This script measures
that cheaply -- no training, no weight updates, just the base model playing the
environment -- across a few settings, so the recipe can be chosen with numbers
instead of hope.

It deliberately reuses the REAL pieces of the training path, so the numbers
mean what training would see:
  - the real MeridianEnvironmentManager (prompts, history, anchors),
  - the real projection function (what counts as a valid action),
  - the same chat-template call verl-agent makes (rollout_loop.py), including
    data.apply_chat_template_kwargs,
  - the same sampling (temperature 1.0, top_p 1.0, no top-k) and token caps.
Only the text generator is swapped for a standalone vLLM engine.

Settings compared by default:
  think-512        Qwen3 thinking on, 512-token cap      (what the smoke tests used)
  think-1024       thinking on, 1024-token cap           (does more room fix truncation?)
  nothink-512      thinking OFF via the chat template    (training equivalent:
                   +data.apply_chat_template_kwargs.enable_thinking=False)
  softnothink-512  thinking off via "/no_think" in the prompt (prompt-only switch)

Metrics (per step unless noted):
  fmt_valid     the response contained a parseable <action>{json}</action> block.
                This is what training logs as episode/valid_action_ratio.
  env_ok        of the format-valid steps, the fraction the environment accepted
                (no "Last action FAILED"): valid JSON is not the same as a sensible move.
  truncated     the response hit the token cap. The action comes AFTER the
                reasoning, so a cut-off response almost never contains one.
  progress      the environment gave a POSITIVE reward on the step (a real milestone
                or success), the clearest sign the model is doing something right.
  ep_reward     mean episode return from the ENVIRONMENT only. Training also
                subtracts invalid_action_penalty_coef (0.1) per invalid action.
  spread        mean over tasks of the std-dev of episode reward across the replicas
                of that task. GRPO/GiGPO learn from DIFFERENCES within a group; if
                spread is ~0 every attempt looks the same and there is nothing to
                learn from.
  max_prompt    longest prompt seen, in tokens. The training scripts use
                data.max_prompt_length=2048 with data.truncation='error', which makes
                verl-agent raise a RuntimeError the first time a prompt is longer. A
                run that is fine for 20 steps can still die at step 21, so this is
                checked here, where it costs minutes instead of hours.

Plain-language warnings are printed under the table. Their thresholds are
heuristics (named constants below), not laws of nature: they are there so a
quick skim is enough, and the numbers are always printed next to them.

Usage (on the GPU machine, from the repo root):
    python training/preflight_generation.py
    python training/preflight_generation.py --max-steps 12 --replicas 2 --settings think-512,nothink-512   # quick look
    python training/preflight_generation.py --dry-run     # no GPU/vLLM: tests the script itself
If vLLM refuses to start, add --in-process (runs its engine inside this process,
the way the training run does).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.env_integration.envs import build_meridian_envs
from training.env_integration.manager import MeridianEnvironmentManager
from training.env_integration.projection import meridian_projection

try:  # progress bars are a convenience, never a requirement
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

ALL_TASKS = ["easy", "medium", "hard", "crisis"]
ENV_STEP_CAP = 35                 # WorkforceEnv.MAX_STEPS: the environment's own episode limit
TRAIN_MAX_PROMPT_LENGTH = 2048    # data.max_prompt_length in the training scripts

# Heuristic thresholds behind the plain-language warnings.
LOW_FORMAT_VALID = 0.30   # below this, most steps earn only the invalid-action penalty
HIGH_TRUNCATED = 0.25     # above this, a quarter or more of the responses never reach an answer
LOW_ENV_OK = 0.50         # of well-formed actions, fewer than this are accepted by the environment
MIN_SPREAD = 0.01         # below this, attempts at a task are effectively identical

# ---------------------------------------------------------------------------
# Settings and the generator interface
# ---------------------------------------------------------------------------

@dataclass
class Setting:
    name: str
    max_tokens: int
    chat_template_kwargs: dict = field(default_factory=dict)  # same slot as data.apply_chat_template_kwargs
    user_suffix: str = ""                                     # appended to the user message

DEFAULT_SETTINGS: List[Setting] = [
    Setting("think-512", 512),
    Setting("think-1024", 1024),
    Setting("nothink-512", 512, chat_template_kwargs={"enable_thinking": False}),
    Setting("softnothink-512", 512, user_suffix="\n/no_think"),
]

@dataclass
class Generation:
    text: str
    finish_reason: str          # "stop" (model ended its answer) or "length" (hit the token cap)
    n_tokens: int
    n_prompt_tokens: int = 0    # prompt length as the engine tokenized it

# A generator takes N prompt strings and a token cap, and returns N Generations in order.
GenerateFn = Callable[[List[str], int], List[Generation]]

def select_settings(spec: str) -> List[Setting]:
    by_name = {s.name: s for s in DEFAULT_SETTINGS}
    if spec.strip().lower() == "all":
        return list(DEFAULT_SETTINGS)
    chosen = []
    for name in [n.strip() for n in spec.split(",") if n.strip()]:
        if name not in by_name:
            raise SystemExit(f"Unknown setting '{name}'. Choose from: {', '.join(by_name)} (or 'all').")
        chosen.append(by_name[name])
    return chosen

def build_chat_prompt(tokenizer, obs_text: str, setting: Setting) -> str:
    """Exactly what verl-agent's rollout loop does: the observation text becomes
    one user message, then apply_chat_template(add_generation_prompt=True, **kwargs)."""
    chat = [{"role": "user", "content": obs_text + setting.user_suffix}]
    return tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False, **setting.chat_template_kwargs
    )

# ---------------------------------------------------------------------------
# Running one setting
# ---------------------------------------------------------------------------

def run_setting(
    setting: Setting,
    generate_fn: GenerateFn,
    tokenizer,
    *,
    tasks: List[str],
    replicas: int,
    max_steps: int,
    history_length: int = 5,
    max_prompt_length: int = TRAIN_MAX_PROMPT_LENGTH,
    seed: int = 0,
    show_progress: bool = True,
) -> dict:
    """Play `replicas` episodes of every task with the base model, in lockstep,
    through the real manager. Returns raw per-step and per-episode records."""
    started = time.perf_counter()
    envs, task_names = build_meridian_envs(
        seed=seed, env_num=len(tasks), group_n=replicas, task_pool=tasks, is_train=False
    )
    config = SimpleNamespace(env=SimpleNamespace(history_length=history_length))
    manager = MeridianEnvironmentManager(envs, meridian_projection, config, task_pool=tasks)
    obs, _ = manager.reset(kwargs={"task_names": task_names})

    n = envs.num_envs
    active = [True] * n
    episodes = [
        {"task": task_names[i], "steps": 0, "reward": 0.0, "won": False,
         "status": "in_progress", "progress_steps": 0}
        for i in range(n)
    ]
    steps: List[dict] = []
    # First example of each kind of response, for eyeballing what the model actually writes.
    samples: Dict[str, Optional[dict]] = {"valid": None, "cut_off": None, "bad_format": None}

    bar = tqdm(range(max_steps), desc=setting.name, disable=not show_progress, leave=False)
    for t in bar:
        idx = [i for i in range(n) if active[i]]
        if not idx:
            break
        prompts = [build_chat_prompt(tokenizer, obs["text"][i], setting) for i in idx]
        gens = generate_fn(prompts, setting.max_tokens)
        assert len(gens) == len(idx), f"generator returned {len(gens)} outputs for {len(idx)} prompts"

        texts = [""] * n  # finished environments get an ignored placeholder
        for i, g in zip(idx, gens):
            texts[i] = g.text
        obs, rewards, dones, infos = manager.step(texts)

        for i, g in zip(idx, gens):
            info = infos[i]
            valid = bool(info["is_action_valid"])
            reward = float(rewards[i])
            truncated = g.finish_reason == "length"
            action_type = meridian_projection([g.text])[0][0][0] if valid else "(unparseable)"
            steps.append({
                "task": task_names[i], "t": t, "format_valid": valid,
                "env_error": bool(info["raw_observation"].get("last_action_error")),
                "reward": reward, "truncated": truncated,
                "n_tokens": g.n_tokens, "action_type": action_type,
                "n_prompt_tokens": g.n_prompt_tokens,
                "over_limit": g.n_prompt_tokens > max_prompt_length,
            })

            ep = episodes[i]
            ep["steps"] += 1
            ep["reward"] += reward
            ep["progress_steps"] += int(reward > 0)
            ep["status"] = info["raw_observation"]["state"]["status"]
            ep["won"] = bool(info["won"])
            if bool(dones[i]):
                active[i] = False

            kind = "valid" if valid else ("cut_off" if truncated else "bad_format")
            if samples[kind] is None:
                samples[kind] = {"task": task_names[i], "step": t, "finish_reason": g.finish_reason,
                                 "n_tokens": g.n_tokens, "text": g.text}
        if show_progress and hasattr(bar, "set_postfix"):
            bar.set_postfix(active=sum(active))

    return {
        "setting": setting.name, "max_tokens": setting.max_tokens,
        "chat_template_kwargs": dict(setting.chat_template_kwargs), "user_suffix": setting.user_suffix,
        "max_prompt_length": max_prompt_length,
        "seconds": time.perf_counter() - started,
        "episodes": episodes, "steps": steps, "samples": samples,
    }

# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------

def _rate(records: List[dict], pred) -> float:
    return sum(1 for r in records if pred(r)) / len(records) if records else 0.0

def _task_block(steps: List[dict], episodes: List[dict]) -> dict:
    valid = [s for s in steps if s["format_valid"]]
    return {
        "episodes": len(episodes),
        "steps": len(steps),
        "format_valid": _rate(steps, lambda s: s["format_valid"]),
        "env_ok_given_valid": (sum(1 for s in valid if not s["env_error"]) / len(valid)) if valid else None,
        "truncated": _rate(steps, lambda s: s["truncated"]),
        "mean_tokens": statistics.fmean(s["n_tokens"] for s in steps) if steps else 0.0,
        "progress": _rate(steps, lambda s: s["reward"] > 0),
        "mean_step_reward": statistics.fmean(s["reward"] for s in steps) if steps else 0.0,
        "mean_episode_reward": statistics.fmean(e["reward"] for e in episodes) if episodes else 0.0,
        "mean_episode_len": statistics.fmean(e["steps"] for e in episodes) if episodes else 0.0,
        "success": _rate(episodes, lambda e: e["won"]),
        "mean_prompt_tokens": statistics.fmean(s["n_prompt_tokens"] for s in steps) if steps else 0.0,
        "max_prompt_tokens": max((s["n_prompt_tokens"] for s in steps), default=0),
        "over_limit_steps": sum(1 for s in steps if s["over_limit"]),
    }

def summarize(run: dict) -> dict:
    steps, episodes = run["steps"], run["episodes"]
    overall = _task_block(steps, episodes)

    per_task, spreads, mixed = {}, [], 0
    for task in sorted({e["task"] for e in episodes}, key=lambda t: ALL_TASKS.index(t) if t in ALL_TASKS else 99):
        t_eps = [e for e in episodes if e["task"] == task]
        t_steps = [s for s in steps if s["task"] == task]
        per_task[task] = _task_block(t_steps, t_eps)
        rewards = [e["reward"] for e in t_eps]
        spreads.append(statistics.pstdev(rewards) if len(rewards) >= 2 else 0.0)
        wins = sum(1 for e in t_eps if e["won"])
        mixed += int(0 < wins < len(t_eps))

    overall["reward_spread"] = statistics.fmean(spreads) if spreads else 0.0
    overall["tasks_with_mixed_outcomes"] = mixed
    overall["top_actions"] = Counter(
        s["action_type"] for s in steps if s["format_valid"]
    ).most_common(5)
    overall["seconds"] = run["seconds"]
    return {"setting": run["setting"], "max_tokens": run["max_tokens"],
            "max_prompt_length": run.get("max_prompt_length", TRAIN_MAX_PROMPT_LENGTH),
            "overall": overall, "per_task": per_task}

def diagnose(summary: dict) -> List[str]:
    """Plain-language warnings for one setting. Heuristic: see the thresholds at the top."""
    o = summary["overall"]
    limit = summary.get("max_prompt_length", TRAIN_MAX_PROMPT_LENGTH)
    notes: List[str] = []
    if o["over_limit_steps"] > 0:
        notes.append(
            f"CRASH RISK: {o['over_limit_steps']} prompt(s) were longer than {limit} tokens "
            f"(longest: {o['max_prompt_tokens']}). With data.truncation='error', training stops with a "
            f"RuntimeError the first time that happens. Shorten env.history_length or raise data.max_prompt_length.")
    if o["format_valid"] < LOW_FORMAT_VALID:
        notes.append(
            f"Only {100 * o['format_valid']:.0f}% of steps contain a parseable action, so most steps "
            f"would earn nothing but the invalid-action penalty.")
    if o["truncated"] > HIGH_TRUNCATED:
        notes.append(
            f"{100 * o['truncated']:.0f}% of responses were cut off at the {summary['max_tokens']}-token cap "
            f"before reaching an answer.")
    ev = o["env_ok_given_valid"]
    if ev is not None and o["format_valid"] >= LOW_FORMAT_VALID and ev < LOW_ENV_OK:
        notes.append(
            f"The model writes well-formed actions, but the environment rejects {100 * (1 - ev):.0f}% of them "
            f"(wrong action name or target).")
    if o["reward_spread"] < MIN_SPREAD:
        notes.append(
            "Reward spread is about 0: every attempt at a task earns the same return, so group-relative "
            "advantages (GRPO and GiGPO alike) would be about 0 and nothing would be learned from this setting.")
    return notes

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"

def _clip(text: str, limit: int = 420) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]} ... [{len(text) - limit} chars omitted] ... {text[-half:]}"

# One list drives the header AND the rows, so they cannot drift out of alignment.
_COLUMNS: List[Tuple[str, int]] = [
    ("setting", 16), ("steps", 6), ("fmt_valid", 9), ("env_ok", 7), ("truncated", 9), ("tok/resp", 8),
    ("progress", 8), ("ep_reward", 9), ("success", 7), ("spread", 6), ("max_prompt", 10), ("time", 6),
]

_TASK_COLUMNS: List[Tuple[str, int]] = [
    ("setting", 16), ("task", 6), ("steps", 6), ("fmt_valid", 9), ("env_ok", 7),
    ("truncated", 9), ("ep_reward", 9), ("success", 7),
]

def _row(cells: List[str], columns: List[Tuple[str, int]] = _COLUMNS) -> str:
    assert len(cells) == len(columns), (len(cells), len(columns))
    parts = []
    for i, (cell, (_, width)) in enumerate(zip(cells, columns)):
        parts.append(cell.ljust(width) if i == 0 else cell.rjust(width))
    return "  ".join(parts)

SUMMARY_HEADER = _row([name for name, _ in _COLUMNS])
TASK_HEADER = _row([name for name, _ in _TASK_COLUMNS], _TASK_COLUMNS)

def summary_line(s: dict) -> str:
    o = s["overall"]
    return _row([
        s["setting"], str(o["steps"]), _pct(o["format_valid"]), _pct(o["env_ok_given_valid"]),
        _pct(o["truncated"]), f"{o['mean_tokens']:.0f}", _pct(o["progress"]),
        f"{o['mean_episode_reward']:.2f}", _pct(o["success"]), f"{o['reward_spread']:.2f}",
        str(o["max_prompt_tokens"]), f"{o['seconds'] / 60:.1f}m",
    ])

_SAMPLE_TITLES = {
    "valid": "a response with a parseable action",
    "cut_off": "a response cut off at the token cap",
    "bad_format": "a response that finished but had no parseable action",
}

def format_report(summaries: List[dict], runs: List[dict], args_info: dict) -> str:
    out: List[str] = []
    rule = "=" * len(SUMMARY_HEADER)
    out.append(rule)
    out.append("PRE-FLIGHT: base-model generation check (no training)")
    out.append(f"model={args_info['model']}  tasks={args_info['tasks']}  replicas/task={args_info['replicas']}  "
               f"max_steps={args_info['max_steps']}  history={args_info['history_length']}  "
               f"temperature={args_info['temperature']}")
    out.append(f"gpu={args_info['gpu']}  dtype={args_info['dtype']}  max_prompt_length={args_info['max_prompt_length']}")
    out.append(rule)
    out.append(SUMMARY_HEADER)
    out.append("-" * len(SUMMARY_HEADER))
    out.extend(summary_line(s) for s in summaries)

    out.append("")
    out.append("WARNINGS (heuristic; the numbers above are what to trust)")
    for s in summaries:
        notes = diagnose(s)
        if not notes:
            out.append(f"  [{s['setting']}] none")
            continue
        for note in notes:
            out.append(f"  [{s['setting']}] {note}")

    out.append("")
    out.append("PER TASK")
    out.append(TASK_HEADER)
    out.append("-" * len(TASK_HEADER))
    for s in summaries:
        for task, b in s["per_task"].items():
            out.append(_row([
                s["setting"], task, str(b["steps"]), _pct(b["format_valid"]), _pct(b["env_ok_given_valid"]),
                _pct(b["truncated"]), f"{b['mean_episode_reward']:.2f}", _pct(b["success"]),
            ], _TASK_COLUMNS))

    out.append("")
    out.append("MOST COMMON VALID ACTIONS")
    for s in summaries:
        out.append(f"  {s['setting']:<16} " + (", ".join(f"{a} x{c}" for a, c in s["overall"]["top_actions"]) or "(none)"))

    out.append("")
    out.append("SAMPLE RESPONSES (first of each kind, shortened)")
    for run in runs:
        for kind in ("valid", "cut_off", "bad_format"):
            smp = run["samples"][kind]
            if smp is None:
                out.append(f"  [{run['setting']}] no example of {_SAMPLE_TITLES[kind]} occurred")
                continue
            out.append(f"  [{run['setting']}] {_SAMPLE_TITLES[kind]}  (task={smp['task']}, step={smp['step']}, "
                       f"finish={smp['finish_reason']}, tokens={smp['n_tokens']}):")
            out.append("      " + _clip(smp["text"]).replace("\n", "\n      "))

    out.append("")
    out.append("HOW TO READ THIS")
    out.append("  * Calibration: the training smoke runs (think-512, Qwen3-0.6B) logged valid_action_ratio 0.10-0.125 and")
    out.append("    ~56-60% of responses cut off at the cap, from only ~32 steps each. If the think-512 row is far from that")
    out.append("    (more than ~10 points), this harness is not matching training.")
    out.append("  * fmt_valid is the single most important column: below ~30%, most steps earn only the invalid-action penalty.")
    out.append("  * spread near 0 means all attempts at a task look alike, so group-relative methods (GRPO, GiGPO) get no signal.")
    out.append("  * progress > 0 means the model is already earning milestone rewards without any training.")
    out.append("  * ep_reward is the environment's reward only; training also subtracts 0.1 per invalid action.")
    out.append("  * max_prompt above the limit in the header would crash training (data.truncation='error').")
    out.append("  * nothink-512 maps to +data.apply_chat_template_kwargs.enable_thinking=False in the training script;")
    out.append("    softnothink-512 needs a one-line change to the prompt instead.")
    return "\n".join(out)

# ---------------------------------------------------------------------------
# GPU probe and generators: real (vLLM) and fake (dry run)
# ---------------------------------------------------------------------------

_GPU_PROBE_CODE = (
    "import json, torch\n"
    "if not torch.cuda.is_available():\n"
    "    print(json.dumps(None))\n"
    "else:\n"
    "    p = torch.cuda.get_device_properties(0)\n"
    "    print(json.dumps({'name': p.name, 'major': p.major, 'minor': p.minor,\n"
    "                      'total_gb': round(p.total_memory / 2**30, 1)}))\n"
)

def parse_gpu_probe(stdout: str) -> Optional[dict]:
    """The probe prints one JSON line last; anything before it (library warnings) is ignored."""
    for line in reversed([ln.strip() for ln in stdout.splitlines() if ln.strip()]):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data is None:
            return None
        if isinstance(data, dict) and {"name", "major", "minor", "total_gb"} <= set(data):
            return data
    return None

def probe_gpu() -> Tuple[Optional[dict], str]:
    """Ask a throwaway subprocess about the GPU, so CUDA is never initialised in THIS process:
    vLLM switches to a slower, fussier start method when its parent already holds a CUDA context."""
    try:
        done = subprocess.run([sys.executable, "-c", _GPU_PROBE_CODE],
                              capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{type(e).__name__}: {e}"
    gpu = parse_gpu_probe(done.stdout)
    detail = "" if gpu else (done.stderr.strip().splitlines() or [""])[-1]
    return gpu, detail

def make_vllm_generate_fn(
    model: str,
    *,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    dtype: Optional[str],
    seed: int,
    gpu: Optional[dict],
    in_process: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> GenerateFn:
    """A standalone vLLM engine configured like the training rollout: same dtype
    rule, same eager mode, T4-safe attention backend, temperature 1.0, top_p 1.0."""
    pre_ampere = bool(gpu) and gpu["major"] < 8       # a T4 (7.5) has no bf16 and no FlashAttention-2
    if dtype is None:
        dtype = "float16" if pre_ampere else ("bfloat16" if gpu else "auto")
    if pre_ampere:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    if in_process:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # what the training run does via external_launcher

    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization, enforce_eager=True,
              max_num_seqs=max_num_seqs, max_num_batched_tokens=4096, seed=seed)

    def generate(prompts: List[str], max_tokens: int) -> List[Generation]:
        params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        outputs = llm.generate(prompts, params, use_tqdm=False)   # returned in prompt order
        gens = []
        for o in outputs:
            c = o.outputs[0]
            gens.append(Generation(
                text=c.text,
                finish_reason=c.finish_reason or "stop",
                n_tokens=len(c.token_ids),
                n_prompt_tokens=len(o.prompt_token_ids) if o.prompt_token_ids is not None else 0,
            ))
        return gens

    generate.dtype = dtype  # type: ignore[attr-defined]  # reported in the header
    return generate

class _DryRunTokenizer:
    """Stands in for a HF tokenizer so --dry-run needs no downloads."""
    def apply_chat_template(self, chat, add_generation_prompt=True, tokenize=False, **kwargs):
        flag = "" if kwargs.get("enable_thinking", True) else "<think></think>"
        return f"<|user|>{chat[0]['content']}<|assistant|>{flag}"

def make_dry_run_generate_fn(seed: int) -> GenerateFn:
    """A canned 'model' for testing the script itself: half well-formed actions,
    half rambling that gets cut off at the cap. Not a measurement of anything."""
    rng = random.Random(seed)
    moves = [("request_document", "passport"), ("verify_document", "passport"),
             ("approve_hr", ""), ("set_tax_id", "Germany")]

    def generate(prompts: List[str], max_tokens: int) -> List[Generation]:
        gens = []
        for p in prompts:
            n_prompt = len(p) // 4   # crude stand-in for a token count
            if rng.random() < 0.5:
                a, t = rng.choice(moves)
                text = f'<think>ok</think><action>{{"action_type": "{a}", "target": "{t}"}}</action>'
                gens.append(Generation(text, "stop", 25, n_prompt))
            else:
                gens.append(Generation("Okay, let's think about this " * 20, "length", max_tokens, n_prompt))
        return gens

    generate.dtype = "(dry run)"  # type: ignore[attr-defined]
    return generate

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generation-only pre-flight check for the Meridian training setup.")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--settings", default="all", help="comma list of: " + ", ".join(s.name for s in DEFAULT_SETTINGS) + ", or 'all'")
    p.add_argument("--tasks", default=",".join(ALL_TASKS))
    p.add_argument("--replicas", type=int, default=4, help="episodes per task (also the GRPO/GiGPO group size to emulate)")
    p.add_argument("--max-steps", type=int, default=ENV_STEP_CAP, help=f"steps per episode (the environment itself ends episodes at {ENV_STEP_CAP})")
    p.add_argument("--history-length", type=int, default=5, help="match env.history_length in the training script")
    p.add_argument("--max-prompt-length", type=int, default=TRAIN_MAX_PROMPT_LENGTH, help="match data.max_prompt_length in the training script")
    p.add_argument("--temperature", type=float, default=1.0, help="rollout temperature (training default: 1.0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default=None, help="default: float16 on pre-Ampere GPUs (e.g. T4), else bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--in-process", action="store_true", help="run vLLM's engine inside this process (VLLM_ENABLE_V1_MULTIPROCESSING=0); try this if startup fails")
    p.add_argument("--out", default="preflight_results.json", help="JSON results; the text report is written next to it as .txt")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="use a canned fake model: tests this script without a GPU, vLLM or downloads")
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    settings = select_settings(args.settings)
    if not settings:
        raise SystemExit("No settings selected. Use --settings all, or a comma list such as think-512,nothink-512.")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"Unknown task(s) {unknown}. Choose from: {', '.join(ALL_TASKS)}.")
    n_episodes = len(tasks) * args.replicas

    gpu: Optional[dict] = None
    if args.dry_run:
        tokenizer = _DryRunTokenizer()
        generate_fn = make_dry_run_generate_fn(args.seed)
    else:
        gpu, detail = probe_gpu()
        if gpu is None:
            print("No CUDA GPU is available to PyTorch, so vLLM cannot run.", file=sys.stderr)
            if detail:
                print(f"  probe said: {detail}", file=sys.stderr)
            print("  In Colab: Runtime > Change runtime type > T4 GPU, then restart the session.", file=sys.stderr)
            return 2
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        generate_fn = make_vllm_generate_fn(
            args.model,
            # Roomy on purpose: the KV cache is sized by memory share, not by this number, and a
            # prompt over the training limit should be MEASURED and flagged, not crash the engine.
            max_model_len=max(8192, args.max_prompt_length + max(s.max_tokens for s in settings) + 1024),
            max_num_seqs=max(16, n_episodes),
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype, seed=args.seed, gpu=gpu,
            in_process=args.in_process, temperature=args.temperature,
        )

    args_info = {
        "model": args.model if not args.dry_run else "(dry run)",
        "tasks": ",".join(tasks), "replicas": args.replicas, "max_steps": args.max_steps,
        "history_length": args.history_length, "max_prompt_length": args.max_prompt_length,
        "temperature": args.temperature,
        "gpu": (f"{gpu['name']} ({gpu['total_gb']} GB, compute {gpu['major']}.{gpu['minor']})" if gpu else "(none)"),
        "dtype": getattr(generate_fn, "dtype", "?"),
    }
    print(f"Running {len(settings)} setting(s) x {n_episodes} episodes "
          f"(up to {args.max_steps} steps each). Each row prints as soon as it finishes.\n")
    print(SUMMARY_HEADER)
    print("-" * len(SUMMARY_HEADER))

    json_path = args.out
    txt_path = os.path.splitext(args.out)[0] + ".txt"
    if txt_path == json_path:  # --out was given a .txt name: never let the report overwrite the JSON
        txt_path = json_path + ".report.txt"
    runs, summaries = [], []
    for setting in settings:
        run = run_setting(setting, generate_fn, tokenizer, tasks=tasks, replicas=args.replicas,
                          max_steps=args.max_steps, history_length=args.history_length,
                          max_prompt_length=args.max_prompt_length,
                          seed=args.seed, show_progress=not args.no_progress)
        summary = summarize(run)
        runs.append(run)
        summaries.append(summary)
        print(summary_line(summary), flush=True)
        with open(json_path, "w") as f:  # rewritten after every setting, so a disconnect loses nothing
            json.dump({"args": args_info, "summaries": summaries,
                       "warnings": {s["setting"]: diagnose(s) for s in summaries},
                       "samples": {r["setting"]: r["samples"] for r in runs},
                       "episodes": {r["setting"]: r["episodes"] for r in runs},
                       "steps": {r["setting"]: r["steps"] for r in runs}}, f, indent=1)

    report = format_report(summaries, runs, args_info)
    with open(txt_path, "w") as f:
        f.write(report + "\n")
    print()
    print(report)
    print(f"\nFull results written to {json_path} and {txt_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())