<div align="center">

# Meridian

**An RL environment and training testbed for cross-border workforce compliance agents**

![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![Status](https://img.shields.io/badge/status-Phase%200%20complete-yellow)

</div>

---

## What this is

An agent must relocate employees across countries — India to Germany, Singapore, or the UAE — by requesting and verifying documents, routing approvals through HR, Legal, and Finance, and setting up tax and payroll correctly under each country's rules. Some cases include a mid-episode regulatory disruption: a visa program gets suspended partway through, and the agent has to notice, adapt, and finish the case under the new rule.

The project exists to answer two specific questions:

1. Does a credit-assignment-aware RL algorithm outperform standard GRPO on a long, rule-heavy task like this one — and does that gap change under the mid-episode disruption?
2. Does a policy trained on a fixed set of hand-authored cases generalize to genuinely novel ones it has never seen, or does it just memorize the training set?

See `docs/architecture.md` for the full design rationale and every alternative that was weighed to get here.

## Status

**Phase 0 — done.** The environment engine, rules, graders, and API are ported and verified working (33/33 tests passing). Nothing about training, generation, or evaluation has been built yet — that's Phases 1-3.

| Phase | What it covers | Status |
|---|---|---|
| 0 | Environment core, rules, graders, API | ✅ done |
| 1 | GRPO baseline, multi-seed | not started |
| 2 | GiGPO comparison arm | not started |
| 3 | LLM-generated held-out generalization test | not started |
| 4 | Stretch: RAG-grounded rules, extra countries | not started |

## Quick start

```bash
pip install -r requirements.txt

# run the full test suite (33 tests across env logic, crisis flow, and the API)
chmod +x run_tests.sh
./run_tests.sh

# run the API server directly
python main.py
# -> http://localhost:7860/docs for interactive API docs
```

## Layout

```
env/            the environment engine: state machine, rules, actions, reward, validators
fixtures/       country rules, visa types, tax treaties (structured JSON, not code)
graders/        deterministic scoring for each task
tests/          33 tests covering env logic, the crisis flow, and the API contract
main.py         FastAPI server exposing reset / step / state / grade
openenv.yaml    environment metadata, Gymnasium/OpenEnv-style reset-step-state interface
docs/           architecture and design-rationale documents
```
