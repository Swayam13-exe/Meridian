"""
test_api.py
===========
HTTP / API contract tests for meridian.

This layer is complementary to:
  - test_eval.py    -> env logic (full happy paths, graders, prereqs)
  - test_crisis.py  -> crisis flow + regression on easy/hard

What test_api.py uniquely covers:
  - Endpoint contracts (status codes, response shapes) for every route
  - Session lifecycle: reset -> step -> state -> grade -> step (until done)
  - Bodyless POST /reset (used by OpenEnv health checks; main.py docstring
    explicitly states this must work)
  - Invalid-task -> HTTP 400 (not silent fallback)
  - Cross-task uniformity (all 4 tasks reach the same observation shape)
  - Crisis behavior reproduced via the API surface (not just the env class)
  - Full easy happy-path expressed at the HTTP layer -> done=True + status=success

Why this matters: validation.sh only pings /reset once. If /step's session
lookup, /grade's flatten helper, or any other HTTP-layer logic regresses,
env tests pass but the live Space fails. This catches that.

Run from project root:
    python test_api.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# TestClient -- exercises the FastAPI app in-process, no server needed
# ---------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
except ImportError:
    print("ERROR: fastapi.testclient not available. Install: pip install httpx fastapi")
    sys.exit(2)

from main import app  # main.py at project root

client = TestClient(app)

VALID_TASKS = ("easy", "medium", "hard", "crisis")


# ---------------------------------------------------------------------------
# Test runner helper
# ---------------------------------------------------------------------------

def run_test(name: str, fn) -> bool:
    try:
        fn()
        print(f"  ✓  {name}")
        return True
    except AssertionError as e:
        print(f"  ✗  {name}")
        print(f"       AssertionError: {e}")
        return False
    except Exception as e:
        print(f"  ✗  {name}")
        print(f"       {type(e).__name__}: {e}")
        return False


# ===========================================================================
# Health & discovery
# ===========================================================================

def test_health_root():
    """GET / returns 200 with status=ok."""
    r = client.get("/")
    assert r.status_code == 200, f"GET / -> {r.status_code}"
    body = r.json()
    assert body.get("status") == "ok", f"unexpected body: {body}"
    assert body.get("environment") == "meridian"


def test_health_endpoint():
    """GET /health returns 200 with status=ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_tasks_lists_all_four():
    """GET /tasks lists easy, medium, hard, crisis."""
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json().get("tasks", [])
    for t in VALID_TASKS:
        assert t in tasks, f"task '{t}' missing from /tasks: {tasks}"


def test_tasks_has_descriptions():
    """GET /tasks includes a non-trivial description for each task."""
    descs = client.get("/tasks").json().get("descriptions", {})
    for t in VALID_TASKS:
        assert t in descs, f"description for '{t}' missing"
        assert len(descs[t]) > 20, f"description for '{t}' too short: {descs[t]!r}"


# ===========================================================================
# Reset semantics
# ===========================================================================

def test_reset_each_task_returns_session():
    """POST /reset works for each of the 4 tasks and returns a session_id + observation."""
    for task in VALID_TASKS:
        r = client.post("/reset", json={"task_name": task})
        assert r.status_code == 200, f"/reset {task} -> HTTP {r.status_code}: {r.text}"
        body = r.json()
        assert "session_id" in body, f"no session_id for task {task}"
        assert "observation" in body, f"no observation for task {task}"
        state = body["observation"]["state"]
        assert state["task_name"] == task, f"state.task_name should be '{task}', got '{state['task_name']}'"


def test_reset_bodyless_defaults_to_easy():
    """POST /reset with NO body must work (OpenEnv health checks do this)."""
    r = client.post("/reset")
    assert r.status_code == 200, f"bodyless /reset -> {r.status_code}: {r.text}"
    state = r.json()["observation"]["state"]
    assert state["task_name"] == "easy", f"bodyless reset should default to easy, got {state['task_name']}"


def test_reset_empty_json_body_defaults_to_easy():
    """POST /reset with `{}` body must also work."""
    r = client.post("/reset", json={})
    assert r.status_code == 200
    assert r.json()["observation"]["state"]["task_name"] == "easy"


def test_reset_invalid_task_returns_400():
    """POST /reset with unknown task_name -> 400 (not silent fallback)."""
    r = client.post("/reset", json={"task_name": "nightmare_mode"})
    assert r.status_code == 400, f"invalid task should 400, got {r.status_code}"


# ===========================================================================
# Step semantics
# ===========================================================================

def test_step_returns_full_shape():
    """POST /step returns observation + reward + done + info."""
    client.post("/reset", json={"task_name": "easy"})
    r = client.post("/step", json={"action_type": "request_document", "target": "passport"})
    assert r.status_code == 200, f"/step -> {r.status_code}: {r.text}"
    body = r.json()
    for key in ("observation", "reward", "done", "info"):
        assert key in body, f"missing '{key}' in /step response"
    assert isinstance(body["reward"], (int, float))
    assert isinstance(body["done"], bool)
    assert body["done"] is False, "easy task can't be done in 1 step"


def test_step_success_for_valid_first_action():
    """First request_document on a fresh easy episode should succeed."""
    client.post("/reset", json={"task_name": "easy"})
    r = client.post("/step", json={"action_type": "request_document", "target": "passport"})
    assert r.json()["info"].get("result") == "success"


# ===========================================================================
# State endpoint
# ===========================================================================

def test_state_after_reset_is_well_formed():
    """GET /state returns the full state model with required fields."""
    client.post("/reset", json={"task_name": "medium"})
    r = client.get("/state")
    assert r.status_code == 200
    state = r.json()
    for key in ("task_name", "documents", "departments", "compliance", "status"):
        assert key in state, f"missing '{key}' in /state response"
    assert state["task_name"] == "medium"


# ===========================================================================
# Grade endpoint
# ===========================================================================

def test_grade_returns_score_in_unit_interval():
    """POST /grade returns a score in [0, 1] for the current session."""
    client.post("/reset", json={"task_name": "easy"})
    r = client.post("/grade", json={})
    assert r.status_code == 200, f"/grade -> {r.status_code}: {r.text}"
    body = r.json()
    score = body.get("score")
    assert score is not None, f"no score in body: {body}"
    assert 0.0 <= score <= 1.0, f"score out of [0,1]: {score}"
    assert body.get("task") == "easy"


def test_grade_with_explicit_task_name():
    """POST /grade with explicit task_name overrides current state task."""
    client.post("/reset", json={"task_name": "easy"})
    r = client.post("/grade", json={"task_name": "easy"})
    assert r.status_code == 200
    assert r.json().get("task") == "easy"


# ===========================================================================
# Crisis behavior reproduced via the API surface
# ===========================================================================

def test_crisis_initial_state_via_api():
    """After POST /reset crisis, /state shows no ict_permit and event not fired."""
    client.post("/reset", json={"task_name": "crisis"})
    state = client.get("/state").json()
    assert state["task_name"] == "crisis"
    assert "ict_permit" not in state["documents"], \
        f"ict_permit should NOT be in initial documents: {list(state['documents'].keys())}"
    assert state["regulatory_event_fired"] is False


def test_crisis_event_fires_at_step_8_via_api():
    """Stepping through 8 actions via /step fires the regulatory event."""
    client.post("/reset", json={"task_name": "crisis"})
    actions = [
        ("request_document", "passport"),
        ("verify_document",  "passport"),
        ("request_document", "visa"),
        ("verify_document",  "visa"),
        ("request_document", "employment_letter"),
        ("verify_document",  "employment_letter"),
        ("approve_hr",       ""),
        ("request_document", "work_permit"),  # step 8 -- event fires here
    ]
    for atype, target in actions:
        client.post("/step", json={"action_type": atype, "target": target})
    state = client.get("/state").json()
    assert state["regulatory_event_fired"] is True, "regulatory event should have fired by step 8"
    assert "ict_permit" in state["documents"], "ict_permit should be injected after event"


# ===========================================================================
# Full happy path expressed at the HTTP layer
# ===========================================================================

def test_easy_happy_path_via_api_done_true_status_success():
    """Full easy happy path through /step must reach done=True with status=success."""
    client.post("/reset", json={"task_name": "easy"})
    actions = [
        ("request_document", "passport"),
        ("verify_document",  "passport"),
        ("request_document", "visa"),
        ("verify_document",  "visa"),
        ("request_document", "employment_letter"),
        ("verify_document",  "employment_letter"),
        ("request_document", "work_permit"),
        ("verify_document",  "work_permit"),
        ("approve_hr",       ""),
        ("set_tax_id",       ""),
        ("set_payroll",      ""),
    ]
    for atype, target in actions:
        client.post("/step", json={"action_type": atype, "target": target})

    r = client.post("/step", json={"action_type": "finalize_case", "target": ""})
    body = r.json()
    assert body["done"] is True, f"episode should be done after finalize_case, info: {body['info']}"
    assert body["info"].get("status") == "success", \
        f"expected status=success, got {body['info'].get('status')}, blockers: {body['info'].get('blockers')}"
    final_score = body["info"].get("final_score", 0.0)
    assert final_score >= 0.70, f"easy final_score should be >= 0.70, got {final_score}"


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Meridian -- API / HTTP Contract Tests")
    print("=" * 60)

    tests = [
        ("GET /            -> 200 ok",                          test_health_root),
        ("GET /health      -> 200 ok",                          test_health_endpoint),
        ("GET /tasks       -> all 4 tasks listed",              test_tasks_lists_all_four),
        ("GET /tasks       -> descriptions present",            test_tasks_has_descriptions),
        ("POST /reset      -> works for every task",            test_reset_each_task_returns_session),
        ("POST /reset      -> bodyless defaults to easy",       test_reset_bodyless_defaults_to_easy),
        ("POST /reset      -> empty {} body defaults to easy",  test_reset_empty_json_body_defaults_to_easy),
        ("POST /reset      -> invalid task = HTTP 400",         test_reset_invalid_task_returns_400),
        ("POST /step       -> returns full shape",              test_step_returns_full_shape),
        ("POST /step       -> first action succeeds",           test_step_success_for_valid_first_action),
        ("GET /state       -> well-formed state",               test_state_after_reset_is_well_formed),
        ("POST /grade      -> score in [0,1]",                  test_grade_returns_score_in_unit_interval),
        ("POST /grade      -> explicit task_name works",        test_grade_with_explicit_task_name),
        ("Crisis: initial /state has no ict_permit",            test_crisis_initial_state_via_api),
        ("Crisis: event fires at step 8 via API",               test_crisis_event_fires_at_step_8_via_api),
        ("Easy: full happy path via API -> done + success",     test_easy_happy_path_via_api_done_true_status_success),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    mark = "PASS" if failed == 0 else "FAIL"
    print(f"Results: {passed}/{len(tests)} passed [{mark}]")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)