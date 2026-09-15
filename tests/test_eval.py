"""
test_eval.py
============
Evaluation tests for meridian.

Tests match the FIXED architecture:
  - Easy task:   4 docs (passport/visa/employment_letter/work_permit) + HR only
  - Medium task: 3 docs + HR + Legal + payroll + pdpa + shadow_payroll (NO tax_id)
  - Hard task:   4 docs + HR + Legal + Finance + tax_id(Germany) + payroll + resolve_conflict

Run with:
    python test_eval.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.getcwd())

from env.environment import WorkforceEnv
from env.models import Action


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def step_and_check(env: WorkforceEnv, atype: str, target: str) -> bool:
    """Apply one action and print result. Returns True if success."""
    res = env.step(Action(action_type=atype, target=target))
    result = res.info.get("result", "?")
    if result != "success":
        print(f"  FAILED:  {atype}:{target or '(empty)'} → {result} | {res.info.get('error', '')}")
        return False
    else:
        print(f"  OK:      {atype}:{target or '(empty)'}")
        return True


# ---------------------------------------------------------------------------
# Test 1 — Easy task: full correct sequence → status=success
# ---------------------------------------------------------------------------

def test_easy_full_success():
    print("--- Test 1: Easy Full Success (India → Germany) ---")
    env = WorkforceEnv()
    env.reset("easy")

    # FIX: easy task uses work_permit NOT degree_certificate
    # FIX: easy task only requires HR (not Legal)
    actions = [
        ("request_document",  "passport"),
        ("verify_document",   "passport"),
        ("request_document",  "visa"),
        ("verify_document",   "visa"),
        ("request_document",  "employment_letter"),
        ("verify_document",   "employment_letter"),
        ("request_document",  "work_permit"),
        ("verify_document",   "work_permit"),
        ("approve_hr",        ""),
        ("set_tax_id",        ""),
        ("set_payroll",       ""),
    ]

    for atype, target in actions:
        step_and_check(env, atype, target)

    res = env.step(Action(action_type="finalize_case", target=""))
    print(f"  FINALIZE → result={res.info.get('result')} status={res.info.get('status')} score={res.info.get('final_score')}")

    assert res.done, "Episode should be done after finalize_case"
    assert res.info.get("status") == "success", (
        f"Expected status=success, got {res.info.get('status')}. "
        f"Blockers: {res.info.get('blockers')}"
    )
    assert res.info.get("final_score", 0.0) >= 0.70, (
        f"Easy score should be >= 0.70, got {res.info.get('final_score')}"
    )
    print("  PASSED ✓\n")


# ---------------------------------------------------------------------------
# Test 2 — Medium task: full correct sequence → status=success
# ---------------------------------------------------------------------------

def test_medium_full_success():
    print("--- Test 2: Medium Full Success (India → Singapore) ---")
    env = WorkforceEnv()
    env.reset("medium")

    # FIX: medium requires HR + Legal (not Finance)
    # FIX: Singapore does NOT require tax_id
    # FIX: medium requires payroll + pdpa + shadow_payroll
    actions = [
        ("request_document",  "passport"),
        ("verify_document",   "passport"),
        ("request_document",  "visa"),
        ("verify_document",   "visa"),
        ("request_document",  "employment_letter"),
        ("verify_document",   "employment_letter"),
        ("approve_hr",        ""),
        ("approve_legal",     ""),
        ("set_payroll",       ""),
        ("set_pdpa",          ""),
        ("set_shadow_payroll", ""),
    ]

    for atype, target in actions:
        step_and_check(env, atype, target)

    res = env.step(Action(action_type="finalize_case", target=""))
    print(f"  FINALIZE → result={res.info.get('result')} status={res.info.get('status')} score={res.info.get('final_score')}")

    assert res.done, "Episode should be done after finalize_case"
    assert res.info.get("status") == "success", (
        f"Expected status=success, got {res.info.get('status')}. "
        f"Blockers: {res.info.get('blockers')}"
    )
    assert res.info.get("final_score", 0.0) >= 0.40, (
        f"Medium score should be >= 0.40, got {res.info.get('final_score')}"
    )
    print("  PASSED ✓\n")


# ---------------------------------------------------------------------------
# Test 3 — Hard task: UAE tax trap → rule_violation penalty
# ---------------------------------------------------------------------------

def test_hard_uae_tax_trap():
    print("--- Test 3: Hard UAE Tax Trap (set_tax_id:UAE = rule_violation) ---")
    env = WorkforceEnv()
    env.reset("hard")

    # Calling set_tax_id for UAE must return rule_violation
    res = env.step(Action(action_type="set_tax_id", target="UAE"))
    result = res.info.get("result")
    print(f"  set_tax_id:UAE → {result} | reward={res.reward:.2f}")

    assert result == "rule_violation", (
        f"Expected rule_violation for set_tax_id:UAE, got '{result}'"
    )
    assert res.reward <= -0.20, (
        f"Expected penalty reward <= -0.20 for UAE tax violation, got {res.reward}"
    )
    print("  PASSED ✓\n")


# ---------------------------------------------------------------------------
# Test 4 — Hard task: full correct sequence → status=success
# ---------------------------------------------------------------------------

def test_hard_full_success():
    print("--- Test 4: Hard Full Success (India → Germany + UAE) ---")
    env = WorkforceEnv()
    env.reset("hard")

    # Hard task: 4 docs + HR + Legal + resolve_conflict + tax_id(DE) + payroll + Finance
    actions = [
        ("request_document",  "passport"),
        ("verify_document",   "passport"),
        ("request_document",  "visa"),
        ("verify_document",   "visa"),
        ("request_document",  "employment_letter"),
        ("verify_document",   "employment_letter"),
        ("request_document",  "work_permit"),
        ("verify_document",   "work_permit"),
        ("approve_hr",        ""),
        ("approve_legal",     ""),
        ("set_tax_id",        ""),        # Germany only (not UAE)
        ("set_payroll",       ""),
        ("resolve_conflict",  ""),        # must come before approve_finance
        ("approve_finance",   ""),
    ]

    for atype, target in actions:
        step_and_check(env, atype, target)

    res = env.step(Action(action_type="finalize_case", target=""))
    print(f"  FINALIZE → result={res.info.get('result')} status={res.info.get('status')} score={res.info.get('final_score')}")

    assert res.done, "Episode should be done after finalize_case"
    assert res.info.get("status") == "success", (
        f"Expected status=success, got {res.info.get('status')}. "
        f"Blockers: {res.info.get('blockers')}"
    )
    assert res.info.get("final_score", 0.0) >= 0.20, (
        f"Hard score should be >= 0.20, got {res.info.get('final_score')}"
    )
    print("  PASSED ✓\n")


# ---------------------------------------------------------------------------
# Test 5 — Grader scores are always in [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_grader_score_range():
    print("--- Test 5: Grader Scores in [0.0, 1.0] ---")
    from graders.graders import grade

    # Test with empty state (minimum possible score)
    empty_state = {
        "task_name": "easy",
        "documents": {"passport": {"status": "missing", "is_valid": False}},
        "departments": {"HR": False, "Legal": False, "Finance": False},
        "compliance": {"tax_id": False, "payroll": False, "pdpa": False, "shadow_payroll": False},
        "conflicts": [],
        "previous_actions": [],
        "status": "in_progress",
        "progress": 0.0,
    }

    for task in ["easy", "medium", "hard"]:
        empty_state["task_name"] = task
        score = grade(task, empty_state)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for task {task}"
        print(f"  {task}: empty state score = {score:.4f} ✓")

    print("  PASSED ✓\n")


# ---------------------------------------------------------------------------
# Test 6 — repeat action gets penalized
# ---------------------------------------------------------------------------

def test_repeat_action_penalty():
    print("--- Test 6: Repeat Action Penalty ---")
    env = WorkforceEnv()
    env.reset("easy")

    # First request — should succeed
    res1 = env.step(Action(action_type="request_document", target="passport"))
    assert res1.info.get("result") == "success", "First request should succeed"

    # Second request of same doc — should get penalized
    res2 = env.step(Action(action_type="request_document", target="passport"))
    result2 = res2.info.get("result")
    print(f"  Second request_document:passport → {result2} | reward={res2.reward:.2f}")

    # Should be wrong_action (doc already submitted) not success
    assert result2 != "success", f"Repeat action should not succeed, got '{result2}'"
    print("  PASSED ✓\n")


# ---------------------------------------------------------------------------
# Test 7 — Legal blocked until docs verified
# ---------------------------------------------------------------------------

def test_legal_prereq_enforced():
    print("--- Test 7: Legal Cannot Approve Before Docs Verified ---")
    env = WorkforceEnv()
    env.reset("medium")

    # Try to approve Legal immediately without any docs
    res = env.step(Action(action_type="approve_legal", target=""))
    result = res.info.get("result")
    print(f"  approve_legal (no docs) → {result} | error={res.info.get('error', '')[:60]}")

    assert result in ("prereq_violated", "rule_violation"), (
        f"Expected prereq_violated, got '{result}'"
    )
    print("  PASSED ✓\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_easy_full_success,
        test_medium_full_success,
        test_hard_uae_tax_trap,
        test_hard_full_success,
        test_grader_score_range,
        test_repeat_action_penalty,
        test_legal_prereq_enforced,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ASSERTION FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}\n")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 50)

    if failed > 0:
        sys.exit(1)