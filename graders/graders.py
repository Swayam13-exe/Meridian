"""
graders/graders.py
==================
Deterministic graders for all meridian tasks.

SCORING DESIGN — NO CEILINGS
────────────────────────────
Scores are derived purely from weighted requirement completion.
No artificial per-task ceilings. Difficulty is encoded directly in the weights,
so a perfect agent naturally lands at a task-appropriate score:

  Task    Perfect Score   What earns it
  ──────  ─────────────   ─────────────
  easy    ~0.95           4 docs + HR + tax_id + payroll + finalize
  medium  ~0.80           3 docs + HR + Legal + payroll + pdpa + shadow + finalize
  crisis  ~0.75           4 docs (incl ict_permit) + HR + Legal + tax_id + payroll
                          + regulatory event acknowledged + finalize
  hard    ~0.65           4 docs + HR + Legal + Finance + tax_id + payroll
                          + conflict resolved + finalize (+ UAE trap avoided)

Why this ordering? Harder tasks have MORE requirements, so each requirement
is worth proportionally less. A perfect Hard score is lower than a perfect
Easy score because the same 100% completion represents more work on Hard.

Penalties:
  UAE tax violation (hard):       -0.25
  Visa-after-event (crisis):      -0.20 per occurrence
  Parsimony (all tasks):          -0.03 per junk action, capped at -0.15

Epsilon floor:
  Scores are clamped to (0.0001, 0.9999) — strictly between 0 and 1.
  This is the ONLY clamp; no task-specific ceilings.

Public API:
  grade(task_name, state) → float   main dispatcher (used by environment.py)
  explain(task_name, state) → str   human-readable breakdown
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Epsilon — the only score boundary
# ---------------------------------------------------------------------------

_EPSILON = 0.0001


# ---------------------------------------------------------------------------
# Documents per task — must match tasks.py exactly
# ---------------------------------------------------------------------------

_TASK_DOCS: dict[str, list[str]] = {
    "easy":   ["passport", "visa", "employment_letter", "work_permit"],
    "medium": ["passport", "visa", "employment_letter"],
    "hard":   ["passport", "visa", "employment_letter", "work_permit"],
    # crisis: visa is replaced by ict_permit after the event fires
    "crisis": ["passport", "employment_letter", "work_permit", "ict_permit"],
}

_TASK_COMPLIANCE: dict[str, list[str]] = {
    "easy":   ["tax_id", "payroll"],
    "medium": ["payroll", "pdpa", "shadow_payroll"],
    "hard":   ["tax_id", "payroll"],
    "crisis": ["tax_id", "payroll"],
}

_TASK_DEPARTMENTS: dict[str, list[str]] = {
    "easy":   ["HR"],
    "medium": ["HR", "Legal"],
    "hard":   ["HR", "Legal", "Finance"],
    "crisis": ["HR", "Legal"],
}


# ---------------------------------------------------------------------------
# Valid actions per task — for parsimony penalty
# ---------------------------------------------------------------------------

_EASY_VALID_ACTIONS: set[str] = {
    "request_document:passport",    "verify_document:passport",
    "request_document:visa",        "verify_document:visa",
    "request_document:employment_letter", "verify_document:employment_letter",
    "request_document:work_permit", "verify_document:work_permit",
    "approve_hr", "set_payroll", "set_tax_id", "finalize_case",
}

_MEDIUM_VALID_ACTIONS: set[str] = {
    "request_document:passport",    "verify_document:passport",
    "request_document:visa",        "verify_document:visa",
    "request_document:employment_letter", "verify_document:employment_letter",
    "approve_hr", "approve_legal",
    "set_payroll", "set_pdpa", "set_shadow_payroll",
    "finalize_case",
}

_HARD_VALID_ACTIONS: set[str] = {
    "request_document:passport",    "verify_document:passport",
    "request_document:visa",        "verify_document:visa",
    "request_document:employment_letter", "verify_document:employment_letter",
    "request_document:work_permit", "verify_document:work_permit",
    "approve_hr", "approve_legal", "approve_finance",
    "set_payroll", "set_tax_id", "resolve_conflict",
    "finalize_case",
}

_CRISIS_VALID_ACTIONS: set[str] = {
    "request_document:passport",    "verify_document:passport",
    "request_document:visa",        "verify_document:visa",      # valid BEFORE event
    "request_document:employment_letter", "verify_document:employment_letter",
    "request_document:work_permit", "verify_document:work_permit",
    "request_document:ict_permit",  "verify_document:ict_permit",  # valid AFTER event
    "acknowledge_regulatory_change",
    "approve_hr", "approve_legal",
    "set_payroll", "set_tax_id",
    "finalize_case",
}

_VALID_ACTIONS_MAP: dict[str, set[str]] = {
    "easy":   _EASY_VALID_ACTIONS,
    "medium": _MEDIUM_VALID_ACTIONS,
    "hard":   _HARD_VALID_ACTIONS,
    "crisis": _CRISIS_VALID_ACTIONS,
}


# ---------------------------------------------------------------------------
# Parsimony penalty — discourages junk actions
# ---------------------------------------------------------------------------

def _parsimony_penalty(prev_actions: list[str], valid_actions: set[str]) -> float:
    """
    -0.03 per action not in valid set. Capped at -0.15.
    System event markers ([SYSTEM_EVENT:...]) are never penalised.
    """
    junk = sum(
        1 for a in prev_actions
        if a not in valid_actions and not a.startswith("[SYSTEM_EVENT:")
    )
    return min(0.15, junk * 0.03)


# ---------------------------------------------------------------------------
# Score finalizer — the ONLY clamp
# ---------------------------------------------------------------------------

def _finalize_score(raw: float) -> float:
    """
    Clamp to strictly (0, 1). No task-specific ceilings applied.
    """
    score = round(raw, 4)
    if score <= 0.0:
        return _EPSILON
    if score >= 1.0:
        return 1.0 - _EPSILON
    return score


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def grade(task_name: str, state: dict[str, Any]) -> float:
    """
    Main entry point. Called by environment.py after every episode end.

    Returns:
        Float strictly in (0.0, 1.0). No ceilings — difficulty is encoded
        in the per-task weights below.
    """
    graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
        "crisis": grade_crisis,
    }
    if task_name not in graders:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid: {list(graders.keys())}"
        )
    score = graders[task_name](state)
    assert 0.0 < score < 1.0, f"Grader returned out-of-range score: {score}"
    return score


def explain(task_name: str, state: dict[str, Any]) -> str:
    """Return a human-readable scoring breakdown string."""
    reporters = {
        "easy":   _explain_easy,
        "medium": _explain_medium,
        "hard":   _explain_hard,
        "crisis": _explain_crisis,
    }
    if task_name not in reporters:
        raise ValueError(f"Unknown task '{task_name}'")
    return reporters[task_name](state)


# ===========================================================================
# EASY grader — perfect score ≈ 0.95
# ===========================================================================
# Weights (sum = 0.95):
#   Documents verified    0.40  (4 docs × 0.10)
#   HR approved           0.20
#   tax_id                0.15
#   payroll               0.15
#   Finalized             0.05
# ===========================================================================

def grade_easy(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["easy"]

    # Documents (0.40 — 0.10 per doc)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.40

    # Department (0.20)
    hr_score = 0.20 if depts.get("HR", False) else 0.0

    # Compliance (0.30 split as 0.15 each)
    tax_score     = 0.15 if compliance.get("tax_id", False) else 0.0
    payroll_score = 0.15 if compliance.get("payroll", False) else 0.0

    # Finalized (0.05)
    fin_score = 0.05 if status == "success" else 0.0

    raw = doc_score + hr_score + tax_score + payroll_score + fin_score

    # Parsimony penalty
    raw -= _parsimony_penalty(prev_acts, _EASY_VALID_ACTIONS)

    return _finalize_score(raw)


# ===========================================================================
# MEDIUM grader — perfect score ≈ 0.80
# ===========================================================================
# Weights (sum = 0.80):
#   Documents verified    0.24  (3 docs × 0.08)
#   HR approved           0.12
#   Legal approved        0.14
#   payroll               0.08
#   pdpa                  0.08
#   shadow_payroll        0.08
#   Finalized             0.06
# ===========================================================================

def grade_medium(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["medium"]

    # Documents (0.24 — 0.08 per doc)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.24

    # Departments
    hr_score    = 0.12 if depts.get("HR", False) else 0.0
    legal_score = 0.14 if depts.get("Legal", False) else 0.0

    # Compliance (0.08 each)
    payroll_score = 0.08 if compliance.get("payroll", False) else 0.0
    pdpa_score    = 0.08 if compliance.get("pdpa", False) else 0.0
    shadow_score  = 0.08 if compliance.get("shadow_payroll", False) else 0.0

    # Finalized
    fin_score = 0.06 if status == "success" else 0.0

    raw = (
        doc_score + hr_score + legal_score
        + payroll_score + pdpa_score + shadow_score
        + fin_score
    )

    raw -= _parsimony_penalty(prev_acts, _MEDIUM_VALID_ACTIONS)

    return _finalize_score(raw)


# ===========================================================================
# HARD grader — perfect score ≈ 0.65 (AFTER UAE trap avoided)
# ===========================================================================
# Weights (sum = 0.65):
#   Documents verified    0.20  (4 docs × 0.05)
#   HR approved           0.06
#   Legal approved        0.08
#   Finance approved      0.08
#   tax_id                0.05
#   payroll               0.05
#   Conflict resolved     0.08
#   Finalized             0.05
#
# Penalty: UAE tax violation = -0.25
# ===========================================================================

def grade_hard(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    conflicts  = state.get("conflicts", [])
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["hard"]

    # Documents (0.20 — 0.05 per doc)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.20

    # Departments
    hr_score      = 0.06 if depts.get("HR", False) else 0.0
    legal_score   = 0.08 if depts.get("Legal", False) else 0.0
    finance_score = 0.08 if depts.get("Finance", False) else 0.0

    # Compliance
    tax_score     = 0.05 if compliance.get("tax_id", False) else 0.0
    payroll_score = 0.05 if compliance.get("payroll", False) else 0.0

    # Conflict resolution
    if conflicts:
        all_resolved = all(c.get("resolved", False) for c in conflicts)
        conflict_score = 0.08 if all_resolved else 0.0
    else:
        conflict_score = 0.0

    # Finalized
    fin_score = 0.05 if status == "success" else 0.0

    raw = (
        doc_score + hr_score + legal_score + finance_score
        + tax_score + payroll_score + conflict_score + fin_score
    )

    # UAE tax violation penalty
    uae_tax_called = "set_tax_id:UAE" in prev_acts
    if uae_tax_called:
        raw -= 0.25

    raw -= _parsimony_penalty(prev_acts, _HARD_VALID_ACTIONS)

    return _finalize_score(raw)


# ===========================================================================
# CRISIS grader — perfect score ≈ 0.75 (AFTER event handled correctly)
# ===========================================================================
# Weights (sum = 0.75):
#   Documents verified    0.32  (4 docs × 0.08)
#   HR approved           0.08
#   Legal approved        0.10
#   tax_id                0.08
#   payroll               0.07
#   Regulatory handled    0.05  (event fired + acknowledged)
#   Finalized             0.05
#
# Penalties:
#   Using visa after event: -0.20 per occurrence
#   Parsimony: standard
# ===========================================================================

def grade_crisis(state: dict[str, Any]) -> float:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["crisis"]

    # Documents (0.32 — 0.08 per doc)
    verified = sum(
        1 for d in required_docs
        if docs.get(d, {}).get("status") == "verified"
    )
    doc_score = (verified / len(required_docs)) * 0.32

    # Departments
    hr_score    = 0.08 if depts.get("HR", False) else 0.0
    legal_score = 0.10 if depts.get("Legal", False) else 0.0

    # Compliance
    tax_score     = 0.08 if compliance.get("tax_id", False) else 0.0
    payroll_score = 0.07 if compliance.get("payroll", False) else 0.0

    # Regulatory event handling (0.05)
    event_fired = state.get("regulatory_event_fired", False)
    event_ack   = state.get("regulatory_event_acknowledged", False)
    if event_fired and event_ack:
        regulatory_score = 0.05
    elif event_fired and not event_ack:
        regulatory_score = 0.015   # partial — fired but not handled
    else:
        regulatory_score = 0.0     # hasn't fired yet

    # Finalized
    fin_score = 0.05 if status == "success" else 0.0

    raw = (
        doc_score + hr_score + legal_score
        + tax_score + payroll_score
        + regulatory_score + fin_score
    )

    # Penalty: using invalidated visa AFTER the system event marker
    system_event_idx = next(
        (i for i, a in enumerate(prev_acts) if a.startswith("[SYSTEM_EVENT:")),
        None
    )
    if system_event_idx is not None:
        post_event_actions = prev_acts[system_event_idx:]
        visa_violations = sum(
            1 for a in post_event_actions
            if a in ("verify_document:visa", "request_document:visa")
        )
        raw -= visa_violations * 0.20

    raw -= _parsimony_penalty(prev_acts, _CRISIS_VALID_ACTIONS)

    return _finalize_score(raw)


# ---------------------------------------------------------------------------
# Explain helpers — human-readable breakdown
# ---------------------------------------------------------------------------

def _explain_easy(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")

    required_docs = _TASK_DOCS["easy"]
    required_comp = _TASK_COMPLIANCE["easy"]

    verified  = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done = [c for c in required_comp if compliance.get(c, False)]

    lines = [
        "=== EASY TASK SCORE BREAKDOWN ===",
        f"Documents:   {len(verified)}/{len(required_docs)} verified {verified}",
        f"HR:          {'✓' if depts.get('HR') else '✗'}",
        f"Compliance:  {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"Status:      {status}",
        f"Score:       {grade_easy(state):.4f}  (perfect ≈ 0.95)",
    ]
    return "\n".join(lines)


def _explain_medium(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")

    required_docs = _TASK_DOCS["medium"]
    required_comp = _TASK_COMPLIANCE["medium"]

    verified  = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done = [c for c in required_comp if compliance.get(c, False)]

    lines = [
        "=== MEDIUM TASK SCORE BREAKDOWN ===",
        f"Documents:   {len(verified)}/{len(required_docs)} verified {verified}",
        f"HR:          {'✓' if depts.get('HR') else '✗'}",
        f"Legal:       {'✓' if depts.get('Legal') else '✗'}",
        f"Compliance:  {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"Status:      {status}",
        f"Score:       {grade_medium(state):.4f}  (perfect ≈ 0.80)",
    ]
    return "\n".join(lines)


def _explain_hard(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    conflicts  = state.get("conflicts", [])
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["hard"]
    required_comp = _TASK_COMPLIANCE["hard"]

    verified       = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done      = [c for c in required_comp if compliance.get(c, False)]
    uae_tax_called = "set_tax_id:UAE" in prev_acts
    resolved       = all(c.get("resolved", False) for c in conflicts) if conflicts else True

    lines = [
        "=== HARD TASK SCORE BREAKDOWN ===",
        f"Documents:        {len(verified)}/{len(required_docs)} verified {verified}",
        f"HR:               {'✓' if depts.get('HR') else '✗'}",
        f"Legal:            {'✓' if depts.get('Legal') else '✗'}",
        f"Finance:          {'✓' if depts.get('Finance') else '✗'}",
        f"Compliance:       {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"UAE no-tax rule:  {'✗ VIOLATED (-0.25)' if uae_tax_called else '✓ respected'}",
        f"Conflicts:        {'✓ resolved' if resolved else '✗ unresolved'}",
        f"Status:           {status}",
        f"Score:            {grade_hard(state):.4f}  (perfect ≈ 0.65)",
    ]
    return "\n".join(lines)


def _explain_crisis(state: dict[str, Any]) -> str:
    docs       = state.get("documents", {})
    depts      = state.get("departments", {})
    compliance = state.get("compliance", {})
    status     = state.get("status", "in_progress")
    prev_acts  = state.get("previous_actions", [])

    required_docs = _TASK_DOCS["crisis"]
    required_comp = _TASK_COMPLIANCE["crisis"]

    verified    = [d for d in required_docs if docs.get(d, {}).get("status") == "verified"]
    comp_done   = [c for c in required_comp if compliance.get(c, False)]
    event_fired = state.get("regulatory_event_fired", False)
    event_ack   = state.get("regulatory_event_acknowledged", False)

    system_event_idx = next(
        (i for i, a in enumerate(prev_acts) if a.startswith("[SYSTEM_EVENT:")),
        None
    )
    visa_violations = 0
    if system_event_idx is not None:
        post_event = prev_acts[system_event_idx:]
        visa_violations = sum(
            1 for a in post_event
            if a in ("verify_document:visa", "request_document:visa")
        )

    lines = [
        "=== CRISIS TASK SCORE BREAKDOWN ===",
        f"Documents:          {len(verified)}/{len(required_docs)} verified {verified}",
        f"  (requires ict_permit, NOT visa)",
        f"HR:                 {'✓' if depts.get('HR') else '✗'}",
        f"Legal:              {'✓' if depts.get('Legal') else '✗'}",
        f"Compliance:         {len(comp_done)}/{len(required_comp)} done {comp_done}",
        f"Event fired:        {'✓' if event_fired else '✗ (not yet)'}",
        f"Event acknowledged: {'✓' if event_ack else '✗'}",
        f"Visa violations:    {visa_violations} × -0.20",
        f"Status:             {status}",
        f"Score:              {grade_crisis(state):.4f}  (perfect ≈ 0.75)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch utility used by inference.py
# ---------------------------------------------------------------------------

def grade_all(states: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Grade all tasks. Missing tasks default to epsilon floor."""
    results: dict[str, float] = {}
    for task_name in ["easy", "medium", "hard", "crisis"]:
        if task_name in states:
            try:
                results[task_name] = grade(task_name, states[task_name])
            except Exception as exc:
                results[task_name] = _EPSILON
                print(f"[grader] ERROR grading '{task_name}': {exc}")
        else:
            results[task_name] = _EPSILON
    return results