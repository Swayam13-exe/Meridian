"""
env/reward.py
=============
Reward computation for meridian.

Design principles:
  - All reward values clamped to [-1.0, 1.0] per step
  - Cumulative episode score clamped to [0.0, 1.0]
  - No randomness — same (action, result, state) always returns same reward
  - Progress delta bonus incentivises efficient completion
  - Milestone bonuses for completing entire document/department/compliance groups
  - No circular imports — state is checked directly, not via rules_engine
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

REWARDS: dict[str, float] = {
    # Positive
    "success":               0.30,   # clean successful action
    "milestone_docs":        0.20,   # all documents in case verified
    "milestone_depts":       0.20,   # all required departments approved
    "milestone_compliance":  0.15,   # all required compliance items done
    "conflict_resolved":     0.25,   # hard task: conflict resolved
    "episode_complete":      0.50,   # finalize_case succeeded

    # Negative
    "wrong_action":          -0.10,  # valid type, wrong context
    "prereq_violated":       -0.20,  # prerequisite not met
    "rule_violation":        -0.30,  # breaks a country rule (e.g. UAE tax)
    "invalid_action":        -0.30,  # unknown action_type or bad target
    "repeated_action":       -0.10,  # same action called again
}

# Base reward per result code
_RESULT_BASE: dict[str, float] = {
    "success":          REWARDS["success"],
    "wrong_action":     REWARDS["wrong_action"],
    "prereq_violated":  REWARDS["prereq_violated"],
    "rule_violation":   REWARDS["rule_violation"],
    "invalid_action":   REWARDS["invalid_action"],
    "repeated_action":  REWARDS["repeated_action"],
}


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------


def compute_reward(
    action: Any,
    result: str,
    state: dict[str, Any],
    prev_progress: float,
) -> float:
    """
    Compute the per-step reward for a completed action.

    NOTE: This is called AFTER state mutation and AFTER progress update
    in environment.py, so state["progress"] reflects the new value.

    Steps:
      1. Base reward from result code
      2. Progress delta bonus (only on success)
      3. Milestone bonuses (only on success)
      4. Episode completion bonus
      5. Clamp to [-1.0, 1.0]

    Args:
        action:        The Action that was executed.
        result:        Result code ("success" | "wrong_action" | etc.)
        state:         Current state dict (already updated with new progress).
        prev_progress: Progress value BEFORE this action was applied.

    Returns:
        Float clamped to [-1.0, 1.0].
    """
    reward = _RESULT_BASE.get(result, 0.0)

    if result == "success":

        # ── Progress delta bonus ─────────────────────────────────────────────
        new_progress = state.get("progress", prev_progress)
        delta = max(0.0, new_progress - prev_progress)
        if delta > 0:
            # Each 0.1 of progress gained adds 0.05 reward
            # Max bonus per step: 1.0 * 0.5 = 0.5 (if progress jumps from 0→1)
            reward += round(min(delta * 0.5, 0.20), 4)

        # ── Milestone bonuses ────────────────────────────────────────────────
        reward += _check_milestones(action, state)

        # ── Episode completion bonus ─────────────────────────────────────────
        if (
            action.action_type == "finalize_case"
            and state.get("status") == "success"
        ):
            reward += REWARDS["episode_complete"]

    # ── Clamp ────────────────────────────────────────────────────────────────
    return round(max(-1.0, min(1.0, reward)), 4)


# ---------------------------------------------------------------------------
# Milestone checker
# ---------------------------------------------------------------------------


def _check_milestones(action: Any, state: dict[str, Any]) -> float:
    """
    Return bonus reward if a milestone was just completed.

    Checks state directly — no country rule re-derivation.
    Uses required_departments and required_compliance from state
    (set at task load time) so it matches task definitions exactly.
    """
    bonus = 0.0
    atype = action.action_type

    # ── Document milestone ───────────────────────────────────────────────────
    # Triggered when a verify_document action succeeds
    if atype == "verify_document":
        documents = state.get("documents", {})
        if documents and all(
            doc.get("status") == "verified" for doc in documents.values()
        ):
            bonus += REWARDS["milestone_docs"]

    # ── Department milestone ─────────────────────────────────────────────────
    # Triggered when any approve_* action succeeds
    if atype in {"approve_hr", "approve_legal", "approve_finance"}:
        departments   = state.get("departments", {})
        required_depts = state.get("required_departments", list(departments.keys()))
        if required_depts and all(
            departments.get(d, False) for d in required_depts
        ):
            bonus += REWARDS["milestone_depts"]

    # ── Compliance milestone ─────────────────────────────────────────────────
    # Triggered when any set_* compliance action succeeds
    if atype in {"set_payroll", "set_tax_id", "set_shadow_payroll", "set_pdpa"}:
        compliance    = state.get("compliance", {})
        required_comp = state.get("required_compliance", [])
        if required_comp and all(
            compliance.get(c, False) for c in required_comp
        ):
            bonus += REWARDS["milestone_compliance"]

    # ── Conflict resolution bonus ────────────────────────────────────────────
    # Hard task only — triggered when resolve_conflict succeeds
    if atype == "resolve_conflict":
        conflicts = state.get("conflicts", [])
        # Check that at least one conflict was just resolved
        resolved_count = sum(1 for c in conflicts if c.get("resolved", False))
        if resolved_count > 0:
            bonus += REWARDS["conflict_resolved"]

    return round(bonus, 4)