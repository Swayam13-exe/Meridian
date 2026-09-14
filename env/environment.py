"""
env/environment.py
==================
Core OpenEnv environment class for meridian.

Implements the full OpenEnv interface:
  - reset(task_name)  → Observation
  - step(action)      → StepResult
  - state()           → WorkforceState
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from env.models import (
    Action,
    ComplianceStatus,
    ConflictRecord,
    DepartmentStatus,
    DocumentRecord,
    EmployeeRecord,
    Observation,
    StepResult,
    WorkforceState,
)
from env.reward import REWARDS, compute_reward
from env.rules import COUNTRY_RULES, DEPARTMENT_DEPENDENCIES
from env.validators import (
    validate_compliance_action,
    validate_department_prerequisites,
    validate_document,
)
from env.tasks import TASKS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS: int = 35   # raised from 25 to accommodate the crisis task (35 days)

VALID_ACTION_TYPES: set[str] = {
    "request_document",
    "verify_document",
    "approve_hr",
    "approve_legal",
    "approve_finance",
    "set_payroll",
    "set_tax_id",
    "set_shadow_payroll",
    "set_pdpa",
    "resolve_conflict",
    "acknowledge_regulatory_change",   # crisis task
    "finalize_case",
}

VALID_DOCUMENTS: set[str] = {
    "passport",
    "visa",
    "employment_letter",
    "degree_certificate",
    "work_permit",          # Germany
    "employment_pass",      # Singapore
    "residence_permit",     # UAE
    "tax_form",
    "ict_permit",           # Germany crisis task — injected at step 8
}

VALID_DEPARTMENTS: set[str] = {"HR", "Legal", "Finance"}

# Actions that legitimately have no target (empty string is fine)
NO_TARGET_ACTIONS: set[str] = {
    "approve_hr",
    "approve_legal",
    "approve_finance",
    "set_payroll",
    "set_tax_id",
    "set_shadow_payroll",
    "set_pdpa",
    "resolve_conflict",
    "acknowledge_regulatory_change",
    "finalize_case",
}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class WorkforceEnv:
    """
    Stateful OpenEnv environment simulating employee relocation workflows.

    Episode termination conditions:
        finalize_case with no blockers  → status="success", done=True
        deadline_days reaches 0         → status="failed",  done=True
        steps_taken reaches MAX_STEPS   → status="failed",  done=True
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}
        self._task_name: str = ""
        self._steps_taken: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._last_action_result: str = "Episode not started. Call reset() first."
        self._last_action_error: str | None = None
        self._session_id: str = str(uuid.uuid4())

    # -----------------------------------------------------------------------
    # Public OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self, task_name: str = "easy") -> Observation:
        """
        Start a new episode for the given task.

        Args:
            task_name: One of "easy", "medium", "hard", "crisis".

        Returns:
            Observation with the initial state and available actions.
        """
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid tasks: {list(TASKS.keys())}"
            )

        self._state = copy.deepcopy(TASKS[task_name])
        self._task_name = task_name
        self._steps_taken = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_action_result = "Episode started. Make your first action."
        self._last_action_error = None
        self._session_id = str(uuid.uuid4())

        # Ensure required keys exist
        if "conflicts" not in self._state:
            self._state["conflicts"] = []
        if "required_departments" not in self._state:
            self._state["required_departments"] = []
        if "required_compliance" not in self._state:
            self._state["required_compliance"] = []

        # Ensure crisis fields exist for non-crisis tasks too (safe defaults)
        if "regulatory_event_fired" not in self._state:
            self._state["regulatory_event_fired"] = False
        if "regulatory_event_acknowledged" not in self._state:
            self._state["regulatory_event_acknowledged"] = False
        if "regulatory_event_step" not in self._state:
            self._state["regulatory_event_step"] = 9999
        if "regulatory_event" not in self._state:
            self._state["regulatory_event"] = None

        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """
        Apply one agent action and return the resulting state + reward.

        Args:
            action: Action(action_type=str, target=str)

        Returns:
            StepResult with updated observation, per-step reward, done flag,
            and an info dict containing the grader score if done=True.
        """
        # Guard: episode already finished
        if self._done:
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
                info={"error": "episode_already_done"},
            )

        # ── Crisis: fire regulatory event if step threshold reached ────────
        self._maybe_fire_regulatory_event()

        prev_progress = self._state.get("progress", 0.0)

        # ── Validate action format ──────────────────────────────────────────
        format_error = self._validate_action_format(action)
        if format_error:
            return self._penalise(
                result="invalid_action",
                error=format_error,
                reward_key="invalid_action",
            )

        # ── Crisis: check for invalidated document usage ───────────────────
        crisis_violation = self._crisis_rule_check(action)
        if crisis_violation:
            result, error, extra_info = crisis_violation
            reward_val = -0.3
            self._state["previous_actions"].append(action.to_key())
            self._state["progress"] = self._compute_progress()
            self._steps_taken += 1
            self._state["deadline_days"] = max(0, self._state["deadline_days"] - 1)
            self._cumulative_reward = max(
                0.0, min(1.0, self._cumulative_reward + reward_val)
            )
            self._last_action_result = result
            self._last_action_error = error
            self._check_terminal_conditions()
            return StepResult(
                observation=self._build_observation(),
                reward=max(-1.0, min(1.0, reward_val)),
                done=self._done,
                info={"result": result, "error": error, **(extra_info or {})},
            )

        # ── Check for repeated action ───────────────────────────────────────
        action_key = action.to_key()
        if action_key in self._state["previous_actions"]:
            return self._penalise(
                result="repeated_action",
                error=f"Action '{action_key}' already performed.",
                reward_key="repeated_action",
            )

        # ── Dispatch to action handler ──────────────────────────────────────
        result, error, extra_info = self._dispatch(action)

        # ── Compute per-step reward ─────────────────────────────────────────
        reward_val = compute_reward(
            action=action,
            result=result,
            state=self._state,
            prev_progress=prev_progress,
        )

        # ── Record action in history ────────────────────────────────────────
        # Only lock out actions that SUCCEEDED or hit penalties we want to
        # prevent repeating. Actions that failed with recoverable errors
        # (like "event not fired yet", "dependency not met") should NOT be
        # blocked from future retry — the agent may need to call them later
        # when conditions change.
        RECORDED_RESULTS = {
            "success",              # succeeded — legitimately done
            "rule_violation",       # hit a rule trap — don't let agent retry the same mistake
            "invalid_action",       # malformed — don't spam it
        }
        if result in RECORDED_RESULTS:
            self._state["previous_actions"].append(action_key)

        # ── Update progress ─────────────────────────────────────────────────
        self._state["progress"] = self._compute_progress()

        # ── Tick clock ──────────────────────────────────────────────────────
        self._steps_taken += 1
        self._state["deadline_days"] = max(0, self._state["deadline_days"] - 1)

        # ── Accumulate episode reward ───────────────────────────────────────
        self._cumulative_reward = max(
            0.0, min(1.0, self._cumulative_reward + reward_val)
        )

        # ── Update last action metadata ─────────────────────────────────────
        self._last_action_result = result
        self._last_action_error = error

        # ── Check deadline / step-limit termination ─────────────────────────
        if not self._done:
            self._check_terminal_conditions()

        # ── Build info payload ──────────────────────────────────────────────
        info: dict[str, Any] = {"result": result}
        if error:
            info["error"] = error
        if extra_info:
            info.update(extra_info)

        # Always attach grader score when episode ends
        if self._done and "final_score" not in info:
            from env.graders import grade
            info["final_score"] = grade(self._task_name, self._state)
            info["status"] = self._state["status"]

        return StepResult(
            observation=self._build_observation(),
            reward=max(-1.0, min(1.0, reward_val)),
            done=self._done,
            info=info,
        )

    def state(self) -> WorkforceState:
        """Return the current full environment state as a typed Pydantic model."""
        return self._dict_to_model()

    # -----------------------------------------------------------------------
    # Penalty helper (for invalid / repeated actions)
    # -----------------------------------------------------------------------

    def _penalise(
        self, result: str, error: str, reward_key: str
    ) -> StepResult:
        """
        Apply a penalty step without mutating case state.
        Still ticks the clock and accumulates reward.
        """
        reward_val = REWARDS.get(reward_key, -0.2)
        self._last_action_result = result
        self._last_action_error = error
        self._steps_taken += 1
        self._state["deadline_days"] = max(0, self._state["deadline_days"] - 1)
        self._cumulative_reward = max(
            0.0, min(1.0, self._cumulative_reward + reward_val)
        )
        self._check_terminal_conditions()
        return StepResult(
            observation=self._build_observation(),
            reward=max(-1.0, min(1.0, reward_val)),
            done=self._done,
            info={"result": result, "error": error},
        )

    # -----------------------------------------------------------------------
    # Action dispatcher
    # -----------------------------------------------------------------------

    def _dispatch(
        self, action: Action
    ) -> tuple[str, str | None, dict[str, Any]]:
        """Route action to the correct handler."""
        handlers = {
            "request_document":               self._handle_request_document,
            "verify_document":                self._handle_verify_document,
            "approve_hr":                     self._handle_approve_department,
            "approve_legal":                  self._handle_approve_department,
            "approve_finance":                self._handle_approve_department,
            "set_payroll":                    self._handle_set_compliance,
            "set_tax_id":                     self._handle_set_compliance,
            "set_shadow_payroll":             self._handle_set_compliance,
            "set_pdpa":                       self._handle_set_compliance,
            "resolve_conflict":               self._handle_resolve_conflict,
            "acknowledge_regulatory_change":  self._handle_acknowledge_regulatory_change,
            "finalize_case":                  self._handle_finalize_case,
        }
        handler = handlers.get(action.action_type)
        if not handler:
            return "invalid_action", f"No handler for '{action.action_type}'", {}
        return handler(action)

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _handle_request_document(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        doc_name = action.target
        doc = self._state["documents"].get(doc_name)
        if not doc:
            return "invalid_action", f"Document '{doc_name}' not in this case", {}
        if doc["status"] != "missing":
            return "wrong_action", f"Document '{doc_name}' already {doc['status']}", {}

        self._state["documents"][doc_name]["status"] = "submitted"
        self._state["documents"][doc_name]["is_valid"] = True  # deterministic
        return "success", None, {"detail": f"Document '{doc_name}' submitted"}

    def _handle_verify_document(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        doc_name = action.target
        doc = self._state["documents"].get(doc_name)
        if not doc:
            return "invalid_action", f"Document '{doc_name}' not in this case", {}

        status = doc.get("status", "missing")
        if status == "missing":
            return (
                "prereq_violated",
                f"Document '{doc_name}' has not been submitted yet. "
                f"Call request_document first.",
                {},
            )
        if status == "verified":
            return "wrong_action", f"Document '{doc_name}' is already verified", {}
        if status == "rejected":
            return (
                "wrong_action",
                f"Document '{doc_name}' was rejected. Re-request before verifying.",
                {},
            )

        is_approved, reason = validate_document(self._state, doc_name)

        if is_approved:
            self._state["documents"][doc_name]["status"] = "verified"
            return "success", None, {"detail": f"Document '{doc_name}' verified"}
        else:
            self._state["documents"][doc_name]["status"] = "rejected"
            return "wrong_action", f"Document '{doc_name}' rejected: {reason}", {}

    def _handle_approve_department(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        dept_map = {
            "approve_hr":      "HR",
            "approve_legal":   "Legal",
            "approve_finance": "Finance",
        }
        dept = dept_map[action.action_type]

        if self._state["departments"][dept]:
            return "wrong_action", f"{dept} already approved", {}

        ok, reason = validate_department_prerequisites(self._state, dept)
        if not ok:
            return "prereq_violated", reason, {}

        self._state["departments"][dept] = True
        extra: dict[str, Any] = {"detail": f"{dept} approved"}
        if all(self._state["departments"].values()):
            extra["milestone"] = "all_departments_approved"
        return "success", None, extra

    def _handle_set_compliance(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """
        Unified handler for set_payroll, set_tax_id, set_shadow_payroll, set_pdpa.
        """
        action_type = action.action_type

        compliance_field_map = {
            "set_payroll":        "payroll",
            "set_tax_id":         "tax_id",
            "set_shadow_payroll": "shadow_payroll",
            "set_pdpa":           "pdpa",
        }
        field = compliance_field_map[action_type]

        if self._state["compliance"][field]:
            return "wrong_action", f"{field} already configured", {}

        country = action.target or (
            self._state["countries"][0] if self._state["countries"] else ""
        )
        ok, reason = validate_compliance_action(self._state, action_type, country)
        if not ok:
            return "rule_violation", reason, {}

        self._state["compliance"][field] = True
        return "success", None, {"detail": f"{field} configured for {country}"}

    def _handle_resolve_conflict(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """Handle resolve_conflict for hard task."""
        conflicts = self._state.get("conflicts", [])
        unresolved = [c for c in conflicts if not c.get("resolved", False)]

        if not unresolved:
            return "wrong_action", "No unresolved conflicts in this case", {}

        for c in self._state["conflicts"]:
            if not c.get("resolved", False):
                c["resolved"] = True
                return "success", None, {
                    "detail": f"Conflict resolved: {c['rule']}",
                    "milestone": "conflict_resolved",
                }

        return "wrong_action", "No conflicts to resolve", {}

    # -----------------------------------------------------------------------
    # CRISIS TASK — Regulatory event injection + acknowledgement handler
    # -----------------------------------------------------------------------

    def _maybe_fire_regulatory_event(self) -> None:
        """
        Called at the START of each step() before action dispatch.
        Fires the crisis task's regulatory event when step threshold is reached.
        Mutates state to inject ict_permit and invalidate the old visa.

        Safe no-op for all non-crisis tasks.
        """
        if self._task_name != "crisis":
            return

        event_fired  = self._state.get("regulatory_event_fired", False)
        event_step   = self._state.get("regulatory_event_step", 8)

        # Only fire once, at the designated step.
        # _steps_taken is 0-indexed and increments AFTER each step completes,
        # so at the start of the Nth step, _steps_taken == N-1.
        # To fire "at step 8" (the 8th action), we fire when _steps_taken == 7.
        if event_fired or self._steps_taken < event_step - 1:
            return

        # ── Fire the event ──────────────────────────────────────────────────
        self._state["regulatory_event_fired"] = True
        event = self._state.get("regulatory_event", {}) or {}

        # 1. Handle the invalidated document.
        #    Critical fix: we mark it as status="verified" so downstream
        #    validators (Legal/Finance "all docs verified") pass. The crisis
        #    rule check in _crisis_rule_check() still traps any request/verify
        #    attempts on this doc after the event, so agents can't use the
        #    old workflow. Effectively: the doc is "archived" — its
        #    compliance requirement is satisfied by the ict_permit replacement.
        invalidated_doc = event.get("invalidates_document", "visa")
        if invalidated_doc in self._state["documents"]:
            self._state["documents"][invalidated_doc]["status"] = "verified"
            self._state["documents"][invalidated_doc]["is_valid"] = False   # logical flag for grader

        # 2. Inject the new required document (ict_permit) into state
        new_doc = event.get("requires_new_document", "ict_permit")
        if new_doc not in self._state["documents"]:
            self._state["documents"][new_doc] = {
                "status":   "missing",
                "is_valid": True,
            }

        # 3. Log event as a system marker in action history
        event_id = event.get("id", "REGULATORY_EVENT")
        self._state["previous_actions"].append(f"[SYSTEM_EVENT:{event_id}]")

    def _handle_acknowledge_regulatory_change(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """
        Handler for acknowledge_regulatory_change.

        Only valid in crisis task after the regulatory event has fired.
        """
        if self._task_name != "crisis":
            return (
                "wrong_action",
                "acknowledge_regulatory_change is only valid in the 'crisis' task",
                {},
            )

        event_fired = self._state.get("regulatory_event_fired", False)
        if not event_fired:
            return (
                "wrong_action",
                (
                    "No regulatory event has fired yet. "
                    "Continue the normal relocation workflow."
                ),
                {},
            )

        already_acknowledged = self._state.get("regulatory_event_acknowledged", False)
        if already_acknowledged:
            return (
                "wrong_action",
                "Regulatory change already acknowledged.",
                {},
            )

        # ── Acknowledge ─────────────────────────────────────────────────────
        self._state["regulatory_event_acknowledged"] = True
        event = self._state.get("regulatory_event", {}) or {}

        return (
            "success",
            None,
            {
                "detail": (
                    f"Regulatory change acknowledged: {event.get('title', 'Unknown event')}. "
                    f"Old visa document invalidated. "
                    f"New document required: {event.get('requires_new_document', 'ict_permit')}. "
                    f"Request and verify 'ict_permit' to continue."
                ),
                "milestone": "regulatory_change_acknowledged",
                "new_document_required": event.get("requires_new_document", "ict_permit"),
            },
        )

    def _crisis_rule_check(
        self, action: Action
    ) -> tuple[str, str | None, dict] | None:
        """
        Called inside step() BEFORE dispatching any action.
        Returns a rule_violation result if the agent tries to use the
        invalidated visa document after the regulatory event has fired.

        Returns None if no violation — normal dispatch continues.
        """
        if self._task_name != "crisis":
            return None

        event_fired = self._state.get("regulatory_event_fired", False)
        if not event_fired:
            return None

        event = self._state.get("regulatory_event", {}) or {}
        invalidated_doc = event.get("invalidates_document", "visa")

        # Penalise any attempt to request or verify the invalidated document
        if action.action_type in ("request_document", "verify_document"):
            if action.target == invalidated_doc:
                return (
                    "rule_violation",
                    (
                        f"Cannot use '{invalidated_doc}' after the regulatory event. "
                        f"The Blue Card visa has been suspended. "
                        f"Acknowledge the change and use 'ict_permit' instead."
                    ),
                    {"penalty": event.get("penalty_if_used_after_event", -0.3)},
                )
        return None

    # -----------------------------------------------------------------------
    # Finalize handler
    # -----------------------------------------------------------------------

    def _handle_finalize_case(
        self, action: Action
    ) -> tuple[str, str | None, dict]:
        """
        Attempt to close the relocation case.
        Accepts both target="" and target="all".
        """
        if action.target not in ("", "all"):
            return (
                "invalid_action",
                f"finalize_case target must be '' or 'all', got '{action.target}'",
                {},
            )

        blockers = self._get_blockers()
        if blockers:
            return (
                "rule_violation",
                "Cannot finalize — mandatory blockers remain",
                {"blockers": blockers},
            )

        from env.graders import grade
        final_score: float = grade(self._task_name, self._state)

        self._state["status"] = "success"
        self._done = True

        return "success", None, {
            "final_score": final_score,
            "status":      "success",
            "milestone":   "episode_complete",
            "detail":      f"Case finalized. Score: {final_score:.4f}",
        }

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    def _validate_action_format(self, action: Action) -> str | None:
        """Return error string if action is structurally invalid, else None."""
        if action.action_type not in VALID_ACTION_TYPES:
            return (
                f"Unknown action_type '{action.action_type}'. "
                f"Valid: {sorted(VALID_ACTION_TYPES)}"
            )

        if action.action_type in {"request_document", "verify_document"}:
            if not action.target:
                return "Document actions require a non-empty target"
            if action.target not in VALID_DOCUMENTS:
                return (
                    f"Unknown document '{action.target}'. "
                    f"Valid: {sorted(VALID_DOCUMENTS)}"
                )

        return None

    # -----------------------------------------------------------------------
    # Progress + blocker computation
    # -----------------------------------------------------------------------

    def _compute_progress(self) -> float:
        """Compute weighted progress fraction [0.0, 1.0]."""
        docs      = self._state.get("documents", {})
        depts     = self._state.get("departments", {})
        comp      = self._state.get("compliance", {})
        conflicts = self._state.get("conflicts", [])
        req_depts = self._state.get("required_departments", list(depts.keys()))
        req_comp  = self._state.get("required_compliance", [])

        # Document progress: verified=1.0, submitted=0.5, else=0.0
        if docs:
            doc_scores = []
            for d in docs.values():
                if d["status"] == "verified":
                    doc_scores.append(1.0)
                elif d["status"] == "submitted":
                    doc_scores.append(0.5)
                else:
                    doc_scores.append(0.0)
            doc_prog = sum(doc_scores) / len(doc_scores)
        else:
            doc_prog = 0.0

        # Department progress
        if req_depts:
            dept_prog = sum(1 for d in req_depts if depts.get(d, False)) / len(req_depts)
        else:
            dept_prog = 1.0

        # Compliance progress
        if req_comp:
            comp_prog = sum(1 for c in req_comp if comp.get(c, False)) / len(req_comp)
        else:
            comp_prog = 1.0

        # Conflict resolution progress
        if conflicts:
            conflict_prog = sum(1 for c in conflicts if c.get("resolved", False)) / len(conflicts)
        else:
            conflict_prog = 1.0

        # Crisis: regulatory event acknowledgement adds to progress
        crisis_prog = 1.0
        if self._task_name == "crisis":
            event_fired = self._state.get("regulatory_event_fired", False)
            event_ack   = self._state.get("regulatory_event_acknowledged", False)
            if event_fired and not event_ack:
                crisis_prog = 0.5
            elif not event_fired:
                crisis_prog = 1.0  # hasn't fired yet — not blocking
            else:
                crisis_prog = 1.0  # fired and acked

        total = (
            0.38 * doc_prog
            + 0.32 * dept_prog
            + 0.15 * comp_prog
            + 0.08 * conflict_prog
            + 0.07 * crisis_prog
        )
        return round(min(max(total, 0.0), 1.0), 4)

    def _get_blockers(self) -> list[str]:
        """Return human-readable list of reasons finalize_case cannot proceed."""
        blockers: list[str] = []
        docs      = self._state.get("documents", {})
        depts     = self._state.get("departments", {})
        comp      = self._state.get("compliance", {})
        conflicts = self._state.get("conflicts", [])
        req_depts = self._state.get("required_departments", [])
        req_comp  = self._state.get("required_compliance", [])

        # All documents must be verified
        unverified = [k for k, v in docs.items() if v["status"] != "verified"]
        if unverified:
            blockers.append(f"Unverified documents: {unverified}")

        # Required departments must approve
        missing_depts = [d for d in req_depts if not depts.get(d, False)]
        if missing_depts:
            blockers.append(f"Pending department approvals: {missing_depts}")

        # Required compliance must be set
        missing_comp = [c for c in req_comp if not comp.get(c, False)]
        if missing_comp:
            blockers.append(f"Incomplete compliance items: {missing_comp}")

        # All conflicts must be resolved
        unresolved = [c["rule"] for c in conflicts if not c.get("resolved", False)]
        if unresolved:
            blockers.append(f"Unresolved conflicts: {unresolved}")

        # Crisis task: block finalization until regulatory event is acknowledged
        if self._task_name == "crisis":
            event_fired = self._state.get("regulatory_event_fired", False)
            event_ack   = self._state.get("regulatory_event_acknowledged", False)
            if event_fired and not event_ack:
                blockers.append(
                    "Regulatory event unacknowledged: call acknowledge_regulatory_change first"
                )

        return blockers

    # -----------------------------------------------------------------------
    # Terminal condition check
    # -----------------------------------------------------------------------

    def _check_terminal_conditions(self) -> None:
        """Handle deadline and step-limit termination only."""
        if self._done:
            return
        if self._state.get("deadline_days", 1) <= 0:
            self._state["status"] = "failed"
            self._done = True
            self._last_action_error = "Episode failed: deadline reached"
        elif self._steps_taken >= MAX_STEPS:
            self._state["status"] = "failed"
            self._done = True
            self._last_action_error = f"Episode failed: max steps ({MAX_STEPS}) reached"

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct the Observation returned to the agent after every step."""
        return Observation(
            state=self._dict_to_model(),
            available_actions=self._get_available_actions(),
            current_blockers=self._get_blockers(),
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            steps_taken=self._steps_taken,
            done=self._done,
        )

    def _get_available_actions(self) -> list[str]:
        """Return useful actions the agent can take. Format: 'action_type:target'"""
        available: list[str] = []
        docs      = self._state.get("documents", {})
        depts     = self._state.get("departments", {})
        comp      = self._state.get("compliance", {})
        countries = self._state.get("countries", [])
        conflicts = self._state.get("conflicts", [])

        # Crisis task: if event fired and unacknowledged, surface it first
        if self._task_name == "crisis":
            event_fired = self._state.get("regulatory_event_fired", False)
            event_ack   = self._state.get("regulatory_event_acknowledged", False)
            if event_fired and not event_ack:
                available.insert(0, "acknowledge_regulatory_change")
                # Don't return here — still show other possible actions

        # Document actions
        for doc_name, doc in docs.items():
            if doc["status"] == "missing":
                available.append(f"request_document:{doc_name}")
            elif doc["status"] == "submitted":
                available.append(f"verify_document:{doc_name}")

        # Department actions
        if not depts.get("HR", False):
            available.append("approve_hr")
        if not depts.get("Legal", False):
            ok, _ = validate_department_prerequisites(self._state, "Legal")
            if ok:
                available.append("approve_legal")
        if not depts.get("Finance", False):
            ok, _ = validate_department_prerequisites(self._state, "Finance")
            if ok:
                available.append("approve_finance")

        # Compliance actions
        for country in countries:
            rules = COUNTRY_RULES.get(country, {})
            if rules.get("requires_payroll") and not comp.get("payroll"):
                available.append(f"set_payroll:{country}")
            if rules.get("requires_tax_id") and not comp.get("tax_id"):
                available.append(f"set_tax_id:{country}")
            if rules.get("requires_pdpa") and not comp.get("pdpa"):
                available.append(f"set_pdpa:{country}")
            if rules.get("requires_shadow_payroll") and not comp.get("shadow_payroll"):
                available.append(f"set_shadow_payroll:{country}")

        # Conflict resolution
        unresolved = [c for c in conflicts if not c.get("resolved", False)]
        if unresolved:
            available.append("resolve_conflict")

        # Finalize
        if not self._get_blockers():
            available.append("finalize_case")

        return available

    # -----------------------------------------------------------------------
    # State conversion: dict → WorkforceState Pydantic model
    # -----------------------------------------------------------------------

    def _dict_to_model(self) -> WorkforceState:
        """Convert internal state dict to a typed WorkforceState Pydantic model."""
        s = self._state
        if not s:
            return WorkforceState(
                case_id="none",
                task_name="easy",
                employee=EmployeeRecord(role="Engineer", has_dependents=False),
                countries=["Germany"],
            )

        return WorkforceState(
            case_id=s["case_id"],
            task_name=s.get("task_name", self._task_name),
            employee=EmployeeRecord(
                role=s["employee"]["role"],
                has_dependents=s["employee"]["has_dependents"],
            ),
            countries=s["countries"],
            documents={
                name: DocumentRecord(
                    status=doc["status"],
                    is_valid=doc["is_valid"],
                )
                for name, doc in s["documents"].items()
            },
            departments=DepartmentStatus(
                HR=s["departments"]["HR"],
                Legal=s["departments"]["Legal"],
                Finance=s["departments"]["Finance"],
            ),
            compliance=ComplianceStatus(
                tax_id=s["compliance"]["tax_id"],
                payroll=s["compliance"]["payroll"],
                pdpa=s["compliance"]["pdpa"],
                shadow_payroll=s["compliance"]["shadow_payroll"],
            ),
            conflicts=[
                ConflictRecord(
                    countries=c["countries"],
                    rule=c["rule"],
                    resolved=c.get("resolved", False),
                )
                for c in s.get("conflicts", [])
            ],
            deadline_days=s["deadline_days"],
            previous_actions=s["previous_actions"],
            progress=s.get("progress", 0.0),
            status=s["status"],
            required_departments=s.get("required_departments", []),
            required_compliance=s.get("required_compliance", []),
            # Crisis fields
            regulatory_event_fired=s.get("regulatory_event_fired", False),
            regulatory_event_acknowledged=s.get("regulatory_event_acknowledged", False),
            regulatory_event=s.get("regulatory_event", None),
            regulatory_event_step=s.get("regulatory_event_step", 9999),
        )