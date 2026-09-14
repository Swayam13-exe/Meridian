"""
env/models.py
=============
All Pydantic v2 models for meridian.

Schema hierarchy:
  EmployeeRecord          — who is being relocated
  DocumentRecord          — a single document's status + validity
  DepartmentStatus        — HR / Legal / Finance approval flags
  ComplianceStatus        — tax, payroll, PDPA, shadow payroll flags
  WorkforceState          — complete episode state (single source of truth)
  Action                  — one agent action: {action_type, target}
  Observation             — what the agent sees after each step
  Reward                  — per-step reward with reason
  StepResult              — full return value of step()
  ResetResult             — return value of reset()
  TaskInfo                — metadata for a single task
  EpisodeSummary          — end-of-episode summary with final grader score

All models use Pydantic v2 syntax exclusively.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

DocumentStatus = Literal["missing", "submitted", "verified", "rejected"]
EpisodeStatus  = Literal["in_progress", "success", "failed"]
TaskName       = Literal["easy", "medium", "hard", "crisis"]   # ← crisis added
CountryName    = Literal["Germany", "Singapore", "UAE"]
RoleName       = Literal["Engineer", "Manager", "Director", "Analyst"]
DepartmentName = Literal["HR", "Legal", "Finance"]

ActionType = Literal[
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
    "acknowledge_regulatory_change",   # ← crisis task action
    "finalize_case",
]

DocumentName = Literal[
    "passport",
    "visa",
    "employment_letter",
    "degree_certificate",
    "work_permit",          # Germany
    "employment_pass",      # Singapore
    "residence_permit",     # UAE
    "tax_form",
    "ict_permit",           # Germany crisis task — replaces visa after event
]


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class EmployeeRecord(BaseModel):
    """
    Describes the employee being relocated.

    Fields:
        role:            Job title / seniority level.
        has_dependents:  True if spouse or children are moving too.
    """

    role: RoleName = Field(
        ...,
        description="Employee job role",
        examples=["Engineer", "Manager"],
    )
    has_dependents: bool = Field(
        default=False,
        description="Whether dependents are included in the relocation",
    )

    model_config = {"frozen": True}


class DocumentRecord(BaseModel):
    """
    Tracks a single relocation document's lifecycle.

    Status transitions:
        missing → submitted → verified
                           → rejected
    """

    status: DocumentStatus = Field(
        default="missing",
        description="Current document lifecycle stage",
    )
    is_valid: bool = Field(
        default=False,
        description="Ground-truth validity. Determines verify_document outcome.",
    )

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        allowed = {"missing", "submitted", "verified", "rejected"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v


class DepartmentStatus(BaseModel):
    """
    Tracks approval status across the three internal departments.

    Approval ordering constraint (enforced by validators.py):
        HR      → no prerequisites
        Legal   → all required documents must be verified first
        Finance → Legal must be approved first
    """

    HR:      bool = Field(default=False, description="HR department approval")
    Legal:   bool = Field(default=False, description="Legal department approval")
    Finance: bool = Field(default=False, description="Finance department approval")

    def approved_count(self) -> int:
        return sum([self.HR, self.Legal, self.Finance])

    def all_approved(self) -> bool:
        return self.HR and self.Legal and self.Finance


class ComplianceStatus(BaseModel):
    """
    Tracks country-specific compliance items.

    Which fields are required depends on destination country:
        Germany:   tax_id + payroll
        Singapore: payroll + pdpa + shadow_payroll  (no tax_id)
        UAE:       payroll only (no income tax — tax_id must NOT be set)
    """

    tax_id:         bool = Field(default=False, description="Tax ID registered")
    payroll:        bool = Field(default=False, description="Payroll configured")
    pdpa:           bool = Field(default=False, description="PDPA consent collected")
    shadow_payroll: bool = Field(default=False, description="Shadow payroll enabled")

    def completed_count(self) -> int:
        return sum([self.tax_id, self.payroll, self.pdpa, self.shadow_payroll])


class ConflictRecord(BaseModel):
    """
    A rule conflict that the agent must resolve before Finance can approve.
    Only present in the hard task (Germany + UAE tax conflict).
    """

    countries: list[str] = Field(..., description="Countries involved in conflict")
    rule:      str        = Field(..., description="Description of the conflicting rule")
    resolved:  bool       = Field(default=False, description="Whether conflict is resolved")


# ---------------------------------------------------------------------------
# Master state model
# ---------------------------------------------------------------------------


class WorkforceState(BaseModel):
    """
    The complete episode state — single source of truth for the environment.
    Returned by state() and embedded in every Observation.
    """

    case_id:           str                          = Field(..., description="Unique case identifier")
    task_name:         TaskName                     = Field(..., description="Task being run")
    employee:          EmployeeRecord               = Field(..., description="Employee being relocated")
    countries:         list[str]                    = Field(..., min_length=1, max_length=2, description="Destination countries")
    documents:         dict[str, DocumentRecord]    = Field(default_factory=dict, description="Document name → status")
    departments:       DepartmentStatus             = Field(default_factory=DepartmentStatus, description="Department approvals")
    compliance:        ComplianceStatus             = Field(default_factory=ComplianceStatus, description="Compliance flags")
    conflicts:         list[ConflictRecord]         = Field(default_factory=list, description="Active rule conflicts")
    deadline_days:     int                          = Field(default=5, ge=0, description="Steps before auto-fail")
    previous_actions:  list[str]                    = Field(default_factory=list, description="Action history")
    progress:          float                        = Field(default=0.0, ge=0.0, le=1.0, description="Completion fraction")
    status:            EpisodeStatus                = Field(default="in_progress", description="Episode status")

    # Track required items per task (set at reset time, never change)
    required_departments: list[str] = Field(default_factory=list, description="Departments required for this task")
    required_compliance:  list[str] = Field(default_factory=list, description="Compliance items required for this task")

    # ── Crisis task fields (default False/None — ignored for easy/medium/hard) ─
    regulatory_event_fired:        bool       = Field(default=False, description="True after event fires at step N")
    regulatory_event_acknowledged: bool       = Field(default=False, description="True after agent acknowledges")
    regulatory_event:              dict | None = Field(default=None,  description="Event payload (crisis only)")
    regulatory_event_step:         int        = Field(default=9999,  description="Step at which event fires")

    @field_validator("countries")
    @classmethod
    def countries_must_be_valid(cls, v: list[str]) -> list[str]:
        valid = {"Germany", "Singapore", "UAE"}
        for c in v:
            if c not in valid:
                raise ValueError(f"Country '{c}' not supported. Valid: {valid}")
        return v

    @field_validator("progress")
    @classmethod
    def progress_in_range(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    @field_validator("deadline_days")
    @classmethod
    def deadline_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("deadline_days cannot be negative")
        return v

    def is_done(self) -> bool:
        return self.status in {"success", "failed"}

    def get_verified_documents(self) -> list[str]:
        return [name for name, doc in self.documents.items() if doc.status == "verified"]

    def get_missing_documents(self) -> list[str]:
        return [name for name, doc in self.documents.items() if doc.status == "missing"]

    def get_submitted_documents(self) -> list[str]:
        return [name for name, doc in self.documents.items() if doc.status == "submitted"]

    def get_rejected_documents(self) -> list[str]:
        return [name for name, doc in self.documents.items() if doc.status == "rejected"]

    def all_docs_verified(self) -> bool:
        return all(doc.status == "verified" for doc in self.documents.values())

    def unresolved_conflicts(self) -> list[ConflictRecord]:
        return [c for c in self.conflicts if not c.resolved]


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------


class Action(BaseModel):
    """
    One agent action submitted to step().

    Valid action_type + target combinations:
        request_document              → document name (e.g. "passport")
        verify_document               → document name
        approve_hr                    → "" or "HR"
        approve_legal                 → "" or "Legal"
        approve_finance               → "" or "Finance"
        set_payroll                   → "" or country name
        set_tax_id                    → "" or country name
        set_shadow_payroll            → "" or "Singapore"
        set_pdpa                      → "" or "Singapore"
        resolve_conflict              → "" or conflict description
        acknowledge_regulatory_change → "" (crisis task only)
        finalize_case                 → ""
    """

    action_type: str = Field(
        ...,
        description="The type of action to perform",
        examples=["verify_document", "approve_hr", "finalize_case"],
    )
    target: str = Field(
        default="",
        description="Subject of the action. Empty string for department/compliance actions.",
        examples=["passport", "HR", "Germany", ""],
    )

    @field_validator("action_type")
    @classmethod
    def action_type_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("action_type cannot be empty")
        return v.strip().lower()

    @field_validator("target")
    @classmethod
    def target_strip(cls, v: str) -> str:
        return v.strip() if v else ""

    def to_key(self) -> str:
        """Return 'action_type:target' string for repeat detection."""
        if self.target:
            return f"{self.action_type}:{self.target}"
        return self.action_type


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------


class Reward(BaseModel):
    """Per-step reward returned alongside the observation."""

    value:  float = Field(..., description="Reward value")
    reason: str   = Field(..., description="Human-readable explanation")

    @field_validator("value")
    @classmethod
    def clamp_reward(cls, v: float) -> float:
        return round(max(-1.0, min(1.0, v)), 4)


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """
    The full observation returned to the agent by reset() and step().

    The agent receives the complete WorkforceState so it can plan
    multi-step sequences (document → verify → approve chain).
    """

    state:               WorkforceState = Field(..., description="Complete current episode state")
    available_actions:   list[str]      = Field(default_factory=list, description="Suggested valid actions")
    current_blockers:    list[str]      = Field(default_factory=list, description="Reasons case cannot be finalized")
    last_action_result:  str            = Field(default="", description="Result code of last action")
    last_action_error:   str | None     = Field(default=None, description="Error detail if last action failed")
    steps_taken:         int            = Field(default=0, ge=0, description="Steps taken this episode")
    done:                bool           = Field(default=False, description="True when episode ended")

    def is_success(self) -> bool:
        return self.done and self.state.status == "success"

    def is_failed(self) -> bool:
        return self.done and self.state.status == "failed"

    def progress_pct(self) -> str:
        return f"{self.state.progress * 100:.1f}%"


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """Complete return value of environment.step(action)."""

    observation: Observation    = Field(..., description="Updated observation post-action")
    reward:      float          = Field(..., description="Per-step reward in [-1.0, 1.0]")
    done:        bool           = Field(..., description="Episode termination flag")
    info:        dict[str, Any] = Field(default_factory=dict, description="Extra context")

    @field_validator("reward")
    @classmethod
    def clamp_step_reward(cls, v: float) -> float:
        return round(max(-1.0, min(1.0, v)), 4)

    def final_score(self) -> float | None:
        return self.info.get("final_score")

    def had_error(self) -> bool:
        return "error" in self.info


# ---------------------------------------------------------------------------
# ResetResult
# ---------------------------------------------------------------------------


class ResetResult(BaseModel):
    """Return value of environment.reset(task_name)."""

    observation: Observation = Field(..., description="Initial episode observation")
    task_name:   TaskName    = Field(..., description="Task that was loaded")
    session_id:  str         = Field(..., description="Unique session ID for this episode")
    task_info:   "TaskInfo"  = Field(..., description="Metadata about the loaded task")


# ---------------------------------------------------------------------------
# TaskInfo
# ---------------------------------------------------------------------------


class TaskInfo(BaseModel):
    """Metadata about a single environment task."""

    name:                 TaskName              = Field(..., description="Task identifier")
    description:          str                   = Field(..., description="Task description")
    countries:            list[str]             = Field(..., description="Destination countries")
    max_steps:            int                   = Field(..., gt=0, description="Max steps before auto-fail")
    expected_score_range: tuple[float, float]   = Field(..., description="(min, max) expected baseline score")
    departments_required: list[str]             = Field(default_factory=list)
    compliance_required:  list[str]             = Field(default_factory=list)
    key_rules:            list[str]             = Field(default_factory=list)

    @field_validator("expected_score_range")
    @classmethod
    def score_range_valid(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"Score range must be 0.0 <= lo <= hi <= 1.0, got {v}")
        return v


# ---------------------------------------------------------------------------
# EpisodeSummary
# ---------------------------------------------------------------------------


class EpisodeSummary(BaseModel):
    """Summary produced at the end of an episode (when done=True)."""

    task_name:             TaskName      = Field(..., description="Task that was run")
    session_id:            str           = Field(..., description="Session identifier")
    status:                EpisodeStatus = Field(..., description="Episode outcome")
    final_score:           float         = Field(..., ge=0.0, le=1.0, description="Grader score")
    steps_taken:           int           = Field(..., ge=0, description="Total steps used")
    cumulative_reward:     float         = Field(..., ge=0.0, le=1.0, description="Accumulated reward")
    verified_documents:    list[str]     = Field(default_factory=list)
    departments_approved:  list[str]     = Field(default_factory=list)
    compliance_completed:  list[str]     = Field(default_factory=list)
    remaining_blockers:    list[str]     = Field(default_factory=list)
    action_history:        list[str]     = Field(default_factory=list)

    @field_validator("final_score", "cumulative_reward")
    @classmethod
    def clamp_scores(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 4)

    def passed(self) -> bool:
        return self.status == "success"

    def efficiency_ratio(self) -> float:
        if self.steps_taken == 0:
            return 0.0
        return round(self.final_score / self.steps_taken, 4)


# ---------------------------------------------------------------------------
# HTTP request/response models for server/app.py
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    """Request body for POST /reset."""
    task_name:  TaskName    = Field(default="easy", description="Task to load")
    session_id: str | None  = Field(default=None, description="Optional session ID")


class StepRequest(BaseModel):
    """Request body for POST /step."""
    session_id: str    = Field(..., description="Session ID from reset()")
    action:     Action = Field(..., description="Action to execute")


class StateRequest(BaseModel):
    """Query parameters for GET /state."""
    session_id: str = Field(..., description="Session ID from reset()")


class HealthResponse(BaseModel):
    """Response for GET /health — used by HuggingFace Space ping check."""
    status:  Literal["ok"] = Field(default="ok")
    name:    str            = Field(default="meridian")
    version: str            = Field(default="1.0.0")


# ---------------------------------------------------------------------------
# Update forward references
# ---------------------------------------------------------------------------

ResetResult.model_rebuild()