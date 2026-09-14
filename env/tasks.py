"""
env/tasks.py
============
Task fixtures for meridian.

TASKS is a dict mapping task_name → initial state dict.
environment.py deep-copies the relevant entry on reset() so the fixture
is never mutated between episodes.

Task summary:
  easy:   India → Germany, Engineer, no dependents, 20 days
          Required: 4 docs + HR approval + tax_id + payroll
          
  medium: India → Singapore, Manager with dependents, 25 days
          Required: 3 docs + HR + Legal + payroll + pdpa + shadow_payroll
          
  hard:   India → Germany + UAE simultaneously, Director, 30 days
          Required: 4 docs + HR + Legal + Finance + tax_id(DE only) + payroll(both)
          KEY TRAP: UAE has no income tax — set_tax_id:UAE = rule_violation

  crisis: India → Germany, Manager, 35 days
          Required: 4 docs (passport, employment_letter, work_permit, ict_permit)
                    + HR + Legal + tax_id + payroll
          KEY MECHANIC: At step 8 a regulatory event fires — Blue Card visa
          suspended. Agent must acknowledge and switch to ict_permit.
          Using old visa after event = rule_violation penalty.

State dict schema matches WorkforceState in env/models.py exactly.
"""

from __future__ import annotations


TASKS: dict[str, dict] = {

    # =========================================================================
    # TASK 1 — EASY
    # India → Germany, Engineer, no dependents
    #
    # Optimal sequence (~9 steps):
    #   request_document x4 → verify_document x4 → approve_hr →
    #   set_tax_id → set_payroll → finalize_case
    #
    # All documents is_valid=True — no traps.
    # Only HR required (Legal/Finance not required for easy).
    # =========================================================================
    "easy": {
        "case_id":   "CASE-001-EASY",
        "task_name": "easy",
        "employee": {
            "role":           "Engineer",
            "has_dependents": False,
        },
        "countries": ["Germany"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "work_permit": {
                "status":   "missing",
                "is_valid": True,
            },
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,
            "shadow_payroll": False,
        },
        "conflicts": [],
        "deadline_days":      20,
        "previous_actions":   [],
        "progress":           0.0,
        "status":             "in_progress",
        "required_departments": ["HR"],
        "required_compliance":  ["tax_id", "payroll"],
        # Crisis fields — inactive for this task
        "regulatory_event_fired":        False,
        "regulatory_event_acknowledged": False,
        "regulatory_event_step":         9999,
        "regulatory_event":              None,
    },

    # =========================================================================
    # TASK 2 — MEDIUM
    # India → Singapore, Manager with dependents
    #
    # Optimal sequence (~12 steps):
    #   request_document x3 → verify_document x3 → approve_hr →
    #   approve_legal → set_payroll → set_pdpa → set_shadow_payroll →
    #   finalize_case
    # =========================================================================
    "medium": {
        "case_id":   "CASE-002-MEDIUM",
        "task_name": "medium",
        "employee": {
            "role":           "Manager",
            "has_dependents": True,
        },
        "countries": ["Singapore"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,
            "shadow_payroll": False,
        },
        "conflicts": [],
        "deadline_days":      25,
        "previous_actions":   [],
        "progress":           0.0,
        "status":             "in_progress",
        "required_departments": ["HR", "Legal"],
        "required_compliance":  ["payroll", "pdpa", "shadow_payroll"],
        # Crisis fields — inactive for this task
        "regulatory_event_fired":        False,
        "regulatory_event_acknowledged": False,
        "regulatory_event_step":         9999,
        "regulatory_event":              None,
    },

    # =========================================================================
    # TASK 3 — HARD
    # India → Germany + UAE simultaneously, Director with dependents
    #
    # Optimal sequence (~16 steps):
    #   request_document x4 → verify_document x4 → approve_hr →
    #   approve_legal → set_tax_id (Germany ONLY) → set_payroll →
    #   resolve_conflict → approve_finance → finalize_case
    #
    # KEY TRAPS:
    #   1. UAE has NO income tax → set_tax_id:UAE = -0.3 rule_violation
    #   2. Finance requires Legal + all conflicts resolved
    # =========================================================================
    "hard": {
        "case_id":   "CASE-003-HARD",
        "task_name": "hard",
        "employee": {
            "role":           "Director",
            "has_dependents": True,
        },
        "countries": ["Germany", "UAE"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "work_permit": {
                "status":   "missing",
                "is_valid": True,
            },
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,
            "shadow_payroll": False,
        },
        "conflicts": [
            {
                "countries": ["Germany", "UAE"],
                "rule": (
                    "tax_conflict: Germany requires tax_id registration "
                    "but UAE has no income tax. "
                    "Register tax_id for Germany only. "
                    "Do NOT call set_tax_id for UAE."
                ),
                "resolved": False,
            }
        ],
        "deadline_days":      30,
        "previous_actions":   [],
        "progress":           0.0,
        "status":             "in_progress",
        "required_departments": ["HR", "Legal", "Finance"],
        "required_compliance":  ["tax_id", "payroll"],
        # Crisis fields — inactive for this task
        "regulatory_event_fired":        False,
        "regulatory_event_acknowledged": False,
        "regulatory_event_step":         9999,
        "regulatory_event":              None,
    },

    # =========================================================================
    # TASK 4 — CRISIS
    # India → Germany, Manager, no dependents — mid-episode regulatory change
    #
    # Phase 1 (steps 1-7): Normal Germany relocation flow.
    #   request/verify: passport, employment_letter, work_permit
    #   approve_hr
    #
    # Phase 2 (auto-triggered at step 8):
    #   Regulatory event fires:
    #   "Germany has suspended the Blue Card visa — switch to ICT Permit"
    #
    #   The agent MUST:
    #     1. Call acknowledge_regulatory_change
    #     2. Call request_document:ict_permit  (new doc injected into state)
    #     3. Call verify_document:ict_permit
    #     4. NOT use old "visa" doc (= rule_violation -0.3)
    #
    # Phase 3 (steps ~13-20): Resume normal flow.
    #   approve_legal → set_tax_id → set_payroll → finalize_case
    #
    # Score ceiling: 0.89
    # Expected range: 0.40 – 0.90
    # =========================================================================
    "crisis": {
        "case_id":   "CASE-004-CRISIS",
        "task_name": "crisis",
        "employee": {
            "role":           "Manager",
            "has_dependents": False,
        },
        "countries": ["Germany"],
        "documents": {
            "passport": {
                "status":   "missing",
                "is_valid": True,
            },
            "visa": {
                "status":   "missing",
                "is_valid": True,
            },
            "employment_letter": {
                "status":   "missing",
                "is_valid": True,
            },
            "work_permit": {
                "status":   "missing",
                "is_valid": True,
            },
            # NOTE: "ict_permit" is NOT here at episode start.
            # It is injected dynamically when the regulatory event fires at step 8.
        },
        "departments": {
            "HR":      False,
            "Legal":   False,
            "Finance": False,
        },
        "compliance": {
            "tax_id":         False,
            "payroll":        False,
            "pdpa":           False,
            "shadow_payroll": False,
        },
        "conflicts": [],
        "deadline_days":      35,
        "previous_actions":   [],
        "progress":           0.0,
        "status":             "in_progress",
        "required_departments": ["HR", "Legal"],
        "required_compliance":  ["tax_id", "payroll"],

        # ── Crisis-specific fields ──────────────────────────────────────────
        "regulatory_event_fired":        False,
        "regulatory_event_acknowledged": False,
        "regulatory_event_step":         8,
        "regulatory_event": {
            "id":          "DE-VISA-SUSPENSION-2024",
            "title":       "Germany Blue Card Visa Suspension",
            "description": (
                "REGULATORY ALERT: Germany has suspended the Blue Card visa "
                "programme effective immediately. All pending Blue Card visas "
                "are invalidated. Affected employees must switch to the "
                "ICT (Intra-Company Transfer) Permit. "
                "Action required: acknowledge this change and request an "
                "ict_permit document to replace the invalidated visa."
            ),
            "invalidates_document":      "visa",
            "requires_new_document":     "ict_permit",
            "penalty_if_used_after_event": -0.3,
        },
    },
}


# ---------------------------------------------------------------------------
# Task metadata — used by /tasks endpoint and TASK_INFO lookups
# ---------------------------------------------------------------------------

TASK_INFO: dict[str, dict] = {
    "easy": {
        "name":        "easy",
        "description": (
            "Relocate an Engineer from India to Germany. "
            "Single country, no dependents, all documents valid. "
            "Requires: 4 docs verified + HR approval + tax_id + payroll."
        ),
        "countries":            ["Germany"],
        "max_steps":            25,
        "expected_score_range": (0.70, 1.00),
        "departments_required": ["HR"],
        "compliance_required":  ["tax_id", "payroll"],
        "key_rules": [
            "Germany requires visa and work_permit",
            "Tax ID registration required for Germany",
            "Only HR approval required for easy task",
        ],
    },
    "medium": {
        "name":        "medium",
        "description": (
            "Relocate a Manager with dependents from India to Singapore. "
            "Requires Employment Pass, PDPA consent, shadow payroll. "
            "HR + Legal approval required."
        ),
        "countries":            ["Singapore"],
        "max_steps":            25,
        "expected_score_range": (0.40, 0.80),
        "departments_required": ["HR", "Legal"],
        "compliance_required":  ["payroll", "pdpa", "shadow_payroll"],
        "key_rules": [
            "Singapore requires PDPA consent before data processing",
            "Shadow payroll mandatory for home-country tax tracking",
            "Singapore does NOT require tax_id for foreign workers",
            "Legal must approve before finalize",
        ],
    },
    "hard": {
        "name":        "hard",
        "description": (
            "Simultaneous relocation of a Director from India to Germany + UAE. "
            "Multi-country compliance with conflicting tax rules. "
            "KEY TRAP: UAE has no income tax — set_tax_id:UAE is a rule violation."
        ),
        "countries":            ["Germany", "UAE"],
        "max_steps":            25,
        "expected_score_range": (0.20, 0.60),
        "departments_required": ["HR", "Legal", "Finance"],
        "compliance_required":  ["tax_id", "payroll"],
        "key_rules": [
            "UAE has NO income tax — set_tax_id:UAE = -0.3 penalty",
            "Germany requires tax_id — call set_tax_id for Germany only",
            "Both countries require payroll configuration",
            "Finance requires Legal approval AND all conflicts resolved",
            "resolve_conflict must be called before approve_finance",
        ],
    },
    "crisis": {
        "name":        "crisis",
        "description": (
            "Relocate a Manager from India to Germany — but mid-episode "
            "a regulatory alert fires: Germany suspends the Blue Card visa. "
            "The agent must detect the change, acknowledge it, swap to ICT Permit, "
            "and complete the relocation. Tests long-horizon planning + adaptation."
        ),
        "countries":            ["Germany"],
        "max_steps":            35,
        "expected_score_range": (0.40, 0.90),
        "departments_required": ["HR", "Legal"],
        "compliance_required":  ["tax_id", "payroll"],
        "key_rules": [
            "Regulatory event fires at step 8 — Blue Card visa suspended",
            "Agent must call acknowledge_regulatory_change to clear the event",
            "Old 'visa' document becomes invalid after the event fires",
            "New 'ict_permit' document must be requested and verified",
            "Using visa after event fires = -0.3 rule_violation penalty",
            "Germany requires tax_id and payroll",
            "HR + Legal approval required",
        ],
    },
}