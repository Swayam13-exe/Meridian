"""
env/rules.py
============
Rules engine for meridian.

Responsibilities:
  - Load and cache country_rules.json, visa_types.json, tax_treaties.json
  - Expose get_rules(home, host)           → dict of rules for a country pair
  - Expose validate_action(action, state)  → (is_valid, reason, penalty_type)
  - Expose get_blockers(state)             → list[str] of unresolved blockers
  - Expose get_required_documents(country) → list[str]
  - Expose get_required_compliance(country)→ list[str]

All functions are pure — no side effects, no external calls.
All return values are deterministic given the same inputs.

Design principle: rules live in JSON fixtures, not in Python logic.
Adding a new country means adding a JSON entry, not changing code.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from env.models import Action

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

_COUNTRY_RULES_PATH  = _FIXTURES_DIR / "country_rules.json"
_VISA_TYPES_PATH     = _FIXTURES_DIR / "visa_types.json"
_TAX_TREATIES_PATH   = _FIXTURES_DIR / "tax_treaties.json"


# ---------------------------------------------------------------------------
# Fixture loaders (cached — loaded once at startup)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_country_rules() -> dict[str, Any]:
    """Load and cache country_rules.json."""
    if not _COUNTRY_RULES_PATH.exists():
        # Fallback to inline defaults if fixture file not found
        return _DEFAULT_COUNTRY_RULES
    with open(_COUNTRY_RULES_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_visa_types() -> dict[str, Any]:
    """Load and cache visa_types.json."""
    if not _VISA_TYPES_PATH.exists():
        return _DEFAULT_VISA_TYPES
    with open(_VISA_TYPES_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_tax_treaties() -> dict[str, Any]:
    """Load and cache tax_treaties.json."""
    if not _TAX_TREATIES_PATH.exists():
        return _DEFAULT_TAX_TREATIES
    with open(_TAX_TREATIES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inline defaults (used when fixture files are not present)
# These mirror the fixture files exactly — single source of truth
# ---------------------------------------------------------------------------

_DEFAULT_COUNTRY_RULES: dict[str, Any] = {
    "Germany": {
        "requires_visa":            True,
        "correct_visa":             "EU Blue Card",
        "visa_options":             ["EU Blue Card", "Work Permit", "Job Seeker Visa"],
        "requires_tax_id":          True,
        "tax_authority":            "Bundeszentralamt für Steuern",
        "requires_payroll":         True,
        "requires_local_contract":  True,
        "requires_pdpa":            False,
        "requires_shadow_payroll":  False,
        "has_income_tax":           True,
        "tax_rate_approx_pct":      42,
        "social_security_treaty":   True,
        "data_privacy_law":         "GDPR",
        "required_documents": [
            "passport",
            "visa",
            "employment_letter",
            "degree_certificate"
        ],
        "compliance_items": [
            "tax_id",
            "payroll"
        ],
        "departments_required": ["HR", "Legal"],
        "notes": "EU Blue Card preferred for skilled workers. GDPR applies."
    },
    "Singapore": {
        "requires_visa":            True,
        "correct_visa":             "Employment Pass",
        "visa_options":             ["Employment Pass", "S Pass", "Work Permit"],
        "requires_tax_id":          True,
        "tax_authority":            "Inland Revenue Authority of Singapore",
        "requires_payroll":         True,
        "requires_local_contract":  True,
        "requires_pdpa":            True,
        "requires_shadow_payroll":  True,
        "has_income_tax":           True,
        "tax_rate_approx_pct":      22,
        "social_security_treaty":   False,
        "data_privacy_law":         "PDPA",
        "required_documents": [
            "passport",
            "visa",
            "employment_letter"
        ],
        "compliance_items": [
            "tax_id",
            "payroll",
            "pdpa",
            "shadow_payroll"
        ],
        "departments_required": ["HR", "Legal", "Finance"],
        "notes": (
            "Employment Pass required for professionals earning above threshold. "
            "Shadow payroll mandatory for home-country tax tracking. "
            "PDPA consent required before any data processing."
        )
    },
    "UAE": {
        "requires_visa":            True,
        "correct_visa":             "Employment Visa",
        "visa_options":             ["Employment Visa", "Freelance Permit"],
        "requires_tax_id":          False,
        "tax_authority":            None,
        "requires_payroll":         True,
        "requires_local_contract":  True,
        "requires_pdpa":            False,
        "requires_shadow_payroll":  False,
        "has_income_tax":           False,
        "tax_rate_approx_pct":      0,
        "social_security_treaty":   False,
        "data_privacy_law":         "UAE PDPL",
        "required_documents": [
            "passport",
            "visa",
            "employment_letter"
        ],
        "compliance_items": [
            "payroll"
        ],
        "departments_required": ["HR", "Legal", "Finance"],
        "notes": (
            "UAE has NO income tax. Calling set_tax_id for UAE is a rule violation. "
            "Payroll must still be configured. UAE PDPL applies to data handling."
        )
    }
}

_DEFAULT_VISA_TYPES: dict[str, Any] = {
    "EU Blue Card": {
        "country":          "Germany",
        "category":         "skilled_worker",
        "min_salary_eur":   56400,
        "requires_degree":  True,
        "duration_years":   4,
        "notes":            "Preferred route for non-EU skilled workers in Germany"
    },
    "Work Permit": {
        "country":          "Germany",
        "category":         "general",
        "min_salary_eur":   0,
        "requires_degree":  False,
        "duration_years":   2,
        "notes":            "Fallback if EU Blue Card salary threshold not met"
    },
    "Employment Pass": {
        "country":          "Singapore",
        "category":         "professional",
        "min_salary_sgd":   5000,
        "requires_degree":  True,
        "duration_years":   2,
        "notes":            "For professionals, managers, executives. Correct visa for Manager/Engineer roles."
    },
    "S Pass": {
        "country":          "Singapore",
        "category":         "mid_skilled",
        "min_salary_sgd":   3000,
        "requires_degree":  False,
        "duration_years":   2,
        "notes":            "Wrong choice for senior roles — triggers wrong_action penalty"
    },
    "Employment Visa": {
        "country":          "UAE",
        "category":         "employment",
        "min_salary_aed":   0,
        "requires_degree":  False,
        "duration_years":   2,
        "notes":            "Standard employment visa for UAE. No income tax obligations."
    }
}

_DEFAULT_TAX_TREATIES: dict[str, Any] = {
    "IN-DE": {
        "home":             "India",
        "host":             "Germany",
        "treaty_id":        "IN-DE-2011",
        "avoids_double_tax": True,
        "social_security":  True,
        "notes":            "India-Germany DTAA 2011. Prevents double taxation on salary income."
    },
    "IN-SG": {
        "home":             "India",
        "host":             "Singapore",
        "treaty_id":        "IN-SG-1994",
        "avoids_double_tax": True,
        "social_security":  False,
        "notes":            "India-Singapore DTAA 1994. No social security agreement."
    },
    "IN-UAE": {
        "home":             "India",
        "host":             "UAE",
        "treaty_id":        "IN-UAE-1992",
        "avoids_double_tax": True,
        "social_security":  False,
        "notes":            (
            "India-UAE DTAA 1992. UAE has no income tax so treaty mainly "
            "governs investment income."
        )
    }
}


# ---------------------------------------------------------------------------
# Department dependency rules (code — not JSON, intentional)
# These are structural constraints that don't vary by country
# ---------------------------------------------------------------------------

DEPARTMENT_DEPENDENCIES: dict[str, list[str]] = {
    "HR":      [],
    "Legal":   ["documents_verified"],
    "Finance": ["legal_approved"],
}

# Required document sets per country (convenience alias — mirrors JSON)
REQUIRED_DOCUMENTS: dict[str, list[str]] = {
    country: data["required_documents"]
    for country, data in _DEFAULT_COUNTRY_RULES.items()
}

# Which compliance items each country needs (convenience alias)
REQUIRED_COMPLIANCE: dict[str, list[str]] = {
    country: data["compliance_items"]
    for country, data in _DEFAULT_COUNTRY_RULES.items()
}

# Full country rules dict (convenience alias for direct import)
COUNTRY_RULES: dict[str, Any] = _DEFAULT_COUNTRY_RULES


# ---------------------------------------------------------------------------
# Public API — get_rules
# ---------------------------------------------------------------------------

def get_rules(home: str, host: str) -> dict[str, Any]:
    """
    Return the combined rules for a country pair (home → host).

    Args:
        home: Origin country (currently always "India").
        host: Destination country ("Germany" | "Singapore" | "UAE").

    Returns:
        Dict containing:
          - All host country rules from country_rules.json
          - tax_treaty: treaty dict if one exists for this pair, else None
          - pair_key: "IN-{host_code}" string

    Raises:
        ValueError: If host country is not supported.
    """
    country_rules = _load_country_rules()
    tax_treaties  = _load_tax_treaties()

    if host not in country_rules:
        raise ValueError(
            f"Country '{host}' not supported. "
            f"Supported countries: {list(country_rules.keys())}"
        )

    # Build country-code pair key for tax treaty lookup
    home_code = _country_code(home)
    host_code = _country_code(host)
    pair_key  = f"{home_code}-{host_code}"

    rules = dict(country_rules[host])  # shallow copy — don't mutate fixture
    rules["tax_treaty"] = tax_treaties.get(pair_key)
    rules["pair_key"]   = pair_key
    rules["home"]       = home
    rules["host"]       = host

    return rules


def get_visa_info(visa_type: str) -> dict[str, Any] | None:
    """
    Return metadata for a specific visa type.

    Args:
        visa_type: e.g. "EU Blue Card", "Employment Pass"

    Returns:
        Visa metadata dict, or None if not found.
    """
    return _load_visa_types().get(visa_type)


def get_tax_treaty(home: str, host: str) -> dict[str, Any] | None:
    """
    Return the tax treaty for a country pair, or None if none exists.

    Args:
        home: Origin country.
        host: Destination country.
    """
    home_code = _country_code(home)
    host_code = _country_code(host)
    pair_key  = f"{home_code}-{host_code}"
    return _load_tax_treaties().get(pair_key)


def get_required_documents(country: str) -> list[str]:
    """
    Return the list of required documents for a destination country.

    Args:
        country: Destination country name.

    Returns:
        List of document name strings.

    Raises:
        ValueError: If country is not supported.
    """
    rules = _load_country_rules()
    if country not in rules:
        raise ValueError(f"Country '{country}' not supported")
    return list(rules[country]["required_documents"])


def get_required_compliance(country: str) -> list[str]:
    """
    Return the list of required compliance items for a destination country.

    Args:
        country: Destination country name.

    Returns:
        List of compliance item strings (e.g. ["tax_id", "payroll"]).
    """
    rules = _load_country_rules()
    if country not in rules:
        raise ValueError(f"Country '{country}' not supported")
    return list(rules[country]["compliance_items"])


# ---------------------------------------------------------------------------
# Public API — validate_action
# ---------------------------------------------------------------------------

def validate_action(
    action: Action,
    state: dict[str, Any],
) -> tuple[bool, str, str]:
    """
    Validate an agent action against current state and country rules.

    This is the central rule-checking function. It is called by environment.py
    before any state mutation occurs.

    Args:
        action: The Action model submitted by the agent.
        state:  Current episode state dict.

    Returns:
        Tuple of (is_valid: bool, reason: str, penalty_type: str)

        penalty_type is one of:
          ""                — no penalty (action is valid)
          "invalid_action"  — unknown action or malformed
          "rule_violation"  — action violates a country rule (-0.3)
          "prereq_violated" — prerequisite not met (-0.3)
          "wrong_action"    — action is valid type but wrong for current state (-0.2)
          "repeated_action" — same action already in history (-0.1)
    """
    action_type = action.action_type
    target      = action.target
    countries   = state.get("countries", [])

    # ── 1. Repeated action check ────────────────────────────────────────────
    action_key = action.to_key()
    if action_key in state.get("previous_actions", []):
        return False, f"Action '{action_key}' already performed this episode", "repeated_action"

    # ── 2. Route to per-action validator ───────────────────────────────────
    validators = {
        "request_document":   _validate_request_document,
        "verify_document":    _validate_verify_document,
        "approve_hr":         _validate_approve_department,
        "approve_legal":      _validate_approve_department,
        "approve_finance":    _validate_approve_department,
        "set_payroll":        _validate_set_payroll,
        "set_tax_id":         _validate_set_tax_id,
        "set_shadow_payroll": _validate_set_shadow_payroll,
        "set_pdpa":           _validate_set_pdpa,
        "finalize_case":      _validate_finalize_case,
    }

    if action_type not in validators:
        return (
            False,
            f"Unknown action_type '{action_type}'",
            "invalid_action",
        )

    return validators[action_type](target, state, countries)


# ---------------------------------------------------------------------------
# Per-action validators — each returns (is_valid, reason, penalty_type)
# ---------------------------------------------------------------------------

def _validate_request_document(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    valid_docs = {"passport", "visa", "employment_letter", "degree_certificate"}
    if target not in valid_docs:
        return False, f"Unknown document '{target}'", "invalid_action"

    doc = state["documents"].get(target)
    if not doc:
        return False, f"Document '{target}' not part of this case", "invalid_action"

    if doc["status"] != "missing":
        return (
            False,
            f"Document '{target}' is already {doc['status']} — cannot request again",
            "wrong_action",
        )

    # Check whether this document is actually required for any country in case
    required_for_any = False
    for country in countries:
        if target in get_required_documents(country):
            required_for_any = True
            break

    if not required_for_any:
        return (
            False,
            f"Document '{target}' is not required for any country in this case",
            "wrong_action",
        )

    return True, "ok", ""


def _validate_verify_document(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    valid_docs = {"passport", "visa", "employment_letter", "degree_certificate"}
    if target not in valid_docs:
        return False, f"Unknown document '{target}'", "invalid_action"

    doc = state["documents"].get(target)
    if not doc:
        return False, f"Document '{target}' not part of this case", "invalid_action"

    if doc["status"] == "missing":
        return (
            False,
            f"Document '{target}' has not been submitted yet — call request_document first",
            "prereq_violated",
        )

    if doc["status"] in {"verified", "rejected"}:
        return (
            False,
            f"Document '{target}' is already {doc['status']}",
            "wrong_action",
        )

    return True, "ok", ""


def _validate_approve_department(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    dept_map = {
        "approve_hr":      "HR",
        "approve_legal":   "Legal",
        "approve_finance": "Finance",
    }
    # target here is the action_type being passed through — normalise
    # NOTE: environment.py passes action.action_type as target for dept actions
    # but validate_action receives action.target. We accept either.
    dept = dept_map.get(target) or (target if target in {"HR", "Legal", "Finance"} else None)
    if not dept:
        return False, f"Unknown department '{target}'", "invalid_action"

    if state["departments"].get(dept):
        return False, f"{dept} already approved", "wrong_action"

    # Check prerequisites using DEPARTMENT_DEPENDENCIES
    prereqs = DEPARTMENT_DEPENDENCIES.get(dept, [])
    for prereq in prereqs:
        if prereq == "documents_verified":
            for country in countries:
                for doc in get_required_documents(country):
                    if state["documents"].get(doc, {}).get("status") != "verified":
                        return (
                            False,
                            f"Legal requires all documents verified first. "
                            f"Document '{doc}' for {country} is not yet verified.",
                            "prereq_violated",
                        )
        elif prereq == "legal_approved":
            if not state["departments"].get("Legal"):
                return (
                    False,
                    "Finance requires Legal approval first",
                    "prereq_violated",
                )

    return True, "ok", ""


def _validate_set_payroll(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    if target not in countries:
        return (
            False,
            f"Country '{target}' is not part of this relocation case. "
            f"Case countries: {countries}",
            "invalid_action",
        )

    if state["compliance"].get("payroll"):
        return False, "Payroll already configured", "wrong_action"

    rules = _load_country_rules().get(target, {})
    if not rules.get("requires_payroll", False):
        return (
            False,
            f"Payroll not required for {target}",
            "rule_violation",
        )

    return True, "ok", ""


def _validate_set_tax_id(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    """
    Critical rule: UAE has no income tax.
    Calling set_tax_id with target=UAE is always a rule_violation.
    This is the key differentiator in the hard task.
    """
    if target not in countries:
        return (
            False,
            f"Country '{target}' is not part of this relocation case",
            "invalid_action",
        )

    rules = _load_country_rules().get(target, {})

    # ── THE KEY UAE RULE ────────────────────────────────────────────────────
    if not rules.get("has_income_tax", True):
        return (
            False,
            f"Rule violation: {target} has no income tax. "
            f"Tax ID registration is not required and must not be called for {target}. "
            f"This action incurs a -0.3 penalty.",
            "rule_violation",
        )

    if not rules.get("requires_tax_id", False):
        return (
            False,
            f"Tax ID not required for {target}",
            "rule_violation",
        )

    if state["compliance"].get("tax_id"):
        return False, "Tax ID already registered", "wrong_action"

    return True, "ok", ""


def _validate_set_shadow_payroll(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    if target not in countries:
        return (
            False,
            f"Country '{target}' is not part of this case",
            "invalid_action",
        )

    rules = _load_country_rules().get(target, {})
    if not rules.get("requires_shadow_payroll", False):
        return (
            False,
            f"Shadow payroll is not required for {target}. "
            f"Only Singapore requires shadow payroll.",
            "rule_violation",
        )

    if state["compliance"].get("shadow_payroll"):
        return False, "Shadow payroll already enabled", "wrong_action"

    return True, "ok", ""


def _validate_set_pdpa(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    if target not in countries:
        return (
            False,
            f"Country '{target}' is not part of this case",
            "invalid_action",
        )

    rules = _load_country_rules().get(target, {})
    if not rules.get("requires_pdpa", False):
        return (
            False,
            f"PDPA is not required for {target}. "
            f"Only Singapore requires PDPA consent.",
            "rule_violation",
        )

    if state["compliance"].get("pdpa"):
        return False, "PDPA consent already collected", "wrong_action"

    return True, "ok", ""


def _validate_finalize_case(
    target: str,
    state: dict[str, Any],
    countries: list[str],
) -> tuple[bool, str, str]:
    if target != "all":
        return (
            False,
            f"finalize_case target must be 'all', got '{target}'",
            "invalid_action",
        )

    blockers = get_blockers(state)
    if blockers:
        return (
            False,
            f"Cannot finalize — {len(blockers)} blocker(s) remain: {blockers[0]}",
            "rule_violation",
        )

    if state.get("status") == "success":
        return False, "Case already finalized", "wrong_action"

    return True, "ok", ""


# ---------------------------------------------------------------------------
# Public API — get_blockers
# ---------------------------------------------------------------------------

def get_blockers(state: dict[str, Any]) -> list[str]:
    """
    Return all unresolved blockers preventing case finalization.

    An empty list means finalize_case is valid to call.

    Blockers are checked in this order:
      1. Documents — all required docs must be verified
      2. Departments — required departments must have approved (in order)
      3. Compliance — all country-specific compliance items must be complete

    Args:
        state: Current episode state dict.

    Returns:
        List of human-readable blocker strings. Empty = no blockers.
    """
    blockers: list[str] = []
    countries   = state.get("countries", [])
    documents   = state.get("documents", {})
    departments = state.get("departments", {})
    compliance  = state.get("compliance", {})

    # ── 1. Document blockers ────────────────────────────────────────────────
    for country in countries:
        required_docs = get_required_documents(country)
        for doc_name in required_docs:
            doc = documents.get(doc_name, {})
            status = doc.get("status", "missing")
            if status != "verified":
                blockers.append(
                    f"[DOCUMENT] '{doc_name}' not verified for {country} "
                    f"(current status: '{status}')"
                )

    # ── 2. Department blockers ──────────────────────────────────────────────
    # Determine which departments are required for this case
    required_depts = _get_required_departments(countries)

    for dept in ["HR", "Legal", "Finance"]:   # enforce ordering in blocker list
        if dept in required_depts and not departments.get(dept):
            blockers.append(
                f"[DEPARTMENT] {dept} approval pending"
            )

    # ── 3. Compliance blockers ──────────────────────────────────────────────
    for country in countries:
        required_compliance = get_required_compliance(country)

        for item in required_compliance:
            if not compliance.get(item):
                label = _compliance_label(item, country)
                blockers.append(
                    f"[COMPLIANCE] {label} not completed for {country}"
                )

    # ── 4. Special: multi-country Finance requirement ───────────────────────
    if len(countries) > 1 and "Finance" not in _get_required_departments(countries):
        if not departments.get("Finance"):
            blockers.append(
                "[DEPARTMENT] Finance approval required for multi-country cases"
            )

    return blockers


def get_blockers_summary(state: dict[str, Any]) -> dict[str, list[str]]:
    """
    Return blockers grouped by category for structured reporting.

    Returns:
        Dict with keys "documents", "departments", "compliance",
        each containing a list of blocker strings for that category.
    """
    all_blockers = get_blockers(state)
    grouped: dict[str, list[str]] = {
        "documents":   [],
        "departments": [],
        "compliance":  [],
    }
    for blocker in all_blockers:
        if blocker.startswith("[DOCUMENT]"):
            grouped["documents"].append(blocker)
        elif blocker.startswith("[DEPARTMENT]"):
            grouped["departments"].append(blocker)
        elif blocker.startswith("[COMPLIANCE]"):
            grouped["compliance"].append(blocker)
    return grouped


# ---------------------------------------------------------------------------
# Progress computation (used by environment.py and graders.py)
# ---------------------------------------------------------------------------

def compute_checklist(state: dict[str, Any]) -> dict[str, Any]:
    """
    Compute a structured checklist of all required items and their completion.

    Used by:
      - environment.py  → _compute_progress()
      - graders.py      → weighted scoring

    Returns:
        Dict with keys:
          "total":     int  — total checklist items
          "completed": int  — completed items
          "progress":  float [0.0, 1.0]
          "items":     list of { "category", "name", "country", "done" }
    """
    countries   = state.get("countries", [])
    documents   = state.get("documents", {})
    departments = state.get("departments", {})
    compliance  = state.get("compliance", {})
    items: list[dict[str, Any]] = []

    # Documents
    for country in countries:
        for doc_name in get_required_documents(country):
            done = documents.get(doc_name, {}).get("status") == "verified"
            items.append({
                "category": "document",
                "name":     doc_name,
                "country":  country,
                "done":     done,
            })

    # Departments
    required_depts = _get_required_departments(countries)
    for dept in ["HR", "Legal", "Finance"]:
        if dept in required_depts:
            items.append({
                "category": "department",
                "name":     dept,
                "country":  "all",
                "done":     bool(departments.get(dept)),
            })

    # Compliance
    for country in countries:
        for item in get_required_compliance(country):
            items.append({
                "category": "compliance",
                "name":     item,
                "country":  country,
                "done":     bool(compliance.get(item)),
            })

    total     = len(items)
    completed = sum(1 for i in items if i["done"])
    progress  = round(completed / total, 4) if total > 0 else 0.0

    return {
        "total":     total,
        "completed": completed,
        "progress":  progress,
        "items":     items,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_required_departments(countries: list[str]) -> list[str]:
    """
    Determine which departments must approve for the given country list.

    Rules:
      - HR always required
      - Legal always required
      - Finance required for: Singapore, multi-country, or any country
        whose rules specify Finance in departments_required
    """
    required = {"HR", "Legal"}
    country_rules = _load_country_rules()

    for country in countries:
        rules = country_rules.get(country, {})
        for dept in rules.get("departments_required", []):
            required.add(dept)

    # Multi-country always requires Finance
    if len(countries) > 1:
        required.add("Finance")

    return [d for d in ["HR", "Legal", "Finance"] if d in required]


def _compliance_label(item: str, country: str) -> str:
    """Return a human-readable label for a compliance item."""
    labels = {
        "tax_id":         f"Tax ID registration ({country})",
        "payroll":        f"Host-country payroll setup ({country})",
        "pdpa":           "PDPA consent (Singapore)",
        "shadow_payroll": "Shadow payroll (Singapore)",
    }
    return labels.get(item, item)


def _country_code(country: str) -> str:
    """Map country name to ISO 2-letter code for treaty key construction."""
    codes = {
        "India":     "IN",
        "Germany":   "DE",
        "Singapore": "SG",
        "UAE":       "UAE",
    }
    return codes.get(country, country[:2].upper())


# ---------------------------------------------------------------------------
# Convenience exports (for direct import in other modules)
# ---------------------------------------------------------------------------

__all__ = [
    "COUNTRY_RULES",
    "DEPARTMENT_DEPENDENCIES",
    "REQUIRED_DOCUMENTS",
    "REQUIRED_COMPLIANCE",
    "get_rules",
    "get_visa_info",
    "get_tax_treaty",
    "get_required_documents",
    "get_required_compliance",
    "validate_action",
    "get_blockers",
    "get_blockers_summary",
    "compute_checklist",
]
