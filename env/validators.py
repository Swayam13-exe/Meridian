"""
env/validators.py
=================
Pure validation functions for meridian.

All functions:
  - Are pure — no side effects, no state mutation
  - Return (is_valid: bool, reason: str)
  - Are called by environment.py BEFORE any state is mutated

Design principle: validators check ONLY what they need to.
  - Document validation: only checks is_valid flag (status already
    checked by environment.py before calling this)
  - Department prerequisites: checks actual state dict directly,
    NOT re-derived from country rules
  - Compliance actions: checks country rules for special restrictions
"""

from __future__ import annotations

from typing import Any

# Country rules embedded directly — avoids circular imports and JSON file reads
# on every validation call. Mirrors fixtures/country_rules.json.
_COUNTRY_RULES: dict[str, dict[str, Any]] = {
    "Germany": {
        "requires_visa":           True,
        "requires_tax_id":         True,
        "has_income_tax":          True,
        "requires_payroll":        True,
        "requires_pdpa":           False,
        "requires_shadow_payroll": False,
        "departments_required":    ["HR", "Legal", "Finance"],
    },
    "Singapore": {
        "requires_visa":           True,
        "requires_tax_id":         False,
        "has_income_tax":          True,
        "requires_payroll":        True,
        "requires_pdpa":           True,
        "requires_shadow_payroll": True,
        "departments_required":    ["HR", "Legal"],
    },
    "UAE": {
        "requires_visa":           True,
        "requires_tax_id":         False,
        "has_income_tax":          False,   # ← KEY: no income tax
        "requires_payroll":        True,
        "requires_pdpa":           False,
        "requires_shadow_payroll": False,
        "departments_required":    ["HR", "Legal", "Finance"],
    },
}


# ---------------------------------------------------------------------------
# validate_document
# ---------------------------------------------------------------------------


def validate_document(
    state: dict[str, Any],
    doc_name: str,
) -> tuple[bool, str]:
    """
    Validate whether a submitted document passes verification.

    Called by environment._handle_verify_document AFTER confirming:
      - Document exists in state
      - Document status is "submitted"

    This function only needs to check the is_valid ground-truth flag.
    Status checks are NOT repeated here to avoid double-validation.

    Args:
        state:    Current episode state dict.
        doc_name: Name of the document to validate.

    Returns:
        (True, "ok") if document passes.
        (False, reason_string) if it fails.
    """
    docs = state.get("documents", {})
    doc = docs.get(doc_name)

    if doc is None:
        return False, f"Document '{doc_name}' does not exist in this case"

    # Primary check: ground-truth validity flag
    # This is set to True when request_document is called (deterministic)
    if not doc.get("is_valid", False):
        return (
            False,
            f"Document '{doc_name}' failed validation — "
            f"the submitted document is not valid for this relocation case.",
        )

    return True, "ok"


# ---------------------------------------------------------------------------
# validate_department_prerequisites
# ---------------------------------------------------------------------------


def validate_department_prerequisites(
    state: dict[str, Any],
    dept: str,
) -> tuple[bool, str]:
    """
    Validate that all prerequisites for a department approval are met.

    Checks actual state dict directly — does NOT re-derive requirements
    from country rules to avoid mismatches with task definitions.

    Ordering constraints:
        HR:      No prerequisites.
        Legal:   ALL documents in state["documents"] must be "verified".
        Finance: Legal must already be approved.
                 All unresolved conflicts must be resolved.

    Args:
        state: Current episode state dict.
        dept:  "HR" | "Legal" | "Finance"

    Returns:
        (True, "ok") if prerequisites met.
        (False, reason) if not.
    """
    departments = state.get("departments", {})
    documents   = state.get("documents", {})
    conflicts   = state.get("conflicts", [])

    if dept == "HR":
        return True, "ok"

    elif dept == "Legal":
        # All documents in the case must be verified
        unverified = [
            name for name, doc in documents.items()
            if doc.get("status") != "verified"
        ]
        if unverified:
            return (
                False,
                f"Legal approval requires all documents to be verified first. "
                f"Not yet verified: {unverified}",
            )
        return True, "ok"

    elif dept == "Finance":
        # Legal must be approved first
        if not departments.get("Legal", False):
            return (
                False,
                "Finance approval requires Legal approval first. "
                "Approve Legal before Finance.",
            )
        # All conflicts must be resolved (hard task)
        unresolved = [c["rule"] for c in conflicts if not c.get("resolved", False)]
        if unresolved:
            return (
                False,
                f"Finance approval blocked by unresolved conflicts: {unresolved}. "
                f"Call resolve_conflict first.",
            )
        return True, "ok"

    else:
        return False, f"Unknown department '{dept}'"


# ---------------------------------------------------------------------------
# validate_compliance_action
# ---------------------------------------------------------------------------


def validate_compliance_action(
    state: dict[str, Any],
    action_type: str,
    country: str,
) -> tuple[bool, str]:
    """
    Validate a compliance-setting action against country-specific rules.

    Critical domain rules enforced here:

      set_tax_id:
        UAE has NO income tax — this is ALWAYS a rule_violation for UAE.
        This is the key trap in the hard task (Germany + UAE).

      set_shadow_payroll:
        Only Singapore requires shadow payroll.

      set_pdpa:
        Only Singapore requires PDPA consent.

      set_payroll:
        Required for all countries — always valid if not yet set.

    Args:
        state:       Current episode state dict.
        action_type: "set_tax_id" | "set_shadow_payroll" | "set_pdpa" | "set_payroll"
        country:     Target country name (may be empty string)

    Returns:
        (True, "ok") if compliant.
        (False, reason) if rule violation.
    """
    # If no country specified, infer from case
    if not country:
        countries = state.get("countries", [])
        country = countries[0] if countries else ""

    country_rules = _COUNTRY_RULES.get(country, {})
    case_countries = state.get("countries", [])

    # Country must be part of this case
    if country and country not in case_countries:
        return (
            False,
            f"Country '{country}' is not part of this relocation case. "
            f"Case countries: {case_countries}",
        )

    if action_type == "set_tax_id":
        # CRITICAL: UAE no-income-tax rule
        # Check ALL case countries — if any has no income tax, warn
        for c in case_countries:
            rules = _COUNTRY_RULES.get(c, {})
            if not rules.get("has_income_tax", True):
                # Only block if this specific country has no tax
                if c == country or not country:
                    return (
                        False,
                        f"RULE VIOLATION: {c} has no income tax. "
                        f"Do NOT call set_tax_id for {c}. "
                        f"Only call set_tax_id for countries with income tax (e.g. Germany).",
                    )

        if not country_rules.get("requires_tax_id", False):
            return (
                False,
                f"Tax ID is not required for {country}.",
            )
        return True, "ok"

    elif action_type == "set_shadow_payroll":
        if not country_rules.get("requires_shadow_payroll", False):
            return (
                False,
                f"Shadow payroll is not required for {country}. "
                f"Only Singapore requires shadow payroll.",
            )
        return True, "ok"

    elif action_type == "set_pdpa":
        if not country_rules.get("requires_pdpa", False):
            return (
                False,
                f"PDPA consent is not required for {country}. "
                f"Only Singapore requires PDPA.",
            )
        return True, "ok"

    elif action_type == "set_payroll":
        if not country_rules.get("requires_payroll", True):
            return (
                False,
                f"Payroll is not required for {country}.",
            )
        return True, "ok"

    # Unknown action — pass through
    return True, "ok"