"""
env/rules.py
============
Re-export layer for the rules engine.

All actual rule data and logic lives in env/rules_engine.py.
This module provides the canonical import path used by environment.py
and other env/ modules:

    from env.rules import COUNTRY_RULES, DEPARTMENT_DEPENDENCIES, REQUIRED_DOCUMENTS
"""

from env.rules_engine import (  # noqa: F401  (re-exports)
    COUNTRY_RULES,
    DEPARTMENT_DEPENDENCIES,
    REQUIRED_COMPLIANCE,
    REQUIRED_DOCUMENTS,
    compute_checklist,
    get_blockers,
    get_blockers_summary,
    get_required_compliance,
    get_required_documents,
    get_rules,
    get_tax_treaty,
    get_visa_info,
    validate_action,
)

__all__ = [
    "COUNTRY_RULES",
    "DEPARTMENT_DEPENDENCIES",
    "REQUIRED_COMPLIANCE",
    "REQUIRED_DOCUMENTS",
    "compute_checklist",
    "get_blockers",
    "get_blockers_summary",
    "get_required_compliance",
    "get_required_documents",
    "get_rules",
    "get_tax_treaty",
    "get_visa_info",
    "validate_action",
]
