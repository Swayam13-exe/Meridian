"""
env/graders.py
==============
Re-export shim for the graders module.

All actual grading logic lives in graders/graders.py.
This module exposes the canonical import path used by environment.py:

    from env.graders import grade
"""

from graders.graders import (
    grade,
    grade_easy,
    grade_medium,
    grade_hard,
    grade_crisis,      # ← ADDED for crisis task support
    grade_all,
    explain,
)

__all__ = [
    "grade",
    "grade_easy",
    "grade_medium",
    "grade_hard",
    "grade_crisis",    # ← ADDED
    "grade_all",
    "explain",
]