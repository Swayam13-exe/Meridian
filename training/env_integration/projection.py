"""
training/env_integration/projection.py
========================================
Parses the model's raw text response into an (action_type, target) pair
the environment understands.

Split responsibility, on purpose:
  - THIS file only catches response-FORMAT problems: no <action> tag, or
    the tag doesn't contain valid JSON with the right keys. Those cases
    get flagged invalid (valid=0) so verl-agent's bookkeeping knows the
    model failed to follow instructions, not just made a bad move.
  - Whether the *content* is a legal move (a real action_type, a sensible
    target) is the environment's job, not this file's. WorkforceEnv
    already handles an unrecognized action_type safely -- confirmed by
    testing it directly: no crash, a -0.3 penalty, and a clear error
    message. Duplicating that validation here would just be two places
    that can disagree about what's legal.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

_ACTION_TAG = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)


def meridian_projection(text_actions: List[str]) -> Tuple[List[Tuple[str, str]], List[int]]:
    actions: List[Tuple[str, str]] = []
    valids: List[int] = []
    for text in text_actions:
        action_type, target, is_valid = _parse_one(text)
        actions.append((action_type, target))
        valids.append(is_valid)
    return actions, valids


def _parse_one(text: str) -> Tuple[str, str, int]:
    match = _ACTION_TAG.search(text)
    if not match:
        return "malformed_response", "", 0

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return "malformed_response", "", 0

    action_type = parsed.get("action_type")
    if not isinstance(action_type, str) or not action_type.strip():
        return "malformed_response", "", 0

    target = parsed.get("target", "")
    if not isinstance(target, str):
        target = str(target)

    return action_type.strip(), target, 1