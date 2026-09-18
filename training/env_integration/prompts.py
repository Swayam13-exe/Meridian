"""
training/env_integration/prompts.py
====================================
Prompt templates for the Meridian agent.

Design principle, stated explicitly because it's a real decision and not
an accident: this prompt gives the model the general RULES of the domain
(the same ones a new hire would get in an onboarding doc), but never the
solution to a specific case. The original hackathon baseline agent's
prompt spelled out the exact action sequence for the hardest task --
that's fine for a demo, but it means the "baseline" isn't really testing
reasoning, just instruction-following. This prompt is deliberately built
so a model has to actually work out what to do from the rules and the
current state, which is what makes training on it (and comparing
GRPO vs GiGPO) mean something.

Two templates:
  - RULES_HEADER: shown once, at the start of every episode
  - STEP_TEMPLATE: shown every step, with recent history folded in
"""

RULES_HEADER = """You are an AI agent handling an employee international relocation case.

EMPLOYEE: {role}, dependents: {has_dependents}
DESTINATION: {countries}

DOMAIN RULES (apply to every case):
- Germany requires visa, work_permit, and tax_id registration.
- Singapore requires an Employment Pass, PDPA consent, and shadow payroll.
- UAE has NO income tax -- calling set_tax_id for UAE is a rule violation.
- Legal approval requires ALL documents to be verified first.
- Finance approval requires Legal approval first.
- If two countries are involved with conflicting rules, resolve_conflict
  must be called before Finance approval.
- A mid-episode regulatory change may occur. If it does, you must call
  acknowledge_regulatory_change before continuing, and switch to whatever
  replacement document/process the change specifies.

RESPONSE FORMAT (every turn, exactly this shape):
<think>
your reasoning about what to do next, briefly
</think>
<action>{{"action_type": "...", "target": "..."}}</action>

Use "" as the target for actions that don't need one (e.g. approve_hr).
Only one action per turn. You will see the result before your next turn.
"""

STEP_TEMPLATE = """{rules_header}
RECENT HISTORY (last {history_length} steps):
{action_history}

CURRENT STEP: {current_step}
{current_observation}

What do you do next? Respond in the exact format above.
"""


def build_initial_prompt(task_state: dict) -> str:
    """The very first prompt of an episode: rules + task description, no history yet."""
    employee = task_state["employee"]
    return RULES_HEADER.format(
        role=employee.get("role", "unknown"),
        has_dependents=employee.get("has_dependents", False),
        countries=", ".join(task_state.get("countries", [])),
    )


def build_step_prompt(
    task_state: dict,
    current_observation_text: str,
    action_history: list[dict],
    history_length: int,
    current_step: int,
) -> str:
    """Every subsequent prompt: rules header + recent history + current state."""
    rules_header = build_initial_prompt(task_state)

    recent = action_history[-history_length:]
    if recent:
        history_lines = []
        for i, record in enumerate(recent):
            step_num = current_step - len(recent) + i
            history_lines.append(
                f"Step {step_num}: {record['action']}\nResult: {record['result']}"
            )
        history_text = "\n\n".join(history_lines)
    else:
        history_text = "(none yet)"

    return STEP_TEMPLATE.format(
        rules_header=rules_header,
        history_length=history_length,
        action_history=history_text,
        current_step=current_step,
        current_observation=current_observation_text,
    )