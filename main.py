"""
main.py
=======
FastAPI application for meridian.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import WorkforceEnv
from env.models import Action

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action_type: str = "request_document"
    target: str = ""
    session_id: Optional[str] = None

class GradeRequest(BaseModel):
    session_id: Optional[str] = None
    task_name: Optional[str] = None

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

_sessions: dict[str, WorkforceEnv] = {}
_last_session: str | None = None
MAX_SESSIONS = 50

VALID_TASKS = ("easy", "medium", "hard", "crisis")

def _get_session(session_id: str | None) -> WorkforceEnv:
    sid = session_id or _last_session
    if not sid or sid not in _sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /reset first.",
        )
    return _sessions[sid]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _last_session
    try:
        env = WorkforceEnv()
        env.reset("easy")
        sid = str(uuid.uuid4())
        _sessions[sid] = env
        _last_session = sid
    except Exception:
        pass
    yield

app = FastAPI(
    title="meridian",
    description="Global Mobility & Compliance Orchestrator — OpenEnv RL Environment",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "environment": "meridian",
        "version": "2.0.0",
    }

@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    return {
        "tasks": list(VALID_TASKS),
        "descriptions": {
            "easy":   "India → Germany | Engineer | HR + 4 docs + tax_id + payroll",
            "medium": "India → Singapore | Manager | HR + Legal + 3 docs + PDPA + shadow payroll",
            "hard":   "India → Germany + UAE | Director | All depts + conflict resolution",
            "crisis": "India → Germany | Manager | Mid-episode regulatory disruption — Blue Card suspended, switch to ICT Permit",
        },
    }

@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None) -> dict[str, Any]:
    """
    Start a new episode.
    Body is fully optional — defaults to task_name="easy".
    POST /reset with no body is valid (used by OpenEnv health checks).
    """
    global _last_session

    if request is None:
        request = ResetRequest()

    if request.task_name not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task_name}'. Valid: {list(VALID_TASKS)}",
        )

    if len(_sessions) >= MAX_SESSIONS:
        oldest = next(iter(_sessions))
        _sessions.pop(oldest)

    env = WorkforceEnv()
    try:
        observation = env.reset(task_name=request.task_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    sid = request.session_id or env._session_id
    _sessions[sid] = env
    _last_session = sid

    return {
        "session_id": sid,
        "observation": observation.model_dump(),
    }

@app.post("/step")
async def step(request: StepRequest) -> dict[str, Any]:
    env = _get_session(request.session_id)
    try:
        action = Action(
            action_type=request.action_type,
            target=request.target,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action: {exc}")

    try:
        result = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if result.done:
        sid = request.session_id or _last_session
        _sessions.pop(sid, None)

    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }

@app.get("/state")
async def state(session_id: Optional[str] = None) -> dict[str, Any]:
    env = _get_session(session_id)
    try:
        return env.state().model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/grade")
async def grade_endpoint(request: GradeRequest) -> dict[str, Any]:
    env = _get_session(request.session_id)
    try:
        from graders.graders import grade
        current_state = env.state()
        task_name = request.task_name or current_state.task_name or "easy"
        state_dict = _flatten_for_grader(current_state)
        score = grade(task_name, state_dict)
        return {
            "task": task_name,
            "score": round(score, 4),
            "status": current_state.status,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/sessions", include_in_schema=False)
async def list_sessions() -> dict[str, Any]:
    return {"count": len(_sessions), "ids": list(_sessions.keys())}

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _flatten_for_grader(ws: Any) -> dict[str, Any]:
    d = ws.model_dump()
    docs = {}
    for name, doc in d.get("documents", {}).items():
        docs[name] = doc if isinstance(doc, dict) else {
            "status": doc.status, "is_valid": doc.is_valid
        }
    d["documents"] = docs

    depts = d.get("departments", {})
    if not isinstance(depts, dict):
        depts = {"HR": depts.HR, "Legal": depts.Legal, "Finance": depts.Finance}
    d["departments"] = depts

    comp = d.get("compliance", {})
    if not isinstance(comp, dict):
        comp = {
            "tax_id": comp.tax_id,
            "payroll": comp.payroll,
            "pdpa": comp.pdpa,
            "shadow_payroll": comp.shadow_payroll,
        }
    d["compliance"] = comp

    conflicts = d.get("conflicts", [])
    d["conflicts"] = [
        c if isinstance(c, dict) else {
            "countries": c.countries,
            "rule": c.rule,
            "resolved": c.resolved,
        }
        for c in conflicts
    ]
    return d

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()