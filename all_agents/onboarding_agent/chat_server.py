"""
Onboarding Agent Chat Server — interactive browser UI for testing the onboarding flow.

Proxies messages to the LangGraph onboarding agent running on localhost:2024.
When onboarding completes, automatically sends parsed data to WithCare's /onboarding/ingest.

Usage:
    python chat_server.py                # starts on http://localhost:8001
    python chat_server.py --port 8002
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

import uuid
from datetime import datetime, timedelta

import boto3
from boto3.dynamodb.conditions import Key
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
_logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://localhost:2024")
GRAPH_NAME = os.getenv("ONBOARDING_GRAPH_NAME", "onboarding_agent")
WITHCARE_AGENT_URL = os.getenv("WITHCARE_AGENT_URL", "http://localhost:8000")
# The browser-facing URL for the WithCare chat UI (may differ from internal API URL)
WITHCARE_CHAT_URL = os.getenv("WITHCARE_CHAT_URL", WITHCARE_AGENT_URL)

# ── DDB config for greeting message writes ──
CHAT_MESSAGES_TABLE = os.getenv("CHAT_MESSAGES_TABLE", "ChatMessages")
USER_CONVERSATION_TABLE = os.getenv("USER_CONVERSATION_TABLE", "WithCare_UserConversationTable")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")

# ── App setup ─────────────────────────────────────────────────────────

app = FastAPI(title="Onboarding Agent Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session state ─────────────────────────────────────────────────────

# {session_key: {thread_id, config, is_first_message, care_recipient, ...}}
_sessions: Dict[str, Dict[str, Any]] = {}

DEFAULT_CARE_RECIPIENT = {
    "address": "11650 National Boulevard, Los Angeles, California 90064, United States",
    "dateOfBirth": "1954-04-11",
    "dependentStatus": "Not a child/dependent",
    "firstName": "Yiming",
    "gender": "Male",
    "isSelf": False,
    "lastName": "Gong",
    "legalName": "Yiming Gong",
    "pronouns": "he/him/his",
    "relationship": "dad",
    "veteranStatus": "Veteran",
}


# ── LangGraph SDK helpers ─────────────────────────────────────────────


async def _create_thread() -> str:
    """Create a new LangGraph thread and return its thread_id."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{LANGGRAPH_URL}/threads", json={})
        resp.raise_for_status()
        return resp.json()["thread_id"]


async def _invoke_graph(thread_id: str, state: dict) -> dict:
    """Invoke the onboarding graph with the given state."""
    payload = {
        "input": state,
        "config": {"configurable": {"thread_id": thread_id}},
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{LANGGRAPH_URL}/runs/wait",
            json={
                "assistant_id": GRAPH_NAME,
                "input": state,
                "config": {"configurable": {"thread_id": thread_id}},
            },
        )
        resp.raise_for_status()
        return resp.json()


async def _get_state(thread_id: str) -> dict:
    """Get current state from the LangGraph thread."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{LANGGRAPH_URL}/threads/{thread_id}/state",
        )
        resp.raise_for_status()
        return resp.json()


async def _write_greeting_to_ddb(user_id: str, greeting_text: str) -> None:
    """Write the onboarding greeting to ChatMessages + WithCare_UserConversationTable."""
    if not user_id or not greeting_text:
        return

    try:
        ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
    except Exception as e:
        _logger.error(f"Failed to create DDB resource for greeting: {e}")
        return

    now_iso = datetime.utcnow().isoformat()
    today = now_iso[:10]
    msg_id = f"msg_greeting_{uuid.uuid4().hex[:12]}"

    # Look up user's latest conversation_id
    conv_id = None
    try:
        conv_table = ddb.Table(USER_CONVERSATION_TABLE)
        for day_offset in range(7):
            date_str = (datetime.utcnow() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            result = conv_table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(f"USER#{user_id}#DATE#{date_str}"),
                ScanIndexForward=False,
                Limit=1,
                ProjectionExpression="conversation_id",
            )
            if result.get("Items"):
                conv_id = result["Items"][0].get("conversation_id")
                break
    except Exception as e:
        _logger.warning(f"Failed to look up conversation_id for greeting: {e}")

    if not conv_id:
        conv_id = str(uuid.uuid4())
        _logger.info(f"No existing conversation found for greeting, created conv_id: {conv_id}")

    # Write to ChatMessages
    try:
        ddb.Table(CHAT_MESSAGES_TABLE).put_item(Item={
            "user_Id": user_id,
            "sort_key": f"{now_iso}#{msg_id}",
            "chat_Id": conv_id,
            "role": "assistant",
            "text": greeting_text,
            "message_Id": msg_id,
            "dateSent": now_iso,
        })
        _logger.info(f"Greeting written to ChatMessages for {user_id}")
    except Exception as e:
        _logger.error(f"ChatMessages greeting write failed: {e}")

    # Write to WithCare_UserConversationTable
    try:
        ddb.Table(USER_CONVERSATION_TABLE).put_item(Item={
            "pk": f"CONV#{conv_id}",
            "sk": f"MSG#{now_iso}#{msg_id}",
            "gsi1pk": f"USER#{user_id}#DATE#{today}",
            "gsi1sk": f"MSG#{now_iso}#{msg_id}",
            "entity": "message",
            "conversation_id": conv_id,
            "user_id": user_id,
            "role": "assistant",
            "content": greeting_text,
            "message_id": msg_id,
            "timestamp": now_iso,
            "metadata": {"source": "onboarding_greeting"},
        })
        _logger.info(f"Greeting written to UserConversationTable for {user_id}")
    except Exception as e:
        _logger.error(f"UserConversationTable greeting write failed: {e}")


async def _send_to_withcare(
    user_id: str,
    care_recipient: dict,
    tasks: list,
    assessment_score: Optional[int],
    assessment_answer: Optional[list],
) -> dict:
    """Send onboarding data to WithCare's /onboarding/ingest endpoint."""
    relationship = care_recipient.get("relationship", "")

    # Parse assessment answers from the Q&A message list
    assessment_answers = None
    if assessment_answer:
        mental_questions = [
            "concentration", "sleep", "isolation",
            "lost_interest", "anxiety", "depression",
        ]
        answers = []
        q_idx = 0
        for msg in assessment_answer:
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            role = msg.get("type", "") if isinstance(msg, dict) else ""
            if role == "human" and q_idx < len(mental_questions):
                answers.append({
                    "question": mental_questions[q_idx],
                    "answer": content.strip().lower(),
                })
                q_idx += 1
        if answers:
            assessment_answers = answers

    payload = {
        "user_id": user_id,
        "care_recipients": [
            {"relationship": relationship, "data": care_recipient}
        ] if care_recipient else [],
        "assessment_score": assessment_score,
        "assessment_answers": assessment_answers,
        "tasks": tasks if tasks else None,
    }

    _logger.info(f"Sending onboarding ingest to {WITHCARE_AGENT_URL}: {json.dumps(payload, indent=2)[:500]}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{WITHCARE_AGENT_URL}/onboarding/ingest",
                json=payload,
            )
            resp.raise_for_status()
            result = resp.json()
            _logger.info(f"WithCare ingest result: {result}")
            return result
    except Exception as e:
        _logger.error(f"WithCare ingest failed: {e}")
        return {"status": "error", "error": str(e)}


# ── Request / response models ────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    care_recipient: Optional[dict] = None
    veteran_status: Optional[str] = None


class ResetRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id
    user_id = req.user_id or "test-user"
    care_recipient = req.care_recipient or DEFAULT_CARE_RECIPIENT
    veteran_status = req.veteran_status or care_recipient.get("veteranStatus", "Not a veteran")

    try:
        if not session_id or session_id not in _sessions:
            # ── First message: create thread and invoke with initial state ──
            thread_id = await _create_thread()
            session_id = thread_id  # use thread_id as session_id

            initial_greeting = (
                "Hello! Welcome to WithCare. I'm your AI Care Navigator "
                "trained by licensed clinicians to support you 24/7. How are you today?"
            )

            initial_state = {
                "node": "root",
                "tasks": [],
                "chat_history": [],
                "veteranStatus": veteran_status,
                "real_chat_history": [
                    {"type": "ai", "content": initial_greeting},
                    {"type": "human", "content": req.message},
                ],
                "last_step": "start",
                "current_tree": "IntroAssessmentTree",
                "care_recipient": care_recipient,
                "direct_record_answer": False,
                "user_id": user_id,
            }

            _logger.info(f"Creating new thread {thread_id} for user {user_id}")
            result = await _invoke_graph(thread_id, initial_state)

            _sessions[session_id] = {
                "thread_id": thread_id,
                "care_recipient": care_recipient,
                "veteran_status": veteran_status,
                "user_id": user_id,
            }

        else:
            # ── Subsequent message: get state, append message, re-invoke ──
            session = _sessions[session_id]
            thread_id = session["thread_id"]

            state_resp = await _get_state(thread_id)
            current_values = state_resp.get("values", {})

            # Append user message to real_chat_history
            real_chat_history = current_values.get("real_chat_history", [])
            real_chat_history.append({"type": "human", "content": req.message})

            updated_state = dict(current_values)
            updated_state["real_chat_history"] = real_chat_history

            _logger.info(f"Continuing thread {thread_id}, tree={current_values.get('current_tree')}")
            result = await _invoke_graph(thread_id, updated_state)

        # ── Extract response fields ──
        question = result.get("question", "")
        completed = result.get("completed_whole_process", False)
        tasks = result.get("tasks", [])
        current_tree = result.get("current_tree", "")
        node = result.get("node", "")
        assessment_score = result.get("assessment_score", 0)
        assessment_answer = result.get("assessment_answer", [])
        route = result.get("route", "")

        debug = {
            "thread_id": thread_id if 'thread_id' in dir() else session_id,
            "current_tree": current_tree,
            "node": node,
            "route": route,
            "completed_whole_process": completed,
            "tasks": tasks,
            "assessment_score": assessment_score,
        }

        # ── If onboarding complete, send to WithCare + write greeting ──
        ingest_result = None
        if completed:
            session_data = _sessions.get(session_id, {})
            ingest_result = await _send_to_withcare(
                user_id=user_id,
                care_recipient=session_data.get("care_recipient", care_recipient),
                tasks=tasks,
                assessment_score=assessment_score,
                assessment_answer=assessment_answer,
            )
            debug["withcare_ingest"] = ingest_result

            # Write greeting message to DDB (ChatMessages + UserConversationTable)
            greeting_message = result.get("greeting_message")
            if greeting_message:
                try:
                    await _write_greeting_to_ddb(user_id, greeting_message)
                    debug["greeting_written"] = True
                except Exception as e:
                    _logger.error(f"Greeting DDB write failed: {e}")
                    debug["greeting_written"] = False

        return {
            "reply": question,
            "session_id": session_id,
            "completed": completed,
            "withcare_chat_url": WITHCARE_CHAT_URL,
            "debug": debug,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        _logger.error(f"Chat error: {exc}\n{tb}")
        return {
            "reply": f"[Server error] {exc}",
            "session_id": session_id or "",
            "completed": False,
            "debug": {"error": str(exc), "traceback": tb},
        }


@app.post("/reset")
async def reset(req: ResetRequest):
    _sessions.pop(req.session_id, None)
    return {"status": "ok", "session_id": req.session_id}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sessions": len(_sessions),
        "langgraph_url": LANGGRAPH_URL,
        "withcare_api_url": WITHCARE_AGENT_URL,
        "withcare_chat_url": WITHCARE_CHAT_URL,
    }


# ── Static frontend ──────────────────────────────────────────────────

_UI_DIR = Path(__file__).parent / "chat_ui"


@app.get("/")
async def index():
    return FileResponse(_UI_DIR / "index.html")


if _UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Onboarding Agent Chat Server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting Onboarding Agent chat server on http://localhost:{args.port}")
    print(f"  LangGraph API: {LANGGRAPH_URL}")
    print(f"  WithCare API:  {WITHCARE_AGENT_URL}")
    uvicorn.run(app, host=args.host, port=args.port)
