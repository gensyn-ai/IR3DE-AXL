"""On-disk chat storage and in-memory manipulation helpers.

A chat is a plain dict matching this shape:

    {
      "chat_id":     str,            # e.g. "1719673205-3f4a9b21"
      "title":       str | None,     # set after first answer (round 1)
      "created_at":  str,            # ISO 8601 UTC
      "updated_at":  str,            # ISO 8601 UTC, bumped on every mutation
      "summary":     str | None,     # populated once summarisation kicks in
      "messages": [
        {
          "role":   "user" | "agent",
          "text":   str,
          "ts":     str,                              # ISO 8601 UTC
          "status": "ok" | "failed",                  # 'failed' only on user role
          "expert": {                                 # agent role only
            "peer_pk":   str,
            "model_idx": int,
            "tag":       str | None,
          } | None,
        },
        ...
      ]
    }

The on-wire history that gets sent to an expert is a *filtered* projection of
`messages` — see `history_for_expert`. Persisted state intentionally keeps
failed turns (so the UI can render the 'failed' marker) but they're excluded
from what the expert sees on subsequent rounds.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


CHATS_DIR = Path("chats")


# ───────────────────────── helpers ─────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_chat_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def chat_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


# ───────────────────────── constructors ─────────────────────────

def new_chat(peer_name: str | None = None) -> dict:
    """Build a fresh empty chat. Caller is responsible for persisting it."""
    now = _now_iso()
    return {
        "chat_id":    _new_chat_id(),
        "title":      None,
        "created_at": now,
        "updated_at": now,
        "summary":    None,
        "peer_name":  peer_name,
        "messages":   [],
    }


# ───────────────────────── mutators ─────────────────────────

def add_user_message(chat: dict, text: str) -> dict:
    """Append a user message in 'ok' state. Returns the appended message dict."""
    msg = {
        "role":   "user",
        "text":   text,
        "ts":     _now_iso(),
        "status": "ok",
    }
    chat["messages"].append(msg)
    chat["updated_at"] = _now_iso()
    return msg


def add_agent_message(
    chat: dict,
    text: str,
    peer_pk: str,
    model_idx: int,
    tag: str | None = None,
) -> dict:
    """Append an agent message tied to the expert that produced it."""
    msg = {
        "role":   "agent",
        "text":   text,
        "ts":     _now_iso(),
        "status": "ok",
        "expert": {
            "peer_pk":   peer_pk,
            "model_idx": model_idx,
            "tag":       tag,
        },
    }
    chat["messages"].append(msg)
    chat["updated_at"] = _now_iso()
    return msg


def mark_last_user_failed(chat: dict) -> None:
    """Flip the most recent user message to 'failed'. No-op if last message
    isn't a user message or doesn't exist."""
    if not chat["messages"]:
        return
    last = chat["messages"][-1]
    if last["role"] != "user":
        return
    last["status"] = "failed"
    chat["updated_at"] = _now_iso()


def set_title(chat: dict, title: str | None) -> None:
    chat["title"] = title
    chat["updated_at"] = _now_iso()


def set_summary(chat: dict, summary: str | None) -> None:
    chat["summary"] = summary
    chat["updated_at"] = _now_iso()


def trim_oldest_pair(chat: dict) -> tuple[dict, dict] | None:
    """Pop the oldest (user, agent) couple from the front of messages, skipping
    leading failed turns. Returns the removed pair, or None if there isn't
    a complete couple at the head. Callers use this when building the
    summarisation input — see step 3 of the implementation plan."""
    msgs = chat["messages"]
    # Find first 'ok' user followed by an 'ok' agent
    for i in range(len(msgs) - 1):
        if (msgs[i]["role"] == "user" and msgs[i].get("status", "ok") == "ok"
                and msgs[i+1]["role"] == "agent"):
            user_msg = msgs.pop(i)
            agent_msg = msgs.pop(i)        # now at the same index after first pop
            chat["updated_at"] = _now_iso()
            return user_msg, agent_msg
    return None


# ───────────────────────── projections ─────────────────────────

def history_for_expert(chat: dict) -> list[dict]:
    """The slice of messages that should be sent to the expert on the next turn:
    all 'ok' messages, in order. Failed user turns are excluded so the agent
    reasons about a clean transcript with no gaps."""
    return [m for m in chat["messages"] if m.get("status", "ok") == "ok"]


def history_char_count(chat: dict) -> int:
    """Sum of character lengths of the to-be-sent history (excluding system
    prompt and any prepended summary). Use as the budget input for trimming."""
    return sum(len(m["text"]) for m in history_for_expert(chat))


# ───────────────────────── persistence ─────────────────────────

def save_chat(chat: dict) -> None:
    """Atomically write a chat to disk. Creates chats/ on first call."""
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    path = chat_path(chat["chat_id"])
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(chat, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_chat(chat_id: str) -> dict:
    with open(chat_path(chat_id), "r", encoding="utf-8") as f:
        return json.load(f)


def delete_chat(chat_id: str) -> bool:
    path = chat_path(chat_id)
    if path.exists():
        path.unlink()
        return True
    return False


def list_chats() -> list[dict]:
    """Lightweight directory scan. Returns metadata only (chat_id, title,
    created_at, updated_at, num_messages), sorted by updated_at descending.
    Malformed files are skipped silently. Use load_chat(chat_id) to get the
    full message list."""
    if not CHATS_DIR.exists():
        return []
    out = []
    for path in CHATS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        out.append({
            "chat_id":      data.get("chat_id", path.stem),
            "title":        data.get("title"),
            "created_at":   data.get("created_at", ""),
            "updated_at":   data.get("updated_at", ""),
            "num_messages": len(data.get("messages", []) or []),
        })
    out.sort(key=lambda c: c["updated_at"], reverse=True)
    return out