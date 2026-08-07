"""
On-disk chat storage and in-memory manipulation helpers.

A chat is a plain dict matching this shape:

    {
      "chat_id":             str,
      "title":               str | None,
      "created_at":          str,
      "updated_at":          str,
      "summary":             str | None,
      "history_start_index": int,       # messages[:history_start_index] are covered by 'summary'
      "peer_id":             int | str | None,  # owning peer's stable id; used to decide which chats a peer loads on startup
      "peer_name":           str | None,         # display-only; not unique, not used for ownership
      "open_in_ui":          bool,
      "messages":            [
      
        {
          "role":   "user" | "router" | "agent",
          "text":   str,
          "ts":     str,                              # ISO 8601 UTC
          "status": "ok" | "failed",                  # 'failed' only on user role
          "expert": {                                 # agent role only
            "peer_pk":   str,
            "model_idx": int,
            "tag":       str | None,
            "peer_name": str | None,                  # display label in chat transcript
          } | None,
        },
        ...
      ],
    }

The on-wire history that gets sent to an expert is a *filtered* projection of
`messages` starting at `history_start_index` — see `history_for_expert`. The UI
still renders the full `messages` list; the index is purely a pointer for the
LLM-facing slice, so nothing ever disappears from the persisted transcript.
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
    """Current UTC time as an ISO 8601 string, for created_at/updated_at/ts fields."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_chat_id() -> str:
    """Time-prefixed unique id for a new chat (sortable by creation order)."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def chat_path(chat_id: str) -> Path:
    """On-disk JSON path for a given chat_id, under CHATS_DIR."""
    return CHATS_DIR / f"{chat_id}.json"


# ───────────────────────── constructors ─────────────────────────


def new_chat(peer_id: int | str | None = None, peer_name: str | None = None) -> dict:
    """Build a fresh empty chat. Caller is responsible for persisting it."""
    now = _now_iso()
    return {
        "chat_id":             _new_chat_id(),
        "title":               None,
        "created_at":          now,
        "updated_at":          now,
        "summary":             None,
        "history_start_index": 0,
        "peer_id":             peer_id,
        "peer_name":           peer_name,
        "open_in_ui":          False,
        "messages":            [],
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


def add_router_message(chat: dict, text: str) -> dict:
    """Append a display-only router selection line (not sent to experts)."""
    msg = {
        "role":   "router",
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
    peer_name: str | None = None,
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
            "peer_name": peer_name,
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
    """Set (or clear) the chat's title. Called from peer.py after a title is
    generated, and from the UI's rename dialog (IR3DEApp._on_chat_renamed)."""
    chat["title"] = title
    chat["updated_at"] = _now_iso()


def set_summary(chat: dict, summary: str | None) -> None:
    """Replace the chat's rolling summary, set by peer.py's _handle_summary
    once older turns are summarized past the character budget."""
    chat["summary"] = summary
    chat["updated_at"] = _now_iso()


def trim_oldest_pair(chat: dict) -> tuple[dict, dict] | None:
    """Advance the summarization boundary past the next (user, agent) couple
    in the live history. Returns the pair (still present in chat['messages'])
    so the caller can feed it into the summarizer, or None if there isn't a
    complete couple at the head of the live slice yet.

    Unlike the old behavior, this does NOT mutate chat['messages']. The
    couple stays in the persisted transcript so the UI can keep rendering
    the full history; only the LLM-facing view (see history_for_expert)
    shrinks."""
    msgs = chat["messages"]
    start = chat.get("history_start_index", 0)
    i = start
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i].get("status", "ok") == "ok":
            j = i + 1
            while j < len(msgs) and msgs[j]["role"] == "router":
                j += 1
            if j < len(msgs) and msgs[j]["role"] == "agent":
                chat["history_start_index"] = j + 1
                chat["updated_at"] = _now_iso()
                return msgs[i], msgs[j]
        i += 1
    return None


# ───────────────────────── projections ─────────────────────────

def history_for_expert(chat: dict) -> list[dict]:
    """The slice of messages sent to the expert on the next turn: all 'ok'
    user/agent messages from history_start_index onward. Earlier turns are
    folded into chat['summary']; failed user turns and display-only router
    lines are excluded so the transcript has no gaps."""
    msgs = chat["messages"]
    start = chat.get("history_start_index", 0)
    return [
        m for m in msgs[start:]
        if m.get("status", "ok") == "ok" and m["role"] in ("user", "agent")
    ]


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
    """Read one chat's full record from disk. Raises if it doesn't exist."""
    with open(chat_path(chat_id), "r", encoding="utf-8") as f:
        return json.load(f)


def delete_chat(chat_id: str) -> bool:
    """Unlink a chat's file, if present. Returns whether it existed. Called
    by Peer.delete_chat and IR3DEApp._close_chat_tab/_delete_chat."""
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

def set_ui_open(chat: dict, open_: bool) -> None:
    """Mark whether the chat is currently mounted as a tab."""
    chat["open_in_ui"] = bool(open_)