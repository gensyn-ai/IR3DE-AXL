import argparse
import contextlib
import io
import ipaddress
import json
import os
import signal
import statistics
import struct
import sys
import threading
import time
import uuid
import warnings
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from rich.text import Text
from safetensors import safe_open
from termcolor import colored
from textual.widgets import RichLog
from tqdm import tqdm as _tqdm
from transformers import AutoModelForCausalLM

import chats

_tqdm.set_lock(threading.RLock())

# Model loading/generation run on background threads, but heavy CPU-bound
# PyTorch/HF work there can still starve the main thread's Textual event loop
# under the GIL — it only gets released between bytecode "ticks". Shortening
# the switch interval (default 5ms) makes the interpreter check for a thread
# switch far more often, so the UI thread gets scheduled promptly even while
# a background thread is deep in a long CPU-bound call.
sys.setswitchinterval(0.001)

# Cap PyTorch's intra-op thread pool so model loading/generation don't
# saturate every CPU core — leaves headroom for the main/UI thread to run.
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

IR3DE_STATS_REPO_ID = "Erosinho/IR3DE-stats"

ACTIVATE_UI = True
MAX_TITLE_CHARS = 60

_log_widget = None
_output_widgets: dict[str, "RichLog"] = {}

_LOG_BUFFER_MAX = 10_000
_log_buffer: deque = deque(maxlen=_LOG_BUFFER_MAX)
_log_lock = threading.Lock()

_filter_predicate = lambda msg_type: True

 
def set_log_widget(widget):
    """Register the Logs tab's RichLog as log()'s target for left-side entries.
    Called once from IR3DEApp.on_mount."""
    global _log_widget
    _log_widget = widget


def set_output_widget(chat_id: str, widget) -> None:
    """Register a chat's RichLog as log()'s target for that chat's right-side
    entries. Called from IR3DEApp._mount_chat_tab when a chat tab is opened."""
    _output_widgets[chat_id] = widget


def unset_output_widget(chat_id: str) -> None:
    """Drop a chat's output widget registration when its tab closes.
    Called from IR3DEApp._close_chat_tab/_delete_chat."""
    _output_widgets.pop(chat_id, None)


def get_output_widget(chat_id: str):
    """The RichLog currently registered for a chat_id, or None if its tab isn't open."""
    return _output_widgets.get(chat_id)

 
MSG_TYPE_COLORS = {
    "text":          "#e2aaf8",
    "warning":       "#ff0000",
    "newnode":       "#ff00ff",
    "greeting":      "#00ffff",
    "knowledge":     "#ffff00",
    "info":          "#0088ff",
    "stats-req":     "#ff8800",
    "stats":         "#ff8800",
    "ir3de":         "#8800ff",
    "summary-req":   "#5f0000",
    "summary":       "#5f0000",
    "greeting-ack":  "#00ff00",
    "knowledge-ack": "#00ff00",
    "info-ack":      "#00ff00",
    "stats-ack":     "#00ff00",
    "ir3de-ack":     "#00ff00",
    "summary-ack":     "#00ff00",
}
 

_TAG_SYMBOL_RULES: list[tuple[tuple[str, ...], str]] = [
    (("cod", "program", "dev"),                        "</>"),
    (("math", "arithmetic", "algebra", "calculus"),    "Σ"),
    (("medic", "health", "medicine", "doctor"),        "✚"),
    (("chat", "general", "conversation", "dialog"),    "⊟"),
    (("history", "historical"),                        "◷"),
    (("philosoph",),                                   "◐"),
    (("physic",),                                      "⚛"),
    (("biolog", "bio"),                                "❀"),
    (("chem",),                                        "⚗"),
    (("art", "draw", "design"),                        "✎"),
    (("music", "audio", "sound"),                      "♪"),
    (("lingu", "language", "writing", "literature"),   "❡"),
    (("science",),                                     "⚛"),
    (("geo", "earth"),                                 "◯"),
    (("law", "legal"),                                 "§"),
    (("finance", "money", "econ"),                     "$"),
]


AGENT_SYSTEM_PROMPT = (
    "You are one agent in a multi-agent chat system.\n\n"
    "You will receive the full conversation history so far.\n"
    "The history contains messages from the user and from previous agents.\n"
    "Use the history as context.\n"
    "Answer the latest user message.\n\n"
    "Important:\n"
    "- Previous agent messages are context, not guaranteed truth.\n"
    "- The user's messages define the actual request.\n"
    "- Do not assume hidden information outside the transcript."
)


SUMMARY_SYSTEM_PROMPT = (
    "You are a summarizer. Produce a concise factual summary of the "
    "conversation excerpt below in at most {max_chars} characters. "
    "Capture key topics, decisions, named entities, and any context "
    "later turns might need. Do not invent details. Output ONLY the "
    "summary text — no preamble, no apologies, no formatting."
)


TITLE_SYSTEM_PROMPT = (
    "Read the exchange below and produce a concise chat title (at most "
    "6 words, no quotes, no trailing punctuation) that captures the topic. "
    "Output ONLY the title text — no preamble, no explanation."
)

def format_prompt_for_expert(chat: dict) -> str:
    """Build the text prompt sent to the expert from the chat's sendable
    history. Plain 'User:' / 'Agent:' role markers — model-agnostic. The
    trailing 'Agent: ' primes the model to continue.

    Note: we deliberately do *not* use tokenizer chat templates here because
    different experts in the network use different tokenizers. Plain text
    works on all of them; quality is marginally below template-formatted
    chat but uniform across the network.
    """
    parts = [AGENT_SYSTEM_PROMPT, ""]
    if chat.get("summary"):
        parts.append("Summary of the chat: " + chat["summary"])
        parts.append("")
    for m in chats.history_for_expert(chat):
        prefix = "User" if m["role"] == "user" else "Agent"
        parts.append(f"{prefix}: {m['text']}")
    parts.append("Agent: ")
    return "\n".join(parts)


def set_filter_predicate(fn):
    """Register the msg_type -> bool predicate log() consults for left-side
    entries. Set to IR3DEApp._should_show_log in IR3DEApp.on_mount."""
    global _filter_predicate
    _filter_predicate = fn


def iter_log_buffer():
    """Thread-safe snapshot iteration for re-rendering. Used by
    IR3DEApp._rerender_logs when the Logs tab's filters change."""
    with _log_lock:
        return list(_log_buffer)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse a '#rgb' or '#rrggbb' string into an (r, g, b) tuple."""
    hex_color = hex_color.lstrip("#")

    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)

    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")

    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def log(message, node_id, msg_type=None, msg_id=None, right=False, chat_id=None):
    """Central logging entry point, called throughout peer.py, run.py, and
    the UI. Two destinations depending on `right`: False (default) writes to
    the Logs tab via the widget set by set_log_widget, filtered by
    set_filter_predicate; True writes to a specific chat's transcript via
    the widget registered for `chat_id` in set_output_widget. Every entry is
    also kept in _log_buffer so the Logs tab can be re-filtered without
    losing history."""

    current_time = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:03d}"

    if ACTIVATE_UI:

        msg_id_str = f" [{msg_id[:8]}]" if msg_id is not None else ""
        node_prefix = f"[NODE {node_id}] " if node_id != 'USER' else "[USER] "
        time_prefix = f"[{current_time}] "

        line = Text()
        if right:
            line.append(node_prefix, style="bold dim white")
        else:
            line.append(node_prefix, style=f"dim white")
        line.append(time_prefix, style="dim white")

        if msg_type is not None:
            if not (msg_type == 'text' and right):
                color = MSG_TYPE_COLORS.get(msg_type, "#ffffff")
                line.append(f"[{msg_type.upper()}]", style=f"bold {color}")
                line.append(msg_id_str, style="#ffffff")
            line.append(f" {message}", style="#ffffff")
        else:
            line.append(msg_id_str, style="#ffffff")
            line.append(message, style="#ffffff")

        entry = (time.time(), node_id, msg_type, line)
        with _log_lock:
            _log_buffer.append(entry)

        if right:
            target = _output_widgets.get(chat_id) if chat_id else None
        else:
            if not _filter_predicate(msg_type):
                return
            target = _log_widget

        if target is not None:
            target.write(line)
        else:
            print(line.plain)
    else:
        msg_id_str = f" [{msg_id[:8]}]" if msg_id is not None else ""
        if msg_type is None:
            print(f"[NODE {node_id}] [{current_time}]{msg_id_str} {message}")
        else:
            color = MSG_TYPE_COLORS.get(msg_type, "white")
            print(f"[NODE {node_id}] [{current_time}] {colored(f'[{msg_type.upper()}]', hex_to_rgb(color))}{msg_id_str} {message}")
    

def ipv6_from_pubkey(pubkey_hex: str) -> str:
    """Yggdrasil 0.5 IPv6 derivation. Works with either the full 64-char
    pubkey or AXL's truncated X-From-Peer-Id — they yield the same address."""
    key = bytes.fromhex(pubkey_hex)
    if len(key) != 32:
        raise ValueError(f"expected 32 bytes, got {len(key)}")

    # 1. Invert the pubkey bit by bit
    inverted = bytes(b ^ 0xFF for b in key)

    # 2. Convert to a bit string for easy slicing
    bits = "".join(f"{b:08b}" for b in inverted)

    # 3. Count leading 1s (= leading 0s in the original pubkey)
    K = 0
    while K < len(bits) and bits[K] == "1":
        K += 1

    # 4. Drop those K leading 1s AND the terminator 0 that follows them
    remaining = bits[K + 1 :]

    # 5. Take the next 112 bits — that's the 14 bytes of address content
    content_bits = remaining[:112]
    content = bytes(int(content_bits[i : i + 8], 2) for i in range(0, 112, 8))

    # 6. Address = 0x02 prefix byte + K byte + 14 content bytes = 16 bytes total
    addr = b"\x02" + bytes([K]) + content
    return str(ipaddress.IPv6Address(addr))


MAX_CHUNK_SIZE = 1024 * 1024 * 15  # 15 MiB to be safe, since AXL has a 16 MiB limit including headers
ARRAY_PREFIX = "__array_"


def serialize_safe(obj, orig_msg_id, msg_id, msg_type, pk_from, pk_to):
    """Pack an object (tensors/ndarrays included) into one or more chunked,
    length-prefixed byte packets under AXL's 16 MiB message limit. Used by
    Peer.send(..., large=True) to transmit IR3DE stats between peers; each
    chunk is sent separately and reassembled by deserialize_safe on the
    receiving end. Returns (packets, chunk_msg_ids)."""
    arrays = {}
    counter = 0

    def encode(x):
        """Recursively replace tensors/ndarrays with name references into
        `arrays`, leaving everything else (dicts, lists, plain values) as-is."""
        nonlocal counter

        if isinstance(x, torch.Tensor):
            name = f"{ARRAY_PREFIX}{counter}"
            counter += 1
            arrays[name] = x.detach().cpu().numpy()
            return {"__type__": "tensor", "name": name}

        if isinstance(x, np.ndarray):
            name = f"{ARRAY_PREFIX}{counter}"
            counter += 1
            arrays[name] = x
            return {"__type__": "ndarray", "name": name}

        if isinstance(x, dict):
            return {k: encode(v) for k, v in x.items()}

        if isinstance(x, list):
            return [encode(v) for v in x]

        return x

    structure = encode(obj)
    metadata_bytes = json.dumps(structure).encode("utf-8")
    arrays["__metadata_json__"] = np.frombuffer(metadata_bytes, dtype=np.uint8)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    data = buffer.getvalue()

    raw_chunks = [
        data[i:i + MAX_CHUNK_SIZE]
        for i in range(0, len(data), MAX_CHUNK_SIZE)
    ]

    num_chunks = len(raw_chunks)
    result = []
    chunks_msg_ids = []

    for chunk_idx, chunk_data in enumerate(raw_chunks):
        chunk_msg_id = str(uuid.uuid4())
        chunks_msg_ids.append(chunk_msg_id)
        header = {
            "orig_msg_id": orig_msg_id,
            "msg_id": msg_id,
            "chunk_msg_id": chunk_msg_id,
            "msg_type": msg_type,
            "pk_from": pk_from,
            "pk_to": pk_to,
            "chunk_idx": chunk_idx,
            "num_chunks": num_chunks,
        }

        header_bytes = json.dumps(header).encode("utf-8")
        packet = (
            struct.pack(">I", len(header_bytes))
            + header_bytes
            + chunk_data
        )
        result.append(packet)

    return result, chunks_msg_ids


def deserialize_safe(chunks):
    """Reassemble the packets produced by serialize_safe back into the
    original object, after validating they all belong together and no chunk
    is missing/duplicated. Called by Peer.recv_loop once every chunk for a
    message has arrived."""
    if not chunks:
        raise ValueError("No chunks provided")

    parsed = []

    for packet in chunks:
        if len(packet) < 4:
            raise ValueError("Invalid packet: too short")

        header_len = struct.unpack(">I", packet[:4])[0]
        header_start = 4
        header_end = 4 + header_len

        if len(packet) < header_end:
            raise ValueError("Invalid packet: incomplete header")

        header = json.loads(packet[header_start:header_end].decode("utf-8"))
        chunk_data = packet[header_end:]

        parsed.append((header, chunk_data))

    first_header = parsed[0][0]

    orig_msg_id = first_header["orig_msg_id"]
    msg_id = first_header["msg_id"]
    msg_type = first_header["msg_type"]
    pk_from = first_header["pk_from"]
    pk_to = first_header["pk_to"]
    num_chunks = first_header["num_chunks"]

    if len(parsed) != num_chunks:
        raise ValueError(f"Expected {num_chunks} chunks, got {len(parsed)}")

    seen_chunk_msg_ids = set()
    seen_chunk_indices = set()

    for header, _ in parsed:
        if header["orig_msg_id"] != orig_msg_id:
            raise ValueError("Chunks have different orig_msg_id")

        if header["msg_id"] != msg_id:
            raise ValueError("Chunks have different msg_id")

        if header["msg_type"] != msg_type:
            raise ValueError("Chunks have different msg_type")

        if header["pk_from"] != pk_from:
            raise ValueError("Chunks have different pk_from")

        if header["pk_to"] != pk_to:
            raise ValueError("Chunks have different pk_to")

        if header["num_chunks"] != num_chunks:
            raise ValueError("Chunks disagree on num_chunks")

        chunk_msg_id = header["chunk_msg_id"]
        if chunk_msg_id in seen_chunk_msg_ids:
            raise ValueError(f"Duplicate chunk_msg_id: {chunk_msg_id}")
        seen_chunk_msg_ids.add(chunk_msg_id)

        chunk_idx = header["chunk_idx"]
        if chunk_idx in seen_chunk_indices:
            raise ValueError(f"Duplicate chunk_idx: {chunk_idx}")
        seen_chunk_indices.add(chunk_idx)

    parsed.sort(key=lambda x: x[0]["chunk_idx"])

    for expected_idx, (header, _) in enumerate(parsed):
        if header["chunk_idx"] != expected_idx:
            raise ValueError(f"Missing chunk {expected_idx}")

    data = b"".join(chunk_data for _, chunk_data in parsed)

    buffer = io.BytesIO(data)

    with np.load(buffer, allow_pickle=False) as loaded:
        metadata_bytes = bytes(loaded["__metadata_json__"])
        structure = json.loads(metadata_bytes.decode("utf-8"))

        def decode(x):
            """Inverse of encode: resolve name references back into
            tensors/ndarrays, recursing through dicts and lists."""
            if isinstance(x, dict) and x.get("__type__") == "tensor":
                return torch.from_numpy(loaded[x["name"]])

            if isinstance(x, dict) and x.get("__type__") == "ndarray":
                return loaded[x["name"]]

            if isinstance(x, dict):
                return {k: decode(v) for k, v in x.items()}

            if isinstance(x, list):
                return [decode(v) for v in x]

            return x

        obj = decode(structure)

    return obj


def deserialize_chunk_header(packet):
    """Parse just one packet's header (not its payload) — used by
    Peer.recv_loop to route/track an incoming chunk before the full message
    has been reassembled."""

    if len(packet) < 4:
        raise ValueError("Invalid packet: too short")

    header_len = struct.unpack(">I", packet[:4])[0]
    header_start = 4
    header_end = 4 + header_len

    if len(packet) < header_end:
        raise ValueError("Invalid packet: incomplete header")

    header = json.loads(packet[header_start:header_end].decode("utf-8"))

    return header


def format_params(n: int) -> str:
    """Format a parameter count for display, e.g. 1234567 -> '1.23M'. Used
    by the Statistics tab's models table and latency plot."""
    if n >= 1e9:  return f"{n / 1e9:.2f}B"
    if n >= 1e6:  return f"{n / 1e6:.2f}M"
    if n >= 1e3:  return f"{n / 1e3:.2f}K"
    return str(n)


def symbol_for_tag(tag: str) -> str:
    """Pick a display glyph for an expertise tag by keyword match, for the
    Control Panel's tag buttons and expertise sections."""
    t = tag.lower()
    for keywords, symbol in _TAG_SYMBOL_RULES:
        if any(kw in t for kw in keywords):
            return symbol
    return "◆"


def redirect_prints_safe(func, *args, **kwargs):
    """Run `func` with all noisy output suppressed, without freezing Textual.

    - fd 2 (stderr) is redirected to /dev/null at the OS level, catching
      C/C++ warnings from PyTorch/CUDA that bypass Python's `sys.stderr`.
    - fd 1 (stdout) is LEFT ALONE, because Textual writes ANSI escape
      sequences there and any redirect would freeze the UI.
    - Python-level `print()` to stdout is captured into a discard buffer.
    - Python `warnings.warn(...)` is silenced.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    finally:
        os.dup2(saved_stderr, 2)
        os.close(devnull)
        os.close(saved_stderr)


def ensure_stats_file(path: str, peer_id: str) -> str:
    """Return `path`, downloading it from the IR3DE-stats HF repo into its
    parent directory first if it isn't present locally yet."""
    if os.path.isfile(path):
        return path
    filename = os.path.basename(path)
    local_dir = os.path.dirname(path) or "."
    log(f"Stats file '{path}' not found locally; downloading '{filename}' from "
        f"'{IR3DE_STATS_REPO_ID}'...", peer_id, msg_type=None)
    return hf_hub_download(repo_id=IR3DE_STATS_REPO_ID, filename=filename, local_dir=local_dir)


_EMBED_WEIGHT_KEY_CANDIDATES = ("model.embed_tokens.weight", "transformer.wte.weight", "embed_tokens.weight")


def load_embedder_only(model_name: str) -> torch.nn.Embedding:
    """Load just a causal LM's input-embedding weight matrix, without
    building (or downloading the full weights of) the rest of the model —
    attention/MLP layers, lm_head, etc. IR3DE only ever needs embedding
    lookups from these models, so instantiating the whole multi-billion-
    parameter network just to keep one of its weight tensors is enormous,
    avoidable overhead. Falls back to the full-model load if the checkpoint
    doesn't use a recognized safetensors layout.
    """
    try:
        try:
            index_path = hf_hub_download(model_name, "model.safetensors.index.json")
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            embed_key = next(k for k in _EMBED_WEIGHT_KEY_CANDIDATES if k in weight_map)
            shard_path = hf_hub_download(model_name, weight_map[embed_key])
        except EntryNotFoundError:
            shard_path = hf_hub_download(model_name, "model.safetensors")
            with safe_open(shard_path, framework="pt") as f:
                embed_key = next(k for k in _EMBED_WEIGHT_KEY_CANDIDATES if k in f.keys())

        with safe_open(shard_path, framework="pt") as f:
            weight = f.get_tensor(embed_key).to(torch.float32)
        return torch.nn.Embedding.from_pretrained(weight)

    except Exception as e:
        log(f"Fast embedder-only load failed for '{model_name}' ({e}); "
            f"falling back to loading the full model.", "SYSTEM", msg_type="warning")
        model = redirect_prints_safe(AutoModelForCausalLM.from_pretrained, model_name)
        embedder = deepcopy(model.model.embed_tokens).to(torch.float32)
        del model
        return embedder


def format_mean_std(values, unit="s", scale=1.0, precision=2):
    """Render a list of samples as 'mean ± std unit'. Empty → '—', single → 'value unit'."""
    if not values:
        return "—"
    scaled = [v * scale for v in values]
    if len(scaled) == 1:
        return f"{scaled[0]:.{precision}f} {unit}"
    m = statistics.mean(scaled)
    s = statistics.stdev(scaled)
    return f"{m:.{precision}f} ± {s:.{precision}f} {unit}"

def render_chat_history_into(chat: dict, widget) -> None:
    """Replay a chat's persisted message list into a RichLog. Mirrors the
    visual format of log(..., right=True) without going through it (so we
    don't pollute _log_buffer with replays)."""
    for m in chat.get("messages", []):
        ts = m.get("ts", "")
        time_part = ts.split("T", 1)[1][:8] if "T" in ts else ""
        if m["role"] == "user":
            node_label = "[USER]"
        else:
            expert = m.get("expert") or {}
            pk = expert.get("peer_pk", "") or ""
            node_label = f"[NODE {pk[:8]}]" if pk else "[AGENT]"
        line = Text()
        line.append(f"{node_label} ", style="bold dim white")
        line.append(f"[{time_part}] ", style="dim white")
        if m.get("status") == "failed":
            line.append("(failed) ", style="bold red")
        line.append(m.get("text", ""), style="#ffffff")
        widget.write(line)


# ───────────────────────── run.py support ─────────────────────────
# The functions below back run.py's entry point (get_args, main, run_peer)
# but don't need to live there themselves — they're generic argument
# parsing and process/signal-cleanup helpers with no dependency on Peer or
# IR3DEApp beyond the objects passed to them.

def get_args():
    """Parse this peer's CLI flags: AXL node identity, gossip/timing
    intervals, IR3DE routing parameters, and which tokenizer/embedder pair
    (--tok-type) this peer is active for."""
    parser = argparse.ArgumentParser(description="Test one peer locally")
    parser.add_argument("--peer-id", type=int, required=False, default=None,
                        help="Which simulated local node to run, matching a "
                             "local_nodes/configNN.json/metadataNN.json pair "
                             "created by scripts/add_local_node.sh. Omit to "
                             "run the default local peer (local_nodes/config.json).")
    parser.add_argument("--discover-peers-interval", type=int, default=60,
                        help="Seconds between rounds of greeting a random "
                             "sample of known peers to discover the network.")
    parser.add_argument("--check-acks-interval", type=int, default=100,
                        help="Seconds between checking for expired "
                             "(un-acked) sent messages and pruning peers "
                             "that have gone silent.")
    parser.add_argument("--greeting-timeout", type=int, default=60,
                        help="Timeout in seconds for each greeting message "
                             "sent to a peer during discovery.")
    parser.add_argument("--knowledge-timeout", type=int, default=60,
                        help="Timeout in seconds for each known-peers "
                             "gossip message sent to a peer.")
    parser.add_argument("--ack-timeout", type=int, default=60,
                        help="Seconds to wait for an ACK before considering "
                             "a sent message expired.")
    parser.add_argument("--share-knowledge-interval", type=int, default=30,
                        help="Seconds between rounds of gossiping this "
                             "peer's known-peers list to a random sample of them.")
    parser.add_argument("--num-peers-to-greet", type=int, default=5,
                        help="How many random known peers to greet each "
                             "discovery round.")
    parser.add_argument("--num-peers-to-share", type=int, default=5,
                        help="How many random known peers to gossip "
                             "knowledge/models-info/stats-info to each round.")
    parser.add_argument("--share-info-interval", type=int, default=60,
                        help="Seconds between rounds of sharing this peer's "
                             "local model/stats metadata (not the stats "
                             "matrices themselves) with the network.")
    parser.add_argument("--share-stats-interval", type=int, default=60,
                        help="Seconds between rounds of asking the network "
                             "for IR3DE stats matrices matching this peer's "
                             "active tokenizer/embedder.")
    parser.add_argument("--stats-timeout", type=int, default=120,
                        help="Timeout in seconds for stats-sharing/"
                             "requesting messages (stats payloads can be "
                             "large and chunked).")
    parser.add_argument("--ir3de-lambda", type=float, default=0.01,
                        help="Ridge-regression regularization strength used "
                             "when building the IR3DE token router.")
    parser.add_argument("--ir3de-entropy-top-k", type=int, default=10,
                        help="Number of lowest-entropy tokens in the user's "
                             "message to vote on when picking its routed tag.")
    parser.add_argument("--max-answer-length", type=int, default=256,
                        help="Maximum number of new tokens an expert may "
                             "generate per answer.")
    parser.add_argument("--answer-timeout", type=int, default=300,
                        help="Seconds to wait for an expert's answer (local generation or a "
                             "remote peer's reply) before giving up.")
    parser.add_argument( "--num-characters-conversation-history", type=int, default=1000,
                        help="Maximum character budget for the prompt sent to experts (system "
                             "prompt + summary + history). When exceeded, oldest user/agent pairs "
                             "are summarized. Set to 0 to disable.")
    parser.add_argument('--tok-type', type=str, default='mistral', choices=['llama', 'mistral'], help='Type of tokenizer to use')
    return parser.parse_args()


def _install_node_signal_cleanup(peer):
    """Headless mode (ACTIVATE_UI=False, main thread): stop the AXL node on
    SIGTERM/SIGHUP so an externally-killed peer process doesn't leave an
    orphaned node. The text UI (TUI) has its own variant,
    _install_tui_node_cleanup; Ctrl+C is handled separately (headless:
    KeyboardInterrupt -> run_peer; TUI: action_quit).
    """
    def _handler(signum, frame):
        """Stop the node and exit with the conventional 128+signum code."""
        peer.stop_node()
        sys.stdout.flush()
        os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or signal unavailable on this platform.
            pass


def _install_tui_node_cleanup(app):
    """Text UI mode (main thread): stop the AXL node on SIGTERM/SIGHUP — the
    catchable exits Textual's Ctrl+C handling doesn't cover. Closing the
    terminal window or an SSH drop delivers SIGHUP; an external kill/pkill/
    IDE-stop delivers SIGTERM. Installed before app.run(). Stops the node
    directly, then asks Textual to exit so the terminal is restored; falls
    back to SystemExit if app.exit() can't run. The window before app.peer
    is set is covered by the Peer's atexit (registered when the node spawns).
    """
    def _handler(signum, frame):
        """Stop the node, then ask Textual to exit cleanly."""
        peer = getattr(app, "peer", None)
        if peer is not None:
            peer.stop_node()
        try:
            app.exit()
        except Exception:
            raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or signal unavailable on this platform.
            pass


def handle_input(app, args, user_input):
    """The TUI's input_handler: log the user's message into its chat, show
    the "waiting for answer" indicator, and dispatch it to the peer. Run on
    its own background thread per submission, spawned from
    IR3DEApp.on_submittable_text_area_submitted."""
    chat_id = getattr(app.peer, "active_chat_id", None)
    log(f"{user_input}", node_id="USER", msg_type=None, right=True, chat_id=chat_id)
    app.call_from_thread(app.start_waiting_for_answer, chat_id)
    try:
        app.peer.handle_user_input(user_input, timeout=args.answer_timeout)
    finally:
        app.call_from_thread(app.maybe_stop_waiting_for_answer, chat_id)