import ipaddress, time, io, struct, os, uuid, threading, json

import torch
import numpy as np

from rich.text import Text
from termcolor import colored
from collections import deque


ACTIVATE_UI = True

# Global reference to the RichLog widget — set by the app on mount
_log_widget = None
_output_widget = None

_LOG_BUFFER_MAX = 10_000
_log_buffer: deque = deque(maxlen=_LOG_BUFFER_MAX)
_log_lock = threading.Lock()

_filter_predicate = lambda msg_type: True

 
def set_log_widget(widget):
    global _log_widget
    _log_widget = widget


def set_output_widget(widget):
    global _output_widget
    _output_widget = widget

 
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
    "budget":        "#efc81b",
    "greeting-ack":  "#00ff00",
    "knowledge-ack": "#00ff00",
    "info-ack":      "#00ff00",
    "stats-ack":     "#00ff00",
    "ir3de-ack":     "#00ff00",
}
 

def set_filter_predicate(fn):
    global _filter_predicate
    _filter_predicate = fn


def iter_log_buffer():
    """Thread-safe snapshot iteration for re-rendering."""
    with _log_lock:
        return list(_log_buffer)
    
 
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:

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


def log(message, node_id, msg_type=None, msg_id=None, right=False):

    current_time = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:03d}"

    if ACTIVATE_UI:

        msg_id_str = f" [{msg_id[:8]}]" if msg_id is not None else ""
        prefix = f"[NODE {node_id}] [{current_time}] "

        line = Text()
        line.append(prefix, style="dim white")

        if msg_type is not None:
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

        target = _output_widget if right else _log_widget

        if target is _log_widget and not _filter_predicate(msg_type):
            return                 # filter applies only to the left logs pane

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
    arrays = {}
    counter = 0

    def encode(x):
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
    
    if len(packet) < 4:
        raise ValueError("Invalid packet: too short")

    header_len = struct.unpack(">I", packet[:4])[0]
    header_start = 4
    header_end = 4 + header_len

    if len(packet) < header_end:
        raise ValueError("Invalid packet: incomplete header")

    header = json.loads(packet[header_start:header_end].decode("utf-8"))

    return header


def redirect_prints(func, *args, **kwargs):
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1); saved_stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        out = func(*args, **kwargs)
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        for fd in (devnull, saved_stdout, saved_stderr):
            os.close(fd)
    return out


def format_params(n: int) -> str:
    if n >= 1e9:  return f"{n / 1e9:.2f}B"
    if n >= 1e6:  return f"{n / 1e6:.2f}M"
    if n >= 1e3:  return f"{n / 1e3:.2f}K"
    return str(n)