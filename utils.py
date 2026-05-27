import ipaddress
import time

from rich.text import Text


# Global reference to the RichLog widget — set by the app on mount
_log_widget = None
 
def set_log_widget(widget):
    global _log_widget
    _log_widget = widget
 
 
MSG_TYPE_COLORS = {
    "text":          "#ffffff",
    "greeting":      "#00ffff",
    "greeting-ack":  "#00ff00",
    "knowledge":     "#ffff00",
    "knowledge-ack": "#00ff00",
    "warning":       "#ff0000",
    "newnode":       "#ff00ff",
}
 
 
def log(message, node_id, msg_type=None):
    current_time = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:03d}"
    prefix = f"[NODE {node_id}] [{current_time}] "
 
    line = Text()
    line.append(prefix, style="dim white")
 
    if msg_type is not None:
        color = MSG_TYPE_COLORS.get(msg_type, "#ffffff")
        line.append(f"[{msg_type.upper()}]", style=f"bold {color}")
        line.append(f" {message}", style="#ffffff")
    else:
        line.append(message, style="#ffffff")
 
    if _log_widget is not None:
        # write() is thread-safe in Textual
        _log_widget.write(line)
    else:
        # Fallback if called before the UI is ready
        print(line.plain)


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