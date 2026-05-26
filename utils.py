import ipaddress
import time

from termcolor import colored


def log(message, node_id, msg_type=None):
    current_time = time.strftime('%H:%M:%S') + f".{int(time.time() * 1000) % 1000:03d}"
    if msg_type is None:
        print(f"[NODE {node_id}] [{current_time}] {message}")
    else:
        color = {
            "text": "white",
            "greeting": "cyan",
            "greeting-ack": "green",
            "knowledge": "yellow",
            "knowledge-ack": "green",
            "warning": "red",
            "newnode": "magenta"
        }.get(msg_type, "white")
        print(f"[NODE {node_id}] [{current_time}] {colored(f'[{msg_type.upper()}]', color)} {message}")


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