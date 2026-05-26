from termcolor import colored


def log(message, node_id, msg_type=None):
    if msg_type is None:
        print(f"[NODE {node_id}] {message}")
    else:
        color = {
            "text": "white",
            "greeting": "cyan",
            "greeting-ack": "green",
            "knowledge": "yellow",
            "knowledge-ack": "green",
            "warning": "red"
        }.get(msg_type, "white")
        print(f"[NODE {node_id}] {colored(f'[{msg_type.upper()}]', color)} {message}")
    