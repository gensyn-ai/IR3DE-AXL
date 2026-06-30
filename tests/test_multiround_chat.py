#!/usr/bin/env python3
"""
Test the multiround chat functionality added to peer.py.

Two stages:
  1. Unit tests for conversation management methods (no model needed)
  2. Integration test: download small experts, load one, run a 2-turn conversation
     and verify that the second answer is generated with the full chat history
     in the prompt.
"""

# python test_multiround_chat.py --tokenizer <tokenizer_path>    full test
# python test_multiround_chat.py --skip-integration              unit tests only


import sys, os, uuid

# ── resolve paths so we can import from src/ ──────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# Replicate just the conversation-management logic from peer.py so we can test
# it without spinning up AXL, loading models, or reading local_nodes/.
# ──────────────────────────────────────────────────────────────────────────────
from datetime import datetime

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


def _new_conversation(chat_id: str) -> dict:
    return {"conversation_id": chat_id, "history": []}


def _append_turn(conversation: dict, speaker_type: str, speaker_id: str, content: str):
    turn = len(conversation["history"]) + 1
    conversation["history"].append({
        "id": str(uuid.uuid4()),
        "turn": turn,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "speaker_type": speaker_type,
        "speaker_id": speaker_id,
        "content": content,
    })


def _render_messages(conversation: dict) -> list:
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    for entry in conversation["history"]:
        if entry["speaker_type"] == "user":
            messages.append({"role": "user", "content": entry["content"]})
        else:
            messages.append({
                "role": "assistant",
                "content": f"[agent_id: {entry['speaker_id']}]\n{entry['content']}",
            })
    return messages


def _format_chat_prompt(messages: list) -> str:
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(f"System: {m['content']}")
        elif m["role"] == "user":
            parts.append(f"User: {m['content']}")
        else:
            parts.append(f"Assistant: {m['content']}")
    return "\n\n".join(parts) + "\n\nAssistant: "


# ──────────────────────────────────────────────────────────────────────────────
# UNIT TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_new_conversation():
    conv = _new_conversation("chat_001")
    assert conv["conversation_id"] == "chat_001"
    assert conv["history"] == []
    print("  [PASS] _new_conversation creates empty history with correct id")


def test_append_turn_turn_numbers():
    conv = _new_conversation("t1")
    _append_turn(conv, "user", "user", "Hello")
    _append_turn(conv, "agent", "NodeA", "Hi there")
    _append_turn(conv, "user", "user", "How are you?")
    assert len(conv["history"]) == 3
    assert conv["history"][0]["turn"] == 1
    assert conv["history"][1]["turn"] == 2
    assert conv["history"][2]["turn"] == 3
    print("  [PASS] _append_turn assigns sequential turn numbers")


def test_append_turn_speaker_fields():
    conv = _new_conversation("t2")
    _append_turn(conv, "user", "user", "My question")
    _append_turn(conv, "agent", "NodeX", "My answer")
    assert conv["history"][0]["speaker_type"] == "user"
    assert conv["history"][0]["speaker_id"]   == "user"
    assert conv["history"][1]["speaker_type"] == "agent"
    assert conv["history"][1]["speaker_id"]   == "NodeX"
    print("  [PASS] _append_turn stores speaker_type and speaker_id correctly")


def test_render_messages_structure():
    conv = _new_conversation("t3")
    _append_turn(conv, "user",  "user",  "Turn 1 user")
    _append_turn(conv, "agent", "NodeA", "Turn 1 agent")
    _append_turn(conv, "user",  "user",  "Turn 2 user")

    msgs = _render_messages(conv)
    # system + 3 history turns
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"   and msgs[1]["content"] == "Turn 1 user"
    assert msgs[2]["role"] == "assistant"
    assert "NodeA" in msgs[2]["content"]
    assert "Turn 1 agent" in msgs[2]["content"]
    assert msgs[3]["role"] == "user"   and msgs[3]["content"] == "Turn 2 user"
    print("  [PASS] _render_messages builds correct role sequence with agent_id prefix")


def test_format_chat_prompt_contains_history():
    conv = _new_conversation("t4")
    _append_turn(conv, "user",  "user",  "What is 2+2?")
    _append_turn(conv, "agent", "NodeA", "2+2 is 4.")
    _append_turn(conv, "user",  "user",  "And 3+3?")

    msgs = _render_messages(conv)
    prompt = _format_chat_prompt(msgs)

    assert "System: " in prompt
    assert "User: What is 2+2?" in prompt
    assert "Assistant: " in prompt
    assert "2+2 is 4." in prompt
    assert "User: And 3+3?" in prompt
    assert prompt.endswith("Assistant: "), f"Prompt should end with 'Assistant: ', got: {prompt[-30:]!r}"
    print("  [PASS] _format_chat_prompt includes all history turns and ends with 'Assistant: '")


def test_setdefault_conversation_isolation():
    """Separate chat_ids must not share history."""
    histories: dict = {}
    for chat_id in ["alice", "bob"]:
        conv = histories.setdefault(chat_id, _new_conversation(chat_id))
        _append_turn(conv, "user", "user", f"Message from {chat_id}")

    assert len(histories["alice"]["history"]) == 1
    assert len(histories["bob"]["history"])   == 1
    assert histories["alice"]["history"][0]["content"] == "Message from alice"
    assert histories["bob"]["history"][0]["content"]   == "Message from bob"
    print("  [PASS] Separate chat_ids maintain isolated conversation histories")


def test_single_turn_prompt_no_history():
    """First message: prompt should contain only system + one user turn."""
    conv = _new_conversation("t5")
    _append_turn(conv, "user", "user", "Hello world")
    msgs = _render_messages(conv)
    prompt = _format_chat_prompt(msgs)

    assert "User: Hello world" in prompt
    assert prompt.endswith("Assistant: ")
    # No spurious 'User:' occurrences beyond the one we added
    assert prompt.count("User:") == 1
    print("  [PASS] Single-turn prompt has exactly one User: section")


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST (download + generate)
# ──────────────────────────────────────────────────────────────────────────────

def download_models(checkpoints_dir):
    """Download small expert models if not already present."""
    from huggingface_hub import snapshot_download
    os.makedirs(checkpoints_dir, exist_ok=True)
    # Check if already downloaded (look for any .pth file)
    pth_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    if pth_files:
        print(f"  Models already present in {checkpoints_dir}: {pth_files[:3]}")
        return pth_files
    print(f"  Downloading M2D2-Llama-115M experts to {checkpoints_dir} ...")
    snapshot_download(
        repo_id="Erosinho/M2D2-Llama-115M-fixed-attn-experts",
        local_dir=checkpoints_dir,
        local_dir_use_symlinks=False,
    )
    pth_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    print(f"  Downloaded {len(pth_files)} .pth checkpoint(s).")
    return pth_files


def load_first_expert(checkpoints_dir, device, tokenizer_name_or_path):
    """Load the first available expert model + its tokenizer."""
    import torch
    from transformers import AutoTokenizer
    from ir3de_stats.models.llama_experts import get_llama_expert

    pth_files = sorted(f for f in os.listdir(checkpoints_dir) if f.endswith(".pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth checkpoints found in {checkpoints_dir}")

    ckpt_path = os.path.join(checkpoints_dir, pth_files[0])
    print(f"  Loading expert from: {ckpt_path}")

    model, _, _ = get_llama_expert(1.15e8)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print(f"  Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    return {"model": model, "tokenizer": tokenizer, "path": ckpt_path}


def generate_answer(text, model_utils, device, max_length=128):
    """Replicate peer.generate_answer (prompt-aware version)."""
    import torch
    enc = model_utils["tokenizer"](text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model_utils["model"].generate(
            input_ids=enc["input_ids"],
            max_length=max_length,
        )
    decoded = model_utils["tokenizer"].decode(out[0], skip_special_tokens=True)
    # Strip the prompt prefix to get only the new tokens
    answer = decoded[len(text):]
    return answer.strip()


def test_multiround_generation(model_utils, device):
    """
    Two-turn conversation:
      Turn 1: user asks "What is the capital of France?"
              agent answers → stored in history
      Turn 2: user asks "What language do they speak there?"
              We verify the prompt sent for Turn 2 includes Turn 1 history,
              and that the model produces a non-empty completion.
    """
    chat_id = "integration_test"
    histories: dict = {}

    # --- Turn 1 ---
    user_msg_1 = "What is the capital of France?"
    conv = histories.setdefault(chat_id, _new_conversation(chat_id))
    _append_turn(conv, "user", "user", user_msg_1)
    msgs = _render_messages(conv)
    prompt_1 = _format_chat_prompt(msgs)

    print(f"\n  [Turn 1] prompt length: {len(prompt_1)} chars")
    answer_1 = generate_answer(prompt_1, model_utils, device, max_length=200)
    print(f"  [Turn 1] answer: {answer_1!r}")
    assert len(answer_1) > 0, "Turn 1 answer must be non-empty"

    _append_turn(conv, "agent", "TestNode", answer_1)

    # --- Turn 2 ---
    user_msg_2 = "What language do they speak there?"
    _append_turn(conv, "user", "user", user_msg_2)
    msgs = _render_messages(conv)
    prompt_2 = _format_chat_prompt(msgs)

    print(f"\n  [Turn 2] prompt length: {len(prompt_2)} chars")

    # Verify history is present in Turn 2 prompt
    assert user_msg_1 in prompt_2, "Turn 1 user message must appear in Turn 2 prompt"
    assert answer_1   in prompt_2, "Turn 1 agent answer must appear in Turn 2 prompt"
    assert user_msg_2 in prompt_2, "Turn 2 user message must appear in Turn 2 prompt"
    assert prompt_2.count("User:") == 2, "Turn 2 prompt must contain exactly 2 User: sections"

    answer_2 = generate_answer(prompt_2, model_utils, device, max_length=256)
    print(f"  [Turn 2] answer: {answer_2!r}")
    assert len(answer_2) > 0, "Turn 2 answer must be non-empty"

    # Turn 2 prompt is strictly longer (has extra history)
    assert len(prompt_2) > len(prompt_1), "Turn 2 prompt must be longer than Turn 1 (includes history)"

    print("  [PASS] Multiround generation: prompt grows correctly and both turns produce output")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test multiround chat functionality")
    parser.add_argument("--skip-integration", action="store_true",
                        help="Run only unit tests (no model download/generation)")
    parser.add_argument("--checkpoints-dir", default=os.path.join(REPO_ROOT, "checkpoints"),
                        help="Where to store/find downloaded expert checkpoints")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B",
                        help="HuggingFace model ID or local path for the tokenizer "
                             "(default: meta-llama/Meta-Llama-3-8B, requires HF auth). "
                             "Pass a local path to avoid the gated-repo requirement.")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("UNIT TESTS — conversation management")
    print("="*60)
    test_new_conversation()
    test_append_turn_turn_numbers()
    test_append_turn_speaker_fields()
    test_render_messages_structure()
    test_format_chat_prompt_contains_history()
    test_setdefault_conversation_isolation()
    test_single_turn_prompt_no_history()
    print("\nAll unit tests passed.\n")

    if args.skip_integration:
        print("Skipping integration test (--skip-integration flag set).")
        return

    print("="*60)
    print("INTEGRATION TEST — download + generate")
    print("="*60)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    pth_files = download_models(args.checkpoints_dir)
    if not pth_files:
        print("  No model files found after download. Skipping generation test.")
        return

    model_utils = load_first_expert(args.checkpoints_dir, device, args.tokenizer)
    test_multiround_generation(model_utils, device)

    print("\nAll tests passed.\n")


if __name__ == "__main__":
    main()
