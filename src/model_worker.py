"""Loads and runs expert models in a separate OS process, submitted to the
ProcessPoolExecutor(max_workers=1, initializer=init_worker) that Peer holds
as self.model_executor. A separate process (not just a thread) is required
because a single from_pretrained()/generate() call holds the GIL for its
whole duration, which would freeze the Textual UI if run in-process.

Models are loaded lazily: the first generate() call for a given model_idx
downloads/loads it, and it then stays resident in _MODELS (kept in
least-recently-used order) for as long as this worker process lives. If
free memory is running low when a new model is needed, the
least-recently-used resident model is evicted first to make room. Eviction
only ever drops this process's own in-memory model/tokenizer objects — the
metadata Peer keeps about that model (tags, size, num_requests, latencies,
...) lives entirely in the main process and is never touched here.
"""
import gc
import os
import time
from collections import OrderedDict

import psutil
import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          StopStringCriteria, StoppingCriteriaList)

_MODELS: "OrderedDict[int, dict]" = OrderedDict()
_NEXT_TURN_MARKERS = ("\nUser:", "\nAgent:", "\nAssistant:")

# Always leave at least this much free memory for the OS and everything
# else running, on top of whatever the model being loaded needs.
_MIN_FREE_BYTES = 2 * 1024 ** 3  # 2 GiB


def init_worker() -> None:
    """Runs once when the worker process starts (passed as Peer's
    ProcessPoolExecutor initializer=). Silences this process's stdout/stderr,
    since it shares the main process's terminal and stray output — HF
    downloads, tokenizer warnings — would corrupt Textual's rendering."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


def _device() -> torch.device:
    """CUDA if available in this worker process, else CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _available_bytes() -> int:
    """Free memory in whichever pool models get loaded into on this device
    — VRAM if this worker runs on a GPU, system RAM otherwise."""
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return free
    return psutil.virtual_memory().available


def get_model_info(model_info: dict) -> dict:
    """Cheap metadata lookup for one model — its config and safetensors
    header, never the weights themselves — used for the Statistics tab and
    for estimating how much memory loading it will need later. Called once
    per configured model by Peer.get_models_info() during Peer.__init__;
    the weights aren't actually downloaded until generate() first needs
    them (see _ensure_loaded).

    The safetensors header fetch is a network call and transient
    connection resets are common, so it's retried a few times before
    giving up — a size of 0 here isn't just cosmetic (it's shown on the
    Statistics tab), it also feeds _estimated_resident_bytes, so silently
    accepting the first failure would under-reserve memory for this model
    later."""
    if "hf_name" not in model_info:
        raise ValueError(
            f"Model info must contain 'hf_name' (only remote Hugging Face "
            f"models are supported). Provided info: {model_info}."
        )
    config = AutoConfig.from_pretrained(model_info["hf_name"])
    from huggingface_hub import get_safetensors_metadata
    size = 0
    for attempt in range(3):
        try:
            meta = get_safetensors_metadata(model_info["hf_name"])
            size = sum(meta.parameter_count.values())
            break
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return {"size": size, "type": getattr(config, "model_type", "unknown")}


def _estimated_resident_bytes(size: int) -> int:
    """Rough resident-memory estimate for a model with `size` parameters,
    loaded in bfloat16 (2 bytes/param), padded for activations and
    framework overhead. Falls back to a conservative flat guess if `size`
    is unknown (e.g. get_model_info couldn't read the safetensors header)."""
    if size <= 0:
        return 4 * 1024 ** 3
    return int(size * 2 * 1.2)


def _evict_lru_until_it_fits(needed_bytes: int) -> list[int]:
    """Evict resident models least-recently-used first (the oldest entry in
    _MODELS, an OrderedDict kept in access order) until enough memory is
    free for `needed_bytes`, or nothing's left resident to evict. Returns
    the model_idx of every model evicted, in eviction order, so the caller
    (this process's stdout/stderr are silenced — see init_worker — so it
    can't log anything itself) can report it back to the main process."""
    evicted = []
    while _MODELS and _available_bytes() < needed_bytes + _MIN_FREE_BYTES:
        oldest_idx, _ = next(iter(_MODELS.items()))
        del _MODELS[oldest_idx]
        evicted.append(oldest_idx)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return evicted


def _model_chat_template(model_info: dict, tokenizer):
    """Resolve a template without changing the configured tokenizer.

    Prefer the configured tokenizer's template. If it has none, read only the
    template metadata from the model repository, which lets heterogeneous
    experts supply their own format without model-family conditionals.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return template
    if model_info["hf_name"] == model_info["tokenizer"]:
        return None
    try:
        model_tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"])
    except Exception:
        return None
    return getattr(model_tokenizer, "chat_template", None)


def _plain_text_messages(messages: list[dict[str, str]]) -> str:
    """Fallback serialization for experts that publish no chat template."""
    labels = {"system": "System", "user": "User", "assistant": "Agent"}
    parts = [
        f"{labels.get(message['role'], message['role'].title())}: "
        f"{message['content']}"
        for message in messages
    ]
    parts.append("Agent: ")
    return "\n".join(parts)


def _encode_generation_input(model_utils: dict, message, device):
    tokenizer = model_utils["tokenizer"]
    if isinstance(message, list) and model_utils.get("chat_template"):
        try:
            input_ids = tokenizer.apply_chat_template(
                message,
                chat_template=model_utils["chat_template"],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(device)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        except Exception:
            # A third-party template may be incompatible with the configured
            # tokenizer. Preserve service by falling back to plain text.
            pass

    text = _plain_text_messages(message) if isinstance(message, list) else message
    return tokenizer(text, return_tensors="pt").to(device)


def _trim_next_turn(answer: str) -> str:
    positions = [
        answer.find(marker)
        for marker in _NEXT_TURN_MARKERS
        if marker in answer
    ]
    return answer[:min(positions)].rstrip() if positions else answer.strip()


def _ensure_loaded(model_idx: int, model_info: dict) -> tuple[bool, list[int]]:
    """Load model_idx into _MODELS if it isn't resident yet — evicting the
    least-recently-used resident model(s) first if memory is tight — and
    mark it as most-recently-used either way. Returns (loaded_now,
    evicted_indices): loaded_now is False if model_idx was already
    resident (nothing to report), True if it was just loaded."""
    if model_idx in _MODELS:
        _MODELS.move_to_end(model_idx)
        return False, []

    needed = _estimated_resident_bytes(model_info.get("size", 0))
    evicted = _evict_lru_until_it_fits(needed)

    device = _device()
    model = AutoModelForCausalLM.from_pretrained(model_info["hf_name"], dtype=torch.bfloat16)
    model.to(device)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
    chat_template = _model_chat_template(model_info, tokenizer)
    try:
        stopping_criteria = StoppingCriteriaList([
            StopStringCriteria(tokenizer, list(_NEXT_TURN_MARKERS))
        ])
    except Exception:
        stopping_criteria = None
    _MODELS[model_idx] = {
        "model": model,
        "tokenizer": tokenizer,
        "chat_template": chat_template,
        "stopping_criteria": stopping_criteria,
    }
    return True, evicted


def generate(model_idx: int, model_info: dict, message: str | list[dict[str, str]], max_new_tokens: int) -> tuple[str, int, int, bool, list[int]]:
    """Generate a reply from model_idx, loading it first if it isn't
    already resident (see _ensure_loaded). Returns (answer,
    num_input_tokens, num_output_tokens, loaded_now, evicted_indices) —
    the last two let the caller log the load/eviction in the main process,
    since this one's own stdout/stderr are silenced (see init_worker).
    Called from peer.py's recv_loop (answering a remote request) and
    _continue_handle_user_input (answering the local user), both via
    Peer.model_executor.submit()."""
    loaded_now, evicted = _ensure_loaded(model_idx, model_info)
    model_utils = _MODELS[model_idx]
    device = _device()
    encoding = _encode_generation_input(model_utils, message, device)
    input_ids = encoding['input_ids']
    num_input_tokens = int(input_ids.shape[1])
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": encoding.get("attention_mask"),
        "max_new_tokens": max_new_tokens,
    }
    if model_utils["stopping_criteria"] is not None:
        generation_kwargs["stopping_criteria"] = model_utils["stopping_criteria"]
    out = model_utils['model'].generate(**generation_kwargs)
    new_tokens = out[0, num_input_tokens:]
    answer = _trim_next_turn(
        model_utils['tokenizer'].decode(new_tokens, skip_special_tokens=True)
    )
    num_output_tokens = int(new_tokens.shape[0])
    return answer, num_input_tokens, num_output_tokens, loaded_now, evicted
