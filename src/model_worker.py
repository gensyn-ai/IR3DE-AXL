"""Loads and runs expert models in a separate OS process.

A single long-running C call inside from_pretrained()/generate() can hold
the GIL for its entire duration, regardless of Python-level thread-priority
tuning (sys.setswitchinterval only affects how often the interpreter checks
for a pending switch *between bytecode instructions* — it can't interrupt a
call that never returns control to the bytecode loop). Running that work in
a genuinely separate process sidesteps the GIL entirely: each process has
its own, and the OS scheduler — not CPython — arbitrates between them, so
the main process's Textual event loop is never blocked by it.

Every function here runs *inside* the worker process, submitted via a
ProcessPoolExecutor(max_workers=1, initializer=init_worker) held by Peer.
Module-level state persists across calls for as long as that one worker
stays alive, so a model loaded by load_model() remains resident in _MODELS
for later generate() calls — it's never pickled back to the caller.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODELS: dict[int, dict] = {}


def init_worker() -> None:
    """ProcessPoolExecutor initializer. Silences this process's own
    stdout/stderr permanently, since it inherits the same terminal Textual
    renders to in the main process — any stray print/progress-bar output
    here (HF Hub downloads, tokenizer warnings, ...) would corrupt that
    rendering otherwise."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))


def _device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_idx: int, model_info: dict) -> dict:
    """Load one expert model + tokenizer into this worker process and keep
    it resident in _MODELS. Returns lightweight metadata only (size/type) —
    the model itself never leaves this process."""
    device = _device()
    model = AutoModelForCausalLM.from_pretrained(model_info["hf_name"])
    model.to(device)  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
    _MODELS[model_idx] = {"model": model, "tokenizer": tokenizer}

    return {
        "size": sum(p.numel() for p in model.parameters()),
        "type": getattr(model.config, "model_type", "unknown"),
    }


def generate(model_idx: int, message: str, max_new_tokens: int) -> tuple[str, int, int]:
    """Generate a reply from an already-loaded model. Returns
    (answer, num_input_tokens, num_output_tokens)."""
    model_utils = _MODELS[model_idx]
    device = _device()
    encoding = model_utils['tokenizer'](message, return_tensors='pt').to(device)
    input_ids = encoding['input_ids']
    num_input_tokens = int(input_ids.shape[1])
    out = model_utils['model'].generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
    new_tokens = out[0, num_input_tokens:]
    answer = model_utils['tokenizer'].decode(new_tokens, skip_special_tokens=True)
    num_output_tokens = int(new_tokens.shape[0])
    return answer, num_input_tokens, num_output_tokens
