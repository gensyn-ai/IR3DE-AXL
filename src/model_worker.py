"""Loads and runs expert models in a separate OS process, submitted to the
ProcessPoolExecutor(max_workers=1, initializer=init_worker) that Peer holds
as self.model_executor. A separate process (not just a thread) is required
because a single from_pretrained()/generate() call holds the GIL for its
whole duration, which would freeze the Textual UI if run in-process.

_MODELS is per-worker-process state that outlives any single call: a model
loaded by load_model() stays resident there for later generate() calls and
is never pickled back to the caller.
"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODELS: dict[int, dict] = {}


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


def load_model(model_idx: int, model_info: dict) -> dict:
    """Load one expert model + tokenizer into this worker process and keep
    it resident in _MODELS. Returns lightweight metadata only (size/type) —
    the model itself never leaves this process. Called once per configured
    model by Peer.get_models_info() during Peer.__init__."""
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
    (answer, num_input_tokens, num_output_tokens). Called from peer.py's
    recv_loop (answering a remote request) and _continue_handle_user_input
    (answering the local user), both via Peer.model_executor.submit()."""
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
