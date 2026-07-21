"""The Peer class: one node in the IR3DE-AXL P2P network. Not run directly —
constructed by run.py, which owns the background threads that drive it
(recv_loop for inbound messages, and the periodic gossip loop in run_peer).

A Peer wraps three things:
  - The AXL node: a Go P2P subprocess (started in __init__) reached over a
    local HTTP API for sending/receiving messages and reading topology.
  - IR3DE routing: builds a ridge-regression router from local + gossiped
    per-tag (A, b) stats matrices (get_token_router), used to pick which
    expert model should answer a given user message (find_best_tag,
    find_best_model), for whichever tags the user has selected.
  - Chats: this peer's own conversation history (via chats.py), including
    dispatching user messages to local or remote experts
    (handle_user_input) and summarizing older turns once a chat exceeds its
    character budget (_ensure_within_budget).

Expert models themselves are loaded and run in a separate OS process (see
model_worker.py) reached through self.model_executor, not in this process.
"""

import atexit
import json
import multiprocessing
import os
import pathlib
import random
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import chats
import model_worker
from utils import (deserialize_chunk_header, deserialize_safe, ensure_stats_file,
                    format_prompt_for_expert, ipv6_from_pubkey, load_embedder_only, log,
                    MAX_TITLE_CHARS, serialize_safe, SUMMARY_SYSTEM_PROMPT, TITLE_SYSTEM_PROMPT)


AXL = "http://127.0.0.1:91"

PEER_DEAD_TIMEOUT = 300  # seconds with no inbound from a peer before it's pruned (refreshed on any inbound)

LOCAL_PEER_ID_PATH = pathlib.Path("local_nodes/local_peer_id.txt")


def _load_or_create_local_peer_id() -> str:
    """Stable id for the no-`--peer-id` peer, persisted alongside its
    pk.pem/metadata.json so it survives across runs instead of being
    regenerated (and thus changing) every launch."""
    if LOCAL_PEER_ID_PATH.is_file():
        existing = LOCAL_PEER_ID_PATH.read_text().strip()
        if existing:
            return existing
    new_id = uuid.uuid4().hex[:8]
    LOCAL_PEER_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_PEER_ID_PATH.write_text(new_id)
    return new_id


class Peer:

    def __init__(self, peer_id, ir3de_lambda=0.01, ir3de_entropy_top_k=10, max_answer_length=256,
                 num_characters_conversation_history=8000, on_axl_ready=None, tokenizer_name="mistralai/Mistral-7B-v0.1"):
        """Start this peer's AXL node subprocess, load its expert models and
        matching IR3DE stats, and load (or create) its chats. Constructed
        once by run.py's run_peer on a background thread; blocks for several
        seconds while the AXL node comes up and models load."""

        with open(f"ir3de_stats/default_stats.json", "r") as f:
            default_stats = json.load(f)

        if peer_id is None:
            metadata_path = f"local_nodes/metadata.json"
            log_path = f"axl-logs/node.log"
        else:
            metadata_path = f"local_nodes/metadata{peer_id:02d}.json"
            log_path = f"axl-logs/node-{peer_id:02d}.log"

        self.metadata_path = metadata_path
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pathlib.Path("axl-logs").mkdir(exist_ok=True)
        log_fp = open(log_path, "w", buffering=1)
        start_node_args = ["./scripts/start_node.sh", str(peer_id)] if peer_id is not None else ["./scripts/start_node.sh"]
        self.proc = subprocess.Popen(
            start_node_args,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so stop_node() can kill the whole node tree
        )
        atexit.register(self.stop_node)  # deterministic node cleanup on normal interpreter exit
        sleep_time = 5
        self.peer_id = peer_id if peer_id is not None else _load_or_create_local_peer_id()
        self._axl_suffix = f"{peer_id:02d}" if isinstance(peer_id, int) else "00"
        self.node_name = metadata.get("node_name", f"Node {self.peer_id}")
        log(f"Waiting {sleep_time} seconds for node {self.peer_id} to initialize...", self.peer_id, msg_type=None)
        time.sleep(sleep_time)

        if on_axl_ready is not None:
            try:
                on_axl_ready()
            except Exception as e:
                log(f"on_axl_ready callback failed: {e}", self.peer_id, msg_type="warning")

        self.session = requests.Session()
        self.topology = self.get_topology(self.session)
        self.public_key = self.topology['our_public_key']
        self.ipv6_address = self.topology['our_ipv6']
        self.known_public_keys = {}
        self.awaiting_acks = {}
        self.last_seen = {}  # pk -> last time we received anything from it; drives pruning

        self.known_tags = []
        self.selected_tags: set[str] = set()
        self.selected_models: dict[str, tuple[str, int]] = {}
        self.default_tokenizer_name = self.default_embedder_name = tokenizer_name

        # Models load/generate in model_worker's own process (see its module
        # docstring for why). mp_context="spawn" because this process may
        # already hold a CUDA context by now (e.g. torch.cuda.is_available()
        # above) — the default "fork" start method would clone that context
        # into the worker, which CUDA doesn't allow. atexit-registered since
        # the pool is a real OS process that must be torn down on exit.
        self.model_executor = ProcessPoolExecutor(
            max_workers=1,
            initializer=model_worker.init_worker,
            mp_context=multiprocessing.get_context("spawn"),
        )
        atexit.register(self.model_executor.shutdown)

        self.models_info = metadata.get("models", [])
        self.models = self.get_models_info()

        # Only load stats matching this peer's active (tokenizer, embedder)
        # identifier — entries for other identifiers (e.g. Llama stats while
        # running --tok-type mistral) are skipped before ever downloading or
        # torch.load-ing them.
        all_stats_info = default_stats.get("stats", []) + metadata.get("stats", [])
        seen_paths = set()
        deduped_stats_info = []
        for s in all_stats_info:
            if s["path"] in seen_paths:
                continue  # e.g. metadata.json re-listing a stat already in default_stats.json
            seen_paths.add(s["path"])
            deduped_stats_info.append(s)
        self.stats_info = [
            s for s in deduped_stats_info
            if s.get("tokenizer_name") == self.default_tokenizer_name
            and s.get("embedder_name") == self.default_embedder_name
        ]
        skipped = len(deduped_stats_info) - len(self.stats_info)
        if skipped:
            log(f"Skipped {skipped} stats entr{'y' if skipped == 1 else 'ies'} not matching "
                f"active identifier ({self.default_tokenizer_name}, {self.default_embedder_name}).",
                self.peer_id, msg_type=None)
        self.stats = self.get_stats_info()

        self.default_lambda = ir3de_lambda
        self.entropy_top_k = ir3de_entropy_top_k
        self.max_answer_length = max_answer_length
        self.budget_chars = num_characters_conversation_history

        self.local_A = {}
        self.local_b = {}
        for stats_info, stats in zip(self.stats_info, self.stats):
            identifier = (stats_info['tokenizer_name'], stats_info['embedder_name'])
            if identifier not in self.local_A:
                self.local_A[identifier] = {}
                self.local_b[identifier] = {}
            for tag, A, b in zip(stats['tags'], stats['A'], stats['b']):
                if tag in self.local_A[identifier]:
                    self.local_A[identifier][tag] = self.local_A[identifier][tag] + A
                    self.local_b[identifier][tag] = self.local_b[identifier][tag] + b
                else:
                    self.local_A[identifier][tag] = A
                    self.local_b[identifier][tag] = b

        self.chunks = {}

        self.chats: dict[str, dict] = {}
        self.active_chat_id: str | None = None
        self.tmp_trimmed_pairs: dict[str, list] = {}
        self.pending_continuations: dict[str, dict] = {}
        # Explicit "is chat X still waiting for an answer" tracking, for the UI's
        # waiting indicator. pending_continuations/awaiting_acks are narrow,
        # network-ack-specific bookkeeping (they don't cover local generation at
        # all, and awaiting_acks is keyed by msg_id with mixed entry types) — this
        # is the one authoritative signal, marked once per user message at
        # dispatch and resolved exactly once at whichever terminal point the
        # request reaches, regardless of local/remote or summary-continuation hops.
        self.pending_answers: dict[str, int] = {}

        # Load chats authored by THIS peer; create one new chat if none.
        existing = chats.list_chats()
        for meta in existing:
            try:
                chat = chats.load_chat(meta["chat_id"])
            except Exception as e:
                log(f"Failed to load chat {meta['chat_id']}: {e}",
                    self.peer_id, msg_type="warning")
                continue
            if chat.get("peer_id") != self.peer_id:
                continue                                # belongs to another peer
            self.chats[chat["chat_id"]] = chat

        if self.chats:
            # list_chats() returns metadata sorted by updated_at desc, and we
            # insert in that order, so next(iter(...)) is the most-recent.
            self.active_chat_id = next(iter(self.chats))
        else:
            self.new_chat()
        
        atexit.register(self._save_all_chats_safely)
    
    def get_topology(self, session):
        """Fetch this node's AXL topology (public key, IPv6, known peers)
        once at startup, right after the node subprocess comes up."""
        resp = session.get(f"{AXL}{self._axl_suffix}/topology", timeout=5)
        resp.raise_for_status()
        topology = resp.json()
        return topology

    def get_models_info(self):
        """Fetch cheap metadata (config + safetensors header, no weights)
        for every model listed in this peer's metadata.json and collect
        their tags. Called once from __init__; feeds the Statistics tab's
        local models table and the Control Panel's expertise sections. The
        weights themselves aren't downloaded/loaded until a model is first
        actually used — see model_worker.generate."""
        models = []
        for i, model_info in enumerate(self.models_info):
            display_name = model_info.get("hf_name", "unknown")
            log(f"Fetching info for expert model {display_name}", self.peer_id, msg_type=None)

            try:
                meta = self.model_executor.submit(model_worker.get_model_info, model_info).result()
            except ValueError as e:
                raise ValueError(f"{e} Check the {self.metadata_path} file.") from e

            model_tags = model_info.get("tags", [])
            for tag in model_tags:
                if tag not in self.known_tags:
                    self.known_tags.append(tag)

            models.append({"tags": model_tags})
            model_info["size"] = meta["size"]
            model_info["type"] = meta["type"]
            if meta["size"] <= 0:
                log(f"Could not determine parameter count for {display_name} "
                    f"(safetensors metadata fetch failed after retries); its size will "
                    f"show as 0 and its memory estimate will fall back to a flat guess.",
                    self.peer_id, msg_type="warning")
        return models

    def _log_model_load_events(self, model_idx, loaded_now, evicted):
        """Log a generate() call's lazy-load/eviction activity, in this
        (the main) process — model_worker's own process has its
        stdout/stderr silenced (see model_worker.init_worker), so it
        reports what happened back through generate()'s return value
        instead of logging it directly."""
        if loaded_now:
            hf_name = self.models_info[model_idx].get("hf_name", "unknown")
            log(f"Loaded expert model {hf_name} into memory (first use since this node started).",
                self.peer_id, msg_type=None)
        for evicted_idx in evicted:
            hf_name = self.models_info[evicted_idx].get("hf_name", "unknown")
            log(f"Evicted expert model {hf_name} from memory to free up space.",
                self.peer_id, msg_type=None)

    def get_stats_info(self):
        """Download (if needed) and load every stats entry in self.stats_info
        that matches this peer's active tokenizer/embedder, deduplicating
        shared tokenizer/embedder loads. Called once from __init__, after
        which self.local_A/local_b are built from the result."""
        all_stats = []
        valid_stats_info = []  # kept in lockstep with all_stats; entries that fail to
                                # load (even after a re-download) are dropped from both,
                                # so callers zip()-ing them together never desync.
        loaded = {}  # (tokenizer_name, embedder_name) -> (tokenizer, embedder); several
                     # default stats entries share the same pair, so load each only once.

        for stats in self.stats_info:

            stats_path = ensure_stats_file(stats["path"], self.peer_id)
            try:
                stats_data = torch.load(stats_path, map_location='cpu')
            except Exception:
                os.remove(stats_path)
                stats_path = ensure_stats_file(stats["path"], self.peer_id)
                try:
                    stats_data = torch.load(stats_path, map_location='cpu')
                except Exception:
                    continue

            key = (stats_data["tokenizer"], stats_data["embedder"])
            if key not in loaded:
                tokenizer = AutoTokenizer.from_pretrained(stats_data["tokenizer"])
                embedder = load_embedder_only(stats_data["embedder"])
                loaded[key] = (tokenizer, embedder)
            tokenizer, embedder = loaded[key]

            stats['tokenizer_name'] = stats_data["tokenizer"]
            stats['embedder_name'] = stats_data["embedder"]
            valid_stats_info.append(stats)
            A = stats_data["A"]
            b = stats_data["b"]
            all_stats.append({
                "A": A,
                "b": b,
                "tokenizer": tokenizer,
                "embedder": embedder,
                "tags": stats_data["domain_tags"],
                "datasets": stats_data["datasets_names"],
                "owner_public_key": self.public_key
            })

            for tag in stats_data["domain_tags"]:
                if tag not in self.known_tags:
                    self.known_tags.append(tag)

        self.stats_info = valid_stats_info
        return all_stats

    def send(self, message, peer_public_key, timeout=5, large=False):
        """POST one JSON message to a peer via the local AXL node. If
        `large`, chunk it through serialize_safe instead (used for IR3DE
        stats payloads, which can exceed AXL's message-size limit). Returns
        whether the send succeeded; never raises."""
        try:
            if not large:
                self.session.post(
                    f"{AXL}{self._axl_suffix}/send",
                    headers={"X-Destination-Peer-Id": peer_public_key},
                    data=json.dumps(message),
                    timeout=timeout
                )
            else:
                orig_msg_id = message.get("orig_msg_id")
                msg_id = message.get("msg_id")
                msg_type = message.get("type")
                pk_from = self.public_key
                pk_to = peer_public_key
                chunks, chunks_msg_ids = serialize_safe(message, orig_msg_id, msg_id, msg_type, pk_from, pk_to)
                for i, chunk in enumerate(chunks):
                    log(f"Sending chunk {i+1}/{len(chunks)} to {peer_public_key[:8]}...", self.peer_id, msg_type="stats")
                    self.session.post(
                        f"{AXL}{self._axl_suffix}/send",
                        headers={"X-Destination-Peer-Id": peer_public_key},
                        data=chunk,
                        timeout=timeout
                    )
                    self.awaiting_acks[chunks_msg_ids[i]] = {
                        "type": msg_type,
                        "receiver": pk_to,
                        "timestamp": time.time()
                    }
        except requests.exceptions.Timeout:
            log(f"send to {peer_public_key[:8]} timed out. msg_id = {message.get('msg_id')}, msg_type = {message.get('type')}", self.peer_id, msg_type="warning")
            return False
        except requests.exceptions.RequestException as e:
            log(f"send to {peer_public_key[:8]} failed: {e}. msg_id = {message.get('msg_id')}, msg_type = {message.get('type')}", self.peer_id, msg_type="warning")
            return False
        except Exception as e:
            log(f"Unexpected error when sending to {peer_public_key[:8]}: {e}. msg_id = {message.get('msg_id')}, msg_type = {message.get('type')}", self.peer_id, msg_type="warning")
            return False
        return True

    def new_peer_discovered(self, msg, sender):
        """Register a not-yet-known public key as a peer, starting its
        liveness clock. Called from recv_loop's greeting/knowledge/info/
        stats-req handlers whenever `sender` isn't already in known_public_keys."""
        log(f"New node discovered with ID = {msg.get('peer_id')}!", self.peer_id, msg_type="newnode")
        self.known_public_keys[sender] = {}
        self.known_public_keys[sender]["peer_id"] = msg.get("peer_id")
        self.known_public_keys[sender]["peer_name"] = msg.get("peer_name")
        self.last_seen[sender] = time.time()  # start the liveness clock at discovery

    def recv_loop(self, timeout=120):
        """Poll the AXL node for inbound messages and dispatch each by type
        (text/answer/summary/greeting/knowledge/info/stats-req/*-ack). Runs
        forever on its own daemon thread, started by run.py's run_peer right
        after Peer() is constructed. Self-heals if the AXL node goes down,
        and never lets one bad message kill the loop."""
        node_unreachable = False

        while True:

            try:
                resp = self.session.get(f"{AXL}{self._axl_suffix}/recv")
            except requests.exceptions.RequestException as e:
                # The AXL node subprocess died, was killed, or is mid-restart.
                # Left unguarded, this exception kills the whole recv_loop
                # thread permanently — the peer would silently stop receiving
                # anything for the rest of the run. Keep retrying instead, so
                # it self-heals if the node comes back; log the transition
                # once rather than once per attempt to avoid flooding the log
                # tab while it's down.
                if not node_unreachable:
                    log(f"Lost connection to the AXL node ({e}). Will keep retrying in the background.",
                        self.peer_id, msg_type="warning")
                    node_unreachable = True
                time.sleep(2)
                continue

            if node_unreachable:
                log("Connection to the AXL node restored.", self.peer_id, msg_type=None)
                node_unreachable = False

            if resp.status_code == 200 and len(resp.content) > 0 and not resp.content.startswith(b"{"):
                
                try:
                    
                    msg = deserialize_chunk_header(resp.content)
                    
                    if msg['msg_type'] == 'stats':
                        
                        if msg['pk_from'] not in self.chunks:
                            self.chunks[msg['pk_from']] = {}
                        
                        if (msg['orig_msg_id'], msg['msg_id']) not in self.chunks[msg['pk_from']]:
                            self.chunks[msg['pk_from']][(msg['orig_msg_id'], msg['msg_id'])] = []
                        
                        if msg['chunk_idx'] not in [idx for idx, _ in self.chunks[msg['pk_from']][(msg['orig_msg_id'], msg['msg_id'])]]:
                            self.chunks[msg['pk_from']][(msg['orig_msg_id'], msg['msg_id'])].append((msg['chunk_idx'], resp.content))
                        else:
                            log(f"Received duplicate chunk {msg['chunk_idx']} for message ID {msg['msg_id'][:8]}... from {msg['pk_from'][:8]}... Ignoring duplicate chunk.", self.peer_id, msg_type="warning")
                        
                        log(f"Received chunk {msg['chunk_idx']} for message ID {msg['msg_id'][:8]}... from {msg['pk_from'][:8]}... Total chunks received for this message: {len(self.chunks[msg['pk_from']][(msg['orig_msg_id'], msg['msg_id'])])}", self.peer_id, msg_type="stats")
                        
                        stats_ack = {
                            "msg_id": msg.get("chunk_msg_id"),
                            "type": "stats-ack",
                            "from": self.public_key,
                            "peer_id": self.peer_id,
                            "peer_name": self.node_name
                        }

                        self.send(stats_ack, msg['pk_from'], timeout=timeout)

                        if len(self.chunks[msg['pk_from']][(msg['orig_msg_id'], msg['msg_id'])]) == msg["num_chunks"]:
                            
                            log(f"All chunks received for message ID {msg['msg_id'][:8]}... from {msg['pk_from'][:8]}... Reconstructing message...", self.peer_id, msg_type="stats")
                            
                            try:
                                stats = deserialize_safe([chunk for _, chunk in self.chunks[msg['pk_from']][(msg['orig_msg_id'], msg['msg_id'])]])
                                log(f"Successfully reconstructed message ID {msg['msg_id'][:8]}... from {msg['pk_from'][:8]}... !", self.peer_id, msg_type="stats")
                                sender = resp.headers.get("X-From-Peer-Id")
                                self.known_public_keys[sender]["stats"] = stats['stats']  # type: ignore

                            except Exception as e:
                                log(f"Failed to reconstruct message ID {msg['msg_id'][:8]}... from {msg['pk_from'][:8]}... Error: {e}", self.peer_id, msg_type="warning")
                            
                            del self.chunks[msg['pk_from']][(msg['orig_msg_id'], msg['msg_id'])]

                    else:
                        log(f"Received non-JSON message with unknown type {msg.get('type')} from {msg.get('pk_from')[:8]}... Ignoring.", self.peer_id, msg_type="warning")

                except Exception as e:
                    log(f"Failed to deserialize received message: {e}. Ignoring. Message content (truncated): {str(resp.content)[:100]}...", self.peer_id, msg_type="warning")
                    time.sleep(0.5)
                    continue
            
            elif resp.status_code == 200:
                
                sender = resp.headers.get("X-From-Peer-Id")
                if sender is None:
                    log(f"Received message without sender information. Ignoring. Message content (truncated): {str(resp.text)[:100]}...", self.peer_id, msg_type="warning")
                    time.sleep(0.5)
                    continue
                self.last_seen[sender] = time.time()  # any inbound message proves the peer is alive
                if resp.text is None or resp.text == "":
                    log(f"Received empty message from {sender[:8]}.... Ignoring.", self.peer_id, msg_type="warning")
                    time.sleep(0.5)
                    continue
                try:
                    msg = json.loads(resp.text)
                except json.JSONDecodeError:
                    log(f"Failed to decode JSON message from {sender[:8]}.... Ignoring. Message content (truncated): {str(resp.text)[:100]}...", self.peer_id, msg_type="warning")
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    log(f"Unexpected error when decoding message from {sender[:8]}: {e}. Ignoring. Message content (truncated): {str(resp.text)[:100]}...", self.peer_id, msg_type="warning")
                    time.sleep(0.5)
                    continue
                
                try:
                    if msg.get("type") == "text":

                        message = msg.get('message')
                        log(f"From {sender[:8]}...: {message}", self.peer_id, msg_type="text")

                        model_idx = msg['selected_model'][1]
                        future = self.model_executor.submit(model_worker.generate, model_idx, self.models_info[model_idx], message, self.max_answer_length)
                        try:
                            answer, num_input_tokens, num_output_tokens, loaded_now, evicted = future.result(timeout=timeout)
                            self._log_model_load_events(model_idx, loaded_now, evicted)
                        except FuturesTimeoutError:
                            log(f"Generation timed out after {timeout}s for request from {sender[:8]}...", self.peer_id, msg_type="warning")
                            time.sleep(0.5)
                            continue
                        except Exception as e:
                            log(f"Generation failed for request from {sender[:8]}...: {e}", self.peer_id, msg_type="warning")
                            time.sleep(0.5)
                            continue
                        log(f"To {sender[:8]}...: {answer}", self.peer_id, msg_type="text")

                        # If the requester asked for a title, generate one with the same expert
                        # that just answered. The user's text is the last 'User:' block of the
                        # received prompt; for the first turn that's the only one.
                        title = ""
                        if msg.get("request_title"):
                            user_text = ""
                            if isinstance(message, str):
                                # Take everything after the last "User:" up to the next "Agent:"
                                # (works because we always end the prompt with "Agent: ").
                                after_user = message.rsplit("User:", 1)
                                if len(after_user) == 2:
                                    user_text = after_user[1].split("Agent:", 1)[0].strip()
                            if user_text:
                                title = self._generate_title(user_text, answer, msg['selected_model'][1])


                        msg_id = str(uuid.uuid4())
                        answer_msg = {
                            "orig_msg_id":       msg.get("msg_id"),
                            "msg_id":            msg_id,
                            "type":              "answer",
                            "from":              self.public_key,
                            "peer_id":           self.peer_id,
                            "peer_name":         self.node_name,
                            "message":           answer,
                            "selected_model":    msg.get("selected_model"),
                            "num_input_tokens":  num_input_tokens,
                            "num_output_tokens": num_output_tokens,
                            "title":             title,
                        }
                        self.send(answer_msg, sender, timeout=timeout)
                
                    elif msg.get("type") == "answer":

                        log(f"Received answer from {sender[:8]}...", self.peer_id, msg_type="text")
                        orig = msg.get("orig_msg_id")
                        pending = self.awaiting_acks.pop(orig, None) if orig else None
                        chat_id_for_log = pending.get("chat_id") if pending else None
                        log(f"{msg.get('message')}", msg.get("peer_id"), msg_type="text", right=True, chat_id=chat_id_for_log)

                        if pending is not None:
                            log(f"Removing message ID {orig[:8]}... from awaiting ACKs.", self.peer_id, msg_type="ir3de-ack")

                            prompt_latency = time.time() - pending["timestamp"]
                            num_input_tokens = int(msg.get("num_input_tokens"))
                            num_output_tokens = int(msg.get("num_output_tokens"))
                            num_total_tokens = num_input_tokens + num_output_tokens

                            latency_per_total = prompt_latency / num_total_tokens
                            latency_per_input = prompt_latency / num_input_tokens

                            sel = msg.get("selected_model")
                            model_idx = sel[1]
                            info = self.known_public_keys[sender]
                            latencies_dict = info.setdefault("model_latencies", {})
                            entries = latencies_dict.setdefault(model_idx, [])
                            entries.append({
                                "prompt_latency":          prompt_latency,
                                "num_input_tokens":        num_input_tokens,
                                "num_output_tokens":       num_output_tokens,
                                "latency_per_total_token": latency_per_total,
                                "latency_per_input_token": latency_per_input,
                            })

                            log(f"Prompt latency to peer {sender[:8]}... (model idx {model_idx}): "
                                f"{prompt_latency:.4f} s [in={num_input_tokens} tok, out={num_output_tokens} tok]",
                                self.peer_id, msg_type="text")
                            log(f"Latency per total token: {latency_per_total*1000:.2f} ms/tok "
                                f"({num_total_tokens} tokens total)",
                                self.peer_id, msg_type="text")
                            log(f"Latency per input token: {latency_per_input*1000:.2f} ms/tok",
                                self.peer_id, msg_type="text")

                            chat_id = pending.get("chat_id")
                            if chat_id:
                                self._mark_answer_resolved(chat_id)
                            chat = self.chats.get(chat_id) if chat_id else None
                            if sel is not None and chat is not None:
                                chats.add_agent_message(
                                    chat, msg.get("message", ""),
                                    peer_pk=sel[0],
                                    model_idx=sel[1],
                                    tag=pending.get("tag"),
                                )
                                title = msg.get("title")
                                if title and chat.get("title") is None:
                                    cleaned = self._clean_title(title)
                                    if cleaned:
                                        chats.set_title(chat, cleaned)
                                        log(f"Chat title set: '{cleaned}'", self.peer_id, msg_type="text")

                    elif msg.get("type") == 'summary-req':
                        log(f"Received summary request from {sender[:8]}...", self.peer_id, msg_type="summary-req")
                        sum_prompt = msg.get("message")
                        max_chars = msg.get("max_chars")
                        model_idx = msg.get("model_idx")
                        new_summary = self._summarize(sum_prompt, max_chars, model_idx=model_idx)
                        msg_id = str(uuid.uuid4())
                        reply = {
                            "msg_id":         msg_id,
                            "orig_msg_id":    msg.get("msg_id"),
                            "type":           "summary",
                            "from":           self.public_key,
                            "to":             msg.get("from"),
                            "summary":        new_summary
                        }
                        self.send(reply, msg.get("from"), timeout=timeout)

                    elif msg.get("type") == 'summary':
                        log(f"Received summary from {sender[:8]}...", self.peer_id, msg_type="summary")
                        orig = msg.get("orig_msg_id")
                        pending = self.awaiting_acks.pop(orig, None) if orig else None
                        if pending is not None:
                            log(f"Removed message ID {orig[:8]}... from awaiting ACKs.", self.peer_id, msg_type="summary-ack")

                        chat_id = pending.get("chat_id") if pending else None
                        chat = self.chats.get(chat_id) if chat_id else None
                        trimmed_pairs = self.tmp_trimmed_pairs.pop(chat_id, None) if chat_id else None

                        new_summary = msg.get("summary")
                        if new_summary and chat is not None:
                            self._handle_summary(new_summary, chat, trimmed_pairs or [])
                        else:
                            log("Remote summarization produced an empty result; keeping previous summary.", self.peer_id, msg_type="warning")

                        if chat_id and chat_id in self.pending_continuations:
                            state = self.pending_continuations.pop(chat_id)
                            threading.Thread(
                                target=self._continue_handle_user_input,
                                kwargs=state,
                                daemon=True,
                                name="continue-user-input",
                            ).start()
                
                    elif msg.get("type") in ("greeting", "greeting-ack"):

                        log(f"Received greeting from {sender[:8]}: {msg.get('message')}", self.peer_id, msg_type=msg.get("type"))

                        if sender not in self.known_public_keys and msg.get("peer_id") != self.peer_id:
                            log(f"New node discovered with ID = {msg.get('peer_id')}!", self.peer_id, msg_type="newnode")
                            self.known_public_keys[sender] = {}
                            self.known_public_keys[sender]["peer_id"] = msg.get("peer_id")
                            self.known_public_keys[sender]["peer_name"] = msg.get("peer_name")

                        if msg.get("type") == "greeting":
                            log(f"Sending greeting back to Node {msg.get('peer_id')}...", self.peer_id, msg_type="greeting-ack")
                            greetings = {
                                "msg_id": msg.get("msg_id"),
                                "type": "greeting-ack",
                                "from": self.public_key,
                                "peer_id": self.peer_id,
                                "peer_name": self.node_name,
                                "message": f"Hello from node {self.peer_id}!"
                            }
                            self.send(greetings, sender, timeout=timeout)
                    
                        if msg.get("type") == "greeting-ack":
                            pending = self.awaiting_acks.pop(msg.get("msg_id"), None)
                            if pending is not None:
                                comm_latency = time.time() - pending["timestamp"]
                                log(f"Received ACK for greeting from Node {msg.get('peer_id')}. Removing from awaiting ACKs.", self.peer_id, msg_type="greeting-ack")
                                if sender in self.known_public_keys:
                                    self.known_public_keys[sender].setdefault("comm_latencies", []).append(comm_latency)
                                log(f"Communication latency to peer {sender[:8]}... (Node {msg.get('peer_id')}): {comm_latency*1000:.2f} ms", self.peer_id, msg_type="greeting-ack")

                    elif msg.get("type") == "knowledge":

                        log(f"Received peers knowledge from {sender[:8]}. Num entries = {len(msg.get('known_peers', {}))}", self.peer_id, msg_type="knowledge")
                        if sender not in self.known_public_keys:
                            self.new_peer_discovered(msg, sender)

                        new_peers = 0
                        for pk, info in msg.get("known_peers", {}).items():
                            peer_id = info.get("peer_id")
                            if pk not in self.known_public_keys and ipv6_from_pubkey(pk) != ipv6_from_pubkey(self.public_key):
                                self.new_peer_discovered({"peer_id": peer_id}, pk)
                                new_peers += 1
                    
                        log(f"Updated known peers with knowledge from Node {msg.get('peer_id')}. New peers added: {new_peers}. Total known peers: {len(self.known_public_keys)}.", self.peer_id, msg_type="knowledge")
                    
                        knowledge_ack = {
                            "msg_id": msg.get("msg_id"),
                            "type": "knowledge-ack",
                            "from": self.public_key,
                            "peer_id": self.peer_id,
                            "peer_name": self.node_name
                        }
                        self.send(knowledge_ack, sender, timeout=timeout)
                
                    elif msg.get("type") == "info":

                        log(f"Received peers models and stats info from {sender[:8]}. Num models = {len(msg.get('models_info', {}))}, num stats = {len(msg.get('stats_info', {}))}", self.peer_id, msg_type="info")
                        if sender not in self.known_public_keys:
                            self.new_peer_discovered(msg, sender)
                    
                        # Preserve locally-tracked num_requests across info updates
                        new_models_info = msg.get("models_info", [])
                        old_models_info = self.known_public_keys[sender].get("models_info", [])
                        for i, new_mi in enumerate(new_models_info):
                            if i < len(old_models_info):
                                if "num_requests" in old_models_info[i]:
                                    new_mi["num_requests"] = old_models_info[i]["num_requests"]
                    
                        self.known_public_keys[sender]["models_info"] = new_models_info
                        self.known_public_keys[sender]["stats_info"] = msg.get("stats_info", [])
                    
                        for model_info in msg.get("models_info", []):
                            for tag in model_info.get("tags", []):
                                if tag not in self.known_tags:
                                    self.known_tags.append(tag)
                                    log(f"Discovered new tag '{tag}' from Node {msg.get('peer_id')}'s shared models info!", self.peer_id, msg_type="info")
                    
                        for stats_info in msg.get("stats_info", []):
                            for tag in stats_info.get("tags", []):
                                if tag not in self.known_tags:
                                    self.known_tags.append(tag)
                                    log(f"Discovered new tag '{tag}' from Node {msg.get('peer_id')}'s shared stats info!", self.peer_id, msg_type="info")

                        info_ack = {
                            "msg_id": msg.get("msg_id"),
                            "type": "info-ack",
                            "from": self.public_key,
                            "peer_id": self.peer_id,
                            "peer_name": self.node_name
                        }
                        self.send(info_ack, sender, timeout=timeout)
                
                    elif msg.get("type") == "stats-req":

                        tokenizer_name = msg.get("tokenizer_name")
                        embedder_name = msg.get("embedder_name")
                        log(f"Received stats request from {sender[:8]} for tokenizer {tokenizer_name} and embedder {embedder_name}.", self.peer_id, msg_type="stats-req")
                        if sender not in self.known_public_keys:
                            self.new_peer_discovered(msg, sender)

                        self.share_stats(msg.get("msg_id"), tokenizer_name, embedder_name, sender, timeout=timeout)

                    elif msg.get("type") == "knowledge-ack":
                        log(f"Received ACK for knowledge from Node {msg.get('peer_id')}.", self.peer_id, msg_type="knowledge-ack")
                        if msg.get("msg_id") in self.awaiting_acks:
                            log(f"Removing message ID {msg.get('msg_id')[:8]}... from awaiting ACKs.", self.peer_id, msg_type="knowledge-ack")
                            del self.awaiting_acks[msg.get("msg_id")]

                    elif msg.get("type") == "info-ack":
                        log(f"Received ACK for model and stats info from Node {msg.get('peer_id')}.", self.peer_id, msg_type="info-ack")
                        if msg.get("msg_id") in self.awaiting_acks:
                            log(f"Removing message ID {msg.get('msg_id')[:8]}... from awaiting ACKs.", self.peer_id, msg_type="info-ack")
                            del self.awaiting_acks[msg.get("msg_id")]

                    elif msg.get("type") == "stats-ack":
                        log(f"Received ACK for stats from Node {msg.get('peer_id')}.", self.peer_id, msg_type="stats-ack")
                        if msg.get("msg_id") in self.awaiting_acks:
                            log(f"Removing message ID {msg.get('msg_id')[:8]}... from awaiting ACKs.", self.peer_id, msg_type="stats-ack")
                            del self.awaiting_acks[msg.get("msg_id")]

                    else:
                        log(f"Received unknown message type from {sender[:8]}, type={msg.get('type')}.", self.peer_id, msg_type="warning")
                except Exception as e:
                    log(f"Error handling message type '{msg.get('type')}' from {sender[:8]}...: {e}", 
                        self.peer_id, msg_type="warning")

            time.sleep(0.5)

    def send_greetings(self, num_peers_to_greet=5, timeout=5):
        """Greet a random sample of known/topology peers to discover the
        network and measure comm latency. Called periodically from run.py's
        run_peer loop, on the --discover-peers-interval schedule."""

        log(f"Discovering peers in the network...", self.peer_id, msg_type=None)
        
        if self.topology['peers'] is None and len(self.known_public_keys) == 0:
            log(f"Attempted sending greetings to known peers, but no known public keys found.", self.peer_id, msg_type=None)
            return
        
        if self.topology['peers'] is not None:
            if all(pk['public_key'] == '' for pk in self.topology['peers']) and len(self.known_public_keys) == 0:
                log(f"Attempted sending greetings to known peers, but all have empty public keys, meaning they are offline.", self.peer_id, msg_type=None)
                return
        
        if self.topology['peers'] is not None:
            known_public_keys_from_topo = set([k['public_key'] for k in self.topology['peers']])
            if '' in known_public_keys_from_topo:
                known_public_keys_from_topo.remove('')
        else:
            known_public_keys_from_topo = set()

        known_public_keys = list(known_public_keys_from_topo | set(list(self.known_public_keys.keys())))
        random.shuffle(known_public_keys)
        known_public_keys = known_public_keys[:num_peers_to_greet]

        greetings_sent = 0
        
        for pk in known_public_keys:

            greetings = {
                "msg_id": str(uuid.uuid4()),
                "type": "greeting",
                "from": self.public_key,
                "peer_id": self.peer_id,
                "peer_name": self.node_name,
                "message": f"Hello from node {self.peer_id}!",
            }
            send_response =self.send(greetings, pk, timeout=timeout)
            if not send_response:
                continue

            self.awaiting_acks[greetings["msg_id"]] = {
                "type": "greeting",
                "receiver": pk,
                "timestamp": time.time()
            }
            greetings_sent += 1
        
        log(f"Sent greetings to {greetings_sent} known peers.", self.peer_id, msg_type='greeting')

    def share_knowledge(self, num_peers_to_share=5, timeout=5):
        """Gossip this peer's known-peers list to a random sample of them,
        so peer discovery propagates transitively. Called periodically from
        run.py's run_peer loop, on the --share-knowledge-interval schedule."""

        log(f"Sharing known peers with the network...", self.peer_id, msg_type='knowledge')
        known_pks = list(self.known_public_keys.keys())
        random.shuffle(known_pks)

        knowledge_shared = 0
        known_public_keys = {}
        for pk, info in self.known_public_keys.items():
            known_public_keys[pk] = {
                "peer_id": info.get("peer_id"),
                "peer_name": info.get("peer_name")
            }

        for pk in known_pks[:num_peers_to_share]:
            log(f"Sharing knowledge with peer {pk[:8]}, ID={self.known_public_keys[pk]['peer_id']}...", self.peer_id, msg_type='knowledge')
            knowledge_msg = {
                "msg_id": str(uuid.uuid4()),
                "type": "knowledge",
                "from": self.public_key,
                "peer_id": self.peer_id,
                "peer_name": self.node_name,
                "known_peers": known_public_keys
            }

            send_response = self.send(knowledge_msg, pk, timeout=timeout)
            if not send_response:
                continue

            self.awaiting_acks[knowledge_msg["msg_id"]] = {
                "type": "knowledge",
                "receiver": pk,
                "timestamp": time.time()
            }
            knowledge_shared += 1

        log(f"Shared knowledge with {knowledge_shared} peers.", self.peer_id, msg_type='knowledge')
    
    def check_acks(self, timeout=30):
        """Time out anything in awaiting_acks past `timeout` seconds (marking
        failed user messages, unblocking stalled summary continuations) and
        prune peers that have gone silent for too long. Called periodically
        from run.py's run_peer loop, on the --check-acks-interval schedule."""
        current_time = time.time()
        expired_acks = [msg_id for msg_id, info in self.awaiting_acks.items()
                        if current_time - info['timestamp'] > timeout]

        for msg_id in expired_acks:
            info = self.awaiting_acks[msg_id]
            pk = info['receiver']

            if pk in self.known_public_keys:
                peer_id = self.known_public_keys[pk]['peer_id']
                log(f"No ACK received from peer {pk[:8]}... with ID {peer_id} for "
                    f"message ID {msg_id[:8]}..., message type "
                    f"{info.get('type', 'unknown')} after {timeout} seconds.",
                    self.peer_id, msg_type="warning")
            else:
                log(f"No ACK received from peer {pk[:8]}... with unknown ID, for "
                    f"message ID {msg_id[:8]}..., message type "
                    f"{info.get('type', 'unknown')} after {timeout} seconds.",
                    self.peer_id, msg_type="warning")

            chat_id = info.get('chat_id')
            chat = self.chats.get(chat_id) if chat_id else None

            if info.get('type') == 'text':
                if chat is not None:
                    chats.mark_last_user_failed(chat)
                if chat_id:
                    self._mark_answer_resolved(chat_id)

            elif info.get('type') == 'summary-req':
                log("Summary request expired; proceeding with trimmed history (no summary).",
                    self.peer_id, msg_type="warning")
                if chat_id:
                    self.tmp_trimmed_pairs.pop(chat_id, None)
                    if chat_id in self.pending_continuations:
                        state = self.pending_continuations.pop(chat_id)
                        threading.Thread(
                            target=self._continue_handle_user_input,
                            kwargs=state,
                            daemon=True,
                            name="continue-user-input",
                        ).start()

            del self.awaiting_acks[msg_id]

        # Drop peers we haven't heard anything from in a while. last_seen is refreshed on
        # every inbound message (recv_loop), and a live peer gossips greetings/knowledge/info
        # regularly — so this only fires for peers that have genuinely gone silent.
        current_time = time.time()
        for pk in [p for p in self.known_public_keys
                   if current_time - self.last_seen.get(p, current_time) > PEER_DEAD_TIMEOUT]:
            self.prune_peer(pk)

    def prune_peer(self, pk):
        """Forget a peer we've stopped hearing from, and any state that references it."""
        info = self.known_public_keys.pop(pk, None)
        self.last_seen.pop(pk, None)
        self.chunks.pop(pk, None)
        for mid in [m for m, a in self.awaiting_acks.items() if a.get("receiver") == pk]:
            del self.awaiting_acks[mid]
        peer_id = info.get("peer_id") if info else "?"
        log(f"Pruned peer {pk[:8]}... (ID {peer_id}) — no contact for over {PEER_DEAD_TIMEOUT}s.",
            self.peer_id, msg_type="warning")

    def share_stats_and_models_info(self, num_peers_to_share=5, timeout=5):
        """Push this peer's model/stats metadata (tags, sizes, dataset names
        — not the stats matrices themselves) to a random sample of known
        peers. Called periodically from run.py's run_peer loop, on the
        --share-info-interval schedule."""
        log(f"Sharing IR3DE local stats and local models info with the network...", self.peer_id, msg_type='info')
        known_pks = list(self.known_public_keys.keys())
        random.shuffle(known_pks)

        info_shared = 0
        for pk in known_pks[:num_peers_to_share]:
            msg_id = str(uuid.uuid4())
            log(f"Sharing IR3DE local stats info and local models info with peer {pk[:8]}, ID={self.known_public_keys[pk]['peer_id']}...", self.peer_id, msg_type='info', msg_id=msg_id)
            info_msg = {
                "msg_id": msg_id,
                "type": "info",
                "from": self.public_key,
                "peer_id": self.peer_id,
                "peer_name": self.node_name,
                "models_info": [],
                "stats_info": []
            }
            for model_info in self.models_info:
                info_msg["models_info"].append({
                    "tags": model_info.get("tags"),
                    "tokenizer_name": model_info.get("tokenizer"),
                    "size": model_info.get("size"),
                    "type": model_info.get("type"),
                    "name": model_info.get("hf_name", "unknown"),
                })
            for stats in self.stats_info:
                info_msg["stats_info"].append({
                    "tags": stats.get("tags"),
                    "tokenizer_name": stats.get("tokenizer_name"),
                    "embedder_name": stats.get("embedder_name"),
                    "datasets_names": stats.get("datasets_names")
                })
            send_response = self.send(info_msg, pk, timeout=timeout)
            if not send_response:
                continue

            self.awaiting_acks[info_msg["msg_id"]] = {
                "type": "info",
                "receiver": pk,
                "timestamp": time.time()
            }
            info_shared += 1

        log(f"Shared IR3DE local stats info and local models info with {info_shared} peers.", self.peer_id, msg_type='info')
            

    def ask_stats(self, num_peers_to_ask=5, timeout=5):
        """Request the actual IR3DE stats matrices from known peers that
        advertised matching tokenizer/embedder stats we haven't asked for
        yet. Called periodically from run.py's run_peer loop, on the
        --share-stats-interval schedule; answered by share_stats on the
        receiving peer."""
        log(f"Asking for IR3DE stats from the network...", self.peer_id, msg_type='stats-req')
        known_pks = list(self.known_public_keys.keys())
        known_pks = [pk for pk in known_pks if "stats_info" in self.known_public_keys[pk] and len(self.known_public_keys[pk]["stats_info"]) > 0] # Filter only peers that have shared stats info, since asking for stats to peers that haven't shared stats info would be pointless. In the future, we might want to allow asking for stats even to peers that haven't shared stats info, in case they have the stats but just haven't shared them for some reason.
        valid_pks = []
        for pk in known_pks:
            stats_info = self.known_public_keys[pk]["stats_info"]
            if any(stats.get("tokenizer_name") == self.default_tokenizer_name and 
                   stats.get("embedder_name") == self.default_embedder_name for stats in stats_info) and \
                   "stats" not in self.known_public_keys[pk]: # Also check that we haven't already asked this peer for stats, since asking multiple times for stats to the same peer would be redundant and could be considered spamming. In the future, we might want to allow asking multiple times for stats to the same peer, in case they have updated stats or if we want to ask for stats with different tags or something like that.
                valid_pks.append(pk)
        if len(valid_pks) == 0:
            log(f"No known peers which shared stats info, never shared stats before and with stats info matching our default tokenizer and embedder. Skipping.", self.peer_id, msg_type='stats-req')
            return

        log(f"Found {len(valid_pks)} peers to ask for IR3DE stats. Asking from {min(len(valid_pks), num_peers_to_ask)} peers.", self.peer_id, msg_type='stats-req')
        random.shuffle(valid_pks)

        stats_asked = 0
        for pk in valid_pks[:num_peers_to_ask]:
            msg_id = str(uuid.uuid4())
            log(f"Asking for IR3DE stats from peer {pk[:8]}, ID={self.known_public_keys[pk]['peer_id']}...", self.peer_id, msg_type='stats-req', msg_id=msg_id)
            info_msg = {
                "msg_id": msg_id,
                "type": "stats-req",
                "from": self.public_key,
                "peer_id": self.peer_id,
                "peer_name": self.node_name,
                "tokenizer_name": self.default_tokenizer_name,
                "embedder_name": self.default_embedder_name
            }
            send_response = self.send(info_msg, pk, timeout=timeout)
            if not send_response:
                continue

            self.awaiting_acks[info_msg["msg_id"]] = {
                "type": "stats-req",
                "receiver": pk,
                "timestamp": time.time()
            }
            stats_asked += 1

        log(f"Asked for IR3DE stats from {stats_asked} peers.", self.peer_id, msg_type='stats-req')

    def share_stats(self, orig_msg_id, tokenizer_name, embedder_name, sender, timeout=5):
        """Reply to a "stats-req" with this peer's stats matching the
        requested tokenizer/embedder, chunked via send(..., large=True).
        Called from recv_loop's "stats-req" handler."""

        stats_to_share = []
        
        for stats_info, stats in zip(self.stats_info, self.stats):
            if stats_info['tokenizer_name'] == tokenizer_name and stats_info['embedder_name'] == embedder_name:
                stats_to_share.append({
                    "A": [A.cpu() for A in stats["A"]],
                    "b": [b.cpu() for b in stats["b"]],
                    "tags": stats["tags"],
                })
        msg_id = str(uuid.uuid4())
        log(f"Sharing stats with {sender[:8]}... in response to stats request ({orig_msg_id[:8]}...). Stats match found: {len(stats_to_share) > 0}.", self.peer_id, msg_type="stats-req", msg_id=msg_id)
        
        stats_msg = {
            "orig_msg_id": orig_msg_id,
            "msg_id": msg_id,
            "type": "stats",
            "from": self.public_key,
            "peer_id": self.peer_id,
            "peer_name": self.node_name,
            "stats": stats_to_share
        }
        send_response = self.send(stats_msg, sender, timeout=timeout, large=True)
        if not send_response:
            log(f"Failed to send stats response to {sender[:8]}... for the received stats request.", self.peer_id, msg_type="warning", msg_id=msg_id)
            return
        
        log(f"Shared stats with {sender[:8]}... in response to stats request. msg_id = {msg_id}", self.peer_id, msg_type="stats", msg_id=msg_id)

    def get_token_router(self):
        """Build the ridge-regression router (a Linear layer) from the
        combined local + known-peers IR3DE stats for the currently-selected
        tags. Called from handle_user_input before routing a message; its
        output feeds find_best_tag. Returns None (after logging why) if no
        stats are available yet for the selected tags."""

        identifier = (self.default_tokenizer_name, self.default_embedder_name)
        ref_stats = None
        for stats_info, stats in zip(self.stats_info, self.stats):
            if stats_info['tokenizer_name'] == self.default_tokenizer_name and stats_info['embedder_name'] == self.default_embedder_name:
                ref_stats = stats
                break
        if ref_stats is None:
            log(f"No reference stats found for tokenizer {self.default_tokenizer_name} and embedder {self.default_embedder_name}. Cannot handle user input.", self.peer_id, msg_type="warning")
            return
        
        with torch.no_grad():
            tokenizer = ref_stats['tokenizer']
            embedder = ref_stats['embedder'].to(self.device)
            known_tags = self.known_tags
            emb_dim = embedder.weight.shape[1]
            A = torch.zeros((emb_dim + 1, emb_dim + 1), dtype=torch.float32, device=self.device)
            b_dict = {}
            for pk in self.known_public_keys:
                if "stats" in self.known_public_keys[pk]:
                    for stats, stats_info in zip(self.known_public_keys[pk]["stats"], self.known_public_keys[pk]["stats_info"]):
                        if stats_info['tokenizer_name'] == self.default_tokenizer_name and stats_info['embedder_name'] == self.default_embedder_name:
                            for tag, A_peer, b_peer in zip(stats['tags'], stats['A'], stats['b']):
                                if tag not in known_tags:
                                    log(f"Received stats with unknown tag '{tag}' from peer {pk[:8]}... New tag found!", self.peer_id, msg_type="warning")
                                    self.known_tags.append(tag)
                                if tag in self.selected_tags:
                                    A_peer = A_peer.to(self.device)
                                    b_peer = b_peer.to(self.device)
                                    A += A_peer
                                    if tag not in b_dict:
                                        b_dict[tag] = b_peer
                                    else:
                                        b_dict[tag] += b_peer                                    
            
            for tag in self.local_A[identifier]:
                if tag in self.selected_tags:
                    A += self.local_A[identifier][tag].to(self.device)
                    if tag not in b_dict:
                        b_dict[tag] = self.local_b[identifier][tag].to(self.device)
                    else:
                        b_dict[tag] += self.local_b[identifier][tag].to(self.device)

            if not b_dict:
                if not self.selected_tags:
                    log("No expertise selected. Activate at least one in the Control Panel.",
                        self.peer_id, msg_type="warning")
                else:
                    log(f"No stats available yet for the selected tags "
                        f"{sorted(self.selected_tags)}. The local peer doesn't carry stats "
                        f"for these tags and no peer has shared matching stats yet.",
                        self.peer_id, msg_type="warning")
                return None

            b = torch.zeros((emb_dim + 1, len(b_dict)), dtype=torch.float32, device=self.device)
            for i, tag in enumerate(b_dict):
                b[:, i] = b_dict[tag]
            
            W = torch.linalg.solve(A + self.default_lambda * torch.eye(A.shape[0], device=A.device), b)
            bias = W[-1, :]
            W = W[:-1, :]
            norm = torch.norm(W, dim=0, keepdim=True)
            if torch.any(norm == 0.0):
                print("WARNING: 0 encountered in norm, substituting with 1e-6")
                norm[norm == 0.0] = 1e-6
            W = W / norm
            bias = bias / norm[0, :]

        router = torch.nn.Linear(W.shape[0], W.shape[1]).to(self.device)
        router.weight.data = W.T
        router.bias.data = bias.T

        return router, tokenizer, embedder, list(b_dict.keys())


    def find_best_tag(self, router, tokenizer, embedder, tags, user_input):
        """Embed the user's input, route it through `router`, and pick the
        tag with lowest-entropy (most confident) prediction across the
        input's tokens. Called from handle_user_input with get_token_router's
        output; its result feeds find_best_model."""
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(self.device)
        X = embedder(input_ids)
        batch_size = X.size(0)
        X = X.reshape(-1, X.size(-1))
        outputs = router(X)
        outputs = outputs.view(batch_size, -1, router.out_features)
        probs = F.softmax(outputs, dim=2)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=2)
        k = min(self.entropy_top_k, entropy.size(1))
        _, idx = torch.topk(entropy, k=k, largest=False, dim=1)
        mask = torch.zeros_like(entropy, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, router.out_features)
        outputs = outputs.gather(1, idx_expanded)
        predicted_tags = torch.argmax(outputs, dim=-1)
        assigned_tag = tags[predicted_tags.view(batch_size, -1).mode(dim=1)[0]]
        return assigned_tag


    def find_best_model(self, assigned_tag):
        """Resolve the user's chosen model for `assigned_tag` (set via the
        Control Panel's expertise sections, mirrored into
        self.selected_models) into a validated (peer_pk, model_idx), bumping
        its request counter. Called from handle_user_input after find_best_tag."""
        selected = self.selected_models.get(assigned_tag)
        if selected is None:
            log(f"No model selected for tag '{assigned_tag}'. Skipping.",
                self.peer_id, msg_type="warning")
            return None

        peer_pk, model_idx = selected
        is_local = (peer_pk == self.public_key)

        # Validate the selection still points at a live model
        if is_local:
            if model_idx >= len(self.models):
                log(f"Selected local model idx {model_idx} for tag '{assigned_tag}' is out of range. Skipping.", self.peer_id, msg_type="warning")
                return None
            target = self.models[model_idx]
        else:
            info = self.known_public_keys.get(peer_pk)
            if info is None:
                log(f"Selected peer {peer_pk[:8]}... for tag '{assigned_tag}' is no longer known. Skipping.", self.peer_id, msg_type="warning")
                return None
            models_info = info.get("models_info") or []
            if model_idx >= len(models_info):
                log(f"Selected remote model idx {model_idx} from peer {peer_pk[:8]}... for tag '{assigned_tag}' is out of range. Skipping.", self.peer_id, msg_type="warning")
                return None
            target = models_info[model_idx]

        if is_local:
            log(f"Using user-selected local model (idx {model_idx}) for tag '{assigned_tag}'.", self.peer_id, msg_type="ir3de")
        else:
            log(f"Using user-selected model (idx {model_idx}) from peer {peer_pk[:8]}... for tag '{assigned_tag}'.", self.peer_id, msg_type="ir3de")

        # Bump num_requests on whichever side hosts the chosen model
        target['num_requests'] = target.get('num_requests', 0) + 1

        return (peer_pk, model_idx)

    def _mark_answer_pending(self, chat_id) -> None:
        """Increment chat_id's in-flight-answer count. Called once at the
        top of handle_user_input, per dispatched message."""
        self.pending_answers[chat_id] = self.pending_answers.get(chat_id, 0) + 1

    def _mark_answer_resolved(self, chat_id) -> None:
        """Decrement chat_id's in-flight-answer count, dropping the entry at
        zero. Called at every terminal point a dispatched message can reach
        (local generation done, remote ack received, timeout, error)."""
        if chat_id not in self.pending_answers:
            return
        self.pending_answers[chat_id] -= 1
        if self.pending_answers[chat_id] <= 0:
            del self.pending_answers[chat_id]

    def is_awaiting_answer(self, chat_id) -> bool:
        """True iff a user message for this chat has been dispatched (locally
        or to a remote peer, possibly via a summary-continuation detour) and
        hasn't reached a terminal outcome yet. Authoritative for the UI's
        waiting indicator."""
        return self.pending_answers.get(chat_id, 0) > 0

    def handle_user_input(self, user_input, timeout=60):
        """Route a submitted message to an expert (skipping routing if only
        one model is selected overall) and dispatch it, deferring to a
        remote summary first if the conversation exceeds its character
        budget. Called from run.py's handle_input, on its own thread per
        submission."""
        chat = self.get_active_chat()
        if chat is None:
            log("No active chat. Cannot handle user input.", self.peer_id, msg_type="warning")
            return
        chat_id = chat["chat_id"]

        if chat_id in self.pending_continuations:
            log("This chat is still waiting for a remote summary. Please wait for it to complete before submitting another message.", self.peer_id, msg_type="warning")
            log("Waiting for previous response to complete...", self.peer_id, msg_type="warning", right=True, chat_id=chat_id)
            return

        # Marked once here, resolved exactly once at whichever terminal point
        # this request eventually reaches below (including inside
        # _continue_handle_user_input, possibly after a summary-continuation
        # detour) — see is_awaiting_answer().
        self._mark_answer_pending(chat_id)

        start_time = time.time()

        unique_models = set(self.selected_models.values())
        if len(unique_models) == 1:
            selected_model = next(iter(unique_models))
            msg = "Only one model is currently selected across all tags. Skipping token routing and dispatching directly to it."
            log(msg, self.peer_id, msg_type="warning")
            log(msg, self.peer_id, msg_type="warning", right=True, chat_id=chat_id)

            peer_pk, model_idx = selected_model
            if peer_pk == self.public_key:
                if model_idx < len(self.models):
                    target = self.models[model_idx]
                    target['num_requests'] = target.get('num_requests', 0) + 1
            else:
                info = self.known_public_keys.get(peer_pk)
                if info is not None:
                    models_info = info.get("models_info") or []
                    if model_idx < len(models_info):
                        target = models_info[model_idx]
                        target['num_requests'] = target.get('num_requests', 0) + 1

            assigned_tag = next(iter(self.selected_models.keys()), "(unrouted)")

        else:
            out = self.get_token_router()
            if out is None:
                log("Cannot handle user input because token router could not be constructed.",
                    self.peer_id, msg_type="warning")
                log("Cannot handle user input. Please select at least one expertise in the Control Panel.",
                    self.peer_id, msg_type="warning", right=True, chat_id=chat_id)
                self._mark_answer_resolved(chat_id)
                return

            assigned_tag = self.find_best_tag(*out, user_input)
            log(f"User input assigned to tag '{assigned_tag}'", self.peer_id, msg_type="ir3de")

            selected_model = self.find_best_model(assigned_tag)
            if selected_model is None:
                err = f"Cannot handle user input because no suitable model was found for the assigned tag '{assigned_tag}'."
                log(err, self.peer_id, msg_type="warning")
                log(err, self.peer_id, msg_type="warning", right=True, chat_id=chat_id)
                self._mark_answer_resolved(chat_id)
                return

        tag_for_msg = assigned_tag if assigned_tag and assigned_tag != "(unrouted)" else None
        chats.add_user_message(chat, user_input)

        needs_remote_summary = self._ensure_within_budget(chat_id, selected_model)

        if needs_remote_summary:
            self.pending_continuations[chat_id] = {
                "chat_id":        chat_id,
                "selected_model": selected_model,
                "assigned_tag":   assigned_tag,
                "tag_for_msg":    tag_for_msg,
                "start_time":     start_time,
                "timeout":        timeout,
            }
            log("Waiting for remote summary before generating the answer.",
                self.peer_id, msg_type="summary")
            return

        self._continue_handle_user_input(
            chat_id=chat_id,
            selected_model=selected_model,
            assigned_tag=assigned_tag,
            tag_for_msg=tag_for_msg,
            start_time=start_time,
            timeout=timeout,
        )

    def _continue_handle_user_input(self, chat_id, selected_model, assigned_tag, tag_for_msg, start_time, timeout=60):
        """Part B of handle_user_input: build the final prompt and dispatch the
        user's request to the expert. Invoked either inline by handle_user_input
        (no remote summary needed / local summary done) or by a worker thread
        spawned from recv_loop's 'summary' handler / check_acks on summary-req
        expiry.
        """
        
        chat = self.chats.get(chat_id)
        if chat is None:
            log(f"Cannot continue user input: chat {chat_id} not found.",
                self.peer_id, msg_type="warning")
            self._mark_answer_resolved(chat_id)
            return
        # Resolved in the finally below on every exit path except the one
        # where the remote send actually succeeds — that stays pending until
        # recv_loop's 'answer' handler or check_acks' timeout resolves it.
        resolved_later = False
        try:
            prompt = format_prompt_for_expert(chat)

            if ipv6_from_pubkey(selected_model[0]) == ipv6_from_pubkey(self.public_key):

                future = self.model_executor.submit(model_worker.generate, selected_model[1], self.models_info[selected_model[1]], prompt, self.max_answer_length)

                try:
                    answer, num_input_tokens, num_output_tokens, loaded_now, evicted = future.result(timeout=timeout)
                    self._log_model_load_events(selected_model[1], loaded_now, evicted)
                except FuturesTimeoutError:
                    err = f"Generation timed out after {timeout}s."
                    log(err, self.peer_id, msg_type="warning")
                    log(err, self.peer_id, msg_type="warning", right=True, chat_id=chat_id)
                    chats.mark_last_user_failed(chat)
                    return
                except Exception as e:
                    err = f"Generation failed: {e}"
                    log(err, self.peer_id, msg_type="warning")
                    log(err, self.peer_id, msg_type="warning", right=True, chat_id=chat_id)
                    chats.mark_last_user_failed(chat)
                    return

                log(f"Answer processed locally.", self.peer_id, msg_type="text")
                log(f"{answer}", self.peer_id, msg_type="text", right=True, chat_id=chat_id)

                prompt_latency = time.time() - start_time
                num_total_tokens = num_input_tokens + num_output_tokens
                latency_per_total = prompt_latency / num_total_tokens
                latency_per_input = prompt_latency / num_input_tokens

                local_model = self.models[selected_model[1]]
                local_model.setdefault('latencies', []).append({
                    "prompt_latency":          prompt_latency,
                    "num_input_tokens":        num_input_tokens,
                    "num_output_tokens":       num_output_tokens,
                    "latency_per_total_token": latency_per_total,
                    "latency_per_input_token": latency_per_input,
                })

                log(f"Prompt latency (local model idx {selected_model[1]}): {prompt_latency:.4f} s "
                    f"[in={num_input_tokens} tok, out={num_output_tokens} tok]",
                    self.peer_id, msg_type="text")
                log(f"Latency per total token: {latency_per_total*1000:.2f} ms/tok "
                    f"({num_total_tokens} tokens total)", self.peer_id, msg_type="text")
                log(f"Latency per input token: {latency_per_input*1000:.2f} ms/tok",
                    self.peer_id, msg_type="text")

                chats.add_agent_message(
                    chat,
                    answer,
                    peer_pk=self.public_key,
                    model_idx=selected_model[1],
                    tag=tag_for_msg,
                )

                if chat.get("title") is None:
                    msgs = chat.get("messages", [])
                    user_msg  = next((m for m in msgs if m["role"] == "user"  and m.get("status", "ok") == "ok"), None)
                    agent_msg = next((m for m in msgs if m["role"] == "agent" and m.get("status", "ok") == "ok"), None)
                    if user_msg is not None and agent_msg is not None:
                        title = self._generate_title(user_msg["text"], agent_msg["text"], selected_model[1])
                        if title:
                            chats.set_title(chat, title)
                            log(f"Chat title set: '{title}'", self.peer_id, msg_type="text")
                return

            # Remote path
            msg_id = str(uuid.uuid4())
            msg = {
                "msg_id":         msg_id,
                "type":           "text",
                "from":           self.public_key,
                "peer_id":        self.peer_id,
                "peer_name":      self.node_name,
                "assigned_tag":   assigned_tag,
                "selected_model": selected_model,
                "message":        prompt,
                "request_title":  chat.get("title") is None,
            }
            sent = self.send(msg, selected_model[0], timeout=timeout)
            if not sent:
                log(f"Could not reach the peer hosting your selected model ({selected_model[0][:8]}...). "
                    f"It will be deselected automatically if it stays offline; please try again.",
                    self.peer_id, msg_type="warning", right=True, chat_id=chat_id)
                chats.mark_last_user_failed(chat)
                return

            self.awaiting_acks[msg_id] = {
                "type":       "text",
                "chat_id":    chat_id,
                "model_idx":  selected_model[1],
                "receiver":   selected_model[0],
                "tag":        tag_for_msg,
                "timestamp":  time.time(),
            }
            resolved_later = True

        except Exception as e:
            log(f"Error in _continue_handle_user_input: {e}", self.peer_id, msg_type="warning")
        finally:
            if not resolved_later:
                self._mark_answer_resolved(chat_id)

    def stop_node(self, timeout=5):
        """Stop the AXL node subprocess deterministically and idempotently.

        SIGTERM the node's process group, wait, then SIGKILL if it's still alive.
        Group-aware (the node is started with start_new_session=True) with a
        single-pid fallback. Safe to call multiple times or if the node is already
        gone. Does not rely on the node handling SIGTERM gracefully — the SIGKILL
        fallback covers an unresponsive node.
        """
        proc = getattr(self, "proc", None)
        if proc is None or proc.poll() is not None:
            return

        pid = proc.pid

        def _signal(sig):
            """Send `sig` to the node's whole process group, falling back
            to just its own pid if the group is gone or inaccessible."""
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.send_signal(sig)
                except ProcessLookupError:
                    pass

        _signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            log(f"Node {getattr(self, 'peer_id', '?')} did not exit after SIGTERM; sending SIGKILL.",
                getattr(self, "peer_id", "?"), msg_type="warning")

        _signal(signal.SIGKILL)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def __del__(self):
        """Best-effort fallback node cleanup if this Peer is garbage
        collected without an interpreter exit (stop_node is also registered
        via atexit in __init__, which is the primary cleanup path)."""
        try:
            self.stop_node()
        except Exception:
            pass
          
    def _save_all_chats_safely(self) -> None:
        """Persist every non-empty in-memory chat on process exit. Empty
        chats are never persisted — their disk files are removed if they
        were left over from before this policy was in place."""
        for chat in list(self.chats.values()):
            chat_id = chat.get("chat_id")
            if not chat_id:
                continue
            if chat.get("messages"):
                chats.save_chat(chat)
            else:
                chats.delete_chat(chat_id)

    def _text_to_summarize(self, existing_summary, trimmed_pairs, max_chars):
        """Build the summarizer prompt from the trimmed (user, agent) pairs
        and any existing summary. Called from _ensure_within_budget."""
        parts = [SUMMARY_SYSTEM_PROMPT.format(max_chars=max_chars), ""]
        if existing_summary:
            parts.append("Existing summary so far:")
            parts.append(existing_summary)
            parts.append("")
            parts.append("Additional turns to fold into it:")
        else:
            parts.append("Conversation to summarize:")
        for user_msg, agent_msg in trimmed_pairs:
            parts.append(f"User: {user_msg['text']}")
            parts.append(f"Agent: {agent_msg['text']}")
        parts.append("")
        parts.append("Summary:")
        sum_prompt = "\n".join(parts)

        return sum_prompt

    def _summarize(self, sum_prompt, max_chars, model_idx=0):
        """Run the summarizer prompt through a local model and return the
        result truncated to max_chars (or "" on failure). Called from
        _ensure_within_budget (local path) and recv_loop's "summary-req"
        handler (when this peer is asked to summarize for a remote peer)."""
        max_tokens = max(20, min(self.max_answer_length, max_chars // 3 + 10))
        try:
            future = self.model_executor.submit(model_worker.generate, model_idx, self.models_info[model_idx], sum_prompt, max_tokens)
            summary, _, _, loaded_now, evicted = future.result(timeout=120)
            self._log_model_load_events(model_idx, loaded_now, evicted)
        except Exception as e:
            log(f"Summarization failed: {e}", self.peer_id, msg_type="warning")
            return ""
        return summary.strip()[:max_chars]

    def _handle_summary(self, new_summary, chat, trimmed_pairs):
        """Store a freshly-produced summary on `chat`. Called once a summary
        is available, whether generated locally (_ensure_within_budget) or
        received from a remote peer (recv_loop's "summary" handler)."""
        chats.set_summary(chat, new_summary)
        log(f"Summary updated ({len(new_summary)} chars covering {len(trimmed_pairs)} dropped pairs).", self.peer_id, msg_type="summary")

    def _ensure_within_budget(self, chat_id, selected_model, timeout=60) -> bool:
        """Trim oldest user/agent couples and summarize them via the routed
        expert if the prompt exceeds self.budget_chars.

        Returns True iff a *remote* summary request was just sent and the caller
        must defer subsequent work until the summary arrives. Returns False in
        all other cases (no trim needed, no complete pair to trim, local summary
        completed inline, remote send failed).
        """

        if self.budget_chars <= 0:
            return False

        chat = self.chats.get(chat_id)
        if chat is None:
            return False

        current_size = len(format_prompt_for_expert(chat))
        if current_size <= self.budget_chars:
            return False

        log(f"Conversation history ({current_size} chars) exceeds budget "
            f"({self.budget_chars}). Trimming and summarizing.",
            self.peer_id, msg_type="summary")

        target = int(0.8 * self.budget_chars)
        trimmed_pairs = []
        while len(format_prompt_for_expert(chat)) > target:
            pair = chats.trim_oldest_pair(chat)
            if pair is None:
                break
            trimmed_pairs.append(pair)

        if not trimmed_pairs:
            log("Budget exceeded but no complete (user, agent) couple at the "
                "head of the history to drop. Sending unchanged.",
                self.peer_id, msg_type="warning")
            return False

        room = int(0.8 * (self.budget_chars - len(format_prompt_for_expert(chat))))
        sum_prompt = self._text_to_summarize(chat.get("summary"), trimmed_pairs, room)

        if ipv6_from_pubkey(selected_model[0]) == ipv6_from_pubkey(self.public_key):
            new_summary = self._summarize(sum_prompt, room, model_idx=selected_model[1])
            if new_summary:
                self._handle_summary(new_summary, chat, trimmed_pairs)
            else:
                log("Local summarization produced an empty result; keeping previous summary.",
                    self.peer_id, msg_type="warning")
            return False

        # Remote path
        self.tmp_trimmed_pairs[chat_id] = trimmed_pairs                # ← per-chat
        msg_id = str(uuid.uuid4())
        msg = {
            "msg_id":    msg_id,
            "type":      "summary-req",
            "from":      self.public_key,
            "peer_id":   self.peer_id,
            "peer_name": self.node_name,
            "model_idx": selected_model[1],
            "message":   sum_prompt,
            "max_chars": room,
        }
        if not self.send(msg, selected_model[0], timeout=timeout):
            log("Failed to send summary-req; proceeding with trimmed history (no summary).",
                self.peer_id, msg_type="warning")
            self.tmp_trimmed_pairs.pop(chat_id, None)
            return False

        self.awaiting_acks[msg_id] = {
            "type":      "summary-req",
            "chat_id":   chat_id,
            "model_idx": selected_model[1],
            "receiver":  selected_model[0],
            "timestamp": time.time(),
        }
        return True

    def _clean_title(self, title):
        """Strip a raw model-generated title down to one line, no quotes,
        at most 6 words and MAX_TITLE_CHARS characters. Called from
        _generate_title and recv_loop's "answer" handler (for titles
        generated by a remote expert)."""
        if not title:
            return ""
        title = title.strip().split("\n", 1)[0].strip().strip('"\'').strip()
        words = title.split()
        if len(words) > 6:
            title = " ".join(words[:6])
        return title[:MAX_TITLE_CHARS]


    def _generate_title(self, user_text, agent_text, model_idx):
        """Generate a chat title from a user/agent exchange using a local model.
        Returns the cleaned title string, or '' on failure. Synchronous."""
        if model_idx >= len(self.models):
            return ""
        title_prompt = (
            f"{TITLE_SYSTEM_PROMPT}\n\n"
            f"User: {user_text}\n"
            f"Agent: {agent_text}\n\n"
            f"Title:"
        )
        try:
            future = self.model_executor.submit(model_worker.generate, model_idx, self.models_info[model_idx], title_prompt, 20)
            title, _, _, loaded_now, evicted = future.result(timeout=60)
            self._log_model_load_events(model_idx, loaded_now, evicted)
        except Exception as e:
            log(f"Title generation failed: {e}", self.peer_id, msg_type="warning")
            return ""
        return self._clean_title(title)

    def new_chat(self) -> str:
        """Create a new empty chat owned by this peer, register it, and return
        its chat_id."""
        chat = chats.new_chat(peer_id=self.peer_id, peer_name=self.node_name)
        cid = chat["chat_id"]
        self.chats[cid] = chat
        if self.active_chat_id is None:
            self.active_chat_id = cid
        return cid

    def get_active_chat(self) -> dict | None:
        """The chat the UI currently has selected, or None if none is active."""
        if self.active_chat_id is None:
            return None
        return self.chats.get(self.active_chat_id)

    def get_chat(self, chat_id: str) -> dict | None:
        """Look up any of this peer's chats by id, active or not."""
        return self.chats.get(chat_id)

    def delete_chat(self, chat_id: str) -> bool:
        """Remove a chat from memory, clear its in-flight state, and erase its
        on-disk file. Returns True if the chat existed. If the active chat was
        deleted, picks an arbitrary remaining chat (or None) as active."""
        if chat_id not in self.chats:
            return False
        del self.chats[chat_id]
        self.tmp_trimmed_pairs.pop(chat_id, None)
        self.pending_continuations.pop(chat_id, None)
        chats.delete_chat(chat_id)
        if self.active_chat_id == chat_id:
            self.active_chat_id = next(iter(self.chats), None)
        return True

    @property
    def current_chat(self) -> dict | None:
        """Backward-compat shim for code that hasn't been ported off the
        single-chat API. Resolves to the active chat. Remove when step 5.b
        finishes wiring the UI to address chats by id."""
        return self.get_active_chat()

    def close_chat(self, chat_id: str) -> None:
        """Mark a chat as closed (i.e., no longer displayed as a tab). The
        chat stays in self.chats so it can be reopened later without hitting
        disk again. Non-empty chats are flushed defensively; empty chats are
        never persisted (and are removed from disk if they were previously
        saved)."""
        if chat_id not in self.chats:
            return
        chat = self.chats[chat_id]
        chats.set_ui_open(chat, False)
        if chat.get("messages"):
            chats.save_chat(chat)
        else:
            chats.delete_chat(chat_id)

        if self.active_chat_id == chat_id:
            self.active_chat_id = None