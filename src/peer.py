from copy import deepcopy
from datetime import datetime
import os, pathlib, subprocess, uuid, requests, json, time, random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from ir3de_stats.models.llama_experts import get_llama_expert
from utils import ipv6_from_pubkey, log, serialize_safe, deserialize_safe, deserialize_chunk_header, redirect_prints

AXL = "http://127.0.0.1:91"

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



class Peer:

    def __init__(self, peer_id, ir3de_lambda=0.01, ir3de_entropy_top_k=10, max_answer_length=256):
        
        with open(f"ir3de_stats/default_stats.json", "r") as f:
            default_stats = json.load(f)

        with open(f"local_nodes/metadata{peer_id:02d}.json", "r") as f:
            metadata = json.load(f)
            self.node_name = metadata.get("node_name", f"Node {peer_id}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pathlib.Path("axl-logs").mkdir(exist_ok=True)
        log_fp = open(f"axl-logs/node-{peer_id:02d}.log", "w", buffering=1)
        self.proc = subprocess.Popen(
            ["./scripts/start_node.sh", str(peer_id)],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        sleep_time = 5
        log(f"Waiting {sleep_time} seconds for node {peer_id} to initialize...", peer_id, msg_type=None)
        time.sleep(sleep_time)

        self.peer_id = peer_id
        self.session = requests.Session()
        self.topology = self.get_topology(self.session)
        self.public_key = self.topology['our_public_key']
        self.ipv6_address = self.topology['our_ipv6']
        self.known_public_keys = {}
        self.awaiting_acks = {}

        self.known_tags = []
        self.models_info = metadata.get("models", [])
        self.models = self.get_models_info()
        self.stats_info = default_stats.get("stats", []) + metadata.get("stats", [])
        self.stats = self.get_stats_info()

        # FUTURE WORK: allow the user for chosing the desired tokenizer and embedder with the UI. Using always the default for now.
        self.default_tokenizer_name = "meta-llama/Meta-Llama-3-8B"
        self.default_embedder_name = "meta-llama/Meta-Llama-3-8B"
        self.default_lambda = ir3de_lambda
        self.entropy_top_k = ir3de_entropy_top_k
        self.max_answer_length = max_answer_length

        self.local_A = {}
        self.local_b = {}
        for stats_info, stats in zip(self.stats_info, self.stats):
            identifier = (stats_info['tokenizer_name'], stats_info['embedder_name'])
            self.local_A[identifier] = {}
            self.local_b[identifier] = {}
            for tag, A, b in zip(stats['tags'], stats['A'], stats['b']):
                self.local_A[identifier][tag] = A
                self.local_b[identifier][tag] = b

        self.chunks = {}
        self.conversation_histories: dict = {}

        self.generation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
    
    def get_topology(self, session):
        resp = session.get(f"{AXL}{self.peer_id:02d}/topology", timeout=5)
        resp.raise_for_status()
        topology = resp.json()
        return topology

    def get_models_info(self):
        models = []
        for model_info in self.models_info:
            if "path" in model_info: # In the future, we might want to allow only hf models
                model, _, _ = get_llama_expert(1.15e8)
                if os.path.isfile(model_info["path"]):
                    log(f"Loading expert model from {model_info['path']}", self.peer_id, msg_type=None)
                    state = torch.load(model_info["path"], map_location='cpu')
                else:
                    raise FileNotFoundError(f"Checkpoint path not found: {model_info['path']}")
                model.load_state_dict(state, strict=False)
                model.to(self.device)
            elif "hf_name" in model_info:
                log(f"Loading expert model from {model_info['hf_name']}", self.peer_id, msg_type=None)
                model = redirect_prints(AutoModelForCausalLM.from_pretrained, model_info["hf_name"])
                model.to(self.device)  # type: ignore
            else:
                raise ValueError(f"Model info for node {self.peer_id} must contain either 'path' or 'hf_name'. Provided info: {model_info}. Check the metadata{self.peer_id:02d}.json file.")

            model_tags = model_info.get("tags", [])
            for tag in model_tags:
                if tag not in self.known_tags:
                    self.known_tags.append(tag)

            tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer"])
            models.append(
                {
                    "model": model,
                    "tokenizer": tokenizer,
                    "tags": model_tags
                }
            )
            model_info["size"] = sum(p.numel() for p in model.parameters())
            model_info["type"] = getattr(model.config, "model_type", "unknown")
        return models

    def get_stats_info(self):
        all_stats = []
        for i, stats in enumerate(self.stats_info):
            stats_data = torch.load(stats["path"], map_location='cpu')
            tokenizer = AutoTokenizer.from_pretrained(stats_data["tokenizer"])
            model = AutoModelForCausalLM.from_pretrained(stats_data["embedder"])
            embedder = deepcopy(model.model.embed_tokens).to(torch.float32)
            self.stats_info[i]['tokenizer_name'] = stats_data["tokenizer"]
            self.stats_info[i]['embedder_name'] = stats_data["embedder"]
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
        return all_stats

    def send(self, message, peer_public_key, timeout=5, large=False):
        try:
            if not large:
                requests.post(
                    f"{AXL}{self.peer_id:02d}/send",
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
                    requests.post(
                        f"{AXL}{self.peer_id:02d}/send",
                        headers={"X-Destination-Peer-Id": peer_public_key},
                        data=chunk,
                        timeout=timeout
                    )
                    self.awaiting_acks[chunks_msg_ids[i]] = {
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
        log(f"New node discovered with ID = {msg.get('peer_id')}!", self.peer_id, msg_type="newnode")
        self.known_public_keys[sender] = {}
        self.known_public_keys[sender]["peer_id"] = msg.get("peer_id")
        self.known_public_keys[sender]["peer_name"] = msg.get("peer_name")

    def recv_loop(self, timeout=120):
        
        while True:
            
            resp = requests.get(f"{AXL}{self.peer_id:02d}/recv")

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
                    continue
            
            elif resp.status_code == 200:
                
                sender = resp.headers.get("X-From-Peer-Id")
                if sender is None:
                    log(f"Received message without sender information. Ignoring. Message content (truncated): {str(resp.text)[:100]}...", self.peer_id, msg_type="warning")
                    continue
                if resp.text is None or resp.text == "":
                    log(f"Received empty message from {sender[:8]}.... Ignoring.", self.peer_id, msg_type="warning")
                    continue
                try:
                    msg = json.loads(resp.text)
                except json.JSONDecodeError:
                    log(f"Failed to decode JSON message from {sender[:8]}.... Ignoring. Message content (truncated): {str(resp.text)[:100]}...", self.peer_id, msg_type="warning")
                    continue
                except Exception as e:
                    log(f"Unexpected error when decoding message from {sender[:8]}: {e}. Ignoring. Message content (truncated): {str(resp.text)[:100]}...", self.peer_id, msg_type="warning")
                    continue
                
                if msg.get("type") == "text":

                    message = msg.get('message')
                    log(f"From {sender[:8]}...: {message}", self.peer_id, msg_type="text")

                    model_utils = self.models[msg['selected_model'][1]]
                    incoming_messages = msg.get("messages")
                    prompt = self._format_chat_prompt(incoming_messages) if incoming_messages else None
                    answer = self.generate_answer(message, model_utils, prompt=prompt)
                    log(f"To {sender[:8]}...: {answer}", self.peer_id, msg_type="text")

                    msg_id = str(uuid.uuid4())
                    answer_msg = {
                        "orig_msg_id": msg.get("msg_id"),
                        "msg_id": msg_id,
                        "type": "answer",
                        "from": self.public_key,
                        "peer_id": self.peer_id,
                        "peer_name": self.node_name,
                        "message": answer,
                        "selected_model": msg.get("selected_model"),
                    }
                    self.send(answer_msg, sender, timeout=timeout)
                
                elif msg.get("type") == "answer":

                    log(f"Received answer from {sender[:8]}...", self.peer_id, msg_type="text")
                    log(f"{msg.get('message')}", msg.get("peer_id"), msg_type="text", right=True)

                    orig_msg_id = msg.get("orig_msg_id")
                    if orig_msg_id in self.awaiting_acks:
                        cb_info = self.awaiting_acks[orig_msg_id]
                        chat_id = cb_info.get("chat_id")
                        answer_text = msg.get("message")
                        peer_name = msg.get("peer_name", "agent")
                        if chat_id and chat_id in self.conversation_histories:
                            self._append_turn(self.conversation_histories[chat_id], "agent", peer_name, answer_text)
                        if cb_info.get("answer_callback") is not None:
                            cb_info["answer_callback"](answer_text, peer_name)
                        log(f"Removing message ID {orig_msg_id[:8]}... from awaiting ACKs.", self.peer_id, msg_type="ir3de-ack")
                        del self.awaiting_acks[orig_msg_id]

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
                    
                    if msg.get("type") == "greeting-ack" and msg.get("msg_id") in self.awaiting_acks:
                        log(f"Received ACK for greeting from Node {msg.get('peer_id')}. Removing from awaiting ACKs.", self.peer_id, msg_type="greeting-ack")
                        del self.awaiting_acks[msg.get("msg_id")]

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
                    log(f"WARNING: Received unknown message type from {sender[:8]}, type={msg.get('type')}.", self.peer_id, msg_type="warning")

            time.sleep(0.2)

    def send_greetings(self, num_peers_to_greet=5, timeout=5):
        
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
            
            if pk in self.known_public_keys:
                continue

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
                "receiver": pk,
                "timestamp": time.time()
            }
            greetings_sent += 1
        
        log(f"Sent greetings to {greetings_sent} known peers.", self.peer_id, msg_type='greeting')

    def share_knowledge(self, num_peers_to_share=5, timeout=5):
        
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
                "receiver": pk,
                "timestamp": time.time()
            }
            knowledge_shared += 1

        log(f"Shared knowledge with {knowledge_shared} peers.", self.peer_id, msg_type='knowledge')
    
    def check_acks(self, timeout=30):
        current_time = time.time()
        expired_acks = [msg_id for msg_id, info in self.awaiting_acks.items() if current_time - info['timestamp'] > timeout]
        for msg_id in expired_acks:
            pk = self.awaiting_acks[msg_id]['receiver']
            if pk in self.known_public_keys:
                peer_id = self.known_public_keys[self.awaiting_acks[msg_id]['receiver']]['peer_id']
                log(f"No ACK received from peer {pk[:8]}... with ID {peer_id} for message ID {msg_id[:8]}..., message type {self.awaiting_acks[msg_id].get('type', 'unknown')} after {timeout} seconds.", self.peer_id, msg_type="warning")
            else:
                log(f"No ACK received from peer {pk[:8]}... with unknown ID, for message ID {msg_id[:8]}..., message type {self.awaiting_acks[msg_id].get('type', 'unknown')} after {timeout} seconds.", self.peer_id, msg_type="warning")
            del self.awaiting_acks[msg_id]

    def share_stats_and_models_info(self, num_peers_to_share=5, timeout=5):

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
                    "type": model_info.get("type")
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
                "receiver": pk,
                "timestamp": time.time()
            }
            info_shared += 1

        log(f"Shared IR3DE local stats info and local models info with {info_shared} peers.", self.peer_id, msg_type='info')
            

    def ask_stats(self, num_peers_to_ask=5, timeout=5):

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
                "receiver": pk,
                "timestamp": time.time()
            }
            stats_asked += 1

        log(f"Asked for IR3DE stats from {stats_asked} peers.", self.peer_id, msg_type='stats-req')

    def share_stats(self, orig_msg_id, tokenizer_name, embedder_name, sender, timeout=5):
        
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
        self.awaiting_acks[msg_id] = {
            "receiver": sender,
            "timestamp": time.time()
        }

    def get_token_router(self):
        
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
                                if tag in known_tags:
                                    A_peer = A_peer.to(self.device)
                                    b_peer = b_peer.to(self.device)
                                    A += A_peer
                                    if tag not in b_dict:
                                        b_dict[tag] = b_peer
                                    else:
                                        b_dict[tag] += b_peer
                                else:
                                    log(f"Received stats with unknown tag '{tag}' from peer {pk[:8]}... New tag found!", self.peer_id, msg_type="warning")
                                    self.known_tags.append(tag)
            
            for tag in self.local_A[identifier]:
                A += self.local_A[identifier][tag].to(self.device)
                if tag not in b_dict:
                    b_dict[tag] = self.local_b[identifier][tag].to(self.device)
                else:
                    b_dict[tag] += self.local_b[identifier][tag].to(self.device)

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
        # expand indices to match last dim
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, router.out_features)     # (16, 10, 5)
        # gather along dim=1
        outputs = outputs.gather(1, idx_expanded)                             # (16, 10, 5)
        predicted_tags = torch.argmax(outputs, dim=-1)
        assigned_tag = tags[predicted_tags.view(batch_size, -1).mode(dim=1)[0]]
        return assigned_tag

    def find_best_model(self, assigned_tag, user_input):
        valid_peers = []
        for pk in self.known_public_keys:
            if "models_info" in self.known_public_keys[pk]:
                for i, model_info in enumerate(self.known_public_keys[pk]["models_info"]):
                    if assigned_tag in model_info.get("tags"):
                        valid_peers.append((pk, i))  # using index to identify which model to use from that peer for now
        for i, model_info in enumerate(self.models_info):
            if assigned_tag in model_info.get("tags"):
                valid_peers.append((None, i))  # None indicates local model
        
        local_models = [i for v, i in valid_peers if v is None]
        if len(local_models) > 0:
            log(f"Local model with tag '{assigned_tag}' found. Using the local model to process the input.", self.peer_id, msg_type="ir3de")
            random.shuffle(local_models)
            selected_model = (self.public_key, local_models[0])
            if 'num_requests' not in self.models[selected_model[1]]:
                self.models[selected_model[1]]['num_requests'] = 0
            self.models[selected_model[1]]['num_requests'] += 1
            return selected_model

        if len(valid_peers) == 0:
            log(f"No known peers with models matching the assigned tag '{assigned_tag}' found. Handling user input with a random local model.", self.peer_id, msg_type="warning")
            all_local_models_indices = list(range(len(self.models_info)))
            random.shuffle(all_local_models_indices)
            selected_model = (self.public_key, all_local_models_indices[0])
            log(f"Selected local model {self.models_info[selected_model[1]]['path']} to handle the user input.", self.peer_id, msg_type="ir3de")
            return selected_model
        
        random.shuffle(valid_peers)
        selected_peer = valid_peers[0]
        log(f"Selected peer {selected_peer[0][:8]}... with model index {selected_peer[1]} to handle the user input.", self.peer_id, msg_type="ir3de")
        
        if 'num_requests' not in self.known_public_keys[selected_peer[0]]["models_info"][selected_peer[1]]:
            self.known_public_keys[selected_peer[0]]["models_info"][selected_peer[1]]['num_requests'] = 0
        self.known_public_keys[selected_peer[0]]["models_info"][selected_peer[1]]['num_requests'] += 1

        return selected_peer


    def _new_conversation(self, chat_id: str) -> dict:
        return {"conversation_id": chat_id, "history": []}

    def _append_turn(self, conversation: dict, speaker_type: str, speaker_id: str, content: str):
        turn = len(conversation["history"]) + 1
        conversation["history"].append({
            "id": str(uuid.uuid4()),
            "turn": turn,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "speaker_type": speaker_type,
            "speaker_id": speaker_id,
            "content": content,
        })

    def _render_messages(self, conversation: dict) -> list:
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

    def _format_chat_prompt(self, messages: list) -> str:
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(f"System: {m['content']}")
            elif m["role"] == "user":
                parts.append(f"User: {m['content']}")
            else:
                parts.append(f"Assistant: {m['content']}")
        return "\n\n".join(parts) + "\n\nAssistant: "

    def handle_user_input(self, user_input, chat_id="default", answer_callback=None, timeout=60):

        out = self.get_token_router()
        if out is None:
            log(f"Cannot handle user input because token router could not be constructed.", self.peer_id, msg_type="warning")
            return

        assigned_tag = self.find_best_tag(*out, user_input)
        log(f"User input assigned to tag '{assigned_tag}'", self.peer_id, msg_type="ir3de")

        selected_model = self.find_best_model(assigned_tag, user_input)
        if selected_model is None:
            log(f"Cannot handle user input because no suitable model was found for the assigned tag '{assigned_tag}'.", self.peer_id, msg_type="warning")
            return

        conv = self.conversation_histories.setdefault(chat_id, self._new_conversation(chat_id))
        self._append_turn(conv, "user", "user", user_input)
        messages = self._render_messages(conv)
        prompt = self._format_chat_prompt(messages)

        if ipv6_from_pubkey(selected_model[0]) == ipv6_from_pubkey(self.public_key):
            model_utils = self.models[selected_model[1]]

            future = self.generation_executor.submit(self.generate_answer, user_input, model_utils, is_local=True, prompt=prompt)

            try:
                answer = future.result(timeout=timeout)
            except FuturesTimeoutError:
                log(f"Generation timed out after {timeout}s.",
                    self.peer_id, msg_type="warning")
                return
            except Exception as e:
                log(f"Generation failed: {e}",
                    self.peer_id, msg_type="warning")
                return

            self._append_turn(conv, "agent", self.node_name, answer)
            log(f"Answer processed locally.", self.peer_id, msg_type="text")
            log(f"{answer}", self.peer_id, msg_type="text", right=True)
            if answer_callback is not None:
                answer_callback(answer, self.node_name)

            if 'num_requests' not in self.models[selected_model[1]]:
                self.models[selected_model[1]]['num_requests'] = 0
            self.models[selected_model[1]]['num_requests'] += 1

            return

        msg_id = str(uuid.uuid4())
        msg = {
            "msg_id": msg_id,
            "type": "text",
            "from": self.public_key,
            "peer_id": self.peer_id,
            "peer_name": self.node_name,
            "assigned_tag": assigned_tag,
            "selected_model": selected_model,
            "message": user_input,
            "messages": messages,
            "chat_id": chat_id,
        }
        self.send(msg, selected_model[0], timeout=timeout)

        self.awaiting_acks[msg_id] = {
            "receiver": selected_model[0],
            "timestamp": time.time(),
            "answer_callback": answer_callback,
            "chat_id": chat_id,
        }
        
    def generate_answer(self, message, model_utils, is_local=False, prompt=None):
        text = prompt if prompt is not None else message
        input_ids = model_utils['tokenizer'](text, return_tensors='pt').to(self.device)
        out = redirect_prints(model_utils['model'].generate, input_ids=input_ids['input_ids'], max_length=self.max_answer_length)
        answer = model_utils['tokenizer'].decode(out[0], skip_special_tokens=True)[len(text):]
        return answer

    def __del__(self):
       if getattr(self, "proc", None) and self.proc.poll() is None:
           self.proc.terminate()
           try: 
               self.proc.wait(timeout=3)
           except subprocess.TimeoutExpired:
               self.proc.kill()