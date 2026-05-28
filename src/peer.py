from copy import deepcopy
import os, pathlib, subprocess, uuid, requests, json, time, random

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from ir3de_stats.models.llama_experts import get_llama_expert
from utils import ipv6_from_pubkey, log

AXL = "http://127.0.0.1:91"



class Peer:

    def __init__(self, peer_id):

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
        self.stats_info = metadata.get("stats", [])
        self.stats = self.get_stats_info()
    
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
                    log(f"Loading checkpoint from {model_info['path']}", self.peer_id, msg_type=None)
                    state = torch.load(model_info["path"], map_location='cpu')
                else:
                    raise FileNotFoundError(f"Checkpoint path not found: {model_info['path']}")
                model.load_state_dict(state, strict=False)
                model.to(self.device)
            elif "hf_name" in model_info:
                model = LlamaForCausalLM.from_pretrained(model_info["hf_name"])
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
        return models

    def get_stats_info(self):
        all_stats = []
        for stats in self.stats_info:
            stats_data = torch.load(stats["path"], map_location='cpu')
            tokenizer = AutoTokenizer.from_pretrained(stats_data["tokenizer"])
            model = LlamaForCausalLM.from_pretrained(stats_data["embedder"])
            embedder = deepcopy(model.model.embed_tokens).to(torch.float32)
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

    def send(self, message, peer_public_key, timeout=5):
        try:
            requests.post(
                f"{AXL}{self.peer_id:02d}/send",
                headers={"X-Destination-Peer-Id": peer_public_key},
                data=json.dumps(message),
                timeout=timeout
            )
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

    def recv_loop(self, timeout=5):
        
        while True:
            
            resp = requests.get(f"{AXL}{self.peer_id:02d}/recv")
            
            if resp.status_code == 200:
                
                sender = resp.headers.get("X-From-Peer-Id")
                assert sender is not None
                msg = json.loads(resp.text)
                
                if msg.get("type") == "text":
                    log(f"From {sender[:8]}...: {msg.get('message')}", self.peer_id, msg_type="text")
                
                elif msg.get("type") in ("greeting", "greeting-ack"):

                    log(f"Received greeting from {sender[:8]}: {msg.get('message')}", self.peer_id, msg_type=msg.get("type"))

                    if sender not in self.known_public_keys and msg.get("peer_id") != self.peer_id:
                        log(f"New node discovered with ID = {msg.get('peer_id')}!", self.peer_id, msg_type="newnode")
                        self.known_public_keys[sender] = {}
                        self.known_public_keys[sender]["peer_id"] = msg.get("peer_id")

                    if msg.get("type") == "greeting":
                        log(f"Sending greeting back to Node {msg.get('peer_id')}...", self.peer_id, msg_type="greeting-ack")
                        greetings = {
                            "msg_id": msg.get("msg_id"),
                            "type": "greeting-ack",
                            "from": self.public_key,
                            "peer_id": self.peer_id,
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
                        "peer_id": self.peer_id
                    }
                    self.send(knowledge_ack, sender, timeout=timeout)
                
                elif msg.get("type") == "info":

                    log(f"Received peers models and stats info from {sender[:8]}. Num models = {len(msg.get('models_info', {}))}, num stats = {len(msg.get('stats_info', {}))}", self.peer_id, msg_type="info")
                    if sender not in self.known_public_keys:
                        self.new_peer_discovered(msg, sender)
                    
                    self.known_public_keys[sender]["models_info"] = msg.get("models_info", [])
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
                        "peer_id": self.peer_id
                    }
                    self.send(info_ack, sender, timeout=timeout)
                
                elif msg.get("type") == "knowledge-ack":
                    log(f"Received ACK for knowledge from Node {msg.get('peer_id')}.", self.peer_id, msg_type="knowledge-ack")
                    if msg.get("msg_id") in self.awaiting_acks:
                        log(f"Removing message ID {msg.get('msg_id')} from awaiting ACKs.", self.peer_id, msg_type="knowledge-ack")
                        del self.awaiting_acks[msg.get("msg_id")]

                elif msg.get("type") == "info-ack":
                    log(f"Received ACK for model and stats info from Node {msg.get('peer_id')}.", self.peer_id, msg_type="info-ack")
                    if msg.get("msg_id") in self.awaiting_acks:
                        log(f"Removing message ID {msg.get('msg_id')} from awaiting ACKs.", self.peer_id, msg_type="info-ack")
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
        for pk in known_pks[:num_peers_to_share]:
            log(f"Sharing knowledge with peer {pk[:8]}, ID={self.known_public_keys[pk]['peer_id']}...", self.peer_id, msg_type='knowledge')
            knowledge_msg = {
                "msg_id": str(uuid.uuid4()),
                "type": "knowledge",
                "from": self.public_key,
                "peer_id": self.peer_id,
                "known_peers": self.known_public_keys
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
                log(f"No ACK received from peer {pk[:8]}... with ID {peer_id} for message ID {msg_id} after {timeout} seconds.", self.peer_id, msg_type="warning")
            else:
                log(f"No ACK received from peer {pk[:8]}... with unknown ID, for message ID {msg_id} after {timeout} seconds.", self.peer_id, msg_type="warning")
            del self.awaiting_acks[msg_id]

    def share_stats_and_models_info(self, num_peers_to_share=5, timeout=5):

        log(f"Sharing IR3DE local stats and local models info with the network...", self.peer_id, msg_type='info')
        known_pks = list(self.known_public_keys.keys())
        random.shuffle(known_pks)

        info_shared = 0
        for pk in known_pks[:num_peers_to_share]:
            log(f"Sharing IR3DE local stats info and local models info with peer {pk[:8]}, ID={self.known_public_keys[pk]['peer_id']}...", self.peer_id, msg_type='info')
            info_msg = {
                "msg_id": str(uuid.uuid4()),
                "type": "info",
                "from": self.public_key,
                "peer_id": self.peer_id,
                "models_info": [],
                "stats_info": []
            }
            for model_info in self.models_info:
                info_msg["models_info"].append({
                    "tags": model_info.get("tags"),
                    "tokenizer_name": model_info.get("tokenizer"),
                    "size": model_info.get("size")
                })
            for stats in self.stats_info:
                info_msg["stats_info"].append({
                    "tags": stats.get("tags"),
                    "tokenizer_name": stats.get("tokenizer"),
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