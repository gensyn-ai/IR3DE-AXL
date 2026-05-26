import subprocess, uuid, requests, json, time, random
from utils import ipv6_from_pubkey, log

AXL = "http://127.0.0.1:91"



class Peer:

    def __init__(self, peer_id, init_node=False):

        if init_node:
            self.proc = subprocess.Popen(
                ["./start_node.sh", str(peer_id)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            sleep_time = 5
            print(f"Waiting {sleep_time} seconds for node {peer_id} to initialize...")
            time.sleep(sleep_time)
        else:
            self.proc = None

        self.peer_id = peer_id
        self.session = requests.Session()
        self.topology = self.get_topology(self.session)
        self.public_key = self.topology['our_public_key']
        self.ipv6_address = self.topology['our_ipv6']
        self.known_public_keys = {}
        self.awaiting_acks = {}

        with open(f"local_nodes/metadata{peer_id:02d}.json", "r") as f:
            metadata = json.load(f)
            self.node_name = metadata.get("node_name", f"Node {peer_id}")
    
    def get_topology(self, session):
        resp = session.get(f"{AXL}{self.peer_id:02d}/topology", timeout=5)
        resp.raise_for_status()
        topology = resp.json()
        return topology

    def send(self, message, peer_public_key):
        requests.post(
            f"{AXL}{self.peer_id:02d}/send",
            headers={"X-Destination-Peer-Id": peer_public_key},
            data=json.dumps(message)
        )

    def recv_loop(self):
        
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
                        self.send(greetings, sender)
                    
                    if msg.get("type") == "greeting-ack" and msg.get("msg_id") in self.awaiting_acks:
                        log(f"Received ACK for greeting from Node {msg.get('peer_id')}. Removing from awaiting ACKs.", self.peer_id, msg_type="greeting-ack")
                        del self.awaiting_acks[msg.get("msg_id")]

                elif msg.get("type") == "knowledge":

                    log(f"Received peers knowledge from {sender[:8]}. Num entries = {len(msg.get('known_peers', {}))}", self.peer_id, msg_type="knowledge")
                    if sender not in self.known_public_keys:
                        log(f"New node discovered with ID = {msg.get('peer_id')}!", self.peer_id, msg_type="newnode")
                        self.known_public_keys[sender] = {}
                        self.known_public_keys[sender]["peer_id"] = msg.get("peer_id")

                    new_peers = 0
                    for pk, info in msg.get("known_peers", {}).items():
                        peer_id = info.get("peer_id")
                        if pk not in self.known_public_keys and ipv6_from_pubkey(pk) != ipv6_from_pubkey(self.public_key):
                            log(f"New node discovered with ID = {peer_id}!", self.peer_id, msg_type="newnode")
                            self.known_public_keys[pk] = {}
                            self.known_public_keys[pk]["peer_id"] = peer_id
                            new_peers += 1
                    
                    log(f"Updated known peers with knowledge from Node {msg.get('peer_id')}. New peers added: {new_peers}. Total known peers: {len(self.known_public_keys)}.", self.peer_id, msg_type="knowledge")
                    
                    knowledge_ack = {
                        "msg_id": msg.get("msg_id"),
                        "type": "knowledge-ack",
                        "from": self.public_key,
                        "peer_id": self.peer_id
                    }
                    self.send(knowledge_ack, sender)
                
                elif msg.get("type") == "knowledge-ack":
                    log(f"Received ACK for knowledge from Node {msg.get('peer_id')}.", self.peer_id, msg_type="knowledge-ack")
                    if msg.get("msg_id") in self.awaiting_acks:
                        log(f"Removing message ID {msg.get('msg_id')} from awaiting ACKs.", self.peer_id, msg_type="knowledge-ack")
                        del self.awaiting_acks[msg.get("msg_id")]

                else:
                    log(f"WARNING: Received unknown message type from {sender[:8]}, type={msg.get('type')}.", self.peer_id, msg_type="warning")

            time.sleep(0.2)

    def send_greetings(self, num_peers_to_greet=5):
        
        log(f"Discovering peers in the network...", self.peer_id, msg_type=None)
        
        if self.topology['peers'] is None:
            log(f"No peers found in the topology.", self.peer_id, msg_type=None)
            return

        known_public_keys = list(set([k['public_key'] for k in self.topology['peers']]) | set(list(self.known_public_keys.keys())))
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
            self.send(greetings, pk)
            self.awaiting_acks[greetings["msg_id"]] = {
                "receiver": pk,
                "timestamp": time.time()
            }
            greetings_sent += 1
        
        log(f"Sent greetings to {greetings_sent} known peers.", self.peer_id, msg_type='greeting')

    def share_knowledge(self, num_peers_to_share=5):
        
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
            self.send(knowledge_msg, pk)
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
            pk = self.awaiting_acks[msg_id]['receiver'][:8]
            log(f"No ACK received from peer {pk}... with ID {self.awaiting_acks[msg_id]['receiver']} for message ID {msg_id} after {timeout} seconds.", self.peer_id, msg_type="warning")
            del self.awaiting_acks[msg_id]
