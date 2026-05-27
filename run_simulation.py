import argparse, threading, time
 
from peer import Peer
from utils import log
 
from ui import SimApp, disable_input, enable_input

 
def get_args():
    parser = argparse.ArgumentParser(description="Test one peer locally")
    parser.add_argument("--peer-id", type=int, required=False, default=1)
    parser.add_argument("--discover-peers-interval", type=int, default=60)
    parser.add_argument("--check-acks-interval", type=int, default=10)
    parser.add_argument("--greeting-timeout", type=int, default=10)
    parser.add_argument("--knowledge-timeout", type=int, default=10)
    parser.add_argument("--ack-timeout", type=int, default=15)
    parser.add_argument("--share-knowledge-interval", type=int, default=30)
    parser.add_argument("--num-peers-to-greet", type=int, default=5)
    parser.add_argument("--num-peers-to-share", type=int, default=5)
    return parser.parse_args()
 
 
def run_peer(app, args):
    """All peer logic runs in this background thread."""

    disable_input(app)

    peer = Peer(args.peer_id, init_node=True)
 
    # Log the peer info now that the widget is available
    log(f"Peer {peer.peer_id} - Public Key: {peer.public_key[:8]}..., IPv6: {peer.ipv6_address}", peer.peer_id)
 
    t = threading.Thread(target=peer.recv_loop, daemon=True, name=str(peer.peer_id))
    t.start()
 
    peer.send_greetings(num_peers_to_greet=args.num_peers_to_greet, timeout=args.greeting_timeout)
    last_greetings = time.time()
    last_check_acks = time.time()
    last_share_knowledge = time.time()
 
    while True:

        current_time = time.time()
        if current_time - last_greetings >= args.discover_peers_interval:
            peer.send_greetings(num_peers_to_greet=args.num_peers_to_greet, timeout=args.greeting_timeout)
            last_greetings = current_time
            time.sleep(1)
 
        current_time = time.time()
        if current_time - last_check_acks >= args.check_acks_interval:
            peer.check_acks(timeout=args.ack_timeout)
            last_check_acks = current_time
            time.sleep(1)
 
        current_time = time.time()
        if current_time - last_share_knowledge >= args.share_knowledge_interval:
            peer.share_knowledge(num_peers_to_share=args.num_peers_to_share, timeout=args.knowledge_timeout)
            last_share_knowledge = current_time
            time.sleep(1)
        
        enable_input(app, peer.peer_id)
 
        time.sleep(0.1)


def handle_input(app, args, user_input):
    log(f"User input: {user_input}", node_id="USER", msg_type=None, right=True)


def main():
    args = get_args()
    SimApp(args, run_peer, handle_input).run()
 
 
if __name__ == "__main__":
    main()