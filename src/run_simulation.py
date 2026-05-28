import os, traceback, argparse, threading, time
 
from peer import Peer
from utils import log
 
from ui.ui import SimApp, disable_input, enable_input

 
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

    try:
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

    except BaseException:

        os.makedirs("logs", exist_ok=True)
        peer_id = app.peer.peer_id if app.peer is not None else "unknown"
        path = f"logs/error_{peer_id}_{int(time.time())}.log"
        
        with open(path, "a") as f:
            traceback.print_exc(file=f)

        # Shut down the Textual app from this thread
        app.call_from_thread(app.exit)


def handle_input(app, args, user_input):
    log(f"User input: {user_input}", node_id="USER", msg_type=None, right=True)


def main():
    args = get_args()
    app = SimApp(args, run_peer, handle_input)
    try:
        app.run()
    except BaseException:
        os.makedirs("logs", exist_ok=True)
        if app.peer is not None:
            path = f"logs/error_{app.peer.peer_id}_{int(time.time())}.log"
        else:
            path = f"logs/error_{int(time.time())}.log"
        with open(path, "a") as f:
            traceback.print_exc(file=f)
        if app.peer is not None and app.peer.proc is not None:
            log(f"Terminating peer process (PID: {app.peer.proc.pid})", node_id=app.peer.peer_id)
            app.peer.proc.terminate()
            app.peer.proc.wait()
        raise

 
if __name__ == "__main__":
    main()