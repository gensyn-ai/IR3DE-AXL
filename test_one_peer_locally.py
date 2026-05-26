import argparse, threading, time

from peer import Peer


def get_args():
    parser = argparse.ArgumentParser(description="Test one peer locally")
    parser.add_argument("--peer-id", type=int, required=False, default=4, help="ID of the peer to test")  # TODO: default=None, required=True
    parser.add_argument("--discover-peers-interval", type=int, required=False, default=60, help="Interval in seconds to rediscover peers in the network")
    parser.add_argument("--check-acks-interval", type=int, required=False, default=10, help="Interval in seconds to check for ACKs")
    parser.add_argument("--ack-timeout", type=int, required=False, default=60, help="Time in seconds to wait for an ACK before considering it expired")
    parser.add_argument("--share-knowledge-interval", type=int, required=False, default=30, help="Interval in seconds to share known peers with others")
    parser.add_argument("--num-peers-to-greet", type=int, required=False, default=5, help="Number of known peers to greet")
    parser.add_argument("--num-peers-to-share", type=int, required=False, default=5, help="Number of known peers to share knowledge with")
    return parser.parse_args()


def main():
    args = get_args()
    peer = Peer(args.peer_id, init_node=True)
    print(f"Peer {peer.peer_id} - Public Key: {peer.public_key[:8]}..., IPv6: {peer.ipv6_address}")

    t = threading.Thread(target=peer.recv_loop, daemon=True, name=str(peer.peer_id))
    t.start()

    peer.send_greetings(num_peers_to_greet=args.num_peers_to_greet)
    last_greetings = time.time()
    last_check_acks = time.time()
    last_share_knowledge = time.time()
    
    try:
        while True:

            current_time = time.time()
            if current_time - last_greetings >= args.discover_peers_interval:
                peer.send_greetings(num_peers_to_greet=args.num_peers_to_greet)
                time.sleep(1)
                last_greetings = current_time
            
            current_time = time.time()
            if current_time - last_check_acks >= args.check_acks_interval:
                peer.check_acks(timeout=args.ack_timeout)
                time.sleep(1)
                last_check_acks = current_time

            current_time = time.time()
            if current_time - last_share_knowledge >= args.share_knowledge_interval:
                peer.share_knowledge(num_peers_to_share=args.num_peers_to_share)
                time.sleep(1)
                last_share_knowledge = current_time
                
    except KeyboardInterrupt:
        print("\nShutting down...")   


if __name__ == "__main__":
    main()
