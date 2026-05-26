import argparse
import threading
import time

from peer import Peer


def get_args():
    parser = argparse.ArgumentParser(description="Test one peer locally")
    parser.add_argument("--peer-id", type=int, required=False, default=4, help="ID of the peer to test")  # TODO: default=None, required=True
    parser.add_argument("--depth", type=int, required=False, default=3, help="Depth of the network topology to find peers' public keys")
    parser.add_argument("--discover-peers-interval", type=int, required=False, default=60, help="Interval in seconds to rediscover peers in the network")
    parser.add_argument("--share-knowledge-interval", type=int, required=False, default=120, help="Interval in seconds to share known peers with others")
    return parser.parse_args()


def main():
    args = get_args()
    peer = Peer(args.peer_id, init_node=True)
    print(f"Peer {peer.peer_id} - Public Key: {peer.public_key[:8]}..., IPv6: {peer.ipv6_address}")

    t = threading.Thread(target=peer.recv_loop, daemon=True, name=str(peer.peer_id))
    t.start()

    peer.know_peers()
    last_know_peers = time.time()
    last_share_knowledge = time.time()
    
    try:
        while True:

            current_time = time.time()
            if current_time - last_know_peers >= args.discover_peers_interval:
                peer.know_peers()
                time.sleep(1)
                last_know_peers = current_time

            current_time = time.time()
            if current_time - last_share_knowledge >= args.share_knowledge_interval:
                peer.share_knowledge()
                time.sleep(1)
                last_share_knowledge = current_time
                
    except KeyboardInterrupt:
        print("\nShutting down...")   


if __name__ == "__main__":
    main()
