import argparse
import threading
import time

from peer import Peer


def get_args():
    parser = argparse.ArgumentParser(description="Test one peer locally")
    parser.add_argument("--peer-id", type=int, required=False, default=4, help="ID of the peer to test")  # TODO: default=None, required=True
    parser.add_argument("--depth", type=int, required=False, default=3, help="Depth of the network topology to find peers' public keys")
    parser.add_argument("--check-inactive-interval", type=int, required=False, default=30, help="Interval in seconds to check for inactive peers")
    parser.add_argument("--ping-peers-interval", type=int, required=False, default=60, help="Interval in seconds to ping incactive peers in the network")
    parser.add_argument("--share-knowledge-interval", type=int, required=False, default=120, help="Interval in seconds to share known peers with others")
    parser.add_argument("--num-peers-to-share-knowledge", type=int, required=False, default=5, help="Number of known random peers to share knowledge")
    return parser.parse_args()


def main():

    args = get_args()
    peer = Peer(args.peer_id, init_node=True)
    print(f"Peer {peer.peer_id} - Public Key: {peer.public_key[:8]}..., IPv6: {peer.ipv6_address}")

    t = threading.Thread(target=peer.recv_loop, daemon=True, name=str(peer.peer_id))
    t.start()

    peer.know_peers()
    last_check_inactive_peers = time.time()
    last_ping_inactive_peers = time.time()
    last_share_knowledge = time.time()
    
    try:
        while True:

            current_time = time.time()
            if current_time - last_check_inactive_peers >= args.check_inactive_interval:
                peer.check_inactive_peers()
                time.sleep(1)
                last_check_inactive_peers = current_time

            current_time = time.time()
            if current_time - last_ping_inactive_peers >= args.ping_peers_interval and len(peer.inactive_peers) > 0:
                peer.ping_inactive_peers()
                time.sleep(1)
                last_ping_inactive_peers = current_time

            current_time = time.time()
            if current_time - last_share_knowledge >= args.share_knowledge_interval:
                peer.share_knowledge(args.num_peers_to_share_knowledge)
                time.sleep(1)
                last_share_knowledge = current_time

    except KeyboardInterrupt:
        print("\nShutting down...")   


if __name__ == "__main__":
    main()
