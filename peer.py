# import subprocess
import requests, json, time
import multiprocessing as mp
from multiprocessing.connection import wait
import threading
import signal
import os

AXL = "http://127.0.0.1:91"



class Peer:

    def __init__(self, peer_id):

        # self.proc = subprocess.Popen(
        #     ["./start_node.sh", str(peer_id)],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        #     text=True
        # )

        self.peer_id = peer_id
        self.session = requests.Session()
        self.topology = self.get_topology(self.session)
        self.public_key = self.topology['our_public_key']
        self.ipv6_address = self.topology['our_ipv6']
 
    def get_topology(self, session):
        resp = session.get(f"{AXL}{self.peer_id:02d}/topology", timeout=5)
        resp.raise_for_status()
        topology = resp.json()
        return topology
        
    def send(self, message, peer_public_key):
        requests.post(f"{AXL}{self.peer_id:02d}/send", headers={"X-Destination-Peer-Id": peer_public_key},
        data=json.dumps(message))

    def recv_loop(self):
        while True:
            resp = requests.get(f"{AXL}{self.peer_id:02d}/recv")
            if resp.status_code == 200:
                sender = resp.headers.get("X-From-Peer-Id")
                assert sender is not None
                print(f"[NODE {self.peer_id}] From {sender[:8]}...: {resp.text}")
            time.sleep(0.2)


def monitor_subprocesses(processes):

    sentinel_to_process = {p.sentinel: p for p in processes}
    ready = wait(list(sentinel_to_process.keys()))
    failed_process = sentinel_to_process[ready[0]]
    failed_process.join()

    code = failed_process.exitcode
    if code == 0:
        reason = "exited unexpectedly with code 0"
    elif code < 0:
        reason = f"was killed by signal {-code}"
    else:
        reason = f"crashed with exit code {code}"

    print(f"\n[MONITOR] Subprocess for node {failed_process.name} {reason}.")
    print("[MONITOR] Terminating remaining subprocesses...")

    for p in processes:
        if p.is_alive():
            p.terminate()

    for p in processes:
        p.join(timeout=2)

    os.kill(os.getpid(), signal.SIGINT)


def main():

    hub = Peer(0)
    node1 = Peer(1)
    node2 = Peer(2)
    node3 = Peer(3)
    all_peers = [hub, node1, node2, node3]
    recv_processes = []

    print("Start listening:")
    for peer in all_peers:
        print(f"Peer {peer.peer_id} - Public Key: {peer.public_key[:8]}..., IPv6: {peer.ipv6_address}")
        p = mp.Process(target=peer.recv_loop, daemon=True, name=str(peer.peer_id))
        p.start()
        recv_processes.append(p)

    monitor_thread = threading.Thread(
        target=monitor_subprocesses,
        args=(recv_processes,),
        daemon=True,
    )
    monitor_thread.start()
    
    try:

        while True:

            sender = input("Which node should start sending messages? ")
            while sender not in ['0', '1', '2', '3']:
                sender = input("Invalid input. Please enter 0, 1, 2, or 3.\nWhich node should start sending messages? ")

            receiver = input("Which node should receive messages? ")
            while receiver not in ['0', '1', '2', '3']:
                receiver = input("Invalid input. Please enter 0, 1, 2, or 3.\nWhich node should receive messages? ")

            message = input("What message should be sent? ")

            try:
                all_peers[int(sender)].send(message, all_peers[int(receiver)].public_key)
                print(f"Sent message from node {sender} to node {receiver}.")

            except Exception as e:
                print(f"Error sending message: {e}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:

        for p in recv_processes:
            if p.is_alive():
                p.terminate()
                
        for p in recv_processes:
            p.join(timeout=2)


if __name__ == '__main__':
    main()
