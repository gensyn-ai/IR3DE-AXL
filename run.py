"""Entry point: launches one IR3DE-AXL peer — an AXL P2P node plus the
Textual TUI (or, headless, just the node) that lets it chat, gossip with
other peers, and route messages to experts via IR3DE. Meant for running
local simulations of the P2P network (per peer, in its own terminal), not
a production deployment tool.

Usage:
    python run.py [options]

--peer-id is optional and only needed when simulating several peers on the
same machine at once (see scripts/add_local_node.sh, which creates each
simulated node's config/metadata/key under local_nodes/): omit it to run
the single default local peer (local_nodes/metadata.json), or pass the
node's number to run that one instead (local_nodes/metadataNN.json).

Key options (see utils.get_args() for the full list and defaults):
    --peer-id N           Which simulated local node to run (omit for the
                          default local peer)
    --tok-type {mistral,llama}
                          Which tokenizer/embedder identity this peer is
                          active for — determines which IR3DE stats and
                          which default model it uses
    --answer-timeout SECONDS
                          How long to wait for an expert's answer before
                          giving up
    --num-characters-conversation-history N
                          Prompt character budget before older turns get
                          summarized (0 disables summarization)

Example:
    python run.py --peer-id 01
"""
import os
import sys
import threading
import time
import traceback
from multiprocessing import resource_tracker

# This file lives at the repo root, but every module it and its dependents
# import (peer, utils, ui.*, ...) uses bare imports assuming src/ is on
# sys.path, matching how they import each other internally.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from peer import Peer
from ui.ui import IR3DEApp
from ui.ui_utils import disable_filters, disable_input, enable_filters, enable_input
from utils import (ACTIVATE_UI, _install_node_signal_cleanup, _install_tui_node_cleanup,
                    get_args, handle_input, log)


os.environ.setdefault("COLORTERM", "truecolor")

# Peer spawns a ProcessPoolExecutor (model_worker) from a background thread.
# The first multiprocessing sync primitive created in a process lazily forks+
# execs a resource_tracker helper; doing that for the first time *after*
# Textual's event loop is already running crashes with `ValueError: bad
# value(s) in fds_to_keep` (same class of issue as the tqdm/mp-lock bug fixed
# earlier). Warming it up here, before app.run() starts the event loop, avoids
# it entirely — verified by direct reproduction.
resource_tracker.ensure_running()


def run_peer(app, args):
    """All peer logic runs in this background thread."""

    peer = None
    try:
        if app is not None:
            disable_input(app)
            disable_filters(app)

        def _on_axl_ready():
            """Peer's on_axl_ready callback: swap the loading message once
            the AXL node subprocess is up, before Peer.get_models_info()
            fetches each configured model's cheap metadata (not its weights
            — those load lazily, on first actual use)."""
            if app is not None:
                app.call_from_thread(app._show_loading_models)

        peer = Peer(
            peer_id=args.peer_id,
            ir3de_lambda=args.ir3de_lambda,
            ir3de_entropy_top_k=args.ir3de_entropy_top_k,
            max_answer_length=args.max_answer_length,
            num_characters_conversation_history=args.num_characters_conversation_history,
            on_axl_ready=_on_axl_ready,
            tokenizer_name="mistralai/Mistral-7B-v0.1" if args.tok_type == 'mistral' else "meta-llama/Meta-Llama-3-8B"
        )
        if app is not None:
            app.peer = peer
            app.call_from_thread(app._show_loading_chats)
            time.sleep(0.5)
            app.call_from_thread(app._populate_chat_tabs)
            enable_filters(app)
            enable_input(app, peer.peer_id)
        else:
            _install_node_signal_cleanup(peer)

        # Log the peer info now that the widget is available
        log(f"Peer {peer.peer_id} - Public Key: {peer.public_key[:8]}..., IPv6: {peer.ipv6_address}", peer.peer_id)

        t = threading.Thread(target=peer.recv_loop, daemon=True, name=str(peer.peer_id))
        t.start()

        peer.send_greetings(num_peers_to_greet=args.num_peers_to_greet, timeout=args.greeting_timeout)
        last_greetings = time.time()
        last_check_acks = time.time()
        last_shared_knowledge = time.time()
        last_shared_info = time.time() + 60 # 2 * args.share_knowledge_interval  # Wait a little bit to share info, to increase chances of discovering peers first
        last_shared_stats = time.time() + 60 # 2 * args.share_knowledge_interval
        never_shared_info = True
        never_shared_stats = True

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
            if current_time - last_shared_knowledge >= args.share_knowledge_interval:
                peer.share_knowledge(num_peers_to_share=args.num_peers_to_share, timeout=args.knowledge_timeout)
                last_shared_knowledge = current_time
                time.sleep(1)

            current_time = time.time()
            share_info_time_to_wait = 0 if never_shared_info else args.share_info_interval
            if current_time - last_shared_info >= share_info_time_to_wait:
                peer.share_stats_and_models_info(num_peers_to_share=args.num_peers_to_share, timeout=args.stats_timeout)
                last_shared_info = current_time
                never_shared_info = False
                time.sleep(1)

            current_time = time.time()
            share_stats_time_to_wait = 0 if never_shared_stats else args.share_stats_interval
            if current_time - last_shared_stats >= share_stats_time_to_wait:
                peer.ask_stats(num_peers_to_ask=args.num_peers_to_share, timeout=args.stats_timeout)
                last_shared_stats = current_time
                never_shared_stats = False
                time.sleep(1)

            time.sleep(0.1)

    except BaseException:

        os.makedirs("logs", exist_ok=True)
        if app is not None:
            peer_id = app.peer.peer_id if app.peer is not None else "unknown"
        else:
            peer_id = "unknown"
        path = f"logs/error_{peer_id}_{int(time.time())}.log"

        with open(path, "a") as f:
            traceback.print_exc(file=f)

        # Stop the node directly — App.exit() does not route through action_quit.
        target_peer = app.peer if app is not None else peer
        if target_peer is not None:
            target_peer.stop_node()

        if app is not None:
            try:
                app.call_from_thread(app.exit)
            except RuntimeError:
                # App is already stopping; nothing left to signal.
                pass


def main():
    """Entry point: launch the Textual TUI (or, if ACTIVATE_UI is False, run
    the peer headless in this thread) and route crashes/Ctrl+C to a clean
    node shutdown instead of leaving the AXL subprocess orphaned."""
    args = get_args()

    if ACTIVATE_UI:
        app = IR3DEApp(args, run_peer, handle_input)
        _install_tui_node_cleanup(app)   # stop the node on SIGHUP (window close / SSH drop) and SIGTERM
        try:
            app.run()
        except KeyboardInterrupt:
            # A raw Ctrl+C that escaped Textual's own "ctrl+c" binding (e.g.
            # landing while the terminal is briefly out of raw mode). This is
            # a normal user-initiated shutdown, not a crash: stop the node
            # and exit quietly instead of dumping a stack trace.
            if app.peer is not None:
                app.peer.stop_node()
        except Exception:
            os.makedirs("logs", exist_ok=True)
            if app.peer is not None:
                path = f"logs/error_{app.peer.peer_id}_{int(time.time())}.log"
            else:
                path = f"logs/error_{int(time.time())}.log"
            with open(path, "a") as f:
                traceback.print_exc(file=f)
            if app.peer is not None:
                log(f"Stopping peer node (PID: {app.peer.proc.pid})", node_id=app.peer.peer_id)
                app.peer.stop_node()
            raise

    else:
        run_peer(None, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
