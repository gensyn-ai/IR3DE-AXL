import os, sys, signal, traceback, argparse, threading, time
from multiprocessing import resource_tracker

from peer import Peer
from utils import log, ACTIVATE_UI

from ui.ui import IR3DEApp
from ui.ui_utils import disable_input, enable_input, disable_filters, enable_filters


os.environ.setdefault("COLORTERM", "truecolor")

# Peer spawns a ProcessPoolExecutor (model_worker) from a background thread.
# The first multiprocessing sync primitive created in a process lazily forks+
# execs a resource_tracker helper; doing that for the first time *after*
# Textual's event loop is already running crashes with `ValueError: bad
# value(s) in fds_to_keep` (same class of issue as the tqdm/mp-lock bug fixed
# earlier). Warming it up here, before app.run() starts the event loop, avoids
# it entirely — verified by direct reproduction.
resource_tracker.ensure_running()

 
def get_args():
    parser = argparse.ArgumentParser(description="Test one peer locally")
    parser.add_argument("--peer-id", type=int, required=False, default=None)
    parser.add_argument("--discover-peers-interval", type=int, default=60)
    parser.add_argument("--check-acks-interval", type=int, default=100)
    parser.add_argument("--greeting-timeout", type=int, default=60)
    parser.add_argument("--knowledge-timeout", type=int, default=60)
    parser.add_argument("--ack-timeout", type=int, default=60)
    parser.add_argument("--share-knowledge-interval", type=int, default=30)
    parser.add_argument("--num-peers-to-greet", type=int, default=5)
    parser.add_argument("--num-peers-to-share", type=int, default=5)
    parser.add_argument("--share-info-interval", type=int, default=60)
    parser.add_argument("--share-stats-interval", type=int, default=60)
    parser.add_argument("--stats-timeout", type=int, default=120)
    parser.add_argument("--ir3de-lambda", type=float, default=0.01)
    parser.add_argument("--ir3de-entropy-top-k", type=int, default=10)
    parser.add_argument("--max-answer-length", type=int, default=256)
    parser.add_argument( "--num-characters-conversation-history", type=int, default=1000, 
                        help="Maximum character budget for the prompt sent to experts (system "
                             "prompt + summary + history). When exceeded, oldest user/agent pairs "
                             "are summarised. Set to 0 to disable.")
    parser.add_argument('--tok-type', type=str, default='mistral', choices=['llama', 'mistral'], help='Type of tokenizer to use')
    return parser.parse_args()
 
 
def _install_node_signal_cleanup(peer):
    """Headless (main thread): stop the AXL node on SIGTERM/SIGHUP so an
    externally-killed peer process doesn't leave an orphaned node. The TUI has its
    own variant (_install_tui_node_cleanup); Ctrl+C is handled separately (headless:
    KeyboardInterrupt -> run_peer; TUI: action_quit).
    """
    def _handler(signum, frame):
        peer.stop_node()
        sys.stdout.flush()
        os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or signal unavailable on this platform.
            pass


def _install_tui_node_cleanup(app):
    """TUI (main thread): stop the AXL node on SIGTERM/SIGHUP — the catchable exits
    Textual's Ctrl+C handling doesn't cover. Closing the terminal window or an SSH
    drop delivers SIGHUP; an external kill/pkill/IDE-stop delivers SIGTERM. Installed
    before app.run(). Stops the node directly, then asks Textual to exit so the terminal
    is restored; falls back to SystemExit if app.exit() can't run. The window before
    app.peer is set is covered by the Peer's atexit (registered when the node spawns).
    """
    def _handler(signum, frame):
        peer = getattr(app, "peer", None)
        if peer is not None:
            peer.stop_node()
        try:
            app.exit()
        except Exception:
            raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or signal unavailable on this platform.
            pass


def run_peer(app, args):
    """All peer logic runs in this background thread."""

    peer = None
    try:
        if app is not None:
            disable_input(app)
            disable_filters(app)
        
        def _on_axl_ready():
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


def handle_input(app, args, user_input):
    chat_id = getattr(app.peer, "active_chat_id", None)
    log(f"{user_input}", node_id="USER", msg_type=None, right=True, chat_id=chat_id)
    app.peer.handle_user_input(user_input)


def main():

    args = get_args()

    if ACTIVATE_UI:
        app = IR3DEApp(args, run_peer, handle_input)
        _install_tui_node_cleanup(app)   # stop the node on SIGHUP (window close / SSH drop) and SIGTERM
        try:
            app.run()
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
    main()