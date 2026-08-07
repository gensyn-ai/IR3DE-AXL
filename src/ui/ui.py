"""IR3DEApp: the Textual TUI for one peer. Not run directly — constructed
and run by run.py, which also drives the peer's background threads
(run_peer, handle_input) via app.call_from_thread.

Layout (see compose()):
  - Left pane: a TabbedContent with Control Panel (pick expertise tags and
    a model per tag), Statistics (known peers, local/remote models and
    IR3DE stats, per-tag expert counts, latency plots), and Logs (the raw
    message log, filterable by type).
  - Right pane: one tab per chat, each with its own scrollable transcript
    and the shared message input row at the bottom.

Most of this module is reactive glue: periodic refreshers
(_refresh_stats_tab and friends, on a 2s timer) resync the widgets to
Peer state, and on_* handlers respond to clicks/messages from the
components in ui/components/.
"""

import os
import random
import statistics
import threading
from datetime import datetime, timezone

from rich.markup import escape as _md_escape
from rich.text import Text as RichText
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import DataTable, Log, RichLog, Static, TabbedContent, TabPane, TextArea
from textual.widgets._tabbed_content import ContentTabs
from textual.widgets._tabs import Tab
from textual_plotext import PlotextPlot

import chats
from peer import Peer
from ui.components import *
from ui.glyphs import DIAMOND_FRAMES, DIAMOND_ROTATION, IR3DE_BANNER
from ui.ui_utils import FollowTailLog, read_css
from utils import (format_mean_std, format_params, ipv6_from_pubkey, iter_log_buffer, log,
                    MAX_TITLE_CHARS, MSG_TYPE_COLORS, render_chat_history_into, set_filter_predicate,
                    set_log_widget, set_output_widget, symbol_for_tag, unset_output_widget)


class IR3DEApp(App):
    CSS = read_css()
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, args, logs_function, input_handler):
        """Construct the app shell. `logs_function` (run.py's run_peer) and
        `input_handler` (run.py's handle_input) are stashed for on_mount to
        launch on background threads once the screen exists."""
        super().__init__()
        self.args = args
        self.peer: "Peer | None" = None
        self.logs_function = logs_function
        self.input_handler = input_handler
        self._active_filters: set[str] = set(MSG_TYPE_COLORS.keys()) | {"no-tag"}
        self._filters_enabled = False
        self._selected_peer_pubkey: str | None = None   # None until first refresh; defaults to self
        self._show_all_experts = False
        self._selected_tags: set[str] = set()
        self._selected_models: dict[str, tuple[str, int]] = {}   # tag -> (peer_pk, model_idx)
        self._selection_memory: dict[str, tuple[str, int]] = {}   # survives deactivation
        self._tag_bars_last: tuple | None = None
        self._peers_table_last_sig: tuple | None = None
        self._section_sigs: dict[str, tuple] = {}
        self._sections: dict[str, ExpertiseSection] = {}
        self._loading_dots: int = 0
        self._loading_timer = None
        self._control_spinner_frame: int = 0
        self._control_spinner_timer = None
        self._show_all_latencies = False
        self._latency_plot_last: tuple | None = None
        self._latency_metric = "prompt"   # one of: 'prompt' | 'total' | 'input'
        self._loading_label = "Initializing AXL backend"
        self._right_spinner_active = True
        self._pending_answer_chat_id: str | None = None   # chat_id currently awaiting an answer, if any
        self._waiting_dots: int = 0
        self._waiting_timer = None

    def compose(self) -> ComposeResult:
        """Lay out the whole screen: banner/bars, the left pane (Control
        Panel/Statistics/Logs tabs), the resizable divider, and the right
        pane (chat tabs + input row)."""
        yield Static("", id="top-divider")
        yield Static(IR3DE_BANNER, id="ir3de-banner")
        yield Static("", id="top-bar")

        with Horizontal():

            yield Static("", id="border-left")

            # ───── LEFT PANE: Control Panel / Statistics / Logs ─────
            with Vertical(id="left-pane"):
                with TabbedContent(id="left-tabs"):

                    with TabPane("Control Panel", id="tab-control"):
                        yield Static(DIAMOND_FRAMES[0], id="control-spinner")
                        with VerticalScroll(id="control-scroll"):
                            yield Static("Expertise selection", id="expertise-title",
                                        classes="control-section-title")
                            yield Static(
                                "Choose one or more expertise. Selected expertise will determine the available models.",
                                id="expertise-desc",
                                classes="control-section-desc",
                            )
                            yield Vertical(id="tag-buttons-container")
                            yield Static("Model selection (one per selected expertise)",
                                        id="model-selection-title",
                                        classes="control-section-title")
                            yield Vertical(id="expertise-sections")

                    with TabPane("Statistics", id="tab-stats"):
                        yield Static(DIAMOND_FRAMES[0], id="stats-spinner")
                        with VerticalScroll(id="stats-scroll"):
                            yield Static("Known Peers", id="peers-title",
                                        classes="stats-table-title")
                            yield DataTable(id="peers-table")
                            with Horizontal(id="stats-tables-row"):
                                with Vertical(classes="stats-table-container"):
                                    yield Static("Local models", id="models-title",
                                                classes="stats-table-title")
                                    yield DataTable(id="models-table")
                                with Vertical(classes="stats-table-container"):
                                    yield Static("Local IR3DE stats", id="stats-title",
                                                classes="stats-table-title")
                                    yield DataTable(id="ir3de-stats-table")
                            with Horizontal(id="plots-row"):
                                with Vertical(classes="plot-pane"):
                                    with Horizontal(id="tag-bars-header"):
                                        yield Static("Experts per tag", id="tag-bars-title",
                                                    classes="stats-table-title")
                                        yield ToggleChip("show all", id="show-all-chip")
                                    yield PlotextPlot(id="tag-bars")

                                with Vertical(classes="plot-pane"):
                                    with Horizontal(id="latency-plot-header"):
                                        yield Static("Latency vs parameters", id="latency-plot-title",
                                                    classes="stats-table-title")
                                        yield ToggleChip("show all", id="show-all-latency-chip")
                                    yield Static("[ Prompt latency ▾ ]",
                                                id="latency-metric-button",
                                                classes="metric-button")
                                    yield PlotextPlot(id="latency-plot")

                    with TabPane("Logs", id="tab-logs"):
                        with Horizontal(id="logs-header"):
                            yield Static("filters ▾", id="filter-toggle")
                        yield Vertical(id="filter-drawer")        # empty; populated dynamically
                        yield FollowTailLog(id="logs", highlight=False, markup=False,
                                            auto_scroll=False, wrap=True)

            yield ResizableDivider(id="divider")

            # ───── RIGHT PANE: Chats ─────
            with Vertical(id="right-pane"):
                with TabbedContent(id="right-tabs"):
                    with TabPane("...", id="tab-chat-placeholder"):
                        yield Static(DIAMOND_FRAMES[0], id="right-spinner")
                yield Static("Initializing AXL backend", id="loading-msg")
                with Horizontal(id="copy-response-row"):
                    yield Static("[COPY RESPONSE]", id="copy-response-button", markup=False)
                yield Static("", id="input-divider")
                with Horizontal(id="input-row"):
                    yield Static("> ", id="prompt")
                    yield SubmittableTextArea(id="user-input")
                    yield Static("[SEND]", id="send-button", markup=False)

            yield Static("", id="border-right")

        yield Static("", id="bottom-bar")

    def _enable_horizontal_tab_scroll(self) -> None:
        """Plain mouse-wheel scroll (and a vertical two-finger trackpad swipe)
        always targets vertical scroll by default, rerouting to horizontal
        only with Shift/Ctrl held. The chat tab bar has no vertical content
        at all, so its default handler never consumes the event (nothing to
        scroll) and it just bubbles up doing nothing.

        Textual dispatches `_on_<event>` handlers by looking them up on the
        *class* (`cls.__dict__`), walking the MRO — an instance-level
        attribute override is invisible to it. So this patches the
        `ContentTabs` class (the concrete type TabbedContent always uses
        internally) once, scoped at call time to only the chat tab bar via
        `tabs.tabbed_content.id` — any other TabbedContent (e.g. the left
        Control Panel/Statistics/Logs tabs) keeps Textual's normal vertical
        wheel behavior. Horizontal trackpad swipes already work on their own
        via Textual's distinct MouseScrollLeft/Right events, when the
        terminal reports them — untouched here.
        """
        original_scroll_down = ContentTabs._on_mouse_scroll_down
        original_scroll_up = ContentTabs._on_mouse_scroll_up

        def _on_mouse_scroll_down(tabs: ContentTabs, event: events.MouseScrollDown) -> None:
            """Mouse wheel down on the chat tab bar scrolls it right; any
            other TabbedContent keeps the normal vertical behavior."""
            if tabs.tabbed_content.id == "right-tabs":
                if tabs.query_one("#tabs-scroll")._scroll_right_for_pointer(animate=False):
                    event.stop()
                return
            original_scroll_down(tabs, event)

        def _on_mouse_scroll_up(tabs: ContentTabs, event: events.MouseScrollUp) -> None:
            """Mouse wheel up on the chat tab bar scrolls it left; any
            other TabbedContent keeps the normal vertical behavior."""
            if tabs.tabbed_content.id == "right-tabs":
                if tabs.query_one("#tabs-scroll")._scroll_left_for_pointer(animate=False):
                    event.stop()
                return
            original_scroll_up(tabs, event)

        ContentTabs._on_mouse_scroll_down = _on_mouse_scroll_down
        ContentTabs._on_mouse_scroll_up = _on_mouse_scroll_up

    def on_mount(self):
        """Post-compose setup: fill decorative bars, patch in horizontal tab
        scrolling, wire up the Logs widget, set up the three Statistics
        tables, launch the peer's background thread (logs_function), and
        start the loading/spinner animation timers."""
        self._fill_bars()
        self._enable_horizontal_tab_scroll()

        log_widget = self.query_one("#logs", RichLog)
        set_log_widget(log_widget)

        self.query_one("#user-input", TextArea).focus()

        self.call_after_refresh(self._fill_input_divider)

        threading.Thread(
            target=self.logs_function,
            args=(self, self.args),
            daemon=True,
            name="logs-thread"
        ).start()
    
        set_filter_predicate(self._should_show_log)

        # Statistics tab — peers table
        table = self.query_one("#peers-table", DataTable)
        self._col_name, self._col_pubkey, self._col_ipv6, self._col_peer_comm_lat = table.add_columns(
            "Name", "Public Key", "IPv6 address", "Comm latency"
        )
        self._sort_column_key = self._col_name
        self._sort_reverse = False                  # default: ascending

        table.cursor_type = "row"
        table.zebra_stripes = True

        # Refresh table every 2 seconds while the app is running.
        self.set_interval(2.0, self._refresh_stats_tab)

        # Models table setup
        models_table = self.query_one("#models-table", DataTable)
        self._col_models_type, self._col_models_params, self._col_models_tags, self._col_models_num_requests, \
        self._col_models_prompt_lat, self._col_models_total_tok_lat, self._col_models_input_tok_lat = models_table.add_columns(
            "Model Type  ",
            "Parameters  ",
            "Expertise  ",
            "Num requests  ",
            "Prompt latency  ",
            "Latency / total tok  ",
            "Latency / input tok  ",
        )
        models_table.zebra_stripes = True
        self._models_sort_column_key = self._col_models_type
        self._models_sort_reverse = False

        # IR3DE stats table setup
        stats_table = self.query_one("#ir3de-stats-table", DataTable)
        (self._col_stats_tok, self._col_stats_emb,
         self._col_stats_ds, self._col_stats_tags) = stats_table.add_columns(
            "Tokenizer  ", "Embedder  ", "Dataset  ", "Expertise  "
        )
        stats_table.zebra_stripes = True
        self._stats_sort_column_key = self._col_stats_tok
        self._stats_sort_reverse = False

        self._loading_timer = self.set_interval(0.5, self._tick_loading_msg)
        self._control_spinner_timer = self.set_interval(0.1, self._tick_control_spinner)

    def _update_sort_arrows(self, table, columns, active_key, reverse):
        """Relabel `table`'s headers, appending a ▲/▼ arrow to whichever
        column is currently sorted. Shared by the peers/models/stats tables."""
        arrow = "▲" if reverse else "▼"
        for col_key, base in columns:
            suffix = f" {arrow}" if col_key == active_key else "  "
            text = f"{base}{suffix}"
            table.columns[col_key].label = RichText(text)
        table.refresh()

    
    def _refresh_peers_table(self):
        """Rebuild the Statistics tab's Known Peers table (self + every
        known_public_keys entry), preserving sort/scroll/selection. Called
        every 2s by _refresh_stats_tab and after a row is clicked."""
        if self.peer is None:
            return

        table = self.query_one("#peers-table", DataTable)
        saved_x, saved_y = table.scroll_x, table.scroll_y

        if self._selected_peer_pubkey is None:
            self._selected_peer_pubkey = self.peer.public_key

        rows: list[tuple[str, tuple[str, str, str, str]]] = []
        comm_lat_by_pk: dict[str, float] = {self.peer.public_key: 0.0}

        self_row = (self.peer.public_key, (
            f"{self.peer.node_name} (self)",
            self.peer.public_key[:16] + "...",
            self.peer.ipv6_address,
            "—",                                    # self has no comm latency
        ))

        for pk, info in self.peer.known_public_keys.items():
            if pk == self.peer.public_key:
                continue
            name = info.get("peer_name") or f"Node {info.get('peer_id', '?')}"
            try:
                ipv6 = ipv6_from_pubkey(pk)
            except ValueError:
                ipv6 = "<invalid>"

            comm_lats = info.get("comm_latencies", []) or []
            comm_lat_by_pk[pk] = statistics.mean(comm_lats) if comm_lats else 0.0
            comm_lat_str = format_mean_std(comm_lats, unit="ms", scale=1000.0, precision=1)

            rows.append((pk, (name, pk[:16] + "...", ipv6, comm_lat_str)))

        if self._sort_column_key == self._col_peer_comm_lat:
            rows.sort(key=lambda r: comm_lat_by_pk.get(r[0], 0.0),
                    reverse=self._sort_reverse)
        else:
            col_idx = {
                self._col_name: 0,
                self._col_pubkey: 1,
                self._col_ipv6: 2,
            }[self._sort_column_key]
            rows.sort(key=lambda r: r[1][col_idx].lower(),
                    reverse=self._sort_reverse)

        all_rows = [self_row] + rows

        sig = tuple(r[1] for r in all_rows) + (self._sort_column_key, self._sort_reverse, self._selected_peer_pubkey)
        if getattr(self, "_peers_table_last_sig", None) == sig:
            return
        self._peers_table_last_sig = sig

        table.clear()
        selected_row_index = 0
        for i, (pk, display) in enumerate(all_rows):
            table.add_row(*display, key=pk)
            if pk == self._selected_peer_pubkey:
                selected_row_index = i

        self._update_header_labels()

        if 0 <= selected_row_index < table.row_count:
            table.move_cursor(row=selected_row_index)

        self.call_after_refresh(
            lambda: table.scroll_to(x=saved_x, y=saved_y, animate=False)
        )

    def _update_header_labels(self):
        """Relabel the peers table's headers with a sort arrow (its columns
        aren't in the generic _update_sort_arrows(columns=...) shape)."""
        table = self.query_one("#peers-table", DataTable)
        arrow = "▲" if self._sort_reverse else "▼"
        labels = {
            self._col_name:   "Name",
            self._col_pubkey: "Public Key",
            self._col_ipv6:   "IPv6 address",
        }
        for col_key, base in labels.items():
            text = f"{base} {arrow}" if col_key == self._sort_column_key else base
            table.columns[col_key].label = RichText(text)
        table.refresh()

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected):
        """Clicking a column header sorts by it (toggling direction on a
        repeat click), for whichever of the three Statistics tables it's in."""
        table_id = event.control.id if event.control else None
        if table_id == "peers-table":
            self._sort_reverse = not self._sort_reverse if event.column_key == self._sort_column_key else False
            self._sort_column_key = event.column_key
            self._refresh_peers_table()
        elif table_id == "models-table":
            self._models_sort_reverse = not self._models_sort_reverse if event.column_key == self._models_sort_column_key else False
            self._models_sort_column_key = event.column_key
            self._refresh_models_table()
        elif table_id == "ir3de-stats-table":
            self._stats_sort_reverse = not self._stats_sort_reverse if event.column_key == self._stats_sort_column_key else False
            self._stats_sort_column_key = event.column_key
            self._refresh_ir3de_stats_table()

    def _should_show_log(self, msg_type) -> bool:
        """Whether a log entry of this msg_type passes the Logs tab's
        current filter set. Registered with utils.set_filter_predicate."""
        key = "no-tag" if msg_type is None else msg_type
        return key in self._active_filters

    def on_filter_chip_toggled(self, event: FilterChip.Toggled):
        """A FilterChip was clicked: update the active filter set and
        re-render the Logs tab to match."""
        if event.active:
            self._active_filters.add(event.msg_type)
        else:
            self._active_filters.discard(event.msg_type)
        self._refresh_chip_states()
        self._rerender_logs()

    def _rerender_logs(self):
        """Redraw the Logs tab from the full _log_buffer under the current
        filter set, since filtering can't be done incrementally on already
        -written lines. Called whenever the filter set changes."""
        widget = self.query_one("#logs", FollowTailLog)
        widget.clear()
        for ts, node, mtype, line in iter_log_buffer():
            if self._should_show_log(mtype):
                widget.write(line)

    def _fill_input_divider(self):
        """Redraw the horizontal rule above the input row to the current
        terminal width. Called on mount and on every resize."""
        div = self.query_one("#input-divider", Static)
        div.update("═" * div.size.width)

    def on_submittable_text_area_submitted(self, event: SubmittableTextArea.Submitted):
        """The user pressed Enter/[SEND]: hide the chat's placeholder text
        and hand the message to input_handler (run.py's handle_input) on
        its own background thread."""
        if self.peer is not None and self.peer.active_chat_id is not None:
            self._hide_chat_placeholder(self.peer.active_chat_id)
        user_text = event.value
        if self.input_handler is not None:
            threading.Thread(
                target=self.input_handler,
                args=(self, self.args, user_text),
                daemon=True,
                name="handle-input",
            ).start()

    def _is_chat_awaiting_answer(self, chat_id: str | None) -> bool:
        """Thin wrapper around Peer.is_awaiting_answer, safe to call with a
        None chat_id or before the peer exists."""
        if self.peer is None or chat_id is None:
            return False
        return self.peer.is_awaiting_answer(chat_id)

    def start_waiting_for_answer(self, chat_id: str | None) -> None:
        """Disable the input and show an animated waiting message. Called via
        app.call_from_thread from run.py's handle_input, so this
        already runs on the main thread — do the widget updates directly
        rather than through disable_input(), which does its own
        call_from_thread and would raise (Textual forbids calling
        call_from_thread from the app's own thread)."""
        self._pending_answer_chat_id = chat_id
        self._waiting_dots = 0
        self.query_one("#send-button", Static).disabled = True
        self._refresh_waiting_message()
        if self._waiting_timer is None:
            self._waiting_timer = self.set_interval(0.5, self._tick_waiting_msg)

    def maybe_stop_waiting_for_answer(self, chat_id: str | None) -> None:
        """Called right after input_handler returns. Local generation and
        validation failures resolve synchronously, so this stops the wait
        immediately for those. A remote answer resolves later, in a
        different thread (recv_loop); _tick_waiting_msg's periodic check
        catches that case once it actually lands."""
        if not self._is_chat_awaiting_answer(chat_id):
            self._stop_waiting_for_answer()

    def _tick_waiting_msg(self) -> None:
        """0.5s timer callback: advance the "..." animation, or stop it once
        the pending answer has actually resolved (catches remote answers,
        which land asynchronously on recv_loop's thread)."""
        if not self._is_chat_awaiting_answer(self._pending_answer_chat_id):
            self._stop_waiting_for_answer()
            return
        self._waiting_dots = (self._waiting_dots + 1) % 4
        self._refresh_waiting_message()

    def _refresh_waiting_message(self) -> None:
        """Show the right waiting message for whichever chat is currently
        active — the pending chat itself, or a different one the user
        switched to while it's still processing. Re-asserts disabled=True
        on every call as a safety net."""
        if self._pending_answer_chat_id is None:
            return
        active_chat_id = self.peer.active_chat_id if self.peer is not None else None
        if active_chat_id == self._pending_answer_chat_id:
            label = "Waiting for the answer"
        else:
            label = "Awaiting for an answer in another chat"
        input_widget = self.query_one("#user-input", SubmittableTextArea)
        input_widget.disabled = True
        input_widget.text = label + "." * self._waiting_dots
        self.query_one("#prompt", Static).styles.color = "#444444"

    def _stop_waiting_for_answer(self) -> None:
        """Clear the waiting state: stop the dots timer, re-enable input."""
        self._pending_answer_chat_id = None
        if self._waiting_timer is not None:
            self._waiting_timer.stop()
            self._waiting_timer = None
        input_widget = self.query_one("#user-input", SubmittableTextArea)
        input_widget.text = ""
        input_widget.disabled = False
        input_widget.focus()
        self.query_one("#prompt", Static).styles.color = "#888888"
        self.query_one("#send-button", Static).disabled = False

    def _hide_chat_placeholder(self, chat_id: str) -> None:
        """Hide the 'Write a message below' placeholder for a chat. Uses query()
        (returns [] when the widget isn't there) so an already-non-existent
        placeholder — e.g. because the chat had messages when it was mounted —
        is a silent no-op."""
        for placeholder in self.query(f"#placeholder-{chat_id}"):
            placeholder.styles.display = "none"

    async def action_quit(self):
        """Bound to Ctrl+C: stop the AXL node before exiting Textual."""
        if self.peer is not None:
            self.peer.stop_node()
        self.exit()

    def on_resize(self, event):
        """Terminal was resized: redraw everything sized off the terminal
        width (decorative bars, filter chips, chat tab bar's reserved width)."""
        self._fill_bars()
        self._fill_input_divider()
        drawer = self.query_one("#filter-drawer")
        if drawer.styles.display != "none":
            self._layout_filter_chips()
        self._resize_tabs_scroll_for_buttons()
    
    def _fill_bars(self):
        """Redraw the top/bottom horizontal rules to the current terminal width."""
        bar = "═" * self.app.size.width
        for wid in ("#top-divider", "#top-bar", "#bottom-bar"):
            self.query_one(wid, Static).update(bar)

    def on_click(self, event):
        """Route clicks for controls and chat-tab actions."""
        if event.control is None:
            return
        if event.control.id == "filter-toggle":
            if not self._filters_enabled:
                return  # silently ignore clicks while disabled
            self._toggle_drawer()
        elif event.control.id == "send-button":
            if event.control.disabled:
                return
            self.query_one("#user-input", SubmittableTextArea).submit()
        elif event.control.id == "copy-response-button":
            self._copy_latest_response()
        elif event.control.id == "latency-metric-button":
            self.push_screen(LatencyMetricSelectScreen(), self._on_latency_metric_chosen)
        elif event.control.id == "new-chat-button":
            self._create_new_chat()
        elif event.control.id == "chat-menu-button":
            self._open_chat_menu()

        # Locate the Tab widget under the click, if any.
        tab_widget = next((node for node in event.control.ancestors_with_self if isinstance(node, Tab)), None)

        if tab_widget is not None:
            tab_id = tab_widget.id or ""
            marker = "--content-tab-tab-chat-"
            if tab_id.startswith(marker):
                chat_id_from_tab = tab_id[len(marker):]
                width = tab_widget.size.width

                if width <= event.x <= width + 2:
                    self._close_chat_tab(chat_id_from_tab)
                    event.stop()
                    return

                # Double-click anywhere else on the tab -> rename.
                if event.chain == 2:
                    self._open_chat_rename_dialog()
                    event.stop()
                    return

    def _copy_latest_response(self) -> None:
        """Copy the active chat's most recent successful agent response."""
        if self.peer is None or self.peer.active_chat_id is None:
            self.notify("No active chat to copy from", severity="warning")
            return

        chat = self.peer.chats.get(self.peer.active_chat_id)
        if chat is None:
            self.notify("No active chat to copy from", severity="warning")
            return

        response = next(
            (
                message.get("text", "")
                for message in reversed(chat.get("messages", []))
                if message.get("role") == "agent"
                and message.get("status", "ok") == "ok"
            ),
            "",
        )
        if not response:
            self.notify("No response to copy", severity="warning")
            return

        self.copy_to_clipboard(response)
        self.notify("Latest response copied")

    def _toggle_drawer(self):
        """Show/hide the Logs tab's filter-chip drawer. Called from on_click
        when "filters ▾/▴" is clicked."""
        drawer = self.query_one("#filter-drawer")
        toggle = self.query_one("#filter-toggle", Static)
        is_hidden = drawer.styles.display == "none"
        drawer.styles.display = "block" if is_hidden else "none"
        toggle.update("filters ▴" if is_hidden else "filters ▾")
        if is_hidden:
            self.call_after_refresh(self._layout_filter_chips)
    
    def _layout_filter_chips(self):
        """Rebuild the filter drawer's chips (all/none + one per message
        type), wrapping them into rows that fit the current width. Called
        when the drawer opens and on every resize while it's open."""
        drawer = self.query_one("#filter-drawer", Vertical)
        if drawer.styles.display == "none":
            return
        avail = drawer.size.width
        if avail <= 0:
            self.call_after_refresh(self._layout_filter_chips)
            return

        for child in list(drawer.children):
            child.remove()

        chip_specs: list[tuple[str, str, str]] = [
            ("action", "all",    "#ffffff"),
            ("action", "none",   "#ffffff"),
            ("filter", "no-tag", "#ffffff"),
        ] + [("filter", name, color) for name, color in MSG_TYPE_COLORS.items()]

        # Group chips into rows that fit horizontally
        chip_margin = 1  # must match CSS margin-right
        rows: list[list[tuple[str, str, str]]] = []
        current, used = [], 0
        for kind, name, color in chip_specs:
            chip_w = len(name) + 2 + chip_margin
            if current and used + chip_w > avail:
                rows.append(current)
                current, used = [], 0
            current.append((kind, name, color))
            used += chip_w
        if current:
            rows.append(current)

        # Mount each row as a Horizontal full of chips
        for row in rows:
            h = Horizontal(classes="filter-row")
            drawer.mount(h)
            for kind, name, color in row:
                if kind == "action":
                    chip = ActionChip(name, color, id=f"action-{name}")
                    all_filters = set(MSG_TYPE_COLORS.keys()) | {"no-tag"}
                    if name == "all":
                        chip.active = (self._active_filters == all_filters)
                    elif name == "none":
                        chip.active = (len(self._active_filters) == 0)
                    chip._refresh_label()
                else:  # "filter"
                    chip = FilterChip(name, color, id=f"chip-{name}")
                    if name not in self._active_filters:
                        chip.active = False
                        chip._refresh_label()
                h.mount(chip)

    def on_action_chip_triggered(self, event: ActionChip.Triggered):
        """The "all"/"none" chip was clicked: bulk-set the active log filters."""
        if event.action == "all":
            self._active_filters = set(MSG_TYPE_COLORS.keys()) | {"no-tag"}
        elif event.action == "none":
            self._active_filters = set()
        else:
            return

        self._refresh_chip_states()
        self._rerender_logs()

    def _refresh_chip_states(self):
        """Sync every FilterChip and ActionChip's visual state to current
        _active_filters."""
        all_filters = set(MSG_TYPE_COLORS.keys()) | {"no-tag"}
        is_all  = self._active_filters == all_filters
        is_none = len(self._active_filters) == 0

        for chip in self.query(FilterChip):
            should_be_active = chip.msg_type in self._active_filters
            if chip.active != should_be_active:
                chip.active = should_be_active
                chip._refresh_label()

        for chip in self.query(ActionChip):
            if chip.action == "all":
                chip.set_active(is_all)
            elif chip.action == "none":
                chip.set_active(is_none)
    
    def _refresh_stats_tab(self):
        """Refresh every Statistics-tab widget and the chat tab titles.
        Called every 2s by the timer set in on_mount."""
        if self.peer is None:
            return
        self._refresh_models_table()
        self._refresh_ir3de_stats_table()
        self._refresh_peers_table()
        self._refresh_tag_bars()
        self._refresh_latency_plot()
        self._refresh_tag_buttons()
        self._refresh_expertise_sections()
        self._refresh_chat_tab_title()

    def _refresh_models_table(self):
        """Rebuild the Statistics tab's models table for whichever peer is
        selected in the peers table (self or a remote), preserving sort/scroll."""
        if self.peer is None:
            return

        pk = self._resolve_selected_pk()
        assert pk is not None
        is_self = (pk == self.peer.public_key)

        table = self.query_one("#models-table", DataTable)
        saved_x, saved_y = table.scroll_x, table.scroll_y

        title = self.query_one("#models-title", Static)
        title.update("Local models" if is_self else f"{self._selected_peer_name(pk)} models")

        if is_self:
            raw_models = [
                (
                    mi.get("type", "unknown"),
                    mi.get("size", 0),
                    mi.get("tags", []) or [],
                    self.peer.models[i].get("num_requests", 0),
                    self.peer.models[i].get("latencies", []) or [],
                )
                for i, mi in enumerate(self.peer.models_info)
            ]
        else:
            info = self.peer.known_public_keys.get(pk, {})
            model_lats = info.get("model_latencies", {}) or {}
            raw_models = [
                (
                    mi.get("type", "unknown"),
                    mi.get("size", 0),
                    mi.get("tags", []) or [],
                    mi.get("num_requests", 0),
                    model_lats.get(i, []) or [],
                )
                for i, mi in enumerate(info.get("models_info", []))
            ]


        data = []
        for model_type, num_params, tags, num_requests, latencies in raw_models:
            tags_str = ", ".join(tags)

            prompt_lats    = [e["prompt_latency"] for e in latencies
                            if e.get("prompt_latency") is not None]
            total_tok_lats = [e["latency_per_total_token"] for e in latencies
                            if e.get("latency_per_total_token") is not None]
            input_tok_lats = [e["latency_per_input_token"] for e in latencies
                            if e.get("latency_per_input_token") is not None]

            data.append({
                "raw": {
                    self._col_models_type:           model_type.lower(),
                    self._col_models_params:         num_params,
                    self._col_models_tags:           tags_str.lower(),
                    self._col_models_num_requests:   num_requests,
                    self._col_models_prompt_lat:
                        statistics.mean(prompt_lats) if prompt_lats else 0.0,
                    self._col_models_total_tok_lat:
                        statistics.mean(total_tok_lats) if total_tok_lats else 0.0,
                    self._col_models_input_tok_lat:
                        statistics.mean(input_tok_lats) if input_tok_lats else 0.0,
                },
                "display": (
                    model_type,
                    format_params(num_params),
                    tags_str,
                    str(num_requests),
                    format_mean_std(prompt_lats,    unit="s",  scale=1.0,    precision=2),
                    format_mean_std(total_tok_lats, unit="ms", scale=1000.0, precision=1),
                    format_mean_std(input_tok_lats, unit="ms", scale=1000.0, precision=1),
                ),
            })

        data.sort(key=lambda r: r["raw"][self._models_sort_column_key],
                reverse=self._models_sort_reverse)

        table.clear()
        for r in data:
            table.add_row(*r["display"])

        columns = [
            (self._col_models_type,          "Model Type"),
            (self._col_models_params,        "Parameters"),
            (self._col_models_tags,          "Expertise"),
            (self._col_models_num_requests,  "Num requests"),
            (self._col_models_prompt_lat,    "Prompt latency"),
            (self._col_models_total_tok_lat, "Latency / total tok"),
            (self._col_models_input_tok_lat, "Latency / input tok"),
        ]

        self._update_sort_arrows(table, columns,
                                self._models_sort_column_key,
                                self._models_sort_reverse)

        # Synchronous restore — eliminates the "snap to 0" frame.
        table.scroll_to(x=saved_x, y=saved_y, animate=False)
        self.call_after_refresh(lambda: table.scroll_to(x=saved_x, y=saved_y, animate=False))

    def _refresh_ir3de_stats_table(self):
        """Rebuild the Statistics tab's IR3DE stats table for whichever peer
        is selected in the peers table, preserving sort/scroll."""
        if self.peer is None:
            return

        pk = self._resolve_selected_pk()
        assert pk is not None
        is_self = (pk == self.peer.public_key)

        title = self.query_one("#stats-title", Static)
        title.update("Local IR3DE stats" if is_self else f"{self._selected_peer_name(pk)} IR3DE stats")

        raw_rows = []
        if is_self:
            for stats_info, stats in zip(self.peer.stats_info, self.peer.stats):
                tokenizer = stats_info.get("tokenizer_name", "?")
                embedder  = stats_info.get("embedder_name", "?")
                datasets  = ", ".join(stats.get("datasets", []) or [])
                tags      = ", ".join(stats.get("tags", []) or [])
                raw_rows.append((tokenizer, embedder, datasets, tags))
        else:
            info = self.peer.known_public_keys.get(pk, {})
            for si in info.get("stats_info", []):
                tokenizer = si.get("tokenizer_name", "?")
                embedder  = si.get("embedder_name", "?")
                datasets  = ", ".join(si.get("datasets_names", []) or [])
                tags      = ", ".join(si.get("tags", []) or [])
                raw_rows.append((tokenizer, embedder, datasets, tags))

        data = [{
            "raw": {
                self._col_stats_tok:  tok.lower(),
                self._col_stats_emb:  emb.lower(),
                self._col_stats_ds:   ds.lower(),
                self._col_stats_tags: tg.lower(),
            },
            "display": (tok, emb, ds, tg),
        } for (tok, emb, ds, tg) in raw_rows]

        table = self.query_one("#ir3de-stats-table", DataTable)
        saved_x, saved_y = table.scroll_x, table.scroll_y
        data.sort(key=lambda r: r["raw"][self._stats_sort_column_key],
                reverse=self._stats_sort_reverse)
        table.clear()
        for r in data:
            table.add_row(*r["display"])
        self._update_sort_arrows(table, [
            (self._col_stats_tok,  "Tokenizer"),
            (self._col_stats_emb,  "Embedder"),
            (self._col_stats_ds,   "Dataset"),
            (self._col_stats_tags, "Expertise"),
        ], self._stats_sort_column_key, self._stats_sort_reverse)

        # Synchronous restore — eliminates the "snap to 0" frame.
        table.scroll_to(x=saved_x, y=saved_y, animate=False)
        self.call_after_refresh(lambda: table.scroll_to(x=saved_x, y=saved_y, animate=False))
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Clicking a peers-table row selects that peer, refreshing the
        models/stats tables and plots to show its data instead of self's."""
        if event.control is None or event.control.id != "peers-table":
            return
        pk = event.row_key.value if event.row_key else None
        if not pk:
            return
        self._selected_peer_pubkey = pk
        self._refresh_models_table()
        self._refresh_ir3de_stats_table()
        self._refresh_tag_bars()
        self._refresh_latency_plot()
    
    def _resolve_selected_pk(self) -> str | None:
        """Return the currently-selected pk, falling back to self if it disappears."""
        if self.peer is None:
            return None
        pk = self._selected_peer_pubkey
        if pk is None or (pk != self.peer.public_key and pk not in self.peer.known_public_keys):
            return self.peer.public_key
        return pk

    def _selected_peer_name(self, pk: str) -> str:
        """Display name for a public key: this peer's node_name for self,
        else the remote peer_name/peer_id known_public_keys has for it."""
        assert self.peer is not None
        if pk == self.peer.public_key:
            return self.peer.node_name
        info = self.peer.known_public_keys.get(pk, {})
        return info.get("peer_name") or f"Node {info.get('peer_id', '?')}"

    def _refresh_tag_bars(self):
        """Redraw the "experts per tag" bar chart (Statistics tab) for
        self, the selected peer, or (if "show all" is on) everyone combined."""
        if self.peer is None:
            return

        pk = self._resolve_selected_pk()
        assert pk is not None
        is_self = (pk == self.peer.public_key)

        title = self.query_one("#tag-bars-title", Static)
        if self._show_all_experts:
            title.update("All experts per tag")
        else:
            title.update(
                "Local experts per tag" if is_self
                else f"{self._selected_peer_name(pk)} experts per tag"
            )

        known_tags: set[str] = set(self.peer.known_tags)
        model_tag_lists: list[list[str]] = []

        def add_models_from_self():
            """Collect this peer's own models' tag lists."""
            assert self.peer is not None
            for m in self.peer.models:
                model_tag_lists.append(list(m.get("tags", []) or []))

        def add_models_from_remote(info: dict):
            """Collect one remote peer's models' tag lists."""
            for mi in info.get("models_info", []):
                model_tag_lists.append(list(mi.get("tags", []) or []))

        if self._show_all_experts:
            add_models_from_self()
            for info in self.peer.known_public_keys.values():
                add_models_from_remote(info)
        elif is_self:
            add_models_from_self()
        else:
            add_models_from_remote(self.peer.known_public_keys.get(pk, {}))

        counts = {tag: 0 for tag in sorted(known_tags)}
        for tags in model_tag_lists:
            for tag in tags:
                if tag in counts:
                    counts[tag] += 1

        # Skip the redraw if the data is identical to the last tick.
        signature = (self._show_all_experts, pk, tuple(counts.items()))
        if signature == self._tag_bars_last:
            return
        self._tag_bars_last = signature

        plot = self.query_one("#tag-bars", PlotextPlot)

        num_bars = len(counts)
        plot.styles.height = num_bars + 4

        plot.plt.clear_data()
        plot.plt.clear_figure()
        plot.plt.theme("dark")

        labels = [f" {tag}" for tag in list(counts.keys())]
        values = list(counts.values())

        if values:
            plot.plt.bar(
                labels, values,
                orientation="horizontal",
                width=0.5,
                color="violet",
            )
            max_v = max(values)
            if max_v > 0:
                plot.plt.xticks(list(range(0, max_v + 1)))

        plot.plt.xlabel("# experts")
        plot.refresh()

    def on_toggle_chip_toggled(self, event: ToggleChip.Toggled):
        """Either "show all" chip was toggled: switch its plot between
        self/selected-peer-only and every known peer combined."""
        if event.chip_id == "show-all-chip":
            self._show_all_experts = event.active
            self._refresh_tag_bars()
        elif event.chip_id == "show-all-latency-chip":
            self._show_all_latencies = event.active
            self._refresh_latency_plot()
    

    def _refresh_tag_buttons(self):
        """Rebuild the Control Panel's expertise tag buttons if the known
        tag set changed, wrapping them into rows that fit the panel width."""
        if self.peer is None:
            return

        container = self.query_one("#tag-buttons-container", Vertical)
        avail = container.size.width
        if avail <= 0:
            return

        # Compare desired vs. current button set; rebuild only if they differ
        desired = sorted(set(self.peer.known_tags))
        current = [btn.tag for btn in container.query(TagButton)]
        if desired == current:
            return                          # no churn — preserves selection visuals

        for child in list(container.children):
            child.remove()

        # Pack buttons into rows that fit horizontally
        button_w = 26                       # must match TagButton { width: 26 } in style.css
        margin   = 1
        per_row  = max(1, avail // (button_w + margin))

        row: list[TagButton] = []
        rows: list[list[TagButton]] = []
        for tag in desired:
            sym = symbol_for_tag(tag)
            btn = TagButton(tag, sym, id=f"tag-btn-{tag}")
            if tag in self._selected_tags:
                btn.active = True
                btn._refresh_label()
            row.append(btn)
            if len(row) == per_row:
                rows.append(row)
                row = []
        if row:
            rows.append(row)

        for r in rows:
            h = Horizontal(classes="tag-button-row")
            container.mount(h)
            for btn in r:
                h.mount(btn)
    
    def on_tag_button_toggled(self, event: TagButton.Toggled):
        """A Control Panel tag button was toggled: update the selected-tags
        set (mirrored onto the peer) and refresh the expertise sections."""
        if event.active:
            self._selected_tags.add(event.tag)
        else:
            self._selected_tags.discard(event.tag)

        if self.peer is not None:
            self.peer.selected_tags = set(self._selected_tags)

        self._refresh_expertise_sections()

    def _refresh_expertise_sections(self):
        """Sync the Control Panel's ExpertiseSection cards to the current
        selected tags: drop deactivated ones, mount/update the rest, and
        pick a default model selection for any tag missing one."""
        if self.peer is None:
            return

        container = self.query_one("#expertise-sections", Vertical)
        desired = sorted(self._selected_tags)

        # Drop deactivated sections — uses our own registry
        for tag in list(self._sections.keys()):
            if tag not in desired:
                if tag in self._selected_models:
                    self._selection_memory[tag] = self._selected_models[tag]
                    del self._selected_models[tag]
                self._sections[tag].remove()
                del self._sections[tag]

        # For each desired tag, ensure a section exists and is up to date
        for tag in desired:
            # 1) Make sure _selected_models[tag] has a valid value
            candidates = self._gather_candidates_for_tag(tag)
            valid_keys = {(pk, idx) for pk, idx, *_ in candidates}

            if tag not in self._selected_models:
                if tag in self._selection_memory and self._selection_memory[tag] in valid_keys:
                    self._selected_models[tag] = self._selection_memory[tag]
                else:
                    default = self._pick_default_selection(candidates)
                    if default is not None:
                        self._selected_models[tag] = default
                        self._selection_memory[tag] = default
            elif self._selected_models[tag] not in valid_keys:
                default = self._pick_default_selection(candidates)
                if default is not None:
                    self._selected_models[tag] = default
                    self._selection_memory[tag] = default
                else:
                    self._selected_models.pop(tag, None)

            selected = self._selected_models.get(tag)

            # 2) Mount the section or update the existing one
            if tag not in self._sections:
                section = ExpertiseSection(tag, symbol_for_tag(tag), candidates, selected)
                container.mount(section)
                self._sections[tag] = section
            else:
                self._sections[tag].update_state(candidates, selected)

        if self.peer is not None:
            self.peer.selected_models = dict(self._selected_models)

    def on_model_row_selected(self, event: ModelRow.Selected):
        """A ModelRow was clicked: record it as the chosen model for its
        tag and update every row in that tag's section to reflect it."""
        self._selected_models[event.tag] = (event.peer_pk, event.model_idx)
        self._selection_memory[event.tag] = (event.peer_pk, event.model_idx)   # ← remember

        for section in self.query(ExpertiseSection):
            if section._tag != event.tag:
                continue
            for row in section.query(ModelRow):
                row.set_active((row._peer_pk, row._model_idx) ==
                            (event.peer_pk, event.model_idx))
            section.set_has_selection(True)

        if self.peer is not None:
            self.peer.selected_models = dict(self._selected_models)

    def _pick_default_selection(self, candidates: list[tuple[str, int, str, str, int]]) -> tuple[str, int] | None:
        """Apply the user's default-selection rules to a candidate list."""
        if not candidates:
            return None
        if len(candidates) == 1:
            pk, idx, *_ = candidates[0]
            return (pk, idx)

        assert self.peer is not None
        locals_only = [(pk, idx) for pk, idx, *_ in candidates if pk == self.peer.public_key]
        if locals_only:
            return random.choice(locals_only)

        pk, idx, *_ = random.choice(candidates)
        return (pk, idx)

    def _gather_candidates_for_tag(
    self, tag: str
) -> list[tuple[str, int, str, str, int]]:
        """Collect (peer_pk, idx, display_name, node_label, params) candidates.
        Pure data — no UI access, safe to call synchronously."""
        if self.peer is None:
            return []

        def shorten(s: str, n: int) -> str:
            """Truncate to n chars with a trailing "..." if too long."""
            return s if len(s) <= n else s[:n-3] + "..."

        candidates: list[tuple[str, int, str, str, int]] = []

        # Local
        for i, m_info in enumerate(self.peer.models_info):
            if tag not in (m_info.get("tags") or []):
                continue
            name = m_info.get("hf_name") or (m_info.get("path") or "unknown").rsplit("/", 1)[-1]
            candidates.append((
                self.peer.public_key, i, name,
                shorten(self.peer.node_name, 7) + " (self)",
                m_info.get("size", 0),
            ))

        # Remote
        for pk, info in self.peer.known_public_keys.items():
            node = info.get("peer_name") or f"Node {info.get('peer_id', '?')}"
            for i, mi in enumerate(info.get("models_info") or []):
                if tag not in (mi.get("tags") or []):
                    continue
                candidates.append((
                    pk, i,
                    mi.get("name") or mi.get("type", "?"),
                    shorten(node, 14),
                    mi.get("size", 0),
                ))

        return candidates

    def _tick_loading_msg(self) -> None:
        """Cycle the loading message between 0 and 3 trailing dots."""
        self._loading_dots = (self._loading_dots + 1) % 4
        msg = self.query_one("#loading-msg", Static)
        msg.update(self._loading_label + "." * self._loading_dots)

    def _hide_loading_msg(self) -> None:
        """Hide the loading message/spinners once the peer is ready. Called
        from ui_utils.enable_input, right before the input box is unlocked."""
        if self._loading_timer is not None:
            self._loading_timer.stop()
            self._loading_timer = None
        msg = self.query_one("#loading-msg", Static)
        msg.styles.display = "none"

        # Stop the shared spinner timer
        if self._control_spinner_timer is not None:
            self._control_spinner_timer.stop()
            self._control_spinner_timer = None

        for spinner_id, content_id in (
            ("#control-spinner", "#control-scroll"),
            ("#stats-spinner",   "#stats-scroll"),
        ):
            self.query_one(spinner_id, Static).styles.display = "none"
            self.query_one(content_id).styles.display = "block"

    def _tick_control_spinner(self) -> None:
        """Advance the diamond loading-spinner animation shared by the
        Control Panel, Statistics, and (while active) chat panes."""
        self._control_spinner_frame = (self._control_spinner_frame + 1) % len(DIAMOND_ROTATION)
        frame = DIAMOND_FRAMES[DIAMOND_ROTATION[self._control_spinner_frame]]
        spinners = ["#control-spinner", "#stats-spinner"]
        if self._right_spinner_active:
            spinners.append("#right-spinner")
        for spinner_id in spinners:
            self.query_one(spinner_id, Static).update(frame)

    def _refresh_latency_plot(self):
        """Redraw the "latency vs parameters" scatter plot (Statistics tab)
        for the currently-selected metric (prompt/total/input) and scope
        (self, selected peer, or all if "show all" is on)."""
        if self.peer is None:
            return

        pk = self._resolve_selected_pk()
        assert pk is not None
        is_self = (pk == self.peer.public_key)

        field_map = {
            "prompt": ("prompt_latency",           "s",      1.0),
            "total":  ("latency_per_total_token",  "ms/tok", 1000.0),
            "input":  ("latency_per_input_token",  "ms/tok", 1000.0),
        }
        metric = getattr(self, "_latency_metric", "prompt")
        field, unit, scale = field_map[metric]

        metric_label = {
            "prompt": f"prompt latency ({unit})",
            "total":  f"latency / total tok ({unit})",
            "input":  f"latency / input tok ({unit})",
        }[metric]

        title = self.query_one("#latency-plot-title", Static)
        if self._show_all_latencies:
            title.update("Latency (all models)")
        else:
            title.update(
                "Latency (local models)" if is_self
                else f"Latency ({self._selected_peer_name(pk)} models)"
            )

        points: list[tuple[int, float, str]] = []

        def add_local():
            """Add one (size, latency, label) point per local model with data."""
            assert self.peer is not None
            for i, mi in enumerate(self.peer.models_info):
                size = mi.get("size", 0) or 0
                latencies = self.peer.models[i].get("latencies", []) or []
                vals = [e[field] for e in latencies if e.get(field) is not None]
                if not vals or size <= 0:
                    continue
                label = mi.get("hf_name") or os.path.basename(mi.get("path", "?"))
                points.append((size, statistics.mean(vals) * scale, label))

        def add_remote(info):
            """Add one (size, latency, label) point per one remote peer's
            models with data."""
            models_info_list = info.get("models_info") or []
            model_lats = info.get("model_latencies", {}) or {}
            for i, mi in enumerate(models_info_list):
                size = mi.get("size", 0) or 0
                latencies = model_lats.get(i, []) or []
                vals = [e[field] for e in latencies if e.get(field) is not None]
                if not vals or size <= 0:
                    continue
                label = mi.get("name") or mi.get("type", "?")
                points.append((size, statistics.mean(vals) * scale, label))

        if self._show_all_latencies:
            add_local()
            for ppk, info in self.peer.known_public_keys.items():
                if ppk == self.peer.public_key:
                    continue
                add_remote(info)
        elif is_self:
            add_local()
        else:
            add_remote(self.peer.known_public_keys.get(pk, {}))

        signature = (
            self._show_all_latencies,
            pk,
            metric,
            tuple((s, round(l, 4)) for s, l, _ in points),
        )
        if signature == self._latency_plot_last:
            return
        self._latency_plot_last = signature

        plot = self.query_one("#latency-plot", PlotextPlot)
        plot.styles.height = max(1, len(self.peer.known_tags) + 3)

        # Match the left plot's height minus 1 row (the row we gave to the button).
        tag_bars_plot = self.query_one("#tag-bars", PlotextPlot)
        left_h = tag_bars_plot.styles.height
        # styles.height is a Length object; coerce safely
        try:
            left_h_val = int(left_h.value)        # type: ignore[union-attr]
        except Exception:
            left_h_val = len(self.peer.known_tags) + 4
        plot.styles.height = max(1, left_h_val - 1)

        plot.plt.clear_data()
        plot.plt.clear_figure()
        plot.plt.theme("dark")

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plot.plt.scatter(xs, ys, color="violet", marker="dot")

        plot.plt.xlabel("# params")
        plot.refresh()
    
    def _on_latency_metric_chosen(self, value: str | None) -> None:
        """LatencyMetricSelectScreen callback: switch the latency plot's
        metric and relabel its button, or do nothing if canceled."""
        if value is None:
            return
        self._latency_metric = value
        labels = {
            "prompt": "Prompt latency",
            "total":  "Latency / total token",
            "input":  "Latency / input token",
        }
        self.query_one("#latency-metric-button", Static).update(
            f"[ {labels[value]} ▾ ]"
        )
        self._latency_plot_last = None    # force redraw with new metric
        self._refresh_latency_plot()
    
    def _refresh_chat_tab_title(self) -> None:
        """Sync every open chat tab's label to its chat's current title
        (set once a model generates one, or via manual rename). Called
        every 2s by _refresh_stats_tab."""
        if self.peer is None:
            return
        tabbed = self.query_one("#right-tabs", TabbedContent)
        for chat_id, chat in self.peer.chats.items():
            title = chat.get("title") or "Chat"
            new_label = self._make_tab_label(title)
            try:
                tab = tabbed.get_tab(f"tab-chat-{chat_id}")
                if str(tab.label) != f"{title}  ×":
                    tab.label = new_label
            except Exception:
                continue

    def _open_chat_rename_dialog(self) -> None:
        """Open the rename modal for the active chat. Called from on_click
        on a double-click on a chat tab."""
        if self.peer is None:
            return
        chat = self.peer.current_chat
        if chat is None:
            return
        current = chat.get("title") or ""
        self.push_screen(ChatTitleRenameScreen(current), self._on_chat_renamed)

    def _on_chat_renamed(self, new_title) -> None:
        """ChatTitleRenameScreen callback: apply the new title (or clear it
        back to auto-generated if left blank), or do nothing if canceled."""
        if new_title is None or self.peer is None:
            return
        chat = self.peer.current_chat
        if chat is None:
            return
        new_title = (new_title or "").strip()[:MAX_TITLE_CHARS]
        if new_title:
            chats.set_title(chat, new_title)
        else:
            chats.set_title(chat, None)
        self._refresh_chat_tab_title()
        log(f"Chat renamed to: '{new_title or '(default)'}'", node_id=self.peer.peer_id, msg_type="text")
    
    async def _populate_chat_tabs(self) -> None:
        """Mount tabs for every previously-open chat plus a fresh empty one,
        then the [+]/☰ button pair. Called once from run.py's run_peer
        after Peer() has loaded the peer's chats."""
        if self.peer is None:
            return

        tabbed = self.query_one("#right-tabs", TabbedContent)
        self._right_spinner_active = False
        await tabbed.clear_panes()

        # 1. Restore previously-open chats, oldest-updated first so the most
        #    recent ends up rightmost. Await each mount so TabbedContent's
        #    tab index is fully populated when we later set `active`.
        previously_open = [
            (cid, c) for cid, c in self.peer.chats.items()
            if c.get("open_in_ui") and c.get("messages")
        ]
        previously_open.sort(key=lambda x: x[1].get("updated_at", ""))
        for chat_id, chat in previously_open:
            await self._mount_chat_tab(chat_id, chat)

        # 2. Mount a fresh empty chat rightmost and make it the active tab.
        empty_id = None
        for cid, c in self.peer.chats.items():
            if not c.get("messages"):
                empty_id = cid
                break
        if empty_id is None:
            empty_id = self.peer.new_chat()

        await self._mount_chat_tab(empty_id, self.peer.chats[empty_id])
        self.peer.active_chat_id = empty_id
        tabbed.active = f"tab-chat-{empty_id}"

        # 3. Mount the [+] / ☰ buttons pair.
        tabs_bar = tabbed.query_one("Tabs")
        await tabs_bar.mount(
            Horizontal(
                Static("[+]", id="new-chat-button"),
                Static("☰", id="chat-menu-button"),
                id="tab-buttons",
            )
        )
        # Deferred: right after mount() returns, the buttons' own layout
        # (outer_size) hasn't necessarily settled yet, which would make this
        # compute against a stale/zero size. call_after_refresh guarantees it
        # runs after the next full layout pass. The end-scroll (below) has to
        # wait for a *further* refresh after that — max_scroll_x won't
        # reflect the narrower width until a layout pass has run with it in
        # place — so it's chained from inside this call rather than scheduled
        # up front alongside it.
        self.call_after_refresh(self._resize_tabs_then_scroll_to_end)

    def _resize_tabs_then_scroll_to_end(self) -> None:
        """Resize the tab bar's scroll viewport, then (once that resize has
        gone through a layout pass) scroll it fully into view. Called after
        mounting/creating/reopening a chat tab, so the active tab is never
        left hidden under the [+]/☰ buttons."""
        self._resize_tabs_scroll_for_buttons()
        # The fresh empty chat (mounted rightmost, earlier in
        # _populate_chat_tabs) was made active before the tab row had its
        # final width or the restored tabs' widths had settled, so Textual's
        # own scroll-into-view could land short — with enough restored
        # chats, the active "new chat" tab ends up scrolled out of view on
        # startup. Force it fully into view once the resize above has
        # actually been reflected in a layout pass.
        self.call_after_refresh(self._scroll_chat_tabs_to_end)

    def _scroll_chat_tabs_to_end(self) -> None:
        """Force the chat tab bar's horizontal scroll all the way right."""
        try:
            tabs_scroll = self.query_one("#right-tabs Tabs #tabs-scroll")
        except Exception:
            return
        tabs_scroll.scroll_to(x=tabs_scroll.max_scroll_x, animate=False, force=True)

    def _resize_tabs_scroll_for_buttons(self) -> None:
        """The [+]/☰ buttons sit on the "overlay" layer, docked right —
        that layer has its own independent layout, so it doesn't reserve any
        space within the tab row itself. Textual's own scroll-the-active-
        tab-into-view logic (triggered every time `.active` changes — new
        chat, switching tabs, etc.) doesn't know that, and can end up
        centering a newly active tab (especially the last one, right after
        creating a new chat) partly under the buttons. Padding alone doesn't
        fix this — it changes how content is laid out inside the viewport,
        not how far the viewport can actually scroll. Shrinking the
        viewport's own width does: it genuinely reduces max_scroll_x, so
        even scrolled all the way to the end, tabs never reach the reserved
        area. Called at mount and on every resize, since it's a fixed cell
        count computed from the current terminal width."""
        try:
            tabs_bar = self.query_one("#right-tabs Tabs")
            button_pair = tabs_bar.query_one("#tab-buttons")
            tabs_scroll = tabs_bar.query_one("#tabs-scroll")
        except Exception:
            return
        tabs_scroll.styles.width = max(0, tabs_bar.size.width - button_pair.outer_size.width)

    def _mount_chat_tab(self, chat_id: str, chat: dict):
        """Mount a chat's TabPane and its selectable Log. Returns the AwaitComplete
        from TabbedContent.add_pane so async callers can await pane
        registration before referencing it (e.g. before setting
        TabbedContent.active). Sync callers can ignore the return value."""

        chats.set_ui_open(chat, True)
        title = chat.get("title") or "Chat"
        pane_id = f"tab-chat-{chat_id}"

        pane = TabPane(self._make_tab_label(title), id=pane_id)
        tabbed = self.query_one("#right-tabs", TabbedContent)
        add_result = tabbed.add_pane(pane)

        placeholder = Static("Write a message below", id=f"placeholder-{chat_id}", classes="chat-placeholder")
        if chat.get("messages"):
            placeholder.styles.display = "none"
        pane.mount(placeholder)

        # Log (not RichLog): Textual supports mouse text selection on Log only.
        output = Log(id=f"output-{chat_id}", classes="chat-output", auto_scroll=True)
        pane.mount(output)
        set_output_widget(chat_id, output)
        render_chat_history_into(chat, output)
        return add_result

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """When the user switches tabs, update which chat new submissions go to."""
        if self.peer is None:
            return
        tab_id = event.tab.id or ""
        # Textual auto-prefixes Tab ids with "--content-tab-" derived from the
        # owning TabPane's id. Strip it to recover the pane id we set.
        if tab_id.startswith("--content-tab-"):
            tab_id = tab_id[len("--content-tab-"):]
        prefix = "tab-chat-"
        if tab_id.startswith(prefix):
            chat_id = tab_id[len(prefix):]
            if chat_id in self.peer.chats:
                self.peer.active_chat_id = chat_id
                if self._pending_answer_chat_id is not None:
                    self._refresh_waiting_message()

    def _show_loading_models(self) -> None:
        """Swap the animated loading message to 'Fetching model info' once
        the AXL backend has finished initializing. This is only the cheap
        config/safetensors-header lookup (Peer.get_models_info) — the
        expert models themselves are loaded lazily, on first actual use
        (see model_worker.generate), not here."""
        self._loading_label = "Fetching model info"
        msg = self.query_one("#loading-msg", Static)
        msg.update(self._loading_label + "." * self._loading_dots)

    def _show_loading_chats(self) -> None:
        """Swap the animated loading message to 'Loading chats' (used during the
        brief window between models being loaded and chat tabs being populated)."""
        self._loading_label = "Loading chats"
        msg = self.query_one("#loading-msg", Static)
        msg.update(self._loading_label + "." * self._loading_dots)
    
    def _create_new_chat(self) -> None:
        """Create or reopen an empty chat. Precedence:
        1. An empty chat that's already an open tab -> switch to it.
        2. An empty chat known to the peer but currently closed -> mount
            its tab and activate. This is the case that used to leak
            duplicate empty chats onto disk.
        3. No empty chat exists at all -> create a fresh one."""
        if self.peer is None:
            return

        tabbed = self.query_one("#right-tabs", TabbedContent)

        # Which chats currently have tabs open? Derived from the DOM so we
        # don't need a second source of truth.
        open_ids: set[str] = set()
        for pane in tabbed.query(TabPane):
            pane_id = pane.id or ""
            if pane_id.startswith("tab-chat-"):
                open_ids.add(pane_id[len("tab-chat-"):])

        # 1. Open empty tab -> just switch.
        for chat_id, chat in self.peer.chats.items():
            if chat_id in open_ids and not chat.get("messages"):
                tabbed.active = f"tab-chat-{chat_id}"
                self.call_after_refresh(self._resize_tabs_then_scroll_to_end)
                return

        # 2. Closed (in peer.chats but no tab) empty chat -> reopen it.
        for chat_id, chat in self.peer.chats.items():
            if chat_id not in open_ids and not chat.get("messages"):
                self._mount_chat_tab(chat_id, chat)
                self.peer.active_chat_id = chat_id
                tabbed.active = f"tab-chat-{chat_id}"
                self.call_after_refresh(self._resize_tabs_then_scroll_to_end)
                return

        # 3. Nothing to reuse — create a fresh chat.
        chat_id = self.peer.new_chat()
        chat = self.peer.chats[chat_id]
        self._mount_chat_tab(chat_id, chat)
        tabbed.active = f"tab-chat-{chat_id}"
        self.call_after_refresh(self._resize_tabs_then_scroll_to_end)

    def _close_chat_tab(self, chat_id: str) -> None:
        """Close a chat's tab (marking it closed but keeping it in
        peer.chats so it can be reopened), activating a neighboring tab or
        spawning a fresh empty chat if it was the last one open. Called
        from on_click's × zone handling."""
        if self.peer is None or chat_id not in self.peer.chats:
            return

        chat = self.peer.chats[chat_id]
        is_empty = not chat.get("messages")

        tabbed = self.query_one("#right-tabs", TabbedContent)
        was_active = tabbed.active == f"tab-chat-{chat_id}"

        # Ordered list of currently-open chat ids (from the DOM).
        open_chat_ids: list[str] = []
        for pane in tabbed.query(TabPane):
            pane_id = pane.id or ""
            if pane_id.startswith("tab-chat-"):
                open_chat_ids.append(pane_id[len("tab-chat-"):])

        if chat_id not in open_chat_ids:
            return

        is_only_open = len(open_chat_ids) == 1

        # Sole, active, empty chat — closing it would just recreate it.
        if is_empty and is_only_open and was_active:
            return

        idx = open_chat_ids.index(chat_id)
        next_active_id = None
        if idx > 0:
            next_active_id = open_chat_ids[idx - 1]
        elif idx + 1 < len(open_chat_ids):
            next_active_id = open_chat_ids[idx + 1]

        if was_active and next_active_id is not None:
            try:
                tabbed.active = f"tab-chat-{next_active_id}"
            except Exception:
                pass

        self.peer.close_chat(chat_id)
        unset_output_widget(chat_id)
        tabbed.remove_pane(f"tab-chat-{chat_id}")
        

        if next_active_id is not None:
            self.peer.active_chat_id = next_active_id
        else:
            self.peer.active_chat_id = None
            self._create_new_chat()
    
    def _make_tab_label(self, title: str) -> str:
        """Tab label as a markup string. The trailing '×' is forced to gray via
        inline color markup, so it stays gray even when Tab.-active repaints
        the rest of the label blue. Bold weight in :hover / .-active states is
        inherited automatically."""
        return f"{_md_escape(title)}  [#888888]×[/]"

    def _open_chat_menu(self) -> None:
        """Open the ☰ menu listing every non-empty chat, most recent first.
        Called from on_click on the chat-menu-button."""
        if self.peer is None:
            return

        candidates: list[tuple[str, dict]] = []
        for chat_id, chat in self.peer.chats.items():
            if not chat.get("messages"):
                continue
            candidates.append((chat_id, chat))

        candidates.sort(key=lambda x: x[1].get("updated_at", ""), reverse=True)

        entries: list[tuple[str, str, str]] = []
        for chat_id, chat in candidates:
            title = chat.get("title") or "Chat"
            ts = self._format_relative_time(chat.get("updated_at", ""))
            entries.append((chat_id, title, ts))

        self.push_screen(ChatMenuScreen(entries), self._on_chat_menu_result)

    def _on_chat_menu_result(self, result) -> None:
        """result: None (canceled) | (chat_id, 'open') | (chat_id, 'delete')."""
        if result is None or self.peer is None:
            return
        chat_id, action = result

        if action == "open":
            if chat_id not in self.peer.chats:
                return
            tabbed = self.query_one("#right-tabs", TabbedContent)
            pane_id = f"tab-chat-{chat_id}"
            already_open = any(p.id == pane_id for p in tabbed.query(TabPane))
            if not already_open:
                self._mount_chat_tab(chat_id, self.peer.chats[chat_id])
            def _activate() -> None:
                """Switch to the just-(re)opened tab once it's mounted."""
                tabbed.active = pane_id
            self.call_after_refresh(_activate)
            self.peer.active_chat_id = chat_id

        elif action == "delete":
            chat = self.peer.chats.get(chat_id)
            if chat is None:
                return
            title = chat.get("title") or "Chat"
            def _on_confirm(confirmed: bool | None) -> None:
                """DeleteChatConfirmScreen callback: delete only if confirmed."""
                if confirmed:
                    self._delete_chat(chat_id)
            self.push_screen(DeleteChatConfirmScreen(title), _on_confirm)


    def _delete_chat(self, chat_id: str) -> None:
        """Permanently delete a chat: remove its tab if open, drop from
        peer.chats, unlink its disk file. If it was the currently-active
        tab, activate the left neighbor; if it was the last tab, spawn a
        fresh empty (matches _close_chat_tab's post-close behavior)."""
        if self.peer is None or chat_id not in self.peer.chats:
            return

        tabbed = self.query_one("#right-tabs", TabbedContent)
        open_chat_ids: list[str] = []
        for p in tabbed.query(TabPane):
            if p.id and p.id.startswith("tab-chat-"):
                open_chat_ids.append(p.id[len("tab-chat-"):])

        was_open = chat_id in open_chat_ids
        was_active = was_open and tabbed.active == f"tab-chat-{chat_id}"

        next_active_id = None
        if was_active:
            idx = open_chat_ids.index(chat_id)
            if idx > 0:
                next_active_id = open_chat_ids[idx - 1]
            elif idx + 1 < len(open_chat_ids):
                next_active_id = open_chat_ids[idx + 1]
            if next_active_id is not None:
                tabbed.active = f"tab-chat-{next_active_id}"

        if was_open:
            unset_output_widget(chat_id)
            tabbed.remove_pane(f"tab-chat-{chat_id}")

        # peer.delete_chat removes from self.chats and unlinks the file.
        # It also touches active_chat_id if the deleted was active, but
        # we've already switched to the neighbor above (if any).
        self.peer.delete_chat(chat_id)

        if was_active:
            if next_active_id is not None:
                self.peer.active_chat_id = next_active_id
            else:
                # Last open tab is gone — start on a fresh empty.
                self.peer.active_chat_id = None
                self._create_new_chat()

    def _format_relative_time(self, iso_ts: str) -> str:
        """Render an ISO timestamp as "just now"/"5m ago"/etc. for the ☰
        chat menu's per-row timestamps."""
        if not iso_ts:
            return ""
        try:
            ts = datetime.fromisoformat(iso_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            seconds = int((datetime.now(timezone.utc) - ts).total_seconds())
        except Exception:
            return ""
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        if days < 30:
            return f"{days}d ago"
        months = days // 30
        return f"{months}mo ago"