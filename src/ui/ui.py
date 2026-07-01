import os, random, threading, statistics
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, TextArea, TabbedContent, TabPane, DataTable, OptionList, Input
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual_plotext import PlotextPlot
from textual.screen import ModalScreen
from textual.events import MouseDown, MouseUp, MouseMove
from rich.text import Text

from utils import (format_params, format_mean_std, set_log_widget, set_output_widget, log, MSG_TYPE_COLORS,
                   set_filter_predicate, ipv6_from_pubkey, symbol_for_tag, MAX_TITLE_CHARS, render_chat_history_into)
from ui.glyphs import DIAMOND_FRAMES, DIAMOND_ROTATION, IR3DE_BANNER

if TYPE_CHECKING:
    from peer import Peer


def disable_input(app):
    def _disable():
        app.query_one("#user-input", TextArea).disabled = True
        app.query_one("#prompt", Static).styles.color = "#444444"
        app.query_one("#send-button", Static).disabled = True
    app.call_from_thread(_disable)


def enable_input(app, node_id):
    input_widget = app.query_one("#user-input", TextArea)
    if input_widget.disabled:
        log("User input enabled!", node_id=node_id, msg_type=None, right=True)
        def _enable():
            input_widget.disabled = False
            input_widget.focus()
            app.query_one("#prompt", Static).styles.color = "#888888"
            app.query_one("#send-button", Static).disabled = False
            app._hide_loading_msg()
        app.call_from_thread(_enable)


def disable_filters(app):
    def _disable():
        app.query_one("#filter-toggle", Static).styles.color = "#444444"
        # Force the drawer closed
        drawer = app.query_one("#filter-drawer")
        if drawer.styles.display != "none":
            drawer.styles.display = "none"
            app.query_one("#filter-toggle", Static).update("filters ▾")
        app._filters_enabled = False
    app.call_from_thread(_disable)


def enable_filters(app):
    def _enable():
        if app._filters_enabled:
            return
        app.query_one("#filter-toggle", Static).styles.color = "#888888"
        app._filters_enabled = True
    app.call_from_thread(_enable)


def read_css():
    with open("src/ui/style.css", "r") as f:
        return f.read()


class SubmittableTextArea(TextArea):
    
    class Submitted(Message):
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def submit(self):
        value = self.text
        self.text = ""
        self.post_message(self.Submitted(value))

    def on_key(self, event):
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.submit()
        elif event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")


class ActionChip(Static):
    """Non-toggleable chip that fires a one-shot action when clicked.
    Renders bold when `active`, plain otherwise."""

    class Triggered(Message):
        def __init__(self, action: str):
            super().__init__()
            self.action = action

    def __init__(self, action: str, color: str, **kwargs):
        super().__init__(**kwargs)
        self.action = action
        self.color = color
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        style = f"bold {self.color}" if self.active else self.color
        self.update(Text(f"[{self.action.upper()}]", style=style))

    def set_active(self, active: bool):
        if active == self.active:
            return
        self.active = active
        self._refresh_label()

    def on_click(self, event):
        self.post_message(self.Triggered(self.action))


class FilterChip(Static):

    class Toggled(Message):
        def __init__(self, msg_type: str, active: bool):
            super().__init__()
            self.msg_type = msg_type
            self.active = active

    def __init__(self, msg_type: str, color: str, **kwargs):
        super().__init__(**kwargs)
        self.msg_type = msg_type
        self.color = color
        self.active = True
        self._refresh_label()

    def _refresh_label(self):
        if self.active:
            self.update(Text(f"[{self.msg_type.upper()}]", style=f"bold {self.color}"))
        else:
            self.update(Text(f"[{self.msg_type.upper()}]", style=f"dim {self.color}"))

    def on_click(self, event):
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.msg_type, self.active))


class ToggleChip(Static):
    """Chip that toggles between plain and bold styling on click."""

    class Toggled(Message):
        def __init__(self, chip_id: str, active: bool):
            super().__init__()
            self.chip_id = chip_id
            self.active = active

    def __init__(self, label: str, color: str = "#ffffff", **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.color = color
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        style = f"bold {self.color}" if self.active else self.color
        self.update(Text(f"[{self.label.upper()}]", style=style))

    def on_click(self, event):
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.id or "", self.active))


class FollowTailLog(RichLog):
    """RichLog that only auto-scrolls when the user is already at the bottom."""

    def write(
        self,
        content,
        width=None,
        expand=False,
        shrink=True,
        scroll_end=None,
        animate=False,
    ):
        at_bottom = self.scroll_y >= self.max_scroll_y - 1
        return super().write(
            content,
            width=width,
            expand=expand,
            shrink=shrink,
            scroll_end=at_bottom,
            animate=animate,
        )


class TagButton(Static):

    class Toggled(Message):
        def __init__(self, tag: str, active: bool):
            super().__init__()
            self.tag = tag
            self.active = active

    def __init__(self, tag: str, symbol: str = "◆", **kwargs):
        super().__init__(**kwargs)
        self.tag = tag
        self.symbol = symbol
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        check = "✓" if self.active else " "
        # Pad symbol to 3 cells so glyphs like "</>" and "λ" line up the same.
        sym = self.symbol.ljust(3)
        self.update(Text(f" {sym} {self.tag.capitalize():<12} [{check}]"))
        if self.active:
            self.add_class("-active")
        else:
            self.remove_class("-active")

    def on_click(self, event):
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.tag, self.active))


class ModelRow(Horizontal):
    """One selectable model row inside an ExpertiseSection."""

    class Selected(Message):
        def __init__(self, tag: str, peer_pk: str, model_idx: int):
            super().__init__()
            self.tag = tag
            self.peer_pk = peer_pk
            self.model_idx = model_idx

    def __init__(self, tag: str, peer_pk: str, model_idx: int,
                 name: str, node_name: str, params: int,
                 active: bool = False, **kwargs):
        # Inject "-active" into the classes arg so the CSS engine sees it
        # from the very first render — no on_mount timing dance.
        classes_arg = kwargs.pop("classes", "")
        if active:
            classes_arg = f"{classes_arg} -active".strip()
        if classes_arg:
            kwargs["classes"] = classes_arg
        super().__init__(**kwargs)

        self._tag = tag
        self._peer_pk = peer_pk
        self._model_idx = model_idx
        self._name = name
        self._node_name = node_name
        self._params = params
        self._active = active

    def compose(self) -> ComposeResult:
        yield Static("●" if self._active else "○", classes="model-row-glyph")
        yield Static(self._name, classes="model-row-name")
        yield Static(self._node_name, classes="model-row-node")
        yield Static(format_params(self._params), classes="model-row-params")

    def set_active(self, active: bool):
        if active == self._active:
            return
        self._active = active
        if active:
            self.add_class("-active")
        else:
            self.remove_class("-active")
        self.query_one(".model-row-glyph", Static).update("●" if active else "○")

    def on_click(self, event):
        self.post_message(self.Selected(self._tag, self._peer_pk, self._model_idx))


class ExpertiseSection(Vertical):
    """A bordered card listing models for one selected expertise tag."""

    def __init__(self, tag: str, symbol: str,
                 candidates: list[tuple[str, int, str, str, int]],
                 selected: tuple[str, int] | None,
                 **kwargs):
        super().__init__(**kwargs)
        self._tag = tag
        self._symbol = symbol
        self._candidates = candidates
        self._selected = selected
        self._rows_container = Vertical(classes="expertise-section-rows")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="expertise-section-header"):
            yield Static(f"{self._symbol.ljust(3)} {self._tag.capitalize()}",
                         classes="expertise-section-title")
            yield Static("[SELECTED]", classes="expertise-section-badge")
        yield Static(f"Select the desired model for {self._tag} tasks.",
                     classes="expertise-section-desc")
        yield self._rows_container

    def on_mount(self) -> None:
        # Rows are added here — at this point _rows_container is in the tree.
        self._rebuild_rows()
        self._update_badge()

    def update_state(self, candidates: list[tuple[str, int, str, str, int]], selected: tuple[str, int] | None) -> None:
        
        candidates_changed = candidates != self._candidates
        selected_changed = selected != self._selected

        if not candidates_changed and not selected_changed:
            return

        self._candidates = candidates
        self._selected = selected

        if candidates_changed:
            self._rebuild_rows()
        else:
            for row in self._rows_container.query(ModelRow):
                is_active = self._selected is not None and (row._peer_pk, row._model_idx) == self._selected
                row.set_active(is_active)

        self._update_badge()

    def _rebuild_rows(self) -> None:
        for child in list(self._rows_container.children):
            child.remove()

        if not self._candidates:
            self._rows_container.mount(
                Static("No available models for now.", classes="no-models-msg")
            )
            return

        for pk, idx, name, node, params in self._candidates:
            is_active = self._selected is not None and (pk, idx) == self._selected
            self._rows_container.mount(
                ModelRow(self._tag, pk, idx, name, node, params, active=is_active)
            )

    def _update_badge(self) -> None:
        badge = self.query_one(".expertise-section-badge", Static)
        badge.styles.display = "block" if self._selected is not None else "none"

    def set_has_selection(self, has: bool) -> None:
        # Kept for backwards compatibility with callers that still use it.
        badge = self.query_one(".expertise-section-badge", Static)
        badge.styles.display = "block" if has else "none"


class LatencyMetricSelectScreen(ModalScreen[str]):
    """Popup that returns one of: 'prompt' | 'total' | 'input'."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    OPTIONS = [
        ("Prompt latency",        "prompt"),
        ("Latency / total token", "total"),
        ("Latency / input token", "input"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="latency-metric-dialog"):
            yield Static("Choose latency metric", id="latency-metric-dialog-title")
            yield OptionList(
                *[label for label, _ in self.OPTIONS],
                id="latency-metric-options",
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.OPTIONS[event.option_index][1])

    def action_cancel(self) -> None:
        self.dismiss(None)


class ResizableDivider(Static):
    """1-cell vertical bar between #left-pane and #right-pane that the
    user can click-and-drag to resize the two panes."""

    MIN_PANE_WIDTH = 20    # don't let either pane shrink below this

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dragging = False

    def on_mouse_down(self, event: MouseDown) -> None:
        # Pin both panes to their current pixel widths so subsequent
        # adjustments work in concrete cells, not fr units.
        app = self.app
        left  = app.query_one("#left-pane")
        right = app.query_one("#right-pane")
        left.styles.width  = left.size.width
        right.styles.width = right.size.width

        self._dragging = True
        self.capture_mouse()
        event.stop()

    def on_mouse_up(self, event: MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
            event.stop()

    def on_mouse_move(self, event: MouseMove) -> None:
        if not self._dragging:
            return

        app = self.app
        left  = app.query_one("#left-pane")
        right = app.query_one("#right-pane")

        # The outer Horizontal contains:
        #   #border-left (1) | #left-pane | #divider (1) | #right-pane | #border-right (1)
        # So the cells available to the two panes total app.width - 3.
        total_inner = app.size.width - 3

        # Place the divider directly under the cursor: left pane spans from
        # the first column after #border-left up to (but not including) the divider.
        new_left = event.screen_x - 1
        max_left = total_inner - self.MIN_PANE_WIDTH
        new_left = max(self.MIN_PANE_WIDTH, min(new_left, max_left))

        left.styles.width  = new_left
        right.styles.width = total_inner - new_left
        event.stop()

class ChatTitleRenameScreen(ModalScreen):
    """Single-field modal that returns the new chat title, or None if cancelled."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current_title: str = "", **kwargs):
        super().__init__(**kwargs)
        self._current_title = current_title

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-rename-dialog"):
            yield Static("Rename chat", id="chat-rename-title")
            yield Input(value=self._current_title,
                        placeholder="Chat title (Enter to confirm, Esc to cancel)",
                        id="chat-rename-input",
                        max_length=MAX_TITLE_CHARS)

    def on_mount(self) -> None:
        self.query_one("#chat-rename-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)

class SimApp(App):
    CSS = read_css()
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, args, logs_function, input_handler):
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

    def compose(self) -> ComposeResult:

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
                yield Static("", id="input-divider")
                with Horizontal(id="input-row"):
                    yield Static("> ", id="prompt")
                    yield SubmittableTextArea(id="user-input")
                    yield Static("[SEND]", id="send-button", markup=False)

            yield Static("", id="border-right")

        yield Static("", id="bottom-bar")

    def on_mount(self):

        self._fill_bars()

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
        arrow = "▲" if reverse else "▼"
        for col_key, base in columns:
            suffix = f" {arrow}" if col_key == active_key else "  "
            text = f"{base}{suffix}"
            table.columns[col_key].label = Text(text)
        table.refresh()

    
    def _refresh_peers_table(self):
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
        table = self.query_one("#peers-table", DataTable)
        arrow = "▲" if self._sort_reverse else "▼"
        labels = {
            self._col_name:   "Name",
            self._col_pubkey: "Public Key",
            self._col_ipv6:   "IPv6 address",
        }
        for col_key, base in labels.items():
            text = f"{base} {arrow}" if col_key == self._sort_column_key else base
            table.columns[col_key].label = Text(text)
        table.refresh()

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected):
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
        key = "no-tag" if msg_type is None else msg_type
        return key in self._active_filters

    def on_filter_chip_toggled(self, event: FilterChip.Toggled):
        if event.active:
            self._active_filters.add(event.msg_type)
        else:
            self._active_filters.discard(event.msg_type)
        self._refresh_chip_states()
        self._rerender_logs()

    def _rerender_logs(self):
        from utils import iter_log_buffer
        widget = self.query_one("#logs", FollowTailLog)
        widget.clear()
        for ts, node, mtype, line in iter_log_buffer():
            if self._should_show_log(mtype):
                widget.write(line)

    def _fill_input_divider(self):
        div = self.query_one("#input-divider", Static)
        div.update("═" * div.size.width)

    def on_submittable_text_area_submitted(self, event: SubmittableTextArea.Submitted):
        user_text = event.value
        if self.input_handler is not None:
            self.input_handler(self, self.args, user_text)

    async def action_quit(self):
        if self.peer is not None:
            self.peer.stop_node()
        self.exit()

    def on_resize(self, event):
        self._fill_bars()
        self._fill_input_divider()
        drawer = self.query_one("#filter-drawer")
        if drawer.styles.display != "none":
            self._layout_filter_chips()
    
    def _fill_bars(self):
        bar = "═" * self.app.size.width
        for wid in ("#top-divider", "#top-bar", "#bottom-bar"):
            self.query_one(wid, Static).update(bar)

    def on_click(self, event):

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
        elif event.control.id == "latency-metric-button":
            self.push_screen(LatencyMetricSelectScreen(), self._on_latency_metric_chosen)
        elif event.control.id == "new-chat-button":
            self._create_new_chat()
        
        if event.chain == 2:
            node = event.control
            for _ in range(6):                          # walk a few levels up
                if node is None:
                    break
                node_id = getattr(node, "id", None) or ""
                # match the Tab generated for our "tab-chat" TabPane, but not the
                # scroll container or anything else that happens to contain it
                if "tab-chat" in node_id and "scroll" not in node_id and node_id != "tab-chat":
                    self._open_chat_rename_dialog()
                    return
                # also accept a direct click on the TabPane label area
                if node_id == "tab-chat":
                    self._open_chat_rename_dialog()
                    return
                node = getattr(node, "parent", None)

    def _toggle_drawer(self):
        drawer = self.query_one("#filter-drawer")
        toggle = self.query_one("#filter-toggle", Static)
        is_hidden = drawer.styles.display == "none"
        drawer.styles.display = "block" if is_hidden else "none"
        toggle.update("filters ▴" if is_hidden else "filters ▾")
        if is_hidden:
            self.call_after_refresh(self._layout_filter_chips)
    
    def _layout_filter_chips(self):
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

        data.sort(key=lambda r: r["raw"][self._models_sort_column_key],
                reverse=self._models_sort_reverse)

        table.clear()
        for r in data:
            table.add_row(*r["display"])

        columns = [
            (self._col_models_type,         "Model Type"),
            (self._col_models_params,       "Parameters"),
            (self._col_models_tags,         "Expertise"),
            (self._col_models_num_requests, "Num requests"),
        ]

        self._update_sort_arrows(table, columns,
                                self._models_sort_column_key, self._models_sort_reverse)

        # Synchronous restore — eliminates the "snap to 0" frame.
        table.scroll_to(x=saved_x, y=saved_y, animate=False)
        self.call_after_refresh(lambda: table.scroll_to(x=saved_x, y=saved_y, animate=False))

    def _refresh_ir3de_stats_table(self):
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
        assert self.peer is not None
        if pk == self.peer.public_key:
            return self.peer.node_name
        info = self.peer.known_public_keys.get(pk, {})
        return info.get("peer_name") or f"Node {info.get('peer_id', '?')}"

    def _refresh_tag_bars(self):

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
            assert self.peer is not None
            for m in self.peer.models:
                model_tag_lists.append(list(m.get("tags", []) or []))

        def add_models_from_remote(info: dict):
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
        if event.chip_id == "show-all-chip":
            self._show_all_experts = event.active
            self._refresh_tag_bars()
        elif event.chip_id == "show-all-latency-chip":
            self._show_all_latencies = event.active
            self._refresh_latency_plot()
    

    def _refresh_tag_buttons(self):
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
        button_w = 24                       # must roughly match CSS width
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

        if event.active:
            self._selected_tags.add(event.tag)
        else:
            self._selected_tags.discard(event.tag)

        if self.peer is not None:
            self.peer.selected_tags = set(self._selected_tags)

        self._refresh_expertise_sections()

    def _refresh_expertise_sections(self):
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
        # Chat tab loading line (existing)
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
        self._control_spinner_frame = (self._control_spinner_frame + 1) % len(DIAMOND_ROTATION)
        frame = DIAMOND_FRAMES[DIAMOND_ROTATION[self._control_spinner_frame]]
        for spinner_id in ("#control-spinner", "#stats-spinner", "#right-spinner"):
            self.query_one(spinner_id, Static).update(frame)

    def _refresh_latency_plot(self):
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
        """Make every right-pane tab label mirror its chat's title (or 'Chat' if
        no title has been generated/set yet)."""
        if self.peer is None:
            return
        tabbed = self.query_one("#right-tabs", TabbedContent)
        for chat_id, chat in self.peer.chats.items():
            label = chat.get("title") or "Chat"
            try:
                tab = tabbed.get_tab(f"tab-chat-{chat_id}")
                if str(tab.label) != label:
                    tab.label = label
            except Exception:
                continue

    def _open_chat_rename_dialog(self) -> None:
        if self.peer is None:
            return
        chat = self.peer.current_chat
        if chat is None:
            return
        current = chat.get("title") or ""
        self.push_screen(ChatTitleRenameScreen(current), self._on_chat_renamed)

    def _on_chat_renamed(self, new_title) -> None:
        if new_title is None or self.peer is None:
            return
        chat = self.peer.current_chat
        if chat is None:
            return
        import chats
        new_title = (new_title or "").strip()[:MAX_TITLE_CHARS]
        if new_title:
            chats.set_title(chat, new_title)
        else:
            chats.set_title(chat, None)
        self._refresh_chat_tab_title()
        log(f"Chat renamed to: '{new_title or '(default)'}'", node_id=self.peer.peer_id, msg_type="text")
    
    async def _populate_chat_tabs(self) -> None:

        if self.peer is None:
            return

        tabbed = self.query_one("#right-tabs", TabbedContent)
        await tabbed.remove_pane("tab-chat-placeholder")

        for chat_id, chat in self.peer.chats.items():
            self._mount_chat_tab(chat_id, chat)

        if self.peer.active_chat_id is not None:
            tabbed.active = f"tab-chat-{self.peer.active_chat_id}"

        tabs_bar = tabbed.query_one("Tabs")
        tabs_bar.mount(Static("[+]", id="new-chat-button"))

    def _mount_chat_tab(self, chat_id: str, chat: dict) -> None:
        """Create a new TabPane for `chat`, mount it, register its RichLog
        with utils._output_widgets, and replay the chat's history into it."""

        title = chat.get("title") or "Chat"
        pane_id = f"tab-chat-{chat_id}"

        pane = TabPane(title, id=pane_id)
        # Add the pane and its RichLog
        tabbed = self.query_one("#right-tabs", TabbedContent)
        tabbed.add_pane(pane)
        output = RichLog(id=f"output-{chat_id}", classes="chat-output",
                        highlight=False, markup=False,
                        auto_scroll=True, wrap=True)
        pane.mount(output)

        set_output_widget(chat_id, output)
        render_chat_history_into(chat, output)

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

    def _show_loading_models(self) -> None:
        """Swap the animated loading message to 'Loading local models' once
        the AXL backend has finished initialising."""
        self._loading_label = "Loading local models"
        msg = self.query_one("#loading-msg", Static)
        msg.update(self._loading_label + "." * self._loading_dots)

    def _show_loading_chats(self) -> None:
        """Swap the animated loading message to 'Loading chats' (used during the
        brief window between models being loaded and chat tabs being populated)."""
        self._loading_label = "Loading chats"
        msg = self.query_one("#loading-msg", Static)
        msg.update(self._loading_label + "." * self._loading_dots)
    
    def _create_new_chat(self) -> None:
        """Create a new empty chat and activate it, unless there's already an
        empty chat (no messages) — in that case just switch to it."""
        if self.peer is None:
            return

        tabbed = self.query_one("#right-tabs", TabbedContent)

        # If any existing chat has no messages, just jump to it instead of
        # creating another empty one.
        for existing_id, existing_chat in self.peer.chats.items():
            if not existing_chat.get("messages"):
                tabbed.active = f"tab-chat-{existing_id}"
                return

        chat_id = self.peer.new_chat()
        chat = self.peer.chats[chat_id]
        self._mount_chat_tab(chat_id, chat)

        tabbed.active = f"tab-chat-{chat_id}"
