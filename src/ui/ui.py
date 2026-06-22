import threading

from utils import format_params, set_log_widget, set_output_widget

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, TextArea, TabbedContent, TabPane, DataTable
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual_plotext import PlotextPlot
from rich.text import Text
from utils import log, MSG_TYPE_COLORS, set_filter_predicate, ipv6_from_pubkey

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from peer import Peer


IR3DE_BANNER = (
    " ▝▀▌▛▀  ▌▛▀▀▀▀▖ ▀▀▀▀▀▖ ▌▛▀▀▀▀▖ ▛▀▀▀▀▘\n"
    "   ▌▌   ▌▌    ▌      ▌ ▌▌    ▌ ▌▌    \n"
    "   ▌▌   ▌▛▀▀▀▚   ▀▀▀▚  ▌▌    ▌ ▛▀▀▀  \n"
    "   ▌▌   ▌▌    ▌      ▌ ▌▌   ▗▌ ▌▌    \n"
    " ▝▀▘▀▀  ▘▘    ▘ ▀▀▀▀▀  ▀▀▀▀▀▘  ▀▀▀▀▀▘"
)

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
    """Non-toggleable chip that fires a one-shot action when clicked."""

    class Triggered(Message):
        def __init__(self, action: str):
            super().__init__()
            self.action = action

    def __init__(self, action: str, color: str, **kwargs):
        super().__init__(**kwargs)
        self.action = action
        self.color = color
        self.update(Text(f"[{action.upper()}]", style=f"bold {color}"))

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
            scroll_end=at_bottom,    # still always recomputed
            animate=animate,
        )
    

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

    def compose(self) -> ComposeResult:

        yield Static("", id="top-divider")
        yield Static(IR3DE_BANNER, id="ir3de-banner")
        yield Static("", id="top-bar")

        with Horizontal():

            yield Static("", id="border-left")
            with Vertical(id="left-pane"):
                with Horizontal(id="logs-header"):
                    yield Static("═══ Logs ═══", id="logs-label")
                    yield Static("filters ▾", id="filter-toggle")
                yield Vertical(id="filter-drawer")        # empty; populated dynamically
                yield FollowTailLog(id="logs", highlight=False, markup=False,
                                    auto_scroll=False, wrap=True)
                
            yield Static("", id="divider")

            with Vertical(id="right-pane"):
                with TabbedContent(id="right-tabs"):
                    
                    with TabPane("Input", id="tab-input"):
                        yield RichLog(id="output", highlight=False, markup=False,
                                    auto_scroll=True, wrap=True)
                        yield Static("", id="input-divider")
                        with Horizontal(id="input-row"):
                            yield Static("> ", id="prompt")
                            yield SubmittableTextArea(id="user-input")
                            yield Static("[SEND]", id="send-button", markup=False)
                    
                    with TabPane("Statistics", id="tab-stats"):
                        with VerticalScroll(id="stats-scroll"):
                            yield Static("Known Peers", id="peers-title", classes="stats-table-title")
                            yield DataTable(id="peers-table")
                            with Horizontal(id="stats-tables-row"):
                                with Vertical(classes="stats-table-container"):
                                    yield Static("Local models", id="models-title", classes="stats-table-title")
                                    yield DataTable(id="models-table")
                                with Vertical(classes="stats-table-container"):
                                    yield Static("Local IR3DE stats", id="stats-title", classes="stats-table-title")
                                    yield DataTable(id="ir3de-stats-table")
                            with Horizontal(id="plots-row"):
                                with Vertical(classes="plot-pane"):
                                    with Horizontal(id="tag-bars-header"):
                                        yield Static("Experts per tag", id="tag-bars-title", classes="stats-table-title")
                                        yield ToggleChip("show all", id="show-all-chip")
                                    yield PlotextPlot(id="tag-bars")
            yield Static("", id="border-right")
        yield Static("", id="bottom-bar")

    def on_mount(self):

        self._fill_bars()

        log_widget = self.query_one("#logs", RichLog)
        output_widget = self.query_one("#output", RichLog)
        set_log_widget(log_widget)
        set_output_widget(output_widget)

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

        self._col_name, self._col_pubkey, self._col_ipv6 = table.add_columns(
            "Name", "Public Key", "IPv6 address"
        )
        self._sort_column_key = self._col_name
        self._sort_reverse = False                  # default: ascending

        table.cursor_type = "row"
        table.zebra_stripes = True

        # Refresh table every 2 seconds while the app is running.
        self.set_interval(2.0, self._refresh_stats_tab)

        # Models table setup
        models_table = self.query_one("#models-table", DataTable)
        self._col_models_type, self._col_models_params, self._col_models_tags, self._col_models_num_requests = models_table.add_columns(
            "Model Type  ",
            "Parameters  ",
            "Expertise  ",
            "Num requests  "
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

        # Collect (pubkey, display tuple) pairs so we can use pk as the row key
        rows: list[tuple[str, tuple[str, str, str]]] = []
        self_row = (self.peer.public_key, (
            f"{self.peer.node_name} (self)",
            self.peer.public_key[:16] + "...",
            self.peer.ipv6_address,
        ))
        for pk, info in self.peer.known_public_keys.items():
            name = info.get("peer_name") or f"Node {info.get('peer_id', '?')}"
            try:
                ipv6 = ipv6_from_pubkey(pk)
            except ValueError:
                ipv6 = "<invalid>"
            rows.append((pk, (name, pk[:16] + "...", ipv6)))

        col_idx = {
            self._col_name:   0,
            self._col_pubkey: 1,
            self._col_ipv6:   2,
        }[self._sort_column_key]
        rows.sort(key=lambda r: r[1][col_idx].lower(), reverse=self._sort_reverse)

        # Pin self on top, sorted peers below
        all_rows = [self_row] + rows

        table.clear()
        selected_row_index = 0
        for i, (pk, display) in enumerate(all_rows):
            table.add_row(*display, key=pk)
            if pk == self._selected_peer_pubkey:
                selected_row_index = i

        self._update_header_labels()

        # Move cursor to the currently-selected pk
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
        if self.peer is not None and self.peer.proc is not None:
            self.peer.proc.terminate()
            self.peer.proc.wait()
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
        """Sync every FilterChip's visual state to current _active_filters."""
        for chip in self.query(FilterChip):
            should_be_active = chip.msg_type in self._active_filters
            if chip.active != should_be_active:
                chip.active = should_be_active
                chip._refresh_label()
    
    def _refresh_stats_tab(self):
        if self.peer is None:
            return
        self._refresh_models_table()
        self._refresh_ir3de_stats_table()
        self._refresh_peers_table()
        self._refresh_tag_bars()

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

        # Normalize both sources to the same (model_type, num_params, tags, num_requests) tuple
        if is_self:
            raw_models = [
                (
                    getattr(m["model"].config, "model_type", "unknown"),
                    sum(p.numel() for p in m["model"].parameters()),
                    m.get("tags", []) or [],
                    m.get("num_requests", 0),
                )
                for m in self.peer.models
            ]
        else:
            info = self.peer.known_public_keys.get(pk, {})
            raw_models = [
                (
                    mi.get("type", "unknown"),
                    mi.get("size", 0),
                    mi.get("tags", []) or [],
                    mi.get("num_requests", 0),
                )
                for mi in info.get("models_info", [])
            ]

        data = []
        for model_type, num_params, tags, num_requests in raw_models:
            tags_str = ", ".join(tags)
            data.append({
                "raw": {
                    self._col_models_type:         model_type.lower(),
                    self._col_models_params:       num_params,
                    self._col_models_tags:         tags_str.lower(),
                    self._col_models_num_requests: num_requests,
                },
                "display": (
                    model_type,
                    format_params(num_params),
                    tags_str,
                    str(num_requests),
                ),
            })

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
        # Immediately refresh the dependent tables so the UI updates without waiting 2s
        self._refresh_models_table()
        self._refresh_ir3de_stats_table()
        self._refresh_tag_bars()
    
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
    