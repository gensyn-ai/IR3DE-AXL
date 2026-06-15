import threading

from utils import set_log_widget, set_output_widget

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, TextArea
from textual.containers import Horizontal, Vertical
from textual.message import Message
from rich.text import Text
from utils import log, MSG_TYPE_COLORS, set_filter_predicate


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
            return                       # idempotent, like enable_input
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


class SimApp(App):
    CSS = read_css()
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, args, logs_function, input_handler):
        super().__init__()
        self.args = args
        self.peer = None
        self.logs_function = logs_function
        self.input_handler = input_handler
        self._active_filters: set[str] = set(MSG_TYPE_COLORS.keys()) | {"no-tag"}
        self._filters_enabled = False

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
                yield Static("═══ Input ═══", id="logs-label")
                yield RichLog(id="output", highlight=False, markup=False, auto_scroll=True, wrap=True)
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
        output_widget = self.query_one("#output", RichLog)
        set_log_widget(log_widget)
        set_output_widget(output_widget)

        self.query_one("#user-input", TextArea).focus()  # <-- give input focus

        self.call_after_refresh(self._fill_input_divider)

        threading.Thread(
            target=self.logs_function,
            args=(self, self.args),
            daemon=True,
            name="logs-thread"
        ).start()
    
        set_filter_predicate(self._should_show_log)

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
        self._fill_input_divider()                  # ← duplicate removed
        drawer = self.query_one("#filter-drawer")
        if drawer.styles.display != "none":         # ← only relayout when open
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
                return                       # silently ignore clicks while disabled
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
        if is_hidden:                               # just opened
            self.call_after_refresh(self._layout_filter_chips)
    
    def _layout_filter_chips(self):
        drawer = self.query_one("#filter-drawer", Vertical)
        if drawer.styles.display == "none":
            return                                       # ← bail when hidden
        avail = drawer.size.width
        if avail <= 0:
            self.call_after_refresh(self._layout_filter_chips)
            return

        # Tear down the previous rows
        for child in list(drawer.children):
            child.remove()

        # Spec list: (kind, label, color).
        #   'action' → ActionChip (one-shot click, no toggle state)
        #   'filter' → FilterChip (toggleable membership in _active_filters)
        chip_specs: list[tuple[str, str, str]] = [
            ("action", "all",    "#ffffff"),
            ("action", "none",   "#ffffff"),
            ("filter", "no-tag", "#ffffff"),
        ] + [("filter", name, color) for name, color in MSG_TYPE_COLORS.items()]

        # Group chips into rows that fit horizontally
        chip_margin = 1                                  # must match CSS margin-right
        rows: list[list[tuple[str, str, str]]] = []
        current, used = [], 0
        for kind, name, color in chip_specs:
            chip_w = len(name) + 2 + chip_margin         # "[name]" + margin
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
            return                           # unknown action, ignore

        self._refresh_chip_states()
        self._rerender_logs()

    def _refresh_chip_states(self):
        """Sync every FilterChip's visual state to current _active_filters."""
        for chip in self.query(FilterChip):
            should_be_active = chip.msg_type in self._active_filters
            if chip.active != should_be_active:
                chip.active = should_be_active
                chip._refresh_label()


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
