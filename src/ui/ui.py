import threading

from utils import set_log_widget, set_output_widget

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, Input
from textual.containers import Horizontal, Vertical
from rich.text import Text
from utils import log


IR3DE_BANNER = (
    " ▝▀▌▛▀  ▌▛▀▀▀▀▖ ▀▀▀▀▀▖ ▌▛▀▀▀▀▖ ▛▀▀▀▀▘\n"
    "   ▌▌   ▌▌    ▌      ▌ ▌▌    ▌ ▌▌    \n"
    "   ▌▌   ▌▛▀▀▀▚   ▀▀▀▚  ▌▌    ▌ ▛▀▀▀  \n"
    "   ▌▌   ▌▌    ▌      ▌ ▌▌   ▗▌ ▌▌    \n"
    " ▝▀▘▀▀  ▘▘    ▘ ▀▀▀▀▀  ▀▀▀▀▀▘  ▀▀▀▀▀▘"
)

def disable_input(app):
    def _disable():
        app.query_one("#user-input", Input).disabled = True
        app.query_one("#prompt", Static).styles.color = "#444444"
    app.call_from_thread(_disable)


def enable_input(app, node_id):
    input_widget = app.query_one("#user-input", Input)
    if input_widget.disabled:
        log("User input enabled!", node_id=node_id, msg_type=None, right=True)
        def _enable():
            input_widget.disabled = False
            input_widget.focus()
            app.query_one("#prompt", Static).styles.color = "#888888"
        app.call_from_thread(_enable)


def read_css():
    with open("src/ui/style.css", "r") as f:
        return f.read()


class SimApp(App):
    CSS = read_css()
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, args, logs_function, input_handler):
        super().__init__()
        self.args = args
        self.peer = None
        self.logs_function = logs_function
        self.input_handler = input_handler

    def compose(self) -> ComposeResult:
        yield Static("", id="top-divider")
        yield Static(IR3DE_BANNER, id="ir3de-banner")
        yield Static("", id="top-bar")
        with Horizontal():
            yield Static("", id="border-left")
            with Vertical(id="left-pane"):
                yield FollowTailLog(id="logs", highlight=False, markup=False, auto_scroll=False, wrap=True)
            yield Static("", id="divider")
            with Vertical(id="right-pane"):
                yield RichLog(id="output", highlight=False, markup=False, auto_scroll=True, wrap=True)
                yield Static("", id="input-divider")
                with Horizontal(id="input-row"):
                    yield Static("> ", id="prompt")
                    yield Input(id="user-input")
            yield Static("", id="border-right")
        yield Static("", id="bottom-bar")

    def on_mount(self):
        self._fill_bars()

        log_widget = self.query_one("#logs", RichLog)
        output_widget = self.query_one("#output", RichLog)
        set_log_widget(log_widget)
        set_output_widget(output_widget)
        log_widget.write(Text("═══ Logs ═══", style="bold blue"))
        output_widget.write(Text("═══ Input ═══", style="bold blue"))

        self.query_one("#user-input", Input).focus()  # <-- give input focus

        self.call_after_refresh(self._fill_input_divider)

        threading.Thread(
            target=self.logs_function,
            args=(self, self.args),
            daemon=True,
            name="logs-thread"
        ).start()
    
    def _fill_input_divider(self):
        div = self.query_one("#input-divider", Static)
        div.update("═" * div.size.width)

    def on_input_submitted(self, event: Input.Submitted):
        user_text = event.value
        event.input.value = ""
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
    
    def _fill_bars(self):
        bar = "═" * self.app.size.width
        for wid in ("#top-divider", "#top-bar", "#bottom-bar"):
            self.query_one(wid, Static).update(bar)


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