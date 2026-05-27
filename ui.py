import threading

from utils import set_log_widget, set_output_widget

from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static, Input
from textual.containers import Horizontal, Vertical
from rich.text import Text
from utils import log


def disable_input(app):
    app.call_from_thread(lambda: setattr(app.query_one("#user-input", Input), "disabled", True))


def enable_input(app, node_id):
    input_widget = app.query_one("#user-input", Input)
    if input_widget.disabled:
        log("User input enabled!", node_id=node_id, msg_type=None, right=True)
        def _enable():
            input_widget.disabled = False
            input_widget.focus()
        app.call_from_thread(_enable)


def read_css():
    with open("style.css", "r") as f:
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
        yield Static("", id="top-bar")
        with Horizontal():
            yield Static("", id="border-left")
            with Vertical(id="left-pane"):
                yield RichLog(id="logs", highlight=False, markup=False, auto_scroll=True, wrap=True)
            yield Static("", id="divider")
            with Vertical(id="right-pane"):
                yield RichLog(id="output", highlight=False, markup=False, auto_scroll=True, wrap=True)
                yield Input(id="user-input")
            yield Static("", id="border-right")
        yield Static("", id="bottom-bar")

    def on_mount(self):
        width = self.app.size.width
        bar = "═" * width
        self.query_one("#top-bar", Static).update(bar)
        self.query_one("#bottom-bar", Static).update(bar)

        log_widget = self.query_one("#logs", RichLog)
        output_widget = self.query_one("#output", RichLog)
        set_log_widget(log_widget)
        set_output_widget(output_widget)
        log_widget.write(Text("═══ Logs ═══", style="bold blue"))
        output_widget.write(Text("═══ Input ═══", style="bold blue"))

        self.query_one("#user-input", Input).focus()  # <-- give input focus

        threading.Thread(
            target=self.logs_function,
            args=(self, self.args),
            daemon=True,
            name="logs-thread"
        ).start()

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
