import threading

from utils import set_log_widget
 
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static
from textual.containers import Horizontal, Vertical
from rich.text import Text


def read_css():
    with open("style.css", "r") as f:
        return f.read()

class SimApp(App):
    CSS = read_css()
    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, args, target):
        super().__init__()
        self.args = args
        self.peer = None
        self.target = target

    def compose(self) -> ComposeResult:
        yield Static("", id="top-bar")
        with Horizontal():
            yield Static("", id="border-left")
            with Vertical(id="left-pane"):
                yield RichLog(id="logs", highlight=False, markup=False, auto_scroll=True, wrap=True)
            yield Static("", id="divider")
            with Vertical(id="right-pane"):
                yield Static("[ Input panel — coming soon ]", id="right-placeholder")
            yield Static("", id="border-right")
        yield Static("", id="bottom-bar")

    def on_mount(self):
        width = self.app.size.width
        bar = "═" * width
        self.query_one("#top-bar", Static).update(bar)
        self.query_one("#bottom-bar", Static).update(bar)

        log_widget = self.query_one("#logs", RichLog)
        set_log_widget(log_widget)
        title = Text("═══ Logs ═══", style="bold #7a8aaa")
        log_widget.write(title)

        threading.Thread(
            target=self.target,
            args=(self.args, self),
            daemon=True,
            name="peer-main"
        ).start()

    async def action_quit(self):
        if self.peer is not None and self.peer.proc is not None:
            self.peer.proc.terminate()
            self.peer.proc.wait()
        self.exit()
 