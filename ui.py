import threading

from utils import set_log_widget
 
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static
from textual.containers import Horizontal, Vertical
from rich.text import Text


CSS = """
Screen {
    background: $background;
}

Horizontal {
    height: 1fr;
    background: $background;
}

#left-pane, #right-pane {
    width: 1fr;
    height: 100%;
    padding: 0 1;
    background: $background;
}

#border-left, #border-right, #divider {
    width: 1;
    height: 100%;
    background: #888888;
}

#left-pane, #right-pane {
    width: 1fr;
    height: 100%;
    padding: 0 1;
}

RichLog {
    background: $background;
    scrollbar-size: 0 0;
}

#right-placeholder {
    color: #888888;
    width: 100%;
    height: 100%;
    text-align: center;
    content-align: center middle;
    background: $background;
}

#top-bar, #bottom-bar {
    height: 1;
    width: 100%;
    color: #888888;
    background: $background;
}
"""


class SimApp(App):
    CSS = CSS
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
 