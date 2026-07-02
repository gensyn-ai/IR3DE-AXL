from textual.widgets import Static, TextArea, RichLog
from utils import log


def disable_input(app):
    def _disable():
        app.query_one("#user-input", TextArea).disabled = True
        app.query_one("#prompt", Static).styles.color = "#444444"
        app.query_one("#send-button", Static).disabled = True
    app.call_from_thread(_disable)


def enable_input(app, node_id):
    def _enable():
        input_widget = app.query_one("#user-input", TextArea)
        if not input_widget.disabled:
            return
        log("User input enabled!", node_id=node_id, msg_type=None, right=True)
        input_widget.disabled = False
        input_widget.focus()
        app.query_one("#prompt", Static).styles.color = "#888888"
        app.query_one("#send-button", Static).disabled = False
        app._hide_loading_msg()
    app.call_from_thread(_enable)


def disable_filters(app):
    def _disable():
        app.query_one("#filter-toggle", Static).styles.color = "#444444"
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