from textual.widgets import RichLog, Static, TextArea

from utils import log


def disable_input(app):
    """Gray out and lock the #user-input box + [SEND] button. Called from a
    background thread (run.py's run_peer) while the peer/AXL node is starting up."""
    def _disable():
        """Runs on the app's own thread via call_from_thread."""
        app.query_one("#user-input", TextArea).disabled = True
        app.query_one("#prompt", Static).styles.color = "#444444"
        app.query_one("#send-button", Static).disabled = True
    app.call_from_thread(_disable)


def enable_input(app, node_id):
    """Re-enable #user-input once the peer is ready; called from run.py's
    run_peer after Peer() finishes constructing and chats are loaded."""
    def _enable():
        """Runs on the app's own thread via call_from_thread."""
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
    """Gray out the Logs tab's "filters" toggle and close its drawer if open;
    called from run.py's run_peer while the peer is starting up."""
    def _disable():
        """Runs on the app's own thread via call_from_thread."""
        app.query_one("#filter-toggle", Static).styles.color = "#444444"
        drawer = app.query_one("#filter-drawer")
        if drawer.styles.display != "none":
            drawer.styles.display = "none"
            app.query_one("#filter-toggle", Static).update("filters ▾")
        app._filters_enabled = False
    app.call_from_thread(_disable)


def enable_filters(app):
    """Re-enable the Logs tab's "filters" toggle once the peer is ready;
    called from run.py's run_peer alongside enable_input."""
    def _enable():
        """Runs on the app's own thread via call_from_thread."""
        if app._filters_enabled:
            return
        app.query_one("#filter-toggle", Static).styles.color = "#888888"
        app._filters_enabled = True
    app.call_from_thread(_enable)


def read_css():
    """Load style.css for IR3DEApp's CSS class attribute (evaluated at import time)."""
    with open("src/ui/style.css", "r") as f:
        return f.read()


class FollowTailLog(RichLog):
    """RichLog that only auto-scrolls when the user is already at the bottom.
    Used for the Logs tab's #logs widget, so scrolling up to read past log
    lines doesn't get yanked back down by the next incoming one."""

    def write(
        self,
        content,
        width=None,
        expand=False,
        shrink=True,
        scroll_end=None,
        animate=False,
    ):
        """Same as RichLog.write, but auto-scroll only if already at bottom."""
        at_bottom = self.scroll_y >= self.max_scroll_y - 1
        return super().write(
            content,
            width=width,
            expand=expand,
            shrink=shrink,
            scroll_end=at_bottom,
            animate=animate,
        )