from textual.widgets import Static
from textual.events import MouseDown, MouseUp, MouseMove


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
