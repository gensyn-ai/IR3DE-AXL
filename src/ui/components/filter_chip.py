from rich.text import Text as RichText
from textual.message import Message
from textual.widgets import Static


class FilterChip(Static):
    """One log-message-type toggle in the Logs tab's filter drawer
    (IR3DEApp._layout_filter_chips), one per entry in MSG_TYPE_COLORS."""

    class Toggled(Message):
        """Posted on click; handled by IR3DEApp.on_filter_chip_toggled, which
        adds/removes msg_type from the active log filter set."""
        def __init__(self, msg_type: str, active: bool):
            """Store which message type was toggled and its new state."""
            super().__init__()
            self.msg_type = msg_type
            self.active = active

    def __init__(self, msg_type: str, color: str, **kwargs):
        """Create the chip, active (shown bold) by default."""
        super().__init__(**kwargs)
        self.msg_type = msg_type
        self.color = color
        self.active = True
        self._refresh_label()

    def _refresh_label(self):
        """Redraw the label: bold when active, dim when filtered out."""
        if self.active:
            self.update(RichText(f"[{self.msg_type.upper()}]", style=f"bold {self.color}"))
        else:
            self.update(RichText(f"[{self.msg_type.upper()}]", style=f"dim {self.color}"))

    def on_click(self, event):
        """Toggle this message type in/out of the visible log filter."""
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.msg_type, self.active))