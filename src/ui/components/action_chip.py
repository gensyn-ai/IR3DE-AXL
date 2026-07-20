from rich.text import Text as RichText
from textual.message import Message
from textual.widgets import Static


class ActionChip(Static):
    """Non-toggleable chip that fires a one-shot action when clicked.
    Renders bold when `active`, plain otherwise."""

    class Triggered(Message):
        """Posted on click; handled by IR3DEApp.on_action_chip_triggered, which
        applies the "all"/"none" bulk toggle to the Logs tab's filter chips."""
        def __init__(self, action: str):
            """Store which action ("all" or "none") was triggered."""
            super().__init__()
            self.action = action

    def __init__(self, action: str, color: str, **kwargs):
        """Create the chip (used for the "all"/"none" buttons in the Logs
        tab's filter drawer, see IR3DEApp._layout_filter_chips)."""
        super().__init__(**kwargs)
        self.action = action
        self.color = color
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        """Redraw the label, bold when active."""
        style = f"bold {self.color}" if self.active else self.color
        self.update(RichText(f"[{self.action.upper()}]", style=style))

    def set_active(self, active: bool):
        """Update the active state; called by IR3DEApp._refresh_chip_states
        to keep "all"/"none" in sync with the individual filter chips."""
        if active == self.active:
            return
        self.active = active
        self._refresh_label()

    def on_click(self, event):
        """Fire the chip's action."""
        self.post_message(self.Triggered(self.action))