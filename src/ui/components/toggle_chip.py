from rich.text import Text as RichText
from textual.message import Message
from textual.widgets import Static


class ToggleChip(Static):
    """Chip that toggles between plain and bold styling on click."""

    class Toggled(Message):
        """Posted on click; handled by IR3DEApp.on_toggle_chip_toggled, which
        drives the "show all" toggles above the Statistics tab's two plots."""
        def __init__(self, chip_id: str, active: bool):
            """Store which chip toggled and its new state."""
            super().__init__()
            self.chip_id = chip_id
            self.active = active

    def __init__(self, label: str, color: str = "#ffffff", **kwargs):
        """Create the chip, inactive by default."""
        super().__init__(**kwargs)
        self.label = label
        self.color = color
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        """Redraw the label, bold when active."""
        style = f"bold {self.color}" if self.active else self.color
        self.update(RichText(f"[{self.label.upper()}]", style=style))

    def on_click(self, event):
        """Flip the toggle state."""
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.id or "", self.active))
