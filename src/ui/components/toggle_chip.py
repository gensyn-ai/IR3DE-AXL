from textual.widgets import Static
from textual.message import Message
from rich.text import Text as RichText


class ToggleChip(Static):
    """Chip that toggles between plain and bold styling on click."""

    class Toggled(Message):
        def __init__(self, chip_id: str, active: bool):
            super().__init__()
            self.chip_id = chip_id
            self.active = active

    def __init__(self, label: str, color: str = "#ffffff", **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.color = color
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        style = f"bold {self.color}" if self.active else self.color
        self.update(RichText(f"[{self.label.upper()}]", style=style))

    def on_click(self, event):
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.id or "", self.active))
