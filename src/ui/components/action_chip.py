from textual.widgets import Static
from textual.message import Message
from rich.text import Text as RichText


class ActionChip(Static):
    """Non-toggleable chip that fires a one-shot action when clicked.
    Renders bold when `active`, plain otherwise."""

    class Triggered(Message):
        def __init__(self, action: str):
            super().__init__()
            self.action = action

    def __init__(self, action: str, color: str, **kwargs):
        super().__init__(**kwargs)
        self.action = action
        self.color = color
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        style = f"bold {self.color}" if self.active else self.color
        self.update(RichText(f"[{self.action.upper()}]", style=style))

    def set_active(self, active: bool):
        if active == self.active:
            return
        self.active = active
        self._refresh_label()

    def on_click(self, event):
        self.post_message(self.Triggered(self.action))