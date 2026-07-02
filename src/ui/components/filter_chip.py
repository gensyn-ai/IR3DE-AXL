from textual.widgets import Static
from textual.message import Message
from rich.text import Text as RichText


class FilterChip(Static):

    class Toggled(Message):
        def __init__(self, msg_type: str, active: bool):
            super().__init__()
            self.msg_type = msg_type
            self.active = active

    def __init__(self, msg_type: str, color: str, **kwargs):
        super().__init__(**kwargs)
        self.msg_type = msg_type
        self.color = color
        self.active = True
        self._refresh_label()

    def _refresh_label(self):
        if self.active:
            self.update(RichText(f"[{self.msg_type.upper()}]", style=f"bold {self.color}"))
        else:
            self.update(RichText(f"[{self.msg_type.upper()}]", style=f"dim {self.color}"))

    def on_click(self, event):
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.msg_type, self.active))