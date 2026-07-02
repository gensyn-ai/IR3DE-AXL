from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static


class ChatMenuRow(Horizontal):
    """One row in the ☰ menu: title + timestamp on the left, 🗑 on the
    right. Emits Open or Delete depending on which zone was clicked."""

    class Open(Message):
        def __init__(self, chat_id: str):
            super().__init__()
            self.chat_id = chat_id

    class Delete(Message):
        def __init__(self, chat_id: str):
            super().__init__()
            self.chat_id = chat_id

    def __init__(self, chat_id: str, title: str, timestamp: str, **kwargs):
        super().__init__(**kwargs)
        self._chat_id = chat_id
        self._title = title
        self._timestamp = timestamp

    def compose(self) -> ComposeResult:
        yield Static(f"{self._title:<40} {self._timestamp}",
                     classes="chat-menu-row-label")
        yield Static("🗑", classes="chat-menu-row-delete")

    def on_click(self, event) -> None:
        target = event.control
        if target is not None and target.has_class("chat-menu-row-delete"):
            self.post_message(self.Delete(self._chat_id))
        else:
            self.post_message(self.Open(self._chat_id))
        event.stop()
