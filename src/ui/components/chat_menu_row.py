from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static


class ChatMenuRow(Horizontal):
    """One row in the ☰ menu: title + timestamp on the left, 🗑 on the
    right. Emits Open or Delete depending on which zone was clicked."""

    class Open(Message):
        """Posted when the label is clicked; see ChatMenuScreen.on_chat_menu_row_open."""
        def __init__(self, chat_id: str):
            """Store which chat should be opened."""
            super().__init__()
            self.chat_id = chat_id

    class Delete(Message):
        """Posted when 🗑 is clicked; see ChatMenuScreen.on_chat_menu_row_delete."""
        def __init__(self, chat_id: str):
            """Store which chat should be deleted."""
            super().__init__()
            self.chat_id = chat_id

    def __init__(self, chat_id: str, title: str, timestamp: str, **kwargs):
        """Create one row, mounted inside ChatMenuScreen's #chat-menu-list."""
        super().__init__(**kwargs)
        self._chat_id = chat_id
        self._title = title
        self._timestamp = timestamp

    def compose(self) -> ComposeResult:
        """Lay out the title/timestamp label and the delete glyph."""
        yield Static(f"{self._title:<40} {self._timestamp}",
                     classes="chat-menu-row-label")
        yield Static("🗑", classes="chat-menu-row-delete")

    def on_click(self, event) -> None:
        """Route the click to Open or Delete depending on which zone was hit."""
        target = event.control
        if target is not None and target.has_class("chat-menu-row-delete"):
            self.post_message(self.Delete(self._chat_id))
        else:
            self.post_message(self.Open(self._chat_id))
        event.stop()
