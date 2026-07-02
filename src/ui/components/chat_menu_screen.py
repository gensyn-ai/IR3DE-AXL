from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.containers import VerticalScroll
from .chat_menu_row import ChatMenuRow


class ChatMenuScreen(ModalScreen[tuple[str, str] | None]):
    """Modal listing every non-empty chat. Dismisses with (chat_id, action)
    where action is 'open' or 'delete', or None on cancel."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, entries: list[tuple[str, str, str]], **kwargs):
        # entries: (chat_id, title, timestamp)
        super().__init__(**kwargs)
        self._entries = entries

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-menu-dialog"):
            yield Static("Open chat", id="chat-menu-title")
            if not self._entries:
                yield Static("No chats to open.", id="chat-menu-empty")
            else:
                with VerticalScroll(id="chat-menu-list"):
                    for chat_id, title, ts in self._entries:
                        yield ChatMenuRow(chat_id, title, ts)

    def on_chat_menu_row_open(self, event: ChatMenuRow.Open) -> None:
        self.dismiss((event.chat_id, "open"))

    def on_chat_menu_row_delete(self, event: ChatMenuRow.Delete) -> None:
        self.dismiss((event.chat_id, "delete"))

    def action_cancel(self) -> None:
        self.dismiss(None)
    
    def on_click(self, event) -> None:
        if event.control is self:
            self.dismiss(None)
