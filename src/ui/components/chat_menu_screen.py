from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from .chat_menu_row import ChatMenuRow


class ChatMenuScreen(ModalScreen[tuple[str, str] | None]):
    """Modal listing every non-empty chat. Dismisses with (chat_id, action)
    where action is 'open' or 'delete', or None on cancel. Opened by the ☰
    button next to the chat tabs (IR3DEApp._open_chat_menu ->
    push_screen(..., self._on_chat_menu_result))."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, entries: list[tuple[str, str, str]], **kwargs):
        """`entries` is (chat_id, title, timestamp) for every non-empty chat,
        already sorted most-recent first by IR3DEApp._open_chat_menu."""
        super().__init__(**kwargs)
        self._entries = entries

    def compose(self) -> ComposeResult:
        """Lay out the title and, if any chats exist, a scrollable list of
        ChatMenuRow entries (else an empty-state message)."""
        with Vertical(id="chat-menu-dialog"):
            yield Static("Open chat", id="chat-menu-title")
            if not self._entries:
                yield Static("No chats to open.", id="chat-menu-empty")
            else:
                with VerticalScroll(id="chat-menu-list"):
                    for chat_id, title, ts in self._entries:
                        yield ChatMenuRow(chat_id, title, ts)

    def on_chat_menu_row_open(self, event: ChatMenuRow.Open) -> None:
        """A row's label was clicked: dismiss requesting that chat be opened."""
        self.dismiss((event.chat_id, "open"))

    def on_chat_menu_row_delete(self, event: ChatMenuRow.Delete) -> None:
        """A row's 🗑 was clicked: dismiss requesting that chat be deleted."""
        self.dismiss((event.chat_id, "delete"))

    def action_cancel(self) -> None:
        """Escape: dismiss with no action."""
        self.dismiss(None)

    def on_click(self, event) -> None:
        """Clicking the backdrop cancels, same as Escape."""
        if event.control is self:
            self.dismiss(None)
