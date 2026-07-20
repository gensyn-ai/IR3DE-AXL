from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from utils import MAX_TITLE_CHARS


class ChatTitleRenameScreen(ModalScreen):
    """Single-field modal that returns the new chat title, or None if
    canceled. Opened by double-clicking a chat tab
    (IR3DEApp._open_chat_rename_dialog -> push_screen(..., self._on_chat_renamed))."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current_title: str = "", **kwargs):
        """Create the dialog, pre-filled with the chat's current title."""
        super().__init__(**kwargs)
        self._current_title = current_title

    def compose(self) -> ComposeResult:
        """Lay out the title label and the editable input field."""
        with Vertical(id="chat-rename-dialog"):
            yield Static("Rename chat", id="chat-rename-title")
            yield Input(value=self._current_title,
                        placeholder="Chat title (Enter to confirm, Esc to cancel)",
                        id="chat-rename-input",
                        max_length=MAX_TITLE_CHARS)

    def on_mount(self) -> None:
        """Focus the input so the user can type immediately."""
        self.query_one("#chat-rename-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter: dismiss with the trimmed new title."""
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        """Escape: dismiss without renaming."""
        self.dismiss(None)

    def on_click(self, event) -> None:
        """Clicking the backdrop cancels, same as Escape."""
        if event.control is self:
            self.dismiss(None)
