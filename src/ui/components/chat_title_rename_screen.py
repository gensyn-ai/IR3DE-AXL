from textual.app import ComposeResult
from textual.widgets import Static, Input
from textual.containers import Vertical
from textual.screen import ModalScreen

from utils import MAX_TITLE_CHARS


class ChatTitleRenameScreen(ModalScreen):
    """Single-field modal that returns the new chat title, or None if cancelled."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, current_title: str = "", **kwargs):
        super().__init__(**kwargs)
        self._current_title = current_title

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-rename-dialog"):
            yield Static("Rename chat", id="chat-rename-title")
            yield Input(value=self._current_title,
                        placeholder="Chat title (Enter to confirm, Esc to cancel)",
                        id="chat-rename-input",
                        max_length=MAX_TITLE_CHARS)

    def on_mount(self) -> None:
        self.query_one("#chat-rename-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)
    
    def on_click(self, event) -> None:
        if event.control is self:
            self.dismiss(None)
