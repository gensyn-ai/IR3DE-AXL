from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen


class DeleteChatConfirmScreen(ModalScreen[bool]):
    """Y/n confirmation for chat deletion. Dismisses True or False."""

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("Y", "confirm", "Yes"),
        ("enter", "confirm", "Yes"),
        ("n", "cancel", "No"),
        ("N", "cancel", "No"),
        ("escape", "cancel", "No"),
    ]

    def __init__(self, chat_title: str, **kwargs):
        super().__init__(**kwargs)
        self._chat_title = chat_title

    def compose(self) -> ComposeResult:
        with Vertical(id="delete-chat-dialog"):
            yield Static(f'Delete "{self._chat_title}"?', id="delete-chat-title")
            yield Static("This action cannot be undone.", id="delete-chat-warning")
            with Horizontal(id="delete-chat-buttons"):
                yield Static("[Y] Delete", id="delete-chat-confirm-btn",
                             classes="chat-dialog-btn")
                yield Static("[N] Cancel", id="delete-chat-cancel-btn",
                             classes="chat-dialog-btn")

    def on_click(self, event) -> None:
        if event.control is None:
            return
        if event.control.id == "delete-chat-confirm-btn":
            self.dismiss(True)
        elif event.control.id == "delete-chat-cancel-btn":
            self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
