from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Static


class DeleteChatConfirmScreen(ModalScreen[bool]):
    """Y/n confirmation for chat deletion. Dismisses True or False. Opened
    from the ☰ chat menu's delete action
    (IR3DEApp._on_chat_menu_result -> push_screen(..., _on_confirm))."""

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("Y", "confirm", "Yes"),
        ("enter", "confirm", "Yes"),
        ("n", "cancel", "No"),
        ("N", "cancel", "No"),
        ("escape", "cancel", "No"),
    ]

    def __init__(self, chat_title: str, **kwargs):
        """Create the dialog for the given chat's title."""
        super().__init__(**kwargs)
        self._chat_title = chat_title

    def compose(self) -> ComposeResult:
        """Lay out the warning text and the Delete/Cancel buttons."""
        with Vertical(id="delete-chat-dialog"):
            yield Static(f'Delete "{self._chat_title}"?', id="delete-chat-title")
            yield Static("This action cannot be undone.", id="delete-chat-warning")
            with Horizontal(id="delete-chat-buttons"):
                yield Static("[Y] Delete", id="delete-chat-confirm-btn",
                             classes="chat-dialog-btn")
                yield Static("[N] Cancel", id="delete-chat-cancel-btn",
                             classes="chat-dialog-btn")

    def on_click(self, event) -> None:
        """Route the click to confirm, cancel, or backdrop-cancel."""
        if event.control is None:
            return
        if event.control is self:
            # Clicked the backdrop — treat as cancel.
            self.dismiss(False)
            return
        if event.control.id == "delete-chat-confirm-btn":
            self.dismiss(True)
        elif event.control.id == "delete-chat-cancel-btn":
            self.dismiss(False)

    def action_confirm(self) -> None:
        """Y/Enter: dismiss confirming deletion."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """N/Escape: dismiss without deleting."""
        self.dismiss(False)
