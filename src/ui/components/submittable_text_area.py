from textual.message import Message
from textual.widgets import TextArea


class SubmittableTextArea(TextArea):
    """The #user-input box in the chat pane; adds Enter-to-submit on top of
    TextArea's default (which just inserts a newline)."""

    class Submitted(Message):
        """Posted on submit; handled by IR3DEApp.on_submittable_text_area_submitted,
        which dispatches the text to input_handler (run.py's handle_input)."""
        def __init__(self, value: str):
            """Store the submitted text."""
            super().__init__()
            self.value = value

    def submit(self):
        """Clear the box and post the message, unless it's blank. Called on
        Enter and from IR3DEApp.on_click for the [SEND] button."""
        value = self.text
        if not value.strip():
            return
        self.text = ""
        self.post_message(self.Submitted(value))

    def on_key(self, event):
        """Enter submits; Shift+Enter/Ctrl+J inserts a newline instead."""
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.submit()
        elif event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")