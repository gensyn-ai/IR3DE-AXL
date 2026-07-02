from textual.widgets import TextArea
from textual.message import Message


class SubmittableTextArea(TextArea):
    
    class Submitted(Message):
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def submit(self):
        value = self.text
        self.text = ""
        self.post_message(self.Submitted(value))

    def on_key(self, event):
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.submit()
        elif event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")