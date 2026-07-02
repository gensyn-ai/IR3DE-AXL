from textual.widgets import Static
from textual.message import Message
from rich.text import Text as RichText


class TagButton(Static):

    class Toggled(Message):
        def __init__(self, tag: str, active: bool):
            super().__init__()
            self.tag = tag
            self.active = active

    def __init__(self, tag: str, symbol: str = "◆", **kwargs):
        super().__init__(**kwargs)
        self.tag = tag
        self.symbol = symbol
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        check = "✓" if self.active else " "
        # Pad symbol to 3 cells so glyphs like "</>" and "λ" line up the same.
        sym = self.symbol.ljust(3)
        self.update(RichText(f" {sym} {self.tag.capitalize():<12} [{check}]"))
        if self.active:
            self.add_class("-active")
        else:
            self.remove_class("-active")

    def on_click(self, event):
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.tag, self.active))