from rich.text import Text as RichText
from textual.message import Message
from textual.widgets import Static


class TagButton(Static):
    """One expertise-tag toggle in the Control Panel's #tag-buttons-container
    (IR3DEApp._refresh_tag_buttons), one per known IR3DE domain tag."""

    class Toggled(Message):
        """Posted on click; handled by IR3DEApp.on_tag_button_toggled, which
        adds/removes the tag from the peer's selected expertise set."""
        def __init__(self, tag: str, active: bool):
            """Store which tag was toggled and its new state."""
            super().__init__()
            self.tag = tag
            self.active = active

    def __init__(self, tag: str, symbol: str = "◆", **kwargs):
        """Create the button, inactive (unselected) by default."""
        super().__init__(**kwargs)
        self.tag = tag
        self.symbol = symbol
        self.active = False
        self._refresh_label()

    def _refresh_label(self):
        """Redraw the glyph + tag name + checkbox and the "-active" class."""
        check = "✓" if self.active else " "
        # Pad symbol to 3 cells so glyphs like "</>" and "λ" line up the same.
        sym = self.symbol.ljust(3)
        self.update(RichText(f" {sym} {self.tag.capitalize():<12} [{check}]"))
        if self.active:
            self.add_class("-active")
        else:
            self.remove_class("-active")

    def on_click(self, event):
        """Toggle this tag's selection state."""
        self.active = not self.active
        self._refresh_label()
        self.post_message(self.Toggled(self.tag, self.active))