from textual.app import ComposeResult
from textual.widgets import Static, OptionList
from textual.containers import Vertical
from textual.screen import ModalScreen


class LatencyMetricSelectScreen(ModalScreen[str]):
    """Popup that returns one of: 'prompt' | 'total' | 'input'."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    OPTIONS = [
        ("Prompt latency",        "prompt"),
        ("Latency / total token", "total"),
        ("Latency / input token", "input"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="latency-metric-dialog"):
            yield Static("Choose latency metric", id="latency-metric-dialog-title")
            yield OptionList(
                *[label for label, _ in self.OPTIONS],
                id="latency-metric-options",
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.OPTIONS[event.option_index][1])

    def action_cancel(self) -> None:
        self.dismiss(None)