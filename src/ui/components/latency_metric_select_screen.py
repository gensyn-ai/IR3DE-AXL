from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static


class LatencyMetricSelectScreen(ModalScreen[str]):
    """Popup that returns one of: 'prompt' | 'total' | 'input'. Opened by
    clicking #latency-metric-button above the Statistics tab's latency plot
    (IR3DEApp.on_click -> push_screen(..., self._on_latency_metric_chosen))."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    OPTIONS = [
        ("Prompt latency",        "prompt"),
        ("Latency / total token", "total"),
        ("Latency / input token", "input"),
    ]

    def compose(self) -> ComposeResult:
        """Lay out the title and the option list of the three metrics."""
        with Vertical(id="latency-metric-dialog"):
            yield Static("Choose latency metric", id="latency-metric-dialog-title")
            yield OptionList(
                *[label for label, _ in self.OPTIONS],
                id="latency-metric-options",
            )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Dismiss with the chosen metric's key ('prompt'/'total'/'input')."""
        self.dismiss(self.OPTIONS[event.option_index][1])

    def action_cancel(self) -> None:
        """Escape: dismiss without changing the current metric."""
        self.dismiss(None)

    def on_click(self, event) -> None:
        """Clicking the backdrop cancels, same as Escape."""
        if event.control is self:
            self.dismiss(None)