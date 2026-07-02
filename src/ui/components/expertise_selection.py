from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Horizontal, Vertical

from ui.components import ModelRow


class ExpertiseSection(Vertical):
    """A bordered card listing models for one selected expertise tag."""

    def __init__(self, tag: str, symbol: str,
                 candidates: list[tuple[str, int, str, str, int]],
                 selected: tuple[str, int] | None,
                 **kwargs):
        super().__init__(**kwargs)
        self._tag = tag
        self._symbol = symbol
        self._candidates = candidates
        self._selected = selected
        self._rows_container = Vertical(classes="expertise-section-rows")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="expertise-section-header"):
            yield Static(f"{self._symbol.ljust(3)} {self._tag.capitalize()}",
                         classes="expertise-section-title")
            yield Static("[SELECTED]", classes="expertise-section-badge")
        yield Static(f"Select the desired model for {self._tag} tasks.",
                     classes="expertise-section-desc")
        yield self._rows_container

    def on_mount(self) -> None:
        # Rows are added here — at this point _rows_container is in the tree.
        self._rebuild_rows()
        self._update_badge()

    def update_state(self, candidates: list[tuple[str, int, str, str, int]], selected: tuple[str, int] | None) -> None:
        
        candidates_changed = candidates != self._candidates
        selected_changed = selected != self._selected

        if not candidates_changed and not selected_changed:
            return

        self._candidates = candidates
        self._selected = selected

        if candidates_changed:
            self._rebuild_rows()
        else:
            for row in self._rows_container.query(ModelRow):
                is_active = self._selected is not None and (row._peer_pk, row._model_idx) == self._selected
                row.set_active(is_active)

        self._update_badge()

    def _rebuild_rows(self) -> None:
        for child in list(self._rows_container.children):
            child.remove()

        if not self._candidates:
            self._rows_container.mount(
                Static("No available models for now.", classes="no-models-msg")
            )
            return

        for pk, idx, name, node, params in self._candidates:
            is_active = self._selected is not None and (pk, idx) == self._selected
            self._rows_container.mount(
                ModelRow(self._tag, pk, idx, name, node, params, active=is_active)
            )

    def _update_badge(self) -> None:
        badge = self.query_one(".expertise-section-badge", Static)
        badge.styles.display = "block" if self._selected is not None else "none"

    def set_has_selection(self, has: bool) -> None:
        # Kept for backwards compatibility with callers that still use it.
        badge = self.query_one(".expertise-section-badge", Static)
        badge.styles.display = "block" if has else "none"
