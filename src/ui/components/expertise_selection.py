from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from .model_row import ModelRow


class ExpertiseSection(Vertical):
    """A bordered card listing models for one selected expertise tag."""

    def __init__(self, tag: str, symbol: str,
                 candidates: list[tuple[str, int, str, str, int]],
                 selected: tuple[str, int] | None,
                 **kwargs):
        """Create the section; mounted into the Control Panel's
        #expertise-sections container by IR3DEApp._refresh_expertise_sections,
        one per currently-selected expertise tag."""
        super().__init__(**kwargs)
        self._tag = tag
        self._symbol = symbol
        self._candidates = candidates
        self._selected = selected
        self._rows_container = Vertical(classes="expertise-section-rows")
        self._badge = Static("[SELECTED]", classes="expertise-section-badge")

    def compose(self) -> ComposeResult:
        """Lay out the header (tag name + [SELECTED] badge), description,
        and the (initially empty) row container filled in on_mount."""
        with Horizontal(classes="expertise-section-header"):
            yield Static(f"{self._symbol.ljust(3)} {self._tag.capitalize()}",
                         classes="expertise-section-title")
            yield self._badge
        yield Static(f"Select the desired model for {self._tag} tasks.",
                     classes="expertise-section-desc")
        yield self._rows_container

    def on_mount(self) -> None:
        """Populate rows after children from compose() are attached.

        Parent on_mount runs before yielded children are mounted, so mounting
        into _rows_container here would raise MountError — defer one refresh.
        """
        self.call_after_refresh(self._populate)

    def _populate(self) -> None:
        """Fill model rows and badge once the section is fully in the DOM."""
        if not self.is_attached:
            return
        if not self._rows_container.is_attached or not self._badge.is_attached:
            # Children not ready yet; try again on the next refresh.
            self.call_after_refresh(self._populate)
            return
        self._rebuild_rows()
        self._update_badge()

    def update_state(self, candidates: list[tuple[str, int, str, str, int]], selected: tuple[str, int] | None) -> None:
        """Refresh an already-mounted section in place; called every stats
        tick by IR3DEApp._refresh_expertise_sections instead of remounting."""

        candidates_changed = candidates != self._candidates
        selected_changed = selected != self._selected

        if not candidates_changed and not selected_changed:
            return

        self._candidates = candidates
        self._selected = selected

        if not self._rows_container.is_attached:
            # Still mounting; on_mount's deferred _populate will use the
            # updated _candidates/_selected.
            return

        if candidates_changed:
            self._rebuild_rows()
        else:
            for row in self._rows_container.query(ModelRow):
                is_active = self._selected is not None and (row._peer_pk, row._model_idx) == self._selected
                row.set_active(is_active)

        self._update_badge()

    def _rebuild_rows(self) -> None:
        """Replace all ModelRow children with one per current candidate."""
        if not self._rows_container.is_attached:
            return

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
        """Show the [SELECTED] badge iff a model is chosen for this tag."""
        if not self._badge.is_attached:
            return
        self._badge.styles.display = "block" if self._selected is not None else "none"

    def set_has_selection(self, has: bool) -> None:
        """Kept for backwards compatibility with callers that still use it."""
        if not self._badge.is_attached:
            return
        self._badge.styles.display = "block" if has else "none"
