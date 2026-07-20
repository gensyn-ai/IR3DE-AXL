from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Static

from utils import format_params


class ModelRow(Horizontal):
    """One selectable model row inside an ExpertiseSection."""

    class Selected(Message):
        """Posted on click; handled by IR3DEApp.on_model_row_selected, which
        records this (peer_pk, model_idx) as the chosen model for `tag` in
        the Control Panel's expertise sections."""
        def __init__(self, tag: str, peer_pk: str, model_idx: int):
            """Store which model, hosted by which peer, was picked for `tag`."""
            super().__init__()
            self.tag = tag
            self.peer_pk = peer_pk
            self.model_idx = model_idx

    def __init__(self, tag: str, peer_pk: str, model_idx: int,
                 name: str, node_name: str, params: int,
                 active: bool = False, **kwargs):
        """Create one row for a model candidate, mounted by
        ExpertiseSection._rebuild_rows."""
        # Inject "-active" into the classes arg so the CSS engine sees it
        # from the very first render — no on_mount timing dance.
        classes_arg = kwargs.pop("classes", "")
        if active:
            classes_arg = f"{classes_arg} -active".strip()
        if classes_arg:
            kwargs["classes"] = classes_arg
        super().__init__(**kwargs)

        self._tag = tag
        self._peer_pk = peer_pk
        self._model_idx = model_idx
        self._name = name
        self._node_name = node_name
        self._params = params
        self._active = active

    def compose(self) -> ComposeResult:
        """Lay out the row: selection glyph, model name, host node, size."""
        yield Static("●" if self._active else "○", classes="model-row-glyph")
        yield Static(self._name, classes="model-row-name")
        yield Static(self._node_name, classes="model-row-node")
        yield Static(format_params(self._params), classes="model-row-params")

    def set_active(self, active: bool):
        """Called by ExpertiseSection when the selected model for this tag
        changes, to (de)highlight this row without a full rebuild."""
        if active == self._active:
            return
        self._active = active
        if active:
            self.add_class("-active")
        else:
            self.remove_class("-active")
        self.query_one(".model-row-glyph", Static).update("●" if active else "○")

    def on_click(self, event):
        """Select this row's model for its tag in the Control Panel."""
        self.post_message(self.Selected(self._tag, self._peer_pk, self._model_idx))
