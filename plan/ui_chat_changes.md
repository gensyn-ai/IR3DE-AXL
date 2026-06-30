# UI Changes Required for Multi-Chat Support

## Overview

The current UI has a single hardcoded Input tab with one `RichLog#output` and one `SubmittableTextArea#user-input`. To support multiple simultaneous chats, each needs its own tab with isolated widgets.

---

## Changes to `src/ui/ui.py`

### 1. App state — add to `__init__`

```python
self._chat_counter = 0
```

### 2. New method: `add_chat_tab() -> str`

Called on startup (to create the initial tab) and on "New Chat" button press.

```python
def add_chat_tab(self) -> str:
    import uuid
    chat_id = str(uuid.uuid4())
    self._chat_counter += 1
    label = f"Chat {self._chat_counter}"

    pane = TabPane(label, id=f"chat-{chat_id}")
    # Mount output log + input inside the pane
    # (Textual: use pane.mount() after add_pane())
    tabs = self.query_one("#right-tabs", TabbedContent)
    self.call_after_refresh(self._mount_chat_pane, tabs, pane, chat_id)
    return chat_id

async def _mount_chat_pane(self, tabs, pane, chat_id):
    await tabs.add_pane(pane)
    output = RichLog(id=f"output-{chat_id}", markup=True, highlight=True)
    textarea = SubmittableTextArea(id=f"input-{chat_id}")
    textarea.chat_id = chat_id          # tag the widget with its chat_id
    await pane.mount(output)
    await pane.mount(textarea)
    tabs.active = f"chat-{chat_id}"     # switch to the new tab
```

### 3. New method: `write_to_chat(chat_id, answer, peer_name)`

Called from the answer callback (via `call_from_thread`).

```python
def write_to_chat(self, chat_id: str, answer: str, peer_name: str):
    try:
        output = self.query_one(f"#output-{chat_id}", RichLog)
        output.write(f"[bold]{peer_name}:[/bold] {answer}")
    except Exception:
        pass  # tab may have been closed
```

### 4. Update `on_submittable_text_area_submitted`

Extract `chat_id` from the submitting widget:

```python
def on_submittable_text_area_submitted(self, event: SubmittableTextArea.Submitted):
    user_text = event.value
    chat_id = getattr(event.control, "chat_id", "default")
    if self.input_handler is not None:
        self.input_handler(self, self.args, user_text, chat_id)
```

### 5. Add "New Chat" button

Add to the Input tab toolbar (near existing filter/action buttons):

```python
Button("+ New Chat", id="new-chat-btn", variant="primary")
```

Handler:

```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    if event.button.id == "new-chat-btn":
        self.add_chat_tab()
        # ... existing button handlers below
```

### 6. Remove hardcoded `#output` / `#user-input` in `compose()`

The existing `TabPane("Input", ...)` block with hardcoded `RichLog(id="output")` and `SubmittableTextArea(id="user-input")` should be removed. Instead, call `self.add_chat_tab()` from `on_mount()` to create the first chat tab automatically.

### 7. Update `on_mount()` / `set_output_widget` wiring

Currently `set_output_widget(self.query_one("#output", RichLog))` is called in `on_mount`. After the refactor this global output widget no longer exists. The `log()` / `set_output_widget` path can remain for system logs in the left pane; chat answers go directly via `write_to_chat`.

### 8. `run_simulation.py` wiring update

The `handle_input` signature gains `chat_id`:

```python
def handle_input(app, args, user_input, chat_id):
    app.peer.handle_user_input(
        user_input,
        chat_id,
        answer_callback=lambda answer, peer_name: app.call_from_thread(
            app.write_to_chat, chat_id, answer, peer_name
        )
    )
```

---

## Summary of Widget ID Convention

| Widget | ID pattern |
|--------|-----------|
| TabPane per chat | `chat-{chat_id}` |
| RichLog per chat | `output-{chat_id}` |
| SubmittableTextArea per chat | `input-{chat_id}` |

---

## Notes

- `call_from_thread` is required because the peer's answer handler runs in a background thread; Textual widget updates must happen on the main thread.
- `SubmittableTextArea` needs a `chat_id` instance attribute — Python allows setting arbitrary attributes on subclass instances, so `textarea.chat_id = chat_id` works without modifying the class definition.
- If the user closes a tab (future feature), the corresponding history in `peer.conversation_histories[chat_id]` can be deleted to free memory.
