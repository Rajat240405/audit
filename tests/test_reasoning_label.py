"""Mirror of frontend/src/utils/reasoningLabel.ts — keep in sync."""


def reasoning_wait_message(model: str | None) -> str:
    name = (model or "").strip()
    if not name:
        return "Model is reasoning… (this can take a bit)"
    if "qwen3" in name.lower():
        return f"Model is reasoning… ({name} thinks before answering; this can take a bit)"
    return f"Model is reasoning… ({name}; this can take a bit)"


def test_never_says_qwen3_unless_model_is_qwen3():
    msg = reasoning_wait_message("qwen2.5:7b")
    assert "qwen3" not in msg.lower()
    assert "qwen2.5:7b" in msg


def test_names_qwen3_when_selected():
    msg = reasoning_wait_message("qwen3:8b")
    assert "qwen3:8b" in msg


def test_empty_model_generic():
    assert "qwen3" not in reasoning_wait_message(None).lower()
    assert "qwen3" not in reasoning_wait_message("").lower()
