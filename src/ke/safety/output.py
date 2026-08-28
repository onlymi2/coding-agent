TRUNCATION_MARKER = "\n... [truncated] ...\n"


def truncate_output(text: str, max_chars: int) -> str:
    """Keep the head and tail of output within a fixed character limit."""

    if max_chars < len(TRUNCATION_MARKER) + 2:
        raise ValueError(
            f"max_chars 必须大于等于 {len(TRUNCATION_MARKER) + 2}"
        )
    if len(text) <= max_chars:
        return text

    available = max_chars - len(TRUNCATION_MARKER)
    head_length = available // 2
    tail_length = available - head_length
    return text[:head_length] + TRUNCATION_MARKER + text[-tail_length:]
