def build_prompt_summary(raw_prompt: str, *, max_length: int = 140) -> str:
    normalized = " ".join(raw_prompt.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 1].rstrip()}…"
