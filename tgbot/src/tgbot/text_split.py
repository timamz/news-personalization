"""Split long text into chunks that fit within Telegram's 4096-char message limit.

Prefers paragraph boundaries ("\\n\\n"), then newlines, then sentence boundaries,
and finally a hard character cut. Short text is returned as a single chunk.
"""

TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def split_for_telegram(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    if len(text) <= max_length:
        return [text]

    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2:
        return _split_hard(text, max_length)

    num_parts = _min_parts(paragraphs, max_length)
    return _balanced_split(paragraphs, num_parts)


def _min_parts(paragraphs: list[str], max_length: int) -> int:
    total = sum(len(p) for p in paragraphs) + 2 * (len(paragraphs) - 1)
    parts = max(2, -(-total // max_length))
    while parts <= len(paragraphs):
        chunks = _balanced_split(paragraphs, parts)
        if all(len(c) <= max_length for c in chunks):
            return parts
        parts += 1
    return len(paragraphs)


def _balanced_split(paragraphs: list[str], num_parts: int) -> list[str]:
    total_len = sum(len(p) for p in paragraphs) + 2 * (len(paragraphs) - 1)
    target = total_len / num_parts

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_len = 0

    for para in paragraphs:
        separator_len = 2 if current_paragraphs else 0
        new_len = current_len + separator_len + len(para)

        if current_paragraphs and len(chunks) < num_parts - 1 and new_len > target:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = [para]
            current_len = len(para)
        else:
            current_paragraphs.append(para)
            current_len = new_len

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))
    return chunks


def _split_hard(text: str, max_length: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_length and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)

    result: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_length:
            part = _split_at_sentence(chunk, max_length)
            result.append(part)
            chunk = chunk[len(part) :].lstrip()
        if chunk:
            result.append(chunk)
    return result


def _split_at_sentence(text: str, max_length: int) -> str:
    window = text[:max_length]
    mid = len(window) // 2
    best = -1
    for i in range(mid + 1):
        for pos in (mid + i, mid - i):
            if 0 <= pos < len(window) - 1 and window[pos] == ".":
                next_ch = window[pos + 1]
                if next_ch in (" ", "\n"):
                    best = pos
                    break
        if best != -1:
            break
    if best != -1:
        return text[: best + 1]
    return text[:max_length]
