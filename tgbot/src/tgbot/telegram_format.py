import html
import re

_URL_PATTERN = re.compile(r"https?://[^\s<>\"]+")
_SOURCE_WITH_URL_PATTERN = re.compile(
    r"^\s*(?:source|источник)\s*:\s*(?:(?!https?://).)*?(https?://[^\s<>\"]+)\s*$",
    re.IGNORECASE,
)
_LABELLED_URL_FRAGMENT_PATTERN = re.compile(
    r"(?:(?:source|источник|url|link)\s*:\s*)?(https?://[^\s<>\"]+)",
    re.IGNORECASE,
)
_CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")


def render_html_message(text: str) -> str:
    lines = text.splitlines()
    default_label = _default_link_label(text)
    rendered_lines: list[str] = []
    for line in lines:
        if line.startswith("- "):
            rendered_lines.append(f"• {render_html_line(line[2:], default_label)}")
            continue
        rendered_lines.append(render_html_line(line, default_label))
    return "\n".join(rendered_lines)


def render_html_line(line: str, default_label: str) -> str:
    source_match = _SOURCE_WITH_URL_PATTERN.match(line)
    if source_match is not None:
        return _italic_link(source_match.group(1), _link_label_for_text(line, default_label))

    stripped = line.strip()
    if _URL_PATTERN.fullmatch(stripped):
        return _italic_link(stripped, _link_label_for_text(line, default_label))

    parts: list[str] = []
    last_end = 0
    for match in _LABELLED_URL_FRAGMENT_PATTERN.finditer(line):
        start, end = match.span()
        parts.append(html.escape(line[last_end:start]))
        url = match.group(0)
        source_url = match.group(1)
        if url.lower().startswith(("source:", "источник:", "url:", "link:")):
            parts.append(_italic_link(source_url, _link_label_for_text(line, default_label)))
        else:
            parts.append(_italic_link(source_url, _link_label_for_text(line, default_label)))
        last_end = end
    parts.append(html.escape(line[last_end:]))
    return "".join(parts)


def _default_link_label(text: str) -> str:
    return "Источник" if _CYRILLIC_PATTERN.search(text) else "Source"


def _link_label_for_text(text: str, default_label: str) -> str:
    lowered = text.lower()
    if "источник" in lowered:
        return "Источник"
    if "source" in lowered:
        return "Source"
    return default_label


def _italic_link(url: str, label: str) -> str:
    escaped_url = html.escape(url, quote=True)
    escaped_label = html.escape(label)
    return f'<a href="{escaped_url}"><i>{escaped_label}</i></a>'
