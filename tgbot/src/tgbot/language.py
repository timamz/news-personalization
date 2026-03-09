from dataclasses import dataclass
from typing import Literal

type SupportedLanguage = Literal["en", "ru"]
type DigestLanguage = SupportedLanguage
type UILanguage = SupportedLanguage
type LanguageMode = Literal["ask", "fixed"]


@dataclass(frozen=True, slots=True)
class LanguagePreference:
    mode: LanguageMode
    code: DigestLanguage | None = None


def normalize_language_code(value: str | None) -> SupportedLanguage | None:
    if value is None:
        return None

    normalized = value.strip().lower().split("-", maxsplit=1)[0]
    if normalized == "en":
        return "en"
    if normalized == "ru":
        return "ru"
    return None
