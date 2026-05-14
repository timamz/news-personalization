"""Pipeline guardrails -- content sanitization, injection detection, and
output safety.

Three layers, increasing cost / latency:

1. Boundary tags + length caps. Cheapest. Always on.
2. Regex injection detection via ``clawguard-core`` (MIT, multilingual
   incl. Russian) plus a small in-repo Russian pattern list. The OSS
   set is the single source of truth for English / EU-language tells;
   the in-repo list backstops Russian since ClawGuard's Russian
   coverage is unaudited as of v0.4.0.
3. Optional ML classifier (``meta-llama/Llama-Prompt-Guard-2-86M``)
   loaded lazily on first call. Off by default -- flip
   ``injection_classifier_enabled`` once the model has been downloaded
   and Meta's license accepted on the host.

Output safety is a separate, narrower scan: ``better-profanity`` for
English, a vendored Cyrillic stem list for Russian, and optionally a
multilingual DistilBERT toxicity classifier
(``citizenlab/distilbert-base-multilingual-cased-toxicity``, ~540 MB,
off by default) covering both languages and 100+ others. The goal is
"do not deliver a slur-laden or insulting digest to a Telegram user";
it is NOT a general moderation system. Pair with a hosted classifier
(OpenAI omni-moderation) for anything subtler.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)


_FALLBACK_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"IMPORTANT\s*:\s*override", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a|an|if)", re.IGNORECASE),
    re.compile(r"pretend\s+(you|that)", re.IGNORECASE),
    re.compile(r"(?:```|~~~)\s*(?:system|admin)", re.IGNORECASE),
    re.compile(r"<\s*/?\s*(?:admin|root)\s*>", re.IGNORECASE),
    re.compile(r"BEGIN\s+(?:OVERRIDE|INJECTION|SYSTEM)", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
]
"""Last-resort English patterns kept in-tree so detection still works if
``clawguard-core`` is unavailable. ClawGuard already covers these; the
duplication is intentional belt-and-braces."""


_RUSSIAN_INJECTION_PATTERNS = [
    re.compile(r"игнорируй(те)?\s+(все\s+)?предыдущие\s+(инструкции|указания|команды)", re.I),
    re.compile(r"забудь(те)?\s+(все\s+)?(предыдущие|прежние)\s+", re.I),
    re.compile(r"не\s+следуй(те)?\s+(никаким\s+)?инструкциям", re.I),
    re.compile(r"новые\s+инструкции\s*:", re.I),
    re.compile(r"теперь\s+ты\s+", re.I),
    re.compile(r"представь(те)?\s*,?\s*что\s+ты\s+", re.I),
    re.compile(r"веди\s+себя\s+как\s+", re.I),
    re.compile(r"отмени(те)?\s+(все\s+)?предыдущие\s+", re.I),
    re.compile(r"перепиши(те)?\s+(свою\s+)?инструкцию", re.I),
    re.compile(r"раскрой(те)?\s+(свой\s+)?системн", re.I),
]
"""Russian-language injection tells. ClawGuard v0.4 advertises Russian
support but has not been audited for the patterns above (which were
collected from observed jailbreak attempts in Russian-speaking forums)."""


_OTHER_LANG_INJECTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "de": [
        re.compile(
            r"ignoriere\s+(alle\s+)?(vorherigen|bisherigen|vorigen)\s+"
            r"(anweisungen|anordnungen|befehle)",
            re.I,
        ),
        re.compile(r"vergiss\s+(alle\s+)?(vorherigen|bisherigen)", re.I),
        re.compile(r"du\s+bist\s+(jetzt|nun)\s+", re.I),
        re.compile(r"neue\s+anweisungen\s*:", re.I),
    ],
    "fr": [
        re.compile(
            r"ignore[zr]?\s+(toutes\s+)?(les\s+)?instructions?\s+pr[eé]c[eé]dentes?",
            re.I,
        ),
        re.compile(r"oublie[zr]?\s+(tout\s+)?(ce\s+)?(qui\s+)?pr[eé]c[eé]de", re.I),
        re.compile(r"tu\s+es\s+maintenant\s+", re.I),
        re.compile(r"nouvelles?\s+instructions?\s*:", re.I),
    ],
    "es": [
        re.compile(
            r"ignora\s+(todas\s+)?(las\s+)?instrucciones?\s+(anteriores|previas)",
            re.I,
        ),
        re.compile(r"olvida\s+(todo\s+)?(lo\s+)?anterior", re.I),
        re.compile(r"ahora\s+(t[uú]\s+)?eres\s+", re.I),
    ],
    "it": [
        re.compile(r"ignora\s+(tutte\s+)?(le\s+)?istruzioni\s+precedenti", re.I),
        re.compile(r"dimentica\s+(tutto\s+)?(quanto\s+)?(detto\s+)?prima", re.I),
        re.compile(r"(ora|adesso)\s+sei\s+", re.I),
    ],
    "pt": [
        re.compile(
            r"ignor[ae]\s+(todas\s+)?(as\s+)?instru[cç][oõ]es?\s+anteriores",
            re.I,
        ),
        re.compile(r"esque[cç][ae]\s+(tudo\s+)?(o\s+que\s+)?(foi\s+)?dito\s+antes", re.I),
        re.compile(r"agora\s+voc[eê]\s+[eé]\s+", re.I),
    ],
    "pl": [
        re.compile(
            r"zignoruj\s+(wszystkie\s+)?(poprzednie|wcze[sś]niejsze)\s+"
            r"(instrukcje|polecenia)",
            re.I,
        ),
        re.compile(r"zapomnij\s+(o\s+)?wszystkim", re.I),
    ],
    "nl": [
        re.compile(r"negeer\s+(alle\s+)?(vorige|voorgaande)\s+(instructies|opdrachten)", re.I),
        re.compile(r"vergeet\s+alles", re.I),
    ],
    "tr": [
        re.compile(
            r"t[uü]m\s+[oö]nceki\s+talimatlar[iı]\s+(yoksay|g[oö]z\s+ard[iı]\s+et)",
            re.I,
        ),
        re.compile(r"her\s+[sş]eyi\s+unut", re.I),
    ],
    "zh": [
        re.compile(r"忽略\s*(所有|之前|以前|先前|上述).{0,10}(指令|指示|命令|要求|提示)"),
        re.compile(r"忘记\s*(之前|以前|先前|所有)"),
        re.compile(r"(现在|从现在开始|从此).{0,5}你是"),
    ],
    "ja": [
        re.compile(r"(以前の|これまでの|前の)\s*(指示|命令|指令).{0,5}(無視|忘れて)"),
        re.compile(r"あなたは\s*(今|これから|もう)"),
    ],
    "ko": [
        re.compile(r"(이전의|이전|기존)\s*(지시|명령|지침).{0,10}(무시|잊)"),
        re.compile(r"(이제|지금부터)\s+당신은"),
    ],
    "ar": [
        re.compile(r"تجاهل\s+(جميع|كل)\s+التعليمات\s+السابقة"),
        re.compile(r"ان[سَ]?\s+(كل|جميع)\s+(ما|التعليمات)"),
    ],
    "hi": [
        re.compile(r"(पिछले|पहले|पूर्व)\s+(सभी\s+)?निर्देशों?\s+को\s+(अनदेखा|नज़रअंदाज़)"),
        re.compile(r"अब\s+(तुम|आप)\s+(हो|हैं)"),
    ],
}
"""Curated injection tells for the other 13 languages ClawGuard claims to
cover (German, French, Spanish, Italian, Portuguese, Polish, Dutch,
Turkish, Chinese, Japanese, Korean, Arabic, Hindi). Sized similarly to
the Russian list (2-4 high-precision stems per language) so we keep a
local audit-friendly view alongside ClawGuard. Not exhaustive --
extend as observed."""


_MAX_NOTIFICATION_BODY_LENGTH = 4000
_MAX_DIGEST_TEXT_LENGTH = 100_000


@lru_cache(maxsize=1)
def _clawguard_scan_text():
    """Import ``clawguard_core.scan_text`` lazily so the dep stays soft.

    Returns ``None`` if the package is not installed; callers fall back
    to ``_FALLBACK_INJECTION_PATTERNS`` and the Russian list.
    """
    try:
        from clawguard_core import scan_text as _scan
    except ImportError:
        logger.warning("clawguard-core not installed; using fallback regex set only")
        return None
    return _scan


def wrap_untrusted_content(text: str) -> str:
    """Wrap external content in boundary tags for LLM consumption."""
    return f"<untrusted-content>\n{text}\n</untrusted-content>"


def scan_for_injection(text: str) -> list[str]:
    """Scan text for known prompt injection patterns.

    Runs four regex layers, deduped: (1) ClawGuard's multilingual
    ruleset (when installed), (2) the in-repo Russian list, (3) curated
    audit-friendly lists for the 13 other languages ClawGuard claims
    to cover, (4) a short in-repo English safety net so detection
    survives a ClawGuard regression or uninstall. Returns the list of
    matched identifiers; empty means clean.
    """
    matches: list[str] = []

    scan = _clawguard_scan_text()
    if scan is not None:
        try:
            result = scan(text)
        except Exception:
            logger.exception("ClawGuard scan raised; relying on in-repo patterns")
        else:
            findings = getattr(result, "findings", None) or getattr(result, "threats", None) or []
            for finding in findings:
                rule_id = (
                    getattr(finding, "pattern_name", None)
                    or getattr(finding, "rule_id", None)
                    or repr(finding)
                )
                matches.append(f"clawguard:{rule_id}")

    for pattern in _RUSSIAN_INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(f"ru:{pattern.pattern}")

    for lang, patterns in _OTHER_LANG_INJECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                matches.append(f"{lang}:{pattern.pattern}")

    for pattern in _FALLBACK_INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(f"fallback:{pattern.pattern}")

    return matches


def sanitize_article_content(headline: str, body: str) -> tuple[str, str, list[str]]:
    """Sanitize article content: wrap in boundaries, detect injections.

    Returns (wrapped_headline, wrapped_body, injection_flags).
    """
    flags = scan_for_injection(headline) + scan_for_injection(body)
    if flags:
        logger.warning(
            "Potential injection detected in article content: %s",
            flags[:3],
        )
    return wrap_untrusted_content(headline), wrap_untrusted_content(body), flags


def validate_used_item_ids(
    claimed_ids: list[str],
    candidate_ids: set[str],
) -> list[str]:
    """Verify all used_item_ids exist in the candidate set.

    Returns only valid IDs, logging warnings for phantoms.
    """
    valid = []
    for item_id in claimed_ids:
        if item_id in candidate_ids:
            valid.append(item_id)
        else:
            logger.warning("Phantom used_item_id from LLM: %s", item_id)
    return valid


def validate_cron(cron_str: str) -> bool:
    """Validate a cron expression using croniter.

    Returns True if valid, False otherwise.
    """
    try:
        from croniter import croniter

        croniter(cron_str)
        return True
    except (ValueError, KeyError, TypeError):
        return False


def validate_notification_body(
    body: str,
    is_relevant: bool,
) -> str | None:
    """Validate event notification body.

    Returns the body if valid, None if invalid (with logging). Also
    runs an output-safety scan: profanity is logged but not blocked,
    because rendering a clean redaction would break the alert UX more
    than the alert itself harms.
    """
    if is_relevant and not body.strip():
        logger.warning("Relevant event has empty notification body")
        return None

    if len(body) > _MAX_NOTIFICATION_BODY_LENGTH:
        logger.warning(
            "Notification body exceeds %d chars, truncating", _MAX_NOTIFICATION_BODY_LENGTH
        )
        body = body[:_MAX_NOTIFICATION_BODY_LENGTH] + "..."

    safety_flags = scan_output_safety(body)
    if safety_flags:
        logger.warning("Output-safety flags on notification body: %s", safety_flags)

    return body


def validate_digest_text(
    text: str,
    max_length: int = _MAX_DIGEST_TEXT_LENGTH,
) -> str:
    """Validate and cap digest text length, log output-safety hits."""
    if len(text) > max_length:
        logger.warning("Digest text exceeds %d chars, truncating", max_length)
        text = text[:max_length] + "\n\n..."

    safety_flags = scan_output_safety(text)
    if safety_flags:
        logger.warning("Output-safety flags on digest text: %s", safety_flags)
    return text


def sanitize_for_llm_prompt(label: str, content: str, *, max_chars: int | None = None) -> str:
    """Wrap untrusted content with labeled boundaries for LLM prompts.

    Scans for injection and logs warnings. Caps overlong content at
    ``max_chars`` (defaults to ``settings.max_llm_external_text_chars``)
    so a single hostile blob cannot blow up the prompt or the bill.
    """
    if max_chars is None:
        from news_service.core.config import get_settings

        max_chars = get_settings().max_llm_external_text_chars
    if len(content) > max_chars:
        logger.warning(
            "%s content exceeds %d chars (got %d), truncating", label, max_chars, len(content)
        )
        content = content[:max_chars] + "\n... [truncated]"
    flags = scan_for_injection(content)
    if flags:
        logger.warning("Potential injection in %s: %s", label, flags[:3])
    return f"<untrusted-{label}>\n{content}\n</untrusted-{label}>"


def cap_text_for_embedding(text: str, max_length: int = 8000) -> str:
    """Truncate text for embedding if it exceeds max_length."""
    if len(text) > max_length:
        logger.warning("Text exceeds %d chars, truncating for embedding", max_length)
        return text[:max_length]
    return text


_RUSSIAN_PROFANITY_STEMS = [
    r"\bхуй\w*",
    r"\bпизд\w*",
    r"\bпидор\w*",
    r"\bпидар\w*",
    r"\bбляд\w*",
    r"\bебан\w*",
    r"\bебать\w*",
    r"\bёбан\w*",
    r"\bебуч\w*",
    r"\bхер\w*",
    r"\bговн\w*",
    r"\bсук[ауи]\b",
    r"\bмудак\w*",
    r"\bмудил\w*",
    r"\bжоп\w*",
    r"\bдолбоёб\w*",
    r"\bдолбоеб\w*",
    r"\bохуе\w*",
    r"\bохуё\w*",
    r"\bпохуй\b",
    r"\bохуит\w*",
]
"""Curated Cyrillic profanity stems. Inflection-aware (``\\w*`` suffix)
since Russian morphology defeats flat wordlists. Source: blend of
LDNOOBW ``ru`` and Jenyay/Obscene-Words-List, manually pruned to
high-precision stems."""


_RUSSIAN_PROFANITY_RE = re.compile("|".join(_RUSSIAN_PROFANITY_STEMS), re.IGNORECASE)


@lru_cache(maxsize=1)
def _better_profanity():
    try:
        from better_profanity import profanity

        profanity.load_censor_words()
        return profanity
    except ImportError:
        logger.warning("better-profanity not installed; English output scan disabled")
        return None


@lru_cache(maxsize=1)
def _toxicity_classifier():
    """Lazy-load the multilingual DistilBERT toxicity classifier.

    Default is ``citizenlab/distilbert-base-multilingual-cased-toxicity``
    (~540 MB, MIT, distilbert-multilingual base, 100+ languages incl.
    Russian and English). Binary classification: toxic / non-toxic.
    Runs on CPU in ~150 ms per call. Returns ``None`` if transformers /
    torch is missing or the weights fail to load.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.warning(
            "transformers/torch not installed; toxicity classifier disabled. "
            "Install via: uv sync --extra classifier"
        )
        return None

    from news_service.core.config import get_settings

    model_id = get_settings().output_safety_classifier_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    except Exception:
        logger.exception("Could not load toxicity classifier %r", model_id)
        return None
    return tokenizer, model, torch


def _toxicity_score(text: str) -> float | None:
    """Return the toxic-class probability from the multilingual classifier.

    Looks up the toxic label by name first (``toxic`` / ``LABEL_1`` /
    ``TOXIC``); falls back to index 1 for a typical binary head. Returns
    ``None`` if the classifier is unavailable or inference raised.
    """
    bundle = _toxicity_classifier()
    if bundle is None:
        return None
    tokenizer, model, torch = bundle
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = logits.softmax(-1)[0]
        label2id = {k.lower(): v for k, v in (getattr(model.config, "label2id", {}) or {}).items()}
        for key in ("toxic", "label_1", "1"):
            if key in label2id:
                return float(probs[label2id[key]])
        if probs.shape[0] > 1:
            return float(probs[1])
        return float(probs[0])
    except Exception:
        logger.exception("Toxicity classifier inference failed; ignoring score")
        return None


def scan_output_safety(text: str) -> list[str]:
    """Scan LLM-generated text for profanity / toxicity in EN + RU.

    Three sub-layers, all log-only by default:

    1. Russian profanity regex (``_RUSSIAN_PROFANITY_RE``): inflection-
       aware Cyrillic stems.
    2. English profanity regex (``better-profanity``): ~360 words plus
       leetspeak normalisation.
    3. Multilingual DistilBERT toxicity classifier
       (``citizenlab/distilbert-base-multilingual-cased-toxicity``):
       optional, off by default. Catches insults / threats / hate that
       the regex stems miss, in either language.

    Returns a list of flag identifiers, e.g. ``["ru-profanity",
    "toxicity:0.83"]``. Empty = clean. Caller decides what to do
    (strip & re-render, drop the message, alert).
    """
    from news_service.core.config import get_settings

    flags: list[str] = []
    if _RUSSIAN_PROFANITY_RE.search(text):
        flags.append("ru-profanity")
    prof = _better_profanity()
    if prof is not None and prof.contains_profanity(text):
        flags.append("en-profanity")

    settings = get_settings()
    if settings.output_safety_classifier_enabled:
        score = _toxicity_score(text)
        if score is not None and score >= settings.output_safety_classifier_threshold:
            flags.append(f"toxicity:{score:.2f}")

    return flags


@lru_cache(maxsize=1)
def _injection_classifier():
    """Lazy-load Llama-Prompt-Guard-2-86M for ML-grade injection detection.

    Returns ``None`` (and logs once) if transformers/torch is not
    installed or the model cannot be loaded. Loading happens on first
    call; subsequent calls reuse the cached tokenizer / model pair.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        logger.warning(
            "transformers/torch not installed; injection classifier disabled. "
            "Install via: uv add --optional classifier transformers torch"
        )
        return None

    from news_service.core.config import get_settings

    model_id = get_settings().injection_classifier_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    except Exception:
        logger.exception(
            "Could not load injection classifier %r; verify HF login and license acceptance.",
            model_id,
        )
        return None
    return tokenizer, model, torch


def classify_injection(text: str) -> float | None:
    """Return the classifier's malicious probability in [0, 1], or None.

    ``None`` means the classifier is disabled or failed to load --
    callers should treat that as "no signal" and rely on the regex
    layer, not as a clean verdict.
    """
    from news_service.core.config import get_settings

    if not get_settings().injection_classifier_enabled:
        return None

    bundle = _injection_classifier()
    if bundle is None:
        return None
    tokenizer, model, torch = bundle

    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = logits.softmax(-1)[0]
        label2id = getattr(model.config, "label2id", {}) or {}
        malicious_idx = (
            label2id.get("MALICIOUS") or label2id.get("LABEL_1") or (1 if probs.shape[0] > 1 else 0)
        )
        return float(probs[malicious_idx])
    except Exception:
        logger.exception("Injection classifier inference failed; ignoring score")
        return None
