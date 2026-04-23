"""Exception types for the LLM layer.

Kept in a dedicated module to break the circular import between
``llm.py`` (which needs to raise these) and ``llm_retry.py`` (which
needs to treat some of them as retriable).
"""


class StructuredOutputParseError(Exception):
    """Raised when an LLM returns content that fails Pydantic structured-output parsing.

    This is a retriable error: the model occasionally returns prose or a
    malformed JSON shape instead of the requested schema, and a retry with
    the same prompt often succeeds. The raised message includes a truncated
    snippet of the offending content so operators can see what the model
    actually returned without dumping arbitrarily large payloads into logs.
    """
