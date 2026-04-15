"""Typed exception hierarchy for the news service.

Three tiers match the error handling policy:
- Critical (DigestPipelineError, EventPipelineError): must propagate to task boundary.
- Quality gate: handled inline, never raises.
- Non-blocking: logged and swallowed, never raises.
"""


class NewsServiceError(Exception):
    """Base for all typed news-service errors."""


class DigestPipelineError(NewsServiceError):
    """Critical digest pipeline stage failed after retries exhausted.

    Raised by planner or composer when an LLM call fails and retries
    are exhausted. Caught at the Celery task level in deliver_digest.
    """


class EventPipelineError(NewsServiceError):
    """Critical event pipeline stage failed after retries exhausted.

    Raised by batch assessor when evaluation fails. Caught at the
    Celery task level in deliver_event_notifications_batch.
    """
