"""Event notification pipeline with batch assessment."""

from news_service.agents.event.batch_assessor import (
    BatchAssessmentResult,
    ItemAssessment,
    assess_batch_events,
)

__all__ = [
    "BatchAssessmentResult",
    "ItemAssessment",
    "assess_batch_events",
]
