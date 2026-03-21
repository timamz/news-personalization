import json
import logging
import sys
from datetime import UTC, datetime


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_record["exception"] = self.formatException(record.exc_info)
        for key in ("user_id", "subscription_id", "source_id"):
            value = getattr(record, key, None)
            if value is not None:
                log_record[key] = value
        return json.dumps(log_record)


def setup_logging(level: str = "DEBUG") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.INFO)
