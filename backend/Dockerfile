FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim AS runtime

ARG TARGETARCH

ENV GECKODRIVER_VERSION=0.36.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl firefox-esr tar && \
    case "${TARGETARCH}" in \
        amd64) geckodriver_arch="linux64" ;; \
        arm64) geckodriver_arch="linux-aarch64" ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac && \
    curl -fsSL -o /tmp/geckodriver.tar.gz \
        "https://github.com/mozilla/geckodriver/releases/download/v${GECKODRIVER_VERSION}/geckodriver-v${GECKODRIVER_VERSION}-${geckodriver_arch}.tar.gz" && \
    tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin && \
    chmod +x /usr/local/bin/geckodriver && \
    rm -rf /var/lib/apt/lists/* /tmp/geckodriver.tar.gz && \
    groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY alembic/ alembic/
COPY alembic.ini .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV MOZ_HEADLESS=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "news_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
