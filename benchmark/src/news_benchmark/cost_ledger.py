"""
CostLedger records every litellm call made during a benchmark run.

Works by replacing litellm.acompletion and litellm.aembedding with async
wrappers at harness startup. Each wrapper reads the backend's current
agent ContextVar, asks litellm.completion_cost for USD pricing, and
appends one row to the in-memory ledger.

Captures both:
  - direct news_service.core.llm calls (chat_completion, embed_texts)
  - Google ADK's LiteLlm model adapter, which also calls
    litellm.acompletion under the hood

The ledger stays in memory for the run and is serialized into the run
artifacts by the report module.

Usage:

    from news_benchmark.cost_ledger import LEDGER, install_litellm_wrappers

    install_litellm_wrappers()
    # ... run scenarios ...
    rows = LEDGER.rows()
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

from news_benchmark.clock import CLOCK

_LLM_CALL_TIMEOUT_SECONDS = float(os.environ.get("BENCH_LLM_CALL_TIMEOUT", "120"))


async def _acompletion_with_hard_timeout(*args: Any, **kwargs: Any) -> Any:
    """Run litellm.acompletion with a hard per-call wall-clock ceiling.

    The provider (notably the neuroapi.host gateway in front of DeepSeek)
    occasionally accepts a request and then never responds. Without a
    hard ceiling, the streaming generator's ``finally: await task`` in
    the conversational agent traps the surrounding ``asyncio.wait_for``
    so the test-level timeout never fires either. Translate the timeout
    into a ``litellm.Timeout`` so the retry decorator can give the call
    a fresh chance before failing the turn.
    """
    assert _original_acompletion is not None
    try:
        return await asyncio.wait_for(
            _original_acompletion(*args, **kwargs),
            timeout=_LLM_CALL_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise litellm.Timeout(
            message=f"benchmark hard timeout {int(_LLM_CALL_TIMEOUT_SECONDS)}s",
            model=str(kwargs.get("model", "unknown")),
            llm_provider="benchmark",
        ) from exc

_trace_logger = logging.getLogger("news_benchmark.llm_trace")


def _verbose_enabled() -> bool:
    return os.environ.get("BENCH_VERBOSE_LLM", "").strip() not in ("", "0", "false", "False")


def _trace_enabled() -> bool:
    """Print a short BEGIN/END line for every litellm call.

    Cheaper than ``BENCH_VERBOSE_LLM`` (no message bodies, just timing
    and agent tag). Use whenever a long-running test goes quiet between
    phase boundaries and you want a heartbeat on every LLM call.
    """
    return os.environ.get("BENCH_TRACE_LLM", "").strip() not in ("", "0", "false", "False")


def _trace_print(msg: str) -> None:
    import sys

    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _trim(text: str, limit: int = 400) -> str:
    text = text or ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + f"…(+{len(text) - limit} chars)"


def _summarize_messages(messages: Any) -> str:
    """Return a one-line summary of the last user-ish message in the request."""
    if not isinstance(messages, list) or not messages:
        return "<no-messages>"
    last = messages[-1]
    role = last.get("role", "?") if isinstance(last, dict) else "?"
    content = last.get("content", "") if isinstance(last, dict) else ""
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text", p)))
            else:
                parts.append(str(p))
        content = " ".join(parts)
    return f"[{role}] {_trim(str(content))}"


def _summarize_response(response: Any) -> str:
    """Return a short description of the completion: text + any tool calls."""
    try:
        choices = getattr(response, "choices", None) or response.get("choices")
        if not choices:
            return "<no-choices>"
        msg = getattr(choices[0], "message", None)
        if msg is None and isinstance(choices[0], dict):
            msg = choices[0].get("message")
        text = getattr(msg, "content", "") or ""
        if isinstance(msg, dict):
            text = msg.get("content") or ""
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        tc_summary = ""
        if tool_calls:
            names = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None and isinstance(tc, dict):
                    fn = tc.get("function")
                name = getattr(fn, "name", None) if fn is not None else None
                if name is None and isinstance(fn, dict):
                    name = fn.get("name")
                args = getattr(fn, "arguments", None) if fn is not None else None
                if args is None and isinstance(fn, dict):
                    args = fn.get("arguments")
                if isinstance(args, str):
                    with contextlib.suppress(Exception):
                        args = json.loads(args)
                names.append(f"{name}({_trim(json.dumps(args) if args else '', 150)})")
            tc_summary = " tools=[" + ", ".join(names) + "]"
        return f"{_trim(text or '')}{tc_summary}"
    except Exception as exc:
        return f"<unparseable: {exc}>"


@dataclass
class LedgerRow:
    """One recorded litellm call."""

    run_id: str
    scenario_id: str
    model_column: str
    agent_path: str
    call_type: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usd_cost: float
    wall_ms: float
    fake_clock_iso: str
    subscription_id: str | None = None


@dataclass
class CostLedger:
    """In-memory collector of LedgerRow objects plus run-wide context."""

    run_id: str = ""
    scenario_id: str = ""
    model_column: str = ""
    _rows: list[LedgerRow] = field(default_factory=list)

    def append(self, row: LedgerRow) -> None:
        self._rows.append(row)

    def rows(self) -> list[LedgerRow]:
        return list(self._rows)

    def clear(self) -> None:
        self._rows.clear()

    def total_usd(self) -> float:
        return sum(r.usd_cost for r in self._rows)

    def by_agent(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self._rows:
            out[r.agent_path] = out.get(r.agent_path, 0.0) + r.usd_cost
        return out


LEDGER = CostLedger()


_original_acompletion = None
_original_aembedding = None
_installed = False


_CODE_FENCE_RE = None


def _clean_code_fences(response: Any) -> None:
    """Strip ```json ... ``` and ``` ... ``` markdown fences from ``response.content``.

    Mutates the response in place. DeepSeek sometimes wraps the JSON
    object in code fences even after being told not to, which trips
    ``Pydantic.model_validate_json`` (it expects pure JSON).
    """
    import re

    global _CODE_FENCE_RE
    if _CODE_FENCE_RE is None:
        _CODE_FENCE_RE = re.compile(
            r"^\s*```(?:json|javascript)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE
        )
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None:
                continue
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                continue
            m = _CODE_FENCE_RE.match(content)
            if m:
                cleaned = m.group(1).strip()
                with contextlib.suppress(AttributeError, ValueError):
                    message.content = cleaned
    except Exception:
        return


def _strip_response_format_inject_schema(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``kwargs`` with ``response_format`` removed and JSON-schema in messages.

    Pydantic model is read off ``kwargs["response_format"]`` (it can be the
    class itself or already a json_schema dict). A system message is
    appended to the existing ``messages`` list instructing the model to
    return ONLY a JSON object matching the schema. The backend's
    ``chat_completion`` then parses ``message.content`` with the same
    Pydantic class.
    """
    out = dict(kwargs)
    rf = out.pop("response_format", None)
    messages_in = out.get("messages") or []
    schema: dict[str, Any] | None = None
    if isinstance(rf, type):
        with contextlib.suppress(Exception):
            schema = rf.model_json_schema()  # type: ignore[attr-defined]
    elif isinstance(rf, dict):
        if rf.get("type") == "json_schema":
            schema = (rf.get("json_schema") or {}).get("schema")
        elif rf.get("type") == "json_object":
            schema = {"type": "object"}
    if schema is None:
        out["messages"] = list(messages_in)
        return out
    instruction = (
        "You must respond with ONLY a single JSON object that strictly conforms "
        "to the following JSON Schema. No prose, no markdown, no code fences -- "
        "just the JSON object. Schema:\n\n" + json.dumps(schema, separators=(",", ":"))
    )
    new_messages = list(messages_in)
    new_messages.append({"role": "system", "content": instruction})
    out["messages"] = new_messages
    return out


async def _wrapped_acompletion(*args: Any, **kwargs: Any) -> Any:
    """Wraps litellm.acompletion with cost accounting and agent tagging."""
    assert _original_acompletion is not None
    tag = _resolve_agent_tag()
    sub_id = _current_sub_id()
    model = kwargs.get("model", "unknown")
    verbose = _verbose_enabled()
    trace = _trace_enabled()
    if verbose:
        _trace_logger.info(
            "LLM-IN  tag=%s model=%s msg=%s",
            tag,
            model,
            _summarize_messages(kwargs.get("messages")),
        )
    if trace:
        _trace_print(f"[llm] BEGIN chat tag={tag} model={model}")
    t0 = time.monotonic()
    try:
        response = await _acompletion_with_hard_timeout(*args, **kwargs)
    except Exception as exc:
        # Some models (notably DeepSeek-v4-flash) reject the Pydantic
        # ``response_format`` kwarg with "This response_format type is
        # unavailable now". Recover by:
        #   1. Pulling the JSON schema off the Pydantic class.
        #   2. Appending a strict "return JSON matching this schema"
        #      instruction to the messages list.
        #   3. Re-calling without ``response_format``.
        # The backend's ``chat_completion`` then parses ``message.content``
        # as JSON against the same Pydantic class. Keeps the benchmark
        # unblocked when the configured chat model lacks structured-
        # output support without changing backend code.
        msg = str(exc)
        if "response_format" in msg and "unavailable" in msg and "response_format" in kwargs:
            if trace:
                _trace_print(
                    f"[llm] RETRY chat tag={tag} model={model} "
                    "dropping response_format + injecting schema (model rejected structured output)"
                )
            stripped = _strip_response_format_inject_schema(kwargs)
            try:
                response = await _acompletion_with_hard_timeout(*args, **stripped)
                _clean_code_fences(response)
            except Exception as exc2:
                if trace:
                    _trace_print(
                        f"[llm] FAIL  chat tag={tag} model={model} "
                        f"dur={time.monotonic() - t0:.2f}s err={type(exc2).__name__}: {exc2}"
                    )
                raise
        else:
            if trace:
                _trace_print(
                    f"[llm] FAIL  chat tag={tag} model={model} "
                    f"dur={time.monotonic() - t0:.2f}s err={type(exc).__name__}: {exc}"
                )
            raise
    wall_ms = (time.monotonic() - t0) * 1000.0
    if trace:
        _trace_print(f"[llm] END   chat tag={tag} model={model} dur={wall_ms / 1000:.2f}s")

    usage = _extract_usage(response)
    usd = _safe_completion_cost(response)
    if verbose:
        _trace_logger.info(
            "LLM-OUT tag=%s model=%s ms=%.0f tokens=%d/%d out=%s",
            tag,
            model,
            wall_ms,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            _summarize_response(response),
        )
    LEDGER.append(
        LedgerRow(
            run_id=LEDGER.run_id,
            scenario_id=LEDGER.scenario_id,
            model_column=LEDGER.model_column,
            agent_path=tag,
            call_type="chat",
            model=str(model),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            usd_cost=usd,
            wall_ms=wall_ms,
            fake_clock_iso=CLOCK.now().isoformat(),
            subscription_id=sub_id,
        )
    )
    return response


def _fake_embeddings_enabled() -> bool:
    """When ``BENCH_FAKE_EMBEDDINGS=1`` we bypass the real embedding provider.

    The benchmark's embedding endpoint (neuroapi.host) periodically hangs
    or rate-limits, which locks the whole simulation on the first
    embedding call. For benchmark correctness we only need vectors of
    the right dimensionality; semantic quality is not under test here.
    Hash the input to a deterministic ``float[1536]`` vector so two
    identical strings still embed to the same vector (cosine-self == 1)
    and different strings produce different ones.
    """
    return os.environ.get("BENCH_FAKE_EMBEDDINGS", "").strip() not in ("", "0", "false", "False")


def _hash_embedding(text: str, dim: int = 1536) -> list[float]:
    """Deterministic pseudo-embedding derived from ``hashlib.sha256(text)``.

    Same string -> same vector (so cosine similarity is meaningful for
    dedup checks), different strings -> different vector. Normalized to
    unit length so cosine similarity behaves consistently.
    """
    import hashlib
    import math

    seed = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    out: list[float] = []
    i = 0
    while len(out) < dim:
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        for byte in h:
            out.append((byte / 127.5) - 1.0)
            if len(out) >= dim:
                break
        i += 1
    norm = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / norm for x in out]


class _FakeEmbeddingResponse:
    """Mimics a litellm embedding response shape used by news_service.core.llm."""

    def __init__(self, vectors: list[list[float]], model: str, total_tokens: int) -> None:
        self.data = [{"embedding": v, "index": i} for i, v in enumerate(vectors)]
        self.model = model

        class _Usage:
            def __init__(self, total: int) -> None:
                self.prompt_tokens = total
                self.completion_tokens = 0
                self.total_tokens = total

        self.usage = _Usage(total_tokens)


async def _wrapped_aembedding(*args: Any, **kwargs: Any) -> Any:
    """Wraps litellm.aembedding with cost accounting and agent tagging."""
    assert _original_aembedding is not None
    tag = _resolve_agent_tag()
    sub_id = _current_sub_id()
    model = kwargs.get("model", "unknown")
    trace = _trace_enabled()
    if trace:
        _trace_print(f"[llm] BEGIN embed tag={tag} model={model}")
    t0 = time.monotonic()
    if _fake_embeddings_enabled():
        inputs = kwargs.get("input") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        dim = int(kwargs.get("dimensions") or 1536)
        vectors = [_hash_embedding(text, dim=dim) for text in inputs]
        total_tokens = sum(max(1, len(text.split())) for text in inputs)
        response: Any = _FakeEmbeddingResponse(
            vectors=vectors, model=str(model), total_tokens=total_tokens
        )
    else:
        try:
            response = await _original_aembedding(*args, **kwargs)
        except Exception as exc:
            if trace:
                _trace_print(
                    f"[llm] FAIL  embed tag={tag} model={model} "
                    f"dur={time.monotonic() - t0:.2f}s err={type(exc).__name__}: {exc}"
                )
            raise
    wall_ms = (time.monotonic() - t0) * 1000.0
    if trace:
        _trace_print(f"[llm] END   embed tag={tag} model={model} dur={wall_ms / 1000:.2f}s")

    usage = _extract_usage(response)
    usd = _safe_completion_cost(response)
    LEDGER.append(
        LedgerRow(
            run_id=LEDGER.run_id,
            scenario_id=LEDGER.scenario_id,
            model_column=LEDGER.model_column,
            agent_path=tag,
            call_type="embedding",
            model=str(model),
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=0,
            total_tokens=usage["total_tokens"],
            usd_cost=usd,
            wall_ms=wall_ms,
            fake_clock_iso=CLOCK.now().isoformat(),
            subscription_id=sub_id,
        )
    )
    return response


def _current_sub_id() -> str | None:
    """Read the production ContextVar for current subscription attribution."""
    try:
        from news_service.core.llm_usage import current_subscription_id

        sub = current_subscription_id.get()
        return str(sub) if sub is not None else None
    except Exception:
        return None


def _resolve_agent_tag() -> str:
    """Read the backend's current agent tag for cost attribution."""
    try:
        from news_service.core.llm_usage import current_agent as backend_agent

        name = backend_agent.get()
        if name:
            return str(name)
    except Exception:
        pass
    return "untagged"


def _extract_usage(response: Any) -> dict[str, int]:
    """Pull token counts out of a litellm response, tolerant to shape drift."""
    try:
        u = getattr(response, "usage", None)
        if u is None and isinstance(response, dict):
            u = response.get("usage")
        if u is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        pt = getattr(u, "prompt_tokens", None)
        ct = getattr(u, "completion_tokens", None)
        tt = getattr(u, "total_tokens", None)
        if pt is None and isinstance(u, dict):
            pt, ct, tt = u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens")
        return {
            "prompt_tokens": int(pt or 0),
            "completion_tokens": int(ct or 0),
            "total_tokens": int(tt or (int(pt or 0) + int(ct or 0))),
        }
    except Exception:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _safe_completion_cost(response: Any) -> float:
    """Return USD cost via litellm.completion_cost, 0.0 on any failure."""
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        return 0.0


def install_litellm_wrappers() -> None:
    """Install litellm wrappers globally. Idempotent; must run before news_service imports."""
    global _installed, _original_acompletion, _original_aembedding
    if _installed:
        return
    _original_acompletion = litellm.acompletion
    _original_aembedding = litellm.aembedding
    litellm.acompletion = _wrapped_acompletion  # type: ignore[assignment]
    litellm.aembedding = _wrapped_aembedding  # type: ignore[assignment]
    _installed = True
