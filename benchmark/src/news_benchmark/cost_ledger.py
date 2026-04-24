"""
CostLedger records every litellm call made during a benchmark run.

Works by replacing litellm.acompletion and litellm.aembedding with async
wrappers at harness startup. Each wrapper reads the current agent_tag
(ContextVar, see tagging.py), asks litellm.completion_cost for USD
pricing, and appends one row to the in-memory ledger. Originals are
restored by uninstall_litellm_wrappers().

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

import contextlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import litellm

from news_benchmark.clock import CLOCK
from news_benchmark.tagging import current_tag

_trace_logger = logging.getLogger("news_benchmark.llm_trace")


def _verbose_enabled() -> bool:
    return os.environ.get("BENCH_VERBOSE_LLM", "").strip() not in ("", "0", "false", "False")


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CostLedger:
    """In-memory collector of LedgerRow objects plus run-wide context."""

    run_id: str = ""
    scenario_id: str = ""
    model_column: str = ""
    _rows: list[LedgerRow] = field(default_factory=list)

    def set_context(self, run_id: str, scenario_id: str, model_column: str) -> None:
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.model_column = model_column

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


async def _wrapped_acompletion(*args: Any, **kwargs: Any) -> Any:
    """Wraps litellm.acompletion with cost accounting and agent tagging."""
    assert _original_acompletion is not None
    tag = _resolve_agent_tag()
    sub_id = _current_sub_id()
    model = kwargs.get("model", "unknown")
    verbose = _verbose_enabled()
    if verbose:
        _trace_logger.info(
            "LLM-IN  tag=%s model=%s msg=%s",
            tag,
            model,
            _summarize_messages(kwargs.get("messages")),
        )
    t0 = time.monotonic()
    response = await _original_acompletion(*args, **kwargs)
    wall_ms = (time.monotonic() - t0) * 1000.0

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


async def _wrapped_aembedding(*args: Any, **kwargs: Any) -> Any:
    """Wraps litellm.aembedding with cost accounting and agent tagging."""
    assert _original_aembedding is not None
    tag = _resolve_agent_tag()
    sub_id = _current_sub_id()
    model = kwargs.get("model", "unknown")
    t0 = time.monotonic()
    response = await _original_aembedding(*args, **kwargs)
    wall_ms = (time.monotonic() - t0) * 1000.0

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
    """Prefer the backend's current_agent; fall back to the benchmark stack.

    news_service has its own ``current_agent`` ContextVar set by every
    agent via ``agent_tag(name)``. That is the authoritative source of
    attribution during a live run. The legacy benchmark stack
    (``news_benchmark.tagging``) remains a fallback for tests that wrap
    arbitrary code in ``agent_tag(...)`` of their own.
    """
    try:
        from news_service.core.llm_usage import current_agent as backend_agent

        name = backend_agent.get()
        if name:
            return str(name)
    except Exception:
        pass
    return current_tag()


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


def uninstall_litellm_wrappers() -> None:
    """Restore original litellm functions. Used in teardown / tests."""
    global _installed
    if not _installed:
        return
    if _original_acompletion is not None:
        litellm.acompletion = _original_acompletion  # type: ignore[assignment]
    if _original_aembedding is not None:
        litellm.aembedding = _original_aembedding  # type: ignore[assignment]
    _installed = False
