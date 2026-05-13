"""
Fire every prompt in ``prompts.PROMPTS`` against the live devbox
Conversational Agent in parallel, capture every stream event, then read
the resulting DB state to produce a per-prompt report.

What gets measured, per prompt:

  * Wall-clock duration of the turn (user -> done event).
  * Every event emitted on the NDJSON stream, bucketed by
    ``event`` (``status`` / ``discovery_progress`` / ``done`` / ``error``)
    and, within status, by ``status_key``. That covers the visible
    tool calls: ``status_saving_subscription``,
    ``status_queuing_discovery``, ``status_adding_source``,
    ``status_removing_source``, ``status_queuing_digest``,
    ``status_resolving_timezone``, plus the finder's inner
    ``status_searching_web`` and ``status_validating_source`` (the two
    tools that cost real money - Yandex Search and post-fetch + embed).
  * Discovery pipeline phases (from ``discovery_progress`` events).
  * Final assistant message.
  * Every subscription the user ended up with, with its full source
    list, each source classified into rss / telegram_channel /
    reddit_subreddit by URL pattern, and per-kind counts.

What is NOT measured here (deliberate - separate pass):

  * Silent tool calls (``get_subscriptions``, ``remember``,
    ``close_scenario``, ``delete_subscription``, ``set_user_language``)
    do not emit status events. They can only be recovered from the
    Redis transcript; we intentionally stay API-only in this first
    pass.
  * Per-call LLM token counts / USD cost. Requires instrumenting the
    backend; handled by a later step.

Usage:

    cd benchmark
    uv run python economics/run_baseline.py \\
        --api-url http://100.73.138.67:8000 \\
        --db-url  postgresql://news:news@100.73.138.67:5432/news

Output: one JSON file per run under ``benchmark/economics/results/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import traceback
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import PROMPTS  # noqa: E402, I001


DEFAULT_API_URL = "http://100.73.138.67:8000"
DEFAULT_DB_URL = "postgresql://news:news@100.73.138.67:5432/news"
STREAM_TIMEOUT_SECONDS = 600.0

_TELEGRAM_RE = re.compile(r"^https?://t\.me/", re.IGNORECASE)
_REDDIT_RE = re.compile(r"^https?://(?:www\.|old\.)?reddit\.com/r/", re.IGNORECASE)


def classify_source_kind(url: str) -> str:
    """Classify a source URL into the kind the backend expects."""
    if _TELEGRAM_RE.match(url):
        return "telegram_channel"
    if _REDDIT_RE.match(url):
        return "reddit_subreddit"
    return "rss"


async def create_user(http: httpx.AsyncClient, api_url: str) -> tuple[str, str]:
    resp = await http.post(f"{api_url}/users", timeout=30.0)
    resp.raise_for_status()
    body = resp.json()
    return body["id"], body["api_key"]


async def ack_onboarding(http: httpx.AsyncClient, api_url: str, api_key: str) -> None:
    resp = await http.post(
        f"{api_url}/users/me/acknowledge-onboarding",
        headers={"X-API-Key": api_key},
        timeout=30.0,
    )
    if resp.status_code not in (200, 204):
        resp.raise_for_status()


async def stream_turn(
    http: httpx.AsyncClient,
    api_url: str,
    api_key: str,
    message: str,
    run_id: str,
    user_language: str = "en",
) -> tuple[list[dict[str, Any]], str | None]:
    """POST a single turn, consume the NDJSON stream to completion."""
    events: list[dict[str, Any]] = []
    final_message: str | None = None
    async with http.stream(
        "POST",
        f"{api_url}/subscriptions/conversations/stream",
        headers={
            "X-API-Key": api_key,
            "X-Run-Id": run_id,
            "Accept": "application/x-ndjson",
            "Content-Type": "application/json",
        },
        json={"message": message, "user_language": user_language},
        timeout=STREAM_TIMEOUT_SECONDS,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                events.append({"event": "_unparsed", "raw": line})
                continue
            events.append(ev)
            if ev.get("event") == "done":
                final_message = (ev.get("output") or {}).get("message")
                break
            if ev.get("event") == "error":
                break
    return events, final_message


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: Counter[str] = Counter()
    by_status_key: Counter[str] = Counter()
    discovery_phases: list[str] = []
    for ev in events:
        kind = ev.get("event") or "(missing)"
        by_type[kind] += 1
        if kind == "status":
            by_status_key[ev.get("status_key") or "(none)"] += 1
        elif kind == "discovery_progress":
            phase = ev.get("phase") or "(none)"
            discovery_phases.append(phase)
    return {
        "total": len(events),
        "by_type": dict(by_type),
        "by_status_key": dict(by_status_key),
        "discovery_phases": discovery_phases,
    }


_SUBSCRIPTIONS_SQL = """
    SELECT
        id::text AS id,
        delivery_mode,
        schedule_cron,
        digest_language,
        is_active,
        user_spec,
        created_at
    FROM subscriptions
    WHERE user_id = $1
    ORDER BY created_at ASC
"""

_SOURCES_SQL = """
    SELECT s.url,
           s.title,
           ss.is_user_specified,
           ss.created_at AS linked_at
    FROM sources s
    JOIN subscription_sources ss ON ss.source_id = s.id
    WHERE ss.subscription_id = $1
    ORDER BY ss.created_at ASC
"""

_USAGE_ROWS_SQL = """
    SELECT agent,
           model,
           call_type,
           prompt_tokens,
           completion_tokens,
           total_tokens,
           cached_tokens,
           reasoning_tokens,
           cost_usd,
           latency_ms,
           error,
           created_at
    FROM llm_usage
    WHERE run_id = $1
       OR user_id = $2
       OR subscription_id = ANY($3::uuid[])
    ORDER BY created_at ASC
"""


_WAIT_FOR_DISCOVERY_STARTED_SECONDS = 120.0
_WAIT_FOR_DISCOVERY_IDLE_SECONDS = 90.0
_WAIT_FOR_DISCOVERY_DEADLINE_SECONDS = 900.0


async def _fetch_discovery_activity(
    pool: asyncpg.Pool, subscription_ids: list[str]
) -> tuple[int, float | None]:
    """Return (count, max_created_at_epoch) for Finder/Discovery rows for these subs.

    Celery-dispatched discovery tags rows with ``subscription_id`` (not
    ``user_id``), so the wait has to key off the subscription IDs the
    Conversational Agent already created before the HTTP stream finished.
    """
    if not subscription_ids:
        return 0, None
    sub_uuids = [uuid.UUID(sid) for sid in subscription_ids]
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT COUNT(*) AS n, EXTRACT(EPOCH FROM MAX(created_at)) AS ts "
            "FROM llm_usage "
            "WHERE subscription_id = ANY($1::uuid[]) "
            "  AND agent IN ('finder','discovery')",
            sub_uuids,
        )
    n = int(row["n"] or 0)
    ts = float(row["ts"]) if row["ts"] is not None else None
    return n, ts


async def wait_for_discovery_to_finish(
    pool: asyncpg.Pool,
    subscription_ids: list[str],
    *,
    started_seconds: float = _WAIT_FOR_DISCOVERY_STARTED_SECONDS,
    idle_seconds: float = _WAIT_FOR_DISCOVERY_IDLE_SECONDS,
    deadline_seconds: float = _WAIT_FOR_DISCOVERY_DEADLINE_SECONDS,
) -> str:
    """Block until Celery-dispatched discovery is finished for these subs.

    Returns one of: "idle", "never_started", "timed_out", "no_subs".
    """
    import asyncio as _asyncio  # local to avoid shadowing module import

    if not subscription_ids:
        return "no_subs"

    start = _asyncio.get_event_loop().time()
    seen_any = False
    last_ts: float | None = None
    last_update_wall = start

    while True:
        now = _asyncio.get_event_loop().time()
        if now - start > deadline_seconds:
            return "timed_out"
        n, ts = await _fetch_discovery_activity(pool, subscription_ids)
        if n > 0:
            if not seen_any or (ts is not None and ts != last_ts):
                last_ts = ts
                last_update_wall = now
            seen_any = True
            if now - last_update_wall > idle_seconds:
                return "idle"
        elif now - start > started_seconds:
            return "never_started"
        await _asyncio.sleep(2.5)


async def fetch_user_subscriptions(pool: asyncpg.Pool, user_id: str) -> list[dict[str, Any]]:
    user_uuid = uuid.UUID(user_id)
    async with pool.acquire() as conn:
        sub_rows = await conn.fetch(_SUBSCRIPTIONS_SQL, user_uuid)
        out: list[dict[str, Any]] = []
        for sub in sub_rows:
            src_rows = await conn.fetch(_SOURCES_SQL, uuid.UUID(sub["id"]))
            sources = []
            by_kind: Counter[str] = Counter()
            for src in src_rows:
                kind = classify_source_kind(src["url"])
                by_kind[kind] += 1
                sources.append(
                    {
                        "url": src["url"],
                        "title": src["title"],
                        "kind": kind,
                        "is_user_specified": src["is_user_specified"],
                    }
                )
            out.append(
                {
                    "id": sub["id"],
                    "delivery_mode": sub["delivery_mode"],
                    "schedule_cron": sub["schedule_cron"],
                    "digest_language": sub["digest_language"],
                    "is_active": sub["is_active"],
                    "user_spec": sub["user_spec"],
                    "sources": sources,
                    "source_count": len(sources),
                    "sources_by_kind": dict(by_kind),
                }
            )
        return out


def _new_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "errors": 0,
    }


async def fetch_usage_for_prompt(
    pool: asyncpg.Pool,
    run_id: str,
    user_id: str,
    subscription_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pull every llm_usage row tied to this prompt's run_id or user, then roll up.

    The backend stamps ``run_id`` from the X-Run-Id header set per prompt, so
    filtering by run_id usually gives us exactly this turn's rows. We also
    union in rows tagged by user_id to catch any straggler writes (e.g.
    embeddings kicked off in ingest tasks that happen to fire at the same
    time but carry no run_id).
    """
    sub_uuids = [uuid.UUID(sid) for sid in subscription_ids]
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            _USAGE_ROWS_SQL,
            run_id,
            uuid.UUID(user_id),
            sub_uuids,
        )
    raw_rows = [dict(row) for row in rows]

    by_agent: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    by_call_type: dict[str, dict[str, Any]] = {}
    web_search_breakdown = {"ok": 0, "errored": 0, "cost_usd": 0.0}
    totals = _new_bucket()
    for r in raw_rows:
        agent_key = r.get("agent") or "(untagged)"
        model_key = r.get("model") or "(unknown)"
        call_type_key = r.get("call_type") or "(unknown)"
        pt = int(r.get("prompt_tokens") or 0)
        ct = int(r.get("completion_tokens") or 0)
        tt = int(r.get("total_tokens") or 0)
        cached = int(r.get("cached_tokens") or 0)
        reasoning = int(r.get("reasoning_tokens") or 0)
        cost_value = r.get("cost_usd")
        cost = float(cost_value) if cost_value is not None else 0.0
        errored = 1 if r.get("error") else 0

        for bucket, key in (
            (totals, None),
            (by_agent, agent_key),
            (by_model, model_key),
            (by_call_type, call_type_key),
        ):
            acc = bucket if key is None else bucket.setdefault(key, _new_bucket())
            acc["calls"] += 1
            acc["prompt_tokens"] += pt
            acc["completion_tokens"] += ct
            acc["total_tokens"] += tt
            acc["cached_tokens"] += cached
            acc["reasoning_tokens"] += reasoning
            acc["cost_usd"] += cost
            acc["errors"] += errored

        if call_type_key == "web_search":
            if errored:
                web_search_breakdown["errored"] += 1
            else:
                web_search_breakdown["ok"] += 1
            web_search_breakdown["cost_usd"] += cost

    totals["cost_usd"] = round(totals["cost_usd"], 8)
    web_search_breakdown["cost_usd"] = round(web_search_breakdown["cost_usd"], 8)
    for bucket in (by_agent, by_model, by_call_type):
        for acc in bucket.values():
            acc["cost_usd"] = round(acc["cost_usd"], 8)

    summary = {
        "totals": totals,
        "by_agent": by_agent,
        "by_model": by_model,
        "by_call_type": by_call_type,
        "web_search": web_search_breakdown,
    }
    return raw_rows, summary


async def run_one_prompt(
    http: httpx.AsyncClient,
    pool: asyncpg.Pool,
    api_url: str,
    run_id: str,
    prompt: dict[str, str],
) -> dict[str, Any]:
    prompt_run_id = f"{run_id}-{prompt['id']}"
    started = datetime.now(UTC)
    user_id: str | None = None
    api_key: str | None = None
    events: list[dict[str, Any]] = []
    final_message: str | None = None
    error: str | None = None
    subscriptions: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    usage_summary: dict[str, Any] = {}

    discovery_wait_outcome: str | None = None
    try:
        user_id, api_key = await create_user(http, api_url)
        await ack_onboarding(http, api_url, api_key)
        events, final_message = await stream_turn(
            http, api_url, api_key, prompt["text"], run_id=prompt_run_id
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    if user_id is not None:
        try:
            subscriptions = await fetch_user_subscriptions(pool, user_id)
        except Exception as exc:
            error = (error or "") + f" | db: {type(exc).__name__}: {exc}"
        try:
            discovery_wait_outcome = await wait_for_discovery_to_finish(
                pool, [sub["id"] for sub in subscriptions]
            )
        except Exception as exc:
            error = (error or "") + f" | wait: {type(exc).__name__}: {exc}"
        try:
            subscriptions = await fetch_user_subscriptions(pool, user_id)
        except Exception as exc:
            error = (error or "") + f" | db2: {type(exc).__name__}: {exc}"

        try:
            usage_rows, usage_summary = await fetch_usage_for_prompt(
                pool,
                run_id=prompt_run_id,
                user_id=user_id,
                subscription_ids=[sub["id"] for sub in subscriptions],
            )
        except Exception as exc:
            error = (error or "") + f" | usage: {type(exc).__name__}: {exc}"

    finished = datetime.now(UTC)
    return {
        "prompt_id": prompt["id"],
        "mode_hint": prompt["mode"],
        "prompt_text": prompt["text"],
        "prompt_run_id": prompt_run_id,
        "user_id": user_id,
        "api_key": api_key,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 2),
        "error": error,
        "final_message": final_message,
        "discovery_wait_outcome": discovery_wait_outcome,
        "events_summary": summarize_events(events),
        "events_raw": events,
        "subscriptions": subscriptions,
        "usage_summary": usage_summary,
        "usage_rows": usage_rows,
    }


def print_summary(run_artifact: dict[str, Any]) -> None:
    print("")
    print(f"run_id:    {run_artifact['run_id']}")
    print(f"duration:  {run_artifact['duration_seconds']}s")
    print(f"api_url:   {run_artifact['api_url']}")
    print("")
    header = (
        f"{'prompt_id':<28} {'mode':<7} {'dur':>6} {'subs':>4} "
        f"{'rss':>3} {'tg':>3} {'rdt':>3} {'calls':>5} "
        f"{'in_tok':>8} {'out_tok':>8} {'srch':>4} {'usd':>10}  status"
    )
    print(header)
    print("-" * len(header))
    run_totals = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "web_search_ok": 0,
        "web_search_err": 0,
    }
    for row in run_artifact["results"]:
        tot_rss = tot_tg = tot_rdt = 0
        for sub in row.get("subscriptions") or []:
            k = sub["sources_by_kind"]
            tot_rss += k.get("rss", 0)
            tot_tg += k.get("telegram_channel", 0)
            tot_rdt += k.get("reddit_subreddit", 0)
        usage_summary = row.get("usage_summary") or {}
        usage_totals = usage_summary.get("totals") or {}
        web_search = usage_summary.get("web_search") or {}
        calls = int(usage_totals.get("calls", 0))
        in_tok = int(usage_totals.get("prompt_tokens", 0))
        out_tok = int(usage_totals.get("completion_tokens", 0))
        cost = float(usage_totals.get("cost_usd", 0.0))
        srch_ok = int(web_search.get("ok", 0))
        srch_err = int(web_search.get("errored", 0))
        run_totals["calls"] += calls
        run_totals["prompt_tokens"] += in_tok
        run_totals["completion_tokens"] += out_tok
        run_totals["cost_usd"] += cost
        run_totals["web_search_ok"] += srch_ok
        run_totals["web_search_err"] += srch_err
        status = "ok" if row.get("error") is None else "ERROR"
        print(
            f"{row['prompt_id']:<28} {row['mode_hint']:<7} "
            f"{row['duration_seconds']:>6.1f} "
            f"{len(row.get('subscriptions') or []):>4} "
            f"{tot_rss:>3} {tot_tg:>3} {tot_rdt:>3} "
            f"{calls:>5} {in_tok:>8} {out_tok:>8} "
            f"{srch_ok:>4} "
            f"{cost:>10.6f}  {status}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<28} {'':<7} {'':>6} {'':>4} {'':>3} {'':>3} {'':>3} "
        f"{run_totals['calls']:>5} "
        f"{run_totals['prompt_tokens']:>8} "
        f"{run_totals['completion_tokens']:>8} "
        f"{run_totals['web_search_ok']:>4} "
        f"{run_totals['cost_usd']:>10.6f}"
    )
    if run_totals["web_search_err"]:
        print(f"(web-search failures across all prompts: {run_totals['web_search_err']})")
    print("")


def _collect_user_ids(results: list[Any]) -> list[str]:
    """Pull every user_id created during the run, including from errored prompts."""
    user_ids: list[str] = []
    for r in results:
        if isinstance(r, BaseException):
            continue
        uid = r.get("user_id") if isinstance(r, dict) else None
        if uid:
            user_ids.append(uid)
    return user_ids


async def delete_synthetic_users(pool: asyncpg.Pool, user_ids: list[str]) -> None:
    """Drop every user the baseline created plus any source it left orphaned.

    The schema cascades from users to subscriptions to subscription_sources,
    sent_items, and source_removal_log. Sources are not cascaded -- they are
    shared across users by design -- so we explicitly delete any source that
    has no surviving subscription_source link after the user delete.
    """
    if not user_ids:
        return
    uuids = [uuid.UUID(uid) for uid in user_ids]
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "DELETE FROM llm_usage WHERE user_id = ANY($1::uuid[])",
            uuids,
        )
        await conn.execute(
            "DELETE FROM users WHERE id = ANY($1::uuid[])",
            uuids,
        )
        await conn.execute(
            "DELETE FROM sources WHERE id NOT IN "
            "(SELECT DISTINCT source_id FROM subscription_sources)"
        )
        await conn.execute(
            "UPDATE sources s SET subscriber_count = COALESCE(sub.cnt, 0) "
            "FROM (SELECT source_id, COUNT(*)::int AS cnt FROM subscription_sources "
            "      GROUP BY source_id) sub "
            "WHERE s.id = sub.source_id"
        )
    print(f"[economics] cleaned up {len(user_ids)} synthetic users and orphaned sources")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--db-url", default=DEFAULT_DB_URL)
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Defaults to economics/results/<run_id>.json",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated prompt ids to run (default: all 10).",
    )
    args = parser.parse_args()

    run_id = uuid.uuid4().hex[:8]
    started = datetime.now(UTC)

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        prompts = [p for p in PROMPTS if p["id"] in wanted]
        if not prompts:
            raise SystemExit(f"No prompts matched --only={args.only!r}")
    else:
        prompts = list(PROMPTS)

    print(f"[economics] run_id={run_id} prompts={len(prompts)} api={args.api_url}")

    async with httpx.AsyncClient() as http:
        pool = await asyncpg.create_pool(args.db_url, min_size=2, max_size=8)
        try:
            results = await asyncio.gather(
                *(run_one_prompt(http, pool, args.api_url, run_id, p) for p in prompts),
                return_exceptions=True,
            )
            user_ids = _collect_user_ids(results)
            try:
                await delete_synthetic_users(pool, user_ids)
            except Exception as exc:
                print(f"[economics] cleanup failed for {len(user_ids)} users: {exc}")
        finally:
            await pool.close()

    finalized: list[dict[str, Any]] = []
    for prompt, result in zip(prompts, results, strict=True):
        if isinstance(result, BaseException):
            finalized.append(
                {
                    "prompt_id": prompt["id"],
                    "mode_hint": prompt["mode"],
                    "prompt_text": prompt["text"],
                    "error": f"{type(result).__name__}: {result}",
                    "traceback": "".join(
                        traceback.format_exception(type(result), result, result.__traceback__)
                    ),
                    "events_summary": {},
                    "events_raw": [],
                    "subscriptions": [],
                    "usage_summary": {},
                    "usage_rows": [],
                    "duration_seconds": 0.0,
                }
            )
        else:
            finalized.append(result)

    finished = datetime.now(UTC)
    run_artifact = {
        "run_id": run_id,
        "api_url": args.api_url,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 2),
        "prompt_count": len(prompts),
        "results": finalized,
    }

    out_path = (
        Path(args.out)
        if args.out
        else (Path(__file__).resolve().parent / "results" / f"{run_id}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(run_artifact, indent=2, default=str))

    print_summary(run_artifact)
    print(f"[economics] wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
