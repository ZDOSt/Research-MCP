"""Opt-in live quality and latency evaluation for the search gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from evidence_quality import root_domain


DEFAULT_CASES = Path(__file__).with_name("evals") / "search_quality_cases.json"


def _domain(url: object) -> str:
    try:
        return (urlsplit(str(url or "")).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def evaluate_case(
    case: dict[str, Any],
    payload: dict[str, Any],
    *,
    status_code: int,
    latency_seconds: float,
) -> dict[str, Any]:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    min_results = int(case.get("min_results", 1))
    domains = [_domain(item.get("url")) for item in results if isinstance(item, dict)]
    owners = {root_domain(domain) for domain in domains if domain}
    searchable = "\n".join(
        f"{item.get('title', '')}\n{item.get('content', '')}".casefold()
        for item in results
        if isinstance(item, dict)
    )
    expected_domains = [str(value).casefold() for value in case.get("expected_domains", [])]
    expected_terms = [str(value).casefold() for value in case.get("expected_any_terms", [])]
    expected_types = {str(value) for value in case.get("expected_source_types", [])}
    source_types = {
        str(item.get("source_type")) for item in results if isinstance(item, dict)
    }
    domain_hit = not expected_domains or any(
        domain == expected or domain.endswith("." + expected)
        for domain in domains
        for expected in expected_domains
    )
    term_hit = not expected_terms or any(term in searchable for term in expected_terms)
    type_hit = not expected_types or bool(source_types & expected_types)
    citation_count = sum(
        bool(item.get("citation_url") or item.get("url"))
        for item in results
        if isinstance(item, dict)
    )
    extracted_count = sum(
        bool(item.get("extraction_method"))
        for item in results
        if isinstance(item, dict)
    )
    dated_count = sum(
        bool(
            item.get("publishedDate")
            or item.get("modifiedDate")
            or item.get("declaredDate")
        )
        for item in results
        if isinstance(item, dict)
    )
    primary_count = sum(
        bool(item.get("primary_source_candidate"))
        for item in results
        if isinstance(item, dict)
    )
    required_checks_passed = domain_hit and term_hit and type_hit
    passed = status_code == 200 and len(results) >= min_results and required_checks_passed
    return {
        "id": case.get("id"),
        "passed": passed,
        "status_code": status_code,
        "latency_seconds": round(latency_seconds, 3),
        "result_count": len(results),
        "independent_source_count": len(owners),
        "primary_source_count": primary_count,
        "dated_source_count": dated_count,
        "extracted_source_count": extracted_count,
        "citation_count": citation_count,
        "expected_domain_hit": domain_hit,
        "expected_term_hit": term_hit,
        "expected_source_type_hit": type_hit,
        "cache": payload.get("cache"),
        "evidence_status": (payload.get("evidence_summary") or {}).get("status"),
        "error": payload.get("error"),
    }


async def _run_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    base_url: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    request = {
        "query": case["query"],
        "mode": case.get("mode", "balanced"),
        "max_results": case.get("max_results", 5),
        "language": case.get("language", "auto"),
        "categories": case.get("categories", []),
    }
    if case.get("time_range"):
        request["time_range"] = case["time_range"]
    async with semaphore:
        started = time.monotonic()
        try:
            response = await client.post(f"{base_url}/v1/research", json=request)
            try:
                payload = response.json()
            except ValueError:
                payload = {"error": "non-JSON response"}
            if not isinstance(payload, dict):
                payload = {"error": "non-object response"}
            status_code = response.status_code
        except Exception as exc:
            payload = {"error": type(exc).__name__}
            status_code = 0
        latency_seconds = time.monotonic() - started
    return evaluate_case(
        case,
        payload,
        status_code=status_code,
        latency_seconds=latency_seconds,
    )


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = sorted(float(row["latency_seconds"]) for row in rows)
    percentile_index = max(0, min(len(latencies) - 1, round(0.95 * len(latencies)) - 1))
    total_results = sum(int(row["result_count"]) for row in rows)
    denominator = max(1, total_results)
    return {
        "case_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "pass_rate": round(sum(bool(row["passed"]) for row in rows) / max(1, len(rows)), 4),
        "median_latency_seconds": round(statistics.median(latencies), 3) if latencies else 0.0,
        "p95_latency_seconds": round(latencies[percentile_index], 3) if latencies else 0.0,
        "citation_coverage": round(sum(row["citation_count"] for row in rows) / denominator, 4),
        "extraction_coverage": round(sum(row["extracted_source_count"] for row in rows) / denominator, 4),
        "primary_source_coverage": round(sum(row["primary_source_count"] for row in rows) / denominator, 4),
        "dated_source_coverage": round(sum(row["dated_source_count"] for row in rows) / denominator, 4),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.getenv("SEARCH_GATEWAY_EVAL_URL", "http://127.0.0.1:8080"),
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--concurrency", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Evaluation case file must contain a JSON list")
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case.get("id") in selected]
    if not cases:
        raise ValueError("No evaluation cases selected")

    base_url = args.base_url.rstrip("/")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout), trust_env=False
    ) as client:
        rows = await asyncio.gather(
            *(_run_case(client, semaphore, base_url, case) for case in cases)
        )
    report = {
        "base_url": base_url,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": _summary(rows),
        "cases": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["summary"]["passed_count"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
