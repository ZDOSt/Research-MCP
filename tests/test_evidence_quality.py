from datetime import datetime, timezone

from evidence_quality import (
    evidence_summary,
    extract_version_markers,
    freshness_score,
    normalize_date,
    source_profile,
    stable_evidence_id,
    temporal_requirement,
)


def test_normalize_date_accepts_iso_rfc_and_epoch_values():
    assert normalize_date("2026-08-18T12:30:00Z") == "2026-08-18T12:30:00Z"
    assert normalize_date("Tue, 18 Aug 2026 12:30:00 GMT") == "2026-08-18T12:30:00Z"
    assert normalize_date(1787056200) == "2026-08-18T12:30:00Z"
    assert normalize_date("not a date") is None


def test_source_hierarchy_prefers_primary_candidates_without_claiming_ownership():
    government = source_profile("https://www.nist.gov/publications/example")
    docs = source_profile(
        "https://docs.example.com/install", query="install Example"
    )
    community = source_profile("https://www.reddit.com/r/example/comments/1")
    social = source_profile("https://www.pinterest.com/pin/1")

    assert government["source_tier"] == 1
    assert docs["source_type"] == "documentation_candidate"
    assert docs["classification_method"] == "domain-path-query heuristic"
    assert government["authority_score"] > docs["authority_score"] > community["authority_score"]
    assert social["source_tier"] == 5


def test_freshness_applies_only_to_temporally_sensitive_queries():
    now = datetime(2026, 8, 18, tzinfo=timezone.utc)
    assert temporal_requirement("What is the latest Docker release?") == "high"
    assert temporal_requirement("How do I install Docker?") == "moderate"
    assert temporal_requirement("Explain binary search") == "none"
    assert freshness_score("2026-08-17", "high", now=now) == 1.0
    assert freshness_score("2020-01-01", "high", now=now) == 0.05
    assert freshness_score("2026-08-17", "none", now=now) == 0.0


def test_version_markers_ignore_calendar_months_and_keep_software_versions():
    markers = extract_version_markers(
        "Ubuntu 24.04 supports Docker v27.1.2; report dated 2026.08"
    )
    assert "Ubuntu 24.04" in markers
    assert "27.1.2" in markers
    assert "2026.08" not in markers


def test_evidence_summary_reports_coverage_and_honest_limitations():
    results = [
        {
            "url": "https://docs.example.com/current",
            "primary_source_candidate": True,
            "modifiedDate": "2026-08-01",
            "extraction_method": "direct",
            "citation_url": "https://docs.example.com/current",
            "version_context": ["4.2.1"],
        },
        {
            "url": "https://independent.test/review",
            "extraction_method": "crawl4ai",
            "citation_url": "https://independent.test/review",
        },
    ]
    summary = evidence_summary(results, "latest Example 4.2 installation")
    assert summary["status"] == "sufficient"
    assert summary["independent_source_count"] == 2
    assert summary["primary_source_candidate_count"] == 1
    assert summary["warnings"] == []
    assert "does not prove" in summary["limitations"]


def test_evidence_ids_are_stable_across_fragments_and_default_ports():
    first = stable_evidence_id("https://Example.com:443/guide#section")
    second = stable_evidence_id("https://example.com/guide")
    assert first == second
    assert first.startswith("ev-")
