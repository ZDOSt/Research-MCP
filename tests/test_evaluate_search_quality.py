from evaluate_search_quality import _summary, evaluate_case


def test_evaluate_case_scores_expected_evidence_fields():
    case = {
        "id": "example",
        "min_results": 2,
        "expected_domains": ["docs.example.com"],
        "expected_any_terms": ["install"],
        "expected_source_types": ["documentation_candidate"],
    }
    payload = {
        "cache": "miss",
        "evidence_summary": {"status": "sufficient"},
        "results": [
            {
                "title": "Install Example",
                "content": "Installation instructions",
                "url": "https://docs.example.com/install",
                "citation_url": "https://docs.example.com/install",
                "source_type": "documentation_candidate",
                "primary_source_candidate": True,
                "modifiedDate": "2026-08-01",
                "extraction_method": "direct",
            },
            {
                "title": "Independent guide",
                "content": "Install notes",
                "url": "https://independent.test/guide",
                "citation_url": "https://independent.test/guide",
                "source_type": "technical_reference",
                "extraction_method": "crawl4ai",
            },
        ],
    }
    row = evaluate_case(case, payload, status_code=200, latency_seconds=1.25)
    assert row["passed"] is True
    assert row["independent_source_count"] == 2
    assert row["primary_source_count"] == 1
    assert row["citation_count"] == 2


def test_evaluation_summary_aggregates_coverage_and_latency():
    rows = [
        {
            "passed": True,
            "latency_seconds": 1.0,
            "result_count": 2,
            "citation_count": 2,
            "extracted_source_count": 2,
            "primary_source_count": 1,
            "dated_source_count": 1,
        },
        {
            "passed": False,
            "latency_seconds": 3.0,
            "result_count": 1,
            "citation_count": 1,
            "extracted_source_count": 0,
            "primary_source_count": 0,
            "dated_source_count": 0,
        },
    ]
    summary = _summary(rows)
    assert summary["pass_rate"] == 0.5
    assert summary["median_latency_seconds"] == 2.0
    assert summary["citation_coverage"] == 1.0
