from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

import shared


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FixedDatetime:
    @staticmethod
    def now(tz=None):
        value = datetime(2026, 7, 13, 2, 27, 16, tzinfo=timezone.utc)
        return value if tz is None else value.astimezone(tz)


def test_runtime_retrieval_context_includes_configured_local_time(monkeypatch):
    monkeypatch.setattr(shared, "datetime", _FixedDatetime)
    monkeypatch.setattr(shared, "RESEARCH_TIMEZONE_NAME", "America/New_York")
    monkeypatch.setattr(
        shared,
        "RESEARCH_TIMEZONE",
        ZoneInfo("America/New_York"),
    )

    context = shared.runtime_retrieval_context()

    assert context["retrieved_at_utc"] == "2026-07-13T02:27:16+00:00"
    assert context["current_date_utc"] == "2026-07-13"
    assert context["timezone"] == "America/New_York"
    assert context["retrieved_at_local"] == "2026-07-12T22:27:16-04:00"
    assert context["current_date_local"] == "2026-07-12"
    assert "current_date_local" in context["guidance"]


def test_runtime_retrieval_context_defaults_to_utc(monkeypatch):
    monkeypatch.setattr(shared, "datetime", _FixedDatetime)
    monkeypatch.setattr(shared, "RESEARCH_TIMEZONE_NAME", "UTC")
    monkeypatch.setattr(shared, "RESEARCH_TIMEZONE", ZoneInfo("UTC"))

    context = shared.runtime_retrieval_context()

    assert context["timezone"] == "UTC"
    assert context["retrieved_at_local"] == context["retrieved_at_utc"]
    assert context["current_date_local"] == context["current_date_utc"]


@pytest.mark.parametrize("value", ["Not/A_Zone", "/etc/passwd", "../UTC"])
def test_research_timezone_rejects_invalid_iana_names(value):
    with pytest.raises(ValueError, match="valid IANA timezone name"):
        shared._load_research_timezone(value)


def test_compose_propagates_research_timezone_to_gateway_and_worker():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text("utf-8"))

    for service_name in ("mcp-gateway", "research-worker"):
        assert compose["services"][service_name]["environment"][
            "RESEARCH_TIMEZONE"
        ] == "${RESEARCH_TIMEZONE:-UTC}"
