import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _compose():
    return yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text("utf-8"))


def test_only_gateway_joins_client_network_and_no_service_publishes_ports():
    compose = _compose()
    attached = []
    for name, service in compose["services"].items():
        assert "ports" not in service
        if "client" in service.get("networks", []):
            attached.append(name)
    assert attached == ["search-gateway"]
    assert compose["networks"]["client"]["external"] is True


def test_web_services_are_isolated_behind_egress_broker():
    compose = _compose()
    services = compose["services"]
    for name in ("crawl4ai", "web-runner"):
        assert services[name]["networks"] == ["web-sandbox"]
        assert "seccomp=./seccomp_profile.json" in services[name]["security_opt"]
    assert compose["networks"]["web-sandbox"]["internal"] is True
    assert set(services["safe-egress"]["networks"]) == {"web-sandbox", "egress"}


def test_search_and_pdf_control_networks_are_private():
    compose = _compose()
    services = compose["services"]
    assert compose["networks"]["backend"]["internal"] is True
    assert compose["networks"]["search-control"]["internal"] is True
    assert compose["networks"]["pdf-sandbox"]["internal"] is True
    assert "search-control" in services["search-gateway"]["networks"]
    assert services["searxng"]["networks"] == ["search-control", "egress"]
    assert services["pdf-runner"]["networks"] == ["pdf-sandbox"]


def test_runtime_services_are_hardened_and_have_resource_limits():
    compose = _compose()
    for name, service in compose["services"].items():
        assert service["read_only"] is True, name
        assert service["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in service["security_opt"], name
        assert "mem_limit" in service, name
        assert "cpus" in service, name


def test_documented_runtime_limits_are_wired_to_their_consumers():
    services = _compose()["services"]
    expected = {
        "search-gateway": {
            "GATEWAY_MAX_PASSAGE_CHARS",
            "GATEWAY_MAX_CONTENT_CHARS",
            "GATEWAY_QUERY_MAX_CHARS",
            "GATEWAY_MAX_REQUEST_BYTES",
            "GATEWAY_MAX_CONCURRENT_REQUESTS",
            "GATEWAY_ADMISSION_TIMEOUT_SECONDS",
            "GATEWAY_FINALIZATION_RESERVE_SECONDS",
            "GATEWAY_CANDIDATE_RERANKER_TIMEOUT_SECONDS",
            "GATEWAY_CACHE_MAX_ENTRIES",
            "SAFE_EGRESS_DNS_TIMEOUT_SECONDS",
        },
        "safe-egress": {
            "SAFE_EGRESS_ALLOWED_PORTS",
            "SAFE_EGRESS_DENY_CIDRS",
            "SAFE_EGRESS_DNS_TIMEOUT_SECONDS",
            "SAFE_EGRESS_CONNECT_TIMEOUT_SECONDS",
            "SAFE_EGRESS_HANDSHAKE_TIMEOUT_SECONDS",
            "SAFE_EGRESS_IDLE_TIMEOUT_SECONDS",
            "SAFE_EGRESS_MAX_CONNECTION_SECONDS",
            "SAFE_EGRESS_MAX_CONNECTIONS",
            "SAFE_EGRESS_QUEUE_TIMEOUT_SECONDS",
            "SAFE_EGRESS_MAX_BYTES_PER_DIRECTION",
        },
        "web-runner": {
            "CRAWL4AI_MAX_RESPONSE_BYTES",
            "WEB_RUNNER_MAX_REQUEST_BYTES",
            "WEB_RUNNER_MAX_RESPONSE_BYTES",
        },
        "pdf-runner": {
            "PDF_TIMEOUT_SECONDS",
            "PDF_SANDBOX_MEMORY_BYTES",
            "PDF_SANDBOX_CPU_SECONDS",
            "PDF_SANDBOX_OUTPUT_BYTES",
        },
    }
    for service_name, variables in expected.items():
        environment = services[service_name]["environment"]
        assert variables.issubset(environment), service_name


def test_searxng_provider_set_is_bounded_and_has_major_keyless_engines():
    settings = yaml.safe_load((PROJECT_ROOT / "searxng-settings.yml").read_text("utf-8"))
    engines = settings["use_default_settings"]["engines"]["keep_only"]
    assert {"startpage", "bing", "brave", "duckduckgo"}.issubset(engines)
    assert len(engines) <= 30
    assert "json" in settings["search"]["formats"]


def test_seccomp_profile_has_namespace_rule_for_chromium():
    profile = json.loads((PROJECT_ROOT / "seccomp_profile.json").read_text("utf-8"))
    assert any(
        set(rule.get("names", [])) == {"clone", "setns", "unshare"}
        and rule.get("action") == "SCMP_ACT_ALLOW"
        for rule in profile["syscalls"]
    )


def test_web_runner_waits_for_healthy_crawl4ai():
    dependency = _compose()["services"]["web-runner"]["depends_on"]["crawl4ai"]
    assert dependency["condition"] == "service_healthy"


def test_ci_validates_real_searxng_with_required_secrets():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    assert "SEARXNG_IMAGE: alpine" not in workflow
    assert "docker compose up -d --no-deps --wait searxng" in workflow
    assert workflow.count("SEARXNG_SECRET: ci-only-searxng-secret") >= 3
