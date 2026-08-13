import ipaddress
import socket
from unittest.mock import patch

import httpx
import pytest

import gateway_fetch
import safe_egress
import search_gateway


PUBLIC_RECORDS = [
    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/",
        "http://service/",
        "http://metadata.google.internal/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://user:secret@example.com/",
        "https://example.com:22/private-scan",
    ],
)
@pytest.mark.asyncio
async def test_gateway_rejects_unsafe_page_destinations_before_dns(url):
    with patch("socket.getaddrinfo") as getaddrinfo:
        with pytest.raises(gateway_fetch.UnsafeURLError):
            await gateway_fetch.validate_public_url(url)
    getaddrinfo.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_rejects_mixed_public_and_private_dns_answers():
    records = PUBLIC_RECORDS + [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 443)),
    ]
    with patch("socket.getaddrinfo", return_value=records):
        with pytest.raises(gateway_fetch.UnsafeURLError):
            await gateway_fetch.validate_public_url("https://mixed.example/")


def test_optional_deny_list_blocks_public_vps_address():
    with pytest.raises(safe_egress.EgressPolicyError):
        safe_egress.validate_public_address(
            "93.184.216.34",
            [ipaddress.ip_network("93.184.216.34/32")],
        )


@pytest.mark.asyncio
async def test_gateway_health_returns_503_when_search_is_unavailable(monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            if "searxng" in url:
                raise httpx.ConnectError("offline")
            return httpx.Response(200)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    transport = httpx.ASGITransport(app=search_gateway.app)
    async with real_client(transport=transport, base_url="http://gateway") as client:
        response = await client.get("/healthz")
    assert response.status_code == 503
    assert response.json()["ok"] is False


@pytest.mark.asyncio
async def test_gateway_rejects_oversized_declared_request():
    transport = httpx.ASGITransport(app=search_gateway.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        response = await client.post(
            "/v1/research",
            content=b"{}",
            headers={"Content-Length": str(search_gateway.MAX_REQUEST_BYTES + 1)},
        )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_searx_compatible_endpoint_returns_gateway_results(monkeypatch):
    expected = {
        "query": "install example",
        "number_of_results": 1,
        "results": [
            {
                "title": "Guide",
                "url": "https://example.com/guide",
                "content": "Installation steps",
            }
        ],
    }

    async def fake_research(request):
        assert request.query == "install example"
        return expected

    monkeypatch.setattr(search_gateway, "research", fake_research)
    transport = httpx.ASGITransport(app=search_gateway.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        response = await client.get(
            "/search",
            params={"q": "install example", "format": "json"},
        )
    assert response.status_code == 200
    assert response.json() == expected
