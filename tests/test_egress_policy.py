import ipaddress
import socket
import unittest
from unittest.mock import patch

import crawler
import safe_egress


PUBLIC_ADDRESS = "93.184.216.34"
SECOND_PUBLIC_ADDRESS = "8.8.8.8"
PUBLIC_DNS_RECORDS = [
    (socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_ADDRESS, 443)),
]


class _FakeNetworkStream:
    def get_extra_info(self, name):
        return (PUBLIC_ADDRESS, 443) if name == "server_addr" else None


class _RedirectResponse:
    status_code = 302
    encoding = "utf-8"
    extensions = {"network_stream": _FakeNetworkStream()}

    def __init__(self, location: str):
        self.headers = {"location": location}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        return None


class _RedirectClient:
    def __init__(self, location: str):
        self.location = location
        self.request_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method, url, **kwargs):
        self.request_count += 1
        return _RedirectResponse(self.location)


class SharedEgressPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_direct_and_broker_default_to_the_same_web_ports(self):
        expected = frozenset({80, 443, 8080, 8443})
        self.assertEqual(crawler.SAFE_EGRESS_ALLOWED_PORTS, expected)
        self.assertEqual(safe_egress.SAFE_EGRESS_ALLOWED_PORTS, expected)

    async def test_direct_url_rejects_arbitrary_tcp_port_before_dns(self):
        with patch("socket.getaddrinfo") as getaddrinfo:
            with self.assertRaises(crawler.UnsafeURLError):
                await crawler.validate_url_safety("https://example.com:22/private-scan")
        getaddrinfo.assert_not_called()

    async def test_direct_and_broker_both_apply_configured_deny_cidrs(self):
        denied = (ipaddress.ip_network(f"{PUBLIC_ADDRESS}/32"),)
        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS), patch.object(
            crawler, "SAFE_EGRESS_DENY_NETWORKS", denied
        ), patch.object(safe_egress, "SAFE_EGRESS_DENY_NETWORKS", denied):
            with self.assertRaises(crawler.UnsafeURLError) as direct_error:
                await crawler.validate_url_safety("https://same-vps.example/")
            with self.assertRaises(safe_egress.EgressPolicyError) as broker_error:
                await safe_egress.resolve_public_addresses("same-vps.example", 443)

        self.assertNotIn(PUBLIC_ADDRESS, str(direct_error.exception))
        self.assertNotIn(PUBLIC_ADDRESS, str(broker_error.exception))

    async def test_nat64_embedded_private_destination_is_rejected_in_both_paths(self):
        nat64_private = "64:ff9b::a00:1"
        with self.assertRaises(crawler.UnsafeURLError):
            await crawler.validate_url_safety(f"https://[{nat64_private}]/")
        with self.assertRaises(safe_egress.EgressPolicyError):
            await safe_egress.resolve_public_addresses(nat64_private, 443)

    async def test_nat64_dns_answer_is_rejected_in_both_paths(self):
        records = [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("64:ff9b::808:808", 443, 0, 0)),
        ]
        with patch("socket.getaddrinfo", return_value=records):
            with self.assertRaises(crawler.UnsafeURLError):
                await crawler.validate_url_safety("https://nat64.example/")
            with self.assertRaises(safe_egress.EgressPolicyError):
                await safe_egress.resolve_public_addresses("nat64.example", 443)

    async def test_redirect_revalidates_destination_port_before_following(self):
        client = _RedirectClient("https://redirect.example:22/admin")
        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS), patch(
            "crawler.httpx.AsyncClient", return_value=client
        ):
            with self.assertRaises(crawler.UnsafeURLError):
                await crawler.direct_fetch_url("https://public.example/start")

        self.assertEqual(client.request_count, 1)

    async def test_redirect_revalidates_deny_cidr_before_following(self):
        denied = (ipaddress.ip_network(f"{SECOND_PUBLIC_ADDRESS}/32"),)
        redirected_records = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (SECOND_PUBLIC_ADDRESS, 443)),
        ]
        client = _RedirectClient("https://same-vps.example/admin")
        with patch(
            "socket.getaddrinfo",
            side_effect=[PUBLIC_DNS_RECORDS, redirected_records],
        ), patch.object(crawler, "SAFE_EGRESS_DENY_NETWORKS", denied), patch(
            "crawler.httpx.AsyncClient", return_value=client
        ):
            with self.assertRaises(crawler.UnsafeURLError):
                await crawler.direct_fetch_url("https://public.example/start")

        self.assertEqual(client.request_count, 1)


if __name__ == "__main__":
    unittest.main()
