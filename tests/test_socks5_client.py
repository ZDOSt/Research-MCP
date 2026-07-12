import asyncio
import importlib.util
import ipaddress
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from socks5_client import Socks5Error, open_socks5_connection


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_crawl4ai_proxy_overlay():
    egress_broker = types.ModuleType("egress_broker")

    class EgressBlocked(Exception):
        pass

    egress_broker.EgressBlocked = EgressBlocked
    egress_broker.resolve_and_pin = lambda url: None
    spec = importlib.util.spec_from_file_location(
        "_test_crawl4ai_egress_proxy",
        PROJECT_ROOT / "crawl4ai_egress_proxy.py",
    )
    module = importlib.util.module_from_spec(spec)
    environment = {
        "CRAWL4AI_EGRESS_SOCKS_HOST": "safe-egress",
        "CRAWL4AI_EGRESS_SOCKS_PORT": "1080",
        "CRAWL4AI_EGRESS_CONNECT_TIMEOUT_SECONDS": "30",
    }
    with patch.dict(os.environ, environment, clear=False), patch.dict(
        sys.modules,
        {"egress_broker": egress_broker},
    ):
        spec.loader.exec_module(module)
    return module


class Socks5ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_crawl4ai_overlay_tunnels_the_pinned_ip(self):
        module = load_crawl4ai_proxy_overlay()
        pin = SimpleNamespace(ip="93.184.216.34", port=443)
        streams = (object(), object())
        open_tunnel = AsyncMock(return_value=streams)

        with patch.object(module, "open_socks5_connection", open_tunnel):
            result = await module._open_pinned_connection(pin)

        self.assertIs(result, streams)
        open_tunnel.assert_awaited_once_with(
            "safe-egress",
            1080,
            "93.184.216.34",
            443,
            timeout_seconds=30.0,
        )

    async def test_opens_ipv4_tunnel_and_transfers_bytes(self):
        destination = "93.184.216.34"
        destination_port = 443
        request_seen = asyncio.Event()

        async def handle(reader, writer):
            try:
                self.assertEqual(await reader.readexactly(3), b"\x05\x01\x00")
                writer.write(b"\x05\x00")
                await writer.drain()

                self.assertEqual(await reader.readexactly(4), b"\x05\x01\x00\x01")
                self.assertEqual(
                    await reader.readexactly(4),
                    ipaddress.IPv4Address(destination).packed,
                )
                self.assertEqual(
                    int.from_bytes(await reader.readexactly(2), "big"),
                    destination_port,
                )
                request_seen.set()
                writer.write(b"\x05\x00\x00\x01\x7f\x00\x00\x01\x1f\x90")
                await writer.drain()

                self.assertEqual(await reader.readexactly(4), b"ping")
                writer.write(b"pong")
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        proxy_port = server.sockets[0].getsockname()[1]
        writer = None
        try:
            reader, writer = await open_socks5_connection(
                "127.0.0.1",
                proxy_port,
                destination,
                destination_port,
            )
            await asyncio.wait_for(request_seen.wait(), timeout=1)
            writer.write(b"ping")
            await writer.drain()
            self.assertEqual(await reader.readexactly(4), b"pong")
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            server.close()
            await server.wait_closed()

    async def test_closes_connection_when_proxy_rejects_authentication(self):
        client_closed = asyncio.Event()

        async def handle(reader, writer):
            try:
                self.assertEqual(await reader.readexactly(3), b"\x05\x01\x00")
                writer.write(b"\x05\xff")
                await writer.drain()
                self.assertEqual(await reader.read(), b"")
                client_closed.set()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        proxy_port = server.sockets[0].getsockname()[1]
        try:
            with self.assertRaisesRegex(Socks5Error, "authentication method"):
                await open_socks5_connection(
                    "127.0.0.1",
                    proxy_port,
                    "example.com",
                    443,
                )
            await asyncio.wait_for(client_closed.wait(), timeout=1)
        finally:
            server.close()
            await server.wait_closed()


if __name__ == "__main__":
    unittest.main()
