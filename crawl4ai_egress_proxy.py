"""Crawl4AI v0.9.1 egress-proxy overlay for the isolated Docker profile.

The upstream server points Chromium at this localhost HTTP proxy. This derived
version performs a DNS-free destination precheck, then delegates hostname
resolution, public-address validation, and connection pinning to safe-egress.
"""

from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import urlsplit

from egress_policy import (
    DEFAULT_ALLOWED_PORTS,
    DestinationPolicyError,
    parse_allowed_ports,
    parse_denied_networks,
    validate_http_url_without_dns,
)
from socks5_client import open_socks5_connection


logger = logging.getLogger("crawl4ai.egress")

_SOCKS_HOST = os.environ.get("CRAWL4AI_EGRESS_SOCKS_HOST", "").strip()
_SOCKS_PORT = int(os.environ.get("CRAWL4AI_EGRESS_SOCKS_PORT", "1080"))
_SOCKS_TIMEOUT_SECONDS = max(
    0.1, float(os.environ.get("CRAWL4AI_EGRESS_CONNECT_TIMEOUT_SECONDS", "30"))
)
_ALLOWED_PORTS = parse_allowed_ports(
    os.environ.get("SAFE_EGRESS_ALLOWED_PORTS", DEFAULT_ALLOWED_PORTS)
)
_DENIED_NETWORKS = parse_denied_networks(os.environ.get("SAFE_EGRESS_DENY_CIDRS", ""))
if not _SOCKS_HOST:
    raise RuntimeError("CRAWL4AI_EGRESS_SOCKS_HOST is required")

_CONNECT_OK = b"HTTP/1.1 200 Connection established\r\n\r\n"
_BLOCKED = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 11\r\n\r\nURL blocked"
_BAD = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nBad Request"
_MAX_HEADER_BYTES = 64 * 1024


async def _open_proxy_connection(host: str, port: int):
    return await open_socks5_connection(
        _SOCKS_HOST,
        _SOCKS_PORT,
        host,
        port,
        timeout_seconds=_SOCKS_TIMEOUT_SECONDS,
    )


def _validate_target(url: str) -> tuple[str, int]:
    return validate_http_url_without_dns(
        url,
        allowed_ports=_ALLOWED_PORTS,
        denied_networks=_DENIED_NETWORKS,
    )


class PinningProxy:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._host = host
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self.bound_host: str | None = None
        self.bound_port: int | None = None

    @property
    def url(self) -> str | None:
        if self.bound_port is None:
            return None
        return f"http://{self.bound_host}:{self.bound_port}"

    async def start(self) -> str:
        self._server = await asyncio.start_server(self._handle, self._host, self._port)
        socket = self._server.sockets[0]
        self.bound_host, self.bound_port = socket.getsockname()[:2]
        logger.info("egress pinning proxy listening on %s", self.url)
        return self.url

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not request_line:
                return
            parts = request_line.split()
            if len(parts) < 3:
                await self._reply(writer, _BAD)
                return
            method = parts[0].decode("latin-1", "replace").upper()
            target = parts[1].decode("latin-1", "replace")

            if method == "CONNECT":
                await self._handle_connect(target, reader, writer)
            else:
                await self._handle_absolute(method, target, reader, writer)
        except asyncio.TimeoutError:
            await self._reply(writer, _BAD)
        except Exception as exc:
            logger.debug("proxy connection error: %s", type(exc).__name__)
            await self._safe_close(writer)

    async def _handle_connect(self, target, client_reader, client_writer):
        try:
            parsed = urlsplit(f"https://{target}")
            if parsed.port is None or parsed.path or parsed.query or parsed.fragment:
                raise DestinationPolicyError("CONNECT requires a host and port")
            host, port = _validate_target(f"https://{target}")
        except (DestinationPolicyError, ValueError):
            await self._reply(client_writer, _BAD)
            return

        await self._drain_headers(client_reader)
        try:
            upstream_reader, upstream_writer = await _open_proxy_connection(host, port)
        except Exception:
            await self._reply(client_writer, _BLOCKED)
            return

        client_writer.write(_CONNECT_OK)
        await client_writer.drain()
        await self._splice(
            client_reader,
            client_writer,
            upstream_reader,
            upstream_writer,
        )

    async def _handle_absolute(self, method, target, client_reader, client_writer):
        try:
            parsed = urlsplit(target)
            if parsed.scheme != "http" or not parsed.hostname:
                raise DestinationPolicyError("Absolute proxy requests must use HTTP")
            host, port = _validate_target(target)
        except (DestinationPolicyError, ValueError):
            await self._reply(client_writer, _BAD)
            return

        headers = await self._read_headers(client_reader)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        try:
            upstream_reader, upstream_writer = await _open_proxy_connection(host, port)
        except Exception:
            await self._reply(client_writer, _BLOCKED)
            return

        outbound = f"{method} {path} HTTP/1.1\r\n".encode("latin-1")
        host_header = f"[{host}]" if ":" in host else host
        outbound += b"Host: " + host_header.encode("ascii")
        if parsed.port is not None:
            outbound += f":{port}".encode("ascii")
        outbound += b"\r\n" + headers + b"\r\n"
        upstream_writer.write(outbound)
        await upstream_writer.drain()
        await self._splice(
            client_reader,
            client_writer,
            upstream_reader,
            upstream_writer,
        )

    async def _drain_headers(self, reader):
        consumed = 0
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            consumed += len(line)
            if line in (b"\r\n", b"\n", b"") or consumed > _MAX_HEADER_BYTES:
                return

    async def _read_headers(self, reader) -> bytes:
        buffer = b""
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30)
            if line in (b"\r\n", b"\n", b""):
                break
            buffer += line
            if len(buffer) > _MAX_HEADER_BYTES:
                break
        kept = []
        for line in buffer.split(b"\r\n"):
            name = line.split(b":", 1)[0].strip().lower()
            if name in (b"proxy-connection", b"proxy-authorization", b"host"):
                continue
            if line:
                kept.append(line)
        return (b"\r\n".join(kept) + b"\r\n") if kept else b""

    async def _splice(self, client_reader, client_writer, upstream_reader, upstream_writer):
        async def pipe(source, destination):
            try:
                while True:
                    data = await source.read(65536)
                    if not data:
                        break
                    destination.write(data)
                    await destination.drain()
            except Exception:
                pass
            finally:
                await self._safe_close(destination)

        await asyncio.gather(
            pipe(client_reader, upstream_writer),
            pipe(upstream_reader, client_writer),
        )

    async def _reply(self, writer, payload: bytes):
        try:
            writer.write(payload)
            await writer.drain()
        except Exception:
            pass
        await self._safe_close(writer)

    @staticmethod
    async def _safe_close(writer):
        try:
            if not writer.is_closing():
                writer.close()
                await writer.wait_closed()
        except Exception:
            pass
