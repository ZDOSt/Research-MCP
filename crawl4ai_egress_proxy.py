"""Crawl4AI v0.9.1 egress-proxy overlay for the isolated Docker profile.

The upstream server points Chromium at this localhost HTTP proxy. This derived
version retains upstream DNS validation and IP pinning, but opens the pinned
connection through Research MCP's public-only SOCKS5 broker.
"""

from __future__ import annotations

import asyncio
import logging
import os
from urllib.parse import urlsplit

from egress_broker import EgressBlocked, resolve_and_pin
from socks5_client import open_socks5_connection


logger = logging.getLogger("crawl4ai.egress")

_SOCKS_HOST = os.environ.get("CRAWL4AI_EGRESS_SOCKS_HOST", "").strip()
_SOCKS_PORT = int(os.environ.get("CRAWL4AI_EGRESS_SOCKS_PORT", "1080"))
_SOCKS_TIMEOUT_SECONDS = max(
    0.1, float(os.environ.get("CRAWL4AI_EGRESS_CONNECT_TIMEOUT_SECONDS", "30"))
)
if not _SOCKS_HOST:
    raise RuntimeError("CRAWL4AI_EGRESS_SOCKS_HOST is required")

_CONNECT_OK = b"HTTP/1.1 200 Connection established\r\n\r\n"
_BLOCKED = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 11\r\n\r\nURL blocked"
_BAD = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nBad Request"
_MAX_HEADER_BYTES = 64 * 1024


async def _open_pinned_connection(pin):
    return await open_socks5_connection(
        _SOCKS_HOST,
        _SOCKS_PORT,
        pin.ip,
        pin.port,
        timeout_seconds=_SOCKS_TIMEOUT_SECONDS,
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
        host, _, port_text = target.rpartition(":")
        if not host or not port_text.isdigit():
            await self._reply(client_writer, _BAD)
            return
        try:
            pin = resolve_and_pin(f"https://{host}:{port_text}")
        except EgressBlocked:
            await self._reply(client_writer, _BLOCKED)
            return

        await self._drain_headers(client_reader)
        try:
            upstream_reader, upstream_writer = await _open_pinned_connection(pin)
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
        parsed = urlsplit(target)
        if parsed.scheme != "http" or not parsed.hostname:
            await self._reply(client_writer, _BAD)
            return
        port = parsed.port or 80
        try:
            pin = resolve_and_pin(f"http://{parsed.hostname}:{port}")
        except EgressBlocked:
            await self._reply(client_writer, _BLOCKED)
            return

        headers = await self._read_headers(client_reader)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        try:
            upstream_reader, upstream_writer = await _open_pinned_connection(pin)
        except Exception:
            await self._reply(client_writer, _BLOCKED)
            return

        outbound = f"{method} {path} HTTP/1.1\r\n".encode("latin-1")
        outbound += b"Host: " + parsed.hostname.encode("latin-1")
        if parsed.port:
            outbound += f":{parsed.port}".encode("latin-1")
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
