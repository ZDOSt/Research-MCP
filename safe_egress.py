"""A small public-only SOCKS5 CONNECT broker for sandboxed web engines."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import logging
import os
import socket
from contextlib import suppress
from typing import Iterable

from egress_policy import (
    DEFAULT_ALLOWED_PORTS,
    DestinationPolicyError,
    IPNetwork,
    normalize_destination_host as _normalize_destination_host,
    parse_allowed_ports as _parse_allowed_ports,
    parse_denied_networks as _parse_denied_networks,
    resolve_public_addresses as _resolve_public_addresses,
    validate_public_address as _validate_public_address,
)


LOGGER = logging.getLogger("safe-egress")

SOCKS_VERSION = 5
SOCKS_AUTH_NONE = 0
SOCKS_AUTH_UNACCEPTABLE = 0xFF
SOCKS_COMMAND_CONNECT = 1
SOCKS_ADDRESS_IPV4 = 1
SOCKS_ADDRESS_DOMAIN = 3
SOCKS_ADDRESS_IPV6 = 4

REPLY_SUCCEEDED = 0
REPLY_GENERAL_FAILURE = 1
REPLY_NOT_ALLOWED = 2
REPLY_NETWORK_UNREACHABLE = 3
REPLY_HOST_UNREACHABLE = 4
REPLY_CONNECTION_REFUSED = 5
REPLY_COMMAND_UNSUPPORTED = 7
REPLY_ADDRESS_UNSUPPORTED = 8

# The broker is container-only and Compose publishes no port; it must listen on
# both of its Docker interfaces to bridge the internal sandbox to egress.
SAFE_EGRESS_HOST = os.getenv("SAFE_EGRESS_HOST", ipaddress.IPv4Address(0).compressed)
SAFE_EGRESS_PORT = int(os.getenv("SAFE_EGRESS_PORT", "1080"))
SAFE_EGRESS_DNS_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("SAFE_EGRESS_DNS_TIMEOUT_SECONDS", "5"))
)
SAFE_EGRESS_CONNECT_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("SAFE_EGRESS_CONNECT_TIMEOUT_SECONDS", "15"))
)
SAFE_EGRESS_HANDSHAKE_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("SAFE_EGRESS_HANDSHAKE_TIMEOUT_SECONDS", "5"))
)
SAFE_EGRESS_IDLE_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("SAFE_EGRESS_IDLE_TIMEOUT_SECONDS", "45"))
)
SAFE_EGRESS_MAX_CONNECTION_SECONDS = max(
    1.0, float(os.getenv("SAFE_EGRESS_MAX_CONNECTION_SECONDS", "120"))
)
SAFE_EGRESS_MAX_CONNECTIONS = max(1, int(os.getenv("SAFE_EGRESS_MAX_CONNECTIONS", "128")))
SAFE_EGRESS_QUEUE_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("SAFE_EGRESS_QUEUE_TIMEOUT_SECONDS", "1"))
)
SAFE_EGRESS_MAX_BYTES_PER_DIRECTION = max(
    65_536, int(os.getenv("SAFE_EGRESS_MAX_BYTES_PER_DIRECTION", str(64 * 1024 * 1024)))
)


EgressPolicyError = DestinationPolicyError


SAFE_EGRESS_ALLOWED_PORTS = _parse_allowed_ports(
    os.getenv("SAFE_EGRESS_ALLOWED_PORTS", DEFAULT_ALLOWED_PORTS)
)
SAFE_EGRESS_DENY_NETWORKS = _parse_denied_networks(os.getenv("SAFE_EGRESS_DENY_CIDRS", ""))


def validate_public_address(
    address: str,
    denied_networks: Iterable[IPNetwork] | None = None,
) -> str:
    """Return a canonical public address or reject it."""
    policy_denials = SAFE_EGRESS_DENY_NETWORKS if denied_networks is None else denied_networks
    return _validate_public_address(address, policy_denials)


def normalize_destination_host(host: str) -> str:
    return _normalize_destination_host(host, SAFE_EGRESS_DENY_NETWORKS)


async def resolve_public_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve once and require every returned address to be public."""
    return await _resolve_public_addresses(
        host,
        port,
        allowed_ports=SAFE_EGRESS_ALLOWED_PORTS,
        denied_networks=SAFE_EGRESS_DENY_NETWORKS,
        dns_timeout_seconds=SAFE_EGRESS_DNS_TIMEOUT_SECONDS,
    )


async def open_pinned_connection(host: str, port: int):
    """Connect to an already validated address without resolving the hostname again."""
    addresses = await resolve_public_addresses(host, port)
    last_error: Exception | None = None
    for address in addresses:
        family = socket.AF_INET6 if ipaddress.ip_address(address).version == 6 else socket.AF_INET
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(address, port, family=family),
                timeout=SAFE_EGRESS_CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            last_error = exc
    raise ConnectionError("Unable to connect to validated public destination") from last_error


async def _read_exactly(reader: asyncio.StreamReader, size: int) -> bytes:
    return await asyncio.wait_for(
        reader.readexactly(size),
        timeout=SAFE_EGRESS_HANDSHAKE_TIMEOUT_SECONDS,
    )


async def _send_reply(writer: asyncio.StreamWriter, reply: int) -> None:
    writer.write(bytes((SOCKS_VERSION, reply, 0, SOCKS_ADDRESS_IPV4)) + b"\x00" * 6)
    await writer.drain()


async def _read_destination(reader: asyncio.StreamReader, address_type: int) -> str:
    if address_type == SOCKS_ADDRESS_IPV4:
        return str(ipaddress.IPv4Address(await _read_exactly(reader, 4)))
    if address_type == SOCKS_ADDRESS_IPV6:
        return str(ipaddress.IPv6Address(await _read_exactly(reader, 16)))
    if address_type == SOCKS_ADDRESS_DOMAIN:
        length = (await _read_exactly(reader, 1))[0]
        if length == 0:
            raise EgressPolicyError("Empty destination hostname")
        try:
            return (await _read_exactly(reader, length)).decode("idna")
        except UnicodeError as exc:
            raise EgressPolicyError("Invalid destination hostname") from exc
    raise EgressPolicyError("Unsupported SOCKS5 address type")


async def _relay(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    transferred = 0
    while True:
        data = await asyncio.wait_for(reader.read(65_536), timeout=SAFE_EGRESS_IDLE_TIMEOUT_SECONDS)
        if not data:
            with suppress(Exception):
                writer.write_eof()
                await writer.drain()
            return
        transferred += len(data)
        if transferred > SAFE_EGRESS_MAX_BYTES_PER_DIRECTION:
            raise EgressPolicyError("Connection exceeded the byte limit")
        writer.write(data)
        await writer.drain()


class SocksEgressServer:
    def __init__(self, max_connections: int = SAFE_EGRESS_MAX_CONNECTIONS):
        self._slots = asyncio.Semaphore(max_connections)

    async def handle_client(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        acquired = False
        try:
            await asyncio.wait_for(
                self._slots.acquire(),
                timeout=SAFE_EGRESS_QUEUE_TIMEOUT_SECONDS,
            )
            acquired = True
            try:
                await asyncio.wait_for(
                    self._handle_session(client_reader, client_writer),
                    timeout=SAFE_EGRESS_MAX_CONNECTION_SECONDS,
                )
            except (asyncio.IncompleteReadError, ConnectionError, OSError, asyncio.TimeoutError):
                pass
            except EgressPolicyError as exc:
                LOGGER.info("Blocked egress connection: %s", exc)
            except Exception:
                LOGGER.exception("Unexpected safe-egress connection failure")
        except asyncio.TimeoutError:
            pass
        finally:
            if acquired:
                self._slots.release()
            client_writer.close()
            with suppress(Exception):
                await client_writer.wait_closed()

    async def _handle_session(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> None:
        version, method_count = await _read_exactly(client_reader, 2)
        if version != SOCKS_VERSION or method_count == 0 or method_count > 16:
            return
        methods = await _read_exactly(client_reader, method_count)
        if SOCKS_AUTH_NONE not in methods:
            client_writer.write(bytes((SOCKS_VERSION, SOCKS_AUTH_UNACCEPTABLE)))
            await client_writer.drain()
            return
        client_writer.write(bytes((SOCKS_VERSION, SOCKS_AUTH_NONE)))
        await client_writer.drain()

        version, command, reserved, address_type = await _read_exactly(client_reader, 4)
        if version != SOCKS_VERSION or reserved != 0:
            await _send_reply(client_writer, REPLY_GENERAL_FAILURE)
            return
        if command != SOCKS_COMMAND_CONNECT:
            await _send_reply(client_writer, REPLY_COMMAND_UNSUPPORTED)
            return

        try:
            host = await _read_destination(client_reader, address_type)
        except EgressPolicyError:
            await _send_reply(client_writer, REPLY_ADDRESS_UNSUPPORTED)
            return
        port = int.from_bytes(await _read_exactly(client_reader, 2), "big")

        try:
            upstream_reader, upstream_writer = await open_pinned_connection(host, port)
        except EgressPolicyError:
            await _send_reply(client_writer, REPLY_NOT_ALLOWED)
            return
        except ConnectionRefusedError:
            await _send_reply(client_writer, REPLY_CONNECTION_REFUSED)
            return
        except ConnectionError:
            await _send_reply(client_writer, REPLY_HOST_UNREACHABLE)
            return

        await _send_reply(client_writer, REPLY_SUCCEEDED)
        try:
            await asyncio.gather(
                _relay(client_reader, upstream_writer),
                _relay(upstream_reader, client_writer),
            )
        finally:
            upstream_writer.close()
            with suppress(Exception):
                await upstream_writer.wait_closed()


async def healthcheck(host: str = "127.0.0.1", port: int = SAFE_EGRESS_PORT) -> bool:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=2.0,
        )
        writer.write(bytes((SOCKS_VERSION, 1, SOCKS_AUTH_NONE)))
        await writer.drain()
        response = await asyncio.wait_for(reader.readexactly(2), timeout=2.0)
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()
        return response == bytes((SOCKS_VERSION, SOCKS_AUTH_NONE))
    except Exception:
        return False


async def serve() -> None:
    handler = SocksEgressServer()
    server = await asyncio.start_server(handler.handle_client, SAFE_EGRESS_HOST, SAFE_EGRESS_PORT)
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    LOGGER.info("Safe egress SOCKS5 broker listening on %s", addresses)
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Public-only SOCKS5 egress broker")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.healthcheck:
        return 0 if asyncio.run(healthcheck()) else 1
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
