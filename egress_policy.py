"""Shared, dependency-free destination policy for all outbound web traffic."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias
from urllib.parse import urlsplit


DEFAULT_ALLOWED_PORTS = "80,443,8080,8443"
MAX_ALLOWED_PORTS = 32
MAX_DENIED_NETWORKS = 128

BLOCKED_HOSTNAMES = frozenset(
    {
        "instance-data.ec2.internal",
        "localhost",
        "localhost.localdomain",
        "metadata.azure.internal",
        "metadata.google",
        "metadata.google.internal",
    }
)
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal")
NAT64_WELL_KNOWN_NETWORK = ipaddress.ip_network("64:ff9b::/96")

IPNetwork: TypeAlias = ipaddress.IPv4Network | ipaddress.IPv6Network
IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address


class DestinationPolicyError(ValueError):
    """Raised when an outbound destination violates the web egress policy."""


def parse_allowed_ports(
    raw_value: str,
    setting_name: str = "SAFE_EGRESS_ALLOWED_PORTS",
) -> frozenset[int]:
    """Parse a bounded comma-separated destination-port allowlist."""
    ports: set[int] = set()
    items = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if len(items) > MAX_ALLOWED_PORTS:
        raise ValueError(f"{setting_name} cannot contain more than {MAX_ALLOWED_PORTS} entries")
    for item in items:
        try:
            port = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid {setting_name} item: {item}") from exc
        if not 1 <= port <= 65535:
            raise ValueError(f"Invalid {setting_name} item: {item}")
        ports.add(port)
    if not ports:
        raise ValueError(f"{setting_name} must contain at least one port")
    return frozenset(ports)


def parse_denied_networks(
    raw_value: str,
    setting_name: str = "SAFE_EGRESS_DENY_CIDRS",
) -> tuple[IPNetwork, ...]:
    """Parse a bounded comma-separated IP-network denylist."""
    networks: list[IPNetwork] = []
    items = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if len(items) > MAX_DENIED_NETWORKS:
        raise ValueError(f"{setting_name} cannot contain more than {MAX_DENIED_NETWORKS} entries")
    for item in items:
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError as exc:
            raise ValueError(f"Invalid {setting_name} item: {item}") from exc
    return tuple(networks)


def validate_destination_port(port: int, allowed_ports: Iterable[int]) -> int:
    """Return a permitted TCP destination port or reject it before DNS."""
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise DestinationPolicyError("Invalid destination port")
    if port not in allowed_ports:
        raise DestinationPolicyError(f"Destination port {port} is not allowed")
    return port


def _is_denied_address(address: IPAddress, denied_networks: Iterable[IPNetwork]) -> bool:
    for network in denied_networks:
        if address.version == network.version and address in network:
            return True
    return False


def _validate_address_object(
    address: IPAddress,
    denied_networks: tuple[IPNetwork, ...],
) -> None:
    # A host with NAT64 can otherwise translate a syntactically global IPv6 literal
    # into an IPv4 destination that was never represented in the URL policy check.
    if isinstance(address, ipaddress.IPv6Address) and address in NAT64_WELL_KNOWN_NETWORK:
        raise DestinationPolicyError("IPv4-embedded NAT64 destinations are not allowed")

    # Keep the explicit checks for consistent behavior across Python/ipaddress versions.
    if (
        not address.is_global
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    ):
        raise DestinationPolicyError("Destination is not globally routable")

    if _is_denied_address(address, denied_networks):
        raise DestinationPolicyError("Destination is denied by egress policy")

    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        mapped = address.ipv4_mapped
        _validate_address_object(mapped, denied_networks)


def validate_public_address(
    address: str,
    denied_networks: Iterable[IPNetwork] = (),
) -> str:
    """Return a canonical public address after applying the shared deny policy."""
    raw_address = str(address)
    if "%" in raw_address:
        raise DestinationPolicyError("Scoped destination addresses are not allowed")
    try:
        parsed = ipaddress.ip_address(raw_address)
    except ValueError as exc:
        raise DestinationPolicyError("Destination resolved to an invalid IP address") from exc
    normalized_denials = tuple(denied_networks)
    _validate_address_object(parsed, normalized_denials)
    return str(parsed)


def normalize_destination_host(
    host: str,
    denied_networks: Iterable[IPNetwork] = (),
) -> str:
    """Normalize a DNS name or validate an IP-literal destination."""
    normalized = str(host or "").strip().rstrip(".").lower()
    if not normalized or len(normalized) > 253 or "%" in normalized:
        raise DestinationPolicyError("Invalid destination hostname")
    if any(ord(character) < 33 or ord(character) == 127 for character in normalized):
        raise DestinationPolicyError("Invalid destination hostname")

    try:
        literal = ipaddress.ip_address(normalized)
    except ValueError:
        literal = None
    if literal is not None:
        return validate_public_address(str(literal), denied_networks)

    try:
        ascii_host = normalized.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise DestinationPolicyError("Invalid internationalized hostname") from exc
    if not ascii_host or len(ascii_host) > 253:
        raise DestinationPolicyError("Invalid destination hostname")
    if ascii_host in BLOCKED_HOSTNAMES or ascii_host.endswith(BLOCKED_HOST_SUFFIXES):
        raise DestinationPolicyError("Destination hostname is not allowed")
    if "." not in ascii_host:
        raise DestinationPolicyError("Single-label destination hostnames are not allowed")
    labels = ascii_host.split(".")
    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or any(not (character.isascii() and (character.isalnum() or character == "-")) for character in label)
        for label in labels
    ):
        raise DestinationPolicyError("Invalid destination hostname")

    # Browsers accept legacy dotted/octal/hex IPv4 spellings that ipaddress
    # intentionally rejects. Do not let such a spelling pass as a DNS name and
    # later be reinterpreted as a private IP by a different URL parser.
    try:
        socket.inet_aton(ascii_host)
    except OSError:
        pass
    else:
        raise DestinationPolicyError("Ambiguous IPv4 destination is not allowed")
    return ascii_host


def validate_http_url_without_dns(
    url: str,
    *,
    allowed_ports: Iterable[int],
    denied_networks: Iterable[IPNetwork] = (),
) -> tuple[str, int]:
    """Validate a proxied HTTP(S) URL without resolving its hostname locally.

    This is the preflight policy for network-isolated clients. The trusted
    egress broker must still resolve the returned hostname, validate every DNS
    answer, and pin the connection before any bytes reach the destination.
    """
    if not isinstance(url, str) or not url.strip():
        raise DestinationPolicyError("URL must be a non-empty string")

    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as exc:
        raise DestinationPolicyError("Invalid URL") from exc

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise DestinationPolicyError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise DestinationPolicyError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise DestinationPolicyError("URLs containing credentials are not allowed")

    normalized_denials = tuple(denied_networks)
    host = normalize_destination_host(parsed.hostname, normalized_denials)
    destination_port = port if port is not None else (443 if scheme == "https" else 80)
    validate_destination_port(destination_port, allowed_ports)
    return host, destination_port


def validate_dns_records(
    records: Sequence[Any],
    denied_networks: Iterable[IPNetwork] = (),
) -> tuple[str, ...]:
    """Require every getaddrinfo answer to contain an allowed destination."""
    addresses: set[str] = set()
    normalized_denials = tuple(denied_networks)
    for record in records:
        if not isinstance(record, tuple) or len(record) < 5:
            raise DestinationPolicyError("Destination hostname returned an invalid DNS answer")
        socket_address = record[4]
        if not isinstance(socket_address, tuple) or not socket_address:
            raise DestinationPolicyError("Destination hostname returned an invalid DNS answer")
        addresses.add(validate_public_address(str(socket_address[0]), normalized_denials))
    if not addresses:
        raise DestinationPolicyError("Destination hostname returned no addresses")
    return tuple(
        sorted(
            addresses,
            key=lambda value: (
                ipaddress.ip_address(value).version,
                int(ipaddress.ip_address(value)),
            ),
        )
    )


async def resolve_public_addresses(
    host: str,
    port: int,
    *,
    allowed_ports: Iterable[int],
    denied_networks: Iterable[IPNetwork] = (),
    dns_timeout_seconds: float = 5.0,
) -> tuple[str, ...]:
    """Resolve once, requiring the port and every DNS answer to satisfy policy."""
    validate_destination_port(port, allowed_ports)
    normalized_denials = tuple(denied_networks)
    normalized_host = normalize_destination_host(host, normalized_denials)

    try:
        literal = ipaddress.ip_address(normalized_host)
    except ValueError:
        literal = None
    if literal is not None:
        return (validate_public_address(str(literal), normalized_denials),)

    loop = asyncio.get_running_loop()
    try:
        records = await asyncio.wait_for(
            loop.getaddrinfo(
                normalized_host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            ),
            timeout=max(0.1, float(dns_timeout_seconds)),
        )
    except (TimeoutError, socket.gaierror) as exc:
        raise DestinationPolicyError("Unable to resolve destination hostname") from exc
    return validate_dns_records(records, normalized_denials)
