import asyncio
import ipaddress
from contextlib import suppress


class Socks5Error(ConnectionError):
    pass


def _destination_address(host: str) -> tuple[int, bytes]:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            encoded = host.encode("idna")
        except UnicodeError as exc:
            raise Socks5Error("invalid SOCKS5 destination") from exc
        if not encoded or len(encoded) > 255 or any(byte < 33 for byte in encoded):
            raise Socks5Error("invalid SOCKS5 destination")
        return 3, bytes([len(encoded)]) + encoded

    if isinstance(address, ipaddress.IPv4Address):
        return 1, address.packed
    return 4, address.packed


async def _close_writer(writer) -> None:
    with suppress(Exception):
        writer.close()
    with suppress(Exception):
        await writer.wait_closed()


async def open_socks5_connection(
    proxy_host: str,
    proxy_port: int,
    destination_host: str,
    destination_port: int,
    *,
    timeout_seconds: float = 30.0,
):
    """Open a no-auth SOCKS5 tunnel and return its asyncio streams."""
    proxy_host = str(proxy_host).strip()
    if not proxy_host:
        raise Socks5Error("SOCKS5 proxy host is required")
    if not 1 <= int(proxy_port) <= 65535:
        raise Socks5Error("invalid SOCKS5 proxy port")
    if not 1 <= int(destination_port) <= 65535:
        raise Socks5Error("invalid SOCKS5 destination port")

    address_type, encoded_address = _destination_address(str(destination_host).strip())
    writer = None
    try:
        async with asyncio.timeout(max(0.1, float(timeout_seconds))):
            reader, writer = await asyncio.open_connection(proxy_host, int(proxy_port))
            writer.write(b"\x05\x01\x00")
            await writer.drain()
            if await reader.readexactly(2) != b"\x05\x00":
                raise Socks5Error("SOCKS5 proxy rejected authentication method")

            request = (
                b"\x05\x01\x00"
                + bytes([address_type])
                + encoded_address
                + int(destination_port).to_bytes(2, "big")
            )
            writer.write(request)
            await writer.drain()

            response = await reader.readexactly(4)
            if response[:1] != b"\x05" or response[1] != 0:
                raise Socks5Error("SOCKS5 proxy rejected destination")

            reply_type = response[3]
            if reply_type == 1:
                await reader.readexactly(4)
            elif reply_type == 4:
                await reader.readexactly(16)
            elif reply_type == 3:
                length = (await reader.readexactly(1))[0]
                await reader.readexactly(length)
            else:
                raise Socks5Error("SOCKS5 proxy returned an invalid address type")
            await reader.readexactly(2)
            return reader, writer
    except Exception:
        if writer is not None:
            await _close_writer(writer)
        raise
