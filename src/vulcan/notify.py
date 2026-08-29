"""Minimal systemd sd_notify client — raw datagrams, no dependency.

Vulcan's runtime dependency posture is four packages; ``sd_notify`` is one
datagram, so it is implemented here rather than imported. Semantics:

* ``NOTIFY_SOCKET`` unset → every entry point is a no-op (dev, CI, smoke).
* ``NOTIFY_SOCKET`` set → :func:`start` announces ``READY=1`` and posts
  ``WATCHDOG=1`` heartbeats on the calling event loop at half of
  ``WATCHDOG_USEC`` — but only when ``WATCHDOG_PID`` is unset or names this
  process (sd_watchdog_enabled semantics). Heartbeats must run on the uvicorn loop: a wedged loop
  stops heartbeats, which is precisely the failure the watchdog exists to
  detect.
* A set-but-undeliverable ``NOTIFY_SOCKET`` fails startup loudly (systemd
  would kill the service anyway — loud beats silent).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

logger = logging.getLogger("vulcan.notify")


def _socket_path(address: str) -> str:
    """Translate systemd's address form: a leading ``@`` is the abstract
    namespace, transmitted as a NUL byte on the wire."""

    if address.startswith("@"):
        return "\0" + address[1:]
    return address


def _send(state: bytes) -> None:
    """One datagram to the notify socket; a no-op when notify is not in play."""

    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.connect(_socket_path(address))
        sock.sendall(state)


def notify_ready() -> None:
    """Announce startup; raises OSError when the socket is undeliverable."""

    _send(b"READY=1")


def _heartbeat_interval_seconds() -> float | None:
    # sd_watchdog_enabled semantics: a set WATCHDOG_PID addresses exactly one
    # process; heartbeating on another's behalf would mask that process's
    # death. Unset means the watchdog env is ours.
    pid = os.environ.get("WATCHDOG_PID")
    if pid is not None:
        try:
            if int(pid) != os.getpid():
                return None
        except ValueError:
            return None
    usec = os.environ.get("WATCHDOG_USEC")
    if not usec:
        return None
    try:
        value = int(usec)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value / 1_000_000 / 2


async def _heartbeat_loop() -> None:
    interval = _heartbeat_interval_seconds()
    if interval is None:
        return
    while True:
        try:
            _send(b"WATCHDOG=1")
        except OSError:
            # Stop heartbeating: the missed beats trip the watchdog, which is
            # the loud failure path this mechanism exists to provide. Say so
            # once — heartbeats otherwise vanish from the journal unexplained.
            logger.warning("heartbeat_stopped")
            return
        await asyncio.sleep(interval)


def start() -> asyncio.Task[None] | None:
    """Announce readiness and start heartbeats; None when notify is not in play.

    Must be called from the serving event loop (see module docstring). Raises
    OSError when NOTIFY_SOCKET is set but undeliverable.
    """

    if not os.environ.get("NOTIFY_SOCKET"):
        return None
    notify_ready()
    return asyncio.create_task(_heartbeat_loop())
