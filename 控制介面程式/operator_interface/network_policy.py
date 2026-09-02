"""Offline-first network policy: env vars for offline model downloads, and a
Python-level socket guard that denies outbound destinations the interface
mode does not allow.

Split out of the former runtime_safety.py (D2). See session_logs.py for
session JSONL logs, disk_policy.py for disk-pressure/retention, and
arming_gate.py for the autonomy arming gate.
"""
from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any, Callable, Iterable

from backend_contract import InterfaceMode


def configure_offline_environment() -> None:
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "NO_PROXY": "127.0.0.1,localhost,192.168.42.1,192.168.53.1",
    }.items():
        os.environ[name] = value


def network_destination_allowed(
    mode: InterfaceMode,
    host: str,
    *,
    allowed_real_hosts: Iterable[str],
) -> bool:
    text = str(host or "").strip().strip("[]")
    if text.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    if mode is InterfaceMode.SIMULATED_STREAM:
        return False
    allowed = set(str(value).strip().strip("[]") for value in allowed_real_hosts)
    return text in allowed


_NETWORK_GUARD_INSTALLED = False


def _network_address_host(address: Any) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    # AF_UNIX path: local IPC is allowed.
    if isinstance(address, (str, bytes)):
        return None
    return ""


def _require_network_destination(
    mode: InterfaceMode, host: str, allowed: tuple[str, ...]
) -> None:
    if not network_destination_allowed(mode, host, allowed_real_hosts=allowed):
        raise PermissionError(
            f"offline network policy blocked destination {host!r} in {mode.value}"
        )


def _make_guarded_socket_call(
    original: Callable[..., Any], mode: InterfaceMode, allowed: tuple[str, ...]
) -> Callable[..., Any]:
    def guarded(sock: socket.socket, address: Any) -> Any:
        host = _network_address_host(address)
        if host is not None:
            _require_network_destination(mode, host, allowed)
        return original(sock, address)

    return guarded


def _make_guarded_sendto(
    original: Callable[..., Any], mode: InterfaceMode, allowed: tuple[str, ...]
) -> Callable[..., Any]:
    def guarded(sock: socket.socket, data: Any, *args: Any) -> Any:
        if not args:
            return original(sock, data)
        address = args[-1]
        host = _network_address_host(address)
        if host is not None:
            _require_network_destination(mode, host, allowed)
        return original(sock, data, *args)

    return guarded


def _make_guarded_getaddrinfo(
    original: Callable[..., Any], mode: InterfaceMode, allowed: tuple[str, ...]
) -> Callable[..., Any]:
    def guarded(host: Any, *args: Any, **kwargs: Any) -> Any:
        _require_network_destination(mode, str(host), allowed)
        return original(host, *args, **kwargs)

    return guarded


def _make_guarded_create_connection(
    original: Callable[..., Any], mode: InterfaceMode, allowed: tuple[str, ...]
) -> Callable[..., Any]:
    def guarded(address: Any, *args: Any, **kwargs: Any) -> Any:
        host = _network_address_host(address)
        if host is not None:
            _require_network_destination(mode, host, allowed)
        return original(address, *args, **kwargs)

    return guarded


def install_network_guard(
    mode: InterfaceMode,
    *,
    allowed_real_hosts: Iterable[str] = (),
) -> None:
    """Deny Python-level outbound sockets outside the selected offline policy."""
    global _NETWORK_GUARD_INSTALLED
    if _NETWORK_GUARD_INSTALLED:
        return
    allowed = tuple(allowed_real_hosts)
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection
    guarded_call = _make_guarded_socket_call(original_connect, mode, allowed)
    guarded_connect_ex = _make_guarded_socket_call(
        original_connect_ex, mode, allowed
    )
    guarded_sendto = _make_guarded_sendto(original_sendto, mode, allowed)
    guarded_getaddrinfo = _make_guarded_getaddrinfo(
        original_getaddrinfo, mode, allowed
    )
    guarded_create_connection = _make_guarded_create_connection(
        original_create_connection, mode, allowed
    )
    socket.socket.connect = guarded_call  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = guarded_sendto  # type: ignore[method-assign]
    socket.getaddrinfo = guarded_getaddrinfo
    socket.create_connection = guarded_create_connection
    _NETWORK_GUARD_INSTALLED = True


__all__ = [
    "configure_offline_environment",
    "install_network_guard",
    "network_destination_allowed",
]
