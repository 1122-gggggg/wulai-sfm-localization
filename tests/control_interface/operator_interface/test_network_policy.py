from __future__ import annotations

from types import SimpleNamespace

import pytest

import network_policy
from backend_contract import InterfaceMode
from network_policy import install_network_guard, network_destination_allowed


def test_network_policy_allows_only_loopback_for_sim_and_anafi_for_real() -> None:
    assert network_destination_allowed(
        InterfaceMode.SIMULATED_STREAM, "127.0.0.1", allowed_real_hosts=()
    )
    assert not network_destination_allowed(
        InterfaceMode.SIMULATED_STREAM, "8.8.8.8", allowed_real_hosts=()
    )
    assert network_destination_allowed(
        InterfaceMode.REAL_FLIGHT,
        "192.168.53.1",
        allowed_real_hosts=("192.168.53.1",),
    )
    assert not network_destination_allowed(
        InterfaceMode.REAL_FLIGHT,
        "example.com",
        allowed_real_hosts=("192.168.53.1",),
    )


def test_network_guard_blocks_disallowed_destinations_without_real_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def connect(self, address: object) -> tuple[str, object]:
            return "connect", address

        def connect_ex(self, address: object) -> tuple[str, object]:
            return "connect_ex", address

        def sendto(self, data: object, *args: object) -> tuple[str, object, tuple[object, ...]]:
            return "sendto", data, args

    fake_socket = SimpleNamespace(
        socket=FakeSocket,
        getaddrinfo=lambda host, *args, **kwargs: (host, args, kwargs),
        create_connection=lambda address, *args, **kwargs: (
            address,
            args,
            kwargs,
        ),
    )
    monkeypatch.setattr(network_policy, "socket", fake_socket)
    monkeypatch.setattr(network_policy, "_NETWORK_GUARD_INSTALLED", False)

    install_network_guard(
        InterfaceMode.REAL_FLIGHT,
        allowed_real_hosts=("192.168.53.1",),
    )
    sock = FakeSocket()
    with pytest.raises(
        PermissionError,
        match="offline network policy blocked destination '192.168.53.2'",
    ):
        sock.connect(("192.168.53.2", 443))
    assert sock.connect(("192.168.53.1", 443))[0] == "connect"
    assert sock.connect_ex(("127.0.0.1", 443))[0] == "connect_ex"
    assert sock.sendto(b"x", ("127.0.0.1", 443))[0] == "sendto"
    assert sock.sendto(b"x")[0] == "sendto"
    assert sock.connect("/tmp/local.sock")[0] == "connect"
    with pytest.raises(PermissionError):
        fake_socket.getaddrinfo("example.com")
    with pytest.raises(PermissionError):
        fake_socket.create_connection(("8.8.8.8", 443))

    guarded_connect = FakeSocket.connect
    install_network_guard(InterfaceMode.SIMULATED_STREAM)
    assert FakeSocket.connect is guarded_connect
