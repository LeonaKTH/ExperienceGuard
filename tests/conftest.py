"""Default regression tests must never contact a paid or external service."""
import ipaddress
import socket
import pytest


@pytest.fixture(autouse=True)
def offline_test_boundary(monkeypatch):
    for name in ("EG_V2_API_KEY", "EG_REMOTE_API_KEY", "EG_DEEPSEEK_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EG_V2_REMOTE_ENABLE", "false")
    monkeypatch.setenv("EG_REMOTE_ENABLE", "false")
    original = socket.socket.connect

    def local_connect(sock, address):
        host = address[0]
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = host == "localhost"
        if not allowed:
            pytest.fail("Regression suite forbids non-loopback network connections")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connect)
