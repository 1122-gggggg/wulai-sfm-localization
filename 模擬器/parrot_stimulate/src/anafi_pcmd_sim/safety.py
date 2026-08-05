"""Hard safety boundaries for a simulator-only experiment."""

from __future__ import annotations

SPHINX_DRONE_IP = "10.202.0.1"


class SphinxOnlyTargetError(ValueError):
    """Raised when an operation could target a physical aircraft."""


def require_sphinx_target(address: str) -> None:
    """Accept only Sphinx's documented virtual-drone endpoint."""
    if address != SPHINX_DRONE_IP:
        message = (
            f"Refusing target {address!r}. This prototype is Sphinx-only and accepts "
            f"only {SPHINX_DRONE_IP}."
        )
        raise SphinxOnlyTargetError(message)
