import pytest

from anafi_pcmd_sim.safety import SphinxOnlyTargetError, require_sphinx_target


def test_accepts_the_only_virtual_drone_endpoint() -> None:
    require_sphinx_target("10.202.0.1")


@pytest.mark.parametrize("address", ["192.168.42.1", "10.202.0.2", "localhost"])
def test_rejects_every_non_sphinx_endpoint(address: str) -> None:
    with pytest.raises(SphinxOnlyTargetError):
        require_sphinx_target(address)
