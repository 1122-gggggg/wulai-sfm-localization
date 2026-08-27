from __future__ import annotations

# Source modules are supplied by the repository's pytest pythonpath.
import pytest

from localizer_registry import (
    get_localizer_capabilities,
    get_localizer_provider,
    register_localizer_provider,
    registered_localizer_names,
)
from pose_types import LocalizerCapabilities, LocalizerProvider


def test_named_providers_expose_backend_capabilities() -> None:
    assert registered_localizer_names() == ("edm", "xfeat")

    edm = get_localizer_provider("edm")
    assert edm.name == "edm"
    assert edm.capabilities.required_assets == ("localizer_profile",)
    assert edm.capabilities.supports_production_profile is True

    xfeat = get_localizer_capabilities("xfeat")
    assert xfeat.name == "xfeat"
    assert xfeat.required_assets == ()
    assert xfeat.supports_production_profile is False


def test_unknown_backend_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported localizer backend"):
        get_localizer_provider("unknown")


def test_factory_attaches_named_provider_builders() -> None:
    import production_localizer_factory  # noqa: F401

    assert get_localizer_provider("edm").builder is not None
    assert get_localizer_provider("xfeat").builder is not None


def test_provider_with_incompatible_contract_version_is_rejected() -> None:
    provider = LocalizerProvider(
        name="future",
        capabilities=LocalizerCapabilities(
            name="future",
            provider_api_version=2,
        ),
    )

    with pytest.raises(ValueError, match="API version mismatch"):
        register_localizer_provider(provider)
