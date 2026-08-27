"""Backend-neutral registry for production localizer providers.

The registry is intentionally dependency-light so site profiles can validate a
backend before importing CUDA/model implementations.  The production factory
attaches the concrete builders when it is imported; profile validation only
needs the static capabilities below.
"""

from __future__ import annotations

from pose_types import (
    LOCALIZATION_POSE_CONTRACT_VERSION,
    LOCALIZER_PROVIDER_API_VERSION,
    LocalizerCapabilities,
    LocalizerProvider,
)


_PROVIDERS: dict[str, LocalizerProvider] = {}


def register_localizer_provider(provider: LocalizerProvider) -> LocalizerProvider:
    """Register or attach a builder to a named provider.

    Re-registering the same capability contract is allowed so a factory reload
    can replace a stale function object.  A changed contract is rejected to
    avoid silently changing site-profile validation at runtime.
    """
    name = provider.name.strip().lower()
    if not name or name != provider.name:
        raise ValueError("localizer provider name must be a normalized non-empty string")
    if provider.capabilities.name != name:
        raise ValueError("localizer provider capabilities name must match provider name")
    if provider.capabilities.provider_api_version != LOCALIZER_PROVIDER_API_VERSION:
        raise ValueError(
            "localizer provider API version mismatch: "
            f"expected {LOCALIZER_PROVIDER_API_VERSION}, "
            f"got {provider.capabilities.provider_api_version}"
        )
    if provider.capabilities.pose_contract_version != LOCALIZATION_POSE_CONTRACT_VERSION:
        raise ValueError(
            "localizer pose contract version mismatch: "
            f"expected {LOCALIZATION_POSE_CONTRACT_VERSION}, "
            f"got {provider.capabilities.pose_contract_version}"
        )
    existing = _PROVIDERS.get(name)
    if existing is not None and existing.capabilities != provider.capabilities:
        raise ValueError(f"localizer provider {name!r} capability contract conflict")
    _PROVIDERS[name] = provider
    return provider


def registered_localizer_names() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def get_localizer_provider(name: str) -> LocalizerProvider:
    selected = str(name).strip().lower()
    provider = _PROVIDERS.get(selected)
    if provider is None:
        known = ", ".join(registered_localizer_names()) or "<none>"
        raise ValueError(
            f"unsupported localizer backend {name!r}; registered localizer backends: {known}"
        )
    return provider


def get_localizer_capabilities(name: str) -> LocalizerCapabilities:
    return get_localizer_provider(name).capabilities


register_localizer_provider(
    LocalizerProvider(
        name="edm",
        capabilities=LocalizerCapabilities(
            name="edm",
            required_assets=("localizer_profile",),
            optional_assets=("reference_index",),
            supports_production_profile=True,
        ),
    )
)
register_localizer_provider(
    LocalizerProvider(
        name="xfeat",
        capabilities=LocalizerCapabilities(
            name="xfeat",
            optional_assets=("megaloc_cache", "track_landmarks", "reference_index"),
            unsupported_assets=("localizer_profile",),
            supports_production_profile=False,
        ),
    )
)


__all__ = [
    "get_localizer_capabilities",
    "get_localizer_provider",
    "register_localizer_provider",
    "registered_localizer_names",
]
