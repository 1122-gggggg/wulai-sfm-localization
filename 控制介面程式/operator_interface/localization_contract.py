"""Validated, immutable boundary type for live localization results.

The worker still writes JSON and callers may keep using the original mapping.  This
module makes the safety-relevant subset explicit at the process boundary without
requiring the renderer or the tracker to understand a new object protocol.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, KeysView, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast


class InvalidLocalizationResult(ValueError):
    """A worker payload does not satisfy the localization boundary contract."""


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise InvalidLocalizationResult(f"{field_name} must be numeric")
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidLocalizationResult(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise InvalidLocalizationResult(f"{field_name} must be finite")
    return number


def _mono_ns(
    payload: Mapping[str, Any],
    canonical: str,
    *aliases: str,
) -> int:
    for name in (canonical, *aliases):
        if name not in payload or payload[name] is None:
            continue
        value = payload[name]
        if isinstance(value, bool):
            raise InvalidLocalizationResult(f"{canonical} must be integer nanoseconds")
        if name.endswith("_mono"):
            seconds = _finite_number(value, field_name=name)
            if seconds <= 0.0:
                raise InvalidLocalizationResult(f"{canonical} must be positive")
            return int(round(seconds * 1_000_000_000.0))
        if not isinstance(value, int):
            raise InvalidLocalizationResult(f"{canonical} must be integer nanoseconds")
        if value <= 0:
            raise InvalidLocalizationResult(f"{canonical} must be positive")
        return int(value)
    raise InvalidLocalizationResult(f"missing {canonical}")


def _validate_timestamps(
    capture_mono_ns: int,
    pose_mono_ns: int,
    now_mono_ns: int | None,
) -> None:
    current_mono_ns = time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
    if pose_mono_ns < capture_mono_ns:
        raise InvalidLocalizationResult("pose_mono_ns cannot precede capture_mono_ns")
    if capture_mono_ns > current_mono_ns:
        raise InvalidLocalizationResult("capture_mono_ns cannot be in the future")
    if pose_mono_ns > current_mono_ns:
        raise InvalidLocalizationResult("pose_mono_ns cannot be in the future")


def _pose_tuple(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        names = ("x", "y", "z")
        try:
            xyz = [_finite_number(value[name], field_name=f"pose.{name}") for name in names]
            yaw_value = value.get("yaw", value.get("yaw_raw"))
        except KeyError as exc:
            raise InvalidLocalizationResult(f"pose missing {exc.args[0]}") from exc
        if yaw_value is None:
            raise InvalidLocalizationResult("pose missing yaw")
        return (
            xyz[0],
            xyz[1],
            xyz[2],
            _finite_number(yaw_value, field_name="pose.yaw"),
        )
    if isinstance(value, (list, tuple)) and len(value) == 4:
        numbers = tuple(
            _finite_number(item, field_name=f"pose[{index}]") for index, item in enumerate(value)
        )
        return numbers[0], numbers[1], numbers[2], numbers[3]
    raise InvalidLocalizationResult("pose must be a four-value object or sequence")


@dataclass(frozen=True, slots=True)
class Pose:
    """Immutable, validated pose in the worker's map frame.

    ``stamp`` is a monotonic timestamp in seconds, matching the worker's
    ``capture_mono_ns``/``pose_mono_ns`` semantics but exposed as a plain
    ``float`` for callers that historically used ``stamp``.  The class is
    ``frozen`` with ``slots`` and validates finite numerics on construction
    through :meth:`from_dict` / :meth:`from_payload`.
    """

    x: float
    y: float
    z: float
    yaw: float
    stamp: float

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        now_mono_ns: int | None = None,
    ) -> "Pose":
        if not isinstance(payload, Mapping):
            raise InvalidLocalizationResult("payload must be an object")
        for field_name in ("x", "y", "z", "stamp"):
            if field_name not in payload or payload[field_name] is None:
                raise InvalidLocalizationResult(f"missing {field_name}")
        yaw_value = payload.get("yaw", payload.get("yaw_raw"))
        if yaw_value is None:
            raise InvalidLocalizationResult("missing yaw")
        x = _finite_number(payload["x"], field_name="x")
        y = _finite_number(payload["y"], field_name="y")
        z = _finite_number(payload["z"], field_name="z")
        yaw = _finite_number(yaw_value, field_name="yaw")
        stamp = _finite_number(payload["stamp"], field_name="stamp")
        return cls(x=x, y=y, z=z, yaw=yaw, stamp=stamp)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        now_mono_ns: int | None = None,
    ) -> "Pose":
        return cls.from_dict(payload, now_mono_ns=now_mono_ns)

    @classmethod
    def from_json(
        cls,
        raw: str | bytes | bytearray,
        *,
        now_mono_ns: int | None = None,
    ) -> "Pose":
        try:
            payload = json.loads(raw)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidLocalizationResult("payload is not valid JSON") from exc
        return cls.from_dict(payload, now_mono_ns=now_mono_ns)

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "z": self.z, "yaw": self.yaw, "yaw_raw": self.yaw, "stamp": self.stamp}

    def to_payload(self) -> dict[str, Any]:
        return self.to_dict()
    def __getitem__(self, key: str) -> Any:
        if key == "yaw_raw":
            return self.yaw
        if key in ("x", "y", "z", "yaw", "stamp"):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any | None = None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return key in ("x", "y", "z", "yaw", "yaw_raw", "stamp")

    def __setitem__(self, key: str, value: Any) -> None:
        # Pose is frozen; allow overlay for compat but update core fields when possible.
        if key in ("x", "y", "z", "yaw", "yaw_raw", "stamp"):
            # Use object.__setattr__ to mutate frozen slots for dict-like compat.
            target = "yaw" if key in ("yaw", "yaw_raw") else key
            object.__setattr__(self, target, float(value) if isinstance(value, (int, float)) else value)
            return
        raise KeyError(key)

    def keys(self) -> Iterator[str]:
        return ("x", "y", "z", "yaw", "stamp").__iter__()

    def __iter__(self) -> Iterator[str]:
        return iter(("x", "y", "z", "yaw", "stamp"))


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Validated subset of one worker response.

    ``capture_mono_ns`` and ``pose_mono_ns`` are host monotonic nanoseconds, not
    wall-clock timestamps.  ``pose`` is XYZ plus yaw in the worker's declared map
    frame; the original payload remains available through :meth:`to_payload`.
    """

    seq: int
    frame_id: str
    capture_mono_ns: int
    pose_mono_ns: int
    validity: bool
    confidence: float
    pose: tuple[float, float, float, float] | None
    _payload: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        now_mono_ns: int | None = None,
    ) -> "LocalizationResult":
        if not isinstance(payload, Mapping):
            raise InvalidLocalizationResult("payload must be an object")

        raw_seq = payload.get("seq", payload.get("display_seq"))
        if isinstance(raw_seq, bool) or not isinstance(raw_seq, int) or raw_seq < 0:
            raise InvalidLocalizationResult("seq must be a non-negative integer")

        raw_frame_id = payload.get("frame_id", payload.get("frame_name"))
        if not isinstance(raw_frame_id, str) or not raw_frame_id.strip():
            raise InvalidLocalizationResult("frame_id must be a non-empty string")

        capture_mono_ns = _mono_ns(
            payload,
            "capture_mono_ns",
            "source_frame_stamp_mono_ns",
            "source_frame_stamp_mono",
        )
        pose_mono_ns = _mono_ns(
            payload,
            "pose_mono_ns",
            "worker_core_done_mono_ns",
            "client_response_mono_ns",
        )
        _validate_timestamps(capture_mono_ns, pose_mono_ns, now_mono_ns)

        validity = payload.get("validity", payload.get("success"))
        if not isinstance(validity, bool):
            raise InvalidLocalizationResult("validity must be boolean")
        if "success" in payload and not isinstance(payload["success"], bool):
            raise InvalidLocalizationResult("success must be boolean")
        if "success" in payload and payload["success"] is not validity:
            raise InvalidLocalizationResult("success and validity disagree")

        confidence = payload.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise InvalidLocalizationResult("confidence must be numeric")
        confidence = _finite_number(confidence, field_name="confidence")
        if not 0.0 <= confidence <= 1.0:
            raise InvalidLocalizationResult("confidence must be in [0, 1]")

        pose = _pose_tuple(payload.get("pose"))
        if validity and pose is None:
            raise InvalidLocalizationResult("valid result requires pose")

        return cls(
            seq=int(raw_seq),
            frame_id=raw_frame_id.strip(),
            capture_mono_ns=capture_mono_ns,
            pose_mono_ns=pose_mono_ns,
            validity=validity,
            confidence=confidence,
            pose=pose,
            _payload=MappingProxyType(dict(payload)),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        now_mono_ns: int | None = None,
    ) -> "LocalizationResult":
        """Alias for :meth:`from_payload` retained for dict-style callers."""
        return cls.from_payload(payload, now_mono_ns=now_mono_ns)

    @classmethod
    def from_json(
        cls,
        raw: str | bytes | bytearray,
        *,
        now_mono_ns: int | None = None,
    ) -> "LocalizationResult":
        try:
            payload = json.loads(raw)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidLocalizationResult("payload is not valid JSON") from exc
        return cls.from_payload(payload, now_mono_ns=now_mono_ns)

    def to_payload(self) -> dict[str, Any]:
        """Return a dict-compatible copy with canonical safety fields present."""
        payload = dict(self._payload)
        payload.setdefault("seq", self.seq)
        payload.setdefault("frame_id", self.frame_id)
        payload.setdefault("capture_mono_ns", self.capture_mono_ns)
        payload.setdefault("pose_mono_ns", self.pose_mono_ns)
        payload.setdefault("validity", self.validity)
        payload.setdefault("confidence", self.confidence)
        return payload

    def __getitem__(self, key: str) -> Any:
        payload = self.to_payload()
        if key in payload:
            return payload[key]
        if key in self._payload:
            return self._payload[key]
        raise KeyError(key)

    def get(self, key: str, default: Any | None = None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        payload = self.to_payload()
        return key in payload or key in self._payload

    def __setitem__(self, key: str, value: Any) -> None:
        # Dict-like mutation for compat while keeping dataclass frozen semantics.
        # Bypass frozen check via object.__setattr__.
        new_payload = dict(self._payload)
        new_payload[key] = value
        object.__setattr__(self, "_payload", MappingProxyType(new_payload))

    def keys(self) -> KeysView[str]:
        return self.to_payload().keys()

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_payload())

    def __len__(self) -> int:
        return len(self.to_payload())

__all__ = ["InvalidLocalizationResult", "LocalizationResult", "Pose"]
