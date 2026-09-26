"""Direct Parrot ARSDK-NG connection, descriptor, and command bindings.

Load the pinned libraries from ``執行環境/native/arsdk-ng/lib`` by default.
``ARSDK_NG_LIB_DIR`` selects a different build when needed.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import errno
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable


ARSDK_NG_SOURCE_REVISION = "409daec0e8fc3393c31a8d09505a47d056eaafe9"

ARSDK_DEFAULT_PORT = 44445  # Discovery; 44444 is advertised for commands.
ARSDK_DEVICE_TYPE_ANAFI4K = 0x0914
ARSDK_DEVICE_TYPE_SKYCTRL_3 = 0x0918


class ArsdkNgError(RuntimeError):
    """ARSDK-NG load, connection, or command failure."""


class _EnumDesc(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("value", ctypes.c_int32)]


class _ArgDesc(ctypes.Structure):
    pass


_ArgDesc._fields_ = [
    ("name", ctypes.c_char_p),
    ("type", ctypes.c_int),
    ("enum_desc_table", ctypes.POINTER(_EnumDesc)),
    ("enum_desc_count", ctypes.c_uint32),
]


class _CmdDesc(ctypes.Structure):
    pass


_CmdDesc._fields_ = [
    ("name", ctypes.c_char_p),
    ("prj_id", ctypes.c_uint8),
    ("cls_id", ctypes.c_uint8),
    ("cmd_id", ctypes.c_uint16),
    ("list_type", ctypes.c_int),
    ("buffer_type", ctypes.c_int),
    ("timeout_policy", ctypes.c_int),
    ("arg_desc_table", ctypes.POINTER(_ArgDesc)),
    ("arg_desc_count", ctypes.c_uint32),
]


class _Binary(ctypes.Structure):
    _fields_ = [("cdata", ctypes.c_void_p), ("len", ctypes.c_uint32)]


class _ValueData(ctypes.Union):
    _fields_ = [
        ("i8", ctypes.c_int8),
        ("u8", ctypes.c_uint8),
        ("i16", ctypes.c_int16),
        ("u16", ctypes.c_uint16),
        ("i32", ctypes.c_int32),
        ("u32", ctypes.c_uint32),
        ("i64", ctypes.c_int64),
        ("u64", ctypes.c_uint64),
        ("f32", ctypes.c_float),
        ("f64", ctypes.c_double),
        ("str", ctypes.c_char_p),
        ("cstr", ctypes.c_char_p),
        ("binary", _Binary),
    ]


class _Value(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("data", _ValueData)]


class _Cmd(ctypes.Structure):
    _fields_ = [
        ("prj_id", ctypes.c_uint8),
        ("cls_id", ctypes.c_uint8),
        ("cmd_id", ctypes.c_uint16),
        ("id", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("userdata", ctypes.c_void_p),
        ("buffer_type", ctypes.c_int),
    ]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("backend_type", ctypes.c_int),
        ("proto_v", ctypes.c_uint32),
        ("api", ctypes.c_int),
        ("state", ctypes.c_int),
        ("name", ctypes.c_char_p),
        ("type", ctypes.c_int),
        ("addr", ctypes.c_char_p),
        ("port", ctypes.c_uint16),
        ("id", ctypes.c_char_p),
        ("json", ctypes.c_char_p),
    ]


class _BackendNetCfg(ctypes.Structure):
    _fields_ = [
        ("iface", ctypes.c_char_p),
        ("qos_mode_supported", ctypes.c_int),
        ("stream_supported", ctypes.c_int),
        ("proto_v_min", ctypes.c_uint32),
        ("proto_v_max", ctypes.c_uint32),
    ]


class _DiscoveryCfg(ctypes.Structure):
    _fields_ = [("types", ctypes.POINTER(ctypes.c_int)), ("count", ctypes.c_uint32)]


class _DeviceConnCfg(ctypes.Structure):
    _fields_ = [
        ("ctrl_name", ctypes.c_char_p),
        ("ctrl_type", ctypes.c_char_p),
        ("device_id", ctypes.c_char_p),
        ("json", ctypes.c_char_p),
    ]


# Callback ABI declarations follow the public arsdk-ng headers.
_CTRL_DEVICE_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
_DEVICE_INFO_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(_DeviceInfo), ctypes.c_void_p)
_DEVICE_CANCELED_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(_DeviceInfo), ctypes.c_int, ctypes.c_void_p)
_DEVICE_LINK_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(_DeviceInfo), ctypes.c_int, ctypes.c_void_p)
_RECV_CMD_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(_Cmd), ctypes.c_void_p)
_DISPOSE_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)
_IDLE_CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_CMD_SEND_STATUS_CB = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(_Cmd),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint16,
    ctypes.c_int,
    ctypes.c_void_p,
)


class _CtrlDeviceCbs(ctypes.Structure):
    _fields_ = [
        ("userdata", ctypes.c_void_p),
        ("added", _CTRL_DEVICE_CB),
        ("removed", _CTRL_DEVICE_CB),
    ]


class _DeviceConnCbs(ctypes.Structure):
    _fields_ = [
        ("userdata", ctypes.c_void_p),
        ("connecting", _DEVICE_INFO_CB),
        ("connected", _DEVICE_INFO_CB),
        ("disconnected", _DEVICE_INFO_CB),
        ("canceled", _DEVICE_CANCELED_CB),
        ("link_status", _DEVICE_LINK_CB),
    ]


class _CmdItfCbs(ctypes.Structure):
    _fields_ = [
        ("userdata", ctypes.c_void_p),
        ("dispose", _DISPOSE_CB),
        ("recv_cmd", _RECV_CMD_CB),
        ("cmd_log", ctypes.c_void_p),
        ("transport_log", ctypes.c_void_p),
        ("cmd_send_status", ctypes.c_void_p),
        ("pack_send_status", ctypes.c_void_p),
        ("pack_recv_status", ctypes.c_void_p),
        ("link_quality", ctypes.c_void_p),
    ]


_ARG_I8, _ARG_U8, _ARG_I16, _ARG_U16 = range(4)
_ARG_I32, _ARG_U32, _ARG_I64, _ARG_U64 = range(4, 8)
_ARG_FLOAT, _ARG_DOUBLE, _ARG_STRING, _ARG_ENUM, _ARG_BINARY = range(8, 13)
_LIST_NONE, _LIST_ITEM, _MAP_ITEM = range(3)
_BUFFER_NON_ACK, _BUFFER_ACK, _BUFFER_HIGH, _BUFFER_LOW = range(1, 5)


def _text(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def _iter_ptr(table: Any):
    if table:
        for item in table:
            if not item:
                break
            yield item


def _native_library(name: str, lib_dir: Path | None) -> ctypes.CDLL:
    candidates = []
    if lib_dir:
        candidates.extend((lib_dir / f"lib{name}.so", lib_dir / f"lib{name}.so.0"))
    found = ctypes.util.find_library(name)
    if found:
        candidates.append(Path(found))
    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) or "no matching shared library found"
    raise ArsdkNgError(f"cannot load lib{name}: {detail}")


class _Runtime:
    def __init__(self, lib_dir: Path | None = None) -> None:
        if lib_dir is None:
            configured = os.getenv("ARSDK_NG_LIB_DIR")
            built = Path(__file__).resolve().parents[2] / "執行環境" / "native" / "arsdk-ng" / "lib"
            lib_dir = Path(configured) if configured else built if built.is_dir() else None
        if lib_dir is not None:
            # Preload the pinned build's transitive dependencies; dlopen does not
            # search sibling files without an inherited LD_LIBRARY_PATH.
            self._dependencies = tuple(
                _native_library(name, lib_dir) for name in ("ulog", "futils", "pomp", "mux")
            )
            self.pomp = self._dependencies[2]
        else:
            self._dependencies = ()
            self.pomp = _native_library("pomp", None)
        self.arsdk = _native_library("arsdk", lib_dir)
        self.arsdkctrl = _native_library("arsdkctrl", lib_dir)
        self._configure()
        self.catalog = _Catalog(self)

    def _configure(self) -> None:
        self.arsdk.arsdk_get_cmd_table.restype = ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(_CmdDesc))))
        self.arsdk.arsdk_cmd_enc_argv.argtypes = [
            ctypes.POINTER(_Cmd), ctypes.POINTER(_CmdDesc), ctypes.c_size_t, ctypes.POINTER(_Value)
        ]
        self.arsdk.arsdk_cmd_enc_argv.restype = ctypes.c_int
        self.arsdk.arsdk_cmd_get_values.argtypes = [
            ctypes.POINTER(_Cmd), ctypes.POINTER(_Value), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
        ]
        self.arsdk.arsdk_cmd_get_values.restype = ctypes.c_int
        self.arsdk.arsdk_cmd_find_desc.argtypes = [ctypes.POINTER(_Cmd)]
        self.arsdk.arsdk_cmd_find_desc.restype = ctypes.POINTER(_CmdDesc)
        self.arsdk.arsdk_cmd_itf_send.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Cmd), _CMD_SEND_STATUS_CB, ctypes.c_void_p
        ]
        self.arsdk.arsdk_cmd_itf_send.restype = ctypes.c_int
        self.pomp.pomp_buffer_unref.argtypes = [ctypes.c_void_p]
        self.pomp.pomp_buffer_unref.restype = None
        self.pomp.pomp_loop_new.restype = ctypes.c_void_p
        self.pomp.pomp_loop_destroy.argtypes = [ctypes.c_void_p]
        self.pomp.pomp_loop_wait_and_process.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.pomp.pomp_loop_wait_and_process.restype = ctypes.c_int
        self.pomp.pomp_loop_wakeup.argtypes = [ctypes.c_void_p]
        self.pomp.pomp_loop_wakeup.restype = ctypes.c_int
        self.pomp.pomp_loop_idle_add.argtypes = [ctypes.c_void_p, _IDLE_CB, ctypes.c_void_p]
        self.pomp.pomp_loop_idle_add.restype = ctypes.c_int
        self.arsdkctrl.arsdk_ctrl_new.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.arsdkctrl.arsdk_ctrl_new.restype = ctypes.c_int
        self.arsdkctrl.arsdk_ctrl_set_device_cbs.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CtrlDeviceCbs)]
        self.arsdkctrl.arsdk_ctrl_set_device_cbs.restype = ctypes.c_int
        self.arsdkctrl.arsdk_ctrl_destroy.argtypes = [ctypes.c_void_p]
        self.arsdkctrl.arsdk_ctrl_destroy.restype = ctypes.c_int
        self.arsdkctrl.arsdkctrl_backend_net_new.argtypes = [ctypes.c_void_p, ctypes.POINTER(_BackendNetCfg), ctypes.POINTER(ctypes.c_void_p)]
        self.arsdkctrl.arsdkctrl_backend_net_new.restype = ctypes.c_int
        self.arsdkctrl.arsdkctrl_backend_net_destroy.argtypes = [ctypes.c_void_p]
        self.arsdkctrl.arsdkctrl_backend_net_destroy.restype = ctypes.c_int
        self.arsdkctrl.arsdk_discovery_net_new_with_port.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_DiscoveryCfg), ctypes.c_char_p,
            ctypes.c_uint16, ctypes.POINTER(ctypes.c_void_p)
        ]
        self.arsdkctrl.arsdk_discovery_net_new_with_port.restype = ctypes.c_int
        self.arsdkctrl.arsdk_discovery_net_start.argtypes = [ctypes.c_void_p]
        self.arsdkctrl.arsdk_discovery_net_start.restype = ctypes.c_int
        self.arsdkctrl.arsdk_discovery_net_stop.argtypes = [ctypes.c_void_p]
        self.arsdkctrl.arsdk_discovery_net_stop.restype = ctypes.c_int
        self.arsdkctrl.arsdk_discovery_net_destroy.argtypes = [ctypes.c_void_p]
        self.arsdkctrl.arsdk_discovery_net_destroy.restype = ctypes.c_int
        self.arsdkctrl.arsdk_device_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(_DeviceInfo))]
        self.arsdkctrl.arsdk_device_get_info.restype = ctypes.c_int
        self.arsdkctrl.arsdk_device_connect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DeviceConnCfg), ctypes.POINTER(_DeviceConnCbs), ctypes.c_void_p]
        self.arsdkctrl.arsdk_device_connect.restype = ctypes.c_int
        self.arsdkctrl.arsdk_device_disconnect.argtypes = [ctypes.c_void_p]
        self.arsdkctrl.arsdk_device_disconnect.restype = ctypes.c_int
        self.arsdkctrl.arsdk_device_create_cmd_itf.argtypes = [ctypes.c_void_p, ctypes.POINTER(_CmdItfCbs), ctypes.POINTER(ctypes.c_void_p)]
        self.arsdkctrl.arsdk_device_create_cmd_itf.restype = ctypes.c_int

    def _encode(self, spec: "ArsdkMessageSpec", values: dict[str, Any]) -> _Cmd:
        desc = spec._desc.contents
        ordered = spec._ordered_values(values)
        argv = (_Value * len(ordered))()
        keepalive = []
        for idx, value in enumerate(ordered):
            arg = desc.arg_desc_table[idx]
            kind = arg.type
            argv[idx].type = kind
            if kind == _ARG_STRING:
                encoded = str(value).encode("utf-8")
                keepalive.append(encoded)
                argv[idx].data.cstr = encoded
            elif kind == _ARG_BINARY:
                blob = bytes(value)
                buf = ctypes.create_string_buffer(blob)
                keepalive.append(buf)
                argv[idx].data.binary = _Binary(ctypes.cast(buf, ctypes.c_void_p), len(blob))
            elif kind in (_ARG_FLOAT, _ARG_DOUBLE):
                setattr(argv[idx].data, "f32" if kind == _ARG_FLOAT else "f64", float(value))
            elif kind in (_ARG_I8, _ARG_I16, _ARG_I32, _ARG_ENUM):
                setattr(argv[idx].data, {0: "i8", 2: "i16", 4: "i32", 11: "i32"}[kind], int(value))
            else:
                setattr(argv[idx].data, {1: "u8", 3: "u16", 5: "u32", 6: "i64", 7: "u64"}[kind], int(value))
        cmd = _Cmd()
        result = self.arsdk.arsdk_cmd_enc_argv(ctypes.byref(cmd), spec._desc, len(ordered), argv)
        if result < 0:
            raise ArsdkNgError(f"arsdk_cmd_enc_argv({spec.fullname}): {result}")
        return cmd

    def _clear_cmd(self, cmd: _Cmd) -> None:
        if cmd.buf:
            self.pomp.pomp_buffer_unref(cmd.buf)
        ctypes.memset(ctypes.byref(cmd), 0, ctypes.sizeof(cmd))
        cmd.buffer_type = 0


    def _decode(self, cmd: ctypes.POINTER(_Cmd)) -> tuple["ArsdkMessageSpec", dict[str, Any]]:
        desc = self.arsdk.arsdk_cmd_find_desc(cmd)
        if not desc:
            raise ArsdkNgError(f"unknown ARSDK message id 0x{cmd.contents.id:08x}")
        spec = self.catalog.by_id.get(int(cmd.contents.id))
        if spec is None:
            raise ArsdkNgError(f"unindexed ARSDK message id 0x{cmd.contents.id:08x}")
        count = ctypes.c_size_t()
        values = (_Value * desc.contents.arg_desc_count)()
        result = self.arsdk.arsdk_cmd_get_values(cmd, values, len(values), ctypes.byref(count))
        if result < 0:
            raise ArsdkNgError(f"arsdk_cmd_get_values({spec.fullname}): {result}")
        args = {}
        for idx in range(count.value):
            arg = desc.contents.arg_desc_table[idx]
            name = _text(arg.name)
            value = values[idx]
            kind = arg.type
            if kind == _ARG_ENUM:
                raw = int(value.data.i32)
                args[_field_name(name)] = spec._enum_for_index(idx)(raw)
            elif kind == _ARG_STRING:
                args[_field_name(name)] = _text(value.data.str)
            elif kind == _ARG_BINARY:
                length = int(value.data.binary.len)
                pointer = value.data.binary.cdata
                args[_field_name(name)] = ctypes.string_at(pointer, length) if pointer and length else b""
            elif kind == _ARG_FLOAT:
                args[_field_name(name)] = float(value.data.f32)
            elif kind == _ARG_DOUBLE:
                args[_field_name(name)] = float(value.data.f64)
            elif kind == _ARG_I8:
                args[_field_name(name)] = int(value.data.i8)
            elif kind == _ARG_U8:
                args[_field_name(name)] = int(value.data.u8)
            elif kind == _ARG_I16:
                args[_field_name(name)] = int(value.data.i16)
            elif kind == _ARG_U16:
                args[_field_name(name)] = int(value.data.u16)
            elif kind == _ARG_I32:
                args[_field_name(name)] = int(value.data.i32)
            elif kind == _ARG_U32:
                args[_field_name(name)] = int(value.data.u32)
            elif kind == _ARG_I64:
                args[_field_name(name)] = int(value.data.i64)
            elif kind == _ARG_U64:
                args[_field_name(name)] = int(value.data.u64)
        return spec, args


class ArsdkMessageSpec:
    def __init__(self, runtime: _Runtime, desc: ctypes.POINTER(_CmdDesc)) -> None:
        self._runtime = runtime
        self._desc = desc
        raw = desc.contents
        self.fullname = _text(raw.name)
        self.id = (int(raw.prj_id) << 24) | (int(raw.cls_id) << 16) | int(raw.cmd_id)
        self.list_type = int(raw.list_type)
        self.buffer_type = int(raw.buffer_type)
        self._args = tuple(raw.arg_desc_table[idx] for idx in range(raw.arg_desc_count))
        self._names = tuple(_field_name(_text(item.name)) for item in self._args)
        self._enum_cache: dict[int, type[IntEnum]] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> "ArsdkMessageInstance":
        metadata = {key: kwargs.pop(key) for key in tuple(kwargs) if key.startswith("_")}
        if len(args) > len(self._names):
            raise TypeError(f"{self.fullname} accepts {len(self._names)} positional argument(s)")
        values = dict(zip(self._names, args))
        for key, value in kwargs.items():
            normalized = _field_name(key)
            if normalized not in self._names:
                raise TypeError(f"{self.fullname} has no argument {key!r}")
            if normalized in values:
                raise TypeError(f"{self.fullname} got multiple values for {normalized!r}")
            values[normalized] = value
        values = {key: self._normalize_value(self._names.index(key), value) for key, value in values.items()}
        return ArsdkMessageInstance(self, values, metadata)

    def _normalize_value(self, idx: int, value: Any) -> Any:
        kind = self._args[idx].type
        if kind == _ARG_ENUM:
            if isinstance(value, str):
                enum_type = self._enum_for_index(idx)
                normalized = value.replace("-", "_").replace(" ", "_").casefold()
                for member in enum_type:
                    if member.name.casefold() == normalized:
                        return member
                raise ValueError(f"{self.fullname}.{self._names[idx]} has no enum value {value!r}")
            if not isinstance(value, int):
                raise TypeError(f"{self.fullname}.{self._names[idx]} requires an enum name or integer")
            if int(value) not in {member.value for member in self._enum_for_index(idx)}:
                raise ValueError(f"{self.fullname}.{self._names[idx]} has no enum value {value!r}")
            return value
        if kind in (_ARG_I8, _ARG_U8, _ARG_I16, _ARG_U16, _ARG_I32, _ARG_U32, _ARG_I64, _ARG_U64):
            if not isinstance(value, int):
                raise TypeError(f"{self.fullname}.{self._names[idx]} requires an integer")
            bits = { _ARG_I8: 8, _ARG_U8: 8, _ARG_I16: 16, _ARG_U16: 16,
                     _ARG_I32: 32, _ARG_U32: 32, _ARG_I64: 64, _ARG_U64: 64}[kind]
            signed = kind in (_ARG_I8, _ARG_I16, _ARG_I32, _ARG_I64)
            lower = -(1 << (bits - 1)) if signed else 0
            upper = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
            if not lower <= value <= upper:
                raise ValueError(f"{self.fullname}.{self._names[idx]} outside [{lower}, {upper}]")
            return value
        if kind in (_ARG_FLOAT, _ARG_DOUBLE):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{self.fullname}.{self._names[idx]} requires a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{self.fullname}.{self._names[idx]} must be finite")
            return float(value)
        if kind == _ARG_STRING and not isinstance(value, str):
            raise TypeError(f"{self.fullname}.{self._names[idx]} requires text")
        if kind == _ARG_BINARY:
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise TypeError(f"{self.fullname}.{self._names[idx]} requires bytes")
            return bytes(value)
        return value
    def _enum_for_index(self, idx: int) -> type[IntEnum]:
        enum_type = self._enum_cache.get(idx)
        if enum_type is None:
            arg = self._args[idx]
            members = {_text(arg.enum_desc_table[n].name): int(arg.enum_desc_table[n].value) for n in range(arg.enum_desc_count)}
            enum_type = IntEnum(f"{self.fullname.rsplit('.', 1)[-1]}_{self._names[idx]}", members)
            self._enum_cache[idx] = enum_type
        return enum_type

    def _ordered_values(self, values: dict[str, Any]) -> tuple[Any, ...]:
        missing = [name for name in self._names if name not in values]
        extra = [name for name in values if name not in self._names]
        if missing or extra:
            raise TypeError(f"{self.fullname} requires {self._names}; missing={missing}, extra={extra}")
        return tuple(values[name] for name in self._names)

    def __repr__(self) -> str:
        return f"ArsdkMessageSpec({self.fullname})"


def _field_name(value: str) -> str:
    return value[:1].lower() + value[1:] if value else value


class _MessageNamespace:
    def __init__(self, catalog: "_Catalog", prefix: str = "") -> None:
        self._catalog = catalog
        self._prefix = prefix

    def __getattr__(self, name: str) -> Any:
        fullname = f"{self._prefix}.{name}" if self._prefix else name
        exact = self._catalog.by_name.get(fullname)
        if exact is not None:
            return exact
        candidates = self._catalog.prefixes.get(fullname.casefold())
        if candidates is None:
            raise AttributeError(f"ARSDK message path not found: {fullname}")
        if isinstance(candidates, ArsdkMessageSpec):
            return candidates
        return _MessageNamespace(self._catalog, fullname)


class _Catalog:
    def __init__(self, runtime: _Runtime) -> None:
        self.by_name: dict[str, ArsdkMessageSpec] = {}
        self.by_id: dict[int, ArsdkMessageSpec] = {}
        self.prefixes: dict[str, ArsdkMessageSpec | bool] = {}
        table = runtime.arsdk.arsdk_get_cmd_table()
        for project in _iter_ptr(table):
            for feature in _iter_ptr(project):
                for desc in _iter_ptr(feature):
                    spec = ArsdkMessageSpec(runtime, desc)
                    self.by_name[spec.fullname] = spec
                    self.by_id[spec.id] = spec
        for fullname, spec in self.by_name.items():
            parts = fullname.split(".")
            for index in range(1, len(parts)):
                prefix = ".".join(parts[:index]).casefold()
                self.prefixes[prefix] = self.prefixes.get(prefix, False)
            self.prefixes[fullname.casefold()] = spec
        if not self.by_name:
            raise ArsdkNgError("arsdk_get_cmd_table returned no message descriptors")


class ArsdkMessageRoot:
    """Lazy Python namespace over the native ARSDK command descriptor table."""

    def __init__(self, lib_dir: str | os.PathLike[str] | None = None) -> None:
        self._runtime = _Runtime(Path(lib_dir) if lib_dir else None)
        self._namespace = _MessageNamespace(self._runtime.catalog)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._namespace, name)

class _LazyMessageRoot:
    def __init__(self) -> None:
        self._root: ArsdkMessageRoot | None = None

    def __getattr__(self, name: str) -> Any:
        if self._root is None:
            self._root = ArsdkMessageRoot()
        return getattr(self._root, name)


ArsdkMessage = _LazyMessageRoot()



@dataclass(frozen=True)
class _Event:
    args: dict[str, Any]
    monotonic_ns: int
    serial: int


    @property
    def uuid(self) -> str:
        return str(self.serial)


@dataclass(frozen=True)
class _WaitResult:
    _success: bool
    reason: str = ""

    def success(self) -> bool:
        return self._success


class ArsdkExpectation:
    def __init__(
        self,
        client: "ArsdkNgClient",
        command: "ArsdkMessageInstance",
        event: "ArsdkMessageInstance | None" = None,
    ) -> None:
        self._client = client
        self._command = command
        self._event = event
        self._baseline = client._event_serial
        self._future = client._send(command)

    def wait(self, _timeout: float | None = None, **_: Any) -> _WaitResult:
        timeout = 10.0 if _timeout is None else max(0.0, float(_timeout))
        deadline = time.monotonic() + timeout
        remaining = max(0.0, deadline - time.monotonic())
        if not self._future.event.wait(remaining):
            return _WaitResult(False, "command send timed out")
        if not self._future.success:
            return _WaitResult(False, self._future.error or "command was not acknowledged")
        if self._event is None:
            return _WaitResult(True)
        if not self._client._wait_for_event(self._event, self._baseline, deadline):
            return _WaitResult(False, f"event timeout: {self._event.spec.fullname}")
        return _WaitResult(True)

    def __rshift__(self, event: "ArsdkMessageInstance") -> "ArsdkMessageRequest":
        return ArsdkMessageRequest(self._command, event)


@dataclass(frozen=True)
class ArsdkMessageInstance:
    spec: ArsdkMessageSpec
    values: dict[str, Any]
    metadata: dict[str, Any]

    def __rshift__(self, event: "ArsdkMessageInstance") -> "ArsdkMessageRequest":
        return ArsdkMessageRequest(self, event)


@dataclass(frozen=True)
class ArsdkMessageRequest:
    command: ArsdkMessageInstance
    event: ArsdkMessageInstance


@dataclass
class _SendFuture:
    event: threading.Event
    success: bool = False
    error: str = ""


class ArsdkNgClient:
    """Single-device ARSDK-NG client using Parrot's native C interfaces."""

    def __init__(
        self,
        address: str,
        *,
        device_type: int,
        port: int = ARSDK_DEFAULT_PORT,
        controller_name: str = "LocalizationOperator",
        controller_type: str = "Linux",
        lib_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.address = str(address)
        self.port = int(port)
        self.device_type = int(device_type)
        self.controller_name = controller_name
        self.controller_type = controller_type
        self._runtime = _Runtime(Path(lib_dir) if lib_dir else None)
        self.messages = _MessageNamespace(self._runtime.catalog)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._states: dict[int, dict[str, Any]] = {}
        self._map_states: dict[int, dict[Any, dict[str, Any]]] = {}
        self._events: dict[int, _Event] = {}
        self._event_history: dict[int, deque[_Event]] = {}
        self._event_serial = 0
        self._link_ok = False
        self._connected = threading.Event()
        self._disconnected = threading.Event()
        self._connection_error = ""
        self._closed = False
        self._device = ctypes.c_void_p()
        self._cmd_itf = ctypes.c_void_p()
        self._next_token = 1
        self._idle_calls: dict[int, tuple[Callable[[], Any], threading.Event, list[Any]]] = {}
        self._send_futures: dict[int, _SendFuture] = {}
        self._loop = self._runtime.pomp.pomp_loop_new()
        if not self._loop:
            raise ArsdkNgError("pomp_loop_new failed")
        self._loop_stop = threading.Event()
        self._idle_cb = _IDLE_CB(self._on_idle)
        self._ctrl_added_cb = _CTRL_DEVICE_CB(self._on_device_added)
        self._ctrl_removed_cb = _CTRL_DEVICE_CB(self._on_device_removed)
        self._connecting_cb = _DEVICE_INFO_CB(self._on_connecting)
        self._connected_cb = _DEVICE_INFO_CB(self._on_connected)
        self._disconnected_cb = _DEVICE_INFO_CB(self._on_disconnected)
        self._canceled_cb = _DEVICE_CANCELED_CB(self._on_canceled)
        self._link_status_cb = _DEVICE_LINK_CB(self._on_link_status)
        self._recv_cmd_cb = _RECV_CMD_CB(self._on_recv_cmd)
        self._dispose_cb = _DISPOSE_CB(self._on_cmd_itf_dispose)
        self._send_status_cb = _CMD_SEND_STATUS_CB(self._on_send_status)
        self._ctrl_cbs = _CtrlDeviceCbs(None, self._ctrl_added_cb, self._ctrl_removed_cb)
        self._device_cbs = _DeviceConnCbs(
            None, self._connecting_cb, self._connected_cb, self._disconnected_cb,
            self._canceled_cb, self._link_status_cb,
        )
        self._cmd_cbs = _CmdItfCbs(
            None, self._dispose_cb, self._recv_cmd_cb, None, None, None, None, None, None
        )
        self._ctrl = ctypes.c_void_p()
        self._backend = ctypes.c_void_p()
        self._discovery = ctypes.c_void_p()
        self._closing = False
        self._thread = threading.Thread(target=self._run_loop, name="arsdk-ng-loop", daemon=True)
        self._thread.start()
        try:
            self._create_connection()
        except Exception:
            self.close()
            raise

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and self._link_ok and not self._closed

    @property
    def _event_serial(self) -> int:
        with self._lock:
            return self.__event_serial

    @_event_serial.setter
    def _event_serial(self, value: int) -> None:
        self.__event_serial = value

    def _check(self, result: int, operation: str) -> None:
        if result < 0:
            raise ArsdkNgError(f"{operation}: {result}")

    def _run_loop(self) -> None:
        while not self._loop_stop.is_set():
            result = self._runtime.pomp.pomp_loop_wait_and_process(self._loop, 250)
            if (
                result < 0
                and result != -errno.ETIMEDOUT
                and not self._loop_stop.is_set()
            ):
                with self._condition:
                    self._connection_error = f"pomp_loop_wait_and_process: {result}"
                    self._condition.notify_all()
                break

    def _on_idle(self, userdata: int) -> None:
        token = int(ctypes.cast(userdata, ctypes.c_void_p).value or 0)
        call = self._idle_calls.pop(token, None)
        if call is None:
            return
        fn, done, output = call
        try:
            output.append(fn())
        except BaseException as exc:
            output.append(exc)
        finally:
            done.set()

    def _on_loop(self, fn: Callable[[], Any], timeout: float = 5.0) -> Any:
        if self._closed or not self._loop:
            raise ArsdkNgError("ARSDK client is closed")
        token = self._next_token
        self._next_token += 1
        done = threading.Event()
        output: list[Any] = []
        self._idle_calls[token] = (fn, done, output)
        result = self._runtime.pomp.pomp_loop_idle_add(
            self._loop, self._idle_cb, ctypes.c_void_p(token)
        )
        if result < 0:
            self._idle_calls.pop(token, None)
            raise ArsdkNgError(f"pomp_loop_idle_add: {result}")
        self._runtime.pomp.pomp_loop_wakeup(self._loop)
        if not done.wait(timeout):
            self._idle_calls.pop(token, None)
            raise TimeoutError("ARSDK event-loop operation timed out")
        result = output[0]
        if isinstance(result, BaseException):
            raise result
        return result

    def _create_connection(self) -> None:
        self._check(self._runtime.arsdkctrl.arsdk_ctrl_new(self._loop, ctypes.byref(self._ctrl)), "arsdk_ctrl_new")
        self._check(self._runtime.arsdkctrl.arsdk_ctrl_set_device_cbs(self._ctrl, ctypes.byref(self._ctrl_cbs)), "arsdk_ctrl_set_device_cbs")
        cfg = _BackendNetCfg(None, 0, 0, 1, 3)
        self._check(
            self._runtime.arsdkctrl.arsdkctrl_backend_net_new(self._ctrl, ctypes.byref(cfg), ctypes.byref(self._backend)),
            "arsdkctrl_backend_net_new",
        )
        device_types = (ctypes.c_int * 1)(self.device_type)
        discovery_cfg = _DiscoveryCfg(device_types, 1)
        self._check(
            self._runtime.arsdkctrl.arsdk_discovery_net_new_with_port(
                self._ctrl, self._backend, ctypes.byref(discovery_cfg),
                self.address.encode("ascii"), self.port, ctypes.byref(self._discovery),
            ),
            "arsdk_discovery_net_new_with_port",
        )
        self._check(self._runtime.arsdkctrl.arsdk_discovery_net_start(self._discovery), "arsdk_discovery_net_start")

    def _on_device_added(self, device: int, _: int) -> None:
        if self._device.value:
            return
        info = ctypes.POINTER(_DeviceInfo)()
        result = self._runtime.arsdkctrl.arsdk_device_get_info(device, ctypes.byref(info))
        if result < 0 or not info:
            self._connection_error = f"arsdk_device_get_info: {result}"
            with self._condition:
                self._condition.notify_all()
            return
        if info.contents.type != self.device_type:
            return
        address = _text(info.contents.addr)
        if address and address != self.address:
            return
        self._device = ctypes.c_void_p(device)
        cfg = _DeviceConnCfg(
            self.controller_name.encode("utf-8"), self.controller_type.encode("utf-8"), b"", b"{}"
        )
        result = self._runtime.arsdkctrl.arsdk_device_connect(
            self._device, ctypes.byref(cfg), ctypes.byref(self._device_cbs), self._loop
        )
        if result < 0:
            self._connection_error = f"arsdk_device_connect: {result}"
            with self._condition:
                self._condition.notify_all()

    def _on_device_removed(self, device: int, _: int) -> None:
        if self._device.value == device:
            self._on_disconnected(device, None, None)

    def _on_connecting(self, device: int, info: Any, _: int) -> None:
        pass

    def _on_connected(self, device: int, info: Any, _: int) -> None:
        self._device = ctypes.c_void_p(device)
        result = self._runtime.arsdkctrl.arsdk_device_create_cmd_itf(
            self._device, ctypes.byref(self._cmd_cbs), ctypes.byref(self._cmd_itf)
        )
        if result < 0:
            with self._condition:
                self._connection_error = f"arsdk_device_create_cmd_itf: {result}"
                self._condition.notify_all()
            return
        self._connected.set()
        self._link_ok = True
        with self._condition:
            self._condition.notify_all()

    def _on_disconnected(self, device: int, info: Any, _: int) -> None:
        self._connected.clear()
        self._link_ok = False
        self._disconnected.set()
        self._cmd_itf = ctypes.c_void_p()
        with self._condition:
            self._condition.notify_all()

    def _on_canceled(self, device: int, info: Any, reason: int, _: int) -> None:
        with self._condition:
            self._connection_error = f"arsdk_device_connect canceled: {reason}"
            self._condition.notify_all()

    def _on_link_status(self, device: int, info: Any, status: int, _: int) -> None:
        # arsdk-ng defines KO=0 and OK=1.
        with self._condition:
            self._link_ok = status == 1
            self._condition.notify_all()

    def _on_cmd_itf_dispose(self, itf: int, _: int) -> None:
        self._cmd_itf = ctypes.c_void_p()
        self._connected.clear()
        self._link_ok = False
        with self._condition:
            self._condition.notify_all()

    def wait_connected(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._connected.is_set():
                if self._connection_error:
                    raise ArsdkNgError(self._connection_error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"ARSDK device {self.address}:{self.port} did not connect")
                self._condition.wait(remaining)
        return True

    def _on_recv_cmd(self, itf: int, cmd: ctypes.POINTER(_Cmd), _: int) -> None:
        try:
            spec, args = self._runtime._decode(cmd)
        except Exception as exc:
            with self._condition:
                self._connection_error = f"ARSDK event decode failed: {exc}"
                self._condition.notify_all()
            return
        now_mono = time.monotonic_ns()
        with self._condition:
            self._event_serial += 1
            event = _Event(args, now_mono, self._event_serial)
            self._events[spec.id] = event
            if spec.fullname.casefold() in {
                "ardrone3.pilotingstate.attitudechanged", "ardrone3.pilotingstate.speedchanged",
            }:
                self._event_history.setdefault(spec.id, deque(maxlen=512)).append(event)
            if spec.list_type == _LIST_NONE:
                self._states[spec.id] = dict(args)
            else:
                map_key = next(iter(args.values()), None)
                bucket = self._map_states.setdefault(spec.id, {})
                if spec.list_type == _MAP_ITEM:
                    bucket[map_key] = dict(args)
                else:
                    bucket[map_key] = dict(args)
            self._condition.notify_all()

    def _on_send_status(
        self, itf: int, cmd: ctypes.POINTER(_Cmd), buffer_type: int,
        status: int, seq: int, done: int, userdata: int,
    ) -> None:
        if not done:
            return
        token = int(ctypes.cast(userdata, ctypes.c_void_p).value or 0)
        future = self._send_futures.pop(token, None)
        if future is None:
            return
        future.success = status in (1, 2)
        if not future.success:
            future.error = f"ARSDK command send status {status}"
        future.event.set()

    def _send(self, message: ArsdkMessageInstance) -> _SendFuture:
        if not self.connected or not self._cmd_itf.value:
            raise ArsdkNgError("ARSDK device is not connected")
        token = self._next_token
        self._next_token += 1
        future = _SendFuture(threading.Event())
        self._send_futures[token] = future

        def send_on_loop() -> int:
            cmd = self._runtime._encode(message.spec, message.values)
            result = self._runtime.arsdk.arsdk_cmd_itf_send(
                self._cmd_itf, ctypes.byref(cmd), self._send_status_cb, ctypes.c_void_p(token)
            )
            self._runtime._clear_cmd(cmd)
            return result

        try:
            result = self._on_loop(send_on_loop)
        except Exception:
            self._send_futures.pop(token, None)
            raise
        if result < 0:
            self._send_futures.pop(token, None)
            raise ArsdkNgError(f"arsdk_cmd_itf_send({message.spec.fullname}): {result}")
        return future

    def __call__(self, request: ArsdkMessageInstance | ArsdkMessageRequest) -> ArsdkExpectation:
        if isinstance(request, ArsdkMessageRequest):
            return ArsdkExpectation(self, request.command, request.event)
        if isinstance(request, ArsdkMessageInstance):
            return ArsdkExpectation(self, request)
        raise TypeError(f"unsupported ARSDK request: {type(request).__name__}")

    def _wait_for_event(self, expected: ArsdkMessageInstance, after_serial: int, deadline: float) -> bool:
        with self._condition:
            while True:
                event = self._events.get(expected.spec.id)
                if event and event.serial > after_serial and all(
                    _comparable(event.args.get(key)) == _comparable(value)
                    for key, value in expected.values.items()
                ):
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed:
                    return False
                self._condition.wait(remaining)

    def get_state(self, message: ArsdkMessageSpec | ArsdkMessageInstance) -> dict[Any, Any] | None:
        spec = message.spec if isinstance(message, ArsdkMessageInstance) else message
        if not isinstance(spec, ArsdkMessageSpec):
            raise TypeError("get_state expects an ARSDK message descriptor")
        with self._lock:
            if spec.list_type == _LIST_NONE:
                state = self._states.get(spec.id)
                return dict(state) if state is not None else None
            state = self._map_states.get(spec.id)
            return {key: dict(value) for key, value in state.items()} if state is not None else None

    def get_last_event(self, message: ArsdkMessageSpec | ArsdkMessageInstance) -> _Event | None:
        spec = message.spec if isinstance(message, ArsdkMessageInstance) else message
        with self._lock:
            return self._events.get(spec.id)

    def get_event_history(self, message: ArsdkMessageSpec | ArsdkMessageInstance) -> list[_Event]:
        """Bounded attitude/speed receipt history, without consuming other readers' samples."""
        spec = message.spec if isinstance(message, ArsdkMessageInstance) else message
        with self._lock:
            return list(self._event_history.get(spec.id, ()))

    def close(self) -> None:
        if self._closed or self._closing:
            return
        self._closing = True
        try:
            if self._loop and self._thread.is_alive():
                def disconnect_native() -> None:
                    if self._discovery.value:
                        self._runtime.arsdkctrl.arsdk_discovery_net_stop(self._discovery)
                    if self._device.value:
                        self._runtime.arsdkctrl.arsdk_device_disconnect(self._device)

                try:
                    self._on_loop(disconnect_native, timeout=2.0)
                    if self._device.value:
                        self._disconnected.wait(3.0)

                    def destroy_native() -> None:
                        for attr, destroy in (
                            ("_discovery", self._runtime.arsdkctrl.arsdk_discovery_net_destroy),
                            ("_backend", self._runtime.arsdkctrl.arsdkctrl_backend_net_destroy),
                            ("_ctrl", self._runtime.arsdkctrl.arsdk_ctrl_destroy),
                        ):
                            obj = getattr(self, attr)
                            if obj.value:
                                destroy(obj)
                                setattr(self, attr, ctypes.c_void_p())

                    self._on_loop(destroy_native, timeout=2.0)
                except Exception:
                    pass
        finally:
            self._closed = True
            self._loop_stop.set()
            if self._loop:
                self._runtime.pomp.pomp_loop_wakeup(self._loop)
            if self._thread.is_alive() and threading.current_thread() is not self._thread:
                self._thread.join(timeout=2.0)
            for future in self._send_futures.values():
                future.success = False
                future.error = "ARSDK client closed"
                future.event.set()
            self._send_futures.clear()
            if self._loop:
                self._runtime.pomp.pomp_loop_destroy(self._loop)
                self._loop = None

    def disconnect(self) -> None:
        self.close()

    def __enter__(self) -> "ArsdkNgClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _comparable(value: Any) -> Any:
    if isinstance(value, IntEnum):
        return value.name.casefold()
    if isinstance(value, str):
        return value.casefold()
    return value

def arsdk_message(fullname: str) -> ArsdkMessageSpec:
    """Resolve a dotted ARSDK descriptor path without vendor Python bindings."""
    message: Any = ArsdkMessage
    for part in str(fullname).split("."):
        message = getattr(message, part)
    if not isinstance(message, ArsdkMessageSpec):
        raise TypeError(f"not an ARSDK message descriptor: {fullname}")
    return message
