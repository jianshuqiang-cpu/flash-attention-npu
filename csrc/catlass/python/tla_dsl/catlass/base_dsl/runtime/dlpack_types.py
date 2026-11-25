"""DLPack ctypes layouts."""

from __future__ import annotations

import ctypes
import enum


class DLDeviceType(enum.IntEnum):
    """Device types used by TLA DLPack binding (Ascend / torch_npu).

    Wire values follow ``dlpack.h``. ``kDLNpuCandidate1`` is TLA's name for
    device_type ``15`` (PyTorch ``PrivateUse1`` / common ``torch_npu`` export).
    """

    kDLCPU = 1
    kDLExtDev = 12  # legacy torch_npu / extension-device export
    kDLNpuCandidate1 = 15  # torch_npu via PyTorch PrivateUse1 (DLPack device_type 15)


class DLDataTypeCode:
    """Data type codes from the DLPack specification."""

    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaqueHandle = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


# torch_npu may export device_type 15 (``kDLNpuCandidate1``) or legacy 12 (``kDLExtDev``).
ASCEND_DEVICE_TYPES = frozenset({DLDeviceType.kDLExtDev, DLDeviceType.kDLNpuCandidate1})


class DLDevice(ctypes.Structure):
    """Device information in a :class:`DLTensor`."""

    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int32),
    ]


class DLDataType(ctypes.Structure):
    """Element dtype in a :class:`DLTensor`."""

    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    """In-memory tensor view (DLPack).

    ``__dlpack__()`` capsules name ``dltensor`` and point at ``DLManagedTensor*``;
    ``dl_tensor`` is that struct's first member, so the pointer may be cast here.
    """

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]
