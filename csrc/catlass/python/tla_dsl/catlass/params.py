from __future__ import annotations

import enum

from .types import TlaTensor
from dataclasses import dataclass


class QuantMode(enum.IntEnum):
    NO_QUANT = 0
    PER_TENSOR = 1  # need quant_scale
    PER_CHANNEL = 2  # need quant_tensor

    def __str__(self):
        return self.name  # "NO_QUANT"/..


class L0C2UBMode(enum.IntEnum):
    NO_SPLIT_VEC_0 = 0
    NO_SPLIT_VEC_1 = 1
    SPLIT_M = 2
    SPLIT_N = 3

    def __str__(self):
        return self.name  # "NO_SPLIT_VEC_0"/..


class ComputeOrder(enum.IntEnum):
    M_FIRST = 0
    N_FIRST = 1

    def __str__(self):
        return self.name  # "M_FIRST"/"N_FIRST"


class HF32Mode(enum.IntEnum):
    HF32_DISABLE = 0
    HF32_NEAREST_ZERO = 1
    HF32_NEAREST_EVEN = 2

    def __str__(self):
        return self.name  # "HF32_DISABLE"/"HF32_NEAREST_ZERO"/"HF32_NEAREST_EVEN"


class AtomicMode(enum.Enum):
    """Atomic operation mode"""

    NONE = "none"  # No atomic operation
    ADD = "add"  # Set atomic add

    def __str__(self) -> str:
        return self.value


@dataclass
class CopyParams:
    """Marker annotation for base copy params."""

    pass


class PostMode:
    POST_MODE_NORMAL = "norm"
    POST_MODE_UPDATE = "update"


class LoadDist:
    DIST_NORM = "norm"
    DIST_BRC_B32 = "brc_b32"
    DIST_DINTLV_B32 = "dintlv_b32"
    DIST_US_B8 = "us_b8"


class MaskLoadDist:
    """Mask load distribution modes (AscendC ``MaskDist`` / ``plds``)."""

    DIST_NORM = "norm"


class StoreDist:
    DIST_NORM = "norm"
    DIST_PACK_B32 = "pack_b32"
    DIST_PACK_B16 = "pack_b16"


class MaskStoreDist:
    """Mask store distribution modes (AscendC ``MaskDist`` / ``psts``)."""

    DIST_NORM = "norm"


class StoreParams:
    """Marker annotation for tensor tile store params."""

    pass


@dataclass
class NormalStoreParams(StoreParams):
    """Default aligned vector store."""

    store_dist: str = StoreDist.DIST_NORM
    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: int = 0


@dataclass
class UnalignStoreParams(StoreParams):
    """Unaligned UB vector store (lowers with ``unaligned_ub_access``)."""

    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: int = 0


@dataclass
class BlockStoreParams(StoreParams):
    """UB Block store with stride"""

    block_stride: int = 0
    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: int = 0


@dataclass
class MaskStoreParams(StoreParams):
    """Continuous-aligned MaskSSA store (1/2/4-byte UB / ``psts``).

    AscendC ``StoreAlign(__ubuf__ T*, MaskReg&)``: spill size is ``N/8`` from
    the ``MaskSSA`` value. ``N`` must equal ``256 / sizeof(T)`` for the
    destination UB element type (``T`` may be int or float of that width).
    Hardware path is still ``psts.b8`` (``DIST_NORM``).
    """

    store_dist: str = MaskStoreDist.DIST_NORM


class LoadParams:
    """Marker annotation for tensor tile load params."""

    pass


@dataclass
class NormalLoadParams(LoadParams):
    """Default aligned vector load."""

    load_dist: str = LoadDist.DIST_NORM
    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: int = 0


@dataclass
class UnalignLoadParams(LoadParams):
    """Unaligned UB vector load (lowers with ``unaligned_ub_access``)."""

    is_pre: bool = False
    post_mode: str = PostMode.POST_MODE_NORMAL
    post_update_stride: int = 0


@dataclass
class MaskLoadParams(LoadParams):
    """Continuous-aligned MaskSSA load (1/2/4-byte UB / ``plds``).

    AscendC ``LoadAlign(MaskReg&, __ubuf__ T*)``: predicate width
    ``!tla.mask<N>`` is ``N = 256 / sizeof(T)`` from the UB element type
    (same rule as ``create_mask``). ``T`` may be int or float; match the
    companion vector element width (e.g. ``f32`` UB with ``f32`` →
    ``mask<64>``). The source tile only supplies the UB address.
    """

    load_dist: str = MaskLoadDist.DIST_NORM


@dataclass
class CopyUbToGmParams(CopyParams):
    atomic_mode: AtomicMode = AtomicMode.NONE

    def _validate(self):
        VALID_ATOMIC_MODE = (AtomicMode.NONE, AtomicMode.ADD)
        cls_name = type(self).__name__
        if self.atomic_mode not in VALID_ATOMIC_MODE:
            raise ValueError(
                f"{cls_name}.atomic_mode must be one of {VALID_ATOMIC_MODE}"
            )


@dataclass
class CopyL0C2DstParams(CopyParams):
    unit_flag: int = 0
    relu_enable: bool = False
    quant_mode: QuantMode = QuantMode.NO_QUANT
    quant_scale: float | None = None
    quant_tensor: TlaTensor | None = None
    l0c2ub_mode: L0C2UBMode = L0C2UBMode.NO_SPLIT_VEC_0
    atomic_mode: AtomicMode = AtomicMode.NONE

    def _validate(self):
        from .core_api import _category
        from .execution_lowering import TlaLoweringError

        VALID_UNIT_FLAG = (0b00, 0b11)
        VALID_ATOMIC_MODE = (AtomicMode.NONE, AtomicMode.ADD)
        cls_name = type(self).__name__
        if not isinstance(self.unit_flag, int):
            raise TlaLoweringError(f"{cls_name}.unit_flag must be a compile-time int")
        if self.unit_flag not in VALID_UNIT_FLAG:
            raise ValueError(f"{cls_name}.unig_flag must be one of {VALID_UNIT_FLAG}")
        if not isinstance(self.relu_enable, bool):
            raise TlaLoweringError(
                f"{cls_name}.relu_enable must be a compile-time bool"
            )
        if not isinstance(self.quant_mode, QuantMode):
            raise TlaLoweringError(
                f"{cls_name}.quant_mode must be a {QuantMode}, got {type(self.quant_mode).__name__}"
            )
        if not isinstance(self.l0c2ub_mode, L0C2UBMode):
            raise TlaLoweringError(
                f"{cls_name}.l0c2ub_mode must be a {L0C2UBMode}, got {type(self.l0c2ub_mode).__name__}"
            )
        if self.quant_mode == QuantMode.PER_TENSOR:
            if not isinstance(self.quant_scale, float):
                raise TlaLoweringError(
                    f"{cls_name}.quant_scale must be a `float` temporarily, got {type(self.quant_scale).__name__}"
                )
        if self.quant_mode == QuantMode.PER_CHANNEL:
            if _category(self.quant_tensor) != "tensor":
                raise TlaLoweringError(
                    f"{cls_name}.quant_tensor must be a tensor, got {type(self.quant_tensor).__name__}"
                )
        if self.atomic_mode not in VALID_ATOMIC_MODE:
            raise ValueError(
                f"{cls_name}.atomic_mode must be one of {VALID_ATOMIC_MODE}"
            )


# ---------------------------------------------------------------------------
# CastParams: the four knobs of a tla.cast (VectorSSA.to). The enum member
# *values* are the MLIR-asm keywords; the integer codes match the
# I32EnumAttrCase values in Tla.td and the order in the tla.cast trait array.
# ---------------------------------------------------------------------------


class RegSlot(enum.Enum):
    """Packed-register position the narrow cast result lands in.

    For a 2x-width cast (e.g. f32->f16, i32<->i16) this maps to the AVE even/odd
    ``part`` (only ZERO=part_even / ONE=part_odd are valid there). For a 4x-width
    cast (i32<->i8) it maps to the AVE pack pattern ``pp`` and all four values
    select the pack quarter: ZERO=pp0, ONE=pp1, TWO=pp2, THREE=pp3.
    """

    ZERO = "zero"  # part_even / pp0
    ONE = "one"  # part_odd  / pp1
    TWO = "two"  # pp2 (4x casts only)
    THREE = "three"  # pp3 (4x casts only)

    def __str__(self) -> str:
        return self.value


class SatMode(enum.Enum):
    """Overflow behaviour of the cast (AVE ``sat`` BoolAttr)."""

    UNKNOWN = "unknown"
    SAT = "sat"
    NOSAT = "nosat"

    def __str__(self) -> str:
        return self.value


class RoundMode(enum.Enum):
    """Rounding applied on precision loss (HIVM ``round_mode``)."""

    CAST_ROUND = "cast_round"  # round to nearest, tie away from zero
    CAST_FLOOR = "cast_floor"  # round toward -inf
    CAST_CEIL = "cast_ceil"  # round toward +inf
    CAST_TRUNC = "cast_trunc"  # round toward zero

    def __str__(self) -> str:
        return self.value


_REG_SLOT_CODE = {RegSlot.ZERO: 0, RegSlot.ONE: 1, RegSlot.TWO: 2, RegSlot.THREE: 3}
_SAT_MODE_CODE = {SatMode.UNKNOWN: 0, SatMode.SAT: 1, SatMode.NOSAT: 2}
_ROUND_MODE_CODE = {
    RoundMode.CAST_ROUND: 0,
    RoundMode.CAST_FLOOR: 1,
    RoundMode.CAST_CEIL: 2,
    RoundMode.CAST_TRUNC: 3,
}


@dataclass(frozen=True)
class CastParams:
    """The three knobs of a ``tla.cast`` (``VectorSSA.to``).

    Args:
        reg_slot: destination packed-register position (:class:`RegSlot`).
        sat_mode: overflow behaviour (:class:`SatMode`).
        round_mode: rounding on precision loss (:class:`RoundMode`).
    """

    reg_slot: RegSlot = RegSlot.ZERO
    sat_mode: SatMode = SatMode.NOSAT
    round_mode: RoundMode = RoundMode.CAST_ROUND

    def codes(self) -> list[int]:
        """The three enum codes for the tla.cast DenseI32ArrayAttr, in order."""
        return [
            _REG_SLOT_CODE[self.reg_slot],
            _SAT_MODE_CODE[self.sat_mode],
            _ROUND_MODE_CODE[self.round_mode],
        ]


class MemType(enum.IntEnum):
    VEC_STORE = 0
    VEC_LOAD = 1
    SCALAR_STORE = 2
    SCALAR_LOAD = 3
    VEC_ALL = 4
    SCALAR_ALL = 5
