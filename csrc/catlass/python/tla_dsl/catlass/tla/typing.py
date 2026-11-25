"""Tensor typing layer.

Provides :class:`Tensor` as the shared abstract base and :class:`TypedTensor` for
``Tensor[dtype, shape, stride, ...]`` compile-time descriptors. Runtime host tensors
(:mod:`catlass.tla.runtime`) and JIT SSA tensors (:mod:`catlass.tla.tensor`) are
separate concrete implementations of the same ABC.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from ..address_space import AddressSpace


class Tensor(ABC):
    r"""Abstract base for TLA tensor representations.

    Concrete implementations:

    * :class:`~catlass.tla.runtime._Tensor` — runtime host tensor（``make_fake_tensor`` / ``from_dlpack``）
    * :class:`~catlass.tla.tensor._Tensor` — frontend ``!tla.tensor`` SSA value

    Subclasses should expose ``dtype``, ``shape``, and ``stride`` via read-only
    properties backed by private fields (host runtime tensors) or MLIR values (IR tensors).
    """

    def __class_getitem__(cls, args: tuple[Any, ...] | Any) -> TypedTensor:
        if not isinstance(args, tuple):
            args = (args,)
        if len(args) < 3:
            raise TypeError(
                "Tensor[dtype, shape, stride, ...] requires at least three arguments"
            )
        dtype, shape, stride = args[0], args[1], args[2]
        extras = args[3:]
        origin_shape = extras[0] if len(extras) >= 1 else None
        coord = extras[1] if len(extras) >= 2 else None
        addrspace = extras[2] if len(extras) >= 3 else AddressSpace.gm
        layout_tag = extras[3] if len(extras) >= 4 else None
        ptr_alignment = extras[4] if len(extras) >= 5 else None
        return TypedTensor(
            dtype,
            shape,
            stride,
            origin_shape=origin_shape,
            coord=coord,
            addrspace=addrspace,
            layout_tag=layout_tag,
            ptr_alignment=ptr_alignment,
        )

    @property
    def dtype(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not expose dtype")

    @property
    def memspace(self) -> Any:
        """Memory space (alias of ``addrspace`` on host tensors)."""
        addrspace = getattr(self, "addrspace", None)
        if addrspace is None:
            raise NotImplementedError(f"{type(self).__name__} does not expose memspace")
        return addrspace

    @property
    def shape(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not expose shape")

    @property
    def stride(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__} does not expose stride")

    def mark_layout_dynamic(self, leading_dim: int | None = None) -> Tensor:
        """Mark layout shape/stride metadata dynamic (runtime/host tensors only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support mark_layout_dynamic"
        )

    def mark_compact_shape_dynamic(
        self,
        mode: int,
        stride_order: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Mark one compact shape mode dynamic (runtime/host tensors only)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support mark_compact_shape_dynamic"
        )

    def __tla_type__(self) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not expose __tla_type__")


class TypedTensor:
    r"""Compile-time descriptor for a statically-typed TLA tensor.

    Construct via subscript syntax on :class:`Tensor`:

    .. code-block:: python

        ty = Tensor[tla.Float16, (128, 128), (128, 1)]

    ``isinstance`` checks compare the lowered ``!tla.tensor<...>`` MLIR type against
    live host or IR tensor values when both sides expose ``__tla_type__()``.
    """

    def __init__(
        self,
        dtype: Any,
        shape: Any,
        stride: Any,
        *,
        origin_shape: Any | None = None,
        coord: Any | None = None,
        addrspace: Any = AddressSpace.gm,
        layout_tag: Any | None = None,
        ptr_alignment: int | None = None,
    ) -> None:
        self._dtype = dtype
        self._shape = shape
        self._stride = stride
        self._origin_shape = origin_shape if origin_shape is not None else shape
        self._coord = coord if coord is not None else ()
        self._addrspace = addrspace
        self._layout_tag = layout_tag
        self._ptr_alignment = ptr_alignment

    @property
    def element_type(self) -> Any:
        return self._dtype

    @property
    def shape(self) -> Any:
        return self._shape

    @property
    def stride(self) -> Any:
        return self._stride

    @property
    def origin_shape(self) -> Any:
        return self._origin_shape

    @property
    def coord(self) -> Any:
        return self._coord

    @property
    def memspace(self) -> Any:
        return self._addrspace

    @property
    def layout_tag(self) -> Any:
        return self._layout_tag

    def isinstance(self, other: object) -> bool:
        if not isinstance(other, Tensor):
            return False
        try:
            other_type = str(other.__tla_type__())
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            return False
        try:
            return other_type == str(self.__tla_type__())
        except TypeError:
            return False

    def __tla_type__(self) -> str:
        raise TypeError(
            "TypedTensor cannot lower to !tla.tensor without tla.make_shape/make_stride; "
            "use from_dlpack(...), or compare against "
            "an existing tensor with TypedTensor.isinstance(...)"
        )

    def __str__(self) -> str:
        return f"TypedTensor<{self._dtype}, {self._shape}, {self._stride}>"


__all__ = ["Tensor", "TypedTensor"]
