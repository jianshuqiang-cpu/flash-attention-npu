# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Dict

from catlass_cppgen.common.data_type import DataType
from ..library import (
    BroadcastType,
    BroadcastTag,
    CastType,
    EpilogueOp,
    EpilogueScalarOp,
    LayoutType,
)
from .node_impl import (
    AccLoadImpl,
    AuxLoadImpl,
    AuxStoreImpl,
    ComputeImpl,
    CastImpl,
    NoOpImpl,
    RowBroadcastImpl,
    ScalarComputeImpl,
    TopoVisitorImpl,
)

if TYPE_CHECKING:
    from ..evg_extension import EpilogueVisitorGraph
    from ..library import EpilogueOp


@dataclass
class NodeMetadata:
    op: str = ""
    element: DataType = DataType.UNDEFINED
    layout: LayoutType = LayoutType.RowMajor
    broadcast: BroadcastType = BroadcastType.NoBroadcast
    shape: Tuple[int, ...] = ()
    stride: Tuple[int, ...] = ()
    is_output: bool = False


class NodeBase:
    def __init__(self, name: str, metadata: NodeMetadata):
        self.name = name  # unique name for the node
        self.metadata = metadata
        self.disabled = False
        self.impl = None  # the underlying impl of this node

    @property
    def type_name(self) -> str:
        return self.impl.type_name

    def signature_name(self) -> str:
        res = str(self)[:-1]
        extra_attrs = [" "]
        if self.metadata.element != DataType.UNDEFINED:
            extra_attrs.append(self.metadata.element.value)
        if self.metadata.shape:
            extra_attrs.append("x".join(map(str, self.metadata.shape)))
        extra_str = " | ".join(extra_attrs) + ">"
        res += extra_str
        return res

    def get_impl(self):
        raise NotImplementedError(
            f"Function `get_impl` is not overloaded in {self.__class__.__name__}"
        )

    def __repr__(self) -> str:
        return f"<Node {self.name} | {self.metadata.op}>"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: NodeBase) -> bool:
        return isinstance(other, NodeBase) and self.name == other.name

    def __lt__(self, other: NodeBase) -> bool:
        return self.name < other.name


class ComputeNode(NodeBase):
    def __init__(self, name: str, metadata: NodeMetadata, fn: EpilogueOp):
        super().__init__(name, metadata)
        self.fn = fn
        self._scalar_values: Dict[str, Tuple[str, DataType]] = {}

    def get_impl(self):
        if self.fn in EpilogueScalarOp:
            self.impl = ScalarComputeImpl(self, self._scalar_values)
        else:
            self.impl = ComputeImpl(self)


class CastNode(NodeBase):
    def __init__(
        self,
        name: str,
        metadata: NodeMetadata,
        from_element,
        to_element,
        round_type=CastType.NONE,
    ):
        super().__init__(name, metadata)
        self.from_element = from_element
        self.to_element = to_element
        self.round_type = round_type

    def get_impl(self):
        self.impl = CastImpl(self)


class ReduceNode(NodeBase):
    def __init__(self, name: str, metadata: NodeMetadata, reduce_fn):
        super().__init__(name, metadata)
        self.reduce_fn = reduce_fn


class ConstantNode(NodeBase):
    def __init__(self, name: str, metadata: NodeMetadata, value):
        super().__init__(name, metadata)
        self.value = value


class LoadNode(NodeBase):
    def __init__(self, name: str, metadata: NodeMetadata):
        super().__init__(name, metadata)

    # Possible impls:
    #   1. AccLoadImpl
    #   2. AuxLoadImpl
    #   3. RowBroadcastImpl
    def get_impl(self):
        if self.metadata.broadcast != BroadcastType.NoBroadcast:
            if self.metadata.op.lower() != "auxload":
                raise ValueError("For broadcast mode, the evg op must be 'auxload'")
            if self.metadata.broadcast == BroadcastType.RowBroadcast:
                self.impl = RowBroadcastImpl(self)
                return
            raise RuntimeError(
                f"Node `{self.name}` does not support {BroadcastTag[self.metadata.broadcast]}"
            )
        if self.metadata.op.lower() == "accload":
            self.impl = AccLoadImpl(self)
        elif self.metadata.op.lower() == "auxload":
            self.impl = AuxLoadImpl(self)


class StoreNode(NodeBase):
    def __init__(self, name: str, metadata: NodeMetadata):
        super().__init__(name, metadata)

    # Possible impls:
    #   1. AuxStoreImpl
    #   2. NoOpImpl
    def get_impl(self):
        if self.metadata.is_output:
            self.impl = AuxStoreImpl(self)
        else:
            self.impl = NoOpImpl(self)


class TopoVisitorNode(NodeBase):
    def __init__(
        self,
        name: str,
        metadata: NodeMetadata,
        subgraph: EpilogueVisitorGraph,
        output_node: NodeBase,
        lca_output_count: int,
    ):
        super().__init__(name, metadata)
        self.metadata.op = "topo_visitor"
        self.subgraph = subgraph
        self.output_node = output_node
        self.impl = TopoVisitorImpl(self)
        self.lca_output_count = lca_output_count
