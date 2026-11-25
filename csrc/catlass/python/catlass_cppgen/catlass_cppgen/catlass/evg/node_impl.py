# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from re import sub

from ..library import LayoutTag, CastTypeTag, EpilogueOpTag


class ImplBase:
    def __init__(self, node):
        self.node = node
        self.name = node.name
        self.element = node.metadata.element
        self.shape = node.metadata.shape
        self.layout = node.metadata.layout
        self._type_decl = None

    @property
    def type_name(self):
        return sub(r"(_|-)+", " ", self.name).title().replace(" ", "")

    @property
    def args_decl(self):
        return "{}"

    def make_layout(self):
        shape_str = ", ".join(str(x) for x in self.shape)
        t_name = self.type_name
        layout_name = f"layout{t_name}"
        layout_str = f"""
using LayoutTag{t_name} = {LayoutTag[self.layout]};
LayoutTag{t_name} tag{t_name}{{{shape_str}}};
auto {layout_name} = tla::MakeLayoutFromTag(tag{t_name});
"""
        return layout_name, layout_str


class ComputeImplBase(ImplBase):
    """
    Base class for compute node implementations
    """

    def __init__(self, node):
        super().__init__(node)
        self.fn = self.node.fn
        self.compute_element = node.metadata.element


class CastImplBase(ImplBase):
    """
    Base class for cast node implementations
    """

    def __init__(self, node):
        super().__init__(node)
        self.to_element = node.to_element
        self.from_element = node.from_element
        self.round_type = node.round_type


class ReductionImplBase(ImplBase):
    """
    Base class for reduction node implementations
    """

    def __init__(self, node):
        super().__init__(node)
        self.reduce_fn = self.node.reduce_fn


class NoOpImpl(ImplBase):
    """
    The NoOpImpl does nothing but forward its inputs to users.
    """

    def __init__(self, node):
        super().__init__(node)


class AccLoadImpl(ImplBase):
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        #         self._type_decl = f"""
        # using {self.type_name} = Catlass::Epilogue::Fusion::VisitorAccLoad<{self.element.value}, EpilogueDispatchPolicy::USE_UB_WORKSPACE>;
        # """
        self._type_decl = f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorAccLoad<{self.element.value}>;
"""
        return self._type_decl


class AuxLoadImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node)
        self.layout_name = None

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self.layout_name, layout_str = self.make_layout()
        self._type_decl = layout_str
        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorAuxLoad<
    {self.element.value}, decltype({self.layout_name})
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        if not self.layout_name:
            self.layout_name, _ = self.make_layout()
        return f"{{{self.name}_ptr, {self.layout_name}}}"


class AuxStoreImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node)
        self.layout_name = None

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self.layout_name, layout_str = self.make_layout()
        self._type_decl = layout_str
        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorAuxStore<
    {self.element.value}, decltype({self.layout_name})
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        if not self.layout_name:
            self.layout_name, _ = self.make_layout()
        return f"{{deviceC, {self.layout_name}}}"


class CastImpl(CastImplBase):
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorCast<
    {self.to_element.value}, {self.from_element.value},
    {CastTypeTag[self.round_type]}
>;
"""
        return self._type_decl


class ComputeImpl(ComputeImplBase):
    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorCompute<
    {EpilogueOpTag[self.fn]}, {self.compute_element.value}
>;
"""
        return self._type_decl


class ScalarComputeImpl(ComputeImplBase):
    def __init__(self, node, values):
        super().__init__(node)
        self.scalar_values = values

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self._type_decl = ""
        for name, item in self.scalar_values.items():
            self._type_decl += f"""
{item[1].value} {name} = {item[0]};
"""

        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorCompute<
    {EpilogueOpTag[self.fn]}, {self.compute_element.value},"""
        self._type_decl += ", ".join(
            [item[1].value for _, item in self.scalar_values.items()]
        )
        self._type_decl += """
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        return f"{{{{{', '.join([name for name in self.scalar_values])}}}}}"


class RowBroadcastImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node)
        self.layout_name = None

    @property
    def type_decl(self):
        """
        Return the string defining the type
        """
        if self._type_decl is not None:
            return self._type_decl

        self.layout_name, layout_str = self.make_layout()
        self._type_decl = layout_str
        self._type_decl += f"""
using {self.type_name} = Catlass::Epilogue::Fusion::VisitorRowBroadcast<
    {self.element.value}, decltype({self.layout_name})
>;
"""
        return self._type_decl

    @property
    def args_decl(self):
        if not self.layout_name:
            self.layout_name, _ = self.make_layout()
        return f"{{{self.name}_ptr, {self.layout_name}}}"


class TopoVisitorImpl(ImplBase):
    def __init__(self, node):
        super().__init__(node.output_node)
        self.name = node.name
        self.element = node.output_node.metadata.element
