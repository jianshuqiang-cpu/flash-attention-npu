# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.


from catlass_cppgen.kernel.kernel_base import KernelBase


class VisitorKernelBase(KernelBase):
    """Visitor Kernel Base Class

    继承自 KernelBase，提供 EVG 模板生成功能。
    """

    _EVG_TEMPLATE: str = ""  # EVG 模板

    def gen_evg_template(self) -> str:
        """生成 EVG 相关代码.

        子类可以通过定义 _EVG_TEMPLATE 类变量来生成 EVG 相关代码，
        或者重写此方法。
        默认检查 _EVG_TEMPLATE，如果存在则使用模板渲染，否则返回空字符串。

        :return: EVG 相关代码.
        :rtype: str
        """
        if not self._EVG_TEMPLATE:
            return ""

        render_params = self.get_render_params()
        for key, value in render_params.items():
            if hasattr(value, "value"):
                value = value.value
            render_params[key] = str(value)
        return self._EVG_TEMPLATE.format(**render_params)
