# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from abc import abstractmethod
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from catlass_cppgen.catlass.gemm_coord import GemmShape
from catlass_cppgen.common.typing import GM_ADDR
from catlass_cppgen.common.utils import get_type_name

DispatchPolicy = TypeVar("DispatchPolicy")
BlockScheduler = TypeVar("BlockScheduler")


def _get_deprecated_arg(kwargs: Dict, key: Any, sub_key: Optional[str] = None) -> Any:
    if key in kwargs:
        warnings.warn(
            f"{key!r} is deprecated."
            + (f" Use {sub_key!r} instead." if sub_key else ""),
            DeprecationWarning,
            stacklevel=3,
        )
        v = kwargs.pop(key)
    else:
        v = None
    return v


class KernelBase:
    """Kernel Base Class"""

    _INCLUDES: List[str] = []  # 头文件引入
    _PARAMS_DEVICE: List[Tuple[str, Type]] = []  # 核函数参数
    _KERNEL_NAME: str = ""  # 核函数名
    _DISPATCH_POLICY: str = ""  # dispatch policy模板
    _KERNEL_TEMPLATE: str = ""  # 核函数模板
    _INPUT_TEMPLATE: str = ""  # 输入模板（用于定义 m, k, n 等变量）
    _LAYOUT_TEMPLATE: str = ""  # layout模板
    _ADDITIONAL_DEFINITIONS_TEMPLATE: str = ""  # 额外定义模板
    _FEATURES: Dict[str, Any] = {}  # 核函数特性字典
    # 参数插入映射：在哪些参数名之后插入什么参数
    # 例如：{"layoutA": "strideA", "layoutB": "strideB"} 表示在 layoutA 后插入 strideA
    _PARAMS_INSERTIONS: Dict[str, str] = {}  # 参数插入映射

    def __init__(self, *args, **kwargs):
        self.l1_tile_shape, self.l0_tile_shape = self.get_default_tile_shape()
        self.block_scheduler = None
        self.dispatch_policy = self.get_default_dispatch_policy_list()
        self._features = {}
        # 从类属性中读取特性值
        if hasattr(self.__class__, "_FEATURES"):
            self._features.update(self.__class__._FEATURES)
        # 初始化 relu_enable，默认为 False
        self.relu_enable = False

    @abstractmethod
    def get_default_tile_shape(self) -> Tuple[GemmShape, GemmShape]:
        pass

    def get_default_dispatch_policy_list(self) -> List[DispatchPolicy]:
        """获取默认的 dispatch_policy 列表.

        子类可以重写此方法以定义自己的默认 dispatch_policy 列表。
        如果子类不重写，默认返回空列表。

        :return: 默认的 dispatch_policy 列表.
        :rtype: List[DispatchPolicy]
        """
        return []

    """op interface,尚未实现"""

    def get_workspace_size(self) -> int:
        return 0

    def need_workspace(self) -> bool:
        return False

    def get_core_num(self) -> int:
        return 0

    """flag interface"""

    def is_support(self, feature: str) -> bool:
        """检查是否支持某个特性"""
        pass

    def __getattr__(self, name: str) -> Any:
        """通过属性方式访问特性值"""
        # 对于特殊属性（如 __setstate__, __getstate__ 等），直接抛出 AttributeError
        # 避免干扰 deepcopy、pickle 等内部操作
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # 直接检查 __dict__ 而不是使用 hasattr，避免递归
        # _features 在 __init__ 中总是会被初始化，所以可以直接检查
        if "_features" in self.__dict__ and name in self._features:
            return self._features[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    """tune interface"""

    def set_l1_tile_shape(self, l1_tile_shape: GemmShape):
        """设置L1 Tile Shape.

        :param l1_tile_shape: L1 Tile Shape.
        :type l1_tile_shape: GemmShape
        """
        self.l1_tile_shape = l1_tile_shape

    def set_l0_tile_shape(self, l0_tile_shape: GemmShape):
        """设置L0 Tile Shape.

        :param l0_tile_shape: L0 Tile Shape.
        :type l0_tile_shape: GemmShape
        """
        self.l0_tile_shape = l0_tile_shape

    def set_dispatch_policy(
        self, dispatch_policy: Union[DispatchPolicy, List[DispatchPolicy]]
    ):
        """设置Dispatch Policy.

        :param dispatch_policy: Dispatch Policy 或 Dispatch Policy 列表。如果传入单个 policy，会转换为包含该 policy 的列表。
        :type dispatch_policy: Union[DispatchPolicy, List[DispatchPolicy]]
        """
        if isinstance(dispatch_policy, list):
            self.dispatch_policy = dispatch_policy
        else:
            self.dispatch_policy = (
                [dispatch_policy] if dispatch_policy is not None else []
            )

    def get_dispatch_policy(self) -> List[DispatchPolicy]:
        """获取Dispatch Policy 列表.

        :return: Dispatch Policy 列表。默认情况下返回包含默认 policy 的列表。
        :rtype: List[DispatchPolicy]
        """
        return self.dispatch_policy

    def set_block_scheduler(self, block_scheduler: BlockScheduler):
        """设置Block Scheduler.

        :param block_scheduler: Block Scheduler.
        :type block_scheduler: BlockScheduler
        """
        self.block_scheduler = block_scheduler

    def set_relu_enable(self, relu_enable: bool):
        """设置是否启用随路 Relu.

        :param relu_enable: 是否启用随路 Relu.
        :type relu_enable: bool
        """
        self.relu_enable = relu_enable

    def set_use_hf32_mode(self, is_hf32: bool):
        """设置是否启用 HF32 模式.

        此方法会检查当前 dispatch_policy 列表中是否包含 use_hf32_mode 属性，
        如果包含则设置为指定值，如果不包含则抛出提示信息。

        :param is_hf32: 是否启用 HF32 模式.
        :type is_hf32: bool
        :raises ValueError: 如果 dispatch_policy 中没有任何 policy 支持 use_hf32_mode
        """
        if not self.dispatch_policy:
            raise ValueError("dispatch_policy 列表为空，无法设置 use_hf32_mode")

        policies_with_hf32 = []
        policies_without_hf32 = []

        for policy in self.dispatch_policy:
            if hasattr(policy, "use_hf32_mode"):
                policy.use_hf32_mode = is_hf32
                policies_with_hf32.append(policy.__class__.__name__)
            else:
                policies_without_hf32.append(policy.__class__.__name__)

        if not policies_with_hf32:
            policy_names = ", ".join(policies_without_hf32)
            raise ValueError(
                f"当前 dispatch_policy 中的 policy ({policy_names}) 不支持 use_hf32_mode。"
                f"支持 use_hf32_mode 的 policy 包括: MmadPingpong, MmadPreloadAsyncWithCallback, MmadMultiBatch"
            )

    def tune(
        self,
        l1_tile_shape: Optional[GemmShape] = None,
        l0_tile_shape: Optional[GemmShape] = None,
        dispatch_policy: Optional[
            Union[DispatchPolicy, List[DispatchPolicy]]
        ] = None,  # reserved
        block_scheduler: Optional[BlockScheduler] = None,  # reserved
        relu_enable: Optional[bool] = None,
        is_hf32: Optional[bool] = None,
        **kwargs,
    ):
        if_hf32 = _get_deprecated_arg(kwargs, "if_hf32", "is_hf32")
        if is_hf32 is None and if_hf32 is not None:
            is_hf32 = if_hf32
        elif if_hf32 is not None and if_hf32 != is_hf32:
            raise ValueError(
                "There is a conflict between suggested 'is_hf32' and deprecated 'if_hf32'."
            )

        self.set_l1_tile_shape(l1_tile_shape or self.l1_tile_shape)
        self.set_l0_tile_shape(l0_tile_shape or self.l0_tile_shape)
        if dispatch_policy is not None:
            self.set_dispatch_policy(dispatch_policy)
        self.set_block_scheduler(block_scheduler or self.block_scheduler)
        if relu_enable is not None:
            self.set_relu_enable(relu_enable)
        if is_hf32 is not None:
            self.set_use_hf32_mode(is_hf32)

    """codegen interface"""

    @abstractmethod
    def get_render_params(self) -> Dict[str, Any]:
        pass

    def gen_includes(self) -> str:
        """生成头文件引入部分源码.

        :return: 头文件.
        :rtype: str
        """
        return "\n".join(
            ["#include <{}>".format(include) for include in self._INCLUDES]
        )

    def gen_kernel_name(self) -> str:
        """生成核函数名.

        :return: 渲染后的核函数名.
        :rtype: str
        """
        return self._KERNEL_NAME.format(**self.get_render_params())

    def gen_params_device(self, def_mode: bool = False) -> str:
        """生成核函数参数.

        :param def_mode: 是否为定义模式. 在定义模式下，会生成函数定义中的形式，如`int a, int b`.
        否则，生成函数调用中的形式，如`a, b`.
        :type def_mode: bool
        :return: 核函数参数.
        :rtype: str
        """
        if def_mode:
            generated = ", ".join(
                [
                    "{} {}".format(get_type_name(type_str), var_name)
                    for type_str, var_name in self._PARAMS_DEVICE
                ]
            )
        else:
            generated = ", ".join(
                ["{}".format(var_name) for _, var_name in self._PARAMS_DEVICE]
            )
        return generated

    def gen_kernel_template(self) -> str:
        """生成核函数组装部分模板.
        比如，从`using Arch=ArchTag::AtlasA2`到`kernel(params);`之间的部分.

        :return: 渲染后的核函数模板.
        :rtype: str
        """
        render_params = self.get_render_params()
        for key, value in render_params.items():
            if hasattr(value, "value"):
                value = value.value
            render_params[key] = str(value)
        result = self._DISPATCH_POLICY.format(
            **render_params
        ) + self._KERNEL_TEMPLATE.format(**render_params)
        return result

    def gen_input_template(self) -> str:
        """生成输入变量定义代码.
        生成包含 M, K, N 等输入变量的定义代码块.

        :return: 输入变量定义代码，如果不存在 _INPUT_TEMPLATE 则返回空字符串.
        :rtype: str
        """
        if (
            not hasattr(self.__class__, "_INPUT_TEMPLATE")
            or not self.__class__._INPUT_TEMPLATE
        ):
            return ""

        render_params = self.get_render_params()
        for key, value in render_params.items():
            if hasattr(value, "value"):
                value = value.value
            render_params[key] = str(value)

        return self.__class__._INPUT_TEMPLATE.format(**render_params)

    def gen_layout_template(self) -> str:
        """生成 layout 相关信息代码.
        生成包含 M, K, N 定义和 layout tag 的代码块.

        :return: layout 相关代码.
        :rtype: str
        """
        render_params = self.get_render_params()
        for key, value in render_params.items():
            if hasattr(value, "value"):
                value = value.value
            render_params[key] = str(value)
        return self._LAYOUT_TEMPLATE.format(
            **render_params,
        )

    def _insert_params_recursive(self, param: str, params_list: List[str]) -> None:
        """递归插入参数.

        如果 param 在 _PARAMS_INSERTIONS 中，插入对应的参数，并递归检查插入的参数。

        :param param: 当前参数名.
        :type param: str
        :param params_list: 参数列表（会被修改）.
        :type params_list: List[str]
        """
        if param in self._PARAMS_INSERTIONS:
            inserted_param = self._PARAMS_INSERTIONS[param]
            params_list.append(inserted_param)
            # 递归检查插入的参数是否也需要插入其他参数
            self._insert_params_recursive(inserted_param, params_list)

    def transform_params_for_construction(self, params_str: str) -> str:
        """转换参数列表用于构造 Params 对象.

        子类可以重写此方法来修改参数列表，例如在特定参数后插入额外的参数。
        支持链式插入：如果插入的参数本身也在 _PARAMS_INSERTIONS 中，会继续插入。
        例如：在 layoutA 后插入 strideA，在 strideA 后插入 strideB。

        :param params_str: 原始参数字符串，格式如 "param1, param2, param3".
        :type params_str: str
        :return: 转换后的参数字符串.
        :rtype: str
        """
        if not self._PARAMS_INSERTIONS:
            return params_str
        params_list = []
        for param in params_str.split(", "):
            params_list.append(param)
            self._insert_params_recursive(param, params_list)
        return ", ".join(params_list)

    def _gen_kernel_params_for_def(self) -> str:
        """生成函数定义时的参数列表，只包含 GM_ADDR 类型的参数和 M, N, K.

        :return: 函数定义参数列表.
        :rtype: str
        """
        # 只包含 GM_ADDR 类型的参数
        gm_addr_params = [
            (type_str, var_name)
            for type_str, var_name in self._PARAMS_DEVICE
            if type_str == GM_ADDR
        ]
        result = ", ".join(
            [
                "{} {}".format(get_type_name(type_str), var_name)
                for type_str, var_name in gm_addr_params
            ]
        )
        result += ", uint32_t M, uint32_t N, uint32_t K"
        return result
