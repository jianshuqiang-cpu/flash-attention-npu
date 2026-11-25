/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTEST_REGISTER_H
#define OPTEST_REGISTER_H

#include <tuple>

#include <ATen/Tensor.h>
#include <torch/torch.h>

namespace catlass_torch {
namespace _register {
/**
 * @brief Return the singleton torch library used for CATLASS op registration.
 * @return Mutable torch::Library registered under the ``catlass`` namespace.
 */
inline torch::Library& GetTorchLibrary()
{
    thread_local static torch::Library library(torch::Library::DEF, "catlass", c10::nullopt, __FILE__, __LINE__);
    return library;
}

template <typename Func>
struct RegisterFunc {
    /**
     * @brief Register one function as a PrivateUse1 torch operator.
     * @param opFuncName Public operator name under ``torch.ops.catlass``.
     * @param opFunc Callable function pointer or functor used as implementation.
     */
    RegisterFunc(const char* opFuncName, Func&& opFunc)
    {
        c10::OperatorName opName = GetTorchLibrary()._resolve(opFuncName);

        auto opFuncSchema = c10::detail::inferFunctionSchemaFromFunctor<std::decay_t<Func>>();
        auto clonedOpFuncSchema = opFuncSchema->cloneWithName(std::move(opName.name), std::move(opName.overload_name));

        GetTorchLibrary().def(std::move(clonedOpFuncSchema));
        GetTorchLibrary().impl(opFuncName, c10::DispatchKey::PrivateUse1, opFunc);
    }
};
} // namespace _register
} // namespace catlass_torch

// register a single function
#define REGISTER_TORCH_FUNC(opFunc) \
    catlass_torch::_register::RegisterFunc<decltype(&opFunc)> register_##opFunc(#opFunc, &opFunc)
#endif
