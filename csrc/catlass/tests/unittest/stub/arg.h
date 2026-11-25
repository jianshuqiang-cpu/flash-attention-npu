/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_STUB_ARG_H
#define ASCENDC_STUB_ARG_H

#include <typeindex>
#include <memory>
#include <type_traits>
template <typename T>
struct ArgUseRef : std::false_type {};

/**
 * @brief 封装调用某类型的 GetAddr 方法，
 *        并在 T 无 GetAddr() 时 fallback 到零地址
 * @tparam T    待检测类型（通常为 AscendC::LocalTensor 或 AscendC::GlobalTensor，
 *             详见 kernel_tensor.h）
 * @tparam Void SFINAE 辅助参数，主模板中默认为 void，
 *             匹配 T 不存在 GetAddr() 的情形；偏特化匹配 T 存在 GetAddr() 的情形
 */
template <typename T, typename = void>
struct GetAddrDispatcher {
    static uintptr_t Get(const void*) { return 0; }
};

template <typename T>
struct GetAddrDispatcher<T, std::void_t<decltype(std::declval<const T&>().GetAddr())>> {
    static uintptr_t Get(const void* p) {
        return static_cast<const T*>(p)->GetAddr();
    }
};

struct Arg {
    std::type_index type = typeid(void);

    void* value = nullptr;
    std::shared_ptr<void> holder = nullptr;
    uintptr_t (*getAddrFn)(const void*) = nullptr;

    Arg() = default;

    template <typename T>
    Arg(const T& v)
    {
        using RawT = std::remove_cv_t<std::remove_reference_t<T>>;
        type = typeid(RawT);

        holder = std::make_shared<RawT>(v);
        if constexpr (!ArgUseRef<RawT>::value) {
            value = holder.get();
        } else {
            value = reinterpret_cast<void*>(const_cast<RawT*>(&v));
        }
        getAddrFn = &GetAddrDispatcher<RawT>::Get;
    }

    uintptr_t GetInstAddr() const
    {
        return getAddrFn ? getAddrFn(holder.get()) : 0;
    }

    std::type_index Type() const
    {
        return type;
    }

    template <typename T = void>
    T* Value() const
    {
        return reinterpret_cast<T*>(value);
    }

    void* RawValue() const
    {
        return value;
    }

    template <typename T=void>
    T* InstValue() const
    {
        if (type == std::type_index(typeid(T))) {
            return static_cast<T*>(holder.get());
        }
        return nullptr;
    }

    // ========================
    // 类型参数
    // ========================
    template <typename T>
    static Arg MakeArg()
    {
        Arg arg;
        arg.type = typeid(T);
        return arg;
    }

    // ========================
    // 值参数
    // ========================
    template <typename T>
    static Arg MakeArgWithValue(const T& argValue)
    {
        return Arg(argValue);
    }
};

#endif