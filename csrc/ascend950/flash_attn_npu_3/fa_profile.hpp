/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the license.
 *
 * Modified by Minghua Shen, 2026.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

#include <ATen/ATen.h>

#include "acl/acl.h"
#include <aprof_pub.h>
#include "torch_npu/csrc/core/npu/NPUFormat.h"

namespace fa3_profile {

inline std::atomic<bool> tensor_info_enabled{false};

inline int32_t UpdateProfilingState(uint32_t control, void* data, uint32_t length)
{
    if (control != PROF_CTRL_SWITCH || data == nullptr || length != sizeof(MsprofCommandHandle)) {
        return 0;
    }

    const auto& command = *static_cast<const MsprofCommandHandle*>(data);
    switch (command.type) {
    case PROF_COMMANDHANDLE_TYPE_START:
        tensor_info_enabled.store((command.profSwitch & PROF_TASK_TIME_MASK) != 0 &&
                                      (command.profSwitch & PROF_TASK_TIME_L1_MASK) != 0,
                                  std::memory_order_relaxed);
        break;
    case PROF_COMMANDHANDLE_TYPE_STOP:
    case PROF_COMMANDHANDLE_TYPE_FINALIZE:
        tensor_info_enabled.store(false, std::memory_order_relaxed);
        break;
    default:
        break;
    }
    return 0;
}

inline void RegisterProfilingCallback()
{
    // CANN sends its l0 + aicore-shape switch to the ACLNN reporting module.
    constexpr uint32_t nnopbase_module_id = 61;
    (void)MsprofRegisterCallback(nnopbase_module_id, UpdateProfilingState);
}

inline bool IsEnabled()
{
    return tensor_info_enabled.load(std::memory_order_relaxed);
}

inline aclDataType ProfileDataType(at::ScalarType type)
{
    switch (type) {
    case at::kHalf:
        return ACL_FLOAT16;
    case at::kBFloat16:
        return ACL_BF16;
    case at::ScalarType::Float8_e4m3fn:
        return ACL_FLOAT8_E4M3FN;
    case at::kFloat:
        return ACL_FLOAT;
    case at::kInt:
        return ACL_INT32;
    case at::kByte:
        return ACL_UINT8;
    default:
        return ACL_DT_UNDEFINED;
    }
}

inline const at::Tensor* TensorPtr(const std::optional<at::Tensor>& tensor)
{
    return tensor ? &*tensor : nullptr;
}

inline void WriteFloat(std::ostream& stream, float value)
{
    if (std::isfinite(value)) {
        stream.precision(std::numeric_limits<float>::max_digits10);
        stream << value;
    } else {
        stream << (std::isnan(value) ? "\"nan\"" : (value > 0 ? "\"inf\"" : "\"-inf\""));
    }
}

struct TensorArgument {
    const char* name;
    const at::Tensor* tensor;
};

class TensorInfo {
  public:
    TensorInfo(std::initializer_list<const at::Tensor*> inputs, std::initializer_list<const at::Tensor*> outputs)
    {
        if (IsEnabled()) {
            tensors_.reserve(inputs.size() + outputs.size());
            for (const auto* tensor : inputs) {
                Append(tensor, MSPROF_GE_TENSOR_TYPE_INPUT);
            }
            AppendOutputs(outputs);
        }
    }

    template <typename WriteParameters>
    TensorInfo(const char* scope, std::initializer_list<TensorArgument> inputs,
               std::initializer_list<const at::Tensor*> outputs, WriteParameters writeParameters)
    {
        if (!IsEnabled()) {
            return;
        }
        tensors_.reserve(inputs.size() + outputs.size());
        std::ostringstream fields;
        fields.imbue(std::locale::classic());
        fields << std::boolalpha << "{\"fa3_scope\":\"" << scope << "\",\"inputs\":[";
        const char* separator = "";
        for (const auto& input : inputs) {
            if (Append(input.tensor, MSPROF_GE_TENSOR_TYPE_INPUT)) {
                fields << separator << '"' << input.name << '"';
                separator = ",";
            }
        }
        fields << "],\"params\":{";
        writeParameters(fields);
        fields << "}}";
        parameters_ = fields.str();
        AppendOutputs(outputs);
    }

    uint64_t Begin() const
    {
        return !tensors_.empty() && IsEnabled() ? MsprofSysCycleTime() : 0;
    }

    void Report(uint64_t begin) const
    {
        if (begin == 0) {
            return;
        }

        // Keep the task's original device kernel name in msprof.
        constexpr char kernel_name_ref[] = "Refer_to_kernelName";
        MsprofApi launch{};
        launch.endTime = MsprofSysCycleTime();
        launch.beginTime = begin;
        launch.level = MSPROF_REPORT_NODE_LEVEL;
        launch.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
        launch.threadId = static_cast<uint32_t>(syscall(SYS_gettid));
        launch.itemId = MsprofGetHashId(kernel_name_ref, sizeof(kernel_name_ref) - 1);
        if (MsprofReportApi(1, &launch) != 0) {
            return;
        }

        MsprofAdditionalInfo event{};
        event.level = launch.level;
        event.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
        event.threadId = launch.threadId;
        event.timeStamp = begin + 1;
        static_assert(sizeof(MsprofTensorInfo) <= sizeof(event.data));
        event.dataLen = sizeof(MsprofTensorInfo);
        for (size_t offset = 0; offset < tensors_.size(); offset += MSPROF_GE_TENSOR_DATA_NUM) {
            MsprofTensorInfo info{};
            info.opName = launch.itemId;
            info.tensorNum = std::min<size_t>(tensors_.size() - offset, MSPROF_GE_TENSOR_DATA_NUM);
            std::copy_n(tensors_.data() + offset, info.tensorNum, info.tensorData);
            std::memcpy(event.data, &info, sizeof(info));
            (void)MsprofReportAdditionalInfo(1, &event, sizeof(event));
        }

        if (!parameters_.empty()) {
            // The hash dictionary stores the fields for this tensor record.
            MsprofCompactInfo attr{};
            attr.level = launch.level;
            attr.type = MSPROF_REPORT_NODE_ATTR_INFO_TYPE;
            attr.threadId = launch.threadId;
            attr.timeStamp = event.timeStamp;
            const MsprofAttrInfo info{launch.itemId, OP_ATTR, MsprofGetHashId(parameters_.data(), parameters_.size())};
            static_assert(sizeof(info) <= sizeof(attr.data.info));
            attr.dataLen = sizeof(info);
            std::memcpy(attr.data.info, &info, sizeof(info));
            (void)MsprofReportCompactInfo(1, &attr, sizeof(attr));
        }
    }

    template <typename Launch>
    void operator()(Launch&& launch) const
    {
        const uint64_t begin = Begin();
        launch();
        Report(begin);
    }

  private:
    bool Append(const at::Tensor* tensor, uint32_t direction)
    {
        if (tensor == nullptr || !tensor->defined() || tensor->device().type() != at::kPrivateUse1 ||
            tensor->dim() > MSPROF_GE_TENSOR_DATA_SHAPE_LEN) {
            return false;
        }
        const auto sizes = tensor->sizes();
        if (std::any_of(sizes.begin(), sizes.end(),
                        [](int64_t size) { return size > std::numeric_limits<uint32_t>::max(); })) {
            return false;
        }
        auto& desc = tensors_.emplace_back();
        desc.tensorType = direction;
        desc.format = static_cast<uint32_t>(at_npu::native::get_npu_format(*tensor));
        desc.dataType = static_cast<uint32_t>(ProfileDataType(tensor->scalar_type()));
        std::copy(sizes.begin(), sizes.end(), desc.shape);
        return true;
    }

    void AppendOutputs(std::initializer_list<const at::Tensor*> outputs)
    {
        for (const auto* tensor : outputs) {
            Append(tensor, MSPROF_GE_TENSOR_TYPE_OUTPUT);
        }
    }

    // The queued launch owns the descriptors and attributes.
    std::vector<MsrofTensorData> tensors_;
    std::string parameters_;
};

} // namespace fa3_profile
