#pragma once

#include <ATen/core/Tensor.h>

void stage_fp8_kv(const at::Tensor& key, const at::Tensor& value,
                  const at::Tensor& keyScale, const at::Tensor& valueScale,
                  const at::Tensor& sourcePages, const at::Tensor& keyOut,
                  const at::Tensor& valueOut);
