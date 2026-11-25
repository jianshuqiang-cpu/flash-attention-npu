# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
# the software repository for the full text of the License.

from collections.abc import Iterable

import torch
import torch.nn.functional as F

# Hardware configuration
BYTE_PER_C0 = 32
C0 = 16
N0 = 16  # Cout fractal dimension for weight layout KDC1KHKWN1N0C0


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def nchw_to_nc1hwc0(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert NCHW tensor to NC1HWC0 tensor. (For `Fmap`)

    Args:
        tensor (torch.Tensor): Input tensor in NCHW format.

    Returns:
        torch.Tensor: Output tensor in NC1HWC0 format.
    """
    N, C, H, W = tensor.shape
    C1 = (C + C0 - 1) // C0
    pad_c = C1 * C0 - C
    if pad_c > 0:
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_c))
    return tensor.reshape(N, C1, C0, H, W).permute(0, 1, 3, 4, 2).contiguous()


def cihw_to_ci1khkwcoci0(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert OIHW tensor to CI1KHKWCOCI0 tensor. (For `Filter`)

    Args:
        tensor (torch.Tensor): Input tensor in OIHW format.

    Returns:
        torch.Tensor: Output tensor in CI1KHKWCOCI0 format.
    """
    OC, IC, KH, KW = tensor.shape
    IC1 = (IC + C0 - 1) // C0
    pad_ic = IC1 * C0 - IC
    if pad_ic > 0:
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, pad_ic))
    return tensor.reshape(OC, IC1, C0, KH, KW).permute(1, 3, 4, 0, 2).contiguous()


def nc1hwc0_to_nchw(tensor: torch.Tensor, N: int, C: int, H: int, W: int) -> torch.Tensor:
    """
    Convert NC1HWC0 tensor back to NCHW tensor. (For `Fmap`)

    Args:
        tensor (torch.Tensor): Input tensor in NC1HWC0 format.
        N (int): Number of samples in the batch.
        C (int): Number of channels in the feature map.
        H (int): Height of the feature map.
        W (int): Width of the feature map.

    Returns:
        torch.Tensor: Output tensor in NCHW format.
    """
    C1 = (C + C0 - 1) // C0
    return (
        tensor.reshape(N, C1, H, W, C0)
        .permute(0, 1, 4, 2, 3)
        .contiguous()
        .reshape(N, C1 * C0, H, W)[:, :C, :, :]
    )


def get_howo(
    input_size: Iterable[int],  # [h, w]
    filter_size: Iterable[int],  # [kh, kw]
    padding: Iterable[int],  # [pL, pR, pT, pD]
    dilation: Iterable[int],  # [dH, dW]
    stride: Iterable[int],  # [sH, sW]
) -> Iterable[int]:  # [ho, wo]
    """
    Calculate the output height and width of the convolution.
    """
    H, W = input_size
    Kh, Kw = filter_size
    Ho = (H + padding[2] + padding[3] - dilation[0] * (Kh - 1) - 1) // stride[0] + 1
    Wo = (W + padding[0] + padding[1] - dilation[1] * (Kw - 1) - 1) // stride[1] + 1
    return Ho, Wo


def conv_can_implement(
    filter_size: Iterable[int],  # [kh, kw]
    dilation: Iterable[int],  # [dH, dW]
    stride: Iterable[int],  # [sH, sW]
    fmap_l1_shape: Iterable[int] = (8, 12, 8),  # [hoBlock, woBlock, cin1BlockSmall]
    filter_l1_shape: Iterable[int] = (96, 8),  # [cout, cin1BlockBig]
    l0_shape: Iterable[int] = (16, 96, 16),  # [mL0, nL0, kL0]
    l1a_stages: int = 2,
    l1b_stages: int = 2,
    l0a_stages: int = 2,
    l0b_stages: int = 2,
    l1_size: int = 512 * 1024,  # AtlasA2 L1 Size
    l0a_size: int = 64 * 1024,
    l0b_size: int = 64 * 1024,
    l0c_size: int = 128 * 1024,
    sizeof_output: int = 2,
) -> bool:
    """Check whether a conv2d problem fits the on-chip buffers.

    Returns:
        bool: ``True`` if the problem can be implemented, ``False`` otherwise.
    """
    kh, kw = filter_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    fmap_ho, fmap_wo, fmap_cin1 = fmap_l1_shape
    filter_cout, filter_cin1 = filter_l1_shape
    _, l0_n, l0_k = l0_shape

    # MAX_STAGES clamp (kernel: MAX_STAGES = 2)
    max_stages = 2
    l1a_stages = min(l1a_stages, max_stages)
    l1b_stages = min(l1b_stages, max_stages)
    l0a_stages = min(l0a_stages, max_stages)
    l0b_stages = min(l0b_stages, max_stages)

    cin1_l0_block = l0_k // (kh * kw * C0)
    cin1_l0_block = max(cin1_l0_block, 1)

    cout_l0_block = _round_up(l0_n, C0)
    cout_block = _round_up(filter_cout, C0)

    hi_block = (fmap_ho - 1) * stride_h + (kh - 1) * dilation_h + 1
    wi_block = (fmap_wo - 1) * stride_w + (kw - 1) * dilation_w + 1

    l1_data_size = (
        l1a_stages * fmap_cin1 * hi_block * wi_block * BYTE_PER_C0
        + l1b_stages * filter_cin1 * kh * kw * filter_cout * BYTE_PER_C0
    )
    l0a_data_size = l0a_stages * fmap_ho * fmap_wo * (cin1_l0_block * kh * kw * BYTE_PER_C0)
    l0b_data_size = l0b_stages * (cin1_l0_block * kh * kw * BYTE_PER_C0) * cout_l0_block
    l0c_data_size = fmap_ho * fmap_wo * cout_block * sizeof_output

    return not (
        l1_data_size > l1_size
        or l0a_data_size > l0a_size
        or l0b_data_size > l0b_size
        or l0c_data_size > l0c_size
    )


def ncdhw_to_ndc1hwc0(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert NCDHW tensor to 1-D NDC1HWC0 tensor. (For ``conv_bias`` fmap)

    Args:
        tensor: ``(N, C, D, H, W)`` float16.

    Returns:
        1-D contiguous ``(N * D * C1 * H * W * C0,)``.
    """
    N, C, D, H, W = tensor.shape
    C1 = (C + C0 - 1) // C0
    pad_c = C1 * C0 - C
    if pad_c > 0:
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, 0, 0, pad_c))
    tensor = tensor.reshape(N, C1, C0, D, H, W).permute(0, 3, 1, 4, 5, 2)
    return tensor.contiguous().reshape(-1)


def coidkhkw_to_kdc1khkwn1n0c0(tensor: torch.Tensor, n0: int = N0) -> torch.Tensor:
    """
    Convert COIDkKhKw tensor to 1-D KDC1KHKWN1N0C0 tensor. (For ``conv_bias`` weight)

    Args:
        tensor: ``(Cout, Cin, Kd, Kh, Kw)`` float16.

    Returns:
        1-D contiguous ``(Kd * C1 * Kh * Kw * N1 * N0 * C0,)``.
    """
    Cout, Cin, Kd, Kh, Kw = tensor.shape
    c1 = (Cin + C0 - 1) // C0
    n1 = (Cout + n0 - 1) // n0
    pad_c = c1 * C0 - Cin
    pad_n = n1 * n0 - Cout
    if pad_n > 0 or pad_c > 0:
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, 0, 0, pad_c, 0, pad_n))
    tensor = tensor.reshape(n1, n0, c1, C0, Kd, Kh, Kw).permute(4, 2, 5, 6, 0, 1, 3)
    return tensor.contiguous().reshape(-1)


def ndc1hwc0_to_ncdhw(
    tensor: torch.Tensor, N: int, Do: int, Cout: int, Ho: int, Wo: int
) -> torch.Tensor:
    """
    Convert 1-D NDC1HWC0 tensor back to NCDHW tensor. (For ``conv_bias`` output)

    Args:
        tensor: 1-D tensor in NDC1HWC0 layout.
        N: Batch size.
        Do: Output depth.
        Cout: Output channels (real, not padded).
        Ho: Output height.
        Wo: Output width.

    Returns:
        ``(N, Cout, Do, Ho, Wo)`` float16.
    """
    cout1 = (Cout + C0 - 1) // C0
    tensor = tensor.reshape(N, Do, cout1, Ho, Wo, C0).permute(0, 2, 5, 1, 3, 4)
    tensor = tensor.contiguous().reshape(N, cout1 * C0, Do, Ho, Wo)
    return tensor[:, :Cout, :, :, :]
