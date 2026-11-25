# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import struct
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Union, List

import numpy as np
from argparse import ArgumentParser

rng = np.random.default_rng()


class QuantMode(Enum):
    PerTensor = auto()
    PerToken = auto()
    PerChannel = auto()
    Default = auto()

    @classmethod
    def get_conv(cls):
        return {
            "per_tensor": cls.PerTensor,
            "per_token": cls.PerToken,
            "per_channel": cls.PerChannel,
            "default": cls.Default,
        }

    @classmethod
    def from_str(cls, s: str) -> "QuantMode":
        if s not in cls.get_conv():
            raise ValueError(f"Unknown quant mode: {s}")
        return cls.get_conv()[s]

    @classmethod
    def get_available_modes(cls) -> List[str]:
        return list(cls.get_conv().keys())


PerTensorArgs = Tuple[QuantMode, np.ndarray]
PerTokenArgs = Tuple[QuantMode, float]
PerChannelArgs = Tuple[QuantMode, np.ndarray]
DefaultArgs = Tuple[QuantMode]
QuantArgs = Union[PerTensorArgs, PerTokenArgs, PerChannelArgs, DefaultArgs]


@dataclass
class QuantMatmulDequantArgs:
    x1_quant_mode: QuantMode
    x2_quant_mode: QuantMode
    has_bias: bool
    m: int
    n: int
    k: int
    input_dir: Path
    output_dir: Path


def gen_quant_args_for_mode(mode: QuantMode, size: int = -1) -> QuantArgs:
    if mode == QuantMode.PerChannel:
        return (
            QuantMode.PerChannel,
            rng.uniform(-0.01, 0.01, [1, size]).astype(np.float32),
        )
    elif mode == QuantMode.PerToken:
        return (
            QuantMode.PerToken,
            rng.uniform(-0.01, 0.01, [size, 1]).astype(np.float32),
        )
    elif mode == QuantMode.PerTensor:
        return (
            QuantMode.PerTensor,
            rng.uniform(-0.01, 0.01, [1]).astype(np.float32).item(),
        )
    elif mode == QuantMode.Default:
        return (QuantMode.Default,)
    else:
        raise ValueError(f"Unknown quant mode: {mode}")


def save_quant_args(qa: QuantArgs, file_path: Path):
    if qa[0] in (QuantMode.PerChannel, QuantMode.PerToken):
        qa[1].tofile(file_path)
    elif qa[0] == QuantMode.PerTensor:
        with open(file_path, "wb") as f:
            value = struct.pack("f", qa[1])
            f.write(value)
    else:  # QuantMode.Default
        pass


def gen_golden_data_quant_int8_bf16(args: QuantMatmulDequantArgs):
    x1 = rng.integers(-127, 128, (args.m, args.k)).astype(np.int8)
    x2 = rng.integers(-127, 128, (args.k, args.n)).astype(np.int8)

    if args.has_bias:
        bias = rng.integers(-127, 128, (1, args.n)).astype(np.float32)
    else:
        bias = None

    x1_quant_args = gen_quant_args_for_mode(args.x1_quant_mode, args.m)
    x2_quant_args = gen_quant_args_for_mode(args.x2_quant_mode, args.n)

    golden_y = numpy_matmul_quant_int8_fp16(
        x1.astype(np.int32), x2.astype(np.int32), x1_quant_args, x2_quant_args, bias
    )

    x1.tofile(args.input_dir / "x1.bin")
    x2.tofile(args.input_dir / "x2.bin")
    golden_y.tofile(args.output_dir / "golden_o.bin")
    save_quant_args(x1_quant_args, args.input_dir / "x1_scale.bin")
    save_quant_args(x2_quant_args, args.input_dir / "x2_scale.bin")
    if args.has_bias:
        bias.tofile(args.input_dir / "bias.bin")


def numpy_matmul_quant_int8_fp16(
    x1: np.ndarray,
    x2: np.ndarray,
    x1_quant_args: QuantArgs,
    x2_quant_args: QuantArgs,
    bias=None,
):
    result = np.matmul(x1, x2).astype(np.float32)
    if len(x2_quant_args) > 1:
        result = result * x2_quant_args[1]
    if len(x1_quant_args) > 1:
        result = result * x1_quant_args[1]
    if bias is not None:
        result = result + bias

    result = result.astype(np.float32)
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--x1_quant_mode", type=str, choices=QuantMode.get_available_modes()
    )
    parser.add_argument(
        "--x2_quant_mode", type=str, choices=QuantMode.get_available_modes()
    )
    parser.add_argument("--has_bias", action="store_true")
    parser.add_argument("--shape", type=lambda s: tuple(map(int, s.split(" "))))
    args = parser.parse_args()

    m, n, k = args.shape

    input_dir = Path("./input/")
    input_dir.mkdir(exist_ok=True)
    output_dir = Path("./output/")
    output_dir.mkdir(exist_ok=True)

    args = QuantMatmulDequantArgs(
        x1_quant_mode=QuantMode.from_str(args.x1_quant_mode),
        x2_quant_mode=QuantMode.from_str(args.x2_quant_mode),
        has_bias=args.has_bias,
        m=m,
        n=n,
        k=k,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    gen_golden_data_quant_int8_bf16(args)
