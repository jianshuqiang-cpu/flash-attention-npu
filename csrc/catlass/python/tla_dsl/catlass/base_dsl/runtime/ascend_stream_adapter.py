"""Resolve Ascend device id and stream handle for host launch.

Ascend launch takes an integer stream handle from PyACL. Resolution order:

1. ``torch.npu`` current device / ``npu_stream`` (when torch_npu is available)
2. ``CATLASS_DSL_NPU_DEVICE`` env (device only); stream must come from torch or
   an explicit ``stream=`` launch kwarg

Preferring the torch stream matters: example harnesses fill I/O tensors with
``torch.*`` on the torch stream. Launching on a separate ACL stream races with
async torch init/copy and can leave outputs at their sentinel values.
"""

from __future__ import annotations

import os
from typing import Any

from ...execution import TlaRuntimeUnavailableError


def current_device() -> int:
    """Return the device id for kernel load/launch."""
    try:
        import torch

        return int(torch.npu.current_device())
    except Exception:
        pass
    return int(os.getenv("CATLASS_DSL_NPU_DEVICE", "0"))


def current_stream(device: int) -> int:
    """Return the RT stream handle for Ascend kernel launch."""
    try:
        import torch

        return int(torch.npu.current_stream(device).npu_stream)
    except Exception:
        pass
    raise TlaRuntimeUnavailableError(
        "Failed to infer current NPU stream. Install torch_npu or pass "
        "`stream=<aclrtStream integer>`."
    )


def as_stream(stream: Any, *, device: int) -> int:
    """Normalize a user-supplied stream to an RT integer handle."""
    if stream is None:
        return current_stream(device)
    npu_stream = getattr(stream, "npu_stream", None)
    if npu_stream is not None:
        return int(npu_stream)
    return int(stream)


__all__ = ["as_stream", "current_device", "current_stream"]
