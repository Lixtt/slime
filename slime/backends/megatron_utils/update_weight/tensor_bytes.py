from collections.abc import Sequence

import torch


def tensor_to_cpu_byte_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous 1D CPU uint8 view of a tensor payload."""

    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    tensor = tensor.contiguous()
    return tensor.view(torch.uint8).reshape(-1).contiguous()


def restore_tensor_from_cpu_byte_tensor(
    byte_tensor: torch.Tensor,
    *,
    shape: torch.Size | Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore a CPU tensor from a 1D uint8 payload."""

    if byte_tensor.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 byte tensor, got {byte_tensor.dtype}")
    if byte_tensor.device.type != "cpu":
        byte_tensor = byte_tensor.cpu()
    byte_tensor = byte_tensor.contiguous().reshape(-1)

    restored = torch.empty(tuple(shape), dtype=dtype, device="cpu")
    restored.view(torch.uint8).reshape(-1).copy_(byte_tensor)
    return restored
