import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "slime"
    / "backends"
    / "megatron_utils"
    / "update_weight"
    / "tensor_bytes.py"
)
_SPEC = importlib.util.spec_from_file_location("tensor_bytes_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

tensor_to_cpu_byte_tensor = _MODULE.tensor_to_cpu_byte_tensor
restore_tensor_from_cpu_byte_tensor = _MODULE.restore_tensor_from_cpu_byte_tensor


def test_cpu_byte_roundtrip_preserves_multidimensional_bf16_tensor():
    source = torch.arange(576 * 12288, dtype=torch.float32).reshape(576, 12288).to(torch.bfloat16)

    payload = tensor_to_cpu_byte_tensor(source)
    restored = restore_tensor_from_cpu_byte_tensor(payload, shape=source.shape, dtype=source.dtype)

    assert payload.ndim == 1
    assert payload.numel() == source.numel() * source.element_size()
    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    torch.testing.assert_close(restored, source)


def test_cpu_byte_roundtrip_preserves_noncontiguous_float16_tensor():
    source = torch.arange(64 * 32, dtype=torch.float32).reshape(64, 32).t().to(torch.float16)

    payload = tensor_to_cpu_byte_tensor(source)
    restored = restore_tensor_from_cpu_byte_tensor(payload, shape=source.shape, dtype=source.dtype)

    assert payload.ndim == 1
    assert payload.numel() == source.numel() * source.element_size()
    torch.testing.assert_close(restored, source.contiguous())
