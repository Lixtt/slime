import os

if os.environ.get("SLIME_ENABLE_QWEN35_SGLANG_PATCH") == "1":
    from slime.backends.sglang_utils.qwen3_5 import patch_sglang_qwen35
    patch_sglang_qwen35()

try:
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM
except (ImportError, ModuleNotFoundError):
    EntryClass = []
else:
    EntryClass = [Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM]
