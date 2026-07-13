"""Merge a LoRA adapter into a base HuggingFace model and export.

Usage:
    python merge_lora_adapter.py \
        --base-model /path/to/Qwen3-4B \
        --adapter /path/to/lora_checkpoint/model \
        --output /path/to/merged_model

The merged model can then be quantized with existing tools:
    python convert_hf_to_fp8.py ...
    python convert_hf_to_int4.py ...
"""

import argparse
import json
import os
import shutil

import torch


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, required=True, help="Path to base HuggingFace model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    parser.add_argument(
        "--merge-method",
        choices=("auto", "peft", "manual"),
        default="auto",
        help="Use PEFT merge, manual tensor merge, or auto-select PEFT when possible.",
    )
    parser.add_argument("--device", default="cpu", help="Device used for merging, e.g. cpu or cuda")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite output directory if it exists")
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.force:
        raise ValueError(f"Output directory {args.output} already exists. Use --force to overwrite.")

    print(f"Loading base model from {args.base_model}")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model_cls = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
    model = model_cls.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    if args.device != "cpu":
        model = model.to(args.device)

    # Load LoRA adapter weights
    adapter_weights_path = os.path.join(args.adapter, "adapter_weights.pt")
    adapter_config_path = os.path.join(args.adapter, "adapter_config.json")

    if os.path.exists(adapter_config_path) and args.merge_method != "manual":
        # Load via PEFT if config is available
        print(f"Loading LoRA adapter via PEFT from {args.adapter}")
        from peft import LoraConfig, get_peft_model

        with open(adapter_config_path) as f:
            lora_cfg = json.load(f)

        peft_config = LoraConfig(**{k: v for k, v in lora_cfg.items() if k in LoraConfig.__dataclass_fields__})
        model = get_peft_model(model, peft_config)

        if os.path.exists(adapter_weights_path):
            lora_state = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)
            # Load LoRA weights into the PEFT model
            model_state = model.state_dict()
            for k, v in lora_state.items():
                if k in model_state:
                    model_state[k] = v
            model.load_state_dict(model_state, strict=False)

        print("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()
    elif os.path.exists(adapter_weights_path):
        # Manual merge: load LoRA weights and add to base
        print(f"Loading LoRA weights from {adapter_weights_path}")
        lora_state = torch.load(adapter_weights_path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(lora_state)} LoRA tensors")
        lora_cfg = {}
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path) as f:
                lora_cfg = json.load(f)
        scale_default = float(lora_cfg.get("lora_alpha", 1.0)) / float(lora_cfg.get("r", 1.0))

        # Group by module: find pairs of lora_A and lora_B
        # Key pattern: base_model.model.<module>.lora_A.default.weight
        from collections import defaultdict

        modules = defaultdict(dict)
        for k, v in lora_state.items():
            if "lora_A" in k:
                base_key = k.split(".lora_A")[0]
                modules[base_key]["A"] = v
            elif "lora_B" in k:
                base_key = k.split(".lora_B")[0]
                modules[base_key]["B"] = v

        model_state = model.state_dict()
        merged_count = 0
        for module_key, ab in modules.items():
            if "A" not in ab or "B" not in ab:
                continue
            # Strip PEFT prefix to find base model key
            base_key = module_key
            for prefix in ["base_model.model.", "base_model."]:
                if base_key.startswith(prefix):
                    base_key = base_key[len(prefix):]
                    break
            weight_key = f"{base_key}.weight"
            if weight_key in model_state:
                # LoRA merge: W = W + B @ A * (alpha / r)
                weight_dtype = model_state[weight_key].dtype
                delta = (ab["B"].float() @ ab["A"].float()) * scale_default
                delta = delta.to(model_state[weight_key].device)
                model_state[weight_key] = model_state[weight_key].float() + delta
                model_state[weight_key] = model_state[weight_key].to(weight_dtype)
                merged_count += 1

        model.load_state_dict(model_state)
        print(f"Merged {merged_count} LoRA modules into base model")
    else:
        raise FileNotFoundError(f"No adapter found at {args.adapter}. Expected adapter_weights.pt or adapter_config.json")

    # Save merged model
    os.makedirs(args.output, exist_ok=True)
    print(f"Saving merged model to {args.output}")
    model.save_pretrained(args.output, safe_serialization=True)

    # Copy tokenizer and other assets from base model
    for filename in os.listdir(args.base_model):
        if filename.endswith(".safetensors") or filename == "model.safetensors.index.json":
            continue
        src = os.path.join(args.base_model, filename)
        if os.path.isfile(src):
            dst = os.path.join(args.output, filename)
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    print(f"Done! Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
