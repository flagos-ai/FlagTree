#!/usr/bin/env python3
"""
Qwen + FlagGems 简单演示脚本。

先用 CPU baseline 跑一次 warmup 和一次正式推理，再用 FlagGems 跑一次 warmup
和一次正式推理，分别打印两边输出。不做耗时统计，也不做自动比较。
命令行只保留两个常用参数：

    python3 demo_qwen_flaggems.py \
      --model qwen3 \
      --prompt "中国的首都在哪里"

"""

import argparse
import inspect
from contextlib import nullcontext


USE_GEMS = True
WARMUP_RUNS = 1
INFER_RUNS = 1
MAX_NEW_TOKENS = 256
CPU_DTYPE = "fp16"
GEMS_DTYPE = "fp16"
CPU_DEVICE = "cpu"
ATTN_IMPLEMENTATION = "eager"
DO_SAMPLE = False
TRUST_REMOTE_CODE = False
GEMS_LOG = "./demo_flaggems.log"
UNUSED_OPS = ["topk", "multinomial", "index"]
DEMO_STAGES = (
    ("cpu", False),
    ("flaggems", True),
)


class ModelSettings:
    def __init__(self, key, display_name, model_dir, prompt, chat_template_kwargs):
        self.key = key
        self.display_name = display_name
        self.model_dir = model_dir
        self.prompt = prompt
        self.chat_template_kwargs = chat_template_kwargs


MODEL_CONFIGS = {
    "qwen25": {
        "display_name": "Qwen2.5-0.5B-Instruct",
        "model_dir": "./models/Qwen2.5-0.5B-Instruct",
        "default_prompt": "中国的首都在哪里？",
        "aliases": {"qwen25", "qwen2.5", "qwen2.5-0.5b", "qwen2.5-0.5b-instruct"},
        "chat_template_kwargs": {},
    },
    "qwen3": {
        "display_name": "Qwen3-0.6B",
        "model_dir": "./models/Qwen3-0.6B",
        "default_prompt": "中国的首都在哪里？",
        "aliases": {"qwen3", "qwen3-0.6b", "qwen3-0.6"},
        "chat_template_kwargs": {"enable_thinking": False},
    },
}


def log(msg):
    print(f"[demo] {msg}", flush=True)


def canonical_model_key(model_name):
    normalized = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if normalized in config["aliases"]:
            return key
    valid = ", ".join(sorted(alias for cfg in MODEL_CONFIGS.values() for alias in cfg["aliases"]))
    raise ValueError(f"Unsupported model '{model_name}'. Valid values: {valid}")


def resolve_model_settings(model_name, prompt):
    key = canonical_model_key(model_name)
    config = MODEL_CONFIGS[key]
    return ModelSettings(
        key=key,
        display_name=config["display_name"],
        model_dir=config["model_dir"],
        prompt=prompt if prompt is not None else config["default_prompt"],
        chat_template_kwargs=dict(config["chat_template_kwargs"]),
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3", help="Model alias: qwen25 or qwen3.")
    parser.add_argument("--prompt", default=None, help="Input prompt.")
    return parser


def get_dtype(torch, dtype_name):
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def sync_device(torch, device):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    backend = getattr(torch, str(device).split(":")[0], None)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()


def load_model(torch, AutoModelForCausalLM, model_dir, device, dtype_name):
    kwargs = {
        "dtype": get_dtype(torch, dtype_name),
        "device_map": None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    return model.to(device).eval()


def apply_chat_template(tokenizer, prompt, chat_template_kwargs):
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        **chat_template_kwargs,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def build_inputs(tokenizer, settings, device):
    text = apply_chat_template(
        tokenizer,
        settings.prompt,
        settings.chat_template_kwargs,
    )
    return tokenizer([text], return_tensors="pt").to(device)


def flag_gems_context(flag_gems, enabled):
    if not enabled:
        return nullcontext()

    signature = inspect.signature(flag_gems.use_gems)
    kwargs = {}
    if "record" in signature.parameters:
        kwargs["record"] = True
    if "path" in signature.parameters:
        kwargs["path"] = GEMS_LOG
    if "unused" in signature.parameters:
        kwargs["unused"] = UNUSED_OPS
    elif "exclude" in signature.parameters:
        kwargs["exclude"] = UNUSED_OPS
    return flag_gems.use_gems(**kwargs)


def print_device_info(torch, model, inputs, device):
    first_param_device = next(model.parameters()).device
    input_devices = {
        name: str(value.device)
        for name, value in inputs.items()
        if torch.is_tensor(value)
    }
    print("target_device:", device, flush=True)
    print("first_param_device:", first_param_device, flush=True)
    print("input_devices:", input_devices, flush=True)


def generate_once(model, tokenizer, inputs, input_len):
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    )


def run_stage(
    torch,
    AutoModelForCausalLM,
    tokenizer,
    settings,
    stage_name,
    device,
    dtype_name,
    gems_ctx,
):
    log(f"{stage_name}: loading model")
    model = load_model(
        torch,
        AutoModelForCausalLM,
        settings.model_dir,
        device,
        dtype_name,
    )
    log(f"{stage_name}: model loaded")

    inputs = build_inputs(tokenizer, settings, device)
    input_len = inputs["input_ids"].shape[-1]
    print(f"\n[{stage_name}]", flush=True)
    print("dtype:", dtype_name, flush=True)
    print("input_tokens:", input_len, flush=True)
    print_device_info(torch, model, inputs, device)

    answer = ""
    with torch.inference_mode(), gems_ctx:
        for _ in range(WARMUP_RUNS):
            log(f"{stage_name}: warmup generate start")
            generate_once(model, tokenizer, inputs, input_len)
            sync_device(torch, device)
            log(f"{stage_name}: warmup generate done")

        for _ in range(INFER_RUNS):
            log(f"{stage_name}: inference generate start")
            answer = generate_once(model, tokenizer, inputs, input_len)
            sync_device(torch, device)
            log(f"{stage_name}: inference generate done")

    del model
    del inputs
    return answer


def main(argv=None):
    args = build_parser().parse_args(argv)

    import gc
    import flag_gems
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = resolve_model_settings(args.model, args.prompt)
    cpu_device = torch.device(CPU_DEVICE)
    gems_device = torch.device(flag_gems.device)

    print("model:", settings.display_name, flush=True)
    print("model_dir:", settings.model_dir, flush=True)
    print("prompt:", settings.prompt, flush=True)
    print("stages:", DEMO_STAGES, flush=True)
    # print("unused_ops:", UNUSED_OPS, flush=True)
    # print("max_new_tokens:", MAX_NEW_TOKENS, flush=True)
    # print("do_sample:", DO_SAMPLE, flush=True)

    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_dir,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    cpu_answer = run_stage(
        torch,
        AutoModelForCausalLM,
        tokenizer,
        settings,
        stage_name="cpu",
        device=cpu_device,
        dtype_name=CPU_DTYPE,
        gems_ctx=nullcontext(),
    )
    gc.collect()

    gems_answer = run_stage(
        torch,
        AutoModelForCausalLM,
        tokenizer,
        settings,
        stage_name="flaggems",
        device=gems_device,
        dtype_name=GEMS_DTYPE,
        gems_ctx=flag_gems_context(flag_gems, USE_GEMS),
    )

    print("\n[cpu_output]", flush=True)
    print(cpu_answer, flush=True)
    print("\n[flaggems_output]", flush=True)
    print(gems_answer, flush=True)


if __name__ == "__main__":
    main()

