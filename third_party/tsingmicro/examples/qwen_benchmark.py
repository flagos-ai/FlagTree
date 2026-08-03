#!/usr/bin/env python3
"""
Qwen + FlagGems benchmark 使用流程

0. 环境准备

    export PYTHONPATH=/your/path/workspace/FlagGems/src:$PYTHONPATH   #准备flaggems环境
    bash third_party/tsingmicro/scripts/run_tsingmicro.sh  python3 third_party/tsingmicro/examples/bench_qwen25_flaggems.py --warmup 1 --repeat 3 --max-new-tokens 64

1. 支持的模型

   通过 --model 选择模型：

       qwen25  -> ./models/Qwen2.5-0.5B-Instruct
       qwen3   -> ./models/Qwen3-0.6B

   也可以用 --model-dir 覆盖实际路径：
       bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen3 --model-dir /path/to/Qwen3-0.6B

2. 基线测试，不启用 FlagGems：
       bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen25 --warmup 1 --repeat 3
       bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen3  --warmup 1 --repeat 3

3. FlagGems 测试：

       bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen25 --use-gems
       bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen3  --use-gems

   使用--unused-ops跳过算子（不跑该算子的flaggem版本）

        bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen3 --use-gems --unused-ops "cat"

   如果想完全不跳过任何 FlagGems 算子：

       bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen3 --use-gems --unused-ops ""

4. Qwen3 thinking 模式

   Qwen3 默认关闭 thinking，便于 smoke test 和性能对比：

       enable_thinking=False

   如需打开：

       bash ../scripts/run_tsingmicro.sh python3 bench_qwen25_flaggems.py --model qwen3 --qwen3-enable-thinking

5. 主要输出

   [prefill forward]：prompt prefill 的 forward 耗时
   [end-to-end generate]：完整 generate 耗时和 tokens/s

   首次运行包含 Triton JIT 编译，不要作为正式性能数据。
"""

import argparse
import inspect
import statistics
import time
from contextlib import nullcontext


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
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def canonical_model_key(model_name):
    normalized = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if normalized in config["aliases"]:
            return key
    valid = ", ".join(sorted(alias for cfg in MODEL_CONFIGS.values() for alias in cfg["aliases"]))
    raise ValueError(f"Unsupported model '{model_name}'. Valid values: {valid}")


def resolve_model_settings(model_name, model_dir, prompt, qwen3_enable_thinking):
    key = canonical_model_key(model_name)
    config = MODEL_CONFIGS[key]
    chat_template_kwargs = dict(config["chat_template_kwargs"])

    if key == "qwen3":
        chat_template_kwargs["enable_thinking"] = bool(qwen3_enable_thinking)

    return ModelSettings(
        key=key,
        display_name=config["display_name"],
        model_dir=model_dir or config["model_dir"],
        prompt=prompt if prompt is not None else config["default_prompt"],
        chat_template_kwargs=chat_template_kwargs,
    )


def parse_unused_ops(text):
    if text is None:
        return ["index"]
    text = text.strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def get_dtype(torch, name):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def sync_device(torch, device):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return

    backend = getattr(torch, str(device).split(":")[0], None)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize()


def percentile(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = int((len(xs) - 1) * p / 100)
    return xs[idx]


def timed(torch, device, fn):
    sync_device(torch, device)
    t0 = time.perf_counter()
    out = fn()
    sync_device(torch, device)
    return out, time.perf_counter() - t0


def load_model(torch, AutoModelForCausalLM, model_dir, dtype, device, attn_implementation):
    kwargs = {
        "dtype": dtype,
        "device_map": None,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

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
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return prompt


def build_inputs(tokenizer, settings, device):
    text = apply_chat_template(
        tokenizer,
        settings.prompt,
        settings.chat_template_kwargs,
    )
    return tokenizer([text], return_tensors="pt").to(device)


def assert_model_on_device(torch, model, inputs, device):
    first_param_device = next(model.parameters()).device
    input_devices = {
        name: str(value.device)
        for name, value in inputs.items()
        if torch.is_tensor(value)
    }
    print("target_device:", device, flush=True)
    print("first_param_device:", first_param_device, flush=True)
    print("input_devices:", input_devices, flush=True)


def flag_gems_context(flag_gems, unused_ops, record, path):
    signature = inspect.signature(flag_gems.use_gems)
    kwargs = {
        "record": record,
        "path": path,
    }
    if "unused" in signature.parameters:
        kwargs["unused"] = unused_ops
    elif "exclude" in signature.parameters:
        kwargs["exclude"] = unused_ops
    return flag_gems.use_gems(**kwargs)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="qwen25",
        help="Model alias: qwen25 or qwen3. Aliases like qwen3-0.6b are also accepted.",
    )
    parser.add_argument("--model-dir", default=None, help="Override model directory.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--attn", default="eager")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--use-gems", action="store_true")
    parser.add_argument("--qwen3-enable-thinking", action="store_true")
    parser.add_argument(
        "--unused-ops",
        default="index",
        help='Comma-separated FlagGems ops to skip. Use "" to skip none.',
    )
    parser.add_argument("--gems-log", default="./bench_flaggems.log")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    import flag_gems
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = resolve_model_settings(
        model_name=args.model,
        model_dir=args.model_dir,
        prompt=args.prompt,
        qwen3_enable_thinking=args.qwen3_enable_thinking,
    )
    unused_ops = parse_unused_ops(args.unused_ops)

    dtype = get_dtype(torch, args.dtype)
    device = torch.device(flag_gems.device)

    log("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, trust_remote_code=False)

    log("loading model")
    model = load_model(
        torch,
        AutoModelForCausalLM,
        settings.model_dir,
        dtype,
        device,
        args.attn,
    )
    log("model loaded")

    inputs = build_inputs(tokenizer, settings, device)
    input_len = inputs["input_ids"].shape[-1]

    print("torch:", torch.__version__, flush=True)
    print("flag_gems:", getattr(flag_gems, "__version__", "unknown"), flush=True)
    print("flag_gems.device:", flag_gems.device, flush=True)
    print("model:", settings.display_name, flush=True)
    print("model_key:", settings.key, flush=True)
    print("model_dir:", settings.model_dir, flush=True)
    print("chat_template_kwargs:", settings.chat_template_kwargs, flush=True)
    print("use_gems:", args.use_gems, flush=True)
    print("unused_ops:", unused_ops, flush=True)
    print("input_tokens:", input_len, flush=True)
    assert_model_on_device(torch, model, inputs, device)

    gems_ctx = (
        flag_gems_context(
            flag_gems,
            unused_ops=unused_ops,
            record=True,
            path=args.gems_log,
        )
        if args.use_gems
        else nullcontext()
    )

    with torch.inference_mode(), gems_ctx:
        log("start prefill warmup")
        for i in range(args.warmup):
            _, sec = timed(torch, device, lambda: model(**inputs, use_cache=False))
            log(f"prefill warmup {i + 1}/{args.warmup}: {sec * 1000:.3f} ms")
        log("finish prefill warmup")

        log("start prefill benchmark")
        prefill_times = []
        for i in range(args.repeat):
            _, sec = timed(torch, device, lambda: model(**inputs, use_cache=False))
            prefill_times.append(sec)
            log(f"prefill repeat {i + 1}/{args.repeat}: {sec * 1000:.3f} ms")
        log("finish prefill benchmark")

        print("\n[prefill forward]", flush=True)
        print("avg_ms:", statistics.mean(prefill_times) * 1000, flush=True)
        print("p50_ms:", percentile(prefill_times, 50) * 1000, flush=True)
        print("p90_ms:", percentile(prefill_times, 90) * 1000, flush=True)

        def run_generate():
            return model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        log("start generate warmup")
        for i in range(args.warmup):
            output_ids, sec = timed(torch, device, run_generate)
            new_tokens = output_ids.shape[-1] - input_len
            log(
                f"generate warmup {i + 1}/{args.warmup}: "
                f"{sec:.3f} sec, new_tokens={new_tokens}, "
                f"tokens/s={new_tokens / sec:.3f}"
            )
        log("finish generate warmup")

        log("start generate benchmark")
        gen_times = []
        gen_tokens = []
        last_output_ids = None

        for i in range(args.repeat):
            output_ids, sec = timed(torch, device, run_generate)
            new_tokens = output_ids.shape[-1] - input_len
            gen_times.append(sec)
            gen_tokens.append(new_tokens)
            last_output_ids = output_ids
            log(
                f"generate repeat {i + 1}/{args.repeat}: "
                f"{sec:.3f} sec, new_tokens={new_tokens}, "
                f"tokens/s={new_tokens / sec:.3f}"
            )

        log("finish generate benchmark")

        avg_gen_sec = statistics.mean(gen_times)
        avg_new_tokens = statistics.mean(gen_tokens)

        print("\n[end-to-end generate]", flush=True)
        print("avg_sec:", avg_gen_sec, flush=True)
        print("p50_sec:", percentile(gen_times, 50), flush=True)
        print("p90_sec:", percentile(gen_times, 90), flush=True)
        print("avg_new_tokens:", avg_new_tokens, flush=True)
        print("tokens_per_sec_e2e:", avg_new_tokens / avg_gen_sec, flush=True)

        answer = tokenizer.decode(
            last_output_ids[0][input_len:],
            skip_special_tokens=True,
        )
        print("\n[last_output]", flush=True)
        print(answer, flush=True)

    log("done")


if __name__ == "__main__":
    main()
