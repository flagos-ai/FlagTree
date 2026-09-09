"""Prepare prefix snapshots and independent greedy references in native vLLM.

Run in the isolated, pinned vLLM environment, not the FlagTree environment.
Prefix processing is outside the decode workload and is never timed here.
No native kernels are replaced and no FlagMega code is loaded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from time import perf_counter


def snapshot(runner, request_id, prefix_length, torch):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    context = runner.vllm_config.compilation_config.static_forward_context
    request_indexes = runner.input_batch.req_id_to_index
    if len(request_indexes) != 1:
        raise RuntimeError(f"快照只接受一个独立活动请求：{request_indexes}")
    row = next(iter(request_indexes.values()))
    layers, summary = {}, []
    for group_index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        table = runner.input_batch.block_table[group_index]
        block_ids = table.get_cpu_tensor()[row, :table.num_blocks_per_row[row]].tolist()
        if not block_ids or min(block_ids) < 0:
            raise RuntimeError(f"请求没有有效的非空 cache block：{block_ids}")
        for name in group.layer_names:
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match is None:
                raise RuntimeError(f"无法确定原生 layer 序号：{name}")
            index = int(match.group(1))
            if index in layers:
                raise RuntimeError(f"重复的原生 layer 状态：{index}")
            layer = context[name]
            cache = layer.kv_cache
            if name.endswith(".linear_attn"):
                if block_ids[0] == 0:
                    raise RuntimeError("GDN 请求不能引用 NULL_BLOCK_ID=0")
                if not layer.enable_packed_recurrent_decode:
                    raise RuntimeError("原生 GDN 没有启用约定的 packed recurrent decode")
                convolution = cache[0] if is_conv_state_dim_first() else cache[0].transpose(-1, -2)
                # Native [head, value, key] -> logical [head, key, value].
                item = {"kind": "linear_attention", "convolution": convolution[block_ids[0]].cpu().clone(),
                        "recurrent": cache[1][block_ids[0]].transpose(-1, -2).cpu().contiguous()}
                details = {"conv_shape": list(cache[0].shape), "ssm_shape": list(cache[1].shape)}
            else:
                # The pinned BF16 attention ABI is logical [block, head,
                # token, 2*dim], even when its actual strides are NHD.
                head_dim, heads = layer.head_size, layer.num_kv_heads
                if cache.ndim != 4 or cache.shape[1] != heads or cache.shape[-1] != 2 * head_dim:
                    raise RuntimeError(f"未审核的原生 KV ABI：{name}: {cache.shape}, {cache.stride()}")
                block_size = cache.shape[2]
                required = (prefix_length + block_size - 1) // block_size
                if required > len(block_ids):
                    raise RuntimeError("原生 block table 不覆盖完整 prefix")
                pages = cache[torch.tensor(block_ids[:required], device=cache.device)]
                logical = pages.permute(0, 2, 1, 3).reshape(-1, heads, 2 * head_dim)[:prefix_length].cpu()
                item = {"kind": "full_attention", "key": logical[..., :head_dim].contiguous(),
                        "value": logical[..., head_dim:].contiguous()}
                details = {"kv_shape": list(cache.shape), "kv_strides": list(cache.stride()),
                           "backend": type(layer.impl).__name__}
            layers[index] = item
            summary.append({"layer": index, "name": name, "group": group_index, "blocks": block_ids,
                            "kind": item["kind"], **details})
    if set(layers) != set(range(40)):
        raise RuntimeError(f"快照必须包含完整 40 层：{sorted(layers)}")
    return layers, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--contexts", type=int, nargs="+", default=(1, 32, 255, 256))
    args = parser.parse_args()
    if args.steps <= 0 or any(n <= 0 or n + args.steps > 512 for n in args.contexts):
        raise ValueError("上下文和生成长度必须是正数，总长度不能超过配置的 512")
    args.trial.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    import vllm
    from vllm import EngineArgs, SamplingParams
    from vllm.v1.engine.llm_engine import LLMEngine

    if vllm.__version__ != "0.23.1rc1.dev1294+gae10e855a":
        raise RuntimeError(f"参考版本不符合合同：{vllm.__version__}")
    torch.set_num_threads(1)
    started = perf_counter()
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(args.checkpoint.resolve()), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=512, max_num_seqs=1, max_num_batched_tokens=512,
        block_size=256, num_gpu_blocks_override=16, gpu_memory_utilization=.95,
        enable_prefix_caching=False, enable_chunked_prefill=True, language_model_only=True,
        mamba_ssm_cache_dtype="float32", disable_log_stats=True, seed=0,
        compilation_config={"mode": 3, "custom_ops": ["none"], "cudagraph_capture_sizes": [1]},
    ))
    print("原生引擎初始化秒", perf_counter() - started, flush=True)
    runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    # Async scheduling can have a decode in flight before Engine.step returns
    # the prefix token. Snapshot directly after the *original* prefix forward,
    # before the worker can submit a subsequent forward. No backend is changed.
    original_execute = runner.execute_model
    capture = {}

    def execute_and_snapshot(scheduler_output, *positional, **keywords):
        output = original_execute(scheduler_output, *positional, **keywords)
        if capture and scheduler_output.total_num_scheduled_tokens:
            if len(scheduler_output.num_scheduled_tokens) != 1:
                raise RuntimeError("诊断快照要求原生 bs=1")
            if "initial_layers" not in capture:
                if scheduler_output.total_num_scheduled_tokens != capture["length"]:
                    raise RuntimeError("prefix 必须在一个准备 forward 中完成")
                torch.cuda.synchronize()
                capture["initial_layers"], capture["layouts"] = snapshot(
                    runner, capture["request_id"], capture["length"], torch)
            elif "first_decode_layers" not in capture:
                if scheduler_output.total_num_scheduled_tokens != 1:
                    raise RuntimeError("测量范围内只允许单 token decode")
                torch.cuda.synchronize()
                capture["first_decode_layers"], _ = snapshot(
                    runner, capture["request_id"], capture["length"] + 1, torch)
        return output

    runner.execute_model = execute_and_snapshot
    text = engine.tokenizer.encode("Explain why the sky is blue, and how sunlight interacts with air. ",
                                   add_special_tokens=False)
    records = []
    try:
        for request, length in enumerate(args.contexts):
            request_id = f"decode-{request}"
            capture.clear()
            capture.update(request_id=request_id, length=length)
            prompt = [1] if length == 1 else (text * ((length + len(text) - 1) // len(text)))[:length]
            engine.add_request(request_id, {"prompt_token_ids": prompt},
                               SamplingParams(temperature=0, max_tokens=args.steps + 1, ignore_eos=True, logprobs=20))
            first = engine.step()
            while not first and engine.has_unfinished_requests():
                first = engine.step()
            if len(first) != 1 or len(first[0].outputs[0].token_ids) != 1:
                raise RuntimeError("准备阶段必须一次完成 prefix，并产生恰好一个边界输入 token")
            initial_token = int(first[0].outputs[0].token_ids[0])
            layers, summary = capture["initial_layers"], capture["layouts"]
            state_path = args.trial / f"context-{length}-initial.pt"
            torch.save({"schema": "flagmega.qwen35-prepared-decode-state/v1", "prefix_length": length,
                        "initial_token": initial_token, "layers": layers}, state_path)
            del layers
            print("准备完成", length, "initial_token", initial_token, flush=True)
            output = first[0].outputs[0]
            while engine.has_unfinished_requests():
                result = engine.step()
                if result:
                    if len(result) != 1:
                        raise RuntimeError("decode 必须维持 bs=1")
                    output = result[0].outputs[0]
                    print("原生 decode", length, list(output.token_ids)[1:], flush=True)
            tokens = list(output.token_ids)
            if len(tokens) != args.steps + 1 or tokens[0] != initial_token:
                raise RuntimeError("原生没有生成完整的独立 decode 序列")
            torch.save(capture["first_decode_layers"], args.trial / f"context-{length}-after-first.pt")
            records.append({"prefix_length": length, "prompt_token_ids": prompt, "initial_token": initial_token,
                            "token_ids": tokens[1:], "state": state_path.name,
                            "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                            "state_layouts": summary,
                            "logprobs": [{str(k): {"logprob": v.logprob, "rank": v.rank} for k, v in row.items()}
                                         for row in output.logprobs[1:]]})
            report = {"schema": "flagmega.prepared-decode-reference/v1", "vllm": vllm.__version__,
                      "torch": torch.__version__, "engine_config": str(engine.vllm_config),
                      "device": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "performance_valid": False, "records": records,
                      "boundary": "prepared prefix state + initial token; all compared tokens independently generated"}
            (args.trial / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        runner.execute_model = original_execute
        engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
