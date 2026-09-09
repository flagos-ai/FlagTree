"""Measure native vLLM prepared decode, not engine/serving request latency."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("需要正数 repeats")
    args.trial.mkdir(parents=True, exist_ok=False)
    local = Path(__file__).resolve().parent
    sources = [Path(__file__), local / "native_prepared_graph.py", local / "prepare_reference.py"]
    identities = {}
    for source in sources:
        (args.trial / source.name).write_bytes(source.read_bytes())
        identities[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    reference = json.loads(args.reference.read_text())
    if reference["schema"] != "flagmega.prepared-decode-reference/v1":
        raise ValueError("需要已验收的四场景原生参考")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    import vllm
    from vllm import EngineArgs, SamplingParams
    from vllm.forward_context import get_forward_context
    from vllm.v1.engine.llm_engine import LLMEngine
    from native_prepared_graph import PreparedNativeTrace
    from prepare_reference import snapshot

    if vllm.__version__ != "0.23.1rc1.dev1294+gae10e855a":
        raise RuntimeError("固定原生版本不一致")
    torch.set_num_threads(1)
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=str(args.checkpoint.resolve()), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=512, max_num_seqs=1, max_num_batched_tokens=512,
        block_size=256, num_gpu_blocks_override=16, gpu_memory_utilization=.95,
        enable_prefix_caching=False, enable_chunked_prefill=True, language_model_only=True,
        mamba_ssm_cache_dtype="float32", disable_log_stats=True, seed=0,
        compilation_config={"mode": 3, "custom_ops": ["none"], "cudagraph_capture_sizes": [1]},
    ))
    runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    execute, forward = runner.execute_model, runner._model_forward
    current = {}

    def observe_execute(output, *positional, **keywords):
        if output.total_num_scheduled_tokens:
            if len(output.num_scheduled_tokens) != 1:
                raise RuntimeError("只允许独立 bs1")
            current["forward_count"] += 1
            expected = current["target"]["prefix_length"] if current["forward_count"] == 1 else 1
            if output.total_num_scheduled_tokens != expected:
                raise RuntimeError("prefix / decode 调度边界不一致")
        return execute(output, *positional, **keywords)

    def observe_forward(*positional, **keywords):
        if current["forward_count"] > 1:
            if not current["trace"].records:
                target = current["target"]
                path = args.reference.parent / target["state"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != target["state_sha256"]:
                    raise RuntimeError("参考初始状态 hash 错误")
                expected = torch.load(path, map_location="cpu", weights_only=True)["layers"]
                actual, _ = snapshot(runner, current["request_id"], target["prefix_length"], torch)
                if actual.keys() != expected.keys():
                    raise RuntimeError("不包含完整的同一组 40 层状态")
                for layer, state in actual.items():
                    for name, value in state.items():
                        if name != "kind" and not torch.equal(value, expected[layer][name]):
                            raise RuntimeError(f"原生初始状态与 FlagMega 参考输入不同: {layer}, {name}")
                current["initial_exact"] = True
            current["trace"].observe(get_forward_context(), keywords)
        return forward(*positional, **keywords)

    runner.execute_model, runner._model_forward = observe_execute, observe_forward
    scenarios = []
    try:
        for index, target in enumerate(reference["records"]):
            current.clear()
            current.update(forward_count=0, request_id=f"prepared-{index}", target=target,
                           trace=PreparedNativeTrace(runner, target))
            engine.add_request(current["request_id"], {"prompt_token_ids": target["prompt_token_ids"]},
                               SamplingParams(temperature=0, max_tokens=len(target["token_ids"]) + 1,
                                              ignore_eos=True))
            result = None
            while engine.has_unfinished_requests():
                outputs = engine.step()
                if outputs:
                    if len(outputs) != 1:
                        raise RuntimeError("输出不是 bs1")
                    result = outputs[0].outputs[0]
            expected = [target["initial_token"], *target["token_ids"]]
            if result is None or list(result.token_ids) != expected:
                raise RuntimeError("原生去掉 logprob 诊断后完整独立序列不一致")
            print("原生初始状态/完整独立序列一致", target["prefix_length"], flush=True)
            trace = current["trace"]
            torch.save([row["metadata"] for row in trace.records],
                       args.trial / f"context-{target['prefix_length']}-schedule.pt")
            timing = trace.benchmark(repeats=args.repeats)
            scenarios.append({**timing, "initial_state_exact": current["initial_exact"],
                              "initial_state_sha256": target["state_sha256"]})
            report = {"schema": "flagmega.native-prepared-decode/v1", "scenarios": scenarios,
                      "complete": len(scenarios) == len(reference["records"]), "performance_valid": True,
                      "variant": "native_vllm_prepared_graph", "source_sha256": identities,
                      "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                      "vllm": vllm.__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(),
                      "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "engine_config": str(engine.vllm_config),
                      "serving_latency": False, "logprobs_requested": False}
            (args.trial / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("原生 prepared", target["prefix_length"], timing["median_ms"], "ms", flush=True)
    finally:
        runner.execute_model, runner._model_forward = execute, forward
        engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
