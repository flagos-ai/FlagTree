"""Compare 63 decode intervals after native vLLM prefill; no prefill timing."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
import time


class _ThermalMonitor:
    """Condition independent requests and reject hardware throttling."""

    def __init__(self, minimum_idle_seconds):
        if minimum_idle_seconds < .5:
            raise ValueError("Use at least 0.5 untimed idle seconds between requests")
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        self.handle = (pynvml.nvmlDeviceGetHandleByUUID(visible) if visible.startswith("GPU-")
                       else pynvml.nvmlDeviceGetHandleByIndex(int(visible)))
        self.minimum_idle_seconds = minimum_idle_seconds
        self.finished = 0.
        self.fields = (pynvml.NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN,
                       pynvml.NVML_FI_DEV_CLOCKS_EVENT_REASON_HW_THERM_SLOWDOWN)

    def info(self):
        return {"kind": "nvml-clock-event-counters/v1",
                "gpu_uuid": self.nvml.nvmlDeviceGetUUID(self.handle),
                "max_start_temperature_c": 65, "minimum_idle_seconds": self.minimum_idle_seconds}

    def read(self):
        n = self.nvml
        counters = n.nvmlDeviceGetFieldValues(self.handle, self.fields)
        if any(v.nvmlReturn != n.NVML_SUCCESS or v.valueType != n.NVML_VALUE_TYPE_UNSIGNED_LONG_LONG
               for v in counters):
            raise RuntimeError("NVML thermal event counters are unavailable")
        reasons = n.nvmlDeviceGetCurrentClocksEventReasons(self.handle)
        return {"temperature_c": n.nvmlDeviceGetTemperature(self.handle, n.NVML_TEMPERATURE_GPU),
                "sm_clock_mhz": n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_SM),
                "thermal_active": bool(reasons & (n.nvmlClocksEventReasonSwThermalSlowdown
                                                   | n.nvmlClocksEventReasonHwThermalSlowdown)),
                "thermal_slowdown_us": sum(v.value.ullVal for v in counters)}

    def before(self):
        time.sleep(max(0., self.minimum_idle_seconds - (perf_counter() - self.finished)))
        deadline = perf_counter() + 60.
        while True:
            value = self.read()
            if not value["thermal_active"] and value["temperature_c"] <= 65:
                return value
            if perf_counter() >= deadline:
                raise RuntimeError("GPU did not cool below 65 C before the request")
            time.sleep(.5)

    def after(self, before):
        value = self.read()
        self.finished = perf_counter()
        if value["thermal_active"] or value["thermal_slowdown_us"] != before["thermal_slowdown_us"]:
            raise RuntimeError(f"Thermal throttling invalidated this campaign: {before} -> {value}")
        return value

    def close(self):
        self.nvml.nvmlShutdown()


def run(args):
    if args.decode_tokens != 64 or args.eager or args.eager_reference:
        raise ValueError("Native-prefill comparison requires 64 output tokens and captured decode")
    import torch
    from transformers import AutoTokenizer
    from benchmark_requests import PROMPT
    torch.set_num_threads(1)
    reference = json.loads(args.native_prefill_reference.read_text())
    if reference.get("schema") != "flagmega.prepared-decode-reference/v1":
        raise ValueError("Expected independent native prefill snapshots")
    records = {r["prefix_length"]: r for r in reference["records"]}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    source = tokenizer.encode(PROMPT, add_special_tokens=False)
    prompts = {n: (source * ((n + len(source) - 1) // len(source)))[:n] for n in args.prompt_lengths}
    for n, prompt in prompts.items():
        if prompt != records[n]["prompt_token_ids"] or len(records[n]["token_ids"]) < 63:
            raise ValueError("Native prefill prompt or decode reference differs")
    if args.runtime == "chat_cli":
        from native_prefill_backend import NativePrefillBackend
        from triton.flagmega.serving import TextGenerationEngine
        from triton.flagmega.serving.chat import TextStream
        backend = NativePrefillBackend(args.artifact, args.native_prefill_reference, args.checkpoint)
        engine = TextGenerationEngine(backend)
        def request(n):
            engine.reset()
            stream = TextStream(tokenizer, lambda text: None)
            result = engine.generate(prompts[n], max_new_tokens=64, ignore_eos=True, on_token=stream.put)
            stream.finish()
            value = result.to_dict()
            return {"prompt_tokens": prompts[n], "token_ids": value["token_ids"],
                    "decode_step_ms": value["decode_step_ms"]}
        metadata = {"vllm": None, "backend": backend.info(),
            "boundary": "native-prefilled standalone generation, sampling, host delivery and text decoding"}
        close = backend.close
    else:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import vllm
        from vllm import EngineArgs, SamplingParams
        from vllm.v1.engine.llm_engine import LLMEngine
        options = {}
        if args.artifact:
            from vllm import ModelRegistry
            from vllm_adapter import FlagMegaModel
            ModelRegistry.register_model("FlagMegaQwen35ForCausalLM", FlagMegaModel)
            os.environ["FLAGMEGA_REQUEST_ARTIFACT"] = str(args.artifact.resolve())
            os.environ["FLAGMEGA_NATIVE_PREFILL_REFERENCE"] = str(args.native_prefill_reference.resolve())
            os.environ["FLAGMEGA_NATIVE_PREFILL_CHECKPOINT"] = str(args.checkpoint.resolve())
            def overrides(config):
                config.architectures = ["FlagMegaQwen35ForCausalLM"]
                if hasattr(config, "text_config"):
                    config.text_config.architectures = config.architectures
                return config
            options.update(hf_overrides=overrides, enforce_eager=True)
        engine = LLMEngine.from_engine_args(EngineArgs(model=str(args.checkpoint.resolve()), dtype="bfloat16",
            tensor_parallel_size=1, max_model_len=3072, max_num_seqs=1, max_num_batched_tokens=3072,
            block_size=256, num_gpu_blocks_override=16, gpu_memory_utilization=.95,
            enable_prefix_caching=False, enable_chunked_prefill=False, language_model_only=True,
            mamba_ssm_cache_dtype="float32", disable_log_stats=True, seed=0,
            compilation_config={"mode": 3, "custom_ops": ["none"], "cudagraph_capture_sizes": [1]}, **options))
        backend = (engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model.backend
                   if args.artifact else None)
        counter = 0
        def request(n):
            nonlocal counter
            counter += 1
            seed = records[n]["initial_token"]
            if backend is not None:
                backend.restore(n)
            prompt = [seed] if backend is not None else prompts[n]
            count = 63 if backend is not None else 64
            engine.add_request(str(counter), {"prompt_token_ids": prompt},
                               SamplingParams(temperature=0, max_tokens=count, ignore_eos=True))
            torch.cuda.synchronize()
            previous = perf_counter()
            steps, output = [], None
            while engine.has_unfinished_requests():
                results = engine.step()
                now = perf_counter()
                if results:
                    if len(results) != 1:
                        raise RuntimeError("Expected batch and concurrency one")
                    output = results[0].outputs[0]
                    steps.append((now - previous) * 1000)
                    previous = now
            if output is None or len(output.token_ids) != count or len(steps) != count:
                raise RuntimeError("Expected one independently generated token per decode step")
            tokens = [seed, *output.token_ids] if backend is not None else list(output.token_ids)
            if tokens[0] != seed:
                raise RuntimeError("Native prefill boundary token changed")
            return {"prompt_tokens": prompts[n], "token_ids": tokens,
                    "decode_step_ms": steps if backend is not None else steps[1:]}
        metadata = {"vllm": vllm.__version__, "backend": backend.info() if backend is not None else None,
            "boundary": "native-prefilled synchronous engine.step through scheduler, forward, sampler and output delivery",
            "engine_config": str(engine.vllm_config)}
        close = engine.engine_core.shutdown
    try:
        import triton
        monitor = _ThermalMonitor(args.inter_request_pause)
        execute_request = request
        def request(n):
            before = monitor.before()
            value = execute_request(n)
            torch.cuda.synchronize()
            return {**value, "thermal_before": before, "thermal_after": monitor.after(before)}
        if args.runtime == "chat_cli" and any(n == "vllm" or n.startswith("vllm.") for n in sys.modules):
            raise RuntimeError("Standalone chat execution imported vLLM")
        common = {"schema": "flagmega.native-prefill-decode-trial/v1", "label": args.label,
            "runtime": args.runtime, "batch": 1, "concurrency": 1, "prefix_caching": False,
            "prefill": "native_vllm", "prefill_timed": False, "decode_intervals": 63,
            "checkpoint_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
            "checkpoint_config_sha256": hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
            "native_reference_sha256": hashlib.sha256(args.native_prefill_reference.read_bytes()).hexdigest(),
            "device": torch.cuda.get_device_name(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__, "triton": triton.__version__, "performance_valid": True,
            "cuda_graph": True, "vllm_imported": args.runtime == "vllm", "warmups_per_scenario": args.warmups,
            "thermal_monitor": monitor.info(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **metadata}
        for offset in range(args.rounds):
            round_id = args.round + offset
            scenarios = []
            rotation = offset % len(args.prompt_lengths)
            for n in args.prompt_lengths[rotation:] + args.prompt_lengths[:rotation]:
                for _ in range(args.warmups):
                    request(n)
                runs = [request(n) for _ in range(args.repeats)]
                if any(r["token_ids"] != runs[0]["token_ids"] for r in runs):
                    raise RuntimeError("Repeated continuations changed token sequences")
                wanted = [records[n]["initial_token"], *records[n]["token_ids"][:63]]
                if any(r["token_ids"][:4] != wanted[:4] for r in runs):
                    raise RuntimeError(f"Independent three-token decode acceptance failed at prompt {n}")
                if any(len(r["decode_step_ms"]) != 63 or min(r["decode_step_ms"]) <= 0 for r in runs):
                    raise RuntimeError("Decode timing must contain 63 positive host intervals")
                scenarios.append({"prompt_length": n, "decode_tokens": 64, "runs": runs,
                    "native_token_ids": wanted, "initial_state_roundtrip_exact": backend is None or backend.state_roundtrip_exact[n]})
                print(json.dumps({"label": args.label, "round": round_id, "prompt_length": n,
                    "median_ms": statistics.median(x for r in runs for x in r["decode_step_ms"]),
                    "native_token_agreement": sum(a == b for a, b in zip(runs[0]["token_ids"], wanted))}), flush=True)
            output = args.output if args.rounds == 1 else args.output.with_name(f"round{round_id:02d}.{args.label}.json")
            if output.exists():
                raise ValueError("Use a fresh immutable trial path")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({**common, "round": round_id,
                "scenarios": sorted(scenarios, key=lambda s: s["prompt_length"])}, indent=2) + "\n")
    finally:
        close()
        if "monitor" in locals():
            monitor.close()
