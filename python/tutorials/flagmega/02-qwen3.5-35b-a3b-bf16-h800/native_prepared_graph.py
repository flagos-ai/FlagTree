"""Prepared native decode with original FULL graph, private state and token feedback."""

import copy
from dataclasses import fields, is_dataclass
import statistics

import torch


def tensor_key(value):
    return (str(value.device), value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype))


def gpu_tensors(value, path=""):
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            yield path, value
    elif isinstance(value, dict):
        for name, item in sorted(value.items()):
            yield from gpu_tensors(item, f"{path}.{name}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from gpu_tensors(item, f"{path}.{index}")
    elif is_dataclass(value):
        for field in fields(value):
            yield from gpu_tensors(getattr(value, field.name), f"{path}.{field.name}")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"未经审核的原生 metadata 字段: {path}: {type(value)}")


def state_storages(runner):
    """Byte views preserve packed backing aliases rather than copying overlapping logical views."""
    storages = {}
    context = runner.vllm_config.compilation_config.static_forward_context
    for group in runner.kv_cache_config.kv_cache_groups:
        for name in group.layer_names:
            cache = context[name].kv_cache
            for value in cache if isinstance(cache, (tuple, list)) else (cache,):
                storage = value.untyped_storage()
                key = (str(value.device), storage.data_ptr(), storage.nbytes())
                if key not in storages:
                    storages[key] = torch.empty(0, device=value.device, dtype=torch.uint8).set_(
                        storage, 0, (storage.nbytes(),), (1,))
    return tuple(storages.values())


def summarize(samples):
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "mean_ms": statistics.mean(samples),
            "p95_ms": sorted(samples)[min(len(samples) - 1, int(.95 * len(samples)))], "count": len(samples)}


class PreparedNativeTrace:
    def __init__(self, runner, target):
        self.runner, self.target = runner, target
        self.records = []
        self.bindings = {}
        self.aliases = {}
        self.initial_caches = None

    @torch.inference_mode()
    def observe(self, context, kwargs):
        from vllm.config import CUDAGraphMode
        if context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            raise RuntimeError("原生观测必须使用既有 FULL graph")
        if context.dp_metadata is not None or context.ubatch_slices is not None or context.skip_compiled:
            raise RuntimeError("仅实现 TP1、bs1、无 microbatch 的原生 prepared ABI")
        inputs = kwargs["input_ids"]
        if inputs.numel() != 1:
            raise RuntimeError("要求单 token decode")
        raw = {"attn": context.attn_metadata, "slots": context.slot_mapping,
               "padding": context.is_padding, "additional": context.additional_kwargs,
               "kwargs": {key: value for key, value in kwargs.items() if key != "input_ids"}}
        bindings = dict(gpu_tensors(raw))
        if not self.records:
            if int(inputs.item()) != self.target["initial_token"]:
                raise RuntimeError("原生初始 token 与约定 prefix 边界不一致")
            self.kwargs = dict(kwargs)
            self.input_initial = inputs.clone()
            self.caches = state_storages(self.runner)
            self.initial_caches = tuple(value.clone() for value in self.caches)
            self.sampling = copy.deepcopy(self.runner.input_batch.sampling_metadata)
            if (not self.sampling.all_greedy or not self.sampling.no_penalties
                    or self.sampling.max_num_logprobs is not None or self.sampling.allowed_token_ids_mask is not None
                    or self.sampling.bad_words_token_ids or self.sampling.logprob_token_ids):
                raise RuntimeError("计时合同只含无附加约束的 greedy，不含 logprob 诊断")
            unique = {}
            for path, tensor in bindings.items():
                key = tensor_key(tensor)
                canonical = unique.setdefault(key, path)
                self.aliases[path] = canonical
                self.bindings.setdefault(canonical, tensor)
        if bindings.keys() != self.aliases.keys():
            raise RuntimeError("原生 metadata surface 在 decode 中变化")
        for name, tensor in bindings.items():
            if tensor_key(tensor) != tensor_key(self.bindings[self.aliases[name]]):
                raise RuntimeError(f"原生 FULL graph 的 persistent metadata ABI 变化: {name}")
        if tensor_key(inputs) != tensor_key(self.kwargs["input_ids"]):
            raise RuntimeError("原生 input_ids buffer 地址变化")
        self.records.append({"context": context, "metadata": {
            name: tensor.detach().cpu().clone() for name, tensor in self.bindings.items()}})

    @torch.inference_mode()
    def benchmark(self, *, repeats):
        from vllm.compilation.counter import compilation_counter
        from vllm.forward_context import override_forward_context
        if len(self.records) != len(self.target["token_ids"]):
            raise RuntimeError("要求完整 decode 调度 trace")
        steps = len(self.records)
        original_graph_count = compilation_counter.num_cudagraph_captured
        dynamic = tuple(name for name in self.bindings
                        if any(not torch.equal(record["metadata"][name], self.records[0]["metadata"][name])
                               for record in self.records[1:]))
        schedules = [{name: record["metadata"][name].to(self.bindings[name].device) for name in dynamic}
                     for record in self.records]
        initial_metadata = {name: value.to(self.bindings[name].device)
                            for name, value in self.records[0]["metadata"].items()}
        final_caches = tuple(value.clone() for value in self.caches)
        final_metadata = {name: value.clone() for name, value in self.bindings.items()}
        final_input = self.kwargs["input_ids"].clone()
        indices = torch.zeros(1, dtype=torch.int64, device=self.kwargs["input_ids"].device)

        def reset():
            for dst, src in zip(self.caches, self.initial_caches):
                dst.copy_(src)
            for name, tensor in initial_metadata.items():
                self.bindings[name].copy_(tensor)
            self.kwargs["input_ids"].copy_(self.input_initial)

        def prepare_metadata(step):
            for name, value in schedules[step].items():
                self.bindings[name].copy_(value)

        def model_forward(step):
            with override_forward_context(self.records[step]["context"]):
                return self.runner.model(**self.kwargs)

        def sample_and_feedback(hidden):
            logits = self.runner.model.compute_logits(hidden[indices])
            sampled = self.runner.sampler(logits=logits, sampling_metadata=self.sampling).sampled_token_ids
            self.kwargs["input_ids"].copy_(sampled.reshape_as(self.kwargs["input_ids"]))

        rounds = []
        try:
            reset()
            prepare_metadata(0)
            hidden = model_forward(0)
            sample_and_feedback(hidden)
            torch.cuda.synchronize()
            reset()
            metadata_graphs = [torch.cuda.CUDAGraph() for _ in range(steps)]
            for step, graph in enumerate(metadata_graphs):
                with torch.cuda.graph(graph):
                    prepare_metadata(step)
            # PyTorch's graph replay prepares allocator/generator state and
            # cannot itself run inside capture. Keep the existing model graph
            # untouched and capture only LM head + native sampler + feedback.
            # The full three-graph GPU interval, including dispatch gaps, is timed.
            post_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(post_graph):
                sample_and_feedback(hidden)
            if compilation_counter.num_cudagraph_captured != original_graph_count:
                raise RuntimeError("观测不允许重捕获/替换原生 model graph")
            tokens = torch.empty((steps, 1), dtype=self.kwargs["input_ids"].dtype, device=indices.device)
            starts = [torch.cuda.Event(enable_timing=True) for _ in metadata_graphs]
            ends = [torch.cuda.Event(enable_timing=True) for _ in metadata_graphs]
            for repeat in range(repeats + 2):
                reset()
                torch.cuda.synchronize()
                for step, graph in enumerate(metadata_graphs):
                    starts[step].record()
                    graph.replay()
                    model_forward(step)
                    post_graph.replay()
                    ends[step].record()
                    tokens[step].copy_(self.kwargs["input_ids"])
                torch.cuda.synchronize()
                actual = tokens.flatten().tolist()
                if actual != self.target["token_ids"]:
                    raise RuntimeError(f"原生 prepared graph 独立完整序列失败: {repeat}: {actual}")
                if repeat >= 2:
                    values = [a.elapsed_time(b) for a, b in zip(starts, ends)]
                    rounds.append({"token_ids": actual, **summarize(values)})
            samples = [value for run in rounds for value in run["samples_ms"]]
            return {"rounds": rounds, **summarize(samples), "tokens_per_second": 1000 / statistics.mean(samples),
                    "all_sequences_match": True, "prefix_length": self.target["prefix_length"], "warmup_rounds": 2,
                    "metadata_bindings": list(self.bindings), "dynamic_metadata": list(dynamic),
                    "native_full_graph_reused": True, "state_pool_bytes": sum(value.numel() for value in self.caches),
                    "boundary": "GPU event envelope: prepared metadata graph + original native FULL model graph + LM head/sampler/feedback graph",
                    "excluded": "prefix, state restore, schedule preparation, token tracing, host reads, loading, warmup",
                    "cache_policy": "natural sequential decode; no synthetic flush", "serving_latency": False}
        finally:
            for dst, src in zip(self.caches, final_caches):
                dst.copy_(src)
            for name, tensor in final_metadata.items():
                self.bindings[name].copy_(tensor)
            self.kwargs["input_ids"].copy_(final_input)
            torch.cuda.synchronize()
