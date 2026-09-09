"""Prepared-state decode graph timing, separate from serving latency."""

import statistics

import torch


def summarize(samples):
    ordered = sorted(samples)
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * .95))], "count": len(samples)}


def benchmark(runtime, arguments, values, initial, expected, token_buffer, *, repeats):
    """Measure model+greedy+device feedback with real evolving private state.

    Initial-state restoration, token trace copies, compilation, warmup and
    CPU reads are outside timed intervals. The graph includes scheduler-field
    preparation and output-to-input feedback copies. No reference executes.
    """
    if repeats < 1:
        raise ValueError("至少一次正式重复")
    steps = len(expected)

    def reset():
        for name, tensor in initial.items():
            values[name].copy_(tensor)

    def launch():
        values["paged_state.slot_mapping"].copy_(values["paged_state.seq_lens"])
        runtime.run_into(*arguments)
        values["input_ids"].copy_(values[token_buffer])

    reset()
    runtime.prepare(*arguments)
    launch()
    torch.cuda.synchronize()
    reset()
    # Runtime synchronization scratch is stream-owned; use a fresh binding
    # for capture after eager warmup, following the public runtime contract.
    runtime.prepare(*arguments)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    tokens = torch.empty((steps, *values[token_buffer].shape), dtype=values[token_buffer].dtype, device="cuda:0")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    rounds = []
    for repeat in range(repeats + 2):
        reset()
        torch.cuda.synchronize()
        for step in range(steps):
            starts[step].record()
            graph.replay()
            ends[step].record()
            # Diagnostic trace outside the model+feedback event interval.
            tokens[step].copy_(values[token_buffer])
        torch.cuda.synchronize()
        generated = tokens.flatten().tolist()
        if generated != expected:
            raise RuntimeError(f"CUDA graph 独立序列不一致：repeat={repeat}, actual={generated}, expected={expected}")
        initial_length = int(initial["paged_state.seq_lens"].item())
        if int(values["paged_state.seq_lens"].item()) != initial_length + steps:
            raise RuntimeError("CUDA graph 没有逐步推进独立状态")
        if repeat >= 2:
            samples = [a.elapsed_time(b) for a, b in zip(starts, ends)]
            rounds.append({"samples_ms": samples, "token_ids": generated, **summarize(samples)})
    samples = [value for run in rounds for value in run["samples_ms"]]
    return {"rounds": rounds, **summarize(samples), "tokens_per_second": 1000 / statistics.mean(samples),
            "warmup_rounds": 2, "all_sequences_match": True,
            "boundary": "GPU CUDA graph: full decoder + logits + greedy sampling + device token feedback",
            "excluded": "prefix/state restore, token trace copy, host read, compilation, loading, warmup",
            "cache_policy": "natural sequential decode; no synthetic flush", "serving_latency": False}
