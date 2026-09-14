"""Prepared-state decode graph timing, separate from serving latency."""

import statistics
from time import perf_counter

import torch


def summarize(samples):
    ordered = sorted(samples)
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * .95))], "count": len(samples)}


def benchmark(runtime, arguments, values, initial, expected, token_buffer, *, repeats, steps=None,
              host_feedback=False, logits_buffer=None):
    """Measure model+greedy+device feedback with real evolving private state.

    Initial-state restoration, token trace copies, compilation, warmup and
    CPU reads are outside timed intervals. The graph includes scheduler-field
    preparation and output-to-input feedback copies. No reference executes.
    """
    if repeats < 1:
        raise ValueError("At least one measured repeat is required")
    prefix = list(expected)
    steps = len(expected) if steps is None else steps
    if not prefix or type(steps) is not int or steps < len(prefix):
        raise ValueError("Timed sequence cannot be shorter than the validated prefix")

    def reset():
        for name, tensor in initial.items():
            values[name].copy_(tensor)

    def launch():
        values["paged_state.slot_mapping"].copy_(values["paged_state.seq_lens"])
        runtime.run_into(*arguments)
        values["input_ids"].copy_(values[token_buffer])

    reset()
    runtime.prepare(*arguments)
    if steps > len(prefix):
        if logits_buffer is None:
            raise ValueError("Extended independent timing requires a logits buffer for finite-output checks")
        expected = []
        for _ in range(steps):
            launch()
            token = int(values[token_buffer].item())
            if not 0 <= token < values[logits_buffer].numel() or not bool(torch.isfinite(values[logits_buffer]).all().item()):
                raise RuntimeError("Extended independent decode produced non-finite logits")
            expected.append(token)
            if int(values["paged_state.seq_lens"].item()) != int(initial["paged_state.seq_lens"].item()) + len(expected):
                raise RuntimeError("Extended decode did not advance state exactly once per token")
        if expected[:len(prefix)] != prefix:
            raise RuntimeError("Extended independent decode changed the validated prefix")
        if int(values["paged_state.seq_lens"].item()) != int(initial["paged_state.seq_lens"].item()) + steps:
            raise RuntimeError("Extended decode did not advance state exactly once per token")
        reset()
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
            raise RuntimeError(f"CUDA graph independent sequence mismatch: repeat={repeat}, actual={generated}, expected={expected}")
        initial_length = int(initial["paged_state.seq_lens"].item())
        if int(values["paged_state.seq_lens"].item()) != initial_length + steps:
            raise RuntimeError("CUDA graph did not advance independent state on every step")
        if repeat >= 2:
            samples = [a.elapsed_time(b) for a, b in zip(starts, ends)]
            rounds.append({"samples_ms": samples, "token_ids": generated, **summarize(samples)})
    samples = [value for run in rounds for value in run["samples_ms"]]
    report = {"rounds": rounds, **summarize(samples), "tokens_per_second": 1000 / statistics.mean(samples),
            "warmup_rounds": 2, "all_sequences_match": True,
            "timed_steps": steps, "validated_prefix": prefix,
            "independent_eager_token_ids": list(expected),
            "timed_sequence_reference": "reference" if steps == len(prefix) else "candidate independent eager decode",
            "boundary": "GPU CUDA graph: full decoder + logits + greedy sampling + device token feedback",
            "excluded": "prefix/state restore, token trace copy, host read, compilation, loading, warmup",
            "cache_policy": "natural sequential decode; no synthetic flush", "serving_latency": False}
    if host_feedback:
        wall = []
        for repeat in range(repeats + 2):
            reset()
            torch.cuda.synchronize()
            started = perf_counter()
            generated = []
            for _ in range(steps):
                graph.replay()
                generated.append(int(values[token_buffer].item()))
            torch.cuda.synchronize()
            ms = (perf_counter() - started) * 1000 / steps
            if generated != expected:
                raise RuntimeError("Host-feedback graph sequence differs from independent eager decode")
            if int(values["paged_state.seq_lens"].item()) != int(initial["paged_state.seq_lens"].item()) + steps:
                raise RuntimeError("Host-feedback graph did not advance state exactly once per token")
            if repeat >= 2:
                wall.append({"ms_per_token": ms, "token_ids": generated})
        report["wall_feedback"] = {**summarize([row["ms_per_token"] for row in wall]), "rounds": wall,
                                   "boundary": "prepared model + greedy sampler + per-token host observation"}
    return report
