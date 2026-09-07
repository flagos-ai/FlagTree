# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Host-visible generation measurements with explicit token denominators."""

from dataclasses import asdict, dataclass
import math
import statistics


def summarize(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": values[max(0, math.ceil(.95 * len(values)) - 1)]}


@dataclass(frozen=True)
class GenerationMetrics:
    prompt_tokens: int
    cached_tokens: int
    prefill_tokens: int
    output_tokens: int
    prefill_ms: float
    ttft_ms: float
    decode_step_ms: tuple[float, ...]
    total_ms: float
    tokenization_ms: float = 0.0

    def to_dict(self):
        result = asdict(self)
        result["prompt_token_count"] = result.pop("prompt_tokens")
        decode_ms = sum(self.decode_step_ms)
        result.update(
            decode_tokens=len(self.decode_step_ms), decode_ms=decode_ms,
            decode_latency_ms=summarize(self.decode_step_ms),
            prefill_tokens_per_second=(self.prefill_tokens * 1000 / self.prefill_ms
                                       if self.prefill_ms > 0 else None),
            decode_tokens_per_second=(len(self.decode_step_ms) * 1000 / decode_ms
                                      if decode_ms > 0 else None),
            output_tokens_per_second=(self.output_tokens * 1000 / self.total_ms
                                      if self.total_ms > 0 else None),
        )
        return result

    def format(self):
        report = self.to_dict()
        def number(value):
            return "n/a" if value is None else f"{value:.2f}"
        return (
            f"prompt eval: {self.prefill_tokens} tokens, {self.prefill_ms:.2f} ms, "
            f"{number(report['prefill_tokens_per_second'])} tokens/s "
            f"({self.cached_tokens} cached)\n"
            f"TTFT: {self.ttft_ms:.2f} ms; decode: {report['decode_tokens']} tokens, "
            f"{report['decode_ms']:.2f} ms, {number(report['decode_tokens_per_second'])} tokens/s\n"
            f"decode latency: median {number(report['decode_latency_ms']['median'])} ms, "
            f"p95 {number(report['decode_latency_ms']['p95'])} ms; "
            f"total: {self.total_ms:.2f} ms, {self.output_tokens} output tokens"
        )
