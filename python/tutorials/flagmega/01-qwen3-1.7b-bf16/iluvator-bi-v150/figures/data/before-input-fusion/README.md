# BI-V150 CUDA Graph measurements, 2026-10-21

The six `round*.json.gz` files are complete raw reports, including every prompt,
output token, warmup, measured decode interval and dependency/source identity.
The matching compressed logs retain native FULL graph capture confirmation.
`summary.json` is generated from those reports; `provenance.json` records the
artifact, hardware, execution boundaries and acquisition command.

Three alternating rounds, four prompt lengths (32/128/1024/2048), one warmup
and five measured independent requests per scenario, 64 output tokens, BF16,
batch/concurrency one, no prefix caching. Native vLLM uses FULL CUDA Graph;
FlagMega C++ uses an extension-owned decode graph with native prefill and
sampling. H800 measurement values are not used in these charts.

Rebuild all three SVGs from the tutorial directory:

```sh
python render_flagos_results.py --results figures/data --figure-dir figures
```

The renderer validates full token sequences, paired rounds, environment,
execution mode and C++ source/library identity before plotting. Decode
throughput uses the complete decode interval sum; request throughput includes
prefill. Neither is the inverse of median decode latency.

The large model weights and shared library are not embedded here. Rebenchmark
with an artifact rebuilt from the same final IR and current C++ backend, using
the command in the tutorial README. Original local logs are under
`.local/perf-iteration/cpp-vllm-cudagraph-20261021/`.

`boundary-native.json.gz`, `boundary-flagmega.json.gz` and
`boundary-summary.json` contain a separate paired diagnostic at prompt length 32
(three measured requests, 189 decode steps per variant). CUDA events surround
actual graph replay, logits and sampling; a host timer surrounds `engine.step`.
These diagnostics do not enter the charts. The native FULL graph excludes the
logits head, while the C++ graph includes it. Decomposition uses means, with
host time minus device time reported as a residual that includes overlap.
The acquisition script is retained locally at
`.local/perf-iteration/cpp-vllm-cudagraph-20261021/boundary/compare_graph_boundaries.py`.
