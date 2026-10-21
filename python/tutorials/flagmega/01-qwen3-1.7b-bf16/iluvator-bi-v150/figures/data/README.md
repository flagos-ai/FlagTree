# BI-V150 CUDA Graph measurements, 2026-10-21

The six `round*.json.gz` files are complete raw reports, including every prompt,
output token, warmup, measured decode interval and dependency/source identity.
The matching compressed logs retain native FULL graph capture confirmation.
`summary.json` combines those reports with the standalone CLI reports in
`chat-cli/`; `provenance.json` records the
artifact, hardware, execution boundaries and acquisition command.

Three serially rotated rounds across native vLLM, FlagMega-vLLM and CLI, with
a fresh process for each variant/round and scenario start temperature at most
70°C. Cooling waits are outside request timing; raw temperature/clock
observations are retained. Four prompt lengths (32/128/1024/2048), one warmup
and five measured independent requests per scenario, 64 output tokens, BF16,
batch/concurrency one, no prefix caching. Native vLLM uses FULL CUDA Graph;
FlagMega C++ uses an extension-owned decode graph with native prefill and
sampling. H800 measurement values are not used in these charts.

Rebuild all three SVGs from the tutorial directory:

```sh
python render_flagos_results.py --results figures/data \
  --chat-results figures/data/chat-cli \
  --chat-native-reference figures/data/chat-cli/native-reference.json.gz \
  --figure-dir figures
```

The renderer validates full token sequences, paired rounds, environment,
execution mode, process lifecycle, scenario start conditions and C++ source/library identity before plotting. Decode
throughput uses the complete decode interval sum; request throughput includes
prefill. Neither is the inverse of median decode latency.

CLI uses the same C++ source/library, physical GPU and raw prompts, measured in
the same rotated rounds with compiled token-scan prefill. Every CLI sequence is compared
with independent native FlagOS token-scan prefill. Those BF16 sequences differ
from the published native batched-prefill sequences; the charts state this
boundary. CLI TTFT and full request throughput include its prefill costs.
See `chat-cli/README.md` and `chat-cli/provenance.json` for details and evidence.

The large model weights and shared library are not embedded here. Rebenchmark
with an artifact rebuilt from the same final IR and current C++ backend, using
the command in the tutorial README. Current local acquisition logs are under `.local/chat-cli-audit/controlled/`.
The original artifact verification reports are retained byte-for-byte as
`artifact_verify.json.gz` here and in `before-long-attention/`;
`verification-reports.json` records their uncompressed hashes and lengths.
Earlier unconditioned formal measurements are preserved in `before-cli-audit/`;
the earlier continuous CLI measurements and thermal reproducer are documented
in `chat-cli/measurement-audit.zh.md`.

`boundary-native.json.gz`, `boundary-flagmega.json.gz` and
`boundary-summary.json` contain a separate paired diagnostic at prompt length 32
(three measured requests, 189 decode steps per variant). CUDA events surround
actual graph replay, logits and sampling; a host timer surrounds `engine.step`.
These diagnostics do not enter the charts. The native FULL graph excludes the
logits head, while the C++ graph includes it. Decomposition uses means, with
host time minus device time reported as a residual that includes overlap.
The acquisition script is `../../benchmark_graph_boundary.py`; its measured snapshot
is retained as `benchmark_graph_boundary.measured.py`.


Final attention staging uses four concurrent query heads and 64-token KV loads,
while keeping canonical 16-token softmax state updates, ordered sums and explicit
FMA. Shared storage is 116992 bytes within the existing 128 KiB arena. See
`attention-root-cause.zh.md` for causal defects, rejected probes and before/after
evidence. `before-long-attention/` preserves the previously published data.
`rejected-warp-softmax/` preserves candidates that failed additional correctness
checks and is excluded from chart aggregation.

`sme-validation/` retains the earlier SME alignment and shared-reader
fail-before/pass-after logs, canonical readonly layout contracts, independent
five-family bandwidth measurements and the unresolved historical small-file
source-loader failure. Its README separates these diagnostics from final
request-level results.

`validation.json` records 126 passing regressions, bit equality for every logit
across 2112 model steps, complete native token agreement for all formal requests,
and 14 adjacent sampling requests per variant including a 2048-token random
sampling prompt. `attention-performance.json` records measured speedup ratios:
1024/2048 reach about 94%/88% of the short-context speedup, with no claim of an
identical speedup or a user-specified numeric SLA.
