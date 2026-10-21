# Standalone chat CLI measurements

Same final BF16 Qwen3-1.7B C++ source/library and physical BI-V150 GPU 8 as
`../round*.json.gz`. CUDA Graph is enabled; batch/concurrency one, no prefix reuse,
32/128/1024/2048 input tokens and 64 greedy output tokens. Three rounds, five
independent measured requests per length and one warmup per length in each fresh process.
All three implementations are run serially, with scenario start temperature at
most 70°C. Cooling waits occur before warmup and are excluded from request timing.
Temperature and clock observations are retained in each raw report.

Timing includes the compiled token-scan prefill, model graph, greedy sampling,
host token delivery and incremental text decoding. Initialization, graph capture,
cache reset, tokenization, terminal writes and user input wait are excluded.
Decode excludes the first output token. Decode/request throughput use per-request
rates; neither is the reciprocal of the pooled median step latency.

## Measurement audit

The earlier CLI series ran all three rounds in one process and drifted after
continuous long token-scan prefill. It is retained in `superseded-continuous/`.
`measurement-audit.zh.md` records the same-process heating/cooling experiment:
graph execution increases from 6.134 to 8.430 ms and returns to 6.528 ms after
cooling, while outside-graph overhead stays near 0.100 ms. Current charts use
fresh processes and identical scenario start conditions for all three variants.
This is a controlled comparison; it does not claim sustained thermal behavior
has been fixed.

## Correctness boundary

The identical input can produce different BF16 sequences with native batched
prefill and token-scan prefill. The first output token for prompt length 32 is
576 in published native FULL-graph measurements and 81917 in standalone CLI.
Independent native FlagOS vLLM with `max_num_batched_tokens=1` also emits 81917.
`native-reference.json.gz` contains the complete native token-scan sequences for
all four lengths; it is correctness-only and is excluded from chart rates.
CLI compares every warmup and measured sequence against this reference.

`prefill-diagnostic.log.gz` checks old C++ graph, final C++ graph and final eager
on the same 32-token input: every vocabulary logit is bit-identical, max error 0.
This rules out the attention optimization and CUDA Graph capture as sources of
that initial divergence. Selecting native token-scan prefill reproduces the
standalone sequence. The internal native operator responsible for the difference
has not been isolated; shape-dependent kernel/FP32 accumulation is a possible
explanation, not a verified root cause. We do not claim identical output sequences
to native batched prefill or treat lower CLI decode latency as an equivalent
end-to-end request speedup. The original rejected check is retained in
`batched-prefill-mismatch/` and contributes no chart data.

The measured script snapshots are retained. Older native acquisition recorded
`torch.__version__` as 2.10.0; the current acquisition also records installed
COREX distribution metadata 2.10.0+corex.4.5.0.20260509. Device, compiler and
native dependency revisions are checked against the published FlagOS reports.

## Rebuild the combined charts

Run from the tutorial directory; only saved reports and FlagTree Python are
needed, with no GPU, vLLM import or model weights:

```sh
python render_flagos_results.py --results figures/data \
  --chat-results figures/data/chat-cli \
  --chat-native-reference figures/data/chat-cli/native-reference.json.gz \
  --figure-dir figures
```

`summary.json` contains CLI-only metrics; `../summary.json` contains the three
implementations. `provenance.json` records actual source/library identity and
SHA256 of uncompressed measurement files. 55 chart/serving tests passed across two runs (51 focused tests plus four device
graph tests), including
rejection of mismatched libraries, tokens, warmup and native reference identity.
The serving test fixture was updated to include the manifest codegen descriptor
used by the current `ArtifactBackend.info`; `tests-before.log.gz` retains its
original failure. `tests-final-focused.log.gz` and `tests-device-final.log.gz`
retain the final passing checks; the earlier 49-case suite is also retained.
