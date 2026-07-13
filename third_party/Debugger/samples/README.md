# FlagTree Debugger Samples

This directory keeps two small FlagGems regression inputs:

- `001_abs`: a pointwise level-1 debugger case.
- `295_softmax`: a reduction level-1 debugger case.

Each sample contains only a reproducible driver and minimal baseline metadata.
Generated kernels, copied FlagGems sources, compiler caches, debugger reports,
logs, and machine-specific command paths are intentionally excluded.

Replay the curated baseline with:

```bash
python3 third_party/Debugger/tools/flaggems_regression_from_samples.py \
  --samples-root third_party/Debugger/samples \
  --sample-index stable_index.json \
  --flaggems-root /path/to/FlagGems
```

Run outputs are written under the tool's workspace and must not be added back
to this directory.
