# TileIR Tutorials

These tutorials assume FlagTree has been built with the default in-tree backend
set including `tileir`. For build environment requirements, see
[`third_party/tileir/README.md`](../../../third_party/tileir/README.md).

By default, their compilation caches are stored under
`/tmp/flagtree_tileir_tutorial_cache/{01,02,03}`. `TRITON_CACHE_DIR` overrides
the default, and tutorial 03 also accepts `--cache-dir`.

## 01 Load View Token Ordering

Tests:

- TileIR TLE view-token APIs: `tle.make_view`, `tle.load_view_tko`,
  `tle.store_view_tko`
- memory token chaining
- expected native NVIDIA failure when TileIR routing is disabled

Run:

```bash
python3 python/tutorials/tileir/01-load-view-token-ordering.py
```

## 02 Mixed Kernel Routing

Tests:

- a plain Triton kernel routes to TileIR
- a non-TileIR TLE shared-memory kernel falls back to native NVIDIA
- both routes work in one process

Run:

```bash
python3 python/tutorials/tileir/02-mixed-kernel-routing.py
```

## 03 Triton TileIR Benchmarks

Tests:

- selected self-contained Triton benchmark kernels
- native NVIDIA path vs TileIR path
- correctness checks and CUPTI kernel-time measurements

`pair` selects benchmark configurations. Use `all` for every selected case, a
case-local numeric index such as `0` when only one case is selected, or full pair
ids such as `bmm-00,matmul-03`. Full pair ids may be listed in any order; output
is grouped by the selected case order.

`path` selects which backend path to run: `native`, `tileir`, or `both`.

Quickly check one pair on both paths:

```bash
python3 python/tutorials/tileir/03-triton-tileir-benchmarks.py \
  --case bmm --pair 0 --path both
```

Run the CI subset, with three curated representative pairs per selected case:

```bash
python3 python/tutorials/tileir/03-triton-tileir-benchmarks.py \
  --case all --ci-subset --path both
```

The script disables autotuning by default for quick and CI runs. For a full
performance comparison, enable autotuning and run both paths on an idle GPU:

**Always set `TILEGYM_DISABLE_AUTOTUNE=0` for performance measurements.**
Without it, the benchmark uses fixed configurations intended for quick tests.

```bash
TILEGYM_DISABLE_AUTOTUNE=0 \
python3 python/tutorials/tileir/03-triton-tileir-benchmarks.py \
  --case all --pair all --path both \
  --cache-dir /tmp/flagtree_tileir_full_autotune_cache --clear-cache
```

For each pair, `both` runs Native and TileIR sequentially in separate child
processes, checks correctness, and reports both timings and
`native ms / TileIR ms` speedup.

Available cases:

```text
bmm, fmha, linear_bias_act, mla, mla_decoding, matmul, rope
```
