# Ascend split/join correctness and performance

This suite compares compiler-generated split/join with an independent Ascend C
vector implementation. It contains no FFT code or size-specific FFT rewrite.

## What is measured

| Case | Work per row | Observable outputs |
|---|---|---|
| `split` | Deinterleave `2Q` elements | Both channels |
| `join` | Interleave two independent `Q`-element inputs | Interleaved result |
| `nested` | Transpose `(4,Q)`, split twice, reconstruct | Transpose, four channels, reconstruction |

Both implementations read the same inputs and produce the same outputs. The AIV
reference uses `GatherMask` for split and vector `Gather` for join/transpose.
Its index-table HBM load is included. It is a vector reference, not a hardware
peak-throughput claim. `aiv.cpp` treats values as raw 16/32-bit words.

Triton is compiled through JIT once, then every captured call uses the
`CompiledKernel` launcher, matching archived-binary controls. Each run checks
every output before timing. After graph capture it clears all
outputs, replays the graph, and checks them again. Native launches resolve the
current stream inside the launch callback so they are captured on the graph's
stream. Each graph contains 64 kernel calls by default; event time is divided by the
actual call count. This amortizes host replay gaps. Use `--graph-steps` to check
convergence; all compared implementations must use the same protocol. JSON contains all timing rounds and replay verification.

## Build and run

Use a rebuilt Ascend FlagTree package and its matching patched FLIR dependency.
The CANN installation is not modified. Select a free NPU and run one case per
process: a runtime device exception can invalidate that process's NPU context.

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export FLAGTREE_BACKEND=ascend TRITON_BACKEND=torch_npu
export TRITON_JIT_BACKEND=NPU TRITON_ASCEND_ARCH=Ascend910B4
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONNOUSERSITE=1
export ASCEND_VISIBLE_DEVICES=6 ASCEND_RT_VISIBLE_DEVICES=6 NPU_VISIBLE_DEVICES=6
export TRITON_ALL_BLOCKS_PARALLEL=0
export TRITON_NPU_COMPILER_PATH=/usr/local/Ascend/cann-9.0.0/tools/bishengir/bin
# For an isolated in-tree build (adjust the platform/Python suffix):
export PYTHONPATH="$PWD/build/lib.linux-aarch64-cpython-311"
export TRITON_CACHE_DIR=/tmp/structured-permute-cache
bench=third_party/ascend/benchmarks/structured_permute
out=/tmp/structured-permute-results
mkdir -p "$out"
cmake -S "$bench" -B "$out/aiv-build" -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 -DSOC_VERSION=Ascend910B4
cmake --build "$out/aiv-build" -j8

for impl in triton aiv; do
  for case in split join nested; do
    for q in 32 256 1024; do
      python "$bench/bench.py" --impl "$impl" --case "$case" --q "$q" \
        --batch 2048 --aiv-blocks 2048 --dtype float32 \
        --library "$out/aiv-build/libpermute_reference.so" \
        --output "$out/$impl-$case-q$q.json"
    done
  done
done
```

For a separate 40-block comparison, enable the existing compiler auto-blockify
option and use the persistent native reference alongside it. Effective dispatch
is included in both JIT specialization and disk-cache keys. Explicit
`enable_auto_blockify=True/False` overrides the environment. An archived binary
must still be replayed with its own dispatch contract.

```bash
TRITON_ALL_BLOCKS_PARALLEL=1 python "$bench/bench.py" \
  --impl triton --case nested --q 256 --output "$out/triton40-nested-q256.json"
python "$bench/bench.py" --impl aiv --case nested --q 256 --aiv-blocks 40 \
  --library "$out/aiv-build/libpermute_reference.so" \
  --output "$out/aiv40-nested-q256.json"
```

`--aiv-blocks 2048` matches the Triton logical grid. The default AIV grid is a
persistent 40-block loop; report that separately. Use `--dtype float16`,
`bfloat16`, `int16`, or `int32` for the other word types. `--dump-dir` saves the
Triton compiler stages before timing. Incorrect runs raise and produce no
successful timing JSON.

## Regression tests

From the FlagTree root, under the same environment:

```bash
pytest -q third_party/ascend/unittest/pytest_ut/test_split_join_layout.py -x
pytest -q third_party/ascend/unittest/pytest_ut/test_split.py \
  third_party/ascend/unittest/pytest_ut/test_interleave.py \
  third_party/ascend/unittest/pytest_ut/test_interleave_optimizaiton.py \
  third_party/ascend/unittest/pytest_ut/test_ir_builder.py -x
pytest -q third_party/ascend/unittest/pytest_ut/test_custom_bytecode_compat.py -x
pytest -q third_party/ascend/unittest/pytest_ut/test_split_join_values.py -x
pytest -q third_party/ascend/unittest/pytest_ut/test_structured_permutation_p0.py -x
```

The layout suite covers rank/shape changes, multiple consumers, independent and
swapped joins, input/output strides and offsets, output guard elements, computed
values, masked tails, constants, and exact floating-point bit patterns including
NaN payloads. It also exercises the native large-transpose fallback.

## Compiler implementation and limits

The FLIR pass first folds exact inverse `join(split(x))` pairs and moves channel
selection before transpose when the shape identity proves it safe. Small-axis
static transposes use vector index arithmetic and a private UB permutation
primitive; large transposes retain native tiling. These are static permutations,
not dynamic `tl.gather` operations.

Before dialect conversion, the pass proves single-use dense unmasked loads and
contiguous terminal stores. A simple rank-two split meeting that proof retains CANN's native load/store-folding
path. Joins use compact interleave within the resource budget in both dispatch
modes. The explicit `native` join policy requires the same dense graph proof. Transposed, computed,
masked and shared inputs do not qualify. This is a graph property, not a
benchmark-name or size-specific admission rule.

In the packed and proven dense native paths, floating-point joins move values as same-width integers so concat/copy cannot
canonicalize NaN payloads. The c220 vector path is enabled for Ascend 910B targets;
other architectures and encoded join layouts retain the existing lowering.

Private device implementations are built into FlagTree's packaged
`custom_ops.bc` and linked automatically. Split uses `vreducev2` for packed data;
a gather honors the physical UB strides chosen downstream. An older CANN custom
operation schema is adapted at the bytecode boundary. After CANN has planned
buffers and synchronization, a private-operation-only ABI adapter emits device
function calls for the installed hivmc. It preserves existing allocations,
strides and synchronization; no installed CANN template or binary is patched.

The vector fast path requires packed, 32-byte-aligned destinations, indices and
scratch, and supports strided or element-offset sources through vector gather.
Packed split uses vector reduction only when its source is also aligned. Narrow byte values use private byte/word storage conversion around permutations,
which prevents CANN from moving a flat cast back into its broken sub-block 2D
cast path. Nonunit destination/index/scratch strides and unaligned destinations, indices
or scratch have a synchronized scalar correctness path. Do not claim vector performance
for those layouts. Tiny two-scalar joins use two exact element insertions.

Static permutation indices are bounded to 4096 entries (16 KiB). Joins with
more than 2048 elements per input retain the existing interleave lowering before
any byte widening; larger transposes retain native tiling. Zero-offset rank-one
one-element copies normalize irrelevant strides to one, avoiding a separate
strided-byte-load failure without adding scalar data loops. This suite verifies the listed cases and does
not establish correctness or performance for every Triton operation/dtype or
other CANN/SoC versions.

Set `FLAGTREE_PERMUTE_DUMP_DIR` to retain the private operations immediately before
and after late ABI lowering. These artifacts expose actual UB descriptors and
synchronization, so the packed vector-path claim can be checked against the
physical strides rather than inferred from source tensor shapes.


## P0 coverage and acceptance

Run the entire regression list twice, in separate processes, with
`TRITON_ALL_BLOCKS_PARALLEL=0` and `=1`. Tests include batch boundaries
1/39/40/41/2047/2049, either split result alone, masked byte tails, native-path
admission/rejection, index/resource thresholds and 32 seeded rank-2 through rank-5
graphs. All outputs, side outputs and guard elements are checked. The large
computed FP32 join explicitly accepts only exact results or a compile-time UB
resource diagnostic; it does not accept runtime failure. This resource test does
not claim that every oversized combination can run on 192 KiB UB.

HBM views do not establish physical UB strides. The optional ABI harness links
the **same device bitcode shipped by FlagTree**, constructs actual UB memref
descriptors, poisons complete output buffers and repeats calls with reused
scratch. Its 348 cases cover 16/32-bit raw words, byte storage conversions,
strides, offsets and vector/tail boundaries. Reconfigure/rebuild it whenever the
bitcode changes; CMake embeds dependency digests in its generated primary source.

```bash
bc="$PWD/python/triton/experimental/tle/language/dsa/ascend/custom_ops/custom_ops.bc"
cmake -S "$bench" -B "$out/ub-build" -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 \
  -DSOC_VERSION=Ascend910B4 -DFLAGTREE_DEVICE_BITCODE="$bc"
cmake --build "$out/ub-build" --target ub_contract_reference -j8
FLAGTREE_UB_CONTRACT_LIBRARY="$out/ub-build/libub_contract_reference.so" \
  pytest -q "$bench/test_ub_contract.py" -x
```

Cross-process cache checks require a new result directory. Specify compatible
LLVM tools explicitly if they are not on `PATH`; the CANN toolkit may ship
`llvm-link` without `llvm-as`. The suite checks 12 cold/hot, dispatch-override,
missing-final-binary, device-bitcode-change and binary-only contracts. Each child
checks outputs and exits cleanly; it also records compiler subprocesses.

```bash
python "$bench/cache_contract.py" --root "$out/cache-contract" \
  --llvm-as /path/to/compatible/llvm-as \
  --llvm-link /usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/llvm-link
```

For controlled join A/B, use `--join-lowering auto`, `native`, or `packed` with
the same dtype, Q, batch, dispatch and timing protocol. These compiler options
are hashed and cannot bypass safety/resource admission. Report dispatch modes
separately. Build a wheel normally and install it into an empty isolated target;
run tests from outside the checkout and verify Python modules, `libtriton` and
`custom_ops.bc` all resolve under that target. Do not validate a wheel by copying
files into its installation from a previous build.

The FLIR tests `structured_permutation.mlir` and
`structured_permutation_admission.mlir` cover rewrite legality and fallback.
`structured_permutation_encoded.mlir` is a **negative** test: this Ascend pipeline
consumes unencoded TTIR and must not silently accept TTG distributed encodings.
Other architectures are checked for IR fallback only, not device execution.

## Persistent-loop performance and paired ablations

After CANN buffer planning, a typed MLIR pass may reuse the existing private
index and scratch buffers as two absolute address tables. Admission requires
fixed packed physical layouts, aligned tables/output, nonoverlapping storage,
and no other consumer or writer of either table. Preheader DMA reads/writes
and unknown effects also reject reuse: their accesses may still be in flight.
Only the known int32 vector index initializers are admitted before the loop;
their scalar seeds synchronize to PIPE_V internally and the preparation helper
drains PIPE_V before overwriting either table. The input may select between
two planned buffers. Zero-, one- and two-iteration loops keep the original
address calculation; dynamic trip counts use a guarded path. No UB allocation
or hardware-size-specific tensor rewrite is added.

A separate pass coalesces MTE3-only waits between direct UB-to-GM stores when
all destination intervals are proven disjoint relative to the same GM base.
It requires packed aligned UB sources, full bursts and non-atomic stores.
Unknown aliases, intervening writes/control flow, other pipeline events, and
the final wait remain unchanged. Neither pass modifies installed CANN files.
Both options are boolean and participate in compilation/cache keys; the late
optimizer binary and transport helpers are hashed too.

For ordinary `bench.py` runs, use `--no-address-hoisting` and/or
`--no-store-sync-optimization` as compiler controls. For the paired experiment:

```bash
TRITON_ALL_BLOCKS_PARALLEL=1 python "$bench/bench_paired.py" \
  --q 256 --batch 2048 --dtype float32 \
  --library "$out/aiv-build/libpermute_reference.so" \
  --output "$out/paired-q256-float32.json"
```

The five variants disable both optimizations, enable addresses only, enable
store synchronization only, enable both, and run the independent vector AIV
reference. They share the same input/output buffers in one process. Measurement
order rotates, every output is verified before capture and after cleared-output
replay, and all timing rounds and compiler stages are retained. The protocol is
10 warmups, 5 rounds of 100 graphs, 64 calls per graph. Repeat with dispatch mode
0 and other supported Q/dtypes; compare modes separately.

Additional regressions are `test_structured_address_hoist.py` (host admission,
cache options and dependency rejection) and `test_structured_store_values.py`
(device guards, GM offsets and short/long loop paths). Run both alongside the
full list above in each dispatch mode.


## Matrix transport: unchanged short axes and sub-block GM stores

These regressions do not depend on FFT. Run `test_short_axis_transpose.py` and
`test_short_row_store.py` in **separate Python/pytest processes**: a vector-core
exception invalidates the current process's NPU context. Repeat with
`TRITON_ALL_BLOCKS_PARALLEL=0` and `1`. The store suite compares parallel and
serial launches of the **same binary**, exercises contiguous and boundary-checked
block-pointer stores, and checks input/output guards and masked rows/columns.
The transpose suite includes the original masked `[16,4,2] -> [4,16,2]` graph,
with and without split/join, plus 3–5D static permutations and five storage types.

```bash
ut="$src/third_party/ascend/unittest/pytest_ut"
python -m pytest -q "$ut/test_short_axis_transpose.py"
python -m pytest -q "$ut/test_short_row_store.py"
```

FLIR admits packed permutations retaining a short last axis under the existing
910B, static unencoded power-of-two shape and 4096-element index-budget checks.
A separate short-row store lowering handles rank-two FP16/BF16/FP32/I16/I32
values with a statically bounded row shorter than 32 bytes and contiguous inner
GM stride. Its two UB workspaces are planned before allocation, bounded by
8 KiB together. Aligned or fully contiguous physical sources use direct DMA;
compact/misaligned sources use unsigned vector address generation, `vgather`
repacking and byte-counted DMA. Internal DMA is complete before returning to
PIPE_V. This path uses no scalar GM writes, changes no installed SDK files,
and neither serializes programs nor specializes on an application/FFT size.
Other shapes, storage types and architectures retain their existing lowering.

`short_row_store.mlir` checks compiler admission/fallback. The physical ABI
suite additionally calls the packaged store primitive on controlled UB row and
column strides, zero/one/chunk-boundary row counts, aligned/unaligned source
starts and concurrent adjacent GM destinations. Rebuild the ABI harness when
the bitcode changes. The low-bit extraction must use **unsigned** vector shifts;
signed shifts can create negative padding addresses even when numerical output
checks do not observe those lanes.

The independent `matrix_transport_aiv` reference keeps aligned UB rows through
load/add/store and uses a vector gather for the short-axis transpose. It is a
comparison implementation, not a hardware-peak claim. Its transpose index-table
HBM load is timed. The benchmark uses the original shapes, exact outputs and GM
guards, with the dynamic program-base input removed from timing variants. All
three policies share the same buffers/stream; use an unshared device and retain
occupancy samples. Results are **microseconds per whole matrix kernel**, not
FFT transform times.

```bash
TRITON_ALL_BLOCKS_PARALLEL=1 python "$bench/bench_matrix_transport.py" \
  --scene store --library "$out/aiv-build/libmatrix_transport_reference.so" \
  --output "$out/matrix-store.json"
TRITON_ALL_BLOCKS_PARALLEL=1 python "$bench/bench_matrix_transport.py" \
  --scene transpose --library "$out/aiv-build/libmatrix_transport_reference.so" \
  --output "$out/matrix-transpose.json"
```

The protocol is 10 warmups, 5 rotated rounds of 100 graphs, 64 launches per graph.
Each policy verifies exact outputs/guards before capture, after replay into
cleared outputs, and after every timing round. Run `bench_paired.py` as well to
check the previous structured-permutation performance baseline.

Full unmasked short-row strided GM loads use a symmetric compiler-owned
DMA/vector operation with at most 4 KiB of preplanned workspace. Aligned
physical outputs load directly; packed outputs use aligned DMA into the
workspace and `vreducev2` compaction. This is needed because an early padded
alloc or storage-alignment marker can disappear during CANN bufferization.
Default cache/eviction, nonvolatile, static unencoded tiles are admitted;
masked, boundary-checked, dense-GM and nondefault-cache loads keep their
existing lowering. The physical-codegen regression checks the final CANN IR,
so a numerically correct scalar-load regression cannot silently pass.
