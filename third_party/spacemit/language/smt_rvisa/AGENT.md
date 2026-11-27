# smt_rvisa — Writing Raw Operators (Agent Guide)

A practical guide to writing custom operators with the `smt_rvisa` eDSL. Kernels
are written in a restricted Python subset, lowered through the C++ builder API
(no MLIR text), and run on SpacemiT RISC-V (K3) via scalable vectors.

> All examples below are verified on K3 (riscv64). See `python/tests/raw/` for
> 25 working operator families (231 tests). For the full language spec, see
> `SPEC.md` (the smt_rvisa raw eDSL specification).

---

## 0. What is smt_rvisa — the semantic model

**smt_rvisa is a hand-written execution plan at the *vector* level.** Normal
Triton describes *what* to compute over tensors and lets the compiler pick the
vectorization; smt_rvisa lets you write the SEW/VL/tiling/instruction-selection
*yourself*, one vector register at a time. You trade automation for control —
useful when you want a specific RVV instruction sequence (vfwmadot, batched
cube ops) or a specific memory-streaming schedule the autoscheduler won't pick.

### The execution model in four decisions
A raw kernel is you making four choices explicitly (SPEC §2):
- **A — decomposition & vector width**: how the problem splits into VL-wide
  chunks. VL (vector length) is the number of lanes processed per instruction.
- **B — instruction selection**: which op each line maps to (`vmacc`→vfwmacc,
  `vmadot`→cross_batch_matmul, `vreduce_sum`→vector.reduction).
- **C — data orchestration**: value & memory model (below).
- **D — scheduling**: loop structure, iter_args, tiling.

### Scalable vectors & VL (the key mental model)
- The hardware is **RVV scalable vectors**: a vector value is `vector<[n]×T>`
  where the runtime length is `vscale × n`. On K3, VLEN=1024, so for f32
  (SEW=32) one register holds **VL=64** elements; f16 also VL=64 at LMUL=1.
- **You don't see `[n]` / vscale in the DSL** — you write `vector<VL×T>` and the
  compiler makes it scalable. `vconfig(avl, lmul)` sets the active VL:
  `vconfig(-1, 1)` = VLMAX (64), `vconfig(k, 1)` = min(k, 64) for tails.
- **A vector op processes exactly VL lanes.** A loop over `range(0, N, VL)`
  sweeps the data VL elements per iteration. This is the "single-pass
  streaming" model — one memory sweep, register-resident accumulator.

### Value & memory model (SPEC §2.4)
- **Values** are SSA vectors/scalars held in registers. `acc = acc + vx`
  produces a new SSA value; there is no mutable state except loop `iter_args`.
- **Memory** is flat: `vload(ptr, idx)` reads VL elements starting at *flat
  element offset* `idx` (you compute 2D coords yourself: `row*N + col`).
  `vstore` writes a vector back; `sstore` writes a single scalar. Scratch
  memory via `alloc`.
- **Loops carry state via iter_args**: a variable reassigned inside a
  `srv.range` loop becomes a loop-carried value (like `acc` in a reduction).
  This is why tail loops need distinct temp names — see §3.

### raw kernel vs host kernel (the boundary)
- **raw kernel** (`@srv.raw_kernel`): the vector-level body. Runs on the vector
  unit. Every parameter is type-annotated (`mem`/`index`). Emitted as a
  `tle.dsl_region` op, lowered `tle→linalg→memref→llvm` through the C++ builder
  API (no MLIR text).
- **host kernel** (`@triton.jit`): ordinary Triton. Computes `program_id`,
  slices per-program work, and calls `srv_call(raw_kernel, inputs=[...])`. The
  grid (`[(G,)]`) decides how many programs run.
- The boundary: host picks *which slice* each program handles (row, block);
  raw kernel says *how* to compute that slice on the vector unit.

### Lowering chain (what happens after you write it)

**Default path** (high-level primitives: `vload`/`vstore`/`vmacc`/...):
```
@srv.raw_kernel Python
  → AST walk (codegen.py, builder API)   # no MLIR text
  → tle.dsl_region op in TTIR
  → --triton-to-linalg-experimental      # tle → linalg/memref
  → --spine-triton-e2e-pipeline (spine-opt)  # → scalable vectors → LLVM
  → llc → .o → .so                        # RVV machine code
```

**LLVM-direct path** (`llvm_*` / `call_intrinsic` only, detected automatically):
```
@srv.raw_kernel Python
  → AST walk (llvm_direct.py)            # MLIR Python bindings emitter, no text
  → top-level llvm.func module (text)
  → mlir-translate --mlir-to-llvmir      # skip spine-opt, only translate
  → llc → .o → .so                        # RVV machine code
```

Two consequences you'll hit: (1) only some vector ops survive spine-mlir's
`ConvertToScalableVector` (§7); (2) x86 llc can't expand `vector.reduction`, so
reduction-using kernels are **K3-only** for numerical verification. The
LLVM-direct path bypasses both limits but requires you to manage LLVM-level details (§8).

---

## 1. Anatomy of a raw operator

Every operator has **three parts**:

```python
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

import triton.language.extra.smt_rvisa as srv           # MUST import from here
from triton.language.extra.smt_rvisa import call as srv_call

f16 = srv.f16
f32 = srv.f32

# ── PART 1: the raw kernel (runs on the vector unit) ─────────────────────
@srv.raw_kernel
def my_kernel(X: srv.mem(f32), out: srv.mem(f32, out=True), N: srv.index):
    nvl = srv.vconfig(-1, 1)                 # set VL (VLMAX for f32 = 64)
    acc = srv.vzero(f32)
    for i in srv.range(0, N, nvl):
        vx = srv.vload(X, i, dtype=f32)
        acc = acc + vx
    srv.sstore(out, 0, srv.vreduce_sum(acc))

# ── PART 2: the @triton.jit host wrapper (dispatches the kernel) ─────────
@triton.jit
def my_host(X, out, N):
    srv_call(my_kernel, outputs=[], inputs=[X, out, N])

# ── PART 3: the Python launcher ──────────────────────────────────────────
X = torch.randn(256, dtype=torch.float32)
out = torch.zeros(1, dtype=torch.float32)
my_host[(1,)](X, out, 256)                   # grid=(1,)
```

**Rules:**
- Import `srv` ONLY from `triton.language.extra.smt_rvisa` (never a standalone path).
- Kernel params are annotated: `srv.mem(dtype)` (read), `srv.mem(dtype, out=True)`
  (write), `srv.index` (scalar loop bound / offset).
- The host is a normal `@triton.jit` function; it calls `srv_call(kernel, outputs=[], inputs=[...])`.
- `grid=(G,)` launches G programs; use `tl.program_id(0)` in the host to get the
  row/group index and pass it into `inputs`.

---

## 2. Primitive reference

### Config / init
| Primitive | Signature | Notes |
|-----------|-----------|-------|
| `vconfig(avl, lmul)` | → sets active VL | `vconfig(-1, 1)` = VLMAX (f32→64, f16→64). `vconfig(N-i, 1)` = narrow VL for tail. **Must call before any vector op.** |
| `vzero(dtype, group=None)` | → `vector<VL×dtype>` or `vector<group×VL×dtype>` | all-zero accumulator; `group=` for the matrix-engine acc |
| `viota()` | → `vector<VL×f32>` | `[0,1,..,VL-1]`, for index tracking (argmax) |

### Memory
| Primitive | Signature | Notes |
|-----------|-----------|-------|
| `vload(...)` | → `vector<VL×dtype>` | **vector** load (width VL), see forms below |
| `vstore(...)` | store | **vector** store (width VL), see forms below |
| `sload(ptr, idx, dtype=f32)` | → scalar | **scalar** load of one element at dynamic index (gather). `memref.load`, never touches VL |
| `sstore(ptr, idx, scalar)` | store | **scalar** store of one element. `memref.store`, never touches VL |
| `alloc(shape, dtype)` | → ranked memref | scratch buffer, e.g. `alloc((VL,), f32)` |

> **Naming: `v*` = vector op (width = VL), `s*` = scalar op (one element, no VL).**
> `vload`/`vstore` move a full VL-wide vector; `sload`/`sstore` move a single
> scalar. Passing a scalar to `vstore` (or a vector to `sstore`) raises a clear
> `TypeError` — no silent dispatch. (Not named bare `load`/`store`: a kernel may
> also use Triton's `tl.load`/`tl.store` with tensor semantics.)

#### `vload` — the full picture

```
vload(ptr, idx, dtype=f16, fill=0.0, group=None)
```

| kwarg | meaning |
|-------|---------|
| `dtype` | element type of the loaded vector (default **f16**). Load then `cast(v, f32)` for f32 math. |
| `fill` | value used for lanes past the valid bound in a narrowed/tail load (default `0.0`). Use `-1e38` for max-reductions, `1e38` for min-reductions. |
| `group` | load `group` consecutive cubes → `vector<group×VL×dtype>` (for the matrix-engine `vmadot` path). |

> **Effective length is `vconfig`'s job, not `vload`'s.** A narrowing
> `vconfig(N-i, 1)` sets the *active valid* length; `vload` reads exactly that
> many elements and `fill`-pads the rest. There is no explicit `valid=` kwarg —
> `vconfig` is the single source of truth, `vload` only reads + fills.

**Addressing — `idx` has two forms:**
- **Scalar flat offset** (external `mem` pointer): `vload(X, row*N + col)` reads
  VL elements starting at that flat element index. This is the common case.
- **Index tuple** (ranked scratch from `alloc`, or a packed tensor from
  `vpack`): `vload(scratch, (i, j))` / `vload(Bcube, (0, kc), group=B1)` — one
  index per memref dimension.

**Three internal paths (chosen automatically, you don't pick):**
1. **Full-tile fast path** — no active valid, main loop: direct `transfer_read`
   of VL elements. Fastest, zero masking.
2. **Fill-0/`fill=` padded path** — when a narrowing `vconfig(N-i,1)` set the
   active valid (tail loop): reads only the valid elements via a bounded
   `reinterpret_cast`, then `linalg.fill(fill)` + `insert_slice` pads lanes
   `[valid:VL)`. Prevents out-of-bounds reads at the buffer end.
3. **Grouped path** (`group=`) — reads `group×VL` and shape-casts to a
   rank-2 `vector<group×VL×dtype>`.

> **Why the tail needs `fill=`**: padded lanes participate in the following
> arithmetic. For `+`/sum, `fill=0` is correct. For max-reduce a padded 0 could
> beat real negatives → use `fill=-1e38`. For `exp` in softmax, `fill=-1e38` so
> `exp(fill-max)≈0` doesn't inflate the sum.

#### `vstore` / `sstore` — the full picture

```
vstore(ptr, idx, vec)                # 1D vector (width VL)
vstore(ptr, idx, vec, shape=(R, C))  # 2D block
sstore(ptr, idx, scalar)             # single scalar
```

| primitive | `val` type | lowering |
|-----------|-----------|----------|
| `sstore` | f32/f16 scalar | `memref.store` at flat `idx` (scalar op, no VL) |
| `vstore` 1D | `vector<VL×T>` | `transfer_write` of the full VL |
| `vstore` 2D | `vector<R×C×T>` + `shape=(R,C)` | reinterpret_cast to `memref<R×C×T, strided>` + 2D `transfer_write` |

> `vstore` accepts **vectors only** — passing a scalar raises `TypeError`
> pointing you to `sstore` (symmetric with the `vload`/`sload` split on the read
> side). Reduction results (`vreduce_*`), `sload` results, and `scalar/N`
> expressions are scalars → use `sstore`. Values from `vload`/`vzero`/`viota`,
> `*scale`/`*inv` broadcasts, and elementwise vector arithmetic → use `vstore`.

**1D vector store auto-splits into two paths (based on active valid):**
- **Full-tile** (no active valid): reinterpret_cast to a *static* `memref<VL×T>`
  so the VSE writes all VL lanes. (A dynamic `memref<?×T>` would clamp the store
  VL to the descriptor size and only write lane 0 — this was a real bug, now
  handled.)
- **Tail-tile** (active valid set by a narrowing `vconfig`): dynamic
  `memref<?×T>` sized to `valid` + `in_bounds=[false]` → a **masked** partial
  write that respects the bound (prevents writing past the buffer end).

So for arbitrary-N output you use the *same* main+tail loop structure as loads;
the store picks full vs masked automatically from the active `vconfig`.

### Arithmetic (Python operators work directly on vectors/scalars)
- `+ - * / // % & | ^ << >> ~` and comparisons `< <= > >= == !=`
- **Scalar ÷ index auto-promotes**: `vreduce_sum(acc) / N` (f32 ÷ index) just works.
- `vmax(a, b)` / `vmin(a, b)` — elementwise max/min
- `abs(a)`, `sqrt(a)`, `rsqrt(a)`, `vexp(a)`, `vlog(a)` — math
- `cast(a, dtype)` — type conversion (vector or scalar)
- `select(mask, a, b)` — `a if mask else b`

### Reductions (vector → scalar)
| Primitive | Op | Notes |
|-----------|-----|-------|
| `vreduce_sum(v)` | Σ | ✅ hardware |
| `vreduce_max(v)` | max | ✅ hardware (float) |
| `vreduce_min(v)` | min | ✅ hardware (float) |
| `vreduce_mul(v)` | Π | ⚠️ RVV has no hardware reduce-mul → llc crash. Avoid. |

### Shape / broadcast
| Primitive | Signature | Lowering / use |
|-----------|-----------|----------------|
| `vshape(v, shape)` | reshape a vector (same total elems) | `vector.shape_cast`. e.g. flatten `vector<g×VL×T>` ↔ `vector<(g·VL)×T>` |
| `vbroadcast(v, n)` | `vector<VL×T>` → `vector<n×VL×T>` | `vector.broadcast` — adds a leading broadcast dim (replicate a row n times) |

### Scalar / index helpers
| Primitive | Signature | Lowering / use |
|-----------|-----------|----------------|
| `imin(a, b)` | scalar **index** min | `arith.minsi` on `index`. For clamping row/tile bounds, e.g. `valid_rows = imin(MB, M - row_base)`. (Distinct from `vmin`, which is elementwise on vectors.) |
| `sqrt/rsqrt/vexp/vlog/abs(a)` | math | also accept a **scalar** operand (not just vectors) — used in L0 scalar norm math |

### Profiling
| Primitive | Signature | Lowering / use |
|-----------|-----------|----------------|
| `proton_mark(name, is_start)` | emit a timestamp mark | `rdtime` + `func.call @proton_record`. Wrap a region with start/end marks to profile it. Skipped (no-op) in the builder path unless profiling is wired up. |

### Matrix engine (advanced — cube MMA, see SPEC §6.3 & mv/mm tests)
The K3 "cube" matrix unit multiplies small batched tiles. These compose the
mv/mm kernels; you rarely need them for reduce/elementwise ops.

| Primitive | Signature | Lowering / use |
|-----------|-----------|----------------|
| `vmacc(acc, x, y)` | widening FMA accumulate | `acc += x*y` with widening (e.g. f16×f16→f32). The scalar-vector MAC used in `mv_svector`. |
| `vmadot(acc, x, y)` | batched cube matmul | `vector_ext.cross_batch_matmul` → many `smt.vfwmadot`. `acc` rows must equal `b1·b2` of the operands. |
| `vpack(v, group_len)` | cube interleave | `vector_ext.group_interleave` → `smt.vpack.vv`. Interleaves `vector<b×N>` → `vector<(b/2)×2N>` for cube layout. |
| `spread(src, cube_shape=(kc,n,k))` | scalar-broadcast pack | `scf.for` scalar pack → `memref<kc×(n·k)>`. Broadcasts A's n dim, bypassing vscale. |
| `pack(src, src_idx, dst, dst_shape)` | row pack | pack rows into a cube-shaped scratch buffer (style3). |

---

## 3. The tail-loop idiom (arbitrary N)

VL is fixed (64). For N not a multiple of VL, split into a full-tile main loop
plus a narrowed tail loop. **Use distinct temp names in the tail** or they leak
into the outer scope and corrupt iter-arg detection.

```python
nvl = srv.vconfig(-1, 1)
Nfloor = (N // nvl) * nvl                     # largest multiple of VL ≤ N
acc = srv.vzero(f32)
for i in srv.range(0, Nfloor, nvl):           # main: full VL tiles, fast path
    vx = srv.vload(X, i, dtype=f32)
    acc = acc + vx
for i in srv.range(Nfloor, N, nvl):           # tail: runs 0 or 1 times
    nvl_t = srv.vconfig(N - i, 1)             # narrow VL to remaining elements
    tx = srv.vload(X, i, dtype=f32)           # DISTINCT name (tx, not vx)
    acc = acc + tx
```

Padded lanes in the tail read as `fill` (default 0.0). For ops where 0 is wrong
(e.g. min-reduce, softmax-max), pass `fill=`:
- `vreduce_min` / argmin tail: `vload(X, i, fill=1e38)`
- softmax exp-sum tail: `vload(X, i, fill=-1e38)` so `exp(-1e38-max)≈0`

---

## 4. Common patterns (copy these)

### 4.1 Reduce to scalar
```python
@srv.raw_kernel
def sum_kernel(X: srv.mem(f32), out: srv.mem(f32, out=True), N: srv.index):
    nvl = srv.vconfig(-1, 1); Nf = (N // nvl) * nvl
    acc = srv.vzero(f32)
    for i in srv.range(0, Nf, nvl):
        acc = acc + srv.vload(X, i, dtype=f32)
    for i in srv.range(Nf, N, nvl):
        nvl_t = srv.vconfig(N - i, 1)
        acc = acc + srv.vload(X, i, dtype=f32)
    srv.sstore(out, 0, srv.vreduce_sum(acc))
```

### 4.2 Mean / variance — use E[x²]-mean², NOT E[(x-mean)²]
> **Critical**: with fill-0 tail padding, `(0-mean)²=mean²` inflates variance.
> `E[x²]-mean²` is padding-safe because `0²=0`.
```python
acc_sum = srv.vzero(f32); acc_sq = srv.vzero(f32)
for i in srv.range(0, Nf, nvl):
    vx = srv.vload(X, i, dtype=f32)
    acc_sum = acc_sum + vx
    acc_sq  = acc_sq  + vx * vx
# (tail loop analogous)
mean = srv.vreduce_sum(acc_sum) / N
var  = srv.vreduce_sum(acc_sq) / N - mean * mean
```

### 4.3 Normalize (reduce → scalar → broadcast back to vector)
```python
inv = srv.rsqrt(srv.vreduce_sum(acc_sq))       # 1/||x||, scalar
for i in srv.range(0, Nf, nvl):
    nx = srv.vload(X, i, dtype=f32)
    srv.vstore(out, i, nx * inv)               # vec * scalar → auto-broadcast
```

### 4.4 Per-row 2D op (grid parallelism)
```python
@srv.raw_kernel
def row_kernel(X: srv.mem(f32), out: srv.mem(f32, out=True),
               M: srv.index, N: srv.index, row: srv.index):
    nvl = srv.vconfig(-1, 1)
    base = row * N                             # row offset into flat buffer
    # ... reduce X[base : base+N] ...

@triton.jit
def row_host(X, out, M, N):
    row = tl.program_id(0)
    if row < M:
        srv_call(row_kernel, outputs=[], inputs=[X, out, M, N, row])

# launch: row_host[(M,)](X.reshape(-1), out, M, N)   # grid=(M,)
```

### 4.5 Index-tracking reduce (argmax)
```python
lane = srv.viota()                             # [0,1,..,VL-1] as f32
best_val = srv.vload(X, 0, dtype=f32); best_idx = lane
for i in srv.range(0, Nf, nvl):
    vx = srv.vload(X, i, dtype=f32)
    idx = lane + srv.cast(i, f32)              # global indices this tile
    gt = vx > best_val
    best_val = srv.select(gt, vx, best_val)
    best_idx = srv.select(gt, idx, best_idx)
gmax = srv.vreduce_max(best_val)
masked = srv.select(best_val >= gmax, best_idx, srv.vzero(f32) + 1e30)
argmax = srv.vreduce_min(masked)               # first index achieving max
srv.sstore(out, 0, argmax)
```

### 4.6 Activation (elementwise transcendental)
```python
for i in srv.range(0, Nf, nvl):
    vx = srv.vload(X, i, dtype=f32)
    sig = 1.0 / (1.0 + srv.vexp(-vx))          # sigmoid
    srv.vstore(out, i, vx * sig)               # silu = x*sigmoid(x)
```

---

## 5. Gotchas (learned the hard way)

1. **`vconfig` before any vector op** — `vzero`/`vload` need VL set, else
   "svector op used before vconfig() set VL".
2. **Distinct temp names in tail loops** — reusing main-loop names leaks SSA
   across loop regions → dominance errors.
3. **Variance: E[x²]-mean²**, never E[(x-mean)²] (padding inflation, see 4.2).
4. **`vreduce_mul` crashes llc** — RVV has no hardware reduce-mul. Avoid.
5. **`viota` returns f32, not index** — index-element vectors don't lower to
   scalable; f32 composes with float lanes directly.
6. **Small-int host args**: triton may inline `G=1` as `constexpr`. The call
   layer handles this (emits `arith.constant`), so pass them normally.
7. **1D full-vector `vstore`** writes the whole VL — the codegen reinterpret-casts
   to a static `memref<VL×T>` so the store isn't clamped to lane 0.
8. **cumsum/scan**: sequential scalar loop with a running accumulator is the
   simplest correct form; the vectorized block-scan is currently slower on K3
   (single-thread dispatch) — prefer scalar unless multi-thread dispatch lands.

---

## 6. Running & verifying on K3

```bash
# On K3 (riscv64):
source /mnt_ai_ws2/zuoweixia/env175.sh
export PYTHONPATH=<worktree>/build-riscv64/:/mnt_ai_ws2/zuoweixia/triton312/lib/python3.12/site-packages
export TRITON_ALWAYS_COMPILE=1
rm -rf ~/.triton/cache ~/.cache/spine-triton     # clear stale cache
cd /tmp                                          # avoid write-permission issues
$PY -m pytest -p no:cacheprovider <test_file>.py -v --tb=short
```

The MLIR Python bindings (needed by `llvm_direct.py` / `mixed_bridge.py`) are
vendored into `triton/backends/spacemit/mlir_core` by setup.py
(`BackendInstaller.prepare` copies `${LLVM_SYSPATH}/python_packages/mlir_core`)
and resolved automatically at import time
(`language/smt_rvisa/_mlir_loader.py`) — no `PYTHONPATH` for `mlir_core` is
needed on a wheel/installed build. For a bare source-tree run, either export
`PYTHONPATH=<llvm>/python_packages/mlir_core` or place a copy at
`backend/mlir_core` (the source-tree layout the resolver checks).

**After editing `codegen.py`/`builtins.py`/`call_registry.py`**: sync the two
copies (source `language/smt_rvisa/` → `build-riscv64/.../smt_rvisa/` and
`build-x86_64/...`). Only `triton_shared.cc` changes require rebuilding
`libtriton.so`; pure-Python changes just need the file copy.

---

## 7. When you need a NEW primitive

Most operators compose from existing primitives (all 4 activation functions,
all 6 norms, softmax family — zero new primitives). Add a primitive only when
you need a new MLIR op. Steps:

1. **C++ binding** in `triton_shared.cc` (`create_xxx`), rebuild both arches.
2. **Marker** in `builtins.py`: `xxx = _SmtRvisaBuiltin("xxx")`.
3. **Codegen** in `codegen.py`: add name to `_SMT_RVISA_BUILTIN_NAMES`, add
   dispatch in `_gen_call_expr`, write `_gen_xxx` handler.
4. **Export** in `__init__.py`.

**Before adding**: check whether the MLIR op survives spine-mlir's
`ConvertToScalableVector` (only Extract/Insert/Reduction/ShapeCast/Splat/
TransferRead/Write are converted). If not (e.g. `vector.step`), build it from
`memref.alloc` + `scf.for` + `transfer_read` instead — see `viota`.

---

## 8. LLVM-direct primitives: `call_intrinsic` and the bypass path

### What and why

The **LLVM-direct path** is a second compilation route for kernels that use *only* LLVM-level primitives (`llvm_*` / `call_intrinsic`). Instead of emitting `tle.dsl_region` → linalg → spine-opt, these kernels emit a standalone top-level `llvm.func` module as **text**, bypassing triton-to-linalg and spine-opt entirely. The path goes: emit `llvm.func` → `mlir-translate` → `llc(riscv64)`.

**Why a second path?** LLVM ops inside a `func.func` body trip BufferDeallocation's "unknown memory side effects" error. The default path wraps your kernel body in a `func.func` (via `tle.dsl_region`), so LLVM ops can't survive there. The LLVM-direct bypass solves this by producing a *top-level* `llvm.func`, which is a no-op to BufferDeallocation / ConvertToScalableVector.

**When to use it:**
- You need a specific RVV intrinsic not wrapped by the high-level primitives (e.g. `llvm.riscv.vfwmacc`, special vsetvli sequences)
- You're prototyping new hardware instructions before writing a high-level wrapper
- You want full control over the LLVM IR (no linalg abstractions)

**Tradeoffs:**
- ✅ Zero libtriton rebuild (pure Python emitter)
- ✅ Direct LLVM IR control
- ❌ Lower level: you manage loop structure, iter-args, SSA yourself
- ❌ No automatic scalable-vector conversion (you write `vector<[8]xf32>` explicitly)
- ❌ Driver ABI passes rank-0 memref descriptors — `llvm_size` is unavailable, pass shapes as scalar params

### The primitives

| Primitive | Signature | Notes |
|-----------|-----------|-------|
| `call_intrinsic(name, args, result_type)` | Call an LLVM intrinsic | `name` = full intrinsic (e.g. `"llvm.riscv.vle"`), `args` = list of SSA values, `result_type` = MLIR type string or `"()"` for void. Returns the result SSA value. |
| `llvm_const(value, mlir_type)` | Emit a constant | `llvm_const(8, "i64")` = scalar int; `llvm_const("0.0", "vector<[8]xf32>")` = zero scalable vector splat. |
| `llvm_poison(mlir_type)` | Emit poison (uninitialized) | RVV load intrinsics require a "merge" operand (the old register value); poison = "don't care". |
| `llvm_base_ptr(mem)` | Extract data pointer from memref descriptor | Loads the rank-0 `{allocated, aligned, offset}` descriptor and returns `extractvalue[1]` (the aligned data pointer). |
| `llvm_gep(ptr, offset, elem_type)` | Pointer arithmetic | `llvm.getelementptr %ptr[%offset] : (!llvm.ptr, i64) -> !llvm.ptr, elem_type`. |
| `llvm_size(mem, dim)` | ❌ **unavailable** | The driver ABI passes rank-0 descriptors with no shape. Pass dimensions as scalar `index` params instead. |

**Note:** Standard LLVM ops like `llvm.fadd`, `llvm.fmul`, etc. are available via `call_intrinsic("llvm.fadd", [a, b], result_type="vector<[8]xf32>")`. No separate wrappers — `call_intrinsic` is the unified entry point.

### Example: MV with vle/vse

```python
import triton.language.extra.smt_rvisa as srv
from triton.language.extra.smt_rvisa import call as srv_call

@srv.raw_kernel
def llvm_direct_mv(A: srv.mem("f32"), B: srv.mem("f32"),
                   C: srv.mem("f32", out=True), K: srv.index):
    vl = srv.llvm_const(8, "i64")
    acc = srv.llvm_const("0.000000e+00", "vector<[8]xf32>")
    zero = srv.llvm_const(0, "i64")
    # Loop bound comes from the scalar K param (llvm_size unavailable in LLVM-direct ABI)
    for k in srv.range(zero, K, vl):
        pa = srv.llvm_poison("vector<[8]xf32>")
        pb = srv.llvm_poison("vector<[8]xf32>")
        ga = srv.llvm_gep(srv.llvm_base_ptr(A), k, "f32")
        gb = srv.llvm_gep(srv.llvm_base_ptr(B), k, "f32")
        va = srv.call_intrinsic("llvm.riscv.vle", [pa, ga, vl], result_type="vector<[8]xf32>")
        vb = srv.call_intrinsic("llvm.riscv.vle", [pb, gb, vl], result_type="vector<[8]xf32>")
        prod = srv.call_intrinsic("llvm.fmul", [va, vb], result_type="vector<[8]xf32>")
        acc = srv.call_intrinsic("llvm.fadd", [acc, prod], result_type="vector<[8]xf32>")
    gc = srv.llvm_base_ptr(C)
    srv.call_intrinsic("llvm.riscv.vse", [acc, gc, vl], result_type="()")

@triton.jit
def llvm_direct_mv_host(A, B, C, K):
    srv_call(llvm_direct_mv, outputs=[], inputs=[A, B, C, K])

# Launch
A = torch.arange(64, dtype=torch.float32)
B = torch.ones(64, dtype=torch.float32)
C = torch.zeros(8, dtype=torch.float32)
llvm_direct_mv_host[(1,)](A, B, C, 64)
```

**What happens:**
1. `runtime.py: _detect_llvm_direct` scans the AST for `llvm_*` / `call_intrinsic` → marks `fn._llvm_direct = True`
2. `call_registry.py: call()` detects the flag → calls `emit_llvm_direct_module(fn)` (text emitter) → stashes the `llvm.func` text in a process-global `_PENDING_LLVM_DIRECT_MODULE` dict
3. `compiler.py: make_ttir` retrieves it → stashes in `metadata["llvm_direct_module"]`
4. `compiler.py: _linalgdir_stage` detects the metadata key → returns the `llvm.func` text as-is (no triton-to-linalg)
5. `compiler.py: _llir_stage` detects the key → calls `_llvm_direct_to_llir` (only `mlir-translate`, skips `spine-opt`)
6. `so` stage is unchanged (llc → .o → .so)

**Critical: driver ABI for LLVM-direct kernels**

The driver (`backend/driver.py:192`) passes each memref as a **rank-0** `StridedMemRefType<char,0>` = `{allocated_ptr, aligned_ptr, offset}`. There are NO `sizes`/`strides` arrays. The data pointer is `extractvalue[1]` (aligned). `llvm_size` is unavailable → pass all dimensions as scalar `index` params and use them for loop bounds.

**Tests:** `python/tests/raw/test_llvm_direct_k3.py` (K3 numerical validation), `test_llvm_direct_emit.py` (x86→riscv64 .o chain). See commit `c599275` (initial) and `01ef396` (rename mode-1→llvm-direct).

### 8.1 The SPMD contract — offsets live INSIDE the kernel, not in `inputs`

`srv_call` looks like a runtime function call, but in LLVM-direct mode it is **not**.
Understanding why is the difference between correct code and a silent wrong answer.

**What actually happens.** `srv_call(fn, inputs=[...])` does *not* thread `inputs`
into the callee. It emits `fn` as a standalone top-level `llvm.func` and stashes the
text; `make_ttir` then **discards the entire host `@triton.jit` body** and swaps in
that module. At launch, the driver calls the emitted symbol with the **host kernel's
launch arguments**, positionally, each memref as a rank-0 descriptor with `offset=0`
(`driver.py:192`). So:

- The `inputs` list must be the kernel's **bare parameters, 1:1, in order** — it
  mirrors the launch args, nothing more. `call_registry.py` enforces the arity and
  raises `ValueError` on mismatch (fail-loud, not silent garbage).
- A **per-program offset cannot be passed from the host.** Writing
  `srv_call(fn, inputs=[A + row*K, ...])` does **not** work — `A + row*K` is a
  `tt.addptr` in the host body, which is thrown away. Every program would read `A[0]`.

**Why this is not a limitation.** This is exactly the CUDA/SPMD model: you pass the
**base pointer** `A` and compute the per-program slice **inside** the kernel from
`program_id`. You would never pass `A + blockIdx*K` to a CUDA kernel either.

**The idiom.** Compute offsets in the kernel with **natural Python arithmetic** —
the emitter lowers `*` / `+` / `-` to `llvm.mul` / `llvm.add` / `llvm.sub`, so no
`call_intrinsic` boilerplate is needed:

```python
@srv.raw_kernel
def gemv(A: srv.mem("f32"), B: srv.mem("f32"),
         C: srv.mem("f32", out=True), M: srv.index, K: srv.index):
    vl = srv.llvm_const(8, "i64"); eight = srv.llvm_const(8, "i64")
    zero = srv.llvm_const(0, "i64")
    row = srv.program_id(0)                 # ← per-program identity, the ONLY thing that varies
    acc = srv.llvm_const("0.000000e+00", "vector<[8]xf32>")
    for k in srv.range(zero, K, vl):
        ga = srv.llvm_gep(srv.llvm_base_ptr(A), row * K + k, "f32")   # natural arithmetic
        gb = srv.llvm_gep(srv.llvm_base_ptr(B), k, "f32")
        pa = srv.llvm_poison("vector<[8]xf32>"); pb = srv.llvm_poison("vector<[8]xf32>")
        va = srv.call_intrinsic("llvm.riscv.vle", [pa, ga, vl], result_type="vector<[8]xf32>")
        vb = srv.call_intrinsic("llvm.riscv.vle", [pb, gb, vl], result_type="vector<[8]xf32>")
        prod = srv.call_intrinsic("llvm.fmul", [va, vb], result_type="vector<[8]xf32>")
        acc = srv.call_intrinsic("llvm.fadd", [acc, prod], result_type="vector<[8]xf32>")
    gc = srv.llvm_gep(srv.llvm_base_ptr(C), row * eight, "f32")       # natural arithmetic
    srv.call_intrinsic("llvm.riscv.vse", [acc, gc, vl], result_type="()")

@triton.jit
def gemv_host(A, B, C, M, K):
    srv_call(gemv, outputs=[], inputs=[A, B, C, M, K])   # bare params, 1:1 — no A+offset

gemv_host[(M,)](A.flatten(), B, C, M, K)                 # grid=(M,): one program per row
```

**Uniform offsets** (same for every program) need no kernel arithmetic at all — slice
the tensor in Python before launch. The driver bakes the torch storage offset into
`data_ptr()`, so `fn_host[(1,)](X[128:], ...)` hands the kernel a base pointer already
advanced by 128 elements.

**Tests:** `test_llvm_direct_k3.py` — grid=4 GEMV (per-program `program_id` offset,
`max_err=0`) + fail-loud arity guard.

---

## 9. Debugging & IR inspection
