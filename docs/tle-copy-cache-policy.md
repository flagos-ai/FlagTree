# Native Cache Policies for TLE Copy and Global Memory Access

`tle.gpu.copy(..., cache_policy=policy)` and `tl.load(..., cache_policy=policy)` /
`tl.store(..., cache_policy=policy)` accept per-backend native cache-policy
objects that control the global-memory side of an access:

- **NVIDIA**: `tle.gpu.CachePolicy` and `tle.gpu.FractionalEvictionPolicy`
- **MThreads / MUSA**: `tle.gpu.mthreads.TMECachePolicy` (descriptor copies) and `tle.gpu.mthreads.LSUCachePolicy` (pointer copies and global load/store)
- **Hygon / HCU**: `tle.gpu.hcu.CachePolicy`
- **MetaX / MACA**: `tle.gpu.metax.CachePolicy`
- **PPU**: `tle.gpu.ppu.CachePolicy` (LSU pointer copies and global load/store) and `tle.gpu.ppu.AIUCachePolicy` (AIU descriptor and block-pointer copies)

Omit `cache_policy` or pass `None` to preserve the backend's default lowering.
There is no cross-backend portable intent mapping or `backend=` selector
argument; the compilation target must match the policy object's backend, memory
direction, and access path.

```python
import triton.language as tl
from triton.experimental.tle import language as tle

# NVIDIA TMA descriptor copies: L2 eviction policies per operand
first = tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy("evict_first"))
last = tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy("evict_last"))
tle.gpu.copy(a_desc, a_tile, [BM, BK], [m, k], cache_policy=last)
tle.gpu.copy(b_desc, b_tile, [BK, BN], [k, n], cache_policy=first)

# MThreads TME descriptor copy and LSU pointer / global store policy
tme = tle.gpu.mthreads.TMECachePolicy(inner="none", outer="persist")
lsu = tle.gpu.mthreads.LSUCachePolicy(inner_persistence=3, outer_persistence=3, l2_policy=1)

# PPU LSU load/store policy and AIU tensor-copy policy
ppu_lsu = tle.gpu.ppu.CachePolicy(".lu")  # equivalent to ppu.CachePolicy(5)
ppu_aiu = tle.gpu.ppu.AIUCachePolicy(kp=4)

# Hygon HCU and MetaX MACA pointer / global load and store policies
hcu_cs = tle.gpu.hcu.CachePolicy(".cs")
metax_cg = tle.gpu.metax.CachePolicy("cg")
```

Policy objects are immutable compile-time constants (`frozen=True`), compare
and specialize by normalized value, and can be constructed on the host or inside
`@triton.jit` functions (bind local variables with `policy: tl.constexpr = ...`
when forwarding through loops or `warp_specialize` workers). Cache hints do not
supply completion barriers or alter copy synchronization.

---

## 1. Supported Policy Objects by Backend and Path

| Backend | Pointer Access (`tl.load`, `tl.store`, `tle.gpu.copy`) | Descriptor Copy (`tle.gpu.copy`) |
| --- | --- | --- |
| **NVIDIA** | `tle.gpu.CachePolicy`: cache modifier (`ca`, `cg`, `cs`, `lu`, `cv` on load; `wb`, `cg`, `cs`, `wt` on store), L1 eviction, or static L2 eviction (`fraction=1.0`) | `tle.gpu.CachePolicy`: static L2 eviction only (`l2=FractionalEvictionPolicy(...)`), load and store |
| **MThreads / MUSA** | `mthreads.LSUCachePolicy`: native `inner_persistence` (`0..5`), `outer_persistence` (`0..5`), and `l2_policy` (`0..1`) | `mthreads.TMECachePolicy`: `inner` (`0..5` / `"none"`, `"once"`, `"normal"`, `"persist"`), `outer` (`0..3`), and load `prefetch_size` (`0`, `64`, `128`, `256`) |
| **Hygon / HCU** | `hcu.CachePolicy`: native global/buffer modifiers (`""`, `.ca`, `.cg`, `.cs`, `.cv` on load; `""`, `.wb`, `.cg`, `.cs` on store) | Unsupported (raises `ValueError`) |
| **MetaX / MACA** | `metax.CachePolicy`: native SDK modifiers (`ca`, `cg`, `cs`, `lu`, `cv` on load; `wb`, `cg`, `cs`, `wt` on store) | Unsupported (raises `ValueError`) |
| **PPU** | `ppu.CachePolicy`: native LSU modifiers (`""`, `.ca`, `.cg`, `.cs`, `.cv`, `.lu`, `.nc`, `.bl` on load; `""`, `.wb`, `.cg`, `.cs`, `.wt`, `.bl` on store)<br>`ppu.AIUCachePolicy`: eligible block-pointer loads (`tle.gpu.copy`) promoted to AIU (`kp=1..6`) | `ppu.AIUCachePolicy`: rank-2 and rank-3 host/device descriptor loads (`kp=1..6`, with `kp=3` aliasing `kp=2`) |

---

## 2. IR Representation, Pass-Through Rules, and Diagnostics

Both `tle.gpu.copy` and `tl.load` / `tl.store` normalize `cache_policy` into the
memory operation's formal IR attributes **before** creating the operation:

| Control | Formal IR Attribute |
| --- | --- |
| Standard NVIDIA / HCU / MetaX / PPU cache modifier | `cache` (`TT_CacheModifierAttr`, printed as `cacheModifier`) |
| NVIDIA L1-only or L2-only eviction priority | Scoped `evict` (`TT_EvictionPolicyAttr`, e.g. `l1_evict_last`, `l2_evict_first`) |
| Legacy `eviction_policy="evict_first"` / `"evict_last"` | Unscoped `evict` (`evict_first`, `evict_last`) |
| PPU explicit LSU default (`ppu.CachePolicy("")`) | `cache = ppu_lsu_default` |
| PPU AIU `kp=1..6` (`ppu.AIUCachePolicy(kp=N)`) | `cache = ppu_aiu_kpN` |
| MThreads LSU 3-tuple (`mthreads.LSUCachePolicy`) | `cache = musa_iN_oN_lN` (all $6 \times 6 \times 2$ tuples) |
| MThreads TME (`mthreads.TMECachePolicy`) | Existing MUSA TME persistence and prefetch operands |

### Mid-End Transport Independence and Final Instruction Conversion

1. **Zero Mid-End Interference**: Mid-end passes (`LowerLoops`,
   `ConvertToBufferOps`, `OptimizeLocalPointerAsyncStores`,
   `TlePlanLogicalDomains`, `TleLowerTmaCopy`, `TleDowngradeInvalidAsyncCopy`,
   and `TlePromoteAsyncLoadToAIU` for ordinary LSU policies) select transport
   channels solely from layout, vector width, alignment, and masking. Valid
   cache policies never block async copy promotion, HCU buffer conversion, or
   PPU AIU promotion; rewrites forward `getCache()` and `getEvict()` verbatim.
2. **Upstream-Style Instruction Conversion and Remarks**: At final instruction
   selection (`ToLLVM`), the chosen hardware instruction retains all controls
   it can encode and converts or omits unsupported controls according to the
   conversion matrix below. Whenever an explicit control is adjusted, the
   compiler emits an MLIR remark reporting the operation location, requested
   `(cache, evict)`, effective `(cache, evict)`, and reason. To view remarks,
   compile with `MLIR_ENABLE_DIAGNOSTICS=remarks` and a fresh
   `TRITON_CACHE_DIR`.

---

## 3. Cross-Backend Cache-Hint Conversion Matrix

| Backend | Memory Path / Lowered Op | Requested Policy / Modifier | Effective Lowered Control | Target Instruction / LLVM IR / PTX | Conversion & Remark Rule |
| --- | --- | --- | --- | --- | --- |
| **NVIDIA**<br>(`sm_80`+) | `tl.load` / `tl.store` / `tle.gpu.copy` (synchronous pointer) | `cache`: `""`, `ca`, `cg`, `cs`, `lu`, `cv` (load); `""`, `wb`, `cg`, `cs`, `wt` (store)<br>`l1`: `evict_normal`, `evict_first`, `evict_last`, `evict_unchanged`, `no_allocate`<br>`l2`: `evict_normal`, `evict_first`, `evict_last`, `evict_unchanged` | **1:1 pass-through** | `ld.global.{ca,cg,cs,lu,cv}.L1::*.L2::*`<br>`st.global.{wb,cg,cs,wt}.L1::*.L2::*` | Legacy `eviction_policy="evict_first"/"evict_last"` sets both L1 and L2; typed `CachePolicy(l1=..., l2=...)` controls L1 or L2 independently. |
| **NVIDIA** | `cp.async` (`AsyncCopyGlobalToLocalOp`, **16B**) | `cache`: `""`/`ca` $\to$ `ca`; `cg`/`cs`/`lu`/`cv` $\to$ **`cg`**<br>`l1`: any $\to$ **`evict_normal`**<br>`l2`: `evict_first`/`evict_last` | `cache`: `ca` or **`cg`**<br>`l1`: **`evict_normal`**<br>`l2`: **preserved at PTX $\ge$ 9.4**, **`evict_normal` at PTX $<$ 9.4** | `cp.async.{ca,cg}.shared.global[.L2::cache_hint]` (+ `createpolicy.fractional.L2::evict_*`) | 16B `cp.async` supports only `ca` and `cg`; `cs`/`lu`/`cv` convert to `cg`. L1 eviction is omitted. PTX $<$ 9.4 omits L2 policy to avoid a `ptxas` miscompilation (upstream #11660/#11848). |
| **NVIDIA** | `cp.async` (`AsyncCopyGlobalToLocalOp`, **4B / 8B**) | `cache`: any (`cg`/`cs`/`lu`/`cv`) $\to$ **`ca`** (default build)<br>`l1`: any $\to$ **`evict_normal`**<br>`l2`: preserved at PTX $\ge$ 9.4, omitted at PTX $<$ 9.4 | `cache`: **`ca`**<br>`l1`: **`evict_normal`**<br>`l2`: preserved at PTX $\ge$ 9.4 / omitted at $<$ 9.4 | `cp.async.ca.shared.global` (+ PTX $\ge$ 9.4 L2 policy) | 4B/8B `cp.async` does not support `cg`; default builds (`__FLAGTREE_PIPELINE_ENHANCE__`) convert non-`ca` modifiers to `ca` with a remark. |
| **NVIDIA** | TMA (`DescriptorLoadOp` / `AsyncTMACopyGlobalToLocalOp` / `AsyncTMACopyLocalToGlobalOp`) | `l2`: `FractionalEvictionPolicy("evict_first" \| "evict_last" \| ...)` | **1:1 pass-through** (`l2_evict_*`) | `cp.async.bulk.tensor.*.L2::cache_hint` | TMA consumes L2 eviction policies only; specifying `cache_modifier` or `l1` on a TMA descriptor copy raises a frontend `ValueError`. |
| **MThreads / MUSA** | `tl.load` / `tl.store` / `tle.gpu.copy` (LSU pointer) | `LSUCachePolicy(inner_persistence=0..5, outer_persistence=0..5, l2_policy=0..1)` | **`mp_31`**: **1:1 pass-through** (`musa_i*_o*_l*`)<br>**non-`mp_31`**: an explicit `LSUCachePolicy` is rejected | **`mp_31`**: `llvm.musa.lsu.{ld,st}.cache.hint.i`<br>**other targets**: omit the policy for default loads and stores | `None` fields resolve to SDK defaults (`inner=4, outer=2, l2=0(ld)/1(st)`). Explicit LSU policies require `mp_31` on ordinary accesses and pointer copies. |
| **MThreads / MUSA** | `memcpy.g2s` (`AsyncCopyGlobalToLocalOp`) | `LSUCachePolicy` or legacy `cache`/`evict` | **Converted to default `memcpy.g2s` (`NONE` / `NORMAL`)** | `llvm.musa.memcpy.g2s.*` | `memcpy.g2s` has no cache-hint operand; converts to default controls with a remark. |
| **MThreads / MUSA** | TME Descriptor Copy (`tle.gpu.copy`) | `TMECachePolicy(inner=..., outer=..., prefetch_size=...)` | **1:1 pass-through** | `llvm.musa.tme.{ld,st}.tile.*` | `prefetch_size` (`0, 64, 128, 256`) applies to loads; stores require `prefetch_size=0`. |
| **Hygon / HCU**<br>(`gfx928` / `gfx936` / `gfx938`) | `tl.load` / `tl.store` / `tle.gpu.copy` (Buffer & Global Ops) | `cache`:<br>• `""`/`.ca` (load), `""`/`.wb`/`.cg` (store) $\to$ `glc=0, slc=0`<br>• `.cg`/`.cs` (load), `.cs` (store) $\to$ `glc=1, slc=1`<br>• `.cv` (load), legacy `.wt` (store) $\to$ `glc=1, slc=0` | `cache`: **1:1 mapped to `glc`/`slc` bits** (and LLVM `nontemporal`/`volatile` on generic path)<br>`evict`: **`evict_normal`** on Buffer Ops | `buffer_{load,store}_*` (default `AMDGCN_USE_BUFFER_OPS=1`) or `global_{load,store}_*` | Buffer Ops omit `evict` with a remark. On generic non-buffer `gfx936`/`gfx938` global loads, `.cg`/`.cs` constrain the pointer to a VGPR so uniform loads are not scalarized to `s_load`. |
| **Hygon / HCU** | Direct-to-LDS (`AsyncCopyGlobalToLocalOp`) | `cache`: `.ca`/`""`, `.cg`, `.cs`, `.cv`<br>`evict`: any $\to$ **`evict_normal`** | `cache`: **1:1 mapped to `aux` cache bits**<br>`evict`: **`evict_normal`** | `llvm.amdgcn.raw.ptr.buffer.load.lds` | Hardware `buffer_load_lds` carries `glc`/`slc` bits and omits eviction priorities with a remark. |
| **MetaX / MACA** | `tl.load` / `tl.store` / `tle.gpu.copy` (synchronous pointer) | **Load**: `ca`/`cv` (default), `cg` (volatile), `cs`/`lu` (`!nontemporal` + `!metaxgpu.l2rp`)<br>**Store**: `wt` (default), `wb` (`!metaxgpu.devc`), `cg` (`!nontemporal`), `cs` (`!nontemporal` + `!metaxgpu.l2rp`) | **1:1 mapped via `CopyCacheTranslation`** | Default: `llvm.mxc.{ldg,stg}.predicator`<br>With `cg`/`cs`/`lu`/`wb`: inlined `__flagtree_metax_copy_{load,store}_*` helper | `CopyCacheTranslation` attaches `!nontemporal`, `!metaxgpu.l2rp`, and `!metaxgpu.devc` metadata during LLVM IR export. |
| **MetaX / MACA** | `AsyncCopyGlobalToLocalOp` | Any `cache` or `evict` | **Converted to default (`NONE` / `NORMAL`)** | `llvm.mxc.ldg.b*Async` | The selected async copy intrinsic uses its default cache controls and emits a remark. |
| **PPU**<br>(ZW810E) | `tl.load` / `tl.store` / `tle.gpu.copy` (LSU pointer) | **Load**: `""`/`.ca` (default), `.nc`(`kp1`), `.cg`/`.cv`(`kp2`), `.cs`(`kp4`), `.lu`(`kp5`), `.bl`(`kp6`)<br>**Store**: `""`/`.wb` (default), `.cg`/`.wt`(`kp2`), `.cs`(`kp4`), `.bl`(`kp6`) | **1:1 mapped to PPU LSU `kp` mode** | `ld.global.{ca,nc,cg,cs,lu,bl}`<br>`st.global.{wb,cg,cs,bl}` | `.ba` (`kp7`) and LSU `kp3` are rejected in the frontend. |
| **PPU** | `cp.async` (`AsyncCopyGlobalToLocalOp`) | Any LSU modifier / eviction policy | **16B**: **`cg`** (`ca` if explicitly `.ca`)<br>**4B / 8B**: **`ca`**<br>`evict`: **`evict_normal`** | `cp.async.{ca,cg}.shared.global` | Ordinary non-AIU async copies convert LSU modifiers to `cp.async` width controls and emit a remark. |
| **PPU** | AIU Tensor Copy (`AIUCachePolicy` / `TlePromoteAsyncLoadToAIU`) | • `AIUCachePolicy(kp=1..6)` $\to$ **1:1 pass-through** (`kp3` aliases `kp2`)<br>• Ordinary `ppu.CachePolicy` on a block-pointer load promoted to AIU $\to$ **`kp=1` (`DEFAULT_AIU_KP`)** | `AIUCachePolicy`: **`kp1..6`**<br>Promoted `CachePolicy`: **`kp1`** | `vmem.aiu.ld.tsm...kpN` | `AIUCachePolicy(kp=7)` is rejected for read-after-write visibility safety on SDK 2.1. Promoted LSU policies convert to `kp=1` with a remark. |

---

## 4. Backend-Specific Policy Reference

### 4.1 NVIDIA (`tle.gpu.CachePolicy`, `tle.gpu.FractionalEvictionPolicy`)

`CachePolicy` and `FractionalEvictionPolicy` are exported directly from
`triton.experimental.tle.language.gpu`. Calling `CachePolicy()` with no
arguments raises `ValueError`; pass `None` to retain the default behavior.

```python
from triton.experimental.tle.language.gpu import CachePolicy, FractionalEvictionPolicy

l2_retain = CachePolicy(l2=FractionalEvictionPolicy("evict_last"))
l2_stream = CachePolicy(l2=FractionalEvictionPolicy("evict_first"))
l1_stream = CachePolicy(l1="evict_first")
l1_bypass = CachePolicy(cache_modifier="cg")
```

- **Cache modifiers (`cache_modifier`)**:
  - Loads accept `"ca"`, `"cg"`, `"cs"`, `"lu"`, `"cv"` (with or without a leading dot).
  - Stores accept `"wb"`, `"cg"`, `"cs"`, `"wt"` (with or without a leading dot).
- **L1 eviction (`l1`)**: Accepts `"evict_normal"`, `"evict_first"`,
  `"evict_last"`, `"evict_unchanged"`, and `"no_allocate"`. `cache_modifier`
  and explicit `l1` cannot be combined.
- **L2 eviction (`l2`)**: Accepts
  `FractionalEvictionPolicy(primary, fraction=1.0, secondary="evict_unchanged")`
  with `primary` in `{"evict_normal", "evict_first", "evict_last", "evict_unchanged"}`.
  Only `fraction=1.0` and `secondary="evict_unchanged"` are supported.
  Simultaneous `l1` and `l2` eviction settings and non-`None`
  `l2_prefetch_size` are rejected. `cache_modifier` can be combined with `l2`
  on pointer accesses.
- **TMA copies**: Accept `l2` policies only (`sm_90`+).
- **Async-copy L2 policies**: TLE preserves the policy through copy planning,
  including unmasked copies and straight-line warp-specialized producers.
  NVIDIA emits the L2 operand for PTX 9.4 and newer, where the assembler
  correctly initializes the address descriptor. Older targets omit it at
  instruction lowering.

The H800 regression test copies one unmasked 16 × 32 FP32 tile through
shared memory and checks all 512 elements. To reproduce the assembler fault
on SM90a, set `PTXAS_FIXED` to ptxas 13.4.92 and `PTXAS_OLD` to ptxas
12.8.93. Run these commands from the checkout with its FlagTree build installed.

```bash
work=$(mktemp -d)
test_id=python/test/tle/integration/test_tle_nvidia_copy_cache_policy.py::test_unmasked_block_copy_preserves_l2_until_instruction_lowering
export TRITON_PTXAS_PATH="$PTXAS_FIXED"
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$work/dump" TRITON_CACHE_DIR="$work/fixed-cache" \
  python -m pytest -q "$test_id"

# Both assemblers accept PTX 8.7. Keep the instructions and change only the header.
ptx=$(find "$work/dump" -name '_block_copy.ptx' -print -quit)
key=$(basename "$(dirname "$ptx")")
mkdir -p "$work/override/$key"
sed 's/^\.version 9\.4$/.version 8.7/' "$ptx" > "$work/input.ptx"
"$PTXAS_OLD" -arch=sm_90a "$work/input.ptx" -o "$work/override/$key/_block_copy.cubin"
TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR="$work/override" TRITON_CACHE_DIR="$work/old-cache" \
  python -m pytest -q "$test_id"
```

With the same hinted PTX input, ptxas 12.8.93 emitted
`LDGSTS.E.BYPASS.128 [R9+UR0], desc[UR1][R2.64]` without defining
`UR0` or `UR1`. Running that cubin on H800 raised CUDA error 715
(`cudaErrorIllegalInstruction`). Assembling the same input with 13.4.92
produced `[R9+UR4], desc[UR5][R2.64]` with both registers initialized and
copied all 512 elements correctly. The normal FlagTree path also passed the
test with 12.8.93 by omitting the L2 operand at PTX 8.7.

### 4.2 MThreads / MUSA (`TMECachePolicy`, `LSUCachePolicy`)

- **`tle.gpu.mthreads.TMECachePolicy(inner="normal", outer="normal", prefetch_size=0)`**:
  - Controls descriptor transfers (`tle.gpu.copy`).
  - `inner` accepts `"none"`, `"once"`, `"normal"`, `"persist"`, or integer codes `0..5` (`0..3` normalize to the four strings).
  - `outer` accepts `"none"`, `"once"`, `"normal"`, `"persist"`, or integer codes `0..3`.
  - `prefetch_size` accepts `0`, `64`, `128`, or `256` bytes for descriptor loads, and requires `0` for descriptor stores.
- **`tle.gpu.mthreads.LSUCachePolicy(inner_persistence=None, outer_persistence=None, l2_policy=None)`**:
  - Controls pointer copies (`tle.gpu.copy`) and ordinary `tl.load` / `tl.store`.
  - `inner_persistence` and `outer_persistence` accept `None` or integers in `0..5`; `l2_policy` accepts `None` or integers in `0..1`.
  - `None` resolves to the direction's SDK default (`inner=4, outer=2, l2=0` for loads; `inner=4, outer=2, l2=1` for stores).
  - Emits native `llvm.musa.lsu.{ld,st}.cache.hint.i` on `mp_31`; an explicit policy on any other MUSA target raises a compilation error. Omitting the policy keeps the default instruction.

### 4.3 Hygon / HCU (`tle.gpu.hcu.CachePolicy`)

- **`tle.gpu.hcu.CachePolicy(cache_modifier="")`**:
  - Loads accept `""`, `".ca"`, `".cg"`, `".cs"`, `".cv"`.
  - Stores accept `""`, `".wb"`, `".cg"`, `".cs"`.
  - On the default buffer-op path (`AMDGCN_USE_BUFFER_OPS=1`) and direct-to-LDS path, modifiers map to hardware `glc`/`slc` control bits (on `gfx936`: `.cg`/`.cs` load and `.cs` store set `glc slc`; `.cv` load sets `glc`; `.ca` load and `.wb`/`.cg` store use default `0`).
  - On the generic global path, `.cg`/`.cs` loads and `.cs` stores attach LLVM `!nontemporal`, `.cv` loads attach `volatile`, and `gfx936`/`gfx938` `.cg`/`.cs` loads constrain the pointer to a VGPR so uniform loads are not scalarized to `s_load`.
  - Legacy `tl.store(..., cache_modifier=".wt")` remains supported on `tl.store` with its existing wait behavior, while `hcu.CachePolicy` rejects `".wt"`.

### 4.4 MetaX / MACA (`tle.gpu.metax.CachePolicy`)

The standard `FLAGTREE_BACKEND=metax` wheel links the in-tree cache lowering into
`libtriton`. It ships the SDK library alongside it for the vendor MCTLE passes.

- **`tle.gpu.metax.CachePolicy(cache_modifier)`**:
  - Loads accept `"ca"`, `"cg"`, `"cs"`, `"lu"`, `"cv"` (leading dot optional).
  - Stores accept `"wb"`, `"cg"`, `"cs"`, `"wt"` (leading dot optional).
  - Load `"cg"` emits a volatile load; load `"cs"`/`"lu"` emit `!nontemporal` plus `!metaxgpu.l2rp` through an inlined LLVM helper (`__flagtree_metax_copy_load_*`); load `"ca"`/`"cv"` keep the default predicator path.
  - Store `"wb"` emits `!metaxgpu.devc`, store `"cg"` emits `!nontemporal`, and store `"cs"` emits `!nontemporal` plus `!metaxgpu.l2rp` through `__flagtree_metax_copy_store_*`; store `"wt"` keeps the default store path.

### 4.5 PPU (`tle.gpu.ppu.CachePolicy`, `tle.gpu.ppu.AIUCachePolicy`)

- **`tle.gpu.ppu.CachePolicy(cache_modifier="")`**:
  - Controls LSU pointer copies (`tle.gpu.copy`) and ordinary `tl.load` / `tl.store`.
  - Loads accept `""`, `".ca"`, `".cg"`, `".cs"`, `".cv"`, `".lu"`, `".nc"`, `".bl"`, or integer aliases `1, 2, 4, 5, 6` (`.nc` $\to$ `kp1`, `.cg`/`.cv` $\to$ `kp2`, `.cs` $\to$ `kp4`, `.lu` $\to$ `kp5`, `.bl` $\to$ `kp6`).
  - Stores accept `""`, `".wb"`, `".cg"`, `".cs"`, `".wt"`, `".bl"`, or integer aliases `2, 4, 6` (`.cg`/`.wt` $\to$ `kp2`, `.cs` $\to$ `kp4`, `.bl` $\to$ `kp6`).
  - `".ba"` (`kp7`) and LSU `kp3` are rejected.
- **`tle.gpu.ppu.AIUCachePolicy(kp=1)`**:
  - Controls asynchronous AIU tensor copies (`tle.gpu.copy`) for supported rank-2 and rank-3 host/device tensor descriptors and contiguous block-pointer loads (`kp` in `1..6`, where `kp=3` aliases `kp=2` on SDK 2.1, and `kp=7` is rejected).
  - Rank-2 descriptors require packed row-major storage; rank-3 descriptors lower to one 2D AIU transfer per leading slice with the requested `kp`.

---

## 5. Ordinary Global Loads and Stores (`tl.load` / `tl.store`)

`tl.load(pointer, mask=None, other=None, ..., cache_policy=None)` and
`tl.store(pointer, value, mask=None, ..., cache_policy=None)` accept the same
pointer-path native policy objects as `tle.gpu.copy`:

- **Accepted policy classes**: `tle.gpu.CachePolicy` (NVIDIA),
  `tle.gpu.mthreads.LSUCachePolicy` (MThreads), `tle.gpu.hcu.CachePolicy` (Hygon),
  `tle.gpu.metax.CachePolicy` (MetaX), and `tle.gpu.ppu.CachePolicy` (PPU).
- **Copy-only policy rejection**: `TMECachePolicy` and `AIUCachePolicy` apply
  only to `tle.gpu.copy` and raise `ValueError` if passed to `tl.load` or
  `tl.store`.
- **Mutual exclusion with legacy string arguments**: An explicit `cache_policy`
  object cannot be combined with a non-empty `cache_modifier` or
  `eviction_policy` argument (and on `tle.gpu.copy`, `cache_policy` cannot be
  combined with `eviction_policy`).
- **Address-space validation**: `cache_policy` requires a global-memory pointer
  (address space `1`) and preserves register-valued `tl.load` / `tl.store`
  semantics without allocating shared memory.
