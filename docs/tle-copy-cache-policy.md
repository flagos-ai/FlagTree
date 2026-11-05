# Per-copy native cache policies

`tle.gpu.copy(..., cache_policy=policy)` controls the global-memory access of
one copy. `tle.gpu.CachePolicy` always means the NVIDIA native policy, even
when importing the module on another platform. MThreads, PPU, HCU, and MetaX
use their own namespaced policy objects. The compilation target must match
the object's backend and supported copy path.

Omit the argument or pass Python `None` to preserve the backend's defaults.
There is no common intent policy, target-selecting `backend` argument, or
automatic conversion between backends. NVIDIA `CachePolicy()` requires an
explicit hint and raises an error; use `None` for the default. Other native
default constructors keep their existing meanings and are not folded away.

Policies are immutable, compare and specialize by normalized value,
and support host/JIT construction and constexpr worker forwarding. Policy
fields must be compile-time values. Hints do not supply a
barrier, change completion or synchronization, or guarantee cache residency.

```python
first = tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy("evict_first"))
last = tle.gpu.CachePolicy(l2=tle.gpu.FractionalEvictionPolicy("evict_last"))
tle.gpu.copy(a_desc, a_tile, [BM, BK], [m, k], cache_policy=first)
tle.gpu.copy(b_desc, b_tile, [BK, BN], [k, n], cache_policy=last)

tme = tle.gpu.mthreads.TMECachePolicy(inner="none", outer="persist")
lsu = tle.gpu.mthreads.LSUCachePolicy(inner_persistence=5, outer_persistence=2, l2_policy=0)
ppu_lsu = tle.gpu.ppu.CachePolicy(5)  # equivalent to ppu.CachePolicy(".lu")
ppu_aiu = tle.gpu.ppu.AIUCachePolicy(kp=5)
hcu = tle.gpu.hcu.CachePolicy(".cs")
metax = tle.gpu.metax.CachePolicy("cs")
```

Each object applies only on its corresponding backend and path. Similar
modifier spellings across platforms do not establish equivalent ISA semantics.
MUSA `"none"` is a native persistence value, distinct from Python `None`,
explicit normal, and a verified bypass.

## Supported policy objects

| Backend | Pointer copy | Descriptor copy |
| --- | --- | --- |
| NVIDIA | `tle.gpu.CachePolicy`: cache modifier, single-level L1 or static L2 eviction | Same object, L2 policy only; load and store |
| MThreads | `mthreads.LSUCachePolicy`: native inner/outer persistence and L2 controls | `mthreads.TMECachePolicy`: inner/outer persistence; load and store |
| PPU | `ppu.CachePolicy`: native LSU modifier; `ppu.AIUCachePolicy`: supported block-pointer loads lowered to native AIU tensor copies | `ppu.AIUCachePolicy`: rank-2/rank-3 host or device descriptor loads; rank-3 uses 2D AIU slices; kp1–kp6, with kp3 aliasing kp2 |
| HCU gfx936 | `hcu.CachePolicy`: generic global-memory modifier | Explicit policy rejected |
| MetaX | `metax.CachePolicy`: SDK memory modifier | Explicit policy rejected |

An object for another backend or an unsupported direction raises an error.
CommonIR and unsupported backends reject explicit native policies.
Transport selection uses the existing layout, width and scheduling rules.
Hints do not prevent an async, AIU or HCU buffer conversion. The selected
instruction keeps the controls it supports and adapts or omits the rest using
the rules below. An adjustment emits an MLIR remark with the operation
location, requested controls, effective controls and reason. Remarks are
compiler diagnostics, not a persistent runtime warning. Triton's normal
diagnostic filter hides remarks; set the existing
`MLIR_ENABLE_DIAGNOSTICS=remarks` environment variable and use a fresh
`TRITON_CACHE_DIR` to see them. Cached kernels do not compile again. Inspect
the final native instructions when validating a hint.
Mapping to a default is not a claim of equivalent cache behavior.

These rules apply after frontend normalization. Legacy string arguments keep
their own accepted values: for example, NVIDIA `tl.load(cache_modifier=...)`
accepts `""`, `.ca`, `.cg`, `.cv`; `.cs` and `.lu` in this branch use the
native `CachePolicy` object. The conversion does not make an invalid legacy
string valid. When both spellings request the same modifier, they produce
the same `cache` enum. For example, on NVIDIA
`tl.load(p, cache_modifier=".cg")` and
`tl.load(p, cache_policy=tle.gpu.CachePolicy("cg"))` follow identical
lowering. Explicit NVIDIA L1 and L2 eviction still have separate encodings.

Invalid values, backend/direction mismatches, unsupported explicit descriptor
paths, and unsafe PPU modes still fail. Structural fallbacks, such as a copy
too narrow for an async instruction, follow the same rules with and without a
hint and retain the original fields on the replacement load/store. No cache
flush, fence, barrier or synchronization change is introduced by conversion.

## NVIDIA

```python
from triton.experimental.tle.language.gpu import CachePolicy, FractionalEvictionPolicy

reused = CachePolicy(l2=FractionalEvictionPolicy("evict_last"))
streaming = CachePolicy(cache_modifier="cs")
tle.gpu.copy(a_desc, a_smem, [BM, BK], [m, k], cache_policy=reused)
# A surviving register load uses cs; a selected cp.async uses the table below.
b = tl.load(b_ptrs, mask=valid, other=0, cache_policy=streaming)
```

Pointer loads accept `ca`, `cg`, `cs`, `lu`, and `cv`; stores accept `wb`, `cg`,
`cs`, and `wt`. A leading dot is optional. These are PTX cache operators:

| Operator | PTX meaning |
| --- | --- |
| load `ca` | Cache at all levels; ordinary load default |
| load/store `cg` | Cache at the global level, bypass L1 |
| load/store `cs` | Streaming access with evict-first allocation |
| load `lu` | Last-use indication; global-memory behavior follows `cs` |
| load `cv` | For system memory, invalidate matching L2 lines and refetch |
| store `wb` | Write back at all coherent levels; ordinary store default |
| store `wt` | Write through to system memory |

`l1` accepts `evict_normal`, `evict_first`, `evict_last`, `evict_unchanged`,
and `no_allocate`. Cache modifiers and explicit L1 eviction are separate
alternatives and cannot be combined in this API. They are not TMA controls.

`l2=FractionalEvictionPolicy(primary, fraction=1.0, secondary="evict_unchanged")`
accepts primary `evict_normal`, `evict_first`, `evict_last`, or
`evict_unchanged`. Only `fraction=1` and `secondary="evict_unchanged"` are
supported. A non-unit fraction, a different secondary priority, non-None
`l2_prefetch_size`, or setting both `l1` and `l2` raises an error.
The retained constructor names make unsupported old configurations explicit.
Cache modifier plus L2 eviction remains supported where the instruction allows it.
The compiler reuses `createpolicy.fractional` with a constant 1.0, without
hard-coding a hardware policy word. Explicit eviction requires SM80+;
TMA requires SM90+.

### NVIDIA conversion rules

For ordinary `ld.global`/`st.global`, the native modifier and scoped eviction
fields retain their existing lowering. TMA preserves its L2 policy and still
rejects explicit cache operators or L1 fields. TMA has a separate instruction
contract from non-bulk `cp.async`.

For a selected non-bulk `cp.async`, the final per-thread transfer width decides
the cache operator, after layout/vectorization and mask analysis:

| Requested load modifier | 16 bytes | 4 or 8 bytes |
| --- | --- | --- |
| `ca` | `ca` | `ca` |
| Omitted, `cg`, `cs`, `lu`, `cv` | `cg` | `ca` |

This follows upstream Triton's instruction selection. In particular,
`cs -> cg` is an instruction-specific adaptation, not equivalent streaming
semantics. `cg -> ca` for a narrow copy no longer produces a compile error.

| Requested eviction | Effective non-bulk `cp.async` control |
| --- | --- |
| None | None |
| Native L1 `evict_normal/first/last/unchanged/no_allocate` | Omitted; this instruction has no matching L1 eviction operand |
| Native L2 `evict_normal/first/last/unchanged` | L2 policy at PTX 9.4+, otherwise omitted |
| Legacy `eviction_policy="evict_first"/"evict_last"` | L2 policy at PTX 9.4+, otherwise omitted; the legacy values are not L1-only |

The PTX gate follows upstream's ptxas cache-hint operand workaround, not the
first ISA version that introduced the operand. It reuses the NVIDIA target's
already selected PTX version at final lowering; no `ttg.ptx-version` attribute
or extra compiler option is needed.

TLE additionally omits L2 hints on async copies produced by its
load-to-local-pointer-store rewrite when the copy is unmasked or belongs to
a loop-free warp-specialize partition. This guards against invalid uniform
register uses seen on H800 and applies even at PTX 9.4+. Other async copies,
including unmasked pipeline copies, use the PTX rule in the table.

Cache policies are compile-time values. Construct them on the host or inside
a JIT function using compile-time arguments. Dynamic fractional and
address-range policies are not part of this API.

CUDA access-policy windows, set-aside budgets, external policy-handle imports
and separate prefetch/discard instructions are outside this per-copy API.
`ld.global.nc` also remains excluded: its noncoherent read path requires an
additional immutable-data contract, unlike ordinary advisory cache hints.
See the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

## MThreads

`TMECachePolicy(inner="normal", outer="normal")` retains PR #1385's
four public values: `none`, `once`, `normal`, and `persist`. Both descriptor
directions carry their policy. The names describe MUSA persistence hints;
they do not guarantee bypass, a precise reuse count, or locked residency.
Python `None` means use the backend default and differs from `"none"`.
Integer aliases 0/1/2/3 normalize to those four strings for both fields,
including equality and JIT specialization. Inner accepts integers 0–5;
outer accepts 0–3. `TMECachePolicy(inner=0, outer=3)` therefore equals
`TMECachePolicy(inner="none", outer="persist")`.
`inner` also accepts native integer codes `4` and `5`, verified on SDK 5.2/mp31;
their replacement semantics are not assigned invented portable names.
TME loads additionally accept `prefetch_size=0`, `64`, `128`, or `256` bytes.
TME stores require zero. The SDK rejects outer codes 4/5 for both directions,
so they are not exposed.

`LSUCachePolicy(inner_persistence=None, outer_persistence=None, l2_policy=None)`
uses native SDK integer controls. Inner/outer accept 0–5 and L2 accepts 0–1.
The ordinary LSU defaults differ from TME: SDK loads use inner 4, outer 2,
L2 0; stores use inner 4, outer 2, L2 1. The lowering preserves the SDK's
direction-specific coherence parameter instead of exposing it as an eviction
hint. Values 4/5 are native codes, not promises of a portable retention
behavior. The four-value TME vocabulary must not be used to infer all LSU
semantics.

For ordinary LSU instructions these fields retain their SDK encoding. TME
retains the explicit inner/outer/prefetch controls. An ordinary LSU load
selected for `memcpy.g2s` uses that instruction's existing default controls:
the LSU tuple and eviction hint are omitted with a remark. LSU controls are
not reinterpreted as TME persistence, and a hint does not prevent async
selection. Passing the wrong native policy type for an explicit copy path
still raises an error.

The direct load-to-shared-store rewrite keeps a register load for a nonzero
masked `other`: `memcpy.g2s` only zero-fills and this rewrite has no post-load
select. This is a data-semantics restriction shared by default and hinted
copies, independent of cache policy. The separate async-load frontend already
restores nonzero `other` through a select after its local load.

## PPU

PPU has native asynchronous tensor-copy instructions through AIU, with a
per-copy `kp` cache field. The block pointer is a frontend representation:
the supported path lowers `tt.load` to `AIULoadOp`, then to
`AsyncAIUCopyGlobalToLocalOp` and a native `vmem.aiu.ld.tsm...kpN` instruction.
It therefore uses the tensor-copy engine even though the Python operand is
not a `tl.tensor_descriptor`.

`tle.gpu.copy(descriptor, shared, shape, offsets, cache_policy=...)` now
accepts both `tl.make_tensor_descriptor` and host `TensorDescriptor` inputs.
The PPU frontend preserves the descriptor's base, shapes, strides and offsets
through descriptor expansion into the native AIU path. It does not emit the
unsupported NVIDIA-style `ttg.tma_copy` operation. With `cache_policy=None`,
this newly supported path uses native kp1.

On PPU targets that expand tensor descriptors to pointers, an AIU descriptor load requires
descriptor expansion before block-pointer promotion. Modules without such
a descriptor keep the original pass order, including legacy async block-pointer
loads. The early rewrite is scheduled at module scope in mixed kernels.

Rank-2 descriptors require packed row-major storage. Rank-3 descriptors are
split into one 2D AIU copy per leading tile slice, with the requested kp on
every slice; this is not a single native 3D instruction. The native SDK also
has a 3D AIU helper, which is separate from this implementation. The two
innermost dimensions must be packed; the leading stride is retained.

Load modifiers are `""`, `.ca`, `.cg`, `.cs`, `.cv`, `.lu`, `.nc`, and `.bl`.
Store modifiers are `""`, `.wb`, `.cg`, `.cs`, `.wt`, and `.bl`.
These are native PPU names. On SDK 2.1/ppu0010, `.nc` emits kp1,
`.cg`/`.cv` loads and `.cg`/`.wt` stores emit kp2, `.cs` emits kp4,
`.lu` kp5, and `.bl` kp6. Default/`.ca` loads and
default/`.wb` stores use the default mode. Aliases do not represent distinct
measured hardware behaviors.

`ppu.CachePolicy(1/2/4/5/6)` are load aliases for `.nc/.cg/.cs/.lu/.bl`;
only 2/4/6 are valid stores. For example, `CachePolicy(5)` equals
`CachePolicy(".lu")`; it does not mean L1 mode 5. Ordinary LSU kp3 is rejected
even though AIU kp3 aliases kp2.

The SDK can encode `.ba` (kp7), but loads failed after GPU input updates and
stores failed with prewarmed output buffers. Both directions are rejected in
the Python frontend and C++ lowering. Integer 7 is also rejected.
No NVIDIA-style eviction or prefetch modifier is exposed: the tested SDK
ignores those modifiers.

If an ordinary LSU load is promoted to AIU, its LSU modifier uses the AIU
default path instead of being translated by numeric coincidence. Explicit
`AIUCachePolicy` keeps kp1–kp6 (including the SDK kp3→kp2 alias). Ordinary
non-AIU asynchronous copy uses its existing default instruction controls;
LSU modifiers and eviction hints are omitted with a remark. Promotion is not
blocked by a legal LSU hint. An explicit AIU object on an unsupported shape
or path still fails the structural checks below.
Specifically, the ordinary async TIX emitter uses `cp.async.cg` for a 16-byte
transfer and `cp.async.ca` for a 4/8-byte transfer, irrespective of the LSU
modifier. These compatibility spellings do not assert NVIDIA cache semantics.

`AIUCachePolicy(kp=1)` through `kp=6` selects native PPU0010 AIU modes for
the supported descriptor and block-pointer loads. These are separate from ordinary LSU
modifiers; kp3 is a kp2 alias on SDK 2.1. The existing swizzled load and
acp commit/wait protocol are retained. Tiles require 16..2048 rows and at
least 32 contiguous bytes per row, with 8/16/32-bit elements. The two AIU
dimensions must be in `1..2**23`. X/Z offsets must be nonnegative and the
whole tile must fit the 24-bit coordinate range. Constant invalid layouts
and coordinates are rejected; dynamic constraints have runtime assertions.
Upper-bound tail tiles are zero-filled. Negative leading-plane offsets in a
rank-3 descriptor are handled by zero-filled slices; negative X/Z offsets
are rejected because they faulted on the tested native path. Contiguous
column-major rank-2 block pointers remain supported through their order.
Descriptor block shape, copy shape and shared shape must match exactly.
Only zero padding is supported. Stores, ordinary pointer tensors, logical
non-power-of-two descriptors, explicit TMA barriers and PPU0015 are rejected.
AIU completion is internal commit/wait, not an exposed asynchronous token.

```python
from triton.experimental.tle.language.gpu.ppu import AIUCachePolicy

desc = tl.make_tensor_descriptor(a, [M, N], [N, 1], [32, 32])
tile = tle.gpu.alloc((32, 32), tl.float32, nv_mma_shared_layout=False)
tle.gpu.copy(desc, tile, (32, 32), (row, col),
             cache_policy=AIUCachePolicy(kp=4))
```

**kp7 exists in hardware/SDK, but this API rejects it for AIU copies.**
On SDK 2.1/ZW810E, overwriting an input on the GPU and then copying it with
AIU kp7 returned stale data on block-pointer, device-descriptor and
host-descriptor paths. The same probe with CPU-uploaded inputs passed.
Earlier tests that read input back to the CPU before AIU did not expose this
problem; they are not evidence of general kp7 coherence. No cache flush or
mapping to another kp is silently inserted. The exact replacement/coherence
semantics of kp4–kp7 remain undocumented in the sources available here.

The native AIU intrinsic uses the SDK's typed-pointer ABI. A packaged helper
performs pointer conversion and is assembled/linked by SDK `llvm-as` and
`llvm-link`; the helper is inlined into the final kernel. The compiler cache
key includes the helper's content hash. No generated LLVM IR is patched to
insert cache controls.

## Choosing and validating a policy

Apply requests separately to A, B, and the output copy. A large input read
once can be a streaming candidate when it competes with repeatedly reused
data; the reused input can be a retention candidate. A write-once output can
be a streaming candidate when it is not consumed soon from the cache.
These are workload hypotheses, not default recommendations. In particular,
do not interpret MThreads `none` as a universally beneficial streaming mode.

Compare the complete kernel with unchanged math, tiling and synchronization.
Inspect final native instructions and confirm the compared kernels use the
intended transport. Check GPU-updated inputs, prewarmed outputs and
masked/tail copies before measuring. Repeat timings in independent processes,
with both reused buffers and buffer rotation representative of production.
A fixed-buffer win can disappear with rotation. API equivalence and expression
coverage are not additional performance gains. This API does not change global
cache budgets, stream access-policy windows, or production operator policies.

## HCU gfx936

Load modifiers are `""`, `.ca`, `.cg`, `.cs`, `.cv`; store modifiers are
`""`, `.wb`, `.cg`, `.cs`. `.cg`/`.cs` loads and `.cs` stores select the
generic LLVM non-temporal path; `.cv` loads additionally use volatile.
`.ca` loads and `.wb`/`.cg` stores are default aliases. They do not imply
NVIDIA cache-level semantics. If the optimizer selects buffer/direct-to-LDS
conversion, it forwards the existing `cache` enum to the target's buffer
control-bit mapping. On gfx936 that mapping produces zero/default bits for
these modifiers. Eviction priorities are omitted because buffer ops do not
carry them. A remark reports these adjustments; compilation continues on the
selected buffer path. A hint neither forces the generic path nor inhibits
buffer conversion. This also means a former speedup caused by forcing generic
memory cannot be attributed to a retained buffer cache hint.
For generic gfx936 global `.cg`/`.cs` loads, the address is constrained to a vector register so
uniform-load scalarization does not discard the cache controls. This can add
a scalar-to-vector move or increase register pressure; no additional volatile
semantics are introduced.
The HCU policy object accepts gfx936 global accesses and the modifiers above.
The legacy `tl.store(..., cache_modifier=".wt")` also remains available and
retains its existing wait lowering. The native HCU object does not accept `.wt`.

## MetaX

Load modifiers are `ca`, `cg`, `cs`, `lu`, `cv`; stores accept `wb`, `cg`,
`cs`, `wt`. A leading dot is optional. The MACA SDK mapping is:

| Direction/mode | Compiler control |
| --- | --- |
| load `ca`/`cv` | Default aliases |
| load `cg` | Volatile |
| load `cs`/`lu` | Non-temporal plus L2 replacement metadata |
| store `wb` | Device metadata |
| store `cg` | Non-temporal |
| store `cs` | Non-temporal plus L2 replacement metadata |
| store `wt` | Default alias |

Metadata is attached through LLVM translation and an inlined LLVM helper,
not by textual rewriting of LLVM IR. Copy masks and vector types are
preserved. Descriptor policies remain unsupported. A load selected for native
asynchronous copy keeps that instruction and uses its default cache controls;
the modifier and eviction hint are omitted with a remark because the current
intrinsic has no verified cache-control operand.

Loads with `cg/cs/lu` and stores with `wb/cg/cs` need the LLVM helper to
express the controls in the table. This can change the instruction sequence
of an existing program that specified one of these modifiers. Loads with
`ca/cv` and stores with `wt` retain the ordinary predicator or vector-store
path wherever eligible. Both Python spellings follow the same rule.

## Migration from the experimental common interface

The common intent layer and public `tle.gpu.nvidia` module have been removed.
Move NVIDIA imports directly to `triton.experimental.tle.language.gpu`;
`CachePolicy` and `FractionalEvictionPolicy` are exported there. The old module has no compatibility alias.

| Removed request | Native replacement |
| --- | --- |
| `CachePolicy(l2="reuse")` | `CachePolicy(l2=FractionalEvictionPolicy("evict_last"))` |
| `CachePolicy(l2="streaming")` | `CachePolicy(l2=FractionalEvictionPolicy("evict_first"))` |
| `CachePolicy(l1="reuse")` | `CachePolicy(l1="evict_last")` |
| `CachePolicy(l1="streaming")` | `CachePolicy(l1="evict_first")` |
| `CachePolicy(l1="bypass")` | `CachePolicy(cache_modifier="cg")` |
| Common all-default object | Omit `cache_policy` or pass `None` |

Requests with independent L1 and L2 eviction are no longer supported. A cache
modifier plus L2 eviction remains available where the instruction permits it.
Remove `backend`; the native type and compilation target enforce compatibility. The
removed `reuse`, `streaming`, `bypass`, and `default` strings are rejected as
NVIDIA L1/L2 fields; they are not silently reinterpreted.

## Ordinary global loads and stores

`tl.load(..., cache_policy=policy)` and `tl.store(..., cache_policy=policy)`
accept the same native memory policy objects as pointer copy. They keep the
existing register-valued memory operation and do not allocate shared memory.
The keyword is an extension in this FlagTree branch; upstream Triton language
still exposes `cache_modifier` and `eviction_policy`, while Gluon also has
policy objects.

```python
from triton.experimental.tle import language as tle

x = tl.load(x_ptr + offsets, mask, other=0,
            cache_policy=tle.gpu.ppu.CachePolicy(".lu"))
tl.store(y_ptr + offsets, y, mask,
         cache_policy=tle.gpu.mthreads.LSUCachePolicy(
             inner_persistence=3, outer_persistence=3, l2_policy=1))
```

These two examples target different devices and belong in their respective
backend kernels. NVIDIA uses `tle.gpu.CachePolicy`, HCU uses
`tle.gpu.hcu.CachePolicy`, and MetaX uses `tle.gpu.metax.CachePolicy`.
Passing `None` preserves the original load/store behavior. An explicit object
cannot be combined with a nonempty `cache_modifier` or `eviction_policy`.
The target, global address space, direction, architecture, and native field
constraints are checked. Invalid requests are rejected instead of dropped.
`TMECachePolicy` and `AIUCachePolicy` are copy-path controls and are rejected
by ordinary load/store; use `LSUCachePolicy` and PPU `CachePolicy` there.

Both entry points normalize policies **before creating the memory operation**:

| Control | Formal IR representation |
| --- | --- |
| Ordinary NVIDIA/HCU/MetaX/PPU modifier | Existing `cache` enum (`cacheModifier` in text) |
| NVIDIA L1-only / L2-only priority | Scoped `evict` enum, e.g. `l2_evict_last` |
| PPU explicit LSU default | `cache = ppu_lsu_default`, retaining its identity until instruction selection |
| PPU AIU kp1–6 | `cache = ppu_aiu_kpN`; kp3 still uses the SDK kp2 encoding |
| MThreads LSU tuple | `cache = musa_iN_oN_lN`, covering all 6 × 6 × 2 tuples |
| MThreads TME | Existing backend persistence and prefetch fields |

The existing `evict_first` / `evict_last` values retain their legacy behavior;
they are not renamed to L2-only policies. Native scoped values distinguish the
requested hardware level. A TLE TMA copy and an NVIDIA TMA store now carry the
same defaulted `evict` field that an NVIDIA TMA load already had.

There is no `tt.cache_policy` dictionary, location tag, builder listener or
policy-conservation pass. Rewrites reuse their existing `getCache()` and
`getEvict()` arguments. PPU AIU rewrites that previously defaulted these fields
now pass them explicitly. A new rewrite must still preserve the formal fields;
ODS does not infer the meaning of an arbitrary replacement operation.

Target/direction validation happens in policy normalization and backend
lowering. `CacheControls.h` contains the NVIDIA async mappings and a shared
adjustment-remark helper. Optimizer candidate queries do not filter on policy.
Final lowering consumes supported controls and reports adaptations at the
point where the instruction capability is known. Native PPU/MUSA modes are
not reinterpreted as another backend's ordinary modifiers. Legacy
modifier/eviction arguments follow the same conversion rules because they
use the same IR fields. Changing a source location does not change a policy.

Policies have compile-time fields and can be constructed on the
host or in JIT code; bind a JIT local with
`policy: tl.constexpr = ...` before forwarding it through loops or workers.
Hints change cache/instruction controls, not synchronization or visibility.
This interface does not promise that cache lines survive another launch.


## 中文：精简后的编译约定

公开入口仍为 `tle.gpu.copy(..., cache_policy=...)` 和本分支扩展的
`tl.load/store(..., cache_policy=...)`。各平台使用自己的原生策略类。
对象在前端校验目标、方向和搬运路径，然后转换为既有 ODS `cache` / `evict`
字段。普通 modifier、PPU AIU 模式及 MThreads LSU 三元组都有明确的枚举表示。
旧 `evict_first/last` 的含义不变，NV L2-only 使用 `l2_evict_first/last`。

已删除策略字典、位置标记、builder 监听器和手工 `copyCachePolicy`。
大多数改写直接沿用原有 `getCache()/getEvict()` 参数；TMA store 和 TLE TMA
载体只补充缺失的默认 `evict` 字段。改写仍须传递正式字段，不能假定 ODS 会
自动替任意新建指令补参数。

NV 本轮收窄：`fraction` 只允许 1，`secondary` 只允许 `evict_unchanged`；
不再支持 prefetch 或同时设置独立的 L1 与 L2 eviction。这些配置在构造时
报错。单独 L1、单独 L2、合法的 modifier + L2、TMA first/last 保留。
其他四个平台的原生取值范围保持不变。显式 HCU 空 modifier 规范化为
load `ca` / store `wb`；它不再强制编译器保留 generic global 路径。
最终约定是：先按原有布局、宽度和调度规则选择指令，再按该指令能力转换 hint。
合法 hint 不再阻止 async、AIU 或 HCU buffer 转换，也不再因此编译失败。
无法表达的部分改为该路径支持的控制或省略，并通过 MLIR remark 报告位置、
请求值、实际值和原因。省略 hint 不表示语义等价，更不保证保留原先的性能收益。
如需显示诊断，使用已有的 `MLIR_ENABLE_DIAGNOSTICS=remarks`，并在新的
`TRITON_CACHE_DIR` 下触发编译；默认诊断级别和缓存命中不会显示 remark。

| 平台／最终路径 | 转换规则 |
| --- | --- |
| NV 普通 `ld.global/st.global` | 保留已支持的原生 modifier、单层 eviction |
| NV TMA | 保留 L2；显式 L1、modifier 仍是非法请求 |
| NV 非 bulk `cp.async` | 显式 `ca` 保留；其他 load modifier 在 16B 下变为 `cg`，4/8B 下变为 `ca`；L1 eviction 省略。L2 在 PTX≥9.4 时保留；TLE 的 load→local-pointer store 融合在无 mask 或无循环 worker 中仍省略 L2 |
| MThreads LSU／TME | 普通 LSU 原生三元组、TME persistence/prefetch 各自保留 |
| MThreads `memcpy.g2s` | 使用该异步指令默认控制，不把 LSU 三元组改名为 TME persistence |
| PPU LSU | 保留已有 modifier→kp 映射及方向限制 |
| PPU AIU | 显式 AIU kp1–6 保留，kp3→kp2；普通 LSU 自动提升到 AIU 时使用 AIU 默认，不按数字推断同义 |
| PPU 普通 async | 使用该异步路径原有默认控制 |
| HCU gfx936 generic global | 保留 LLVM non-temporal/volatile 映射；`.cg/.cs` load 的地址使用 VGPR，两种写法一致。旧字符串 `.wt` store 继续走原 wait 路径 |
| HCU gfx936 buffer／direct-to-LDS | 沿用已有 cache→控制位映射，当前结果为默认位 0；不携带 eviction |
| MetaX 普通访存 | Load `cg/cs/lu` 和 store `wb/cg/cs` 经 LLVM helper；load `ca/cv` 和 store `wt` 保留原本可用的 predicator／向量写 |
| MetaX async | 使用该异步路径默认控制 |

后端、方向、非法值、显式 descriptor 路径和危险 PPU kp7 等检查保留。
与 hint 无关的结构性同步回退仍保留原 `cache/evict` 字段。
NV PTX 判断直接使用最终 lowering 的目标信息，删除了额外的
`ttg.ptx-version` module attr。旧 `cache_modifier` / `eviction_policy`
通过同一组正式字段遵循相同转换规则。省略策略仍走原默认路径。

实现依据：[上游 NVIDIA lowering](https://github.com/triton-lang/triton/blob/698e39e0d1e528043351f36a95007f75ea929ae8/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp#L1121-L1185)、
[上游 L1 省略与窄 cg 转 ca 测试](https://github.com/triton-lang/triton/blob/698e39e0d1e528043351f36a95007f75ea929ae8/test/Conversion/tritongpu_to_llvm.mlir#L1087-L1183)。
这里沿用上游“按最终指令支持度处理 hint”的原则，其他后端使用各自已有 SDK
映射；它们不是 NVIDIA cache operator 的跨平台等价实现。
