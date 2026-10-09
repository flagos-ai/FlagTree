# RFC: Custom C-Interoperable Struct & Array Types for FlagTree

| | |
|---|---|
| **Status** | Draft, accepted for implementation |
| **Target backend** | Ascend (design is kept backend-agnostic) |
| **Owners** | FlagTree |

---

## 1. Summary

FlagTree should let a Triton kernel work with **C-defined data structures shared
with the host** (C/C++ and Python). The concrete goal is to pass a small C struct
such as Meta's MTIA `TensorView` to a kernel **by reference** and read or write
its fields, with both sides seeing the same bytes in memory.

This RFC proposes:

- **Two new IR types**, `!tt.struct<...>` and `!tt.array<T, N>`, whose byte
  layout (field offsets, size, alignment) is recorded **explicitly** from
  `ctypes`. They are used only as pointer pointees or as fields nested in another
  struct — **never as SSA values**.
- **A Python API** built on `ctypes`: `@tl.struct_type` on a
  `ctypes.Structure`, with field access (`x_view.base`) and array indexing
  (`x_view.shape[0]`).
- **Two new IR operations**, `tt.struct_gep` and `tt.array_gep`, which compute
  the address of a field or element. Reads and writes then use Triton's existing
  `tt.load` and `tt.store`.
- **A small shared lowering pass** that rewrites those two operations into
  ordinary byte-offset address arithmetic and removes the new types before any
  backend-specific lowering.
- **A host wrapper** (`StructView`, created by `tl.to_device`) that pairs a
  `ctypes` type with a device buffer.

The design is **reference-only**: a struct is always accessed through a pointer
and never exists as a value. This covers the motivating use case and avoids the
large machinery that value semantics would require (scalar replacement of
aggregates, value-manipulation operations, whole-aggregate loads/stores,
rewriting of function calls and control flow). Field **writes are supported** and
cost almost nothing beyond the read case, since a write is just a store to a
computed address.

Early lowering is needed because the Ascend backend does not use Triton's common
GPU-to-LLVM path: it lowers through its own passes into Huawei's *HIVM*
representation and then a proprietary `bishengir` toolchain, which cannot be
expected to understand new types. Removing the types *before* those passes keeps
the feature backend-agnostic and low-risk.

---

## 2. Motivation and example

Meta's MTIA work extends Triton with `ctypes.Structure`-defined custom types
because "many existing MTIA C++ kernels work on custom data structures such as
TensorViews [...] and CoreId". The existing open-source `_aggregate` feature does
not help: it is value-only, has no memory layout, and cannot be shared with C.

The target usage:

```python
import ctypes
import triton
import triton.language as tl

@tl.struct_type
class TensorView(ctypes.Structure):
    _fields_ = [
        ("shape",  ctypes.c_uint64 * 4),
        ("stride", ctypes.c_uint64 * 4),
        ("base",   ctypes.c_void_p),
        ("ndim",   ctypes.c_uint32),
    ]

@triton.jit
def example_kernel(x_view):        # x_view is a pointer to a TensorView
    base  = x_view.base            # read a field
    n     = x_view.ndim
    s0    = x_view.shape[0]        # read an array element
    x_view.ndim = n - 1            # write a field
    ...
```

The host creates a `TensorView` (in Python or C++), uploads its bytes to device
memory, and launches the kernel with a pointer to them. Both sides agree on the
C layout, so the kernel reads and writes the correct fields.

Layout on a 64-bit little-endian host:

| field | type | offset | size |
|---|---|---|---|
| `shape` | `uint64[4]` | 0 | 32 |
| `stride` | `uint64[4]` | 32 | 32 |
| `base` | `void*` | 64 | 8 |
| `ndim` | `uint32` | 72 | 4 |
| **total** | | | **80** (align 8; 4 bytes trailing padding) |

### 2.1 Goals

- Define C-layout structs and arrays in Triton and share their exact bytes with
  host C++/Python.
- Support reference semantics with field reads **and writes**.
- Support nested structs and fixed-size arrays.
- Confine the change to the frontend plus one small shared lowering pass; do not
  modify Ascend's HIVM/`bishengir` pipeline.

### 2.2 Non-goals

- Value semantics (a struct as an SSA value, device-function return, or
  loop-carried variable). See §12.
- Passing a struct by value as a kernel launch argument.
- C unions, bitfields, function pointers, `long double`, complex numbers.
- Platform-dependent integer types (`c_long`, `c_ulong`, `c_size_t`,
  `c_ssize_t`) and ambiguous character types (`c_char`, `c_wchar`, `c_char_p`,
  `c_wchar_p`).
- Merging with, or deprecating, `_aggregate`.
- Keeping the new types alive down to LLVM/HIVM.

---

## 3. Background

Three existing mechanisms shape the design.

### 3.1 `_aggregate`

`python/triton/language/core.py` already has `_aggregate`, a value-semantics
feature for Python classes whose fields are Triton values. It **flattens** each
field into its own SSA value, has no notion of a byte layout, and cannot be used
as a kernel launch argument. It is left unchanged (§5.4).

### 3.2 How a kernel argument gets its type

Kernel argument types are **not** taken from source annotations; they are derived
at launch time from the host values:

1. `specialize_impl` (`python/triton/runtime/jit.py`) inspects each host
   argument and produces a **signature string** (e.g. `*fp32`,
   `tensordesc<...>`).
2. `str_to_ty` (`python/triton/language/__init__.py`) parses that string back
   into a Triton type.
3. The type drives code generation.

Two consequences:

- A struct argument must be expressible as a **self-contained signature string**
  that `str_to_ty` can reconstruct without access to the defining Python class.
- Because a by-reference struct is just a pointer at runtime, the host must pass
  an object that tells `specialize_impl` it is a struct; otherwise the type is
  lost. This motivates the `StructView` wrapper (§6.3).

### 3.3 The Ascend backend is different

- **Ascend overrides the frontend with copies.** When `FLAGTREE_BACKEND=ascend`,
  `python/triton/flagtree_spec.py` makes modules such as `triton.language`,
  `triton.runtime`, and `triton.compiler` resolve to near-copies under
  `python/triton/spec/ascend/...`. New frontend code must therefore be factored
  into a **shared module** rather than edited twice.
- **Ascend does not use the common GPU-to-LLVM path.** It runs the usual early
  Triton passes and then its own passes into HIVM, which is fed to the
  proprietary `bishengir` toolchain. There is no shared `TritonGPUToLLVM` type
  converter in this path.

### 3.4 Resulting constraints

1. A host wrapper must carry the struct type across the boundary.
2. New frontend code lives in a shared module imported by both frontends.
3. The new types must be removed before backend-specific passes.

---

## 4. Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Semantics | **Reference only**; a struct is always accessed through a pointer | Covers the use case; avoids the machinery of value semantics. |
| Mutability | Field **writes** supported | A write is just a store to a computed address, so it is nearly free and enables exchange with the host. |
| Memory layout | **Explicit** byte layout stored in the type | The C/`ctypes` layout is authoritative; a target's own layout rules may silently differ. |
| Type identity | **Structural**: (field names, field types, layout) | Must be reconstructible from a signature string, which cannot carry a Python class. The class name is kept only as a diagnostic label. |
| Field types | **Basic C types only**, plus nesting and arrays | Keeps the feature well-scoped; every field maps to a Triton scalar/pointer/composite. |
| Composition | Nested structs (inline) and fixed-size arrays | The motivating `TensorView` needs array fields. |
| Type position | Only as **pointer pointees** or nested fields; never SSA values | Removes all value-manipulation machinery. |
| Operations | **`tt.struct_gep` and `tt.array_gep`**; reads/writes via existing `tt.load`/`tt.store` | Minimal, verifiable, and lowers to plain address arithmetic. |
| Lowering | A **small shared pass** rewrites the two ops and removes the types | Backend-agnostic; Ascend's HIVM/`bishengir` pipeline stays untouched. |
| Code placement | A **shared frontend module** for both `core.py` copies | Prevents divergence between the main and Ascend frontends. |
| Kernel boundary | Structs pass **by reference** | Matches the use case; avoids a by-value C-struct launch ABI. |
| Host API | `StructView` + `tl.to_device` | The kernel needs the struct type, which a bare pointer would not convey. |
| Layout correctness | Address fields by **explicit byte offsets** | The kernel layout exactly matches the host layout, independent of backend. |
| Existing `_aggregate` | Left **unchanged** | Different purpose (its fields may be Triton tensors). |

---

## 5. Types

### 5.1 Struct and array types

```
!tt.array<i64, 4>                       ; element type, number of elements
!tt.struct<{shape: !tt.array<i64,4> @0,
            ndim:  i32              @32},
           #tt.struct_layout<size=40, align=8}>
```

- `!tt.array<T, N>` is contiguous with `size = N * sizeof(T)` and
  `align = alignof(T)`, with no padding between elements.
- `!tt.struct` is an ordered list of fields `(name, type, offset)` plus a total
  `size` and `align`.
- The layout attribute records offsets/size/align explicitly. Attaching offsets
  directly to fields instead of a separate attribute is an acceptable encoding,
  as long as it is explicit and verifiable.

These types appear in only two positions: as the pointee of a pointer (e.g.
`!tt.ptr<!tt.struct<...>>`), or as a field of another struct. They are never SSA
values, and there is no "load a whole struct" operation.

Both types are defined in `TritonTypes.td`, following the existing pointer and
tensor-descriptor type patterns.

### 5.2 Allowed field types

Only basic C types are allowed, plus nesting and arrays. Sizes/alignments are for
a 64-bit little-endian host.

| `ctypes` type | C type | size / align | Triton type |
|---|---|---|---|
| `c_bool` | `_Bool` | 1 / 1 | `int1` (stored as one byte) |
| `c_byte`, `c_int8` | `int8_t` | 1 / 1 | `int8` |
| `c_ubyte`, `c_uint8` | `uint8_t` | 1 / 1 | `uint8` |
| `c_short`, `c_int16` | `int16_t` | 2 / 2 | `int16` |
| `c_ushort`, `c_uint16` | `uint16_t` | 2 / 2 | `uint16` |
| `c_int`, `c_int32` | `int32_t` | 4 / 4 | `int32` |
| `c_uint`, `c_uint32` | `uint32_t` | 4 / 4 | `uint32` |
| `c_longlong`, `c_int64` | `int64_t` | 8 / 8 | `int64` |
| `c_ulonglong`, `c_uint64` | `uint64_t` | 8 / 8 | `uint64` |

- **Floats:** `c_float` → `float32`; `c_double` → `float64`.
- **Pointers:** `c_void_p` → `ptr<int8>` (opaque); `POINTER(T)` → `ptr<T'>`.
- **Composites:** nested `ctypes.Structure` (inline), `T * N` arrays, arrays of
  arrays, arrays of structs.
- **Rejected:** `c_char`, `c_wchar`, `c_char_p`, `c_wchar_p`, `c_long`,
  `c_ulong`, `c_size_t`, `c_ssize_t`, `c_longdouble`, `ctypes.Union`, bitfields,
  function pointers, `py_object`, `c_void`. Rejections raise a clear error naming
  the offending field and type.

### 5.3 Signature strings

A struct argument is encoded as a self-contained string parsed by `str_to_ty`:

```
*struct<TensorView>{shape:array<u64,4>@0,stride:array<u64,4>@32,base:*i8@64,ndim:u32@72};size=80;align=8
```

The leading `*` denotes by-reference passing. The parser is added to the shared
`str_to_ty`; it is not duplicated for Ascend.

### 5.4 Existing `_aggregate`

`_aggregate` is left unchanged. Its fields may be arbitrary Triton values
(including tensors), whereas `!tt.struct` is restricted to C-representable
fields; the two features serve different purposes.

---

## 6. Frontend and host API

### 6.1 Shared module

All new frontend code lives in `python/triton/language/_custom_types.py`,
imported by both `python/triton/language/core.py` and
`python/triton/spec/ascend/language/core.py`. It contains the `ctypes` mapping,
layout computation, the type classes, the `@struct_type` decorator, signature
serialization, and the host wrapper. The two `core.py` copies only re-export the
public symbols.

### 6.2 Using a struct

The `@tl.struct_type` decorator validates and maps each field, computes the
layout with `ctypes.offsetof`/`sizeof`/`alignment`, and produces a
`struct_type`. Inside a kernel the parameter is a pointer, and fields are read
and written through it (see §2). Nested structs are reached by chaining field
access; array elements are addressed with `x_view.shape[i]` (static or dynamic).

### 6.3 Host API

A by-reference struct is just a pointer at runtime, so a small wrapper carries
the struct type alongside the device buffer:

```python
tv   = TensorView(...)            # ctypes instance, host layout
view = tl.to_device(tv)           # allocate device memory, upload the bytes
example[(1,)](view)               # specialize_impl sees "*struct<TensorView>{...}"
```

`StructView` holds the `ctypes` type and a device buffer exposing `data_ptr()`.
On Ascend the launcher already accepts any object with `data_ptr()`, so no
launcher change is needed for by-reference passing.

**Public surface:** `tl.struct_type`, `tl.to_device`, `StructView`. Array fields
use ordinary `ctypes` syntax (`T * N`), so no separate array helper is required.

---

## 7. IR operations and verification

Only two operations are new; both compute an address, and the read or write uses
existing load/store.

| Operation | Meaning | Lowered form |
|---|---|---|
| `tt.struct_gep` | address of a named field through a struct pointer | `tt.addptr` by the field offset |
| `tt.array_gep` | address of element `i` through an array pointer | byte `addptr` by `i * sizeof(T)` |

Reading `x_view.shape[i]`:

```
%p0 = tt.struct_gep %x_view["shape"] : !tt.ptr<!tt.struct<...>> -> !tt.ptr<!tt.array<i64,4>>
%p1 = tt.array_gep  %p0[%i]          : !tt.ptr<!tt.array<i64,4>> -> !tt.ptr<i64>
%v  = tt.load %p1 : !tt.ptr<i64>
```

A write uses the same address computation followed by `tt.store`. Dynamic
indices are naturally supported because `tt.array_gep` uses byte arithmetic.

**Not permitted with struct/array types:** arithmetic, comparisons,
reductions/scans, `select`/`where`, bitcasts, masked load/store, pointer casts
between unrelated struct types, and use as a tensor element type or as a
by-value function argument/result.

**Verification rules for `!tt.struct`:** unique field names; offsets
non-decreasing, non-overlapping, and within `[0, size)`; each offset a multiple
of the field's alignment; struct alignment the maximum field alignment; size a
multiple of alignment; array count ≥ 1; no by-value self-recursion (only through
a pointer).

---

## 8. Lowering

### 8.1 The `tt-lower-structs` pass

A small shared pass (`lib/Dialect/Triton/Transforms/LowerStructs.cpp`,
registered as `triton-lower-structs`) does two things:

1. Rewrites each `tt.struct_gep`/`tt.array_gep` into a `tt.addptr` by the
   appropriate byte offset.
2. Rewrites pointer types whose pointee is a struct or array to an opaque
   pointer, and fails if any struct/array type remains.

Because structs and arrays are never values, the pass does **not** scalarize
values, rewrite function signatures or calls, or touch `scf` loop-carried values
or returns — those only carry pointers. The pass runs at the end of `make_ttir`
for Ascend, before the Ascend passes.

### 8.2 ABI and layout correctness

By-reference passing reuses the launcher's existing pointer handling (a raw
integer address or any object with `data_ptr()`).

Because field access becomes explicit **byte offsets** and the host builds the
buffer with `ctypes`, the kernel sees exactly the host/C layout, independent of
the backend and its data-layout rules. The compiler adds two safety checks: the
host and target pointer sizes must match, and both must be little-endian.

**Open item (highest risk):** Ascend flattens pointer arguments into a
one-dimensional memref tuple (base pointer, data pointer, offset, shape, stride).
Confirm that `ptr<struct>` uses the simple scalar-pointer path, or special-case
it.

---

## 9. Implementation plan

| Phase | Scope | Key files |
|---|---|---|
| 0 | Shared frontend module: `ctypes` mapping, layout, types, decorator, signature serialization, `StructView`/`to_device` | `python/triton/language/_custom_types.py`; imports in both `core.py` copies |
| 1 | Python plumbing: allow `pointer_type` to point at composite types; extend `str_to_ty`; add the `StructView` case to `specialize_impl` | `core.py`, `language/__init__.py`, `runtime/jit.py`, `spec/ascend/runtime/jit.py` |
| 2 | C++ types, the two address ops, verification rules, Python builder bindings | `TritonTypes.td`, `Types.h/.cpp`, `TritonOps.td`, `Ops.cpp`, `python/src/ir.cc` |
| 3 | The lowering pass and its registration | `Passes.td`, `LowerStructs.cpp`, `python/src/passes.cc`, Ascend `make_ttir` |
| 4 | Ascend host/runtime wiring; interpreter behavior | `third_party/ascend/backend/driver.py`, `spec/ascend/runtime/interpreter.py` |
| 5 | Tests and documentation | `python/test/unit/language/...`, pass tests, end-to-end test |

**Order (vertical slices)**

1. **Scalar fields, by reference.** Implement all phases for scalar fields and
   get one Ascend kernel that reads and writes `x_view.ndim` working end to end.
2. **Arrays and nesting.** Add array fields, `tt.array_gep`, nested structs, and
   `x_view.shape[i]` with static and dynamic indices.
3. **Polish.** Verification-failure tests, interpreter behavior, documentation,
   and clear error messages.

---

## 10. Testing

- **Layout:** computed offsets/size/alignment equal `ctypes.offsetof`/`sizeof`/
  `alignment` for nested, array, and packed structs; rejected fields raise.
- **Frontend/lit:** field reads and writes, nested access, array access with
  static and dynamic indices, by-reference passing.
- **Verification (negative):** malformed layouts and out-of-range indices are
  rejected with clear messages.
- **Pass:** no struct/array type remains and the address ops are rewritten to
  `addptr`.
- **End-to-end:** a `TensorView`-style Ascend kernel that reads and writes
  fields, validated against a host-computed reference.

---

## 11. Risks

1. **Ascend pointer lowering for `ptr<struct>`** — the memref-tuple handling must
   be checked early; highest integration risk.
2. **Python bindings** for the two ops are needed before the frontend can emit
   them.
3. **Frontend module resolution** — both `core.py` copies must import the shared
   module without being shadowed by the Ascend path injection.
4. **Coexistence with `_aggregate`** — independent by design, but should not
   collide on shared names or markers.

---

## 12. Alternatives considered

**Semantics.** Full value + reference semantics (structs as SSA values, returned
from device functions, carried by loops) would require scalar replacement of
aggregates, value-manipulation ops, whole-aggregate loads/stores, dynamic-index
select chains, and rewriting of calls and control flow. Deferred; the motivating
use case does not need it. **Reference-only (chosen)** supports field reads and
writes with far less machinery.

**Type representation.** *Frontend-only flattening* (no IR type, just `addptr`
and loads/stores) is smaller but gives the compiler no struct type to verify.
*A full first-class value type* is covered above. **A real struct type used only
in pointer position (chosen)** is verifiable and drives signature parsing while
still being removed before backend lowering.

**Lowering depth.** *Keep the type down to LLVM/HIVM* would require new
conversions in every backend and changes to Ascend's proprietary toolchain.
**Remove it with a small shared pass (chosen)** is backend-agnostic and low-risk.

**Memory layout.** *Derive the layout from the target's rules* can silently
disagree with the host C layout. **Record the explicit `ctypes` layout (chosen).**

**Kernel boundary.** *By-value struct launch arguments* require C-struct ABI
marshalling in every launcher; deferred. **By-reference (chosen).**

**Type identity.** *The nominal `ctypes` class name* cannot be reconstructed from
a signature string. **Structural identity (chosen)**, with the name kept only as
a diagnostic label.
