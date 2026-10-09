# RFC: Custom C-Interoperable Struct & Array Types for FlagTree

| | |
|---|---|
| **Status** | Draft, accepted for implementation |
| **Target backend** | Ascend (design is kept backend-agnostic) |
| **Owners** | FlagTree |
| **Related** | MTIA custom-types work; existing `_aggregate` support; `tensordesc` |

---

## 0. Quick summary

FlagTree should let a Triton kernel work with **custom data structures that are
defined in C** and are shared with the surrounding software stack (C/C++ and
Python). The concrete goal is to take a small C struct such as Meta's MTIA
`TensorView` (tensor metadata: shape, strides, base pointer, rank), hand it to a
kernel **by reference**, and read its fields inside the kernel — with the kernel
and the host seeing exactly the same bytes in memory.

Today Triton cannot do this. Its only nearby feature, `@triton.language.core._aggregate`,
models Python objects whose fields are Triton values. It has no concept of a C
memory layout, so it cannot be shared with C.

This document specifies the following design:

1. **Two new Triton IR types.**
   - `!tt.struct<{...}, #layout>` — a C-style record whose **byte layout**
     (field offsets, total size, alignment) is recorded explicitly in the type.
   - `!tt.array<T, N>` — a fixed-size C array of `N` elements of type `T`.
2. **A Python-facing API built on `ctypes`.** A user declares a struct with the
   familiar `ctypes.Structure` syntax and marks it with `@tl.struct_type`. Field
   access (`x_view.base`), nested structs, and array fields (`x_view.shape[0]`)
   all work inside a `@triton.jit` kernel.
3. **A small, fixed set of IR operations** to read/write fields and elements,
   plus the ability to load/store a whole struct or array.
4. **An "early lowering" strategy.** The new types are first-class in the
   frontend and in early Triton IR. A single shared compiler pass then
   **removes them entirely** before any backend-specific lowering begins. This
   pass turns struct/array values into ordinary scalar values and turns
   field/element access into byte-offset address arithmetic plus normal
   loads/stores. As a result, no backend — including Ascend — needs to learn
   about the new types.
5. **A host-side wrapper object** (`StructView`, produced by `tl.to_device`)
   that pairs a `ctypes` type with a device buffer. In the first version, structs
   cross the kernel launch boundary **by reference (as a pointer)**; by-value
   structs are available for in-kernel and device-function use.

**Why early lowering matters:** the Ascend backend does not use Triton's common
GPU-to-LLVM lowering path. It lowers kernels through its own passes into
Huawei's "HIVM" intermediate representation and then into a proprietary
`bishengir` compiler toolchain. Teaching that toolchain about brand-new types is
risky and expensive. Removing the new types *before* those passes keeps the
feature backend-agnostic, small, and low-risk, while still giving the frontend
and the IR a real, checkable type.

---

## 1. Introduction and terminology

### 1.1 The problem in one paragraph

Triton normally passes only scalars, pointers, and tensors across the
host/kernel boundary, and it represents them with a small fixed set of types.
Real-world accelerator software often has richer, C-defined metadata structures
that both host code and kernels must manipulate. Without support for such
structures, users must decompose them into loose scalar arguments (error-prone,
and impossible when the structure must be passed by reference or nested), or
duplicate the definition on each side. This RFC adds a first-class way to define
and use such C-compatible structures in Triton.

### 1.2 What this document is

This is a design document ("RFC"). It records the problem, the chosen design,
the reasoning behind each decision, the implementation plan, and the
alternatives that were considered. It assumes no prior discussion.

### 1.3 Glossary

| Term | Meaning |
|---|---|
| **IR** | Intermediate representation. Triton programs are lowered through several IRs before becoming machine code. |
| **Triton IR / `tt` dialect** | The MLIR dialect (named `tt`) that represents a Triton kernel before it is specialized to a particular backend. Types in it are written `!tt.*` and operations `tt.*`. |
| **MLIR** | The multi-level IR framework Triton is built on. |
| **SSA** | Static single assignment: every value is assigned exactly once, which is how MLIR represents data flow. |
| **Lowering** | Translating a program from a higher-level IR to a lower-level one, gradually removing abstraction. |
| **SROA** | "Scalar replacement of aggregates": replacing a struct/array value with its individual fields/elements. |
| **GEP** | "Get element pointer": address arithmetic that computes the address of a field or element. |
| **ABI** | Application binary interface: the concrete rules for how values are passed between caller and callee (host and kernel). |
| **ctypes** | Python's standard library module for defining C-compatible data types (e.g. `ctypes.Structure`, `ctypes.c_uint64`). |
| **HIVM** | Huawei's intermediate representation used by the Ascend backend. |
| **bishengir** | Ascend's proprietary compiler toolchain that consumes HIVM. |
| **TritonGPU / TTGIR** | Triton's GPU-specific dialect, used by the CUDA/AMD backends between Triton IR and LLVM. Ascend does **not** use it. |
| **LLVM** | The compiler infrastructure used by the CUDA/AMD backends for final code generation. |

---

## 2. Motivation and example

Meta's MTIA work describes extending Triton with custom types defined via
`ctypes.Structure`, because "many existing MTIA C++ kernels work on custom data
structures such as TensorViews [...] and CoreId". The existing open-source
`_aggregate` feature does not help here: it is value-only, has no memory layout,
and cannot be shared with C.

The target usage looks like this:

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
def example_kernel(x_view):        # x_view arrives as a pointer to a TensorView
    base  = x_view.base            # read a field
    n     = x_view.ndim
    s0    = x_view.shape[0]        # read an array element
    ...
```

The host creates a `TensorView` (in Python or C++), uploads its bytes to device
memory, and launches the kernel with a pointer to those bytes. Because both
sides agree on the C layout, the kernel reads the correct fields.

### 2.1 Goals

- Define C-layout structs and arrays in Triton and share their **exact bytes**
  with host C++/Python.
- Support both **value semantics** (a struct as a normal SSA value, usable in
  device functions and control flow) and **reference semantics** (a pointer to a
  struct, with field reads and writes).
- Support nested structs and fixed-size arrays.
- Keep the compiler change confined to the frontend plus one shared lowering
  pass; do not modify Ascend's HIVM/`bishengir` pipeline.

### 2.2 Non-goals (first version)

- C unions, bitfields, function pointers, `long double`, complex numbers.
- Platform-dependent integer types (`c_long`, `c_ulong`, `c_size_t`,
  `c_ssize_t`) and ambiguous character types (`c_char`, `c_wchar`, `c_char_p`,
  `c_wchar_p`).
- Passing a struct **by value** as a kernel launch argument (the first version
  passes structs to kernels by reference only).
- Merging with, or deprecating, the existing `_aggregate` feature.
- Keeping the new types alive all the way down to LLVM/HIVM (see
  §15, "Alternatives considered").

---

## 3. Background: how the current system works

Understanding three existing mechanisms is necessary to follow the design.

### 3.1 Existing aggregate support: `_aggregate`

`python/triton/language/core.py` already contains `_aggregate` /
`_aggregate_type`. It lets a user write a Python class whose fields are Triton
values, and it has value semantics. Internally it **flattens** each field into
its own SSA value. It has two limitations that matter here:

- It has no notion of a byte representation, so it cannot be shared with C.
- It works for device functions but not as a kernel launch argument.

Keeping it unchanged is an explicit decision (§5, "Existing `_aggregate`").

### 3.2 How a kernel argument gets its type

Kernel argument types are **not** taken from the Python annotations in the
source. Instead, they are derived at launch time from the actual host values:

1. `specialize_impl` (`python/triton/runtime/jit.py`) inspects each host
   argument and produces a **signature string**, e.g. `*fp32` for a float
   pointer or `tensordesc<...>` for a tensor descriptor.
2. The signature string is parsed back into a Triton type by `str_to_ty`
   (`python/triton/language/__init__.py`).
3. That type drives code generation.

Two consequences follow:

- A new kind of argument (like our struct) must be expressible as a
  **self-contained signature string** that `str_to_ty` can reconstruct without
  access to the original Python class.
- Because a by-reference struct is, at runtime, just a pointer, the host must
  pass an object that tells `specialize_impl` that it is a struct (otherwise the
  struct type is lost). This motivates the `StructView` wrapper (§8.3).

### 3.3 The Ascend backend is architecturally different

FlagTree supports several backends. For Ascend, two facts shape this design:

- **Ascend overrides the Python frontend with copies.** When
  `FLAGTREE_BACKEND=ascend`, the file `python/triton/flagtree_spec.py` arranges
  for modules such as `triton.language`, `triton.runtime`, and
  `triton.compiler` to resolve to near-copies under
  `python/triton/spec/ascend/...` instead of the main sources. Those copies have
  already diverged from the originals. Therefore new frontend code must either
  be edited in both places or, better, factored into a shared module.
- **Ascend does not use the common GPU-to-LLVM path.** Its pipeline runs the
  usual early Triton passes and then its own passes
  (`add_commonir_to_hivm`, `add_triton_to_hivm`, `add_triton_to_llvm`,
  `add_triton_to_linalg`, ...), producing HIVM, which is fed to the proprietary
  `bishengir` toolchain. There is no shared `TritonGPUToLLVM` type converter in
  this path.

### 3.4 Constraints derived from the background

1. Because argument types come from host values, a **host wrapper** must carry
   the struct type across the boundary (§8.3).
2. Because Ascend overrides the frontend with copies, new frontend code lives in
   a **shared module** imported by both copies (§6.1).
3. Because Ascend has a bespoke pipeline, the new types must be **removed before
   backend-specific passes** (§10).

---

## 4. Design decisions at a glance

Each decision is explained in the referenced section. The "Rationale" column
summarizes why it was chosen.

| Decision | Choice | Rationale |
|---|---|---|
| Semantics | Support **both** value and reference semantics | The motivating use case needs by-reference passing; value semantics are needed for device functions and control flow. |
| Memory layout | Store an **explicit byte layout** in the type | The C/`ctypes` layout is authoritative for sharing memory; a target's own layout rules may differ silently. |
| Type identity | **Structural**: identity is (field names, field types, layout) | Types must be reconstructible from a signature string, which has no access to the defining Python class. |
| Field types | **Basic C types only** (plus nesting and arrays) | Keeps the feature well-scoped; every field maps to a Triton scalar/pointer/composite. |
| Composition | Nested structs (inline) and fixed-size arrays | The motivating `TensorView` requires array fields. |
| Struct as a value | A struct is a **single IR value**, not field-flattened | Lets existing control-flow and function-call machinery carry structs unchanged. |
| Arrays as a value | Add a full first-class `!tt.array` type | Needed for array fields inside value structs and whole-array load/store. |
| Operations | A small fixed op set for field/element access and whole-struct/array load/store | Minimal surface; easy to verify; lowers to ordinary operations. |
| Lowering strategy | **Early lowering**: first-class in the frontend/early IR, removed by a shared pass | Ascend cannot consume the types downstream; avoids touching HIVM/`bishengir`. |
| Code placement | A **shared frontend module** for both `core.py` copies | Avoids divergence between the main and Ascend frontends. |
| Kernel boundary | Structs cross the kernel boundary **by reference** in v1 | Matches the use case; avoids inventing a by-value C-struct launch ABI. |
| Host API | A `StructView` wrapper plus `tl.to_device` | The kernel needs the struct type, which a bare pointer would not convey. |
| Layout correctness | Address fields by **explicit byte offsets** | Makes the kernel layout exactly match the host layout, independent of backend. |
| Existing `_aggregate` | Leave it **unchanged** | It serves a different purpose (its fields can be Triton tensors; `!tt.struct` is C-only). |

---

## 5. Type-system decisions in detail

### 5.1 Value and reference semantics

A struct can appear in two forms:

- **By value**, as an SSA value of type `!tt.struct<...>`. This is useful for
  device functions that take or return a struct, and for loop-carried variables.
- **By reference**, as a pointer `!tt.ptr<!tt.struct<...>>`. Reading a field
  loads from memory at a known offset; writing a field stores to it.

Both are supported. The paper's use case ("passing them by reference") is the
by-reference form.

### 5.2 Explicit memory layout

The type records the exact byte layout produced by the host's C/`ctypes`
compiler: each field's byte offset, the total size, and the alignment. The
alternative — letting the target backend decide the layout from field types —
was rejected because the target's layout rules can differ from the host's in
ways that are silent and dangerous: integer alignment, pointer width, struct
packing, and trailing padding can all differ. For a feature whose entire purpose
is binary interoperability with C, the host layout must win.

### 5.3 Type identity

Two struct types are considered the same if they have the same field names,
field types, and layout. The Python class name (e.g. `TensorView`) is kept only
as a human-readable label for diagnostics and does **not** affect identity. This
choice is forced by the requirement that a type be reconstructible from a
signature string (§3.2), since the string cannot carry a Python class object.
A useful consequence: two structs with identical fields but different packing
are correctly treated as different types.

### 5.4 Allowed field types

Only basic C types are allowed, plus nesting and arrays. Sizes and alignments
below are for a 64-bit little-endian host.

**Integers**

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

**Floats:** `c_float` → `float32`; `c_double` → `float64`.

**Pointers:** `c_void_p` → `ptr<int8>` (an opaque pointer); `POINTER(T)` →
`ptr<T'>` where `T'` is the mapped type of `T`.

**Composites:** a nested `ctypes.Structure` is stored inline by value; `T * N`
is a fixed-size array; arrays of arrays and arrays of structs are allowed.

**Rejected in the first version:** `c_char`, `c_wchar`, `c_char_p`,
`c_wchar_p`, `c_long`, `c_ulong`, `c_size_t`, `c_ssize_t`, `c_longdouble`,
`ctypes.Union`, bitfields, function pointers (`CFUNCTYPE`), `py_object`, and
`c_void`. Platform-dependent integer types and ambiguous character types are
rejected because their size or signedness is not fixed across platforms.
Rejections raise a clear error naming the offending field and type.

### 5.5 Existing `_aggregate`

The existing `_aggregate` feature is intentionally left unchanged. Its fields
may be arbitrary Triton values, including tensors, whereas `!tt.struct` is
restricted to C-representable fields. The two features therefore serve different
purposes and remain independent.

---

## 6. Architecture overview

```
              ┌──────────────── shared frontend module ─────────────────┐
  @tl.struct_type ──▶ struct_type / array_type + values + ops + host API   │
              │      (python/triton/language/_custom_types.py)            │
              └──────────────────────────────────────────────────────────┘
                      │ imported by both core.py copies
                      ▼
   Triton IR: `!tt.struct` / `!tt.array` first-class SSA values + new ops
                      │
        ┌─────────────┴───────────────────────────────┐
        │        shared `tt-lower-structs` pass        │
        │  • scalarize aggregate values (SROA)         │
        │  • pointer field access → byte addptr         │
        │  • aggregate load/store → field-wise          │
        └─────────────┬───────────────────────────────┘
                      │  (aggregate types fully removed)
        ┌─────────────┴─────────────┐
        ▼                           ▼
  Ascend passes → HIVM        (future) TritonGPU → LLVM
  → bishengir (unmodified)
```

### 6.1 Shared frontend module

All new frontend code lives in a single new module,
`python/triton/language/_custom_types.py`. This module is imported by both
`python/triton/language/core.py` (the main frontend) and
`python/triton/spec/ascend/language/core.py` (the Ascend copy). It contains the
`ctypes` mapping, layout computation, the type classes, the `@struct_type`
decorator, signature serialization, and the host wrapper. The two `core.py`
copies only re-export the public symbols, so the two frontends cannot drift on
this feature.

---

## 7. The type system

### 7.1 The two new types

```
!tt.array<i64, 4>                       ; element type, number of elements
!tt.struct<{shape: !tt.array<i64,4> @0,
            ndim:  i32              @32},
           #tt.struct_layout<size=40, align=8}>
```

- `!tt.array<T, N>` is contiguous, has `size = N * sizeof(T)` and
  `align = alignof(T)`, with no padding between elements.
- `!tt.struct` has an ordered list of fields `(name, type, offset)` plus a total
  `size` and `align`.
- The layout attribute `#tt.struct_layout<...>` records the offsets/size/align
  explicitly. (The per-field offsets may instead be attached to the fields; the
  exact encoding is a minor implementation choice, as long as it is explicit and
  verifiable.)

Both types are defined in `TritonTypes.td`, following the existing patterns for
the pointer type and the tensor-descriptor type, and are added to the set of
types that generic Triton operations understand (`TT_Type`).

### 7.2 Signature strings

Kernel-argument types are identified by self-contained strings that `str_to_ty`
parses. The proposed encoding is:

```
*struct<TensorView>{shape:array<u64,4>@0,stride:array<u64,4>@32,base:*i8@64,ndim:u32@72};size=80;align=8
```

The leading `*` denotes by-reference passing. The parser is added to the shared
`str_to_ty` function; it is not duplicated for Ascend.

---

## 8. Frontend and host API

### 8.1 Defining a struct

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
```

The `@tl.struct_type` decorator:

1. Validates and maps each field according to the allowlist (§5.4).
2. Computes the layout using `ctypes.offsetof`, `ctypes.sizeof`, and
   `ctypes.alignment`.
3. Produces a `struct_type` and marks the class so the code generator will
   permit references to it from within kernels.

### 8.2 Using a struct inside a kernel

By reference (the kernel parameter is a pointer to the struct):

```python
@triton.jit
def example(x_view):              # pointer to a TensorView
    base = x_view.base           # read a field
    n    = x_view.ndim
    s0   = x_view.shape[0]       # read an array element
    ...
```

By value (for device functions and in-kernel values):

```python
@triton.jit
def use_value(v):                 # v is a struct value
    return v.ndim
```

### 8.3 Using a struct from the host

Because a by-reference struct is, at runtime, just a pointer, and because the
kernel's type information is derived from the host argument (§3.2), a small
wrapper carries the struct type alongside the device buffer:

```python
tv   = TensorView(...)            # a ctypes instance, host layout
view = tl.to_device(tv)           # allocates device memory, uploads the bytes
example[(1,)](view)               # specialize_impl sees "*struct<TensorView>{...}"
```

`StructView` holds the `ctypes` type and a device buffer that exposes a
`data_ptr()` method. On Ascend, the existing launcher already accepts any object
with a `data_ptr()` method, so no launcher change is needed for by-reference
passing.

### 8.4 Public surface

- `tl.struct_type` — the decorator.
- `tl.array_type` — construct `!tt.array` types directly (optional).
- `tl.to_device(instance) -> StructView` — upload a host struct to device memory.
- `StructView` — the host wrapper type.

---

## 9. IR operations

| Operation | Meaning | Lowered form (after the pass) |
|---|---|---|
| `tt.struct_extract` | read a named field of a struct value | field selection |
| `tt.struct_insert` | produce a new struct value with one field replaced | re-pairing of fields |
| `tt.struct_gep` | address of a named field through a struct pointer | `tt.addptr` by the field's byte offset |
| `tt.array_extract` | read element `i` of an array value | direct select (static index) or select chain (dynamic) |
| `tt.array_insert` | produce a new array value with element `i` replaced | re-pairing of elements |
| `tt.array_gep` | address of element `i` through an array pointer | byte `addptr` by `i * sizeof(T)` |
| `tt.load` / `tt.store` | load/store a whole struct or array | field-wise / element-wise access |

**Dynamic indexing of an array value** (i.e. indexing by a runtime value inside
a value struct) is lowered to a chain of `extractvalue` + `select` operations,
which is acceptable because C arrays are small. Indexing through a pointer
(reference form) uses byte arithmetic and supports dynamic indices directly.

**Operations not permitted on structs or arrays:** arithmetic, comparisons,
reductions/scans, `select`/`where`, bitcasts, masked load/store, and pointer
casts between unrelated struct types.

### 9.1 Type verification rules

The compiler verifies the following for every `!tt.struct`:

- field names are unique;
- field offsets are non-decreasing, each field lies entirely within
  `[0, size)`, and fields do not overlap;
- each field's offset is a multiple of that field's alignment;
- the struct's alignment is the maximum of its fields' alignments, and its size
  is a multiple of its alignment;
- a struct may not contain itself by value (that would be infinitely large);
  self-reference is allowed only through a pointer;
- an array's element count is at least 1.

---

## 10. The `tt-lower-structs` lowering pass

A single shared Triton IR pass removes the new types before backend-specific
lowering. It is registered as `triton-lower-structs` and lives in
`lib/Dialect/Triton/Transforms/LowerStructs.cpp`.

It performs four jobs:

1. **Scalarize aggregate values.** Each `!tt.struct` value becomes its
   constituent fields; each `!tt.array` value becomes its elements. The
   field/element access operations are rewritten accordingly.
2. **Lower aggregate pointers.** `tt.struct_gep` becomes `tt.addptr` by the
   field's byte offset; `tt.array_gep` becomes a byte `tt.addptr`. Whole-struct
   and whole-array loads/stores become field-wise or element-wise loads/stores.
3. **Rewrite structural control flow.** Aggregate operands are expanded wherever
   values flow across a boundary: function signatures, function calls,
   `scf.for` loop-carried values, `scf.if`/`scf.while` results and yields, and
   function returns.
4. **Verify elimination.** The pass fails if any struct or array type remains.

The pass is inserted at the end of `make_ttir` for Ascend (before the Ascend
passes begin) and, for future backends, before the GPU conversion. After it
runs, no backend ever sees the new types.

---

## 11. Lowering and ABI details

### 11.1 Ascend

- All frontend and IR additions are shared with the main frontend.
- The lowering pass runs inside `make_ttir`; Ascend's HIVM/`bishengir` pipeline
  is unchanged.
- By-reference passing reuses the launcher's existing pointer handling, which
  accepts a raw integer address or any object with a `data_ptr()` method.
- **Open item:** Ascend flattens pointer arguments into a one-dimensional
  "memref tuple" (base pointer, data pointer, offset, shape, stride). We must
  confirm that a `ptr<struct>` uses the simple scalar-pointer path, or special-case
  it. This is the highest integration risk (§14).

### 11.2 Other backends (future, not in the first version)

If first-class structs are later wanted all the way to LLVM on the CUDA/AMD
backends, the common GPU-to-LLVM type converter would need cases mapping
`!tt.struct` to `!llvm.struct` and `!tt.array` to `!llvm.array`, with care taken
to preserve the explicit layout (for example, by verifying the LLVM layout
against the recorded offsets and inserting padding if they differ). This is
explicitly out of scope for the first version.

### 11.3 Layout correctness

Because field access is lowered to explicit **byte offsets** and the host builds
the buffer with `ctypes`, the layout seen by the kernel is exactly the host/C
layout — independent of the backend and of any target data-layout rules. The
compiler adds two safety checks:

- the host pointer size must equal the target pointer size;
- both must be little-endian.

---

## 12. Implementation plan

| Phase | Scope | Key files |
|---|---|---|
| 0 | Shared frontend module: `ctypes` mapping, layout computation, `struct_type`/`array_type`, decorator, signature serialization, `StructView`/`to_device` | `python/triton/language/_custom_types.py`; imports in both `core.py` copies |
| 1 | Python plumbing: allow `pointer_type` to point at composite types; extend `str_to_ty`; add the `StructView` case to `specialize_impl` | `core.py`, `language/__init__.py`, `runtime/jit.py`, `spec/ascend/runtime/jit.py` |
| 2 | C++ types and operations, verification rules, Python builder bindings | `TritonTypes.td`, `Types.h/.cpp`, `TritonOps.td`, `Ops.cpp`, `python/src/ir.cc` |
| 3 | The `tt-lower-structs` pass and its registration | `Passes.td`, `LowerStructs.cpp`, `python/src/passes.cc`, Ascend `make_ttir` |
| 4 | Ascend host/runtime wiring; interpreter behavior | `third_party/ascend/backend/driver.py` (verify pointer path), `spec/ascend/runtime/interpreter.py` |
| 5 | Tests and documentation | `python/test/unit/language/...`, pass tests, end-to-end test |

**Recommended order (vertical slices)**

1. **Slice 1 — by-reference, scalar fields.** Implement all phases for structs
   with scalar fields only, and get one Ascend kernel reading `x_view.base`
   working end to end. This proves the whole chain before adding breadth.
2. **Slice 2 — arrays and nesting.** Add `array_type`, `array_gep`/`array_extract`,
   nested structs, and `TensorView.shape[0]`.
3. **Slice 3 — value structs and control flow.** Add SROA, `scf.for`, calls and
   returns, and `struct_extract`/`struct_insert`.
4. **Slice 4 — polish.** Verification-failure tests, interpreter support,
   documentation, and clear error messages.

---

## 13. Testing

- **Layout unit tests:** the computed offsets, size, and alignment must equal
  `ctypes.offsetof`, `ctypes.sizeof`, and `ctypes.alignment` for nested, array,
  and packed structs; rejected field types must raise clear errors.
- **Frontend/lit tests:** field reads and writes; nested access; array indexing
  (static and dynamic); by-reference access; value structs passing through
  `scf.for` and function calls.
- **Verification tests (negative):** malformed layouts, by-value recursion, and
  out-of-range indices must be rejected with clear messages.
- **Pass tests:** the `tt-lower-structs` pass must leave no struct/array type
  behind.
- **End-to-end test:** a `TensorView`-style kernel on Ascend, validated against a
  host-computed reference.

---

## 14. Risks and open questions

1. **Ascend pointer lowering for `ptr<struct>`.** Ascend's memref-tuple handling
   of pointers must be checked early; this is the highest integration risk.
2. **Python bindings.** The new operations need builder bindings in
   `python/src/ir.cc` before the frontend can emit them.
3. **Frontend module resolution.** Both `core.py` copies must import the shared
   module, and the Ascend path-injection mechanism must not shadow it.
4. **Dynamic array indexing cost.** Value-form dynamic indexing is lowered to a
   select chain; a size threshold may be needed.
5. **Coexistence with `_aggregate`.** The two features are independent by
   decision, but they should not collide on shared names or markers.

---

## 15. Alternatives considered

**How to represent a struct.**

- *Frontend-only flattening (no new IR type).* The struct would exist only in
  the Python frontend; the IR would see loose scalar values and pointers. This
  is the simplest and most backend-agnostic option, but there would be no struct
  type in the IR, so the compiler could not verify or reason about it. Rejected
  in favor of a real IR type.
- *Struct type only behind pointers.* A middle ground that adds a real type only
  in pointer position. Rejected because value semantics (struct values and
  device-function returns) are also required.
- **First-class struct type (chosen).**

**How deep to keep the type.**

- *Keep the type all the way to LLVM/HIVM.* This would require new type
  conversions and operation lowerings in every backend and, for Ascend, changes
  to the proprietary `bishengir` toolchain. High risk; deferred.
- **Keep the type first-class in the frontend and early IR, then remove it with
  a shared pass (chosen).** Backend-agnostic and low-risk.

**Memory layout.**

- *Derive the layout from the target's data-layout rules.* Canonical, but it can
  silently disagree with the host C layout, which defeats the purpose of the
  feature. Rejected.
- **Record the explicit `ctypes` layout in the type (chosen).**

**Kernel boundary.**

- *Allow by-value struct launch arguments.* This requires implementing C-struct
  ABI marshalling in every backend launcher; deferred. By-value structs remain
  fully available inside the kernel and for device functions.
- **Pass structs to kernels by reference in v1 (chosen).**

**Type identity.**

- *Use the nominal `ctypes` class name.* This cannot be reconstructed from a
  signature string by `str_to_ty`, so it cannot be the basis of identity here.
  Rejected; the name is kept only as a diagnostic label.
- **Use the structural definition, including layout (chosen).**

---

## 16. Appendix: worked example

Layout of `TensorView` on a 64-bit little-endian host:

| field | type | offset | size |
|---|---|---|---|
| `shape` | `uint64[4]` | 0 | 32 |
| `stride` | `uint64[4]` | 32 | 32 |
| `base` | `void*` | 64 | 8 |
| `ndim` | `uint32` | 72 | 4 |
| **total** | | | **80** (alignment 8; 4 bytes of trailing padding) |

Kernel:

```python
@triton.jit
def example_kernel(x_view):
    x_ptr = x_view.base
    ...
```

After the `tt-lower-structs` pass, reading `base` becomes byte address
arithmetic followed by a load (conceptual form):

```
%p    = tt.addptr %x_view, 64          ; byte offset of the "base" field
%raw  = tt.load %p : !tt.ptr<i64>      ; the stored pointer value
```

No `!tt.struct` remains, so Ascend's downstream pipeline is unaffected.
