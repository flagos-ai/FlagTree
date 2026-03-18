[中文版|[English](./tl_primitive_tests.md)]

## FlagTree 原语列表及测试代码全集

本文罗列了后端接入 FlagTree 应支持的 `triton.language` 原语列表及测试代码全集。
测试文件中的 `is_cuda()`、`is_hip*()`、`torch.cuda.get_device_capability()` 等方法可进行后端定制。
原语总数 167，基础原语总数 120，可选原语总数 47（文档中标注 Optional）。

---

## `python/test/unit/language/test_core.py`

**范围（Scope）**: 核心语言原语 —— 算术、内存访问、控制流、归约、点积、原子操作、类型转换、张量操作、调度相关属性等。

- **`tl.load`**: Load tensor elements from memory given a pointer or index expression.
- **`tl.store`**: Store tensor elements to memory at a given pointer or index expression.
- **Binary arithmetic (`+`, `-`, `*`, `/`, `%`, `//`)**: Elementwise addition, subtraction, multiplication, division, remainder, floor-division.
- **Bitwise ops (`&`, `|`, `^`, `<<`, `>>`, `~`)**: Elementwise bitwise and/or/xor, left/right shift, and bitwise not on integer types.
- **Comparisons (`==`, `!=`, `<`, `>`, `<=`, `>=`)**: Elementwise comparison returning boolean masks.
- **`tl.abs`**: Elementwise absolute value for integer and floating-point inputs.
- **`tl.exp`, `tl.log`, `tl.cos`, `tl.sin`, `tl.exp2`, `tl.log2`, `tl.sqrt`, `tl.rsqrt`, `tl.floor`, `tl.ceil`**: Standard elementwise math functions (exponential, logarithm, trig, square root, reciprocal sqrt, floor, ceil).
- **`tl.math.erf`**: Elementwise error function.
- **`tl.math.fma` (Optional)**: Fused multiply-add, computing `x * y + z` with a single rounding.
- **`tl.math.fdiv`, `tl.math.div_rn` (Optional)**: Floating-point division with specific rounding semantics.
- **`tl.math.sqrt_rn` (Optional)**: Square root with defined “round to nearest” semantics.
- **`tl.arange`**: Create a 1D tensor of consecutive indices over a specified range.
- **`tl.full`**: Create a tensor filled with a scalar value, given a shape and dtype.
- **`tl.zeros`**: Create a tensor of zeros for a given shape and dtype.
- **`tl.zeros_like`**: Create a tensor of zeros with the same shape and dtype as a given tensor.
- **`tl.reshape`**: View tensor data with a new shape without changing underlying storage order.
- **`tl.broadcast`, `tl.broadcast_to`**: Expand a tensor to a target shape via broadcasting semantics.
- **`tl.expand_dims` (Optional)**: Insert singleton dimensions into a tensor’s shape at given axes.
- **`tl.slice` (Optional)**: Take a slice view along dimensions with specified start/size.
- **`tl.permute` (Optional)**: Reorder tensor dimensions according to a permutation.
- **`tl.trans`**: Transpose 2D or higher-rank tensors by swapping dimensions.
- **`tl.where`**: Select elements from two tensors based on a boolean condition tensor.
- **`tl.program_id`**: Get current program (block) index along a given axis.
- **`tl.num_programs`**: Get the number of programs (blocks) along a given axis.
- **`tl.static_assert`**: Compile-time assertion on shapes or static properties.
- **`tl.assume` (Optional)**: Hint to the compiler that a runtime predicate always holds, enabling optimizations.
- **`tl.sum`**: Reduce tensor elements along a dimension by summation.
- **`tl.min`, `tl.max`**: Reduce tensor elements along a dimension by minimum or maximum.
- **`tl.argmin`, `tl.argmax`**: Return the index of the min or max along a dimension.
- **`tl.cumsum` / `tl.associative_scan`**: Inclusive scan/reduction over a dimension using an associative operator (e.g. sum).
- **`tl.dot`**: Matrix multiplication or batched dot product between 2D or higher-rank tensors.
- **`tl.dot_scaled` (Optional)**: Matrix multiply with an additional scaling factor (and optional quantization-related behavior).
- **`tl.atomic_add`, `tl.atomic_min`, `tl.atomic_max`, `tl.atomic_cas`, `tl.atomic_and`, `tl.atomic_or`, `tl.atomic_xor` (Optional), `tl.atomic_xchg`**: Atomic read-modify-write operations on memory to safely update shared locations across threads.
- **`tl.cat`**: Concatenate tensors along a given axis.
- **`tl.join`**: Join multiple tensors into a larger tensor in a structured way (also used with MMA layouts).
- **`tl.split`**: Split a tensor into multiple tensors along an axis.
- **`tl.interleave`**: Interleave bits or lanes from two tensors into one (often for packing/unpacking operations).
- **`tl.cast`**: Convert tensor elements to a target type, with optional casting modes.
- **`tl.gather`**: Index-based gather of elements according to an index tensor along a given axis.
- **`tl.histogram` (Optional)**: Build a histogram of values into bins, using atomics for accumulation.
- **`tl.clamp`**: Clamp values within `[min, max]` bounds.
- **`tl.cdiv`**: Integer ceil-division (i.e., `(x + y - 1) // y`).
- **`tl.range`**: Loop-range helper used inside Triton `for` loops with tiling and staging options.
- **`tl.map_elementwise` (Optional)**: Apply a user-provided scalar function to elements of one or more tensors elementwise.
- **`tl.inline_asm_elementwise` (Optional)**: Apply inline assembly to each element or pack of elements.
- **`tl.device_assert`**: Assertion inside a kernel that aborts execution when the predicate fails.
- **`tl.device_print`**: Device-side printing of tensor values for debugging.
- **`tl.num_warps` (Optional), `tl.num_ctas` (Optional), `tl.maxnreg` (Optional) (via tests on kernel configs)**: Knobs that control kernel launch configuration and register usage (tested indirectly through kernel decorators and attributes).
- **`tl.constexpr` / `tl.const` / `tl.constexpr_type` (Optional)**: Mark arguments or values as compile-time constants and represent constexpr types.
- **`tl.tensor`**: Construct a tensor type object (used for meta-programming and type tests).
- **`tl.tuple` (Optional), `tl.tuple_type` (Optional)**: Represent and manipulate tuples at the IR level.
- **`tl.pointer_type`**: Represent pointer types in Triton IR.
- **`tl.dtype`**: Utility to represent and manipulate Triton dtypes.
- **`tl.bfloat16`, `tl.float16`, `tl.float32`, `tl.float64`, `tl.float8e4b15` (Optional), `tl.float8e4nv`, `tl.float8e4b8` (Optional), `tl.float8e5`, `tl.float8e5b16` (Optional)**: Floating-point dtype constants.
- **`tl.int1`, `tl.int8`, `tl.int16`, `tl.int32`, `tl.int64`, `tl.uint8`, `tl.uint16`, `tl.uint32`, `tl.uint64`**: Integer dtype constants.
- **`tl.umulhi` (Optional)**: Unsigned integer multiply returning the high word of the product.
- **`tl.view`**: View tensor with a new shape (akin to reshape with layout guarantees).
- **`tl.make_tensor_descriptor`, `tl.tensor_descriptor` (Optional), `tl.tensor_descriptor_type` (Optional), `tl.load_tensor_descriptor` (Optional), `tl.store_tensor_descriptor` (Optional)**: Describe and manipulate logical tensor layouts and their loads/stores (tested more thoroughly in `test_tensor_descriptor.py`).
- **`tl.make_block_ptr`**: Construct a block pointer describing a tile within a tensor (also tested in `test_block_pointer.py`).
- **`tl.advance`**: Advance a block pointer along specified dimensions by a step.
- **`tl.broadcast_to`**: Broadcast a tensor to a target shape (alias/helper of `broadcast`).
- **`tl.reduce`**: Generic reduction over an axis, parameterized by reduction kind.
- **`tl.static_range`**: Compile-time loop range for unrolled/known-tripcount loops.
- **`tl.static_print` (Optional)**: Print compile-time constants during compilation (not at runtime).
- **`tl.static_range` / unroll attributes**: Used for controlling loop unrolling and LICM behavior.

---

## `python/test/unit/language/test_standard.py`

**范围（Scope）**: 实现在 `triton.language.standard` 中的高层 “standard” 原语。

- **`tl.maximum`**: Elementwise maximum between two tensors, matching NumPy’s `maximum`.
- **`tl.minimum`**: Elementwise minimum between two tensors, matching NumPy’s `minimum`.
- **`tl.sort` (Optional)**: Sort elements along a dimension, returning a tensor of sorted values.
- **`tl.topk` (Optional)**: Extract top‑k elements along a dimension (used through `sort` tests when `k` is not `None`).
- **`tl.flip`**: Reverse elements along a given dimension.
- **`tl.ravel` (Optional)**: Flatten a tensor into a 1D contiguous view.
- **`tl.swizzle2d` (Optional)**: Compute a 2D swizzled index mapping for tiling/permutation of matrix coordinates.

---

## `python/test/unit/language/test_libdevice.py`

**范围（Scope）**: 通过 `triton.language.extra.libdevice` 暴露出来的 CUDA libdevice 数学函数。

- **`libdevice.j0` (Optional)**: Bessel function of the first kind, order 0, computed elementwise.
- **`libdevice.j1` (Optional)**: Bessel function of the first kind, order 1, computed elementwise.
- **`libdevice.y0` (Optional)**: Bessel function of the second kind, order 0, computed elementwise.
- **`libdevice.y1` (Optional)**: Bessel function of the second kind, order 1, computed elementwise.
- **`libdevice.cyl_bessel_i0` (Optional)**: Modified Bessel function of the first kind, order 0.
- **`libdevice.cyl_bessel_i1` (Optional)**: Modified Bessel function of the first kind, order 1.
- **`libdevice.fast_dividef` (Optional) (and its alias `my_fast_dividef` (Optional))**: Fast single‑precision division helper, used as an optimized approximation to `x / y`.

---

## `python/test/unit/language/test_matmul.py`

**范围（Scope）**: 矩阵乘法以及相关的低精度 / 缩放变体。

- **`tl.dot`**: GEMM‑like matrix multiply (including batched and 3D cases) with support for various layouts and dtypes.
- **`tl.dot_scaled` (Optional)**: Matrix multiply where one or both inputs are scaled or quantized, e.g. for MXFP or FP4 formats.
- **`tl.make_tensor_descriptor` / `tl.tensor_descriptor` (Optional)**: Used to describe tiled matmul layouts (also heavily tested in `test_tensor_descriptor.py`).
- **`tl.make_block_ptr` / `tl.advance`**: Used for block‑tiled matrix loads/stores for matmul kernels.

---

## `python/test/unit/language/test_random.py`

**范围（Scope）**: 随机数生成原语。

- **`tl.randint` (Optional)**: Generate uniform integer random values in a specified range, elementwise.
- **`tl.rand` (Optional)**: Generate uniform floating‑point random values in `[0, 1)`, elementwise.
- **`tl.randn` (Optional)**: Generate normal‑distributed random values (mean 0, variance 1), elementwise.
- **`tl.rand4x` (Optional), `tl.randint4x` (Optional), `tl.randn4x` (Optional)**: Vectorized 4‑wide variants producing 4 random values per invocation.
- **`tl.philox`, `tl.philox_impl` (Optional)**: Counter‑based Philox random number generator core.
- **`tl.pair_uniform_to_normal`**: Transform pairs of uniform random numbers into normally distributed numbers (e.g. via Box–Muller).
- **`tl.uint_to_uniform_float`**: Map an unsigned integer to a uniform float in `[0, 1)`.

---

## `python/test/unit/language/test_conversions.py`

**范围（Scope）**: 类型转换，尤其是 FP8 等低精度格式。

- **`x.to(dtype, fp_downcast_rounding=...)` (tensor `.to` used on `tl` tensors)**: Convert tensor to another element type with optional rounding mode for fp downcasts.

---

## `python/test/unit/language/test_block_pointer.py`

**范围（Scope）**: block pointer 创建，高级索引和边界检查。

- **`tl.make_block_ptr`**: Create a block pointer describing a tile (block) view into a 1D/2D/3D tensor with shape, strides, offsets, and block_shape parameters.
- **`tl.advance`**: Move a block pointer forward or backward along given dimensions by multiples of a tile size.
- **`tl.load` / `tl.store` with `boundary_check`**: Load/store via block pointers while automatically applying in‑bounds checks and optional padding options (`zero`, `nan`, or no padding).

---

## `python/test/unit/language/test_tuple.py`

**范围（Scope）**: Triton 内核中的 tuple 构造、索引、赋值和返回。

- **`tl.tuple([...])` (Optional)**: Construct a compile‑time tuple of values usable in Triton kernels.
- **`tl.static_range(len(values))`**: Iterate at compile time over tuple elements with a known number of entries.

---

## `python/test/unit/language/test_tensor_descriptor.py`

**可选测试**
**范围（Scope）**: 张量描述符的创建和使用，包括多维 / 降维场景以及 TMA（Tensor Memory Accelerator）路径。

- **`tl.tensor_descriptor` (Optional) / `tl.tensor_descriptor_type` (Optional)**: Represent a logical tensor layout (shape, strides, order, padding).
- **`tl.make_tensor_descriptor`**: Construct a tensor descriptor from base pointer, shape, and strides, suitable for logical loads/stores.
- **`tl.load_tensor_descriptor` (Optional)**: Load a tensor tile using a tensor descriptor abstraction.
- **`tl.store_tensor_descriptor` (Optional)**: Store a tensor tile using a tensor descriptor abstraction.

---

## `python/test/unit/language/test_pipeliner.py`

**范围（Scope）**: 流水线内核，以及带额外 epilogue 和量化格式的 dot 运算。

- **`tl.range` (loop form)**: Used to express pipelined loops with `num_stages` to overlap memory loads and computation.
- **`tl.dot` / `tl.dot_scaled` (Optional)**: Core compute for GEMM and pipelined matmul kernels, including scaled/quantized paths.
- **`tl.load`, `tl.store`**: Used with carefully constructed pointer arithmetic to form pipelines over matrix tiles.
- **`tl.multiple_of`**: Hint that a value is a multiple of a given number to enable alignment‑aware optimizations.
- **`tl.max_contiguous`**: Hint that the first N elements of a tensor are contiguous, enabling more aggressive vectorization.
- **`tl.static_assert`**: Compile-time checks on shapes and dtypes in pipelined kernels.
