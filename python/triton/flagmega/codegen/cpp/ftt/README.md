# FTT infrastructure

The dimension, shape, vector, storage and operation APIs are adapted from
`FlagLang/ntt` at `6f629828a`. Adapted files retain Canaan's Apache-2.0 notices;
the license is in `NTT-LICENSE`.

Include `<ftt/ftt.h>` for the common API. The common headers support C++17,
CUDA host/device compilation and the K230 freestanding toolchain. Tensor views
use caller-owned memory. Vector values own their fixed lane storage.

```cpp
using namespace ftt;
auto shape = make_shape(2_dim, dim_t(3), 4_dim);
auto strides = default_strides(shape);
auto index = unravel_index(23_dim, shape);
auto offset = linear_offset(index, strides);

auto a = vector<float, 2, 4>::from_scalar(2.0f);
auto b = vector<float, 4>::from_scalar(3.0f);
auto c = mul_add(a, b, a);
auto selected = where(greater(c, 7.0f), c, 0.0f);
float sum = reduce_sum<float>(selected);
```

Dimensions preserve `fixed_dim` results when all inputs are fixed. Shapes,
dimensions and strides support generation, aggregate, contains/index lookup,
append/prepend/concat, slice/select, remove/replace and reverse. Shape utilities
include canonical and regular strides, broadcast strides, linear offsets and
sizes, index unraveling, bounds, contiguous dimensions, squeeze and unsqueeze.
Hosted builds also support shape structured bindings.

`fixed_shape<...>`, `fixed_strides<...>`, `row_major_strides_t` and the existing
tensor view interface remain available. Regular row-major strides retain
nonzero singleton strides; `default_strides` canonicalizes singleton strides to
zero. Use `default_strides<Shape, false>` for regular strides.

Vectors support multidimensional and partial indexing, scalar and trailing-axis
broadcasting, unaligned loads/stores, arithmetic and math, comparisons, selection,
clamp, reductions, products and `vmma`. Binary operations also broadcast singleton
lanes. `vector_storage_traits` and `<ftt/native_vector.h>` provide native storage
adapters; the generic storage uses nested fixed arrays.

`cast_elem` follows NTT's lane transformation: equal-width and boolean casts keep
the shape; widening adds a size-ratio axis to a one-dimensional vector; narrowing
removes that leading axis. Values are converted numerically. `mul_add` and
`mul_sub` use NTT's separate multiply and add/subtract semantics. Native fused
instructions remain in the architecture-specific interfaces.

Shapes, strides, indices and pointer spans must describe valid storage. Ceiling
division requires nonnegative numerators and positive divisors. Native adapters
must provide accessors for their lane storage. This migration supplies common
infrastructure; architecture-specific kernels keep their existing dispatch.

Run `python -m pytest python/test/flagmega/codegen/cpp/test_ftt_dimensions.py`.
The suite runs host checks and compile-failure contracts, builds CUDA PTX when
`nvcc` is available, and builds K230 objects when the toolchain is available.
`K230_LLVM_DIR` and `K230_CLANG` override the local toolchain discovery.
