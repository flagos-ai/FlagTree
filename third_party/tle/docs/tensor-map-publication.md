# Raw tensor-map publication and views

NVIDIA TMA can use a tensor map encoded by the host and stored in global
memory. `tle.gpu.tensor_map_fenceproxy_acquire(pointer)` publishes those
host-written descriptor bytes to the tensor-map proxy before their first TMA
use. `tle.gpu.reinterpret_tensor_map(pointer, destination)` gives the entry a
descriptor type matching the destination's physical block shape and dtype.
The encoded box shape and element type must match that destination.

A table uses ordinary byte-pointer arithmetic; it needs no separate TLE
table-entry primitive. Each CUDA tensor-map entry occupies 128 bytes:

```python
entry = table_uint8 + tl.program_id(0) * 128
tle.gpu.tensor_map_fenceproxy_acquire(entry)
descriptor = tle.gpu.reinterpret_tensor_map(entry, stage)
tle.gpu.copy(descriptor, stage, [8, 8], [0, 0])
```

The view preserves its physical block shape across runtime branches and
buffer arguments passed to helper functions. Existing TMA completion-barrier
and logical-copy-shape call forms remain available. Explicit eviction policies
use the named `eviction_policy` builder argument after the optional logical
shape.

`test_tle_tensor_map_shape.py`, `test_tle_tensor_map_table.py`, and
`test_tle_tma_copy_binding.py` cover selected table entries, publication,
scope cloning, completion barriers, and the current binding contract. The
test table encoder uses the CUDA driver directly and does not depend on
FlagMega.
