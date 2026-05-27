# TLE DSA OP Table

This document summarizes the TLE DSA OP documents under `/documents/tle`.

| OP | Short Description | Detailed Document |
|----|-------------------|-------------------|
| `tle.dsa.alloc` | Allocates a TLE DSA buffer in a specified backend address space. | [tle.dsa.alloc.md](tle.dsa.alloc.md) |
| `tle.dsa.copy` | Copies data between GM pointer tensors and DSA buffers with an explicit copy extent. | [tle.dsa.copy.md](tle.dsa.copy.md) |
| `tle.dsa.extract_element` | Extracts a scalar element from a ranked tensor using the given indices. | [tle.dsa.extract_element.md](tle.dsa.extract_element.md) |
| `tle.dsa.extract_slice` | Extracts a subtensor from an input tensor using offsets, sizes, and strides. | [tle.dsa.extract_slice.md](tle.dsa.extract_slice.md) |
| `tle.dsa.hint` | Passes compile-time hints to TLE DSA builtins through a `with` scope, mainly for `tle.dsa.copy` `inter_no_alias`. | [tle.dsa.hint.md](tle.dsa.hint.md) |
| `tle.dsa.insert_slice` | Inserts a subtensor into a specified slice region and returns the updated full tensor. | [tle.dsa.insert_slice.md](tle.dsa.insert_slice.md) |
| `tle.dsa.parallel` | Represents loop iterations as independent and suitable for parallel semantics in JIT code. | [tle.dsa.parallel.md](tle.dsa.parallel.md) |
| `tle.dsa.subview` | Creates a subview buffer from an existing TLE DSA buffer without copying data. | [tle.dsa.subview.md](tle.dsa.subview.md) |
| `tle.dsa.to_buffer` | Converts a non-scalar `tl.tensor` into a TLE DSA buffer in an explicit address space. | [tle.dsa.to_buffer.md](tle.dsa.to_buffer.md) |
| `tle.dsa.to_tensor` | Converts a TLE DSA buffer into a `tl.tensor`. | [tle.dsa.to_tensor.md](tle.dsa.to_tensor.md) |
