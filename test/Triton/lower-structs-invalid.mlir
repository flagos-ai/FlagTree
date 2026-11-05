// RUN: triton-opt %s --triton-lower-structs --split-input-file -verify-diagnostics

// Custom struct/array types are only supported in global memory (address space
// 1): the host builds the bytes and the kernel reaches the fields through an
// absolute address, which is meaningless in any other address space.  Such
// pointers are rejected instead of being silently lowered to a global access.

tt.func public @non_global_struct(%p: !tt.ptr<!tt.struct<{x: i32:0}, size = 4, align = 4>, 3>) {
  // expected-error @+1 {{custom struct/array types are supported in global memory only, but this pointer is in address space 3}}
  %0 = tt.struct_gep %p, "x" : !tt.ptr<!tt.struct<{x: i32:0}, size = 4, align = 4>, 3> -> !tt.ptr<i32, 3>
  %1 = tt.load %0 : !tt.ptr<i32, 3>
  tt.return
}

// -----

tt.func public @non_global_array(%p: !tt.ptr<!tt.array<i32, 4>, 2>, %i: i32) {
  // expected-error @+1 {{custom struct/array types are supported in global memory only, but this pointer is in address space 2}}
  %0 = tt.array_gep %p, %i : !tt.ptr<!tt.array<i32, 4>, 2>, i32 -> !tt.ptr<i32, 2>
  %1 = tt.load %0 : !tt.ptr<i32, 2>
  tt.return
}
