// RUN: triton-opt %s --triton-lower-structs --split-input-file | FileCheck %s --implicit-check-not "!tt.struct" --implicit-check-not "!tt.array"

// Scalar field at offset 0.

// CHECK-LABEL: tt.func public @scalar_offset_zero
tt.func public @scalar_offset_zero(%p: !tt.ptr<!tt.struct<{x: i32:0}, size = 4, align = 4>>) {
  // CHECK: %[[PI:.*]] = tt.ptr_to_int %arg0 : !tt.ptr<i8> -> i64
  // CHECK: %[[C:.*]] = arith.constant 0 : i64
  // CHECK: %[[SUM:.*]] = arith.addi %[[PI]], %[[C]] : i64
  // CHECK: %[[FP:.*]] = tt.int_to_ptr %[[SUM]] : i64 -> !tt.ptr<i32>
  // CHECK: tt.load %[[FP]] : !tt.ptr<i32>
  %0 = tt.struct_gep %p, "x" : !tt.ptr<!tt.struct<{x: i32:0}, size = 4, align = 4>> -> !tt.ptr<i32>
  %1 = tt.load %0 : !tt.ptr<i32>
  tt.return
}

// -----

// Scalar field at a non-zero offset (byte offset is used directly).

// CHECK-LABEL: tt.func public @scalar_offset_nonzero
tt.func public @scalar_offset_nonzero(%p: !tt.ptr<!tt.struct<{a: i64:0, x: i32:8}, size = 16, align = 8>>) {
  // CHECK: %[[PI:.*]] = tt.ptr_to_int %arg0 : !tt.ptr<i8> -> i64
  // CHECK: %[[C:.*]] = arith.constant 8 : i64
  // CHECK: %[[SUM:.*]] = arith.addi %[[PI]], %[[C]] : i64
  // CHECK: tt.int_to_ptr %[[SUM]] : i64 -> !tt.ptr<i32>
  %0 = tt.struct_gep %p, "x" : !tt.ptr<!tt.struct<{a: i64:0, x: i32:8}, size = 16, align = 8>> -> !tt.ptr<i32>
  %1 = tt.load %0 : !tt.ptr<i32>
  tt.return
}

// -----

// Nested struct field: the outer offset is added to the (already byte) pointer,
// and the intermediate struct pointer is rewritten to an opaque pointer.

// CHECK-LABEL: tt.func public @nested
tt.func public @nested(%p: !tt.ptr<!tt.struct<{inner: !tt.struct<{v: i32:0}, size = 4, align = 4>:0}, size = 4, align = 4>>) {
  // CHECK: arith.constant 0 : i64
  // CHECK: tt.int_to_ptr %{{.*}} : i64 -> !tt.ptr<i8>
  // CHECK: arith.constant 0 : i64
  // CHECK: tt.int_to_ptr %{{.*}} : i64 -> !tt.ptr<i32>
  %0 = tt.struct_gep %p, "inner" : !tt.ptr<!tt.struct<{inner: !tt.struct<{v: i32:0}, size = 4, align = 4>:0}, size = 4, align = 4>> -> !tt.ptr<!tt.struct<{v: i32:0}, size = 4, align = 4>>
  %1 = tt.struct_gep %0, "v" : !tt.ptr<!tt.struct<{v: i32:0}, size = 4, align = 4>> -> !tt.ptr<i32>
  %2 = tt.load %1 : !tt.ptr<i32>
  tt.return
}

// -----

// Array element address: byte offset is index * sizeof(element).

// CHECK-LABEL: tt.func public @array_element
tt.func public @array_element(%p: !tt.ptr<!tt.array<i64, 4>>, %i: i32) {
  // CHECK: %[[PI:.*]] = tt.ptr_to_int %arg0 : !tt.ptr<i8> -> i64
  // CHECK: %[[IDX:.*]] = arith.extsi %arg1 : i32 to i64
  // CHECK: %[[SZ:.*]] = arith.constant 8 : i64
  // CHECK: %[[BOFF:.*]] = arith.muli %[[IDX]], %[[SZ]] : i64
  // CHECK: %[[SUM:.*]] = arith.addi %[[PI]], %[[BOFF]] : i64
  // CHECK: tt.int_to_ptr %[[SUM]] : i64 -> !tt.ptr<i64>
  %0 = tt.array_gep %p, %i : !tt.ptr<!tt.array<i64, 4>>, i32 -> !tt.ptr<i64>
  %1 = tt.load %0 : !tt.ptr<i64>
  tt.return
}

// -----

// Array of structs: the element stride is sizeof(struct), and a field of the
// addressed element is reached with a further struct_gep.

// CHECK-LABEL: tt.func public @array_of_struct
tt.func public @array_of_struct(%p: !tt.ptr<!tt.array<!tt.struct<{a: i32:0, b: f64:8}, size = 16, align = 8>, 2>>, %i: i32) {
  // CHECK: arith.constant 16 : i64
  // CHECK: arith.muli
  // CHECK: arith.constant 8 : i64
  // CHECK: tt.int_to_ptr %{{.*}} : i64 -> !tt.ptr<f64>
  %e = tt.array_gep %p, %i : !tt.ptr<!tt.array<!tt.struct<{a: i32:0, b: f64:8}, size = 16, align = 8>, 2>>, i32 -> !tt.ptr<!tt.struct<{a: i32:0, b: f64:8}, size = 16, align = 8>>
  %f = tt.struct_gep %e, "b" : !tt.ptr<!tt.struct<{a: i32:0, b: f64:8}, size = 16, align = 8>> -> !tt.ptr<f64>
  %v = tt.load %f : !tt.ptr<f64>
  tt.return
}

// -----

// A pointer-typed field: its storage becomes a ptr<i64> and the value is
// restored with an int round-trip, so no pointer-typed memory is created.

// CHECK-LABEL: tt.func public @pointer_field
tt.func public @pointer_field(%p: !tt.ptr<!tt.struct<{base: !tt.ptr<i8>:0}, size = 8, align = 8>>) {
  // CHECK: arith.constant 0 : i64
  // CHECK: tt.int_to_ptr %{{.*}} : i64 -> !tt.ptr<i64>
  // CHECK: tt.load %{{.*}} : !tt.ptr<i64>
  // CHECK: tt.int_to_ptr %{{.*}} : i64 -> !tt.ptr<i8>
  %0 = tt.struct_gep %p, "base" : !tt.ptr<!tt.struct<{base: !tt.ptr<i8>:0}, size = 8, align = 8>> -> !tt.ptr<!tt.ptr<i8>>
  %1 = tt.load %0 : !tt.ptr<!tt.ptr<i8>>
  %2 = tt.ptr_to_int %1 : !tt.ptr<i8> -> i64
  tt.return
}

// -----

// Storing a pointer value: the value is converted to i64 and stored into the
// ptr<i64> storage.

// CHECK-LABEL: tt.func public @pointer_field_store
tt.func public @pointer_field_store(%p: !tt.ptr<!tt.struct<{base: !tt.ptr<i8>:0}, size = 8, align = 8>>, %v: !tt.ptr<i8>) {
  // CHECK: tt.int_to_ptr %{{.*}} : i64 -> !tt.ptr<i64>
  // CHECK: tt.ptr_to_int %arg1 : !tt.ptr<i8> -> i64
  // CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i64>
  %0 = tt.struct_gep %p, "base" : !tt.ptr<!tt.struct<{base: !tt.ptr<i8>:0}, size = 8, align = 8>> -> !tt.ptr<!tt.ptr<i8>>
  tt.store %0, %v : !tt.ptr<!tt.ptr<i8>>
  tt.return
}
