// RUN: triton-opt "--triton-to-linalg=global-kernel=false" --split-input-file %s | FileCheck %s

// Full rows recover a static copy; the tail keeps the exact original extent.
// The condition and allocation must stay outside both row loops.

// CHECK: module attributes {{.*}}tt.full_row_copy
// CHECK-LABEL: func.func @full_row_copy
// CHECK: %[[BUFFER:.*]] = memref.alloc() : memref<32x64xf32>
// CHECK: %[[FULL:.*]] = arith.cmpi eq,
// CHECK: scf.if %[[FULL]] {
// CHECK-NEXT: scf.for
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.if
// CHECK: memref.copy {{.*}} : memref<1x64xf32, {{.*}}> to memref<1x64xf32, {{.*}}>
// CHECK: } else {
// CHECK-NEXT: scf.for
// CHECK-NOT: memref.alloc
// CHECK-NOT: scf.if
// CHECK: memref.copy {{.*}} : memref<1x?xf32, {{.*}}> to memref<1x?xf32, {{.*}}>
// CHECK: return %[[BUFFER]]

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @full_row_copy(%input: memref<?xf32>, %rows: index, %cols: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>

      %read = memref.subview %src[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>


    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}

// -----

// The source row may already have been hoisted. Cloning must reuse that value.
// CHECK: module attributes {{.*}}tt.full_row_copy
// CHECK-LABEL: func.func @invariant_source
// CHECK: %[[SOURCE:.*]] = memref.reinterpret_cast
// CHECK: scf.if
// CHECK-NEXT: scf.for
// CHECK: memref.copy %[[SOURCE]], {{.*}} : memref<1x64xf32, {{.*}}> to memref<1x64xf32, {{.*}}>
// CHECK: } else {
// CHECK: memref.copy {{.*}} : memref<1x?xf32, {{.*}}> to memref<1x?xf32, {{.*}}>
module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @invariant_source(%input: memref<?xf32>, %rows: index, %cols: index, %base: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
    scf.for %i = %c0 to %rows step %c1 {
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %read = memref.subview %src[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}

// -----

// CHECK: module attributes
// CHECK-NOT: tt.full_row_copy
// CHECK-LABEL: func.func @row_dependent_extent
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.copy
// CHECK-NOT: scf.if
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @row_dependent_extent(%input: memref<?xf32>, %rows: index, %cols: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %local = arith.minsi %i, %cols : index
      %read = memref.subview %src[0, 0] [1, %local] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %local] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>


    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @nonzero_column_offset
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.copy
// CHECK-NOT: scf.if
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @nonzero_column_offset(%input: memref<?xf32>, %rows: index, %cols: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>

      %read = memref.subview %src[0, 1] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 1] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>


    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @strided_columns
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.copy
// CHECK-NOT: scf.if
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @strided_columns(%input: memref<?xf32>, %rows: index, %cols: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [128, 2] : memref<?xf32> to memref<1x64xf32, strided<[128, 2], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>

      %read = memref.subview %src[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[128, 2], offset: ?>> to memref<1x?xf32, strided<[128, 2], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[128, 2], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>


    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @unmarked_loop
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.copy
// CHECK-NOT: scf.if
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @unmarked_loop(%input: memref<?xf32>, %rows: index, %cols: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>

      %read = memref.subview %src[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>


    }
    return %buffer : memref<32x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @additional_read
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.copy
// CHECK-NOT: scf.if
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @additional_read(%input: memref<?xf32>, %rows: index, %cols: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>

      %read = memref.subview %src[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>

      %extra = memref.load %input[%i] : memref<?xf32>
      memref.store %extra, %buffer[%i, %c0] : memref<32x64xf32>
    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @multiple_copies
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.copy
// CHECK-NOT: scf.if
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @multiple_copies(%input: memref<?xf32>, %rows: index, %cols: index) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %buffer = memref.alloc() : memref<32x64xf32>
    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>

      %read = memref.subview %src[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>

    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}

// -----

// CHECK-LABEL: func.func @external_destination
// CHECK-NOT: scf.if
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: memref.copy
// CHECK-NOT: scf.if
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  func.func @external_destination(%input: memref<?xf32>, %rows: index, %cols: index, %buffer: memref<32x64xf32>) -> memref<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    scf.for %i = %c0 to %rows step %c1 {
      %base = arith.muli %i, %c64 : index
      %src = memref.reinterpret_cast %input to offset: [%base], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
      %row = memref.subview %buffer[%i, 0] [1, 64] [1, 1] : memref<32x64xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>

      %read = memref.subview %src[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      %write = memref.subview %row[0, 0] [1, %cols] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
      memref.copy %read, %write : memref<1x?xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>


    } {ExtractedLoadOrStore, hivm.parallel_loop}
    return %buffer : memref<32x64xf32>
  }
}
