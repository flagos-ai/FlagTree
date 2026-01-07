// RUN:  triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// === i8 u8 version ===
module {
  tt.func public @fn_npu_u8(
    %arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i8>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i8, %arg3: i8):
      %7 = arith.maxui %arg2, %arg3 : i8
      tt.scan.return %7 : i8
    }) : (tensor<8xi8>) -> tensor<8xi8>
    %5 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i8>>
    tt.return
  }
}

// -----
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi8>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi8{{.*}}> to memref<8xi8>

// CHECK: %[[OUTPUT_BUF:.*]] = memref.alloc() : memref<8xi8>

// Initialize first element
// CHECK: %[[FIRST_VAL:.*]] = memref.load %[[INPUT_BUF]][%c0] : memref<8xi8>
// CHECK: memref.store %[[FIRST_VAL]], %[[OUTPUT_BUF]][%c0] : memref<8xi8>

// Main scan loop
// CHECK: scf.for %{{.*}} = %c1 to %c8 step %c1 {
// CHECK-NEXT:   %[[PREV_IDX:.*]] = arith.subi %{{.*}}, %c1 : index
// CHECK-NEXT:   %[[CURR_INPUT:.*]] = memref.load %[[INPUT_BUF]][%{{.*}}] : memref<8xi8>
// CHECK-NEXT:   %[[PREV_OUTPUT:.*]] = memref.load %[[OUTPUT_BUF]][%[[PREV_IDX]]] : memref<8xi8>
// CHECK-NEXT:   %[[COMBINED:.*]] = arith.maxui %[[PREV_OUTPUT]], %[[CURR_INPUT]] : i8
// CHECK-NEXT:   memref.store %[[COMBINED]], %[[OUTPUT_BUF]][%{{.*}}] : memref<8xi8>
// CHECK-NEXT: }

// Final materialization
// CHECK: bufferization.materialize_in_destination


// === i16 u16 version ===
module {
  tt.func public @fn_npu_u16(
    %arg0: !tt.ptr<i16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i16> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i16>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i16, %arg3: i16):
      %7 = arith.maxui %arg2, %arg3 : i16
      tt.scan.return %7 : i16
    }) : (tensor<8xi16>) -> tensor<8xi16>
    %5 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i16>>
    tt.return
  }
}

// -----
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi16>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi16{{.*}}> to memref<8xi16>

// CHECK: %[[OUTPUT_BUF:.*]] = memref.alloc() : memref<8xi16>

// Initialize first element
// CHECK: %[[FIRST_VAL:.*]] = memref.load %[[INPUT_BUF]][%c0] : memref<8xi16>
// CHECK: memref.store %[[FIRST_VAL]], %[[OUTPUT_BUF]][%c0] : memref<8xi16>

// Main scan loop
// CHECK: scf.for %{{.*}} = %c1 to %c8 step %c1 {
// CHECK-NEXT:   %[[PREV_IDX:.*]] = arith.subi %{{.*}}, %c1 : index
// CHECK-NEXT:   %[[CURR_INPUT:.*]] = memref.load %[[INPUT_BUF]][%{{.*}}] : memref<8xi16>
// CHECK-NEXT:   %[[PREV_OUTPUT:.*]] = memref.load %[[OUTPUT_BUF]][%[[PREV_IDX]]] : memref<8xi16>
// CHECK-NEXT:   %[[COMBINED:.*]] = arith.maxui %[[PREV_OUTPUT]], %[[CURR_INPUT]] : i16
// CHECK-NEXT:   memref.store %[[COMBINED]], %[[OUTPUT_BUF]][%{{.*}}] : memref<8xi16>
// CHECK-NEXT: }

// Final materialization
// CHECK: bufferization.materialize_in_destination


// === i32 u32 version ===
module {
  tt.func public @fn_npu_u32(
    %arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i32>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %7 = arith.maxui %arg2, %arg3 : i32
      tt.scan.return %7 : i32
    }) : (tensor<8xi32>) -> tensor<8xi32>
    %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// -----
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi32>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi32{{.*}}> to memref<8xi32>

// CHECK: %[[OUTPUT_BUF:.*]] = memref.alloc() : memref<8xi32>

// Initialize first element
// CHECK: %[[FIRST_VAL:.*]] = memref.load %[[INPUT_BUF]][%c0] : memref<8xi32>
// CHECK: memref.store %[[FIRST_VAL]], %[[OUTPUT_BUF]][%c0] : memref<8xi32>

// Main scan loop
// CHECK: scf.for %{{.*}} = %c1 to %c8 step %c1 {
// CHECK-NEXT:   %[[PREV_IDX:.*]] = arith.subi %{{.*}}, %c1 : index
// CHECK-NEXT:   %[[CURR_INPUT:.*]] = memref.load %[[INPUT_BUF]][%{{.*}}] : memref<8xi32>
// CHECK-NEXT:   %[[PREV_OUTPUT:.*]] = memref.load %[[OUTPUT_BUF]][%[[PREV_IDX]]] : memref<8xi32>
// CHECK-NEXT:   %[[COMBINED:.*]] = arith.maxui %[[PREV_OUTPUT]], %[[CURR_INPUT]] : i32
// CHECK-NEXT:   memref.store %[[COMBINED]], %[[OUTPUT_BUF]][%{{.*}}] : memref<8xi32>
// CHECK-NEXT: }

// Final materialization
// CHECK: bufferization.materialize_in_destination


// === i64 u64 version ===
module {
  tt.func public @fn_npu_u64(
    %arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i64>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg2: i64, %arg3: i64):
      %7 = arith.maxui %arg2, %arg3 : i64
      tt.scan.return %7 : i64
    }) : (tensor<8xi64>) -> tensor<8xi64>
    %5 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<8x!tt.ptr<i64>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i64>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i64>>
    tt.return
  }
}

// -----
// CHECK: %[[INPUT_BUF:.*]] = memref.alloc() : memref<8xi64>
// CHECK: memref.copy {{.*}}, %[[INPUT_BUF]] : memref<8xi64{{.*}}> to memref<8xi64>

// CHECK: %[[OUTPUT_BUF:.*]] = memref.alloc() : memref<8xi64>

// Initialize first element
// CHECK: %[[FIRST_VAL:.*]] = memref.load %[[INPUT_BUF]][%c0] : memref<8xi64>
// CHECK: memref.store %[[FIRST_VAL]], %[[OUTPUT_BUF]][%c0] : memref<8xi64>

// Main scan loop
// CHECK: scf.for %{{.*}} = %c1 to %c8 step %c1 {
// CHECK-NEXT:   %[[PREV_IDX:.*]] = arith.subi %{{.*}}, %c1 : index
// CHECK-NEXT:   %[[CURR_INPUT:.*]] = memref.load %[[INPUT_BUF]][%{{.*}}] : memref<8xi64>
// CHECK-NEXT:   %[[PREV_OUTPUT:.*]] = memref.load %[[OUTPUT_BUF]][%[[PREV_IDX]]] : memref<8xi64>
// CHECK-NEXT:   %[[COMBINED:.*]] = arith.maxui %[[PREV_OUTPUT]], %[[CURR_INPUT]] : i64
// CHECK-NEXT:   memref.store %[[COMBINED]], %[[OUTPUT_BUF]][%{{.*}}] : memref<8xi64>
// CHECK-NEXT: }

// Final materialization
// CHECK: bufferization.materialize_in_destination