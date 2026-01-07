// RUN:  triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' --split-input-file %s | FileCheck %s

// === xor_sum ===
// === i8 u8 version ===
module {
  tt.func public @fn_npu_u8_xor_sum(
    %arg0: !tt.ptr<i8>,
    %arg1: !tt.ptr<i8>
  ) {
    %c0_i8 = arith.constant 0 : i8
    %input = tt.splat %c0_i8 : i8 -> tensor<64x32xi8>

    %reduced = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%a: i8, %b: i8):
      %xor = arith.xori %a, %b : i8
      tt.reduce.return %xor : i8
    }) : (tensor<64x32xi8>) -> tensor<64xi8>

    %ptrs = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>>
    %offs = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<i8>>, tensor<64xi32>
    tt.store %addrs, %reduced : tensor<64x!tt.ptr<i8>>

    tt.return
  }
}

// CHECK: linalg.reduce ins(%{{.*}} : tensor<64x32xi8>) outs(%{{.*}} : tensor<64xi8>) dimensions = [1]
// CHECK-NEXT: (%{{.*}}: i8, %{{.*}}: i8) {
// CHECK-NEXT:   %{{[0-9]+}} = arith.xori %{{.*}}, %{{.*}} : i8
// CHECK-NEXT:   linalg.yield %{{[0-9]+}} : i8
// CHECK: bufferization.materialize_in_destination


// === i16 u16 version ===
module {
  tt.func public @fn_npu_u16_xor_sum(
    %arg0: !tt.ptr<i16>,
    %arg1: !tt.ptr<i16>
  ) {
    %c0_i16 = arith.constant 0 : i16
    %input = tt.splat %c0_i16 : i16 -> tensor<64x32xi16>

    %reduced = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%a: i16, %b: i16):
      %xor = arith.xori %a, %b : i16
      tt.reduce.return %xor : i16
    }) : (tensor<64x32xi16>) -> tensor<64xi16>

    %ptrs = tt.splat %arg1 : !tt.ptr<i16> -> tensor<64x!tt.ptr<i16>>
    %offs = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<i16>>, tensor<64xi32>
    tt.store %addrs, %reduced : tensor<64x!tt.ptr<i16>>

    tt.return
  }
}

// CHECK: linalg.reduce ins(%{{.*}} : tensor<64x32xi16>) outs(%{{.*}} : tensor<64xi16>) dimensions = [1]
// CHECK-NEXT: (%{{.*}}: i16, %{{.*}}: i16) {
// CHECK-NEXT:   %{{[0-9]+}} = arith.xori %{{.*}}, %{{.*}} : i16
// CHECK-NEXT:   linalg.yield %{{[0-9]+}} : i16
// CHECK: bufferization.materialize_in_destination


// === i32 u32 version ===
module {
  tt.func public @fn_npu_u32_xor_sum(
    %arg0: !tt.ptr<i32>,
    %arg1: !tt.ptr<i32>
  ) {
    %c0_i32 = arith.constant 0 : i32
    %input = tt.splat %c0_i32 : i32 -> tensor<64x32xi32>

    %reduced = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %xor = arith.xori %a, %b : i32
      tt.reduce.return %xor : i32
    }) : (tensor<64x32xi32>) -> tensor<64xi32>

    %ptrs = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %offs = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
    tt.store %addrs, %reduced : tensor<64x!tt.ptr<i32>>

    tt.return
  }
}

// CHECK: linalg.reduce ins(%{{.*}} : tensor<64x32xi32>) outs(%{{.*}} : tensor<64xi32>) dimensions = [1]
// CHECK-NEXT: (%{{.*}}: i32, %{{.*}}: i32) {
// CHECK-NEXT:   %{{[0-9]+}} = arith.xori %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:   linalg.yield %{{[0-9]+}} : i32
// CHECK: bufferization.materialize_in_destination


// === i64 u64 version ===
module {
  tt.func public @fn_npu_u64_xor_sum(
    %arg0: !tt.ptr<i64>,
    %arg1: !tt.ptr<i64>
  ) {
    %c0_i64 = arith.constant 0 : i64
    %input = tt.splat %c0_i64 : i64 -> tensor<64x32xi64>

    %reduced = "tt.reduce"(%input) <{axis = 1 : i32}> ({
    ^bb0(%a: i64, %b: i64):
      %xor = arith.xori %a, %b : i64
      tt.reduce.return %xor : i64
    }) : (tensor<64x32xi64>) -> tensor<64xi64>

    %ptrs = tt.splat %arg1 : !tt.ptr<i64> -> tensor<64x!tt.ptr<i64>>
    %offs = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %addrs = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<i64>>, tensor<64xi32>
    tt.store %addrs, %reduced : tensor<64x!tt.ptr<i64>>

    tt.return
  }
}

// CHECK: linalg.reduce ins(%{{.*}} : tensor<64x32xi64>) outs(%{{.*}} : tensor<64xi64>) dimensions = [1]
// CHECK-NEXT: (%{{.*}}: i64, %{{.*}}: i64) {
// CHECK-NEXT:   %{{[0-9]+}} = arith.xori %{{.*}}, %{{.*}} : i64
// CHECK-NEXT:   linalg.yield %{{[0-9]+}} : i64
// CHECK: bufferization.materialize_in_destination