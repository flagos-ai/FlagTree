// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
    func.func @test_fill_init(%arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}){
    %c1_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32

    %0 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %1 = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_0 : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %91 = scf.for %arg20 = %c0_i32 to %c128_i32 step %c1_i32 iter_args(%arg21 = %1) -> (tensor<32x32xf32>)  : i32 {

        %60 = math.exp %arg21 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

        %182 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%0, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>

        scf.yield {ssbuffer.core_type = "CUBE"} %182 : tensor<32x32xf32>
    } {ssbuffer.core_type = "CUBE"}
    return
}}


// CHECK-LABEL: func.func @test_fill_init
// CHECK: %[[FILL_1:[a-z0-9_]+]] = linalg.fill

// CHECK: scf.for {{.*}} iter_args(%[[ARG_2:[a-z0-9_]+]] = %[[FILL_1]])

// CHECK: math.exp %[[ARG_2:[a-z0-9_]+]]
// CHECK: %[[MATMUL_4:[a-z0-9_]+]] = linalg.matmul

// CHECK: hivm.hir.sync_block_wait
// CHECK: hivm.hir.fixpipe {{.*}} ins(%[[MATMUL_4]] : tensor<32x32xf32>)
// CHECK: hivm.hir.sync_block_set

// CHECK: hivm.hir.sync_block_wait
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_5:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: arith.constant
// CHECK: hivm.hir.sync_block_set

// CHECK: scf.yield {ssbuffer.core_type = "VECTOR"} %[[TENSOR_5]] : tensor<32x32xf32>
// CHECK: } {{{.*}}ssbuffer.core_type = "VECTOR"{{.*}}}
