// RUN: triton-opt --update-for-ops %s --allow-unregistered-dialect | FileCheck %s

// Test update-for-ops pass adds iter_args for block counters, inner dependency conditions and inserts PIPE_S sync
// CHECK: func.func @test_update_for_ops
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_update_for_ops(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 5 : i32} dense<[8, 8, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 5 : i32} dense<[128, 8, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 8 : i32} 0.000000e+00 : f32
    %c28_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 28 : i32
    %c65536_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 65536 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    %cst_2 = arith.constant {ssbuffer.block_id = 8 : i32} 5.000000e-01 : f32
    %cst_3 = arith.constant {ssbuffer.block_id = 8 : i32} 0xFF800000 : f32
    %cst_4 = arith.constant {ssbuffer.block_id = 8 : i32} 1.000000e+00 : f32
    %c128 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : index
    scope.scope : () -> () {
      // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
      %0 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<128x128xf32>
      %1 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<128x128xf32>
      %2 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst_1 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %3 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst_2 : f32) outs(%1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %4 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<128xf32>
      %5 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst_3 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
      %6 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst_4 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
      scf.for %arg16 = %arg13 to %c65536_i32 step %c28_i32  : i32 {
        %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        %alloc_7 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_7 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %7 = bufferization.to_tensor %memspacecast writable : memref<128xf32> to tensor<128xf32>
        %alloc_8 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast_9 = memref.memory_space_cast %alloc_8 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %8 = bufferization.to_tensor %memspacecast_9 writable : memref<128xf32> to tensor<128xf32>
        // CHECK: scf.for %arg17
        // CHECK: iter_args({{.*}}, {{.*}}, {{.*}}, %{{.*}}, %{{.*}}) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, i32, i32)
        %9:3 = scf.for %arg17 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg18 = %6, %arg19 = %2, %arg20 = %5) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>)  : i32 {
          // CHECK: hivm.hir.sync_block_wait[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          %true = arith.constant true
          %10:2 = scf.if %true -> (tensor<128xf32>, tensor<128xf32>) {
            hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
            %memspacecast_11 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 5 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
            %12 = bufferization.to_tensor %memspacecast_11 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32> to tensor<128x128xf32>
            %13 = arith.mulf %12, %3 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
            %reduced = linalg.reduce ins(%13 : tensor<128x128xf32>) outs(%5 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
              (%in: f32, %init: f32) {
                %27 = arith.maximumf %in, %init : f32
                linalg.yield %27 : f32
              }
            %14 = arith.maximumf %arg20, %reduced {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
            %broadcasted = linalg.broadcast ins(%14 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
            %15 = arith.subf %13, %broadcasted {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
            %16 = math.exp %15 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
            %17 = arith.truncf %16 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
            %reshape = tensor.reshape %17(%cst_0) {ssbuffer.block_id = 5 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
            annotation.mark %reshape {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
            %18 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x128x16xf16>
            %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%18 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 5 : i32}
            %reshape_12 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 5 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
            annotation.mark %reshape_12 {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
            %19 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_1 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
            %reduced_13 = linalg.reduce ins(%16 : tensor<128x128xf32>) outs(%19 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
              (%in: f32, %init: f32) {
                %27 = arith.addf %in, %init : f32
                linalg.yield %27 : f32
              }
            %20 = arith.subf %arg20, %14 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
            %21 = math.exp %20 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
            %22 = arith.divui %arg17, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
            %c0_i32_14 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : i32
            %c2_i32 = arith.constant {ssbuffer.block_id = 5 : i32} 2 : i32
            %23 = arith.remsi %22, %c2_i32 {ssbuffer.block_id = 5 : i32} : i32
            %24 = arith.cmpi eq, %23, %c0_i32_14 {ssbuffer.block_id = 5 : i32} : i32
            scf.if %24 {
              bufferization.materialize_in_destination %21 in restrict writable %memspacecast {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
            } else {
              bufferization.materialize_in_destination %21 in restrict writable %memspacecast_9 {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
            } {ssbuffer.block_id = 5 : i32}
            %25 = arith.mulf %arg18, %21 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
            %26 = arith.addf %25, %reduced_13 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
            hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
            hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
            hivm.hir.copy ins(%reshape_12 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}
            hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
            scf.yield %14, %26 : tensor<128xf32>, tensor<128xf32>
          } else {
            scf.yield %arg20, %arg18 : tensor<128xf32>, tensor<128xf32>
          } {ssbuffer.if = 5 : i32}
          %true_10 = arith.constant true
          %11 = scf.if %true_10 -> (tensor<128x128xf32>) {
            hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
            %memspacecast_11 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 6 : i32, ssbuffer.crossDeps = [1, 0], ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
            %12 = bufferization.to_tensor %memspacecast_11 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32> to tensor<128x128xf32>
            %13 = arith.divui %arg17, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
            %c2_i32 = arith.constant {ssbuffer.block_id = 6 : i32} 2 : i32
            %14 = arith.remsi %13, %c2_i32 {ssbuffer.block_id = 6 : i32} : i32
            %c0_i32_12 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : i32
            %15 = arith.cmpi eq, %14, %c0_i32_12 {ssbuffer.block_id = 6 : i32} : i32
            %16 = scf.if %15 -> (tensor<128xf32>) {
              %19 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 6 : i32} : memref<128xf32> to tensor<128xf32>
              scf.yield %19 : tensor<128xf32>
            } else {
              %19 = bufferization.to_tensor %memspacecast_9 restrict writable {ssbuffer.block_id = 6 : i32} : memref<128xf32> to tensor<128xf32>
              scf.yield %19 : tensor<128xf32>
            } {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]}
            %broadcasted = linalg.broadcast ins(%16 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
            %17 = arith.mulf %arg19, %broadcasted {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
            %18 = arith.addf %12, %17 {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
            hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
            scf.yield %18 : tensor<128x128xf32>
          } else {
            scf.yield %arg19 : tensor<128x128xf32>
          } {ssbuffer.if = 6 : i32}
          // CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_S>, <PIPE_S>] flag = 15
          // CHECK: scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
          scf.yield %10#1, %11, %10#0 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>
        } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
      } {ssbuffer.block_id = 10 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
