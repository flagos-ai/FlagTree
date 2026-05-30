// RUN: triton-opt -add-control-flow-condition %s | FileCheck %s

// CHECK-LABEL: module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">}
// CHECK-LABEL: func.func @_attn_fwd(
// CHECK-DAG: [[C3:%[^ ]+]] = arith.constant 3 : i32

// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi slt
// CHECK-DAG: arith.cmpi slt

// CHECK: [[AND1:%[^ ]+]] = arith.andi
// CHECK: [[AND2:%[^ ]+]] = arith.andi [[AND1]]
// CHECK: [[AND3:%[^ ]+]] = arith.andi [[AND2]]
// CHECK: [[AND4:%[^ ]+]] = arith.andi [[AND3]]
// CHECK: [[AND5:%[^ ]+]] = arith.andi [[AND4]]
// CHECK: [[AND6:%[^ ]+]] = arith.andi [[AND5]]
// CHECK: [[AND7:%[^ ]+]] = arith.andi [[AND6]]

// CHECK: [[FINAL_CMP:%[^ ]+]] = arith.cmpi slt
// CHECK: [[AND8:%[^ ]+]] = arith.andi
// CHECK: [[COND:%[^ ]+]] = arith.andi [[AND8]]

// CHECK: scf.if [[COND]]


// 匹配常量 1
// CHECK: [[C1:%[^ ]+]] = arith.constant 1 : i32
// CHECK-DAG: arith.addi %arg24, [[C1]] : i32
// CHECK-DAG: arith.addi %arg25, [[C1]] : i32
// CHECK-DAG: arith.addi %arg23, [[C1]] : i32
// CHECK-DAG: arith.addi %arg27, [[C1]] : i32
// CHECK-DAG: arith.addi %arg26, [[C1]] : i32
// CHECK-DAG: arith.addi %arg28, [[C1]] : i32
// CHECK-DAG: arith.addi %arg29, [[C1]] : i32
// CHECK-DAG: arith.addi %arg30, [[C1]] : i32
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 5 : i32} dense<[8, 8, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 5 : i32} dense<[128, 8, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 8 : i32} 0.000000e+00 : f32
    %c28_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 28 : i32
    %c65536_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 65536 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8192 : i32
    %c128_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : i32
    %c1048576_i64 = arith.constant {ssbuffer.block_id = 8 : i32} 1048576 : i64
    %c8388608_i64 = arith.constant {ssbuffer.block_id = 8 : i32} 8388608 : i64
    %c8_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 8 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 64 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 8 : i32} 0 : i32
    %cst_2 = arith.constant {ssbuffer.block_id = 8 : i32} 5.000000e-01 : f32
    %cst_3 = arith.constant {ssbuffer.block_id = 8 : i32} 0xFF800000 : f32
    %cst_4 = arith.constant {ssbuffer.block_id = 8 : i32} 1.000000e+00 : f32
    %c128 = arith.constant {ssbuffer.block_id = 8 : i32} 128 : index
    scope.scope : () -> () {
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
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32]} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossDeps = [1 : i32, 1 : i32]} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        %alloc_7 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_7 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %7 = bufferization.to_tensor %memspacecast writable : memref<128xf32> to tensor<128xf32>
        %alloc_8 = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
        %memspacecast_9 = memref.memory_space_cast %alloc_8 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
        %8 = bufferization.to_tensor %memspacecast_9 writable : memref<128xf32> to tensor<128xf32>
        %9:3 = scf.for %arg17 = %c0_i32 to %c8192_i32 step %c128_i32 iter_args(%arg18 = %6, %arg19 = %2, %arg20 = %5) -> (tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>)  : i32 {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
          %memspacecast_11 = memref.memory_space_cast %alloc_5 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %33 = bufferization.to_tensor %memspacecast_11 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32> to tensor<128x128xf32>
          %34 = arith.mulf %33, %3 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %reduced = linalg.reduce ins(%34 : tensor<128x128xf32>) outs(%5 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
            (%in: f32, %init: f32) {
              %55 = arith.maximumf %in, %init : f32
              linalg.yield %55 : f32
            }
          %35 = arith.maximumf %arg20, %reduced {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %broadcasted_12 = linalg.broadcast ins(%35 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
          %36 = arith.subf %34, %broadcasted_12 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %37 = math.exp %36 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32>
          %38 = arith.truncf %37 {ssbuffer.block_id = 5 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          %reshape = tensor.reshape %38(%cst_0) {ssbuffer.block_id = 5 : i32} : (tensor<128x128xf16>, tensor<3xi64>) -> tensor<128x8x16xf16>
          annotation.mark %reshape {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<128x8x16xf16>
          %39 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<8x128x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<128x8x16xf16>) outs(%39 : tensor<8x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 5 : i32}
          %reshape_13 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 5 : i32} : (tensor<8x128x16xf16>, tensor<4xi64>) -> tensor<8x8x16x16xf16>
          annotation.mark %reshape_13 {ssbuffer.block_id = 5 : i32, tiling_dim_mapping = {"1" = 1 : index}} : tensor<8x8x16x16xf16>
          %40 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_1 : f32) outs(%4 : tensor<128xf32>) -> tensor<128xf32>
          %reduced_14 = linalg.reduce ins(%37 : tensor<128x128xf32>) outs(%40 : tensor<128xf32>) dimensions = [1]  {ssbuffer.block_id = 5 : i32}
            (%in: f32, %init: f32) {
              %55 = arith.addf %in, %init : f32
              linalg.yield %55 : f32
            }
          %41 = arith.subf %arg20, %35 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %42 = math.exp %41 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %43 = arith.divui %arg17, %c128_i32 {ssbuffer.block_id = 5 : i32} : i32
          %c0_i32_15 = arith.constant {ssbuffer.block_id = 5 : i32} 0 : i32
          %c2_i32 = arith.constant {ssbuffer.block_id = 5 : i32} 2 : i32
          %44 = arith.remsi %43, %c2_i32 {ssbuffer.block_id = 5 : i32} : i32
          %45 = arith.cmpi eq, %44, %c0_i32_15 {ssbuffer.block_id = 5 : i32} : i32
          scf.if %45 {
            bufferization.materialize_in_destination %42 in restrict writable %memspacecast {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
          } else {
            bufferization.materialize_in_destination %42 in restrict writable %memspacecast_9 {ssbuffer.block_id = 5 : i32} : (tensor<128xf32>, memref<128xf32>) -> ()
          } {ssbuffer.block_id = 5 : i32}
          %46 = arith.mulf %arg18, %42 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          %47 = arith.addf %46, %reduced_14 {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
          hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          hivm.hir.copy ins(%reshape_13 : tensor<8x8x16x16xf16>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
          %memspacecast_16 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32, ssbuffer.crossDeps = [1, 0]} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %48 = bufferization.to_tensor %memspacecast_16 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32> to tensor<128x128xf32>
          %49 = arith.divui %arg17, %c128_i32 {ssbuffer.block_id = 6 : i32} : i32
          %c2_i32_17 = arith.constant {ssbuffer.block_id = 6 : i32} 2 : i32
          %50 = arith.remsi %49, %c2_i32_17 {ssbuffer.block_id = 6 : i32} : i32
          %c0_i32_18 = arith.constant {ssbuffer.block_id = 6 : i32} 0 : i32
          %51 = arith.cmpi eq, %50, %c0_i32_18 {ssbuffer.block_id = 6 : i32} : i32
          %producer_buf = memref.alloc() {ssbuffer.block_id = 6 : i32} : memref<128xf32, #hivm.address_space<ub>>
          %producer_cast = memref.memory_space_cast %producer_buf {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
          %producer_tensor = bufferization.to_tensor %producer_cast writable {ssbuffer.block_id = 6 : i32} : memref<128xf32> to tensor<128xf32>

          %c1_buf = memref.alloc() {ssbuffer.block_id = 6 : i32} : memref<128xf32, #hivm.address_space<ub>>
          %c1_cast = memref.memory_space_cast %c1_buf {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
          %c1_tensor = bufferization.to_tensor %c1_cast writable {ssbuffer.block_id = 6 : i32} : memref<128xf32> to tensor<128xf32>
          %use1 = arith.mulf %producer_tensor, %producer_tensor {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : tensor<128xf32>

          %c2_buf = memref.alloc() {ssbuffer.block_id = 6 : i32} : memref<128xf32, #hivm.address_space<ub>>
          %c2_cast = memref.memory_space_cast %c2_buf {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
          %c2_tensor = bufferization.to_tensor %c2_cast writable {ssbuffer.block_id = 6 : i32} : memref<128xf32> to tensor<128xf32>
          %use2 = arith.addf %producer_tensor, %producer_tensor {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : tensor<128xf32>

          %c3_buf = memref.alloc() {ssbuffer.block_id = 6 : i32} : memref<128xf32, #hivm.address_space<ub>>
          %c3_cast = memref.memory_space_cast %c3_buf {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : memref<128xf32, #hivm.address_space<ub>> to memref<128xf32>
          %c3_tensor = bufferization.to_tensor %c3_cast writable {ssbuffer.block_id = 6 : i32} : memref<128xf32> to tensor<128xf32>
          %use3 = arith.subf %producer_tensor, %producer_tensor {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : tensor<128xf32>

          %52 = scf.if %51 -> (tensor<128xf32>) {
            %55 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : memref<128xf32> to tensor<128xf32>
            scf.yield %55 : tensor<128xf32>
          } else {
            %55 = bufferization.to_tensor %memspacecast_9 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32]} : memref<128xf32> to tensor<128xf32>
            scf.yield %55 : tensor<128xf32>
          } {ssbuffer.block_id = 6 : i32}
          %broadcasted_19 = linalg.broadcast ins(%52 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 6 : i32}
          %53 = arith.mulf %arg19, %broadcasted_19 {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
          %54 = arith.addf %48, %53 {ssbuffer.block_id = 6 : i32} : tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
          scf.yield %47, %54, %35 : tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>
        } {ssbuffer.block_id = 9 : i32, ssbuffer.main_loop = 0 : i64}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        %10 = arith.divsi %arg16, %c64_i32 {ssbuffer.block_id = 7 : i32} : i32
        %11 = arith.remsi %arg16, %c64_i32 {ssbuffer.block_id = 7 : i32} : i32
        %12 = arith.divsi %10, %c8_i32 {ssbuffer.block_id = 7 : i32} : i32
        %13 = arith.remsi %10, %c8_i32 {ssbuffer.block_id = 7 : i32} : i32
        %14 = arith.extsi %12 {ssbuffer.block_id = 7 : i32} : i32 to i64
        %15 = arith.muli %14, %c8388608_i64 {ssbuffer.block_id = 7 : i32} : i64
        %16 = arith.extsi %13 {ssbuffer.block_id = 7 : i32} : i32 to i64
        %17 = arith.muli %16, %c1048576_i64 {ssbuffer.block_id = 7 : i32} : i64
        %18 = arith.addi %15, %17 {ssbuffer.block_id = 7 : i32} : i64
        %19 = arith.index_cast %18 {ssbuffer.block_id = 7 : i32} : i64 to index
        %20 = arith.muli %11, %c128_i32 {ssbuffer.block_id = 7 : i32} : i32
        %21 = arith.maxsi %20, %c0_i32 {ssbuffer.block_id = 7 : i32} : i32
        %22 = arith.index_cast %21 {ssbuffer.block_id = 7 : i32} : i32 to index
        %23 = arith.muli %22, %c128 {ssbuffer.block_id = 7 : i32} : index
        %24 = arith.addi %23, %19 {ssbuffer.block_id = 7 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%24], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 7 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
        %25 = math.log %9#0 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
        %26 = arith.addf %9#2, %25 {ssbuffer.block_id = 7 : i32} : tensor<128xf32>
        %broadcasted = linalg.broadcast ins(%9#0 : tensor<128xf32>) outs(%1 : tensor<128x128xf32>) dimensions = [1]  {ssbuffer.block_id = 7 : i32}
        %27 = arith.divf %9#1, %broadcasted {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
        %28 = arith.muli %10, %c8192_i32 {ssbuffer.block_id = 7 : i32} : i32
        %29 = arith.index_cast %28 {ssbuffer.block_id = 7 : i32} : i32 to index
        %30 = arith.index_cast %20 {ssbuffer.block_id = 7 : i32} : i32 to index
        %31 = arith.addi %29, %30 {ssbuffer.block_id = 7 : i32} : index
        %reinterpret_cast_10 = memref.reinterpret_cast %arg6 to offset: [%31], sizes: [128], strides: [1] {ssbuffer.block_id = 7 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        bufferization.materialize_in_destination %26 in writable %reinterpret_cast_10 {ssbuffer.block_id = 7 : i32} : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
        %32 = arith.truncf %27 {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
        bufferization.materialize_in_destination %32 in writable %reinterpret_cast {ssbuffer.block_id = 7 : i32} : (tensor<128x128xf16>, memref<128x128xf16, strided<[128, 1], offset: ?>>) -> ()
      } {ssbuffer.block_id = 10 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
        %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 0 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 9 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
      scope.return
   }{hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
