// RUN: triton-opt --add_multi_buffer_outer_scope %s | FileCheck %s

// When the input IR already uses flags 1..9 and requests inter-core
// double-buffering (ssbuffer.inter_core_buf_count = 2), the 6 main-loop
// transfer keys need to acquire output flags 10..15, pushing the maximum
// flag id past MAX_FLAG_ID (14). The pass must detect the budget overflow
// and fall back to single-buffer mode, leaving the IR unchanged:
// no new output flags, no polling control flow, no output buffers.

// The double-buffer request attribute is preserved (downgrade, not removal).
// CHECK: ssbuffer.inter_core_buf_count = 2
// CHECK-LABEL: func.func @_attn_bwd
// Original flags 1..9 are preserved.
// CHECK: flag = 9
// No new output flag (10..15) is allocated -> confirms single-buffer fallback.
// CHECK-NOT: flag = {{1[0-5]}}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">, ssbuffer.insertionOptimization, ssbuffer.inter_core_buf_count = 2 : i32, ssbuffer.intra_buf_count = 3 : i32, ssbuffer.load_store_buf_count = 1 : i32} {
  func.func @_attn_bwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 2 : i32}, %arg7: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg8: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg9: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg10: memref<?xf32> {tt.tensor_kind = 0 : i32}, %arg11: f32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 19 : i32} dense<[4, 8, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 19 : i32} dense<[128, 4, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 14 : i32} 0.000000e+00 : f32
    %c64_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 64 : i32
    %c8388608_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 8388608 : i32
    %c1048576_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 1048576 : i32
    %c8_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 8 : i32
    %c8192_i32 = arith.constant {ssbuffer.block_id = 13 : i32} 8192 : i32
    %c128_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 128 : i32
    %c131072_i32 = arith.constant {Undefined, ssbuffer.block_id = 13 : i32} 131072 : i32
    %c28_i32 = arith.constant {Undefined, ssbuffer.block_id = 13 : i32} 28 : i32
    %c0_i32 = arith.constant {Undefined, ssbuffer.block_id = 13 : i32} 0 : i32
    %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 13 : i32} 1 : i32
    %c128 = arith.constant {ssbuffer.block_id = 13 : i32} 128 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<128x64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%arg11 : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      scf.for %arg18 = %arg15 to %c131072_i32 step %c28_i32  : i32 {
        %2 = arith.divsi %arg18, %c128_i32 {MixUse, ssbuffer.block_id = 15 : i32} : i32
        %3 = arith.muli %2, %c8192_i32 {ssbuffer.block_id = 15 : i32} : i32
        %4 = arith.remsi %2, %c8_i32 {MixUse, ssbuffer.block_id = 15 : i32} : i32
        %5 = arith.muli %4, %c1048576_i32 {MixUse, ssbuffer.block_id = 15 : i32} : i32
        %6 = arith.divsi %2, %c8_i32 {MixUse, ssbuffer.block_id = 15 : i32} : i32
        %7 = arith.muli %6, %c8388608_i32 {MixUse, ssbuffer.block_id = 15 : i32} : i32
        %8 = arith.addi %5, %7 {MixUse, ssbuffer.block_id = 15 : i32} : i32
        %9 = arith.index_cast %8 {ssbuffer.block_id = 15 : i32} : i32 to index
        %alloc_3 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_4 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_6 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
        %alloc_8 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
        scf.for %arg19 = %c0_i32 to %c64_i32 step %c1_i32  : i32 {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %memspacecast_11 = memref.memory_space_cast %alloc_6 {ssbuffer.block_id = 19 : i32, ssbuffer.crossCoreDeps = [3 : i32, 0 : i32], ssbuffer.transfer_id = 3 : i32} : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
          %27 = bufferization.to_tensor %memspacecast_11 restrict writable {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x64xf32> to tensor<128x64xf32>
          %28 = arith.muli %arg19, %c128_i32 {MixUse, ssbuffer.block_id = 19 : i32} : i32
          %29 = arith.index_cast %28 {ssbuffer.block_id = 19 : i32} : i32 to index
          %30 = arith.index_cast %3 {ssbuffer.block_id = 19 : i32} : i32 to index
          %31 = arith.addi %30, %29 {ssbuffer.block_id = 19 : i32} : index
          %reinterpret_cast_12 = memref.reinterpret_cast %arg9 to offset: [%31], sizes: [128], strides: [1] {ssbuffer.block_id = 19 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
          %alloc_13 = memref.alloc() {ssbuffer.block_id = 19 : i32} : memref<128xf32>
          memref.copy %reinterpret_cast_12, %alloc_13 {ssbuffer.block_id = 19 : i32} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
          %32 = bufferization.to_tensor %alloc_13 restrict writable {ssbuffer.block_id = 19 : i32} : memref<128xf32> to tensor<128xf32>
          %reinterpret_cast_14 = memref.reinterpret_cast %arg10 to offset: [%31], sizes: [128], strides: [1] {ssbuffer.block_id = 19 : i32} : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
          %alloc_15 = memref.alloc() {ssbuffer.block_id = 19 : i32} : memref<128xf32>
          memref.copy %reinterpret_cast_14, %alloc_15 {ssbuffer.block_id = 19 : i32} : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32>
          %33 = bufferization.to_tensor %alloc_15 restrict writable {ssbuffer.block_id = 19 : i32} : memref<128xf32> to tensor<128xf32>
          %broadcasted = linalg.broadcast ins(%32 : tensor<128xf32>) outs(%0 : tensor<128x64xf32>) dimensions = [1]  {ssbuffer.block_id = 19 : i32}
          %34 = arith.mulf %27, %1 {DataUse, ssbuffer.block_id = 19 : i32} : tensor<128x64xf32>
          %35 = arith.subf %34, %broadcasted {DataUse, ssbuffer.block_id = 19 : i32} : tensor<128x64xf32>
          %36 = math.exp %35 {DataUse, ssbuffer.block_id = 19 : i32} : tensor<128x64xf32>
          %37 = arith.truncf %36 {DataUse, ssbuffer.block_id = 19 : i32} : tensor<128x64xf32> to tensor<128x64xf16>
          %reshape = tensor.reshape %37(%cst_0) {ssbuffer.block_id = 19 : i32} : (tensor<128x64xf16>, tensor<3xi64>) -> tensor<128x4x16xf16>
          %38 = tensor.empty() {ssbuffer.block_id = 19 : i32} : tensor<4x128x16xf16>
          %transposed = linalg.transpose ins(%reshape : tensor<128x4x16xf16>) outs(%38 : tensor<4x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 19 : i32}
          %reshape_16 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 19 : i32} : (tensor<4x128x16xf16>, tensor<4xi64>) -> tensor<4x8x16x16xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
          hivm.hir.copy ins(%reshape_16 : tensor<4x8x16x16xf16>) outs(%alloc_3 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 19 : i32, ssbuffer.crossCoreDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          hivm.hir.sync_block_set {ssbuffer.block_id = 19 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 6
          hivm.hir.sync_block_wait {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 7
          %memspacecast_17 = memref.memory_space_cast %alloc_7 {ssbuffer.block_id = 21 : i32, ssbuffer.crossCoreDeps = [4 : i32, 0 : i32], ssbuffer.transfer_id = 4 : i32} : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
          %39 = bufferization.to_tensor %memspacecast_17 restrict writable {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x64xf32> to tensor<128x64xf32>
          %40 = arith.extf %37 {DataUse, ssbuffer.block_id = 21 : i32} : tensor<128x64xf16> to tensor<128x64xf32>
          %broadcasted_18 = linalg.broadcast ins(%33 : tensor<128xf32>) outs(%0 : tensor<128x64xf32>) dimensions = [1]  {ssbuffer.block_id = 21 : i32}
          %41 = arith.subf %39, %broadcasted_18 {DataUse, ssbuffer.block_id = 21 : i32} : tensor<128x64xf32>
          %42 = arith.mulf %40, %41 {DataUse, ssbuffer.block_id = 21 : i32} : tensor<128x64xf32>
          %43 = arith.mulf %42, %1 {DataUse, ssbuffer.block_id = 21 : i32} : tensor<128x64xf32>
          %44 = arith.truncf %43 {DataUse, ssbuffer.block_id = 21 : i32} : tensor<128x64xf32> to tensor<128x64xf16>
          %reshape_19 = tensor.reshape %44(%cst_0) {ssbuffer.block_id = 21 : i32} : (tensor<128x64xf16>, tensor<3xi64>) -> tensor<128x4x16xf16>
          %45 = tensor.empty() {ssbuffer.block_id = 21 : i32} : tensor<4x128x16xf16>
          %transposed_20 = linalg.transpose ins(%reshape_19 : tensor<128x4x16xf16>) outs(%45 : tensor<4x128x16xf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 21 : i32}
          %reshape_21 = tensor.reshape %transposed_20(%cst) {ssbuffer.block_id = 21 : i32} : (tensor<4x128x16xf16>, tensor<4xi64>) -> tensor<4x8x16x16xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
          hivm.hir.copy ins(%reshape_21 : tensor<4x8x16x16xf16>) outs(%alloc_4 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 21 : i32, ssbuffer.crossCoreDeps = [1 : i32, 1 : i32], ssbuffer.transfer_id = 1 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
          hivm.hir.sync_block_wait {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 5
          hivm.hir.copy ins(%reshape_21 : tensor<4x8x16x16xf16>) outs(%alloc_5 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 21 : i32, ssbuffer.crossCoreDeps = [2 : i32, 1 : i32], ssbuffer.transfer_id = 2 : i32}
          hivm.hir.sync_block_set {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
          hivm.hir.sync_block_set {ssbuffer.block_id = 21 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 7
          hivm.hir.sync_block_wait {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 9
          %memspacecast_22 = memref.memory_space_cast %alloc_8 {ssbuffer.block_id = 22 : i32, ssbuffer.crossCoreDeps = [5 : i32, 0 : i32], ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>> to memref<128x128xf32>
          %46 = bufferization.to_tensor %memspacecast_22 restrict writable {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32> to tensor<128x128xf32>
          %47 = arith.muli %arg19, %c128_i32 {MixUse, ssbuffer.block_id = 22 : i32} : i32
          %48 = arith.index_cast %47 {ssbuffer.block_id = 22 : i32} : i32 to index
          %49 = arith.muli %48, %c128 {ssbuffer.block_id = 22 : i32} : index
          %50 = arith.addi %9, %49 {ssbuffer.block_id = 22 : i32} : index
          %reinterpret_cast_23 = memref.reinterpret_cast %arg6 to offset: [%50], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 22 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %51 = arith.truncf %46 {DataUse, ssbuffer.block_id = 22 : i32} : tensor<128x128xf32> to tensor<128x128xf16>
          hivm.hir.store ins(%51 : tensor<128x128xf16>) outs(%reinterpret_cast_23 : memref<128x128xf16, strided<[128, 1], offset: ?>>) {ssbuffer.block_id = 22 : i32} atomic = <add>
          hivm.hir.sync_block_set {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
        } {DataUse, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 0 : i32}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 5
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        hivm.hir.sync_block_wait {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
        %memspacecast = memref.memory_space_cast %alloc_2 {ssbuffer.block_id = 17 : i32, ssbuffer.crossCoreDeps = [7 : i32, 0 : i32], ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %10 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32> to tensor<64x128xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %memspacecast_9 = memref.memory_space_cast %alloc {ssbuffer.block_id = 17 : i32, ssbuffer.crossCoreDeps = [6 : i32, 0 : i32], ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>> to memref<64x128xf32>
        %11 = bufferization.to_tensor %memspacecast_9 restrict writable {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32> to tensor<64x128xf32>
        %12 = arith.divsi %arg18, %c128_i32 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %13 = arith.remsi %12, %c8_i32 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %14 = arith.muli %13, %c1048576_i32 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %15 = arith.divsi %12, %c8_i32 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %16 = arith.muli %15, %c8388608_i32 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %17 = arith.addi %14, %16 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %18 = arith.index_cast %17 {ssbuffer.block_id = 17 : i32} : i32 to index
        %19 = arith.muli %12, %c128_i32 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %20 = arith.subi %arg18, %19 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %21 = arith.muli %20, %c64_i32 {MixUse, ssbuffer.block_id = 17 : i32} : i32
        %22 = arith.index_cast %21 {ssbuffer.block_id = 17 : i32} : i32 to index
        %23 = arith.muli %22, %c128 {ssbuffer.block_id = 17 : i32} : index
        %24 = arith.addi %18, %23 {ssbuffer.block_id = 17 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [%24], sizes: [64, 128], strides: [128, 1] {ssbuffer.block_id = 17 : i32} : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        %reinterpret_cast_10 = memref.reinterpret_cast %arg8 to offset: [%24], sizes: [64, 128], strides: [128, 1] {ssbuffer.block_id = 17 : i32} : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        %25 = arith.truncf %11 {DataUse, ssbuffer.block_id = 17 : i32} : tensor<64x128xf32> to tensor<64x128xf16>
        %26 = arith.truncf %10 {DataUse, ssbuffer.block_id = 17 : i32} : tensor<64x128xf32> to tensor<64x128xf16>
        bufferization.materialize_in_destination %25 in writable %reinterpret_cast {ssbuffer.block_id = 17 : i32} : (tensor<64x128xf16>, memref<64x128xf16, strided<[128, 1], offset: ?>>) -> ()
        bufferization.materialize_in_destination %26 in writable %reinterpret_cast_10 {ssbuffer.block_id = 17 : i32} : (tensor<64x128xf16>, memref<64x128xf16, strided<[128, 1], offset: ?>>) -> ()
        hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.sync_block_set {ssbuffer.block_id = 17 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      } {Undefined, ssbuffer.block_id = 24 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<64x128xf32>
      %1 = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%cst_1 : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
      %2 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x128xf32>
      %3 = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%cst_1 : f32) outs(%2 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %4 = tensor.empty() {ssbuffer.block_id = 1 : i32} : tensor<128x64xf32>
      %5 = linalg.fill {ssbuffer.block_id = 1 : i32} ins(%cst_1 : f32) outs(%4 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
      scf.for %arg18 = %arg15 to %c131072_i32 step %c28_i32  : i32 {
        %6 = arith.divsi %arg18, %c128_i32 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %7 = arith.muli %6, %c128_i32 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %8 = arith.subi %arg18, %7 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %9 = arith.remsi %6, %c8_i32 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %10 = arith.muli %9, %c1048576_i32 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %11 = arith.divsi %6, %c8_i32 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %12 = arith.muli %11, %c8388608_i32 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %13 = arith.addi %10, %12 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %14 = arith.muli %8, %c64_i32 {MixUse, ssbuffer.block_id = 2 : i32} : i32
        %15 = arith.index_cast %13 {ssbuffer.block_id = 2 : i32} : i32 to index
        %16 = arith.index_cast %14 {ssbuffer.block_id = 2 : i32} : i32 to index
        %17 = arith.muli %16, %c128 {ssbuffer.block_id = 2 : i32} : index
        %18 = arith.addi %15, %17 {ssbuffer.block_id = 2 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%18], sizes: [64, 128], strides: [128, 1] {ssbuffer.block_id = 2 : i32} : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        %alloc_3 = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<64x128xf16>
        memref.copy %reinterpret_cast, %alloc_3 {ssbuffer.block_id = 2 : i32} : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
        %19 = bufferization.to_tensor %alloc_3 restrict writable {ssbuffer.block_id = 2 : i32} : memref<64x128xf16> to tensor<64x128xf16>
        %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%18], sizes: [64, 128], strides: [128, 1] {ssbuffer.block_id = 2 : i32} : memref<?xf16> to memref<64x128xf16, strided<[128, 1], offset: ?>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<64x128xf16>
        memref.copy %reinterpret_cast_4, %alloc_5 {ssbuffer.block_id = 2 : i32} : memref<64x128xf16, strided<[128, 1], offset: ?>> to memref<64x128xf16>
        %20 = bufferization.to_tensor %alloc_5 restrict writable {ssbuffer.block_id = 2 : i32} : memref<64x128xf16> to tensor<64x128xf16>
        %21 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<128x64xf16>
        %transposed = linalg.transpose ins(%19 : tensor<64x128xf16>) outs(%21 : tensor<128x64xf16>) permutation = [1, 0]  {ssbuffer.block_id = 2 : i32}
        %transposed_6 = linalg.transpose ins(%20 : tensor<64x128xf16>) outs(%21 : tensor<128x64xf16>) permutation = [1, 0]  {ssbuffer.block_id = 2 : i32}
        %alloc_7 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 3
        %alloc_8 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        %alloc_9 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 5
        %alloc_10 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        %alloc_11 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        %alloc_12 = memref.alloc() {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %22:2 = scf.for %arg19 = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%arg20 = %1, %arg21 = %1) -> (tensor<64x128xf32>, tensor<64x128xf32>)  : i32 {
          %23 = arith.muli %arg19, %c128_i32 {MixUse, ssbuffer.block_id = 4 : i32} : i32
          %24 = arith.index_cast %23 {ssbuffer.block_id = 4 : i32} : i32 to index
          %25 = arith.muli %24, %c128 {ssbuffer.block_id = 4 : i32} : index
          %26 = arith.addi %15, %25 {ssbuffer.block_id = 4 : i32} : index
          %reinterpret_cast_13 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 4 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_14 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_13, %alloc_14 {ssbuffer.block_id = 4 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %27 = bufferization.to_tensor %alloc_14 restrict writable {ssbuffer.block_id = 4 : i32} : memref<128x128xf16> to tensor<128x128xf16>
          %28 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.loop_carried_l0c} ins(%27, %transposed : tensor<128x128xf16>, tensor<128x64xf16>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.crossCoreDeps = [3 : i32, 1 : i32], ssbuffer.transfer_id = 3 : i32} ins(%28 : tensor<128x64xf32>) outs(%alloc_10 : memref<128x64xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
          %reinterpret_cast_15 = memref.reinterpret_cast %arg5 to offset: [%26], sizes: [128, 128], strides: [128, 1] {ssbuffer.block_id = 6 : i32} : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
          %alloc_16 = memref.alloc() {ssbuffer.block_id = 6 : i32} : memref<128x128xf16>
          memref.copy %reinterpret_cast_15, %alloc_16 {ssbuffer.block_id = 6 : i32} : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
          %29 = bufferization.to_tensor %alloc_16 restrict writable {ssbuffer.block_id = 6 : i32} : memref<128x128xf16> to tensor<128x128xf16>
          %30 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<64x128xf16>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
          %31 = hivm.hir.convert_layout %alloc_7 output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 6 : i32, ssbuffer.crossCoreDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : (memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
          %memspacecast = memref.memory_space_cast %31 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
          %32 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf16> to tensor<128x64xf16>
          %transposed_17 = linalg.transpose ins(%32 : tensor<128x64xf16>) outs(%30 : tensor<64x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 6 : i32}
          %33 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.loop_carried_l0c} ins(%transposed_17, %29 : tensor<64x128xf16>, tensor<128x128xf16>) outs(%arg21 : tensor<64x128xf32>) -> tensor<64x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 3
          %34 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 8 : i32, ssbuffer.loop_carried_l0c} ins(%29, %transposed_6 : tensor<128x128xf16>, tensor<128x64xf16>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
          hivm.hir.sync_block_wait {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 8 : i32, ssbuffer.crossCoreDeps = [4 : i32, 1 : i32], ssbuffer.transfer_id = 4 : i32} ins(%34 : tensor<128x64xf32>) outs(%alloc_11 : memref<128x64xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 8 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 7
          hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
          %35 = hivm.hir.convert_layout %alloc_8 output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 10 : i32, ssbuffer.crossCoreDeps = [1 : i32, 0 : i32], ssbuffer.transfer_id = 1 : i32} : (memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
          %memspacecast_18 = memref.memory_space_cast %35 {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
          %36 = bufferization.to_tensor %memspacecast_18 restrict writable {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf16> to tensor<128x64xf16>
          %transposed_19 = linalg.transpose ins(%36 : tensor<128x64xf16>) outs(%30 : tensor<64x128xf16>) permutation = [1, 0]  {ssbuffer.block_id = 10 : i32}
          %37 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 10 : i32, ssbuffer.loop_carried_l0c} ins(%transposed_19, %27 : tensor<64x128xf16>, tensor<128x128xf16>) outs(%arg20 : tensor<64x128xf32>) -> tensor<64x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
          hivm.hir.sync_block_wait {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
          %38 = hivm.hir.convert_layout %alloc_9 output_shape [128, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 12 : i32, ssbuffer.crossCoreDeps = [2 : i32, 0 : i32], ssbuffer.transfer_id = 2 : i32} : (memref<4x8x16x16xf16, #hivm.address_space<cbuf>>) -> memref<128x64xf16, #hivm.address_space<cbuf>>
          %memspacecast_20 = memref.memory_space_cast %38 {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x64xf16, #hivm.address_space<cbuf>> to memref<128x64xf16>
          %39 = bufferization.to_tensor %memspacecast_20 restrict writable {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x64xf16> to tensor<128x64xf16>
          %40 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.loop_carried_l0c} ins(%39, %19 : tensor<128x64xf16>, tensor<64x128xf16>) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
          hivm.hir.sync_block_set {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 5
          hivm.hir.sync_block_wait {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 12 : i32, ssbuffer.crossCoreDeps = [5 : i32, 1 : i32], ssbuffer.transfer_id = 5 : i32} ins(%40 : tensor<128x128xf32>) outs(%alloc_12 : memref<128x128xf32, #hivm.address_space<ub>>)
          hivm.hir.sync_block_set {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 9
          scf.yield {Undefined} %37, %33 : tensor<64x128xf32>, tensor<64x128xf32>
        } {DataUse, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 0 : i32}
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 7
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 23 : i32, ssbuffer.crossCoreDeps = [6 : i32, 1 : i32], ssbuffer.transfer_id = 6 : i32} ins(%22#0 : tensor<64x128xf32>) outs(%alloc : memref<64x128xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 23 : i32, ssbuffer.crossCoreDeps = [7 : i32, 1 : i32], ssbuffer.transfer_id = 7 : i32} ins(%22#1 : tensor<64x128xf32>) outs(%alloc_2 : memref<64x128xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      } {Undefined, ssbuffer.block_id = 24 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      hivm.hir.sync_block_wait {ssbuffer.block_id = 24 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
