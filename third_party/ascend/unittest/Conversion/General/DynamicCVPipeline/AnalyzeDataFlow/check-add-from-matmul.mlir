// RUN: triton-opt --analyze-data-flow --verify-diagnostics %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_add_from_matmul(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg9: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c32 = arith.constant {ssbuffer.block_id = 8 : i32} 32 : index
    %cst = arith.constant {ssbuffer.block_id = 14 : i32} 0.000000e+00 : f32
    %c1_i32 = arith.constant {Undefined, ssbuffer.block_id = 14 : i32} 1 : i32
    %c0_i32 = arith.constant {Undefined, ssbuffer.block_id = 14 : i32} 0 : i32
    %c63_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 63 : i32
    %c32_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 32 : i32
    %c8_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 8 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 14 : i32} 1.000000e+00 : f32
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32} 64 : index
    %cst_1 = arith.constant {ssbuffer.block_id = 14 : i32} 0.000000e+00 : bf16
    %c128 = arith.constant {ssbuffer.block_id = 14 : i32} 128 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<128x64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %2 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst_0 : f32) outs(%0 : tensor<128x64xf32>) -> tensor<128x64xf32>
      %alloc = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %23 = scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg20 = %1) -> (tensor<128x64xf32>)  : i32 {
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %memspacecast_20 = memref.memory_space_cast %alloc {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
        %60 = bufferization.to_tensor %memspacecast_20 restrict writable {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32> to tensor<128x64xf32>
        %61 = arith.addf %60, %arg20 {ssbuffer.block_id = 11 : i32, ssbuffer.add_from_matmul} : tensor<128x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        scf.yield %61 : tensor<128x64xf32>
      } {DataUse, ssbuffer.block_id = 15 : i32, ssbuffer.main_loop = 0 : i32}
      %alloc_1 = memref.alloc() {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
      %24 = scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg20 = %1) -> (tensor<128x64xf32>)  : i32 {
        %500 = math.exp %2 {ssbuffer.block_id = 12 : i32} : tensor<128x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 2
        %memspacecast_200 = memref.memory_space_cast %alloc_1 {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
        %600 = bufferization.to_tensor %memspacecast_200 restrict writable {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x64xf32> to tensor<128x64xf32>
        %610 = arith.addf %600, %arg20 {ssbuffer.block_id = 13 : i32, ssbuffer.add_from_matmul} : tensor<128x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 13 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
        scf.yield %610 : tensor<128x64xf32>
      } {DataUse, ssbuffer.block_id = 16 : i32, ssbuffer.main_loop = 1 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %20 = arith.addi %arg11, %c63_i32 {ssbuffer.block_id = 8 : i32} : i32
      %21 = arith.divsi %20, %c64_i32 {ssbuffer.block_id = 8 : i32} : i32
      %22 = arith.muli %21, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
      %23 = arith.divsi %arg16, %22 {ssbuffer.block_id = 8 : i32} : i32
      %24 = arith.muli %23, %c8_i32 {ssbuffer.block_id = 8 : i32} : i32
      %35 = arith.index_cast %arg11 {ssbuffer.block_id = 7 : i32} : i32 to index
      %36 = arith.maxsi %35, %35 {ssbuffer.block_id = 7 : i32} : index
      %38 = arith.subi %36, %35 {ssbuffer.block_id = 7 : i32} : index
      %alloc = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %41 = arith.muli %arg19, %c32_i32 {ssbuffer.block_id = 6 : i32} : i32
        %42 = arith.index_cast %24 {ssbuffer.block_id = 6 : i32} : i32 to index
        %43 = arith.index_cast %arg12 {ssbuffer.block_id = 6 : i32} : i32 to index
        %44 = arith.index_cast %41 {ssbuffer.block_id = 6 : i32} : i32 to index
        %45 = arith.addi %44, %c128 {ssbuffer.block_id = 6 : i32} : index
        %46 = arith.index_cast %arg10 {ssbuffer.block_id = 6 : i32} : i32 to index
        %47 = arith.maxsi %44, %46 {ssbuffer.block_id = 6 : i32} : index
        %48 = arith.minsi %45, %47 {ssbuffer.block_id = 6 : i32} : index
        %49 = arith.subi %48, %44 {ssbuffer.block_id = 6 : i32} : index
        %50 = arith.addi %44, %c32 {ssbuffer.block_id = 6 : i32} : index
        %51 = arith.maxsi %44, %43 {ssbuffer.block_id = 6 : i32} : index
        %52 = arith.minsi %50, %51 {ssbuffer.block_id = 6 : i32} : index
        %53 = arith.subi %52, %44 {ssbuffer.block_id = 6 : i32} : index
        %54 = arith.minsi %49, %c128 {ssbuffer.block_id = 6 : i32} : index
        %55 = arith.minsi %53, %c32 {ssbuffer.block_id = 6 : i32} : index
        %59 = arith.minsi %38, %c64 {ssbuffer.block_id = 6 : i32} : index
        %60 = arith.cmpi slt, %59, %c64 {ssbuffer.block_id = 6 : i32} : index
        %61 = arith.ori %60, %60 {ssbuffer.block_id = 6 : i32} : i1
        %alloc_3 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<128x32xbf16>
        %alloc_4 = memref.alloc() {ssbuffer.block_id = 4 : i32} : memref<32x64xbf16>
        scf.if %61 {
          linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst_1 : bf16) outs(%alloc_4 : memref<32x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        scf.if %61 {
          linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst_1 : bf16) outs(%alloc_3 : memref<128x32xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        %62 = arith.muli %42, %43 {ssbuffer.block_id = 4 : i32} : index
        %63 = arith.addi %62, %44 {ssbuffer.block_id = 4 : i32} : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%63], sizes: [128, 32], strides: [%43, 1] {ssbuffer.block_id = 4 : i32} : memref<?xbf16> to memref<128x32xbf16, strided<[?, 1], offset: ?>>
        %subview = memref.subview %reinterpret_cast[0, 0] [%54, %55] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<128x32xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %subview_5 = memref.subview %alloc_3[0, 0] [%54, %55] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<128x32xbf16> to memref<?x?xbf16, strided<[32, 1]>>
        memref.copy %subview, %subview_5 {ssbuffer.block_id = 4 : i32} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1]>>
        %64 = bufferization.to_tensor %alloc_3 restrict writable {ssbuffer.block_id = 4 : i32} : memref<128x32xbf16> to tensor<128x32xbf16>
        %65 = arith.muli %44, %35 {ssbuffer.block_id = 4 : i32} : index
        %66 = arith.addi %65, %35 {ssbuffer.block_id = 4 : i32} : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg3 to offset: [%66], sizes: [32, 64], strides: [%35, 1] {ssbuffer.block_id = 4 : i32} : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
        %subview_7 = memref.subview %reinterpret_cast_6[0, 0] [%55, %59] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %subview_8 = memref.subview %alloc_4[0, 0] [%55, %59] [1, 1] {ssbuffer.block_id = 4 : i32} : memref<32x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
        memref.copy %subview_7, %subview_8 {ssbuffer.block_id = 4 : i32} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
        %67 = bufferization.to_tensor %alloc_4 restrict writable {ssbuffer.block_id = 4 : i32} : memref<32x64xbf16> to tensor<32x64xbf16>
        %68 = tensor.empty() {ssbuffer.block_id = 4 : i32} : tensor<128x64xf32>
        %69 = linalg.fill {ssbuffer.block_id = 4 : i32} ins(%cst : f32) outs(%68 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %70 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32} ins(%64, %67 : tensor<128x32xbf16>, tensor<32x64xbf16>) outs(%69 : tensor<128x64xf32>) -> tensor<128x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32} ins(%70 : tensor<128x64xf32>) outs(%alloc : memref<128x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 4 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
      } {DataUse, ssbuffer.block_id = 15 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
      %alloc_1 = memref.alloc() {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
      scf.for %arg19 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
        %410 = arith.muli %arg19, %c32_i32 {ssbuffer.block_id = 7 : i32} : i32
        %420 = arith.index_cast %24 {ssbuffer.block_id = 7 : i32} : i32 to index
        %430 = arith.index_cast %arg12 {ssbuffer.block_id = 7 : i32} : i32 to index
        %440 = arith.index_cast %410 {ssbuffer.block_id = 7 : i32} : i32 to index
        %450 = arith.addi %440, %c128 {ssbuffer.block_id = 7 : i32} : index
        %460 = arith.index_cast %arg10 {ssbuffer.block_id = 7 : i32} : i32 to index
        %470 = arith.maxsi %440, %460 {ssbuffer.block_id = 7 : i32} : index
        %480 = arith.minsi %450, %470 {ssbuffer.block_id = 7 : i32} : index
        %490 = arith.subi %480, %440 {ssbuffer.block_id = 7 : i32} : index
        %500 = arith.addi %440, %c32 {ssbuffer.block_id = 7 : i32} : index
        %510 = arith.maxsi %440, %430 {ssbuffer.block_id = 7 : i32} : index
        %520 = arith.minsi %500, %510 {ssbuffer.block_id = 7 : i32} : index
        %530 = arith.subi %520, %440 {ssbuffer.block_id = 7 : i32} : index
        %540 = arith.minsi %490, %c128 {ssbuffer.block_id = 7 : i32} : index
        %550 = arith.minsi %530, %c32 {ssbuffer.block_id = 7 : i32} : index
        %590 = arith.minsi %38, %c64 {ssbuffer.block_id = 7 : i32} : index
        %600 = arith.cmpi slt, %590, %c64 {ssbuffer.block_id = 7 : i32} : index
        %610 = arith.ori %600, %600 {ssbuffer.block_id = 7 : i32} : i1
        %alloc_30 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<128x32xbf16>
        %alloc_40 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<32x64xbf16>
        scf.if %610 {
          linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_1 : bf16) outs(%alloc_40 : memref<32x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
        scf.if %610 {
          linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_1 : bf16) outs(%alloc_30 : memref<128x32xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
        %620 = arith.muli %420, %430 {ssbuffer.block_id = 5 : i32} : index
        %630 = arith.addi %620, %440 {ssbuffer.block_id = 5 : i32} : index
        %reinterpret_cast0 = memref.reinterpret_cast %arg2 to offset: [%630], sizes: [128, 32], strides: [%430, 1] {ssbuffer.block_id = 5 : i32} : memref<?xbf16> to memref<128x32xbf16, strided<[?, 1], offset: ?>>
        %subview0 = memref.subview %reinterpret_cast0[0, 0] [%540, %550] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<128x32xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %subview_50 = memref.subview %alloc_30[0, 0] [%540, %550] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<128x32xbf16> to memref<?x?xbf16, strided<[32, 1]>>
        memref.copy %subview0, %subview_50 {ssbuffer.block_id = 5 : i32} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[32, 1]>>
        %640 = bufferization.to_tensor %alloc_30 restrict writable {ssbuffer.block_id = 5 : i32} : memref<128x32xbf16> to tensor<128x32xbf16>
        %650 = arith.muli %440, %35 {ssbuffer.block_id = 5 : i32} : index
        %660 = arith.addi %650, %35 {ssbuffer.block_id = 5 : i32} : index
        %reinterpret_cast_60 = memref.reinterpret_cast %arg3 to offset: [%660], sizes: [32, 64], strides: [%35, 1] {ssbuffer.block_id = 5 : i32} : memref<?xbf16> to memref<32x64xbf16, strided<[?, 1], offset: ?>>
        %subview_70 = memref.subview %reinterpret_cast_60[0, 0] [%550, %590] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<32x64xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        %subview_80 = memref.subview %alloc_40[0, 0] [%550, %590] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<32x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
        memref.copy %subview_70, %subview_80 {ssbuffer.block_id = 5 : i32} : memref<?x?xbf16, strided<[?, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
        %670 = bufferization.to_tensor %alloc_40 restrict writable {ssbuffer.block_id = 5 : i32} : memref<32x64xbf16> to tensor<32x64xbf16>
        %680 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128x64xf32>
        %690 = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%680 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %700 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32} ins(%640, %670 : tensor<128x32xbf16>, tensor<32x64xbf16>) outs(%690 : tensor<128x64xf32>) -> tensor<128x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} ins(%700 : tensor<128x64xf32>) outs(%alloc_1 : memref<128x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
      } {DataUse, ssbuffer.block_id = 16 : i32, ssbuffer.main_loop = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 16 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
