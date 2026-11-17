// RUN: triton-opt --analyze-data-flow --verify-diagnostics %s

#map = affine_map<(d0) -> (d0)>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @prepare_wy_repr_bwd_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %arg10: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg11: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg12: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant {ssbuffer.block_id = 22 : i32} dense<[4, 4, 16, 16]> : tensor<4xi64>
    %cst_0 = arith.constant {ssbuffer.block_id = 22 : i32} dense<[64, 4, 16]> : tensor<3xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 31 : i32} 0.000000e+00 : f32
    %c128_i32 = arith.constant {ssbuffer.block_id = 31 : i32} 128 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 31 : i32} 0 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 31 : i32} 64 : i32
    %c8_i32 = arith.constant {ssbuffer.block_id = 31 : i32} 8 : i32
    %c2_i32 = arith.constant {ssbuffer.block_id = 31 : i32} 2 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 31 : i32} 1 : i32
    %c8 = arith.constant {ssbuffer.block_id = 31 : i32} 8 : index
    %c0 = arith.constant {ssbuffer.block_id = 31 : i32} 0 : index
    %c64 = arith.constant {ssbuffer.block_id = 31 : i32} 64 : index
    %cst_2 = arith.constant {ssbuffer.block_id = 31 : i32} 0.000000e+00 : bf16
    %c512 = arith.constant {ssbuffer.block_id = 31 : i32} 512 : index
    %c1024 = arith.constant {ssbuffer.block_id = 31 : i32} 1024 : index
    %c128 = arith.constant {ssbuffer.block_id = 31 : i32} 128 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 31 : i32} : tensor<64x64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 31 : i32} ins(%cst_1 : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %2 = tensor.empty() {ssbuffer.block_id = 31 : i32} : tensor<64xf32>
      %3 = linalg.fill {ssbuffer.block_id = 31 : i32} ins(%cst_1 : f32) outs(%2 : tensor<64xf32>) -> tensor<64xf32>
      %4 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 31 : i32} : i32
      %5 = arith.remsi %arg18, %c8_i32 {ssbuffer.block_id = 31 : i32} : i32
      %6 = arith.muli %4, %arg13 {ssbuffer.block_id = 31 : i32} : i32
      %7 = arith.muli %6, %c8_i32 {ssbuffer.block_id = 31 : i32} : i32
      %8 = arith.addi %7, %5 {ssbuffer.block_id = 31 : i32} : i32
      %9 = arith.muli %arg17, %c64_i32 {ssbuffer.block_id = 31 : i32} : i32
      %10 = arith.maxsi %9, %c0_i32 {ssbuffer.block_id = 31 : i32} : i32
      %11 = arith.index_cast %10 {ssbuffer.block_id = 31 : i32} : i32 to index
      %12 = arith.muli %11, %c8 {ssbuffer.block_id = 31 : i32} : index
      %13 = arith.index_cast %arg13 {ssbuffer.block_id = 31 : i32} : i32 to index
      %14 = arith.divsi %12, %c8 {ssbuffer.block_id = 31 : i32} : index
      %15 = arith.subi %13, %14 {ssbuffer.block_id = 31 : i32} : index
      %16 = arith.maxsi %15, %c0 {ssbuffer.block_id = 31 : i32} : index
      %17 = arith.minsi %16, %c64 {ssbuffer.block_id = 31 : i32} : index
      %18 = arith.subi %c0_i32, %9 {ssbuffer.block_id = 31 : i32} : i32
      %19 = arith.maxsi %18, %c0_i32 {ssbuffer.block_id = 31 : i32} : i32
      %20 = arith.index_cast %19 {ssbuffer.block_id = 31 : i32} : i32 to index
      %21 = arith.minsi %20, %17 {ssbuffer.block_id = 31 : i32} : index
      %22 = arith.subi %17, %21 {ssbuffer.block_id = 31 : i32} : index
      %23 = arith.cmpi slt, %22, %c64 {ssbuffer.block_id = 31 : i32} : index
      %24 = arith.muli %8, %c128_i32 {ssbuffer.block_id = 31 : i32} : i32
      %25 = arith.index_cast %24 {ssbuffer.block_id = 31 : i32} : i32 to index
      %alloc = memref.alloc() {ssbuffer.block_id = 32 : i32} : memref<64xbf16>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 32 : i32} : memref<64xbf16>
      scf.if %23 {
        linalg.fill {ssbuffer.block_id = 32 : i32} ins(%cst_2 : bf16) outs(%alloc : memref<64xbf16>)
        linalg.fill {ssbuffer.block_id = 32 : i32} ins(%cst_2 : bf16) outs(%alloc_3 : memref<64xbf16>)
      }
      %26 = arith.index_cast %8 {ssbuffer.block_id = 32 : i32} : i32 to index
      %27 = arith.addi %12, %26 {ssbuffer.block_id = 32 : i32} : index
      %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%27], sizes: [64], strides: [8] {ssbuffer.block_id = 32 : i32} : memref<?xbf16> to memref<64xbf16, strided<[8], offset: ?>>
      %subview = memref.subview %reinterpret_cast[0] [%22] [1] {ssbuffer.block_id = 32 : i32} : memref<64xbf16, strided<[8], offset: ?>> to memref<?xbf16, strided<[8], offset: ?>>
      %subview_4 = memref.subview %alloc[%21] [%22] [1] {ssbuffer.block_id = 32 : i32} : memref<64xbf16> to memref<?xbf16, strided<[1], offset: ?>>
      memref.copy %subview, %subview_4 {ssbuffer.block_id = 32 : i32} : memref<?xbf16, strided<[8], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      %28 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 32 : i32} : memref<64xbf16> to tensor<64xbf16>
      %29 = tensor.empty() {ssbuffer.block_id = 32 : i32} : tensor<64x64xbf16>
      %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [64], strides: [8] {ssbuffer.block_id = 32 : i32} : memref<?xbf16> to memref<64xbf16, strided<[8], offset: ?>>
      %subview_6 = memref.subview %reinterpret_cast_5[0] [%22] [1] {ssbuffer.block_id = 32 : i32} : memref<64xbf16, strided<[8], offset: ?>> to memref<?xbf16, strided<[8], offset: ?>>
      %subview_7 = memref.subview %alloc_3[%21] [%22] [1] {ssbuffer.block_id = 32 : i32} : memref<64xbf16> to memref<?xbf16, strided<[1], offset: ?>>
      memref.copy %subview_6, %subview_7 {ssbuffer.block_id = 32 : i32} : memref<?xbf16, strided<[8], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
      %30 = bufferization.to_tensor %alloc_3 restrict writable {ssbuffer.block_id = 32 : i32} : memref<64xbf16> to tensor<64xbf16>
      %31 = math.exp %30 {ssbuffer.block_id = 32 : i32} : tensor<64xbf16>
      %32 = arith.mulf %28, %31 {ssbuffer.block_id = 32 : i32} : tensor<64xbf16>
      %expanded = tensor.expand_shape %32 [[0, 1]] output_shape [64, 1] {ssbuffer.block_id = 32 : i32} : tensor<64xbf16> into tensor<64x1xbf16>
      %broadcasted = linalg.broadcast ins(%32 : tensor<64xbf16>) outs(%29 : tensor<64x64xbf16>) dimensions = [1]  {ssbuffer.block_id = 32 : i32}
      %33 = arith.extf %expanded {ssbuffer.block_id = 32 : i32} : tensor<64x1xbf16> to tensor<64x1xf32>
      %collapsed = tensor.collapse_shape %33 [[0, 1]] {ssbuffer.block_id = 32 : i32} : tensor<64x1xf32> into tensor<64xf32>
      %broadcasted_8 = linalg.broadcast ins(%collapsed : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 32 : i32}
      %expanded_9 = tensor.expand_shape %31 [[0, 1]] output_shape [64, 1] {ssbuffer.block_id = 32 : i32} : tensor<64xbf16> into tensor<64x1xbf16>
      %34 = arith.extf %expanded_9 {ssbuffer.block_id = 32 : i32} : tensor<64x1xbf16> to tensor<64x1xf32>
      %collapsed_10 = tensor.collapse_shape %34 [[0, 1]] {ssbuffer.block_id = 32 : i32} : tensor<64x1xf32> into tensor<64xf32>
      %broadcasted_11 = linalg.broadcast ins(%collapsed_10 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 32 : i32}
      %expanded_12 = tensor.expand_shape %28 [[0, 1]] output_shape [64, 1] {ssbuffer.block_id = 32 : i32} : tensor<64xbf16> into tensor<64x1xbf16>
      %broadcasted_13 = linalg.broadcast ins(%28 : tensor<64xbf16>) outs(%29 : tensor<64x64xbf16>) dimensions = [1]  {ssbuffer.block_id = 32 : i32}
      %35 = arith.extf %expanded_12 {ssbuffer.block_id = 32 : i32} : tensor<64x1xbf16> to tensor<64x1xf32>
      %collapsed_14 = tensor.collapse_shape %35 [[0, 1]] {ssbuffer.block_id = 32 : i32} : tensor<64x1xf32> into tensor<64xf32>
      %broadcasted_15 = linalg.broadcast ins(%collapsed_14 : tensor<64xf32>) outs(%0 : tensor<64x64xf32>) dimensions = [1]  {ssbuffer.block_id = 32 : i32}
      %alloc_16 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_16 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_17 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 10 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 10 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 10 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 11
      %alloc_18 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 15 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<15>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 15 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 15 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
      %alloc_19 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 16 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_19 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<16>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 16 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 16 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
      %alloc_20 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 17 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_20 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<17>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 17 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 17 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
      %36:3 = scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg21 = %1, %arg22 = %3, %arg23 = %3) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
        %72 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 21 : i32} : i32
        %73 = arith.maxsi %72, %c0_i32 {ssbuffer.block_id = 21 : i32} : i32
        %74 = arith.index_cast %73 {ssbuffer.block_id = 21 : i32} : i32 to index
        %75 = arith.muli %11, %c1024 {ssbuffer.block_id = 21 : i32} : index
        %76 = arith.addi %75, %25 {ssbuffer.block_id = 21 : i32} : index
        %77 = arith.addi %76, %74 {ssbuffer.block_id = 21 : i32} : index
        %78 = arith.subi %77, %25 {ssbuffer.block_id = 21 : i32} : index
        %79 = arith.divsi %78, %c1024 {ssbuffer.block_id = 21 : i32} : index
        %80 = arith.subi %13, %79 {ssbuffer.block_id = 21 : i32} : index
        %81 = arith.maxsi %80, %c0 {ssbuffer.block_id = 21 : i32} : index
        %82 = arith.minsi %81, %c64 {ssbuffer.block_id = 21 : i32} : index
        %83 = arith.remsi %78, %c1024 {ssbuffer.block_id = 21 : i32} : index
        %84 = arith.subi %c128, %83 {ssbuffer.block_id = 21 : i32} : index
        %85 = arith.maxsi %84, %c0 {ssbuffer.block_id = 21 : i32} : index
        %86 = arith.minsi %85, %c64 {ssbuffer.block_id = 21 : i32} : index
        %87 = arith.minsi %20, %82 {ssbuffer.block_id = 21 : i32} : index
        %88 = arith.subi %82, %87 {ssbuffer.block_id = 21 : i32} : index
        %89 = arith.subi %c0_i32, %72 {ssbuffer.block_id = 21 : i32} : i32
        %90 = arith.maxsi %89, %c0_i32 {ssbuffer.block_id = 21 : i32} : i32
        %91 = arith.index_cast %90 {ssbuffer.block_id = 21 : i32} : i32 to index
        %92 = arith.minsi %91, %86 {ssbuffer.block_id = 21 : i32} : index
        %93 = arith.subi %86, %92 {ssbuffer.block_id = 21 : i32} : index
        %94 = arith.cmpi slt, %88, %c64 {ssbuffer.block_id = 21 : i32} : index
        %95 = arith.cmpi slt, %93, %c64 {ssbuffer.block_id = 21 : i32} : index
        %96 = arith.ori %94, %95 {ssbuffer.block_id = 21 : i32} : i1
        %reinterpret_cast_56 = memref.reinterpret_cast %arg9 to offset: [%77], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 21 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %subview_57 = memref.subview %reinterpret_cast_56[0, 0] [%88, %93] [1, 1] {ssbuffer.block_id = 21 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %alloc_58 = memref.alloc() {ssbuffer.block_id = 22 : i32} : memref<64x64xbf16>
        %reinterpret_cast_59 = memref.reinterpret_cast %arg2 to offset: [%77], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 22 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %subview_60 = memref.subview %reinterpret_cast_59[0, 0] [%88, %93] [1, 1] {ssbuffer.block_id = 22 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %subview_61 = memref.subview %alloc_58[%87, %92] [%88, %93] [1, 1] {ssbuffer.block_id = 22 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        scf.if %96 {
          linalg.fill {ssbuffer.block_id = 22 : i32} ins(%cst_2 : bf16) outs(%alloc_58 : memref<64x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 22 : i32}
        memref.copy %subview_60, %subview_61 {ssbuffer.block_id = 22 : i32} : memref<?x?xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %97 = bufferization.to_tensor %alloc_58 restrict writable {ssbuffer.block_id = 22 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %98 = arith.mulf %97, %broadcasted {ssbuffer.block_id = 22 : i32} : tensor<64x64xbf16>
        %reshape_62 = tensor.reshape %98(%cst_0) {ssbuffer.block_id = 22 : i32} : (tensor<64x64xbf16>, tensor<3xi64>) -> tensor<64x4x16xbf16>
        %99 = tensor.empty() {ssbuffer.block_id = 22 : i32} : tensor<4x64x16xbf16>
        %transposed_63 = linalg.transpose ins(%reshape_62 : tensor<64x4x16xbf16>) outs(%99 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 22 : i32}
        %reshape_64 = tensor.reshape %transposed_63(%cst) {ssbuffer.block_id = 22 : i32} : (tensor<4x64x16xbf16>, tensor<4xi64>) -> tensor<4x4x16x16xbf16>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.copy ins(%reshape_64 : tensor<4x4x16x16xbf16>) outs(%alloc_16 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 22 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 17 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = -1
        %memspacecast_65 = memref.memory_space_cast %alloc_20 {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 17 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %100 = bufferization.to_tensor %memspacecast_65 restrict writable {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 17 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 16 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = -1
        %memspacecast_66 = memref.memory_space_cast %alloc_19 {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 16 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %101 = bufferization.to_tensor %memspacecast_66 restrict writable {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 16 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 15 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = -1
        %memspacecast_67 = memref.memory_space_cast %alloc_18 {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 15 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %102 = bufferization.to_tensor %memspacecast_67 restrict writable {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 15 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 10 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 11
        %memspacecast_68 = memref.memory_space_cast %alloc_17 {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 10 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %103 = bufferization.to_tensor %memspacecast_68 restrict writable {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 10 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        %104 = arith.addf %103, %arg21 {ssbuffer.block_id = 23 : i32} : tensor<64x64xf32>
        %105 = arith.mulf %100, %broadcasted_8 {ssbuffer.block_id = 23 : i32} : tensor<64x64xf32>
        %106 = arith.extf %97 {ssbuffer.block_id = 23 : i32} : tensor<64x64xbf16> to tensor<64x64xf32>
        %107 = arith.mulf %101, %106 {ssbuffer.block_id = 23 : i32} : tensor<64x64xf32>
        %108 = arith.mulf %107, %broadcasted_11 {ssbuffer.block_id = 23 : i32} : tensor<64x64xf32>
        %reduced_69 = linalg.reduce ins(%108 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1]  {ssbuffer.block_id = 23 : i32}
          (%in: f32, %init: f32) {
            %114 = arith.addf %in, %init : f32
            linalg.yield %114 : f32
          }
        %109 = arith.addf %arg22, %reduced_69 {ssbuffer.block_id = 23 : i32} : tensor<64xf32>
        %110 = arith.extf %98 {ssbuffer.block_id = 23 : i32} : tensor<64x64xbf16> to tensor<64x64xf32>
        %111 = arith.mulf %102, %110 {ssbuffer.block_id = 23 : i32} : tensor<64x64xf32>
        %reduced_70 = linalg.reduce ins(%111 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1]  {ssbuffer.block_id = 23 : i32}
          (%in: f32, %init: f32) {
            %114 = arith.addf %in, %init : f32
            linalg.yield %114 : f32
          }
        %112 = arith.addf %arg23, %reduced_70 {ssbuffer.block_id = 23 : i32} : tensor<64xf32>
        %113 = arith.truncf %105 {ssbuffer.block_id = 23 : i32} : tensor<64x64xf32> to tensor<64x64xbf16>
        %extracted_slice_71 = tensor.extract_slice %113[%87, %92] [%88, %93] [1, 1] {ssbuffer.block_id = 23 : i32} : tensor<64x64xbf16> to tensor<?x?xbf16>
        bufferization.materialize_in_destination %extracted_slice_71 in writable %subview_57 {ssbuffer.block_id = 23 : i32} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[1024, 1], offset: ?>>) -> ()
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 10 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 11
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 15 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 16 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.sync_block_set {ssbuffer.block_id = 23 : i32, ssbuffer.transfer_id = 17 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
        scf.yield %104, %109, %112 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
      } {ssbuffer.block_id = 37 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      %alloc_21 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 3 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_21 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 3 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_22 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_22 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
      %alloc_23 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 8 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_23 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 8 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 8 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
      %alloc_24 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 13 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_24 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<13>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 13 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 13 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 14
      %37:2 = scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg21 = %36#0, %arg22 = %36#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
        %72 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 25 : i32} : i32
        %73 = arith.maxsi %72, %c0_i32 {ssbuffer.block_id = 25 : i32} : i32
        %74 = arith.index_cast %73 {ssbuffer.block_id = 25 : i32} : i32 to index
        %75 = arith.muli %11, %c1024 {ssbuffer.block_id = 25 : i32} : index
        %76 = arith.addi %75, %25 {ssbuffer.block_id = 25 : i32} : index
        %77 = arith.addi %76, %74 {ssbuffer.block_id = 25 : i32} : index
        %78 = arith.subi %77, %25 {ssbuffer.block_id = 25 : i32} : index
        %79 = arith.divsi %78, %c1024 {ssbuffer.block_id = 25 : i32} : index
        %80 = arith.subi %13, %79 {ssbuffer.block_id = 25 : i32} : index
        %81 = arith.maxsi %80, %c0 {ssbuffer.block_id = 25 : i32} : index
        %82 = arith.minsi %81, %c64 {ssbuffer.block_id = 25 : i32} : index
        %83 = arith.remsi %78, %c1024 {ssbuffer.block_id = 25 : i32} : index
        %84 = arith.subi %c128, %83 {ssbuffer.block_id = 25 : i32} : index
        %85 = arith.maxsi %84, %c0 {ssbuffer.block_id = 25 : i32} : index
        %86 = arith.minsi %85, %c64 {ssbuffer.block_id = 25 : i32} : index
        %87 = arith.minsi %20, %82 {ssbuffer.block_id = 25 : i32} : index
        %88 = arith.subi %82, %87 {ssbuffer.block_id = 25 : i32} : index
        %89 = arith.subi %c0_i32, %72 {ssbuffer.block_id = 25 : i32} : i32
        %90 = arith.maxsi %89, %c0_i32 {ssbuffer.block_id = 25 : i32} : i32
        %91 = arith.index_cast %90 {ssbuffer.block_id = 25 : i32} : i32 to index
        %92 = arith.minsi %91, %86 {ssbuffer.block_id = 25 : i32} : index
        %93 = arith.subi %86, %92 {ssbuffer.block_id = 25 : i32} : index
        %94 = arith.cmpi slt, %88, %c64 {ssbuffer.block_id = 25 : i32} : index
        %95 = arith.cmpi slt, %93, %c64 {ssbuffer.block_id = 25 : i32} : index
        %96 = arith.ori %94, %95 {ssbuffer.block_id = 25 : i32} : i1
        %reinterpret_cast_56 = memref.reinterpret_cast %arg10 to offset: [%77], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 25 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %subview_57 = memref.subview %reinterpret_cast_56[0, 0] [%88, %93] [1, 1] {ssbuffer.block_id = 25 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %alloc_58 = memref.alloc() {ssbuffer.block_id = 26 : i32} : memref<64x64xbf16>
        %reinterpret_cast_59 = memref.reinterpret_cast %arg3 to offset: [%77], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 26 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %subview_60 = memref.subview %reinterpret_cast_59[0, 0] [%88, %93] [1, 1] {ssbuffer.block_id = 26 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %subview_61 = memref.subview %alloc_58[%87, %92] [%88, %93] [1, 1] {ssbuffer.block_id = 26 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        scf.if %96 {
          linalg.fill {ssbuffer.block_id = 26 : i32} ins(%cst_2 : bf16) outs(%alloc_58 : memref<64x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 26 : i32}
        memref.copy %subview_60, %subview_61 {ssbuffer.block_id = 26 : i32} : memref<?x?xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %97 = bufferization.to_tensor %alloc_58 restrict writable {ssbuffer.block_id = 26 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %98 = arith.mulf %97, %broadcasted_13 {ssbuffer.block_id = 26 : i32} : tensor<64x64xbf16>
        %reshape_62 = tensor.reshape %98(%cst_0) {ssbuffer.block_id = 26 : i32} : (tensor<64x64xbf16>, tensor<3xi64>) -> tensor<64x4x16xbf16>
        %99 = tensor.empty() {ssbuffer.block_id = 26 : i32} : tensor<4x64x16xbf16>
        %transposed_63 = linalg.transpose ins(%reshape_62 : tensor<64x4x16xbf16>) outs(%99 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 26 : i32}
        %reshape_64 = tensor.reshape %transposed_63(%cst) {ssbuffer.block_id = 26 : i32} : (tensor<4x64x16xbf16>, tensor<4xi64>) -> tensor<4x4x16x16xbf16>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        hivm.hir.copy ins(%reshape_64 : tensor<4x4x16x16xbf16>) outs(%alloc_21 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 3 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
        hivm.hir.sync_block_wait {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 13 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 14
        %memspacecast_65 = memref.memory_space_cast %alloc_24 {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 13 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %100 = bufferization.to_tensor %memspacecast_65 restrict writable {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 13 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 8 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 9
        %memspacecast_66 = memref.memory_space_cast %alloc_23 {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 8 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %101 = bufferization.to_tensor %memspacecast_66 restrict writable {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 8 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 8
        %memspacecast_67 = memref.memory_space_cast %alloc_22 {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %102 = bufferization.to_tensor %memspacecast_67 restrict writable {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        %103 = arith.addf %100, %arg21 {ssbuffer.block_id = 27 : i32} : tensor<64x64xf32>
        %104 = arith.mulf %101, %broadcasted_15 {ssbuffer.block_id = 27 : i32} : tensor<64x64xf32>
        %105 = arith.extf %97 {ssbuffer.block_id = 27 : i32} : tensor<64x64xbf16> to tensor<64x64xf32>
        %106 = arith.mulf %102, %105 {ssbuffer.block_id = 27 : i32} : tensor<64x64xf32>
        %reduced_68 = linalg.reduce ins(%106 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1]  {ssbuffer.block_id = 27 : i32}
          (%in: f32, %init: f32) {
            %109 = arith.addf %in, %init : f32
            linalg.yield %109 : f32
          }
        %107 = arith.addf %arg22, %reduced_68 {ssbuffer.block_id = 27 : i32} : tensor<64xf32>
        %108 = arith.truncf %104 {ssbuffer.block_id = 27 : i32} : tensor<64x64xf32> to tensor<64x64xbf16>
        %extracted_slice_69 = tensor.extract_slice %108[%87, %92] [%88, %93] [1, 1] {ssbuffer.block_id = 27 : i32} : tensor<64x64xbf16> to tensor<?x?xbf16>
        bufferization.materialize_in_destination %extracted_slice_69 in writable %subview_57 {ssbuffer.block_id = 27 : i32} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[1024, 1], offset: ?>>) -> ()
        hivm.hir.sync_block_set {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 7 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 8
        hivm.hir.sync_block_set {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 8 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 9
        hivm.hir.sync_block_set {ssbuffer.block_id = 27 : i32, ssbuffer.transfer_id = 13 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 14
        scf.yield %103, %107 : tensor<64x64xf32>, tensor<64xf32>
      } {ssbuffer.block_id = 38 : i32, ssbuffer.main_loop = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 3 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 4
      %38 = tensor.empty() {ssbuffer.block_id = 33 : i32} : tensor<64xi32>
      %39 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%38 : tensor<64xi32>) attrs =  {ssbuffer.block_id = 33 : i32, tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 64 : index} {
      ^bb0(%out: i32):
        %72 = linalg.index 0 : index
        %73 = arith.index_cast %72 : index to i32
        linalg.yield %73 : i32
      } -> tensor<64xi32>
      %40 = linalg.fill {ssbuffer.block_id = 33 : i32} ins(%9 : i32) outs(%38 : tensor<64xi32>) -> tensor<64xi32>
      %41 = arith.addi %40, %39 {ssbuffer.block_id = 33 : i32} : tensor<64xi32>
      %42 = linalg.fill {ssbuffer.block_id = 33 : i32} ins(%arg13 : i32) outs(%38 : tensor<64xi32>) -> tensor<64xi32>
      %43 = arith.cmpi slt, %41, %42 {ssbuffer.block_id = 33 : i32} : tensor<64xi32>
      %44 = tensor.empty() {ssbuffer.block_id = 33 : i32} : tensor<64x64xi32>
      %broadcasted_25 = linalg.broadcast ins(%41 : tensor<64xi32>) outs(%44 : tensor<64x64xi32>) dimensions = [1]  {ssbuffer.block_id = 33 : i32}
      %broadcasted_26 = linalg.broadcast ins(%41 : tensor<64xi32>) outs(%44 : tensor<64x64xi32>) dimensions = [0]  {ssbuffer.block_id = 33 : i32}
      %45 = arith.cmpi sgt, %broadcasted_25, %broadcasted_26 {ssbuffer.block_id = 33 : i32} : tensor<64x64xi32>
      %46 = tensor.empty() {ssbuffer.block_id = 33 : i32} : tensor<64x64xi1>
      %broadcasted_27 = linalg.broadcast ins(%43 : tensor<64xi1>) outs(%46 : tensor<64x64xi1>) dimensions = [1]  {ssbuffer.block_id = 33 : i32}
      %broadcasted_28 = linalg.broadcast ins(%43 : tensor<64xi1>) outs(%46 : tensor<64x64xi1>) dimensions = [0]  {ssbuffer.block_id = 33 : i32}
      %47 = arith.andi %broadcasted_27, %broadcasted_28 {ssbuffer.block_id = 33 : i32} : tensor<64x64xi1>
      %48 = arith.andi %45, %47 {ssbuffer.block_id = 33 : i32} : tensor<64x64xi1>
      %49 = arith.select %48, %37#0, %1 {ssbuffer.block_id = 33 : i32} : tensor<64x64xi1>, tensor<64x64xf32>
      %50 = arith.truncf %49 {ssbuffer.block_id = 33 : i32} : tensor<64x64xf32> to tensor<64x64xbf16>
      %reshape = tensor.reshape %50(%cst_0) {ssbuffer.block_id = 33 : i32} : (tensor<64x64xbf16>, tensor<3xi64>) -> tensor<64x4x16xbf16>
      %51 = tensor.empty() {ssbuffer.block_id = 33 : i32} : tensor<4x64x16xbf16>
      %transposed = linalg.transpose ins(%reshape : tensor<64x4x16xbf16>) outs(%51 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 33 : i32}
      %reshape_29 = tensor.reshape %transposed(%cst) {ssbuffer.block_id = 33 : i32} : (tensor<4x64x16xbf16>, tensor<4xi64>) -> tensor<4x4x16x16xbf16>
      %alloc_30 = memref.alloc() {ssbuffer.block_id = 33 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_30 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 33 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%reshape_29 : tensor<4x4x16x16xbf16>) outs(%alloc_30 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 33 : i32, ssbuffer.transfer_id = 0 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 33 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 9 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 10
      %alloc_31 = memref.alloc() {ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 9 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_31 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>, ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 9 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_31 {ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 9 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %52 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 9 : i32} : memref<64x64xf32> to tensor<64x64xf32>
      %53 = arith.truncf %52 {ssbuffer.block_id = 34 : i32} : tensor<64x64xf32> to tensor<64x64xbf16>
      %reshape_32 = tensor.reshape %53(%cst_0) {ssbuffer.block_id = 34 : i32} : (tensor<64x64xbf16>, tensor<3xi64>) -> tensor<64x4x16xbf16>
      %54 = tensor.empty() {ssbuffer.block_id = 34 : i32} : tensor<4x64x16xbf16>
      %transposed_33 = linalg.transpose ins(%reshape_32 : tensor<64x4x16xbf16>) outs(%54 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 34 : i32}
      %reshape_34 = tensor.reshape %transposed_33(%cst) {ssbuffer.block_id = 34 : i32} : (tensor<4x64x16xbf16>, tensor<4xi64>) -> tensor<4x4x16x16xbf16>
      %alloc_35 = memref.alloc() {ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 4 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_35 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 4 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%reshape_34 : tensor<4x4x16x16xbf16>) outs(%alloc_35 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 4 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 34 : i32, ssbuffer.transfer_id = 4 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
      hivm.hir.sync_block_wait {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 14 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = -1
      %alloc_36 = memref.alloc() {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 14 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_36 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 14 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %memspacecast_37 = memref.memory_space_cast %alloc_36 {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 14 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
      %55 = bufferization.to_tensor %memspacecast_37 restrict writable {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 14 : i32} : memref<64x64xf32> to tensor<64x64xf32>
      %broadcasted_38 = linalg.broadcast ins(%30 : tensor<64xbf16>) outs(%29 : tensor<64x64xbf16>) dimensions = [1]  {ssbuffer.block_id = 35 : i32}
      %broadcasted_39 = linalg.broadcast ins(%30 : tensor<64xbf16>) outs(%29 : tensor<64x64xbf16>) dimensions = [0]  {ssbuffer.block_id = 35 : i32}
      %56 = arith.subf %broadcasted_38, %broadcasted_39 {ssbuffer.block_id = 35 : i32} : tensor<64x64xbf16>
      %57 = arith.extf %56 {ssbuffer.block_id = 35 : i32} : tensor<64x64xbf16> to tensor<64x64xf32>
      %58 = math.exp %57 {ssbuffer.block_id = 35 : i32} : tensor<64x64xf32>
      %59 = arith.mulf %55, %58 {ssbuffer.block_id = 35 : i32} : tensor<64x64xf32>
      %60 = arith.subf %1, %59 {ssbuffer.block_id = 35 : i32} : tensor<64x64xf32>
      %61 = arith.select %48, %60, %1 {ssbuffer.block_id = 35 : i32} : tensor<64x64xi1>, tensor<64x64xf32>
      %62 = arith.truncf %61 {ssbuffer.block_id = 35 : i32} : tensor<64x64xf32> to tensor<64x64xbf16>
      %reshape_40 = tensor.reshape %62(%cst_0) {ssbuffer.block_id = 35 : i32} : (tensor<64x64xbf16>, tensor<3xi64>) -> tensor<64x4x16xbf16>
      %63 = tensor.empty() {ssbuffer.block_id = 35 : i32} : tensor<4x64x16xbf16>
      %transposed_41 = linalg.transpose ins(%reshape_40 : tensor<64x4x16xbf16>) outs(%63 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 35 : i32}
      %reshape_42 = tensor.reshape %transposed_41(%cst) {ssbuffer.block_id = 35 : i32} : (tensor<4x64x16xbf16>, tensor<4xi64>) -> tensor<4x4x16x16xbf16>
      %alloc_43 = memref.alloc() {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_43 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%reshape_42 : tensor<4x4x16x16xbf16>) outs(%alloc_43 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 2 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
      %alloc_44 = memref.alloc() {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 6 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_44 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 6 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.copy ins(%reshape_42 : tensor<4x4x16x16xbf16>) outs(%alloc_44 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 6 : i32}
      hivm.hir.sync_block_set {ssbuffer.block_id = 35 : i32, ssbuffer.transfer_id = 6 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 7
      %alloc_45 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 5 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_45 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 5 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      %alloc_46 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 11 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_46 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 11 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 11 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 12
      %alloc_47 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 12 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_47 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 12 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 12 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 13
      %alloc_48 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 18 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_48 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<18>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 18 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 18 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
      %alloc_49 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 19 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_49 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<19>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 19 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 19 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
      %64:2 = scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg21 = %1, %arg22 = %37#1) -> (tensor<64x64xf32>, tensor<64xf32>)  : i32 {
        %72 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 29 : i32} : i32
        %73 = arith.maxsi %72, %c0_i32 {ssbuffer.block_id = 29 : i32} : i32
        %74 = arith.index_cast %73 {ssbuffer.block_id = 29 : i32} : i32 to index
        %75 = arith.muli %11, %c1024 {ssbuffer.block_id = 29 : i32} : index
        %76 = arith.addi %75, %25 {ssbuffer.block_id = 29 : i32} : index
        %77 = arith.addi %76, %74 {ssbuffer.block_id = 29 : i32} : index
        %reinterpret_cast_56 = memref.reinterpret_cast %arg2 to offset: [%77], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 29 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %alloc_57 = memref.alloc() {ssbuffer.block_id = 29 : i32} : memref<64x64xbf16>
        %78 = arith.subi %77, %25 {ssbuffer.block_id = 29 : i32} : index
        %79 = arith.divsi %78, %c1024 {ssbuffer.block_id = 29 : i32} : index
        %80 = arith.subi %13, %79 {ssbuffer.block_id = 29 : i32} : index
        %81 = arith.maxsi %80, %c0 {ssbuffer.block_id = 29 : i32} : index
        %82 = arith.minsi %81, %c64 {ssbuffer.block_id = 29 : i32} : index
        %83 = arith.remsi %78, %c1024 {ssbuffer.block_id = 29 : i32} : index
        %84 = arith.subi %c128, %83 {ssbuffer.block_id = 29 : i32} : index
        %85 = arith.maxsi %84, %c0 {ssbuffer.block_id = 29 : i32} : index
        %86 = arith.minsi %85, %c64 {ssbuffer.block_id = 29 : i32} : index
        %87 = arith.minsi %20, %82 {ssbuffer.block_id = 29 : i32} : index
        %88 = arith.subi %82, %87 {ssbuffer.block_id = 29 : i32} : index
        %89 = arith.subi %c0_i32, %72 {ssbuffer.block_id = 29 : i32} : i32
        %90 = arith.maxsi %89, %c0_i32 {ssbuffer.block_id = 29 : i32} : i32
        %91 = arith.index_cast %90 {ssbuffer.block_id = 29 : i32} : i32 to index
        %92 = arith.minsi %91, %86 {ssbuffer.block_id = 29 : i32} : index
        %93 = arith.subi %86, %92 {ssbuffer.block_id = 29 : i32} : index
        %94 = arith.cmpi slt, %88, %c64 {ssbuffer.block_id = 29 : i32} : index
        %95 = arith.cmpi slt, %93, %c64 {ssbuffer.block_id = 29 : i32} : index
        %96 = arith.ori %94, %95 {ssbuffer.block_id = 29 : i32} : i1
        %subview_58 = memref.subview %reinterpret_cast_56[0, 0] [%88, %93] [1, 1] {ssbuffer.block_id = 29 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %subview_59 = memref.subview %alloc_57[%87, %92] [%88, %93] [1, 1] {ssbuffer.block_id = 29 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        memref.copy %subview_58, %subview_59 {ssbuffer.block_id = 29 : i32} : memref<?x?xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %97 = bufferization.to_tensor %alloc_57 restrict writable {ssbuffer.block_id = 29 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %98 = arith.mulf %97, %broadcasted_13 {ssbuffer.block_id = 29 : i32} : tensor<64x64xbf16>
        %reshape_60 = tensor.reshape %98(%cst_0) {ssbuffer.block_id = 29 : i32} : (tensor<64x64xbf16>, tensor<3xi64>) -> tensor<64x4x16xbf16>
        %99 = tensor.empty() {ssbuffer.block_id = 29 : i32} : tensor<4x64x16xbf16>
        %transposed_61 = linalg.transpose ins(%reshape_60 : tensor<64x4x16xbf16>) outs(%99 : tensor<4x64x16xbf16>) permutation = [1, 0, 2]  {ssbuffer.block_id = 29 : i32}
        %reshape_62 = tensor.reshape %transposed_61(%cst) {ssbuffer.block_id = 29 : i32} : (tensor<4x64x16xbf16>, tensor<4xi64>) -> tensor<4x4x16x16xbf16>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 29 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
        hivm.hir.copy ins(%reshape_62 : tensor<4x4x16x16xbf16>) outs(%alloc_45 : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 29 : i32, ssbuffer.transfer_id = 5 : i32}
        hivm.hir.sync_block_set {ssbuffer.block_id = 29 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 6
        hivm.hir.sync_block_wait {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 19 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = -1
        %memspacecast_63 = memref.memory_space_cast %alloc_49 {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 19 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %100 = bufferization.to_tensor %memspacecast_63 restrict writable {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 19 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 18 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = -1
        %memspacecast_64 = memref.memory_space_cast %alloc_48 {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 18 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %101 = bufferization.to_tensor %memspacecast_64 restrict writable {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 18 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 12 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 13
        %memspacecast_65 = memref.memory_space_cast %alloc_47 {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 12 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %102 = bufferization.to_tensor %memspacecast_65 restrict writable {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 12 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 11 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 12
        %memspacecast_66 = memref.memory_space_cast %alloc_46 {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 11 : i32} : memref<64x64xf32, #hivm.address_space<ub>> to memref<64x64xf32>
        %103 = bufferization.to_tensor %memspacecast_66 restrict writable {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 11 : i32} : memref<64x64xf32> to tensor<64x64xf32>
        %alloc_67 = memref.alloc() {ssbuffer.block_id = 30 : i32} : memref<64x64xbf16>
        %reinterpret_cast_68 = memref.reinterpret_cast %arg9 to offset: [%77], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 30 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %subview_69 = memref.subview %reinterpret_cast_68[0, 0] [%88, %93] [1, 1] {ssbuffer.block_id = 30 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %subview_70 = memref.subview %alloc_67[%87, %92] [%88, %93] [1, 1] {ssbuffer.block_id = 30 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        scf.if %96 {
          linalg.fill {ssbuffer.block_id = 30 : i32} ins(%cst_2 : bf16) outs(%alloc_67 : memref<64x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 30 : i32}
        %104 = arith.addf %101, %arg21 {ssbuffer.block_id = 30 : i32} : tensor<64x64xf32>
        %105 = arith.extf %97 {ssbuffer.block_id = 30 : i32} : tensor<64x64xbf16> to tensor<64x64xf32>
        %106 = arith.mulf %102, %105 {ssbuffer.block_id = 30 : i32} : tensor<64x64xf32>
        %reduced_71 = linalg.reduce ins(%106 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1]  {ssbuffer.block_id = 30 : i32}
          (%in: f32, %init: f32) {
            %114 = arith.addf %in, %init : f32
            linalg.yield %114 : f32
          }
        %107 = arith.addf %arg22, %reduced_71 {ssbuffer.block_id = 30 : i32} : tensor<64xf32>
        %108 = arith.mulf %103, %broadcasted_15 {ssbuffer.block_id = 30 : i32} : tensor<64x64xf32>
        %transposed_72 = linalg.transpose ins(%100 : tensor<64x64xf32>) outs(%0 : tensor<64x64xf32>) permutation = [1, 0]  {ssbuffer.block_id = 30 : i32}
        %109 = arith.addf %108, %transposed_72 {ssbuffer.block_id = 30 : i32} : tensor<64x64xf32>
        memref.copy %subview_69, %subview_70 {ssbuffer.block_id = 30 : i32} : memref<?x?xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %110 = bufferization.to_tensor %alloc_67 restrict writable {ssbuffer.block_id = 30 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %111 = arith.extf %110 {ssbuffer.block_id = 30 : i32} : tensor<64x64xbf16> to tensor<64x64xf32>
        %112 = arith.addf %109, %111 {ssbuffer.block_id = 30 : i32} : tensor<64x64xf32>
        %113 = arith.truncf %112 {ssbuffer.block_id = 30 : i32} : tensor<64x64xf32> to tensor<64x64xbf16>
        %extracted_slice_73 = tensor.extract_slice %113[%87, %92] [%88, %93] [1, 1] {ssbuffer.block_id = 30 : i32} : tensor<64x64xbf16> to tensor<?x?xbf16>
        bufferization.materialize_in_destination %extracted_slice_73 in writable %subview_69 {ssbuffer.block_id = 30 : i32} : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[1024, 1], offset: ?>>) -> ()
        hivm.hir.sync_block_set {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 11 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 12
        hivm.hir.sync_block_set {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 12 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 13
        hivm.hir.sync_block_set {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 18 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.sync_block_set {ssbuffer.block_id = 30 : i32, ssbuffer.transfer_id = 19 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = -1
        scf.yield %104, %107 : tensor<64x64xf32>, tensor<64xf32>
      } {ssbuffer.block_id = 39 : i32, ssbuffer.main_loop = 2 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 5 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      %reinterpret_cast_50 = memref.reinterpret_cast %arg11 to offset: [%27], sizes: [64], strides: [8] {ssbuffer.block_id = 36 : i32} : memref<?xbf16> to memref<64xbf16, strided<[8], offset: ?>>
      gpu.barrier {ssbuffer.block_id = 36 : i32}
      %65 = arith.truncf %64#1 {ssbuffer.block_id = 36 : i32} : tensor<64xf32> to tensor<64xbf16>
      %extracted_slice = tensor.extract_slice %65[%21] [%22] [1] {ssbuffer.block_id = 36 : i32} : tensor<64xbf16> to tensor<?xbf16>
      %subview_51 = memref.subview %reinterpret_cast_50[0] [%22] [1] {ssbuffer.block_id = 36 : i32} : memref<64xbf16, strided<[8], offset: ?>> to memref<?xbf16, strided<[8], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_51 {ssbuffer.block_id = 36 : i32} : (tensor<?xbf16>, memref<?xbf16, strided<[8], offset: ?>>) -> ()
      %66 = arith.mulf %64#0, %broadcasted_15 {ssbuffer.block_id = 36 : i32} : tensor<64x64xf32>
      %67 = arith.extf %62 {ssbuffer.block_id = 36 : i32} : tensor<64x64xbf16> to tensor<64x64xf32>
      %68 = arith.mulf %67, %66 {ssbuffer.block_id = 36 : i32} : tensor<64x64xf32>
      %reinterpret_cast_52 = memref.reinterpret_cast %arg12 to offset: [%27], sizes: [64], strides: [8] {ssbuffer.block_id = 36 : i32} : memref<?xbf16> to memref<64xbf16, strided<[8], offset: ?>>
      %reduced = linalg.reduce ins(%68 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [1]  {ssbuffer.block_id = 36 : i32}
        (%in: f32, %init: f32) {
          %72 = arith.addf %in, %init : f32
          linalg.yield %72 : f32
        }
      %reduced_53 = linalg.reduce ins(%68 : tensor<64x64xf32>) outs(%3 : tensor<64xf32>) dimensions = [0]  {ssbuffer.block_id = 36 : i32}
        (%in: f32, %init: f32) {
          %72 = arith.addf %in, %init : f32
          linalg.yield %72 : f32
        }
      %69 = arith.subf %reduced, %reduced_53 {ssbuffer.block_id = 36 : i32} : tensor<64xf32>
      %70 = arith.addf %36#2, %69 {ssbuffer.block_id = 36 : i32} : tensor<64xf32>
      %71 = arith.truncf %70 {ssbuffer.block_id = 36 : i32} : tensor<64xf32> to tensor<64xbf16>
      %extracted_slice_54 = tensor.extract_slice %71[%21] [%22] [1] {ssbuffer.block_id = 36 : i32} : tensor<64xbf16> to tensor<?xbf16>
      %subview_55 = memref.subview %reinterpret_cast_52[0] [%22] [1] {ssbuffer.block_id = 36 : i32} : memref<64xbf16, strided<[8], offset: ?>> to memref<?xbf16, strided<[8], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_54 in writable %subview_55 {ssbuffer.block_id = 36 : i32} : (tensor<?xbf16>, memref<?xbf16, strided<[8], offset: ?>>) -> ()
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    scope.scope : () -> () {
      %0 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 16 : i32} : i32
      %1 = arith.remsi %arg18, %c8_i32 {ssbuffer.block_id = 16 : i32} : i32
      %2 = arith.muli %0, %arg13 {ssbuffer.block_id = 16 : i32} : i32
      %3 = arith.muli %2, %c8_i32 {ssbuffer.block_id = 16 : i32} : i32
      %4 = arith.addi %3, %1 {ssbuffer.block_id = 16 : i32} : i32
      %5 = arith.muli %arg17, %c64_i32 {ssbuffer.block_id = 16 : i32} : i32
      %6 = arith.maxsi %5, %c0_i32 {ssbuffer.block_id = 16 : i32} : i32
      %7 = arith.index_cast %6 {ssbuffer.block_id = 16 : i32} : i32 to index
      %8 = arith.index_cast %arg13 {ssbuffer.block_id = 16 : i32} : i32 to index
      %9 = arith.muli %4, %c64_i32 {ssbuffer.block_id = 16 : i32} : i32
      %10 = arith.index_cast %9 {ssbuffer.block_id = 16 : i32} : i32 to index
      %11 = arith.subi %c0_i32, %5 {ssbuffer.block_id = 16 : i32} : i32
      %12 = arith.maxsi %11, %c0_i32 {ssbuffer.block_id = 16 : i32} : i32
      %13 = arith.index_cast %12 {ssbuffer.block_id = 16 : i32} : i32 to index
      %14 = arith.muli %7, %c512 {ssbuffer.block_id = 16 : i32} : index
      %15 = arith.addi %14, %10 {ssbuffer.block_id = 16 : i32} : index
      %16 = arith.divsi %14, %c512 {ssbuffer.block_id = 16 : i32} : index
      %17 = arith.subi %8, %16 {ssbuffer.block_id = 16 : i32} : index
      %18 = arith.maxsi %17, %c0 {ssbuffer.block_id = 16 : i32} : index
      %19 = arith.minsi %18, %c64 {ssbuffer.block_id = 16 : i32} : index
      %20 = arith.remsi %14, %c512 {ssbuffer.block_id = 16 : i32} : index
      %21 = arith.subi %c64, %20 {ssbuffer.block_id = 16 : i32} : index
      %22 = arith.maxsi %21, %c0 {ssbuffer.block_id = 16 : i32} : index
      %23 = arith.minsi %22, %c64 {ssbuffer.block_id = 16 : i32} : index
      %24 = arith.minsi %13, %19 {ssbuffer.block_id = 16 : i32} : index
      %25 = arith.subi %19, %24 {ssbuffer.block_id = 16 : i32} : index
      %26 = arith.minsi %23, %c0 {ssbuffer.block_id = 16 : i32} : index
      %27 = arith.subi %23, %26 {ssbuffer.block_id = 16 : i32} : index
      %28 = tensor.empty() {ssbuffer.block_id = 16 : i32} : tensor<64x64xbf16>
      %29 = arith.muli %4, %c128_i32 {ssbuffer.block_id = 16 : i32} : i32
      %30 = arith.index_cast %29 {ssbuffer.block_id = 16 : i32} : i32 to index
      %31 = tensor.empty() {ssbuffer.block_id = 31 : i32} : tensor<64x64xf32>
      %32 = linalg.fill {ssbuffer.block_id = 31 : i32} ins(%cst_1 : f32) outs(%31 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %33 = arith.divsi %arg18, %c8_i32 {ssbuffer.block_id = 31 : i32} : i32
      %34 = arith.remsi %arg18, %c8_i32 {ssbuffer.block_id = 31 : i32} : i32
      %35 = arith.muli %33, %arg13 {ssbuffer.block_id = 31 : i32} : i32
      %36 = arith.muli %35, %c8_i32 {ssbuffer.block_id = 31 : i32} : i32
      %37 = arith.addi %36, %34 {ssbuffer.block_id = 31 : i32} : i32
      %38 = arith.muli %arg17, %c64_i32 {ssbuffer.block_id = 31 : i32} : i32
      %39 = arith.maxsi %38, %c0_i32 {ssbuffer.block_id = 31 : i32} : i32
      %40 = arith.index_cast %39 {ssbuffer.block_id = 31 : i32} : i32 to index
      %41 = arith.index_cast %arg13 {ssbuffer.block_id = 31 : i32} : i32 to index
      %42 = arith.subi %c0_i32, %38 {ssbuffer.block_id = 31 : i32} : i32
      %43 = arith.maxsi %42, %c0_i32 {ssbuffer.block_id = 31 : i32} : i32
      %44 = arith.index_cast %43 {ssbuffer.block_id = 31 : i32} : i32 to index
      %45 = arith.muli %40, %c512 {ssbuffer.block_id = 31 : i32} : index
      %46 = arith.divsi %45, %c512 {ssbuffer.block_id = 31 : i32} : index
      %47 = arith.subi %41, %46 {ssbuffer.block_id = 31 : i32} : index
      %48 = arith.maxsi %47, %c0 {ssbuffer.block_id = 31 : i32} : index
      %49 = arith.minsi %48, %c64 {ssbuffer.block_id = 31 : i32} : index
      %50 = arith.remsi %45, %c512 {ssbuffer.block_id = 31 : i32} : index
      %51 = arith.subi %c64, %50 {ssbuffer.block_id = 31 : i32} : index
      %52 = arith.maxsi %51, %c0 {ssbuffer.block_id = 31 : i32} : index
      %53 = arith.minsi %52, %c64 {ssbuffer.block_id = 31 : i32} : index
      %54 = arith.minsi %44, %49 {ssbuffer.block_id = 31 : i32} : index
      %55 = arith.subi %49, %54 {ssbuffer.block_id = 31 : i32} : index
      %56 = arith.minsi %53, %c0 {ssbuffer.block_id = 31 : i32} : index
      %57 = arith.subi %53, %56 {ssbuffer.block_id = 31 : i32} : index
      %58 = arith.cmpi slt, %55, %c64 {ssbuffer.block_id = 31 : i32} : index
      %59 = arith.cmpi slt, %57, %c64 {ssbuffer.block_id = 31 : i32} : index
      %60 = arith.ori %58, %59 {ssbuffer.block_id = 31 : i32} : i1
      %61 = arith.muli %37, %c128_i32 {ssbuffer.block_id = 31 : i32} : i32
      %62 = arith.index_cast %61 {ssbuffer.block_id = 31 : i32} : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%15], sizes: [64, 64], strides: [512, 1] {ssbuffer.block_id = 17 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[512, 1], offset: ?>>
      %alloc = memref.alloc() {ssbuffer.block_id = 17 : i32} : memref<64x64xbf16>
      %subview = memref.subview %reinterpret_cast[0, 0] [%25, %27] [1, 1] {ssbuffer.block_id = 17 : i32} : memref<64x64xbf16, strided<[512, 1], offset: ?>> to memref<?x?xbf16, strided<[512, 1], offset: ?>>
      %subview_3 = memref.subview %alloc[%24, %26] [%25, %27] [1, 1] {ssbuffer.block_id = 17 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
      scf.if %60 {
        linalg.fill {ssbuffer.block_id = 17 : i32} ins(%cst_2 : bf16) outs(%alloc : memref<64x64xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 17 : i32}
      memref.copy %subview, %subview_3 {ssbuffer.block_id = 17 : i32} : memref<?x?xbf16, strided<[512, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
      %63 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 17 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
      %transposed = linalg.transpose ins(%63 : tensor<64x64xbf16>) outs(%28 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 17 : i32}
      %alloc_4 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      %alloc_5 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 10 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<10>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 10 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 15 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<15>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 15 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 16 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<16>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 16 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 17 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<17>, ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 17 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %74 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 21 : i32} : i32
        %75 = arith.maxsi %74, %c0_i32 {ssbuffer.block_id = 21 : i32} : i32
        %76 = arith.index_cast %75 {ssbuffer.block_id = 21 : i32} : i32 to index
        %77 = arith.muli %40, %c1024 {ssbuffer.block_id = 21 : i32} : index
        %78 = arith.addi %77, %62 {ssbuffer.block_id = 21 : i32} : index
        %79 = arith.addi %78, %76 {ssbuffer.block_id = 21 : i32} : index
        %80 = arith.subi %79, %62 {ssbuffer.block_id = 21 : i32} : index
        %81 = arith.divsi %80, %c1024 {ssbuffer.block_id = 21 : i32} : index
        %82 = arith.subi %41, %81 {ssbuffer.block_id = 21 : i32} : index
        %83 = arith.maxsi %82, %c0 {ssbuffer.block_id = 21 : i32} : index
        %84 = arith.minsi %83, %c64 {ssbuffer.block_id = 21 : i32} : index
        %85 = arith.remsi %80, %c1024 {ssbuffer.block_id = 21 : i32} : index
        %86 = arith.subi %c128, %85 {ssbuffer.block_id = 21 : i32} : index
        %87 = arith.maxsi %86, %c0 {ssbuffer.block_id = 21 : i32} : index
        %88 = arith.minsi %87, %c64 {ssbuffer.block_id = 21 : i32} : index
        %89 = arith.minsi %44, %84 {ssbuffer.block_id = 21 : i32} : index
        %90 = arith.subi %84, %89 {ssbuffer.block_id = 21 : i32} : index
        %91 = arith.subi %c0_i32, %74 {ssbuffer.block_id = 21 : i32} : i32
        %92 = arith.maxsi %91, %c0_i32 {ssbuffer.block_id = 21 : i32} : i32
        %93 = arith.index_cast %92 {ssbuffer.block_id = 21 : i32} : i32 to index
        %94 = arith.minsi %93, %88 {ssbuffer.block_id = 21 : i32} : index
        %95 = arith.subi %88, %94 {ssbuffer.block_id = 21 : i32} : index
        %96 = arith.cmpi slt, %90, %c64 {ssbuffer.block_id = 21 : i32} : index
        %97 = arith.cmpi slt, %95, %c64 {ssbuffer.block_id = 21 : i32} : index
        %98 = arith.ori %96, %97 {ssbuffer.block_id = 21 : i32} : i1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %99 = hivm.hir.convert_layout %alloc_4 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32} : (memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<64x64xbf16, #hivm.address_space<cbuf>>
        %memspacecast_27 = memref.memory_space_cast %99 {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xbf16, #hivm.address_space<cbuf>> to memref<64x64xbf16>
        %100 = bufferization.to_tensor %memspacecast_27 restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %alloc_28 = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16>
        scf.if %98 {
          linalg.fill {ssbuffer.block_id = 2 : i32} ins(%cst_2 : bf16) outs(%alloc_28 : memref<64x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 2 : i32}
        %101 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 2 : i32} : i32
        %102 = arith.maxsi %101, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %103 = arith.index_cast %102 {ssbuffer.block_id = 2 : i32} : i32 to index
        %104 = arith.muli %7, %c1024 {ssbuffer.block_id = 2 : i32} : index
        %105 = arith.addi %104, %30 {ssbuffer.block_id = 2 : i32} : index
        %106 = arith.addi %105, %103 {ssbuffer.block_id = 2 : i32} : index
        %reinterpret_cast_29 = memref.reinterpret_cast %arg7 to offset: [%106], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 2 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %107 = arith.subi %106, %30 {ssbuffer.block_id = 2 : i32} : index
        %108 = arith.divsi %107, %c1024 {ssbuffer.block_id = 2 : i32} : index
        %109 = arith.subi %8, %108 {ssbuffer.block_id = 2 : i32} : index
        %110 = arith.maxsi %109, %c0 {ssbuffer.block_id = 2 : i32} : index
        %111 = arith.minsi %110, %c64 {ssbuffer.block_id = 2 : i32} : index
        %112 = arith.remsi %107, %c1024 {ssbuffer.block_id = 2 : i32} : index
        %113 = arith.subi %c128, %112 {ssbuffer.block_id = 2 : i32} : index
        %114 = arith.maxsi %113, %c0 {ssbuffer.block_id = 2 : i32} : index
        %115 = arith.minsi %114, %c64 {ssbuffer.block_id = 2 : i32} : index
        %116 = arith.minsi %13, %111 {ssbuffer.block_id = 2 : i32} : index
        %117 = arith.subi %111, %116 {ssbuffer.block_id = 2 : i32} : index
        %118 = arith.subi %c0_i32, %101 {ssbuffer.block_id = 2 : i32} : i32
        %119 = arith.maxsi %118, %c0_i32 {ssbuffer.block_id = 2 : i32} : i32
        %120 = arith.index_cast %119 {ssbuffer.block_id = 2 : i32} : i32 to index
        %121 = arith.minsi %120, %115 {ssbuffer.block_id = 2 : i32} : index
        %122 = arith.subi %115, %121 {ssbuffer.block_id = 2 : i32} : index
        %subview_30 = memref.subview %reinterpret_cast_29[0, 0] [%117, %122] [1, 1] {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %subview_31 = memref.subview %alloc_28[%116, %121] [%117, %122] [1, 1] {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        memref.copy %subview_30, %subview_31 {ssbuffer.block_id = 2 : i32} : memref<?x?xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %123 = bufferization.to_tensor %alloc_28 restrict writable {ssbuffer.block_id = 2 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %transposed_32 = linalg.transpose ins(%100 : tensor<64x64xbf16>) outs(%28 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 2 : i32}
        %124 = tensor.empty() {ssbuffer.block_id = 2 : i32} : tensor<64x64xf32>
        %125 = linalg.fill {ssbuffer.block_id = 2 : i32} ins(%cst_1 : f32) outs(%124 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %126 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32} ins(%123, %transposed_32 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%125 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 10 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 11
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 10 : i32} ins(%126 : tensor<64x64xf32>) outs(%alloc_5 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 2 : i32, ssbuffer.transfer_id = 10 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 11
        %127 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32} ins(%transposed, %123 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%32 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 15 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 15 : i32} ins(%127 : tensor<64x64xf32>) outs(%alloc_6 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 15 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = -1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 16 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 16 : i32} ins(%127 : tensor<64x64xf32>) outs(%alloc_7 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 16 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = -1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 17 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 17 : i32} ins(%127 : tensor<64x64xf32>) outs(%alloc_8 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 3 : i32, ssbuffer.transfer_id = 17 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = -1
      } {ssbuffer.block_id = 37 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 17 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 16 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 15 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 37 : i32, ssbuffer.transfer_id = 10 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 11
      %alloc_9 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 3 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 3 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
      %alloc_10 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_10 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<7>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 7 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_11 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 8 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<8>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 8 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_12 = memref.alloc() {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 13 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<13>, ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 13 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %74 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 25 : i32} : i32
        %75 = arith.maxsi %74, %c0_i32 {ssbuffer.block_id = 25 : i32} : i32
        %76 = arith.index_cast %75 {ssbuffer.block_id = 25 : i32} : i32 to index
        %77 = arith.muli %40, %c1024 {ssbuffer.block_id = 25 : i32} : index
        %78 = arith.addi %77, %62 {ssbuffer.block_id = 25 : i32} : index
        %79 = arith.addi %78, %76 {ssbuffer.block_id = 25 : i32} : index
        %80 = arith.subi %79, %62 {ssbuffer.block_id = 25 : i32} : index
        %81 = arith.divsi %80, %c1024 {ssbuffer.block_id = 25 : i32} : index
        %82 = arith.subi %41, %81 {ssbuffer.block_id = 25 : i32} : index
        %83 = arith.maxsi %82, %c0 {ssbuffer.block_id = 25 : i32} : index
        %84 = arith.minsi %83, %c64 {ssbuffer.block_id = 25 : i32} : index
        %85 = arith.remsi %80, %c1024 {ssbuffer.block_id = 25 : i32} : index
        %86 = arith.subi %c128, %85 {ssbuffer.block_id = 25 : i32} : index
        %87 = arith.maxsi %86, %c0 {ssbuffer.block_id = 25 : i32} : index
        %88 = arith.minsi %87, %c64 {ssbuffer.block_id = 25 : i32} : index
        %89 = arith.minsi %44, %84 {ssbuffer.block_id = 25 : i32} : index
        %90 = arith.subi %84, %89 {ssbuffer.block_id = 25 : i32} : index
        %91 = arith.subi %c0_i32, %74 {ssbuffer.block_id = 25 : i32} : i32
        %92 = arith.maxsi %91, %c0_i32 {ssbuffer.block_id = 25 : i32} : i32
        %93 = arith.index_cast %92 {ssbuffer.block_id = 25 : i32} : i32 to index
        %94 = arith.minsi %93, %88 {ssbuffer.block_id = 25 : i32} : index
        %95 = arith.subi %88, %94 {ssbuffer.block_id = 25 : i32} : index
        %96 = arith.cmpi slt, %90, %c64 {ssbuffer.block_id = 25 : i32} : index
        %97 = arith.cmpi slt, %95, %c64 {ssbuffer.block_id = 25 : i32} : index
        %98 = arith.ori %96, %97 {ssbuffer.block_id = 25 : i32} : i1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
        %99 = hivm.hir.convert_layout %alloc_9 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32} : (memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<64x64xbf16, #hivm.address_space<cbuf>>
        %memspacecast_27 = memref.memory_space_cast %99 {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xbf16, #hivm.address_space<cbuf>> to memref<64x64xbf16>
        %100 = bufferization.to_tensor %memspacecast_27 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %alloc_28 = memref.alloc() {ssbuffer.block_id = 6 : i32} : memref<64x64xbf16>
        scf.if %98 {
          linalg.fill {ssbuffer.block_id = 6 : i32} ins(%cst_2 : bf16) outs(%alloc_28 : memref<64x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 6 : i32}
        %101 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 6 : i32} : i32
        %102 = arith.maxsi %101, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
        %103 = arith.index_cast %102 {ssbuffer.block_id = 6 : i32} : i32 to index
        %104 = arith.muli %7, %c1024 {ssbuffer.block_id = 6 : i32} : index
        %105 = arith.addi %104, %30 {ssbuffer.block_id = 6 : i32} : index
        %106 = arith.addi %105, %103 {ssbuffer.block_id = 6 : i32} : index
        %reinterpret_cast_29 = memref.reinterpret_cast %arg8 to offset: [%106], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 6 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %107 = arith.subi %106, %30 {ssbuffer.block_id = 6 : i32} : index
        %108 = arith.divsi %107, %c1024 {ssbuffer.block_id = 6 : i32} : index
        %109 = arith.subi %8, %108 {ssbuffer.block_id = 6 : i32} : index
        %110 = arith.maxsi %109, %c0 {ssbuffer.block_id = 6 : i32} : index
        %111 = arith.minsi %110, %c64 {ssbuffer.block_id = 6 : i32} : index
        %112 = arith.remsi %107, %c1024 {ssbuffer.block_id = 6 : i32} : index
        %113 = arith.subi %c128, %112 {ssbuffer.block_id = 6 : i32} : index
        %114 = arith.maxsi %113, %c0 {ssbuffer.block_id = 6 : i32} : index
        %115 = arith.minsi %114, %c64 {ssbuffer.block_id = 6 : i32} : index
        %116 = arith.minsi %13, %111 {ssbuffer.block_id = 6 : i32} : index
        %117 = arith.subi %111, %116 {ssbuffer.block_id = 6 : i32} : index
        %118 = arith.subi %c0_i32, %101 {ssbuffer.block_id = 6 : i32} : i32
        %119 = arith.maxsi %118, %c0_i32 {ssbuffer.block_id = 6 : i32} : i32
        %120 = arith.index_cast %119 {ssbuffer.block_id = 6 : i32} : i32 to index
        %121 = arith.minsi %120, %115 {ssbuffer.block_id = 6 : i32} : index
        %122 = arith.subi %115, %121 {ssbuffer.block_id = 6 : i32} : index
        %subview_30 = memref.subview %reinterpret_cast_29[0, 0] [%117, %122] [1, 1] {ssbuffer.block_id = 6 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %subview_31 = memref.subview %alloc_28[%116, %121] [%117, %122] [1, 1] {ssbuffer.block_id = 6 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        memref.copy %subview_30, %subview_31 {ssbuffer.block_id = 6 : i32} : memref<?x?xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %123 = bufferization.to_tensor %alloc_28 restrict writable {ssbuffer.block_id = 6 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %transposed_32 = linalg.transpose ins(%100 : tensor<64x64xbf16>) outs(%28 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 6 : i32}
        %124 = tensor.empty() {ssbuffer.block_id = 6 : i32} : tensor<64x64xf32>
        %125 = linalg.fill {ssbuffer.block_id = 6 : i32} ins(%cst_1 : f32) outs(%124 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %126 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32} ins(%123, %transposed_32 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%125 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 3 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        hivm.hir.sync_block_wait {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 13 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 14
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 13 : i32} ins(%126 : tensor<64x64xf32>) outs(%alloc_12 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 6 : i32, ssbuffer.transfer_id = 13 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 14
        %127 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32} ins(%transposed, %123 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%32 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 7 : i32} ins(%127 : tensor<64x64xf32>) outs(%alloc_10 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 8
        hivm.hir.sync_block_wait {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 8 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 8 : i32} ins(%127 : tensor<64x64xf32>) outs(%alloc_11 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 8 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 9
      } {ssbuffer.block_id = 38 : i32, ssbuffer.main_loop = 1 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 13 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 14
      hivm.hir.sync_block_wait {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 8 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 9
      hivm.hir.sync_block_wait {ssbuffer.block_id = 38 : i32, ssbuffer.transfer_id = 7 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 8
      hivm.hir.sync_block_wait {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
      %alloc_13 = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      %64 = hivm.hir.convert_layout %alloc_13 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 0 : i32} : (memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<64x64xbf16, #hivm.address_space<cbuf>>
      %memspacecast = memref.memory_space_cast %64 {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x64xbf16, #hivm.address_space<cbuf>> to memref<64x64xbf16>
      %65 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 0 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
      %66 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 14 : i32} ins(%65, %transposed : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%32 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %alloc_14 = memref.alloc() {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 9 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<9>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 9 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 9 : i32} ins(%66 : tensor<64x64xf32>) outs(%alloc_14 : memref<64x64xf32, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 14 : i32, ssbuffer.transfer_id = 9 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 10
      hivm.hir.sync_block_wait {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 4 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 5
      %alloc_15 = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 4 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_15 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 4 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      %67 = hivm.hir.convert_layout %alloc_15 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 4 : i32} : (memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<64x64xbf16, #hivm.address_space<cbuf>>
      %memspacecast_16 = memref.memory_space_cast %67 {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xbf16, #hivm.address_space<cbuf>> to memref<64x64xbf16>
      %68 = bufferization.to_tensor %memspacecast_16 restrict writable {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 4 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
      %69 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 15 : i32} ins(%transposed, %68 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%32 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %alloc_17 = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 14 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<14>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 14 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 14 : i32} ins(%69 : tensor<64x64xf32>) outs(%alloc_17 : memref<64x64xf32, #hivm.address_space<ub>>)
      hivm.hir.sync_block_set {ssbuffer.block_id = 15 : i32, ssbuffer.transfer_id = 14 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = -1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 6 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 7
      %alloc_18 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 6 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<6>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 6 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      %70 = hivm.hir.convert_layout %alloc_18 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 6 : i32} : (memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<64x64xbf16, #hivm.address_space<cbuf>>
      %memspacecast_19 = memref.memory_space_cast %70 {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x64xbf16, #hivm.address_space<cbuf>> to memref<64x64xbf16>
      %71 = bufferization.to_tensor %memspacecast_19 restrict writable {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 6 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
      hivm.hir.sync_block_wait {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 3
      %alloc_20 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_20 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 2 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      %72 = hivm.hir.convert_layout %alloc_20 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 2 : i32} : (memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<64x64xbf16, #hivm.address_space<cbuf>>
      %memspacecast_21 = memref.memory_space_cast %72 {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xbf16, #hivm.address_space<cbuf>> to memref<64x64xbf16>
      %73 = bufferization.to_tensor %memspacecast_21 restrict writable {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 2 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
      %alloc_22 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 5 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      annotation.mark %alloc_22 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 5 : i32} : memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>
      hivm.hir.sync_block_set {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
      %alloc_23 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 11 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_23 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<11>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 11 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_24 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 12 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_24 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<12>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 12 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_25 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 18 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_25 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<18>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 18 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_26 = memref.alloc() {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 19 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_26 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<19>, ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 19 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
      scf.for %arg20 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %74 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 29 : i32} : i32
        %75 = arith.maxsi %74, %c0_i32 {ssbuffer.block_id = 29 : i32} : i32
        %76 = arith.index_cast %75 {ssbuffer.block_id = 29 : i32} : i32 to index
        %77 = arith.muli %40, %c1024 {ssbuffer.block_id = 29 : i32} : index
        %78 = arith.addi %77, %62 {ssbuffer.block_id = 29 : i32} : index
        %79 = arith.addi %78, %76 {ssbuffer.block_id = 29 : i32} : index
        %80 = arith.subi %79, %62 {ssbuffer.block_id = 29 : i32} : index
        %81 = arith.divsi %80, %c1024 {ssbuffer.block_id = 29 : i32} : index
        %82 = arith.subi %41, %81 {ssbuffer.block_id = 29 : i32} : index
        %83 = arith.maxsi %82, %c0 {ssbuffer.block_id = 29 : i32} : index
        %84 = arith.minsi %83, %c64 {ssbuffer.block_id = 29 : i32} : index
        %85 = arith.remsi %80, %c1024 {ssbuffer.block_id = 29 : i32} : index
        %86 = arith.subi %c128, %85 {ssbuffer.block_id = 29 : i32} : index
        %87 = arith.maxsi %86, %c0 {ssbuffer.block_id = 29 : i32} : index
        %88 = arith.minsi %87, %c64 {ssbuffer.block_id = 29 : i32} : index
        %89 = arith.minsi %44, %84 {ssbuffer.block_id = 29 : i32} : index
        %90 = arith.subi %84, %89 {ssbuffer.block_id = 29 : i32} : index
        %91 = arith.subi %c0_i32, %74 {ssbuffer.block_id = 29 : i32} : i32
        %92 = arith.maxsi %91, %c0_i32 {ssbuffer.block_id = 29 : i32} : i32
        %93 = arith.index_cast %92 {ssbuffer.block_id = 29 : i32} : i32 to index
        %94 = arith.minsi %93, %88 {ssbuffer.block_id = 29 : i32} : index
        %95 = arith.subi %88, %94 {ssbuffer.block_id = 29 : i32} : index
        %96 = arith.cmpi slt, %90, %c64 {ssbuffer.block_id = 29 : i32} : index
        %97 = arith.cmpi slt, %95, %c64 {ssbuffer.block_id = 29 : i32} : index
        %98 = arith.ori %96, %97 {ssbuffer.block_id = 29 : i32} : i1
        hivm.hir.sync_block_wait {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 6
        %99 = hivm.hir.convert_layout %alloc_22 output_shape [64, 64] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32} : (memref<4x4x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<64x64xbf16, #hivm.address_space<cbuf>>
        %memspacecast_27 = memref.memory_space_cast %99 {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x64xbf16, #hivm.address_space<cbuf>> to memref<64x64xbf16>
        %100 = bufferization.to_tensor %memspacecast_27 restrict writable {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %transposed_28 = linalg.transpose ins(%100 : tensor<64x64xbf16>) outs(%28 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 12 : i32}
        %101 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32} ins(%transposed_28, %71 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%32 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_set {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 5 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 6
        hivm.hir.sync_block_wait {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 19 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 19 : i32} ins(%101 : tensor<64x64xf32>) outs(%alloc_26 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 19 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = -1
        %alloc_29 = memref.alloc() {ssbuffer.block_id = 10 : i32} : memref<64x64xbf16>
        scf.if %98 {
          linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_2 : bf16) outs(%alloc_29 : memref<64x64xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 10 : i32}
        %102 = arith.muli %arg20, %c64_i32 {ssbuffer.block_id = 10 : i32} : i32
        %103 = arith.maxsi %102, %c0_i32 {ssbuffer.block_id = 10 : i32} : i32
        %104 = arith.index_cast %103 {ssbuffer.block_id = 10 : i32} : i32 to index
        %105 = arith.muli %7, %c1024 {ssbuffer.block_id = 10 : i32} : index
        %106 = arith.addi %105, %30 {ssbuffer.block_id = 10 : i32} : index
        %107 = arith.addi %106, %104 {ssbuffer.block_id = 10 : i32} : index
        %reinterpret_cast_30 = memref.reinterpret_cast %arg2 to offset: [%107], sizes: [64, 64], strides: [1024, 1] {ssbuffer.block_id = 10 : i32} : memref<?xbf16> to memref<64x64xbf16, strided<[1024, 1], offset: ?>>
        %108 = arith.subi %107, %30 {ssbuffer.block_id = 10 : i32} : index
        %109 = arith.divsi %108, %c1024 {ssbuffer.block_id = 10 : i32} : index
        %110 = arith.subi %8, %109 {ssbuffer.block_id = 10 : i32} : index
        %111 = arith.maxsi %110, %c0 {ssbuffer.block_id = 10 : i32} : index
        %112 = arith.minsi %111, %c64 {ssbuffer.block_id = 10 : i32} : index
        %113 = arith.remsi %108, %c1024 {ssbuffer.block_id = 10 : i32} : index
        %114 = arith.subi %c128, %113 {ssbuffer.block_id = 10 : i32} : index
        %115 = arith.maxsi %114, %c0 {ssbuffer.block_id = 10 : i32} : index
        %116 = arith.minsi %115, %c64 {ssbuffer.block_id = 10 : i32} : index
        %117 = arith.minsi %13, %112 {ssbuffer.block_id = 10 : i32} : index
        %118 = arith.subi %112, %117 {ssbuffer.block_id = 10 : i32} : index
        %119 = arith.subi %c0_i32, %102 {ssbuffer.block_id = 10 : i32} : i32
        %120 = arith.maxsi %119, %c0_i32 {ssbuffer.block_id = 10 : i32} : i32
        %121 = arith.index_cast %120 {ssbuffer.block_id = 10 : i32} : i32 to index
        %122 = arith.minsi %121, %116 {ssbuffer.block_id = 10 : i32} : index
        %123 = arith.subi %116, %122 {ssbuffer.block_id = 10 : i32} : index
        %subview_31 = memref.subview %reinterpret_cast_30[0, 0] [%118, %123] [1, 1] {ssbuffer.block_id = 10 : i32} : memref<64x64xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[1024, 1], offset: ?>>
        %subview_32 = memref.subview %alloc_29[%117, %122] [%118, %123] [1, 1] {ssbuffer.block_id = 10 : i32} : memref<64x64xbf16> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        memref.copy %subview_31, %subview_32 {ssbuffer.block_id = 10 : i32} : memref<?x?xbf16, strided<[1024, 1], offset: ?>> to memref<?x?xbf16, strided<[64, 1], offset: ?>>
        %124 = bufferization.to_tensor %alloc_29 restrict writable {ssbuffer.block_id = 10 : i32} : memref<64x64xbf16> to tensor<64x64xbf16>
        %transposed_33 = linalg.transpose ins(%124 : tensor<64x64xbf16>) outs(%28 : tensor<64x64xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 10 : i32}
        %125 = tensor.empty() {ssbuffer.block_id = 10 : i32} : tensor<64x64xf32>
        %126 = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_1 : f32) outs(%125 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %127 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 10 : i32} ins(%124, %transposed_33 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%126 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 18 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 18 : i32} ins(%127 : tensor<64x64xf32>) outs(%alloc_25 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 18 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = -1
        %128 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 11 : i32} ins(%73, %124 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%32 : tensor<64x64xf32>) -> tensor<64x64xf32>
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 11 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 12
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 11 : i32} ins(%128 : tensor<64x64xf32>) outs(%alloc_23 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 11 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 12
        hivm.hir.sync_block_wait {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 12 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 13
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 12 : i32} ins(%128 : tensor<64x64xf32>) outs(%alloc_24 : memref<64x64xf32, #hivm.address_space<ub>>)
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 12 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 13
      } {ssbuffer.block_id = 39 : i32, ssbuffer.main_loop = 2 : i32}
      hivm.hir.sync_block_wait {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 19 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 18 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = -1
      hivm.hir.sync_block_wait {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 12 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 13
      hivm.hir.sync_block_wait {ssbuffer.block_id = 39 : i32, ssbuffer.transfer_id = 11 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 12
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
