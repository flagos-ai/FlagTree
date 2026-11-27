// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync %s | FileCheck %s

// Step 2 of the cube-to-vector-direct-store pipeline: after the VECTOR pseudo-op
// (`arith.addf` carrying `ssbuffer.add_from_matmul`) is folded away in a
// preceding pass, the for-loop now yields a VECTOR `bufferization.to_tensor`
// directly (its source memref traces back to a `hivm.tightly_coupled_buffer<N>`
// `memref.alloc`). The yielded value flows out of the loop, through one or
// more `tensor.extract_slice`s, to a post-region store-like op. This pass
// inserts the CUBE.PIPE_FIX -> VECTOR.PIPE_MTE3 set/wait pair that guards
// the post-region store.
// CHECK-LABEL: func.func @pre_process_bwd_kernel_merged
// CHECK-NOT: ssbuffer.add_from_matmul
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: hivm.hir.sync_block_set {{.*}}[<CUBE>, <PIPE_FIX>, <PIPE_MTE3>]
// CHECK: tensor.extract_slice
// CHECK: hivm.hir.sync_block_wait {{.*}}[<VECTOR>, <PIPE_FIX>, <PIPE_MTE3>]
// CHECK: bufferization.materialize_in_destination
// CHECK: return
// [ReorderOpsByBlockIdPass] Output mlir:
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @pre_process_bwd_kernel_merged(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32}, %arg6: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: memref<?xf32> {tt.divisibility = 16 : i32}, %arg8: f32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c256 = arith.constant {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 256 : index
    %c128 = arith.constant {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 128 : index
    %c64 = arith.constant {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 64 : index
    %c0 = arith.constant {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 0 : index
    %c32768_i32 = arith.constant {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 32768 : i32
    %c0_i32 = arith.constant {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
    %c64_i32 = arith.constant {MixUse, ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 64 : i32
    %c128_i32 = arith.constant {MixUse, ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} 128 : i32
    %0 = arith.muli %arg13, %c64_i32 {Undefined, ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %1 = arith.muli %arg14, %c32768_i32 {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %2 = arith.cmpi sge, %0, %c128_i32 {Undefined, ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %c63_i32 = arith.constant {MixUse, ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"} 63 : i32
    %c1_i32 = arith.constant {MixUse, ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32
    %3 = arith.addi %arg9, %c63_i32 {MixUse, ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %4 = arith.divsi %3, %c64_i32 {MixUse, ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"} : i32
    %c1024 = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 1024 : index
    %c128_0 = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 128 : index
    %c64_1 = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 64 : index
    %c0_2 = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 0 : index
    %c512 = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 512 : index
    %c63_i32_3 = arith.constant {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 63 : i32
    %c1_i32_4 = arith.constant {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 1 : i32
    %cst = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 0.000000e+00 : f32
    %c0_i32_5 = arith.constant {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %c64_i32_6 = arith.constant {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 64 : i32
    %c2_i32 = arith.constant {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 2 : i32
    %c128_i64 = arith.constant {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} 128 : i64
    %5 = tensor.empty() {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
    %6 = tensor.empty() {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
    %7 = linalg.fill {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %8 = arith.addi %arg9, %c63_i32_3 {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32
    %9 = arith.divsi %8, %c64_i32_6 {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32
    %10 = arith.divsi %arg14, %c2_i32 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32
    %11 = arith.extsi %10 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32 to i64
    %12 = arith.muli %11, %c128_i64 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i64
    %13 = arith.index_cast %12 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i64 to index
    %14 = arith.extsi %arg14 {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i32 to i64
    %15 = arith.muli %14, %c128_i64 {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i64
    %16 = arith.index_cast %15 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "CUBE"} : i64 to index
    %cst_7 = arith.constant {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %cst_8 = arith.constant {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %c2_i32_9 = arith.constant {MixUse, ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} 2 : i32
    %17 = tensor.empty() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
    %18 = linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_8 : f32) outs(%17 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %19 = linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_7 : f32) outs(%17 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %20 = tensor.empty() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
    %21 = linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_8 : f32) outs(%20 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %22 = linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_7 : f32) outs(%20 : tensor<128x128xf32>) -> tensor<128x128xf32>
    scf.if %2 {
      %23 = arith.subi %arg13, %c2_i32_9 {MixUse, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %24 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128xi32>
      %25 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%24 : tensor<128xi32>) attrs =  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR", tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 128 : index} {
      ^bb0(%out: i32):
        %59 = linalg.index 0 : index
        %60 = arith.index_cast %59 : index to i32
        linalg.yield %60 : i32
      } -> tensor<128xi32>
      %26 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xi32>
      %27 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%26 : tensor<64xi32>) attrs =  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR", tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 64 : index} {
      ^bb0(%out: i32):
        %59 = linalg.index 0 : index
        %60 = arith.index_cast %59 : index to i32
        linalg.yield %60 : i32
      } -> tensor<64xi32>
      %28 = arith.muli %23, %c64_i32 {MixUse, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %29 = linalg.fill {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} ins(%28 : i32) outs(%26 : tensor<64xi32>) -> tensor<64xi32>
      %30 = arith.addi %27, %29 {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xi32>
      %31 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x64xi32>
      %broadcasted = linalg.broadcast ins(%25 : tensor<128xi32>) outs(%31 : tensor<128x64xi32>) dimensions = [1]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
      %broadcasted_10 = linalg.broadcast ins(%30 : tensor<64xi32>) outs(%31 : tensor<128x64xi32>) dimensions = [0]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
      %32 = arith.cmpi eq, %broadcasted, %broadcasted_10 {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x64xi32>
      %33 = arith.select %32, %18, %19 {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x64xi1>, tensor<128x64xf32>
      %34 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi32>
      %broadcasted_11 = linalg.broadcast ins(%25 : tensor<128xi32>) outs(%34 : tensor<128x128xi32>) dimensions = [1]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
      %broadcasted_12 = linalg.broadcast ins(%25 : tensor<128xi32>) outs(%34 : tensor<128x128xi32>) dimensions = [0]  {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"}
      %35 = arith.cmpi eq, %broadcasted_11, %broadcasted_12 {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi32>
      %36 = arith.select %35, %21, %22 {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xi1>, tensor<128x128xf32>
      %37 = arith.subi %9, %c1_i32_4 {MixUse, ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : i32
      %38 = scf.for %arg16 = %c0_i32_5 to %9 step %c1_i32_4 iter_args(%arg17 = %33) -> (tensor<128x64xf32>)  : i32 {
        %59 = arith.subi %37, %arg16 {MixUse, ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32
        %60 = arith.muli %59, %c64_i32_6 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32
        %61 = arith.maxsi %60, %c0_i32_5 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32
        %62 = arith.index_cast %61 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %63 = arith.muli %62, %c512 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %64 = arith.index_cast %arg9 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %65 = arith.divsi %63, %c512 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %66 = arith.subi %64, %65 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %67 = arith.maxsi %66, %c0_2 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %68 = arith.minsi %67, %c64_1 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %69 = arith.remsi %63, %c512 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %70 = arith.subi %c128_0, %69 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %71 = arith.maxsi %70, %c0_2 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %72 = arith.minsi %71, %c128_0 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %73 = arith.subi %c0_i32_5, %60 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32
        %74 = arith.maxsi %73, %c0_i32_5 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32
        %75 = arith.index_cast %74 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i32 to index
        %76 = arith.minsi %75, %68 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %77 = arith.subi %68, %76 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %78 = arith.minsi %72, %c0_2 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %79 = arith.subi %72, %78 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %80 = arith.cmpi slt, %77, %c64_1 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %81 = arith.cmpi slt, %79, %c128_0 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %82 = arith.ori %80, %81 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i1
        %83 = arith.muli %62, %c1024 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %84 = arith.divsi %83, %c1024 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %85 = arith.subi %64, %84 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %86 = arith.maxsi %85, %c0_2 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %87 = arith.minsi %86, %c64_1 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %88 = arith.remsi %83, %c1024 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %89 = arith.subi %c128_0, %88 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %90 = arith.maxsi %89, %c0_2 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %91 = arith.minsi %90, %c128_0 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %92 = arith.minsi %75, %87 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %93 = arith.subi %87, %92 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %94 = arith.minsi %91, %c0_2 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %95 = arith.subi %91, %94 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %96 = arith.cmpi slt, %93, %c64_1 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %97 = arith.cmpi slt, %95, %c128_0 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : index
        %98 = arith.ori %96, %97 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : i1
        %alloc = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32>
        %alloc_13 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32>
        scf.if %98 {
          linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%alloc_13 : memref<64x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        scf.if %82 {
          linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%alloc : memref<64x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        %99 = arith.addi %63, %13 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_14 = memref.reinterpret_cast %arg3 to offset: [%99], sizes: [64, 128], strides: [512, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf32> to memref<64x128xf32, strided<[512, 1], offset: ?>>
        %subview_15 = memref.subview %reinterpret_cast_14[0, 0] [%77, %79] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %subview_16 = memref.subview %alloc[%76, %78] [%77, %79] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32> to memref<?x?xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_15, %subview_16 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xf32, strided<[512, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
        %100 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32> to tensor<64x128xf32>
        %101 = arith.addi %83, %16 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : index
        %reinterpret_cast_17 = memref.reinterpret_cast %arg4 to offset: [%101], sizes: [64, 128], strides: [1024, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?xf32> to memref<64x128xf32, strided<[1024, 1], offset: ?>>
        %subview_18 = memref.subview %reinterpret_cast_17[0, 0] [%93, %95] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32, strided<[1024, 1], offset: ?>> to memref<?x?xf32, strided<[1024, 1], offset: ?>>
        %subview_19 = memref.subview %alloc_13[%92, %94] [%93, %95] [1, 1] {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32> to memref<?x?xf32, strided<[128, 1], offset: ?>>
        memref.copy %subview_18, %subview_19 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<?x?xf32, strided<[1024, 1], offset: ?>> to memref<?x?xf32, strided<[128, 1], offset: ?>>
        %102 = bufferization.to_tensor %alloc_13 restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : memref<64x128xf32> to tensor<64x128xf32>
        %transposed = linalg.transpose ins(%102 : tensor<64x128xf32>) outs(%5 : tensor<128x64xf32>) permutation = [1, 0]  {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"}
        %103 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%transposed, %100 : tensor<128x64xf32>, tensor<64x128xf32>) outs(%7 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %104 = arith.subf %36, %103 {DataUse, ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %105 = tensor.empty() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x64xf32>
        %106 = linalg.fill {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : f32) outs(%105 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %107 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE", ssbuffer.loop_carried_l0c} ins(%104, %arg17 : tensor<128x128xf32>, tensor<128x64xf32>) outs(%106 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %108 = arith.addf %107, %19 {ssbuffer.add_from_matmul, ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32>
        scf.yield {ssbuffer.core_type = "VECTOR"} %108 : tensor<128x64xf32>
      } {DataUse, ssbuffer.core_type = "VECTOR"}
      %39 = arith.index_cast %1 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %40 = arith.addi %39, %c128 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %41 = arith.maxsi %28, %c0_i32 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %42 = arith.index_cast %41 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %43 = arith.addi %40, %42 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%43], sizes: [128, 64], strides: [256, 1] {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : memref<?xf32> to memref<128x64xf32, strided<[256, 1], offset: ?>>
      %44 = arith.divsi %42, %c256 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %45 = arith.subi %c128, %44 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %46 = arith.maxsi %45, %c0 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %47 = arith.minsi %46, %c128 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %48 = arith.remsi %42, %c256 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %49 = arith.subi %c128, %48 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %50 = arith.maxsi %49, %c0 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %51 = arith.minsi %50, %c64 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %52 = arith.minsi %47, %c0 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %53 = arith.subi %47, %52 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %54 = arith.subi %c0_i32, %28 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %55 = arith.maxsi %54, %c0_i32 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : i32
      %56 = arith.index_cast %55 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index
      %57 = arith.minsi %56, %51 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %58 = arith.subi %51, %57 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : index
      %extracted_slice = tensor.extract_slice %38[%52, %57] [%53, %58] [1, 1] {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x64xf32> to tensor<?x?xf32>
      %subview = memref.subview %reinterpret_cast[0, 0] [%53, %58] [1, 1] {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x64xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<?x?xf32>, memref<?x?xf32, strided<[256, 1], offset: ?>>) -> ()
    } {Undefined}
    return
  }
}
