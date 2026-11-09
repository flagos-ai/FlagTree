// RUN: triton-opt --clone-ops %s --allow-unregistered-dialect | FileCheck %s

// Unit test for CloneOps pass: cloning ops with SSA dependency chains.
//
// Block 10 holds a chain of i32 ops:
//   %23 = arith.cmpi eq, %arg24, %20
//   %24 = arith.addi %arg24, %c1_i32
//   %25 = arith.select %23, %c0_i32, %24
//   %28 = arith.cmpi eq, %25, %20
// Block 12 uses %28 via:
//   %54 = arith.select %28, %2, %51
//
// After --clone-ops, the chain in block 10 must be cloned into block 12 and the
// cloned %28 must reference the cloned %25 (not the original %25). Otherwise
// the precursor clones (%23, %24, %25) become dead and are erased by the
// cleanup pass. Each cloned op must keep the `clone = 10` attribute and its
// operand must be replaced with the cloned value.

// CHECK-LABEL: func.func @_kernel_matmul_fp8_row_no_fast_acc
// Verify the full cloned SSA chain in block 12 (%23, %24, %25, %28 -> %23', %24',
// %25', %28'). The cloned cmpi (clone of %28) must read the cloned select
// (clone of %25), not the original %25.
// CHECK: arith.cmpi eq, {{.*}}, {{.*}} {MixUse, ssbuffer.block_id = 12 : i32, ssbuffer.clone = 10 : i32} : i32
// CHECK-NEXT: arith.addi {{.*}} {MixUse, ssbuffer.block_id = 12 : i32, ssbuffer.clone = 10 : i32} : i32
// CHECK-NEXT: arith.select {{.*}} {MixUse, ssbuffer.block_id = 12 : i32, ssbuffer.clone = 10 : i32} : i32
// CHECK-NEXT: arith.cmpi eq, {{.*}}, {{.*}} {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.clone = 10 : i32, ssbuffer.dep_mark = [3 : i32]} : i32
// The block-12 consumer (clone of %54) must reference the cloned cmpi.
// CHECK: arith.select {{.*}} {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [3 : i32]} : tensor<32x32xf32>
// Also verify the same chain is cloned into block 9 (the last block).
// CHECK: arith.cmpi eq, {{.*}}, {{.*}} {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.clone = 10 : i32} : i32
// CHECK-NEXT: arith.addi {{.*}} {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.clone = 10 : i32} : i32
// CHECK-NEXT: arith.select {{.*}} {MixUse, ssbuffer.block_id = 9 : i32, ssbuffer.clone = 10 : i32} : i32
// CHECK-NEXT: arith.cmpi eq, {{.*}}, {{.*}} {DataUse, ssbuffer.block_id = 9 : i32, ssbuffer.clone = 10 : i32, ssbuffer.dep_mark = [3 : i32]} : i32

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func private @triton_indirect_load(memref<?xf32>, tensor<32x32xi64>, tensor<32x32xi1>, tensor<32x32xf32>) -> tensor<32x32xf32>
  func.func @_kernel_matmul_fp8_row_no_fast_acc(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg12: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg13: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "mix_simd_simt"} {
    %c32_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 32 : i32
    %c0_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 0 : i32
    %c-1_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} -1 : i32
    %c1_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 1 : i32
    %c80_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 80 : i32
    %c1 = arith.constant {Undefined, ssbuffer.block_id = 13 : i32} 1 : index
    %c0 = arith.constant {DataUse, ssbuffer.block_id = 13 : i32} 0 : index
    %c32 = arith.constant {DataUse, ssbuffer.block_id = 13 : i32} 32 : index
    %cst = arith.constant {ssbuffer.block_id = 13 : i32} 0.000000e+00 : f32
    %c31_i32 = arith.constant {MixUse, ssbuffer.block_id = 13 : i32} 31 : i32
    %c4_i32 = arith.constant {MixUse, ssbuffer.block_id = 14 : i32} 4 : i32
    %cst_0 = arith.constant {ssbuffer.block_id = 11 : i32} dense<[4, 2, 16, 8]> : tensor<4xi64>
    %cst_1 = arith.constant {ssbuffer.block_id = 11 : i32} dense<[32, 4, 8]> : tensor<3xi64>
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<32xi32>
      %1 = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<32x32xf32>
      %2 = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %3 = arith.addi %arg5, %c31_i32 {MixUse, ssbuffer.block_id = 13 : i32} : i32
      %4 = arith.divsi %3, %c32_i32 {MixUse, ssbuffer.block_id = 13 : i32} : i32
      %5 = arith.addi %arg6, %c31_i32 {MixUse, ssbuffer.block_id = 13 : i32} : i32
      %6 = arith.divsi %5, %c32_i32 {MixUse, ssbuffer.block_id = 13 : i32} : i32
      %7 = arith.muli %4, %6 {Undefined, ssbuffer.block_id = 13 : i32} : i32
      %8 = arith.divsi %7, %c80_i32 {Undefined, ssbuffer.block_id = 13 : i32} : i32
      %9 = arith.remsi %7, %c80_i32 {Undefined, ssbuffer.block_id = 13 : i32} : i32
      %10 = arith.cmpi slt, %arg20, %9 {Undefined, ssbuffer.block_id = 13 : i32} : i32
      %11 = arith.subi %arg20, %c80_i32 {MixUse, ssbuffer.block_id = 13 : i32} : i32
      %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%0 : tensor<32xi32>) attrs =  {ssbuffer.block_id = 13 : i32, tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 32 : index} {
      ^bb0(%out: i32):
        %23 = linalg.index 0 : index
        %24 = arith.index_cast %23 : index to i32
        linalg.yield %24 : i32
      } -> tensor<32xi32>
      %13 = scf.if %10 -> (i32) {
        %23 = arith.addi %8, %c1_i32 {Undefined, ssbuffer.block_id = 13 : i32} : i32
        scf.yield {Undefined} %23 : i32
      } else {
        scf.yield {Undefined} %8 : i32
      } {Undefined, ssbuffer.block_id = 13 : i32}
      %14 = tensor.empty() {ssbuffer.block_id = 15 : i32} : tensor<1x32xi32>
      %15 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%arg15 : i32) outs(%14 : tensor<1x32xi32>) -> tensor<1x32xi32>
      %expanded = tensor.expand_shape %12 [[0, 1]] output_shape [32, 1] {ssbuffer.block_id = 15 : i32} : tensor<32xi32> into tensor<32x1xi32>
      %16 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%c0_i32 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
      %17 = arith.muli %6, %c4_i32 {MixUse, ssbuffer.block_id = 14 : i32} : i32
      %18 = arith.addi %arg7, %c31_i32 {MixUse, ssbuffer.block_id = 16 : i32} : i32
      %19 = arith.divsi %18, %c32_i32 {MixUse, ssbuffer.block_id = 16 : i32} : i32
      %20 = arith.subi %19, %c1_i32 {MixUse, ssbuffer.block_id = 16 : i32} : i32
      %21 = arith.muli %19, %13 {Undefined, ssbuffer.block_id = 17 : i32} : i32
      %alloc = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      %alloc_2 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      %alloc_3 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.crossDeps = [2 : i32, 1 : i32], ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>>
      hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
      %alloc_4 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
      %memspacecast = memref.memory_space_cast %alloc_4 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
      %alloc_5 = memref.alloc() : memref<32x32xf32, #hivm.address_space<ub>>
      %memspacecast_6 = memref.memory_space_cast %alloc_5 {ssbuffer.intraDeps = [0 : i32, 1 : i32]} : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
      %alloc_7 = memref.alloc() : memref<32xi32, #hivm.address_space<ub>>
      %memspacecast_8 = memref.memory_space_cast %alloc_7 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<32xi32, #hivm.address_space<ub>> to memref<32xi32>
      %alloc_9 = memref.alloc() : memref<32xi32, #hivm.address_space<ub>>
      %memspacecast_10 = memref.memory_space_cast %alloc_9 {ssbuffer.intraDeps = [1 : i32, 1 : i32]} : memref<32xi32, #hivm.address_space<ub>> to memref<32xi32>
      %22:7 = scf.for %arg23 = %c0_i32 to %21 step %c1_i32 iter_args(%arg24 = %c-1_i32, %arg25 = %11, %arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %12, %arg29 = %12, %arg30 = %2) -> (i32, i32, i32, i32, tensor<32xi32>, tensor<32xi32>, tensor<32x32xf32>)  : i32 {
        %23 = arith.cmpi eq, %arg24, %20 {MixUse, ssbuffer.block_id = 10 : i32} : i32
        %24 = arith.addi %arg24, %c1_i32 {MixUse, ssbuffer.block_id = 10 : i32} : i32
        %25 = arith.select %23, %c0_i32, %24 {MixUse, ssbuffer.block_id = 10 : i32} : i32
        %26 = arith.cmpi eq, %25, %c0_i32 {Undefined, ssbuffer.block_id = 10 : i32, ssbuffer.dep_mark = [1 : i32]} : i32
        %27 = arith.muli %25, %c32_i32 {MixUse, ssbuffer.block_id = 10 : i32, ssbuffer.dep_mark = [2 : i32]} : i32
        %28 = arith.cmpi eq, %25, %20 {DataUse, ssbuffer.block_id = 10 : i32, ssbuffer.dep_mark = [3 : i32]} : i32
        %29:5 = scf.if %26 -> (i32, i32, i32, tensor<32xi32>, tensor<32xi32>) {
          %65 = arith.addi %arg25, %c80_i32 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %66 = arith.divsi %65, %17 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %67 = arith.muli %66, %c4_i32 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %68 = arith.subi %4, %67 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %69 = arith.minsi %68, %c4_i32 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %70 = arith.remsi %65, %69 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %71 = arith.addi %67, %70 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %72 = arith.remsi %65, %17 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %73 = arith.divsi %72, %69 {MixUse, ssbuffer.block_id = 3 : i32} : i32
          %74 = arith.muli %71, %c32_i32 {DataUse, ssbuffer.block_id = 3 : i32} : i32
          %75 = arith.muli %73, %c32_i32 {DataUse, ssbuffer.block_id = 3 : i32} : i32
          %76 = linalg.fill {ssbuffer.block_id = 3 : i32} ins(%74 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
          %77 = arith.addi %76, %12 {DataUse, ssbuffer.block_id = 3 : i32} : tensor<32xi32>
          %78 = linalg.fill {ssbuffer.block_id = 3 : i32} ins(%75 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
          %79 = arith.addi %78, %12 {DataUse, ssbuffer.block_id = 3 : i32} : tensor<32xi32>
          %80 = arith.index_cast %74 {DataUse, ssbuffer.block_id = 3 : i32} : i32 to index
          %81 = arith.addi %80, %c32 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %82 = arith.index_cast %arg5 {DataUse, ssbuffer.block_id = 3 : i32} : i32 to index
          %83 = arith.maxsi %80, %82 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %84 = arith.minsi %81, %83 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %85 = arith.subi %84, %80 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %extracted_slice_24 = tensor.extract_slice %77[0] [%85] [1] {DataUse, ssbuffer.block_id = 3 : i32} : tensor<32xi32> to tensor<?xi32>
          %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %16[0] [%85] [1] {DataUse, ssbuffer.block_id = 3 : i32} : tensor<?xi32> into tensor<32xi32>
          %86 = arith.index_cast %75 {DataUse, ssbuffer.block_id = 3 : i32} : i32 to index
          %87 = arith.addi %86, %c32 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %88 = arith.index_cast %arg6 {DataUse, ssbuffer.block_id = 3 : i32} : i32 to index
          %89 = arith.maxsi %86, %88 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %90 = arith.minsi %87, %89 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %91 = arith.subi %90, %86 {DataUse, ssbuffer.block_id = 3 : i32} : index
          %extracted_slice_26 = tensor.extract_slice %79[0] [%91] [1] {DataUse, ssbuffer.block_id = 3 : i32} : tensor<32xi32> to tensor<?xi32>
          %inserted_slice_27 = tensor.insert_slice %extracted_slice_26 into %16[0] [%91] [1] {DataUse, ssbuffer.block_id = 3 : i32} : tensor<?xi32> into tensor<32xi32>
          scf.yield {Undefined} %65, %71, %73, %inserted_slice_25, %inserted_slice_27 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
        } else {
          scf.yield {Undefined} %arg25, %arg26, %arg27, %arg28, %arg29 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
        } {MixUse, ssbuffer.block_id = 3 : i32, ssbuffer.dep_mark = [1 : i32]}
        %c2_i32 = arith.constant {ssbuffer.block_id = 3 : i32} 2 : i32
        %30 = arith.remsi %arg23, %c2_i32 {ssbuffer.block_id = 3 : i32} : i32
        %c0_i32_11 = arith.constant {ssbuffer.block_id = 3 : i32} 0 : i32
        %31 = arith.cmpi eq, %30, %c0_i32_11 {ssbuffer.block_id = 3 : i32} : i32
        scf.if %31 {
          hivm.hir.copy ins(%29#3 : tensor<32xi32>) outs(%memspacecast_8 : memref<32xi32>) {ssbuffer.block_id = 3 : i32}
        } else {
          hivm.hir.copy ins(%29#3 : tensor<32xi32>) outs(%memspacecast_10 : memref<32xi32>) {ssbuffer.block_id = 3 : i32}
        } {ssbuffer.block_id = 3 : i32, ssbuffer.intra_buffer}
        %alloc_12 = memref.alloc() {ssbuffer.block_id = 11 : i32} : memref<32x32xf32>
        scf.for %arg31 = %c0 to %c32 step %c1 {
          %c2_i32_24 = arith.constant {ssbuffer.block_id = 11 : i32} 2 : i32
          %65 = arith.remsi %arg23, %c2_i32_24 {ssbuffer.block_id = 11 : i32} : i32
          %c0_i32_25 = arith.constant {ssbuffer.block_id = 11 : i32} 0 : i32
          %66 = arith.cmpi eq, %65, %c0_i32_25 {ssbuffer.block_id = 11 : i32} : i32
          %67 = scf.if %66 -> (tensor<32xi32>) {
            %72 = bufferization.to_tensor %memspacecast_8 restrict writable : memref<32xi32> to tensor<32xi32>
            scf.yield %72 : tensor<32xi32>
          } else {
            %72 = bufferization.to_tensor %memspacecast_10 restrict writable : memref<32xi32> to tensor<32xi32>
            scf.yield %72 : tensor<32xi32>
          } {ssbuffer.block_id = 11 : i32, ssbuffer.intraDeps = [1 : i32, 0 : i32], ssbuffer.intra_buffer}
          %extracted = tensor.extract %67[%arg31] {DiscreteMemAccess, ssbuffer.block_id = 11 : i32} : tensor<32xi32>
          %68 = arith.muli %extracted, %arg14 {ssbuffer.block_id = 11 : i32} : i32
          %69 = arith.index_cast %68 {ssbuffer.block_id = 11 : i32} : i32 to index
          %70 = arith.index_cast %27 {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [2 : i32]} : i32 to index
          %71 = arith.addi %69, %70 {ssbuffer.block_id = 11 : i32} : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%71], sizes: [1, 32], strides: [32, 1] {ssbuffer.block_id = 11 : i32} : memref<?xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
          %subview = memref.subview %alloc_12[%arg31, 0] [1, 32] [1, 1] {ssbuffer.block_id = 11 : i32} : memref<32x32xf32> to memref<1x32xf32, strided<[32, 1], offset: ?>>
          memref.copy %reinterpret_cast, %subview {ssbuffer.block_id = 11 : i32} : memref<1x32xf32, strided<[32, 1], offset: ?>> to memref<1x32xf32, strided<[32, 1], offset: ?>>
        } {DataUse, ExtractedLoadOrStore, hivm.parallel_loop, ssbuffer.block_id = 11 : i32}
        %32 = linalg.fill {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [2 : i32]} ins(%27 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
        %33 = arith.addi %32, %12 {MixUse, ssbuffer.block_id = 11 : i32} : tensor<32xi32>
        %expanded_13 = tensor.expand_shape %29#4 [[0, 1]] output_shape [1, 32] {ssbuffer.block_id = 11 : i32} : tensor<32xi32> into tensor<1x32xi32>
        %34 = arith.muli %expanded_13, %15 {DataUse, ssbuffer.block_id = 11 : i32} : tensor<1x32xi32>
        %35 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<32x32xi32>
        %broadcasted = linalg.broadcast ins(%33 : tensor<32xi32>) outs(%35 : tensor<32x32xi32>) dimensions = [1]  {ssbuffer.block_id = 11 : i32}
        %collapsed = tensor.collapse_shape %34 [[0, 1]] {ssbuffer.block_id = 11 : i32} : tensor<1x32xi32> into tensor<32xi32>
        %broadcasted_14 = linalg.broadcast ins(%collapsed : tensor<32xi32>) outs(%35 : tensor<32x32xi32>) dimensions = [0]  {ssbuffer.block_id = 11 : i32}
        %36 = arith.addi %broadcasted, %broadcasted_14 {DataUse, ssbuffer.block_id = 11 : i32} : tensor<32x32xi32>
        %37 = arith.extsi %36 {DataUse, ssbuffer.block_id = 11 : i32} : tensor<32x32xi32> to tensor<32x32xi64>
        %38 = arith.subi %arg7, %27 {DataUse, ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [2 : i32]} : i32
        %39 = bufferization.to_tensor %alloc_12 restrict writable {ssbuffer.block_id = 11 : i32} : memref<32x32xf32> to tensor<32x32xf32>
        %40 = arith.index_cast %38 {DataUse, ssbuffer.block_id = 11 : i32} : i32 to index
        %41 = arith.maxsi %40, %c0 {DataUse, ssbuffer.block_id = 11 : i32} : index
        %42 = arith.minsi %41, %c32 {DataUse, ssbuffer.block_id = 11 : i32} : index
        %extracted_slice = tensor.extract_slice %39[0, 0] [32, %42] [1, 1] {DataUse, ssbuffer.block_id = 11 : i32} : tensor<32x32xf32> to tensor<32x?xf32>
        %inserted_slice = tensor.insert_slice %extracted_slice into %2[0, 0] [32, %42] [1, 1] {DataUse, ssbuffer.block_id = 11 : i32} : tensor<32x?xf32> into tensor<32x32xf32>
        %43 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<32x1xi32>
        %44 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%38 : i32) outs(%43 : tensor<32x1xi32>) -> tensor<32x1xi32>
        %45 = arith.cmpi slt, %expanded, %44 {DataUse, ssbuffer.block_id = 11 : i32} : tensor<32x1xi32>
        %46 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<32x32xi1>
        %collapsed_15 = tensor.collapse_shape %45 [[0, 1]] {ssbuffer.block_id = 11 : i32} : tensor<32x1xi1> into tensor<32xi1>
        %broadcasted_16 = linalg.broadcast ins(%collapsed_15 : tensor<32xi1>) outs(%46 : tensor<32x32xi1>) dimensions = [1]  {ssbuffer.block_id = 11 : i32}
        %47 = func.call @triton_indirect_load(%arg3, %37, %broadcasted_16, %2) {ssbuffer.block_id = 11 : i32} : (memref<?xf32>, tensor<32x32xi64>, tensor<32x32xi1>, tensor<32x32xf32>) -> tensor<32x32xf32>
        %reshape = tensor.reshape %inserted_slice(%cst_1) {ssbuffer.block_id = 11 : i32} : (tensor<32x32xf32>, tensor<3xi64>) -> tensor<32x4x8xf32>
        %48 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<4x32x8xf32>
        %transposed = linalg.transpose ins(%reshape : tensor<32x4x8xf32>) outs(%48 : tensor<4x32x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 11 : i32}
        %reshape_17 = tensor.reshape %transposed(%cst_0) {ssbuffer.block_id = 11 : i32} : (tensor<4x32x8xf32>, tensor<4xi64>) -> tensor<4x2x16x8xf32>
        %reshape_18 = tensor.reshape %47(%cst_1) {ssbuffer.block_id = 11 : i32} : (tensor<32x32xf32>, tensor<3xi64>) -> tensor<32x4x8xf32>
        %49 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<4x32x8xf32>
        %transposed_19 = linalg.transpose ins(%reshape_18 : tensor<32x4x8xf32>) outs(%49 : tensor<4x32x8xf32>) permutation = [1, 0, 2]  {ssbuffer.block_id = 11 : i32}
        %reshape_20 = tensor.reshape %transposed_19(%cst_0) {ssbuffer.block_id = 11 : i32} : (tensor<4x32x8xf32>, tensor<4xi64>) -> tensor<4x2x16x8xf32>
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.copy ins(%reshape_17 : tensor<4x2x16x8xf32>) outs(%alloc : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
        hivm.hir.copy ins(%reshape_20 : tensor<4x2x16x8xf32>) outs(%alloc_2 : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>) {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3
        %memspacecast_21 = memref.memory_space_cast %alloc_3 {ssbuffer.block_id = 12 : i32, ssbuffer.crossDeps = [2 : i32, 0 : i32], ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32, #hivm.address_space<ub>> to memref<32x32xf32>
        %50 = bufferization.to_tensor %memspacecast_21 restrict writable {ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32} : memref<32x32xf32> to tensor<32x32xf32>
        %51 = arith.addf %50, %arg30 {ssbuffer.add_from_matmul, ssbuffer.block_id = 12 : i32} : tensor<32x32xf32>
        %c2_i32_22 = arith.constant {ssbuffer.block_id = 12 : i32} 2 : i32
        %52 = arith.remsi %arg23, %c2_i32_22 {ssbuffer.block_id = 12 : i32} : i32
        %c0_i32_23 = arith.constant {ssbuffer.block_id = 12 : i32} 0 : i32
        %53 = arith.cmpi eq, %52, %c0_i32_23 {ssbuffer.block_id = 12 : i32} : i32
        scf.if %53 {
          hivm.hir.copy ins(%51 : tensor<32x32xf32>) outs(%memspacecast : memref<32x32xf32>) {ssbuffer.block_id = 12 : i32}
        } else {
          hivm.hir.copy ins(%51 : tensor<32x32xf32>) outs(%memspacecast_6 : memref<32x32xf32>) {ssbuffer.block_id = 12 : i32}
        } {ssbuffer.block_id = 12 : i32, ssbuffer.intra_buffer}
        %54 = arith.select %28, %2, %51 {DataUse, ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [3 : i32]} : tensor<32x32xf32>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 12 : i32, ssbuffer.transfer_id = 2 : i32}[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
        %55 = arith.cmpi eq, %arg24, %20 {MixUse, ssbuffer.block_id = 9 : i32} : i32
        %56 = arith.addi %arg24, %c1_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
        %57 = arith.select %55, %c0_i32, %56 {MixUse, ssbuffer.block_id = 9 : i32} : i32
        %58 = arith.cmpi eq, %57, %c0_i32 {Undefined, ssbuffer.block_id = 9 : i32} : i32
        %59:5 = scf.if %58 -> (i32, i32, i32, tensor<32xi32>, tensor<32xi32>) {
          %65 = arith.addi %arg25, %c80_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %66 = arith.divsi %65, %17 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %67 = arith.muli %66, %c4_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %68 = arith.subi %4, %67 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %69 = arith.minsi %68, %c4_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %70 = arith.remsi %65, %69 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %71 = arith.addi %67, %70 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %72 = arith.remsi %65, %17 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %73 = arith.divsi %72, %69 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %74 = arith.muli %71, %c32_i32 {DataUse, ssbuffer.block_id = 9 : i32} : i32
          %75 = arith.muli %73, %c32_i32 {DataUse, ssbuffer.block_id = 9 : i32} : i32
          %76 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%74 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
          %77 = arith.addi %76, %12 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32>
          %78 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%75 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
          %79 = arith.addi %78, %12 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32>
          %80 = arith.index_cast %74 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %81 = arith.addi %80, %c32 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %82 = arith.index_cast %arg5 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %83 = arith.maxsi %80, %82 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %84 = arith.minsi %81, %83 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %85 = arith.subi %84, %80 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %extracted_slice_24 = tensor.extract_slice %77[0] [%85] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32> to tensor<?xi32>
          %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %16[0] [%85] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<?xi32> into tensor<32xi32>
          %86 = arith.index_cast %75 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %87 = arith.addi %86, %c32 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %88 = arith.index_cast %arg6 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %89 = arith.maxsi %86, %88 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %90 = arith.minsi %87, %89 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %91 = arith.subi %90, %86 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %extracted_slice_26 = tensor.extract_slice %79[0] [%91] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32> to tensor<?xi32>
          %inserted_slice_27 = tensor.insert_slice %extracted_slice_26 into %16[0] [%91] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<?xi32> into tensor<32xi32>
          scf.yield {Undefined, ssbuffer.block_id = 9 : i32} %65, %71, %73, %inserted_slice_25, %inserted_slice_27 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
        } else {
          scf.yield {Undefined, ssbuffer.block_id = 9 : i32} %arg25, %arg26, %arg27, %arg28, %arg29 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
        } {MixUse, ssbuffer.block_id = 9 : i32}
        %60 = arith.cmpi eq, %arg24, %20 {MixUse, ssbuffer.block_id = 9 : i32} : i32
        %61 = arith.addi %arg24, %c1_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
        %62 = arith.select %60, %c0_i32, %61 {MixUse, ssbuffer.block_id = 9 : i32} : i32
        %63 = arith.cmpi eq, %62, %c0_i32 {Undefined, ssbuffer.block_id = 9 : i32} : i32
        %64:5 = scf.if %63 -> (i32, i32, i32, tensor<32xi32>, tensor<32xi32>) {
          %65 = arith.addi %arg25, %c80_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %66 = arith.divsi %65, %17 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %67 = arith.muli %66, %c4_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %68 = arith.subi %4, %67 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %69 = arith.minsi %68, %c4_i32 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %70 = arith.remsi %65, %69 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %71 = arith.addi %67, %70 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %72 = arith.remsi %65, %17 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %73 = arith.divsi %72, %69 {MixUse, ssbuffer.block_id = 9 : i32} : i32
          %74 = arith.muli %71, %c32_i32 {DataUse, ssbuffer.block_id = 9 : i32} : i32
          %75 = arith.muli %73, %c32_i32 {DataUse, ssbuffer.block_id = 9 : i32} : i32
          %76 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%74 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
          %77 = arith.addi %76, %12 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32>
          %78 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%75 : i32) outs(%0 : tensor<32xi32>) -> tensor<32xi32>
          %79 = arith.addi %78, %12 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32>
          %80 = arith.index_cast %74 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %81 = arith.addi %80, %c32 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %82 = arith.index_cast %arg5 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %83 = arith.maxsi %80, %82 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %84 = arith.minsi %81, %83 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %85 = arith.subi %84, %80 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %extracted_slice_24 = tensor.extract_slice %77[0] [%85] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32> to tensor<?xi32>
          %inserted_slice_25 = tensor.insert_slice %extracted_slice_24 into %16[0] [%85] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<?xi32> into tensor<32xi32>
          %86 = arith.index_cast %75 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %87 = arith.addi %86, %c32 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %88 = arith.index_cast %arg6 {DataUse, ssbuffer.block_id = 9 : i32} : i32 to index
          %89 = arith.maxsi %86, %88 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %90 = arith.minsi %87, %89 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %91 = arith.subi %90, %86 {DataUse, ssbuffer.block_id = 9 : i32} : index
          %extracted_slice_26 = tensor.extract_slice %79[0] [%91] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32xi32> to tensor<?xi32>
          %inserted_slice_27 = tensor.insert_slice %extracted_slice_26 into %16[0] [%91] [1] {DataUse, ssbuffer.block_id = 9 : i32} : tensor<?xi32> into tensor<32xi32>
          scf.yield {Undefined, ssbuffer.block_id = 9 : i32} %65, %71, %73, %inserted_slice_25, %inserted_slice_27 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
        } else {
          scf.yield {Undefined, ssbuffer.block_id = 9 : i32} %arg25, %arg26, %arg27, %arg28, %arg29 : i32, i32, i32, tensor<32xi32>, tensor<32xi32>
        } {MixUse, ssbuffer.block_id = 9 : i32}
        scf.if %28 {
          %65 = arith.muli %59#1, %c32_i32 {ssbuffer.block_id = 9 : i32} : i32
          %66 = arith.muli %64#2, %c32_i32 {ssbuffer.block_id = 9 : i32} : i32
          %67 = arith.index_cast %65 {ssbuffer.block_id = 9 : i32} : i32 to index
          %alloc_24 = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<32xf32>
          %68 = arith.addi %67, %c32 {ssbuffer.block_id = 9 : i32} : index
          %69 = arith.index_cast %arg5 {ssbuffer.block_id = 9 : i32} : i32 to index
          %70 = arith.maxsi %67, %69 {ssbuffer.block_id = 9 : i32} : index
          %71 = arith.minsi %68, %70 {ssbuffer.block_id = 9 : i32} : index
          %72 = arith.subi %71, %67 {ssbuffer.block_id = 9 : i32} : index
          %73 = arith.cmpi slt, %72, %c32 {ssbuffer.block_id = 9 : i32} : index
          %74 = arith.index_cast %66 {ssbuffer.block_id = 9 : i32} : i32 to index
          %alloc_25 = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<32xf32>
          %75 = arith.addi %74, %c32 {ssbuffer.block_id = 9 : i32} : index
          %76 = arith.index_cast %arg6 {ssbuffer.block_id = 9 : i32} : i32 to index
          %77 = arith.maxsi %74, %76 {ssbuffer.block_id = 9 : i32} : index
          %78 = arith.minsi %75, %77 {ssbuffer.block_id = 9 : i32} : index
          %79 = arith.subi %78, %74 {ssbuffer.block_id = 9 : i32} : index
          %80 = arith.cmpi slt, %79, %c32 {ssbuffer.block_id = 9 : i32} : index
          %alloc_26 = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<32xf32>
          scf.if %80 {
            linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%alloc_26 : memref<32xf32>)
            linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%alloc_25 : memref<32xf32>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 9 : i32}
          scf.if %73 {
            linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%alloc_24 : memref<32xf32>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 9 : i32}
          %reinterpret_cast = memref.reinterpret_cast %arg11 to offset: [%67], sizes: [32], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
          %subview = memref.subview %reinterpret_cast[0] [%72] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
          %subview_27 = memref.subview %alloc_24[0] [%72] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32> to memref<?xf32, strided<[1]>>
          memref.copy %subview, %subview_27 {ssbuffer.block_id = 9 : i32} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
          %81 = bufferization.to_tensor %alloc_24 restrict writable {ssbuffer.block_id = 9 : i32} : memref<32xf32> to tensor<32xf32>
          %reinterpret_cast_28 = memref.reinterpret_cast %arg12 to offset: [%74], sizes: [32], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
          %subview_29 = memref.subview %reinterpret_cast_28[0] [%79] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
          %subview_30 = memref.subview %alloc_25[0] [%79] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32> to memref<?xf32, strided<[1]>>
          memref.copy %subview_29, %subview_30 {ssbuffer.block_id = 9 : i32} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
          %82 = bufferization.to_tensor %alloc_25 restrict writable {ssbuffer.block_id = 9 : i32} : memref<32xf32> to tensor<32xf32>
          %broadcasted_31 = linalg.broadcast ins(%81 : tensor<32xf32>) outs(%1 : tensor<32x32xf32>) dimensions = [1]  {ssbuffer.block_id = 9 : i32}
          %broadcasted_32 = linalg.broadcast ins(%82 : tensor<32xf32>) outs(%1 : tensor<32x32xf32>) dimensions = [0]  {ssbuffer.block_id = 9 : i32}
          %83 = arith.mulf %broadcasted_31, %broadcasted_32 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x32xf32>
          %c2_i32_33 = arith.constant {ssbuffer.block_id = 9 : i32} 2 : i32
          %84 = arith.remsi %arg23, %c2_i32_33 {ssbuffer.block_id = 9 : i32} : i32
          %c0_i32_34 = arith.constant {ssbuffer.block_id = 9 : i32} 0 : i32
          %85 = arith.cmpi eq, %84, %c0_i32_34 {ssbuffer.block_id = 9 : i32} : i32
          %86 = scf.if %85 -> (tensor<32x32xf32>) {
            %95 = bufferization.to_tensor %memspacecast restrict writable : memref<32x32xf32> to tensor<32x32xf32>
            scf.yield %95 : tensor<32x32xf32>
          } else {
            %95 = bufferization.to_tensor %memspacecast_6 restrict writable : memref<32x32xf32> to tensor<32x32xf32>
            scf.yield %95 : tensor<32x32xf32>
          } {ssbuffer.block_id = 9 : i32, ssbuffer.intraDeps = [0 : i32, 0 : i32], ssbuffer.intra_buffer}
          %87 = arith.mulf %86, %83 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x32xf32>
          %reinterpret_cast_35 = memref.reinterpret_cast %arg13 to offset: [%74], sizes: [32], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
          %subview_36 = memref.subview %reinterpret_cast_35[0] [%79] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
          %subview_37 = memref.subview %alloc_26[0] [%79] [1] {ssbuffer.block_id = 9 : i32} : memref<32xf32> to memref<?xf32, strided<[1]>>
          memref.copy %subview_36, %subview_37 {ssbuffer.block_id = 9 : i32} : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
          %88 = bufferization.to_tensor %alloc_26 restrict writable {ssbuffer.block_id = 9 : i32} : memref<32xf32> to tensor<32xf32>
          %broadcasted_38 = linalg.broadcast ins(%88 : tensor<32xf32>) outs(%1 : tensor<32x32xf32>) dimensions = [0]  {ssbuffer.block_id = 9 : i32}
          %89 = arith.addf %87, %broadcasted_38 {DataUse, ssbuffer.block_id = 9 : i32} : tensor<32x32xf32>
          %90 = arith.index_cast %arg16 {ssbuffer.block_id = 9 : i32} : i32 to index
          %91 = arith.muli %67, %90 {ssbuffer.block_id = 9 : i32} : index
          %92 = arith.addi %91, %74 {ssbuffer.block_id = 9 : i32} : index
          %reinterpret_cast_39 = memref.reinterpret_cast %arg4 to offset: [%92], sizes: [32, 32], strides: [%90, 1] {ssbuffer.block_id = 9 : i32} : memref<?xf32> to memref<32x32xf32, strided<[?, 1], offset: ?>>
          %93 = arith.minsi %72, %c32 {ssbuffer.block_id = 9 : i32} : index
          %94 = arith.minsi %79, %c32 {ssbuffer.block_id = 9 : i32} : index
          %extracted_slice_40 = tensor.extract_slice %89[0, 0] [%93, %94] [1, 1] {ssbuffer.block_id = 9 : i32} : tensor<32x32xf32> to tensor<?x?xf32>
          %subview_41 = memref.subview %reinterpret_cast_39[0, 0] [%93, %94] [1, 1] {ssbuffer.block_id = 9 : i32} : memref<32x32xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          bufferization.materialize_in_destination %extracted_slice_40 in writable %subview_41 {ssbuffer.block_id = 9 : i32} : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
        } {Undefined, ssbuffer.block_id = 9 : i32}
        scf.yield %25, %29#0, %29#1, %29#2, %29#3, %29#4, %54 : i32, i32, i32, i32, tensor<32xi32>, tensor<32xi32>, tensor<32x32xf32>
      } {Undefined, ssbuffer.block_id = 18 : i32, ssbuffer.main_loop = 0 : i32}
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 1 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 18 : i32, ssbuffer.transfer_id = 0 : i32}[<VECTOR>, <PIPE_M>, <PIPE_MTE3>] flag = 1
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
