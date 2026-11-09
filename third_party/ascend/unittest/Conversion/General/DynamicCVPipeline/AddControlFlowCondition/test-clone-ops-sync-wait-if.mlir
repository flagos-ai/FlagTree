// RUN: triton-opt --clone-ops --allow-unregistered-dialect %s | FileCheck %s

// After --clone-ops, no cloned scf.if whose body contains only
// sync_block_wait/sync_block_set/fixpipe ops should remain.
// Such cloned ifOps are erased by cleanup Rule 3 (via the
// isIfOpWithOnlySyncOps check in shouldEraseOpForCube).
//
// A cloned scf.if carries `ssbuffer.clone` only on its closing
// attribute line — body ops are not cloned individually. We
// therefore distinguish the bad pattern (cloned sync-only-ifOp,
// whose closing attrs include `ssbuffer.cross_buffer = 1 : i32`)
// from the allowed pattern (cloned non-sync-only ifOp, whose
// closing attrs include `hivm.unlikely_condition` instead).

// CHECK-LABEL: func.func @_swa_paged_prefill_small_kernel
// CHECK-NOT: } {ssbuffer.block_id = {{.+}} : i32, ssbuffer.clone = {{.+}} : i32, ssbuffer.cross_buffer = 1 : i32}
// CHECK: return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_swa_paged_prefill_small_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg3: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xbf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg6: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg7: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg8: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg9: f32, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32, %arg21: memref<?xi8> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg22: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg23: memref<?xi64> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg24: i32, %arg25: i32, %arg26: i32, %arg27: i32, %arg28: i32, %arg29: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c4_i64 = arith.constant {ssbuffer.block_id = 19 : i32} 4 : i64
    %c0_i8 = arith.constant {ssbuffer.block_id = 19 : i32} 0 : i8
    %cst = arith.constant {ssbuffer.block_id = 19 : i32} -1.000000e+06 : f32
    %c0_i32 = arith.constant {ssbuffer.block_id = 18 : i32} 0 : i32
    %c128 = arith.constant {ssbuffer.block_id = 18 : i32} 128 : index
    %cst_0 = arith.constant {ssbuffer.block_id = 17 : i32} 0.000000e+00 : bf16
    %cst_1 = arith.constant {ssbuffer.block_id = 16 : i32} 0.000000e+00 : f32
    %c32_i64 = arith.constant {ssbuffer.block_id = 16 : i32} 32 : i64
    %c1_i64 = arith.constant {ssbuffer.block_id = 16 : i32} 1 : i64
    %c128_i64 = arith.constant {ssbuffer.block_id = 16 : i32} 128 : i64
    %c5_i64 = arith.constant {ssbuffer.block_id = 16 : i32} 5 : i64
    %c2_i32 = arith.constant {ssbuffer.block_id = 16 : i32} 2 : i32
    %c0_i64 = arith.constant {ssbuffer.block_id = 16 : i32} 0 : i64
    %c31_i64 = arith.constant {Undefined, ssbuffer.block_id = 16 : i32} 31 : i64
    %c-1283_i64 = arith.constant {ssbuffer.block_id = 16 : i32} -1283 : i64
    %c1283_i64 = arith.constant {ssbuffer.block_id = 16 : i32} 1283 : i64
    %cst_2 = arith.constant {ssbuffer.block_id = 16 : i32} 0xFF800000 : f32
    %c0 = arith.constant {ssbuffer.block_id = 16 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 16 : i32} 1 : index
    %c2 = arith.constant {ssbuffer.block_id = 16 : i32} 2 : index
    %c3 = arith.constant {ssbuffer.block_id = 16 : i32} 3 : index
    %c1539 = arith.constant {ssbuffer.block_id = 16 : i32} 1539 : index
    %cst_3 = arith.constant {ssbuffer.block_id = 12 : i32} dense<[2, 8, 16, 16]> : tensor<4xi64>
    %cst_4 = arith.constant {ssbuffer.block_id = 12 : i32} dense<[128, 2, 16]> : tensor<3xi64>
    %c32 = arith.constant {ssbuffer.block_id = 11 : i32} 32 : index
    scope.scope : () -> () {
      %0 = arith.muli %arg27, %c2_i32 {ssbuffer.block_id = 16 : i32} : i32
      %1 = arith.index_cast %0 {ssbuffer.block_id = 16 : i32} : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg23 to offset: [%1], sizes: [1], strides: [1] {ssbuffer.block_id = 16 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %2 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 16 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %3 = arith.addi %1, %c1 {ssbuffer.block_id = 16 : i32} : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg23 to offset: [%3], sizes: [1], strides: [1] {ssbuffer.block_id = 16 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
      %4 = memref.load %reinterpret_cast_5[%c0] {ssbuffer.block_id = 16 : i32} : memref<1xi64, strided<[1], offset: ?>>
      %5 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<128x32xf32>
      %6 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst_1 : f32) outs(%5 : tensor<128x32xf32>) -> tensor<128x32xf32>
      %7 = arith.extsi %arg13 {ssbuffer.block_id = 11 : i32} : i32 to i64
      %8 = arith.extsi %arg20 {ssbuffer.block_id = 11 : i32} : i32 to i64
      %9 = arith.extsi %arg15 {ssbuffer.block_id = 11 : i32} : i32 to i64
      %10 = arith.extsi %arg16 {ssbuffer.block_id = 11 : i32} : i32 to i64
      %11 = arith.extsi %arg18 {ssbuffer.block_id = 11 : i32} : i32 to i64
      %12 = arith.extsi %arg19 {ssbuffer.block_id = 11 : i32} : i32 to i64
      scf.for %arg30 = %2 to %4 step %c1_i64  : i64 {
        %13 = arith.muli %arg30, %c5_i64 {ssbuffer.block_id = 14 : i32} : i64
        %14 = arith.index_cast %13 {ssbuffer.block_id = 14 : i32} : i64 to index
        %15 = arith.addi %14, %c2 {ssbuffer.block_id = 14 : i32} : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg22 to offset: [%15], sizes: [1], strides: [1] {ssbuffer.block_id = 14 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %16 = memref.load %reinterpret_cast_6[%c0] {ssbuffer.block_id = 14 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %17 = arith.addi %14, %c3 {ssbuffer.block_id = 14 : i32} : index
        %reinterpret_cast_7 = memref.reinterpret_cast %arg22 to offset: [%17], sizes: [1], strides: [1] {ssbuffer.block_id = 14 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %18 = memref.load %reinterpret_cast_7[%c0] {ssbuffer.block_id = 14 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %19 = arith.index_cast %16 {ssbuffer.block_id = 14 : i32} : i64 to index
        %reinterpret_cast_8 = memref.reinterpret_cast %arg6 to offset: [%19], sizes: [1], strides: [1] {ssbuffer.block_id = 14 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %20 = memref.load %reinterpret_cast_8[%c0] {ssbuffer.block_id = 14 : i32} : memref<1xi32, strided<[1], offset: ?>>
        %21 = arith.addi %19, %c1 {ssbuffer.block_id = 14 : i32} : index
        %reinterpret_cast_9 = memref.reinterpret_cast %arg6 to offset: [%21], sizes: [1], strides: [1] {ssbuffer.block_id = 14 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %22 = memref.load %reinterpret_cast_9[%c0] {ssbuffer.block_id = 14 : i32} : memref<1xi32, strided<[1], offset: ?>>
        %23 = arith.subi %22, %20 {ssbuffer.block_id = 14 : i32} : i32
        %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%19], sizes: [1], strides: [1] {ssbuffer.block_id = 14 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %24 = memref.load %reinterpret_cast_10[%c0] {ssbuffer.block_id = 14 : i32} : memref<1xi32, strided<[1], offset: ?>>
        %25 = arith.subi %24, %23 {ssbuffer.block_id = 14 : i32} : i32
        %26 = arith.muli %18, %c128_i64 {ssbuffer.block_id = 14 : i32} : i64
        %27 = arith.addi %26, %c128_i64 {Undefined, ssbuffer.block_id = 14 : i32} : i64
        %28 = arith.extsi %23 {ssbuffer.block_id = 14 : i32} : i32 to i64
        %29 = arith.minsi %27, %28 {Undefined, ssbuffer.block_id = 14 : i32} : i64
        %30 = arith.subi %29, %26 {Undefined, ssbuffer.block_id = 14 : i32} : i64
        %31 = arith.extsi %25 {ssbuffer.block_id = 14 : i32} : i32 to i64
        %32 = arith.addi %26, %31 {ssbuffer.block_id = 14 : i32} : i64
        %33 = arith.addi %32, %30 {Undefined, ssbuffer.block_id = 14 : i32} : i64
        %34 = arith.addi %33, %c31_i64 {Undefined, ssbuffer.block_id = 14 : i32} : i64
        %35 = arith.divsi %34, %c32_i64 {Undefined, ssbuffer.block_id = 14 : i32} : i64
        %36 = arith.muli %arg30, %c5_i64 {ssbuffer.block_id = 9 : i32} : i64
        %37 = arith.index_cast %36 {ssbuffer.block_id = 9 : i32} : i64 to index
        %38 = arith.addi %37, %c1 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast_11 = memref.reinterpret_cast %arg22 to offset: [%38], sizes: [1], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %39 = memref.load %reinterpret_cast_11[%c0] {ssbuffer.block_id = 9 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %40 = arith.addi %37, %c2 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast_12 = memref.reinterpret_cast %arg22 to offset: [%40], sizes: [1], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %41 = memref.load %reinterpret_cast_12[%c0] {ssbuffer.block_id = 9 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %42 = arith.addi %37, %c3 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast_13 = memref.reinterpret_cast %arg22 to offset: [%42], sizes: [1], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
        %43 = memref.load %reinterpret_cast_13[%c0] {ssbuffer.block_id = 9 : i32} : memref<1xi64, strided<[1], offset: ?>>
        %44 = arith.index_cast %41 {ssbuffer.block_id = 9 : i32} : i64 to index
        %reinterpret_cast_14 = memref.reinterpret_cast %arg6 to offset: [%44], sizes: [1], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %45 = memref.load %reinterpret_cast_14[%c0] {ssbuffer.block_id = 9 : i32} : memref<1xi32, strided<[1], offset: ?>>
        %46 = arith.addi %44, %c1 {ssbuffer.block_id = 9 : i32} : index
        %reinterpret_cast_15 = memref.reinterpret_cast %arg6 to offset: [%46], sizes: [1], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %47 = memref.load %reinterpret_cast_15[%c0] {ssbuffer.block_id = 9 : i32} : memref<1xi32, strided<[1], offset: ?>>
        %48 = arith.subi %47, %45 {ssbuffer.block_id = 9 : i32} : i32
        %reinterpret_cast_16 = memref.reinterpret_cast %arg7 to offset: [%44], sizes: [1], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
        %49 = memref.load %reinterpret_cast_16[%c0] {ssbuffer.block_id = 9 : i32} : memref<1xi32, strided<[1], offset: ?>>
        %50 = arith.divsi %39, %c4_i64 {ssbuffer.block_id = 9 : i32} : i64
        %51 = arith.muli %43, %c128_i64 {ssbuffer.block_id = 9 : i32} : i64
        %52 = arith.muli %45, %arg12 {ssbuffer.block_id = 9 : i32} : i32
        %53 = arith.index_cast %52 {ssbuffer.block_id = 9 : i32} : i32 to index
        %54 = arith.muli %39, %7 {ssbuffer.block_id = 9 : i32} : i64
        %55 = arith.index_cast %54 {ssbuffer.block_id = 9 : i32} : i64 to index
        %56 = arith.addi %53, %55 {ssbuffer.block_id = 9 : i32} : index
        %57 = arith.trunci %51 {ssbuffer.block_id = 9 : i32} : i64 to i32
        %58 = arith.maxsi %57, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
        %59 = arith.index_cast %58 {ssbuffer.block_id = 9 : i32} : i32 to index
        %60 = arith.index_cast %arg12 {ssbuffer.block_id = 9 : i32} : i32 to index
        %61 = arith.muli %59, %60 {ssbuffer.block_id = 9 : i32} : index
        %62 = arith.addi %61, %56 {ssbuffer.block_id = 9 : i32} : index
        %63 = arith.index_cast %48 {ssbuffer.block_id = 9 : i32} : i32 to index
        %reinterpret_cast_17 = memref.reinterpret_cast %arg3 to offset: [%62], sizes: [128, 128], strides: [%60, 1] {ssbuffer.block_id = 9 : i32} : memref<?xbf16> to memref<128x128xbf16, strided<[?, 1], offset: ?>>
        %64 = arith.divsi %61, %60 {ssbuffer.block_id = 9 : i32} : index
        %65 = arith.subi %63, %64 {ssbuffer.block_id = 9 : i32} : index
        %66 = arith.maxsi %65, %c0 {ssbuffer.block_id = 9 : i32} : index
        %67 = arith.minsi %66, %c128 {ssbuffer.block_id = 9 : i32} : index
        %68 = arith.subi %c0_i32, %57 {ssbuffer.block_id = 9 : i32} : i32
        %69 = arith.maxsi %68, %c0_i32 {ssbuffer.block_id = 9 : i32} : i32
        %70 = arith.index_cast %69 {ssbuffer.block_id = 9 : i32} : i32 to index
        %71 = arith.minsi %70, %67 {ssbuffer.block_id = 9 : i32} : index
        %72 = arith.subi %67, %71 {ssbuffer.block_id = 9 : i32} : index
        %73 = arith.cmpi slt, %72, %c128 {ssbuffer.block_id = 9 : i32} : index
        %subview = memref.subview %reinterpret_cast_17[0, 0] [%72, 128] [1, 1] {ssbuffer.block_id = 9 : i32} : memref<128x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
        %74 = arith.extsi %49 {ssbuffer.block_id = 9 : i32} : i32 to i64
        %75 = arith.muli %41, %8 {ssbuffer.block_id = 9 : i32} : i64
        %76 = arith.muli %50, %9 {ssbuffer.block_id = 9 : i32} : i64
        %77 = arith.muli %50, %11 {ssbuffer.block_id = 9 : i32} : i64
        %alloc = memref.alloc() {ssbuffer.block_id = 10 : i32} : memref<128x128xbf16>
        %subview_18 = memref.subview %alloc[%71, 0] [%72, 128] [1, 1] {ssbuffer.block_id = 10 : i32} : memref<128x128xbf16> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
        scf.if %73 {
          linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst_0 : bf16) outs(%alloc : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 10 : i32}
        memref.copy %subview, %subview_18 {ssbuffer.block_id = 10 : i32} : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
        %78 = bufferization.to_tensor %alloc restrict writable {ssbuffer.block_id = 10 : i32} : memref<128x128xbf16> to tensor<128x128xbf16>
        %alloc_19 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32} : memref<2x8x16x16xbf16, #hivm.address_space<cbuf>>
        %alloc_20 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.crossDeps = [0 : i32, 1 : i32], ssbuffer.transfer_id = 0 : i32} : memref<2x8x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_20 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<3>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x8x16x16xbf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_19 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32} : memref<2x8x16x16xbf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
        hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
        %alloc_21 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x32xf32, #hivm.address_space<ub>>
        %alloc_22 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_22 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<5>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x32xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_21 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32} : memref<128x32xf32, #hivm.address_space<ub>>
        %alloc_23 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        %alloc_24 = memref.alloc() {ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_24 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<4>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_23 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32} : memref<128x128xf32, #hivm.address_space<ub>>
        scf.for %arg31 = %c0_i64 to %35 step %c1_i64  : i64 {
          %79 = arith.muli %arg31, %c32_i64 {ssbuffer.block_id = 8 : i32} : i64
          %80 = arith.addi %79, %c32_i64 {ssbuffer.block_id = 8 : i32} : i64
          %81 = arith.minsi %80, %74 {ssbuffer.block_id = 8 : i32} : i64
          %82 = arith.subi %81, %79 {ssbuffer.block_id = 8 : i32} : i64
          %83 = arith.index_cast %arg16 {ssbuffer.block_id = 8 : i32} : i32 to index
          %84 = arith.index_cast %82 {ssbuffer.block_id = 8 : i32} : i64 to index
          %85 = arith.index_cast %arg19 {ssbuffer.block_id = 8 : i32} : i32 to index
          %86 = arith.divsi %c0, %83 {ssbuffer.block_id = 8 : i32} : index
          %87 = arith.subi %84, %86 {ssbuffer.block_id = 8 : i32} : index
          %88 = arith.maxsi %87, %c0 {ssbuffer.block_id = 8 : i32} : index
          %89 = arith.minsi %88, %c32 {ssbuffer.block_id = 8 : i32} : index
          %90 = arith.minsi %89, %c0 {ssbuffer.block_id = 8 : i32} : index
          %91 = arith.subi %89, %90 {ssbuffer.block_id = 8 : i32} : index
          %92 = arith.cmpi slt, %91, %c32 {ssbuffer.block_id = 8 : i32} : index
          %93 = arith.divsi %c0, %85 {ssbuffer.block_id = 8 : i32} : index
          %94 = arith.subi %84, %93 {ssbuffer.block_id = 8 : i32} : index
          %95 = arith.maxsi %94, %c0 {ssbuffer.block_id = 8 : i32} : index
          %96 = arith.minsi %95, %c32 {ssbuffer.block_id = 8 : i32} : index
          %97 = arith.minsi %96, %c0 {ssbuffer.block_id = 8 : i32} : index
          %98 = arith.subi %96, %97 {ssbuffer.block_id = 8 : i32} : index
          %99 = arith.cmpi slt, %98, %c32 {ssbuffer.block_id = 8 : i32} : index
          %alloc_25 = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<32x128xbf16>
          scf.if %92 {
            linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst_0 : bf16) outs(%alloc_25 : memref<32x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
          %100 = arith.divsi %79, %c32_i64 {ssbuffer.block_id = 5 : i32} : i64
          %101 = arith.remsi %79, %c32_i64 {ssbuffer.block_id = 5 : i32} : i64
          %102 = arith.addi %75, %100 {ssbuffer.block_id = 5 : i32} : i64
          %103 = arith.index_cast %102 {ssbuffer.block_id = 5 : i32} : i64 to index
          %reinterpret_cast_26 = memref.reinterpret_cast %arg8 to offset: [%103], sizes: [1], strides: [1] {ssbuffer.block_id = 5 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
          %104 = memref.load %reinterpret_cast_26[%c0] {ssbuffer.block_id = 5 : i32} : memref<1xi32, strided<[1], offset: ?>>
          %105 = arith.muli %104, %arg14 {ssbuffer.block_id = 5 : i32} : i32
          %106 = arith.index_cast %105 {ssbuffer.block_id = 5 : i32} : i32 to index
          %107 = arith.muli %101, %10 {ssbuffer.block_id = 5 : i32} : i64
          %108 = arith.addi %76, %107 {ssbuffer.block_id = 5 : i32} : i64
          %109 = arith.index_cast %108 {ssbuffer.block_id = 5 : i32} : i64 to index
          %110 = arith.addi %106, %109 {ssbuffer.block_id = 5 : i32} : index
          %reinterpret_cast_27 = memref.reinterpret_cast %arg4 to offset: [%110], sizes: [32, 128], strides: [%83, 1] {ssbuffer.block_id = 5 : i32} : memref<?xbf16> to memref<32x128xbf16, strided<[?, 1], offset: ?>>
          %subview_28 = memref.subview %reinterpret_cast_27[0, 0] [%91, 128] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<32x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
          %subview_29 = memref.subview %alloc_25[%90, 0] [%91, 128] [1, 1] {ssbuffer.block_id = 5 : i32} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_28, %subview_29 {ssbuffer.block_id = 5 : i32} : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          %111 = bufferization.to_tensor %alloc_25 restrict writable {ssbuffer.block_id = 5 : i32} : memref<32x128xbf16> to tensor<32x128xbf16>
          %112 = tensor.empty() {ssbuffer.block_id = 5 : i32} : tensor<128x32xbf16>
          %transposed = linalg.transpose ins(%111 : tensor<32x128xbf16>) outs(%112 : tensor<128x32xbf16>) permutation = [1, 0]  {ssbuffer.block_id = 5 : i32}
          %113 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 5 : i32, ssbuffer.loop_carried_l0c} ins(%78, %transposed : tensor<128x128xbf16>, tensor<128x32xbf16>) outs(%6 : tensor<128x32xf32>) -> tensor<128x32xf32>
          %114 = arith.divsi %arg31, %c1_i64 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} : i64
          %c2_i64 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} 2 : i64
          %115 = arith.remsi %114, %c2_i64 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} : i64
          %c0_i64_30 = arith.constant {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} 0 : i64
          %116 = arith.cmpi eq, %115, %c0_i64_30 {ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} : i64
          scf.if %116 {
            hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
          } else {
            hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
          } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32}
          scf.if %116 {
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} ins(%113 : tensor<128x32xf32>) outs(%alloc_21 : memref<128x32xf32, #hivm.address_space<ub>>)
          } else {
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32} ins(%113 : tensor<128x32xf32>) outs(%alloc_22 : memref<128x32xf32, #hivm.address_space<ub>>)
          } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 1 : i32}
          scf.if %116 {
            hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 2
          } else {
            hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 5 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 6
          } {ssbuffer.block_id = 5 : i32, ssbuffer.cross_buffer = 1 : i32}
          %alloc_31 = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16>
          scf.if %99 {
            linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst_0 : bf16) outs(%alloc_31 : memref<32x128xbf16>)
          } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
          %117 = arith.muli %104, %arg17 {ssbuffer.block_id = 7 : i32} : i32
          %118 = arith.index_cast %117 {ssbuffer.block_id = 7 : i32} : i32 to index
          %119 = arith.muli %101, %12 {ssbuffer.block_id = 7 : i32} : i64
          %120 = arith.addi %77, %119 {ssbuffer.block_id = 7 : i32} : i64
          %121 = arith.index_cast %120 {ssbuffer.block_id = 7 : i32} : i64 to index
          %122 = arith.addi %118, %121 {ssbuffer.block_id = 7 : i32} : index
          %reinterpret_cast_32 = memref.reinterpret_cast %arg5 to offset: [%122], sizes: [32, 128], strides: [%85, 1] {ssbuffer.block_id = 7 : i32} : memref<?xbf16> to memref<32x128xbf16, strided<[?, 1], offset: ?>>
          %subview_33 = memref.subview %reinterpret_cast_32[0, 0] [%98, 128] [1, 1] {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[?, 1], offset: ?>>
          %subview_34 = memref.subview %alloc_31[%97, 0] [%98, 128] [1, 1] {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          memref.copy %subview_33, %subview_34 {ssbuffer.block_id = 7 : i32} : memref<?x128xbf16, strided<[?, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1], offset: ?>>
          %123 = bufferization.to_tensor %alloc_31 restrict writable {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16> to tensor<32x128xbf16>
          %124 = tensor.empty() {ssbuffer.block_id = 7 : i32} : tensor<128x128xf32>
          %125 = linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst_1 : f32) outs(%124 : tensor<128x128xf32>) -> tensor<128x128xf32>
          %126 = arith.divsi %arg31, %c1_i64 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} : i64
          %c2_i64_35 = arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} 2 : i64
          %127 = arith.remsi %126, %c2_i64_35 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} : i64
          %c0_i64_36 = arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} 0 : i64
          %128 = arith.cmpi eq, %127, %c0_i64_36 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} : i64
          scf.if %128 {
            hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          } else {
            hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 4
          } {ssbuffer.block_id = 7 : i32, ssbuffer.cross_buffer = 1 : i32}
          %129 = scf.if %128 -> (tensor<128x32xbf16>) {
            %134 = hivm.hir.convert_layout %alloc_19 output_shape [128, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 7 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : (memref<2x8x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<128x32xbf16, #hivm.address_space<cbuf>>
            %memspacecast = memref.memory_space_cast %134 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x32xbf16, #hivm.address_space<cbuf>> to memref<128x32xbf16>
            %135 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x32xbf16> to tensor<128x32xbf16>
            scf.yield %135 : tensor<128x32xbf16>
          } else {
            %134 = hivm.hir.convert_layout %alloc_20 output_shape [128, 32] {dstLayout = #hivm.data_layout<ND>, srcLayout = #hivm.data_layout<nZ>, ssbuffer.block_id = 7 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32], ssbuffer.transfer_id = 0 : i32} : (memref<2x8x16x16xbf16, #hivm.address_space<cbuf>>) -> memref<128x32xbf16, #hivm.address_space<cbuf>>
            %memspacecast = memref.memory_space_cast %134 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x32xbf16, #hivm.address_space<cbuf>> to memref<128x32xbf16>
            %135 = bufferization.to_tensor %memspacecast restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32} : memref<128x32xbf16> to tensor<128x32xbf16>
            scf.yield %135 : tensor<128x32xbf16>
          } {ssbuffer.block_id = 7 : i32, ssbuffer.crossDeps = [0 : i32, 0 : i32], ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 0 : i32}
          %130 = linalg.matmul {input_precision = "ieee", ssbuffer.adep, ssbuffer.block_id = 7 : i32, ssbuffer.loop_carried_l0c} ins(%129, %123 : tensor<128x32xbf16>, tensor<32x128xbf16>) outs(%125 : tensor<128x128xf32>) -> tensor<128x128xf32>
          scf.if %128 {
            hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 1
          } else {
            hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 4
          } {ssbuffer.block_id = 7 : i32, ssbuffer.cross_buffer = 1 : i32}
          %131 = arith.divsi %arg31, %c1_i64 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : i64
          %c2_i64_37 = arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} 2 : i64
          %132 = arith.remsi %131, %c2_i64_37 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : i64
          %c0_i64_38 = arith.constant {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} 0 : i64
          %133 = arith.cmpi eq, %132, %c0_i64_38 {ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} : i64
          scf.if %133 {
            hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
          } else {
            hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
          } {ssbuffer.block_id = 7 : i32, ssbuffer.cross_buffer = 1 : i32}
          scf.if %133 {
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} ins(%130 : tensor<128x128xf32>) outs(%alloc_23 : memref<128x128xf32, #hivm.address_space<ub>>)
          } else {
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32} ins(%130 : tensor<128x128xf32>) outs(%alloc_24 : memref<128x128xf32, #hivm.address_space<ub>>)
          } {ssbuffer.block_id = 7 : i32, ssbuffer.cross_buffer = 1 : i32, ssbuffer.transfer_id = 2 : i32}
          scf.if %133 {
            hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 3
          } else {
            hivm.hir.sync_block_set {ssbuffer.analyze_flag_id, ssbuffer.block_id = 7 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 5
          } {ssbuffer.block_id = 7 : i32, ssbuffer.cross_buffer = 1 : i32}
        } {DataUse, ssbuffer.block_id = 20 : i32, ssbuffer.flowOpt, ssbuffer.main_loop = 0 : i32}
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 2 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 5
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
        hivm.hir.sync_block_wait {ssbuffer.analyze_flag_id, ssbuffer.block_id = 20 : i32, ssbuffer.transfer_id = 1 : i32}[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 6
      } {Undefined, ssbuffer.block_id = 21 : i32}
      scope.return
    } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
