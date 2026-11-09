// RUN: triton-opt --clone-ops %s --allow-unregistered-dialect | FileCheck %s

// CHECK: func.func @test_clone_ops
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_clone_ops(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16>, %arg3: memref<?xf16>, %arg4: memref<?xf16>, %arg5: memref<?xi64>, %arg6: memref<?xi64>, %arg7: memref<?xi32>, %arg8: memref<?xf32>, %arg9: memref<?xf16>, %arg10: f32, %arg11: f32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : i64
    %c1_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 1 : i64
    %c8_i64 = arith.constant {ssbuffer.block_id = 18 : i32} 8 : i64
    %c63_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 63 : i32
    %c64_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 64 : i32
    %c256_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 256 : i32
    %c256_i64 = arith.constant {ssbuffer.block_id = 17 : i32} 256 : i64
    %c0_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : i32
    %c3_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 3 : i32
    %c1_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 1 : i32
    %c2_i32 = arith.constant {ssbuffer.block_id = 17 : i32} 2 : i32
    %c0 = arith.constant {ssbuffer.block_id = 17 : i32} 0 : index
    %c1 = arith.constant {ssbuffer.block_id = 17 : i32} 1 : index
    %c256 = arith.constant {ssbuffer.block_id = 17 : i32} 256 : index
    scope.scope : () -> () {
      %0 = tensor.empty() {ssbuffer.block_id = 17 : i32} : tensor<256x64xf32>
      %1 = linalg.fill {ssbuffer.block_id = 17 : i32} ins(%arg11 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
      %2 = linalg.fill {ssbuffer.block_id = 17 : i32} ins(%arg11 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
      %3 = arith.cmpi sle, %arg12, %c256_i32 {ssbuffer.block_id = 17 : i32} : i32
      %4 = scf.if %3 -> (i64) {
        %ext = arith.extsi %c2_i32 {ssbuffer.block_id = 17 : i32} : i32 to i64
        scf.yield %ext : i64
      } else {
        %reinterpret_cast = memref.reinterpret_cast %arg7 to offset: [2], sizes: [1], strides: [1] {ssbuffer.block_id = 7 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: 2>>
        %24 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 7 : i32} : memref<1xi32, strided<[1], offset: 2>>
        %25 = arith.extsi %24 {ssbuffer.block_id = 7 : i32} : i32 to i64
        scf.yield %25 : i64
      } {ssbuffer.block_id = 19 : i32}
      %5 = arith.muli %4, %c8_i64 {ssbuffer.block_id = 18 : i32} : i64
      %6 = arith.extsi %arg15 {ssbuffer.block_id = 18 : i32} : i32 to i64
      %7 = arith.minsi %6, %5 {ssbuffer.block_id = 18 : i32} : i64
      %8 = arith.divsi %5, %7 {ssbuffer.block_id = 18 : i32} : i64
      %9 = arith.addi %8, %c1_i64 {ssbuffer.block_id = 18 : i32} : i64
      %10 = arith.remsi %5, %7 {ssbuffer.block_id = 18 : i32} : i64
      %11 = arith.extsi %arg18 {ssbuffer.block_id = 18 : i32} : i32 to i64
      %12 = arith.cmpi slt, %11, %7 {ssbuffer.block_id = 18 : i32} : i64
      %13 = arith.cmpi slt, %11, %10 {ssbuffer.block_id = 18 : i32} : i64
      %14 = arith.muli %11, %9 {ssbuffer.block_id = 18 : i32} : i64
      %15 = arith.muli %10, %9 {ssbuffer.block_id = 18 : i32} : i64
      %16 = arith.subi %11, %10 {ssbuffer.block_id = 18 : i32} : i64
      %17 = arith.muli %16, %8 {ssbuffer.block_id = 18 : i32} : i64
      %18 = arith.addi %15, %17 {ssbuffer.block_id = 18 : i32} : i64
      %19 = arith.select %13, %14, %18 {ssbuffer.block_id = 18 : i32} : i64
      %20 = arith.select %12, %19, %c0_i64 {ssbuffer.block_id = 18 : i32} : i64
      %21 = arith.select %13, %9, %8 {ssbuffer.block_id = 18 : i32} : i64
      %22 = arith.select %12, %21, %c0_i64 {ssbuffer.block_id = 18 : i32} : i64
      %23 = arith.cmpi sge, %11, %7 {ssbuffer.block_id = 18 : i32} : i64
      scf.if %23 {
      } else {
        %25 = arith.addi %arg13, %c63_i32 {ssbuffer.block_id = 15 : i32} : i32
        %26 = arith.divsi %25, %c64_i32 {ssbuffer.block_id = 15 : i32} : i32
        %27 = arith.extsi %26 {ssbuffer.block_id = 15 : i32} : i32 to i64
        %28 = arith.muli %22, %27 {ssbuffer.block_id = 15 : i32} : i64
        %29 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%arg10 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %30 = linalg.fill {ssbuffer.block_id = 15 : i32} ins(%arg11 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %alloc = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 0 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_4 = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        %alloc_5 = memref.alloc() {ssbuffer.block_id = 26 : i32, ssbuffer.transfer_id = 2 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        scf.for %arg21 = %c0_i64 to %28 step %c1_i64  : i64 {
          %31 = arith.divsi %arg21, %27 {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [14 : i32]} : i64
          %32 = arith.addi %20, %31 {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [5 : i32]} : i64
          %33 = arith.remsi %32, %4 {ssbuffer.block_id = 11 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]} : i64
          // Test: op within the region depends on the preceding op in another block
          // CHECK: arith.divsi {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.clone = 11 : i32, ssbuffer.dep_mark = [14 : i32]}
          // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.clone = 11 : i32, ssbuffer.dep_mark = [5 : i32]}
          // CHECK: arith.remsi {{.*}} {ssbuffer.block_id = 22 : i32, ssbuffer.clone = 11 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]}
          %34:2 = scf.if %3 -> (i64, i64) {
            scf.yield %33, %c0_i64 : i64, i64
          } else {
            %66:2 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %c0_i32, %arg24 = %c3_i32) -> (i32, i32)  : i32 {
              %73 = arith.addi %arg23, %arg24 {ssbuffer.block_id = 9 : i32} : i32
              %74 = arith.divsi %73, %c2_i32 {ssbuffer.block_id = 9 : i32} : i32
              %75 = arith.index_cast %74 {ssbuffer.block_id = 9 : i32} : i32 to index
              %reinterpret_cast_11 = memref.reinterpret_cast %arg7 to offset: [%75], sizes: [1], strides: [1] {ssbuffer.block_id = 9 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
              %76 = memref.load %reinterpret_cast_11[%c0] {ssbuffer.block_id = 9 : i32} : memref<1xi32, strided<[1], offset: ?>>
              %77 = arith.extsi %76 {ssbuffer.block_id = 9 : i32} : i32 to i64
              %78 = arith.cmpi sle, %77, %33 {ssbuffer.block_id = 9 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]} : i64
              %79 = arith.select %78, %arg24, %74 {ssbuffer.block_id = 9 : i32} : i32
              %80 = scf.if %78 -> (i32) {
                %81 = arith.addi %74, %c1_i32 {ssbuffer.block_id = 8 : i32} : i32
                scf.yield %81 : i32
              } else {
                scf.yield %arg23 : i32
              } {ssbuffer.block_id = 20 : i32}
              scf.yield %80, %79 : i32, i32
            } {ssbuffer.block_id = 21 : i32}
            %67 = arith.subi %66#0, %c1_i32 {ssbuffer.block_id = 10 : i32} : i32
            %68 = arith.extsi %67 {ssbuffer.block_id = 10 : i32} : i32 to i64
            %69 = arith.index_cast %67 {ssbuffer.block_id = 10 : i32} : i32 to index
            %reinterpret_cast_10 = memref.reinterpret_cast %arg7 to offset: [%69], sizes: [1], strides: [1] {ssbuffer.block_id = 10 : i32} : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
            %70 = memref.load %reinterpret_cast_10[%c0] {ssbuffer.block_id = 10 : i32} : memref<1xi32, strided<[1], offset: ?>>
            %71 = arith.extsi %70 {ssbuffer.block_id = 10 : i32} : i32 to i64
            %72 = arith.subi %33, %71 {ssbuffer.block_id = 10 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]} : i64
            scf.yield %68, %72 : i64, i64
          } {ssbuffer.block_id = 22 : i32, ssbuffer.dep_mark = [1 : i32, 2 : i32]}
          // Test: op depends on the preceding region op in another block
          // CHECK: arith.divsi {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.clone = 11 : i32, ssbuffer.dep_mark = [14 : i32]}
          // CHECK: arith.addi {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.clone = 11 : i32, ssbuffer.dep_mark = [5 : i32]}
          // CHECK: arith.remsi {{.*}} {ssbuffer.block_id = 12 : i32, ssbuffer.clone = 11 : i32, ssbuffer.dep_mark = [3 : i32, 4 : i32]}
          // CHECK: } {ssbuffer.block_id = 12 : i32, ssbuffer.clone = 22 : i32, ssbuffer.dep_mark = [1 : i32, 2 : i32]}
          %35 = arith.index_cast %34#0 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [1 : i32]} : i64 to index
          %reinterpret_cast = memref.reinterpret_cast %arg5 to offset: [%35], sizes: [1], strides: [1] {ssbuffer.block_id = 12 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %36 = memref.load %reinterpret_cast[%c0] {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [6 : i32]} : memref<1xi64, strided<[1], offset: ?>>
          %37 = arith.addi %35, %c1 {ssbuffer.block_id = 12 : i32} : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg5 to offset: [%37], sizes: [1], strides: [1] {ssbuffer.block_id = 12 : i32} : memref<?xi64> to memref<1xi64, strided<[1], offset: ?>>
          %38 = memref.load %reinterpret_cast_6[%c0] {ssbuffer.block_id = 12 : i32} : memref<1xi64, strided<[1], offset: ?>>
          %39 = arith.subi %38, %36 {ssbuffer.block_id = 12 : i32} : i64
          %40 = arith.muli %34#1, %c256_i64 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [2 : i32]} : i64
          %41 = arith.index_cast %40 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [7 : i32]} : i64 to index
          %42 = arith.addi %41, %c256 {ssbuffer.block_id = 12 : i32} : index
          %43 = arith.index_cast %39 {ssbuffer.block_id = 12 : i32} : i64 to index
          %44 = arith.maxsi %41, %43 {ssbuffer.block_id = 12 : i32} : index
          %45 = arith.minsi %42, %44 {ssbuffer.block_id = 12 : i32} : index
          %46 = arith.subi %45, %41 {ssbuffer.block_id = 12 : i32, ssbuffer.dep_mark = [8 : i32]} : index
          // Test: op depends on the preceding op in another block
          // CHECK: arith.divsi {{.*}} {ssbuffer.block_id = 13 : i32, ssbuffer.clone = 11 : i32, ssbuffer.dep_mark = [14 : i32]}
          %47 = arith.addi %20, %31 {ssbuffer.block_id = 13 : i32, ssbuffer.dep_mark = [14 : i32]} : i64
        } {ssbuffer.block_id = 26 : i32, ssbuffer.main_loop = 0 : i32}
      } {ssbuffer.block_id = 27 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
