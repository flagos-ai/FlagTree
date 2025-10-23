// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
tt.func public @assert_lol(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
  %1 = tt.splat %0 : i1 -> tensor<1xi1>
  tt.assert %1, "lol" : tensor<1xi1>
  tt.return
}

// CHECK: (%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:   return
// CHECK: }
