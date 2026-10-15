// REQUIRES: flagtree-common-ir
// RUN: triton-opt %s --convert-common-ir-to-ttgir='enable-async-copy=true' | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module {
  // CHECK-LABEL: @independent_copies
  // CHECK: %[[A:.*]] = ttg.async_copy_global_to_local
  // CHECK-NOT: ttg.async_wait
  // CHECK: %[[B:.*]] = ttg.async_copy_global_to_local
  // CHECK: %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[A]], %[[B]]
  // CHECK: ttg.async_wait %[[COMMIT]]
  // CHECK-NOT: ttg.async_wait
  tt.func @independent_copies(%src: tensor<16x!tt.ptr<f32>>) {
    %a = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    %b = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    tile.copy %src -> %a : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    %c0 = arith.constant dense<0> : tensor<16xi32>
    %ptr = tt.addptr %src, %c0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tile.copy %ptr -> %b : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    tt.return
  }

  // CHECK-LABEL: @same_buffer
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_wait
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_wait
  tt.func @same_buffer(%src: tensor<16x!tt.ptr<f32>>) {
    %a = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    tile.copy %src -> %a : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    tile.copy %src -> %a : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    tt.return
  }

  // CHECK-LABEL: @read_between_copies
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_wait
  // CHECK: ttg.local_load
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_wait
  tt.func @read_between_copies(%src: tensor<16x!tt.ptr<f32>>) -> tensor<16xf32> {
    %a = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    %b = tile.alloc {space = 7 : i64, tle.gpu_layout = #shared} : <[16], f32, shared>
    tile.copy %src -> %a : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    %value = tile.to_tensor %a : <[16], f32, shared> -> tensor<16xf32>
    tile.copy %src -> %b : tensor<16x!tt.ptr<f32>>, !tile.buf<[16], f32, shared>
    tt.return %value : tensor<16xf32>
  }
}
