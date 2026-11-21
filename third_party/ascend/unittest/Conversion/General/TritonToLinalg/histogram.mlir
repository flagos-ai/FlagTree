// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=true" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_histogram_i32
// CHECK: %[[INPUT:.*]] = bufferization.to_tensor {{.*}} : memref<1024xi32> to tensor<1024xi32>
// CHECK: %[[INIT:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%{{.*}} : tensor<256xi32>) -> tensor<256xi32>
// CHECK: %[[HIST:.*]] = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>, symbol = "__builtin_histogram"} "__builtin_histogram" ins(%[[INPUT]], %{{.*}} : tensor<1024xi32>, i64) outs(%[[INIT]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK: bufferization.materialize_in_destination %[[HIST]] in writable {{.*}} : (tensor<256xi32>, memref<256xi32, strided<[1]>>) -> ()
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @test_histogram_i32(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %in_offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %out_offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %in_addrs = tt.addptr %in_ptrs, %in_offsets : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<i32>>
    %hist = tt.histogram %input : tensor<1024xi32> -> tensor<256xi32>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %out_addrs = tt.addptr %out_ptrs, %out_offsets : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    tt.store %out_addrs, %hist : tensor<256x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @test_histogram_i16
// CHECK: %[[INPUT:.*]] = bufferization.to_tensor {{.*}} : memref<16xi16> to tensor<16xi16>
// CHECK: %[[INIT:.*]] = linalg.fill ins(%{{.*}} : i16) outs(%{{.*}} : tensor<8xi16>) -> tensor<8xi16>
// CHECK: %[[HIST:.*]] = hivm.hir.custom {gm_addr_args_indices = array<i32>, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, hivm.vf_mode = #hivm.vf_mode<SIMT>, symbol = "__builtin_histogram"} "__builtin_histogram" ins(%[[INPUT]], %{{.*}} : tensor<16xi16>, i64) outs(%[[INIT]] : tensor<8xi16>) -> tensor<8xi16>
// CHECK: bufferization.materialize_in_destination %[[HIST]] in writable {{.*}} : (tensor<8xi16>, memref<8xi16, strided<[1]>>) -> ()
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @test_histogram_i16(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>) attributes {noinline = false} {
    %in_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %out_offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<i16> -> tensor<16x!tt.ptr<i16>>
    %in_addrs = tt.addptr %in_ptrs, %in_offsets : tensor<16x!tt.ptr<i16>>, tensor<16xi32>
    %input = tt.load %in_addrs : tensor<16x!tt.ptr<i16>>
    %hist = tt.histogram %input : tensor<16xi16> -> tensor<8xi16>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
    %out_addrs = tt.addptr %out_ptrs, %out_offsets : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
    tt.store %out_addrs, %hist : tensor<8x!tt.ptr<i16>>
    tt.return
  }
}
