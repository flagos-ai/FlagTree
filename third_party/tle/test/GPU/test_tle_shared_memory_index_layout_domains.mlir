// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' | FileCheck %s

// A front-end CSE is allowed to share pure range/mask expressions.  Each
// memory access nevertheless owns an independent layout domain: the shared
// expressions below must be rematerialized instead of forcing the load and
// store to use one encoding.

#load_layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#store_layout = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

// CHECK-DAG: [[$LOAD:#[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
// CHECK-DAG: [[$STORE:#[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module {
  // CHECK-LABEL: tt.func public @shared_memory_index_layout_domains
  tt.func public @shared_memory_index_layout_domains(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %limit = arith.constant dense<127> : tensor<128xi32>
    %mask = arith.cmpi sle, %range, %limit : tensor<128xi32>
    %zero = arith.constant dense<0.0> : tensor<128xf32>

    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input_base, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    // CHECK: tt.load {{.*}} : tensor<128x!tt.ptr<f32>, [[$LOAD]]>
    %loaded = tt.load %input_ptr, %mask, %zero : tensor<128x!tt.ptr<f32>>
    %encoded_load = tle.gpu.set_layout %loaded {target_encoding = #load_layout} : tensor<128xf32> -> tensor<128xf32>

    %one = arith.constant dense<1.0> : tensor<128xf32>
    %encoded_store = tle.gpu.set_layout %one {target_encoding = #store_layout} : tensor<128xf32> -> tensor<128xf32>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_base, %range : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    // CHECK: tt.store {{.*}} : tensor<128x!tt.ptr<f32>, [[$STORE]]>
    tt.store %output_ptr, %encoded_store, %mask : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
