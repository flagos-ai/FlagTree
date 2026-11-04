// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' | FileCheck %s
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' -tritongpu-coalesce -tritongpu-remove-layout-conversions | FileCheck %s

// With no explicit result/data or pointer layout, preserve a mask/fallback
// hint when selecting the memory access layout. Its full access tuple must
// become consistent before coalescing and RLC inspect explicit attributes.
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
// CHECK-DAG: [[$B:#[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module {
  // CHECK-LABEL: tt.func @fallback_layout
  tt.func @fallback_layout(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %limit = arith.constant dense<512> : tensor<1024xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<1024xi32>
    %fallback = arith.constant dense<1.0> : tensor<1024xf32>
    %encoded_fallback = tle.gpu.set_layout %fallback {target_encoding = #b} : tensor<1024xf32> -> tensor<1024xf32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input_base, %range : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // CHECK: tt.load {{.*}}tle.explicit_memory_encoding = [[$B]]{{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    // CHECK-NOT: tt.load
    %loaded = tt.load %input_ptr, %mask, %encoded_fallback {isVolatile = true} : tensor<1024x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_base, %range : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // CHECK: tt.store
    tt.store %output_ptr, %loaded : tensor<1024x!tt.ptr<f32>>
    tt.return
  }

  // CHECK-LABEL: tt.func @mask_layout
  tt.func @mask_layout(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %limit = arith.constant dense<512> : tensor<1024xi32>
    %mask = arith.cmpi slt, %range, %limit : tensor<1024xi32>
    %encoded_mask = tle.gpu.set_layout %mask {target_encoding = #b} : tensor<1024xi1> -> tensor<1024xi1>
    %fallback = arith.constant dense<1.0> : tensor<1024xf32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input_base, %range : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // CHECK: tt.load {{.*}}tle.explicit_memory_encoding = [[$B]]{{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    // CHECK-NOT: tt.load
    %loaded = tt.load %input_ptr, %encoded_mask, %fallback {isVolatile = true} : tensor<1024x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_base, %range : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // CHECK: tt.store
    tt.store %output_ptr, %loaded : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
