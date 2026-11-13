// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' | FileCheck %s --check-prefix=LEGAL
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8' -tritongpu-remove-layout-conversions -cse | FileCheck %s --check-prefix=RLC

// Layout selection must not recursively clone a shared address/mask DAG.
// Conversion legalizes each memory access with explicit layout conversions;
// RLC rematerializes the pure indices only where different layouts require it.
// The volatile load makes duplication/deletion of the memory effect observable.
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

// LEGAL-DAG: [[$A:#[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
// LEGAL-DAG: [[$B:#[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
// RLC-DAG: [[$A:#[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
// RLC-DAG: [[$B:#[a-zA-Z0-9_]+]] = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module {
  // LEGAL-LABEL: tt.func @shared_mask_and_address
  // RLC-LABEL: tt.func @shared_mask_and_address
  // RLC-DAG: %[[OFFSET_B:.*]] = arith.muli {{.*}} : tensor<1024xi32, [[$B]]>
  // RLC-DAG: %[[MASK_B:.*]] = arith.cmpi {{.*}} : tensor<1024xi32, [[$B]]>
  tt.func @shared_mask_and_address(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %stride: i32) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %strides = tt.splat %stride : i32 -> tensor<1024xi32>
    // LEGAL: %[[OFFSET:.*]] = arith.muli
    %offset = arith.muli %range, %strides : tensor<1024xi32>
    %limit = arith.constant dense<1024> : tensor<1024xi32>
    // LEGAL: %[[MASK:.*]] = arith.cmpi
    %mask = arith.cmpi slt, %offset, %limit : tensor<1024xi32>
    %zero = arith.constant dense<0.0> : tensor<1024xf32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input_base, %offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // LEGAL-NOT: arith.muli
    // LEGAL-NOT: arith.cmpi
    // LEGAL: tt.load {{.*}} %[[MASK]]{{.*}} : tensor<1024x!tt.ptr<f32>, [[$A]]>
    // RLC: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, [[$A]]>
    // RLC-NOT: tt.load
    %loaded = tt.load %input_ptr, %mask, %zero {isVolatile = true} : tensor<1024x!tt.ptr<f32>>
    %value_a = tle.gpu.set_layout %loaded {target_encoding = #a} : tensor<1024xf32> -> tensor<1024xf32>
    %value_b = tle.gpu.set_layout %value_a {target_encoding = #b} : tensor<1024xf32> -> tensor<1024xf32>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_base, %offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // LEGAL-NOT: arith.muli
    // LEGAL-NOT: arith.cmpi
    // LEGAL: ttg.convert_layout {{.*}} -> tensor<1024x!tt.ptr<f32>, [[$B]]>
    // LEGAL: ttg.convert_layout %[[MASK]] {{.*}} -> tensor<1024xi1, [[$B]]>
    // LEGAL: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    // RLC: %[[VALUE_B:.*]] = ttg.convert_layout {{.*}} -> tensor<1024xf32, [[$B]]>
    // RLC-NEXT: %[[BASE_B:.*]] = tt.splat
    // RLC-NEXT: %[[PTR_B:.*]] = tt.addptr %[[BASE_B]], %[[OFFSET_B]]
    // RLC-NEXT: tt.store %[[PTR_B]], %[[VALUE_B]], %[[MASK_B]] {{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    tt.store %output_ptr, %value_b, %mask : tensor<1024x!tt.ptr<f32>>
    // LEGAL-NOT: arith.muli
    // LEGAL-NOT: arith.cmpi
    // RLC-NOT: tt.load
    // RLC-NOT: tt.store
    tt.return
  }

  // Pointer and loaded value may each have an explicit (different) layout.
  // Neither set_layout is a demand to retag the other side of the access.
  // LEGAL-LABEL: tt.func @explicit_pointer_and_value
  // RLC-LABEL: tt.func @explicit_pointer_and_value
  tt.func @explicit_pointer_and_value(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %encoded_range = tle.gpu.set_layout %range {target_encoding = #a} : tensor<1024xi32> -> tensor<1024xi32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input_base, %encoded_range : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // LEGAL: ttg.convert_layout {{.*}} -> tensor<1024x!tt.ptr<f32>, [[$B]]>
    // LEGAL: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    // RLC: tt.load {{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    %loaded = tt.load %input_ptr {isVolatile = true} : tensor<1024x!tt.ptr<f32>>
    %encoded_value = tle.gpu.set_layout %loaded {target_encoding = #b} : tensor<1024xf32> -> tensor<1024xf32>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_base, %encoded_range : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // LEGAL: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    // RLC-NOT: tt.load
    // RLC: tt.store {{.*}} : tensor<1024x!tt.ptr<f32>, [[$B]]>
    tt.store %output_ptr, %encoded_value : tensor<1024x!tt.ptr<f32>>
    tt.return
  }

  // Matching access layouts keep the same mask and address arithmetic.
  // LEGAL-LABEL: tt.func @same_layout_reuses_mask
  // RLC-LABEL: tt.func @same_layout_reuses_mask
  tt.func @same_layout_reuses_mask(%first: !tt.ptr<f32>, %second: !tt.ptr<f32>, %stride: i32) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
    %strides = tt.splat %stride : i32 -> tensor<1024xi32>
    // LEGAL: %[[SHARED_OFFSET:.*]] = arith.muli
    // RLC: %[[SHARED_OFFSET:.*]] = arith.muli
    %offset = arith.muli %range, %strides : tensor<1024xi32>
    %limit = arith.constant dense<1024> : tensor<1024xi32>
    // LEGAL: %[[SHARED_MASK:.*]] = arith.cmpi
    // RLC: %[[SHARED_MASK:.*]] = arith.cmpi
    %mask = arith.cmpi slt, %offset, %limit : tensor<1024xi32>
    %one = arith.constant dense<1.0> : tensor<1024xf32>
    %value = tle.gpu.set_layout %one {target_encoding = #a} : tensor<1024xf32> -> tensor<1024xf32>
    %first_base = tt.splat %first : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %first_ptr = tt.addptr %first_base, %offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // LEGAL-NOT: arith.muli
    // LEGAL-NOT: arith.cmpi
    // RLC-NOT: arith.muli
    // RLC-NOT: arith.cmpi
    // LEGAL: tt.store %{{.*}}, %{{.*}}, %[[SHARED_MASK]]
    // RLC: tt.store %{{.*}}, %{{.*}}, %[[SHARED_MASK]]
    tt.store %first_ptr, %value, %mask : tensor<1024x!tt.ptr<f32>>
    %second_base = tt.splat %second : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // LEGAL-NOT: arith.muli
    // LEGAL-NOT: arith.cmpi
    // RLC-NOT: arith.muli
    // RLC-NOT: arith.cmpi
    // LEGAL: tt.addptr %{{.*}}, %[[SHARED_OFFSET]]
    // RLC: tt.addptr %{{.*}}, %[[SHARED_OFFSET]]
    %second_ptr = tt.addptr %second_base, %offset : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // LEGAL: tt.store %{{.*}}, %{{.*}}, %[[SHARED_MASK]]
    // RLC: tt.store %{{.*}}, %{{.*}}, %[[SHARED_MASK]]
    tt.store %second_ptr, %value, %mask : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
