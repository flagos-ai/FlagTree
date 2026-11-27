// RUN: triton-opt %s --triton-to-linalg='compile-on-910-95=true' --split-input-file | FileCheck %s --check-prefixes=ON-CHECK-LABEL,ON-CHECK-SAME
// RUN: triton-opt %s --triton-to-linalg='compile-on-910-95=false' --split-input-file | FileCheck %s --check-prefixes=OFF-CHECK-LABEL,OFF-CHECK-SAME

// isSIMTOp() routes math.sin / math.cos on f16/f32 tensor inputs to the SIMT
// template.  Downstream justification (Ascend 950 RegBase normalize, enable-high-
// precision defaults to true): these ops are rewritten into a Payne-Hanek
// range reduction that looks up a 320xi32 2/pi limbs table with two
// hfusion.gather ops, so the kernel must be launched as mix_simd_simt to
// reserve localMemorySize for the SIMT template (otherwise VEC UB
// out-of-bounds, error 341).  Observable signal: the converted func.func
// carries parallel_mode = "mix_simd_simt".  bf16 is the negative control:
// the downstream high-precision trait only accepts f16/f32, so bf16 sin
// stays SIMD and produces no table gather.

// ON-CHECK-LABEL: func.func @sin_f32
// ON-CHECK-SAME: parallel_mode = "mix_simd_simt"
// OFF-CHECK-LABEL: func.func @sin_f32
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @sin_f32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f32>>
    %result = math.sin %input : tensor<1024xf32>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// ON-CHECK-LABEL: func.func @cos_f32
// ON-CHECK-SAME: parallel_mode = "mix_simd_simt"
// OFF-CHECK-LABEL: func.func @cos_f32
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @cos_f32(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f32>>
    %result = math.cos %input : tensor<1024xf32>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// ON-CHECK-LABEL: func.func @sin_f16
// ON-CHECK-SAME: parallel_mode = "mix_simd_simt"
// OFF-CHECK-LABEL: func.func @sin_f16
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @sin_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f16>>
    %result = math.sin %input : tensor<1024xf16>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// ON-CHECK-LABEL: func.func @cos_f16
// ON-CHECK-SAME: parallel_mode = "mix_simd_simt"
// OFF-CHECK-LABEL: func.func @cos_f16
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @cos_f16(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<f16>>
    %result = math.cos %input : tensor<1024xf16>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<f16>>
    tt.return
  }
}

// -----

// Negative control: bf16 input does not match the f16/f32 filter, so sin
// stays SIMD (no downstream high-precision table gather for bf16).
// ON-CHECK-LABEL: func.func @sin_bf16
// ON-CHECK-SAME: parallel_mode = "simd"
// OFF-CHECK-LABEL: func.func @sin_bf16
// OFF-CHECK-SAME: parallel_mode = "simd"
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @sin_bf16(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %in_ptrs = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %in_addrs = tt.addptr %in_ptrs, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    %input = tt.load %in_addrs : tensor<1024x!tt.ptr<bf16>>
    %result = math.sin %input : tensor<1024xbf16>
    %out_ptrs = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %out_addrs = tt.addptr %out_ptrs, %offsets : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    tt.store %out_addrs, %result : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}
