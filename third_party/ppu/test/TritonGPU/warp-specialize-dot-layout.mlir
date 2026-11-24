// RUN: triton-opt %s -split-input-file --convert-triton-to-tritongpu="target=ppu:80 num-warps=4" | FileCheck %s

// A tt.dot in a warp_specialize partition gets the result layout of that
// partition's warp count, not of the default region's. The default region runs
// 4 warps. A layout of the wrong warp count fails the verifier ("Layout has 4
// warps per CTA, but the context requires 8").

// A partition wider than the default region.

// CHECK-DAG: #[[$DEFAULT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$WORKER:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-LABEL: @dot_in_wider_partition
// CHECK: ttg.warp_specialize
// CHECK: default {
// CHECK: ttg.convert_layout {{.*}} -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DEFAULT]]}>>
// CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DEFAULT]]>
// CHECK: partition0({{.*}}) num_warps(8) {
// CHECK: ttg.convert_layout {{.*}} -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$WORKER]]}>>
// CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$WORKER]]>

module {
  tt.func public @dot_in_wider_partition(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    ttg.warp_specialize(%a, %b) attributes {requestedRegisters = array<i32: 128>}
    default {
      %pa = tt.splat %a : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %pb = tt.splat %b : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %x = tt.load %pa : tensor<64x64x!tt.ptr<bf16>>
      %y = tt.load %pb : tensor<64x64x!tt.ptr<bf16>>
      %d = tt.dot %x, %y, %zero : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      ttg.warp_yield
    }
    partition0(%a0: !tt.ptr<bf16>, %b0: !tt.ptr<bf16>) num_warps(8) {
      %z = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
      %pa = tt.splat %a0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %pb = tt.splat %b0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %x = tt.load %pa : tensor<64x64x!tt.ptr<bf16>>
      %y = tt.load %pb : tensor<64x64x!tt.ptr<bf16>>
      %d = tt.dot %x, %y, %z : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      ttg.warp_return
    } : (!tt.ptr<bf16>, !tt.ptr<bf16>) -> ()
    tt.return
  }
}

// -----

// A partition narrower than the default region.

// CHECK-DAG: #[[$DEFAULT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$WORKER:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-LABEL: @dot_in_narrower_partition
// CHECK: ttg.warp_specialize
// CHECK: default {
// CHECK: ttg.convert_layout {{.*}} -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DEFAULT]]}>>
// CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DEFAULT]]>
// CHECK: partition0({{.*}}) num_warps(1) {
// CHECK: ttg.convert_layout {{.*}} -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$WORKER]]}>>
// CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$WORKER]]>

module {
  tt.func public @dot_in_narrower_partition(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    ttg.warp_specialize(%a, %b) attributes {requestedRegisters = array<i32: 128>}
    default {
      %pa = tt.splat %a : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %pb = tt.splat %b : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %x = tt.load %pa : tensor<64x64x!tt.ptr<bf16>>
      %y = tt.load %pb : tensor<64x64x!tt.ptr<bf16>>
      %d = tt.dot %x, %y, %zero : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      ttg.warp_yield
    }
    partition0(%a0: !tt.ptr<bf16>, %b0: !tt.ptr<bf16>) num_warps(1) {
      %z = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
      %pa = tt.splat %a0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %pb = tt.splat %b0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %x = tt.load %pa : tensor<64x64x!tt.ptr<bf16>>
      %y = tt.load %pb : tensor<64x64x!tt.ptr<bf16>>
      %d = tt.dot %x, %y, %z : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
      ttg.warp_return
    } : (!tt.ptr<bf16>, !tt.ptr<bf16>) -> ()
    tt.return
  }
}

// -----

// A loop-carried accumulator in a wider partition: the loop's block arguments
// and results are converted while the conversion moves the loop body, so the
// partition's warp count is recorded before the conversion starts. The loop
// carries the 8-warp default layout of its partition, and the dot works in its
// own 8-warp layout; the layouts of 4 warps do not appear in the partition.

// CHECK-DAG: #[[$CARRIED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
// CHECK-DAG: #[[$DOT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-LABEL: @dot_in_loop_of_wider_partition
// CHECK: partition0({{.*}}) num_warps(8) {
// CHECK: scf.for {{.*}} iter_args({{.*}} = {{.*}}) -> (tensor<64x64xf32, #[[$CARRIED]]>)
// CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DOT]]>
// CHECK: scf.yield {{.*}} : tensor<64x64xf32, #[[$CARRIED]]>

module {
  tt.func public @dot_in_loop_of_wider_partition(%a: !tt.ptr<bf16>, %b: !tt.ptr<bf16>, %n: i32) {
    ttg.warp_specialize(%a, %b, %n) attributes {requestedRegisters = array<i32: 128>}
    default {
      ttg.warp_yield
    }
    partition0(%a0: !tt.ptr<bf16>, %b0: !tt.ptr<bf16>, %n0: i32) num_warps(8) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      %z = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
      %pa = tt.splat %a0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %pb = tt.splat %b0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>>
      %x = tt.load %pa : tensor<64x64x!tt.ptr<bf16>>
      %y = tt.load %pb : tensor<64x64x!tt.ptr<bf16>>
      %r = scf.for %i = %c0 to %n0 step %c1 iter_args(%acc = %z) -> (tensor<64x64xf32>) : i32 {
        %d = tt.dot %x, %y, %acc : tensor<64x64xbf16> * tensor<64x64xbf16> -> tensor<64x64xf32>
        scf.yield %d : tensor<64x64xf32>
      }
      ttg.warp_return
    } : (!tt.ptr<bf16>, !tt.ptr<bf16>, i32) -> ()
    tt.return
  }
}

// -----

// tt.split of a source that is not in the layout split needs builds a default
// layout for the conversion. In a wider partition that layout has the
// partition's warp count too.

// CHECK-DAG: #[[$SPLIT4:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
// CHECK-DAG: #[[$SPLIT8:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 2, 1], order = [2, 1, 0]}>
// CHECK-LABEL: @split_in_wider_partition
// CHECK: default {
// CHECK: tt.split {{.*}} : tensor<64x64x2xf32, #[[$SPLIT4]]>
// CHECK: partition0({{.*}}) num_warps(8) {
// CHECK: tt.split {{.*}} : tensor<64x64x2xf32, #[[$SPLIT8]]>

module {
  tt.func public @split_in_wider_partition(%a: !tt.ptr<f32>) {
    ttg.warp_specialize(%a) attributes {requestedRegisters = array<i32: 128>}
    default {
      %p = tt.splat %a : !tt.ptr<f32> -> tensor<64x64x2x!tt.ptr<f32>>
      %v = tt.load %p : tensor<64x64x2x!tt.ptr<f32>>
      %l, %r = tt.split %v : tensor<64x64x2xf32> -> tensor<64x64xf32>
      ttg.warp_yield
    }
    partition0(%a0: !tt.ptr<f32>) num_warps(8) {
      %p = tt.splat %a0 : !tt.ptr<f32> -> tensor<64x64x2x!tt.ptr<f32>>
      %v = tt.load %p : tensor<64x64x2x!tt.ptr<f32>>
      %l, %r = tt.split %v : tensor<64x64x2xf32> -> tensor<64x64xf32>
      ttg.warp_return
    } : (!tt.ptr<f32>) -> ()
    tt.return
  }
}
