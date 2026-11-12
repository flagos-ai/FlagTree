// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -tritongpu-assign-latencies=num-stages=3 -tritongpu-schedule-loops -tritongpu-pipeline="num-stages=3 enable-pipeline-enhance=true" -canonicalize | FileCheck %s --check-prefixes=COMMON,OFF,NOCARRY
// RUN: triton-opt %s -tritongpu-assign-latencies=num-stages=3 -tritongpu-schedule-loops -tritongpu-pipeline="num-stages=3 enable-pipeline-enhance=true hoist-async-copy=true wgmma-carry=true" -canonicalize | FileCheck %s --check-prefixes=COMMON,HOIST,CARRY
// The two options only act together with enable-pipeline-enhance.
// RUN: triton-opt %s -tritongpu-assign-latencies=num-stages=3 -tritongpu-schedule-loops -tritongpu-pipeline="num-stages=3 hoist-async-copy=true wgmma-carry=true" -canonicalize | FileCheck %s --check-prefixes=COMMON,OFF,NOCARRY

#AL = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#NL = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#KL = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma32 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#mma64 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The block-scale epilogue uses the wgmma result inside the iteration, so the
  // scheduler waits for it with pendings = 0. The prefetch of the next block
  // does not depend on the result and writes a slot the in-flight wgmma does
  // not read: with hoist-async-copy it is issued before the wait.
  // COMMON-LABEL: @block_scale_gemm
  // COMMON: scf.for
  // COMMON: ttng.warp_group_dot %
  // COMMON-NEXT: ttng.warp_group_dot_commit
  // HOIST: ttg.async_copy_global_to_local
  // HOIST: ttg.async_copy_global_to_local
  // COMMON: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // COMMON-NEXT: arith.mulf
  // OFF: ttg.async_copy_global_to_local
  // OFF: ttg.async_copy_global_to_local
  // COMMON: scf.yield
  tt.func public @block_scale_gemm(%lb : i32, %ub : i32, %step : i32,
      %a_ptr_init : tensor<128x64x!tt.ptr<f16>, #AL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %b_ptr_init : tensor<64x128x!tt.ptr<f16>, #BL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 128]> : tensor<2xi32>},
      %scale : tensor<128x128xf32, #mma>) -> tensor<128x128xf32, #mma> {
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %a_off = arith.constant dense<64> : tensor<128x64xi32, #AL>
    %b_off = arith.constant dense<8192> : tensor<64x128xi32, #BL>
    %res:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %acc = %zero) -> (tensor<128x64x!tt.ptr<f16>, #AL>, tensor<64x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) : i32 {
      %a_ = tt.load %a_ptr : tensor<128x64x!tt.ptr<f16>, #AL>
      %a = ttg.local_alloc %a_ : (tensor<128x64xf16, #AL>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_ = tt.load %b_ptr : tensor<64x128x!tt.ptr<f16>, #BL>
      %b = ttg.local_alloc %b_ : (tensor<64x128xf16, #BL>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %d = ttng.warp_group_dot %a, %b, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
      %ds = arith.mulf %d, %scale : tensor<128x128xf32, #mma>
      %acc2 = arith.addf %acc, %ds : tensor<128x128xf32, #mma>
      %a_next = tt.addptr %a_ptr, %a_off : tensor<128x64x!tt.ptr<f16>, #AL>, tensor<128x64xi32, #AL>
      %b_next = tt.addptr %b_ptr, %b_off : tensor<64x128x!tt.ptr<f16>, #BL>, tensor<64x128xi32, #BL>
      scf.yield %a_next, %b_next, %acc2 : tensor<128x64x!tt.ptr<f16>, #AL>, tensor<64x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
    }
    tt.return %res#2 : tensor<128x128xf32, #mma>
  }

  // The accumulator is only passed to the next iteration: with wgmma-carry the
  // group of this iteration stays in flight (pendings = 1 after the commit,
  // before the prefetch writes shared memory) and is waited for after the loop.
  // COMMON-LABEL: @plain_gemm
  // COMMON: scf.for
  // COMMON: ttng.warp_group_dot %
  // COMMON-NEXT: ttng.warp_group_dot_commit
  // CARRY-NEXT: ttng.warp_group_dot_wait {{.*}} {pendings = 1 : i32}
  // NOCARRY-NOT: pendings = 1
  // COMMON: ttg.async_copy_global_to_local
  // COMMON: ttg.async_copy_global_to_local
  // NOCARRY: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // COMMON: scf.yield
  // CARRY: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  tt.func public @plain_gemm(%lb : i32, %ub : i32, %step : i32,
      %a_ptr_init : tensor<128x64x!tt.ptr<f16>, #AL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %b_ptr_init : tensor<64x128x!tt.ptr<f16>, #BL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 128]> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %a_off = arith.constant dense<64> : tensor<128x64xi32, #AL>
    %b_off = arith.constant dense<8192> : tensor<64x128xi32, #BL>
    %res:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %acc = %zero) -> (tensor<128x64x!tt.ptr<f16>, #AL>, tensor<64x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) : i32 {
      %a_ = tt.load %a_ptr : tensor<128x64x!tt.ptr<f16>, #AL>
      %a = ttg.local_alloc %a_ : (tensor<128x64xf16, #AL>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_ = tt.load %b_ptr : tensor<64x128x!tt.ptr<f16>, #BL>
      %b = ttg.local_alloc %b_ : (tensor<64x128xf16, #BL>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %d = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
      %a_next = tt.addptr %a_ptr, %a_off : tensor<128x64x!tt.ptr<f16>, #AL>, tensor<128x64xi32, #AL>
      %b_next = tt.addptr %b_ptr, %b_off : tensor<64x128x!tt.ptr<f16>, #BL>, tensor<64x128xi32, #BL>
      scf.yield %a_next, %b_next, %d : tensor<128x64x!tt.ptr<f16>, #AL>, tensor<64x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
    }
    tt.return %res#2 : tensor<128x128xf32, #mma>
  }

  // BLOCK_N < 64: the carry does not pay off and is not applied.
  // COMMON-LABEL: @narrow_gemm
  // COMMON: scf.for
  // COMMON-NOT: pendings = 1
  // COMMON: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // COMMON-NEXT: scf.yield
  tt.func public @narrow_gemm(%lb : i32, %ub : i32, %step : i32,
      %a_ptr_init : tensor<128x64x!tt.ptr<f16>, #AL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %b_ptr_init : tensor<64x32x!tt.ptr<f16>, #NL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 32]> : tensor<2xi32>}) -> tensor<128x32xf32, #mma32> {
    %zero = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #mma32>
    %a_off = arith.constant dense<64> : tensor<128x64xi32, #AL>
    %b_off = arith.constant dense<2048> : tensor<64x32xi32, #NL>
    %res:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %acc = %zero) -> (tensor<128x64x!tt.ptr<f16>, #AL>, tensor<64x32x!tt.ptr<f16>, #NL>, tensor<128x32xf32, #mma32>) : i32 {
      %a_ = tt.load %a_ptr : tensor<128x64x!tt.ptr<f16>, #AL>
      %a = ttg.local_alloc %a_ : (tensor<128x64xf16, #AL>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %b_ = tt.load %b_ptr : tensor<64x32x!tt.ptr<f16>, #NL>
      %b = ttg.local_alloc %b_ : (tensor<64x32xf16, #NL>) -> !ttg.memdesc<64x32xf16, #shared64, #smem>
      %d = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x32xf16, #shared64, #smem> -> tensor<128x32xf32, #mma32>
      %a_next = tt.addptr %a_ptr, %a_off : tensor<128x64x!tt.ptr<f16>, #AL>, tensor<128x64xi32, #AL>
      %b_next = tt.addptr %b_ptr, %b_off : tensor<64x32x!tt.ptr<f16>, #NL>, tensor<64x32xi32, #NL>
      scf.yield %a_next, %b_next, %d : tensor<128x64x!tt.ptr<f16>, #AL>, tensor<64x32x!tt.ptr<f16>, #NL>, tensor<128x32xf32, #mma32>
    }
    tt.return %res#2 : tensor<128x32xf32, #mma32>
  }

  // The second wgmma consumes the first one of the same iteration (S -> P -> PV):
  // the prefetch stays after the first wait.
  // COMMON-LABEL: @attention_chain
  // COMMON: scf.for
  // COMMON: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // COMMON-NEXT: arith.truncf
  // COMMON: ttng.warp_group_dot %
  // COMMON: ttg.async_copy_global_to_local
  // COMMON: ttg.async_copy_global_to_local
  // COMMON: ttng.warp_group_dot_wait {{.*}} {pendings = 0 : i32}
  // COMMON-NEXT: scf.yield
  tt.func public @attention_chain(%lb : i32, %ub : i32, %step : i32, %q : !ttg.memdesc<128x64xf16, #shared, #smem>,
      %k_ptr_init : tensor<64x64x!tt.ptr<f16>, #KL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>},
      %v_ptr_init : tensor<64x128x!tt.ptr<f16>, #BL> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 128]> : tensor<2xi32>}) -> tensor<128x128xf32, #mma> {
    %zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %zs = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma64>
    %k_off = arith.constant dense<4096> : tensor<64x64xi32, #KL>
    %v_off = arith.constant dense<8192> : tensor<64x128xi32, #BL>
    %res:3 = scf.for %iv = %lb to %ub step %step iter_args(%k_ptr = %k_ptr_init, %v_ptr = %v_ptr_init, %o = %zero) -> (tensor<64x64x!tt.ptr<f16>, #KL>, tensor<64x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) : i32 {
      %k_ = tt.load %k_ptr : tensor<64x64x!tt.ptr<f16>, #KL>
      %k = ttg.local_alloc %k_ : (tensor<64x64xf16, #KL>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %s = ttng.warp_group_dot %q, %k, %zs {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma64>
      %p = arith.truncf %s : tensor<128x64xf32, #mma64> to tensor<128x64xf16, #mma64>
      %pa = ttg.convert_layout %p : tensor<128x64xf16, #mma64> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %v_ = tt.load %v_ptr : tensor<64x128x!tt.ptr<f16>, #BL>
      %v = ttg.local_alloc %v_ : (tensor<64x128xf16, #BL>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      %o2 = ttng.warp_group_dot %pa, %v, %o {inputPrecision = 0 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x128xf16, #shared, #smem> -> tensor<128x128xf32, #mma>
      %k_next = tt.addptr %k_ptr, %k_off : tensor<64x64x!tt.ptr<f16>, #KL>, tensor<64x64xi32, #KL>
      %v_next = tt.addptr %v_ptr, %v_off : tensor<64x128x!tt.ptr<f16>, #BL>, tensor<64x128xi32, #BL>
      scf.yield %k_next, %v_next, %o2 : tensor<64x64x!tt.ptr<f16>, #KL>, tensor<64x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
    }
    tt.return %res#2 : tensor<128x128xf32, #mma>
  }
}
