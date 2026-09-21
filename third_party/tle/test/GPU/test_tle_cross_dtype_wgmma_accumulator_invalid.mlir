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

// RUN: split-file %s %t
// RUN: not triton-opt %t/unmarked_warps.mlir --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' 2>&1 | FileCheck %s --check-prefix=LAYOUT
// RUN: not triton-opt %t/marked_warps.mlir --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' 2>&1 | FileCheck %s --check-prefix=LAYOUT
// RUN: not triton-opt %t/mismatched_types.mlir --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' 2>&1 | FileCheck %s --check-prefix=TYPES
// RUN: not triton-opt %t/ieee_precision.mlir --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' 2>&1 | FileCheck %s --check-prefix=TYPES

// LAYOUT: error: incompatible register-A and accumulator layouts for Hopper WGMMA
// TYPES: error: unsupported operand types or precision for Hopper WGMMA instruction shape

//--- unmarked_warps.mlir
#acc = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#current = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #current, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @different_warp_distribution(%a: tensor<128x64xbf16, #dot>, %b: !ttg.memdesc<64x128xbf16, #shared, #smem>, %c: tensor<128x128xf32, #acc>) {
    %d = ttng.warp_group_dot %a, %b, %c {inputPrecision = 0 : i32, isAsync = true } : tensor<128x64xbf16, #dot> * !ttg.memdesc<64x128xbf16, #shared, #smem> -> tensor<128x128xf32, #acc>
    tt.return
  }
}

//--- marked_warps.mlir
#acc = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#current = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #current, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @different_warp_distribution(%a: tensor<128x64xbf16, #dot>, %b: !ttg.memdesc<64x128xbf16, #shared, #smem>, %c: tensor<128x128xf32, #acc>) {
    %d = ttng.warp_group_dot %a, %b, %c {inputPrecision = 0 : i32, isAsync = true, tle.wgmma_accumulator_chain_c } : tensor<128x64xbf16, #dot> * !ttg.memdesc<64x128xbf16, #shared, #smem> -> tensor<128x128xf32, #acc>
    tt.return
  }
}

//--- mismatched_types.mlir
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#sa = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_f16_bf16(%a: !ttg.memdesc<64x64xf16, #sa, #smem>, %b: !ttg.memdesc<64x64xbf16, #sb, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %res = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 2147483647 : i32} : !ttg.memdesc<64x64xf16, #sa, #smem> * !ttg.memdesc<64x64xbf16, #sb, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

//--- ieee_precision.mlir
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#sa = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#sb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_ieee(%a: !ttg.memdesc<64x64xf32, #sa, #smem>, %b: !ttg.memdesc<64x64xf32, #sb, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %res = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 2 : i32, isAsync = true, maxNumImpreciseAcc = 2147483647 : i32} : !ttg.memdesc<64x64xf32, #sa, #smem> * !ttg.memdesc<64x64xf32, #sb, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
