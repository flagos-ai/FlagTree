// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-nv-gpu-to-llvm | FileCheck %s --check-prefix=LLVM

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_nonzero_acc_precedes_wgmma_fence
  // CHECK: llvm.insertvalue
  // CHECK: nvvm.wgmma.fence.aligned
  // CHECK-NOT: llvm.insertvalue
  // CHECK: nvg.wgmma
  // LLVM-LABEL: @dot_nonzero_acc_precedes_wgmma_fence
  // LLVM: "wgmma.fence.sync.aligned;\0A\09wgmma.mma_async.sync.aligned
  tt.func @dot_nonzero_acc_precedes_wgmma_fence(%a: !ttg.memdesc<64x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x64xf16, #shared, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc { inputPrecision = 0 : i32 }:
      !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
