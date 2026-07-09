// RUN: triton-opt %s --verify-diagnostics -o /dev/null

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @invalid_non_shared_wgmma_shared_operand_fence(
      %dep: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
    // expected-error @+1 {{expects only shared-memory operands}}
    tle.wgmma_shared_operand_fence %dep {bCluster = false} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
