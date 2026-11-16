// RUN: triton-opt %s --canonicalize --cse --convert-triton-gpu-to-llvm="compute-capability=90 ptx-version=80" | FileCheck %s
// RUN: not triton-opt %s --convert-triton-gpu-to-llvm="compute-capability=80 ptx-version=80" 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: not triton-opt %s --convert-triton-gpu-to-llvm="compute-capability=90 ptx-version=78" 2>&1 | FileCheck %s --check-prefix=REJECT

// Both observable fences survive CSE even without pointer operands. They order
// memory through proxies; they neither rendezvous threads nor publish a map.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: llvm.func @global_data_proxy
  // CHECK: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.global>}
  // CHECK-NEXT: nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.global>}
  // CHECK-NEXT: llvm.return
  // REJECT: requires SM90 or newer and PTX ISA 8.0 or newer
  tt.func public @global_data_proxy() {
    "tle.fence_proxy_async_global"() : () -> ()
    "tle.fence_proxy_async_global"() : () -> ()
    tt.return
  }
}
