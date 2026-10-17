// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

// FlagCX's C bitcode ABI uses generic pointers and void barrier results.
// Global memory remains AS1 on either side of the external call boundary.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: llvm.func @flagcxDevCommGetIntraRank(!llvm.ptr) -> i32
  // CHECK-DAG: llvm.func @flagcxDevCommGetRank(!llvm.ptr) -> i32
  // CHECK-DAG: llvm.func @flagcxDevCommGetIntraSize(!llvm.ptr) -> i32
  // CHECK-DAG: llvm.func @flagcxGetIntraPointerC(!llvm.ptr, i64, i32) -> !llvm.ptr
  // CHECK-DAG: llvm.func @flagcxIntraBarrierArriveS(!llvm.ptr, i32, i32, i1, i32)
  // CHECK-DAG: llvm.func @flagcxIntraBarrierWaitS(!llvm.ptr, i32, i32, i1, i32)
  // CHECK-DAG: llvm.func @flagcxIntraBarrierSyncS(!llvm.ptr, i32, i32, i1, i32)
  // CHECK-LABEL: llvm.func @kernel
  // CHECK: llvm.inttoptr %{{.*}} : i64 to !llvm.ptr
  // CHECK: llvm.call @flagcxDevCommGetIntraRank({{.*}}) : (!llvm.ptr) -> i32
  // CHECK: llvm.call @flagcxDevCommGetRank({{.*}}) : (!llvm.ptr) -> i32
  // CHECK: llvm.call @flagcxDevCommGetIntraSize({{.*}}) : (!llvm.ptr) -> i32
  // CHECK: %[[PEER:.*]] = llvm.call @flagcxGetIntraPointerC({{.*}}) : (!llvm.ptr, i64, i32) -> !llvm.ptr
  // CHECK: llvm.addrspacecast %[[PEER]] : !llvm.ptr to !llvm.ptr<1>
  // CHECK: llvm.call @flagcxIntraBarrierArriveS({{.*}}) : (!llvm.ptr, i32, i32, i1, i32) -> ()
  // CHECK: llvm.call @flagcxIntraBarrierWaitS({{.*}}) : (!llvm.ptr, i32, i32, i1, i32) -> ()
  // CHECK: llvm.call @flagcxIntraBarrierSyncS({{.*}}) : (!llvm.ptr, i32, i32, i1, i32) -> ()
  tt.func public @kernel(%comm: i64, %memory: i64, %output: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i64
    %rank = "tle.get_device_id"(%comm) : (i64) -> i32
    %world = "tle.get_world_rank"(%comm) : (i64) -> i32
    %size = "tle.get_num_pes"(%comm) : (i64) -> i32
    %peer = "tle.remote_pointers"(%memory, %rank, %zero) <{operandSegmentSizes = array<i32: 1, 1, 1>, space = "device"}> : (i64, i32, i64) -> !tt.ptr<i32>
    %value = tt.load %peer : !tt.ptr<i32>
    %a = arith.addi %value, %world : i32
    %b = arith.addi %a, %size : i32
    tt.store %output, %b : !tt.ptr<i32>
    "tle.distributed_barrier"(%comm) <{barrier_index = 0 : i32, barrier_type = "arrive", group_kind = "block", order = "acqrel", space = "device"}> : (i64) -> ()
    "tle.distributed_barrier"(%comm) <{barrier_index = 0 : i32, barrier_type = "wait", group_kind = "block", order = "acqrel", space = "device"}> : (i64) -> ()
    "tle.distributed_barrier"(%comm) <{barrier_index = 0 : i32, barrier_type = "sync", group_kind = "block", order = "acqrel", space = "device"}> : (i64) -> ()
    tt.return
  }
}
