// RUN: triton-opt %s -split-input-file --tritongpu-global-scratch-memory-allocation --convert-triton-gpu-to-llvm="compute-capability=90 ptx-version=87" -reconcile-unrealized-casts | FileCheck %s --check-prefix=DEFER --implicit-check-not=ld.acquire.gpu
// RUN: triton-opt %s -split-input-file --tritongpu-global-scratch-memory-allocation --convert-triton-gpu-to-llvm="compute-capability=90 ptx-version=85" -reconcile-unrealized-casts | FileCheck %s --check-prefix=LEGACY --implicit-check-not=ld.relaxed.gpu --implicit-check-not=fence.acquire.gpu
// RUN: triton-opt %s -split-input-file --tritongpu-global-scratch-memory-allocation --convert-triton-gpu-to-llvm="compute-capability=80 ptx-version=87" -reconcile-unrealized-casts | FileCheck %s --check-prefix=LEGACY --implicit-check-not=ld.relaxed.gpu --implicit-check-not=fence.acquire.gpu

// The successful strong read must precede the acquire fence. The retry edge
// goes directly back to the relaxed read; nonleaders join only at the original
// trailing CTA barrier. Both full-grid and partitioned groups use this protocol.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // DEFER-LABEL: llvm.func @grid
  // DEFER: nvvm.barrier0
  // DEFER: atom.add.release.gpu.u32
  // DEFER: llvm.br ^[[WAIT:bb[0-9]+]](
  // DEFER: ^[[WAIT]](%[[OLD:[0-9]+]]: i32)
  // DEFER-NEXT: %[[READ:[0-9]+]] = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.u32
  // DEFER-NEXT: %[[XOR:[0-9]+]] = llvm.xor %[[OLD]], %[[READ]] : i32
  // DEFER: llvm.cond_br {{.*}}, ^[[SUCCESS:bb[0-9]+]], ^[[WAIT]](%[[OLD]] : i32)
  // DEFER: ^[[SUCCESS]]:
  // DEFER-NEXT: llvm.inline_asm has_side_effects {{.*}}"fence.acquire.gpu{{ *}};"
  // DEFER-NEXT: llvm.br ^[[JOIN:bb[0-9]+]]
  // DEFER-NEXT: ^[[JOIN]]:
  // DEFER-NEXT: nvvm.barrier0
  // DEFER-NEXT: llvm.return
  // LEGACY-LABEL: llvm.func @grid
  // LEGACY: nvvm.barrier0
  // LEGACY: atom.add.release.gpu.u32
  // LEGACY: ld.acquire.gpu.u32
  // LEGACY: nvvm.barrier0
  // LEGACY: llvm.return
  tt.func public @grid() {
    "tle.distributed_barrier"() <{group_kind = "grid"}> : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // DEFER-LABEL: llvm.func @axis8
  // DEFER: nvvm.barrier0
  // DEFER: atom.add.release.gpu.u32
  // DEFER: ld.relaxed.gpu.u32
  // DEFER: llvm.cond_br
  // DEFER: llvm.inline_asm has_side_effects {{.*}}"fence.acquire.gpu{{ *}};"
  // DEFER: nvvm.barrier0
  // DEFER: llvm.return
  // LEGACY-LABEL: llvm.func @axis8
  // LEGACY: nvvm.barrier0
  // LEGACY: atom.add.release.gpu.u32
  // LEGACY: ld.acquire.gpu.u32
  // LEGACY: nvvm.barrier0
  // LEGACY: llvm.return
  tt.func public @axis8() {
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 8>, group_axes = array<i32: 0>, group_domain_shape = array<i32: 8, 16>}> : () -> ()
    tt.return
  }
}
