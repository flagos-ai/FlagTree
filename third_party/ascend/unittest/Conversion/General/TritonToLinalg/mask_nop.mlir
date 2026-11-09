// RUN: triton-opt --triton-to-linalg --mlir-print-debuginfo --split-input-file %s \
// RUN:   | FileCheck %s --check-prefix=NODEBUG
// RUN: env LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 triton-opt --triton-to-linalg \
// RUN:   --mlir-print-debuginfo --split-input-file %s | FileCheck %s --check-prefix=DEBUG

#loc     = loc("mask.py":1:0)
#loc_off = loc("mask.py":10:11)
#loc_mcmp = loc("mask.py":11:16)
#loc_msk = loc("offs_lt_n"(#loc_mcmp))

tt.func public @mask_kernel(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                            %out: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                            %n: i32) {
  %offs = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32> loc(#loc_off)
  %ns   = tt.splat %n : i32 -> tensor<128xi32>
  %mask = arith.cmpi slt, %offs, %ns : tensor<128xi32> loc(#loc_msk)
  %xptr = tt.splat %x   : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %optr = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %xp   = tt.addptr %xptr, %offs : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %op   = tt.addptr %optr, %offs : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  %v = tt.load %xp, %mask : tensor<128x!tt.ptr<f32>>
  tt.store %op, %v, %mask : tensor<128x!tt.ptr<f32>>
  tt.return
} loc(#loc)

// DEBUG-LABEL: func.func @mask_kernel
// DEBUG: llvm.inline_asm has_side_effects asm_dialect = att "nop"
// DEBUG-DAG: loc("mask.py":11
// The mask line (11) must appear as an anchored nop, not only as the cmpi.

// NODEBUG-LABEL: func.func @mask_kernel
// NODEBUG-NOT: llvm.inline_asm
