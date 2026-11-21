// Test: insertDebugNop emits llvm.inline_asm "nop" ops only when
// TRITON_DEBUG=1 and LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 is set, with the original source location preserved
// on each NOP. Without TRITON_DEBUG and LLVM_EXTRACT_DI_LOCAL_VARIABLES, no inline asm is emitted.
//
// RUN: triton-opt --triton-to-linalg --mlir-print-debuginfo --split-input-file %s | FileCheck %s --check-prefix=NODEBUG
// RUN: env TRITON_DEBUG=1 LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 triton-opt --triton-to-linalg --mlir-print-debuginfo --split-input-file %s | FileCheck %s --check-prefix=DEBUG

#loc = loc("test.py":6:0)
#loc1 = loc("test.py":11:24)
#loc2 = loc("test.py":12:32)
#loc3 = loc("test.py":16:28)
#loc4 = loc("test.py":18:34)
#loc_pid = loc("pid"(#loc1))
#loc_numprogs = loc("num_progs"(#loc2))
#loc_offsets = loc("offsets"(#loc3))
#loc_full = loc("scalar_one"(#loc4))

tt.func public @test_kernel(
    %ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %n: i32 {tt.divisibility = 16 : i32},
    %arg2: i32, %arg3: i32, %arg4: i32,
    %arg5: i32, %arg6: i32, %arg7: i32
) attributes {noinline = false} {
  %pid = tt.get_program_id x : i32 loc(#loc_pid)
  %np  = tt.get_num_programs x : i32 loc(#loc_numprogs)
  %c128 = arith.constant 128 : i32
  %block_start = arith.muli %pid, %c128 : i32
  %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %splat_bs = tt.splat %block_start : i32 -> tensor<128xi32>
  %abs_offs = arith.addi %splat_bs, %offs : tensor<128xi32> loc(#loc_offsets)
  %splat_ptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
  %addptr = tt.addptr %splat_ptr, %abs_offs : tensor<128x!tt.ptr<f32>>, tensor<128xi32> loc(#loc_offsets)
  // tl.full((128,), 1.0) lowers to: scalar constant + tt.splat to tensor<128xf32>.
  // Without TRITON_DEBUG and LLVM_EXTRACT_DI_LOCAL_VARIABLES, SplatOp::fold collapses this into a dense<1.0> constant
  // and the loc on the splat is lost. With TRITON_DEBUG=1 and LLVM_EXTRACT_DI_LOCAL_VARIABLES=1, the fold is suppressed
  // and SplatConverter emits a NOP carrying loc("scalar_one").
  %cst_one = arith.constant 1.000000e+00 : f32
  %full = tt.splat %cst_one : f32 -> tensor<128xf32> loc(#loc_full)
  %sum = arith.addi %pid, %np : i32
  %splat_sum = tt.splat %sum : i32 -> tensor<128xi32>
  %sum_f32 = arith.sitofp %splat_sum : tensor<128xi32> to tensor<128xf32>
  %final = arith.addf %sum_f32, %full : tensor<128xf32>
  tt.store %addptr, %final : tensor<128x!tt.ptr<f32>>
  tt.return
} loc(#loc)

// DEBUG-LABEL: func.func @test_kernel
// DEBUG-COUNT-5: llvm.inline_asm has_side_effects asm_dialect = att "nop"
// DEBUG-DAG: loc("pid"
// DEBUG-DAG: loc("num_progs"
// DEBUG-DAG: loc("offsets"
// DEBUG-DAG: loc("scalar_one"

// NODEBUG-LABEL: func.func @test_kernel
// NODEBUG-NOT: llvm.inline_asm
