// RUN: env LLVM_EXTRACT_DI_LOCAL_VARIABLES=1 triton-opt %s \
// RUN:   --deduplicate-debug-nops | FileCheck %s --check-prefix=DEDUP
//
// RUN: env LLVM_EXTRACT_DI_LOCAL_VARIABLES=0 triton-opt %s \
// RUN:   --deduplicate-debug-nops | FileCheck %s --check-prefix=NODEDUP

// Four NOPs at line 40 (different columns). Three at line 42.
// Expected after dedup: 1 NOP at line 40, 1 NOP at line 42.

module {
  func.func @kernel() {
    llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc(#loc_40_a)
    llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc(#loc_40_b)
    llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc(#loc_40_c)
    llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc(#loc_40_d)
    llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc(#loc_42_a)
    llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc(#loc_42_b)
    llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc(#loc_42_c)
    return
  } loc(#loc_fn)
}

#loc_fn   = loc("test.py":1:0)
#loc_40_a = loc("test.py":40:10)
#loc_40_b = loc("test.py":40:20)
#loc_40_c = loc("test.py":40:30)
#loc_40_d = loc("test.py":40:40)
#loc_42_a = loc("test.py":42:10)
#loc_42_b = loc("test.py":42:20)
#loc_42_c = loc("test.py":42:30)

// DEDUP:       func @kernel
// DEDUP-COUNT-2: llvm.inline_asm{{.*}}"nop"
// DEDUP-NOT:   llvm.inline_asm{{.*}}"nop"

// NODEDUP:       func @kernel
// NODEDUP-COUNT-7: llvm.inline_asm{{.*}}"nop"
