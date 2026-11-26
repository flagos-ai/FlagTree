// RUN: triton-opt %s -split-input-file --normalize-debug-line-locations --allow-unregistered-dialect --mlir-print-debuginfo | FileCheck %s

//===----------------------------------------------------------------------===//
// control_ops
//
// Verifies that control-flow operations remain user-visible debug anchors.
// cf.br, scf.if, scf.for, and func.return must be classified as
// "control" and must keep their original real source locations.
//===----------------------------------------------------------------------===//

module {
func.func @control_ops(%cond: i1) {
cf.br ^bb1 loc("control.py":2:3)
^bb1:
scf.if %cond {
} else {
} loc("control.py":3:5)
%c0 = arith.constant 0 : index loc("control.py":4:5)
%c1 = arith.constant 1 : index loc("control.py":4:9)
scf.for %i = %c0 to %c1 step %c1 {
scf.yield loc("control.py":5:9)
} loc("control.py":5:5)
func.return loc("control.py":6:3)
}
}

// CHECK-LABEL: func.func @control_ops
// CHECK: cf.br
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[CONTROL_BR_LOC:[A-Za-z0-9_]+]])
// CHECK: scf.if
// CHECK: } {triton.debug_line.class = "control"} loc(#[[CONTROL_IF_LOC:[A-Za-z0-9_]+]])
// CHECK: scf.for
// CHECK: } {triton.debug_line.class = "control"} loc(#[[CONTROL_FOR_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[CONTROL_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[CONTROL_BR_LOC]] = loc("control.py":2:3)
// CHECK-DAG: #[[CONTROL_IF_LOC]] = loc("control.py":3:5)
// CHECK-DAG: #[[CONTROL_FOR_LOC]] = loc("control.py":5:5)
// CHECK-DAG: #[[CONTROL_RETURN_LOC]] = loc("control.py":6:3)

// -----

//===----------------------------------------------------------------------===//
// synthetic_glue
//
// Verifies the default handling of helper/glue operations that do not have a
// unique user-visible store anchor. unrealized_conversion_cast and
// memref.subview must be classified as "synthetic", preserve their original
// location in triton.debug_line.origin, and move to glue.py:0:0. The
// surrounding control operations must remain at their real source locations.
//===----------------------------------------------------------------------===//

module {
func.func @synthetic_glue(%arg0: memref<8xf32>, %idx: index, %cond: i1) {
%cast = builtin.unrealized_conversion_cast %idx : index to i64 loc("glue.py":2:5)
%sub = memref.subview %arg0[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("glue.py":3:5)
scf.if %cond {
} loc("glue.py":4:5)
func.return loc("glue.py":5:3)
}
}

// CHECK-DAG: #[[$GLUE_CAST_ORIGIN:[A-Za-z0-9_]+]] = loc("glue.py":2:5)
// CHECK-DAG: #[[$GLUE_SUBVIEW_ORIGIN:[A-Za-z0-9_]+]] = loc("glue.py":3:5)

// CHECK-LABEL: func.func @synthetic_glue
// CHECK: builtin.unrealized_conversion_cast
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$GLUE_CAST_ORIGIN]]
// CHECK-SAME: loc(#[[GLUE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: memref.subview
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$GLUE_SUBVIEW_ORIGIN]]
// CHECK-SAME: : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[GLUE_SYNTH_LOC]])
// CHECK: scf.if
// CHECK: } {triton.debug_line.class = "control"} loc(#[[GLUE_IF_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[GLUE_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[GLUE_SYNTH_LOC]] = loc("glue.py":0:0)
// CHECK-DAG: #[[GLUE_IF_LOC]] = loc("glue.py":4:5)
// CHECK-DAG: #[[GLUE_RETURN_LOC]] = loc("glue.py":5:3)

// -----

//===----------------------------------------------------------------------===//
// floating_point_conversion_classification
//
// Verifies that an arith.extf/truncf on its own source line remains semantic,
// while the same operations are synthetic when a load/store already provides
// the semantic anchor for that source line.
//===----------------------------------------------------------------------===//

module {
func.func @floating_point_conversion_classification(%src: memref<4xf16>, %dst: memref<4xf16>, %wide: f32) {
%c0 = arith.constant 0 : index loc("fp_conversion.py":10:3)
%loaded = memref.load %src[%c0] : memref<4xf16> loc("fp_conversion.py":10:5)
%aux_ext = arith.extf %loaded : f16 to f32 loc("fp_conversion.py":10:20)
%standalone_ext = arith.extf %loaded : f16 to f32 loc("fp_conversion.py":11:5)
%standalone_trunc = arith.truncf %standalone_ext : f32 to f16 loc("fp_conversion.py":12:5)
%aux_trunc = arith.truncf %wide : f32 to f16 loc("fp_conversion.py":13:5)
memref.store %aux_trunc, %dst[%c0] : memref<4xf16> loc("fp_conversion.py":13:20)
func.return loc("fp_conversion.py":14:3)
}
}

// CHECK-DAG: #[[$FP_AUX_EXT_ORIGIN:[A-Za-z0-9_]+]] = loc("fp_conversion.py":10:20)
// CHECK-DAG: #[[$FP_AUX_TRUNC_ORIGIN:[A-Za-z0-9_]+]] = loc("fp_conversion.py":13:5)

// CHECK-LABEL: func.func @floating_point_conversion_classification
// CHECK: %[[FP_LOADED:[A-Za-z0-9_]+]] = memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FP_LOAD_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[FP_AUX_EXT:[A-Za-z0-9_]+]] = arith.extf %[[FP_LOADED]]
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$FP_AUX_EXT_ORIGIN]]
// CHECK-SAME: loc(#[[FP_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[FP_STANDALONE_EXT:[A-Za-z0-9_]+]] = arith.extf %[[FP_LOADED]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FP_STANDALONE_EXT_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[FP_STANDALONE_TRUNC:[A-Za-z0-9_]+]] = arith.truncf %[[FP_STANDALONE_EXT]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FP_STANDALONE_TRUNC_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[FP_AUX_TRUNC:[A-Za-z0-9_]+]] = arith.truncf
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$FP_AUX_TRUNC_ORIGIN]]
// CHECK-SAME: loc(#[[FP_SYNTH_LOC]])
// CHECK: memref.store %[[FP_AUX_TRUNC]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FP_STORE_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[FP_SYNTH_LOC]] = loc("fp_conversion.py":0:0)
// CHECK-DAG: #[[FP_LOAD_LOC]] = loc("fp_conversion.py":10:5)
// CHECK-DAG: #[[FP_STANDALONE_EXT_LOC]] = loc("fp_conversion.py":11:5)
// CHECK-DAG: #[[FP_STANDALONE_TRUNC_LOC]] = loc("fp_conversion.py":12:5)
// CHECK-DAG: #[[FP_STORE_LOC]] = loc("fp_conversion.py":13:20)

// -----

//===----------------------------------------------------------------------===//
// source_line_write_anchor
//
// Verifies the main write-anchor normalization case modeled after the loop
// reordering problem. Address/view computations are retargeted only when their
// original source line matches the user-visible store line. The ticket index is
// produced on line 23 and must therefore remain synthetic instead of being
// retargeted to the store on line 24. Same-line view and tensor.insert paths may
// still be retargeted. The test also covers NameLoc preservation for the load.
//===----------------------------------------------------------------------===//

module {
func.func @source_line_write_anchor(%trace: memref<8xi32>, %counter: memref<1xi32>) {
%c0 = arith.constant 0 : index loc("loop_reordering.py":23:21)
%ticket = memref.load %counter[%c0] : memref<1xi32> loc("ticket"("loop_reordering.py":23:21))
%ticket_idx = arith.index_cast %ticket : i32 to index loc("ticket"("loop_reordering.py":23:21))
%c200_i32 = arith.constant 200 : i32 loc("loop_reordering.py":24:33)
%empty = tensor.empty() : tensor<1xi32> loc("loop_reordering.py":24:33)
%filled = linalg.fill ins(%c200_i32 : i32) outs(%empty : tensor<1xi32>) -> tensor<1xi32> loc("loop_reordering.py":24:33)
%view = memref.reinterpret_cast %trace to offset: [%ticket_idx], sizes: [1], strides: [1] : memref<8xi32> to memref<1xi32, strided<[1], offset: ?>> loc("loop_reordering.py":24:33)
bufferization.materialize_in_destination %filled in writable %view : (tensor<1xi32>, memref<1xi32, strided<[1], offset: ?>>) -> () loc("loop_reordering.py":24:33)
%c1_i32 = arith.constant 1 : i32 loc("loop_reordering.py":25:35)
%next_ticket = arith.addi %ticket, %c1_i32 : i32 loc("loop_reordering.py":25:35)
%inserted = tensor.insert %next_ticket into %empty[%c0] : tensor<1xi32> loc("loop_reordering.py":25:26)
bufferization.materialize_in_destination %inserted in writable %counter : (tensor<1xi32>, memref<1xi32>) -> () loc("loop_reordering.py":25:26)
func.return loc("loop_reordering.py":26:3)
}
}

// CHECK-DAG: #[[$LOOP_TICKET_FILE_LOC:[A-Za-z0-9_]+]] = loc("loop_reordering.py":23:21)
// CHECK-DAG: #[[$LOOP_TICKET_LOC:[A-Za-z0-9_]+]] = loc("ticket"(#[[$LOOP_TICKET_FILE_LOC]]))
// CHECK-DAG: #[[$LOOP_WRITE_LOC:[A-Za-z0-9_]+]] = loc("loop_reordering.py":24:33)
// CHECK-DAG: #[[$LOOP_NEXT_VALUE_LOC:[A-Za-z0-9_]+]] = loc("loop_reordering.py":25:35)

// CHECK-LABEL: func.func @source_line_write_anchor
// CHECK: %[[LOOP_TICKET:[A-Za-z0-9_]+]] = memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : memref<1xi32>
// CHECK-SAME: loc(#[[$LOOP_TICKET_LOC]])

// CHECK: %[[LOOP_TICKET_IDX:[A-Za-z0-9_]+]] = arith.index_cast %[[LOOP_TICKET]]
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$LOOP_TICKET_LOC]]
// CHECK-SAME: : i32 to index
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC:[A-Za-z0-9_]+]])

// CHECK: %[[LOOP_C200:[A-Za-z0-9_]+]] = arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$LOOP_WRITE_LOC]]
// CHECK-SAME: 200 : i32
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC]])

// CHECK: %[[LOOP_EMPTY:[A-Za-z0-9_]+]] = tensor.empty
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$LOOP_WRITE_LOC]]
// CHECK-SAME: tensor<1xi32>
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC]])

// CHECK: %[[LOOP_FILLED:[A-Za-z0-9_]+]] = linalg.fill
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$LOOP_WRITE_LOC]]
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC]])

// CHECK: %[[LOOP_VIEW:[A-Za-z0-9_]+]] = memref.reinterpret_cast
// CHECK-SAME: %[[LOOP_TICKET_IDX]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: triton.debug_line.origin = #[[$LOOP_WRITE_LOC]]
// CHECK-SAME: memref<8xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[$LOOP_WRITE_LOC]])

// CHECK: bufferization.materialize_in_destination %[[LOOP_FILLED]] in writable %[[LOOP_VIEW]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[$LOOP_WRITE_LOC]])

// CHECK: %[[LOOP_C1:[A-Za-z0-9_]+]] = arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$LOOP_NEXT_VALUE_LOC]]
// CHECK-SAME: 1 : i32
// CHECK-SAME: loc(#[[LOOP_SYNTH_LOC]])

// CHECK: %[[LOOP_NEXT:[A-Za-z0-9_]+]] = arith.addi %[[LOOP_TICKET]], %[[LOOP_C1]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: triton.debug_line.origin = #[[$LOOP_NEXT_VALUE_LOC]]
// CHECK-SAME: : i32
// CHECK-SAME: loc(#[[LOOP_NEXT_STORE_LOC:[A-Za-z0-9_]+]])

// CHECK: %[[LOOP_INSERTED:[A-Za-z0-9_]+]] = tensor.insert %[[LOOP_NEXT]] into %[[LOOP_EMPTY]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: tensor<1xi32>
// CHECK-SAME: loc(#[[LOOP_NEXT_STORE_LOC]])

// CHECK: bufferization.materialize_in_destination %[[LOOP_INSERTED]] in writable
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[LOOP_NEXT_STORE_LOC]])

// CHECK-DAG: #[[LOOP_SYNTH_LOC]] = loc("loop_reordering.py":0:0)
// CHECK-DAG: #[[LOOP_NEXT_STORE_LOC]] = loc("loop_reordering.py":25:26)

// -----

//===----------------------------------------------------------------------===//
// relocate_unique_load_debug_nop
//
// Verifies that a debug NOP is moved immediately before the unique memref.copy
// carrying the same source line. The helper constant originally between the
// NOP and the copy remains in place and becomes synthetic.
//===----------------------------------------------------------------------===//

module {
func.func @relocate_unique_load_debug_nop(%src: memref<4xf32>, %dst: memref<4xf32>) {
llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc("load.py":10:5)
%c0 = arith.constant 0 : i32 loc("load.py":10:10)
memref.copy %src, %dst : memref<4xf32> to memref<4xf32> loc("load.py":10:20)
func.return loc("load.py":11:3)
}
}

// CHECK-LABEL: func.func @relocate_unique_load_debug_nop
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-NEXT: llvm.inline_asm
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[LOAD_NOP_LOC:[A-Za-z0-9_]+]])
// CHECK-NEXT: memref.copy
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[LOAD_COPY_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[LOAD_NOP_LOC]] = loc("load.py":10:5)
// CHECK-DAG: #[[LOAD_COPY_LOC]] = loc("load.py":10:20)

// -----

//===----------------------------------------------------------------------===//
// keep_ambiguous_load_debug_nop
//
// Verifies that a debug NOP is not moved when multiple memref.copy anchors use
// the same source line. Relocation is skipped because the matching load anchor
// is ambiguous.
//===----------------------------------------------------------------------===//

module {
func.func @keep_ambiguous_load_debug_nop(%src0: memref<4xf32>, %src1: memref<4xf32>, %dst0: memref<4xf32>, %dst1: memref<4xf32>) {
llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc("ambiguous_load.py":20:5)
%c0 = arith.constant 0 : i32 loc("ambiguous_load.py":20:10)
memref.copy %src0, %dst0 : memref<4xf32> to memref<4xf32> loc("ambiguous_load.py":20:20)
memref.copy %src1, %dst1 : memref<4xf32> to memref<4xf32> loc("ambiguous_load.py":20:30)
func.return loc("ambiguous_load.py":21:3)
}
}

// CHECK-LABEL: func.func @keep_ambiguous_load_debug_nop
// CHECK: llvm.inline_asm
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[AMBIGUOUS_LOAD_NOP_LOC:[A-Za-z0-9_]+]])
// CHECK-NEXT: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-NEXT: memref.copy
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-NEXT: memref.copy
// CHECK-SAME: triton.debug_line.class = "semantic"

// CHECK-DAG: #[[AMBIGUOUS_LOAD_NOP_LOC]] = loc("ambiguous_load.py":20:5)

// -----

//===----------------------------------------------------------------------===//
// demote_redundant_store_debug_nop
//
// Verifies that a debug NOP becomes synthetic when the same source line already
// has a materialize_in_destination store anchor. The store remains semantic and
// the redundant NOP is moved to store.py:0:0.
//===----------------------------------------------------------------------===//

module {
func.func @demote_redundant_store_debug_nop(%src: tensor<4xf32>, %dst: memref<4xf32>) {
llvm.inline_asm has_side_effects asm_dialect = att "nop", "" : () -> () loc("store.py":30:5)
bufferization.materialize_in_destination %src in writable %dst : (tensor<4xf32>, memref<4xf32>) -> () loc("store.py":30:20)
func.return loc("store.py":31:3)
}
}

// CHECK-DAG: #[[$STORE_NOP_ORIGIN:[A-Za-z0-9_]+]] = loc("store.py":30:5)

// CHECK-LABEL: func.func @demote_redundant_store_debug_nop
// CHECK: llvm.inline_asm
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$STORE_NOP_ORIGIN]]
// CHECK-SAME: loc(#[[STORE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: bufferization.materialize_in_destination
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[STORE_ANCHOR_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[STORE_SYNTH_LOC]] = loc("store.py":0:0)
// CHECK-DAG: #[[STORE_ANCHOR_LOC]] = loc("store.py":30:20)

// -----

//===----------------------------------------------------------------------===//
// memref_fill_anchor
//
// Verifies that linalg.fill writing directly to a memref is treated as a real
// semantic memory write anchor. The scalar constant used as the fill value is
// synthetic and is moved to memfill.py:0:0, while the memref fill keeps its
// user-visible source location.
//===----------------------------------------------------------------------===//

module {
func.func @memref_fill_anchor(%dst: memref<4xf32>) {
%cst = arith.constant 0.000000e+00 : f32 loc("memfill.py":3:5)
linalg.fill ins(%cst : f32) outs(%dst : memref<4xf32>) loc("memfill.py":3:7)
func.return loc("memfill.py":4:3)
}
}

// CHECK-DAG: #[[$MEMFILL_CONST_ORIGIN:[A-Za-z0-9_]+]] = loc("memfill.py":3:5)

// CHECK-LABEL: func.func @memref_fill_anchor
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$MEMFILL_CONST_ORIGIN]]
// CHECK-SAME: loc(#[[MEMFILL_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: linalg.fill
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: outs
// CHECK-SAME: loc(#[[MEMFILL_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[MEMFILL_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[MEMFILL_SYNTH_LOC]] = loc("memfill.py":0:0)
// CHECK-DAG: #[[MEMFILL_LOC]] = loc("memfill.py":3:7)
// CHECK-DAG: #[[MEMFILL_RETURN_LOC]] = loc("memfill.py":4:3)

// -----

//===----------------------------------------------------------------------===//
// future_line_constant
//
// Verifies the backward-step heuristic and helper constant handling. Constants
// and reordered arithmetic that would otherwise introduce misleading source
// locations are classified as synthetic and moved to future.py:0:0, while the
// real load and return keep their visible locations.
//===----------------------------------------------------------------------===//

module {
func.func @future_line_constant(%arg0: memref<4xf32>) {
%c0 = arith.constant 0 : index loc("future.py":8:5)
%c1 = arith.constant 1 : index loc("future.py":24:5)
%v = memref.load %arg0[%c0] : memref<4xf32> loc("future.py":24:7)
%sum = arith.addi %c0, %c1 : index loc("future.py":13:5)
func.return loc("future.py":14:3)
}
}

// CHECK-DAG: #[[$FUTURE_C0_ORIGIN:[A-Za-z0-9_]+]] = loc("future.py":8:5)
// CHECK-DAG: #[[$FUTURE_C1_ORIGIN:[A-Za-z0-9_]+]] = loc("future.py":24:5)
// CHECK-DAG: #[[$FUTURE_ADDI_ORIGIN:[A-Za-z0-9_]+]] = loc("future.py":13:5)

// CHECK-LABEL: func.func @future_line_constant
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$FUTURE_C0_ORIGIN]]
// CHECK-SAME: 0 : index
// CHECK-SAME: loc(#[[FUTURE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$FUTURE_C1_ORIGIN]]
// CHECK-SAME: 1 : index
// CHECK-SAME: loc(#[[FUTURE_SYNTH_LOC]])
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FUTURE_LOAD_LOC:[A-Za-z0-9_]+]])
// CHECK: arith.addi
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$FUTURE_ADDI_ORIGIN]]
// CHECK-SAME: : index
// CHECK-SAME: loc(#[[FUTURE_SYNTH_LOC]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[FUTURE_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[FUTURE_SYNTH_LOC]] = loc("future.py":0:0)
// CHECK-DAG: #[[FUTURE_LOAD_LOC]] = loc("future.py":24:7)
// CHECK-DAG: #[[FUTURE_RETURN_LOC]] = loc("future.py":14:3)

// -----

//===----------------------------------------------------------------------===//
// llvm_constant
//
// Verifies LLVM dialect helper constants. llvm.mlir.constant is classified as
// synthetic, its original location is preserved in triton.debug_line.origin,
// and the operation location is rewritten to llvm.py:0:0. llvm.return remains
// a control operation at its real source location.
//===----------------------------------------------------------------------===//

module {
llvm.func @llvm_constant() {
%0 = llvm.mlir.constant(7 : i32) : i32 loc("llvm.py":20:5)
llvm.return loc("llvm.py":21:3)
}
}

// CHECK-DAG: #[[$LLVM_CONST_ORIGIN:[A-Za-z0-9_]+]] = loc("llvm.py":20:5)

// CHECK-LABEL: llvm.func @llvm_constant
// CHECK: llvm.mlir.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$LLVM_CONST_ORIGIN]]
// CHECK-SAME: loc(#[[LLVM_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: llvm.return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[LLVM_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[LLVM_SYNTH_LOC]] = loc("llvm.py":0:0)
// CHECK-DAG: #[[LLVM_RETURN_LOC]] = loc("llvm.py":21:3)

// -----

//===----------------------------------------------------------------------===//
// nameloc_is_source / fused_scope_is_preserved
//
// Verifies location canonicalization rules for semantic operations. A
// non-internal NameLoc such as "trace_ptr" must be preserved as source-like
// metadata. A fused location carrying an LLVM DIScope must also be preserved
// intact, so already materialized debug-scope metadata is not stripped. Helper
// constants in both functions still become synthetic zero-line operations.
//===----------------------------------------------------------------------===//

#di_file = #llvm.di_file<"scope.py" in "/tmp">
#di_cu = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, producer = "triton", isOptimized = true, emissionKind = LineTablesOnly>
#di_sp = #llvm.di_subprogram<compileUnit = #di_cu, scope = #di_file, name = "scoped", file = #di_file, subprogramFlags = "Definition|Optimized">

module {
func.func @nameloc_is_source(%arg0: memref<4xf32>) {
%c0 = arith.constant 0 : index loc("name.py":2:5)
%v = memref.load %arg0[%c0] : memref<4xf32> loc("trace_ptr"("name.py":7:9))
func.return loc("name.py":8:3)
}

func.func @fused_scope_is_preserved(%arg0: memref<4xf32>) {
%c0 = arith.constant 0 : index loc("scope.py":2:5)
%v = memref.load %arg0[%c0] : memref<4xf32> loc(fused<#di_sp>["scope.py":7:9])
func.return loc("scope.py":8:3)
}
}

// CHECK-DAG: #[[$NAME_CONST_ORIGIN:[A-Za-z0-9_]+]] = loc("name.py":2:5)
// CHECK-DAG: #[[$SCOPE_CONST_ORIGIN:[A-Za-z0-9_]+]] = loc("scope.py":2:5)

// CHECK-LABEL: func.func @nameloc_is_source
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$NAME_CONST_ORIGIN]]
// CHECK-SAME: loc(#[[$NAME_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[$TRACE_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[$NAME_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-LABEL: func.func @fused_scope_is_preserved
// CHECK: arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$SCOPE_CONST_ORIGIN]]
// CHECK-SAME: loc(#[[SCOPE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: memref.load
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[FUSED_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: loc(#[[SCOPE_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[$NAME_SYNTH_LOC]] = loc("name.py":0:0)
// CHECK-DAG: #[[$NAME_FILE_LOC:[A-Za-z0-9_]+]] = loc("name.py":7:9)
// CHECK-DAG: #[[$TRACE_LOC]] = loc("trace_ptr"(#[[$NAME_FILE_LOC]]))
// CHECK-DAG: #[[$NAME_RETURN_LOC]] = loc("name.py":8:3)
// CHECK-DAG: #[[SCOPE_SYNTH_LOC]] = loc("scope.py":0:0)
// CHECK-DAG: #[[SCOPE_FILE_LOC:[A-Za-z0-9_]+]] = loc("scope.py":7:9)
// CHECK-DAG: #[[FUSED_LOC]] = loc(fused<{{.*}}>[#[[SCOPE_FILE_LOC]]])
// CHECK-DAG: #[[SCOPE_RETURN_LOC]] = loc("scope.py":8:3)

// -----

//===----------------------------------------------------------------------===//
// shape_is_unchanged
//
// Verifies that the pass does not change IR semantics or structure. The test
// captures function arguments, result types, operands, produced values, and the
// returned value. Only operation locations and diagnostic debug-line
// attributes are expected to change.
//===----------------------------------------------------------------------===//

module {
func.func @shape_is_unchanged(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %idx: index) -> i32 {
%sub = memref.subview %arg0[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("shape.py":2:5)
%v = memref.load %arg1[%idx] : memref<8xf32> loc("shape.py":3:5)
%c1 = arith.constant 1 : i32 loc("shape.py":4:5)
%r = arith.addi %c1, %c1 : i32 loc("shape.py":5:5)
return %r : i32 loc("shape.py":6:3)
}
}

// CHECK-DAG: #[[$SHAPE_SUBVIEW_ORIGIN:[A-Za-z0-9_]+]] = loc("shape.py":2:5)
// CHECK-DAG: #[[$SHAPE_CONST_ORIGIN:[A-Za-z0-9_]+]] = loc("shape.py":4:5)

// CHECK-LABEL: func.func @shape_is_unchanged(
// CHECK-SAME: %[[SHAPE_ARG0:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[SHAPE_ARG1:[A-Za-z0-9_]+]]: memref<8xf32>
// CHECK-SAME: %[[SHAPE_IDX:[A-Za-z0-9_]+]]: index
// CHECK-SAME: -> i32
// CHECK: %[[SHAPE_SUB:[A-Za-z0-9_]+]] = memref.subview %[[SHAPE_ARG0]][%[[SHAPE_IDX]]] [4] [1]
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$SHAPE_SUBVIEW_ORIGIN]]
// CHECK-SAME: : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-SAME: loc(#[[SHAPE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[SHAPE_V:[A-Za-z0-9_]+]] = memref.load %[[SHAPE_ARG1]][%[[SHAPE_IDX]]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : memref<8xf32>
// CHECK-SAME: loc(#[[SHAPE_LOAD_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[SHAPE_C1:[A-Za-z0-9_]+]] = arith.constant
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$SHAPE_CONST_ORIGIN]]
// CHECK-SAME: 1 : i32
// CHECK-SAME: loc(#[[SHAPE_SYNTH_LOC]])
// CHECK: %[[SHAPE_R:[A-Za-z0-9_]+]] = arith.addi %[[SHAPE_C1]], %[[SHAPE_C1]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: : i32
// CHECK-SAME: loc(#[[SHAPE_ADDI_LOC:[A-Za-z0-9_]+]])
// CHECK: return
// CHECK-SAME: triton.debug_line.class = "control"
// CHECK-SAME: %[[SHAPE_R]] : i32
// CHECK-SAME: loc(#[[SHAPE_RETURN_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[SHAPE_SYNTH_LOC]] = loc("shape.py":0:0)
// CHECK-DAG: #[[SHAPE_LOAD_LOC]] = loc("shape.py":3:5)
// CHECK-DAG: #[[SHAPE_ADDI_LOC]] = loc("shape.py":5:5)
// CHECK-DAG: #[[SHAPE_RETURN_LOC]] = loc("shape.py":6:3)

// -----

//===----------------------------------------------------------------------===//
// direct_destination_view_ambiguity
//
// Verifies deterministic behavior when one destination view has multiple
// direct materialize_in_destination users with distinct store locations. The
// view must not be retargeted based on getUsers() iteration order; it remains
// synthetic and moves to direct_ambiguity.py:0:0, while each store keeps its
// own semantic location.
//===----------------------------------------------------------------------===//

module {
func.func @direct_destination_view_ambiguity(%src: tensor<4xf32>, %dst: memref<8xf32>, %idx: index) {
%view = memref.subview %dst[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("direct_ambiguity.py":10:5)
bufferization.materialize_in_destination %src in writable %view : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("direct_ambiguity.py":10:20)
bufferization.materialize_in_destination %src in writable %view : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("direct_ambiguity.py":10:30)
func.return loc("direct_ambiguity.py":11:3)
}
}

// CHECK-DAG: #[[$DIRECT_AMBIGUITY_ORIGIN:[A-Za-z0-9_]+]] = loc("direct_ambiguity.py":10:5)

// CHECK-LABEL: func.func @direct_destination_view_ambiguity
// CHECK: %[[DIRECT_AMBIGUOUS_VIEW:[A-Za-z0-9_]+]] = memref.subview
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$DIRECT_AMBIGUITY_ORIGIN]]
// CHECK-SAME: loc(#[[DIRECT_AMBIGUITY_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[DIRECT_AMBIGUOUS_VIEW]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[DIRECT_STORE_0_LOC:[A-Za-z0-9_]+]])
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[DIRECT_AMBIGUOUS_VIEW]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[DIRECT_STORE_1_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[DIRECT_AMBIGUITY_SYNTH_LOC]] = loc("direct_ambiguity.py":0:0)
// CHECK-DAG: #[[DIRECT_STORE_0_LOC]] = loc("direct_ambiguity.py":10:20)
// CHECK-DAG: #[[DIRECT_STORE_1_LOC]] = loc("direct_ambiguity.py":10:30)

// -----

//===----------------------------------------------------------------------===//
// through_destination_view_ambiguity
//
// Verifies deterministic behavior when one operation feeds multiple
// destination views. The shared index_cast has two distinct store candidates
// through different subviews, so it must remain synthetic instead of choosing
// an arbitrary user. Each subview has a unique store and may be retargeted to
// its own semantic store location.
//===----------------------------------------------------------------------===//

module {
func.func @through_destination_view_ambiguity(%src: tensor<4xf32>, %dst0: memref<8xf32>, %dst1: memref<8xf32>, %offset: i32) {
%idx = arith.index_cast %offset : i32 to index loc("through_view_ambiguity.py":20:5)
%view0 = memref.subview %dst0[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("through_view_ambiguity.py":21:5)
bufferization.materialize_in_destination %src in writable %view0 : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("through_view_ambiguity.py":21:20)
%view1 = memref.subview %dst1[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("through_view_ambiguity.py":22:5)
bufferization.materialize_in_destination %src in writable %view1 : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("through_view_ambiguity.py":22:20)
func.return loc("through_view_ambiguity.py":23:3)
}
}

// CHECK-DAG: #[[$THROUGH_VIEW_AMBIGUITY_ORIGIN:[A-Za-z0-9_]+]] = loc("through_view_ambiguity.py":20:5)

// CHECK-LABEL: func.func @through_destination_view_ambiguity
// CHECK: %[[THROUGH_VIEW_IDX:[A-Za-z0-9_]+]] = arith.index_cast
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$THROUGH_VIEW_AMBIGUITY_ORIGIN]]
// CHECK-SAME: loc(#[[THROUGH_VIEW_AMBIGUITY_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[THROUGH_VIEW_0:[A-Za-z0-9_]+]] = memref.subview {{.*}}[%[[THROUGH_VIEW_IDX]]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[THROUGH_VIEW_STORE_0_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[THROUGH_VIEW_1:[A-Za-z0-9_]+]] = memref.subview {{.*}}[%[[THROUGH_VIEW_IDX]]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[THROUGH_VIEW_STORE_1_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[THROUGH_VIEW_AMBIGUITY_SYNTH_LOC]] = loc("through_view_ambiguity.py":0:0)
// CHECK-DAG: #[[THROUGH_VIEW_STORE_0_LOC]] = loc("through_view_ambiguity.py":21:20)
// CHECK-DAG: #[[THROUGH_VIEW_STORE_1_LOC]] = loc("through_view_ambiguity.py":22:20)

// -----

//===----------------------------------------------------------------------===//
// through_tensor_insert_ambiguity
//
// Verifies deterministic behavior for the tensor.insert retargeting path. The
// same arithmetic result is inserted into two tensors and reaches two distinct
// stores. Because there is no unique canonical store location, the arithmetic
// operation remains synthetic and moves to tensor_insert_ambiguity.py:0:0.
//===----------------------------------------------------------------------===//

module {
func.func @through_tensor_insert_ambiguity(%lhs: f32, %rhs: f32, %base0: tensor<1xf32>, %base1: tensor<1xf32>, %dst0: memref<1xf32>, %dst1: memref<1xf32>, %idx: index) {
%sum = arith.addf %lhs, %rhs : f32 loc("tensor_insert_ambiguity.py":30:5)
%inserted0 = tensor.insert %sum into %base0[%idx] : tensor<1xf32> loc("tensor_insert_ambiguity.py":30:10)
bufferization.materialize_in_destination %inserted0 in writable %dst0 : (tensor<1xf32>, memref<1xf32>) -> () loc("tensor_insert_ambiguity.py":30:30)
%inserted1 = tensor.insert %sum into %base1[%idx] : tensor<1xf32> loc("tensor_insert_ambiguity.py":30:11)
bufferization.materialize_in_destination %inserted1 in writable %dst1 : (tensor<1xf32>, memref<1xf32>) -> () loc("tensor_insert_ambiguity.py":30:40)
func.return loc("tensor_insert_ambiguity.py":31:3)
}
}

// CHECK-DAG: #[[$TENSOR_INSERT_AMBIGUITY_ORIGIN:[A-Za-z0-9_]+]] = loc("tensor_insert_ambiguity.py":30:5)

// CHECK-LABEL: func.func @through_tensor_insert_ambiguity
// CHECK: %[[TENSOR_INSERT_SUM:[A-Za-z0-9_]+]] = arith.addf
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$TENSOR_INSERT_AMBIGUITY_ORIGIN]]
// CHECK-SAME: loc(#[[TENSOR_INSERT_AMBIGUITY_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[TENSOR_INSERTED_0:[A-Za-z0-9_]+]] = tensor.insert %[[TENSOR_INSERT_SUM]]
// CHECK: bufferization.materialize_in_destination %[[TENSOR_INSERTED_0]]
// CHECK-SAME: loc(#[[TENSOR_INSERT_STORE_0_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[TENSOR_INSERTED_1:[A-Za-z0-9_]+]] = tensor.insert %[[TENSOR_INSERT_SUM]]
// CHECK: bufferization.materialize_in_destination %[[TENSOR_INSERTED_1]]
// CHECK-SAME: loc(#[[TENSOR_INSERT_STORE_1_LOC:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[TENSOR_INSERT_AMBIGUITY_SYNTH_LOC]] = loc("tensor_insert_ambiguity.py":0:0)
// CHECK-DAG: #[[TENSOR_INSERT_STORE_0_LOC]] = loc("tensor_insert_ambiguity.py":30:30)
// CHECK-DAG: #[[TENSOR_INSERT_STORE_1_LOC]] = loc("tensor_insert_ambiguity.py":30:40)

// -----

//===----------------------------------------------------------------------===//
// duplicate_canonical_store_candidates
//
// Verifies that duplicate candidates with the same canonical source location
// are treated as one unique store anchor. One store location is wrapped in an
// internal NameLoc and the other is a direct FileLineColLoc, but both
// canonicalize to duplicate_candidates.py:40:30, so the view can be safely
// retargeted there.
//===----------------------------------------------------------------------===//

module {
func.func @duplicate_canonical_store_candidates(%src: tensor<4xf32>, %dst: memref<8xf32>, %idx: index) {
%view = memref.subview %dst[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("duplicate_candidates.py":40:5)
bufferization.materialize_in_destination %src in writable %view : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("synthetic_store"("duplicate_candidates.py":40:30))
bufferization.materialize_in_destination %src in writable %view : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("duplicate_candidates.py":40:30)
func.return loc("duplicate_candidates.py":41:3)
}
}

// CHECK-DAG: #[[$DUPLICATE_VIEW_ORIGIN:[A-Za-z0-9_]+]] = loc("duplicate_candidates.py":40:5)

// CHECK-LABEL: func.func @duplicate_canonical_store_candidates
// CHECK: %[[DUPLICATE_VIEW:[A-Za-z0-9_]+]] = memref.subview
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: triton.debug_line.origin = #[[$DUPLICATE_VIEW_ORIGIN]]
// CHECK-SAME: loc(#[[DUPLICATE_STORE_LOC:[A-Za-z0-9_]+]])
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[DUPLICATE_VIEW]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[DUPLICATE_STORE_LOC]])
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[DUPLICATE_VIEW]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[DUPLICATE_STORE_LOC]])

// CHECK-DAG: #[[DUPLICATE_STORE_LOC]] = loc("duplicate_candidates.py":40:30)

// -----

//===----------------------------------------------------------------------===//
// direct_and_nested_destination_view_ambiguity
//
// Verifies ambiguity detection across direct and nested destination-view
// paths. The base view has several distinct possible store anchors, including
// direct stores and a store through a nested subview. The pass must not select
// one arbitrarily, so the base view remains synthetic and moves to
// cross_view_ambiguity.py:0:0.
//===----------------------------------------------------------------------===//

module {
func.func @direct_and_nested_destination_view_ambiguity(%src: tensor<4xf32>, %dst: memref<8xf32>, %idx: index) {
%view = memref.subview %dst[%idx] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>> loc("cross_view_ambiguity.py":50:5)
bufferization.materialize_in_destination %src in writable %view : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("cross_view_ambiguity.py":50:20)
bufferization.materialize_in_destination %src in writable %view : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("cross_view_ambiguity.py":50:30)
%nested = memref.subview %view[0] [4] [1] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>> loc("cross_view_ambiguity.py":51:5)
bufferization.materialize_in_destination %src in writable %nested : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> () loc("cross_view_ambiguity.py":51:20)
func.return loc("cross_view_ambiguity.py":52:3)
}
}

// CHECK-DAG: #[[$CROSS_VIEW_ORIGIN:[A-Za-z0-9_]+]] = loc("cross_view_ambiguity.py":50:5)

// CHECK-LABEL: func.func @direct_and_nested_destination_view_ambiguity
// CHECK: %[[CROSS_VIEW:[A-Za-z0-9_]+]] = memref.subview
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$CROSS_VIEW_ORIGIN]]
// CHECK-SAME: loc(#[[CROSS_VIEW_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[CROSS_VIEW]]
// CHECK-SAME: loc(#[[CROSS_VIEW_STORE_A:[A-Za-z0-9_]+]])
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[CROSS_VIEW]]
// CHECK-SAME: loc(#[[CROSS_VIEW_STORE_B:[A-Za-z0-9_]+]])
// CHECK: %[[CROSS_NESTED:[A-Za-z0-9_]+]] = memref.subview %[[CROSS_VIEW]]
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[CROSS_NESTED]]
// CHECK-SAME: loc(#[[CROSS_VIEW_STORE_C:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[CROSS_VIEW_SYNTH_LOC]] = loc("cross_view_ambiguity.py":0:0)
// CHECK-DAG: #[[CROSS_VIEW_STORE_A]] = loc("cross_view_ambiguity.py":50:20)
// CHECK-DAG: #[[CROSS_VIEW_STORE_B]] = loc("cross_view_ambiguity.py":50:30)
// CHECK-DAG: #[[CROSS_VIEW_STORE_C]] = loc("cross_view_ambiguity.py":51:20)

// -----

//===----------------------------------------------------------------------===//
// destination_view_and_tensor_insert_ambiguity
//
// Verifies ambiguity detection across different retargeting mechanisms. The
// same index_cast feeds both a destination-view store path and a tensor.insert
// store path, and the two paths resolve to different store locations. Since the
// combined candidate set is not unique, the index_cast remains synthetic and
// moves to cross_path_ambiguity.py:0:0.
//===----------------------------------------------------------------------===//

module {
func.func @destination_view_and_tensor_insert_ambiguity(%offset: i32, %src: tensor<1xindex>, %base: tensor<1xindex>, %dst0: memref<4xindex>, %dst1: memref<1xindex>) {
%c0 = arith.constant 0 : index loc("cross_path_ambiguity.py":60:3)
%idx = arith.index_cast %offset : i32 to index loc("cross_path_ambiguity.py":60:5)
%view = memref.subview %dst0[%idx] [1] [1] : memref<4xindex> to memref<1xindex, strided<[1], offset: ?>> loc("cross_path_ambiguity.py":61:5)
bufferization.materialize_in_destination %src in writable %view : (tensor<1xindex>, memref<1xindex, strided<[1], offset: ?>>) -> () loc("cross_path_ambiguity.py":61:20)
%inserted = tensor.insert %idx into %base[%c0] : tensor<1xindex> loc("cross_path_ambiguity.py":62:5)
bufferization.materialize_in_destination %inserted in writable %dst1 : (tensor<1xindex>, memref<1xindex>) -> () loc("cross_path_ambiguity.py":62:20)
func.return loc("cross_path_ambiguity.py":63:3)
}
}

// CHECK-DAG: #[[$CROSS_PATH_AMBIGUITY_ORIGIN:[A-Za-z0-9_]+]] = loc("cross_path_ambiguity.py":60:5)

// CHECK-LABEL: func.func @destination_view_and_tensor_insert_ambiguity
// CHECK: %[[CROSS_PATH_IDX:[A-Za-z0-9_]+]] = arith.index_cast
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$CROSS_PATH_AMBIGUITY_ORIGIN]]
// CHECK-SAME: loc(#[[CROSS_PATH_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[CROSS_PATH_VIEW:[A-Za-z0-9_]+]] = memref.subview {{.*}}[%[[CROSS_PATH_IDX]]]
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[CROSS_PATH_VIEW]]
// CHECK-SAME: loc(#[[CROSS_PATH_STORE_A:[A-Za-z0-9_]+]])
// CHECK: %[[CROSS_PATH_INSERTED:[A-Za-z0-9_]+]] = tensor.insert %[[CROSS_PATH_IDX]]
// CHECK: bufferization.materialize_in_destination %[[CROSS_PATH_INSERTED]]
// CHECK-SAME: loc(#[[CROSS_PATH_STORE_B:[A-Za-z0-9_]+]])

// CHECK-DAG: #[[CROSS_PATH_SYNTH_LOC]] = loc("cross_path_ambiguity.py":0:0)
// CHECK-DAG: #[[CROSS_PATH_STORE_A]] = loc("cross_path_ambiguity.py":61:20)
// CHECK-DAG: #[[CROSS_PATH_STORE_B]] = loc("cross_path_ambiguity.py":62:20)

// -----

//===----------------------------------------------------------------------===//
// matching_destination_view_and_tensor_insert_candidates
//
// Verifies that uniqueness alone does not allow cross-line retargeting. The
// index_cast feeds both a destination-view store path and a tensor.insert store
// path, and all candidates canonicalize to cross_path_unique.py:71:20. However,
// the index_cast originates on line 70, so it remains synthetic. The destination
// view originates on line 71 and may still be retargeted to the store anchor.
//===----------------------------------------------------------------------===//

module {
func.func @matching_destination_view_and_tensor_insert_candidates(%offset: i32, %src: tensor<1xindex>, %base: tensor<1xindex>, %dst0: memref<4xindex>, %dst1: memref<1xindex>) {
%c0 = arith.constant 0 : index loc("cross_path_unique.py":70:3)
%idx = arith.index_cast %offset : i32 to index loc("cross_path_unique.py":70:5)
%view = memref.subview %dst0[%idx] [1] [1] : memref<4xindex> to memref<1xindex, strided<[1], offset: ?>> loc("cross_path_unique.py":71:5)
bufferization.materialize_in_destination %src in writable %view : (tensor<1xindex>, memref<1xindex, strided<[1], offset: ?>>) -> () loc("synthetic_store"("cross_path_unique.py":71:20))
%inserted = tensor.insert %idx into %base[%c0] : tensor<1xindex> loc("cross_path_unique.py":71:6)
bufferization.materialize_in_destination %inserted in writable %dst1 : (tensor<1xindex>, memref<1xindex>) -> () loc("cross_path_unique.py":71:20)
func.return loc("cross_path_unique.py":72:3)
}
}

// CHECK-DAG: #[[$CROSS_PATH_UNIQUE_ORIGIN:[A-Za-z0-9_]+]] = loc("cross_path_unique.py":70:5)

// CHECK-LABEL: func.func @matching_destination_view_and_tensor_insert_candidates
// CHECK: %[[CROSS_PATH_UNIQUE_IDX:[A-Za-z0-9_]+]] = arith.index_cast
// CHECK-SAME: triton.debug_line.class = "synthetic"
// CHECK-SAME: triton.debug_line.origin = #[[$CROSS_PATH_UNIQUE_ORIGIN]]
// CHECK-SAME: loc(#[[CROSS_PATH_UNIQUE_SYNTH_LOC:[A-Za-z0-9_]+]])
// CHECK: %[[CROSS_PATH_UNIQUE_VIEW:[A-Za-z0-9_]+]] = memref.subview {{.*}}[%[[CROSS_PATH_UNIQUE_IDX]]]
// CHECK-SAME: triton.debug_line.class = "semantic"
// CHECK-SAME: loc(#[[CROSS_PATH_UNIQUE_STORE:[A-Za-z0-9_]+]])
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[CROSS_PATH_UNIQUE_VIEW]]
// CHECK-SAME: loc(#[[CROSS_PATH_UNIQUE_STORE]])
// CHECK: %[[CROSS_PATH_UNIQUE_INSERTED:[A-Za-z0-9_]+]] = tensor.insert %[[CROSS_PATH_UNIQUE_IDX]]
// CHECK: bufferization.materialize_in_destination %[[CROSS_PATH_UNIQUE_INSERTED]]
// CHECK-SAME: loc(#[[CROSS_PATH_UNIQUE_STORE]])

// CHECK-DAG: #[[CROSS_PATH_UNIQUE_SYNTH_LOC]] = loc("cross_path_unique.py":0:0)
// CHECK-DAG: #[[CROSS_PATH_UNIQUE_STORE]] = loc("cross_path_unique.py":71:20)
