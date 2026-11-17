/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_CFG_PASSES
#define TRITON_TO_CFG_PASSES

include "mlir/Pass/PassBase.td"

def BuildCFG : Pass<"build-cfg", "mlir::ModuleOp"> {
  let summary = "Build Control Flow Graph from TTIR";
  let description = [{
    This pass analyzes the TTIR and builds a Control Flow Graph (CFG)
    representation. It exports the CFG to multiple formats (text, DOT, JSON, HTML)
    for visualization and analysis.

    The output files will be named as <function_name>_cfg.<ext> and placed in
    the directory specified by the output-dir option.
  }];
  let constructor = "mlir::triton::cfg::createBuildCFGPass()";
  let options = [
    Option<"output-dir", "outputDir", "std::string", "\".\"",
           "Directory where CFG files will be generated">,
  ];
}

def MergeConcatLoadBuffer : Pass<"merge-concat-load-buffer", "mlir::ModuleOp"> {
  let summary = "Merge disjoint masked-load UB buffers concatenated by insert_slice";
  let description = [{
    After triton-to-linalg, each masked tt.load allocates a full-tile UB buffer,
    fills it with the padding value, and copies only the masked subview. A
    subsequent cat/join is lowered to extract_slice + insert_slice, so two
    allocs that write disjoint regions of the same static shape can be merged
    into one memref.alloc.

    If the union of copy destinations is proven to cover every later read of
    the concatenated tensor, both linalg.fill ops are deleted. Otherwise a
    single fill is kept (when both fills are bare and use the same value), or
    the rewrite is abandoned.

    A fill only counts as padding when it runs before the copies that overwrite
    it, so a fill placed after them is what the concatenated value observes and
    the rewrite is abandoned. The fill that survives a merge must additionally
    precede the copies of both buffers, since it initialises the shared one.

    The merged buffer ends up written by two disjoint memref.copy ops. On
    910_9362 the generated code only makes the copy next to the surviving
    to_tensor visible and the rest of the tile reads back the padding value, so
    the pipeline only enables this pass on 910_95/950.
  }];
  let constructor = "mlir::triton::cfg::createMergeConcatLoadBufferPass()";
}

def GraphOptimize : Pass<"graph-optimize", "mlir::ModuleOp"> {
  let summary = "Optimize Triton IR with graph analyses";
  let constructor = "mlir::triton::cfg::createGraphOptimizePass()";
  let options = [
    Option<"ruleMask", "rule-mask", "uint64_t", "3071",
           "Bitmask of known 32-bit graph rule identities; default 3071 preserves the legacy native bundle and enables IAT/PTSM. RLF/IPBE/SCP/CBAF remain explicit opt-in">,
    Option<"maxRewritesPerFunction", "max-rewrites-per-function", "uint64_t",
           "64", "Maximum graph rewrites per function">,
    Option<"ubCapacityBytes", "ub-capacity-bytes", "uint64_t", "0",
           "Legacy unified-buffer budget; explicit mapping/store budgets fall back to this value when omitted">,
    Option<"mappingUBCapacityBytes", "mapping-ub-capacity-bytes", "uint64_t", "0",
           "Physical unified-buffer capacity available to program-mapping resource checks">,
    Option<"storeCoalescingUBBudgetBytes", "store-coalescing-ub-budget-bytes", "uint64_t", "0",
           "Conservative unified-buffer budget available to StoreCoalescing">,
    Option<"ubSafetyPercent", "ub-safety-percent", "uint64_t", "80",
           "Percent of UB capacity available to resource candidates">,
    Option<"reservedUBBytes", "reserved-ub-bytes", "uint64_t", "0",
           "UB bytes reserved for lowering/backend workspace before candidate budgeting">,
    Option<"compileMode", "compile-mode", "std::string", "\"simd_simt_template\"",
           "Compilation mode; legacy elementwise RowCoalescing is enabled only for simt_only">,
  ];
}

#endif // TRITON_TO_CFG_PASSES
