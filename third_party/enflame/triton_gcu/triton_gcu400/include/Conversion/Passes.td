/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GCU_CONVERSION_PASSES
#define GCU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// TritonToGCU
//===----------------------------------------------------------------------===//
def ConvertTritonToGCUPass: Pass<"convert-triton-to-gcu", "gpu::GPUModuleOp"> {
  let summary = "Convert triton ir to gcu operations.";
  let description = [{
    This pass converts triton ir to gcu operations.
  }];
  let options = [
    Option<"arch", "arch", "std::string",
           /*default=*/"\"gcu400\"",
           "Architecture that these operations will run on">,
    Option<"vectorLength", "vector-length", "unsigned", /*default=*/"512u",
           "Length of vector in byte">,
    Option<"vectorizationMaxLength", "vectorization-max-length", "unsigned", /*default=*/"16384u",
            "Maximum vector length in byte for vectorization">,
  ];
}

//===----------------------------------------------------------------------===//
// TrtionFunsion
//===----------------------------------------------------------------------===//
def GCUTritonFusionPass: Pass<"gcu-triton-fusion", "gpu::GPUModuleOp"> {
  let summary = "Fuse triton ir to gcu operations.";
  let description = [{
    This pass fuse triton ir to gcu operations.
  }];
  let options = [
     Option<"arch", "arch", "std::string",
           /*default=*/"\"gcu400\"",
           "Architecture that these operations will run on">
  ];
}

//===----------------------------------------------------------------------===//
// TritonLoadStoreToGCUDma
//===----------------------------------------------------------------------===//
def ConvertTritonLoadStoreToGCUDmaPass: Pass<"convert-triton-load-store-to-gcu-dma", "gpu::GPUModuleOp"> {
  let summary = "Convert triton load/store ir to gcu dma operations.";
  let description = [{
    This pass converts triton load/store ir to gcu dma operations.
  }];
  let options = [
    Option<"arch", "arch", "std::string",
           /*default=*/"\"gcu400\"",
           "Architecture that these operations will run on">,
    Option<"support_stride0", "support_stride0", "bool", /*default=*/"false",
          "support broadcast for stride = 0 ">
  ];
}


//===----------------------------------------------------------------------===//
// TritonGCULayoutOptimize
//===----------------------------------------------------------------------===//
 def TritonGCULayoutOptimizePass: Pass<"triton-gcu-data-layout-optimize", "gpu::GPUModuleOp"> {
   let summary = "optimize data layout for gcu target";
   let description = [{
    This pass gerneric optimize triton  layout for gcu target.
   }];
   let options = [
     Option<"arch", "arch", "std::string",
           /*default=*/"\"gcu400\"",
           "Architecture that these operations will run on">
   ];
}

//===----------------------------------------------------------------------===//
// TritonDotLayoutOptimize
//===----------------------------------------------------------------------===//
def TritonGCUDotLayoutOptimizePass: Pass<"triton-gcu-dot-layout-optimize", "gpu::GPUModuleOp"> {
  let summary = "optimize dot layout for gcu target";
  let description = [{
    This pass specified optimize triton layout for dot.
  }];
  let options = [
    Option<"arch", "arch", "std::string",
           /*default=*/"\"gcu400\"",
           "Architecture that these operations will run on">
  ];
}

//===----------------------------------------------------------------------===//
// FlattenTritonFunc
//===----------------------------------------------------------------------===//
def GCUFlattenTritonFuncPass: Pass<"flatten-triton-func", "gpu::GPUModuleOp"> {
  let summary = "flatten func op for triton.";
  let description = [{
    This pass flatten func op which is called multiple times.
  }];
  let options = [
  ];
}

//===----------------------------------------------------------------------===//
// ConvertTensorPointer
//===----------------------------------------------------------------------===//
def ConvertTensorPointerPass: Pass<"convert-tensor-pointer", "gpu::GPUModuleOp"> {
  let summary = "Convert triton make_tensor_ptr ir to make_range operations.";
  let description = [{
    This pass converts triton make_tensor_ptr ir to make_range operations.
  }];
  let options = [
  ];
}

//===----------------------------------------------------------------------===//
// TritonGPUtoTritonGCU
//===----------------------------------------------------------------------===//
def TritonGPUToTritonGCUPass: Pass<"triton-gpu-to-triton-gcu", "gpu::GPUModuleOp"> {
  let summary = "change some special triton GPU op to gcu ";
  let description = [{
    This pass change some special triton GPU op to gcu .
  }];
  let options = [
  ];
}

//===----------------------------------------------------------------------===//
// TritonGCUAccelerateMatmul
//===----------------------------------------------------------------------===//
def TritonGCUAccelerateMatmul : Pass<"tritongcu-accelerate-matmul", "gpu::GPUModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., Nvidia tensor cores)
  }];

  let dependentDialects = ["mlir::triton::gcu::TritonGCUDialect"];
}

//===----------------------------------------------------------------------===//
// TritonWGDotToGCU
//===----------------------------------------------------------------------===//
def TritonWGDotToGCU: Pass<"triton-wgdot-to-gcu", "gpu::GPUModuleOp"> {
  let summary = "Convert warp_group_dot and related ops for GCU execution.";
  let description = [{
    This pass converts warp_group_dot and related operations for GCU:
    1. Converts warp_group_dot to tt.dot with blocked layout.
    2. Fuses load + local_alloc into load_global_to_share DMA operations
       for dot operands.
    3. Folds redundant convert_layout round-trips around dot.
  }];
  let options = [
  ];

  let dependentDialects = ["mlir::triton::gcu::TritonGCUDialect"];
}

//===----------------------------------------------------------------------===//
// TritonGCUAllocateWarpGroups
//===----------------------------------------------------------------------===//
def TritonGCUAllocateWarpGroups : Pass<"triton-gcu-allocate-warp-groups", "gpu::GPUModuleOp"> {
  let summary = "Allocate warp groups";

  let description = [{
    The `triton-gcu-allocate-warp-groups` pass performs warpgroup allocation for
    a GCU program. When a GCU program contains warp specialization, additional
    warps are launched in addition to the "default" warp group. The "default"
    warpgroup executes top-level code in a `tt.func` and its size is specified
    by the user via the `num_warps` argument.

    This pass analyzes `ttg.warp_specialize` ops in the program and determines
    the total number of needed warps, then attaches the range of warp IDs to
    each warpgroup function.
  }];
}

//===----------------------------------------------------------------------===//
// TLEToTritonGCU
//===----------------------------------------------------------------------===//
def TLEToTritonGCUPass
    : Pass<"tle-to-triton-gcu", "gpu::GPUModuleOp"> {
  let summary = "Unified FlagTree TLE op lowering for enflame GCU backend.";
  let description = [{
    This pass handles all TLE-related op lowering for the enflame backend.
    All patterns use string-based RewritePattern matching.
  }];
  let options = [];
  let dependentDialects = ["mlir::triton::gcu::TritonGCUDialect"];
}

//===----------------------------------------------------------------------===//
// AnnotateDotAccReuse
//===----------------------------------------------------------------------===//
def AnnotateDotAccReusePass
    : Pass<"annotate-dot-acc-reuse", "gpu::GPUModuleOp"> {
  let summary = "Annotate tt.dot ops eligible for in-place accumulator reuse.";
  let description = [{
    Pre-conversion analysis pass that identifies tt.dot ops where the
    accumulator buffer (C) can be reused as the output buffer (D) to avoid
    allocating a separate buffer (in-place MMA pattern: D = A * B + C).

    Qualifying ops are annotated with the "acc_reuse_candidate" attribute.
    The downstream convert-triton-to-gcu pass checks this attribute and
    performs the remaining type-compatibility verification at conversion time.
  }];
  let options = [];
  let dependentDialects = [];
}

#endif // GCU_CONVERSION_PASSES
