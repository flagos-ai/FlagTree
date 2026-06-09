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
           /*default=*/"\"gcu300\"",
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
           /*default=*/"\"gcu300\"",
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
           /*default=*/"\"gcu300\"",
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
           /*default=*/"\"gcu300\"",
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
           /*default=*/"\"gcu300\"",
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
// TritonGCUPingpongPass
//===----------------------------------------------------------------------===//
def TritonGCUPingpongPass: Pass<"triton-gcu-pingpong", "gpu::GPUModuleOp"> {
  let summary = "optimize dot layout for gcu target";
   let description = [{
     This pass async load global to share  and maybe only for scorpio.
   }];
  let options = [
     Option<"arch", "arch", "std::string",
            /*default=*/"\"gcu300\"",
            "Architecture that these operations will run on">,
     Option<"numStages", "num_stages", "unsigned", /*default=*/"3u",
             "number stage of pipeline">
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
// ConvertTritonToTritonGPU (GCU variant)
//===----------------------------------------------------------------------===//
def GCUConvertTritonToTritonGPUPass
    : Pass<"gcu-convert-triton-to-tritongpu", "mlir::ModuleOp"> {
  let summary = "Convert Triton IR to TritonGPU IR with GCU-aware layout.";
  let description = [{
    This pass annotates ranked tensors with TritonGPU layout encodings.
    Unlike the upstream pass, it analyses reduce ops in the module to
    determine the reduce axis, then constructs an order that gives the
    reduce axis the lowest priority (most-strided position). This keeps
    threads contiguous along non-reduce dimensions, which is beneficial
    for GCU targets where cross-thread reduction is expensive.
  }];
  let options = [
    Option<"numWarps", "num-warps", "unsigned", /*default=*/"4",
           "Number of warps">,
    Option<"threadsPerWarp", "threads-per-warp", "unsigned", /*default=*/"32",
           "Number of threads per warp">,
    Option<"numCTAs", "num-ctas", "unsigned", /*default=*/"1",
           "Number of CTAs in a CGA">,
    Option<"target", "target", "std::string", /*default=*/"\"\"",
           "Target triple, e.g. gcu:300">
  ];
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::TritonDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::ub::UBDialect"
  ];
}

#endif // GCU_CONVERSION_PASSES
