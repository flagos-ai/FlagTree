/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
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
#ifndef GCU_TRANSFORMS_PASSES
#define GCU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "triton/Dialect/Triton/IR/TritonOps.td"

def GCU64TypeVerifierPass: Pass<"gcu64-type-verifier", "gpu::GPUModuleOp"> {
  let summary = "Verify that all types are GCU64 types.";
  let description = "Verify that no 64bit type on func arguments.";
  let dependentDialects = [
    "gpu::GPUDialect",
    "triton::TritonDialect",
  ];
  let options = [
    Option<"arch", "arch", "std::string",
           /*default=*/"\"gcu300\"",
           "Architecture that these operations will run on">,
    Option<"test_mode", "test-mode", "bool", /*default=*/"false",
           "test mode for filecheck">
  ];
}

def GCUCombineOps : Pass<"gcu-combine-ops", "gpu::GPUModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the specific patterns.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "triton::TritonDialect"
  ];
}

#endif // GCU_TRANSFORMS_PASSES
