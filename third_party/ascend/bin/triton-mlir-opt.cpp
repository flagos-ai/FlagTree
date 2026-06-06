//===- triton-mlir-opt.cpp - Triton-Ascend Optimizer Driver -----------*- C++ -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Main entry function for triton-mlir-opt built as standalone binary.
// This tool combines MLIR dialects and BishengIR dialects.
//
//===----------------------------------------------------------------------===//

#include "./RegisterTritonDialects.h"

// BishengIR includes
#include "bishengir/InitAllDialects.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  // Register all MLIR dialects and passes
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  // Register BishengIR dialects and passes
  bishengir::registerAllDialects(registry);


  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Triton-Ascend optimizer driver\n", registry));
}
