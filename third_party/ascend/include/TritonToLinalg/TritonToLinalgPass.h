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

#ifndef TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
#define TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H

#include "ascend/include/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <string>

#define GEN_PASS_CLASSES
#include "ascend/include/TritonToLinalg/Passes.h.inc"

extern int nd2nzFlag;
extern bool compileOn91095Flag;
extern bool existDotFlag;
extern mlir::triton::ascend::CompileMode compileModeFlag;

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToLinalgPass();

std::unique_ptr<OperationPass<ModuleOp>>
createTritonToLinalgPass(bool, bool, bool, bool, bool,
                         const std::string &compileMode = "simd_simt_template");

} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace triton;
const std::string globalKernelAttr = "global_kernel";
const std::string kernelMixModeName = "mix_mode";
const std::string kernelParallelModeName = "parallel_mode";

class TritonTypeConverter : public mlir::TypeConverter {
public:
  explicit TritonTypeConverter();
};

class TritonToLinalgPass : public TritonToLinalgBase<TritonToLinalgPass> {

  static auto constexpr LAUNCH_GRID_RANK = getMaxEnumValForProgramIDDim() + 1;
  static unsigned int constexpr TRITON_PROGRAM_INFO_ARG_COUNT =
      LAUNCH_GRID_RANK * 2;

private:
  // grid构造 num_programs 3维, program_id 3维
  // remember 'xxxOp' is usually a Pointer, so that we can change target memory
  // without giving a reference argument
  void addProgramInfo(triton::FuncOp func, bool globalKernel);

  void convertTTFunc(triton::FuncOp func, const bool existDot,
                     const bool existSIMTOp);

  LogicalResult convertMultipleBlockControlFlow(Operation *funcOp,
                                                OpBuilder &builder);
  // 处理嵌套的if/else
  scf::IfOp transformNestedIfElse(Operation &nestedBranch, OpBuilder &builder);

  void addDynamicLegal(ConversionTarget &target,
                       TritonTypeConverter &tritonTypeConverter);

  void
  populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns);

  void populateTritonToLinalgConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns,
                                                unsigned int launchGridRank);

  LogicalResult processDescriptorOperations(ModuleOp moduleOp);
  LogicalResult processPtrBroadcastOperations(ModuleOp moduleOp);
  LogicalResult processImplicitPermuteOperations(ModuleOp moduleOp);
  LogicalResult processStridedLoadStoreRewriteOperations(ModuleOp moduleOp);
  LogicalResult processLegalStrideOperations(ModuleOp moduleOp);

public:
  TritonToLinalgPass() = default;

  TritonToLinalgPass(bool globalKernel, bool namedOps, bool enableNd2nzOnVector,
                     bool enableSelectAnalysis, bool compileOn91095,
                     const std::string &compileMode = "simd_simt_template") {
    this->globalKernel = globalKernel;
    this->namedOps = namedOps;
    this->enableNd2nzOnVector = enableNd2nzOnVector;
    this->enableSelectAnalysis = enableSelectAnalysis;
    this->compileOn91095 = compileOn91095;
    this->compileMode = compileMode;
  };

  void getDependentDialects(DialectRegistry &registry) const override;

  void runOnOperation() override;
};

#endif // TRITON_ADAPTER_CONVERSION_TRITONTOLINALG_H
