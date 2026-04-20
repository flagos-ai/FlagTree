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

#include <map>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
namespace {
struct TritonMakeRangeOpLowering
    : SharedConversionPattern<triton::MakeRangeOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op);
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();
    auto lastUser =
        userAnalysis.getLastUser(op.getOperation()->getResults()[0]);
    auto warpIds = getWarpIds(rewriter, loc, op.getType());
    auto slicedAxies = getSlicedAxies(op.getType());
    auto numElems = triton::gcu::getTotalElemsPerThread(op.getType());
    auto resultType =
        dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto output = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                              replaced2Origin, resultType);

    Value start = rewriter
                      .create<arith::ConstantIntOp>(
                          loc, resultType.getElementType(), op.getStart())
                      .getResult();
    if (!slicedAxies.empty()) {
      start = rewriter.create<arith::AddIOp>(
          loc,
          rewriter.create<arith::MulIOp>(
              loc,
              rewriter.create<arith::IndexCastOp>(
                  loc, resultType.getElementType(), warpIds.front()),
              rewriter.create<arith::ConstantIntOp>(
                  loc, resultType.getElementType(), numElems)),
          start);
    }
    auto vectorLength =
        oaccSizeInBytes / triton::gcu::getBpe(resultType.getElementType());

    auto vectorType = VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      resultType.getElementType());
    Value initValue =
        rewriter.create<gcu::VectorStepOp>(loc, vectorType, start).getResult();
    Value step = rewriter.create<vector::BroadcastOp>(
        loc, vectorType,
        rewriter.create<arith::ConstantIntOp>(loc, resultType.getElementType(),
                                              vectorLength));
    rewriter.create<scf::ForOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
        rewriter.create<arith::ConstantIndexOp>(loc, numElems),
        rewriter.create<arith::ConstantIndexOp>(loc, vectorLength),
        ValueRange{initValue},
        [&](OpBuilder &builder, Location loc, Value iters,
            ValueRange iterArgs) {
          builder.create<vector::StoreOp>(loc, iterArgs[0], output, iters);
          builder.create<scf::YieldOp>(
              loc, ValueRange{
                       builder.create<arith::AddIOp>(loc, iterArgs[0], step)});
        });
    leaveTritionOp(rewriter, op);
    rewriter.replaceOp(op, output);
    return success();
  }
};
} // namespace

void mlir::triton::populateMakeRangeOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<TritonMakeRangeOpLowering>(converter, patterns.getContext(),
                                          userAnalysis, replaced2Origin,
                                          pTagPool);
}
