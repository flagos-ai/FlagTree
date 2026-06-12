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

#include <functional>
#include <map>
#include <string>
#include <utility>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Conversion/TritonToGCU/ReduceScanCommon.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace {

constexpr unsigned kOaccSizeInBytes = 512;
constexpr unsigned kLoopUnrollTimes = 16;

enum class ReduceExecutionMode {
  kScalar,
  kVectorized,
};

struct VectorAxisCandidate {
  int64_t vectorAxis;
  std::array<int64_t, 2> peerDims;
};

struct InputPaddingInfo {
  bool needsPadding = false;
  TypedAttr paddingValue;
};

struct TransposeInfo {
  bool doTranspose = false;
  std::array<int64_t, 3> transposeLayout = {0, 1, 2};
};

struct VectorizationPolicy {
  int64_t vectorAxis;
  unsigned vectorLength;
  int64_t unrollAxis;
  unsigned unrollCount;
  SmallVector<VectorType> loadTypes;
  std::optional<TransposeInfo> transposeInfo = std::nullopt;
  std::optional<SmallVector<InputPaddingInfo>> inputPaddingInfos = std::nullopt;
};

struct DispatchPlan {
  ReduceExecutionMode executionMode;
  std::optional<VectorizationPolicy> vectorizationPolicy;
};

struct ReductionCoord3D {
  Value iter0;
  Value iter1;
  Value iter2;
};

struct Reduction3DContext {
  std::array<int64_t, 3> inputDims;
  std::array<int64_t, 3> outputDims;
  int64_t reductionAxis;
};

static bool isSupportedElementType(Type elementType) {
  return elementType.isBF16() || elementType.isF16() || elementType.isF32() ||
         elementType.isInteger(1) || elementType.isInteger(8) ||
         elementType.isInteger(16) || elementType.isInteger(32);
}

static SmallVector<VectorType> buildVectorLoadTypes(ArrayRef<Value> inputs,
                                                    unsigned vectorLength) {
  return llvm::to_vector(llvm::map_range(inputs, [&](auto input) {
    return VectorType::get(ArrayRef<int64_t>{vectorLength},
                           cast<MemRefType>(input.getType()).getElementType());
  }));
}

static bool needsVectorPadding(int64_t numElements, Type elementType) {
  auto bpe = mlir::triton::gcu::getBpe(elementType);
  return bpe * numElements < kOaccSizeInBytes;
}

static Value emitVectorLoad(OpBuilder &builder, Location loc,
                            VectorType loadType, Value memref,
                            ValueRange indices, Value mask,
                            const InputPaddingInfo &paddingInfo) {
  if (paddingInfo.needsPadding) {
    auto paddingValue = builder.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(loadType, paddingInfo.paddingValue));
    Value q = builder.create<vector::MaskedLoadOp>(loc, loadType, memref,
                                                   indices, mask, paddingValue);
    return q;
  } else {
    return builder.create<vector::LoadOp>(loc, loadType, memref, indices);
  }
}

static void emitVectorStore(OpBuilder &builder, Location loc, Value value,
                            Value memref, ValueRange indices, Value mask,
                            bool needsPadding) {
  if (needsPadding) {
    builder.create<vector::MaskedStoreOp>(loc, memref, indices, mask, value);
  } else {
    builder.create<vector::StoreOp>(loc, value, memref, indices);
  }
}

static void doTranspose(OpBuilder &builder, Location loc, ArrayRef<Value> dsts,
                        ArrayRef<Value> srcs,
                        const std::array<int64_t, 3> &transposeLayout,
                        const triton::gcu::TagInfo &tag) {
  SmallVector<Value> layoutValues =
      llvm::to_vector(llvm::map_range(transposeLayout, [&](auto dim) {
        return builder.create<arith::ConstantIntOp>(loc, dim, 32).getResult();
      }));

  for (auto [dst, src] : llvm::zip_equal(dsts, srcs)) {
    auto memrefTy = cast<MemRefType>(src.getType());
    builder.create<memref_ext::TransposeStartOp>(
        loc, dst, src, layoutValues, tag.getTag(), ValueRange{tag.getIdx()});
    builder.create<memref::DmaWaitOp>(
        loc, tag.getTag(), ValueRange{tag.getIdx()},
        builder.create<arith::ConstantIndexOp>(loc, memrefTy.getNumElements()));
  }
}

static SmallVector<Value>
prepareTransposedOutputs(OpBuilder &builder, Location loc,
                         ArrayRef<Value> outputs,
                         const std::array<int64_t, 3> &transposedOutputDims,
                         const std::array<int64_t, 3> &transposeLayout) {
  SmallVector<Value> transposedOutputs;
  if (transposeLayout[2] == 0) {
    llvm::transform(
        outputs, std::back_inserter(transposedOutputs), [&](auto output) {
          return builder.create<memref::AllocOp>(
              loc, MemRefType::get(
                       ArrayRef<int64_t>{transposedOutputDims},
                       cast<MemRefType>(output.getType()).getElementType()));
        });
  } else {
    llvm::transform(
        outputs, std::back_inserter(transposedOutputs), [&](auto output) {
          return builder.create<memref::ReinterpretCastOp>(
              loc,
              MemRefType::get(
                  ArrayRef<int64_t>{transposedOutputDims},
                  cast<MemRefType>(output.getType()).getElementType()),
              output, 0, ArrayRef<int64_t>{transposedOutputDims},
              ArrayRef<int64_t>{transposedOutputDims[2] *
                                    transposedOutputDims[1],
                                transposedOutputDims[2], 1});
        });
  }
  return transposedOutputs;
}

static SmallVector<Value>
reshapeToContiguous3D(OpBuilder &builder, Location loc, ValueRange buffers,
                      const std::array<int64_t, 3> &dims) {
  SmallVector<Value> results;
  results.reserve(buffers.size());
  for (auto buffer : buffers) {
    assert(isa<MemRefType>(buffer.getType()) && "buffer must be a memref");
    auto memrefTy = cast<MemRefType>(buffer.getType());
    auto elementTy = memrefTy.getElementType();
    if (elementTy.isInteger(1)) {
      auto ptr = builder.create<mlir::gcu::MemRefToPtrOp>(
          loc, mlir::gcu::PtrType::get(builder.getContext(), elementTy),
          buffer);
      elementTy = builder.getI8Type();
      buffer = builder.create<mlir::gcu::PtrToMemRefOp>(
          loc,
          MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic}, elementTy,
                          MemRefLayoutAttrInterface{},
                          memrefTy.getMemorySpace()),
          ptr);
    }
    results.emplace_back(builder.create<memref::ReinterpretCastOp>(
        loc,
        MemRefType::get(dims, elementTy, MemRefLayoutAttrInterface{},
                        memrefTy.getMemorySpace()),
        buffer, 0, ArrayRef<int64_t>{dims},
        ArrayRef<int64_t>{dims[1] * dims[2], dims[2], 1}));
  }
  return results;
}

static SmallVector<Value> emitUnrolledStrideLoads(
    OpBuilder &builder, Location loc, const VectorizationPolicy &policy,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const ReductionCoord3D &coord) {
  int64_t loadStride = 1;
  for (unsigned i = policy.vectorAxis + 1; i < context.inputDims.size(); ++i)
    loadStride *= context.inputDims[i];
  int64_t unrollStep =
      (policy.unrollAxis == policy.vectorAxis) ? policy.vectorLength : 1;
  SmallVector<Value> values;
  auto loadStrideVal =
      builder.create<arith::ConstantIntOp>(loc, loadStride, 64);
  for (unsigned i = 0; i < policy.unrollCount; ++i) {
    Value iter0 = coord.iter0;
    Value iter1 = coord.iter1;
    Value iter2 = coord.iter2;
    auto offset = builder.create<arith::ConstantIndexOp>(loc, i * unrollStep);
    switch (policy.unrollAxis) {
    case 0:
      iter0 = builder.create<arith::AddIOp>(loc, offset, iter0);
      break;
    case 1:
      iter1 = builder.create<arith::AddIOp>(loc, offset, iter1);
      break;
    case 2:
      iter2 = builder.create<arith::AddIOp>(loc, offset, iter2);
      break;
    }
    for (auto [loadType, input] : llvm::zip_equal(policy.loadTypes, inputs)) {
      values.emplace_back(builder.create<gcu::LoadStrideOp>(
          loc, loadType, input, ValueRange{iter0, iter1, iter2},
          loadStrideVal));
    }
  }
  return values;
}

static SmallVector<Value>
emitUnrollCombine(OpBuilder &builder, Location loc,
                  const triton::gcu::CombineOpDesc &combineOpDesc,
                  ValueRange accumulators, ArrayRef<Value> curValues,
                  unsigned unrollCount, unsigned numOutputs,
                  unsigned vectorLength) {
  SmallVector<Value> results;
  SmallVector<Value> combineOperands(numOutputs * 2);
  for (unsigned i = 0; i < unrollCount; ++i) {
    for (unsigned j = 0; j < numOutputs; ++j) {
      combineOperands[j] = accumulators[i * numOutputs + j];
      combineOperands[numOutputs + j] = curValues[i * numOutputs + j];
    }
    results.append(combineOpDesc.applyVectorizedCombine(
        builder, loc, combineOperands, vectorLength));
  }
  return results;
}

static SmallVector<Value>
reduceUnrolledPartials(OpBuilder &builder, Location loc,
                       const triton::gcu::CombineOpDesc &combineOpDesc,
                       ArrayRef<Value> partials, unsigned numOutputs,
                       unsigned vectorLength) {
  unsigned numPartials = partials.size() / numOutputs;
  SmallVector<Value> results(partials);
  while (numPartials != 1) {
    numPartials /= 2;
    for (unsigned i = 0; i < numPartials; ++i) {
      auto combineResults = combineOpDesc.applyVectorizedCombine(
          builder, loc,
          ValueRange(ArrayRef<Value>(results).slice(2 * i * numOutputs,
                                                    2 * numOutputs)),
          vectorLength);
      llvm::copy(combineResults, results.begin() + i * numOutputs);
    }
  }
  results.truncate(numOutputs);
  return results;
}

static unsigned computeUnrollCount(const VectorizationPolicy &policy,
                                   const Reduction3DContext &context) {
  unsigned dimSize = context.inputDims[policy.unrollAxis];
  unsigned numTilesAlongUnrollAxis = (policy.unrollAxis == policy.vectorAxis)
                                         ? dimSize / policy.vectorLength
                                         : dimSize;
  return std::min<unsigned>(kLoopUnrollTimes, numTilesAlongUnrollAxis);
}

class ReduceGenerator {
public:
  explicit ReduceGenerator(triton::ReduceOp op) : op(op), combineOpDesc(op) {
    vectorLength = computeVectorLength();
  }
  static Reduction3DContext
  normalizeReductionTo3D(ArrayRef<unsigned> elemsPerThread, unsigned axis);

  DispatchPlan buildDispatchPlan(OpBuilder &builder, ArrayRef<Value> inputs,
                                 const Reduction3DContext &context) const;

  void applyReduce(OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
                   ArrayRef<Value> inputs, const Reduction3DContext &context,
                   const DispatchPlan &dispatchPlan) const;

private:
  std::optional<VectorizationPolicy>
  tryBuildReduceAxis2Policy(OpBuilder &builder, ArrayRef<Value> inputs,
                            const Reduction3DContext &context) const;

  std::optional<VectorizationPolicy>
  tryBuildReduceAxis1Policy(OpBuilder &builder, ArrayRef<Value> inputs,
                            const Reduction3DContext &context) const;

  std::optional<VectorizationPolicy>
  tryBuildReduceAxis2WithPaddingPolicy(OpBuilder &builder,
                                       ArrayRef<Value> inputs,
                                       const Reduction3DContext &context) const;
  VectorizationPolicy
  buildReduceAxis1WithPaddingPolicy(OpBuilder &builder, ArrayRef<Value> inputs,
                                    const Reduction3DContext &context) const;

private:
  std::optional<unsigned> computeVectorLength() const;

  void applyReduceAxis2VecAxis2UnrollAxis2(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis2VecAxis2UnrollAxis1(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis2VecAxis1UnrollAxis2(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis2VecAxis1UnrollAxis1(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis2WithPadding(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis2VecAxis2WithPadding(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis2UnrollAxis2(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis2UnrollAxis1(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis2UnrollAxis0(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis1UnrollAxis0(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis1UnrollAxis1(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis1UnrollAxis2(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis0UnrollAxis0(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis0UnrollAxis1(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyReduceAxis1VecAxis0UnrollAxis2(
      OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
      ArrayRef<Value> inputs, const Reduction3DContext &context,
      const VectorizationPolicy &policy) const;

  void applyScalarImpl(OpBuilder &builder, Location loc,
                       ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                       const Reduction3DContext &context) const;

private:
  triton::ReduceOp op;
  std::optional<unsigned> vectorLength;
  triton::gcu::CombineOpDesc combineOpDesc;
};

Reduction3DContext
ReduceGenerator::normalizeReductionTo3D(ArrayRef<unsigned> elemsPerThread,
                                        unsigned axis) {
  auto [reduceInputDims, reduceAxis] =
      triton::gcu::foldTo3D(elemsPerThread, axis);
  std::array<int64_t, 3> reduceOutputDims = reduceInputDims;
  reduceOutputDims[reduceAxis] = 1;
  assert((reduceAxis == 1 || reduceAxis == 2) &&
         "normalize N-D reduction to 3D, the reduce axis must be 1 or 2");
  return {reduceInputDims, reduceOutputDims, reduceAxis};
}

std::optional<unsigned> ReduceGenerator::computeVectorLength() const {
  unsigned maxBpe = 1;
  unsigned minBpe = 8;
  for (auto elementType : combineOpDesc.getElementTypes()) {
    if (!isSupportedElementType(elementType)) {
      return std::nullopt;
    }
    auto bpe = mlir::triton::gcu::getBpe(elementType);
    maxBpe = std::max(maxBpe, bpe);
    minBpe = std::min(minBpe, bpe);
  }
  auto numOacc = maxBpe / minBpe;
  if (numOacc > 4) {
    return std::nullopt;
  }
  return kOaccSizeInBytes / minBpe;
}

void ReduceGenerator::applyReduceAxis2VecAxis2UnrollAxis2(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.inputDims[0] == 1 && "input dim 0 must be 1");
  assert(context.reductionAxis == 2 && "reduce axis must be 2");
  assert(policy.vectorAxis == 2 && "vector axis must be 2");
  assert(policy.unrollAxis == 2 && "unroll axis must be 2");

  SmallVector<Value> tarAddrs;
  SmallVector<Value> inputTarStrides;
  unsigned numInputs = inputs.size();
  unsigned numOutputs = outputs.size();
  triton::gcu::TritonGCUBuilder b(loc, builder);
  for (auto [loadType, input] : llvm::zip_equal(policy.loadTypes, inputs)) {
    tarAddrs.emplace_back(b.tarAddr(input));
    auto bpe = mlir::triton::gcu::getBpe(loadType.getElementType());
    inputTarStrides.emplace_back(
        b.tarStride(loadType, policy.vectorLength * bpe));
  }
  auto fetchInputValues = [&](OpBuilder &builder, Location loc,
                              SmallVector<Value> &tarAddrs) {
    auto b = triton::gcu::TritonGCUBuilder(loc, builder);
    SmallVector<Value> values;
    for (unsigned i = 0; i < policy.unrollCount; ++i) {
      for (auto [loadType, tarAddr, inputTarStride] :
           llvm::zip_equal(policy.loadTypes, tarAddrs, inputTarStrides)) {
        values.emplace_back(b.tarLoad(loadType, tarAddr, inputTarStride));
      }
    }
    return values;
  };

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto vectorTileSize = builder.create<arith::ConstantIndexOp>(
      loc, policy.unrollCount * policy.vectorLength);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[1]), one,
      tarAddrs,
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs0) {
        SmallVector<Value> initArgs(iterArgs0);
        initArgs.append(fetchInputValues(builder, loc, initArgs));
        auto loop1 = builder.create<scf::ForOp>(
            loc, vectorTileSize,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            vectorTileSize, initArgs,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              SmallVector<Value> inputTarAddrs(iterArgs1.take_front(numInputs));
              auto curValues = fetchInputValues(builder, loc, inputTarAddrs);
              SmallVector<Value> results(inputTarAddrs);
              results.append(emitUnrollCombine(
                  builder, loc, combineOpDesc,
                  ValueRange(iterArgs1).drop_front(numInputs), curValues,
                  policy.unrollCount, numOutputs, policy.vectorLength));
              builder.create<scf::YieldOp>(loc, results);
            });
        SmallVector<Value> partials(loop1.getResults().drop_front(numInputs));
        assert(partials.size() == policy.unrollCount * numOutputs);
        auto results =
            reduceUnrolledPartials(builder, loc, combineOpDesc, partials,
                                   numOutputs, policy.vectorLength);
        results = reduceVectorLanes(builder, loc, combineOpDesc, results);
        for (unsigned i = 0; i < numOutputs; ++i) {
          builder.create<memref::StoreOp>(loc, results[i], outputs[i],
                                          ValueRange{zero, iter0, zero});
        }
        builder.create<scf::YieldOp>(
            loc, ValueRange(loop1.getResults().take_front(numInputs)));
      });
}

void ReduceGenerator::applyReduceAxis2VecAxis2UnrollAxis1(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.inputDims[0] == 1 && "input dim 0 must be 1");
  assert(context.reductionAxis == 2 && "reduce axis must be 2");
  assert(policy.vectorAxis == 2 && "vector axis must be 2");
  assert(policy.unrollAxis == 1 && "unroll axis must be 1");

  unsigned numOutputs = outputs.size();
  unsigned numInputs = inputs.size();
  SmallVector<Value> tarAddrs;
  SmallVector<SmallVector<Value>> inputTarStridesTable(numInputs);
  triton::gcu::TritonGCUBuilder b(loc, builder);

  for (auto [loadType, input, inputTarStrides] :
       llvm::zip_equal(policy.loadTypes, inputs, inputTarStridesTable)) {
    tarAddrs.emplace_back(b.tarAddr(input));
    auto bpe = mlir::triton::gcu::getBpe(loadType.getElementType());
    inputTarStrides.emplace_back(
        b.tarStride(loadType, context.inputDims[2] * bpe));
    inputTarStrides.emplace_back(
        b.tarStride(loadType, (policy.vectorLength - (policy.unrollCount - 1) *
                                                         context.inputDims[2]) *
                                  bpe));
    inputTarStrides.emplace_back(
        b.tarValue((policy.unrollCount - 1) * context.inputDims[2] * bpe));
  }
  auto fetchInputValues = [&](OpBuilder &builder, Location loc,
                              SmallVector<Value> &tarAddrs) {
    auto b = triton::gcu::TritonGCUBuilder(loc, builder);
    SmallVector<Value> values;
    for (unsigned i = 0; i < policy.unrollCount - 1; ++i) {
      for (auto [loadType, tarAddr, inputTarStrides] :
           llvm::zip_equal(policy.loadTypes, tarAddrs, inputTarStridesTable)) {
        values.emplace_back(b.tarLoad(loadType, tarAddr, inputTarStrides[0]));
      }
    }
    for (auto [loadType, tarAddr, inputTarStrides] :
         llvm::zip_equal(policy.loadTypes, tarAddrs, inputTarStridesTable)) {
      values.emplace_back(b.tarLoad(loadType, tarAddr, inputTarStrides[1]));
    }
    return values;
  };
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[1]),
      builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount), tarAddrs,
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs0) {
        SmallVector<Value> initArgs(iterArgs0);
        initArgs.append(fetchInputValues(builder, loc, initArgs));
        auto loop1 = builder.create<scf::ForOp>(
            loc,
            builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength),
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength),
            initArgs,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              SmallVector<Value> inputTarAddrs(iterArgs1.take_front(numInputs));
              auto curValues = fetchInputValues(builder, loc, inputTarAddrs);
              SmallVector<Value> results(inputTarAddrs);
              results.append(emitUnrollCombine(
                  builder, loc, combineOpDesc,
                  ValueRange(iterArgs1).drop_front(numInputs), curValues,
                  policy.unrollCount, numOutputs, policy.vectorLength));
              builder.create<scf::YieldOp>(loc, results);
            });
        triton::gcu::TritonGCUBuilder b(loc, builder);
        for (unsigned i = 0; i < numInputs; ++i) {
          initArgs[i] = loop1.getResult(i);
          b.tarJump(initArgs[i], inputTarStridesTable[i][2]);
        }
        for (unsigned i = 0; i < policy.unrollCount; ++i) {
          auto results =
              reduceVectorLanes(builder, loc, combineOpDesc,
                                ValueRange(loop1.getResults().slice(
                                    numInputs + i * numOutputs, numOutputs)));
          for (unsigned j = 0; j < numOutputs; ++j) {
            builder.create<memref::StoreOp>(
                loc, results[j], outputs[j],
                ValueRange{zero,
                           builder.create<arith::AddIOp>(
                               loc,
                               builder.create<arith::ConstantIndexOp>(loc, i),
                               iter0),
                           zero});
          }
        }
        builder.create<scf::YieldOp>(
            loc, ValueRange(initArgs).take_front(numInputs));
      });
}

void ReduceGenerator::applyReduceAxis2VecAxis2WithPadding(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 2 && "reduce axis must be 2");
  assert(policy.vectorAxis == 2 && "vector axis must be 2");
  assert(policy.unrollAxis == 1 && "unroll axis must be 1");
  assert(policy.inputPaddingInfos.has_value() &&
         "input padding infos must be present");

  unsigned numOutputs = outputs.size();
  auto mask = builder.create<vector::ConstantMaskOp>(
      loc,
      VectorType::get(ArrayRef<int64_t>{policy.vectorLength},
                      builder.getI1Type()),
      ArrayRef<int64_t>{context.inputDims[2]});
  auto fetchInputValues = [&](OpBuilder &builder, Location loc,
                              const ReductionCoord3D &coord) {
    SmallVector<Value> values;
    for (unsigned i = 0; i < policy.unrollCount; ++i) {
      auto iter1 = builder.create<arith::AddIOp>(
          loc, builder.create<arith::ConstantIndexOp>(loc, i), coord.iter1);
      for (auto [loadType, input, inputPaddingInfo] : llvm::zip_equal(
               policy.loadTypes, inputs, policy.inputPaddingInfos.value())) {
        values.emplace_back(
            emitVectorLoad(builder, loc, loadType, input,
                           ValueRange{coord.iter0, iter1, coord.iter2}, mask,
                           inputPaddingInfo));
      }
    }
    return values;
  };

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[1]),
      builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount),
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter, ValueRange) {
        auto curValues =
            fetchInputValues(builder, loc, ReductionCoord3D{zero, iter, zero});
        for (unsigned i = 0; i < policy.unrollCount; ++i) {
          auto results = reduceVectorLanes(
              builder, loc, combineOpDesc,
              ValueRange(curValues).slice(i * numOutputs, numOutputs));
          for (auto [result, output] : llvm::zip_equal(results, outputs)) {
            builder.create<memref::StoreOp>(
                loc, result, output,
                ValueRange{zero,
                           builder.create<arith::AddIOp>(
                               loc,
                               builder.create<arith::ConstantIndexOp>(loc, i),
                               iter),
                           zero});
          }
        }
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis2VecAxis1UnrollAxis2(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.inputDims[0] == 1 && "input dim 0 must be 1");
  assert(context.reductionAxis == 2 && "reduce axis must be 2");
  assert(policy.vectorAxis == 1 && "vector axis must be 1");
  assert(policy.unrollAxis == 2 && "unroll axis must be 2");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[1]),
      builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength),
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs0) {
        SmallVector<Value> initArgs = emitUnrolledStrideLoads(
            builder, loc, policy, inputs, context, {zero, iter0, zero});
        auto loop1 = builder.create<scf::ForOp>(
            loc,
            builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount),
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount),
            initArgs,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              auto curValues = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {zero, iter0, iter1});
              SmallVector<Value> results = emitUnrollCombine(
                  builder, loc, combineOpDesc, ValueRange(iterArgs1), curValues,
                  policy.unrollCount, numOutputs, policy.vectorLength);
              builder.create<scf::YieldOp>(loc, results);
            });
        SmallVector<Value> partials(loop1.getResults());
        assert(partials.size() == policy.unrollCount * numOutputs);
        auto results =
            reduceUnrolledPartials(builder, loc, combineOpDesc, partials,
                                   numOutputs, policy.vectorLength);
        for (auto [result, output] : llvm::zip_equal(results, outputs)) {
          builder.create<vector::StoreOp>(loc, result, output,
                                          ValueRange{zero, iter0, zero});
        }
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis2VecAxis1UnrollAxis1(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.inputDims[0] == 1 && "input dim 0 must be 1");
  assert(context.reductionAxis == 2 && "reduce axis must be 2");
  assert(policy.vectorAxis == 1 && "vector axis must be 1");
  assert(policy.unrollAxis == 1 && "unroll axis must be 1");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[1]),
      builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount *
                                                      policy.vectorLength),
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs0) {
        SmallVector<Value> initArgs = emitUnrolledStrideLoads(
            builder, loc, policy, inputs, context, {zero, iter0, zero});
        auto loop = builder.create<scf::ForOp>(
            loc, one,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            one, initArgs,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              auto curValues = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {zero, iter0, iter1});
              SmallVector<Value> results = emitUnrollCombine(
                  builder, loc, combineOpDesc, ValueRange(iterArgs1), curValues,
                  policy.unrollCount, numOutputs, policy.vectorLength);
              builder.create<scf::YieldOp>(loc, results);
            });
        for (unsigned i = 0; i < policy.unrollCount; ++i) {
          for (unsigned j = 0; j < numOutputs; ++j) {
            builder.create<vector::StoreOp>(
                loc, loop.getResult(i * numOutputs + j), outputs[j],
                ValueRange{zero,
                           builder.create<arith::AddIOp>(
                               loc, iter0,
                               builder.create<arith::ConstantIndexOp>(
                                   loc, i * policy.vectorLength)),
                           zero});
          }
        }
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis2UnrollAxis2(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 2 && "vector axis must be 2");
  assert(policy.unrollAxis == 2 && "unroll axis must be 2");

  unsigned numInputs = inputs.size();
  unsigned numOutputs = outputs.size();

  SmallVector<Value> tarAddrs;
  SmallVector<SmallVector<Value>> tarStridesTable(numInputs);
  triton::gcu::TritonGCUBuilder b(loc, builder);

  for (auto input : inputs) {
    tarAddrs.emplace_back(b.tarAddr(input));
  }

  for (auto output : outputs) {
    tarAddrs.emplace_back(b.tarAddr(output));
  }

  for (auto [loadType, tarStrides] :
       llvm::zip_equal(policy.loadTypes, tarStridesTable)) {
    auto bpe = mlir::triton::gcu::getBpe(loadType.getElementType());
    tarStrides.emplace_back(b.tarStride(loadType, policy.vectorLength * bpe));
    tarStrides.emplace_back(
        b.tarStride(loadType, (context.inputDims[2] -
                               (policy.unrollCount - 1) * policy.vectorLength) *
                                  bpe));
    tarStrides.emplace_back(
        b.tarValue((policy.vectorLength * policy.unrollCount -
                    context.inputDims[1] * context.inputDims[2]) *
                   bpe));
    tarStrides.emplace_back(
        b.tarValue((context.inputDims[1] - 1) * context.inputDims[2] * bpe));
  }

  auto fetchInputValues = [&](OpBuilder &builder, Location loc,
                              SmallVector<Value> &tarAddrs) {
    auto b = triton::gcu::TritonGCUBuilder(loc, builder);
    SmallVector<Value> values;
    for (unsigned i = 0; i < policy.unrollCount - 1; ++i) {
      for (auto [loadType, tarAddr, tarStrides] :
           llvm::zip_equal(policy.loadTypes, tarAddrs, tarStridesTable)) {
        values.emplace_back(b.tarLoad(loadType, tarAddr, tarStrides[0]));
      }
    }
    for (auto [loadType, tarAddr, tarStrides] :
         llvm::zip_equal(policy.loadTypes, tarAddrs, tarStridesTable)) {
      values.emplace_back(b.tarLoad(loadType, tarAddr, tarStrides[1]));
    }
    return values;
  };

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]), one,
      tarAddrs,
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs0) {
        auto loop1 = builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength *
                                                            policy.unrollCount),
            iterArgs0,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              SmallVector<Value> outputTarAddrs(
                  iterArgs1.slice(numInputs, numOutputs));
              SmallVector<Value> initArgs(iterArgs1.take_front(numInputs));
              initArgs.append(fetchInputValues(builder, loc, initArgs));
              auto loop2 = builder.create<scf::ForOp>(
                  loc, one,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  one, initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter2,
                      ValueRange iterArgs2) {
                    SmallVector<Value> inputTarAddrs(
                        iterArgs2.take_front(numInputs));
                    auto curValues =
                        fetchInputValues(builder, loc, inputTarAddrs);
                    SmallVector<Value> results(inputTarAddrs);
                    results.append(emitUnrollCombine(
                        builder, loc, combineOpDesc,
                        ValueRange(iterArgs2).drop_front(numInputs), curValues,
                        policy.unrollCount, numOutputs, policy.vectorLength));
                    builder.create<scf::YieldOp>(loc, results);
                  });
              SmallVector<Value> results(
                  loop2.getResults().take_front(numInputs));
              for (unsigned i = 0; i < numInputs; ++i) {
                b.tarJump(results[i], tarStridesTable[i][2]);
              }
              for (unsigned i = 0; i < policy.unrollCount; ++i) {
                for (unsigned j = 0; j < numOutputs; ++j) {
                  b.tarStore(loop2.getResult(numInputs + i * numOutputs + j),
                             outputTarAddrs[j], tarStridesTable[j][0]);
                }
              }
              results.append(outputTarAddrs);
              builder.create<scf::YieldOp>(loc, results);
            });
        SmallVector<Value> results(loop1.getResults());
        for (unsigned i = 0; i < numInputs; ++i) {
          b.tarJump(results[i], tarStridesTable[i][3]);
        }
        builder.create<scf::YieldOp>(loc, results);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis2UnrollAxis1(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 2 && "vector axis must be 2");
  assert(policy.unrollAxis == 1 && "unroll axis must be 1");

  unsigned numInputs = inputs.size();
  unsigned numOutputs = outputs.size();

  SmallVector<Value> tarAddrs;
  SmallVector<SmallVector<Value>> tarStridesTable(numInputs);
  triton::gcu::TritonGCUBuilder b(loc, builder);

  for (auto input : inputs) {
    tarAddrs.emplace_back(b.tarAddr(input));
  }

  for (auto output : outputs) {
    tarAddrs.emplace_back(b.tarAddr(output));
  }

  for (auto [loadType, tarStrides] :
       llvm::zip_equal(policy.loadTypes, tarStridesTable)) {
    auto bpe = mlir::triton::gcu::getBpe(loadType.getElementType());
    tarStrides.emplace_back(b.tarStride(loadType, context.inputDims[2] * bpe));
    tarStrides.emplace_back(b.tarValue(
        (policy.vectorLength - context.inputDims[1] * context.inputDims[2]) *
        bpe));
    tarStrides.emplace_back(
        b.tarValue((context.inputDims[1] - 1) * context.inputDims[2] * bpe));
    tarStrides.emplace_back(b.tarValue(policy.vectorLength * bpe));
  }

  auto fetchInputValues = [&](OpBuilder &builder, Location loc,
                              SmallVector<Value> &tarAddrs) {
    auto b = triton::gcu::TritonGCUBuilder(loc, builder);
    SmallVector<Value> values;
    for (unsigned i = 0; i < policy.unrollCount; ++i) {
      for (auto [loadType, tarAddr, tarStrides] :
           llvm::zip_equal(policy.loadTypes, tarAddrs, tarStridesTable)) {
        values.emplace_back(b.tarLoad(loadType, tarAddr, tarStrides[0]));
      }
    }
    return values;
  };

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto unrollCountValue =
      builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]), one,
      tarAddrs,
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs0) {
        auto loop1 = builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength),
            iterArgs0,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              SmallVector<Value> initArgs(iterArgs1.take_front(numInputs));
              SmallVector<Value> outputTarAddrs(
                  iterArgs1.slice(numInputs, numOutputs));
              initArgs.append(fetchInputValues(builder, loc, initArgs));
              auto loop2 = builder.create<scf::ForOp>(
                  loc, unrollCountValue,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  unrollCountValue, initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter2,
                      ValueRange iterArgs2) {
                    SmallVector<Value> inputTarAddrs(
                        iterArgs2.take_front(numInputs));
                    auto curValues =
                        fetchInputValues(builder, loc, inputTarAddrs);
                    SmallVector<Value> results(inputTarAddrs);
                    results.append(emitUnrollCombine(
                        builder, loc, combineOpDesc,
                        ValueRange(iterArgs2).drop_front(numInputs), curValues,
                        policy.unrollCount, numOutputs, policy.vectorLength));
                    builder.create<scf::YieldOp>(loc, results);
                  });
              SmallVector<Value> partialResults(
                  loop2.getResults().drop_front(numInputs));
              assert(partialResults.size() == policy.unrollCount * numOutputs);
              SmallVector<Value> results(
                  loop2.getResults().take_front(numInputs));
              for (unsigned i = 0; i < numInputs; ++i) {
                b.tarJump(results[i], tarStridesTable[i][1]);
              }
              for (auto [result, outputTarAddr, tarStrides] : llvm::zip_equal(
                       reduceUnrolledPartials(builder, loc, combineOpDesc,
                                              partialResults, numOutputs,
                                              policy.vectorLength),
                       outputTarAddrs, tarStridesTable)) {
                b.tarStore(result, outputTarAddr, tarStrides[3]);
              }
              results.append(outputTarAddrs);
              builder.create<scf::YieldOp>(loc, results);
            });
        SmallVector<Value> results(loop1.getResults());
        for (unsigned i = 0; i < numInputs; ++i) {
          b.tarJump(results[i], tarStridesTable[i][2]);
        }
        builder.create<scf::YieldOp>(loc, results);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis2UnrollAxis0(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 2 && "vector axis must be 2");
  assert(policy.unrollAxis == 0 && "unroll axis must be 0");

  unsigned numInputs = inputs.size();
  unsigned numOutputs = outputs.size();

  SmallVector<Value> tarAddrs;
  SmallVector<SmallVector<Value>> tarStridesTable(numInputs);
  triton::gcu::TritonGCUBuilder b(loc, builder);

  for (auto input : inputs) {
    tarAddrs.emplace_back(b.tarAddr(input));
  }

  for (auto output : outputs) {
    tarAddrs.emplace_back(b.tarAddr(output));
  }

  for (auto [loadType, tarStrides] :
       llvm::zip_equal(policy.loadTypes, tarStridesTable)) {
    auto bpe = mlir::triton::gcu::getBpe(loadType.getElementType());
    tarStrides.emplace_back(b.tarStride(
        loadType, context.inputDims[1] * context.inputDims[2] * bpe));
    tarStrides.emplace_back(
        b.tarStride(loadType, (context.inputDims[2] -
                               (policy.unrollCount - 1) * context.inputDims[1] *
                                   context.inputDims[2]) *
                                  bpe));
    tarStrides.emplace_back(b.tarValue(
        (policy.vectorLength - context.inputDims[1] * context.inputDims[2]) *
        bpe));
    tarStrides.emplace_back(
        b.tarValue(((policy.unrollCount * context.inputDims[1] - 1) *
                    context.inputDims[2]) *
                   bpe));
    tarStrides.emplace_back(b.tarStride(loadType, context.inputDims[2] * bpe));
    tarStrides.emplace_back(
        b.tarStride(loadType, (policy.vectorLength - (policy.unrollCount - 1) *
                                                         context.inputDims[2]) *
                                  bpe));
    tarStrides.emplace_back(
        b.tarValue((policy.unrollCount - 1) * context.inputDims[2] * bpe));
  }

  auto fetchInputValues = [&](OpBuilder &builder, Location loc,
                              SmallVector<Value> &tarAddrs) {
    auto b = triton::gcu::TritonGCUBuilder(loc, builder);
    SmallVector<Value> values;
    for (unsigned i = 0; i < policy.unrollCount - 1; ++i) {
      for (auto [loadType, tarAddr, tarStrides] :
           llvm::zip_equal(policy.loadTypes, tarAddrs, tarStridesTable)) {
        values.emplace_back(b.tarLoad(loadType, tarAddr, tarStrides[0]));
      }
    }
    for (auto [loadType, tarAddr, tarStrides] :
         llvm::zip_equal(policy.loadTypes, tarAddrs, tarStridesTable)) {
      values.emplace_back(b.tarLoad(loadType, tarAddr, tarStrides[1]));
    }
    return values;
  };

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]),
      builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount), tarAddrs,
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs0) {
        auto loop1 = builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength),
            iterArgs0,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              SmallVector<Value> initArgs(iterArgs1.take_front(numInputs));
              SmallVector<Value> outputTarAddrs(
                  iterArgs1.slice(numInputs, numOutputs));
              initArgs.append(fetchInputValues(builder, loc, initArgs));
              auto loop2 = builder.create<scf::ForOp>(
                  loc, one,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  one, initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter2,
                      ValueRange iterArgs2) {
                    SmallVector<Value> inputTarAddrs(
                        iterArgs2.take_front(numInputs));
                    auto curValues =
                        fetchInputValues(builder, loc, inputTarAddrs);
                    SmallVector<Value> results(inputTarAddrs);
                    results.append(emitUnrollCombine(
                        builder, loc, combineOpDesc,
                        ValueRange(iterArgs2).drop_front(numInputs), curValues,
                        policy.unrollCount, numOutputs, policy.vectorLength));
                    builder.create<scf::YieldOp>(loc, results);
                  });
              SmallVector<Value> results(
                  loop2.getResults().take_front(numInputs));
              for (unsigned i = 0; i < numInputs; ++i) {
                b.tarJump(results[i], tarStridesTable[i][2]);
              }
              for (unsigned i = 0; i < policy.unrollCount - 1; ++i) {
                for (unsigned j = 0; j < numOutputs; ++j) {
                  b.tarStore(loop2.getResult(numInputs + i * numOutputs + j),
                             outputTarAddrs[j], tarStridesTable[j][4]);
                }
              }
              for (unsigned j = 0; j < numOutputs; ++j) {
                b.tarStore(
                    loop2.getResult(numInputs +
                                    (policy.unrollCount - 1) * numOutputs + j),
                    outputTarAddrs[j], tarStridesTable[j][5]);
              }
              results.append(outputTarAddrs);
              builder.create<scf::YieldOp>(loc, results);
            });
        SmallVector<Value> results(loop1.getResults());
        for (unsigned i = 0; i < numInputs; ++i) {
          b.tarJump(results[i], tarStridesTable[i][3]);
        }
        for (unsigned i = 0; i < numOutputs; ++i) {
          b.tarJump(results[numInputs + i], tarStridesTable[i][6]);
        }
        builder.create<scf::YieldOp>(loc, results);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis2WithPadding(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 2 && "vector axis must be 2");
  assert((policy.unrollAxis == 0 || policy.unrollAxis == 1) &&
         "unroll axis must be 0 or 1");
  assert(policy.inputPaddingInfos.has_value() &&
         "input padding infos must be present");
  unsigned numOutputs = outputs.size();
  auto mask = builder.create<vector::ConstantMaskOp>(
      loc,
      VectorType::get(ArrayRef<int64_t>{policy.vectorLength},
                      builder.getI1Type()),
      ArrayRef<int64_t>{context.inputDims[2]});
  auto fetchInputValues = [&](OpBuilder &builder, Location loc,
                              const ReductionCoord3D &coord) {
    SmallVector<Value> values;
    for (unsigned i = 0; i < policy.unrollCount; ++i) {
      auto offset = builder.create<arith::ConstantIndexOp>(loc, i);
      Value iter0 = coord.iter0;
      Value iter1 = coord.iter1;
      if (policy.unrollAxis == 0) {
        iter0 = builder.create<arith::AddIOp>(loc, offset, iter0);
      } else {
        iter1 = builder.create<arith::AddIOp>(loc, offset, iter1);
      }

      for (auto [loadType, input, inputPaddingInfo] : llvm::zip_equal(
               policy.loadTypes, inputs, policy.inputPaddingInfos.value())) {
        values.emplace_back(emitVectorLoad(
            builder, loc, loadType, input,
            ValueRange{iter0, iter1, coord.iter2}, mask, inputPaddingInfo));
      }
    }
    return values;
  };

  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  unsigned outerStep = policy.unrollAxis == 0 ? policy.unrollCount : 1;
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]),
      builder.create<arith::ConstantIndexOp>(loc, outerStep), ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange) {
        auto initValues =
            fetchInputValues(builder, loc, ReductionCoord3D{iter0, zero, zero});
        auto loopCntValue =
            builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount);
        Value innerLowerBound = policy.unrollAxis == 1 ? loopCntValue : one;
        Value innerUpperBound =
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[1]);
        Value innerStep = policy.unrollAxis == 1 ? loopCntValue : one;
        auto loop1 = builder.create<scf::ForOp>(
            loc, innerLowerBound, innerUpperBound, innerStep, initValues,
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs1) {
              auto curValues = fetchInputValues(
                  builder, loc, ReductionCoord3D{iter0, iter1, zero});
              SmallVector<Value> results = emitUnrollCombine(
                  builder, loc, combineOpDesc, ValueRange(iterArgs1), curValues,
                  policy.unrollCount, numOutputs, policy.vectorLength);
              builder.create<scf::YieldOp>(loc, results);
            });
        if (policy.unrollAxis == 1) {
          SmallVector<Value> partials(loop1.getResults());
          assert(partials.size() == policy.unrollCount * numOutputs);
          auto results =
              reduceUnrolledPartials(builder, loc, combineOpDesc, partials,
                                     numOutputs, policy.vectorLength);
          for (auto [inputPaddingInfo, result, output] : llvm::zip_equal(
                   policy.inputPaddingInfos.value(), results, outputs)) {
            emitVectorStore(builder, loc, result, output,
                            ValueRange{iter0, zero, zero}, mask,
                            inputPaddingInfo.needsPadding);
          }
        } else {
          for (unsigned i = 0; i < policy.unrollCount; ++i) {
            auto storeIter0 = builder.create<arith::AddIOp>(
                loc, builder.create<arith::ConstantIndexOp>(loc, i), iter0);
            for (unsigned j = 0; j < numOutputs; ++j) {
              emitVectorStore(builder, loc, loop1.getResult(i * numOutputs + j),
                              outputs[j], ValueRange{storeIter0, zero, zero},
                              mask,
                              policy.inputPaddingInfos.value()[j].needsPadding);
            }
          }
        }
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis1UnrollAxis0(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 1 && "vector axis must be 1");
  assert(policy.unrollAxis == 0 && "unroll axis must be 0");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]),
      builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount),
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            one, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter2, ValueRange) {
              auto initArgs = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {iter0, zero, iter2});
              auto loop = builder.create<scf::ForOp>(
                  loc,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         policy.vectorLength),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         policy.vectorLength),
                  initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    auto curValues =
                        emitUnrolledStrideLoads(builder, loc, policy, inputs,
                                                context, {iter0, iter1, iter2});
                    SmallVector<Value> results = emitUnrollCombine(
                        builder, loc, combineOpDesc, ValueRange(iterArgs),
                        curValues, policy.unrollCount, numOutputs,
                        policy.vectorLength);
                    builder.create<scf::YieldOp>(loc, results);
                  });
              for (unsigned i = 0; i < policy.unrollCount; ++i) {
                auto results =
                    reduceVectorLanes(builder, loc, combineOpDesc,
                                      ValueRange(loop.getResults())
                                          .slice(i * numOutputs, numOutputs));
                auto storeIter0 = builder.create<arith::AddIOp>(
                    loc, builder.create<arith::ConstantIndexOp>(loc, i), iter0);
                for (unsigned j = 0; j < numOutputs; ++j) {
                  builder.create<memref::StoreOp>(
                      loc, results[j], outputs[j],
                      ValueRange{storeIter0, zero, iter2});
                }
              }
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis1UnrollAxis1(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 1 && "vector axis must be 1");
  assert(policy.unrollAxis == 1 && "unroll axis must be 1");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]), one,
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            one, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter2, ValueRange) {
              auto initArgs = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {iter0, zero, iter2});
              auto loop = builder.create<scf::ForOp>(
                  loc,
                  builder.create<arith::ConstantIndexOp>(
                      loc, policy.unrollCount * policy.vectorLength),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  builder.create<arith::ConstantIndexOp>(
                      loc, policy.unrollCount * policy.vectorLength),
                  initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    auto curValues =
                        emitUnrolledStrideLoads(builder, loc, policy, inputs,
                                                context, {iter0, iter1, iter2});
                    SmallVector<Value> results = emitUnrollCombine(
                        builder, loc, combineOpDesc, ValueRange(iterArgs),
                        curValues, policy.unrollCount, numOutputs,
                        policy.vectorLength);
                    builder.create<scf::YieldOp>(loc, results);
                  });
              SmallVector<Value> partials(loop.getResults());
              assert(partials.size() == policy.unrollCount * numOutputs);
              auto reduced =
                  reduceUnrolledPartials(builder, loc, combineOpDesc, partials,
                                         numOutputs, policy.vectorLength);
              auto results =
                  reduceVectorLanes(builder, loc, combineOpDesc, reduced);
              for (unsigned j = 0; j < numOutputs; ++j) {
                builder.create<memref::StoreOp>(loc, results[j], outputs[j],
                                                ValueRange{iter0, zero, iter2});
              }
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis1UnrollAxis2(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 1 && "vector axis must be 1");
  assert(policy.unrollAxis == 2 && "unroll axis must be 2");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]), one,
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount),
            ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter2, ValueRange) {
              auto initArgs = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {iter0, zero, iter2});
              auto loop = builder.create<scf::ForOp>(
                  loc,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         policy.vectorLength),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         policy.vectorLength),
                  initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    auto curValues =
                        emitUnrolledStrideLoads(builder, loc, policy, inputs,
                                                context, {iter0, iter1, iter2});
                    SmallVector<Value> results = emitUnrollCombine(
                        builder, loc, combineOpDesc, ValueRange(iterArgs),
                        curValues, policy.unrollCount, numOutputs,
                        policy.vectorLength);
                    builder.create<scf::YieldOp>(loc, results);
                  });
              for (unsigned i = 0; i < policy.unrollCount; ++i) {
                auto results =
                    reduceVectorLanes(builder, loc, combineOpDesc,
                                      ValueRange(loop.getResults())
                                          .slice(i * numOutputs, numOutputs));
                auto storeIter2 = builder.create<arith::AddIOp>(
                    loc, builder.create<arith::ConstantIndexOp>(loc, i), iter2);
                for (unsigned j = 0; j < numOutputs; ++j) {
                  builder.create<memref::StoreOp>(
                      loc, results[j], outputs[j],
                      ValueRange{iter0, zero, storeIter2});
                }
              }
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis0UnrollAxis0(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 0 && "vector axis must be 0");
  assert(policy.unrollAxis == 0 && "unroll axis must be 0");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]),
      builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount *
                                                      policy.vectorLength),
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            one, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter2, ValueRange) {
              auto initArgs = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {iter0, zero, iter2});
              auto loop = builder.create<scf::ForOp>(
                  loc, one,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  one, initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    auto curValues =
                        emitUnrolledStrideLoads(builder, loc, policy, inputs,
                                                context, {iter0, iter1, iter2});
                    SmallVector<Value> results = emitUnrollCombine(
                        builder, loc, combineOpDesc, ValueRange(iterArgs),
                        curValues, policy.unrollCount, numOutputs,
                        policy.vectorLength);
                    builder.create<scf::YieldOp>(loc, results);
                  });
              auto storeStrideVal = builder.create<arith::ConstantIntOp>(
                  loc, context.inputDims[2], 64);
              for (unsigned i = 0; i < policy.unrollCount; ++i) {
                auto storeIter0 = builder.create<arith::AddIOp>(
                    loc,
                    builder.create<arith::ConstantIndexOp>(
                        loc, i * policy.vectorLength),
                    iter0);
                for (unsigned j = 0; j < numOutputs; ++j) {
                  builder.create<gcu::StoreStrideOp>(
                      loc, loop.getResult(i * numOutputs + j), outputs[j],
                      ValueRange{storeIter0, zero, iter2}, storeStrideVal);
                }
              }
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis0UnrollAxis1(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 0 && "vector axis must be 0");
  assert(policy.unrollAxis == 1 && "unroll axis must be 1");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]),
      builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength),
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            one, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter2, ValueRange) {
              auto initArgs = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {iter0, zero, iter2});
              auto loop = builder.create<scf::ForOp>(
                  loc,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         policy.unrollCount),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         policy.unrollCount),
                  initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    auto curValues =
                        emitUnrolledStrideLoads(builder, loc, policy, inputs,
                                                context, {iter0, iter1, iter2});
                    SmallVector<Value> results = emitUnrollCombine(
                        builder, loc, combineOpDesc, ValueRange(iterArgs),
                        curValues, policy.unrollCount, numOutputs,
                        policy.vectorLength);
                    builder.create<scf::YieldOp>(loc, results);
                  });
              SmallVector<Value> partials(loop.getResults());
              assert(partials.size() == policy.unrollCount * numOutputs);
              auto results =
                  reduceUnrolledPartials(builder, loc, combineOpDesc, partials,
                                         numOutputs, policy.vectorLength);
              auto storeStrideVal = builder.create<arith::ConstantIntOp>(
                  loc, context.inputDims[2], 64);
              for (unsigned j = 0; j < numOutputs; ++j) {
                builder.create<gcu::StoreStrideOp>(
                    loc, results[j], outputs[j], ValueRange{iter0, zero, iter2},
                    storeStrideVal);
              }
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyReduceAxis1VecAxis0UnrollAxis2(
    OpBuilder &builder, Location loc, ArrayRef<Value> outputs,
    ArrayRef<Value> inputs, const Reduction3DContext &context,
    const VectorizationPolicy &policy) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");
  assert(policy.vectorAxis == 0 && "vector axis must be 0");
  assert(policy.unrollAxis == 2 && "unroll axis must be 2");

  unsigned numOutputs = outputs.size();
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]),
      builder.create<arith::ConstantIndexOp>(loc, policy.vectorLength),
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(loc, context.inputDims[2]),
            builder.create<arith::ConstantIndexOp>(loc, policy.unrollCount),
            ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter2, ValueRange) {
              auto initArgs = emitUnrolledStrideLoads(
                  builder, loc, policy, inputs, context, {iter0, zero, iter2});
              auto loop = builder.create<scf::ForOp>(
                  loc, one,
                  builder.create<arith::ConstantIndexOp>(loc,
                                                         context.inputDims[1]),
                  one, initArgs,
                  [&](OpBuilder &builder, Location loc, Value iter1,
                      ValueRange iterArgs) {
                    auto curValues =
                        emitUnrolledStrideLoads(builder, loc, policy, inputs,
                                                context, {iter0, iter1, iter2});
                    SmallVector<Value> results = emitUnrollCombine(
                        builder, loc, combineOpDesc, ValueRange(iterArgs),
                        curValues, policy.unrollCount, numOutputs,
                        policy.vectorLength);
                    builder.create<scf::YieldOp>(loc, results);
                  });
              auto storeStrideVal = builder.create<arith::ConstantIntOp>(
                  loc, context.inputDims[2], 64);
              for (unsigned i = 0; i < policy.unrollCount; ++i) {
                auto storeIter2 = builder.create<arith::AddIOp>(
                    loc, builder.create<arith::ConstantIndexOp>(loc, i), iter2);
                for (unsigned j = 0; j < numOutputs; ++j) {
                  builder.create<gcu::StoreStrideOp>(
                      loc, loop.getResult(i * numOutputs + j), outputs[j],
                      ValueRange{iter0, zero, storeIter2}, storeStrideVal);
                }
              }
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });
}

void ReduceGenerator::applyScalarImpl(OpBuilder &builder, Location loc,
                                      ArrayRef<Value> outputs,
                                      ArrayRef<Value> inputs,
                                      const Reduction3DContext &context) const {
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto one = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto numOutputs = outputs.size();
  builder.create<scf::ForOp>(
      loc, zero,
      builder.create<arith::ConstantIndexOp>(loc, context.inputDims[0]), one,
      ValueRange{},
      [&](OpBuilder &builder, Location loc, Value iter0, ValueRange iterArgs) {
        builder.create<scf::ForOp>(
            loc, zero,
            builder.create<arith::ConstantIndexOp>(
                loc, context.inputDims[3 - context.reductionAxis]),
            one, ValueRange{},
            [&](OpBuilder &builder, Location loc, Value iter1,
                ValueRange iterArgs) {
              SmallVector<Value> iterators(3);
              iterators[0] = iter0;
              iterators[3 - context.reductionAxis] = iter1;
              iterators[context.reductionAxis] = zero;
              auto initValues =
                  llvm::to_vector(llvm::map_range(inputs, [&](auto input) {
                    return builder.create<memref::LoadOp>(loc, input, iterators)
                        .getResult();
                  }));
              auto loop = builder.create<scf::ForOp>(
                  loc, one,
                  builder.create<arith::ConstantIndexOp>(
                      loc, context.inputDims[context.reductionAxis]),
                  one, initValues,
                  [&](OpBuilder &builder, Location loc, Value iter2,
                      ValueRange iterArgs) {
                    SmallVector<Value> combineOperands(iterArgs.begin(),
                                                       iterArgs.end());
                    SmallVector<Value> coord(iterators);
                    coord[context.reductionAxis] = iter2;
                    llvm::transform(inputs, std::back_inserter(combineOperands),
                                    [&](auto input) {
                                      return builder
                                          .create<memref::LoadOp>(loc, input,
                                                                  coord)
                                          .getResult();
                                    });
                    auto combineFuncResults = combineOpDesc.applyScalarCombine(
                        builder, loc, combineOperands);
                    builder.create<scf::YieldOp>(
                        loc, ValueRange{combineFuncResults});
                  });
              for (unsigned i = 0; i < numOutputs; ++i) {
                builder.create<memref::StoreOp>(loc, loop.getResult(i),
                                                outputs[i], iterators);
              }
              builder.create<scf::YieldOp>(loc);
            });
        builder.create<scf::YieldOp>(loc);
      });
}

std::optional<VectorizationPolicy>
ReduceGenerator::tryBuildReduceAxis2WithPaddingPolicy(
    OpBuilder &builder, ArrayRef<Value> inputs,
    const Reduction3DContext &context) const {
  auto identityAttrs = combineOpDesc.inferIdentityAttrs(builder);
  if (failed(identityAttrs))
    return std::nullopt;

  VectorizationPolicy policy;
  policy.vectorLength = *vectorLength;
  policy.vectorAxis = 2;
  policy.unrollAxis = 1;
  policy.loadTypes = buildVectorLoadTypes(inputs, policy.vectorLength);
  policy.inputPaddingInfos.emplace();
  for (auto [input, identityAttr] :
       llvm::zip_equal(inputs, identityAttrs.value())) {
    Type elementType = cast<MemRefType>(input.getType()).getElementType();
    policy.inputPaddingInfos->emplace_back(InputPaddingInfo{
        needsVectorPadding(context.inputDims[2], elementType), identityAttr});
  }
  return policy;
}

VectorizationPolicy ReduceGenerator::buildReduceAxis1WithPaddingPolicy(
    OpBuilder &builder, ArrayRef<Value> inputs,
    const Reduction3DContext &context) const {
  VectorizationPolicy policy;
  policy.vectorLength = *vectorLength;
  policy.vectorAxis = 2;
  // Prefer dim 0 for unrolling when dim 1 is short and dim 0 is larger;
  // otherwise default to dim 1.
  policy.unrollAxis = (context.inputDims[1] < kLoopUnrollTimes &&
                       context.inputDims[0] > context.inputDims[1])
                          ? 0
                          : 1;
  policy.loadTypes = buildVectorLoadTypes(inputs, policy.vectorLength);
  policy.inputPaddingInfos.emplace();
  for (Value input : inputs) {
    Type elementType = cast<MemRefType>(input.getType()).getElementType();
    policy.inputPaddingInfos->emplace_back(
        InputPaddingInfo{needsVectorPadding(context.inputDims[2], elementType),
                         builder.getZeroAttr(elementType)});
  }
  return policy;
}

std::optional<VectorizationPolicy> ReduceGenerator::tryBuildReduceAxis2Policy(
    OpBuilder &builder, ArrayRef<Value> inputs,
    const Reduction3DContext &context) const {
  assert(context.reductionAxis == 2 && "reduce axis must be 2");

  // No axis is wide enough for a vector tile -> fall back to padding.
  if (context.inputDims[2] < *vectorLength &&
      context.inputDims[1] < *vectorLength)
    return tryBuildReduceAxis2WithPaddingPolicy(builder, inputs, context);

  VectorizationPolicy policy;
  policy.vectorLength = *vectorLength;
  policy.loadTypes = buildVectorLoadTypes(inputs, policy.vectorLength);
  if (context.inputDims[2] >= policy.vectorLength) {
    policy.vectorAxis = 2;
  } else {
    // TODO(peng.tian): vectorDim=1 is not always faster than vectorDim=2 with
    // padding; revisit with a cost model.
    policy.vectorAxis = 1;
  }
  int64_t peerDim = policy.vectorAxis == 2 ? 1 : 2;
  policy.unrollAxis =
      context.inputDims[peerDim] < kLoopUnrollTimes &&
              context.inputDims[policy.vectorAxis] / policy.vectorLength >
                  context.inputDims[peerDim]
          ? policy.vectorAxis
          : peerDim;
  return policy;
}

std::optional<VectorizationPolicy> ReduceGenerator::tryBuildReduceAxis1Policy(
    OpBuilder &builder, ArrayRef<Value> inputs,
    const Reduction3DContext &context) const {
  assert(context.reductionAxis == 1 && "reduce axis must be 1");

  constexpr std::array<VectorAxisCandidate, 3> vectorAxisCandidates = {{
      {2, {1, 0}},
      {0, {1, 2}},
      {1, {2, 0}},
  }};

  // Try to find a dimension wide enough for vectorization, in priority order.
  auto it = llvm::find_if(vectorAxisCandidates, [&](auto candidate) {
    return context.inputDims[candidate.vectorAxis] >= *vectorLength;
  });
  // No dimension is wide enough for a vector tile -> fall back to padding.
  if (it == vectorAxisCandidates.end()) {
    return buildReduceAxis1WithPaddingPolicy(builder, inputs, context);
  }

  VectorizationPolicy policy;
  policy.vectorLength = *vectorLength;
  policy.loadTypes = buildVectorLoadTypes(inputs, policy.vectorLength);
  policy.vectorAxis = it->vectorAxis;

  auto numVectorTiles =
      context.inputDims[policy.vectorAxis] / policy.vectorLength;
  if (numVectorTiles >= kLoopUnrollTimes) {
    policy.unrollAxis = policy.vectorAxis;
    return policy;
  }

  for (auto peerDim : it->peerDims) {
    if (context.inputDims[peerDim] >= kLoopUnrollTimes) {
      policy.unrollAxis = peerDim;
      return policy;
    }
  }

  auto candidatePeerDim =
      context.inputDims[it->peerDims[0]] >= context.inputDims[it->peerDims[1]]
          ? it->peerDims[0]
          : it->peerDims[1];
  policy.unrollAxis = (context.inputDims[candidatePeerDim] > numVectorTiles)
                          ? candidatePeerDim
                          : policy.vectorAxis;
  return policy;
}

DispatchPlan
ReduceGenerator::buildDispatchPlan(OpBuilder &builder, ArrayRef<Value> inputs,
                                   const Reduction3DContext &context) const {
  if (!vectorLength) {
    return DispatchPlan{ReduceExecutionMode::kScalar, std::nullopt};
  }
  std::optional<VectorizationPolicy> policy;
  if (context.reductionAxis == 2) {
    policy = tryBuildReduceAxis2Policy(builder, inputs, context);
  } else if (context.reductionAxis == 1) {
    policy = tryBuildReduceAxis1Policy(builder, inputs, context);
  } else {
    llvm_unreachable("reduce axis must be 1 or 2");
  }
  if (!policy) {
    return DispatchPlan{ReduceExecutionMode::kScalar, std::nullopt};
  }

  policy->unrollCount = computeUnrollCount(*policy, context);
  return DispatchPlan{ReduceExecutionMode::kVectorized, policy};
}

void ReduceGenerator::applyReduce(OpBuilder &builder, Location loc,
                                  ArrayRef<Value> outputs,
                                  ArrayRef<Value> inputs,
                                  const Reduction3DContext &context,
                                  const DispatchPlan &dispatchPlan) const {
  if (dispatchPlan.executionMode == ReduceExecutionMode::kScalar) {
    applyScalarImpl(builder, loc, outputs, inputs, context);
    return;
  }
  assert(dispatchPlan.executionMode == ReduceExecutionMode::kVectorized &&
         "execution mode must be vectorized");
  assert(dispatchPlan.vectorizationPolicy.has_value() &&
         "vectorization policy must be present");
  auto &policy = dispatchPlan.vectorizationPolicy.value();

  if (context.reductionAxis == 2) {
    if (policy.inputPaddingInfos.has_value())
      return applyReduceAxis2VecAxis2WithPadding(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 2 && policy.unrollAxis == 2)
      return applyReduceAxis2VecAxis2UnrollAxis2(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 2 && policy.unrollAxis == 1)
      return applyReduceAxis2VecAxis2UnrollAxis1(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 1 && policy.unrollAxis == 2)
      return applyReduceAxis2VecAxis1UnrollAxis2(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 1 && policy.unrollAxis == 1)
      return applyReduceAxis2VecAxis1UnrollAxis1(builder, loc, outputs, inputs,
                                                 context, policy);
    llvm_unreachable("unhandled vectorAxis/unrollAxis combination");
  } else if (context.reductionAxis == 1) {
    if (policy.inputPaddingInfos.has_value())
      return applyReduceAxis1VecAxis2WithPadding(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 2 && policy.unrollAxis == 2)
      return applyReduceAxis1VecAxis2UnrollAxis2(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 2 && policy.unrollAxis == 1)
      return applyReduceAxis1VecAxis2UnrollAxis1(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 2 && policy.unrollAxis == 0)
      return applyReduceAxis1VecAxis2UnrollAxis0(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 1 && policy.unrollAxis == 0)
      return applyReduceAxis1VecAxis1UnrollAxis0(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 1 && policy.unrollAxis == 1)
      return applyReduceAxis1VecAxis1UnrollAxis1(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 1 && policy.unrollAxis == 2)
      return applyReduceAxis1VecAxis1UnrollAxis2(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 0 && policy.unrollAxis == 0)
      return applyReduceAxis1VecAxis0UnrollAxis0(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 0 && policy.unrollAxis == 1)
      return applyReduceAxis1VecAxis0UnrollAxis1(builder, loc, outputs, inputs,
                                                 context, policy);
    if (policy.vectorAxis == 0 && policy.unrollAxis == 2)
      return applyReduceAxis1VecAxis0UnrollAxis2(builder, loc, outputs, inputs,
                                                 context, policy);
    applyScalarImpl(builder, loc, outputs, inputs, context);
  } else {
    llvm_unreachable("reduce axis must be 1 or 2");
  }
}

struct TTReduceOpLowering : SharedConversionPattern<triton::ReduceOp> {
  using SharedConversionPattern::SharedConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op.getOperation());
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }

    auto loc = op.getLoc();
    auto axis = op.getAxis();
    auto inputTypes = op.getInputTypes();
    auto numOutput = op->getNumResults();
    auto elemsPerThread = triton::gcu::getElemsPerThread(inputTypes[0]);
    SmallVector<int64_t> outputElemsPerThread(elemsPerThread.begin(),
                                              elemsPerThread.end());
    outputElemsPerThread[axis] = 1;
    SmallVector<Value> outputs;
    for (auto result : op.getResults()) {
      auto lastUser = userAnalysis.getLastUser(result);
      outputs.emplace_back(
          syncAllocOp(rewriter, loc, lastUser, userAnalysis, replaced2Origin,
                      MemRefType::get(outputElemsPerThread,
                                      getElementTypeOrSelf(result.getType()))));
    }

    Reduction3DContext context =
        ReduceGenerator::normalizeReductionTo3D(elemsPerThread, axis);
    SmallVector<Value> reduceInputs = reshapeToContiguous3D(
        rewriter, loc, adaptor.getSrcs(), context.inputDims);
    SmallVector<Value> reduceOutputs =
        reshapeToContiguous3D(rewriter, loc, outputs, context.outputDims);

    applyReduce(op, rewriter, reduceOutputs, reduceInputs, context.inputDims,
                context.reductionAxis);

    auto slicedAxies = getSlicedAxies(inputTypes[0]);
    if (slicedAxies.contains(axis)) {
      SmallVector<int64_t> sharedMemShape(inputTypes[0].getShape());
      auto encodingAttr = inputTypes[0].getEncoding();
      // use gcu triton::gcu::getWarpsPerCTA
      auto warpsPerCTA = triton::gcu::getWarpsPerCTA(encodingAttr);
      if (warpsPerCTA.size() != sharedMemShape.size()) {
        op.dump();
        assert(false && "the reduce input layout is not a blockencoding!");
      }

      if (warpsPerCTA[axis] < sharedMemShape[axis]) {
        sharedMemShape[axis] = warpsPerCTA[axis];
      }

      bool isReduce1D =
          sharedMemShape[axis] == std::accumulate(sharedMemShape.begin(),
                                                  sharedMemShape.end(), 1,
                                                  std::multiplies<unsigned>());
      triton::gcu::TagInfo tag;
      if (!isReduce1D) {
        tag = pTagPool.getPrivateSyncTagInfo(op);
      }
      SmallVector<Value> sharedBuffers;
      sharedBuffers.reserve(numOutput);
      auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      for (unsigned i = 0; i < numOutput; ++i) {
        auto elementType = getElementTypeOrSelf(inputTypes[i]);
        elementType = elementType.isInteger(1) && isReduce1D
                          ? rewriter.getI8Type()
                          : elementType;
        auto sharedMemRefType =
            MemRefType::get(sharedMemShape, elementType, AffineMap{},
                            rewriter.getI64IntegerAttr(2) /*shared memory*/);
        sharedBuffers.emplace_back(
            rewriter.create<memref::AllocOp>(loc, sharedMemRefType));
        if (isReduce1D) {
          rewriter.create<memref::StoreOp>(
              loc,
              rewriter.create<memref::LoadOp>(loc, reduceOutputs[i],
                                              ValueRange{zero, zero, zero}),
              sharedBuffers.back(),
              ValueRange{getWarpIds(rewriter, loc, inputTypes[i])});
          rewriter.create<gpu::BarrierOp>(loc);
        } else {
          storeToSharedMem(rewriter, tag, inputTypes[i], sharedBuffers.back(),
                           outputs[i], false);
        }
      }

      Reduction3DContext partialReductionContext = context;

      if (warpsPerCTA[axis] < sharedMemShape[axis]) {
        partialReductionContext
            .inputDims[partialReductionContext.reductionAxis] =
            warpsPerCTA[axis];
      } else {
        partialReductionContext
            .inputDims[partialReductionContext.reductionAxis] =
            sharedMemShape[axis];
      }
      auto loadFromShareForAllReduce =
          [&](OpBuilder &builder, triton::gcu::TagInfo tag, Type type,
              Value buffer, triton::gcu::FirstLastUserAnalysis &userAnalysis,
              std::map<Operation *, Operation *> &replaced2Origin) {
            auto loc = buffer.getLoc();
            auto srcType = dyn_cast<MemRefType>(buffer.getType());
            auto numElems = triton::gcu::getElemsPerThread(type);
            numElems[axis] = warpsPerCTA[axis];
            auto totalNumElems = builder.create<arith::ConstantIndexOp>(
                loc, std::accumulate(numElems.begin(), numElems.end(), 1,
                                     std::multiplies<unsigned>()));
            auto outputType = MemRefType::get(
                SmallVector<int64_t>(numElems.begin(), numElems.end()),
                srcType.getElementType());
            auto warpIds = getWarpIds(builder, loc, type);
            SmallVector<Value, 4> offsets;
            for (unsigned i = 0; i < srcType.getRank(); ++i) {
              if (i == axis) {
                offsets.push_back(
                    builder.create<arith::ConstantIntOp>(loc, 0, 32));
              } else {
                offsets.push_back(builder.create<arith::MulIOp>(
                    loc,
                    builder.create<arith::ConstantIntOp>(loc, numElems[i], 32),
                    builder.create<arith::IndexCastOp>(
                        loc, builder.getI32Type(), warpIds[i])));
              }
            }
            auto output =
                syncAllocOp(builder, loc, std::make_pair(op.getOperation(), -1),
                            userAnalysis, replaced2Origin, outputType);
            auto defaultValue = triton::gcu::createConstantZero(
                builder, loc, srcType.getElementType());
            if (srcType.getRank() > 5) {
              SmallVector<Value, 4> mergedOffsets;
              Value src;
              Value dst;
              mergeContinuousDims(builder, loc, src, dst, offsets,
                                  mergedOffsets, srcType, outputType, buffer,
                                  output);
              builder.create<memref_ext::SliceStartOp>(
                  loc, dst, src, mergedOffsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
              auto [oriOutputStrides, oriOutputOffset] =
                  outputType.getStridesAndOffset();
              builder.create<memref::ReinterpretCastOp>(
                  loc, outputType, dst, oriOutputOffset,
                  SmallVector<int64_t>(numElems.begin(), numElems.end()),
                  oriOutputStrides);
            } else {
              builder.create<memref_ext::SliceStartOp>(
                  loc, output, buffer, offsets, defaultValue, tag.getTag(),
                  ValueRange{tag.getIdx()});
            }
            builder.create<memref::DmaWaitOp>(
                loc, tag.getTag(), ValueRange{tag.getIdx()}, totalNumElems);
            return output;
          };

      SmallVector<Value, 4> warpReduceInputs;
      for (unsigned i = 0; i < numOutput; ++i) {
        if (isReduce1D) {
          warpReduceInputs.push_back(sharedBuffers[i]);
        } else {
          auto tensorType = RankedTensorType::get(
              sharedMemShape, getElementTypeOrSelf(inputTypes[i]),
              encodingAttr);
          warpReduceInputs.emplace_back(loadFromShareForAllReduce(
              rewriter, tag, tensorType, sharedBuffers[i], userAnalysis,
              replaced2Origin));
        }
      }

      warpReduceInputs = reshapeToContiguous3D(
          rewriter, loc, warpReduceInputs, partialReductionContext.inputDims);
      applyReduce(op, rewriter, reduceOutputs, warpReduceInputs,
                  partialReductionContext.inputDims,
                  partialReductionContext.reductionAxis);
      for (auto buffer : sharedBuffers) {
        rewriter.create<memref::DeallocOp>(loc, buffer);
      }
    }

    SmallVector<Value, 4> finalOutputs;
    for (unsigned i = 0; i < numOutput; ++i) {
      auto output = outputs[i];
      if (!isa<ShapedType>(op.getResultTypes()[i])) {
        auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        output = rewriter.create<memref::LoadOp>(loc, output, ValueRange{zero});
      } else {
        auto resultType = dyn_cast<MemRefType>(
            getTypeConverter()->convertType(op.getResultTypes()[i]));
        if (resultType.getNumElements() !=
            dyn_cast<MemRefType>(output.getType()).getNumElements()) {
          return op.emitOpError("element number mismatch: ")
                 << resultType.getNumElements() << " vs "
                 << dyn_cast<MemRefType>(output.getType()).getNumElements();
        }
        auto [strides, offset] = resultType.getStridesAndOffset();
        output = rewriter.create<memref::ReinterpretCastOp>(
            loc, resultType, output, offset, resultType.getShape(), strides);
      }
      finalOutputs.push_back(output);
    }
    leaveTritionOp(rewriter, op.getOperation());
    rewriter.replaceOp(op, finalOutputs);
    return success();
  }

private:
  void applyReduce(triton::ReduceOp op, OpBuilder &rewriter,
                   ArrayRef<Value> outputs, ArrayRef<Value> inputs,
                   const std::array<int64_t, 3> &reduceDims,
                   int64_t reduceAxis) const {
    ReduceGenerator generator(op);
    Reduction3DContext context;
    context.inputDims = reduceDims;
    context.outputDims = reduceDims;
    context.outputDims[reduceAxis] = 1;
    context.reductionAxis = reduceAxis;

    auto plan = generator.buildDispatchPlan(rewriter, inputs, context);
    if (plan.vectorizationPolicy && plan.vectorizationPolicy->transposeInfo &&
        plan.vectorizationPolicy->transposeInfo->doTranspose) {
      auto loc = op.getLoc();
      auto tag = pTagPool.getPrivateSyncTagInfo(op);
      const std::array<int64_t, 3> &layout =
          plan.vectorizationPolicy->transposeInfo->transposeLayout;
      Reduction3DContext transposedContext;
      for (unsigned i = 0; i < 3; ++i) {
        transposedContext.inputDims[i] = context.inputDims[layout[i]];
        transposedContext.outputDims[i] = context.outputDims[layout[i]];
      }
      transposedContext.reductionAxis = layout[context.reductionAxis];

      SmallVector<Value> transposedInputs =
          llvm::to_vector(llvm::map_range(inputs, [&](auto input) {
            Value tmpBuffer = rewriter.create<memref::AllocOp>(
                loc, MemRefType::get(
                         ArrayRef<int64_t>{transposedContext.inputDims},
                         cast<MemRefType>(input.getType()).getElementType()));
            return tmpBuffer;
          }));
      doTranspose(rewriter, loc, transposedInputs, inputs, layout, tag);
      auto transposedOutputs = prepareTransposedOutputs(
          rewriter, loc, outputs, transposedContext.outputDims, layout);
      auto transposedPlan = generator.buildDispatchPlan(
          rewriter, transposedInputs, transposedContext);
      generator.applyReduce(rewriter, loc, transposedOutputs, transposedInputs,
                            transposedContext, transposedPlan);
      if (layout[2] == 0) {
        doTranspose(rewriter, loc, outputs, transposedOutputs, layout, tag);
        for (auto buf : transposedOutputs) {
          rewriter.create<memref::DeallocOp>(loc, buf);
        }
      }
      for (auto buf : transposedInputs) {
        rewriter.create<memref::DeallocOp>(loc, buf);
      }
      return;
    }
    generator.applyReduce(rewriter, op.getLoc(), outputs, inputs, context,
                          plan);
  }
};
} // namespace

void mlir::triton::populateReduceOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    triton::gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::PrivateTagPool &pTagPool) {
  patterns.add<TTReduceOpLowering>(converter, patterns.getContext(),
                                   userAnalysis, replaced2Origin, pTagPool);
}
