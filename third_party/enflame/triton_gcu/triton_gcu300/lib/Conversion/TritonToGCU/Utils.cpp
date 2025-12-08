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
#include "Utils.h"
#include <algorithm>
#include <functional>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"
#define DEBUG_TYPE "triton-ir-to-gcu-ir-util"
namespace mlir {
namespace triton {
namespace gcu {

bool get_bool_env(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  std::string str_value(value);
  std::transform(str_value.begin(), str_value.end(), str_value.begin(),
                 ::tolower);
  return (str_value == "true" || str_value == "1" || str_value == "on" ||
          str_value == "yes");
}

SmallVector<unsigned> getWarpsPerCTA(Attribute layout) {
  if (auto blockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(layout)) {
    return blockEnc.getWarpsPerCTA();

  } else if (auto sliceEnc = dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parent = sliceEnc.getParent();
    SmallVector<unsigned> sliceDims;
    sliceDims.push_back(sliceEnc.getDim());
    while (auto innerSliceEnc =
               dyn_cast<triton::gpu::SliceEncodingAttr>(parent)) {
      auto curSliceDim = innerSliceEnc.getDim();
      for (size_t idx = 0; idx < sliceDims.size(); idx++) {
        if (sliceDims[idx] >= curSliceDim) {
          sliceDims[idx] = sliceDims[idx] + 1;
        }
      }
      sliceDims.push_back(curSliceDim);
      parent = innerSliceEnc.getParent();
    }
    if (!isa<triton::gpu::BlockedEncodingAttr>(parent)) {
      llvm::report_fatal_error("[Error] bad slice layout parent");
      assert(false && "bad slice layout parent");
      triton::gpu::getWarpsPerCTA(layout);
    }
    auto blockEncParent = dyn_cast<triton::gpu::BlockedEncodingAttr>(parent);
    auto parentWarpsPerCTA = blockEncParent.getWarpsPerCTA();
    SmallVector<unsigned> warpsPerCTA;
    for (unsigned i = 0; i < parentWarpsPerCTA.size(); ++i) {
      if (!llvm::is_contained(sliceDims, i)) {
        warpsPerCTA.push_back(parentWarpsPerCTA[i]);
      }
    }
    return warpsPerCTA;

  } else {
    return triton::gpu::getWarpsPerCTA(layout);
  }
}

SmallVector<unsigned> getElemsPerThread(Type type) {
  if (auto tType = dyn_cast<RankedTensorType>(type)) {
    if (auto dotEnc = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
            tType.getEncoding())) {
      // dot lhs and rhs should have different slicing by op id but
      // DotOperandEncodingAttr no supported and currently support 2D dot first
      auto shape = tType.getShape();
      if (auto blockedLayout =
              dyn_cast<triton::gpu::BlockedEncodingAttr>(dotEnc.getParent())) {
        auto rank = shape.size();
        SmallVector<unsigned> elemsPerthread(rank, 1);
        // low 2 rank do dot
        auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
        for (unsigned idx = 0; idx < rank - 2; idx++) {
          elemsPerthread[idx] = shape[idx];
          if (warpsPerCTA[idx] > 1) {
            LLVM_DEBUG({
              llvm::dbgs() << "hi slice should in lower 2 dims for dot \n";
              dotEnc.dump();
            });
          }
          assert((warpsPerCTA[idx] == 1) &&
                 "hi slice should in lower 2 dims for dot\n");
        }
        bool isM = dotEnc.getOpIdx() == 0;
        // only debug check
        if (isM) {
          int64_t k = shape[rank - 1];
          elemsPerthread[rank - 1] = k;
          elemsPerthread[rank - 2] = shape[rank - 2] / warpsPerCTA[rank - 2];
        } else {
          int64_t k = shape[rank - 2];
          elemsPerthread[rank - 2] = k;
          elemsPerthread[rank - 1] = shape[rank - 1] / warpsPerCTA[rank - 1];
        }
        return elemsPerthread;
      }
    } else if (mlir::isa<triton::gpu::SharedEncodingTrait>(
                   tType.getEncoding())) {
      return SmallVector<unsigned>(tType.getShape().begin(),
                                   tType.getShape().end());
    } else if (auto blockEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(
                   tType.getEncoding())) {
      auto shape = tType.getShape();
      size_t rank = shape.size();
      SmallVector<unsigned> sizePerThread(rank, 1);
      auto warpsPerCTA = blockEnc.getWarpsPerCTA();
      auto threadsPerWarp = blockEnc.getThreadsPerWarp();
      auto shapePerCTA = triton::gpu::getShapePerCTA(blockEnc, shape);
      assert(rank == sizePerThread.size() &&
             "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
      SmallVector<unsigned> elemsPerThread(rank);
      for (size_t i = 0; i < rank; ++i) {
        unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
        elemsPerThread[i] =
            ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
      }
      return elemsPerThread;
    } else if (auto linearEnc = dyn_cast<triton::gpu::LinearEncodingAttr>(
                   tType.getEncoding())) {
      auto shape = tType.getShape();
      size_t rank = shape.size();
      SmallVector<unsigned> sizePerThread(rank, 1);
      auto warpsPerCTA = linearEnc.getWarpsPerCTA();
      auto threadsPerWarp = linearEnc.getThreadsPerWarp();
      assert(rank == sizePerThread.size() &&
             "unexpected rank in LinearEncodingAttr::getElemsPerThread");
      SmallVector<unsigned> elemsPerThread(rank);
      for (size_t i = 0; i < rank; ++i) {
        unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
        elemsPerThread[i] = ceil<unsigned>(shape[i], t) * sizePerThread[i];
      }
      return elemsPerThread;
    } else if (auto sliceEnc = dyn_cast<triton::gpu::SliceEncodingAttr>(
                   tType.getEncoding())) {
      auto parent = sliceEnc.getParent();
      auto outShape = sliceEnc.paddedShape(tType.getShape());
      SmallVector<unsigned> sliceDims;
      sliceDims.push_back(sliceEnc.getDim());
      while (auto innerSliceEnc =
                 dyn_cast<triton::gpu::SliceEncodingAttr>(parent)) {
        llvm::ArrayRef<int64_t> inputShpe = outShape;
        outShape = innerSliceEnc.paddedShape(inputShpe);
        auto curSliceDim = innerSliceEnc.getDim();
        for (size_t idx = 0; idx < sliceDims.size(); idx++) {
          if (sliceDims[idx] >= curSliceDim) {
            sliceDims[idx] = sliceDims[idx] + 1;
          }
        }
        sliceDims.push_back(curSliceDim);
        parent = innerSliceEnc.getParent();
      }
      if (!isa<triton::gpu::BlockedEncodingAttr>(parent)) {
        return triton::gpu::getElemsPerThread(type);
      }
      auto blockEncParent = dyn_cast<triton::gpu::BlockedEncodingAttr>(parent);
      size_t rank = outShape.size();
      SmallVector<unsigned> sizePerThread(rank, 1);
      auto warpsPerCTA = blockEncParent.getWarpsPerCTA();
      auto threadsPerWarp = blockEncParent.getThreadsPerWarp();
      auto shapePerCTA = triton::gpu::getShapePerCTA(blockEncParent, outShape);
      assert(rank == sizePerThread.size() &&
             "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
      SmallVector<unsigned> parentElemsPerThread(rank);
      for (size_t i = 0; i < rank; ++i) {
        unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
        parentElemsPerThread[i] =
            ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
      }
      SmallVector<unsigned> elemsPerThread;
      for (unsigned i = 0; i < rank; ++i) {
        if (!llvm::is_contained(sliceDims, i)) {
          elemsPerThread.push_back(parentElemsPerThread[i]);
        }
      }
      return elemsPerThread;
    } else {
      return triton::gpu::getElemsPerThread(type);
    }
  }
  return triton::gpu::getElemsPerThread(type);
}

unsigned getTotalElemsPerThread(Type type) {
  if (auto tType = dyn_cast<RankedTensorType>(type)) {
    if (auto enc = tType.getEncoding()) {
      if (llvm::isa_and_nonnull<triton::gpu::DotOperandEncodingAttr>(enc)) {
        auto elemsPerthread = gcu::getElemsPerThread(type);
        return std::accumulate(elemsPerthread.begin(), elemsPerthread.end(), 1,
                               std::multiplies<unsigned>());
      } else if (mlir::isa<triton::gpu::SharedEncodingTrait>(enc)) {
        return std::accumulate(tType.getShape().begin(), tType.getShape().end(),
                               1, std::multiplies<unsigned>());
      } else if (llvm::isa_and_nonnull<triton::gpu::BlockedEncodingAttr>(
                     tType.getEncoding()) ||
                 llvm::isa_and_nonnull<triton::gpu::LinearEncodingAttr>(
                     tType.getEncoding())) {
        auto elemsPerthread = gcu::getElemsPerThread(type);
        return std::accumulate(elemsPerthread.begin(), elemsPerthread.end(), 1,
                               std::multiplies<unsigned>());

      } else if (llvm::isa_and_nonnull<triton::gpu::SliceEncodingAttr>(
                     tType.getEncoding())) {
        auto elemsPerthread = gcu::getElemsPerThread(type);
        return std::accumulate(elemsPerthread.begin(), elemsPerthread.end(), 1,
                               std::multiplies<unsigned>());
      } else {
        return triton::gpu::getTotalElemsPerThread(type);
      }
    }
  }
  return triton::gpu::getTotalElemsPerThread(type);
}

unsigned getNumWarpsPerCTA(Type type) {
  if (auto tType = dyn_cast<RankedTensorType>(type)) {
    return tType.getEncoding()
               ? triton::gpu::getNumWarpsPerCTA(tType.getEncoding())
               : 1;
  }
  return 1;
}

unsigned getBpe(Type type) {
  assert(type.isIntOrFloat());
  return ((type.getIntOrFloatBitWidth() + 7) / 8);
}

} // namespace gcu
} // namespace triton
} // namespace mlir
