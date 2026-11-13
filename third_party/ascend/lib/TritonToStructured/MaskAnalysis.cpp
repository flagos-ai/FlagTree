/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
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

#include "TritonToStructured/MaskAnalysis.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "TritonToStructured/PtrAnalysis.h"
#include "Utils/Utils.h"

#define DEBUG_TYPE "triton-to-structured-mask-analysis"

namespace TritonToStructured {
using namespace mlir;
using namespace triton;

bool dimInfo::setType(arith::CmpIPredicate Type) {
  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Setting compare type for dimIndex " << dimIndex << "\n";
    llvm::dbgs() << "Type: " << Type << "\n";
    llvm::dbgs() << "----------------------------------------------\n";
  });

  switch (Type) {
  case arith::CmpIPredicate::slt:
    this->currentType = dimInfo::CompareType::slt;
    break;
  case arith::CmpIPredicate::ult:
    this->currentType = dimInfo::CompareType::ult;
    break;
  case arith::CmpIPredicate::sge:
    this->currentType = dimInfo::CompareType::sge;
    break;
  case arith::CmpIPredicate::uge:
    this->currentType = dimInfo::CompareType::uge;
    break;
  default:
    return false;
  }
  return true;
}

bool dimInfo::compareTypeIsLess() const {
  return this->currentType == dimInfo::CompareType::slt ||
         this->currentType == dimInfo::CompareType::ult;
}

void dimInfo::dump() const {
  llvm::dbgs() << "----------------------------------------------\n";
  llvm::dbgs() << "MaskDimInfo: \n";
  llvm::dbgs() << "offset = " << offset << "\n";
  llvm::dbgs() << "shape = " << shape << "\n";
  llvm::dbgs() << "rhs = " << rhs << "\n";
  llvm::dbgs() << "isLessMode = " << compareTypeIsLess() << "\n";
  llvm::dbgs() << "hasBroadCast = " << hasBroadCast << "\n";
  llvm::dbgs() << "----------------------------------------------\n";
};

void MaskState::dump() const {
  llvm::dbgs() << "----------------------------------------------\n";
  llvm::dbgs() << "MaskState :\n";
  llvm::dbgs() << "scalar = " << scalar << "\n";
  llvm::dbgs() << "stateInfo.size = " << stateInfo.size() << "\n";
  for (auto info : stateInfo)
    info.dump();
  llvm::dbgs() << "----------------------------------------------\n";
};

LogicalResult MaskState::parse(Value operand, const Location loc,
                               OpBuilder &builder) {
  if (isa<IntegerType>(operand.getType()) &&
      operand.getType().getIntOrFloatBitWidth() != 1) {
    return this->parseIntScalar(operand, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
    return this->parseConstant(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AddIOp>()) {
    return this->parseAdd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::SubIOp>()) {
    return this->parseSub(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::AndIOp>()) {
    return this->parseAnd(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::CmpIOp>()) {
    return this->parseCmp(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::MakeRangeOp>()) {
    return this->parseMakeRange(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return this->parseBroadcast(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return this->parseSplat(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<triton::ExpandDimsOp>()) {
    return this->parseExpandDims(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::ExtSIOp>()) {
    return this->parseExtSI(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::RemSIOp>()) {
    return this->parseRem(op, loc, builder);
  } else if (auto op = operand.getDefiningOp<arith::DivSIOp>()) {
    return this->parseDiv(op, loc, builder);
  }
  LLVM_DEBUG({
    InFlightDiagnostic diag = emitWarning(loc)
                              << "MaskAnalysis: compare operand produced by an "
                                 "unsupported operation\n";
  });
  return failure();
}

LogicalResult MaskState::parseConstant(arith::ConstantOp constOp,
                                       const Location loc, OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      constOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting constant");
    });
    return failure();
  }

  if (auto intType = dyn_cast<IntegerType>(constOp.getType())) {
    if (intType.getWidth() == 1) {
      LLVM_DEBUG({
        constOp.emitWarning("MaskAnalysis: Unsupported constant for int1");
      });
      return failure();
    }
  }

  if (isa<DenseElementsAttr>(constOp.getValue())) {
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    auto elementType = attr.getElementType();
    if (!attr.isSplat() || !isa<IntegerType>(elementType)) {
      LLVM_DEBUG({
        constOp.emitError("MaskAnalysis: only support splat integer constant");
      });
      return failure();
    }
    auto values = attr.getValues<IntegerAttr>();
    auto value = values[0].getValue();
    auto constAttr = builder.getIndexAttr(value.getSExtValue());
    auto op = arith::ConstantOp::materialize(builder, constAttr,
                                             builder.getIndexType(), loc);
    this->scalar = op.getValue();
  } else {
    auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
    this->scalar = builder.getIndexAttr(value);
  }
  return success();
}

LogicalResult MaskState::parseIntScalar(Value scalar, const Location loc,
                                        OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitError(loc) << "MaskAnalysis: MaskState should be empty when "
                            "visiting integer scalar";
    });
    return failure();
  }
  auto castOp =
      builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), scalar);
  this->scalar = castOp.getResult();
  return success();
}

LogicalResult MaskState::parseMakeRange(triton::MakeRangeOp rangeOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      rangeOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting make_range");
    });
    return failure();
  }

  auto shape = cast<ShapedType>(rangeOp.getType()).getShape();
  auto start = rangeOp.getStart();
  auto end = rangeOp.getEnd();
  auto stride = (end - start + shape[0] - 1) / shape[0];

  if (stride != 1) {
    LLVM_DEBUG(
        {
          InFlightDiagnostic diag =
              emitError(loc)
              << "stride must be 1 for make_range whose result is used "
                 "as load or store masks";
        });
    return failure();
  }

  stateInfo.emplace_back(builder.getIndexAttr(start),
                         builder.getIndexAttr(shape[0]));
  return success();
}

LogicalResult MaskState::parseExtSI(arith::ExtSIOp op, const Location loc,
                                    OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      op->emitError(
          "MaskAnalysis: MaskState should be empty when visiting extsi");
    });
    return failure();
  }
  return parse(op.getIn(), loc, builder);
}

LogicalResult MaskState::parseSplat(triton::SplatOp splatOp, const Location loc,
                                    OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      splatOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting splat");
    });
    return failure();
  }

  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (!isa<IntegerType>(src.getType())) {
    LLVM_DEBUG(
        {
          splatOp.emitError()
              << "splat source must be an integer scalar for load/store masks";
        });
    return failure();
  }

  if (failed(this->parse(src, loc, builder)))
    return failure();

  if (stateInfo.size() > 1 ||
      (stateInfo.size() == 1 && !isOne(stateInfo.back().shape))) {
    LLVM_DEBUG({
      splatOp.emitError() << "splat from a non-scalar source is not supported, "
                             "unless it's state size and shape are 1";
    });
    return failure();
  }

  auto zeroAttr = builder.getIndexAttr(0);
  SmallVector<dimInfo> newStateInfo;
  for (auto [i, shape] : llvm::enumerate(dstShape)) {
    auto shapeAttr = builder.getIndexAttr(shape);
    newStateInfo.emplace_back(zeroAttr, shapeAttr, i, true);
    if (!stateInfo.empty() && shape == 1) {
      newStateInfo.back() = stateInfo.back();
      newStateInfo.back().dimIndex = i;
    }
  }
  this->stateInfo = newStateInfo;
  return success();
}

LogicalResult MaskState::parseExpandDims(triton::ExpandDimsOp expandDimsOp,
                                         const Location loc,
                                         OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      expandDimsOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting expand_dims");
    });
    return failure();
  }

  auto zeroAttr = builder.getIndexAttr(0);
  auto defaultShape = builder.getIndexAttr(1);

  if (failed(this->parse(expandDimsOp.getSrc(), loc, builder)))
    return failure();

  auto dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();
  if (dstShape[axis] != 1) {
    LLVM_DEBUG({
      expandDimsOp.emitError(
          "MaskAnalysis: unexpected dimension size in expand_dims");
    });
    return failure();
  }

  size_t insertPos = 0;
  for (auto &info : stateInfo) {
    if (info.dimIndex >= axis)
      ++info.dimIndex;
    if (info.dimIndex < axis)
      ++insertPos;
  }

  dimInfo insertInfo(zeroAttr, defaultShape, axis, true);
  stateInfo.insert(stateInfo.begin() + insertPos, insertInfo);

  return success();
}

LogicalResult MaskState::parseAdd(arith::AddIOp addOp, const Location loc,
                                  OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      addOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting add");
    });
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(addOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(addOp.getRhs(), loc, builder)))
    return failure();

  return this->addStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::parseSub(arith::SubIOp subOp, const Location loc,
                                  OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      subOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting sub");
    });
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(subOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(subOp.getRhs(), loc, builder)))
    return failure();

  return this->subStates(lhsState, rhsState, loc, builder);
}

LogicalResult MaskState::addStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  if (lhsState.scalar && rhsState.scalar) {
    LLVM_DEBUG(
        {
          InFlightDiagnostic diag =
              emitWarning(loc)
              << "Unexpected case where both lhs and rhs are scalars";
        });
    return failure();
  }

  if (!lhsState.scalar && !rhsState.scalar) {
    LLVM_DEBUG(
        {
          InFlightDiagnostic diag =
              emitWarning(loc)
              << "Unsupported scenario where neither lhs nor rhs is a scalar";
        });
    return failure();
  }

  if (lhsState.scalar)
    return addStateScalar(rhsState, lhsState.scalar, loc, builder);
  else
    return addStateScalar(lhsState, rhsState.scalar, loc, builder);
}

LogicalResult MaskState::addStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  for (auto info : state.stateInfo) {
    info.offset = addOpFoldResult(info.offset, scalar, loc, builder,
                                  builder.getIndexType());
    this->stateInfo.emplace_back(info);
  }
  return success();
}

LogicalResult MaskState::subStates(const MaskState &lhsState,
                                   const MaskState &rhsState, Location loc,
                                   OpBuilder &builder) {
  // Only (range - scalar) keeps a contiguous increasing index range.
  if (!lhsState.scalar && rhsState.scalar)
    return subStateScalar(lhsState, rhsState.scalar, loc, builder);

  LLVM_DEBUG(
      {
        InFlightDiagnostic diag =
            emitWarning(loc)
            << "Unsupported subi for continuous mask: only (range - scalar) is "
               "supported";
      });
  return failure();
}

LogicalResult MaskState::subStateScalar(const MaskState &state,
                                        const OpFoldResult scalar, Location loc,
                                        OpBuilder &builder) {
  for (auto info : state.stateInfo) {
    info.offset = subOpFoldResult(info.offset, scalar, loc, builder);
    if (!info.offset)
      return failure();
    this->stateInfo.emplace_back(info);
  }
  return success();
}

LogicalResult MaskState::parseBroadcast(triton::BroadcastOp broadcastOp,
                                        const Location loc,
                                        OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      broadcastOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting broadcast");
    });
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Parsing BROADCAST operation: " << broadcastOp << "\n";
  });

  auto src = broadcastOp.getSrc();
  auto dst = broadcastOp.getResult();
  if (!isa<ShapedType>(dst.getType())) {
    LLVM_DEBUG({
      broadcastOp.emitError(
          "MaskAnalysis: broadcast dst should be a shaped type");
    });
    return failure();
  }

  auto srcShape = cast<ShapedType>(src.getType()).getShape();
  auto dstShape = cast<ShapedType>(dst.getType()).getShape();
  if (srcShape.size() != dstShape.size()) {
    LLVM_DEBUG({
      broadcastOp.emitError(
          "MaskAnalysis: broadcast src and dst should have the same rank");
    });
    return failure();
  }

  if (failed(parse(src, loc, builder)))
    return failure();

  LLVM_DEBUG({
    llvm::dbgs() << "Before BROADCAST MaskState: \n";
    this->dump();
  });

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == dstShape[i])
      continue;
    else if (srcShape[i] < dstShape[i] && srcShape[i] == 1) {
      for (auto &info : stateInfo) {
        if (info.dimIndex != i)
          continue;
        info.shape = builder.getIndexAttr(dstShape[i]);
        info.hasBroadCast = true;
      }
    } else {
      LLVM_DEBUG({
        broadcastOp.emitError(
            "MaskAnalysis: unexpected dimensions used in broadcast");
      });
      return failure();
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After BROADCAST MaskState: \n";
    this->dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  return success();
}

LogicalResult MaskState::parseCmp(arith::CmpIOp cmpOp, const Location loc,
                                  OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      cmpOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting cmpi");
    });
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Parsing CMP operation: " << cmpOp << "\n";
  });

  if (isa<IntegerType>(cmpOp.getLhs().getType()) &&
      (cmpOp.getLhs().getDefiningOp<arith::CmpIOp>() ||
       cmpOp.getRhs().getDefiningOp<arith::CmpIOp>())) {
    LLVM_DEBUG({
      cmpOp.emitWarning(
          "MaskAnalysis: Unsupported nested cmpi scenario for int1");
    });
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(cmpOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(cmpOp.getRhs(), loc, builder)))
    return failure();

  if (lhsState.scalar) {
    if (lhsState.stateInfo.empty()) {
      lhsState.stateInfo.emplace_back(builder.getIndexAttr(0),
                                      builder.getIndexAttr(1));
    }
    for (auto &info : lhsState.stateInfo) {
      if (isOne(info.shape)) {
        info.offset = lhsState.scalar;
      }
    }
  }

  // lhs must be a Value and rhs must be scalar
  if (!rhsState.scalar) {
    LLVM_DEBUG(
        { cmpOp.emitWarning("MaskAnalysis: Unsupported cmpi scenario"); });
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "LHS MaskState: \n";
    lhsState.dump();
    llvm::dbgs() << "RHS MaskState: \n";
    rhsState.dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  // In the case where the values we are loading are entirely masked off like
  // the following:
  //
  // ---|-------|-----------|
  //    ^       ^           ^
  //   scalar  start       end
  //
  // newEnd = min(end, scalar) = scalar
  // Now scalar < start, so simply doing dim = newEnd - start is incorrect.
  //
  // The correct formula is to optionally move `newDim` back to `start` using
  // max(newEnd, start).
  auto cmpType = cmpOp.getPredicate();
  for (auto &info : lhsState.stateInfo) {
    if (info.hasBroadCast)
      continue;
    if (!info.setType(cmpType)) {
      LLVM_DEBUG({ cmpOp.emitWarning("MaskAnalysis: Unsupported cmpi type"); });
      return failure();
    }
    info.rhs = rhsState.scalar;
  }
  this->stateInfo = lhsState.stateInfo;

  LLVM_DEBUG({
    llvm::dbgs() << "After CMP MaskState: \n";
    this->dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });
  return success();
}

LogicalResult MaskState::parseRem(arith::RemSIOp remOp, const Location loc,
                                  OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      remOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting REMSI");
    });
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(remOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(remOp.getRhs(), loc, builder)))
    return failure();

  if (lhsState.scalar || !rhsState.scalar) {
    LLVM_DEBUG(
        { remOp.emitRemark("MaskAnalysis: Unsupported REMSI scenario"); });
    return failure();
  }

  auto divisorAttr = rhsState.scalar;

  if (!getIntAttr(divisorAttr).has_value()) {
    LLVM_DEBUG({
      remOp.emitError("MaskAnalysis: do not support dynamic divisor in REMSI.");
    });
    return failure();
  }

  SmallVector<dimInfo> newStateInfo;
  auto zeroAttr = builder.getIndexAttr(0);
  for (auto info : lhsState.stateInfo) {
    if (info.hasBroadCast) {
      newStateInfo.emplace_back(info);
      continue;
    }
    if (!isMultiple(divisorAttr, info.shape) &&
        !isMultiple(info.shape, divisorAttr)) {
      LLVM_DEBUG({
        remOp.emitError(
            "MaskAnalysis: do not support dynamic stride before REMSI.");
      });
      return failure();
    }

    auto contiguousSize =
        minOpFoldResult(divisorAttr, info.shape, loc, builder);
    auto nonContiguousSize =
        divOpFoldResult(info.shape, contiguousSize, loc, builder);

    auto staticNonContiguousSize = getIntAttr(nonContiguousSize);
    if (!staticNonContiguousSize.has_value()) {
      LLVM_DEBUG({
        remOp.emitError(
            "MaskAnalysis: do not support dynamic size before REMSI.");
      });
      return failure();
    }

    if (staticNonContiguousSize.value() != 0)
      newStateInfo.emplace_back(zeroAttr, nonContiguousSize, info.dimIndex,
                                true);

    auto newOffset = remOpFoldResult(info.offset, divisorAttr, loc, builder);
    newStateInfo.emplace_back(newOffset, nonContiguousSize, info.dimIndex);
  }

  this->stateInfo = newStateInfo;

  return success();
}

LogicalResult MaskState::parseDiv(arith::DivSIOp divOp, const Location loc,
                                  OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      divOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting DIVSI");
    });
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Parsing DIV operation: " << divOp << "\n";
  });

  MaskState lhsState;
  if (failed(lhsState.parse(divOp.getLhs(), loc, builder)))
    return failure();

  MaskState rhsState;
  if (failed(rhsState.parse(divOp.getRhs(), loc, builder)))
    return failure();

  LLVM_DEBUG({
    llvm::dbgs() << "LHS MaskState: \n";
    lhsState.dump();
    llvm::dbgs() << "RHS MaskState: \n";
    rhsState.dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  if (lhsState.scalar || !rhsState.scalar) {
    LLVM_DEBUG(
        { divOp.emitRemark("MaskAnalysis: Unsupported DIVSI scenario"); });
    return failure();
  }

  auto divisorAttr = rhsState.scalar;

  if (!getIntAttr(divisorAttr).has_value()) {
    LLVM_DEBUG({
      divOp.emitError("MaskAnalysis: do not support dynamix divisor in DIVSI.");
    });
    return failure();
  }

  SmallVector<dimInfo> newStateInfo;
  auto zeroAttr = builder.getIndexAttr(0);
  for (auto info : lhsState.stateInfo) {
    if (info.hasBroadCast) {
      newStateInfo.emplace_back(info);
      continue;
    }
    if (!isMultiple(divisorAttr, info.shape) &&
        !isMultiple(info.shape, divisorAttr)) {
      LLVM_DEBUG({
        divOp.emitError(
            "MaskAnalysis: do not support dynamix stride before DIVSI.");
      });
      return failure();
    }

    auto nonContiguousSize =
        minOpFoldResult(divisorAttr, info.shape, loc, builder);
    auto contiguousSize =
        divOpFoldResult(info.shape, nonContiguousSize, loc, builder);

    auto staticContiguousSize = getIntAttr(contiguousSize);
    if (!staticContiguousSize.has_value()) {
      LLVM_DEBUG({
        divOp.emitError(
            "MaskAnalysis: do not support dynamix size before DIVSI.");
      });
      return failure();
    }

    if (staticContiguousSize.value() != 0) {
      auto newOffset = divOpFoldResult(info.offset, divisorAttr, loc, builder);
      newStateInfo.emplace_back(newOffset, contiguousSize, info.dimIndex);
    }

    newStateInfo.emplace_back(zeroAttr, nonContiguousSize, info.dimIndex, true);
  }

  this->stateInfo = newStateInfo;

  LLVM_DEBUG({
    llvm::dbgs() << "After DIV MaskState: \n";
    this->dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });
  return success();
}

LogicalResult MaskState::parseAnd(arith::AndIOp andOp, const Location loc,
                                  OpBuilder &builder) {
  if (!this->isEmpty()) {
    LLVM_DEBUG({
      andOp.emitError(
          "MaskAnalysis: MaskState should be empty when visiting and");
    });
    return failure();
  }
  auto zeroAttr = builder.getIndexAttr(0);

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Parsing AND operation: " << andOp << "\n";
  });

  if (isa<IntegerType>(andOp.getLhs().getType())) {
    LLVM_DEBUG({
      andOp.emitWarning("MaskAnalysis: Unsupported andi scenario for int1");
    });
    return failure();
  }

  MaskState lhsState;
  if (failed(lhsState.parse(andOp.getLhs(), loc, builder)))
    return failure();
  MaskState rhsState;
  if (failed(rhsState.parse(andOp.getRhs(), loc, builder)))
    return failure();

  LLVM_DEBUG({
    llvm::dbgs() << "LHS MaskState: \n";
    lhsState.dump();
    llvm::dbgs() << "RHS MaskState: \n";
    rhsState.dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });

  SmallVector<dimInfo> newStateInfo;
  auto lIt = lhsState.stateInfo.begin();
  auto rIt = rhsState.stateInfo.begin();

  while (lIt != lhsState.stateInfo.end() && rIt != rhsState.stateInfo.end()) {
    if (lIt->dimIndex != rIt->dimIndex) {
      auto newInfo = lIt->dimIndex < rIt->dimIndex ? *lIt++ : *rIt++;
      newStateInfo.emplace_back(newInfo);
      continue;
    }

    if (!isMultiple(lIt->shape, rIt->shape) &&
        !isMultiple(rIt->shape, lIt->shape)) {
      LLVM_DEBUG({
        llvm::dbgs() << "LHS MaskState: \n";
        lhsState.dump();
        llvm::dbgs() << "RHS MaskState: \n";
        rhsState.dump();
        llvm::dbgs() << "----------------------------------------------\n";
      });
      LLVM_DEBUG({
        andOp.emitError(
            "MaskAnalysis: the add operation have incompatible sizes");
      });
      return failure();
    }

    dimInfo newInfo;
    newInfo.dimIndex = lIt->dimIndex;
    newInfo.shape = minOpFoldResult(lIt->shape, rIt->shape, loc, builder);
    if ((isLess(newInfo.shape, lIt->shape) && !lIt->hasBroadCast ||
         isLess(newInfo.shape, rIt->shape) && !rIt->hasBroadCast)) {
      LLVM_DEBUG({
        llvm::dbgs() << "LHS MaskState: \n";
        lhsState.dump();
        llvm::dbgs() << "RHS MaskState: \n";
        rhsState.dump();
        llvm::dbgs() << "----------------------------------------------\n";
      });
      LLVM_DEBUG({
        andOp.emitError(
            "MaskAnalysis: the add operation have incompatible sizes."
            "Valid dimensions are split.");
      });
      return failure();
    }
    newInfo.currentType =
        lIt->hasBroadCast ? rIt->currentType : lIt->currentType;
    if (lIt->currentType != dimInfo::CompareType::deafaultType &&
        rIt->currentType != dimInfo::CompareType::deafaultType &&
        lIt->currentType != rIt->currentType) {
      LLVM_DEBUG({
        andOp.emitError(
            "MaskAnalysis: do not suppport different compare mode within"
            "the same dimension.");
      });
      return failure();
    }

    if (lIt->hasBroadCast) {
      newInfo.offset = rIt->offset;
      newInfo.rhs = rIt->rhs;
      newInfo.hasBroadCast = rIt->hasBroadCast;
    } else if (rIt->hasBroadCast) {
      newInfo.offset = lIt->offset;
      newInfo.rhs = lIt->rhs;
      newInfo.hasBroadCast = lIt->hasBroadCast;
    } else {
      LLVM_DEBUG({
        andOp.emitError("MaskAnalysis: do not suppport "
                        "and in the same dimension.");
      });
      return failure();
    }

    newStateInfo.emplace_back(newInfo);

    if (isEqual(lIt->shape, newInfo.shape))
      ++lIt;
    else
      lIt->shape = divOpFoldResult(lIt->shape, newInfo.shape, loc, builder);
    if (isEqual(rIt->shape, newInfo.shape))
      ++rIt;
    else
      rIt->shape = divOpFoldResult(rIt->shape, newInfo.shape, loc, builder);
  }

  while (rIt != rhsState.stateInfo.end()) {
    newStateInfo.push_back(*rIt++);
  }
  while (lIt != lhsState.stateInfo.end()) {
    newStateInfo.push_back(*lIt++);
  }

  this->stateInfo = newStateInfo;

  LLVM_DEBUG({
    llvm::dbgs() << "After AND MaskState: \n";
    this->dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });
  return success();
}

LogicalResult MaskState::analysisMask(Value operand) {
  auto op = operand.getDefiningOp();
  if (!op) {
    return failure();
  }
  auto loc = op->getLoc();
  OpBuilder builder(op);

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "Analyzing mask: " << operand << "\n";
  });

  if (this->parse(operand, loc, builder).failed() || this->isEmpty()) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Mask analysis result: \n";
    this->dump();
    llvm::dbgs() << "MaskAnalysis: successfully analyzed mask.\n";
    llvm::dbgs() << "----------------------------------------------\n";
  });
  return success();
}

Value MaskState::createNewMask(const Location loc, OpBuilder &builder) {
  if (isEmpty())
    return nullptr;

  SmallVector<int64_t> shape;
  for (auto info : stateInfo) {
    auto staticShape = getIntAttr(info.shape);
    if (!staticShape.has_value()) {
      LLVM_DEBUG(
          {
            InFlightDiagnostic diag =
                emitError(loc)
                << "MaskAnalysis: dynamic shape is not supported in mask "
                   "generation\n";
          });
      return nullptr;
    }
    shape.emplace_back(staticShape.value());
  }
  SmallVector<Value> cacheResults;
  auto maskShape = RankedTensorType::get(shape, builder.getI1Type());

  auto createRhsValue = [&](OpFoldResult rhs) -> Value {
    if (auto rhsInt = getIntAttr(rhs)) {
      auto rhsAttr =
          builder.getI32IntegerAttr(static_cast<int32_t>(rhsInt.value()));
      return builder.create<arith::ConstantOp>(loc, rhsAttr).getResult();
    }
    Value rhsValue = dyn_cast<Value>(rhs);
    if (rhsValue.getType().isIndex()) {
      rhsValue = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                                    rhsValue);
    }
    return rhsValue;
  };
  for (size_t i = 0; i < stateInfo.size(); ++i) {
    auto info = stateInfo[i];
    if (info.hasBroadCast) {
      continue;
    }
    auto indexI32RowType =
        RankedTensorType::get(shape[i], builder.getI32Type());
    Value newMask =
        builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0, shape[i]);

    Value newOffset = createRhsValue(info.offset);
    if (newOffset.getType().isIndex()) {
      newOffset = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                                     newOffset);
    }
    Value splatRhs =
        builder.create<triton::SplatOp>(loc, indexI32RowType, newOffset);
    newMask = builder.create<arith::AddIOp>(loc, newMask, splatRhs);

    auto rhsValue = createRhsValue(info.rhs);
    auto splatOp =
        builder.create<triton::SplatOp>(loc, indexI32RowType, rhsValue);

    if (info.currentType == dimInfo::CompareType::deafaultType) {
      LLVM_DEBUG({
        InFlightDiagnostic diag = emitError(loc)
                                  << "MaskAnalysis: cannot generate mask when "
                                     "compare type is not set\n";
      });
      return nullptr;
    }
    auto cmpOp = builder.create<arith::CmpIOp>(loc,
                                               info.compareTypeIsLess()
                                                   ? arith::CmpIPredicate::slt
                                                   : arith::CmpIPredicate::sge,
                                               newMask, splatOp.getResult());

    auto expandValue = cmpOp.getResult();
    for (size_t j = 0; j < stateInfo.size(); ++j) {
      if (j == i)
        continue;
      expandValue = builder.create<triton::ExpandDimsOp>(loc, expandValue, j);
    }

    auto broadcastValue =
        builder.create<triton::BroadcastOp>(loc, maskShape, expandValue);

    cacheResults.push_back(broadcastValue);
  }

  if (cacheResults.empty()) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitWarning(loc) << "MaskAnalysis: cannot generate mask when all "
                              "dimensions are broadcasted";
    });
    return nullptr;
  }
  newMask = cacheResults[0];
  for (size_t i = 1; i < cacheResults.size(); ++i) {
    newMask = builder.create<arith::AndIOp>(loc, newMask, cacheResults[i]);
  }
  return newMask;
}

} // namespace TritonToStructured
