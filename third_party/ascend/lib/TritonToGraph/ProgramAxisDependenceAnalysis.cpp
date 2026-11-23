/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
 */

#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cfg;

namespace {

struct OffsetForm {
  bool valid = true;
  bool symbolicStride = false;
  int64_t programCoefficient = 0;
  int64_t laneMin = 0;
  int64_t laneMax = 0;
};

bool addNoOverflow(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

bool multiplyNoOverflow(int64_t lhs, int64_t rhs, int64_t &result) {
  return !__builtin_mul_overflow(lhs, rhs, &result);
}

std::optional<int64_t> getConstantInt(Value value) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

std::optional<int64_t> getUniformConstantInt(Value value) {
  if (std::optional<int64_t> constant = getConstantInt(value))
    return constant;
  DenseElementsAttr elements;
  if (!matchPattern(value, m_Constant(&elements)) || !elements.isSplat() ||
      !isa<IntegerType>(elements.getElementType()))
    return std::nullopt;
  return elements.getSplatValue<APInt>().getSExtValue();
}

OffsetForm invalidOffsetForm() {
  OffsetForm form;
  form.valid = false;
  return form;
}

OffsetForm addOffsetForms(const OffsetForm &lhs, const OffsetForm &rhs,
                          bool subtractRhs) {
  if (!lhs.valid || !rhs.valid)
    return invalidOffsetForm();
  if (subtractRhs && rhs.symbolicStride)
    return invalidOffsetForm();
  if ((lhs.symbolicStride &&
       (rhs.symbolicStride || rhs.programCoefficient != 0)) ||
      (rhs.symbolicStride && lhs.programCoefficient != 0))
    return invalidOffsetForm();
  const int64_t rhsCoefficient =
      subtractRhs ? -rhs.programCoefficient : rhs.programCoefficient;
  const int64_t rhsMin = subtractRhs ? -rhs.laneMax : rhs.laneMin;
  const int64_t rhsMax = subtractRhs ? -rhs.laneMin : rhs.laneMax;
  OffsetForm result;
  result.symbolicStride = lhs.symbolicStride || rhs.symbolicStride;
  if (!addNoOverflow(lhs.programCoefficient, rhsCoefficient,
                     result.programCoefficient) ||
      !addNoOverflow(lhs.laneMin, rhsMin, result.laneMin) ||
      !addNoOverflow(lhs.laneMax, rhsMax, result.laneMax))
    return invalidOffsetForm();
  return result;
}

OffsetForm scaleOffsetForm(const OffsetForm &input, int64_t factor) {
  if (!input.valid)
    return invalidOffsetForm();
  if (input.symbolicStride && factor != 1)
    return invalidOffsetForm();
  OffsetForm result;
  result.symbolicStride = input.symbolicStride;
  if (!multiplyNoOverflow(input.programCoefficient, factor,
                          result.programCoefficient) ||
      !multiplyNoOverflow(input.laneMin, factor, result.laneMin) ||
      !multiplyNoOverflow(input.laneMax, factor, result.laneMax))
    return invalidOffsetForm();
  if (factor < 0)
    std::swap(result.laneMin, result.laneMax);
  return result;
}

bool valueDependsOnAxis(Value value, int32_t targetAxis, DenseSet<Value> &seen);

bool isRuntimeIntegerEntryArgument(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto integer = dyn_cast<IntegerType>(argument.getType());
    return integer && argument.getOwner() &&
           isa<triton::FuncOp>(argument.getOwner()->getParentOp());
  }
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return isRuntimeIntegerEntryArgument(splat.getSrc());
  if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>())
    return isRuntimeIntegerEntryArgument(broadcast.getSrc());
  if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>())
    return isRuntimeIntegerEntryArgument(expand.getSrc());
  return false;
}

OffsetForm analyzeOffset(Value value, int32_t targetAxis,
                         DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return invalidOffsetForm();

  auto finish = [&](OffsetForm form) {
    visited.erase(value);
    return form;
  };

  if (auto pid = value.getDefiningOp<triton::GetProgramIdOp>()) {
    if (pid.getAxisAsInt() != targetAxis)
      return finish(invalidOffsetForm());
    OffsetForm form;
    form.programCoefficient = 1;
    return finish(form);
  }
  if (value.getDefiningOp<triton::GetNumProgramsOp>())
    return finish(invalidOffsetForm());
  if (std::optional<int64_t> constant = getConstantInt(value)) {
    OffsetForm form;
    form.laneMin = *constant;
    form.laneMax = *constant;
    return finish(form);
  }
  if (auto range = value.getDefiningOp<triton::MakeRangeOp>()) {
    if (range.getEnd() <= range.getStart())
      return finish(invalidOffsetForm());
    OffsetForm form;
    form.laneMin = range.getStart();
    form.laneMax = range.getEnd() - 1;
    return finish(form);
  }
  DenseSet<Value> dependenceSeen;
  if (!valueDependsOnAxis(value, targetAxis, dependenceSeen))
    return finish(OffsetForm{});
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return finish(analyzeOffset(splat.getSrc(), targetAxis, visited));
  if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>())
    return finish(analyzeOffset(broadcast.getSrc(), targetAxis, visited));
  if (auto expand = value.getDefiningOp<triton::ExpandDimsOp>())
    return finish(analyzeOffset(expand.getSrc(), targetAxis, visited));
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    OffsetForm lhs = analyzeOffset(add.getLhs(), targetAxis, visited);
    OffsetForm rhs = analyzeOffset(add.getRhs(), targetAxis, visited);
    return finish(addOffsetForms(lhs, rhs, false));
  }
  if (auto subtract = value.getDefiningOp<arith::SubIOp>()) {
    OffsetForm lhs = analyzeOffset(subtract.getLhs(), targetAxis, visited);
    OffsetForm rhs = analyzeOffset(subtract.getRhs(), targetAxis, visited);
    return finish(addOffsetForms(lhs, rhs, true));
  }
  if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
    if (std::optional<int64_t> lhs = getConstantInt(multiply.getLhs()))
      return finish(scaleOffsetForm(
          analyzeOffset(multiply.getRhs(), targetAxis, visited), *lhs));
    if (std::optional<int64_t> rhs = getConstantInt(multiply.getRhs()))
      return finish(scaleOffsetForm(
          analyzeOffset(multiply.getLhs(), targetAxis, visited), *rhs));
    DenseSet<Value> lhsSeen;
    DenseSet<Value> rhsSeen;
    const bool lhsDepends =
        valueDependsOnAxis(multiply.getLhs(), targetAxis, lhsSeen);
    const bool rhsDepends =
        valueDependsOnAxis(multiply.getRhs(), targetAxis, rhsSeen);
    if (lhsDepends == rhsDepends)
      return finish(invalidOffsetForm());
    Value pidTerm = lhsDepends ? multiply.getLhs() : multiply.getRhs();
    Value strideTerm = lhsDepends ? multiply.getRhs() : multiply.getLhs();
    DenseSet<Value> pidSeen;
    OffsetForm pidForm = analyzeOffset(pidTerm, targetAxis, pidSeen);
    if (!pidForm.valid || pidForm.symbolicStride ||
        pidForm.programCoefficient != 1 || pidForm.laneMin != 0 ||
        pidForm.laneMax != 0 || !isRuntimeIntegerEntryArgument(strideTerm))
      return finish(invalidOffsetForm());
    OffsetForm form;
    form.symbolicStride = true;
    return finish(form);
  }
  return finish(invalidOffsetForm());
}

bool valueDependsOnAxis(Value value, int32_t targetAxis,
                        DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return false;
  if (auto pid = value.getDefiningOp<triton::GetProgramIdOp>())
    return pid.getAxisAsInt() == targetAxis;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return valueDependsOnAxis(operand, targetAxis, seen);
  });
}

OffsetForm analyzePointerOffset(Value pointer, int32_t targetAxis,
                                DenseSet<Value> &visited) {
  if (!visited.insert(pointer).second)
    return invalidOffsetForm();
  auto finish = [&](OffsetForm form) {
    visited.erase(pointer);
    return form;
  };

  if (auto addPtr = pointer.getDefiningOp<triton::AddPtrOp>()) {
    OffsetForm base =
        analyzePointerOffset(addPtr.getPtr(), targetAxis, visited);
    DenseSet<Value> offsetVisited;
    OffsetForm offset =
        analyzeOffset(addPtr.getOffset(), targetAxis, offsetVisited);
    return finish(addOffsetForms(base, offset, false));
  }
  if (auto splat = pointer.getDefiningOp<triton::SplatOp>())
    return finish(analyzePointerOffset(splat.getSrc(), targetAxis, visited));
  if (auto broadcast = pointer.getDefiningOp<triton::BroadcastOp>())
    return finish(
        analyzePointerOffset(broadcast.getSrc(), targetAxis, visited));
  if (auto expand = pointer.getDefiningOp<triton::ExpandDimsOp>())
    return finish(analyzePointerOffset(expand.getSrc(), targetAxis, visited));

  DenseSet<Value> dependenceSeen;
  return finish(valueDependsOnAxis(pointer, targetAxis, dependenceSeen)
                    ? invalidOffsetForm()
                    : OffsetForm{});
}

std::optional<int64_t>
getConjunctiveOffsetUpperBound(Value mask, Value offset,
                               DenseSet<Value> &visited) {
  if (!mask || !visited.insert(mask).second)
    return std::nullopt;
  if (auto compare = mask.getDefiningOp<arith::CmpIOp>()) {
    if (compare.getPredicate() == arith::CmpIPredicate::slt &&
        compare.getLhs() == offset) {
      std::optional<int64_t> bound = getUniformConstantInt(compare.getRhs());
      if (bound && *bound > 0)
        return bound;
    }
    return std::nullopt;
  }
  if (auto conjunction = mask.getDefiningOp<arith::AndIOp>()) {
    std::optional<int64_t> lhs =
        getConjunctiveOffsetUpperBound(conjunction.getLhs(), offset, visited);
    std::optional<int64_t> rhs =
        getConjunctiveOffsetUpperBound(conjunction.getRhs(), offset, visited);
    if (lhs && rhs)
      return std::min(*lhs, *rhs);
    return lhs ? lhs : rhs;
  }
  return std::nullopt;
}

Value peelPointerShapeOps(Value pointer) {
  while (true) {
    if (auto splat = pointer.getDefiningOp<triton::SplatOp>()) {
      pointer = splat.getSrc();
      continue;
    }
    if (auto broadcast = pointer.getDefiningOp<triton::BroadcastOp>()) {
      pointer = broadcast.getSrc();
      continue;
    }
    if (auto expand = pointer.getDefiningOp<triton::ExpandDimsOp>()) {
      pointer = expand.getSrc();
      continue;
    }
    return pointer;
  }
}

OffsetForm analyzeStorePointerOffset(triton::StoreOp store,
                                     int32_t targetAxis) {
  Value pointer = peelPointerShapeOps(store.getPtr());
  auto finalAddPtr = pointer.getDefiningOp<triton::AddPtrOp>();
  if (!finalAddPtr) {
    DenseSet<Value> pointerSeen;
    return analyzePointerOffset(store.getPtr(), targetAxis, pointerSeen);
  }

  DenseSet<Value> baseSeen;
  OffsetForm base =
      analyzePointerOffset(finalAddPtr.getPtr(), targetAxis, baseSeen);
  DenseSet<Value> offsetSeen;
  OffsetForm offset =
      analyzeOffset(finalAddPtr.getOffset(), targetAxis, offsetSeen);
  if (!base.valid || !offset.valid)
    return invalidOffsetForm();

  DenseSet<Value> maskSeen;
  std::optional<int64_t> exclusiveBound = getConjunctiveOffsetUpperBound(
      store.getMask(), finalAddPtr.getOffset(), maskSeen);
  if (exclusiveBound && *exclusiveBound > offset.laneMin) {
    const int64_t maskedMax = *exclusiveBound - 1;
    if (maskedMax < offset.laneMax)
      offset.laneMax = maskedMax;
  }
  return addOffsetForms(base, offset, false);
}

StoreAddressIndependence classifyStoreAddress(triton::StoreOp store,
                                              int32_t targetAxis,
                                              bool &pointerDependsOnAxis) {
  DenseSet<Value> dependencySeen;
  pointerDependsOnAxis =
      valueDependsOnAxis(store.getPtr(), targetAxis, dependencySeen);
  if (!pointerDependsOnAxis)
    return StoreAddressIndependence::NotProgramDependent;

  OffsetForm offset = analyzeStorePointerOffset(store, targetAxis);
  if (!offset.valid ||
      (!offset.symbolicStride && offset.programCoefficient == 0) ||
      offset.laneMax < offset.laneMin)
    return StoreAddressIndependence::Unknown;

  if (offset.symbolicStride)
    return StoreAddressIndependence::CanonicalDynamicStride;

  const uint64_t span = static_cast<uint64_t>(offset.laneMax) -
                        static_cast<uint64_t>(offset.laneMin);
  const uint64_t coefficient =
      offset.programCoefficient < 0
          ? static_cast<uint64_t>(-(offset.programCoefficient + 1)) + 1
          : static_cast<uint64_t>(offset.programCoefficient);
  return coefficient > span ? StoreAddressIndependence::ProvenDisjoint
                            : StoreAddressIndependence::Unknown;
}

bool isControlFlowEscape(Operation *operation) {
  if (isa<triton::ReturnOp, CallOpInterface>(operation))
    return true;
  StringRef name = operation->getName().getStringRef();
  return name == "cf.br" || name == "cf.cond_br" || name.ends_with(".yield");
}

bool hasUnsupportedEffect(Operation *operation) {
  if (isa<triton::StoreOp, triton::LoadOp>(operation))
    return false;
  if (isa<CallOpInterface>(operation))
    return true;
  StringRef name = operation->getName().getStringRef();
  if (name.contains("atomic") || name.contains("barrier") ||
      name.contains("print"))
    return true;
  auto effects = dyn_cast<MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effectInstances;
  effects.getEffects(effectInstances);
  return llvm::any_of(effectInstances, [](const auto &effect) {
    return isa<MemoryEffects::Write, MemoryEffects::Allocate,
               MemoryEffects::Free>(effect.getEffect());
  });
}

void recordOperationFacts(Operation *operation, ProgramAxisDependence &info,
                          DenseSet<Operation *> &seenStores) {
  if (auto reduce = dyn_cast<triton::ReduceOp>(operation))
    info.reductionAxes.push_back(reduce.getAxis());
  if (auto scan = dyn_cast<triton::ScanOp>(operation))
    info.reductionAxes.push_back(scan.getAxis());

  if (isControlFlowEscape(operation))
    info.escapes = true;
  if (auto store = dyn_cast<triton::StoreOp>(operation)) {
    info.hasSideEffects = true;
    if (seenStores.insert(operation).second) {
      bool pointerDependsOnAxis = false;
      StoreAddressIndependence independence =
          classifyStoreAddress(store, info.axis, pointerDependsOnAxis);
      info.stores.push_back({operation, pointerDependsOnAxis, independence});
    }
    return;
  }
  if (hasUnsupportedEffect(operation)) {
    info.hasSideEffects = true;
    info.hasUnsupportedSideEffects = true;
  }
}

void buildClosure(ProgramAxisDependence &info) {
  DenseSet<Value> visitedValues;
  DenseSet<Operation *> visitedOperations;
  DenseSet<Operation *> seenStores;
  SmallVector<Value> worklist(info.programIds.begin(), info.programIds.end());

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visitedValues.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *operation = use.getOwner();
      if (!visitedOperations.insert(operation).second)
        continue;
      info.dependenceClosure.push_back(operation);
      recordOperationFacts(operation, info, seenStores);
      for (Value result : operation->getResults())
        worklist.push_back(result);
    }
  }
}

} // namespace

bool ProgramAxisDependence::hasOnlyDisjointStoreAddresses() const {
  return !stores.empty() && llvm::all_of(stores, [](const auto &store) {
    return store.pointerDependsOnAxis &&
           store.independence == StoreAddressIndependence::ProvenDisjoint;
  });
}

bool ProgramAxisDependence::hasOnlyLiftableStoreAddressesForProgramMapping()
    const {
  return !stores.empty() && llvm::all_of(stores, [](const auto &store) {
    return store.pointerDependsOnAxis &&
           (store.independence == StoreAddressIndependence::ProvenDisjoint ||
            store.independence ==
                StoreAddressIndependence::CanonicalDynamicStride);
  });
}

bool ProgramAxisDependence::isIndependentAxisTransformCandidate() const {
  return hasProgramId() && !escapes && !hasUnsupportedSideEffects &&
         !readsNumPrograms && hasOnlyDisjointStoreAddresses();
}

bool ProgramAxisDependence::isProgramMappingTransformCandidate() const {
  return hasProgramId() && !escapes && !hasUnsupportedSideEffects &&
         !readsNumPrograms && hasOnlyLiftableStoreAddressesForProgramMapping();
}

ProgramAxisDependenceAnalysis::ProgramAxisDependenceAnalysis(
    triton::FuncOp function) {
  for (int32_t axis = 0; axis < 3; ++axis)
    axes[axis].axis = axis;

  function.walk([&](triton::GetProgramIdOp pid) {
    const int32_t axis = pid.getAxisAsInt();
    if (axis >= 0 && axis < static_cast<int32_t>(axes.size()))
      axes[axis].programIds.push_back(pid.getResult());
  });
  function.walk([&](triton::GetNumProgramsOp programs) {
    const int32_t axis = programs.getAxisAsInt();
    if (axis >= 0 && axis < static_cast<int32_t>(axes.size()))
      axes[axis].readsNumPrograms = true;
  });
  for (ProgramAxisDependence &axis : axes)
    buildClosure(axis);
}

const ProgramAxisDependence &
ProgramAxisDependenceAnalysis::get(int32_t axis) const {
  assert(axis >= 0 && axis < static_cast<int32_t>(axes.size()) &&
         "program axis must be x, y, or z");
  return axes[axis];
}
