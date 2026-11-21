//===- Utils.h - Common utility functions for AscendModel --------*- C++
//-*-===//
//
// This file defines common utility functions used across multiple passes.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_UTILS_H
#define ASCENDMODEL_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <optional>
#include <variant>

namespace mlir {
namespace ascend {
namespace utils {

//===----------------------------------------------------------------------===//
// Tensor/Type Utilities
//===----------------------------------------------------------------------===//

/// Get the number of elements from a tensor type.
/// Returns 1024 for dynamic dimensions as a default estimate.
inline int64_t getNumElements(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    int64_t count = 1;
    for (int64_t dim : tensorType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return 1024; // Default estimate for dynamic shapes
      count *= dim;
    }
    return count;
  }
  return 1;
}

/// Get the byte size of a tensor type.
/// Returns 0 if the type is not a ranked tensor or has dynamic dimensions.
inline int64_t getByteSize(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType)
    return 0;

  int64_t count = 1;
  for (int64_t dim : tensorType.getShape()) {
    if (dim == ShapedType::kDynamic)
      return 0;
    count *= dim;
  }

  Type elemType = tensorType.getElementType();
  int64_t elemBits = 32; // Default
  if (elemType.isF16() || elemType.isBF16())
    elemBits = 16;
  else if (elemType.isF32())
    elemBits = 32;
  else if (elemType.isF64())
    elemBits = 64;
  else if (auto intType = dyn_cast<IntegerType>(elemType))
    elemBits = intType.getWidth();

  return count * elemBits / 8;
}

/// Get the element type bit width.
inline int64_t getElementBitWidth(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type elemType = tensorType.getElementType();
    if (elemType.isF16() || elemType.isBF16())
      return 16;
    else if (elemType.isF32())
      return 32;
    else if (elemType.isF64())
      return 64;
    else if (auto intType = dyn_cast<IntegerType>(elemType))
      return intType.getWidth();
  }
  return 32; // Default
}

//===----------------------------------------------------------------------===//
// Constant Extraction Utilities
//===----------------------------------------------------------------------===//

/// Try to extract a constant integer value from a Value.
/// Uses MLIR's matchPattern and fallback to arith.constant.
inline std::optional<int64_t> getConstantIntValue(Value v) {
  // Try using MLIR's matchPattern for constant matching
  IntegerAttr intAttr;
  if (matchPattern(v, m_Constant(&intAttr))) {
    return intAttr.getInt();
  }

  // Fallback: manually check for arith.constant
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return attr.getInt();
    }
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Dynamic Loop Bound Analysis
//===----------------------------------------------------------------------===//

/// Represents a symbolic expression for loop bounds.
/// Can be a constant, a function argument, or a simple expression.
struct SymbolicBound {
  enum Kind {
    Constant,   // A compile-time constant
    Argument,   // A function argument (e.g., %arg2)
    ProgramId,  // A tt.get_program_id
    Expression, // A computed expression depending on arguments
    Unknown     // Cannot be determined
  };

  Kind kind = Unknown;
  int64_t constantValue = 0; // Valid if kind == Constant
  unsigned argIndex = 0;     // Valid if kind == Argument
  std::string programIdDim;  // Valid if kind == ProgramId ("x", "y", "z")
  SmallVector<unsigned> dependentArgs; // Arguments this expression depends on
  SmallVector<std::string> dependentProgramIds; // Program IDs this depends on
  std::string description;                      // Human-readable description

  bool isConstant() const { return kind == Constant; }
  bool isArgument() const { return kind == Argument; }
  bool isProgramId() const { return kind == ProgramId; }
  bool hasDependencies() const {
    return !dependentArgs.empty() || !dependentProgramIds.empty();
  }
};

/// Analyze a value to determine its symbolic representation.
/// Returns information about whether it's constant, an argument, or depends on
/// arguments.
inline SymbolicBound analyzeValue(Value v, Operation *funcOp = nullptr) {
  SymbolicBound result;

  // Check if it's a constant
  if (auto constVal = getConstantIntValue(v)) {
    result.kind = SymbolicBound::Constant;
    result.constantValue = *constVal;
    result.description = std::to_string(*constVal);
    return result;
  }

  // Check if it's a block argument (function parameter)
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    result.kind = SymbolicBound::Argument;
    result.argIndex = blockArg.getArgNumber();
    result.dependentArgs.push_back(result.argIndex);
    result.description = "%arg" + std::to_string(result.argIndex);
    return result;
  }

  // Check if it's computed from arguments
  if (Operation *defOp = v.getDefiningOp()) {
    result.kind = SymbolicBound::Expression;

    // Recursively find all argument and program_id dependencies
    SmallVector<Value> worklist;
    llvm::DenseSet<Value> visited;
    worklist.push_back(v);

    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;

      if (auto blockArg = dyn_cast<BlockArgument>(current)) {
        result.dependentArgs.push_back(blockArg.getArgNumber());
        continue;
      }

      if (Operation *op = current.getDefiningOp()) {
        // Check for tt.get_program_id
        if (op->getName().getStringRef() == "tt.get_program_id") {
          if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis")) {
            int axis = axisAttr.getInt();
            std::string dim = (axis == 0) ? "x" : (axis == 1) ? "y" : "z";
            result.dependentProgramIds.push_back(dim);
          } else {
            // Try to get from operand or other attribute
            result.dependentProgramIds.push_back("x"); // Default
          }
          continue;
        }

        for (Value operand : op->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }

    // Sort and deduplicate
    llvm::sort(result.dependentArgs);
    result.dependentArgs.erase(
        std::unique(result.dependentArgs.begin(), result.dependentArgs.end()),
        result.dependentArgs.end());

    llvm::sort(result.dependentProgramIds);
    result.dependentProgramIds.erase(
        std::unique(result.dependentProgramIds.begin(),
                    result.dependentProgramIds.end()),
        result.dependentProgramIds.end());

    // Build description
    if (result.dependentArgs.empty() && result.dependentProgramIds.empty()) {
      result.kind = SymbolicBound::Unknown;
      result.description = "<unknown>";
    } else {
      result.description = "expr(";
      bool first = true;
      for (unsigned arg : result.dependentArgs) {
        if (!first)
          result.description += ", ";
        first = false;
        result.description += "%arg" + std::to_string(arg);
      }
      for (const auto &pid : result.dependentProgramIds) {
        if (!first)
          result.description += ", ";
        first = false;
        result.description += "program_id_" + pid;
      }
      result.description += ")";
    }

    return result;
  }

  result.kind = SymbolicBound::Unknown;
  result.description = "<unknown>";
  return result;
}

/// Try to evaluate a value given argument and program_id bindings.
/// Returns std::nullopt if evaluation fails.
inline std::optional<int64_t>
evaluateValue(Value v, const llvm::DenseMap<unsigned, int64_t> &argBindings,
              const llvm::StringMap<int64_t> &programIdBindings = {}) {

  // Check if it's a constant
  if (auto constVal = getConstantIntValue(v)) {
    return *constVal;
  }

  // Check if it's a bound argument
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    auto it = argBindings.find(blockArg.getArgNumber());
    if (it != argBindings.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  // Try to evaluate the defining operation
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  // Handle tt.get_program_id
  if (defOp->getName().getStringRef() == "tt.get_program_id") {
    std::string dim = "x";
    if (auto axisAttr = defOp->getAttrOfType<mlir::IntegerAttr>("axis")) {
      int axis = axisAttr.getInt();
      dim = (axis == 0) ? "x" : (axis == 1) ? "y" : "z";
    }
    auto it = programIdBindings.find(dim);
    if (it != programIdBindings.end()) {
      return it->second;
    }
    // Also try "program_id_x" format
    auto it2 = programIdBindings.find("program_id_" + dim);
    if (it2 != programIdBindings.end()) {
      return it2->second;
    }
    return std::nullopt;
  }

  // Handle tt.get_num_programs
  if (defOp->getName().getStringRef() == "tt.get_num_programs") {
    std::string dim = "x";
    if (auto axisAttr = defOp->getAttrOfType<mlir::IntegerAttr>("axis")) {
      int axis = axisAttr.getInt();
      dim = (axis == 0) ? "x" : (axis == 1) ? "y" : "z";
    }
    auto it = programIdBindings.find("num_programs_" + dim);
    if (it != programIdBindings.end()) {
      return it->second;
    }
    auto it2 = programIdBindings.find("num_programs" + dim);
    if (it2 != programIdBindings.end()) {
      return it2->second;
    }
    return std::nullopt;
  }

  // Handle common arithmetic operations
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    auto lhs = evaluateValue(addOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(addOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs + *rhs;
  } else if (auto subOp = dyn_cast<arith::SubIOp>(defOp)) {
    auto lhs = evaluateValue(subOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(subOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs - *rhs;
  } else if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    auto lhs = evaluateValue(mulOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(mulOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs * *rhs;
  } else if (auto divOp = dyn_cast<arith::DivSIOp>(defOp)) {
    auto lhs = evaluateValue(divOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(divOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs && *rhs != 0)
      return *lhs / *rhs;
  } else if (auto divOp = dyn_cast<arith::DivUIOp>(defOp)) {
    auto lhs = evaluateValue(divOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(divOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs && *rhs != 0)
      return static_cast<int64_t>(static_cast<uint64_t>(*lhs) /
                                  static_cast<uint64_t>(*rhs));
  } else if (auto minOp = dyn_cast<arith::MinSIOp>(defOp)) {
    auto lhs = evaluateValue(minOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(minOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return std::min(*lhs, *rhs);
  } else if (auto maxOp = dyn_cast<arith::MaxSIOp>(defOp)) {
    auto lhs = evaluateValue(maxOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(maxOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return std::max(*lhs, *rhs);
  } else if (auto minOp = dyn_cast<arith::MinUIOp>(defOp)) {
    auto lhs = evaluateValue(minOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(minOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return static_cast<int64_t>(
          std::min(static_cast<uint64_t>(*lhs), static_cast<uint64_t>(*rhs)));
  } else if (auto maxOp = dyn_cast<arith::MaxUIOp>(defOp)) {
    auto lhs = evaluateValue(maxOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(maxOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return static_cast<int64_t>(
          std::max(static_cast<uint64_t>(*lhs), static_cast<uint64_t>(*rhs)));
  } else if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp)) {
    return evaluateValue(extOp.getIn(), argBindings, programIdBindings);
  } else if (auto extOp = dyn_cast<arith::ExtUIOp>(defOp)) {
    return evaluateValue(extOp.getIn(), argBindings, programIdBindings);
  } else if (auto truncOp = dyn_cast<arith::TruncIOp>(defOp)) {
    return evaluateValue(truncOp.getIn(), argBindings, programIdBindings);
  } else if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
    auto lhs = evaluateValue(remOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(remOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs && *rhs != 0)
      return *lhs % *rhs;
  } else if (auto remOp = dyn_cast<arith::RemUIOp>(defOp)) {
    auto lhs = evaluateValue(remOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(remOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs && *rhs != 0)
      return static_cast<int64_t>(static_cast<uint64_t>(*lhs) %
                                  static_cast<uint64_t>(*rhs));
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    auto cond =
        evaluateValue(selectOp.getCondition(), argBindings, programIdBindings);
    auto trueVal =
        evaluateValue(selectOp.getTrueValue(), argBindings, programIdBindings);
    auto falseVal =
        evaluateValue(selectOp.getFalseValue(), argBindings, programIdBindings);
    if (cond && trueVal && falseVal) {
      return *cond ? *trueVal : *falseVal;
    }
  } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(defOp)) {
    auto lhs = evaluateValue(cmpOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(cmpOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs) {
      switch (cmpOp.getPredicate()) {
      case arith::CmpIPredicate::eq:
        return *lhs == *rhs ? 1 : 0;
      case arith::CmpIPredicate::ne:
        return *lhs != *rhs ? 1 : 0;
      case arith::CmpIPredicate::slt:
        return *lhs < *rhs ? 1 : 0;
      case arith::CmpIPredicate::sle:
        return *lhs <= *rhs ? 1 : 0;
      case arith::CmpIPredicate::sgt:
        return *lhs > *rhs ? 1 : 0;
      case arith::CmpIPredicate::sge:
        return *lhs >= *rhs ? 1 : 0;
      case arith::CmpIPredicate::ult:
        return static_cast<uint64_t>(*lhs) < static_cast<uint64_t>(*rhs) ? 1
                                                                         : 0;
      case arith::CmpIPredicate::ule:
        return static_cast<uint64_t>(*lhs) <= static_cast<uint64_t>(*rhs) ? 1
                                                                          : 0;
      case arith::CmpIPredicate::ugt:
        return static_cast<uint64_t>(*lhs) > static_cast<uint64_t>(*rhs) ? 1
                                                                         : 0;
      case arith::CmpIPredicate::uge:
        return static_cast<uint64_t>(*lhs) >= static_cast<uint64_t>(*rhs) ? 1
                                                                          : 0;
      }
    }
  } else if (auto andOp = dyn_cast<arith::AndIOp>(defOp)) {
    auto lhs = evaluateValue(andOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(andOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs & *rhs;
  } else if (auto orOp = dyn_cast<arith::OrIOp>(defOp)) {
    auto lhs = evaluateValue(orOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(orOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs | *rhs;
  } else if (auto xorOp = dyn_cast<arith::XOrIOp>(defOp)) {
    auto lhs = evaluateValue(xorOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(xorOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs ^ *rhs;
  } else if (auto shlOp = dyn_cast<arith::ShLIOp>(defOp)) {
    auto lhs = evaluateValue(shlOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(shlOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs << *rhs;
  } else if (auto shrOp = dyn_cast<arith::ShRSIOp>(defOp)) {
    auto lhs = evaluateValue(shrOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(shrOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return *lhs >> *rhs;
  } else if (auto shrOp = dyn_cast<arith::ShRUIOp>(defOp)) {
    auto lhs = evaluateValue(shrOp.getLhs(), argBindings, programIdBindings);
    auto rhs = evaluateValue(shrOp.getRhs(), argBindings, programIdBindings);
    if (lhs && rhs)
      return static_cast<int64_t>(static_cast<uint64_t>(*lhs) >> *rhs);
  } else if (auto indexCastOp = dyn_cast<arith::IndexCastOp>(defOp)) {
    return evaluateValue(indexCastOp.getIn(), argBindings, programIdBindings);
  } else if (auto indexCastOp = dyn_cast<arith::IndexCastUIOp>(defOp)) {
    return evaluateValue(indexCastOp.getIn(), argBindings, programIdBindings);
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Loop Trip Count Analysis (with dynamic bound support)
//===----------------------------------------------------------------------===//

/// Result of loop trip count analysis.
struct LoopTripCountResult {
  bool isStatic = false;       // True if trip count is compile-time constant
  int64_t staticTripCount = 1; // Valid if isStatic
  SmallVector<unsigned> dependentArgs; // Arguments affecting trip count
  std::string errorMsg;                // Error message if analysis fails

  // Detailed bound info
  SymbolicBound lowerBound;
  SymbolicBound upperBound;
  SymbolicBound step;
};

/// Analyze scf.for loop to determine trip count or required argument bindings.
inline LoopTripCountResult analyzeScfForTripCount(scf::ForOp forOp) {
  LoopTripCountResult result;

  result.lowerBound = analyzeValue(forOp.getLowerBound());
  result.upperBound = analyzeValue(forOp.getUpperBound());
  result.step = analyzeValue(forOp.getStep());

  // Collect all dependent arguments
  for (unsigned arg : result.lowerBound.dependentArgs)
    result.dependentArgs.push_back(arg);
  for (unsigned arg : result.upperBound.dependentArgs)
    result.dependentArgs.push_back(arg);
  for (unsigned arg : result.step.dependentArgs)
    result.dependentArgs.push_back(arg);

  // Sort and deduplicate
  llvm::sort(result.dependentArgs);
  result.dependentArgs.erase(
      std::unique(result.dependentArgs.begin(), result.dependentArgs.end()),
      result.dependentArgs.end());

  // Check if all bounds are static
  if (result.lowerBound.isConstant() && result.upperBound.isConstant() &&
      result.step.isConstant()) {
    result.isStatic = true;
    int64_t lb = result.lowerBound.constantValue;
    int64_t ub = result.upperBound.constantValue;
    int64_t step = result.step.constantValue;
    if (step != 0 && ub > lb) {
      result.staticTripCount = (ub - lb + step - 1) / step;
    } else {
      result.staticTripCount = 0;
    }
  }

  return result;
}

/// Get trip count for scf.for loop with optional argument and program_id
/// bindings. If bindings are provided, tries to evaluate dynamic bounds. If
/// bindings are missing for required args/program_ids, returns error in result.
inline LoopTripCountResult getScfForTripCountWithBindings(
    scf::ForOp forOp,
    const llvm::DenseMap<unsigned, int64_t> &argBindings =
        llvm::DenseMap<unsigned, int64_t>(),
    const llvm::StringMap<int64_t> &programIdBindings =
        llvm::StringMap<int64_t>()) {

  LoopTripCountResult result = analyzeScfForTripCount(forOp);

  // If already static, we're done
  if (result.isStatic) {
    return result;
  }

  // Check if we have all required bindings
  SmallVector<std::string> missingBindings;
  for (unsigned argIdx : result.dependentArgs) {
    if (argBindings.find(argIdx) == argBindings.end()) {
      missingBindings.push_back("%arg" + std::to_string(argIdx));
    }
  }
  for (const auto &pid : result.lowerBound.dependentProgramIds) {
    if (programIdBindings.find(pid) == programIdBindings.end() &&
        programIdBindings.find("program_id_" + pid) ==
            programIdBindings.end()) {
      missingBindings.push_back("pid_" + pid);
    }
  }
  for (const auto &pid : result.upperBound.dependentProgramIds) {
    if (programIdBindings.find(pid) == programIdBindings.end() &&
        programIdBindings.find("program_id_" + pid) ==
            programIdBindings.end()) {
      // Avoid duplicates
      std::string name = "pid_" + pid;
      bool found = false;
      for (const auto &m : missingBindings) {
        if (m == name) {
          found = true;
          break;
        }
      }
      if (!found)
        missingBindings.push_back(name);
    }
  }

  if (!missingBindings.empty()) {
    result.errorMsg = "Loop trip count depends on unbound values: ";
    for (size_t i = 0; i < missingBindings.size(); ++i) {
      if (i > 0)
        result.errorMsg += ", ";
      result.errorMsg += missingBindings[i];
    }
    result.errorMsg += ". Please provide via --arg-bindings option.";
    return result;
  }

  // Try to evaluate with bindings
  auto lbVal =
      evaluateValue(forOp.getLowerBound(), argBindings, programIdBindings);
  auto ubVal =
      evaluateValue(forOp.getUpperBound(), argBindings, programIdBindings);
  auto stepVal = evaluateValue(forOp.getStep(), argBindings, programIdBindings);

  if (lbVal && ubVal && stepVal && *stepVal != 0) {
    result.isStatic = true;
    int64_t lb = *lbVal;
    int64_t ub = *ubVal;
    int64_t step = *stepVal;
    if (ub > lb) {
      result.staticTripCount = (ub - lb + step - 1) / step;
    } else {
      result.staticTripCount = 0;
    }
    result.errorMsg.clear();
  } else {
    result.errorMsg = "Failed to evaluate loop bounds with provided bindings";
  }

  return result;
}

/// Simple wrapper that returns trip count (1 if unknown).
/// For backwards compatibility.
inline int64_t getScfForTripCount(scf::ForOp forOp) {
  auto result = analyzeScfForTripCount(forOp);
  return result.isStatic ? result.staticTripCount : 1;
}

/// Get the loop multiplier for an operation by walking up the parent chain.
/// This handles nested loops correctly by multiplying all enclosing loop trip
/// counts.
inline int64_t getLoopMultiplier(Operation *op) {
  int64_t multiplier = 1;
  Operation *parent = op->getParentOp();

  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      // Check if trip_count was already computed and stored
      if (auto tripCountAttr =
              forOp->getAttrOfType<IntegerAttr>("ascend.trip_count")) {
        multiplier *= tripCountAttr.getInt();
      } else {
        multiplier *= getScfForTripCount(forOp);
      }
    }
    parent = parent->getParentOp();
  }

  return multiplier;
}

/// Check if an operation is inside any loop.
inline bool isInsideLoop(Operation *op) {
  return op->getParentOfType<scf::ForOp>() != nullptr;
}

/// Get the innermost enclosing loop for an operation.
inline scf::ForOp getInnermostLoop(Operation *op) {
  return op->getParentOfType<scf::ForOp>();
}

/// Get all enclosing loops for an operation (innermost first).
inline SmallVector<scf::ForOp> getEnclosingLoops(Operation *op) {
  SmallVector<scf::ForOp> loops;
  Operation *parent = op->getParentOp();

  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      loops.push_back(forOp);
    }
    parent = parent->getParentOp();
  }

  return loops;
}

//===----------------------------------------------------------------------===//
// Argument Binding Parsing (Generic Programming)
//===----------------------------------------------------------------------===//

/// Generic binding value that can hold different types.
/// Uses std::variant for type-safe value storage.
class BindingValue {
public:
  using ValueType = std::variant<int64_t, double, std::string>;

private:
  ValueType value_;

public:
  BindingValue() : value_(int64_t(0)) {}
  BindingValue(int64_t v) : value_(v) {}
  BindingValue(double v) : value_(v) {}
  BindingValue(const std::string &v) : value_(v) {}
  BindingValue(llvm::StringRef v) : value_(v.str()) {}

  /// Check type
  template <typename T> bool is() const {
    return std::holds_alternative<T>(value_);
  }

  bool isInt() const { return is<int64_t>(); }
  bool isFloat() const { return is<double>(); }
  bool isString() const { return is<std::string>(); }

  /// Get value with type conversion
  template <typename T> T get() const {
    if constexpr (std::is_same_v<T, int64_t>) {
      if (auto *v = std::get_if<int64_t>(&value_))
        return *v;
      if (auto *v = std::get_if<double>(&value_))
        return static_cast<int64_t>(*v);
      return 0;
    } else if constexpr (std::is_same_v<T, double>) {
      if (auto *v = std::get_if<double>(&value_))
        return *v;
      if (auto *v = std::get_if<int64_t>(&value_))
        return static_cast<double>(*v);
      return 0.0;
    } else if constexpr (std::is_same_v<T, std::string>) {
      if (auto *v = std::get_if<std::string>(&value_))
        return *v;
      if (auto *v = std::get_if<int64_t>(&value_))
        return std::to_string(*v);
      if (auto *v = std::get_if<double>(&value_))
        return std::to_string(*v);
      return "";
    } else {
      static_assert(sizeof(T) == 0, "Unsupported type for BindingValue::get()");
    }
  }

  /// Convenience accessors
  int64_t asInt() const { return get<int64_t>(); }
  double asFloat() const { return get<double>(); }
  std::string asString() const { return get<std::string>(); }

  /// Convert to string representation
  std::string toString() const {
    return std::visit(
        [](auto &&v) -> std::string {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::string>) {
            return v;
          } else {
            return std::to_string(v);
          }
        },
        value_);
  }
};

/// Generic bindings container with type-safe access.
class Bindings {
public:
  using MapType = llvm::StringMap<BindingValue>;

private:
  MapType bindings_;

public:
  Bindings() = default;

  /// Set a binding
  template <typename T> void set(llvm::StringRef name, T value) {
    bindings_[name] = BindingValue(value);
  }

  /// Get a binding with type conversion
  template <typename T> std::optional<T> get(llvm::StringRef name) const {
    auto it = bindings_.find(name);
    if (it == bindings_.end())
      return std::nullopt;
    return it->second.get<T>();
  }

  /// Get raw BindingValue
  std::optional<BindingValue> getValue(llvm::StringRef name) const {
    auto it = bindings_.find(name);
    if (it == bindings_.end())
      return std::nullopt;
    return it->second;
  }

  /// Check if binding exists
  bool has(llvm::StringRef name) const {
    return bindings_.find(name) != bindings_.end();
  }

  /// Get with default value
  template <typename T> T getOr(llvm::StringRef name, T defaultValue) const {
    auto it = bindings_.find(name);
    if (it == bindings_.end())
      return defaultValue;
    return it->second.get<T>();
  }

  /// Iterate over all bindings
  auto begin() const { return bindings_.begin(); }
  auto end() const { return bindings_.end(); }
  size_t size() const { return bindings_.size(); }
  bool empty() const { return bindings_.empty(); }

  /// Get argument binding (by index)
  std::optional<int64_t> getArg(unsigned idx) const {
    return get<int64_t>("arg" + std::to_string(idx));
  }

  /// Get program_id binding
  std::optional<int64_t> getProgramId(llvm::StringRef dim) const {
    // Try multiple formats
    if (auto v = get<int64_t>(dim))
      return v;
    if (auto v = get<int64_t>("pid_" + dim.str()))
      return v;
    if (auto v = get<int64_t>("program_id_" + dim.str()))
      return v;
    return std::nullopt;
  }

  /// Export to legacy formats for backwards compatibility
  void exportArgBindings(llvm::DenseMap<unsigned, int64_t> &out) const {
    for (const auto &kv : bindings_) {
      llvm::StringRef key = kv.first();
      if (key.starts_with("arg")) {
        unsigned idx;
        if (!key.substr(3).getAsInteger(10, idx)) {
          out[idx] = kv.second.asInt();
        }
      }
    }
  }

  void exportProgramIdBindings(llvm::StringMap<int64_t> &out) const {
    for (const auto &kv : bindings_) {
      llvm::StringRef key = kv.first();
      if (key.starts_with("pid_") || key.starts_with("program_id_") ||
          key.starts_with("num_programs") || key == "x" || key == "y" ||
          key == "z") {
        out[key] = kv.second.asInt();
      }
    }
  }
};

/// Parse bindings from string format.
/// Supports: "arg2=100,arg3=1.5,pid_x=0,scale=0.125,name=hello"
/// Returns Bindings object on success, error message on failure.
inline llvm::Expected<Bindings> parseBindings(llvm::StringRef input) {
  Bindings result;

  if (input.empty())
    return result;

  SmallVector<StringRef> pairs;
  input.split(pairs, ',');

  for (StringRef pair : pairs) {
    pair = pair.trim();
    if (pair.empty())
      continue;

    size_t eqPos = pair.find('=');
    if (eqPos == StringRef::npos) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Invalid binding format: '%s' (expected 'key=value')",
          pair.str().c_str());
    }

    StringRef key = pair.substr(0, eqPos).trim();
    StringRef val = pair.substr(eqPos + 1).trim();

    // Determine value type and parse
    BindingValue bindingVal;

    // Try integer first
    int64_t intVal;
    if (!val.getAsInteger(10, intVal)) {
      bindingVal = BindingValue(intVal);
    }
    // Try float (contains '.', 'e', or 'E')
    else if (val.contains('.') || val.contains('e') || val.contains('E')) {
      std::string valStr = val.str();
      char *endPtr = nullptr;
      double floatVal = std::strtod(valStr.c_str(), &endPtr);
      if (endPtr != valStr.c_str() && *endPtr == '\0') {
        bindingVal = BindingValue(floatVal);
      } else {
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Invalid float value: '%s'",
                                       val.str().c_str());
      }
    }
    // Treat as string
    else {
      bindingVal = BindingValue(val.str());
    }

    // Normalize key and store
    std::string normalizedKey = key.str();
    result.set(normalizedKey, bindingVal);

    // Also store normalized versions for common patterns
    if (key.starts_with("arg")) {
      // Already in correct format
    } else if (key[0] >= '0' && key[0] <= '9') {
      // Numeric key -> argN
      result.set("arg" + key.str(), bindingVal);
    } else if (key == "pid_x" || key == "pidx" || key == "program_id_x") {
      result.set("x", bindingVal);
      result.set("pid_x", bindingVal);
      result.set("program_id_x", bindingVal);
    } else if (key == "pid_y" || key == "pidy" || key == "program_id_y") {
      result.set("y", bindingVal);
      result.set("pid_y", bindingVal);
      result.set("program_id_y", bindingVal);
    } else if (key == "pid_z" || key == "pidz" || key == "program_id_z") {
      result.set("z", bindingVal);
      result.set("pid_z", bindingVal);
      result.set("program_id_z", bindingVal);
    } else if (key == "num_programs_x" || key == "num_programsx") {
      result.set("num_programs_x", bindingVal);
      result.set("num_programsx", bindingVal);
    } else if (key == "num_programs_y" || key == "num_programsy") {
      result.set("num_programs_y", bindingVal);
      result.set("num_programsy", bindingVal);
    } else if (key == "num_programs_z" || key == "num_programsz") {
      result.set("num_programs_z", bindingVal);
      result.set("num_programsz", bindingVal);
    }
  }

  return result;
}

/// Legacy parseBindings for backwards compatibility.
inline bool parseBindings(llvm::StringRef input,
                          llvm::DenseMap<unsigned, int64_t> &argBindings,
                          llvm::StringMap<int64_t> &programIdBindings,
                          std::string &error) {
  auto result = parseBindings(input);
  if (!result) {
    error = llvm::toString(result.takeError());
    return false;
  }

  result->exportArgBindings(argBindings);
  result->exportProgramIdBindings(programIdBindings);
  return true;
}

/// Backwards-compatible wrapper that only parses arg bindings.
inline bool parseArgBindings(llvm::StringRef input,
                             llvm::DenseMap<unsigned, int64_t> &bindings,
                             std::string &error) {
  llvm::StringMap<int64_t> programIdBindings; // Ignored
  return parseBindings(input, bindings, programIdBindings, error);
}

/// Parse loop trip counts from string format: "4,100,6588" (in loop order)
/// Returns true on success, false on parse error.
inline bool parseLoopTripCounts(llvm::StringRef input,
                                SmallVector<int64_t> &tripCounts,
                                std::string &error) {
  if (input.empty())
    return true;

  SmallVector<StringRef> parts;
  input.split(parts, ',');

  for (StringRef part : parts) {
    part = part.trim();
    if (part.empty())
      continue;

    int64_t value;
    if (part.getAsInteger(10, value)) {
      error = "Invalid trip count: " + part.str();
      return false;
    }

    if (value <= 0) {
      error = "Trip count must be positive: " + part.str();
      return false;
    }

    tripCounts.push_back(value);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Operation Classification
//===----------------------------------------------------------------------===//

/// Compute path classification for operations.
enum class ComputePath {
  Cube,   // Matrix operations: matmul
  Vector, // SIMD operations: elementwise, reduce, etc.
  Memory, // Memory operations: load/store
  Unknown
};

/// Get a string name for ComputePath.
inline StringRef getComputePathName(ComputePath path) {
  switch (path) {
  case ComputePath::Cube:
    return "Cube";
  case ComputePath::Vector:
    return "Vector";
  case ComputePath::Memory:
    return "Memory";
  default:
    return "Unknown";
  }
}

} // namespace utils
} // namespace ascend
} // namespace mlir

#endif // ASCENDMODEL_UTILS_H
