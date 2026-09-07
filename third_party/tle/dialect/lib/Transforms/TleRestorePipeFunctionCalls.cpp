#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include <map>

namespace mlir::triton::tle {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttnvws = mlir::triton::nvws;

#define GEN_PASS_DEF_TRITONTLERESTOREPIPEFUNCTIONCALLS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

struct CallScope {
  PipeCallBeginOp begin;
  PipeCallEndOp end;
};

struct OutlinedFunction {
  tt::FuncOp function;
  SmallVector<Type> argumentTypes;
  SmallVector<std::string> argumentSources;
  unsigned callArgumentCount;
  int32_t numWarps;
};

static std::string describeValue(Value value) {
  std::string description;
  llvm::raw_string_ostream stream(description);
  if (Operation *definition = value.getDefiningOp()) {
    stream << definition->getName();
  } else if (auto argument = dyn_cast<BlockArgument>(value)) {
    stream << "block argument " << argument.getArgNumber();
  } else {
    stream << "unknown value";
  }
  return description;
}

static bool isNvwsTokenValue(Value value) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  return tensorType && isa<ttnvws::TokenType>(tensorType.getElementType());
}

static bool isDefinedInScope(Value value,
                             const DenseSet<Operation *> &rootOps) {
  Operation *op = value.getDefiningOp();
  if (!op) {
    auto blockArg = dyn_cast<BlockArgument>(value);
    op = blockArg ? blockArg.getOwner()->getParentOp() : nullptr;
  }
  for (; op; op = op->getParentOp()) {
    if (rootOps.contains(op))
      return true;
  }
  return false;
}

static SmallVector<Value> collectCaptures(ArrayRef<Operation *> operations,
                                          const DenseSet<Operation *> &rootOps,
                                          ValueRange callArguments,
                                          ValueRange callAliases) {
  DenseSet<Value> seen;
  SmallVector<Value> captures(callArguments.begin(), callArguments.end());
  seen.insert(callArguments.begin(), callArguments.end());
  seen.insert(callAliases.begin(), callAliases.end());
  for (Operation *root : operations) {
    root->walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (!isDefinedInScope(operand, rootOps) && seen.insert(operand).second)
          captures.push_back(operand);
      }
    });
  }
  return captures;
}

static std::string getConstantKey(Operation *constant) {
  std::string key;
  llvm::raw_string_ostream stream(key);
  stream << constant->getName() << " " << constant->getAttrDictionary()
         << " : ";
  llvm::interleaveComma(constant->getResultTypes(), stream);
  return key;
}

static void normalizeConstants(ArrayRef<Operation *> operations,
                               const DenseSet<Operation *> &rootOps,
                               PipeCallBeginOp begin,
                               ArrayRef<Operation *> requiredConstants = {}) {
  if (operations.empty())
    return;

  DenseSet<Value> callArguments;
  if (begin) {
    callArguments.insert(begin.getAliases().begin(), begin.getAliases().end());
    for (Value argument : begin.getArguments()) {
      if (isNvwsTokenValue(argument))
        callArguments.insert(argument);
    }
  }
  SetVector<Operation *> constants;
  for (Operation *root : operations) {
    root->walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::ConstantLike>())
        constants.insert(op);
      for (Value operand : op->getOperands()) {
        if (callArguments.contains(operand))
          continue;
        Operation *definition = operand.getDefiningOp();
        if (definition && definition->hasTrait<OpTrait::ConstantLike>())
          constants.insert(definition);
      }
    });
  }
  constants.insert(requiredConstants.begin(), requiredConstants.end());
  if (constants.empty())
    return;

  std::map<std::string, SmallVector<Operation *>> constantsByKey;
  for (Operation *constant : constants)
    constantsByKey[getConstantKey(constant)].push_back(constant);

  OpBuilder builder(operations.front());
  IRMapping mapping;
  for (auto &[key, equivalentConstants] : constantsByKey) {
    Operation *clone = builder.clone(*equivalentConstants.front());
    for (Operation *constant : equivalentConstants) {
      for (auto [source, result] :
           llvm::zip(constant->getResults(), clone->getResults()))
        mapping.map(source, result);
    }
  }
  for (Operation *root : operations) {
    root->walk([&](Operation *op) {
      for (OpOperand &operand : op->getOpOperands()) {
        if (Value replacement = mapping.lookupOrNull(operand.get()))
          operand.set(replacement);
      }
    });
  }
  for (Operation *constant : llvm::reverse(constants)) {
    if (isDefinedInScope(constant->getResult(0), rootOps) &&
        constant->use_empty())
      constant->erase();
  }
}

static SmallVector<Operation *>
collectTopLevelConstants(ArrayRef<Operation *> operations) {
  SmallVector<Operation *> constants;
  for (Operation *op : operations) {
    if (op->hasTrait<OpTrait::ConstantLike>())
      constants.push_back(op);
  }
  return constants;
}

static LogicalResult verifyNoEscapingResults(ArrayRef<Operation *> operations,
                                             const DenseSet<Operation *> &rootOps,
                                             PipeCallBeginOp begin) {
  for (Operation *root : operations) {
    for (Value result : root->getResults()) {
      for (Operation *user : result.getUsers()) {
        bool inside = false;
        for (Operation *op = user; op; op = op->getParentOp()) {
          if (rootOps.contains(op)) {
            inside = true;
            break;
          }
        }
        if (!inside)
          return begin.emitOpError("cannot restore pipe helper with a result "
                                   "that escapes its marked call scope");
      }
    }
  }
  return success();
}

static FailureOr<SmallVector<CallScope>> collectCallScopes(ModuleOp module) {
  SmallVector<CallScope> scopes;
  WalkResult result = module.walk([&](Block *block) -> WalkResult {
    SmallVector<PipeCallBeginOp> stack;
    for (Operation &op : *block) {
      if (auto begin = dyn_cast<PipeCallBeginOp>(op)) {
        stack.push_back(begin);
        continue;
      }
      auto end = dyn_cast<PipeCallEndOp>(op);
      if (!end)
        continue;
      if (stack.empty()) {
        end.emitOpError("has no matching pipe.call_begin");
        return WalkResult::interrupt();
      }
      PipeCallBeginOp begin = stack.pop_back_val();
      if (begin.getCallId() != end.getCallId() ||
          begin.getCallee() != end.getCallee()) {
        end.emitOpError("does not match the innermost pipe.call_begin");
        return WalkResult::interrupt();
      }
      scopes.push_back({begin, end});
    }
    if (!stack.empty()) {
      stack.back().emitOpError("has no matching pipe.call_end");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return scopes;
}

static FailureOr<OutlinedFunction>
createOutlinedFunction(ModuleOp module, StringRef name,
                       ArrayRef<Operation *> operations,
                       ArrayRef<Value> captures, ValueRange callAliases,
                       int32_t numWarps, Location loc) {
  if (module.lookupSymbol(name))
    return emitError(loc) << "cannot restore pipe helper " << name
                          << " because that symbol already exists";

  SmallVector<Type> argumentTypes =
      llvm::to_vector(llvm::map_range(captures, [](Value value) {
        return value.getType();
      }));
  OpBuilder builder = OpBuilder::atBlockEnd(module.getBody());
  auto functionType = builder.getFunctionType(argumentTypes, {});
  auto function = tt::FuncOp::create(builder, loc, name, functionType);
  function.setVisibility(SymbolTable::Visibility::Private);
  function->setAttr("noinline", builder.getBoolAttr(true));
  function->setAttr(ttg::AttrNumWarpsName,
                    builder.getI32IntegerAttr(numWarps));
  Block *entry = function.addEntryBlock();

  IRMapping mapping;
  unsigned callArgumentCount = callAliases.size();
  for (unsigned index = 0; index < callArgumentCount; ++index) {
    mapping.map(callAliases[index], entry->getArgument(index));
    if (isNvwsTokenValue(captures[index]))
      mapping.map(captures[index], entry->getArgument(index));
  }
  for (unsigned index = callArgumentCount; index < captures.size(); ++index)
    mapping.map(captures[index], entry->getArgument(index));

  builder.setInsertionPointToStart(entry);
  for (Operation *op : operations)
    builder.clone(*op, mapping);
  tt::ReturnOp::create(builder, loc);
  SmallVector<std::string> argumentSources = llvm::to_vector(
      llvm::map_range(captures, describeValue));
  return OutlinedFunction{function, std::move(argumentTypes),
                          std::move(argumentSources), callArgumentCount,
                          numWarps};
}

static LogicalResult verifyEquivalentBody(OutlinedFunction &outlined,
                                          ArrayRef<Operation *> operations,
                                          ArrayRef<Value> captures,
                                          ValueRange callAliases,
                                          PipeCallBeginOp begin) {
  Block &entry = outlined.function.getBody().front();
  SmallVector<Operation *> outlinedOperations;
  for (Operation &op : entry.without_terminator())
    outlinedOperations.push_back(&op);

  DenseMap<Value, Value> equivalentValues;
  unsigned callArgumentCount = callAliases.size();
  for (unsigned index = 0; index < callArgumentCount; ++index)
    equivalentValues.try_emplace(entry.getArgument(index), callAliases[index]);
  for (unsigned index = callArgumentCount; index < captures.size(); ++index)
    equivalentValues.try_emplace(entry.getArgument(index), captures[index]);

  auto checkEquivalent = [&](Value lhs, Value rhs) -> LogicalResult {
    auto it = equivalentValues.find(lhs);
    if (it != equivalentValues.end() && it->second == rhs)
      return success();
    auto argument = dyn_cast<BlockArgument>(lhs);
    if (!argument || argument.getOwner() != &entry ||
        argument.getArgNumber() >= callArgumentCount)
      return failure();
    unsigned index = argument.getArgNumber();
    return success(rhs == callAliases[index] ||
                   (isNvwsTokenValue(captures[index]) &&
                    rhs == captures[index]));
  };
  auto markEquivalent = [&](Value lhs, Value rhs) {
    auto [it, inserted] = equivalentValues.try_emplace(lhs, rhs);
    assert(inserted || it->second == rhs);
  };
  unsigned commonOperations =
      std::min(outlinedOperations.size(), operations.size());
  for (unsigned index = 0; index < commonOperations; ++index) {
    Operation *lhs = outlinedOperations[index];
    Operation *rhs = operations[index];
    if (!OperationEquivalence::isEquivalentTo(
            lhs, rhs, checkEquivalent, markEquivalent,
            OperationEquivalence::IgnoreLocations)) {
      InFlightDiagnostic diagnostic =
          begin.emitOpError("calls to pipe helper ");
      diagnostic << outlined.function.getName()
                 << " do not have one structurally equivalent lowered body: "
                 << "operation " << index << " expected " << lhs->getName()
                 << " " << lhs->getAttrDictionary() << " but got "
                 << rhs->getName() << " " << rhs->getAttrDictionary();
      for (auto [operandIndex, lhsOperand, rhsOperand] :
           llvm::enumerate(lhs->getOperands(), rhs->getOperands())) {
        diagnostic << "; operand " << operandIndex << " expected "
                   << lhsOperand.getType() << " from "
                   << describeValue(lhsOperand) << " but got "
                   << rhsOperand.getType() << " from "
                   << describeValue(rhsOperand);
        for (unsigned argumentIndex = 0;
             argumentIndex < callArgumentCount; ++argumentIndex) {
          if (rhsOperand == captures[argumentIndex])
            diagnostic << " (call operand " << argumentIndex << ")";
          if (rhsOperand == callAliases[argumentIndex])
            diagnostic << " (call alias " << argumentIndex << ")";
        }
      }
      return failure();
    }
  }
  if (outlinedOperations.size() != operations.size()) {
    Operation *extra = outlinedOperations.size() > commonOperations
                           ? outlinedOperations[commonOperations]
                           : operations[commonOperations];
    return begin.emitOpError("calls to pipe helper ")
           << outlined.function.getName()
           << " do not have one structurally equivalent lowered body: "
           << "expected " << outlinedOperations.size()
           << " top-level operations but got " << operations.size()
           << "; first extra operation is " << extra->getName() << " "
           << extra->getAttrDictionary();
  }
  return success();
}

class TritonTleRestorePipeFunctionCallsPass
    : public impl::TritonTleRestorePipeFunctionCallsBase<
          TritonTleRestorePipeFunctionCallsPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<SmallVector<CallScope>> scopes = collectCallScopes(module);
    if (failed(scopes)) {
      signalPassFailure();
      return;
    }

    std::map<std::string, OutlinedFunction> functions;
    // collectCallScopes records an inner scope before its enclosing scope.
    for (CallScope scope : *scopes) {
      if (!scope.begin || !scope.end)
        continue;
      if (scope.begin->getBlock() != scope.end->getBlock()) {
        scope.begin.emitOpError("pipe helper call markers must share a block");
        signalPassFailure();
        return;
      }

      SmallVector<Operation *> operations;
      for (Operation *op = scope.begin->getNextNode(); op && op != scope.end;
           op = op->getNextNode())
        operations.push_back(op);
      if (operations.empty()) {
        scope.begin.emitOpError("cannot restore an empty pipe helper call");
        signalPassFailure();
        return;
      }

      DenseSet<Operation *> rootOps(operations.begin(), operations.end());
      normalizeConstants(operations, rootOps, scope.begin);
      operations.clear();
      for (Operation *op = scope.begin->getNextNode(); op && op != scope.end;
           op = op->getNextNode())
        operations.push_back(op);
      rootOps = DenseSet<Operation *>(operations.begin(), operations.end());
      int32_t numWarps = ttg::lookupNumWarps(scope.begin);
      std::string name = scope.begin.getCallee().str();
      auto it = functions.find(name);
      if (it != functions.end()) {
        Block &entry = it->second.function.getBody().front();
        SmallVector<Operation *> functionOperations;
        for (Operation &op : entry.without_terminator())
          functionOperations.push_back(&op);
        DenseSet<Operation *> functionRootOps(functionOperations.begin(),
                                              functionOperations.end());
        SmallVector<Operation *> currentConstants =
            collectTopLevelConstants(operations);
        normalizeConstants(functionOperations, functionRootOps,
                           PipeCallBeginOp{}, currentConstants);

        functionOperations.clear();
        for (Operation &op : entry.without_terminator())
          functionOperations.push_back(&op);
        SmallVector<Operation *> canonicalConstants =
            collectTopLevelConstants(functionOperations);
        normalizeConstants(operations, rootOps, scope.begin,
                           canonicalConstants);

        operations.clear();
        for (Operation *op = scope.begin->getNextNode(); op && op != scope.end;
             op = op->getNextNode())
          operations.push_back(op);
        rootOps = DenseSet<Operation *>(operations.begin(), operations.end());
      }
      if (failed(verifyNoEscapingResults(operations, rootOps, scope.begin))) {
        signalPassFailure();
        return;
      }
      SmallVector<Value> captures =
          collectCaptures(operations, rootOps, scope.begin.getArguments(),
                          scope.begin.getAliases());
      SmallVector<Type> argumentTypes =
          llvm::to_vector(llvm::map_range(captures, [](Value value) {
            return value.getType();
          }));
      if (it == functions.end()) {
        FailureOr<OutlinedFunction> outlined = createOutlinedFunction(
            module, name, operations, captures,
            scope.begin.getAliases(), numWarps, scope.begin.getLoc());
        if (failed(outlined)) {
          signalPassFailure();
          return;
        }
        it = functions.emplace(name, std::move(*outlined)).first;
      } else if (it->second.argumentTypes != argumentTypes ||
                 it->second.numWarps != numWarps) {
        InFlightDiagnostic diagnostic =
            scope.begin.emitOpError("calls to pipe helper ");
        diagnostic << name << " do not have one structural lowered ABI: "
                   << "expected " << it->second.argumentTypes.size()
                   << " arguments and " << it->second.numWarps
                   << " warps, got " << argumentTypes.size()
                   << " arguments and " << numWarps << " warps; call ABI has "
                   << it->second.callArgumentCount << " versus "
                   << scope.begin.getArguments().size() << " arguments";
        unsigned commonArguments =
            std::min(it->second.argumentTypes.size(), argumentTypes.size());
        for (unsigned index = 0; index < commonArguments; ++index) {
          Type expected = it->second.argumentTypes[index];
          Type actual = argumentTypes[index];
          if (expected == actual)
            continue;
          diagnostic << "; argument " << index << " expected " << expected
                     << " from " << it->second.argumentSources[index]
                     << " but got " << actual << " from "
                     << describeValue(captures[index]);
          break;
        }
        signalPassFailure();
        return;
      } else if (failed(verifyEquivalentBody(it->second, operations, captures,
                                             scope.begin.getAliases(),
                                             scope.begin))) {
        signalPassFailure();
        return;
      }

      OpBuilder builder(scope.begin);
      tt::CallOp::create(builder, scope.begin.getLoc(),
                         it->second.function.getName(), TypeRange{}, captures);
      for (Operation *op : llvm::reverse(operations))
        op->erase();
      scope.end.erase();
      scope.begin.erase();
    }

    bool hasMarker = false;
    module.walk([&](Operation *op) {
      if (isa<PipeCallBeginOp, PipeCallEndOp>(op))
        hasMarker = true;
    });
    if (hasMarker) {
      module.emitError("failed to restore every marked pipe helper call");
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::triton::tle
