#include "TritonToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>
#include <string>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_NORMALIZEDEBUGLINELOCATIONS
#include "ascend/include/TritonToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kOriginAttr = "triton.debug_line.origin";
constexpr llvm::StringLiteral kClassAttr = "triton.debug_line.class";

enum class DebugLineLocClass { Semantic, Control, Synthetic };

struct SourceLine {
  StringAttr file;
  unsigned line = 0;

  explicit operator bool() const { return file && line != 0; }

  bool operator==(const SourceLine &rhs) const {
    return file == rhs.file && line == rhs.line;
  }

  bool operator!=(const SourceLine &rhs) const { return !(*this == rhs); }

  std::string getKey() const {
    if (!*this)
      return {};

    std::string key = file.getValue().str();
    key += ":";
    key += std::to_string(line);
    return key;
  }
};

bool isInternalName(StringRef name) {
  return name.contains("synthetic") || name.contains("lowering") ||
         name.contains("helper") || name.contains("tmp") ||
         name.contains("__") || name.starts_with("llvm.") ||
         name.starts_with("triton.");
}

bool hasRealFileLineColLoc(Location loc) {
  return static_cast<bool>(loc->findInstanceOf<FileLineColLoc>());
}

Location canonicalizeSourceLoc(Location loc) {
  if (isa<UnknownLoc>(loc))
    return loc;

  // If debug scopes were already materialized, keep the fused metadata intact.
  if (loc->findInstanceOf<FusedLocWith<LLVM::DIScopeAttr>>())
    return loc;

  if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
    Location child = nameLoc.getChildLoc();
    if (hasRealFileLineColLoc(child) &&
        isInternalName(nameLoc.getName().getValue()))
      return child;
  }

  return loc;
}

SourceLine getSourceLine(Location loc) {
  if (auto fileLoc = loc->findInstanceOf<FileLineColLoc>())
    return {fileLoc.getFilename(), fileLoc.getLine()};

  return {};
}

bool hasSourceLikeLocation(Operation *op) {
  Location loc = canonicalizeSourceLoc(op->getLoc());
  return !isa<UnknownLoc>(loc) && hasRealFileLineColLoc(loc);
}

bool isContainerOp(Operation *op) {
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Case("builtin.module", true)
      .Case("func.func", true)
      .Case("llvm.func", true)
      .Case("tt.func", true)
      .Default(false);
}

bool isControlOp(Operation *op) {
  StringRef name = op->getName().getStringRef();

  return llvm::StringSwitch<bool>(name)
      .Case("scf.for", true)
      .Case("scf.while", true)
      .Case("scf.if", true)
      .Case("scf.condition", true)
      .Case("cf.br", true)
      .Case("cf.cond_br", true)
      .Case("cf.switch", true)
      .Case("llvm.br", true)
      .Case("llvm.cond_br", true)
      .Case("llvm.switch", true)
      .Case("llvm.return", true)
      .Case("func.return", true)
      .Default(false);
}

bool isAlwaysSyntheticOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  // Do not classify memref.copy as always synthetic. In TTIR-to-TTAdapter
  // lowering, a user-visible `tt.load` may become a memref.copy-based sequence.
  // Keeping memref.copy source locations preserves stable DWARF anchors for
  // load lines that otherwise may disappear from the final line table.
  return llvm::StringSwitch<bool>(name)
      .Case("scf.yield", true)
      .Case("builtin.unrealized_conversion_cast", true)
      .Case("llvm.mlir.undef", true)
      .Case("llvm.mlir.zero", true)
      .Case("llvm.mlir.constant", true)
      .Case("llvm.mlir.poison", true)
      .Case("llvm.mlir.none", true)
      .Default(false);
}

bool isTensorOnlyLinalgFillOp(Operation *op) {
  if (op->getName().getStringRef() != "linalg.fill")
    return false;

  if (op->getNumResults() == 0)
    return false;

  return llvm::all_of(op->getResultTypes(),
                      [](Type type) { return isa<TensorType>(type); });
}

bool isExplicitlyMarkedSynthetic(Operation *op) {
  if (auto classAttr = op->getAttrOfType<StringAttr>(kClassAttr))
    return classAttr.getValue() == "synthetic";

  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if ((name.contains("synthetic") || name.contains("lowering")) &&
        (isa<UnitAttr>(attr.getValue()) ||
         (isa<BoolAttr>(attr.getValue()) &&
          cast<BoolAttr>(attr.getValue()).getValue())))
      return true;
  }

  return false;
}

bool isRealMemoryOrCallOp(Operation *op) {
  StringRef name = op->getName().getStringRef();

  return name.ends_with(".load") || name.ends_with(".store") ||
         name.contains("atomic") || name.ends_with(".call") ||
         name == "func.call" || name == "llvm.call";
}

bool isArithmeticOrCastOp(Operation *op) {
  StringRef name = op->getName().getStringRef();

  return name.starts_with("arith.") || name.ends_with(".add") ||
         name.ends_with(".sub") || name.ends_with(".mul") ||
         name.ends_with(".div") || name.ends_with(".rem") ||
         name.ends_with(".and") || name.ends_with(".or") ||
         name.ends_with(".xor") || name.ends_with(".shl") ||
         name.ends_with(".shr") || name.contains(".cmp") ||
         name.contains("cast") || name.contains("ext") ||
         name.contains("trunc");
}

bool isUserVisibleLoadAnchorOp(Operation *op) {
  return op->getName().getStringRef() == "memref.copy";
}

bool isUserVisibleStoreAnchorOp(Operation *op) {
  return op->getName().getStringRef() ==
         "bufferization.materialize_in_destination";
}

bool isDebugNop(Operation *op) {
  auto inlineAsm = dyn_cast<LLVM::InlineAsmOp>(op);
  return inlineAsm && inlineAsm.getAsmString() == "nop";
}

bool isDestinationViewOp(Operation *op) {
  StringRef name = op->getName().getStringRef();

  return name == "memref.reinterpret_cast" || name == "memref.subview";
}

bool isValuePreparationOp(Operation *op) {
  StringRef name = op->getName().getStringRef();

  return llvm::StringSwitch<bool>(name)
             .Case("arith.constant", true)
             .Case("tensor.empty", true)
             .Case("tensor.from_elements", true)
             .Case("bufferization.alloc_tensor", true)
             .Case("bufferization.to_tensor", true)
             .Case("bufferization.to_buffer", true)
             .Default(false) ||
         isTensorOnlyLinalgFillOp(op);
}

bool isFloatingPointExtensionOrTruncationOp(Operation *op) {
  return llvm::StringSwitch<bool>(op->getName().getStringRef())
      .Case("arith.extf", true)
      .Case("arith.truncf", true)
      .Default(false);
}

bool isAddressComputationOp(Operation *op) {
  StringRef name = op->getName().getStringRef();

  if (isFloatingPointExtensionOrTruncationOp(op))
    return false;

  return llvm::StringSwitch<bool>(name)
             .Case("memref.reinterpret_cast", true)
             .Case("memref.subview", true)
             .Case("memref.memory_space_cast", true)
             .Default(false) ||
         name.contains("cast") || name.contains("ext") ||
         name.contains("trunc");
}

// `previousLine` is the last seen source line of a semantic or control
// operation. A semantic arithmetic operation that points to an earlier source
// line was likely reordered by lowering, so treat it as synthetic.
bool isReorderedBackwardStep(Operation *op, DebugLineLocClass locClass,
                             SourceLine previousLine) {
  if (locClass != DebugLineLocClass::Semantic || !previousLine)
    return false;

  SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
  if (!line || line.file != previousLine.file || line.line >= previousLine.line)
    return false;

  return isArithmeticOrCastOp(op);
}

std::optional<Location> getUniqueStoreLoc(llvm::ArrayRef<Location> candidates) {
  std::optional<Location> uniqueLoc;

  for (Location candidate : candidates) {
    Location canonicalLoc = canonicalizeSourceLoc(candidate);
    if (!uniqueLoc) {
      uniqueLoc = canonicalLoc;
      continue;
    }

    if (*uniqueLoc != canonicalLoc)
      return std::nullopt;
  }

  return uniqueLoc;
}

void collectUserVisibleStoreLocsForDestinationView(
    Operation *view, llvm::SmallVectorImpl<Location> &candidates) {
  if (!isDestinationViewOp(view) || view->getNumResults() != 1)
    return;

  SourceLine viewLine = getSourceLine(canonicalizeSourceLoc(view->getLoc()));
  if (!viewLine)
    return;

  Value result = view->getResult(0);

  for (Operation *user : result.getUsers()) {
    if (!isUserVisibleStoreAnchorOp(user))
      continue;

    SourceLine userLine = getSourceLine(canonicalizeSourceLoc(user->getLoc()));

    if (!userLine || userLine != viewLine)
      continue;

    for (OpOperand &operand : user->getOpOperands()) {
      if (operand.get() != result)
        continue;

      // bufferization.materialize_in_destination:
      //   operand #0 = source tensor/value
      //   operand #1 = destination memref/view
      if (operand.getOperandNumber() == 1)
        candidates.push_back(user->getLoc());
    }
  }
}

void collectUserVisibleStoreLocsThroughDestinationView(
    Operation *op, llvm::SmallVectorImpl<Location> &candidates) {
  if (op->getNumResults() != 1)
    return;

  Value result = op->getResult(0);

  for (Operation *viewUser : result.getUsers()) {
    collectUserVisibleStoreLocsForDestinationView(viewUser, candidates);
  }
}

void collectUserVisibleStoreLocsThroughTensorInsert(
    Operation *op, llvm::SmallVectorImpl<Location> &candidates) {
  if (op->getNumResults() != 1)
    return;

  Value result = op->getResult(0);

  for (Operation *insertUser : result.getUsers()) {
    if (insertUser->getName().getStringRef() != "tensor.insert")
      continue;

    // tensor.insert:
    //   operand #0 = scalar/value being inserted
    if (insertUser->getNumOperands() == 0 ||
        insertUser->getOperand(0) != result)
      continue;

    if (insertUser->getNumResults() != 1)
      continue;

    SourceLine insertLine =
        getSourceLine(canonicalizeSourceLoc(insertUser->getLoc()));
    if (!insertLine)
      continue;

    Value insertedTensor = insertUser->getResult(0);

    for (Operation *storeUser : insertedTensor.getUsers()) {
      if (!isUserVisibleStoreAnchorOp(storeUser))
        continue;

      SourceLine storeLine =
          getSourceLine(canonicalizeSourceLoc(storeUser->getLoc()));

      if (!storeLine || storeLine != insertLine)
        continue;

      for (OpOperand &operand : storeUser->getOpOperands()) {
        if (operand.get() != insertedTensor)
          continue;

        // bufferization.materialize_in_destination:
        //   operand #0 = source tensor/value
        //   operand #1 = destination memref/view
        if (operand.getOperandNumber() == 0)
          candidates.push_back(storeUser->getLoc());
      }
    }
  }
}

DebugLineLocClass
classifyOperation(Operation *op,
                  const llvm::StringMap<unsigned> &semanticAnchors) {
  if (isContainerOp(op))
    return DebugLineLocClass::Semantic;

  if (isControlOp(op))
    return DebugLineLocClass::Control;

  if (isExplicitlyMarkedSynthetic(op))
    return DebugLineLocClass::Synthetic;

  if (!hasSourceLikeLocation(op))
    return DebugLineLocClass::Synthetic;

  if (isUserVisibleLoadAnchorOp(op) || isUserVisibleStoreAnchorOp(op))
    return DebugLineLocClass::Semantic;

  if (isRealMemoryOrCallOp(op))
    return DebugLineLocClass::Semantic;

  if (isValuePreparationOp(op))
    return DebugLineLocClass::Synthetic;

  SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
  bool hasSemanticAnchorOnSameLine =
      line && semanticAnchors.lookup(line.getKey()) > 0;

  if (isFloatingPointExtensionOrTruncationOp(op))
    return hasSemanticAnchorOnSameLine ? DebugLineLocClass::Synthetic
                                       : DebugLineLocClass::Semantic;

  if (isAddressComputationOp(op))
    return DebugLineLocClass::Synthetic;

  if (isArithmeticOrCastOp(op)) {
    if (hasSemanticAnchorOnSameLine)
      return DebugLineLocClass::Synthetic;
    return DebugLineLocClass::Semantic;
  }

  if (isAlwaysSyntheticOp(op))
    return DebugLineLocClass::Synthetic;

  return DebugLineLocClass::Semantic;
}

StringRef stringifyClass(DebugLineLocClass locClass) {
  switch (locClass) {
  case DebugLineLocClass::Semantic:
    return "semantic";
  case DebugLineLocClass::Control:
    return "control";
  case DebugLineLocClass::Synthetic:
    return "synthetic";
  }
  llvm_unreachable("unknown debug line location class");
}

Location makeSyntheticDebugLoc(Operation *op, Location originalLoc) {
  MLIRContext *context = op->getContext();

  Location sourceLoc = canonicalizeSourceLoc(originalLoc);

  if (auto fileLoc = sourceLoc->findInstanceOf<FileLineColLoc>()) {
    // Do not use UnknownLoc here: downstream LLVM/Ascend debug-info lowering
    // expects a file-like location for some operations and may fail on
    // UnknownLoc.
    //
    // DWARF line 0 is used as a non-user source anchor: it preserves a valid
    // file identity while preventing synthetic/helper ops from being associated
    // with real Python source lines.
    return FileLineColLoc::get(context, fileLoc.getFilename().getValue(), 0, 0);
  }

  return originalLoc;
}

void preserveOrigin(Operation *op, Location originalLoc) {
  if (!isa<UnknownLoc>(originalLoc) && !op->hasAttr(kOriginAttr))
    op->setAttr(kOriginAttr, originalLoc);
}

void markRetargetedSyntheticOp(Operation *op, Location originalLoc,
                               Location storeLoc) {
  MLIRContext *context = op->getContext();

  op->setAttr(
      kClassAttr,
      StringAttr::get(context, stringifyClass(DebugLineLocClass::Semantic)));
  preserveOrigin(op, originalLoc);
  op->setLoc(storeLoc);
}

bool retargetSyntheticOpToStoreLoc(Operation *op, Location originalLoc) {
  llvm::SmallVector<Location> candidates;

  if (isAddressComputationOp(op) || isArithmeticOrCastOp(op)) {
    collectUserVisibleStoreLocsForDestinationView(op, candidates);
    collectUserVisibleStoreLocsThroughDestinationView(op, candidates);
  }

  if (isArithmeticOrCastOp(op))
    collectUserVisibleStoreLocsThroughTensorInsert(op, candidates);

  if (std::optional<Location> storeLoc = getUniqueStoreLoc(candidates)) {
    SourceLine originalLine = getSourceLine(canonicalizeSourceLoc(originalLoc));
    SourceLine targetLine = getSourceLine(canonicalizeSourceLoc(*storeLoc));

    if (!originalLine || !targetLine || originalLine != targetLine)
      return false;

    markRetargetedSyntheticOp(op, originalLoc, *storeLoc);
    return true;
  }

  return false;
}

void markOperation(Operation *op, DebugLineLocClass locClass) {
  MLIRContext *context = op->getContext();

  Location originalLoc = op->getLoc();
  Location loc = canonicalizeSourceLoc(originalLoc);

  op->setAttr(kClassAttr, StringAttr::get(context, stringifyClass(locClass)));

  if (locClass == DebugLineLocClass::Synthetic) {
    if (retargetSyntheticOpToStoreLoc(op, originalLoc))
      return;

    preserveOrigin(op, originalLoc);

    op->setLoc(makeSyntheticDebugLoc(op, originalLoc));
    return;
  }

  if (loc != originalLoc)
    op->setLoc(loc);
}

void collectSemanticAnchors(Block &block,
                            llvm::StringMap<unsigned> &semanticAnchors) {
  for (Operation &op : block) {
    if (isContainerOp(&op) || isControlOp(&op) ||
        isExplicitlyMarkedSynthetic(&op) || !hasSourceLikeLocation(&op))
      continue;

    if (!isUserVisibleLoadAnchorOp(&op) && !isUserVisibleStoreAnchorOp(&op) &&
        !isRealMemoryOrCallOp(&op))
      continue;

    if (SourceLine line = getSourceLine(canonicalizeSourceLoc(op.getLoc())))
      ++semanticAnchors[line.getKey()];
  }
}

void relocateLoadDebugNops(Operation *root) {
  llvm::StringMap<Operation *> loadAnchors;
  llvm::StringMap<bool> ambiguousLoadLines;

  root->walk([&](Operation *op) {
    if (!isUserVisibleLoadAnchorOp(op))
      return;

    SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
    if (!line)
      return;

    std::string key = line.getKey();
    auto [it, inserted] = loadAnchors.try_emplace(key, op);
    if (!inserted && it->second != op)
      ambiguousLoadLines[key] = true;
  });

  struct Relocation {
    Operation *nop;
    Operation *anchor;
  };

  llvm::SmallVector<Relocation> relocations;
  llvm::StringMap<bool> scheduledLines;

  root->walk([&](Operation *op) {
    if (!isDebugNop(op))
      return;

    SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
    if (!line)
      return;

    std::string key = line.getKey();
    if (ambiguousLoadLines.lookup(key) || scheduledLines.lookup(key))
      return;

    auto anchorIt = loadAnchors.find(key);
    if (anchorIt == loadAnchors.end())
      return;

    Operation *anchor = anchorIt->second;

    // If the NOP is already immediately before the load anchor, keep it.
    if (op->getBlock() == anchor->getBlock() && op->getNextNode() == anchor) {
      scheduledLines[key] = true;
      return;
    }

    scheduledLines[key] = true;
    relocations.push_back({op, anchor});
  });

  // A debug NOP has no operands or results, so it is safe to move it across
  // blocks within the same function and place it immediately before the
  // corresponding memref.copy.
  for (const Relocation &relocation : relocations)
    relocation.nop->moveBefore(relocation.anchor);
}

void demoteRedundantStoreDebugNops(Operation *root) {
  llvm::StringMap<unsigned> storeAnchorLines;

  root->walk([&](Operation *op) {
    if (!isUserVisibleStoreAnchorOp(op))
      return;

    if (SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc())))
      ++storeAnchorLines[line.getKey()];
  });

  llvm::SmallVector<Operation *> redundantNops;
  root->walk([&](Operation *op) {
    if (!isDebugNop(op))
      return;

    SourceLine line = getSourceLine(canonicalizeSourceLoc(op->getLoc()));
    if (!line || storeAnchorLines.lookup(line.getKey()) == 0)
      return;

    redundantNops.push_back(op);
  });

  for (Operation *nop : redundantNops)
    markOperation(nop, DebugLineLocClass::Synthetic);
}

struct NormalizeDebugLineLocationsPass
    : public triton::impl::NormalizeDebugLineLocationsBase<
          NormalizeDebugLineLocationsPass> {
  void processBlock(Block &block) {
    llvm::StringMap<unsigned> semanticAnchors;
    collectSemanticAnchors(block, semanticAnchors);

    SourceLine previousLine;
    for (Operation &op : block) {
      if (isContainerOp(&op))
        continue;

      DebugLineLocClass locClass = classifyOperation(&op, semanticAnchors);
      if (isReorderedBackwardStep(&op, locClass, previousLine))
        locClass = DebugLineLocClass::Synthetic;

      markOperation(&op, locClass);

      if (locClass == DebugLineLocClass::Semantic ||
          locClass == DebugLineLocClass::Control) {
        if (SourceLine line = getSourceLine(op.getLoc()))
          previousLine = line;
      }
    }
  }

  void processRegions(Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        processBlock(block);
        for (Operation &nestedOp : block)
          processRegions(&nestedOp);
      }
    }
  }

  void runOnOperation() override {
    Operation *root = getOperation().getOperation();

    relocateLoadDebugNops(root);

    processRegions(root);

    demoteRedundantStoreDebugNops(root);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createNormalizeDebugLineLocationsPass() {
  return std::make_unique<NormalizeDebugLineLocationsPass>();
}
