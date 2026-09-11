#include "Gluon/GluonLayoutPlaceholders.h"

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <iterator>

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_dialect = ::mlir::triton::gluon;

namespace mlir::triton::gpu::metax::gluon {

Attribute unwrapNoVerifyEncoding(Attribute attr) {
  if (auto noVerify =
          dyn_cast_or_null<gluon_dialect::NoVerifyEncodingAttr>(attr))
    return noVerify.getLayout();
  return attr;
}

Type cloneTypeWithEncoding(Type type, Attribute encoding) {
  encoding = unwrapNoVerifyEncoding(encoding);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensorTy.getShape(),
                                 tensorTy.getElementType(), encoding);
  if (auto memDescTy = dyn_cast<ttg::MemDescType>(type))
    return ttg::MemDescType::get(
        memDescTy.getShape(), memDescTy.getElementType(), encoding,
        memDescTy.getMemorySpace(), memDescTy.getMutableMemory(),
        memDescTy.getAllocShape());
  return type;
}

static Type unwrapNoVerifyType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return cloneTypeWithEncoding(
        tensorTy, unwrapNoVerifyEncoding(tensorTy.getEncoding()));
  if (auto memDescTy = dyn_cast<ttg::MemDescType>(type))
    return cloneTypeWithEncoding(
        memDescTy, unwrapNoVerifyEncoding(memDescTy.getEncoding()));
  return type;
}

unsigned unwrapNoVerifyEncodings(Operation *root) {
  unsigned replacements = 0;
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](gluon_dialect::NoVerifyEncodingAttr encoding) -> Attribute {
        ++replacements;
        return encoding.getLayout();
      });
  replacer.addReplacement(
      [](RankedTensorType type) -> std::optional<Type> {
        auto encoding = dyn_cast_or_null<gluon_dialect::NoVerifyEncodingAttr>(
            type.getEncoding());
        if (!encoding)
          return std::nullopt;
        return type.cloneWithEncoding(encoding.getLayout());
      });
  replacer.addReplacement([](ttg::MemDescType type) -> std::optional<Type> {
    auto encoding = dyn_cast_or_null<gluon_dialect::NoVerifyEncodingAttr>(
        type.getEncoding());
    if (!encoding)
      return std::nullopt;
    return cloneTypeWithEncoding(type, encoding.getLayout());
  });
  replacer.addReplacement(
      [](DenseElementsAttr elements) -> std::optional<Attribute> {
        auto type = dyn_cast<RankedTensorType>(elements.getType());
        if (!type)
          return std::nullopt;
        auto encoding = dyn_cast_or_null<gluon_dialect::NoVerifyEncodingAttr>(
            type.getEncoding());
        if (!encoding)
          return std::nullopt;
        return elements.reshape(type.cloneWithEncoding(encoding.getLayout()));
      });
  replacer.addReplacement(
      [](DenseResourceElementsAttr elements) -> std::optional<Attribute> {
        auto type = dyn_cast<RankedTensorType>(elements.getType());
        if (!type)
          return std::nullopt;
        auto encoding = dyn_cast_or_null<gluon_dialect::NoVerifyEncodingAttr>(
            type.getEncoding());
        if (!encoding)
          return std::nullopt;
        auto newType = type.cloneWithEncoding(encoding.getLayout());
        return DenseResourceElementsAttr::get(newType,
                                              elements.getRawHandle());
      });
  replacer.addReplacement(
      [](SparseElementsAttr elements) -> std::optional<Attribute> {
        auto type = dyn_cast<RankedTensorType>(elements.getType());
        if (!type)
          return std::nullopt;
        auto encoding = dyn_cast_or_null<gluon_dialect::NoVerifyEncodingAttr>(
            type.getEncoding());
        if (!encoding)
          return std::nullopt;
        auto newType = type.cloneWithEncoding(encoding.getLayout());
        return SparseElementsAttr::get(newType, elements.getIndices(),
                                       elements.getValues());
      });
  replacer.recursivelyReplaceElementsIn(
      root, /*replaceAttrs=*/true, /*replaceLocs=*/false,
      /*replaceTypes=*/true);
  return replacements;
}

void synchronizeFunctionTypeAfterNoVerifyUnwrap(tt::FuncOp func) {
  SmallVector<Type> inputs;
  if (func.getBody().empty()) {
    llvm::transform(func.getFunctionType().getInputs(),
                    std::back_inserter(inputs), unwrapNoVerifyType);
  } else {
    llvm::append_range(inputs, func.getBody().front().getArgumentTypes());
  }
  SmallVector<Type> results;
  llvm::transform(func.getFunctionType().getResults(),
                  std::back_inserter(results), unwrapNoVerifyType);
  func.setFunctionType(FunctionType::get(func.getContext(), inputs, results));
}

bool hasAutoEncoding(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  return tensorTy &&
         isa<gluon_dialect::AutoEncodingAttr>(tensorTy.getEncoding());
}

static bool containsDeferredEncoding(Attribute attr, bool autoEncoding);
static bool containsDeferredEncoding(Type type, bool autoEncoding);

static bool containsDeferredEncoding(Attribute attr, bool autoEncoding) {
  if (!attr)
    return false;
  if (autoEncoding ? isa<gluon_dialect::AutoEncodingAttr>(attr)
                   : isa<gluon_dialect::NoVerifyEncodingAttr>(attr))
    return true;
  bool found = false;
  attr.walkImmediateSubElements(
      [&](Attribute nested) {
        found |= containsDeferredEncoding(nested, autoEncoding);
      },
      [&](Type nested) {
        found |= containsDeferredEncoding(nested, autoEncoding);
      });
  return found;
}

static bool containsDeferredEncoding(Type type, bool autoEncoding) {
  if (!type)
    return false;
  Attribute encoding;
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    encoding = tensor.getEncoding();
  else if (auto memdesc = dyn_cast<ttg::MemDescType>(type))
    encoding = memdesc.getEncoding();
  if (containsDeferredEncoding(encoding, autoEncoding))
    return true;
  bool found = false;
  type.walkImmediateSubElements(
      [&](Attribute nested) {
        found |= containsDeferredEncoding(nested, autoEncoding);
      },
      [&](Type nested) {
        found |= containsDeferredEncoding(nested, autoEncoding);
      });
  return found;
}

bool containsAutoEncoding(Type type) {
  return containsDeferredEncoding(type, true);
}

bool containsNoVerifyEncoding(Attribute attr) {
  return containsDeferredEncoding(attr, false);
}

bool containsNoVerifyEncoding(Type type) {
  return containsDeferredEncoding(type, false);
}

LogicalResult verifyNoResidualCalls(ModuleOp module, StringRef boundary) {
  WalkResult result = module.walk([&](CallOpInterface call) {
    call.emitError()
        << "unresolved call-like operation reached " << boundary
        << "; the Gluon inliner must eliminate every call before MetaX "
           "layout analysis because layout constraints are intraprocedural";
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

static LogicalResult verifyNoResidualEncoding(ModuleOp module,
                                              StringRef boundary,
                                              bool autoEncoding) {
  auto containsType = [&](Type type) {
    return containsDeferredEncoding(type, autoEncoding);
  };
  auto containsAttr = [&](Attribute attr) {
    return containsDeferredEncoding(attr, autoEncoding);
  };
  auto report = [&](Operation *op, const Twine &owner) {
    op->emitError() << "unresolved gluon."
                    << (autoEncoding ? "auto_encoding" : "no_verify_encoding")
                    << " on " << owner << " reached " << boundary;
  };
  WalkResult result = module.walk([&](Operation *op) {
    for (auto [index, type] : llvm::enumerate(op->getResultTypes())) {
      if (!containsType(type))
        continue;
      report(op, "result type #" + Twine(index));
      return WalkResult::interrupt();
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (auto [index, type] : llvm::enumerate(block.getArgumentTypes()))
          if (containsType(type)) {
            report(op, "block argument type #" + Twine(index));
            return WalkResult::interrupt();
          }
    for (NamedAttribute attr : op->getAttrs())
      if (containsAttr(attr.getValue())) {
        report(op, Twine("operation attribute '") +
                       attr.getName().strref() + "'");
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult verifyNoResidualNoVerifyEncodings(ModuleOp module,
                                                StringRef boundary) {
  return verifyNoResidualEncoding(module, boundary, false);
}

LogicalResult verifyNoResidualAutoEncodings(ModuleOp module,
                                            StringRef boundary) {
  return verifyNoResidualEncoding(module, boundary, true);
}

} // namespace mlir::triton::gpu::metax::gluon
