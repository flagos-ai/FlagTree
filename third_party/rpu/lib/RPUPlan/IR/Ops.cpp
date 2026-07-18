#include "RPUPlan/IR/Dialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::rpu::plan;

namespace {

bool isSupportedPattern(StringRef pattern) {
  return llvm::StringSwitch<bool>(pattern)
      .Case("add", true)
      .Case("gemm", true)
      .Case("softmax", true)
      .Case("convkxk", true)
      .Case("resnet_block", true)
      .Case("resnet50_bottleneck", true)
      .Default(false);
}

IntegerAttr getIntegerField(DictionaryAttr dict, StringRef name) {
  if (!dict)
    return nullptr;
  return dyn_cast_or_null<IntegerAttr>(dict.get(name));
}

StringAttr getStringField(DictionaryAttr dict, StringRef name) {
  if (!dict)
    return nullptr;
  return dyn_cast_or_null<StringAttr>(dict.get(name));
}

LogicalResult requirePositiveShape(KernelOp op, DictionaryAttr shape,
                                   StringRef name) {
  IntegerAttr value = getIntegerField(shape, name);
  if (!value || value.getInt() <= 0)
    return op.emitOpError("shape field ")
           << name << " must be a positive integer";
  return success();
}

LogicalResult requireArg(KernelOp op, DictionaryAttr args, StringRef name,
                         int64_t paramCount) {
  IntegerAttr value = getIntegerField(args, name);
  if (!value)
    return op.emitOpError("requires args field ") << name;
  if (value.getInt() < 0 || value.getInt() >= paramCount)
    return op.emitOpError("argument index ")
           << value.getInt() << " is outside signature parameter range";
  return success();
}

LogicalResult verifyRequiredShapes(KernelOp op, StringRef pattern,
                                   DictionaryAttr shape) {
  if (pattern == "add") {
    if (failed(requirePositiveShape(op, shape, "n")) ||
        failed(requirePositiveShape(op, shape, "logical_n")))
      return failure();
    return success();
  }
  if (pattern == "gemm") {
    if (failed(requirePositiveShape(op, shape, "m")) ||
        failed(requirePositiveShape(op, shape, "n")) ||
        failed(requirePositiveShape(op, shape, "k")))
      return failure();
    return success();
  }
  if (pattern == "softmax")
    return requirePositiveShape(op, shape, "n");
  if (pattern == "convkxk") {
    for (StringRef name :
         {"m", "kernel_size", "in_channels", "out_channels", "input_width"})
      if (failed(requirePositiveShape(op, shape, name)))
        return failure();
    return success();
  }
  if (pattern == "resnet_block") {
    for (StringRef name : {"m", "channels", "hidden"})
      if (failed(requirePositiveShape(op, shape, name)))
        return failure();
    return success();
  }
  if (pattern == "resnet50_bottleneck") {
    for (StringRef name :
         {"m", "kernel_size", "channels", "bottleneck", "input_width"})
      if (failed(requirePositiveShape(op, shape, name)))
        return failure();
    return success();
  }
  return success();
}

LogicalResult verifyRequiredArgs(KernelOp op, StringRef pattern,
                                 DictionaryAttr args, int64_t paramCount) {
  if (pattern == "add" || pattern == "gemm") {
    for (StringRef name : {"out", "lhs", "rhs"})
      if (failed(requireArg(op, args, name, paramCount)))
        return failure();
    return success();
  }
  if (pattern == "softmax") {
    for (StringRef name : {"out", "input"})
      if (failed(requireArg(op, args, name, paramCount)))
        return failure();
    return success();
  }
  if (pattern == "convkxk") {
    for (StringRef name : {"out", "input", "weight"})
      if (failed(requireArg(op, args, name, paramCount)))
        return failure();
    return success();
  }
  if (pattern == "resnet_block") {
    for (StringRef name : {"out", "x", "w1", "w2"})
      if (failed(requireArg(op, args, name, paramCount)))
        return failure();
    return success();
  }
  if (pattern == "resnet50_bottleneck") {
    for (StringRef name : {"out", "input", "w1", "w2", "w3"})
      if (failed(requireArg(op, args, name, paramCount)))
        return failure();
    return success();
  }
  return success();
}

} // namespace

LogicalResult KernelOp::verify() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module || getOperation()->getParentOp() != module.getOperation())
    return emitOpError("must be a top-level operation in builtin.module");

  if (getVersion() != 1)
    return emitOpError("requires version 1");

  if (getSymName().empty())
    return emitOpError("requires non-empty sym_name");

  if (getKernelName().empty())
    return emitOpError("requires non-empty kernel_name");

  StringRef sourceName = getSourceFuncAttr().getValue();
  if (getSymName() == sourceName)
    return emitOpError("sym_name must not equal source_func");

  auto source = module.lookupSymbol<triton::FuncOp>(sourceName);
  if (!source)
    return emitOpError("source_func must reference an existing tt.func");

  std::optional<StringRef> visibility = source.getSymVisibility();
  if (visibility && *visibility != "public")
    return emitOpError("source_func must reference a public tt.func");

  StringRef pattern = getPattern();
  if (!isSupportedPattern(pattern))
    return emitOpError("unsupported pattern ") << pattern;

  DictionaryAttr signature = getSignature();
  if (!signature)
    return emitOpError("requires signature dictionary");

  StringAttr returnType = getStringField(signature, "return_type");
  if (!returnType || returnType.getValue() != "void")
    return emitOpError("signature.return_type must be void");

  auto params = dyn_cast_or_null<ArrayAttr>(signature.get("params"));
  if (!params)
    return emitOpError("requires signature.params array");

  for (auto [index, paramAttr] : llvm::enumerate(params)) {
    auto param = dyn_cast<DictionaryAttr>(paramAttr);
    if (!param)
      return emitOpError("signature.params entries must be dictionaries");

    IntegerAttr paramIndex = getIntegerField(param, "index");
    if (!paramIndex || paramIndex.getInt() != static_cast<int64_t>(index))
      return emitOpError("signature.params indices must be contiguous");

    StringAttr kind = getStringField(param, "kind");
    if (!kind || kind.getValue() != "ptr")
      return emitOpError("signature.params kind must be ptr");

    StringAttr elementType = getStringField(param, "element_type");
    if (!elementType || elementType.getValue() != "f16")
      return emitOpError("signature.params element_type must be f16");
  }

  DictionaryAttr args = getArgs();
  if (!args)
    return emitOpError("requires args dictionary");
  if (failed(verifyRequiredArgs(*this, pattern, args, params.size())))
    return failure();

  DictionaryAttr shape = getShape();
  if (!shape)
    return emitOpError("requires shape dictionary");
  if (failed(verifyRequiredShapes(*this, pattern, shape)))
    return failure();

  DictionaryAttr mask = getMask();
  if (!mask)
    return emitOpError("requires mask dictionary");
  if (!isa_and_nonnull<BoolAttr>(mask.get("masked")))
    return emitOpError("requires mask.masked boolean");

  DictionaryAttr layout = getLayout();
  if (!layout)
    return emitOpError("requires layout dictionary");
  if (!layout.get("memory") && !layout.get("access") && !layout.get("order") &&
      !layout.get("tile") && !layout.get("window"))
    return emitOpError(
        "layout requires stable memory/access/order/tile/window fields");

  for (Attribute feature : getRequiredDslFeatures()) {
    if (!isa<StringAttr>(feature))
      return emitOpError("required_dsl_features entries must be strings");
  }

  if (!getEmission())
    return emitOpError("requires emission dictionary");

  return success();
}

#define GET_OP_CLASSES
#include "RPUPlan/IR/RPUPlanOps.cpp.inc"
