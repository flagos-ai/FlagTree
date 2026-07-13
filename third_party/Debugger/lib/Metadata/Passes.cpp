#include "Debugger/Metadata/Passes.h"

#include "Debugger/IR/Dialect.h"
#include "Debugger/Metadata/TrackedOpTable.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#if __has_include("mlir/Interfaces/FunctionInterfaces.h")
#include "mlir/Interfaces/FunctionInterfaces.h"
#else
#include "mlir/IR/FunctionInterfaces.h"
#endif
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace flagtree {
namespace debugger {

namespace {

constexpr StringLiteral kAttrScopeId = "flagtree.debug.scope_id";
constexpr StringLiteral kAttrOpId = "flagtree.debug.op_id";
constexpr StringLiteral kAttrIsMemoryOp = "flagtree.debug.is_memory_op";
constexpr StringLiteral kAttrRecordLevel = "flagtree.debug.record_level";
constexpr StringLiteral kAttrAddrLevel = "flagtree.debug.addr_level";
constexpr StringLiteral kAttrFallbackAddrLevel = "debug_addr_level";
constexpr StringLiteral kAttrOpCategory = "flagtree.debug.op_category";
constexpr StringLiteral kAttrTrackedTableJson =
    "flagtree.debug.tracked_table_json";
constexpr StringLiteral kAttrMetadataJson = "flagtree.debug.metadata_json";
constexpr StringLiteral kAttrKernelId = "flagtree.debug.kernel_id";
constexpr StringLiteral kAttrKernelIdSeed = "flagtree.debug.kernel_id_seed";
constexpr StringLiteral kAttrScopeCount = "flagtree.debug.scope_count";
constexpr StringLiteral kAttrTrackedOpCount = "flagtree.debug.tracked_op_count";
constexpr StringLiteral kAttrStatementId = "flagtree.debug.statement_id";
constexpr StringLiteral kAttrStatementResultName =
    "flagtree.debug.statement_result_name";
constexpr StringLiteral kAttrOperandCaptureMap =
    "flagtree.debug.operand_capture_map";

std::string stringify(Type type) {
  if (!type)
    return "";
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return os.str();
}

std::string stringify(Location loc) {
  std::string text;
  llvm::raw_string_ostream os(text);
  loc.print(os);
  return os.str();
}

std::string stringify(Attribute attr) {
  if (!attr)
    return "";
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  return os.str();
}

uint32_t fnv1a32(StringRef text) {
  uint32_t hash = 2166136261u;
  for (char ch : text) {
    hash ^= static_cast<uint8_t>(ch);
    hash *= 16777619u;
  }
  return hash == 0 ? 1 : hash;
}

int hexDigit(char ch) {
  if (ch >= '0' && ch <= '9')
    return ch - '0';
  if (ch >= 'a' && ch <= 'f')
    return ch - 'a' + 10;
  if (ch >= 'A' && ch <= 'F')
    return ch - 'A' + 10;
  return -1;
}

bool parseFirstEightHexDigits(StringRef text, uint32_t &value) {
  if (text.size() < 8)
    return false;

  uint32_t parsed = 0;
  for (char ch : text.take_front(8)) {
    int digit = hexDigit(ch);
    if (digit < 0)
      return false;
    parsed = (parsed << 4) | static_cast<uint32_t>(digit);
  }
  value = parsed == 0 ? 1 : parsed;
  return true;
}

uint32_t kernelIdFromSeed(StringRef seed) {
  uint32_t parsed = 0;
  if (parseFirstEightHexDigits(seed, parsed))
    return parsed;
  return fnv1a32(seed);
}

bool isMemoryLikeOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.contains("load") || name.contains("store") ||
         name.contains("atomic") || name.contains("async_copy") ||
         name.contains("async_tma_copy") || name == "memref.copy" ||
         name == "tt.experimental_tensormap_create" ||
         name == "tt.experimental_tensormap_fenceproxy_acquire";
}

bool isTerminatorLike(Operation *op) {
  return op->hasTrait<OpTrait::IsTerminator>() ||
         op->getName().getStringRef().contains("return") ||
         op->getName().getStringRef().contains("yield");
}

bool isCallLikeOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.contains(".call") || name.contains("Call") ||
         name == "gpu.launch_func";
}

bool isTritonCombinerOpName(StringRef name) {
  return name == "tt.reduce" || name == "tt.scan";
}

bool isInsideTritonCombinerRegion(Operation *op) {
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (isTritonCombinerOpName(parent->getName().getStringRef()))
      return true;
  }
  return false;
}

bool isTritonTensorPointerLikeType(Type type) {
  if (!type)
    return false;
  if (triton::isTensorPointerType(type))
    return true;
  if (auto shaped = dyn_cast<ShapedType>(type))
    return triton::isTensorPointerType(shaped.getElementType());
  return false;
}

bool opUsesTritonTensorPointerType(Operation *op) {
  for (Type type : op->getOperandTypes())
    if (isTritonTensorPointerLikeType(type))
      return true;
  for (Type type : op->getResultTypes())
    if (isTritonTensorPointerLikeType(type))
      return true;
  return false;
}

std::optional<StringRef> calledCalleeName(Operation *op) {
  if (!isCallLikeOp(op))
    return std::nullopt;
  if (auto callee = op->getAttrOfType<FlatSymbolRefAttr>("callee"))
    return callee.getValue();
  if (auto callee = op->getAttrOfType<SymbolRefAttr>("callee"))
    return callee.getRootReference().getValue();
  return std::nullopt;
}

FunctionOpInterface parentFunction(Operation *op) {
  for (Operation *parent = op; parent; parent = parent->getParentOp()) {
    if (auto func = dyn_cast<FunctionOpInterface>(parent))
      return func;
  }
  return {};
}

llvm::StringSet<> collectCalledFunctions(ModuleOp module) {
  llvm::StringSet<> calledFunctions;
  module.walk([&](Operation *op) {
    if (std::optional<StringRef> callee = calledCalleeName(op))
      calledFunctions.insert(*callee);
  });
  return calledFunctions;
}

bool isInsideCalledFunction(Operation *op,
                            const llvm::StringSet<> &calledFunctions) {
  FunctionOpInterface func = parentFunction(op);
  return func && calledFunctions.contains(func.getName());
}

void clearDebugTrackingAttrs(Operation *op) {
  op->removeAttr(kAttrScopeId);
  op->removeAttr(kAttrOpId);
  op->removeAttr(kAttrIsMemoryOp);
  op->removeAttr(kAttrRecordLevel);
  op->removeAttr(kAttrAddrLevel);
  op->removeAttr(kAttrOpCategory);
  op->removeAttr(kAttrStatementId);
  op->removeAttr(kAttrStatementResultName);
  op->removeAttr(kAttrOperandCaptureMap);
  op->removeAttr("scope_id");
  op->removeAttr("op_id");
}

bool shouldTrack(Operation *op) {
  if (isa<CollectBeginOp, CollectEndOp>(op))
    return false;
  if (isa<ModuleOp>(op) || dyn_cast<FunctionOpInterface>(op))
    return false;
  if (isInsideTritonCombinerRegion(op))
    return false;
  if (isTerminatorLike(op))
    return false;
  // The debugger ABI is not call-graph aware yet.  Tracking tt.call/func.call
  // would require threading any debug hidden argument through both callee
  // signatures and all callsites, so skip calls until that support exists.
  if (isCallLikeOp(op))
    return false;
  return op->getNumResults() > 0 || isMemoryLikeOp(op);
}

uint32_t getIntegerAttr(Operation *op, StringRef name, uint32_t fallback = 0) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return fallback;
  int64_t value = attr.getInt();
  return value < 0 ? fallback : static_cast<uint32_t>(value);
}

uint32_t getUnsignedIntegerAttr(Operation *op, StringRef name,
                                uint32_t fallback = 0) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return fallback;

  const llvm::APInt &value = attr.getValue();
  if (value.getActiveBits() > 32)
    return fallback;
  uint64_t extended = value.getZExtValue();
  return extended > UINT32_MAX ? fallback : static_cast<uint32_t>(extended);
}

std::string getStringAttr(Operation *op, StringRef name,
                          StringRef fallback = "") {
  auto attr = op->getAttrOfType<StringAttr>(name);
  return attr ? attr.getValue().str() : fallback.str();
}

std::string classifyOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "memref.copy")
    return "copy";
  if (name.contains("async_copy") || name.contains("async_tma_copy"))
    return "async_copy";
  if (name == "tt.experimental_tensormap_create")
    return "tensormap_create";
  if (name == "tt.experimental_tensormap_fenceproxy_acquire")
    return "tensormap_fenceproxy_acquire";
  if (name.contains("load"))
    return "load";
  if (name.contains("store"))
    return "store";
  if (name.contains("atomic"))
    return "atomic";
  if (name.contains("dot"))
    return "dot";
  if (name.contains("reduce"))
    return "reduce";
  if (name.contains("reshape") || name.contains("transpose") ||
      name.contains("broadcast") || name.contains("convert_layout"))
    return "layout_convert";
  if (name.contains("scf.") || name.contains("cf."))
    return "control";
  return "compute";
}

std::string accessTypeForOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name == "tt.atomic_rmw")
    return "atomic_rmw";
  if (name == "tt.atomic_cas")
    return "atomic_cas";
  if (name == "tt.descriptor_load")
    return "descriptor_load";
  if (name == "tt.descriptor_store")
    return "descriptor_store";
  if (name == "tt.experimental_descriptor_load")
    return "experimental_descriptor_load";
  if (name == "tt.experimental_descriptor_store")
    return "experimental_descriptor_store";
  if (name == "tt.experimental_tensormap_create")
    return "tensormap_create";
  if (name == "tt.experimental_tensormap_fenceproxy_acquire")
    return "tensormap_fenceproxy_acquire";
  if (name == "memref.copy")
    return "memref_copy";
  if (name == "tptr.load")
    return "tptr_load";
  if (name == "tptr.store")
    return "tptr_store";
  if (name == "tts.load")
    return "tts_load";
  if (name == "tts.store")
    return "tts_store";
  if (name == "tts.atomic_rmw")
    return "tts_atomic_rmw";
  if (name == "tts.indexed_atomic_rmw")
    return "tts_indexed_atomic_rmw";
  if (name == "tts.atomic_cas")
    return "tts_atomic_cas";
  if (name.contains("async_tma_copy"))
    return "async_tma_copy";
  if (name.contains("async_copy"))
    return "async_copy";
  return classifyOp(op);
}

std::string roleForOp(Operation *op) {
  std::string category = classifyOp(op);
  if (category == "load")
    return "load";
  if (category == "store")
    return "store";
  if (category == "atomic")
    return "atomic";
  if (category == "async_copy")
    return "async_copy";
  if (category == "copy")
    return "copy";
  if (category == "tensormap_create")
    return "tensormap_create";
  if (category == "tensormap_fenceproxy_acquire")
    return "tensormap_fenceproxy_acquire";
  return "";
}

Type elementType(Type type) {
  if (!type)
    return {};
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (auto tensor = dyn_cast<UnrankedTensorType>(type))
    type = tensor.getElementType();
  if (auto vector = dyn_cast<VectorType>(type))
    type = vector.getElementType();
  if (auto memref = dyn_cast<MemRefType>(type))
    type = memref.getElementType();
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    return ptr.getPointeeType();
  return type;
}

bool isTritonPointerLikeType(Type type) {
  if (!type)
    return false;
  if (isa<triton::PointerType>(type))
    return true;
  if (auto shaped = dyn_cast<ShapedType>(type))
    return isa<triton::PointerType>(shaped.getElementType());
  return false;
}

bool isPtrDialectPointerType(Type type) {
  return type && type.getDialect().getNamespace() == "ptr";
}

bool isAddressCaptureType(Type type) {
  return isTritonPointerLikeType(type) || isa<MemRefType>(type) ||
         isPtrDialectPointerType(type);
}

uint32_t elementBits(Type type) {
  if (!type)
    return 0;
  Type elem = elementType(type);
  if (auto floatType = dyn_cast<FloatType>(elem))
    return floatType.getWidth();
  if (auto intType = dyn_cast<IntegerType>(elem))
    return intType.getWidth();
  if (isa<IndexType>(elem))
    return 64;
  return 0;
}

std::string shapeString(Type type) {
  if (!type)
    return "[]";
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    std::string result = "[";
    llvm::raw_string_ostream os(result);
    llvm::interleaveComma(tensor.getShape(), os);
    os << "]";
    return os.str();
  }
  if (auto vector = dyn_cast<VectorType>(type)) {
    std::string result = "[";
    llvm::raw_string_ostream os(result);
    llvm::interleaveComma(vector.getShape(), os);
    os << "]";
    return os.str();
  }
  if (auto memref = dyn_cast<MemRefType>(type)) {
    std::string result = "[";
    llvm::raw_string_ostream os(result);
    llvm::interleaveComma(memref.getShape(), os);
    os << "]";
    return os.str();
  }
  return "[]";
}

uint32_t rankOf(Type type) {
  if (!type)
    return 0;
  if (auto shaped = dyn_cast<ShapedType>(type); shaped && shaped.hasRank())
    return static_cast<uint32_t>(shaped.getRank());
  return 0;
}

uint32_t vecWidth(Type type) {
  if (!type)
    return 0;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    int64_t count = tensor.getNumElements();
    return count > 0 && count <= UINT32_MAX ? static_cast<uint32_t>(count) : 0;
  }
  if (auto vector = dyn_cast<VectorType>(type)) {
    int64_t count = vector.getNumElements();
    return count > 0 && count <= UINT32_MAX ? static_cast<uint32_t>(count) : 0;
  }
  return 1;
}

std::string valueKind(Type type) {
  if (!type)
    return "unknown";
  if (isa<triton::PointerType>(type))
    return "pointer";
  if (isPtrDialectPointerType(type))
    return "pointer";
  if (isa<RankedTensorType, UnrankedTensorType>(type))
    return "tensor";
  if (isa<VectorType>(type))
    return "tensor";
  if (isa<MemRefType>(type))
    return "pointer";
  return "scalar";
}

std::string layoutOf(Type type) {
  if (!type)
    return "unknown";
  Attribute encoding;
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    encoding = tensor.getEncoding();
  if (encoding)
    return stringify(encoding);
  if (isa<MemRefType>(type))
    return "memref";
  if (isa<VectorType>(type))
    return "vector";
  return "unknown";
}

std::string addressSpaceName(int addressSpace) {
  if (addressSpace == 1)
    return "global";
  if (addressSpace == 0)
    return "generic";
  return "addrspace(" + std::to_string(addressSpace) + ")";
}

std::string addrSpaceOfMemorySpace(Attribute memorySpace) {
  if (!memorySpace)
    return "memory";
  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace))
    return addressSpaceName(static_cast<int>(intAttr.getInt()));
  return stringify(memorySpace);
}

std::string addrSpaceOf(Type type) {
  if (!type)
    return "";
  if (auto memref = dyn_cast<MemRefType>(type))
    return addrSpaceOfMemorySpace(memref.getMemorySpace());
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    return addressSpaceName(ptr.getAddressSpace());
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (auto ptr = dyn_cast<triton::PointerType>(shaped.getElementType()))
      return addressSpaceName(ptr.getAddressSpace());
  }
  return "";
}

StaticValueInfo valueInfo(Type type) {
  StaticValueInfo info;
  info.valueKind = valueKind(type);
  info.dtype = stringify(type);
  info.elementDtype = stringify(elementType(type));
  info.shape = shapeString(type);
  info.stride = "unknown";
  info.layout = layoutOf(type);
  info.encoding = info.layout == "unknown" ? "" : info.layout;
  info.addrSpace = addrSpaceOf(type);
  info.rank = rankOf(type);
  info.elementBits = elementBits(type);
  info.vecWidth = vecWidth(type);
  return info;
}

Type primaryResultOrValueType(Operation *op) {
  if (op->getNumResults() > 0)
    return op->getResult(0).getType();
  if (op->getNumOperands() > 0)
    return op->getOperand(0).getType();
  return {};
}

bool isDescriptorLoadStoreOpName(StringRef name) {
  return name == "tt.descriptor_load" || name == "tt.descriptor_store";
}

bool isExperimentalDescriptorLoadStoreOpName(StringRef name) {
  return name == "tt.experimental_descriptor_load" ||
         name == "tt.experimental_descriptor_store";
}

bool isAsyncCopyOpName(StringRef name) {
  return name.contains("async_copy") || name.contains("async_tma_copy");
}

bool isCopyOpName(StringRef name) { return name == "memref.copy"; }

Value pointerIfTritonPointerLike(Value value) {
  if (!value || !isAddressCaptureType(value.getType()))
    return {};
  return value;
}

Value descriptorBasePointer(Value descriptor) {
  Operation *def = descriptor ? descriptor.getDefiningOp() : nullptr;
  if (!def || def->getName().getStringRef() != "tt.make_tensor_descriptor" ||
      def->getNumOperands() == 0)
    return {};
  return pointerIfTritonPointerLike(def->getOperand(0));
}

struct MemoryAddressTarget {
  Value pointer;
  uint32_t operandIndex = 0;
  StringRef role;
};

void addMemoryAddressTarget(SmallVectorImpl<MemoryAddressTarget> &targets,
                            Operation *op, uint32_t operandIndex,
                            StringRef role) {
  if (operandIndex >= op->getNumOperands())
    return;
  if (Value pointer = pointerIfTritonPointerLike(op->getOperand(operandIndex)))
    targets.push_back(MemoryAddressTarget{pointer, operandIndex, role});
}

SmallVector<MemoryAddressTarget> memoryAddressTargets(Operation *op) {
  SmallVector<MemoryAddressTarget> targets;
  StringRef name = op->getName().getStringRef();
  if (name == "memref.store" && op->getNumOperands() > 1) {
    addMemoryAddressTarget(targets, op, 1, "ptr");
    return targets;
  }
  if (isCopyOpName(name)) {
    addMemoryAddressTarget(targets, op, 0, "src");
    addMemoryAddressTarget(targets, op, 1, "dst");
    return targets;
  }
  if (name == "tptr.store" && op->getNumOperands() > 1) {
    addMemoryAddressTarget(targets, op, 1, "ptr");
    return targets;
  }
  if (name == "tt.experimental_tensormap_create") {
    addMemoryAddressTarget(targets, op, 0, "desc_ptr");
    addMemoryAddressTarget(targets, op, 1, "global_address");
    return targets;
  }
  if (name == "tt.experimental_tensormap_fenceproxy_acquire") {
    addMemoryAddressTarget(targets, op, 0, "desc_ptr");
    return targets;
  }
  if (isDescriptorLoadStoreOpName(name) && op->getNumOperands() > 0) {
    if (Value pointer = descriptorBasePointer(op->getOperand(0)))
      targets.push_back(MemoryAddressTarget{pointer, 0, "descriptor_base"});
    return targets;
  }
  if (isExperimentalDescriptorLoadStoreOpName(name) &&
      op->getNumOperands() > 0) {
    addMemoryAddressTarget(targets, op, 0, "desc_ptr");
    return targets;
  }
  if (isAsyncCopyOpName(name)) {
    addMemoryAddressTarget(targets, op, 0, "src");
    addMemoryAddressTarget(targets, op, 1, "dst");
    return targets;
  }
  if (op->getNumOperands() > 0)
    addMemoryAddressTarget(targets, op, 0, "ptr");
  return targets;
}

Value memoryPointerOperand(Operation *op) {
  SmallVector<MemoryAddressTarget> targets = memoryAddressTargets(op);
  if (!targets.empty())
    return targets.front().pointer;
  return {};
}

Value storedValueOperand(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  if (!opName.contains("store"))
    return {};
  if (opName == "memref.store")
    return op->getNumOperands() > 0 ? op->getOperand(0) : Value();
  if (opName == "tptr.store")
    return op->getNumOperands() > 0 ? op->getOperand(0) : Value();
  return op->getNumOperands() > 1 ? op->getOperand(1) : Value();
}

Type pointerPointeeType(Type type) {
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    return ptr.getPointeeType();
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (auto ptr = dyn_cast<triton::PointerType>(shaped.getElementType()))
      return ptr.getPointeeType();
  }
  if (auto memref = dyn_cast<MemRefType>(type))
    return memref.getElementType();
  return {};
}

Type memoryAccessValueType(Operation *op) {
  if (Value stored = storedValueOperand(op))
    return stored.getType();
  if (op->getNumResults() > 0)
    return op->getResult(0).getType();
  if (Value ptr = memoryPointerOperand(op))
    return pointerPointeeType(ptr.getType());
  return primaryResultOrValueType(op);
}

uint32_t accessBytes(Operation *op) {
  Type type = memoryAccessValueType(op);
  uint32_t bits = elementBits(type);
  return bits == 0 ? 0 : std::max<uint32_t>(1, bits / 8);
}

bool hasMask(Operation *op) {
  for (Value operand : op->getOperands()) {
    Type type = operand.getType();
    if (auto intType = dyn_cast<IntegerType>(elementType(type));
        intType && intType.getWidth() == 1)
      return true;
  }
  return false;
}

std::string maskDtype(Operation *op) {
  for (Value operand : op->getOperands()) {
    Type type = operand.getType();
    if (auto intType = dyn_cast<IntegerType>(elementType(type));
        intType && intType.getWidth() == 1)
      return stringify(type);
  }
  return "";
}

std::string operandRole(Operation *op, uint32_t index) {
  Value operand = op->getOperand(index);
  StringRef opName = op->getName().getStringRef();
  if (isDescriptorLoadStoreOpName(opName) && index == 0)
    return "descriptor";
  for (const MemoryAddressTarget &target : memoryAddressTargets(op)) {
    if (target.operandIndex == index)
      return target.role.str();
  }
  if (isMemoryLikeOp(op) && operand == memoryPointerOperand(op))
    return "ptr";
  if (operand == storedValueOperand(op))
    return "value";
  if (index == 0)
    return "lhs";
  if (index == 1)
    return "rhs";
  return "operand" + std::to_string(index);
}

bool isKernelArgument(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return isa<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
  return false;
}

bool isConstant(Value value) {
  Operation *def = value.getDefiningOp();
  return def && def->getName().getStringRef().contains("constant");
}

std::string constantRepr(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || !def->getName().getStringRef().contains("constant"))
    return "";
  if (Attribute attr = def->getAttr("value"))
    return stringify(attr);
  return "";
}

bool isIdentifierStart(char ch) {
  return std::isalpha(static_cast<unsigned char>(ch)) || ch == '_';
}

bool isIdentifierContinue(char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

bool isIgnoredStatementIdentifier(StringRef ident) {
  return ident == "tl" || ident == "triton" || ident == "math" ||
         ident == "constexpr" || ident == "True" || ident == "False" ||
         ident == "None";
}

std::string compactSourceExpr(StringRef expr) {
  expr = expr.trim();
  std::string compact;
  llvm::raw_string_ostream os(compact);
  bool previousSpace = false;
  for (char ch : expr) {
    bool isSpace = std::isspace(static_cast<unsigned char>(ch));
    if (isSpace) {
      previousSpace = true;
      continue;
    }
    if (previousSpace && !compact.empty())
      os << ' ';
    os << ch;
    previousSpace = false;
  }
  return os.str();
}

bool isSimpleIdentifierExpr(StringRef expr) {
  expr = expr.trim();
  if (expr.empty() || !isIdentifierStart(expr.front()))
    return false;
  for (char ch : expr.drop_front()) {
    if (!isIdentifierContinue(ch))
      return false;
  }
  return true;
}

size_t matchingParen(StringRef text, size_t open) {
  int depth = 0;
  char quote = 0;
  for (size_t i = open; i < text.size(); ++i) {
    char ch = text[i];
    if (quote != 0) {
      if (ch == quote)
        quote = 0;
      continue;
    }
    if (ch == '\'' || ch == '"') {
      quote = ch;
      continue;
    }
    if (ch == '(' || ch == '[' || ch == '{') {
      ++depth;
      continue;
    }
    if (ch == ')' || ch == ']' || ch == '}') {
      --depth;
      if (depth == 0)
        return i;
    }
  }
  return StringRef::npos;
}

std::optional<StringRef> findCallArgs(StringRef rhs, StringRef callee) {
  size_t pos = rhs.find(callee);
  if (pos == StringRef::npos)
    return std::nullopt;
  size_t open = rhs.find('(', pos + callee.size());
  if (open == StringRef::npos)
    return std::nullopt;
  size_t close = matchingParen(rhs, open);
  if (close == StringRef::npos || close <= open)
    return std::nullopt;
  return rhs.slice(open + 1, close);
}

llvm::SmallVector<StringRef> splitTopLevelArgs(StringRef args) {
  llvm::SmallVector<StringRef> result;
  size_t start = 0;
  int depth = 0;
  char quote = 0;
  for (size_t i = 0; i < args.size(); ++i) {
    char ch = args[i];
    if (quote != 0) {
      if (ch == quote)
        quote = 0;
      continue;
    }
    if (ch == '\'' || ch == '"') {
      quote = ch;
      continue;
    }
    if (ch == '(' || ch == '[' || ch == '{') {
      ++depth;
      continue;
    }
    if (ch == ')' || ch == ']' || ch == '}') {
      --depth;
      continue;
    }
    if (ch == ',' && depth == 0) {
      result.push_back(args.slice(start, i).trim());
      start = i + 1;
    }
  }
  StringRef tail = args.drop_front(start).trim();
  if (!tail.empty())
    result.push_back(tail);
  return result;
}

std::optional<size_t> findTopLevelEqual(StringRef arg) {
  int depth = 0;
  char quote = 0;
  for (size_t i = 0; i < arg.size(); ++i) {
    char ch = arg[i];
    if (quote != 0) {
      if (ch == quote)
        quote = 0;
      continue;
    }
    if (ch == '\'' || ch == '"') {
      quote = ch;
      continue;
    }
    if (ch == '(' || ch == '[' || ch == '{') {
      ++depth;
      continue;
    }
    if (ch == ')' || ch == ']' || ch == '}') {
      --depth;
      continue;
    }
    if (ch == '=' && depth == 0)
      return i;
  }
  return std::nullopt;
}

std::string inferStatementResultName(StringRef statement) {
  std::optional<size_t> eq = findTopLevelEqual(statement);
  if (!eq)
    return "";
  StringRef lhs = statement.take_front(*eq).trim();
  if (lhs.empty())
    return "";
  size_t comma = lhs.find(',');
  if (comma != StringRef::npos)
    lhs = lhs.take_front(comma).trim();
  if (lhs.starts_with("("))
    lhs = lhs.drop_front().trim();
  if (lhs.ends_with(")"))
    lhs = lhs.drop_back().trim();
  return lhs.str();
}

std::optional<StringRef> positionalArg(ArrayRef<StringRef> args,
                                       size_t positionalIndex) {
  size_t current = 0;
  for (StringRef arg : args) {
    if (findTopLevelEqual(arg))
      continue;
    if (current == positionalIndex)
      return arg;
    ++current;
  }
  return std::nullopt;
}

std::optional<StringRef> keywordArg(ArrayRef<StringRef> args, StringRef key) {
  for (StringRef arg : args) {
    std::optional<size_t> eq = findTopLevelEqual(arg);
    if (!eq)
      continue;
    if (arg.take_front(*eq).trim() == key)
      return arg.drop_front(*eq + 1).trim();
  }
  return std::nullopt;
}

std::string keywordDisplayName(StringRef key, std::optional<StringRef> value) {
  if (!value)
    return key.str();
  if (isSimpleIdentifierExpr(*value))
    return compactSourceExpr(*value);
  return key.str();
}

std::optional<llvm::SmallVector<std::string>>
inferLoadOperandNames(StringRef rhs) {
  std::optional<StringRef> body = findCallArgs(rhs, "tl.load");
  if (!body)
    body = findCallArgs(rhs, "load");
  if (!body)
    return std::nullopt;

  llvm::SmallVector<StringRef> args = splitTopLevelArgs(*body);
  llvm::SmallVector<std::string> names;
  std::optional<StringRef> ptr = positionalArg(args, 0);
  names.push_back(ptr ? compactSourceExpr(*ptr) : "ptr");

  std::optional<StringRef> mask = keywordArg(args, "mask");
  if (!mask)
    mask = positionalArg(args, 1);
  names.push_back(keywordDisplayName("mask", mask));

  std::optional<StringRef> other = keywordArg(args, "other");
  if (!other)
    other = positionalArg(args, 2);
  names.push_back(keywordDisplayName("other", other));
  return names;
}

std::optional<llvm::SmallVector<std::string>>
inferStoreOperandNames(StringRef statement) {
  std::optional<StringRef> body = findCallArgs(statement, "tl.store");
  if (!body)
    body = findCallArgs(statement, "store");
  if (!body)
    return std::nullopt;

  llvm::SmallVector<StringRef> args = splitTopLevelArgs(*body);
  llvm::SmallVector<std::string> names;

  std::optional<StringRef> ptr = keywordArg(args, "pointer");
  if (!ptr)
    ptr = positionalArg(args, 0);
  names.push_back(ptr ? compactSourceExpr(*ptr) : "ptr");

  std::optional<StringRef> value = keywordArg(args, "value");
  if (!value)
    value = positionalArg(args, 1);
  names.push_back(value ? compactSourceExpr(*value) : "value");

  std::optional<StringRef> mask = keywordArg(args, "mask");
  if (!mask)
    mask = positionalArg(args, 2);
  names.push_back(keywordDisplayName("mask", mask));
  return names;
}

llvm::SmallVector<std::string> inferStatementOperandNames(StringRef statement) {
  if (std::optional<llvm::SmallVector<std::string>> storeNames =
          inferStoreOperandNames(statement))
    return *storeNames;

  std::optional<size_t> eq = findTopLevelEqual(statement);
  StringRef rhs = eq ? statement.drop_front(*eq + 1) : statement;
  if (std::optional<llvm::SmallVector<std::string>> loadNames =
          inferLoadOperandNames(rhs))
    return *loadNames;

  llvm::SmallVector<std::string> names;
  for (size_t i = 0; i < rhs.size();) {
    if (!isIdentifierStart(rhs[i])) {
      ++i;
      continue;
    }
    size_t start = i++;
    while (i < rhs.size() && isIdentifierContinue(rhs[i]))
      ++i;
    StringRef ident = rhs.slice(start, i);
    if (isIgnoredStatementIdentifier(ident))
      continue;
    if (start > 0 && rhs[start - 1] == '.')
      continue;
    size_t next = i;
    while (next < rhs.size() &&
           std::isspace(static_cast<unsigned char>(rhs[next])))
      ++next;
    if (next < rhs.size() && rhs[next] == '=') {
      continue;
    }
    if (next < rhs.size() && rhs[next] == '(')
      continue;
    names.push_back(ident.str());
  }
  return names;
}

std::string fallbackOperandName(const OperandStaticInfo &info) {
  if (!info.operandRole.empty())
    return info.operandRole;
  return "operand" + std::to_string(info.operandIndex);
}

bool shouldCaptureOperandAtStatement(Value operand,
                                     const OperandStaticInfo &info) {
  if (!operand || info.producerOpId != 0 || info.isConstant)
    return false;
  if (isTritonPointerLikeType(operand.getType()))
    return false;
  StaticValueInfo value = valueInfo(operand.getType());
  return value.valueKind != "pointer" && value.elementBits != 0;
}

std::string
serializeOperandCaptureMap(ArrayRef<std::pair<uint32_t, uint32_t>> captures) {
  std::string text;
  llvm::raw_string_ostream os(text);
  bool first = true;
  for (const auto &capture : captures) {
    if (!first)
      os << ",";
    first = false;
    os << capture.first << ":" << capture.second;
  }
  return os.str();
}

StatementValueInfo makeResultStatementValue(const TrackedOpEntry &entry) {
  StatementValueInfo value;
  value.sourceName =
      entry.statementResultName.empty() ? "result" : entry.statementResultName;
  value.sourceRole = "result";
  value.hasOperandIndex = false;
  value.captureOpId = entry.opId;
  value.capturePolicy = "captured_current_op";
  value.value = entry.result;
  return value;
}

StatementValueInfo makeOperandStatementValue(const OperandStaticInfo &operand,
                                             StringRef sourceName,
                                             uint32_t captureOpId,
                                             StringRef capturePolicy) {
  StatementValueInfo value;
  value.sourceName = sourceName.str();
  value.sourceRole = "operand";
  value.hasOperandIndex = true;
  value.operandIndex = operand.operandIndex;
  value.producerOpId = operand.producerOpId;
  value.captureOpId = captureOpId;
  value.capturePolicy = capturePolicy.str();
  value.isConstant = operand.isConstant;
  value.constantValueRepr = operand.constantValueRepr;
  value.value = operand.value;
  return value;
}

TrackedOpEntry
makeSyntheticOperandCaptureEntry(const TrackedOpEntry &anchor,
                                 const OperandStaticInfo &operand,
                                 uint32_t captureOpId, StringRef sourceName) {
  TrackedOpEntry entry;
  entry.opId = captureOpId;
  entry.scopeId = anchor.scopeId;
  entry.resultIndex = 0;
  entry.statementId = anchor.statementId;
  entry.isSyntheticStatementCapture = true;
  entry.isMemoryOp = false;
  entry.opCategory = "statement_operand_capture";
  entry.role = "operand";
  entry.mlirOpName = "flagtree.debug.operand_capture";
  entry.sourceLoc = anchor.sourceLoc;
  entry.tritonStatement = anchor.tritonStatement;
  entry.statementResultName = sourceName.str();
  entry.inlineCallPath = anchor.inlineCallPath;
  entry.result = operand.value;
  return entry;
}

OperandStaticInfo
makeOperandInfo(Operation *op, uint32_t operandIndex,
                const llvm::DenseMap<Operation *, uint32_t> &producerOpIds) {
  Value operand = op->getOperand(operandIndex);
  OperandStaticInfo info;
  info.operandIndex = operandIndex;
  info.operandRole = operandRole(op, operandIndex);
  if (Operation *def = operand.getDefiningOp()) {
    auto it = producerOpIds.find(def);
    if (it != producerOpIds.end())
      info.producerOpId = it->second;
  }
  info.isConstant = isConstant(operand);
  info.isPredicate = false;
  if (auto intType = dyn_cast<IntegerType>(elementType(operand.getType())))
    info.isPredicate = intType.getWidth() == 1;
  info.isKernelArgument = isKernelArgument(operand);
  info.constantValueRepr = constantRepr(operand);
  info.value = valueInfo(operand.getType());
  return info;
}

TrackedOpEntry
makeEntry(Operation *op, uint32_t opId,
          const llvm::DenseMap<Operation *, uint32_t> &producerOpIds,
          uint32_t &nextOpId, Builder &builder,
          llvm::SmallVectorImpl<TrackedOpEntry> &syntheticEntries) {
  TrackedOpEntry entry;
  entry.opId = opId;
  entry.scopeId = getIntegerAttr(op, kAttrScopeId);
  entry.resultIndex = 0;
  entry.statementId = getIntegerAttr(op, kAttrStatementId, opId);
  if (entry.statementId == 0)
    entry.statementId = opId;
  entry.isMemoryOp = isMemoryLikeOp(op);
  entry.opCategory = entry.isMemoryOp ? classifyOp(op) : "";
  entry.role = entry.isMemoryOp ? roleForOp(op) : "";
  entry.mlirOpName = op->getName().getStringRef().str();
  entry.sourceLoc = stringify(op->getLoc());
  entry.tritonStatement =
      getStringAttr(op, "flagtree.debug.triton_statement", entry.mlirOpName);
  entry.statementResultName =
      getStringAttr(op, kAttrStatementResultName,
                    inferStatementResultName(entry.tritonStatement));
  entry.inlineCallPath = "";
  entry.result = valueInfo(entry.isMemoryOp ? memoryAccessValueType(op)
                                            : primaryResultOrValueType(op));
  entry.addrSpace = entry.isMemoryOp && memoryPointerOperand(op)
                        ? addrSpaceOf(memoryPointerOperand(op).getType())
                        : "";
  entry.accessType = entry.isMemoryOp ? accessTypeForOp(op) : "";
  entry.accessBytes = entry.isMemoryOp ? accessBytes(op) : 0;
  entry.alignmentRequired = entry.accessBytes;
  entry.hasMask = entry.isMemoryOp && hasMask(op);
  entry.maskDtype = entry.hasMask ? maskDtype(op) : "";
  entry.cacheModifier = getStringAttr(op, "cache", "");
  entry.evictionPolicy = getStringAttr(op, "eviction_policy", "");
  entry.isVolatile = op->hasAttr("volatile");
  entry.boundaryCheckPolicy = getStringAttr(op, "boundary_check", "");
  entry.paddingSemantics = getStringAttr(op, "padding", "");

  const bool hasSourceStatement =
      op->hasAttr("flagtree.debug.triton_statement") ||
      op->hasAttr(kAttrStatementResultName);
  llvm::SmallVector<std::string> operandNames;
  llvm::SmallVector<std::pair<uint32_t, uint32_t>> operandCaptures;

  if (hasSourceStatement)
    operandNames = inferStatementOperandNames(entry.tritonStatement);

  if (hasSourceStatement && op->getNumResults() > 0)
    entry.statementValues.push_back(makeResultStatementValue(entry));

  for (uint32_t i = 0, e = op->getNumOperands(); i < e; ++i)
    entry.operands.push_back(makeOperandInfo(op, i, producerOpIds));

  if (hasSourceStatement) {
    for (OperandStaticInfo &operand : entry.operands) {
      Value operandValue = op->getOperand(operand.operandIndex);
      std::string sourceName = operand.operandIndex < operandNames.size()
                                   ? operandNames[operand.operandIndex]
                                   : fallbackOperandName(operand);
      uint32_t captureOpId = operand.producerOpId;
      StringRef capturePolicy = "reused_producer";
      if (operand.isConstant) {
        captureOpId = 0;
        capturePolicy = "constant";
      } else if (shouldCaptureOperandAtStatement(operandValue, operand)) {
        captureOpId = nextOpId++;
        capturePolicy = "captured_at_current_statement";
        operandCaptures.push_back({operand.operandIndex, captureOpId});
        syntheticEntries.push_back(makeSyntheticOperandCaptureEntry(
            entry, operand, captureOpId, sourceName));
      } else if (captureOpId == 0) {
        capturePolicy =
            operand.isKernelArgument ? "kernel_argument" : "not_captured";
      }
      entry.statementValues.push_back(makeOperandStatementValue(
          operand, sourceName, captureOpId, capturePolicy));
    }
  }

  if (!operandCaptures.empty()) {
    op->setAttr(
        kAttrOperandCaptureMap,
        builder.getStringAttr(serializeOperandCaptureMap(operandCaptures)));
  } else {
    op->removeAttr(kAttrOperandCaptureMap);
  }

  return entry;
}

std::string entryFunctionName(ModuleOp module) {
  std::string fallback = "unknown_kernel";
  for (Operation &op : module.getBody()->getOperations()) {
    if (auto func = dyn_cast<FunctionOpInterface>(&op))
      return func.getName().str();
    if (auto name =
            op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      fallback = name.getValue().str();
  }
  return fallback;
}

std::string kernelIdSeed(ModuleOp module, StringRef kernelName) {
  std::string seed = getStringAttr(module.getOperation(), kAttrKernelIdSeed);
  if (!seed.empty())
    return seed;

  std::string target =
      getStringAttr(module.getOperation(), "flagtree.debug.target", "");
  if (!target.empty()) {
    std::string combined = kernelName.str();
    combined += ":";
    combined += target;
    return combined;
  }
  return kernelName.str();
}

} // namespace

LogicalResult assignDebugOpIdsAndMetadataWithoutPassManager(ModuleOp module) {
  Builder builder(module.getContext());
  TrackedOpTable table;
  llvm::DenseMap<Operation *, uint32_t> producerOpIds;
  llvm::SmallVector<Operation *> trackedOps;
  llvm::StringSet<> calledFunctions = collectCalledFunctions(module);
  uint32_t nextOpId = 1;

  module.walk([&](Operation *op) {
    uint32_t explicitOpId = getIntegerAttr(op, kAttrOpId);
    if (explicitOpId != 0)
      nextOpId = std::max(nextOpId, explicitOpId + 1);
  });

  module.walk([&](Operation *op) {
    if (isCallLikeOp(op) || isInsideCalledFunction(op, calledFunctions) ||
        isInsideTritonCombinerRegion(op)) {
      clearDebugTrackingAttrs(op);
      return;
    }

    if (!op->hasAttr(kAttrScopeId) || !shouldTrack(op))
      return;

    uint32_t opId = getIntegerAttr(op, kAttrOpId);
    if (opId == 0)
      opId = nextOpId++;
    else
      nextOpId = std::max(nextOpId, opId + 1);

    op->setAttr(kAttrOpId, builder.getI32IntegerAttr(opId));
    bool memory = isMemoryLikeOp(op);
    op->setAttr(kAttrIsMemoryOp, builder.getBoolAttr(memory));
    if (memory)
      op->setAttr(kAttrOpCategory, builder.getStringAttr(classifyOp(op)));
    else
      op->removeAttr(kAttrOpCategory);

    producerOpIds[op] = opId;
    trackedOps.push_back(op);
  });

  uint32_t nextSyntheticOpId = nextOpId;
  for (Operation *op : trackedOps) {
    uint32_t opId = getIntegerAttr(op, kAttrOpId);
    llvm::SmallVector<TrackedOpEntry, 1> syntheticEntries;
    table.push_back(makeEntry(op, opId, producerOpIds, nextSyntheticOpId,
                              builder, syntheticEntries));
    for (TrackedOpEntry &synthetic : syntheticEntries)
      table.push_back(std::move(synthetic));
  }

  KernelDebugMetadata metadata;
  metadata.kernelName = entryFunctionName(module);
  metadata.backendName =
      getStringAttr(module.getOperation(), "flagtree.debug.backend", "");
  metadata.targetName =
      getStringAttr(module.getOperation(), "flagtree.debug.target", "");
  metadata.debugKernelId =
      kernelIdFromSeed(kernelIdSeed(module, metadata.kernelName));
  metadata.scopeCount = getIntegerAttr(module.getOperation(), kAttrScopeCount);
  metadata.trackedOpCount = static_cast<uint32_t>(table.size());
  metadata.trackedOps = table;

  std::string trackedJson = serializeTrackedOpTableToJson(table);
  std::string metadataJson = serializeKernelDebugMetadataToJson(metadata);

  module->setAttr(kAttrKernelId, builder.getI64IntegerAttr(static_cast<int64_t>(
                                     metadata.debugKernelId)));
  module->setAttr(kAttrTrackedOpCount,
                  builder.getI32IntegerAttr(metadata.trackedOpCount));
  module->setAttr(kAttrTrackedTableJson, builder.getStringAttr(trackedJson));
  module->setAttr(kAttrMetadataJson, builder.getStringAttr(metadataJson));
  return success();
}

void eraseDebugCollectMarkers(ModuleOp module) {
  llvm::SmallVector<Operation *> toErase;
  module.walk([&](Operation *op) {
    if (isa<CollectBeginOp, CollectEndOp>(op))
      toErase.push_back(op);
  });
  for (Operation *op : toErase)
    op->erase();
}

bool hasTritonTensorPointerTypes(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (opUsesTritonTensorPointerType(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

LogicalResult assignDebugCollectScopeIdsWithoutErase(ModuleOp module) {
  Builder builder(module.getContext());
  llvm::SmallVector<Operation *> orderedOps;
  module.walk([&](Operation *op) { orderedOps.push_back(op); });
  llvm::StringSet<> calledFunctions = collectCalledFunctions(module);

  int32_t nextScopeId = 1;
  uint32_t activeScopeId = 0;
  uint32_t activeLevel = 1;
  uint32_t moduleAddrLevel = getIntegerAttr(
      module.getOperation(), kAttrAddrLevel,
      getIntegerAttr(module.getOperation(), kAttrFallbackAddrLevel, 0));
  moduleAddrLevel = std::min<uint32_t>(moduleAddrLevel, 2);
  uint32_t activeAddrLevel = moduleAddrLevel;
  CollectBeginOp activeBegin;
  module->setAttr(kAttrAddrLevel, builder.getI32IntegerAttr(moduleAddrLevel));

  for (Operation *op : orderedOps) {
    if (isInsideCalledFunction(op, calledFunctions)) {
      clearDebugTrackingAttrs(op);
      continue;
    }

    if (auto begin = dyn_cast<CollectBeginOp>(op)) {
      if (activeBegin) {
        begin.emitError() << "illegal nested debug collect region (Phase 1 "
                             "forbids nesting)";
        return failure();
      }
      activeScopeId = static_cast<uint32_t>(nextScopeId++);
      activeLevel = getIntegerAttr(begin.getOperation(), "level", 1);
      activeAddrLevel = std::min<uint32_t>(
          getIntegerAttr(begin.getOperation(), "addr_level", moduleAddrLevel),
          2);
      auto scopeAttr = builder.getI32IntegerAttr(activeScopeId);
      auto addrLevelAttr = builder.getI32IntegerAttr(activeAddrLevel);
      begin->setAttr("scope_id", scopeAttr);
      begin->setAttr(kAttrScopeId, scopeAttr);
      begin->setAttr(kAttrRecordLevel, builder.getI32IntegerAttr(activeLevel));
      begin->setAttr(kAttrAddrLevel, addrLevelAttr);
      activeBegin = begin;
      continue;
    }

    if (auto end = dyn_cast<CollectEndOp>(op)) {
      if (!activeBegin) {
        end.emitError() << "debug collect_end without matching collect_begin";
        return failure();
      }
      auto scopeAttr = activeBegin->getAttr("scope_id");
      if (scopeAttr) {
        end->setAttr("scope_id", scopeAttr);
        end->setAttr(kAttrScopeId, scopeAttr);
      }
      activeBegin = CollectBeginOp();
      activeScopeId = 0;
      activeLevel = 1;
      activeAddrLevel = moduleAddrLevel;
      continue;
    }

    if (isCallLikeOp(op)) {
      clearDebugTrackingAttrs(op);
      continue;
    }

    if (activeScopeId != 0 && !isa<ModuleOp>(op) &&
        !dyn_cast<FunctionOpInterface>(op)) {
      op->setAttr(kAttrScopeId, builder.getI32IntegerAttr(activeScopeId));
      op->setAttr(kAttrRecordLevel, builder.getI32IntegerAttr(activeLevel));
      op->setAttr(kAttrAddrLevel, builder.getI32IntegerAttr(activeAddrLevel));
    }
  }

  if (activeBegin) {
    activeBegin.emitError()
        << "debug collect_begin without matching collect_end";
    return failure();
  }

  module->setAttr(kAttrScopeCount, builder.getI32IntegerAttr(nextScopeId - 1));
  return success();
}

namespace {

struct ResolveDebugScopePass
    : public PassWrapper<ResolveDebugScopePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveDebugScopePass);

  void runOnOperation() override {
    auto module = getOperation();
    if (failed(assignDebugCollectScopeIdsWithoutErase(module))) {
      signalPassFailure();
      return;
    }
    llvm::SmallVector<Operation *> toErase;
    module.walk([&](Operation *op) {
      if (isa<CollectBeginOp, CollectEndOp>(op))
        toErase.push_back(op);
    });
    for (Operation *op : toErase)
      op->erase();
  }

  StringRef getArgument() const final { return "flagtree-resolve-debug-scope"; }
  StringRef getDescription() const final {
    return "Resolve debug collect scopes and validate begin/end pairs";
  }
};

struct AssignOpIdPass
    : public PassWrapper<AssignOpIdPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignOpIdPass);

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(assignDebugOpIdsAndMetadataWithoutPassManager(module)))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "flagtree-assign-debug-op-id"; }
  StringRef getDescription() const final {
    return "Assign stable debug op ids and collect static metadata";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createResolveDebugScopePass() {
  return std::make_unique<ResolveDebugScopePass>();
}

std::unique_ptr<mlir::Pass> createAssignOpIdPass() {
  return std::make_unique<AssignOpIdPass>();
}

bool hasDebugCollectMarkers(mlir::Operation *op) {
  if (!op)
    return false;

  bool found = false;
  op->walk([&](mlir::Operation *nestedOp) {
    auto name = nestedOp->getName().getStringRef();
    if (name == "flagtree_debug.collect_begin" ||
        name == "flagtree_debug.collect_end") {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

LogicalResult insertDefaultDebugCollectMarkers(ModuleOp module, int32_t level,
                                               int32_t addrLevel) {
  Builder moduleBuilder(module.getContext());
  IntegerAttr levelAttr = moduleBuilder.getI32IntegerAttr(level);
  IntegerAttr addrAttr;
  if (addrLevel >= 0)
    addrAttr = moduleBuilder.getI32IntegerAttr(std::min<int32_t>(addrLevel, 2));

  llvm::SmallVector<Operation *> existingMarkers;
  module.walk([&](Operation *op) {
    if (isa<CollectBeginOp, CollectEndOp>(op))
      existingMarkers.push_back(op);
  });
  for (Operation *op : existingMarkers)
    op->erase();

  llvm::StringSet<> calledFunctions = collectCalledFunctions(module);
  bool inserted = false;
  for (Operation &op : module.getBody()->getOperations()) {
    auto func = dyn_cast<FunctionOpInterface>(&op);
    if (!func || calledFunctions.contains(func.getName()))
      continue;
    if (func.isExternal())
      continue;

    Region *body = func.getCallableRegion();
    if (!body || body->empty())
      continue;

    Block &entryBlock = body->front();
    if (entryBlock.empty())
      continue;

    OpBuilder entryBuilder(module.getContext());
    entryBuilder.setInsertionPoint(&entryBlock.front());
    entryBuilder.create<CollectBeginOp>(op.getLoc(), levelAttr, addrAttr,
                                        /*scope_id=*/nullptr);

    llvm::SmallVector<Operation *> returns;
    for (Block &block : body->getBlocks()) {
      if (Operation *terminator = block.getTerminator())
        returns.push_back(terminator);
    }
    if (returns.empty())
      return failure();
    for (Operation *returnOp : returns) {
      OpBuilder returnBuilder(returnOp);
      returnBuilder.create<CollectEndOp>(returnOp->getLoc(),
                                         /*scope_id=*/nullptr);
    }
    inserted = true;
  }

  return inserted ? success() : failure();
}

std::string getDebugTrackedOpTableJson(ModuleOp module) {
  return getStringAttr(module.getOperation(), kAttrTrackedTableJson, "[]");
}

std::string getDebugKernelMetadataJson(ModuleOp module) {
  return getStringAttr(module.getOperation(), kAttrMetadataJson, "{}");
}

uint32_t getDebugKernelId(ModuleOp module) {
  return getUnsignedIntegerAttr(module.getOperation(), kAttrKernelId);
}

void setDebugKernelIdSeed(ModuleOp module, StringRef seed) {
  Builder builder(module.getContext());
  module->setAttr(kAttrKernelIdSeed, builder.getStringAttr(seed));
}

void registerFlagTreeDebuggerMetadataPasses() {
  PassRegistration<ResolveDebugScopePass>();
  PassRegistration<AssignOpIdPass>();
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
