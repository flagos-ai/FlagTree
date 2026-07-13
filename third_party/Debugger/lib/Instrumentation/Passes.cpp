#include "Debugger/Instrumentation/Passes.h"

#include "Debugger/IR/Dialect.h"
#include "Debugger/Instrumentation/Collectors.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#if __has_include("mlir/Interfaces/FunctionInterfaces.h")
#include "mlir/Interfaces/FunctionInterfaces.h"
#else
#include "mlir/IR/FunctionInterfaces.h"
#endif
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

constexpr StringLiteral kAttrOpId = "flagtree.debug.op_id";
constexpr StringLiteral kAttrFallbackOpId = "op_id";
constexpr StringLiteral kAttrScopeId = "flagtree.debug.scope_id";
constexpr StringLiteral kAttrFallbackScopeId = "scope_id";
constexpr StringLiteral kAttrIsMemoryOp = "flagtree.debug.is_memory_op";
constexpr StringLiteral kAttrFallbackIsMemoryOp = "is_memory_op";
constexpr StringLiteral kAttrRecordLevel = "flagtree.debug.record_level";
constexpr StringLiteral kAttrFallbackRecordLevel = "debug_record_level";
constexpr StringLiteral kAttrAddrLevel = "flagtree.debug.addr_level";
constexpr StringLiteral kAttrFallbackAddrLevel = "debug_addr_level";
constexpr StringLiteral kAttrInstrumented = "flagtree.debug.instrumented";
constexpr StringLiteral kAttrRecordKinds = "flagtree.debug.record_kinds";
constexpr StringLiteral kAttrSummaryCollectors =
    "flagtree.debug.summary_collectors";
constexpr StringLiteral kAttrMemoryEventKind =
    "flagtree.debug.memory_event_kind";
constexpr StringLiteral kAttrFullValueRef = "flagtree.debug.full_value_ref";
constexpr StringLiteral kAttrHiddenArg = "flagtree.debug.hidden_arg";
constexpr StringLiteral kAttrHiddenArgAbiEnabled =
    "flagtree.debug.enable_hidden_arg_abi";
constexpr StringLiteral kAttrTimelineEnabled =
    "flagtree.debug.timeline_enabled";
constexpr StringLiteral kAttrTimelineOnly = "flagtree.debug.timeline_only";
constexpr StringLiteral kAttrHiddenArgIndex = "flagtree.debug.hidden_arg_index";
constexpr StringLiteral kAttrHiddenArgType = "flagtree.debug.hidden_arg_type";
constexpr StringLiteral kAttrLogicalInstanceFormula =
    "flagtree.debug.logical_instance_id_formula";
constexpr StringLiteral kAttrInstrumentationInserted =
    "flagtree.debug.instrumentation_inserted";

// Level-1 dynamic collection must stay light enough for CANN9's UB limits.
// Large tensor summaries and address reductions can make otherwise valid
// kernels fail in BiShengIR with local-buffer overflow; keep those cases
// metadata-only unless a higher collection level is explicitly requested.
constexpr uint64_t kLevel1AuxSummaryElementLimit = 256;
constexpr uint64_t kLevel1LoadSummaryElementLimit = 256;
constexpr uint64_t kLevel1AddressSummaryElementLimit = 256;
constexpr StringLiteral kAttrRecordIndex = "flagtree.debug.record_index";
constexpr StringLiteral kAttrRecordsPerInstance =
    "flagtree.debug.records_per_instance";
constexpr StringLiteral kAttrRecordSize = "flagtree.debug.record_size";
constexpr StringLiteral kAttrRecordLayout = "flagtree.debug.record_layout";
constexpr StringLiteral kAttrRecordPlan = "flagtree.debug.record_plan";
constexpr StringLiteral kAttrOperandCaptureMap =
    "flagtree.debug.operand_capture_map";
constexpr StringLiteral kAttrFullDumpPlan = "flagtree.debug.full_dump_plan";
constexpr StringLiteral kAttrFullDumpPayloadBytesPerInstance =
    "flagtree.debug.full_dump_payload_bytes_per_instance";
constexpr StringLiteral kAttrFullDumpKind = "flagtree.debug.full_dump_kind";
constexpr StringLiteral kAttrFullDumpSource = "flagtree.debug.full_dump_source";
constexpr StringLiteral kAttrFullDumpArtifactDtype =
    "flagtree.debug.full_dump_artifact_dtype";
constexpr StringLiteral kAttrFullDumpElementCount =
    "flagtree.debug.full_dump_element_count";
constexpr StringLiteral kAttrFullDumpElementBytes =
    "flagtree.debug.full_dump_element_bytes";
constexpr StringLiteral kAttrFullDumpPayloadOffset =
    "flagtree.debug.full_dump_payload_offset";
constexpr StringLiteral kAttrFullDumpPayloadLength =
    "flagtree.debug.full_dump_payload_length";
constexpr StringLiteral kRecordLayoutDeterministicCompactV1 =
    "deterministic_compact_v1";
constexpr StringLiteral kRecordSummaryOpName = "flagtree_debug.record_summary";
constexpr StringLiteral kRecordSummaryBundleOpName =
    "flagtree_debug.record_summary_bundle";
constexpr StringLiteral kRecordMemoryEventOpName =
    "flagtree_debug.record_memory_event";
constexpr StringLiteral kCaptureMemoryAddressOpName =
    "flagtree_debug.capture_memory_address";
constexpr StringLiteral kRecordFullValueRefOpName =
    "flagtree_debug.record_full_value_ref";
constexpr StringLiteral kRecordTimelineOpName =
    "flagtree_debug.record_timeline";
constexpr StringLiteral kHiddenArgName = "__debug_ctrl_ptr";
constexpr StringLiteral kLogicalInstanceFormula =
    "pid0 + pid1 * num_programs0 + pid2 * num_programs0 * num_programs1";
constexpr StringLiteral kAttrDeviceLowered = "flagtree.debug.device_lowered";
constexpr StringLiteral kMemrefStoreOpName = "memref.store";
constexpr StringLiteral kMemrefReinterpretCastOpName =
    "memref.reinterpret_cast";
constexpr StringLiteral kMemrefCastOpName = "memref.cast";
constexpr StringLiteral kMemrefSubviewOpName = "memref.subview";
constexpr StringLiteral kTensorInsertOpName = "tensor.insert";
constexpr StringLiteral kLinalgFillOpName = "linalg.fill";
constexpr StringLiteral kMaterializeInDestinationOpName =
    "bufferization.materialize_in_destination";

// Protocol.h ABI constants.  Keep these local and literal so the generated
// device IR does not depend on host-only headers.
constexpr int64_t kRingHeaderBytes = 32;
constexpr int64_t kLegacyRecordBytes = 32;
constexpr int64_t kRecordBytes = 64;
constexpr int64_t kHeaderWriteIdxOffset = 0;
constexpr int64_t kHeaderCapacityOffset = 4;
constexpr int64_t kHeaderOverflowCountOffset = 8;
constexpr int64_t kHeaderFlagsOffset = 12;
constexpr int64_t kHeaderPayloadOffsetOffset = 20;
constexpr uint32_t kRecordKindSummary = 1;
constexpr uint32_t kRecordKindMemoryEvent = 2;
constexpr uint32_t kRecordKindFullValue = 3;
constexpr uint32_t kRecordKindSummaryCountBundleU64 = 4;
constexpr uint32_t kRecordKindSummaryValueBundleF32 = 5;
constexpr uint32_t kRecordKindTimeline = 6;
constexpr uint32_t kCollectorNanCount = 1;
constexpr uint32_t kCollectorInfCount = 2;
constexpr uint32_t kCollectorMeanFinite = 3;
constexpr uint32_t kCollectorMinFinite = 4;
constexpr uint32_t kCollectorMaxFinite = 5;
constexpr uint32_t kCollectorElementCount = 6;
constexpr uint32_t kCollectorZeroCount = 7;
constexpr uint32_t kCollectorL2Norm = 8;
constexpr uint32_t kResultTypeU64 = 1;
constexpr uint32_t kResultTypeF32 = 2;
constexpr uint32_t kMemoryEventLastAlignedAddr = 1;
constexpr uint32_t kMemoryEventBaseAlignedAddr = 2;
constexpr uint32_t kMemoryEventFirstAddr = 3;
constexpr uint32_t kMemoryEventLastAddr = 4;
constexpr uint32_t kMemoryEventMinAddr = 5;
constexpr uint32_t kMemoryEventMaxAddr = 6;
constexpr uint32_t kMemoryEventActiveLaneCount = 7;
constexpr uint32_t kMemoryEventAddressSpanBytes = 8;
constexpr uint32_t kRingFlagOverflow = 1;
constexpr int64_t kLegacyRecordWords = kLegacyRecordBytes / 4;
constexpr int64_t kRecordWords = kRecordBytes / 4;
constexpr int32_t kMaxPointerTraceDepth = 32;

constexpr StringLiteral kEventLastAlignedAddr = "LAST_ALIGNED_ADDR";
constexpr StringLiteral kEventBaseAlignedAddr = "BASE_ALIGNED_ADDR";
constexpr StringLiteral kEventFirstAddr = "FIRST_ADDR";
constexpr StringLiteral kEventLastAddr = "LAST_ADDR";
constexpr StringLiteral kEventMinAddr = "MIN_ADDR";
constexpr StringLiteral kEventMaxAddr = "MAX_ADDR";
constexpr StringLiteral kEventActiveLaneCount = "ACTIVE_LANE_COUNT";
constexpr StringLiteral kEventAddressSpanBytes = "ADDRESS_SPAN_BYTES";
constexpr StringLiteral kEventAddressSummary = "ADDRESS_SUMMARY";
constexpr StringLiteral kFullDumpKindValue = "value";
constexpr StringLiteral kFullDumpKindMemoryAddress = "memory_address";
constexpr StringLiteral kFullDumpSourceResult = "result";
constexpr StringLiteral kFullDumpSourceStoreValue = "store_value";
constexpr StringLiteral kFullDumpSourceAddress = "address";
constexpr StringLiteral kFullDumpSourceStatementOperand = "statement_operand";

IntegerAttr getIntAttr(Operation *op, StringRef primary, StringRef fallback) {
  if (!op) {
    return {};
  }
  if (auto attr = op->getAttrOfType<IntegerAttr>(primary)) {
    return attr;
  }
  return op->getAttrOfType<IntegerAttr>(fallback);
}

BoolAttr getBoolAttr(Operation *op, StringRef primary, StringRef fallback) {
  if (!op) {
    return {};
  }
  if (auto attr = op->getAttrOfType<BoolAttr>(primary)) {
    return attr;
  }
  return op->getAttrOfType<BoolAttr>(fallback);
}

bool isMemoryLikeOp(Operation *op) {
  if (op->getName().getDialectNamespace() == "flagtree_debug")
    return false;

  if (auto attr = getBoolAttr(op, kAttrIsMemoryOp, kAttrFallbackIsMemoryOp)) {
    return attr.getValue();
  }

  StringRef opName = op->getName().getStringRef();
  return opName.contains("load") || opName.contains("store") ||
         opName.contains("atomic") || opName.contains("async_copy") ||
         opName.contains("async_tma_copy") || opName == "memref.copy" ||
         opName == "tt.experimental_tensormap_create" ||
         opName == "tt.experimental_tensormap_fenceproxy_acquire";
}

bool shouldEmitTimeline(Operation *op) {
  StringRef dialect = op->getName().getDialectNamespace();
  StringRef name = op->getName().getStringRef();
  if (dialect == "flagtree_debug")
    return false;
  if (name == "arith.constant")
    return false;
  return true;
}

bool isCallLikeOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName.contains(".call") || opName.contains("Call") ||
         opName == "gpu.launch_func";
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

FunctionOpInterface parentFunction(Operation *op) {
  for (Operation *parent = op; parent; parent = parent->getParentOp()) {
    if (auto func = dyn_cast<FunctionOpInterface>(parent))
      return func;
  }
  return {};
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

int32_t getI32AttrValue(IntegerAttr attr, int32_t fallback = 0) {
  if (!attr)
    return fallback;
  return static_cast<int32_t>(attr.getInt());
}

bool isPointerLikeType(Type type);
Value storedValueOperand(Operation *op);

struct MemoryAddressTarget {
  Value pointer;
  uint32_t operandIndex = 0;
  StringRef role;
};

bool isDescriptorLoadStoreOpName(StringRef opName) {
  return opName == "tt.descriptor_load" || opName == "tt.descriptor_store";
}

bool isExperimentalDescriptorLoadStoreOpName(StringRef opName) {
  return opName == "tt.experimental_descriptor_load" ||
         opName == "tt.experimental_descriptor_store";
}

bool isTensormapOpName(StringRef opName) {
  return opName == "tt.experimental_tensormap_create" ||
         opName == "tt.experimental_tensormap_fenceproxy_acquire";
}

bool isAsyncCopyOpName(StringRef opName) {
  return opName.contains("async_copy") || opName.contains("async_tma_copy");
}

bool isCopyOpName(StringRef opName) { return opName == "memref.copy"; }

bool isPtrDialectPointerType(Type type) {
  return type && type.getDialect().getNamespace() == "ptr";
}

bool isMemRefPointerType(Type type) { return isa<MemRefType>(type); }

bool isAddressCaptureType(Type type) {
  return isPointerLikeType(type) || isMemRefPointerType(type) ||
         isPtrDialectPointerType(type);
}

Value pointerIfPointerLike(Value value) {
  if (!value || !isAddressCaptureType(value.getType()))
    return {};
  return value;
}

Value descriptorBasePointer(Value descriptor) {
  Operation *def = descriptor ? descriptor.getDefiningOp() : nullptr;
  if (!def || def->getName().getStringRef() != "tt.make_tensor_descriptor" ||
      def->getNumOperands() == 0)
    return {};
  return pointerIfPointerLike(def->getOperand(0));
}

void addMemoryAddressTarget(SmallVectorImpl<MemoryAddressTarget> &targets,
                            Operation *op, uint32_t operandIndex,
                            StringRef role) {
  if (operandIndex >= op->getNumOperands())
    return;
  if (Value pointer = pointerIfPointerLike(op->getOperand(operandIndex)))
    targets.push_back(MemoryAddressTarget{pointer, operandIndex, role});
}

SmallVector<MemoryAddressTarget> memoryAddressTargets(Operation *op) {
  SmallVector<MemoryAddressTarget> targets;
  StringRef opName = op->getName().getStringRef();
  if (opName == "memref.store" && op->getNumOperands() > 1) {
    addMemoryAddressTarget(targets, op, 1, "ptr");
    return targets;
  }
  if (isCopyOpName(opName)) {
    addMemoryAddressTarget(targets, op, 0, "src");
    addMemoryAddressTarget(targets, op, 1, "dst");
    return targets;
  }
  if (opName == "tptr.store" && op->getNumOperands() > 1) {
    addMemoryAddressTarget(targets, op, 1, "ptr");
    return targets;
  }
  if (opName == "tt.experimental_tensormap_create") {
    addMemoryAddressTarget(targets, op, 0, "desc_ptr");
    addMemoryAddressTarget(targets, op, 1, "global_address");
    return targets;
  }
  if (opName == "tt.experimental_tensormap_fenceproxy_acquire") {
    addMemoryAddressTarget(targets, op, 0, "desc_ptr");
    return targets;
  }
  if (isDescriptorLoadStoreOpName(opName) && op->getNumOperands() > 0) {
    if (Value pointer = descriptorBasePointer(op->getOperand(0)))
      targets.push_back(MemoryAddressTarget{pointer, 0, "descriptor_base"});
    return targets;
  }
  if (isExperimentalDescriptorLoadStoreOpName(opName) &&
      op->getNumOperands() > 0) {
    addMemoryAddressTarget(targets, op, 0, "desc_ptr");
    return targets;
  }
  if (isAsyncCopyOpName(opName)) {
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

bool isTensorPointerLikeType(Type type) {
  return isa<RankedTensorType>(type) && isPointerLikeType(type);
}

bool isTritonTensorPointerType(Type type) {
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
    if (isTritonTensorPointerType(type))
      return true;
  for (Type type : op->getResultTypes())
    if (isTritonTensorPointerType(type))
      return true;
  return false;
}

bool hasDynamicShapedType(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && (!shaped.hasRank() || !shaped.hasStaticShape());
}

bool opUsesDynamicShapedType(Operation *op) {
  for (Type type : op->getOperandTypes())
    if (hasDynamicShapedType(type))
      return true;
  for (Type type : op->getResultTypes())
    if (hasDynamicShapedType(type))
      return true;
  return false;
}

bool isSafeEffectingDebugSource(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName == "tt.load" || opName == "memref.load" ||
         opName == "tt.reduce" || opName == "tt.scan";
}

bool hasUnsupportedSideEffectsForDynamicDebug(Operation *op) {
  if (isSafeEffectingDebugSource(op))
    return false;
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  iface.getEffects(effects);
  return llvm::any_of(
      effects,
      [](const SideEffects::EffectInstance<MemoryEffects::Effect> &effect) {
        return !isa<MemoryEffects::Read>(effect.getEffect());
      });
}

bool isSafeDynamicDebugValue(Operation *op) {
  // Keep static metadata for all tracked operations, but only attach
  // device-side observers to values whose use is known to be backend-safe.
  // CANN9's current path is particularly sensitive to Triton block/tensor
  // pointers
  // (!tt.ptr<tensor<...>>) and mutation/alias-heavy ops: a debug side-use can
  // change legality or scheduling even when the original kernel is valid.
  return !opUsesTritonTensorPointerType(op) && !opUsesDynamicShapedType(op) &&
         !hasUnsupportedSideEffectsForDynamicDebug(op);
}

bool isSafeDynamicDebugMemoryEvent(Operation *op) {
  // Store/atomic ops have side effects by definition, but pointer-only memory
  // event capture is safe for ordinary scalar/tensor-of-scalar-pointer forms.
  // Triton block pointers (!tt.ptr<tensor<...>>) stay metadata-only until a
  // backend-specific address-summary lowering is implemented for them.
  return !opUsesTritonTensorPointerType(op) && !opUsesDynamicShapedType(op);
}

StringRef memoryEventKindForPointer(Value pointer) {
  if (pointer && isTensorPointerLikeType(pointer.getType()))
    return kEventBaseAlignedAddr;
  return kEventLastAlignedAddr;
}

bool isAddressSummaryEventKind(StringRef eventKind) {
  return eventKind == kEventFirstAddr || eventKind == kEventLastAddr ||
         eventKind == kEventMinAddr || eventKind == kEventMaxAddr ||
         eventKind == kEventActiveLaneCount ||
         eventKind == kEventAddressSpanBytes;
}

std::optional<unsigned> memoryPointerOperandIndex(Operation *op) {
  SmallVector<MemoryAddressTarget> targets = memoryAddressTargets(op);
  if (!targets.empty())
    return targets.front().operandIndex;
  return std::nullopt;
}

uint64_t getStaticElementCount(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return 0;
    uint64_t count = 1;
    for (int64_t dim : shaped.getShape())
      count *= static_cast<uint64_t>(dim);
    return count;
  }
  return 1;
}

std::optional<uint32_t> collectorIdForName(StringRef name) {
  return llvm::StringSwitch<std::optional<uint32_t>>(name)
      .Case("nan_count", kCollectorNanCount)
      .Case("inf_count", kCollectorInfCount)
      .Case("zero_count", kCollectorZeroCount)
      .Case("mean_finite", kCollectorMeanFinite)
      .Case("min_finite", kCollectorMinFinite)
      .Case("max_finite", kCollectorMaxFinite)
      .Case("l2_norm", kCollectorL2Norm)
      .Case("element_count", kCollectorElementCount)
      .Default(std::nullopt);
}

Type getElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

bool isFloatValueType(Type type) {
  return isa<FloatType>(getElementType(type));
}

struct FullDumpSpec {
  std::string kind;
  std::string source;
  std::string artifactDtype;
  uint64_t elementCount = 0;
  uint32_t elementBytes = 0;
};

uint64_t alignTo(uint64_t value, uint64_t alignment) {
  if (alignment == 0)
    return value;
  uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + (alignment - remainder);
}

std::optional<FullDumpSpec> getFullDumpSpecForValue(Type type,
                                                    StringRef source) {
  if (isPointerLikeType(type))
    return std::nullopt;
  uint64_t elementCount = getStaticElementCount(type);
  if (elementCount == 0)
    return std::nullopt;

  Type elementType = getElementType(type);
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    unsigned width = floatType.getWidth();
    if (width <= 32)
      return FullDumpSpec{kFullDumpKindValue.str(), source.str(), "float32",
                          elementCount, 4};
    if (width == 64)
      return FullDumpSpec{kFullDumpKindValue.str(), source.str(), "float64",
                          elementCount, 8};
    return std::nullopt;
  }

  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    if (intType.getWidth() <= 64)
      return FullDumpSpec{kFullDumpKindValue.str(), source.str(), "int64",
                          elementCount, 8};
  }
  return std::nullopt;
}

std::optional<FullDumpSpec> getFullDumpSpecForObservedValue(Type type,
                                                            StringRef source) {
  if (isPointerLikeType(type)) {
    uint64_t elementCount = getStaticElementCount(type);
    if (elementCount == 0)
      return std::nullopt;
    return FullDumpSpec{kFullDumpKindMemoryAddress.str(), source.str(),
                        "uint64", elementCount, 8};
  }
  return getFullDumpSpecForValue(type, source);
}

std::optional<FullDumpSpec> getFullDumpSpecForMemoryAddress(Value pointer) {
  if (!pointer || !isPointerLikeType(pointer.getType()))
    return std::nullopt;
  uint64_t elementCount = getStaticElementCount(pointer.getType());
  if (elementCount == 0)
    return std::nullopt;
  return FullDumpSpec{kFullDumpKindMemoryAddress.str(),
                      kFullDumpSourceAddress.str(), "uint64", elementCount, 8};
}

Value observedFullDumpValue(Operation *op, StringRef &source) {
  if (op->getNumResults() > 0) {
    source = kFullDumpSourceResult;
    return op->getResult(0);
  }
  if (Value stored = storedValueOperand(op)) {
    source = kFullDumpSourceStoreValue;
    return stored;
  }
  return {};
}

llvm::SmallVector<std::pair<uint32_t, uint32_t>>
parseOperandCaptureMap(StringRef text) {
  llvm::SmallVector<std::pair<uint32_t, uint32_t>> captures;
  while (!text.empty()) {
    StringRef item;
    std::tie(item, text) = text.split(',');
    item = item.trim();
    if (item.empty())
      continue;
    StringRef lhs;
    StringRef rhs;
    std::tie(lhs, rhs) = item.split(':');
    uint32_t operandIndex = 0;
    uint32_t captureOpId = 0;
    if (lhs.trim().getAsInteger(10, operandIndex) ||
        rhs.trim().getAsInteger(10, captureOpId) || captureOpId == 0)
      continue;
    captures.push_back({operandIndex, captureOpId});
  }
  return captures;
}

bool isLowerableSummaryCollector(CollectorKind kind, Type valueType) {
  if (isPointerLikeType(valueType))
    return false;
  if (kind == CollectorKind::ELEMENT_COUNT)
    return getStaticElementCount(valueType) != 0;
  return getStaticElementCount(valueType) != 0 && isFloatValueType(valueType);
}

ArrayAttr buildCollectorArrayForValue(Builder &builder, RecordLevel level,
                                      Type valueType) {
  SmallVector<Attribute> collectors;
  for (CollectorKind kind : getEnabledCollectors(level)) {
    if (!isLowerableSummaryCollector(kind, valueType))
      continue;
    std::string_view name = getCollectorName(kind);
    if (!name.empty())
      collectors.push_back(builder.getStringAttr(name));
  }
  return builder.getArrayAttr(collectors);
}

bool isLevel1LargeAuxiliarySummary(Operation *op, Type resultType,
                                   RecordLevel level) {
  if (level != RecordLevel::LEVEL_SUMMARY)
    return false;

  StringRef opName = op->getName().getStringRef();
  if (opName == "tt.load" || opName == "memref.load" || opName == "tt.dot" ||
      opName == "tt.reduce" || opName == "tt.scan")
    return false;

  uint64_t elementCount = getStaticElementCount(resultType);
  return elementCount > kLevel1AuxSummaryElementLimit;
}

bool isLevel1LargeLoadSummary(Operation *op, Type resultType,
                              RecordLevel level) {
  if (level != RecordLevel::LEVEL_SUMMARY)
    return false;

  StringRef opName = op->getName().getStringRef();
  if (opName != "tt.load" && opName != "memref.load")
    return false;

  uint64_t elementCount = getStaticElementCount(resultType);
  return elementCount > kLevel1LoadSummaryElementLimit;
}

bool shouldEmitDynamicSummary(Operation *op, Type resultType,
                              RecordLevel level) {
  StringRef opName = op->getName().getStringRef();
  if (opName == "tt.load" || opName == "memref.load")
    return !isLevel1LargeLoadSummary(op, resultType, level);
  if (opName == "tt.dot")
    return true;
  if (opName == "tt.reduce" || opName == "tt.scan")
    return true;

  StringRef dialect = op->getName().getDialectNamespace();
  if ((dialect == "arith" || dialect == "math") &&
      isFloatValueType(resultType) &&
      !isLevel1LargeAuxiliarySummary(op, resultType, level))
    return true;
  return false;
}

bool recordSummaryCanLower(Operation *op) {
  auto collectors = op->getAttrOfType<ArrayAttr>("collectors");
  if (!collectors || op->getNumOperands() != 1)
    return false;
  const Type valueType = op->getOperand(0).getType();
  for (Attribute attr : collectors) {
    auto name = dyn_cast<StringAttr>(attr);
    if (!name || !collectorIdForName(name.getValue()))
      continue;
    if (name.getValue() == "element_count")
      return getStaticElementCount(valueType) != 0;
    if (isFloatValueType(valueType) && getStaticElementCount(valueType) != 0)
      return true;
  }
  return false;
}

bool isPointerLikeType(Type type) {
  return isa<triton::PointerType>(getElementType(type));
}

bool isIntegerValueType(Type type) {
  return isa<IntegerType>(getElementType(type));
}

uint32_t elementBits(Type type) {
  Type element = getElementType(type);
  if (auto intType = dyn_cast<IntegerType>(element))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(element))
    return floatType.getWidth();
  if (auto ptrType = dyn_cast<triton::PointerType>(element))
    return elementBits(ptrType.getPointeeType());
  return 0;
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

Type memoryAccessValueType(Operation *op) {
  if (Value stored = storedValueOperand(op))
    return stored.getType();
  if (op->getNumResults() > 0)
    return op->getResult(0).getType();
  if (Value ptr = memoryPointerOperand(op))
    return pointerPointeeType(ptr.getType());
  return {};
}

uint32_t accessBytes(Operation *op) {
  uint32_t bits = elementBits(memoryAccessValueType(op));
  return bits == 0 ? 0 : std::max<uint32_t>(1, bits / 8);
}

Value memoryMaskOperand(Operation *op) {
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return load.getMask();
  if (auto store = dyn_cast<triton::StoreOp>(op))
    return store.getMask();
  if (auto atomic = dyn_cast<triton::AtomicRMWOp>(op))
    return atomic.getMask();
  StringRef opName = op->getName().getStringRef();
  if (isAsyncCopyOpName(opName) && op->getNumOperands() > 2) {
    Value candidate = op->getOperand(2);
    if (isa<IntegerType>(getElementType(candidate.getType())))
      return candidate;
  }
  return {};
}

Value hiddenDebugArgument(FunctionOpInterface func) {
  auto indexAttr = func->getAttrOfType<IntegerAttr>(kAttrHiddenArgIndex);
  if (!indexAttr)
    return {};
  int64_t index = indexAttr.getInt();
  if (index < 0 || index >= static_cast<int64_t>(func.getNumArguments()))
    return {};
  return func.getArgument(static_cast<unsigned>(index));
}

Value createIntegerConstant(OpBuilder &builder, Location loc, unsigned width,
                            uint64_t value) {
  auto type = builder.getIntegerType(width);
  return builder.create<arith::ConstantOp>(
      loc, type, IntegerAttr::get(type, llvm::APInt(width, value)));
}

Value createI32Constant(OpBuilder &builder, Location loc, uint32_t value) {
  return createIntegerConstant(builder, loc, 32, value);
}

Value createI64Constant(OpBuilder &builder, Location loc, uint64_t value) {
  return createIntegerConstant(builder, loc, 64, value);
}

Value createBoolConstant(OpBuilder &builder, Location loc, bool value) {
  return builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(value));
}

Value addWordOffset(OpBuilder &builder, Location loc, Value wordPtr,
                    Value offset) {
  return builder.create<triton::AddPtrOp>(loc, wordPtr.getType(), wordPtr,
                                          offset);
}

Value addWordOffsetLike(OpBuilder &builder, Location loc, Value wordPtr,
                        Value offset) {
  if (auto rankedOffset = dyn_cast<RankedTensorType>(offset.getType())) {
    Type resultType = RankedTensorType::get(
        rankedOffset.getShape(), wordPtr.getType(), rankedOffset.getEncoding());
    Value ptr = builder.create<triton::SplatOp>(loc, resultType, wordPtr);
    return builder.create<triton::AddPtrOp>(loc, resultType, ptr, offset);
  }
  return builder.create<triton::AddPtrOp>(loc, wordPtr.getType(), wordPtr,
                                          offset);
}

Value fieldPointer(OpBuilder &builder, Location loc, Value baseBytePtr,
                   Value baseOffset, int64_t fieldOffset, Type elementType) {
  assert(elementType == builder.getI32Type() &&
         "debug ring device stores are emitted as i32 words");
  assert(fieldOffset % 4 == 0 &&
         "debug ring field offsets must be word-aligned");
  Value offset = baseOffset;
  if (fieldOffset != 0) {
    offset = builder.create<arith::AddIOp>(
        loc, offset,
        createI64Constant(builder, loc,
                          static_cast<uint64_t>(fieldOffset / 4)));
  }
  return addWordOffset(builder, loc, baseBytePtr, offset);
}

Value absoluteFieldPointer(OpBuilder &builder, Location loc, Value baseBytePtr,
                           int64_t fieldOffset, Type elementType) {
  return fieldPointer(builder, loc, baseBytePtr,
                      createI64Constant(builder, loc, 0), fieldOffset,
                      elementType);
}

Value loadI32(OpBuilder &builder, Location loc, Value ptr) {
  return builder.create<triton::LoadOp>(loc, ptr, triton::CacheModifier::NONE,
                                        triton::EvictionPolicy::NORMAL, false);
}

Value loadRingHeaderI32(OpBuilder &builder, Location loc, Value ctrlWordPtr,
                        int64_t fieldOffset) {
  assert(fieldOffset >= 0 && fieldOffset < kRingHeaderBytes &&
         fieldOffset % 4 == 0 && "ring header field must be word-aligned");
  constexpr int64_t kHeaderWords = kRingHeaderBytes / 4;
  auto i32TensorType =
      RankedTensorType::get({kHeaderWords}, builder.getI32Type());
  Value offsets = builder.create<triton::MakeRangeOp>(
      loc, i32TensorType, 0, static_cast<int32_t>(kHeaderWords));
  Value ptrs = addWordOffsetLike(builder, loc, ctrlWordPtr, offsets);
  Value header =
      builder.create<triton::LoadOp>(loc, ptrs, triton::CacheModifier::NONE,
                                     triton::EvictionPolicy::NORMAL, false);
  Value index = builder.create<arith::ConstantIndexOp>(loc, fieldOffset / 4);
  return builder.create<tensor::ExtractOp>(loc, header, ValueRange{index});
}

void storeValue(OpBuilder &builder, Location loc, Value ptr, Value value,
                Value mask = {}) {
  if (mask) {
    builder.create<triton::StoreOp>(loc, ptr, value, mask,
                                    triton::CacheModifier::NONE,
                                    triton::EvictionPolicy::NORMAL);
    return;
  }
  builder.create<triton::StoreOp>(loc, ptr, value, triton::CacheModifier::NONE,
                                  triton::EvictionPolicy::NORMAL);
}

void storeI32(OpBuilder &builder, Location loc, Value baseBytePtr,
              Value recordOffset, int64_t fieldOffset, uint32_t value,
              Value mask = {}) {
  storeValue(builder, loc,
             fieldPointer(builder, loc, baseBytePtr, recordOffset, fieldOffset,
                          builder.getI32Type()),
             createI32Constant(builder, loc, value), mask);
}

void storeI32Value(OpBuilder &builder, Location loc, Value baseBytePtr,
                   Value recordOffset, int64_t fieldOffset, Value value,
                   Value mask = {}) {
  storeValue(builder, loc,
             fieldPointer(builder, loc, baseBytePtr, recordOffset, fieldOffset,
                          builder.getI32Type()),
             value, mask);
}

void storeI64(OpBuilder &builder, Location loc, Value baseBytePtr,
              Value recordOffset, int64_t fieldOffset, Value value,
              Value mask = {}) {
  Value low = builder.create<arith::TruncIOp>(loc, builder.getI32Type(), value);
  Value shifted = builder.create<arith::ShRUIOp>(
      loc, value, createI64Constant(builder, loc, 32));
  Value high =
      builder.create<arith::TruncIOp>(loc, builder.getI32Type(), shifted);
  storeI32Value(builder, loc, baseBytePtr, recordOffset, fieldOffset, low,
                mask);
  storeI32Value(builder, loc, baseBytePtr, recordOffset, fieldOffset + 4, high,
                mask);
}

void storeI64Constant(OpBuilder &builder, Location loc, Value baseBytePtr,
                      Value recordOffset, int64_t fieldOffset, uint64_t value,
                      Value mask = {}) {
  storeI32(builder, loc, baseBytePtr, recordOffset, fieldOffset,
           static_cast<uint32_t>(value & 0xffffffffu), mask);
  storeI32(builder, loc, baseBytePtr, recordOffset, fieldOffset + 4,
           static_cast<uint32_t>((value >> 32) & 0xffffffffu), mask);
}

Type withElementType(Type type, Type elementType);
Value createIntegerConstantLike(OpBuilder &builder, Location loc, Type type,
                                uint64_t value);

std::pair<Value, Value> splitI64ToI32Words(OpBuilder &builder, Location loc,
                                           Value value) {
  Type i32Like = withElementType(value.getType(), builder.getI32Type());
  Value low = builder.create<arith::TruncIOp>(loc, i32Like, value);
  Value shifted = builder.create<arith::ShRUIOp>(
      loc, value, createIntegerConstantLike(builder, loc, value.getType(), 32));
  Value high = builder.create<arith::TruncIOp>(loc, i32Like, shifted);
  return {low, high};
}

void storeRecordWords(OpBuilder &builder, Location loc, Value ctrlBytePtr,
                      Value recordOffset, ArrayRef<Value> scalarWords,
                      Value mask) {
  assert((static_cast<int64_t>(scalarWords.size()) == kLegacyRecordWords ||
          static_cast<int64_t>(scalarWords.size()) == kRecordWords) &&
         "debug records are fixed-width legacy or bundle records");
  const int64_t wordCount = static_cast<int64_t>(scalarWords.size());
  auto i32TensorType = RankedTensorType::get({wordCount}, builder.getI32Type());

  Value lanesI32 = builder.create<triton::MakeRangeOp>(
      loc, i32TensorType, 0, static_cast<int32_t>(wordCount));
  Value recordOffsetI32 =
      builder.create<arith::TruncIOp>(loc, builder.getI32Type(), recordOffset);
  Value recordOffsetSplat =
      builder.create<triton::SplatOp>(loc, i32TensorType, recordOffsetI32);
  Value wordOffsets =
      builder.create<arith::AddIOp>(loc, recordOffsetSplat, lanesI32);
  Value ptrs = addWordOffsetLike(builder, loc, ctrlBytePtr, wordOffsets);

  Value values = createIntegerConstantLike(builder, loc, i32TensorType, 0);
  for (auto indexed : llvm::enumerate(scalarWords)) {
    Value lane =
        createIntegerConstantLike(builder, loc, i32TensorType, indexed.index());
    Value laneMask = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lanesI32, lane);
    Value word =
        builder.create<triton::SplatOp>(loc, i32TensorType, indexed.value());
    values = builder.create<arith::SelectOp>(loc, laneMask, word, values);
  }

  builder.create<scf::IfOp>(loc, mask,
                            [&](OpBuilder &thenBuilder, Location thenLoc) {
                              storeValue(thenBuilder, thenLoc, ptrs, values);
                              thenBuilder.create<scf::YieldOp>(thenLoc);
                            });
}

Value createAtomicRMW(OpBuilder &builder, Location loc, triton::RMWOp rmwOp,
                      Value ptr, Value value, Value mask = {}) {
  if (!mask)
    mask = createBoolConstant(builder, loc, true);
  constexpr int64_t kAtomicReservationLanes = 4;
  auto resultType =
      RankedTensorType::get({kAtomicReservationLanes}, value.getType());
  auto ptrType =
      RankedTensorType::get({kAtomicReservationLanes}, ptr.getType());
  auto maskType =
      RankedTensorType::get({kAtomicReservationLanes}, mask.getType());
  Value ptrSplat = builder.create<triton::SplatOp>(loc, ptrType, ptr);
  Value valueSplat = builder.create<triton::SplatOp>(loc, resultType, value);
  Value maskSplat = builder.create<triton::SplatOp>(loc, maskType, mask);
  auto laneType =
      RankedTensorType::get({kAtomicReservationLanes}, builder.getI32Type());
  Value lanes = builder.create<triton::MakeRangeOp>(
      loc, laneType, 0, static_cast<int32_t>(kAtomicReservationLanes));
  Value zero = createI32Constant(builder, loc, 0);
  Value zeroSplat = builder.create<triton::SplatOp>(loc, laneType, zero);
  Value lane0Mask = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  lanes, zeroSplat);
  Value effectiveMask =
      builder.create<arith::AndIOp>(loc, maskSplat, lane0Mask);
  Value atomicResult = builder.create<triton::AtomicRMWOp>(
      loc, resultType, rmwOp, ptrSplat, valueSplat, effectiveMask,
      triton::MemSemantic::ACQUIRE_RELEASE, triton::MemSyncScope::GPU);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  return builder.create<tensor::ExtractOp>(loc, atomicResult,
                                           ValueRange{zeroIndex});
}

Value computeLogicalInstanceId(OpBuilder &builder, Location loc) {
  Value pid0 = builder.create<triton::GetProgramIdOp>(loc, 0);
  Value pid1 = builder.create<triton::GetProgramIdOp>(loc, 1);
  Value pid2 = builder.create<triton::GetProgramIdOp>(loc, 2);
  Value numPrograms0 = builder.create<triton::GetNumProgramsOp>(loc, 0);
  Value numPrograms1 = builder.create<triton::GetNumProgramsOp>(loc, 1);

  Value pid0I64 =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), pid0);
  Value pid1I64 =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), pid1);
  Value pid2I64 =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), pid2);
  Value numPrograms0I64 =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), numPrograms0);
  Value numPrograms1I64 =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), numPrograms1);

  Value pid1Term = builder.create<arith::MulIOp>(loc, pid1I64, numPrograms0I64);
  Value numPrograms01 =
      builder.create<arith::MulIOp>(loc, numPrograms0I64, numPrograms1I64);
  Value pid2Term = builder.create<arith::MulIOp>(loc, pid2I64, numPrograms01);
  Value partial = builder.create<arith::AddIOp>(loc, pid0I64, pid1Term);
  return builder.create<arith::AddIOp>(loc, partial, pid2Term);
}

struct ReservedRecordSlot {
  Value slot;
  Value recordOffset;
  Value inBounds;
};

int32_t getRecordIndex(Operation *recordOp) {
  auto attr = recordOp->getAttrOfType<IntegerAttr>(kAttrRecordIndex);
  if (!attr)
    return 0;
  return static_cast<int32_t>(attr.getInt());
}

int32_t getRecordsPerInstance(Operation *recordOp) {
  for (Operation *cursor = recordOp; cursor; cursor = cursor->getParentOp()) {
    if (auto attr = cursor->getAttrOfType<IntegerAttr>(kAttrRecordsPerInstance))
      return static_cast<int32_t>(attr.getInt());
  }
  return 1;
}

struct RecordLoweringContext {
  Value ctrlBytePtr;
  Value logicalInstanceId;
  Value instanceBase;
  Value capacityI64;
  Value payloadOffsetBytes;
  Value payloadInstanceBaseBytes;
  int32_t recordsPerInstance = 1;
  uint64_t payloadBytesPerInstance = 0;
};

RecordLoweringContext
createRecordLoweringContext(OpBuilder &builder, FunctionOpInterface func,
                            Value ctrlBytePtr, int32_t recordsPerInstance,
                            uint64_t payloadBytesPerInstance) {
  Region &body = func.getFunctionBody();
  Block &entry = body.front();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  Location loc = func.getLoc();
  int32_t normalizedRecordsPerInstance =
      std::max<int32_t>(1, recordsPerInstance);
  Value logicalInstanceId = computeLogicalInstanceId(builder, loc);
  Value recordsPerInstanceI64 = createI64Constant(
      builder, loc, static_cast<uint64_t>(normalizedRecordsPerInstance));
  Value instanceBase = builder.create<arith::MulIOp>(loc, logicalInstanceId,
                                                     recordsPerInstanceI64);
  Value capacity =
      loadRingHeaderI32(builder, loc, ctrlBytePtr, kHeaderCapacityOffset);
  Value capacityI64 =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), capacity);
  Value payloadOffset =
      loadRingHeaderI32(builder, loc, ctrlBytePtr, kHeaderPayloadOffsetOffset);
  Value payloadOffsetBytes =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), payloadOffset);
  Value payloadInstanceBaseBytes = builder.create<arith::MulIOp>(
      loc, logicalInstanceId,
      createI64Constant(builder, loc, payloadBytesPerInstance));
  return RecordLoweringContext{ctrlBytePtr,
                               logicalInstanceId,
                               instanceBase,
                               capacityI64,
                               payloadOffsetBytes,
                               payloadInstanceBaseBytes,
                               normalizedRecordsPerInstance,
                               payloadBytesPerInstance};
}

ReservedRecordSlot reserveDeterministicRecordSlot(
    OpBuilder &builder, Location loc, Operation *recordOp,
    const RecordLoweringContext &ctx, int32_t indexOffset = 0) {
  const int32_t recordIndex = getRecordIndex(recordOp) + indexOffset;
  Value slotI64 = builder.create<arith::AddIOp>(
      loc, ctx.instanceBase,
      createI64Constant(builder, loc, static_cast<uint64_t>(recordIndex)));
  Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 slotI64, ctx.capacityI64);

  Value slotWords = builder.create<arith::MulIOp>(
      loc, slotI64,
      createI64Constant(builder, loc, static_cast<uint64_t>(kRecordWords)));
  Value recordOffset = builder.create<arith::AddIOp>(
      loc,
      createI64Constant(builder, loc,
                        static_cast<uint64_t>(kRingHeaderBytes / 4)),
      slotWords);

  Value slotI32 =
      builder.create<arith::TruncIOp>(loc, builder.getI32Type(), slotI64);
  return ReservedRecordSlot{slotI32, recordOffset, inBounds};
}

ReservedRecordSlot reserveRecordSlot(OpBuilder &builder, Location loc,
                                     Value ctrlBytePtr) {
  Value writeIdxPtr = absoluteFieldPointer(
      builder, loc, ctrlBytePtr, kHeaderWriteIdxOffset, builder.getI32Type());
  Value one = createI32Constant(builder, loc, 1);
  Value slot =
      createAtomicRMW(builder, loc, triton::RMWOp::ADD, writeIdxPtr, one);

  Value capacityPtr = absoluteFieldPointer(
      builder, loc, ctrlBytePtr, kHeaderCapacityOffset, builder.getI32Type());
  Value capacity = loadI32(builder, loc, capacityPtr);
  Value inBounds = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 slot, capacity);

  Value slotI64 =
      builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), slot);
  Value slotWords = builder.create<arith::MulIOp>(
      loc, slotI64,
      createI64Constant(builder, loc, static_cast<uint64_t>(kRecordBytes / 4)));
  Value recordOffset = builder.create<arith::AddIOp>(
      loc,
      createI64Constant(builder, loc,
                        static_cast<uint64_t>(kRingHeaderBytes / 4)),
      slotWords);
  return ReservedRecordSlot{slot, recordOffset, inBounds};
}

void emitOverflowUpdate(OpBuilder &builder, Location loc, Value ctrlBytePtr,
                        Value mask) {
  Value one = createI32Constant(builder, loc, 1);
  Value overflowCountPtr =
      absoluteFieldPointer(builder, loc, ctrlBytePtr,
                           kHeaderOverflowCountOffset, builder.getI32Type());
  createAtomicRMW(builder, loc, triton::RMWOp::ADD, overflowCountPtr, one,
                  mask);

  Value flagsPtr = absoluteFieldPointer(
      builder, loc, ctrlBytePtr, kHeaderFlagsOffset, builder.getI32Type());
  createAtomicRMW(builder, loc, triton::RMWOp::OR, flagsPtr,
                  createI32Constant(builder, loc, kRingFlagOverflow), mask);
}

void emitRecordHeaderStores(OpBuilder &builder, Location loc, Value ctrlBytePtr,
                            Value recordOffset, uint32_t recordKind,
                            uint32_t opId, Value logicalInstanceId,
                            Value mask) {
  storeI32(builder, loc, ctrlBytePtr, recordOffset, 0, recordKind, mask);
  storeI32(builder, loc, ctrlBytePtr, recordOffset, 4, opId, mask);
  storeI64(builder, loc, ctrlBytePtr, recordOffset, 8, logicalInstanceId, mask);
}

Type withElementType(Type type, Type elementType) {
  if (auto ranked = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(ranked.getShape(), elementType,
                                 ranked.getEncoding());
  return elementType;
}

Value createIntegerConstantLike(OpBuilder &builder, Location loc, Type type,
                                uint64_t value) {
  Attribute attr;
  Type elementType = getElementType(type);
  auto intType = dyn_cast<IntegerType>(elementType);
  if (!intType)
    return {};
  attr = IntegerAttr::get(elementType, llvm::APInt(intType.getWidth(), value));
  if (auto ranked = dyn_cast<RankedTensorType>(type))
    attr = DenseElementsAttr::get(ranked, attr);
  return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(attr));
}

Value createBoolConstantLike(OpBuilder &builder, Location loc, Type type,
                             bool value) {
  return createIntegerConstantLike(builder, loc, type, value ? 1 : 0);
}

Value createFloatConstantLike(OpBuilder &builder, Location loc, Type type,
                              double value) {
  Type elementType = getElementType(type);
  if (!isa<FloatType>(elementType))
    return {};
  Attribute attr = builder.getFloatAttr(elementType, value);
  if (auto ranked = dyn_cast<RankedTensorType>(type))
    attr = DenseElementsAttr::get(ranked, attr);
  return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(attr));
}

Value flattenTensorForSummary(OpBuilder &builder, Location loc, Value value) {
  auto ranked = dyn_cast<RankedTensorType>(value.getType());
  if (!ranked)
    return value;
  const uint64_t elementCount = getStaticElementCount(ranked);
  if (elementCount == 0)
    return {};
  if (ranked.getRank() == 1 &&
      ranked.getDimSize(0) == static_cast<int64_t>(elementCount))
    return value;
  auto flatType =
      RankedTensorType::get({static_cast<int64_t>(elementCount)},
                            ranked.getElementType(), ranked.getEncoding());
  return builder.create<triton::ReshapeOp>(loc, flatType, value,
                                           /*allow_reorder=*/true,
                                           /*efficient_layout=*/false);
}

Value castFloatValueToF32(OpBuilder &builder, Location loc, Value value) {
  Type elementType = getElementType(value.getType());
  if (!isa<FloatType>(elementType))
    return {};
  // CANN9-compatible summary lowering currently computes all device-side
  // floating summaries through f32 reductions.  This keeps the record ABI and
  // current backend lowering simple, but f64 precision-sensitive summaries need
  // a future dtype-preserving collector path.
  Type f32Like = withElementType(value.getType(), builder.getF32Type());
  if (value.getType() == f32Like)
    return value;
  const unsigned width = elementType.getIntOrFloatBitWidth();
  if (width < 32)
    return builder.create<arith::ExtFOp>(loc, f32Like, value);
  if (width > 32)
    return builder.create<arith::TruncFOp>(loc, f32Like, value);
  return value;
}

template <typename CombineFn>
Value reduceTensor(OpBuilder &builder, Location loc, Value value,
                   CombineFn combine) {
  Value flat = flattenTensorForSummary(builder, loc, value);
  if (!flat)
    return {};
  auto ranked = dyn_cast<RankedTensorType>(flat.getType());
  if (!ranked)
    return flat;

  auto reduce = builder.create<triton::ReduceOp>(loc, ValueRange{flat}, 0);
  Region &region = reduce.getCombineOp();
  Block *block = new Block();
  region.push_back(block);
  Type elementType = ranked.getElementType();
  block->addArgument(elementType, loc);
  block->addArgument(elementType, loc);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);
  Value combined =
      combine(builder, loc, block->getArgument(0), block->getArgument(1));
  builder.create<triton::ReduceReturnOp>(loc, ValueRange{combined});
  return reduce->getResult(0);
}

Value reduceAddI32(OpBuilder &builder, Location loc, Value value) {
  return reduceTensor(
      builder, loc, value,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) -> Value {
        return builder.create<arith::AddIOp>(loc, lhs, rhs);
      });
}

Value reduceMaxUI64(OpBuilder &builder, Location loc, Value value) {
  return reduceTensor(
      builder, loc, value,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) -> Value {
        return builder.create<arith::MaxUIOp>(loc, lhs, rhs);
      });
}

Value reduceMinUI(OpBuilder &builder, Location loc, Value value) {
  return reduceTensor(
      builder, loc, value,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) -> Value {
        return builder.create<arith::MinUIOp>(loc, lhs, rhs);
      });
}

Value reduceMaxUI(OpBuilder &builder, Location loc, Value value) {
  return reduceTensor(
      builder, loc, value,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) -> Value {
        return builder.create<arith::MaxUIOp>(loc, lhs, rhs);
      });
}

Value reduceAddF32(OpBuilder &builder, Location loc, Value value) {
  return reduceTensor(
      builder, loc, value,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) -> Value {
        return builder.create<arith::AddFOp>(loc, lhs, rhs);
      });
}

Value reduceMinF32(OpBuilder &builder, Location loc, Value value) {
  return reduceTensor(
      builder, loc, value,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) -> Value {
        return builder.create<arith::MinNumFOp>(loc, lhs, rhs);
      });
}

Value reduceMaxF32(OpBuilder &builder, Location loc, Value value) {
  return reduceTensor(
      builder, loc, value,
      [](OpBuilder &builder, Location loc, Value lhs, Value rhs) -> Value {
        return builder.create<arith::MaxNumFOp>(loc, lhs, rhs);
      });
}

Value boolToI32(OpBuilder &builder, Location loc, Value value) {
  Type i32Like = withElementType(value.getType(), builder.getI32Type());
  return builder.create<arith::ExtUIOp>(loc, i32Like, value);
}

Value countTrueToI64(OpBuilder &builder, Location loc, Value predicate) {
  Value countI32 =
      reduceAddI32(builder, loc, boolToI32(builder, loc, predicate));
  return builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), countI32);
}

struct PointerAddressSlice {
  Value base;
  SmallVector<Value> offsets;
  bool tensorPointer = false;
};

bool isShapeCompatibleForPointerTrace(Type srcType, Type dstType) {
  uint64_t srcCount = getStaticElementCount(srcType);
  uint64_t dstCount = getStaticElementCount(dstType);
  return srcCount != 0 && dstCount != 0 &&
         (srcCount == dstCount || srcCount == 1);
}

std::optional<PointerAddressSlice> tracePointerAddressSlice(Value pointer) {
  if (!pointer || !isPointerLikeType(pointer.getType()))
    return std::nullopt;

  PointerAddressSlice slice;
  slice.tensorPointer = isa<RankedTensorType>(pointer.getType());
  Type finalType = pointer.getType();
  Value cursor = pointer;

  for (int32_t depth = 0; depth < kMaxPointerTraceDepth && cursor; ++depth) {
    if (!isa<RankedTensorType>(cursor.getType())) {
      if (!isPointerLikeType(cursor.getType()))
        return std::nullopt;
      slice.base = cursor;
      return slice;
    }

    Operation *def = cursor.getDefiningOp();
    if (!def)
      return std::nullopt;

    if (auto addPtr = dyn_cast<triton::AddPtrOp>(def)) {
      Value offset = addPtr.getOffset();
      if (!isIntegerValueType(offset.getType()) ||
          !isShapeCompatibleForPointerTrace(offset.getType(), finalType))
        return std::nullopt;
      slice.offsets.push_back(offset);
      cursor = addPtr.getPtr();
      continue;
    }

    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      Value src = splat.getSrc();
      if (!isPointerLikeType(src.getType()))
        return std::nullopt;
      slice.base = src;
      return slice;
    }

    if (auto bitcast = dyn_cast<triton::BitcastOp>(def)) {
      cursor = bitcast.getSrc();
      continue;
    }

    StringRef name = def->getName().getStringRef();
    if ((name == "tt.reshape" || name == "tt.broadcast" ||
         name == "tt.expand_dims") &&
        def->getNumOperands() == 1 &&
        isShapeCompatibleForPointerTrace(def->getOperand(0).getType(),
                                         finalType)) {
      cursor = def->getOperand(0);
      continue;
    }

    return std::nullopt;
  }

  return std::nullopt;
}

bool canComputeAddressSummary(Value pointer) {
  return tracePointerAddressSlice(pointer).has_value();
}

bool isIntegerConstantSplat(Value value, int64_t &constant) {
  Operation *def = value ? value.getDefiningOp() : nullptr;
  if (!def)
    return false;
  auto constOp = dyn_cast<arith::ConstantOp>(def);
  if (!constOp)
    return false;
  Attribute attr = constOp.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    constant = intAttr.getInt();
    return true;
  }
  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return false;
  constant = (*dense.value_begin<llvm::APInt>()).getSExtValue();
  return true;
}

bool isScalarLikeIntegerOffset(Value value) {
  if (!value || !isIntegerValueType(value.getType()))
    return false;
  if (!isa<RankedTensorType>(value.getType()))
    return true;
  if (getStaticElementCount(value.getType()) == 1)
    return true;
  if (auto splat = dyn_cast_or_null<triton::SplatOp>(value.getDefiningOp()))
    return isIntegerValueType(splat.getSrc().getType());
  int64_t ignored = 0;
  return isIntegerConstantSplat(value, ignored);
}

bool isContiguousLaneOffset(Value value, int32_t depth = 0) {
  if (!value || depth >= kMaxPointerTraceDepth ||
      !isIntegerValueType(value.getType()))
    return false;
  auto ranked = dyn_cast<RankedTensorType>(value.getType());
  if (!ranked || getStaticElementCount(ranked) == 0)
    return false;

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;

  if (auto range = dyn_cast<triton::MakeRangeOp>(def)) {
    return static_cast<int64_t>(getStaticElementCount(ranked)) ==
           static_cast<int64_t>(range.getEnd() - range.getStart());
  }

  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    Value lhs = add.getLhs();
    Value rhs = add.getRhs();
    return (isContiguousLaneOffset(lhs, depth + 1) &&
            isScalarLikeIntegerOffset(rhs)) ||
           (isScalarLikeIntegerOffset(lhs) &&
            isContiguousLaneOffset(rhs, depth + 1));
  }

  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    return isContiguousLaneOffset(sub.getLhs(), depth + 1) &&
           isScalarLikeIntegerOffset(sub.getRhs());
  }

  if (auto ext = dyn_cast<arith::ExtSIOp>(def))
    return isContiguousLaneOffset(ext.getIn(), depth + 1);
  if (auto ext = dyn_cast<arith::ExtUIOp>(def))
    return isContiguousLaneOffset(ext.getIn(), depth + 1);
  if (auto trunc = dyn_cast<arith::TruncIOp>(def))
    return isContiguousLaneOffset(trunc.getIn(), depth + 1);
  if (auto bitcast = dyn_cast<triton::BitcastOp>(def))
    return isContiguousLaneOffset(bitcast.getSrc(), depth + 1);

  StringRef name = def->getName().getStringRef();
  if ((name == "tt.reshape" || name == "tt.broadcast" ||
       name == "tt.expand_dims") &&
      def->getNumOperands() == 1)
    return isContiguousLaneOffset(def->getOperand(0), depth + 1);

  return false;
}

bool hasContiguousAddressOffsets(const PointerAddressSlice &slice) {
  if (slice.offsets.empty())
    return !slice.tensorPointer;
  bool sawContiguous = false;
  for (Value offset : slice.offsets) {
    if (isContiguousLaneOffset(offset)) {
      if (sawContiguous)
        return false;
      sawContiguous = true;
      continue;
    }
    if (!isScalarLikeIntegerOffset(offset))
      return false;
  }
  return sawContiguous;
}

bool isAllTrueMask(Value mask) {
  if (!mask)
    return true;
  Operation *def = mask.getDefiningOp();
  if (!def)
    return false;
  auto constOp = dyn_cast<arith::ConstantOp>(def);
  if (!constOp)
    return false;
  Attribute attr = constOp.getValue();
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return boolAttr.getValue();
  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  if (!dense || !dense.isSplat())
    return false;
  return (*dense.value_begin<llvm::APInt>()).isAllOnes();
}

bool isSupportedPrefixMask(Value mask, ArrayRef<Value> offsets) {
  if (!mask || isAllTrueMask(mask))
    return true;
  auto cmp = dyn_cast_or_null<arith::CmpIOp>(mask.getDefiningOp());
  if (!cmp)
    return false;
  switch (cmp.getPredicate()) {
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ult:
  case arith::CmpIPredicate::ule:
    break;
  default:
    return false;
  }
  Value lhs = cmp.getLhs();
  if (!isContiguousLaneOffset(lhs) || !isScalarLikeIntegerOffset(cmp.getRhs()))
    return false;
  for (Value offset : offsets) {
    if (offset == lhs)
      return true;
    if (isContiguousLaneOffset(offset))
      return true;
  }
  return false;
}

bool shouldUseRepresentativeAddressOnly(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return isTensormapOpName(opName) || isAsyncCopyOpName(opName) ||
         isCopyOpName(opName);
}

bool canComputeAddressSummaryForMemoryOp(Operation *op) {
  if (shouldUseRepresentativeAddressOnly(op))
    return false;
  Value pointer = memoryPointerOperand(op);
  if (!pointer || !isPointerLikeType(pointer.getType()) || accessBytes(op) == 0)
    return false;
  std::optional<PointerAddressSlice> slice = tracePointerAddressSlice(pointer);
  if (!slice || !slice->base || !hasContiguousAddressOffsets(*slice))
    return false;
  return isSupportedPrefixMask(memoryMaskOperand(op), slice->offsets);
}

bool canComputeAddressSummaryForMemoryTarget(
    Operation *op, const MemoryAddressTarget &target) {
  if (shouldUseRepresentativeAddressOnly(op))
    return false;
  Value pointer = target.pointer;
  if (!pointer || !isPointerLikeType(pointer.getType()) || accessBytes(op) == 0)
    return false;
  std::optional<PointerAddressSlice> slice = tracePointerAddressSlice(pointer);
  if (!slice || !slice->base || !hasContiguousAddressOffsets(*slice))
    return false;
  return isSupportedPrefixMask(memoryMaskOperand(op), slice->offsets);
}

StringRef memoryEventKindForMemoryOp(Operation *op) {
  if (canComputeAddressSummaryForMemoryOp(op))
    return kEventAddressSummary;
  return memoryEventKindForPointer(memoryPointerOperand(op));
}

StringRef memoryEventKindForMemoryTarget(Operation *op,
                                         const MemoryAddressTarget &target) {
  if (canComputeAddressSummaryForMemoryTarget(op, target))
    return kEventAddressSummary;
  return memoryEventKindForPointer(target.pointer);
}

struct FloatPredicates {
  Value nan;
  Value inf;
  Value finite;
};

FloatPredicates buildFloatPredicates(OpBuilder &builder, Location loc,
                                     Value value) {
  Value posInf = createFloatConstantLike(
      builder, loc, value.getType(), std::numeric_limits<float>::infinity());
  Value negInf = createFloatConstantLike(
      builder, loc, value.getType(), -std::numeric_limits<float>::infinity());
  Value nan = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                            value, value);
  Value posInfEq = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                 value, posInf);
  Value negInfEq = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                 value, negInf);
  Value inf = builder.create<arith::OrIOp>(loc, posInfEq, negInfEq);
  Value nonFinite = builder.create<arith::OrIOp>(loc, nan, inf);
  Value finite = builder.create<arith::XOrIOp>(
      loc, nonFinite,
      createBoolConstantLike(builder, loc, nonFinite.getType(), true));
  return FloatPredicates{nan, inf, finite};
}

Value computeFiniteCountI32(OpBuilder &builder, Location loc,
                            const FloatPredicates &predicates) {
  return reduceAddI32(builder, loc, boolToI32(builder, loc, predicates.finite));
}

Value selectFiniteOrZero(OpBuilder &builder, Location loc, Value value,
                         const FloatPredicates &predicates) {
  Value zero = createFloatConstantLike(builder, loc, value.getType(), 0.0);
  return builder.create<arith::SelectOp>(loc, predicates.finite, value, zero);
}

Value computeNanCount(OpBuilder &builder, Location loc, Value f32Value) {
  return countTrueToI64(builder, loc,
                        buildFloatPredicates(builder, loc, f32Value).nan);
}

Value computeInfCount(OpBuilder &builder, Location loc, Value f32Value) {
  return countTrueToI64(builder, loc,
                        buildFloatPredicates(builder, loc, f32Value).inf);
}

Value computeZeroCount(OpBuilder &builder, Location loc, Value f32Value) {
  FloatPredicates predicates = buildFloatPredicates(builder, loc, f32Value);
  Value zero = createFloatConstantLike(builder, loc, f32Value.getType(), 0.0);
  Value isZero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               f32Value, zero);
  Value finiteZero =
      builder.create<arith::AndIOp>(loc, predicates.finite, isZero);
  return countTrueToI64(builder, loc, finiteZero);
}

Value computeMeanFinite(OpBuilder &builder, Location loc, Value f32Value) {
  FloatPredicates predicates = buildFloatPredicates(builder, loc, f32Value);
  Value finiteCountI32 = computeFiniteCountI32(builder, loc, predicates);
  Value hasFinite = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, finiteCountI32,
      createI32Constant(builder, loc, 0));

  Value finiteSum = reduceAddF32(
      builder, loc, selectFiniteOrZero(builder, loc, f32Value, predicates));
  Value countF32 = builder.create<arith::UIToFPOp>(loc, builder.getF32Type(),
                                                   finiteCountI32);
  Value denom = builder.create<arith::SelectOp>(
      loc, hasFinite, countF32,
      createFloatConstantLike(builder, loc, builder.getF32Type(), 1.0));
  Value mean = builder.create<arith::DivFOp>(loc, finiteSum, denom);
  return builder.create<arith::SelectOp>(
      loc, hasFinite, mean,
      createFloatConstantLike(builder, loc, builder.getF32Type(), 0.0));
}

Value computeMinFinite(OpBuilder &builder, Location loc, Value f32Value) {
  FloatPredicates predicates = buildFloatPredicates(builder, loc, f32Value);
  Value finiteCountI32 = computeFiniteCountI32(builder, loc, predicates);
  Value hasFinite = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, finiteCountI32,
      createI32Constant(builder, loc, 0));
  Value minInput = builder.create<arith::SelectOp>(
      loc, predicates.finite, f32Value,
      createFloatConstantLike(builder, loc, f32Value.getType(),
                              std::numeric_limits<float>::infinity()));
  Value minValue = reduceMinF32(builder, loc, minInput);
  return builder.create<arith::SelectOp>(
      loc, hasFinite, minValue,
      createFloatConstantLike(builder, loc, builder.getF32Type(), 0.0));
}

Value computeMaxFinite(OpBuilder &builder, Location loc, Value f32Value) {
  FloatPredicates predicates = buildFloatPredicates(builder, loc, f32Value);
  Value finiteCountI32 = computeFiniteCountI32(builder, loc, predicates);
  Value hasFinite = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, finiteCountI32,
      createI32Constant(builder, loc, 0));
  Value maxInput = builder.create<arith::SelectOp>(
      loc, predicates.finite, f32Value,
      createFloatConstantLike(builder, loc, f32Value.getType(),
                              -std::numeric_limits<float>::infinity()));
  Value maxValue = reduceMaxF32(builder, loc, maxInput);
  return builder.create<arith::SelectOp>(
      loc, hasFinite, maxValue,
      createFloatConstantLike(builder, loc, builder.getF32Type(), 0.0));
}

Value computeL2Norm(OpBuilder &builder, Location loc, Value f32Value) {
  FloatPredicates predicates = buildFloatPredicates(builder, loc, f32Value);
  Value finiteValue = selectFiniteOrZero(builder, loc, f32Value, predicates);
  Value square = builder.create<arith::MulFOp>(loc, finiteValue, finiteValue);
  Value squareSum = reduceAddF32(builder, loc, square);
  return builder.create<math::SqrtOp>(loc, squareSum);
}

void emitSummaryU64Stores(OpBuilder &builder, Location loc, Operation *recordOp,
                          Value ctrlBytePtr, Value recordOffset,
                          Value logicalInstanceId, uint32_t collectorId,
                          Value value, Value mask) {
  auto opIdAttr = recordOp->getAttrOfType<IntegerAttr>("op_id");
  const uint32_t packedCollectorAndType = collectorId | (kResultTypeU64 << 16);
  auto [logicalLow, logicalHigh] =
      splitI64ToI32Words(builder, loc, logicalInstanceId);
  auto [valueLow, valueHigh] = splitI64ToI32Words(builder, loc, value);

  SmallVector<Value, kRecordWords> words = {
      createI32Constant(builder, loc, kRecordKindSummary),
      createI32Constant(builder, loc, static_cast<uint32_t>(opIdAttr.getInt())),
      logicalLow,
      logicalHigh,
      createI32Constant(builder, loc, packedCollectorAndType),
      createI32Constant(builder, loc, 0),
      valueLow,
      valueHigh,
  };
  storeRecordWords(builder, loc, ctrlBytePtr, recordOffset, words, mask);
}

void emitSummaryF32Stores(OpBuilder &builder, Location loc, Operation *recordOp,
                          Value ctrlBytePtr, Value recordOffset,
                          Value logicalInstanceId, uint32_t collectorId,
                          Value value, Value mask) {
  auto opIdAttr = recordOp->getAttrOfType<IntegerAttr>("op_id");
  const uint32_t packedCollectorAndType = collectorId | (kResultTypeF32 << 16);
  auto [logicalLow, logicalHigh] =
      splitI64ToI32Words(builder, loc, logicalInstanceId);
  Value valueBits =
      builder.create<arith::BitcastOp>(loc, builder.getI32Type(), value);

  SmallVector<Value, kRecordWords> words = {
      createI32Constant(builder, loc, kRecordKindSummary),
      createI32Constant(builder, loc, static_cast<uint32_t>(opIdAttr.getInt())),
      logicalLow,
      logicalHigh,
      createI32Constant(builder, loc, packedCollectorAndType),
      createI32Constant(builder, loc, 0),
      valueBits,
      createI32Constant(builder, loc, 0),
  };
  storeRecordWords(builder, loc, ctrlBytePtr, recordOffset, words, mask);
}

void emitSummaryBundleStores(OpBuilder &builder, Location loc,
                             Operation *recordOp, Value ctrlBytePtr,
                             Value countRecordOffset, Value valueRecordOffset,
                             Value countMask, Value valueMask) {
  Value observed = recordOp->getOperand(0);
  Value f32Value = flattenTensorForSummary(
      builder, loc, castFloatValueToF32(builder, loc, observed));
  if (!f32Value)
    return;

  FloatPredicates predicates = buildFloatPredicates(builder, loc, f32Value);
  Value nanCount = countTrueToI64(builder, loc, predicates.nan);
  Value infCount = countTrueToI64(builder, loc, predicates.inf);
  Value zero = createFloatConstantLike(builder, loc, f32Value.getType(), 0.0);
  Value isZero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                               f32Value, zero);
  Value finiteZero =
      builder.create<arith::AndIOp>(loc, predicates.finite, isZero);
  Value zeroCount = countTrueToI64(builder, loc, finiteZero);
  Value elementCount = createI64Constant(
      builder, loc, getStaticElementCount(observed.getType()));

  // Deterministic compact records rely on the runtime-cleared buffer and the
  // static record plan.  Only dynamic payload fields are written on device;
  // record kind, op id, logical instance id, and reserved zeros are
  // reconstructed by the host decoder from slot index.
  storeI64(builder, loc, ctrlBytePtr, countRecordOffset, 16, nanCount,
           countMask);
  storeI64(builder, loc, ctrlBytePtr, countRecordOffset, 24, infCount,
           countMask);
  storeI64(builder, loc, ctrlBytePtr, countRecordOffset, 32, zeroCount,
           countMask);
  storeI64(builder, loc, ctrlBytePtr, countRecordOffset, 40, elementCount,
           countMask);

  Value finiteCountI32 = computeFiniteCountI32(builder, loc, predicates);
  Value hasFinite = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, finiteCountI32,
      createI32Constant(builder, loc, 0));
  Value finiteValue = selectFiniteOrZero(builder, loc, f32Value, predicates);
  Value finiteSum = reduceAddF32(builder, loc, finiteValue);
  Value countF32 = builder.create<arith::UIToFPOp>(loc, builder.getF32Type(),
                                                   finiteCountI32);
  Value denom = builder.create<arith::SelectOp>(
      loc, hasFinite, countF32,
      createFloatConstantLike(builder, loc, builder.getF32Type(), 1.0));
  Value rawMean = builder.create<arith::DivFOp>(loc, finiteSum, denom);
  Value mean = builder.create<arith::SelectOp>(
      loc, hasFinite, rawMean,
      createFloatConstantLike(builder, loc, builder.getF32Type(), 0.0));

  Value minInput = builder.create<arith::SelectOp>(
      loc, predicates.finite, f32Value,
      createFloatConstantLike(builder, loc, f32Value.getType(),
                              std::numeric_limits<float>::infinity()));
  Value minValue = builder.create<arith::SelectOp>(
      loc, hasFinite, reduceMinF32(builder, loc, minInput),
      createFloatConstantLike(builder, loc, builder.getF32Type(), 0.0));

  Value maxInput = builder.create<arith::SelectOp>(
      loc, predicates.finite, f32Value,
      createFloatConstantLike(builder, loc, f32Value.getType(),
                              -std::numeric_limits<float>::infinity()));
  Value maxValue = builder.create<arith::SelectOp>(
      loc, hasFinite, reduceMaxF32(builder, loc, maxInput),
      createFloatConstantLike(builder, loc, builder.getF32Type(), 0.0));

  Value square = builder.create<arith::MulFOp>(loc, finiteValue, finiteValue);
  Value l2Norm =
      builder.create<math::SqrtOp>(loc, reduceAddF32(builder, loc, square));

  Value meanBits =
      builder.create<arith::BitcastOp>(loc, builder.getI32Type(), mean);
  Value minBits =
      builder.create<arith::BitcastOp>(loc, builder.getI32Type(), minValue);
  Value maxBits =
      builder.create<arith::BitcastOp>(loc, builder.getI32Type(), maxValue);
  Value l2Bits =
      builder.create<arith::BitcastOp>(loc, builder.getI32Type(), l2Norm);

  storeI32Value(builder, loc, ctrlBytePtr, valueRecordOffset, 16, meanBits,
                valueMask);
  storeI32Value(builder, loc, ctrlBytePtr, valueRecordOffset, 20, minBits,
                valueMask);
  storeI32Value(builder, loc, ctrlBytePtr, valueRecordOffset, 24, maxBits,
                valueMask);
  storeI32Value(builder, loc, ctrlBytePtr, valueRecordOffset, 28, l2Bits,
                valueMask);
}

void emitSummaryCollectorStores(OpBuilder &builder, Location loc,
                                Operation *recordOp, Value ctrlBytePtr,
                                Value recordOffset, Value logicalInstanceId,
                                Value mask) {
  auto collectors = recordOp->getAttrOfType<ArrayAttr>("collectors");
  if (!collectors || collectors.empty())
    return;
  auto collectorName = dyn_cast<StringAttr>(collectors[0]);
  if (!collectorName)
    return;
  std::optional<uint32_t> collectorId =
      collectorIdForName(collectorName.getValue());
  if (!collectorId)
    return;

  Value observed = recordOp->getOperand(0);
  if (*collectorId == kCollectorElementCount) {
    emitSummaryU64Stores(
        builder, loc, recordOp, ctrlBytePtr, recordOffset, logicalInstanceId,
        *collectorId,
        createI64Constant(builder, loc,
                          getStaticElementCount(observed.getType())),
        mask);
    return;
  }

  Value f32Value = flattenTensorForSummary(
      builder, loc, castFloatValueToF32(builder, loc, observed));
  if (!f32Value)
    return;

  switch (*collectorId) {
  case kCollectorNanCount:
    emitSummaryU64Stores(builder, loc, recordOp, ctrlBytePtr, recordOffset,
                         logicalInstanceId, *collectorId,
                         computeNanCount(builder, loc, f32Value), mask);
    return;
  case kCollectorInfCount:
    emitSummaryU64Stores(builder, loc, recordOp, ctrlBytePtr, recordOffset,
                         logicalInstanceId, *collectorId,
                         computeInfCount(builder, loc, f32Value), mask);
    return;
  case kCollectorZeroCount:
    emitSummaryU64Stores(builder, loc, recordOp, ctrlBytePtr, recordOffset,
                         logicalInstanceId, *collectorId,
                         computeZeroCount(builder, loc, f32Value), mask);
    return;
  case kCollectorMeanFinite:
    emitSummaryF32Stores(builder, loc, recordOp, ctrlBytePtr, recordOffset,
                         logicalInstanceId, *collectorId,
                         computeMeanFinite(builder, loc, f32Value), mask);
    return;
  case kCollectorMinFinite:
    emitSummaryF32Stores(builder, loc, recordOp, ctrlBytePtr, recordOffset,
                         logicalInstanceId, *collectorId,
                         computeMinFinite(builder, loc, f32Value), mask);
    return;
  case kCollectorMaxFinite:
    emitSummaryF32Stores(builder, loc, recordOp, ctrlBytePtr, recordOffset,
                         logicalInstanceId, *collectorId,
                         computeMaxFinite(builder, loc, f32Value), mask);
    return;
  case kCollectorL2Norm:
    emitSummaryF32Stores(builder, loc, recordOp, ctrlBytePtr, recordOffset,
                         logicalInstanceId, *collectorId,
                         computeL2Norm(builder, loc, f32Value), mask);
    return;
  default:
    return;
  }
}

Value splatScalarToType(OpBuilder &builder, Location loc, Type type,
                        Value scalar) {
  if (isa<RankedTensorType>(type))
    return builder.create<triton::SplatOp>(loc, type, scalar);
  return scalar;
}

Value castIntegerValueToI64Like(OpBuilder &builder, Location loc, Value value,
                                Type targetType) {
  Value normalized = value;
  if (auto rankedTarget = dyn_cast<RankedTensorType>(targetType)) {
    if (isa<RankedTensorType>(value.getType())) {
      normalized = flattenTensorForSummary(builder, loc, value);
      if (!normalized)
        return {};
    } else {
      Type splatType = RankedTensorType::get(rankedTarget.getShape(),
                                             getElementType(value.getType()),
                                             rankedTarget.getEncoding());
      normalized = builder.create<triton::SplatOp>(loc, splatType, value);
    }
  }

  Type targetI64 = withElementType(targetType, builder.getI64Type());
  if (normalized.getType() == targetI64)
    return normalized;

  auto intType = dyn_cast<IntegerType>(getElementType(normalized.getType()));
  if (!intType)
    return {};

  unsigned width = intType.getWidth();
  if (width < 64)
    return builder.create<arith::ExtSIOp>(loc, targetI64, normalized);
  if (width > 64)
    return builder.create<arith::TruncIOp>(loc, targetI64, normalized);
  return normalized;
}

Value buildFlatAddressValue(OpBuilder &builder, Location loc,
                            Operation *recordOp) {
  if (recordOp->getNumOperands() == 0)
    return {};
  Value observed = recordOp->getOperand(0);
  if (!isPointerLikeType(observed.getType()))
    return {};

  std::optional<PointerAddressSlice> slice = tracePointerAddressSlice(observed);
  if (!slice || !slice->base)
    return {};

  Type addressType = builder.getI64Type();
  if (auto ranked = dyn_cast<RankedTensorType>(observed.getType())) {
    uint64_t count = getStaticElementCount(ranked);
    if (count == 0)
      return {};
    addressType =
        RankedTensorType::get({static_cast<int64_t>(count)},
                              builder.getI64Type(), ranked.getEncoding());
  }

  Value baseAddress = builder.create<triton::PtrToIntOp>(
      loc, builder.getI64Type(), slice->base);
  Value address = splatScalarToType(builder, loc, addressType, baseAddress);
  uint32_t bytes = 1;
  if (auto attr = recordOp->getAttrOfType<IntegerAttr>("access_bytes"))
    bytes = static_cast<uint32_t>(std::max<int64_t>(1, attr.getInt()));

  for (Value offset : slice->offsets) {
    Value offsetI64 =
        castIntegerValueToI64Like(builder, loc, offset, address.getType());
    if (!offsetI64)
      return {};
    Value byteScale =
        createIntegerConstantLike(builder, loc, offsetI64.getType(), bytes);
    Value byteOffset = builder.create<arith::MulIOp>(loc, offsetI64, byteScale);
    address = builder.create<arith::AddIOp>(loc, address, byteOffset);
  }

  return flattenTensorForSummary(builder, loc, address);
}

Value createPtrDialectToI64(OpBuilder &builder, Location loc, Value pointer) {
  OperationState state(loc, "tptr.ptrtoint");
  state.addOperands(pointer);
  state.addTypes(builder.getI64Type());
  Operation *op = builder.create(state);
  return op->getResult(0);
}

Value createMemRefBaseAddressToI64(OpBuilder &builder, Location loc,
                                   Value memrefValue) {
  OperationState state(loc, "memref.extract_aligned_pointer_as_index");
  state.addOperands(memrefValue);
  state.addTypes(builder.getIndexType());
  Operation *op = builder.create(state);
  return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                            op->getResult(0));
}

Value fallbackPointerAddressForMemoryEvent(OpBuilder &builder, Location loc,
                                           Operation *recordOp) {
  if (recordOp->getNumOperands() == 0)
    return {};
  Value observed = recordOp->getOperand(0);
  if (isMemRefPointerType(observed.getType()))
    return createMemRefBaseAddressToI64(builder, loc, observed);
  if (isPtrDialectPointerType(observed.getType()))
    return createPtrDialectToI64(builder, loc, observed);

  if (!isPointerLikeType(observed.getType()))
    return {};
  if (!isa<RankedTensorType>(observed.getType())) {
    return builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(),
                                              observed);
  }

  std::optional<PointerAddressSlice> slice = tracePointerAddressSlice(observed);
  if (!slice || !slice->base)
    return createI64Constant(builder, loc, 0);
  return builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(),
                                            slice->base);
}

Value extractLaneValue(OpBuilder &builder, Location loc, Value value,
                       int64_t lane) {
  Value flat = flattenTensorForSummary(builder, loc, value);
  if (!flat)
    return {};
  if (!isa<RankedTensorType>(flat.getType()))
    return flat;
  Value index = builder.create<arith::ConstantIndexOp>(loc, lane);
  return builder.create<tensor::ExtractOp>(loc, flat, ValueRange{index});
}

Value combineOffsetsToI64(OpBuilder &builder, Location loc, Type targetType,
                          ArrayRef<Value> offsets) {
  Value combined = createIntegerConstantLike(builder, loc, targetType, 0);
  for (Value offset : offsets) {
    Value offsetI64 =
        castIntegerValueToI64Like(builder, loc, offset, targetType);
    if (!offsetI64)
      return {};
    combined = builder.create<arith::AddIOp>(loc, combined, offsetI64);
  }
  return combined;
}

Value activeLaneCountForSummary(OpBuilder &builder, Location loc,
                                Operation *recordOp, int64_t laneCount) {
  if (recordOp->getNumOperands() < 2)
    return createI64Constant(builder, loc, static_cast<uint64_t>(laneCount));

  Value mask = recordOp->getOperand(1);
  if (!mask || !isa<IntegerType>(getElementType(mask.getType())))
    return createI64Constant(builder, loc, static_cast<uint64_t>(laneCount));
  if (isAllTrueMask(mask))
    return createI64Constant(builder, loc, static_cast<uint64_t>(laneCount));
  if (!isa<RankedTensorType>(mask.getType()))
    return builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), mask);
  Value flatMask = flattenTensorForSummary(builder, loc, mask);
  if (!flatMask)
    return createI64Constant(builder, loc, static_cast<uint64_t>(laneCount));
  return countTrueToI64(builder, loc, flatMask);
}

struct FastAddressSummaryValues {
  Value baseAddress;
  Value firstOffset;
  Value activeCount;
  Value hasActive;
  int64_t laneCount = 1;
  uint32_t bytes = 1;
};

std::optional<FastAddressSummaryValues>
buildFastAddressSummaryValues(OpBuilder &builder, Location loc,
                              Operation *recordOp) {
  if (recordOp->getNumOperands() == 0)
    return std::nullopt;
  Value observed = recordOp->getOperand(0);
  if (!isPointerLikeType(observed.getType()))
    return std::nullopt;

  std::optional<PointerAddressSlice> slice = tracePointerAddressSlice(observed);
  if (!slice || !slice->base || !hasContiguousAddressOffsets(*slice))
    return std::nullopt;
  Value mask =
      recordOp->getNumOperands() > 1 ? recordOp->getOperand(1) : Value();
  if (!isSupportedPrefixMask(mask, slice->offsets))
    return std::nullopt;

  uint32_t bytes = 1;
  if (auto attr = recordOp->getAttrOfType<IntegerAttr>("access_bytes"))
    bytes = static_cast<uint32_t>(std::max<int64_t>(1, attr.getInt()));

  int64_t laneCount = 1;
  Type offsetType = builder.getI64Type();
  if (auto ranked = dyn_cast<RankedTensorType>(observed.getType())) {
    laneCount = static_cast<int64_t>(getStaticElementCount(ranked));
    if (laneCount <= 0)
      return std::nullopt;
    offsetType = RankedTensorType::get({laneCount}, builder.getI64Type(),
                                       ranked.getEncoding());
  }

  Value combinedOffset =
      combineOffsetsToI64(builder, loc, offsetType, slice->offsets);
  if (!combinedOffset)
    return std::nullopt;
  Value firstOffset = extractLaneValue(builder, loc, combinedOffset, 0);
  if (!firstOffset)
    return std::nullopt;

  Value activeCount =
      activeLaneCountForSummary(builder, loc, recordOp, laneCount);
  Value hasActive =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, activeCount,
                                    createI64Constant(builder, loc, 0));
  Value baseAddress = builder.create<triton::PtrToIntOp>(
      loc, builder.getI64Type(), slice->base);

  return FastAddressSummaryValues{baseAddress, firstOffset, activeCount,
                                  hasActive,   laneCount,   bytes};
}

Value addressForElementOffset(OpBuilder &builder, Location loc,
                              const FastAddressSummaryValues &values,
                              Value offset) {
  Value byteOffset = builder.create<arith::MulIOp>(
      loc, offset, createI64Constant(builder, loc, values.bytes));
  return builder.create<arith::AddIOp>(loc, values.baseAddress, byteOffset);
}

Value selectOrZero(OpBuilder &builder, Location loc, Value condition,
                   Value value);

Value computeFastAddressSummaryValue(OpBuilder &builder, Location loc,
                                     Operation *recordOp, StringRef eventKind) {
  std::optional<FastAddressSummaryValues> values =
      buildFastAddressSummaryValues(builder, loc, recordOp);
  if (!values)
    return {};

  if (eventKind == kEventActiveLaneCount)
    return values->activeCount;
  if (eventKind == kEventAddressSpanBytes) {
    Value span = builder.create<arith::MulIOp>(
        loc, values->activeCount,
        createI64Constant(builder, loc, values->bytes));
    return selectOrZero(builder, loc, values->hasActive, span);
  }

  Value firstAddr =
      addressForElementOffset(builder, loc, *values, values->firstOffset);
  if (eventKind == kEventFirstAddr || eventKind == kEventMinAddr)
    return selectOrZero(builder, loc, values->hasActive, firstAddr);

  Value lastDelta = builder.create<arith::SubIOp>(
      loc, values->activeCount, createI64Constant(builder, loc, 1));
  Value lastOffset =
      builder.create<arith::AddIOp>(loc, values->firstOffset, lastDelta);
  Value lastAddr = addressForElementOffset(builder, loc, *values, lastOffset);
  if (eventKind == kEventLastAddr || eventKind == kEventMaxAddr)
    return selectOrZero(builder, loc, values->hasActive, lastAddr);

  return {};
}

Value maskForAddressValue(OpBuilder &builder, Location loc, Operation *recordOp,
                          Value address) {
  Type maskType = withElementType(address.getType(), builder.getI1Type());
  if (recordOp->getNumOperands() < 2)
    return createBoolConstantLike(builder, loc, maskType, true);

  Value mask = recordOp->getOperand(1);
  if (!mask || !isa<IntegerType>(getElementType(mask.getType())))
    return createBoolConstantLike(builder, loc, maskType, true);

  if (isa<RankedTensorType>(address.getType())) {
    if (isa<RankedTensorType>(mask.getType())) {
      Value flatMask = flattenTensorForSummary(builder, loc, mask);
      if (flatMask && getStaticElementCount(flatMask.getType()) ==
                          getStaticElementCount(address.getType()))
        return flatMask;
      return createBoolConstantLike(builder, loc, maskType, true);
    }
    return builder.create<triton::SplatOp>(loc, maskType, mask);
  }

  if (!isa<RankedTensorType>(mask.getType()))
    return mask;
  return createBoolConstant(builder, loc, true);
}

Value selectOrZero(OpBuilder &builder, Location loc, Value condition,
                   Value value) {
  return builder.create<arith::SelectOp>(
      loc, condition, value,
      createIntegerConstantLike(builder, loc, value.getType(), 0));
}

Value computeMaskedMinAddress(OpBuilder &builder, Location loc, Value address,
                              Value activeMask, Value hasActive) {
  Value maxSentinel = createIntegerConstantLike(
      builder, loc, address.getType(), std::numeric_limits<uint64_t>::max());
  Value maskedAddress =
      builder.create<arith::SelectOp>(loc, activeMask, address, maxSentinel);
  Value minAddress = reduceMinUI(builder, loc, maskedAddress);
  return selectOrZero(builder, loc, hasActive, minAddress);
}

Value computeMaskedMaxAddress(OpBuilder &builder, Location loc, Value address,
                              Value activeMask, Value hasActive) {
  Value zero = createIntegerConstantLike(builder, loc, address.getType(), 0);
  Value maskedAddress =
      builder.create<arith::SelectOp>(loc, activeMask, address, zero);
  Value maxAddress = reduceMaxUI(builder, loc, maskedAddress);
  return selectOrZero(builder, loc, hasActive, maxAddress);
}

Value computeFirstOrLastAddress(OpBuilder &builder, Location loc, Value address,
                                Value activeMask, Value hasActive, bool first) {
  if (!isa<RankedTensorType>(address.getType()))
    return selectOrZero(builder, loc, hasActive, address);

  auto ranked = cast<RankedTensorType>(address.getType());
  int64_t count = ranked.getDimSize(0);
  auto laneType = RankedTensorType::get({count}, builder.getI32Type(),
                                        ranked.getEncoding());
  Value lanes = builder.create<triton::MakeRangeOp>(
      loc, laneType, 0, static_cast<int32_t>(count));

  Value inactiveLane = createIntegerConstantLike(
      builder, loc, laneType, first ? static_cast<uint64_t>(count) : 0);
  Value selectedLanes =
      builder.create<arith::SelectOp>(loc, activeMask, lanes, inactiveLane);
  Value selectedLane = first ? reduceMinUI(builder, loc, selectedLanes)
                             : reduceMaxUI(builder, loc, selectedLanes);
  Value selectedLaneSplat =
      builder.create<triton::SplatOp>(loc, laneType, selectedLane);
  Value selectedLaneMask = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lanes, selectedLaneSplat);
  Value selectedActiveMask =
      builder.create<arith::AndIOp>(loc, selectedLaneMask, activeMask);
  Value maxSentinel = createIntegerConstantLike(
      builder, loc, address.getType(), std::numeric_limits<uint64_t>::max());
  Value selectedAddress = builder.create<arith::SelectOp>(
      loc, selectedActiveMask, address, maxSentinel);
  Value addressValue = reduceMinUI(builder, loc, selectedAddress);
  return selectOrZero(builder, loc, hasActive, addressValue);
}

Value computeAddressSummaryValue(OpBuilder &builder, Location loc,
                                 Operation *recordOp, StringRef eventKind) {
  if (Value fastValue =
          computeFastAddressSummaryValue(builder, loc, recordOp, eventKind))
    return fastValue;

  // Unsupported pointer/mask shapes should not take the dynamic summary path.
  // Return zero instead of building generic tensor pointer reductions, because
  // CANN9 currently compiles those i64 min/max reductions extremely slowly.
  return createI64Constant(builder, loc, 0);
}

Value memoryEventValue(OpBuilder &builder, Location loc, Operation *recordOp) {
  auto kindAttr = recordOp->getAttrOfType<StringAttr>("event_kind");
  StringRef eventKind = kindAttr ? kindAttr.getValue() : StringRef();
  if (isAddressSummaryEventKind(eventKind))
    return computeAddressSummaryValue(builder, loc, recordOp, eventKind);
  return fallbackPointerAddressForMemoryEvent(builder, loc, recordOp);
}

uint32_t memoryEventKindId(StringRef eventKind) {
  return llvm::StringSwitch<uint32_t>(eventKind)
      .Case(kEventBaseAlignedAddr, kMemoryEventBaseAlignedAddr)
      .Case(kEventFirstAddr, kMemoryEventFirstAddr)
      .Case(kEventLastAddr, kMemoryEventLastAddr)
      .Case(kEventMinAddr, kMemoryEventMinAddr)
      .Case(kEventMaxAddr, kMemoryEventMaxAddr)
      .Case(kEventActiveLaneCount, kMemoryEventActiveLaneCount)
      .Case(kEventAddressSpanBytes, kMemoryEventAddressSpanBytes)
      .Default(kMemoryEventLastAlignedAddr);
}

uint32_t memoryEventKindForRecord(Operation *recordOp) {
  auto kindAttr = recordOp->getAttrOfType<StringAttr>("event_kind");
  return memoryEventKindId(kindAttr ? kindAttr.getValue() : StringRef());
}

Value pointerAddressForMemoryEventWithKind(OpBuilder &builder, Location loc,
                                           Operation *recordOp,
                                           uint32_t &eventKind) {
  eventKind = memoryEventKindForRecord(recordOp);
  Value value = memoryEventValue(builder, loc, recordOp);
  if (!value)
    return createI64Constant(builder, loc, 0);
  return value;
}

bool hasPointerOperand(Operation *recordOp) {
  return recordOp->getNumOperands() > 0 &&
         isAddressCaptureType(recordOp->getOperand(0).getType());
}

void emitMemoryEventStores(OpBuilder &builder, Location loc,
                           Operation *recordOp, Value ctrlBytePtr,
                           Value recordOffset, Value observedAddr, Value mask) {
  storeI64(builder, loc, ctrlBytePtr, recordOffset, 16, observedAddr, mask);
  uint32_t ext0 = 0;
  if (auto attr = recordOp->getAttrOfType<IntegerAttr>("operand_index"))
    ext0 = static_cast<uint32_t>(std::max<int64_t>(0, attr.getInt()));
  storeI32(builder, loc, ctrlBytePtr, recordOffset, 28, ext0, mask);
}

Value canonicalizeFullDumpValue(OpBuilder &builder, Location loc,
                                Operation *recordOp) {
  auto kindAttr = recordOp->getAttrOfType<StringAttr>(kAttrFullDumpKind);
  StringRef kind =
      kindAttr ? kindAttr.getValue() : StringRef(kFullDumpKindValue);
  if (kind == kFullDumpKindMemoryAddress)
    return buildFlatAddressValue(builder, loc, recordOp);

  if (recordOp->getNumOperands() == 0)
    return {};
  Value value = flattenTensorForSummary(builder, loc, recordOp->getOperand(0));
  if (!value)
    return {};

  auto dtypeAttr =
      recordOp->getAttrOfType<StringAttr>(kAttrFullDumpArtifactDtype);
  StringRef dtype = dtypeAttr ? dtypeAttr.getValue() : StringRef();
  if (dtype == "float32")
    return castFloatValueToF32(builder, loc, value);
  if (dtype == "float64") {
    auto floatType = dyn_cast<FloatType>(getElementType(value.getType()));
    if (floatType && floatType.getWidth() == 64)
      return value;
    return {};
  }
  if (dtype == "int64") {
    Type i64Like = withElementType(value.getType(), builder.getI64Type());
    if (value.getType() == i64Like)
      return value;
    auto intType = dyn_cast<IntegerType>(getElementType(value.getType()));
    if (!intType || intType.getWidth() > 64)
      return {};
    if (intType.getWidth() == 1)
      return builder.create<arith::ExtUIOp>(loc, i64Like, value);
    return builder.create<arith::ExtSIOp>(loc, i64Like, value);
  }
  return {};
}

Value bitcastPayloadValueToI64(OpBuilder &builder, Location loc, Value value) {
  Type i64Like = withElementType(value.getType(), builder.getI64Type());
  if (value.getType() == i64Like)
    return value;
  auto floatType = dyn_cast<FloatType>(getElementType(value.getType()));
  if (floatType && floatType.getWidth() == 64)
    return builder.create<arith::BitcastOp>(loc, i64Like, value);
  return {};
}

Value payloadLaneOffsets(OpBuilder &builder, Location loc, Type valueType,
                         Value payloadWordOffset, uint32_t wordsPerElement) {
  auto ranked = dyn_cast<RankedTensorType>(valueType);
  if (!ranked)
    return payloadWordOffset;
  int64_t count = static_cast<int64_t>(getStaticElementCount(ranked));
  auto laneI32Type = RankedTensorType::get({count}, builder.getI32Type(),
                                           ranked.getEncoding());
  Value lanes = builder.create<triton::MakeRangeOp>(
      loc, laneI32Type, 0, static_cast<int32_t>(count));
  auto laneI64Type = RankedTensorType::get({count}, builder.getI64Type(),
                                           ranked.getEncoding());
  Value lanesI64 = builder.create<arith::ExtUIOp>(loc, laneI64Type, lanes);
  if (wordsPerElement != 1) {
    Value scale =
        createIntegerConstantLike(builder, loc, laneI64Type, wordsPerElement);
    lanesI64 = builder.create<arith::MulIOp>(loc, lanesI64, scale);
  }
  Value base =
      builder.create<triton::SplatOp>(loc, laneI64Type, payloadWordOffset);
  return builder.create<arith::AddIOp>(loc, base, lanesI64);
}

Value maskForPayloadStore(OpBuilder &builder, Location loc, Type valueType,
                          Value inBounds) {
  if (auto ranked = dyn_cast<RankedTensorType>(valueType)) {
    auto maskType = RankedTensorType::get(
        ranked.getShape(), builder.getI1Type(), ranked.getEncoding());
    return builder.create<triton::SplatOp>(loc, maskType, inBounds);
  }
  return inBounds;
}

void emitFullDumpPayloadStores(OpBuilder &builder, Location loc,
                               Operation *recordOp,
                               const RecordLoweringContext &ctx,
                               Value absolutePayloadOffsetBytes,
                               Value inBounds) {
  auto elementBytesAttr =
      recordOp->getAttrOfType<IntegerAttr>(kAttrFullDumpElementBytes);
  if (!elementBytesAttr)
    return;
  uint32_t elementBytes =
      static_cast<uint32_t>(std::max<int64_t>(1, elementBytesAttr.getInt()));
  Value value = canonicalizeFullDumpValue(builder, loc, recordOp);
  if (!value)
    return;
  Value payloadWordOffset = builder.create<arith::DivUIOp>(
      loc, absolutePayloadOffsetBytes, createI64Constant(builder, loc, 4));
  Value mask = maskForPayloadStore(builder, loc, value.getType(), inBounds);

  if (elementBytes == 4) {
    Value wordValue = value;
    if (isa<FloatType>(getElementType(value.getType()))) {
      Type i32Like = withElementType(value.getType(), builder.getI32Type());
      wordValue = builder.create<arith::BitcastOp>(loc, i32Like, value);
    }
    Value offsets =
        payloadLaneOffsets(builder, loc, value.getType(), payloadWordOffset, 1);
    Value ptr = addWordOffsetLike(builder, loc, ctx.ctrlBytePtr, offsets);
    storeValue(builder, loc, ptr, wordValue, mask);
    return;
  }

  if (elementBytes == 8) {
    Value i64Value = bitcastPayloadValueToI64(builder, loc, value);
    if (!i64Value)
      return;
    auto [low, high] = splitI64ToI32Words(builder, loc, i64Value);
    Value lowOffsets = payloadLaneOffsets(builder, loc, i64Value.getType(),
                                          payloadWordOffset, 2);
    Value one =
        createIntegerConstantLike(builder, loc, lowOffsets.getType(), 1);
    Value highOffsets = builder.create<arith::AddIOp>(loc, lowOffsets, one);
    storeValue(builder, loc,
               addWordOffsetLike(builder, loc, ctx.ctrlBytePtr, lowOffsets),
               low, mask);
    storeValue(builder, loc,
               addWordOffsetLike(builder, loc, ctx.ctrlBytePtr, highOffsets),
               high, mask);
  }
}

void emitFullValueRefStores(OpBuilder &builder, Location loc,
                            Operation *recordOp,
                            const RecordLoweringContext &ctx,
                            Value recordOffset, Value inBounds) {
  auto payloadLengthAttr =
      recordOp->getAttrOfType<IntegerAttr>(kAttrFullDumpPayloadLength);
  auto payloadOffsetAttr =
      recordOp->getAttrOfType<IntegerAttr>(kAttrFullDumpPayloadOffset);
  if (!payloadLengthAttr || !payloadOffsetAttr)
    return;

  Value absolutePayloadOffsetBytes = builder.create<arith::AddIOp>(
      loc, ctx.payloadOffsetBytes, ctx.payloadInstanceBaseBytes);
  absolutePayloadOffsetBytes = builder.create<arith::AddIOp>(
      loc, absolutePayloadOffsetBytes,
      createI64Constant(builder, loc,
                        static_cast<uint64_t>(payloadOffsetAttr.getInt())));
  Value payloadOffsetI32 = builder.create<arith::TruncIOp>(
      loc, builder.getI32Type(), absolutePayloadOffsetBytes);
  storeI32Value(builder, loc, ctx.ctrlBytePtr, recordOffset, 16,
                payloadOffsetI32, inBounds);
  storeI32(builder, loc, ctx.ctrlBytePtr, recordOffset, 20,
           static_cast<uint32_t>(payloadLengthAttr.getInt()), inBounds);
  emitFullDumpPayloadStores(builder, loc, recordOp, ctx,
                            absolutePayloadOffsetBytes, inBounds);
}

void emitTimelineStores(OpBuilder &builder, Location loc, Operation *recordOp,
                        Value ctrlBytePtr, Value recordOffset, Value inBounds) {
  Value startCycle = recordOp->getOperand(0);
  Value endCycle = recordOp->getOperand(1);
  if (startCycle.getType() != builder.getI64Type())
    startCycle =
        builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), startCycle);
  if (endCycle.getType() != builder.getI64Type())
    endCycle =
        builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), endCycle);
  Value durationCycle =
      builder.create<arith::SubIOp>(loc, endCycle, startCycle);
  auto [startLow, startHigh] = splitI64ToI32Words(builder, loc, startCycle);
  auto [endLow, endHigh] = splitI64ToI32Words(builder, loc, endCycle);
  auto [durationLow, durationHigh] =
      splitI64ToI32Words(builder, loc, durationCycle);
  Value zero = createI32Constant(builder, loc, 0);
  SmallVector<Value, kRecordWords> words = {
      zero,   zero,    zero,        zero,         startLow, startHigh,
      endLow, endHigh, durationLow, durationHigh, zero,     zero,
      zero,   zero,    zero,        zero,
  };
  storeRecordWords(builder, loc, ctrlBytePtr, recordOffset, words, inBounds);
}

bool canLowerRecordOp(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  if (opName == kRecordSummaryOpName)
    return op->getNumOperands() == 1 && recordSummaryCanLower(op) &&
           getStaticElementCount(op->getOperand(0).getType()) != 0 &&
           op->getAttrOfType<IntegerAttr>("op_id");
  if (opName == kRecordSummaryBundleOpName)
    return op->getNumOperands() == 1 && recordSummaryCanLower(op) &&
           isFloatValueType(op->getOperand(0).getType()) &&
           getStaticElementCount(op->getOperand(0).getType()) != 0 &&
           op->getAttrOfType<IntegerAttr>("op_id");
  if (opName == kRecordMemoryEventOpName)
    return op->getAttrOfType<IntegerAttr>("op_id") && hasPointerOperand(op);
  if (opName == kCaptureMemoryAddressOpName)
    return op->getAttrOfType<IntegerAttr>("op_id") && hasPointerOperand(op);
  if (opName == kRecordFullValueRefOpName)
    return op->getNumOperands() == 1 && op->getAttrOfType<IntegerAttr>("op_id");
  if (opName == kRecordTimelineOpName)
    return op->getNumOperands() == 2 && op->getAttrOfType<IntegerAttr>("op_id");
  return false;
}

void lowerOneRecordOp(OpBuilder &builder, Operation *recordOp,
                      const RecordLoweringContext &ctx) {
  Location loc = recordOp->getLoc();
  builder.setInsertionPoint(recordOp);
  StringRef opName = recordOp->getName().getStringRef();
  if (opName == kRecordSummaryBundleOpName) {
    ReservedRecordSlot countSlot =
        reserveDeterministicRecordSlot(builder, loc, recordOp, ctx);
    ReservedRecordSlot valueSlot = reserveDeterministicRecordSlot(
        builder, loc, recordOp, ctx, /*indexOffset=*/1);
    emitSummaryBundleStores(builder, loc, recordOp, ctx.ctrlBytePtr,
                            countSlot.recordOffset, valueSlot.recordOffset,
                            countSlot.inBounds, valueSlot.inBounds);
  } else if (opName == kRecordSummaryOpName) {
    ReservedRecordSlot slot =
        reserveDeterministicRecordSlot(builder, loc, recordOp, ctx);
    emitSummaryCollectorStores(builder, loc, recordOp, ctx.ctrlBytePtr,
                               slot.recordOffset, ctx.logicalInstanceId,
                               slot.inBounds);
  } else if (opName == kRecordMemoryEventOpName ||
             opName == kCaptureMemoryAddressOpName) {
    ReservedRecordSlot slot =
        reserveDeterministicRecordSlot(builder, loc, recordOp, ctx);
    uint32_t eventKind = kMemoryEventLastAlignedAddr;
    Value addr =
        pointerAddressForMemoryEventWithKind(builder, loc, recordOp, eventKind);
    emitMemoryEventStores(builder, loc, recordOp, ctx.ctrlBytePtr,
                          slot.recordOffset, addr, slot.inBounds);
  } else if (opName == kRecordFullValueRefOpName) {
    ReservedRecordSlot slot =
        reserveDeterministicRecordSlot(builder, loc, recordOp, ctx);
    emitFullValueRefStores(builder, loc, recordOp, ctx, slot.recordOffset,
                           slot.inBounds);
  } else if (opName == kRecordTimelineOpName) {
    ReservedRecordSlot slot =
        reserveDeterministicRecordSlot(builder, loc, recordOp, ctx);
    emitTimelineStores(builder, loc, recordOp, ctx.ctrlBytePtr,
                       slot.recordOffset, slot.inBounds);
  }

  recordOp->setAttr(kAttrDeviceLowered, builder.getBoolAttr(true));
  recordOp->erase();
}

void lowerRecordOpsWithHiddenArg(ModuleOp module, OpBuilder &builder) {
  struct LowerableRecordOp {
    Operation *op = nullptr;
    Value hiddenArg;
    FunctionOpInterface func;
  };
  llvm::SmallVector<LowerableRecordOp> lowerable;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != kRecordSummaryOpName &&
        op->getName().getStringRef() != kRecordSummaryBundleOpName &&
        op->getName().getStringRef() != kRecordMemoryEventOpName &&
        op->getName().getStringRef() != kCaptureMemoryAddressOpName &&
        op->getName().getStringRef() != kRecordFullValueRefOpName &&
        op->getName().getStringRef() != kRecordTimelineOpName)
      return;
    if (!canLowerRecordOp(op))
      return;
    FunctionOpInterface func;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      func = dyn_cast<FunctionOpInterface>(parent);
      if (func)
        break;
    }
    if (!func)
      return;
    Value hiddenArg = hiddenDebugArgument(func);
    if (!hiddenArg)
      return;
    lowerable.push_back(LowerableRecordOp{op, hiddenArg, func});
  });

  llvm::DenseMap<Operation *, unsigned> contextIndexByFunc;
  llvm::SmallVector<RecordLoweringContext> contexts;
  for (LowerableRecordOp &item : lowerable) {
    Operation *funcOp = item.func.getOperation();
    auto found = contextIndexByFunc.find(funcOp);
    if (found == contextIndexByFunc.end()) {
      const unsigned contextIndex = static_cast<unsigned>(contexts.size());
      uint64_t payloadBytesPerInstance = 0;
      for (Operation *cursor = item.op; cursor;
           cursor = cursor->getParentOp()) {
        if (auto attr = cursor->getAttrOfType<IntegerAttr>(
                kAttrFullDumpPayloadBytesPerInstance)) {
          payloadBytesPerInstance =
              static_cast<uint64_t>(std::max<int64_t>(0, attr.getInt()));
          break;
        }
      }
      contexts.push_back(createRecordLoweringContext(
          builder, item.func, item.hiddenArg, getRecordsPerInstance(item.op),
          payloadBytesPerInstance));
      found = contextIndexByFunc.insert({funcOp, contextIndex}).first;
    }
    lowerOneRecordOp(builder, item.op, contexts[found->second]);
  }
}

RecordLevel parseRecordLevelAttr(Attribute attr) {
  if (!attr) {
    return RecordLevel::LEVEL_SUMMARY;
  }

  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
    switch (intAttr.getInt()) {
    case 2:
      return RecordLevel::LEVEL_TENSOR_FULL;
    case 1:
    default:
      return RecordLevel::LEVEL_SUMMARY;
    }
  }

  if (auto strAttr = mlir::dyn_cast<StringAttr>(attr)) {
    if (strAttr.getValue() == "LEVEL_TENSOR_FULL" ||
        strAttr.getValue() == "2") {
      return RecordLevel::LEVEL_TENSOR_FULL;
    }
  }

  return RecordLevel::LEVEL_SUMMARY;
}

RecordLevel getRecordLevel(Operation *op) {
  for (Operation *cursor = op; cursor; cursor = cursor->getParentOp()) {
    if (Attribute attr = cursor->getAttr(kAttrRecordLevel)) {
      return parseRecordLevelAttr(attr);
    }
    if (Attribute attr = cursor->getAttr(kAttrFallbackRecordLevel)) {
      return parseRecordLevelAttr(attr);
    }
  }
  return RecordLevel::LEVEL_SUMMARY;
}

int32_t parseAddrLevelAttr(Attribute attr) {
  if (!attr)
    return 0;
  int64_t value = 0;
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(attr)) {
    value = intAttr.getInt();
  } else if (auto strAttr = mlir::dyn_cast<StringAttr>(attr)) {
    if (strAttr.getValue().getAsInteger(10, value))
      value = 0;
  }
  return static_cast<int32_t>(std::clamp<int64_t>(value, 0, 2));
}

int32_t getAddrLevel(Operation *op) {
  for (Operation *cursor = op; cursor; cursor = cursor->getParentOp()) {
    if (Attribute attr = cursor->getAttr(kAttrAddrLevel))
      return parseAddrLevelAttr(attr);
    if (Attribute attr = cursor->getAttr(kAttrFallbackAddrLevel))
      return parseAddrLevelAttr(attr);
  }
  return 0;
}

ArrayAttr buildRecordKindArray(Builder &builder, bool hasSummary,
                               bool hasMemoryEvent, bool hasFullValueRef,
                               bool hasTimeline) {
  SmallVector<Attribute> recordKinds;
  if (hasTimeline) {
    recordKinds.push_back(builder.getStringAttr("timeline"));
  }
  if (hasSummary) {
    recordKinds.push_back(builder.getStringAttr("summary"));
  }
  if (hasMemoryEvent) {
    recordKinds.push_back(builder.getStringAttr("memory_event"));
  }
  if (hasFullValueRef) {
    recordKinds.push_back(builder.getStringAttr("full_value"));
  }
  return builder.getArrayAttr(recordKinds);
}

void annotateFunction(FunctionOpInterface func, Builder &builder) {
  func->setAttr(kAttrHiddenArg, builder.getStringAttr(kHiddenArgName));
  func->setAttr(kAttrLogicalInstanceFormula,
                builder.getStringAttr(kLogicalInstanceFormula));
}

LogicalResult ensureHiddenDebugArgument(FunctionOpInterface func,
                                        Builder &builder,
                                        bool enableHiddenArgAbi) {
  if (!enableHiddenArgAbi)
    return success();

  if (func->getName().getStringRef() != "tt.func")
    return success();

  if (func->hasAttr(kAttrHiddenArgIndex))
    return success();

  const unsigned argIndex = func.getNumArguments();
  Type hiddenArgType = triton::PointerType::get(builder.getI32Type(), 1);
  auto hiddenArgAttrs = builder.getDictionaryAttr({builder.getNamedAttr(
      kAttrHiddenArg, builder.getStringAttr(kHiddenArgName))});
  if (failed(func.insertArgument(argIndex, hiddenArgType, hiddenArgAttrs,
                                 func.getLoc()))) {
    func.emitError("failed to insert the debugger hidden argument");
    return failure();
  }
  func->setAttr(kAttrHiddenArgIndex,
                builder.getI32IntegerAttr(static_cast<int32_t>(argIndex)));
  func->setAttr(kAttrHiddenArgType, builder.getStringAttr("!tt.ptr<i32>"));
  return success();
}

struct InstrumentationTarget {
  Operation *op = nullptr;
  Value observedValue;
  int32_t opId = 0;
  int32_t scopeId = 0;
  RecordLevel level = RecordLevel::LEVEL_SUMMARY;
  int32_t addrLevel = 0;
  bool hasSummary = false;
  bool hasMemoryEvent = false;
  bool hasFullValueRef = false;
  bool hasFullAddressRef = false;
  bool hasTimeline = false;
  ArrayAttr collectors;
  FullDumpSpec valueDump;
  FullDumpSpec addressDump;
};

struct RecordPlanEntry {
  int32_t recordIndex = 0;
  int32_t opId = 0;
  int32_t scopeId = 0;
  uint32_t recordKind = 0;
  uint32_t collectorKind = 0;
  uint32_t resultType = 0;
  uint32_t eventKind = 0;
};

struct FullDumpPlanEntry {
  int32_t recordIndex = 0;
  int32_t opId = 0;
  int32_t scopeId = 0;
  std::string kind;
  std::string source;
  std::string artifactDtype;
  uint64_t elementCount = 0;
  uint32_t elementBytes = 0;
  uint64_t payloadOffset = 0;
  uint64_t payloadLength = 0;
};

std::string serializeRecordPlanToJson(ArrayRef<RecordPlanEntry> entries) {
  llvm::json::Array array;
  for (const RecordPlanEntry &entry : entries) {
    array.push_back(llvm::json::Object{
        {"record_index", entry.recordIndex},
        {"op_id", entry.opId},
        {"scope_id", entry.scopeId},
        {"record_kind", entry.recordKind},
        {"collector_kind", entry.collectorKind},
        {"result_type", entry.resultType},
        {"event_kind", entry.eventKind},
    });
  }

  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::json::Value(std::move(array));
  return text;
}

bool isDotOp(Operation *op) {
  return op && op->getName().getStringRef() == "tt.dot";
}

Value createDynamicFloatOneLike(OpBuilder &builder, Location loc, Type type) {
  Type elementType = getElementType(type);
  auto floatType = dyn_cast<FloatType>(elementType);
  if (!floatType)
    return {};

  Value numPrograms = builder.create<triton::GetNumProgramsOp>(loc, 0);
  Value oneI32 = builder.create<arith::DivUIOp>(loc, numPrograms, numPrograms);
  Value one = builder.create<arith::UIToFPOp>(loc, floatType, oneI32);
  if (isa<ShapedType>(type))
    return builder.create<triton::SplatOp>(loc, type, one);
  return one;
}

Value createDotIdentityProxy(OpBuilder &builder,
                             const InstrumentationTarget &target,
                             Operation *&anchor) {
  Value observed = target.observedValue;
  if (!observed)
    return {};
  if (!isDotOp(target.op) || target.op->getNumResults() == 0)
    return observed;

  Value dotResult = target.op->getResult(0);
  if (observed != dotResult)
    return observed;
  if (!isFloatValueType(dotResult.getType()))
    return dotResult;

  builder.setInsertionPointAfter(target.op);
  Value one = createDynamicFloatOneLike(builder, target.op->getLoc(),
                                        dotResult.getType());
  if (!one)
    return dotResult;

  // Ascend CANN9 cannot safely observe a raw tt.dot/cube accumulator through a
  // debugger side-use: it may force an unsupported UB memref.alloc during HIVM
  // lowering.  Route the value through an identity epilogue op and keep the
  // metadata attached to the original dot op_id, so reports still show tt.dot.
  Value proxy =
      builder.create<arith::MulFOp>(target.op->getLoc(), dotResult, one);
  proxy.getDefiningOp()->setAttr("flagtree.debug.proxy_for_op_id",
                                 builder.getI32IntegerAttr(target.opId));
  dotResult.replaceUsesWithIf(proxy, [&](OpOperand &use) {
    return use.getOwner() != proxy.getDefiningOp();
  });
  anchor = proxy.getDefiningOp();
  return proxy;
}

std::string serializeFullDumpPlanToJson(ArrayRef<FullDumpPlanEntry> entries) {
  llvm::json::Array array;
  for (const FullDumpPlanEntry &entry : entries) {
    llvm::json::Array shape;
    shape.push_back(static_cast<int64_t>(entry.elementCount));
    array.push_back(llvm::json::Object{
        {"record_index", entry.recordIndex},
        {"op_id", entry.opId},
        {"scope_id", entry.scopeId},
        {"kind", entry.kind},
        {"source", entry.source},
        {"artifact_dtype", entry.artifactDtype},
        {"shape", std::move(shape)},
        {"element_count", static_cast<int64_t>(entry.elementCount)},
        {"element_bytes", static_cast<int64_t>(entry.elementBytes)},
        {"payload_offset", static_cast<int64_t>(entry.payloadOffset)},
        {"payload_length", static_cast<int64_t>(entry.payloadLength)},
    });
  }

  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::json::Value(std::move(array));
  return text;
}

uint64_t
appendFullDumpPlanEntry(llvm::SmallVectorImpl<FullDumpPlanEntry> &fullDumpPlan,
                        int32_t recordIndex,
                        const InstrumentationTarget &target,
                        const FullDumpSpec &spec, uint64_t nextPayloadOffset) {
  uint64_t alignedOffset = alignTo(nextPayloadOffset, spec.elementBytes);
  uint64_t payloadLength =
      spec.elementCount * static_cast<uint64_t>(spec.elementBytes);
  fullDumpPlan.push_back(FullDumpPlanEntry{
      recordIndex,
      target.opId,
      target.scopeId,
      spec.kind,
      spec.source,
      spec.artifactDtype,
      spec.elementCount,
      spec.elementBytes,
      alignedOffset,
      payloadLength,
  });
  return alignedOffset + payloadLength;
}

const FullDumpPlanEntry *
findFullDumpPlanEntry(ArrayRef<FullDumpPlanEntry> fullDumpPlan,
                      int32_t recordIndex) {
  for (const FullDumpPlanEntry &entry : fullDumpPlan) {
    if (entry.recordIndex == recordIndex)
      return &entry;
  }
  return nullptr;
}

Operation *createRecordSummaryOp(OpBuilder &builder, Operation *anchor,
                                 const InstrumentationTarget &target,
                                 StringAttr collector, int32_t recordIndex) {
  if (!target.op || !target.observedValue)
    return anchor;

  Value summaryValue = createDotIdentityProxy(builder, target, anchor);
  OperationState state(target.op->getLoc(), kRecordSummaryOpName);
  state.addOperands(summaryValue);
  state.addAttribute("op_id", builder.getI32IntegerAttr(target.opId));
  state.addAttribute("scope_id", builder.getI32IntegerAttr(target.scopeId));
  state.addAttribute("record_level", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(target.level)));
  state.addAttribute("collectors", builder.getArrayAttr({collector}));
  state.addAttribute("result_index", builder.getI32IntegerAttr(0));
  state.addAttribute(kAttrRecordIndex, builder.getI32IntegerAttr(recordIndex));

  builder.setInsertionPointAfter(anchor);
  return builder.create(state);
}

Operation *createRecordSummaryBundleOp(OpBuilder &builder, Operation *anchor,
                                       const InstrumentationTarget &target,
                                       int32_t recordIndex) {
  if (!target.op || !target.observedValue)
    return anchor;

  Value summaryValue = createDotIdentityProxy(builder, target, anchor);
  OperationState state(target.op->getLoc(), kRecordSummaryBundleOpName);
  state.addOperands(summaryValue);
  state.addAttribute("op_id", builder.getI32IntegerAttr(target.opId));
  state.addAttribute("scope_id", builder.getI32IntegerAttr(target.scopeId));
  state.addAttribute("record_level", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(target.level)));
  state.addAttribute("collectors", target.collectors);
  state.addAttribute("result_index", builder.getI32IntegerAttr(0));
  state.addAttribute(kAttrRecordIndex, builder.getI32IntegerAttr(recordIndex));

  builder.setInsertionPointAfter(anchor);
  return builder.create(state);
}

Operation *
createCaptureMemoryAddressOp(OpBuilder &builder, Operation *anchor,
                             const InstrumentationTarget &target,
                             const MemoryAddressTarget &addressTarget,
                             StringRef eventKind, int32_t recordIndex) {
  Value pointer = addressTarget.pointer;
  if (!pointer)
    return anchor;

  OperationState state(target.op->getLoc(), kCaptureMemoryAddressOpName);
  state.addOperands(pointer);
  if (isAddressSummaryEventKind(eventKind)) {
    if (Value mask = memoryMaskOperand(target.op))
      state.addOperands(mask);
  }
  state.addAttribute("op_id", builder.getI32IntegerAttr(target.opId));
  state.addAttribute("scope_id", builder.getI32IntegerAttr(target.scopeId));
  state.addAttribute("operand_index",
                     builder.getI32IntegerAttr(
                         static_cast<int32_t>(addressTarget.operandIndex)));
  state.addAttribute("event_kind", builder.getStringAttr(eventKind));
  state.addAttribute("access_bytes",
                     builder.getI32IntegerAttr(accessBytes(target.op)));
  state.addAttribute("lowering_policy",
                     builder.getStringAttr(isAddressSummaryEventKind(eventKind)
                                               ? "cann9_address_summary"
                                               : "backend_sensitive"));
  state.addAttribute(kAttrRecordIndex, builder.getI32IntegerAttr(recordIndex));

  builder.setInsertionPointAfter(anchor);
  return builder.create(state);
}

Operation *createRecordFullValueRefOp(OpBuilder &builder, Operation *anchor,
                                      const InstrumentationTarget &target,
                                      int32_t recordIndex,
                                      const FullDumpPlanEntry &planEntry) {
  if (!target.op)
    return anchor;

  Value observed;
  if (planEntry.kind == kFullDumpKindMemoryAddress.str() &&
      planEntry.source == kFullDumpSourceAddress.str()) {
    observed = memoryPointerOperand(target.op);
  } else if (planEntry.kind == kFullDumpKindValue.str() &&
             planEntry.source == kFullDumpSourceResult.str()) {
    observed = createDotIdentityProxy(builder, target, anchor);
  } else {
    observed = target.observedValue;
    if (!observed) {
      StringRef ignoredSource;
      observed = observedFullDumpValue(target.op, ignoredSource);
    }
  }
  if (!observed)
    return anchor;

  OperationState state(target.op->getLoc(), kRecordFullValueRefOpName);
  state.addOperands(observed);
  state.addAttribute("op_id", builder.getI32IntegerAttr(target.opId));
  state.addAttribute("scope_id", builder.getI32IntegerAttr(target.scopeId));
  state.addAttribute("record_level", builder.getI32IntegerAttr(
                                         static_cast<int32_t>(target.level)));
  state.addAttribute("result_index", builder.getI32IntegerAttr(0));
  state.addAttribute(kAttrFullDumpKind,
                     builder.getStringAttr(StringRef(planEntry.kind)));
  state.addAttribute(kAttrFullDumpSource,
                     builder.getStringAttr(StringRef(planEntry.source)));
  state.addAttribute(kAttrFullDumpArtifactDtype,
                     builder.getStringAttr(StringRef(planEntry.artifactDtype)));
  state.addAttribute(
      kAttrFullDumpElementCount,
      builder.getI64IntegerAttr(static_cast<int64_t>(planEntry.elementCount)));
  state.addAttribute(
      kAttrFullDumpElementBytes,
      builder.getI32IntegerAttr(static_cast<int32_t>(planEntry.elementBytes)));
  state.addAttribute(
      kAttrFullDumpPayloadOffset,
      builder.getI64IntegerAttr(static_cast<int64_t>(planEntry.payloadOffset)));
  state.addAttribute(
      kAttrFullDumpPayloadLength,
      builder.getI64IntegerAttr(static_cast<int64_t>(planEntry.payloadLength)));
  if (planEntry.kind == kFullDumpKindMemoryAddress.str()) {
    state.addAttribute("access_bytes",
                       builder.getI32IntegerAttr(accessBytes(target.op)));
  }
  state.addAttribute(kAttrRecordIndex, builder.getI32IntegerAttr(recordIndex));

  builder.setInsertionPointAfter(anchor);
  return builder.create(state);
}

Value createDeviceCycleRead(OpBuilder &builder, Location loc) {
  OperationState state(loc, "tt.elementwise_inline_asm");
  state.addAttribute("asm_string", builder.getStringAttr("MOV $0, SYS_CNT"));
  state.addAttribute("constraints", builder.getStringAttr("=l"));
  state.addAttribute("pure", builder.getBoolAttr(false));
  state.addAttribute("packed_element", builder.getI32IntegerAttr(1));
  state.addTypes(builder.getI64Type());
  return builder.create(state)->getResult(0);
}

Operation *createRecordTimelineOp(OpBuilder &builder, Operation *anchor,
                                  const InstrumentationTarget &target,
                                  int32_t recordIndex) {
  if (!target.op)
    return anchor;

  builder.setInsertionPoint(target.op);
  Value startCycle = createDeviceCycleRead(builder, target.op->getLoc());

  builder.setInsertionPointAfter(anchor);
  Value endCycle = createDeviceCycleRead(builder, target.op->getLoc());

  OperationState state(target.op->getLoc(), kRecordTimelineOpName);
  state.addOperands({startCycle, endCycle});
  state.addAttribute("op_id", builder.getI32IntegerAttr(target.opId));
  state.addAttribute("scope_id", builder.getI32IntegerAttr(target.scopeId));
  state.addAttribute(kAttrRecordIndex, builder.getI32IntegerAttr(recordIndex));
  builder.setInsertionPointAfter(endCycle.getDefiningOp());
  return builder.create(state);
}

uint32_t resultTypeForCollector(uint32_t collectorKind) {
  switch (collectorKind) {
  case kCollectorNanCount:
  case kCollectorInfCount:
  case kCollectorElementCount:
  case kCollectorZeroCount:
    return kResultTypeU64;
  case kCollectorMeanFinite:
  case kCollectorMinFinite:
  case kCollectorMaxFinite:
  case kCollectorL2Norm:
    return kResultTypeF32;
  default:
    return 0;
  }
}

void appendRecordPlanEntry(llvm::SmallVectorImpl<RecordPlanEntry> &recordPlan,
                           int32_t recordIndex,
                           const InstrumentationTarget &target,
                           uint32_t recordKind, uint32_t collectorKind = 0,
                           uint32_t resultType = 0, uint32_t eventKind = 0) {
  recordPlan.push_back(RecordPlanEntry{recordIndex, target.opId, target.scopeId,
                                       recordKind, collectorKind, resultType,
                                       eventKind});
}

void insertRecordOps(OpBuilder &builder, const InstrumentationTarget &target,
                     int32_t &nextRecordIndex,
                     llvm::SmallVectorImpl<RecordPlanEntry> &recordPlan,
                     llvm::SmallVectorImpl<FullDumpPlanEntry> &fullDumpPlan,
                     uint64_t &nextPayloadOffset) {
  if (!target.op || target.op->hasTrait<OpTrait::IsTerminator>())
    return;

  Operation *anchor = target.op;
  if (target.hasTimeline) {
    appendRecordPlanEntry(recordPlan, nextRecordIndex, target,
                          kRecordKindTimeline);
    anchor = createRecordTimelineOp(builder, anchor, target, nextRecordIndex++);
  }
  if (target.hasSummary) {
    if (target.observedValue &&
        isFloatValueType(target.observedValue.getType())) {
      appendRecordPlanEntry(recordPlan, nextRecordIndex, target,
                            kRecordKindSummaryCountBundleU64);
      appendRecordPlanEntry(recordPlan, nextRecordIndex + 1, target,
                            kRecordKindSummaryValueBundleF32);
      anchor =
          createRecordSummaryBundleOp(builder, anchor, target, nextRecordIndex);
      nextRecordIndex += 2;
    } else {
      for (Attribute attr : target.collectors) {
        auto collector = dyn_cast<StringAttr>(attr);
        if (!collector)
          continue;
        std::optional<uint32_t> collectorKind =
            collectorIdForName(collector.getValue());
        appendRecordPlanEntry(
            recordPlan, nextRecordIndex, target, kRecordKindSummary,
            collectorKind.value_or(0),
            collectorKind ? resultTypeForCollector(*collectorKind) : 0);
        anchor = createRecordSummaryOp(builder, anchor, target, collector,
                                       nextRecordIndex++);
      }
    }
  }
  if (target.hasMemoryEvent) {
    SmallVector<MemoryAddressTarget> addressTargets =
        memoryAddressTargets(target.op);
    for (const MemoryAddressTarget &addressTarget : addressTargets) {
      bool emitAddressSummary =
          target.addrLevel >= 1 &&
          canComputeAddressSummaryForMemoryTarget(target.op, addressTarget);
      if (emitAddressSummary && target.level == RecordLevel::LEVEL_SUMMARY) {
        uint64_t elementCount =
            getStaticElementCount(addressTarget.pointer.getType());
        emitAddressSummary = elementCount != 0 &&
                             elementCount <= kLevel1AddressSummaryElementLimit;
      }
      if (emitAddressSummary) {
        StringRef summaryKinds[] = {
            kEventFirstAddr, kEventLastAddr,        kEventMinAddr,
            kEventMaxAddr,   kEventActiveLaneCount, kEventAddressSpanBytes,
        };
        for (StringRef eventKind : summaryKinds) {
          appendRecordPlanEntry(recordPlan, nextRecordIndex, target,
                                kRecordKindMemoryEvent, /*collectorKind=*/0,
                                /*resultType=*/0, memoryEventKindId(eventKind));
          anchor = createCaptureMemoryAddressOp(builder, anchor, target,
                                                addressTarget, eventKind,
                                                nextRecordIndex++);
        }
      } else {
        StringRef eventKind = memoryEventKindForPointer(addressTarget.pointer);
        appendRecordPlanEntry(recordPlan, nextRecordIndex, target,
                              kRecordKindMemoryEvent, /*collectorKind=*/0,
                              /*resultType=*/0, memoryEventKindId(eventKind));
        anchor =
            createCaptureMemoryAddressOp(builder, anchor, target, addressTarget,
                                         eventKind, nextRecordIndex++);
      }
    }
  }
  if (target.hasFullValueRef) {
    int32_t recordIndex = nextRecordIndex;
    nextPayloadOffset = appendFullDumpPlanEntry(
        fullDumpPlan, recordIndex, target, target.valueDump, nextPayloadOffset);
    appendRecordPlanEntry(recordPlan, nextRecordIndex, target,
                          kRecordKindFullValue);
    const FullDumpPlanEntry *planEntry =
        findFullDumpPlanEntry(fullDumpPlan, recordIndex);
    if (planEntry)
      anchor = createRecordFullValueRefOp(builder, anchor, target, recordIndex,
                                          *planEntry);
    ++nextRecordIndex;
  }
  if (target.hasFullAddressRef) {
    int32_t recordIndex = nextRecordIndex;
    nextPayloadOffset =
        appendFullDumpPlanEntry(fullDumpPlan, recordIndex, target,
                                target.addressDump, nextPayloadOffset);
    appendRecordPlanEntry(recordPlan, nextRecordIndex, target,
                          kRecordKindFullValue);
    const FullDumpPlanEntry *planEntry =
        findFullDumpPlanEntry(fullDumpPlan, recordIndex);
    if (planEntry)
      createRecordFullValueRefOp(builder, anchor, target, recordIndex,
                                 *planEntry);
    ++nextRecordIndex;
  }
}

bool isI32Scalar(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  return intType && intType.getWidth() == 32;
}

bool isSingleElementI32Tensor(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  return ranked && ranked.hasStaticShape() && ranked.getRank() == 1 &&
         ranked.getDimSize(0) == 1 && isI32Scalar(ranked.getElementType());
}

bool isSingleElementI32MemRef(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.hasStaticShape() && memref.getRank() == 1 &&
         memref.getDimSize(0) == 1 && isI32Scalar(memref.getElementType());
}

bool isDebugMemrefViewOp(Operation *op) {
  if (!op || op->getNumOperands() == 0 || op->getNumResults() != 1)
    return false;
  StringRef name = op->getName().getStringRef();
  return name == kMemrefReinterpretCastOpName || name == kMemrefCastOpName ||
         name == kMemrefSubviewOpName;
}

Value traceMemrefRoot(Value value) {
  for (int depth = 0; depth < 16; ++depth) {
    if (isa<BlockArgument>(value))
      return value;
    Operation *def = value.getDefiningOp();
    if (!isDebugMemrefViewOp(def))
      return value;
    value = def->getOperand(0);
  }
  return value;
}

bool isDebugHiddenArgMemref(Value value) {
  Value root = traceMemrefRoot(value);
  auto blockArg = dyn_cast<BlockArgument>(root);
  if (!blockArg)
    return false;

  Operation *parent = blockArg.getOwner()->getParentOp();
  auto func = dyn_cast_or_null<FunctionOpInterface>(parent);
  if (!func)
    return false;

  if (blockArg.getArgNumber() >= func.getNumArguments())
    return false;

  auto attr = dyn_cast_or_null<StringAttr>(
      func.getArgAttr(blockArg.getArgNumber(), kAttrHiddenArg));
  return attr && attr.getValue() == kHiddenArgName;
}

Value extractI32WordFromSingleElementTensor(Value source) {
  if (!isSingleElementI32Tensor(source.getType()))
    return {};

  Operation *def = source.getDefiningOp();
  if (!def || def->getNumOperands() == 0)
    return {};

  StringRef name = def->getName().getStringRef();
  if (name != kTensorInsertOpName && name != kLinalgFillOpName)
    return {};

  Value word = def->getOperand(0);
  if (!isI32Scalar(word.getType()))
    return {};
  return word;
}

bool simplifyOneDebugRecordMaterialize(
    OpBuilder &builder, Operation *op,
    SmallVectorImpl<Operation *> &maybeDead) {
  if (op->getName().getStringRef() != kMaterializeInDestinationOpName ||
      op->getNumOperands() < 2)
    return false;

  Value source = op->getOperand(0);
  Value dest = op->getOperand(1);
  if (!isSingleElementI32Tensor(source.getType()) ||
      !isSingleElementI32MemRef(dest.getType()) ||
      !isDebugHiddenArgMemref(dest))
    return false;

  Value word = extractI32WordFromSingleElementTensor(source);
  if (!word)
    return false;

  Operation *sourceDef = source.getDefiningOp();
  builder.setInsertionPoint(op);
  Value zeroIndex = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
  OperationState storeState(op->getLoc(), kMemrefStoreOpName);
  storeState.addOperands({word, dest, zeroIndex});
  builder.create(storeState);
  op->erase();

  if (sourceDef)
    maybeDead.push_back(sourceDef);
  return true;
}

struct SimplifyRecordMemrefWritesPass
    : public PassWrapper<SimplifyRecordMemrefWritesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyRecordMemrefWritesPass);

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> materializes;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == kMaterializeInDestinationOpName)
        materializes.push_back(op);
    });

    OpBuilder builder(module.getContext());
    SmallVector<Operation *> maybeDead;
    for (Operation *op : materializes) {
      if (!op->getBlock())
        continue;
      simplifyOneDebugRecordMaterialize(builder, op, maybeDead);
    }

    llvm::SmallPtrSet<Operation *, 32> seen;
    for (Operation *op : maybeDead) {
      if (!op || !op->use_empty() || !seen.insert(op).second)
        continue;
      StringRef name = op->getName().getStringRef();
      if (name == kTensorInsertOpName || name == kLinalgFillOpName)
        op->erase();
    }
  }

  StringRef getArgument() const final {
    return "flagtree-simplify-debug-record-memref-writes";
  }
  StringRef getDescription() const final {
    return "Rewrite debugger record materialize writes to scalar memref.store";
  }
};

struct InsertInstrumentationPass
    : public PassWrapper<InsertInstrumentationPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertInstrumentationPass);

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<FlagTreeDebugDialect, arith::ArithDialect, math::MathDialect,
                memref::MemRefDialect, ptr::PtrDialect, scf::SCFDialect,
                tensor::TensorDialect, triton::TritonDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Idempotency guard: skip if this pass has already run on this module.
    if (module->hasAttr(kAttrInstrumentationInserted))
      return;

    Builder builder(module.getContext());
    llvm::SmallVector<InstrumentationTarget, 8> targets;
    bool anyInstrumented = false;
    bool level2Failure = false;
    auto hiddenArgAbiAttr =
        module->getAttrOfType<BoolAttr>(kAttrHiddenArgAbiEnabled);
    const bool enableHiddenArgAbi =
        hiddenArgAbiAttr && hiddenArgAbiAttr.getValue();
    const bool metadataOnlyCompilePath =
        hiddenArgAbiAttr && !hiddenArgAbiAttr.getValue();
    auto timelineEnabledAttr =
        module->getAttrOfType<BoolAttr>(kAttrTimelineEnabled);
    const bool timelineEnabled =
        timelineEnabledAttr && timelineEnabledAttr.getValue();
    auto timelineOnlyAttr = module->getAttrOfType<BoolAttr>(kAttrTimelineOnly);
    const bool timelineOnly = timelineOnlyAttr && timelineOnlyAttr.getValue();
    llvm::DenseSet<Operation *> functionsWithCalls;
    llvm::StringSet<> calledFunctions;
    if (enableHiddenArgAbi) {
      module.walk([&](FunctionOpInterface func) {
        bool hasCallLikeOp = false;
        func.walk([&](Operation *nestedOp) {
          if (nestedOp == func.getOperation())
            return WalkResult::advance();
          if (isCallLikeOp(nestedOp)) {
            hasCallLikeOp = true;
            if (std::optional<StringRef> callee = calledCalleeName(nestedOp))
              calledFunctions.insert(*callee);
          }
          return WalkResult::advance();
        });
        if (hasCallLikeOp)
          functionsWithCalls.insert(func.getOperation());
      });
    }

    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "flagtree_debug")
        return;
      if (isCallLikeOp(op))
        return;
      if (isInsideTritonCombinerRegion(op))
        return;
      if (enableHiddenArgAbi) {
        FunctionOpInterface func = parentFunction(op);
        if (func && (functionsWithCalls.contains(func.getOperation()) ||
                     calledFunctions.contains(func.getName()))) {
          // Hidden-arg ABI is not call-graph aware yet.  If a function contains
          // tt.call/func.call, or if it is referenced by such a call, adding
          // the debug hidden argument can invalidate callee/callsite
          // signatures.  Keep such functions metadata-only until call-graph
          // argument threading is implemented.
          return;
        }
      }

      auto opIdAttr = getIntAttr(op, kAttrOpId, kAttrFallbackOpId);
      if (!opIdAttr || opIdAttr.getInt() == 0)
        return;
      auto captureMap = op->getAttrOfType<StringAttr>(kAttrOperandCaptureMap);

      const RecordLevel level = getRecordLevel(op);
      const int32_t addrLevel = getAddrLevel(op);
      const bool level2 = level == RecordLevel::LEVEL_TENSOR_FULL;
      const bool canEmitDynamicSummary = isSafeDynamicDebugValue(op);
      const bool canEmitDynamicMemory = isSafeDynamicDebugMemoryEvent(op);

      auto addValueTarget = [&](Value observedValue, int32_t targetOpId,
                                StringRef fullDumpSource) -> bool {
        if (timelineOnly || !observedValue || targetOpId == 0)
          return false;
        ArrayAttr collectors = buildCollectorArrayForValue(
            builder, level, observedValue.getType());
        const bool valueHasSummary =
            canEmitDynamicSummary && collectors && !collectors.empty() &&
            (shouldEmitDynamicSummary(op, observedValue.getType(), level) ||
             fullDumpSource == kFullDumpSourceStatementOperand);
        std::optional<FullDumpSpec> dump =
            level2 && canEmitDynamicSummary
                ? getFullDumpSpecForObservedValue(observedValue.getType(),
                                                  fullDumpSource)
                : std::nullopt;
        if (!valueHasSummary && !dump)
          return false;
        targets.push_back(InstrumentationTarget{
            op, observedValue, targetOpId,
            getI32AttrValue(getIntAttr(op, kAttrScopeId, kAttrFallbackScopeId)),
            level, addrLevel, valueHasSummary, false, dump.has_value(), false,
            false, valueHasSummary ? collectors : ArrayAttr(),
            dump.value_or(FullDumpSpec{}), FullDumpSpec{}});
        return true;
      };

      // Ops with lowerable result collectors get summary records; memory-like
      // ops get memory-event records only when address collection is enabled.
      StringRef valueSource;
      Value fullValue = observedFullDumpValue(op, valueSource);
      ArrayAttr summaryCollectors;
      if (!timelineOnly && canEmitDynamicSummary && fullValue)
        summaryCollectors =
            buildCollectorArrayForValue(builder, level, fullValue.getType());
      const bool hasSummary =
          !timelineOnly && canEmitDynamicSummary && summaryCollectors &&
          !summaryCollectors.empty() &&
          shouldEmitDynamicSummary(op, fullValue.getType(), level);
      const bool hasMemoryEvent = !timelineOnly && canEmitDynamicMemory &&
                                  addrLevel > 0 && isMemoryLikeOp(op) &&
                                  memoryPointerOperand(op);
      const bool fullValueAllowed =
          canEmitDynamicSummary || valueSource == kFullDumpSourceStoreValue;
      std::optional<FullDumpSpec> valueDump =
          !timelineOnly && level2 && fullValue && fullValueAllowed
              ? getFullDumpSpecForObservedValue(fullValue.getType(),
                                                valueSource)
              : std::nullopt;
      std::optional<FullDumpSpec> addressDump =
          !timelineOnly && level2 && canEmitDynamicMemory && addrLevel >= 2 &&
                  isMemoryLikeOp(op) && memoryPointerOperand(op)
              ? getFullDumpSpecForMemoryAddress(memoryPointerOperand(op))
              : std::nullopt;
      if (addressDump && !canComputeAddressSummaryForMemoryOp(op)) {
        op->emitError()
            << "level-2 debugger cannot dump full memory lane addresses for "
               "this pointer/mask pattern on the current backend";
        level2Failure = true;
        return;
      }
      if (!timelineOnly && level2 && !valueDump && !addressDump &&
          !captureMap) {
        op->emitError()
            << "level-2 debugger requires a statically shaped scalar/tensor "
               "value or supported memory address to dump";
        level2Failure = true;
        return;
      }
      const bool hasFullValueRef = valueDump.has_value();
      const bool hasFullAddressRef = addressDump.has_value();
      const bool hasTimeline = timelineEnabled && shouldEmitTimeline(op);
      const bool hasPrimaryRecords = hasSummary || hasMemoryEvent ||
                                     hasFullValueRef || hasFullAddressRef ||
                                     hasTimeline;

      if (hasPrimaryRecords) {
        op->setAttr(kAttrInstrumented, builder.getBoolAttr(true));
        op->setAttr(kAttrRecordKinds,
                    buildRecordKindArray(builder, hasSummary, hasMemoryEvent,
                                         hasFullValueRef || hasFullAddressRef,
                                         hasTimeline));

        if (hasSummary)
          op->setAttr(kAttrSummaryCollectors, summaryCollectors);

        if (hasMemoryEvent) {
          // Use canonical Protocol enum names when a single event is emitted,
          // and ADDRESS_SUMMARY when addr_level requests the CANN9 summary
          // bundle.
          op->setAttr(kAttrMemoryEventKind,
                      builder.getStringAttr(memoryEventKindForMemoryOp(op)));
        }

        if (hasFullValueRef || hasFullAddressRef)
          op->setAttr(kAttrFullValueRef, builder.getBoolAttr(true));

        targets.push_back(InstrumentationTarget{
            op, fullValue, getI32AttrValue(opIdAttr),
            getI32AttrValue(getIntAttr(op, kAttrScopeId, kAttrFallbackScopeId)),
            level, addrLevel, hasSummary, hasMemoryEvent, hasFullValueRef,
            hasFullAddressRef, hasTimeline,
            hasSummary ? summaryCollectors : ArrayAttr(),
            valueDump.value_or(FullDumpSpec{}),
            addressDump.value_or(FullDumpSpec{})});
        anyInstrumented = true;
      }

      if (captureMap) {
        for (const auto &[operandIndex, captureOpId] :
             parseOperandCaptureMap(captureMap.getValue())) {
          if (operandIndex >= op->getNumOperands())
            continue;
          if (addValueTarget(op->getOperand(operandIndex),
                             static_cast<int32_t>(captureOpId),
                             kFullDumpSourceStatementOperand)) {
            anyInstrumented = true;
            op->setAttr(kAttrInstrumented, builder.getBoolAttr(true));
          }
        }
      }
    });

    if (level2Failure) {
      signalPassFailure();
      return;
    }

    if (!anyInstrumented)
      return;

    int32_t recordsPerInstance = 0;
    OpBuilder opBuilder(module.getContext());
    llvm::SmallVector<RecordPlanEntry> recordPlan;
    llvm::SmallVector<FullDumpPlanEntry> fullDumpPlan;
    uint64_t payloadBytesPerInstance = 0;
    if (!metadataOnlyCompilePath) {
      for (const InstrumentationTarget &target : targets)
        insertRecordOps(opBuilder, target, recordsPerInstance, recordPlan,
                        fullDumpPlan, payloadBytesPerInstance);
    }

    // Mark module as instrumented before annotating functions so that the
    // idempotency guard above fires correctly on any re-run.
    module->setAttr(kAttrInstrumentationInserted, builder.getBoolAttr(true));
    module->setAttr(kAttrRecordsPerInstance,
                    builder.getI32IntegerAttr(recordsPerInstance));
    module->setAttr(kAttrRecordSize, builder.getI32IntegerAttr(kRecordBytes));
    if (!fullDumpPlan.empty()) {
      module->setAttr(kAttrFullDumpPayloadBytesPerInstance,
                      builder.getI64IntegerAttr(
                          static_cast<int64_t>(payloadBytesPerInstance)));
      module->setAttr(
          kAttrFullDumpPlan,
          builder.getStringAttr(serializeFullDumpPlanToJson(fullDumpPlan)));
    }
    if (!recordPlan.empty()) {
      module->setAttr(
          kAttrRecordLayout,
          builder.getStringAttr(kRecordLayoutDeterministicCompactV1));
      module->setAttr(
          kAttrRecordPlan,
          builder.getStringAttr(serializeRecordPlanToJson(recordPlan)));
    }

    bool hiddenArgInsertFailed = false;
    module.walk([&](Operation *op) {
      auto func = dyn_cast<FunctionOpInterface>(op);
      if (!func)
        return;

      bool hasInstrumentedOps = false;
      func.walk([&](Operation *nestedOp) {
        if (nestedOp->hasAttr(kAttrInstrumented)) {
          hasInstrumentedOps = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasInstrumentedOps && !metadataOnlyCompilePath)
        annotateFunction(func, builder);
      if (hasInstrumentedOps && enableHiddenArgAbi)
        hiddenArgInsertFailed |= failed(
            ensureHiddenDebugArgument(func, builder, enableHiddenArgAbi));
    });

    if (hiddenArgInsertFailed) {
      signalPassFailure();
      return;
    }

    if (enableHiddenArgAbi)
      lowerRecordOpsWithHiddenArg(module, opBuilder);
  }

  StringRef getArgument() const final {
    return "flagtree-insert-debug-records";
  }
  StringRef getDescription() const final {
    return "Insert debug summary and memory-event instrumentation";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createInsertInstrumentationPass() {
  return std::make_unique<InsertInstrumentationPass>();
}

std::unique_ptr<mlir::Pass> createSimplifyRecordMemrefWritesPass() {
  return std::make_unique<SimplifyRecordMemrefWritesPass>();
}

void setDebugHiddenArgAbiEnabled(ModuleOp module, bool enabled) {
  Builder builder(module.getContext());
  module->setAttr(kAttrHiddenArgAbiEnabled, builder.getBoolAttr(enabled));
}

void setDebugAddrLevel(ModuleOp module, int32_t addrLevel) {
  Builder builder(module.getContext());
  int32_t normalized = std::clamp<int32_t>(addrLevel, 0, 2);
  module->setAttr(kAttrAddrLevel, builder.getI32IntegerAttr(normalized));
}

void setDebugTimelineEnabled(ModuleOp module, bool enabled) {
  Builder builder(module.getContext());
  module->setAttr(kAttrTimelineEnabled, builder.getBoolAttr(enabled));
}

void setDebugTimelineOnly(ModuleOp module, bool enabled) {
  Builder builder(module.getContext());
  module->setAttr(kAttrTimelineOnly, builder.getBoolAttr(enabled));
}

uint32_t getDebugRecordsPerInstance(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(kAttrRecordsPerInstance))
    return static_cast<uint32_t>(std::max<int64_t>(0, attr.getInt()));
  return 0;
}

uint32_t getDebugRecordSize(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(kAttrRecordSize))
    return static_cast<uint32_t>(std::max<int64_t>(0, attr.getInt()));
  return kLegacyRecordBytes;
}

std::string getDebugRecordLayout(ModuleOp module) {
  if (auto attr = module->getAttrOfType<StringAttr>(kAttrRecordLayout))
    return attr.getValue().str();
  return "";
}

std::string getDebugRecordPlanJson(ModuleOp module) {
  if (auto attr = module->getAttrOfType<StringAttr>(kAttrRecordPlan))
    return attr.getValue().str();
  return "[]";
}

uint64_t getDebugFullDumpPayloadBytesPerInstance(ModuleOp module) {
  if (auto attr = module->getAttrOfType<IntegerAttr>(
          kAttrFullDumpPayloadBytesPerInstance))
    return static_cast<uint64_t>(std::max<int64_t>(0, attr.getInt()));
  return 0;
}

std::string getDebugFullDumpPlanJson(ModuleOp module) {
  if (auto attr = module->getAttrOfType<StringAttr>(kAttrFullDumpPlan))
    return attr.getValue().str();
  return "[]";
}

void registerFlagTreeDebuggerInstrumentationPasses() {
  PassRegistration<InsertInstrumentationPass>();
  PassRegistration<SimplifyRecordMemrefWritesPass>();
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
