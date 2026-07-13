#include "Debugger/Metadata/TrackedOpTable.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <utility>

namespace mlir {
namespace flagtree {
namespace debugger {

namespace {

using llvm::json::Array;
using llvm::json::Object;
using llvm::json::Value;

std::string jsonToString(Value value) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << value;
  return text;
}

Object toJson(const StaticValueInfo &info) {
  return Object{{"valueKind", info.valueKind},
                {"dtype", info.dtype},
                {"elementDtype", info.elementDtype},
                {"shape", info.shape},
                {"stride", info.stride},
                {"layout", info.layout},
                {"encoding", info.encoding},
                {"addrSpace", info.addrSpace},
                {"rank", info.rank},
                {"elementBits", info.elementBits},
                {"vecWidth", info.vecWidth}};
}

Object toJson(const OperandStaticInfo &info) {
  return Object{{"operandIndex", info.operandIndex},
                {"operandRole", info.operandRole},
                {"producerOpId", info.producerOpId},
                {"isConstant", info.isConstant},
                {"isPredicate", info.isPredicate},
                {"isKernelArgument", info.isKernelArgument},
                {"constantValueRepr", info.constantValueRepr},
                {"value", toJson(info.value)}};
}

Object toJson(const StatementValueInfo &info) {
  return Object{{"sourceName", info.sourceName},
                {"sourceRole", info.sourceRole},
                {"hasOperandIndex", info.hasOperandIndex},
                {"operandIndex", info.operandIndex},
                {"producerOpId", info.producerOpId},
                {"captureOpId", info.captureOpId},
                {"capturePolicy", info.capturePolicy},
                {"isConstant", info.isConstant},
                {"constantValueRepr", info.constantValueRepr},
                {"value", toJson(info.value)}};
}

Object toJson(const TrackedOpEntry &entry) {
  Array operands;
  for (const auto &operand : entry.operands)
    operands.push_back(toJson(operand));

  Array statementValues;
  for (const auto &value : entry.statementValues)
    statementValues.push_back(toJson(value));

  return Object{
      {"opId", entry.opId},
      {"scopeId", entry.scopeId},
      {"resultIndex", entry.resultIndex},
      {"statementId", entry.statementId},
      {"isSyntheticStatementCapture", entry.isSyntheticStatementCapture},
      {"isMemoryOp", entry.isMemoryOp},
      {"opCategory", entry.opCategory},
      {"role", entry.role},
      {"mlirOpName", entry.mlirOpName},
      {"sourceLoc", entry.sourceLoc},
      {"tritonStatement", entry.tritonStatement},
      {"statementResultName", entry.statementResultName},
      {"inlineCallPath", entry.inlineCallPath},
      {"result", toJson(entry.result)},
      {"operands", std::move(operands)},
      {"statementValues", std::move(statementValues)},
      {"addrSpace", entry.addrSpace},
      {"accessType", entry.accessType},
      {"accessBytes", entry.accessBytes},
      {"alignmentRequired", entry.alignmentRequired},
      {"hasMask", entry.hasMask},
      {"maskDtype", entry.maskDtype},
      {"cacheModifier", entry.cacheModifier},
      {"evictionPolicy", entry.evictionPolicy},
      {"isVolatile", entry.isVolatile},
      {"boundaryCheckPolicy", entry.boundaryCheckPolicy},
      {"paddingSemantics", entry.paddingSemantics}};
}

bool getString(const Object &object, const char *key, std::string &out,
               std::string *errorMessage) {
  auto value = object.getString(key);
  if (!value) {
    if (errorMessage)
      *errorMessage = std::string("missing or non-string key: ") + key;
    return false;
  }
  out = value->str();
  return true;
}

bool getOptionalString(const Object &object, const char *key, std::string &out,
                       std::string_view fallback = "") {
  auto value = object.getString(key);
  out = value ? value->str() : std::string(fallback);
  return true;
}

bool getUInt32(const Object &object, const char *key, uint32_t &out,
               std::string *errorMessage) {
  auto value = object.getInteger(key);
  if (!value || *value < 0 || *value > UINT32_MAX) {
    if (errorMessage)
      *errorMessage = std::string("missing or invalid uint32 key: ") + key;
    return false;
  }
  out = static_cast<uint32_t>(*value);
  return true;
}

bool getOptionalUInt32(const Object &object, const char *key, uint32_t &out,
                       uint32_t fallback = 0) {
  auto value = object.getInteger(key);
  if (!value || *value < 0 || *value > UINT32_MAX) {
    out = fallback;
    return true;
  }
  out = static_cast<uint32_t>(*value);
  return true;
}

bool getBool(const Object &object, const char *key, bool &out,
             std::string *errorMessage) {
  auto value = object.getBoolean(key);
  if (!value) {
    if (errorMessage)
      *errorMessage = std::string("missing or non-bool key: ") + key;
    return false;
  }
  out = *value;
  return true;
}

bool getOptionalBool(const Object &object, const char *key, bool &out,
                     bool fallback = false) {
  auto value = object.getBoolean(key);
  out = value ? *value : fallback;
  return true;
}

bool parseStaticValueInfo(const Object &object, StaticValueInfo &info,
                          std::string *errorMessage) {
  return getString(object, "valueKind", info.valueKind, errorMessage) &&
         getString(object, "dtype", info.dtype, errorMessage) &&
         getString(object, "elementDtype", info.elementDtype, errorMessage) &&
         getString(object, "shape", info.shape, errorMessage) &&
         getString(object, "stride", info.stride, errorMessage) &&
         getString(object, "layout", info.layout, errorMessage) &&
         getString(object, "encoding", info.encoding, errorMessage) &&
         getString(object, "addrSpace", info.addrSpace, errorMessage) &&
         getUInt32(object, "rank", info.rank, errorMessage) &&
         getUInt32(object, "elementBits", info.elementBits, errorMessage) &&
         getUInt32(object, "vecWidth", info.vecWidth, errorMessage);
}

bool parseOperandStaticInfo(const Object &object, OperandStaticInfo &info,
                            std::string *errorMessage) {
  const auto *valueObject = object.getObject("value");
  if (!valueObject) {
    if (errorMessage)
      *errorMessage = "missing or non-object key: value";
    return false;
  }
  return getUInt32(object, "operandIndex", info.operandIndex, errorMessage) &&
         getString(object, "operandRole", info.operandRole, errorMessage) &&
         getUInt32(object, "producerOpId", info.producerOpId, errorMessage) &&
         getBool(object, "isConstant", info.isConstant, errorMessage) &&
         getBool(object, "isPredicate", info.isPredicate, errorMessage) &&
         getBool(object, "isKernelArgument", info.isKernelArgument,
                 errorMessage) &&
         getString(object, "constantValueRepr", info.constantValueRepr,
                   errorMessage) &&
         parseStaticValueInfo(*valueObject, info.value, errorMessage);
}

bool parseStatementValueInfo(const Object &object, StatementValueInfo &info,
                             std::string *errorMessage) {
  const auto *valueObject = object.getObject("value");
  if (!valueObject) {
    if (errorMessage)
      *errorMessage = "missing or non-object key: value";
    return false;
  }
  return getString(object, "sourceName", info.sourceName, errorMessage) &&
         getString(object, "sourceRole", info.sourceRole, errorMessage) &&
         getBool(object, "hasOperandIndex", info.hasOperandIndex,
                 errorMessage) &&
         getUInt32(object, "operandIndex", info.operandIndex, errorMessage) &&
         getUInt32(object, "producerOpId", info.producerOpId, errorMessage) &&
         getUInt32(object, "captureOpId", info.captureOpId, errorMessage) &&
         getString(object, "capturePolicy", info.capturePolicy, errorMessage) &&
         getBool(object, "isConstant", info.isConstant, errorMessage) &&
         getString(object, "constantValueRepr", info.constantValueRepr,
                   errorMessage) &&
         parseStaticValueInfo(*valueObject, info.value, errorMessage);
}

bool parseTrackedOpEntry(const Object &object, TrackedOpEntry &entry,
                         std::string *errorMessage) {
  if (!getUInt32(object, "opId", entry.opId, errorMessage))
    return false;

  const auto *resultObject = object.getObject("result");
  if (!resultObject) {
    if (errorMessage)
      *errorMessage = "missing or non-object key: result";
    return false;
  }

  const auto *operandsArray = object.getArray("operands");
  if (!operandsArray) {
    if (errorMessage)
      *errorMessage = "missing or non-array key: operands";
    return false;
  }

  const auto *statementValuesArray = object.getArray("statementValues");

  if (!(getUInt32(object, "scopeId", entry.scopeId, errorMessage) &&
        getUInt32(object, "resultIndex", entry.resultIndex, errorMessage) &&
        getOptionalUInt32(object, "statementId", entry.statementId) &&
        getOptionalBool(object, "isSyntheticStatementCapture",
                        entry.isSyntheticStatementCapture) &&
        getBool(object, "isMemoryOp", entry.isMemoryOp, errorMessage) &&
        getString(object, "opCategory", entry.opCategory, errorMessage) &&
        getString(object, "role", entry.role, errorMessage) &&
        getString(object, "mlirOpName", entry.mlirOpName, errorMessage) &&
        getString(object, "sourceLoc", entry.sourceLoc, errorMessage) &&
        getString(object, "tritonStatement", entry.tritonStatement,
                  errorMessage) &&
        getOptionalString(object, "statementResultName",
                          entry.statementResultName) &&
        getString(object, "inlineCallPath", entry.inlineCallPath,
                  errorMessage) &&
        parseStaticValueInfo(*resultObject, entry.result, errorMessage) &&
        getString(object, "addrSpace", entry.addrSpace, errorMessage) &&
        getString(object, "accessType", entry.accessType, errorMessage) &&
        getUInt32(object, "accessBytes", entry.accessBytes, errorMessage) &&
        getUInt32(object, "alignmentRequired", entry.alignmentRequired,
                  errorMessage) &&
        getBool(object, "hasMask", entry.hasMask, errorMessage) &&
        getString(object, "maskDtype", entry.maskDtype, errorMessage) &&
        getString(object, "cacheModifier", entry.cacheModifier, errorMessage) &&
        getString(object, "evictionPolicy", entry.evictionPolicy,
                  errorMessage) &&
        getBool(object, "isVolatile", entry.isVolatile, errorMessage) &&
        getString(object, "boundaryCheckPolicy", entry.boundaryCheckPolicy,
                  errorMessage) &&
        getString(object, "paddingSemantics", entry.paddingSemantics,
                  errorMessage))) {
    return false;
  }

  entry.operands.clear();
  for (const auto &operandValue : *operandsArray) {
    const auto *operandObject = operandValue.getAsObject();
    if (!operandObject) {
      if (errorMessage)
        *errorMessage = "operands entries must be objects";
      return false;
    }
    OperandStaticInfo operand;
    if (!parseOperandStaticInfo(*operandObject, operand, errorMessage))
      return false;
    entry.operands.push_back(std::move(operand));
  }

  entry.statementValues.clear();
  if (statementValuesArray) {
    for (const auto &statementValue : *statementValuesArray) {
      const auto *statementObject = statementValue.getAsObject();
      if (!statementObject) {
        if (errorMessage)
          *errorMessage = "statementValues entries must be objects";
        return false;
      }
      StatementValueInfo value;
      if (!parseStatementValueInfo(*statementObject, value, errorMessage))
        return false;
      entry.statementValues.push_back(std::move(value));
    }
  }

  return true;
}

} // namespace

const TrackedOpEntry *findTrackedOp(const TrackedOpTable &table,
                                    uint32_t opId) {
  for (const auto &entry : table) {
    if (entry.opId == opId) {
      return &entry;
    }
  }
  return nullptr;
}

TrackedOpEntry *findTrackedOp(TrackedOpTable &table, uint32_t opId) {
  for (auto &entry : table) {
    if (entry.opId == opId) {
      return &entry;
    }
  }
  return nullptr;
}

std::string serializeTrackedOpTableToJson(const TrackedOpTable &table) {
  Array array;
  for (const auto &entry : table)
    array.push_back(toJson(entry));
  return jsonToString(std::move(array));
}

bool parseTrackedOpTableFromJson(std::string_view text, TrackedOpTable &table,
                                 std::string *errorMessage) {
  table.clear();
  llvm::Expected<Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    if (errorMessage) {
      std::string err;
      llvm::raw_string_ostream os(err);
      os << llvm::toString(parsed.takeError());
      *errorMessage = os.str();
    }
    return false;
  }

  const auto *array = parsed->getAsArray();
  if (!array) {
    if (errorMessage)
      *errorMessage = "TrackedOpTable JSON must be an array";
    return false;
  }

  for (const auto &value : *array) {
    const auto *object = value.getAsObject();
    if (!object) {
      if (errorMessage)
        *errorMessage = "TrackedOpTable entries must be objects";
      return false;
    }
    TrackedOpEntry entry;
    if (!parseTrackedOpEntry(*object, entry, errorMessage))
      return false;
    table.push_back(std::move(entry));
  }

  return true;
}

std::string
serializeKernelDebugMetadataToJson(const KernelDebugMetadata &metadata) {
  Array trackedOps;
  for (const auto &entry : metadata.trackedOps)
    trackedOps.push_back(toJson(entry));

  Object object{{"debugKernelId", metadata.debugKernelId},
                {"kernelName", metadata.kernelName},
                {"backendName", metadata.backendName},
                {"targetName", metadata.targetName},
                {"scopeCount", metadata.scopeCount},
                {"trackedOpCount", metadata.trackedOpCount},
                {"trackedOps", std::move(trackedOps)}};
  return jsonToString(std::move(object));
}

bool parseKernelDebugMetadataFromJson(std::string_view text,
                                      KernelDebugMetadata &metadata,
                                      std::string *errorMessage) {
  metadata = {};
  llvm::Expected<Value> parsed = llvm::json::parse(text);
  if (!parsed) {
    if (errorMessage) {
      std::string err;
      llvm::raw_string_ostream os(err);
      os << llvm::toString(parsed.takeError());
      *errorMessage = os.str();
    }
    return false;
  }

  const auto *object = parsed->getAsObject();
  if (!object) {
    if (errorMessage)
      *errorMessage = "KernelDebugMetadata JSON must be an object";
    return false;
  }

  const auto *trackedOps = object->getArray("trackedOps");
  if (!trackedOps) {
    if (errorMessage)
      *errorMessage = "missing or non-array key: trackedOps";
    return false;
  }

  if (!(getUInt32(*object, "debugKernelId", metadata.debugKernelId,
                  errorMessage) &&
        getString(*object, "kernelName", metadata.kernelName, errorMessage) &&
        getString(*object, "backendName", metadata.backendName, errorMessage) &&
        getString(*object, "targetName", metadata.targetName, errorMessage) &&
        getUInt32(*object, "scopeCount", metadata.scopeCount, errorMessage) &&
        getUInt32(*object, "trackedOpCount", metadata.trackedOpCount,
                  errorMessage))) {
    return false;
  }

  metadata.trackedOps.clear();
  for (const auto &value : *trackedOps) {
    const auto *entryObject = value.getAsObject();
    if (!entryObject) {
      if (errorMessage)
        *errorMessage = "trackedOps entries must be objects";
      return false;
    }
    TrackedOpEntry entry;
    if (!parseTrackedOpEntry(*entryObject, entry, errorMessage))
      return false;
    metadata.trackedOps.push_back(std::move(entry));
  }

  return true;
}

} // namespace debugger
} // namespace flagtree
} // namespace mlir
