#include "Debugger/Metadata/TrackedOpTable.h"

#include <gtest/gtest.h>

namespace mlir {
namespace flagtree {
namespace debugger {
namespace {

TrackedOpEntry makeEntry(uint32_t opId, bool memory) {
  TrackedOpEntry entry;
  entry.opId = opId;
  entry.scopeId = 1;
  entry.resultIndex = 0;
  entry.isMemoryOp = memory;
  entry.opCategory = memory ? "load" : "";
  entry.role = memory ? "load" : "";
  entry.mlirOpName = memory ? "memref.load" : "arith.addf";
  entry.sourceLoc = "unit.mlir:1:1";
  entry.tritonStatement = entry.mlirOpName;
  entry.inlineCallPath = "";
  entry.result.valueKind = "tensor";
  entry.result.dtype = "tensor<4xf32>";
  entry.result.elementDtype = "f32";
  entry.result.shape = "[4]";
  entry.result.stride = "unknown";
  entry.result.layout = "unknown";
  entry.result.encoding = "";
  entry.result.addrSpace = memory ? "global" : "";
  entry.result.rank = 1;
  entry.result.elementBits = 32;
  entry.result.vecWidth = 4;

  OperandStaticInfo operand;
  operand.operandIndex = 0;
  operand.operandRole = memory ? "ptr" : "lhs";
  operand.producerOpId = 0;
  operand.isConstant = false;
  operand.isPredicate = false;
  operand.isKernelArgument = true;
  operand.constantValueRepr = "";
  operand.value = entry.result;
  entry.operands.push_back(operand);

  entry.addrSpace = memory ? "global" : "";
  entry.accessType = memory ? "load" : "";
  entry.accessBytes = memory ? 4 : 0;
  entry.alignmentRequired = memory ? 4 : 0;
  entry.hasMask = false;
  entry.maskDtype = "";
  entry.cacheModifier = "";
  entry.evictionPolicy = "";
  entry.isVolatile = false;
  entry.boundaryCheckPolicy = "";
  entry.paddingSemantics = "";
  return entry;
}

TEST(FlagTreeDebuggerMetadataTest, FindsTrackedOpById) {
  TrackedOpTable table{makeEntry(1, true), makeEntry(2, false)};
  ASSERT_NE(findTrackedOp(table, 1), nullptr);
  EXPECT_EQ(findTrackedOp(table, 1)->mlirOpName, "memref.load");
  EXPECT_EQ(findTrackedOp(table, 99), nullptr);
}

TEST(FlagTreeDebuggerMetadataTest, TrackedOpTableJsonRoundTripsStrictSchema) {
  TrackedOpTable table{makeEntry(1, true), makeEntry(2, false)};

  std::string json = serializeTrackedOpTableToJson(table);
  EXPECT_NE(json.find("\"opId\""), std::string::npos);
  EXPECT_NE(json.find("\"operands\""), std::string::npos);
  EXPECT_NE(json.find("\"accessBytes\""), std::string::npos);

  TrackedOpTable parsed;
  std::string error;
  ASSERT_TRUE(parseTrackedOpTableFromJson(json, parsed, &error)) << error;
  ASSERT_EQ(parsed.size(), 2u);
  EXPECT_EQ(parsed[0].opId, 1u);
  EXPECT_TRUE(parsed[0].isMemoryOp);
  ASSERT_EQ(parsed[0].operands.size(), 1u);
  EXPECT_TRUE(parsed[0].operands[0].isKernelArgument);
  EXPECT_EQ(parsed[1].mlirOpName, "arith.addf");
}

TEST(FlagTreeDebuggerMetadataTest, StatementValueMetadataJsonRoundTrips) {
  TrackedOpEntry entry = makeEntry(2, false);
  entry.statementId = 12345;
  entry.statementResultName = "y";

  StatementValueInfo result;
  result.sourceName = "y";
  result.sourceRole = "result";
  result.captureOpId = 2;
  result.capturePolicy = "captured_current_op";
  result.value = entry.result;
  entry.statementValues.push_back(result);

  StatementValueInfo operand;
  operand.sourceName = "a";
  operand.sourceRole = "operand";
  operand.hasOperandIndex = true;
  operand.operandIndex = 0;
  operand.captureOpId = 5;
  operand.capturePolicy = "captured_at_current_statement";
  operand.value = entry.result;
  entry.statementValues.push_back(operand);

  TrackedOpTable table{entry};
  std::string json = serializeTrackedOpTableToJson(table);
  EXPECT_NE(json.find("\"statementValues\""), std::string::npos);
  EXPECT_NE(json.find("\"statementResultName\""), std::string::npos);

  TrackedOpTable parsed;
  std::string error;
  ASSERT_TRUE(parseTrackedOpTableFromJson(json, parsed, &error)) << error;
  ASSERT_EQ(parsed.size(), 1u);
  EXPECT_EQ(parsed[0].statementId, 12345u);
  EXPECT_EQ(parsed[0].statementResultName, "y");
  ASSERT_EQ(parsed[0].statementValues.size(), 2u);
  EXPECT_EQ(parsed[0].statementValues[0].sourceName, "y");
  EXPECT_EQ(parsed[0].statementValues[0].captureOpId, 2u);
  EXPECT_EQ(parsed[0].statementValues[1].sourceRole, "operand");
  EXPECT_TRUE(parsed[0].statementValues[1].hasOperandIndex);
  EXPECT_EQ(parsed[0].statementValues[1].operandIndex, 0u);
  EXPECT_EQ(parsed[0].statementValues[1].captureOpId, 5u);
}

TEST(FlagTreeDebuggerMetadataTest, KernelMetadataJsonRoundTrips) {
  KernelDebugMetadata metadata;
  metadata.debugKernelId = 17;
  metadata.kernelName = "kernel";
  metadata.backendName = "cuda";
  metadata.targetName = "unit";
  metadata.scopeCount = 1;
  metadata.trackedOpCount = 2;
  metadata.trackedOps = {makeEntry(1, true), makeEntry(2, false)};

  std::string json = serializeKernelDebugMetadataToJson(metadata);
  KernelDebugMetadata parsed;
  std::string error;
  ASSERT_TRUE(parseKernelDebugMetadataFromJson(json, parsed, &error)) << error;
  EXPECT_EQ(parsed.debugKernelId, 17u);
  EXPECT_EQ(parsed.kernelName, "kernel");
  EXPECT_EQ(parsed.trackedOpCount, 2u);
  ASSERT_EQ(parsed.trackedOps.size(), 2u);
  EXPECT_EQ(parsed.trackedOps[0].accessType, "load");
}

TEST(FlagTreeDebuggerMetadataTest, RejectsMalformedAndTypeInvalidJson) {
  TrackedOpTable table;
  std::string error;
  EXPECT_FALSE(parseTrackedOpTableFromJson("{", table, &error));
  EXPECT_FALSE(
      parseTrackedOpTableFromJson("[{\"opId\":\"bad\"}]", table, &error));
  EXPECT_NE(error.find("opId"), std::string::npos);

  KernelDebugMetadata metadata;
  EXPECT_FALSE(parseKernelDebugMetadataFromJson("[]", metadata, &error));
  EXPECT_NE(error.find("object"), std::string::npos);
}

} // namespace
} // namespace debugger
} // namespace flagtree
} // namespace mlir
