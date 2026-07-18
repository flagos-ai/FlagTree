#include "RPUPlan/IR/Dialect.h"
#include "RPUPlanModel.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace mlir {
namespace rpu {

static llvm::json::Object
mapStringInt(const std::map<std::string, int64_t> &input) {
  llvm::json::Object object;
  for (const auto &[key, value] : input)
    object[key] = value;
  return object;
}

static std::optional<llvm::json::Value> attrToJsonValue(Attribute attr);

static std::optional<llvm::json::Object>
dictionaryAttrToJsonObject(DictionaryAttr dict) {
  llvm::json::Object object;
  for (NamedAttribute named : dict) {
    std::optional<llvm::json::Value> value = attrToJsonValue(named.getValue());
    if (!value)
      return std::nullopt;
    object[named.getName().str()] = std::move(*value);
  }
  return object;
}

static std::optional<llvm::json::Value> attrToJsonValue(Attribute attr) {
  if (auto boolAttr = dyn_cast<BoolAttr>(attr))
    return llvm::json::Value(boolAttr.getValue());
  if (auto integerAttr = dyn_cast<IntegerAttr>(attr))
    return llvm::json::Value(integerAttr.getInt());
  if (auto stringAttr = dyn_cast<StringAttr>(attr))
    return llvm::json::Value(stringAttr.getValue().str());
  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    llvm::json::Array array;
    for (Attribute entry : arrayAttr) {
      std::optional<llvm::json::Value> value = attrToJsonValue(entry);
      if (!value)
        return std::nullopt;
      array.push_back(std::move(*value));
    }
    return llvm::json::Value(std::move(array));
  }
  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    std::optional<llvm::json::Object> object =
        dictionaryAttrToJsonObject(dictAttr);
    if (!object)
      return std::nullopt;
    return llvm::json::Value(std::move(*object));
  }
  return std::nullopt;
}

static std::optional<std::map<std::string, int64_t>>
dictionaryAttrToIntMap(DictionaryAttr dict) {
  std::map<std::string, int64_t> map;
  for (NamedAttribute named : dict) {
    auto value = dyn_cast<IntegerAttr>(named.getValue());
    if (!value)
      return std::nullopt;
    map[named.getName().str()] = value.getInt();
  }
  return map;
}

static std::optional<std::vector<llvm::json::Object>>
signatureParamsFromAttr(DictionaryAttr signature) {
  auto params = dyn_cast_or_null<ArrayAttr>(signature.get("params"));
  if (!params)
    return std::nullopt;

  std::vector<llvm::json::Object> result;
  for (Attribute entry : params) {
    auto param = dyn_cast<DictionaryAttr>(entry);
    if (!param)
      return std::nullopt;
    std::optional<llvm::json::Object> object =
        dictionaryAttrToJsonObject(param);
    if (!object)
      return std::nullopt;
    result.push_back(std::move(*object));
  }
  return result;
}

static std::string serializeRPUPlanToJson(const RPUPlan &plan) {
  llvm::json::Array params;
  for (const auto &param : plan.signatureParams)
    params.push_back(llvm::json::Object(param));

  llvm::json::Array features;
  for (const auto &feature : plan.requiredDslFeatures)
    features.push_back(feature);

  llvm::json::Object root;
  root["args"] = mapStringInt(plan.args);
  root["emission"] = llvm::json::Object(plan.emission);
  root["kernel_name"] = plan.kernelName;
  root["layout"] = llvm::json::Object(plan.layout);
  root["mask"] = llvm::json::Object(plan.mask);
  root["pattern"] = plan.pattern;
  root["required_dsl_features"] = std::move(features);
  root["shape"] = mapStringInt(plan.shape);
  root["signature"] = llvm::json::Object{{"params", std::move(params)},
                                         {"return_type", plan.returnType}};
  root["version"] = plan.version;
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str();
}

std::optional<RPUPlan> rpuPlanFromKernelOp(plan::KernelOp op) {
  RPUPlan plan;
  plan.version = static_cast<int>(op.getVersion());
  plan.kernelName = op.getKernelName().str();
  plan.pattern = op.getPattern().str();

  DictionaryAttr signature = op.getSignature();
  std::optional<std::vector<llvm::json::Object>> params =
      signatureParamsFromAttr(signature);
  if (!params)
    return std::nullopt;
  plan.signatureParams = std::move(*params);

  auto returnType = dyn_cast_or_null<StringAttr>(signature.get("return_type"));
  if (!returnType)
    return std::nullopt;
  plan.returnType = returnType.getValue().str();

  std::optional<std::map<std::string, int64_t>> shape =
      dictionaryAttrToIntMap(op.getShape());
  std::optional<std::map<std::string, int64_t>> args =
      dictionaryAttrToIntMap(op.getArgs());
  std::optional<llvm::json::Object> layout =
      dictionaryAttrToJsonObject(op.getLayout());
  std::optional<llvm::json::Object> mask =
      dictionaryAttrToJsonObject(op.getMask());
  std::optional<llvm::json::Object> emission =
      dictionaryAttrToJsonObject(op.getEmission());
  if (!shape || !args || !layout || !mask || !emission)
    return std::nullopt;

  plan.shape = std::move(*shape);
  plan.args = std::move(*args);
  plan.layout = std::move(*layout);
  plan.mask = std::move(*mask);
  plan.emission = std::move(*emission);

  for (Attribute feature : op.getRequiredDslFeatures()) {
    auto stringFeature = dyn_cast<StringAttr>(feature);
    if (!stringFeature)
      return std::nullopt;
    plan.requiredDslFeatures.push_back(stringFeature.getValue().str());
  }

  return plan;
}

std::optional<std::string> serializeRPUPlanKernelOpToJson(plan::KernelOp op) {
  std::optional<RPUPlan> plan = rpuPlanFromKernelOp(op);
  if (!plan)
    return std::nullopt;
  return serializeRPUPlanToJson(*plan);
}

} // namespace rpu
} // namespace mlir
