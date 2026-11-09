/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */

#include "TritonToGraph/ProgramMappingScalarSpecialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

constexpr llvm::StringLiteral kVersion = "version";
constexpr llvm::StringLiteral kArguments = "arguments";
constexpr llvm::StringLiteral kIndex = "index";
constexpr llvm::StringLiteral kName = "name";
constexpr llvm::StringLiteral kValue = "value";

bool hasExactKeys(DictionaryAttr dictionary, ArrayRef<llvm::StringRef> keys) {
  if (dictionary.size() != keys.size())
    return false;
  return llvm::all_of(keys,
                      [&](llvm::StringRef key) { return dictionary.get(key); });
}

std::optional<int64_t> getInteger(DictionaryAttr dictionary,
                                  llvm::StringRef name) {
  auto value = dyn_cast_or_null<IntegerAttr>(dictionary.get(name));
  if (!value)
    return std::nullopt;
  return value.getInt();
}

std::optional<llvm::StringRef> getString(DictionaryAttr dictionary,
                                         llvm::StringRef name) {
  auto value = dyn_cast_or_null<StringAttr>(dictionary.get(name));
  if (!value || value.getValue().empty())
    return std::nullopt;
  return value.getValue();
}

void setAttrOnFunctions(ModuleOp module, llvm::StringRef name,
                        Attribute attribute) {
  module.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() == "tt.func")
      operation->setAttr(name, attribute);
  });
}

void removeAttrFromFunctions(ModuleOp module, llvm::StringRef name) {
  module.walk([&](Operation *operation) {
    if (operation->getName().getStringRef() == "tt.func")
      operation->removeAttr(name);
  });
}

bool isPublicEntry(triton::FuncOp function) {
  auto visibility = function->getAttrOfType<StringAttr>("sym_visibility");
  return !visibility || visibility.getValue() == "public";
}

bool sameScalarSpecialization(const ProgramMappingScalarSpecialization &lhs,
                              const ProgramMappingScalarSpecialization &rhs) {
  if (lhs.version != rhs.version ||
      lhs.arguments.size() != rhs.arguments.size())
    return false;
  for (unsigned index = 0; index < lhs.arguments.size(); ++index) {
    const ProgramMappingScalarArgument &left = lhs.arguments[index];
    const ProgramMappingScalarArgument &right = rhs.arguments[index];
    if (left.index != right.index || left.name != right.name ||
        left.value != right.value)
      return false;
  }
  return true;
}

bool isRepresentableInIntegerType(int64_t value, IntegerType type) {
  const unsigned width = type.getWidth();
  if (width == 0)
    return false;
  if (width >= 64)
    return true;
  const int64_t minimum = -(int64_t{1} << (width - 1));
  const int64_t maximum = (int64_t{1} << (width - 1)) - 1;
  return value >= minimum && value <= maximum;
}

std::optional<BlockArgument> findEntryArgumentByName(triton::FuncOp function,
                                                     llvm::StringRef name) {
  std::optional<BlockArgument> found;
  for (unsigned index = 0; index < function.getNumArguments(); ++index) {
    BlockArgument argument = function.getArgument(index);
    auto namedLocation = dyn_cast<NameLoc>(argument.getLoc());
    if (!namedLocation || namedLocation.getName().getValue() != name)
      continue;
    if (found)
      return std::nullopt;
    found = argument;
  }
  return found;
}

} // namespace

FailureOr<ProgramMappingScalarSpecialization>
mlir::triton::cfg::parseProgramMappingScalarSpecialization(
    Attribute attribute) {
  auto dictionary = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!dictionary || !hasExactKeys(dictionary, {kVersion, kArguments}))
    return failure();

  std::optional<int64_t> version = getInteger(dictionary, kVersion);
  auto arguments = dyn_cast_or_null<ArrayAttr>(dictionary.get(kArguments));
  if (!version ||
      (*version != kProgramMappingScalarSpecializationLegacyVersion &&
       *version != kProgramMappingScalarSpecializationVersion) ||
      !arguments || arguments.empty())
    return failure();

  ProgramMappingScalarSpecialization parsed;
  parsed.version = *version;
  std::optional<int64_t> previousIndex;
  for (Attribute attribute : arguments) {
    auto argument = dyn_cast<DictionaryAttr>(attribute);
    const bool isLegacy =
        *version == kProgramMappingScalarSpecializationLegacyVersion;
    if (!argument || (isLegacy && !hasExactKeys(argument, {kIndex, kValue})) ||
        (!isLegacy && !hasExactKeys(argument, {kIndex, kName, kValue})))
      return failure();
    std::optional<int64_t> index = getInteger(argument, kIndex);
    std::optional<int64_t> value = getInteger(argument, kValue);
    std::optional<llvm::StringRef> name =
        isLegacy ? std::optional<llvm::StringRef>(llvm::StringRef())
                 : getString(argument, kName);
    if (!index || !value || *index < 0 ||
        *index > std::numeric_limits<uint32_t>::max() || !name ||
        (previousIndex && *index <= *previousIndex))
      return failure();
    parsed.arguments.push_back(ProgramMappingScalarArgument{
        static_cast<uint32_t>(*index), name->str(), *value});
    previousIndex = *index;
  }
  return parsed;
}

DictionaryAttr mlir::triton::cfg::serializeProgramMappingScalarSpecialization(
    MLIRContext *context,
    const ProgramMappingScalarSpecialization &specialization) {
  Builder builder(context);
  SmallVector<Attribute> arguments;
  arguments.reserve(specialization.arguments.size());
  for (const ProgramMappingScalarArgument &argument :
       specialization.arguments) {
    SmallVector<NamedAttribute> fields;
    fields.emplace_back(builder.getStringAttr(kIndex),
                        builder.getI64IntegerAttr(argument.index));
    if (specialization.version == kProgramMappingScalarSpecializationVersion)
      fields.emplace_back(builder.getStringAttr(kName),
                          builder.getStringAttr(argument.name));
    fields.emplace_back(builder.getStringAttr(kValue),
                        builder.getI64IntegerAttr(argument.value));
    arguments.push_back(DictionaryAttr::get(context, fields));
  }
  return DictionaryAttr::get(
      context, {{builder.getStringAttr(kVersion),
                 builder.getI64IntegerAttr(specialization.version)},
                {builder.getStringAttr(kArguments),
                 ArrayAttr::get(context, arguments)}});
}

LogicalResult mlir::triton::cfg::setProgramMappingScalarSpecialization(
    ModuleOp module, const ProgramMappingScalarSpecialization &specialization) {
  DictionaryAttr serialized = serializeProgramMappingScalarSpecialization(
      module.getContext(), specialization);
  if (failed(parseProgramMappingScalarSpecialization(serialized)))
    return failure();
  module->setAttr(kProgramMappingScalarSpecializationAttr, serialized);
  setAttrOnFunctions(module, kProgramMappingScalarSpecializationAttr,
                     serialized);
  return success();
}

void mlir::triton::cfg::clearProgramMappingScalarSpecialization(
    ModuleOp module) {
  module->removeAttr(kProgramMappingScalarSpecializationAttr);
  removeAttrFromFunctions(module, kProgramMappingScalarSpecializationAttr);
}

LogicalResult
mlir::triton::cfg::applyProgramMappingScalarSpecialization(ModuleOp module) {
  Attribute attribute =
      module->getAttr(kProgramMappingScalarSpecializationAttr);
  if (!attribute) {
    clearProgramMappingScalarSpecialization(module);
    return success();
  }

  FailureOr<ProgramMappingScalarSpecialization> specialization =
      parseProgramMappingScalarSpecialization(attribute);
  if (failed(specialization)) {
    clearProgramMappingScalarSpecialization(module);
    return failure();
  }

  for (triton::FuncOp function : module.getOps<triton::FuncOp>()) {
    Attribute functionAttribute =
        function->getAttr(kProgramMappingScalarSpecializationAttr);
    if (functionAttribute) {
      FailureOr<ProgramMappingScalarSpecialization> functionSpecialization =
          parseProgramMappingScalarSpecialization(functionAttribute);
      if (failed(functionSpecialization) ||
          !sameScalarSpecialization(*specialization, *functionSpecialization)) {
        clearProgramMappingScalarSpecialization(module);
        return failure();
      }
    }
    if (!isPublicEntry(function) || function->getNumRegions() != 1 ||
        function.getBody().empty())
      continue;

    Block &entry = function.getBody().front();
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToStart(&entry);
    for (const ProgramMappingScalarArgument &argument :
         specialization->arguments) {
      std::optional<BlockArgument> blockArgument;
      if (specialization->version ==
          kProgramMappingScalarSpecializationLegacyVersion) {
        if (argument.index < function.getNumArguments())
          blockArgument = function.getArgument(argument.index);
      } else {
        blockArgument = findEntryArgumentByName(function, argument.name);
      }
      if (!blockArgument)
        continue;
      auto integerType = dyn_cast<IntegerType>(blockArgument->getType());
      if (!integerType ||
          !isRepresentableInIntegerType(argument.value, integerType))
        continue;
      Value constant = builder.create<arith::ConstantIntOp>(
          function.getLoc(), argument.value, integerType.getWidth());
      blockArgument->replaceAllUsesWith(constant);
    }
  }

  clearProgramMappingScalarSpecialization(module);
  return success();
}
