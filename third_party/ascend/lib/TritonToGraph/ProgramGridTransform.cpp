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
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToGraph/ProgramGridTransform.h"

#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <array>
#include <optional>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

constexpr llvm::StringLiteral kVersion = "version";
constexpr llvm::StringLiteral kExtentSource = "extent_source";
constexpr llvm::StringLiteral kHiddenExtentAxes = "hidden_extent_axes";
constexpr llvm::StringLiteral kHiddenArgumentOrder = "hidden_argument_order";
constexpr llvm::StringLiteral kHiddenArgumentTypes = "hidden_argument_types";
constexpr llvm::StringLiteral kTransforms = "transforms";
constexpr llvm::StringLiteral kOrder = "order";
constexpr llvm::StringLiteral kKind = "kind";
constexpr llvm::StringLiteral kAxis = "axis";
constexpr llvm::StringLiteral kFactor = "factor";
constexpr llvm::StringLiteral kPersistentCoverage = "persistent_coverage";
constexpr llvm::StringLiteral kGridStrideAbiVerified =
    "grid_stride_abi_verified";
constexpr llvm::StringLiteral kCeilDiv = "ceil_div";
constexpr llvm::StringLiteral kRuntimeOriginalGrid = "runtime_original_grid";
constexpr llvm::StringLiteral kOriginalGridX = "originalGridX";
constexpr llvm::StringLiteral kOriginalGridY = "originalGridY";
constexpr llvm::StringLiteral kI32 = "i32";
constexpr llvm::StringLiteral kHiddenXName = "__hacc_original_grid_x";
constexpr llvm::StringLiteral kHiddenYName = "__hacc_original_grid_y";

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

std::optional<bool> getBoolean(DictionaryAttr dictionary,
                               llvm::StringRef name) {
  auto value = dyn_cast_or_null<BoolAttr>(dictionary.get(name));
  if (!value)
    return std::nullopt;
  return value.getValue();
}

bool hasExactIntegerArray(Attribute attribute, ArrayRef<int64_t> expected) {
  auto values = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!values || values.size() != expected.size())
    return false;
  for (auto [value, wanted] : llvm::zip(values, expected)) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer || integer.getInt() != wanted)
      return false;
  }
  return true;
}

bool hasExactStringArray(Attribute attribute,
                         ArrayRef<llvm::StringRef> expected) {
  auto values = dyn_cast_or_null<ArrayAttr>(attribute);
  if (!values || values.size() != expected.size())
    return false;
  for (auto [value, wanted] : llvm::zip(values, expected)) {
    auto string = dyn_cast<StringAttr>(value);
    if (!string || string.getValue() != wanted)
      return false;
  }
  return true;
}

bool isPublicEntry(triton::FuncOp function) {
  auto visibility = function->getAttrOfType<StringAttr>("sym_visibility");
  return !visibility || visibility.getValue() == "public";
}

bool hasNamedI32Argument(BlockArgument argument, llvm::StringRef name) {
  auto integer = dyn_cast<IntegerType>(argument.getType());
  auto location = dyn_cast<NameLoc>(argument.getLoc());
  return integer && integer.getWidth() == 32 && location &&
         location.getName().getValue() == name;
}

bool isIAT16(const ProgramGridTransform &transform) {
  return transform.axis == 1 && transform.factor == 16 &&
         !transform.persistentCoverage && !transform.gridStrideAbiVerified;
}

bool isPTSM4(const ProgramGridTransform &transform) {
  return transform.axis == 0 && transform.factor == 4 &&
         transform.persistentCoverage && transform.gridStrideAbiVerified;
}

bool isPTSM64(const ProgramGridTransform &transform) {
  return transform.axis == 0 && transform.factor == 64 &&
         transform.persistentCoverage && transform.gridStrideAbiVerified;
}

bool isDotRowTransform(const ProgramGridTransform &transform) {
  return transform.axis == 0 && !transform.persistentCoverage &&
         !transform.gridStrideAbiVerified &&
         (transform.factor == 2 || transform.factor == 4 ||
          transform.factor == 8 || transform.factor == 16 ||
          transform.factor == 32);
}

bool isSupportedSequence(ArrayRef<ProgramGridTransform> transforms) {
  if (transforms.size() == 1)
    return isIAT16(transforms.front()) || isPTSM64(transforms.front()) ||
           isDotRowTransform(transforms.front());
  return transforms.size() == 2 && isIAT16(transforms[0]) &&
         isPTSM4(transforms[1]);
}

} // namespace

bool mlir::triton::cfg::hasProgramGridHiddenExtentArguments(
    triton::FuncOp function) {
  if (function.getNumArguments() < 2)
    return false;
  const unsigned xIndex = function.getNumArguments() - 2;
  return hasNamedI32Argument(function.getArgument(xIndex), kHiddenXName) &&
         hasNamedI32Argument(function.getArgument(xIndex + 1), kHiddenYName);
}

LogicalResult mlir::triton::cfg::addProgramGridHiddenExtentArguments(
    triton::FuncOp function) {
  if (hasProgramGridHiddenExtentArguments(function))
    return success();
  for (BlockArgument argument : function.getArguments()) {
    auto location = dyn_cast<NameLoc>(argument.getLoc());
    if (location && (location.getName().getValue() == kHiddenXName ||
                     location.getName().getValue() == kHiddenYName))
      return failure();
  }

  MLIRContext *context = function.getContext();
  SmallVector<DictionaryAttr> originalArgAttrs;
  const bool hasArgAttrs = static_cast<bool>(function.getAllArgAttrs());
  if (hasArgAttrs) {
    function.getAllArgAttrs(originalArgAttrs);
    if (originalArgAttrs.size() != function.getNumArguments())
      return failure();
  }

  Type i32 = IntegerType::get(context, 32);
  DictionaryAttr hiddenArgAttrs =
      hasArgAttrs ? DictionaryAttr::get(context) : DictionaryAttr();
  if (failed(function.insertArgument(
          function.getNumArguments(), i32, hiddenArgAttrs,
          NameLoc::get(StringAttr::get(context, kHiddenXName),
                       function.getLoc()))))
    return failure();
  if (failed(function.insertArgument(
          function.getNumArguments(), i32, hiddenArgAttrs,
          NameLoc::get(StringAttr::get(context, kHiddenYName),
                       function.getLoc()))))
    return failure();

  if (hasArgAttrs) {
    originalArgAttrs.push_back(DictionaryAttr::get(context));
    originalArgAttrs.push_back(DictionaryAttr::get(context));
    function.setAllArgAttrs(originalArgAttrs);
  } else {
    function->removeAttr("arg_attrs");
  }
  return success();
}

LogicalResult mlir::triton::cfg::commitProgramGridFunctionFromSandbox(
    triton::FuncOp destination, triton::FuncOp source) {
  if (source.getNumArguments() != source.getFunctionType().getNumInputs() ||
      !hasProgramGridHiddenExtentArguments(source))
    return failure();

  SmallVector<DictionaryAttr> sourceArgAttrs;
  const bool sourceHasArgAttrs = static_cast<bool>(source.getAllArgAttrs());
  if (sourceHasArgAttrs) {
    source.getAllArgAttrs(sourceArgAttrs);
    if (sourceArgAttrs.size() != source.getNumArguments())
      return failure();
  }

  destination.setFunctionType(source.getFunctionType());
  if (sourceHasArgAttrs)
    destination.setAllArgAttrs(sourceArgAttrs);
  else
    destination->removeAttr("arg_attrs");
  destination.getRegion().takeBody(source.getRegion());
  return success();
}

FailureOr<ProgramGridTransformContract>
mlir::triton::cfg::parseProgramGridTransformContract(Attribute attribute) {
  auto contract = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!contract)
    return failure();
  if (!hasExactKeys(contract,
                    {kVersion, kExtentSource, kHiddenExtentAxes,
                     kHiddenArgumentOrder, kHiddenArgumentTypes, kTransforms}))
    return failure();

  std::optional<int64_t> version = getInteger(contract, kVersion);
  auto extentSource = dyn_cast_or_null<StringAttr>(contract.get(kExtentSource));
  constexpr std::array<int64_t, 2> kAxes = {0, 1};
  constexpr std::array<llvm::StringRef, 2> kArgumentOrder = {kOriginalGridX,
                                                             kOriginalGridY};
  constexpr std::array<llvm::StringRef, 2> kArgumentTypes = {kI32, kI32};
  if (!version || *version != kProgramGridTransformsVersion || !extentSource ||
      extentSource.getValue() != kRuntimeOriginalGrid ||
      !hasExactIntegerArray(contract.get(kHiddenExtentAxes), kAxes) ||
      !hasExactStringArray(contract.get(kHiddenArgumentOrder),
                           kArgumentOrder) ||
      !hasExactStringArray(contract.get(kHiddenArgumentTypes), kArgumentTypes))
    return failure();

  auto transforms = dyn_cast_or_null<ArrayAttr>(contract.get(kTransforms));
  if (!transforms || transforms.empty() || transforms.size() > 2)
    return failure();

  ProgramGridTransformContract parsed;
  parsed.version = *version;
  bool seenAxis[2] = {false, false};
  for (auto [expectedOrder, rawTransform] : llvm::enumerate(transforms)) {
    auto transform = dyn_cast<DictionaryAttr>(rawTransform);
    if (!transform ||
        !hasExactKeys(transform, {kOrder, kKind, kAxis, kFactor,
                                  kPersistentCoverage, kGridStrideAbiVerified}))
      return failure();
    std::optional<int64_t> order = getInteger(transform, kOrder);
    auto kind = dyn_cast_or_null<StringAttr>(transform.get(kKind));
    std::optional<int64_t> axis = getInteger(transform, kAxis);
    std::optional<int64_t> factor = getInteger(transform, kFactor);
    std::optional<bool> persistent = getBoolean(transform, kPersistentCoverage);
    std::optional<bool> gridStride =
        getBoolean(transform, kGridStrideAbiVerified);
    if (!order || !kind || !axis || !factor || !persistent || !gridStride ||
        *order != static_cast<int64_t>(expectedOrder) ||
        kind.getValue() != kCeilDiv || *axis < 0 || *axis > 1 || *factor < 2 ||
        *persistent != *gridStride || seenAxis[*axis])
      return failure();
    seenAxis[*axis] = true;
    parsed.transforms.push_back(ProgramGridTransform{
        static_cast<int32_t>(*order), static_cast<int32_t>(*axis), *factor,
        *persistent, *gridStride});
  }
  if (!isSupportedSequence(parsed.transforms))
    return failure();
  return parsed;
}

DictionaryAttr mlir::triton::cfg::serializeProgramGridTransformContract(
    MLIRContext *context, const ProgramGridTransformContract &contract) {
  Builder builder(context);
  SmallVector<Attribute> transforms;
  transforms.reserve(contract.transforms.size());
  for (const ProgramGridTransform &transform : contract.transforms) {
    transforms.push_back(DictionaryAttr::get(
        context,
        {{builder.getStringAttr(kOrder),
          builder.getI64IntegerAttr(transform.order)},
         {builder.getStringAttr(kKind), builder.getStringAttr(kCeilDiv)},
         {builder.getStringAttr(kAxis),
          builder.getI64IntegerAttr(transform.axis)},
         {builder.getStringAttr(kFactor),
          builder.getI64IntegerAttr(transform.factor)},
         {builder.getStringAttr(kPersistentCoverage),
          builder.getBoolAttr(transform.persistentCoverage)},
         {builder.getStringAttr(kGridStrideAbiVerified),
          builder.getBoolAttr(transform.gridStrideAbiVerified)}}));
  }
  return DictionaryAttr::get(
      context,
      {{builder.getStringAttr(kVersion),
        builder.getI64IntegerAttr(contract.version)},
       {builder.getStringAttr(kExtentSource),
        builder.getStringAttr(kRuntimeOriginalGrid)},
       {builder.getStringAttr(kHiddenExtentAxes),
        ArrayAttr::get(context, {builder.getI64IntegerAttr(0),
                                 builder.getI64IntegerAttr(1)})},
       {builder.getStringAttr(kHiddenArgumentOrder),
        ArrayAttr::get(context, {builder.getStringAttr(kOriginalGridX),
                                 builder.getStringAttr(kOriginalGridY)})},
       {builder.getStringAttr(kHiddenArgumentTypes),
        ArrayAttr::get(context, {builder.getStringAttr(kI32),
                                 builder.getStringAttr(kI32)})},
       {builder.getStringAttr(kTransforms),
        ArrayAttr::get(context, transforms)}});
}

LogicalResult mlir::triton::cfg::setProgramGridTransformContract(
    ModuleOp module, const ProgramGridTransformContract &contract) {
  DictionaryAttr serialized =
      serializeProgramGridTransformContract(module.getContext(), contract);
  if (failed(parseProgramGridTransformContract(serialized)))
    return failure();

  triton::FuncOp entry;
  for (triton::FuncOp function : module.getOps<triton::FuncOp>()) {
    if (!isPublicEntry(function))
      continue;
    if (entry)
      return failure();
    entry = function;
  }
  if (!entry || !hasProgramGridHiddenExtentArguments(entry))
    return failure();
  module->setAttr(kProgramGridTransformsAttr, serialized);
  return success();
}
