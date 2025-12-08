/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <functional>
#include <string>

#include "ConstantUtil.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace gcu {

Value createConstantZero(OpBuilder &builder, Location loc, Type elemType) {
  if (elemType.isIntOrIndex()) {
    return builder.create<arith::ConstantIntOp>(loc, 0, elemType);
  } else if (elemType.isF32()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::IEEEsingle(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (elemType.isF16()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::IEEEhalf(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (elemType.isBF16()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::BFloat(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (elemType.isF64()) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::IEEEdouble(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (llvm::isa<Float8E4M3B11FNUZType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::Float8E4M3B11FNUZ(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (llvm::isa<Float8E4M3FNUZType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::Float8E4M3FNUZ(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (llvm::isa<Float8E5M2FNUZType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::Float8E5M2FNUZ(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (llvm::isa<Float8E4M3FNType>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::Float8E4M3FN(), "0"),
        dyn_cast<FloatType>(elemType));
  } else if (llvm::isa<Float8E5M2Type>(elemType)) {
    return builder.create<arith::ConstantFloatOp>(
        loc, APFloat(llvm::APFloatBase::Float8E5M2(), "0"),
        dyn_cast<FloatType>(elemType));
  } else {
    std::string o;
    llvm::raw_string_ostream os(o);
    elemType.print(os);
    llvm_unreachable((o + " is unsupported").c_str());
  }
  return Value();
}

Value createConstantNaN(OpBuilder &builder, Location loc, Type elemType) {
  const llvm::fltSemantics *sem = nullptr;
  if (elemType.isF32()) {
    sem = &llvm::APFloatBase::IEEEsingle();
  } else if (elemType.isBF16()) {
    sem = &llvm::APFloatBase::BFloat();
  } else if (elemType.isF16()) {
    sem = &llvm::APFloatBase::IEEEhalf();
  } else if (elemType.isF64()) {
    sem = &llvm::APFloatBase::IEEEdouble();
  } else if (llvm::isa<Float8E4M3B11FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3B11FNUZ();
  } else if (llvm::isa<Float8E4M3FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FNUZ();
  } else if (llvm::isa<Float8E5M2FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2FNUZ();
  } else if (llvm::isa<Float8E4M3FNType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FN();
  } else if (llvm::isa<Float8E5M2Type>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2();
  } else {
    std::string o;
    llvm::raw_string_ostream os(o);
    elemType.print(os);
    llvm_unreachable((o + " is unsupported").c_str());
  }
  return builder.create<arith::ConstantFloatOp>(loc, APFloat::getNaN(*sem),
                                                dyn_cast<FloatType>(elemType));
}

Value createConstantInf(OpBuilder &builder, Location loc, Type elemType,
                        bool isNegative) {
  const llvm::fltSemantics *sem = nullptr;
  if (elemType.isF32()) {
    sem = &llvm::APFloatBase::IEEEsingle();
  } else if (elemType.isBF16()) {
    sem = &llvm::APFloatBase::BFloat();
  } else if (elemType.isF16()) {
    sem = &llvm::APFloatBase::IEEEhalf();
  } else if (elemType.isF64()) {
    sem = &llvm::APFloatBase::IEEEdouble();
  } else if (llvm::isa<Float8E4M3B11FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3B11FNUZ();
  } else if (llvm::isa<Float8E4M3FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FNUZ();
  } else if (llvm::isa<Float8E5M2FNUZType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2FNUZ();
  } else if (llvm::isa<Float8E4M3FNType>(elemType)) {
    sem = &llvm::APFloatBase::Float8E4M3FN();
  } else if (llvm::isa<Float8E5M2Type>(elemType)) {
    sem = &llvm::APFloatBase::Float8E5M2();
  } else {
    std::string o;
    llvm::raw_string_ostream os(o);
    elemType.print(os);
    llvm_unreachable((o + " is unsupported").c_str());
  }
  return builder.create<arith::ConstantFloatOp>(
      loc, APFloat::getInf(*sem, isNegative), dyn_cast<FloatType>(elemType));
}

} // namespace gcu
} // namespace triton
} // namespace mlir
