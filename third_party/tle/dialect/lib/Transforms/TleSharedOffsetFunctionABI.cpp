// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"

namespace mlir::triton::tle {
#define GEN_PASS_DEF_TRITONTLESHAREDOFFSETFUNCTIONABI
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {
constexpr StringLiteral kOffsetABI = "tle.shared_offset_abi";

// Memdesc values contain a Shared base plus scalar offsets. Only the pointer
// leaves change representation; global pointers and scalar offsets do not.
static Type offsetType(Type type) {
  if (auto ptr = dyn_cast<LLVM::LLVMPointerType>(type))
    return ptr.getAddressSpace() == 3 ? IntegerType::get(type.getContext(), 32)
                                      : type;
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (structure.isOpaque())
      return type;
    SmallVector<Type> elements;
    for (Type element : structure.getBody())
      elements.push_back(offsetType(element));
    if (llvm::equal(elements, structure.getBody()))
      return type;
    return LLVM::LLVMStructType::getLiteral(type.getContext(), elements,
                                           structure.isPacked());
  }
  if (auto array = dyn_cast<LLVM::LLVMArrayType>(type))
    return LLVM::LLVMArrayType::get(offsetType(array.getElementType()),
                                    array.getNumElements());
  return type;
}

static Value convertValue(OpBuilder &builder, Location loc, Value value,
                          Type originalType, Value arena, bool encode) {
  Type encodedType = offsetType(originalType);
  if (encodedType == originalType)
    return value;
  if (isa<LLVM::LLVMPointerType>(originalType)) {
    auto i32 = builder.getI32Type();
    Value base = LLVM::PtrToIntOp::create(builder, loc, i32, arena);
    if (encode) {
      Value pointer = LLVM::PtrToIntOp::create(builder, loc, i32, value);
      return LLVM::SubOp::create(builder, loc, pointer, base);
    }
    Value pointer = LLVM::AddOp::create(builder, loc, value, base);
    return LLVM::IntToPtrOp::create(builder, loc, originalType, pointer);
  }
  Type resultType = encode ? encodedType : originalType;
  Value result = LLVM::UndefOp::create(builder, loc, resultType);
  SmallVector<Type> elements;
  if (auto structure = dyn_cast<LLVM::LLVMStructType>(originalType))
    elements.append(structure.getBody().begin(), structure.getBody().end());
  else {
    auto array = cast<LLVM::LLVMArrayType>(originalType);
    elements.assign(array.getNumElements(), array.getElementType());
  }
  for (auto [index, elementType] : llvm::enumerate(elements)) {
    SmallVector<int64_t> position{static_cast<int64_t>(index)};
    Value element = LLVM::ExtractValueOp::create(builder, loc, value, position);
    element = convertValue(builder, loc, element, elementType, arena, encode);
    result = LLVM::InsertValueOp::create(builder, loc, result, element, position);
  }
  return result;
}

struct TritonTleSharedOffsetFunctionABI
    : impl::TritonTleSharedOffsetFunctionABIBase<TritonTleSharedOffsetFunctionABI> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto arena = module.lookupSymbol<LLVM::GlobalOp>("global_smem");
    if (!arena || arena.getAddrSpace() != 3)
      return;
    llvm::MapVector<Operation *, SmallVector<Type>> functions;
    for (auto function : module.getOps<LLVM::LLVMFuncOp>()) {
      if (function.isExternal() || function.getLinkage() != LLVM::Linkage::Internal ||
          function->hasAttr(kOffsetABI))
        continue;
      auto inputs = function.getFunctionType().getParams();
      if (llvm::none_of(inputs, [](Type type) { return offsetType(type) != type; }))
        continue;
      if (function.getFunctionType().isVarArg()) {
        function.emitError("Shared offset ABI requires non-variadic functions");
        return signalPassFailure();
      }
      for (auto [index, type] : llvm::enumerate(inputs)) {
        if (offsetType(type) == type)
          continue;
        for (StringRef attr : {"llvm.byval", "llvm.byref", "llvm.inalloca",
                               "llvm.preallocated", "llvm.sret"}) {
          if (function.getArgAttr(index, attr)) {
            function.emitError("Shared offset ABI cannot change copy/return argument conventions");
            return signalPassFailure();
          }
        }
      }
      // This is a closed, direct-call ABI. An address-taken function could be
      // called by code outside this module whose argument convention is fixed.
      auto uses = SymbolTable::getSymbolUses(function, module);
      if (!uses || llvm::any_of(*uses, [](const SymbolTable::SymbolUse &use) {
            return !isa<LLVM::CallOp>(use.getUser());
          })) {
        function.emitError("Shared offset ABI requires direct calls only");
        return signalPassFailure();
      }
      functions[function] = SmallVector<Type>(inputs);
    }
    OpBuilder builder(module.getContext());
    module.walk([&](LLVM::CallOp call) {
      if (!call.getCallee())
        return;
      auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(*call.getCallee());
      auto found = functions.find(function);
      if (found == functions.end())
        return;
      builder.setInsertionPoint(call);
      Value base = LLVM::AddressOfOp::create(builder, call.getLoc(), arena);
      SmallVector<Value> arguments;
      for (auto [value, type] : llvm::zip_equal(call.getCalleeOperands(), found->second))
        arguments.push_back(convertValue(builder, call.getLoc(), value, type, base, true));
      call.getCalleeOperandsMutable().assign(arguments);
    });
    for (auto &[operation, inputs] : functions) {
      auto function = cast<LLVM::LLVMFuncOp>(operation);
      builder.setInsertionPointToStart(&function.getBody().front());
      Value base = LLVM::AddressOfOp::create(builder, function.getLoc(), arena);
      SmallVector<Type> encodedInputs;
      for (auto [argument, type] : llvm::zip_equal(function.getArguments(), inputs)) {
        Type encoded = offsetType(type);
        encodedInputs.push_back(encoded);
        if (encoded == type)
          continue;
        // These facts constrain the original pointer, not its integer offset
        // (offset zero is legal even for a nonnull pointer). Do not attach
        // pointer-only optimization attributes to the new integer argument.
        for (StringRef attr : {"llvm.align", "llvm.noalias", "llvm.nonnull",
                               "llvm.dereferenceable", "llvm.dereferenceable_or_null",
                               "llvm.nocapture", "llvm.readonly", "llvm.writeonly",
                               "llvm.readnone"})
          function.removeArgAttr(argument.getArgNumber(), attr);
        SmallVector<OpOperand *> uses;
        for (OpOperand &use : argument.getUses())
          uses.push_back(&use);
        argument.setType(encoded);
        Value pointer = convertValue(builder, function.getLoc(), argument, type, base, false);
        for (OpOperand *use : uses)
          use->set(pointer);
      }
      auto original = function.getFunctionType();
      function.setFunctionType(LLVM::LLVMFunctionType::get(
          original.getReturnType(), encodedInputs, original.isVarArg()));
      function->setAttr(kOffsetABI, builder.getUnitAttr());
    }
  }
};
} // namespace

} // namespace mlir::triton::tle
