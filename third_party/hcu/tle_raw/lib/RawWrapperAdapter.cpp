#include "hcu/tle_raw/include/RawWrapperAdapter.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace mlir::triton::HCU::tle_raw {

FailureOr<LLVM::LLVMFuncOp>
createRawWrapperAdapter(ModuleOp module, LLVM::LLVMFuncOp wrapper,
                        unsigned inputCount,
                        ArrayRef<RawFragmentOutput> outputs) {
  if (!wrapper)
    return failure();
  auto type = wrapper.getFunctionType();
  if (wrapper.isExternal() || type.isVarArg() ||
      !isa<LLVM::LLVMVoidType>(type.getReturnType()) ||
      type.getNumParams() != inputCount + outputs.size()) {
    wrapper.emitError("raw wrapper must return void and take ")
        << inputCount << " inputs followed by " << outputs.size()
        << " output pointers";
    return failure();
  }
  for (unsigned i = 0; i < type.getNumParams(); ++i) {
    if (wrapper.getArgAttr(i, "llvm.sret") ||
        wrapper.getArgAttr(i, "llvm.byval")) {
      wrapper.emitError("raw wrapper must not use sret or byval");
      return failure();
    }
  }
  if (outputs.empty())
    return wrapper;

  SmallVector<Type> resultTypes;
  for (auto [i, output] : llvm::enumerate(outputs)) {
    auto pointer =
        dyn_cast<LLVM::LLVMPointerType>(type.getParamType(inputCount + i));
    if (!pointer) {
      wrapper.emitError("HIP wrapper output must be a pointer at ") << i;
      return failure();
    }
    resultTypes.append(output.count, type.getParamType(output.inputOffset));
  }

  std::string name = wrapper.getSymName().str() + "_adapter_";
  if (module.lookupSymbol(name)) {
    wrapper.emitError("raw adapter symbol already exists: ") << name;
    return failure();
  }
  DataLayout layout(module);
  auto allocaSpace =
      dyn_cast_or_null<IntegerAttr>(layout.getAllocaMemorySpace());
  if (!allocaSpace) {
    wrapper.emitError("raw adapter requires an alloca memory space in "
                      "the source data layout");
    return failure();
  }

  OpBuilder builder(module.getContext());
  Location loc = wrapper.getLoc();
  builder.setInsertionPointAfter(wrapper);
  auto resultType =
      LLVM::LLVMStructType::getLiteral(module.getContext(), resultTypes);
  auto adapterType = LLVM::LLVMFunctionType::get(
      resultType, type.getParams().take_front(inputCount), false);
  auto adapter = builder.create<LLVM::LLVMFuncOp>(loc, name, adapterType,
                                                  LLVM::Linkage::Internal);

  SymbolTable::setSymbolVisibility(adapter, SymbolTable::Visibility::Private);
  adapter.setPassthroughAttr(
      builder.getArrayAttr({builder.getStringAttr("alwaysinline")}));
  // The adapter has a different result and performs its own memory accesses;
  // only input argument attributes and the calling convention are reusable.
  if (auto attrs = wrapper.getArgAttrsAttr())
    adapter.setArgAttrsAttr(
        builder.getArrayAttr(attrs.getValue().take_front(inputCount)));
  adapter.setCConv(wrapper.getCConv());

  Block *entry = adapter.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  auto privatePointer = LLVM::LLVMPointerType::get(
      module.getContext(), allocaSpace.getValue().getZExtValue());
  auto one = builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(),
                                              builder.getI32IntegerAttr(1));

  SmallVector<Value> arguments(entry->getArguments());
  SmallVector<Value> buffers;
  for (auto [i, output] : llvm::enumerate(outputs)) {
    Type element = type.getParamType(output.inputOffset);
    auto array = LLVM::LLVMArrayType::get(element, output.count);
    uint64_t alignment = layout.getTypeABIAlignment(array);
    if (auto required =
            wrapper.getArgAttrOfType<IntegerAttr>(inputCount + i, "llvm.align"))
      alignment = std::max(alignment, required.getValue().getZExtValue());
    Value buffer = builder.create<LLVM::AllocaOp>(loc, privatePointer, array,
                                                  one, alignment);
    buffers.push_back(buffer);
    Type target = type.getParamType(inputCount + i);
    if (target != privatePointer)
      buffer = builder.create<LLVM::AddrSpaceCastOp>(loc, target, buffer);
    arguments.push_back(buffer);
  }

  auto call = builder.create<LLVM::CallOp>(loc, wrapper, arguments);
  call.setAlwaysInline(true);
  call.setCConv(wrapper.getCConv());
  if (auto attrs = wrapper.getArgAttrsAttr())
    call.setArgAttrsAttr(attrs);

  Value result = builder.create<LLVM::UndefOp>(loc, resultType);
  unsigned field = 0;
  for (auto [i, output] : llvm::enumerate(outputs)) {
    Type element = type.getParamType(output.inputOffset);
    auto array = LLVM::LLVMArrayType::get(element, output.count);
    for (unsigned j = 0; j < output.count; ++j) {
      auto address = builder.create<LLVM::GEPOp>(
          loc, privatePointer, array, buffers[i],
          ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(j)});
      auto value = builder.create<LLVM::LoadOp>(
          loc, element, address, layout.getTypeABIAlignment(element));
      result = builder.create<LLVM::InsertValueOp>(loc, result, value,
                                                   ArrayRef<int64_t>{field++});
    }
  }
  builder.create<LLVM::ReturnOp>(loc, result);
  return adapter;
}

} // namespace mlir::triton::HCU::tle_raw
