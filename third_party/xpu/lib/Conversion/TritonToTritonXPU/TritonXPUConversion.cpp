#include "triton/Dialect/TritonXPU/Transforms/TritonXPUConversion.h"

#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include <cstdint>

using namespace mlir;

//
// TypeConverter
//
TritonXPUTypeConverter::TritonXPUTypeConverter(MLIRContext *context,
                                               uint32_t buffer_size,
                                               uint32_t core_num, bool isTLE)
    : context(context), buffer_size(buffer_size), core_num(core_num),
      isTLE(isTLE) {

  addConversion([](Type type) { return type; });

  bool isTLELocal = this->isTLE;
  addConversion([this,
                 isTLELocal](RankedTensorType tensorType) -> RankedTensorType {
    if (tensorType.getEncoding())
      return tensorType;

    // TLE path only: LOCAL-memory pointer-element tensors (address space 0,
    // produced by tle_local_ptr) stay unencoded and are lowered directly in
    // TritonXPUToLLVM without a ClusterLayout. The normal path always assigns a
    // ClusterLayout (including to GM pointer tensors feeding gm2lm / the source
    // of tle_normcopy_g2l), so this exception is gated behind isTLE and
    // restricted to address space 0 — GM pointer tensors (address space != 0)
    // must still receive a ClusterLayout like every other tensor, otherwise
    // they reach the LLVM conversion with a null encoding and crash.
    if (isTLELocal) {
      if (auto ptrTy =
              dyn_cast<triton::PointerType>(tensorType.getElementType())) {
        if (ptrTy.getAddressSpace() == 0)
          return tensorType;
      }
    }

    ArrayRef<int64_t> shape = tensorType.getShape();
    triton::xpu::ClusterLayoutAttr encoding =
        triton::xpu::getDefaultClusterEncoding(
            this->context, shape, this->buffer_size, this->core_num);
    return RankedTensorType::get(shape, tensorType.getElementType(), encoding);
  });

  // TODO[dyq]: check addConversion for triton::PointerType

  //
  // Materializations
  //
  // Note: addArgumentMaterialization was removed in newer MLIR. Argument
  // remats now go through addSourceMaterialization.
  // If the origValue still has live user(s), use this to
  // convert origValue to newValue
  addSourceMaterialization([isTLELocal](
                               OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) -> Value {
    // TLE path only: a legal op (e.g. tt.load on an LM ptr-tensor) produces an
    // unencoded result consumed by a converted op (e.g. arith.addf); bridge the
    // type gap with a ConvertLayoutOp. In the normal path this must never
    // happen, so keep the original llvm_unreachable to avoid silently altering
    // non-TLE lowering.
    if (!isTLELocal) {
      llvm_unreachable("Source rematerialization should not happen in Triton "
                       "-> TritonXPU Conversion");
      return Value();
    }
    auto cast =
        builder.create<triton::xpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return cast.getResult();
  });

  // This will be called when (desiredType != newOperandType)
  // where, desiredType = typeConverter->convertType(origType)
  // NOTE: only for remapped values.
  addTargetMaterialization([&](OpBuilder &builder, RankedTensorType tensorType,
                               ValueRange inputs, Location loc) -> Value {
    auto cast =
        builder.create<triton::xpu::ConvertLayoutOp>(loc, tensorType, inputs);
    return cast.getResult();
  });
}

//
// TritonXPUConversion
//
TritonXPUConversionTarget::TritonXPUConversionTarget(
    MLIRContext &context, TritonXPUTypeConverter &typeConverter, bool isTLE)
    : ConversionTarget(context) {

  addLegalDialect<triton::xpu::TritonXPUDialect>();

  if (isTLE) {
    // TLE path only: allow ttg ops (ttg::LocalAllocOp/LocalLoadOp/LocalStoreOp)
    // to survive TritonToTritonXPU; they are lowered in TritonXPUToLLVM.
    addLegalDialect<triton::gpu::TritonGPUDialect>();

    // GM normcopy ops must consume ClusterLayout-encoded pointer tensors (like
    // the regular gm2lm path). Mark them illegal while their pointer operand is
    // still unencoded so the GenericOpPattern rebuilds them with the converted
    // (encoded) operand. Otherwise a target materialization inserts an
    // encoded->unencoded ConvertLayout bridge that cannot be lowered in
    // TritonXPUToLLVM (the op then "fails to legalize").
    addDynamicallyLegalOp<triton::xpu::TLENormCopyGlobalToLocalOp,
                          triton::xpu::TLENormCopyLocalToGlobalOp>(
        [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });
  }

  // Some ops from SCF are illegal
  // TODO[dyq]: addIllegalOp necessary?
  //   addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
  //                scf::ReduceReturnOp>();

  addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect,
                             triton::TritonDialect, cf::ControlFlowDialect,
                             scf::SCFDialect>(
      [&typeConverter, isTLE](Operation *op) {
        // TLE path only: tt.load/tt.store operating on LOCAL-memory
        // pointer-element tensors (address space 0, from tle_local_ptr) are
        // legal as-is — they bypass type conversion and are lowered directly in
        // TritonXPUToLLVM. Gated behind isTLE and restricted to address space 0
        // so GM loads/stores (also ptr-element tensors) keep being converted to
        // gm2lm.
        if (isTLE) {
          if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
            if (auto ptrTy =
                    dyn_cast<RankedTensorType>(loadOp.getPtr().getType())) {
              if (auto elemPtrTy =
                      dyn_cast<triton::PointerType>(ptrTy.getElementType()))
                if (elemPtrTy.getAddressSpace() == 0)
                  return true;
            }
          }
          if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
            if (auto ptrTy =
                    dyn_cast<RankedTensorType>(storeOp.getPtr().getType())) {
              if (auto elemPtrTy =
                      dyn_cast<triton::PointerType>(ptrTy.getElementType()))
                if (elemPtrTy.getAddressSpace() == 0)
                  return true;
            }
          }
        }
        bool hasLegalRegions = true;
        for (auto &region : op->getRegions()) {
          hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
        }
        if (hasLegalRegions && typeConverter.isLegal(op)) {
          return true;
        }
        return false;
      });

  // TODO[dyq]: XPUSDNN-CHECK check addDynamicallyLegalDialect for triton::DotOp
}
