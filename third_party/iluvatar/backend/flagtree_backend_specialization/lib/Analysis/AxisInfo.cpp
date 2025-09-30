#include "triton/Analysis/AxisInfo.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton {

template <class T>
void AxisInfo::initPessimisticStateFromFunc(int argNumber, T funcOp,
                                            DimVectorT *contiguity,
                                            DimVectorT *divisibility,
                                            DimVectorT *constancy,
                                            DimVectorT *corexFlag) {
  // liast of attributes that we care about
  SmallVector<std::pair<DimVectorT *, std::string>> retVecs;
  retVecs.push_back({contiguity, "tt.contiguity"});
  retVecs.push_back({divisibility, "tt.divisibility"});
  retVecs.push_back({constancy, "tt.constancy"});
  retVecs.push_back({corexFlag, "tt.corex_stride"});

  // initialize attributes one by one
  for (auto [vec, attrName] : retVecs) {
    Attribute attr = funcOp.getArgAttr(argNumber, attrName);
    if (auto int_attr = dyn_cast_or_null<IntegerAttr>(attr))
      *vec = DimVectorT(contiguity->size(), int_attr.getValue().getZExtValue());
    if (auto dense_attr = dyn_cast_or_null<DenseElementsAttr>(attr)) {
      auto vals = dense_attr.getValues<int>();
      *vec = DimVectorT(vals.begin(), vals.end());
    }
  }
}

template void AxisInfo::initPessimisticStateFromFunc<mlir::FunctionOpInterface>(
    int argNumber, mlir::FunctionOpInterface funcOp, AxisInfo::DimVectorT *contiguity,
    AxisInfo::DimVectorT *divisibility, AxisInfo::DimVectorT *constancy,
    FLAGTREE_SPEC_AxisInfo_initPessimisticStateFromFunc_ARG spec_arg);

template void AxisInfo::initPessimisticStateFromFunc<mlir::LLVM::LLVMFuncOp>(
    int argNumber, mlir::LLVM::LLVMFuncOp funcOp, AxisInfo::DimVectorT *contiguity,
    AxisInfo::DimVectorT *divisibility, AxisInfo::DimVectorT *constancy,
    FLAGTREE_SPEC_AxisInfo_initPessimisticStateFromFunc_ARG spec_arg);

}

