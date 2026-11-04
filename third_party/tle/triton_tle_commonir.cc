// [metax+CommonIR] Slim CommonIR-only TLE bindings.
// Mirrors only the __FLAGTREE_COMMON_IR__ tile.* builder methods from triton_tle.cc.
// Depends on FLIR CommonIR (tile) only — no TLE-raw/Lite and no ttng Hopper.
#include "Python.h"
#include "ir.h"
#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/PassManager.h"
#include "CommonIRToTTGIR/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>

namespace py = pybind11;
using namespace mlir;
namespace tile = triton::tile;

static std::string attrToLowerString(Attribute attr) {
  if (!attr) return "";
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  os.flush();
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return text;
}
static tile::MemorySpace attrToCommonIRMemorySpace(Attribute attr) {
  auto text = attrToLowerString(attr);
  if (text.find("register") != std::string::npos) return tile::MemorySpace::Register;
  if (text.find("shared") != std::string::npos || text.find("smem") != std::string::npos) return tile::MemorySpace::Shared;
  if (text.find("global") != std::string::npos) return tile::MemorySpace::Global;
  if (text.find("local") != std::string::npos) return tile::MemorySpace::Local;
  return tile::MemorySpace::Shared;
}

void init_triton_tle_passes_commonir(py::module &&m) {
  m.def("add_to_ttgir", [](mlir::PassManager &pm, bool enableAsyncCopy) {
    mlir::triton::TleCommonIRToTTGIROptions opts;
    opts.enableAsyncCopy = enableAsyncCopy;
    pm.addPass(mlir::triton::createTleCommonIRToTTGIR(opts));
  });
}

void init_triton_tle(py::module &&m) {
  m.def("is_common_ir_enabled", []() { return true; });
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    context.loadAllAvailableDialects();
  });
  m.def("load_tile_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::tile::CommonIRDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
  init_triton_tle_passes_commonir(m.def_submodule("passes").def_submodule("commonir"));
  auto &builder_cls = *ir::getBuilderClass();
  builder_cls
      .def("tile_get_string_attr",
           [](TritonOpBuilder &self, const std::string &name) -> Attribute {
             return self.getBuilder().getStringAttr(name);
           })
      .def("tile_get_buffer_type",
           [](TritonOpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const Attribute &memorySpace) -> Type {
             auto memSpace = attrToCommonIRMemorySpace(memorySpace);
             return tile::BufType::get(self.getBuilder().getContext(), shape,
                                       elementType, memSpace);
           })
      .def("create_tile_alloc",
           [](TritonOpBuilder &self, Type tileBufType,
              Attribute targetLayout) -> Value {
             auto bufType = mlir::cast<tile::BufType>(tileBufType);
             auto op = self.create<tile::AllocOp>(
                 tileBufType, bufType.getMemorySpace(),
                 /*shape=*/mlir::ArrayAttr(), /*dtype=*/mlir::TypeAttr(),
                 /*policy=*/tile::PolicyAttr(),
                 /*layout=*/
                 tile::LayoutAttr::get(self.getBuilder().getContext(),
                                       tile::Layout::ND),
                 /*lifetime=*/tile::LifetimeAttr(),
                 /*comment=*/mlir::StringAttr());
             op->setAttr("tle.gpu_layout", targetLayout);
             return op.getResult();
           })
      .def("create_tile_copy",
           [](TritonOpBuilder &self, Value &src, Value &dst,
              bool interNoAlias) -> void {
             auto op = self.create<tile::CopyOp>(
                 src, dst, /*engine=*/tile::EngineAttr(),
                 /*src_layout=*/
                 tile::LayoutAttr::get(self.getBuilder().getContext(),
                                       tile::Layout::ND),
                 /*dst_nz_layout=*/tile::NZLayoutAttr(),
                 /*transpose=*/mlir::UnitAttr(),
                 /*comment=*/mlir::StringAttr());
             if (interNoAlias)
               op->setAttr("inter_no_alias",
                           self.getBuilder().getBoolAttr(true));
           })
      .def("create_tile_get_memdesc",
           [](TritonOpBuilder &self, Type resultTy, Value source) -> Value {
             return self
                 .create<UnrealizedConversionCastOp>(TypeRange{resultTy},
                                                     ValueRange{source})
                 .getResult(0);
           })
      .def("create_tile_subview",
           [](TritonOpBuilder &self, Value source, std::vector<Value> &offsets,
              const std::vector<int64_t> &sizes,
              const std::vector<int64_t> &strides,
              Attribute targetLayout) -> Value {
             SmallVector<Value> indexOffsets;
             auto &builder = self.getBuilder();
             auto indexType = builder.getIndexType();
             for (Value offset : offsets) {
               if (offset.getType() != indexType)
                 offset = self.create<arith::IndexCastOp>(indexType, offset);
               indexOffsets.push_back(offset);
             }
             auto srcBuf = mlir::cast<tile::BufType>(source.getType());
             auto resTy = tile::BufType::get(builder.getContext(), sizes,
                                             srcBuf.getElementType(),
                                             srcBuf.getMemorySpace());
             auto op = self.create<tile::SubViewOp>(
                 resTy, source, indexOffsets, builder.getI64ArrayAttr(sizes),
                 builder.getI64ArrayAttr(strides));
             op->setAttr("tle.gpu_layout", targetLayout);
             return op.getResult();
           })
      .def("create_tile_to_tensor",
           [](TritonOpBuilder &self, Value &src, bool /*writable*/) -> Value {
             auto srcBuf = mlir::cast<tile::BufType>(src.getType());
             auto resTy = RankedTensorType::get(srcBuf.getShape(),
                                                srcBuf.getElementType());
             return self.create<tile::ToTensorOp>(resTy, src).getResult();
           })
      .def("create_tile_store_tensor",
           [](TritonOpBuilder &self, Value &src, Value &dst) -> void {
             self.create<tile::StoreTensorOp>(src, dst);
           })
      .def("create_tile_gm_offset",
           [](TritonOpBuilder &self, Value &base, std::vector<Value> &indices,
              std::vector<Value> &strides) -> Value {
             SmallVector<Value> indexValues;
             SmallVector<Value> strideValues;
             auto &builder = self.getBuilder();
             auto indexType = builder.getIndexType();
             for (Value index : indices) {
               if (index.getType() != indexType)
                 index = self.create<arith::IndexCastOp>(indexType, index);
               indexValues.push_back(index);
             }
             for (Value stride : strides) {
               if (stride.getType() != indexType)
                 stride = self.create<arith::IndexCastOp>(indexType, stride);
               strideValues.push_back(stride);
             }
             return self
                 .create<tile::GmOffsetOp>(base.getType(), base, indexValues,
                                           strideValues)
                 .getResult();
           })
      ;
}
