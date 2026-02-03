/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
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

#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/Instructions.h"

using namespace mlir;
namespace py = pybind11;

struct AscendNPUIROpBuilder : public TritonOpBuilder {
  std::string target;
  static constexpr char kTarget910_95[] = "Ascend910_95";

  explicit AscendNPUIROpBuilder(MLIRContext *context, std::string target = "")
      : TritonOpBuilder(context), target(target) {}

  bool is_910_95() {
    // TODO: Use enum instead of strings after enabling HACC in satandalone
    // build
    constexpr size_t kTargetLen = sizeof(kTarget910_95) - 1;
    return target.size() >= kTargetLen &&
           target.compare(0, kTargetLen, kTarget910_95) == 0;
  }
};

namespace {
struct ModeAndPipes {
  hivm::SyncBlockModeAttr modeAttr = {};
  hivm::PipeAttr cubePipe = {};
  hivm::PipeAttr vectorPipe = {};
};

hivm::TCoreTypeAttr GetCore(MLIRContext *ctx, llvm::StringRef opName,
                            llvm::StringRef sender) {
  // Decide core type
  hivm::TCoreTypeAttr core;
  if (sender == "cube") {
    if (opName == "sync_block_set")
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
    else
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
  } else {
    if (sender != "vector") {
      throw std::runtime_error(
          "sync_block_set/wait only supports 'cube' or 'vector' as sender");
    }
    if (opName == "sync_block_set")
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
    else
      core = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
  }

  return core;
}

void buildSyncBlockOp(AscendNPUIROpBuilder &self, const std::string &opName,
                      std::string &sender, std::string &receiver, Value id,
                      hivm::PIPE senderPipe, hivm::PIPE receiverPipe) {
  auto *ctx = self.getBuilder().getContext();
  hivm::TCoreTypeAttr coreAttr = GetCore(ctx, opName, sender);
  hivm::PipeAttr prodPipe = hivm::PipeAttr::get(ctx, senderPipe);
  hivm::PipeAttr consPipe = hivm::PipeAttr::get(ctx, receiverPipe);
  const size_t I64 = 64;
  auto i64Ty = IntegerType::get(ctx, I64);
  Value idI64 = id;
  if (!id.getType().isInteger(I64)) {
    idI64 = mlir::convertScalarToDtype(self.getBuilder(), id.getLoc(), id,
                                       i64Ty, true);
  }
  if (opName == "sync_block_set") {
    self.create<hivm::SyncBlockSetOp>(coreAttr, prodPipe, consPipe, idI64);
  } else if (opName == "sync_block_wait") {
    self.create<hivm::SyncBlockWaitOp>(coreAttr, prodPipe, consPipe, idI64);
  } else {
    throw std::runtime_error("Unsupported operation name for SyncBlockOp");
  }
}

ModeAndPipes GetSyncBlockModeAndPipes(MLIRContext *ctx,
                                      const std::string &mode) {
  hivm::SyncBlockModeAttr modeAttr = {};
  hivm::PipeAttr cubePipe = {};
  hivm::PipeAttr vectorPipe = {};

  if (mode == "all_cube") {
    modeAttr = hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_CUBE);
    cubePipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    vectorPipe = hivm::PipeAttr{};
  } else if (mode == "all_vector") {
    modeAttr =
        hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_VECTOR);
    cubePipe = hivm::PipeAttr{};
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else if (mode == "all") {
    modeAttr = hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL);
    cubePipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else if (mode == "all_sub_vector") {
    modeAttr =
        hivm::SyncBlockModeAttr::get(ctx, hivm::SyncBlockMode::ALL_SUB_VECTOR);
    cubePipe = hivm::PipeAttr{};
    vectorPipe = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
  } else {
    llvm::report_fatal_error(
        llvm::StringRef("Invalid sync-block mode: " + mode));
  }
  return {modeAttr, cubePipe, vectorPipe};
}
} // namespace

void init_ascend_ir(py::module &&m) {
  py::enum_<hivm::AddressSpace>(m, "AddressSpace", py::module_local())
      .value("L1", hivm::AddressSpace::L1)
      .value("UB", hivm::AddressSpace::UB)
      .value("L0A", hivm::AddressSpace::L0A)
      .value("L0B", hivm::AddressSpace::L0B)
      .value("L0C", hivm::AddressSpace::L0C)
      .export_values();

  py::enum_<hivm::TCoreType>(m, "CoreType", py::module_local())
      .value("CUBE", hivm::TCoreType::CUBE)
      .value("VECTOR", hivm::TCoreType::VECTOR)
      .value("CUBE_OR_VECTOR", hivm::TCoreType::CUBE_OR_VECTOR)
      .value("CUBE_AND_VECTOR", hivm::TCoreType::CUBE_AND_VECTOR)
      .export_values();

  py::enum_<hivm::PIPE>(m, "PIPE", py::module_local())
      .value("PIPE_S", hivm::PIPE::PIPE_S)
      .value("PIPE_V", hivm::PIPE::PIPE_V)
      .value("PIPE_M", hivm::PIPE::PIPE_M)
      .value("PIPE_MTE1", hivm::PIPE::PIPE_MTE1)
      .value("PIPE_MTE2", hivm::PIPE::PIPE_MTE2)
      .value("PIPE_MTE3", hivm::PIPE::PIPE_MTE3)
      .value("PIPE_ALL", hivm::PIPE::PIPE_ALL)
      .value("PIPE_FIX", hivm::PIPE::PIPE_FIX)
      .export_values();

  py::enum_<hivm::VFMode>(m, "MODE", py::module_local())
      .value("SIMD", hivm::VFMode::SIMD)
      .value("SIMT", hivm::VFMode::SIMT)
      .value("MIX", hivm::VFMode::MIX)
      .export_values();

  py::enum_<hivm::FixpipeDMAMode>(m, "FixpipeDMAMode", py::module_local())
      .value("NZ2DN", hivm::FixpipeDMAMode::NZ2DN)
      .value("NZ2ND", hivm::FixpipeDMAMode::NZ2ND)
      .value("NZ2NZ", hivm::FixpipeDMAMode::NZ2NZ)
      .export_values();

  py::enum_<hivm::FixpipeDualDstMode>(m, "FixpipeDualDstMode",
                                      py::module_local())
      .value("NO_DUAL", hivm::FixpipeDualDstMode::NO_DUAL)
      .value("COLUMN_SPLIT", hivm::FixpipeDualDstMode::COLUMN_SPLIT)
      .value("ROW_SPLIT", hivm::FixpipeDualDstMode::ROW_SPLIT)
      .export_values();

  py::enum_<hivm::FixpipePreQuantMode>(m, "FixpipePreQuantMode",
                                       py::module_local())
      .value("NO_QUANT", hivm::FixpipePreQuantMode::NO_QUANT)
      .value("F322BF16", hivm::FixpipePreQuantMode::F322BF16)
      .value("F322F16", hivm::FixpipePreQuantMode::F322F16)
      .value("S322I8", hivm::FixpipePreQuantMode::S322I8)
      .export_values();

  py::enum_<hivm::FixpipePreReluMode>(m, "FixpipePreReluMode",
                                      py::module_local())
      .value("LEAKY_RELU", hivm::FixpipePreReluMode::LEAKY_RELU)
      .value("NO_RELU", hivm::FixpipePreReluMode::NO_RELU)
      .value("NORMAL_RELU", hivm::FixpipePreReluMode::NORMAL_RELU)
      .value("P_RELU", hivm::FixpipePreReluMode::P_RELU)
      .export_values();
  py::enum_<hivm::DataLayout>(m, "DataLayout", py::module_local())
      .value("nZ", hivm::DataLayout::nZ)
      .value("zN", hivm::DataLayout::zN)
      .export_values();

  m.def("load_dialects", [](MLIRContext &context) {
    // Allow unregistered dialects so we can parse HACC attributes without
    // registering the dialect
    context.allowUnregisteredDialects();
    DialectRegistry registry;
    registry.insert<annotation::AnnotationDialect, mlir::hivm::HIVMDialect,
                    scope::ScopeDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<AscendNPUIROpBuilder, TritonOpBuilder>(
      m, "ascendnpu_ir_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *, std::string>(), py::arg("context"),
           py::arg("target") = "")
      .def("get_core_type_attr",
           [](AscendNPUIROpBuilder &self,
              hivm::TCoreType core_type) -> Attribute {
             return self.getBuilder().getAttr<hivm::TCoreTypeAttr>(core_type);
           })
      .def("get_pipe_attr",
           [](AscendNPUIROpBuilder &self, hivm::PIPE pipe) -> Attribute {
             return self.getBuilder().getAttr<hivm::PipeAttr>(pipe);
           })
      .def("get_vf_mode_attr",
           [](AscendNPUIROpBuilder &self, hivm::VFMode mode) -> Attribute {
             return self.getBuilder().getAttr<hivm::VFModeAttr>(mode);
           })
      .def("get_t_core_type_attr_name",
           [](AscendNPUIROpBuilder &self) -> std::string {
             return hivm::TCoreTypeAttr::name.str();
           })
      .def("get_t_core_type_cube_attr",
           [](AscendNPUIROpBuilder &self) -> Attribute {
             return hivm::TCoreTypeAttr::get(self.getBuilder().getContext(),
                                             hivm::TCoreType::CUBE);
           })
      .def("get_t_core_type_vector_attr",
           [](AscendNPUIROpBuilder &self) -> Attribute {
             return hivm::TCoreTypeAttr::get(self.getBuilder().getContext(),
                                             hivm::TCoreType::VECTOR);
           })
      .def("parse_attr",
           [](TritonOpBuilder &self, std::string value) -> Attribute {
             auto *ctx = self.getBuilder().getContext();
             return mlir::parseAttribute(value, ctx);
           })
      .def("create_fixpipe",
           [](AscendNPUIROpBuilder &self, Value src, Value dst,
              hivm::FixpipeDMAMode dma_mode,
              hivm::FixpipeDualDstMode dual_dst_mode,
              hivm::FixpipePreQuantMode pre_quant_mode,
              hivm::FixpipePreReluMode pre_relu_mode) -> void {
             if (!dyn_cast<RankedTensorType>(src.getType())) {
               llvm_unreachable("src is not of RankedTensorType");
             }
             if (!dyn_cast<MemRefType>(dst.getType())) {
               llvm_unreachable("dst is not of MemRefType");
             }
             auto *ctx = self.getBuilder().getContext();
             auto dma_mode_attr =
                 mlir::hivm::FixpipeDMAModeAttr::get(ctx, dma_mode);
             auto dual_dst_mode_attr =
                 mlir::hivm::FixpipeDualDstModeAttr::get(ctx, dual_dst_mode);
             auto pre_quant_mode_attr =
                 mlir::hivm::FixpipePreQuantModeAttr::get(ctx, pre_quant_mode);
             auto pre_relu_mode_attr =
                 mlir::hivm::FixpipePreReluModeAttr::get(ctx, pre_relu_mode);
             auto channel_split = BoolAttr::get(ctx, false);
             auto op = self.create<hivm::FixpipeOp>(
                 mlir::TypeRange{}, src, dst, dma_mode_attr, dual_dst_mode_attr,
                 pre_quant_mode_attr, pre_relu_mode_attr, channel_split);
           })
      .def("create_bind_buffer",
           [](TritonOpBuilder &self, Value &src, Value &alloc) -> void {
             auto ctx = self.getBuilder().getContext();
             auto bind = StringAttr::get(ctx, "bind_buffer");
             self.create<annotation::MarkOp>(src, ValueRange{alloc},
                                             ArrayAttr::get(ctx, bind));
           })
      .def("create_debug_barrier",
           [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
              Attribute &attrVal) {
             auto annotationOp = self.create<annotation::MarkOp>(ptr);
             annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                   attrVal);
           })
      .def("create_custom_op",
           [](AscendNPUIROpBuilder &self, const std::string &name,
              const py::dict &attrs, const std::vector<Value> &ins,
              const std::vector<Value> &outs) -> std::vector<Value> {
             ValueRange inputs{ins};
             ValueRange outputs{outs};
             TypeRange res_types{outputs};
             auto op =
                 self.create<hivm::CustomOp>(res_types, name, inputs, outputs);
             for (auto &attr : attrs) {
               std::string attr_name = py::cast<std::string>(attr.first);
               Attribute attr_value = py::cast<Attribute>(attr.second);
               op->setAttr(attr_name, attr_value);
             }
             auto results = op->getResults();
             return std::vector<Value>(results.begin(), results.end());
           })
      .def("create_scope_op",
           [](AscendNPUIROpBuilder &self, py::dict &scopeAttrs,
              std::vector<Type> resultTypes) -> OpState {
             llvm::SmallVector<NamedAttribute> attrs;
             for (auto item : scopeAttrs) {
               std::string key = py::cast<std::string>(item.first);
               Attribute value = py::cast<Attribute>(item.second);
               attrs.push_back(
                   NamedAttribute(self.getBuilder().getStringAttr(key), value));
             }
             auto scopeOp = self.create<scope::ScopeOp>(TypeRange(resultTypes));
             scopeOp->setAttrs(attrs);
             return OpState(scopeOp);
           })
      .def("scope_return",
           [](AscendNPUIROpBuilder &self,
              std::vector<Value> operands) -> OpState {
             return self.create<scope::ReturnOp>(ValueRange(operands));
           })
      .def("sync_block_set",
           [](AscendNPUIROpBuilder &self, std::string &sender,
              std::string &receiver, Value id, hivm::PIPE senderPipe,
              hivm::PIPE receiverPipe) -> void {
             buildSyncBlockOp(self, "sync_block_set", sender, receiver, id,
                              senderPipe, receiverPipe);
           })
      .def("sync_block_wait",
           [](AscendNPUIROpBuilder &self, std::string &sender,
              std::string &receiver, Value id, hivm::PIPE senderPipe,
              hivm::PIPE receiverPipe) -> void {
             buildSyncBlockOp(self, "sync_block_wait", sender, receiver, id,
                              senderPipe, receiverPipe);
           })
      .def("get_target_attribute",
           [](AscendNPUIROpBuilder &self,
              hivm::AddressSpace &addressSpace) -> Attribute {
             return hivm::AddressSpaceAttr::get(self.getBuilder().getContext(),
                                                addressSpace);
           })
      .def("create_get_sub_vec_id",
           [](AscendNPUIROpBuilder &self) -> Value {
             auto subBlockIdxOp = self.create<hivm::GetSubBlockIdxOp>();
             auto moduleOp = subBlockIdxOp->getParentOfType<ModuleOp>();
             auto *ctx = self.getBuilder().getContext();
             // If user explicitly uses sub.block idx, add attribute to module.
             // NPU compiler will parse this attribute and disable auto tile and
             // bind subblock pass.
             moduleOp->setAttr("hivm.disable_auto_tile_and_bind_subblock",
                               mlir::UnitAttr::get(ctx));
             return subBlockIdxOp;
           })
      .def("sync_block_all",
           [](AscendNPUIROpBuilder &self, std::string &mode, int id) -> void {
             auto *ctx = self.getBuilder().getContext();
             auto [modeAttr, cubePipe, vectorPipe] =
                 GetSyncBlockModeAndPipes(ctx, mode);
             mlir::IndexType indexType = mlir::IndexType::get(ctx);
             mlir::IntegerAttr indexAttribute =
                 mlir::IntegerAttr::get(indexType, static_cast<int64_t>(id));
             self.create<hivm::SyncBlockOp>(
                 modeAttr, indexAttribute, mlir::Value{}, cubePipe, vectorPipe);
           })
      .def("is_910_95",
           [](AscendNPUIROpBuilder &self) -> bool { return self.is_910_95(); })
      .def("create_copy_buffer",
           [](AscendNPUIROpBuilder &self, Value src, Value dst) {
             self.create<hivm::CopyOp>(mlir::TypeRange{}, src, dst);
           })
      .def("create_copy_tensor",
           [](AscendNPUIROpBuilder &self, Value src, Value dst) {
             return self
                 .create<hivm::CopyOp>(mlir::TypeRange{dst.getType()}, src, dst)
                 .getResult(0);
           })
      .def("create_convert_layout",
           [](AscendNPUIROpBuilder &self, Value src, Type memrefType) -> Value {
             // src is a memref
             // the layout is incorrect (temporarily)
             auto *ctx = self.getBuilder().getContext();
             return self
                 .create<hivm::ConvertLayoutOp>(
                     memrefType, src,
                     hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND),
                     hivm::DataLayoutAttr::get(ctx, hivm::DataLayout::ND))
                 .getResult();
           });
}
