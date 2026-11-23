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
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/TritonToGraph/ProgramAxisDependenceAnalysis.h"
#include "ascend/include/TritonToGraph/ProgramGridTransform.h"
#include "ascend/include/TritonToGraph/ProgramMappingScalarSpecialization.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/IR/Instructions.h"
#include <Python.h>
#include <utility>
#include <vector>

using namespace mlir;
namespace py = pybind11;

struct AscendNPUIROpBuilder : public TritonOpBuilder {
  std::string target;
  std::string compile_mode;
  static constexpr char kTarget910_95[] = "Ascend910_95";
  static constexpr char kTarget950[] = "Ascend950";

  explicit AscendNPUIROpBuilder(MLIRContext *context, std::string target = "",
                                std::string compile_mode = "simd")
      : TritonOpBuilder(context), target(target), compile_mode(compile_mode) {}

  bool isSimtMode() const { return compile_mode == "simt"; }

  bool is_910_95() const {
    // TODO: Use enum instead of strings after enabling HACC in satandalone
    // build
    constexpr size_t kLen910 = sizeof(kTarget910_95) - 1;
    bool match_910 = target.size() >= kLen910 &&
                     target.compare(0, kLen910, kTarget910_95) == 0;

    constexpr size_t kLen950 = sizeof(kTarget950) - 1;
    bool match_950 =
        target.size() >= kLen950 && target.compare(0, kLen950, kTarget950) == 0;

    return match_910 || match_950;
  }
};

namespace {
MLIRContext *gDefaultAscendContext = nullptr;

MLIRContext *resolveContext(const py::object &contextObj) {
  if (!contextObj.is_none()) {
    return &py::cast<MLIRContext &>(contextObj);
  }
  if (gDefaultAscendContext) {
    return gDefaultAscendContext;
  }
  throw std::invalid_argument(
      "No default MLIR context. Pass context explicitly or call "
      "ascend_ir.load_dialects(context) first.");
}

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

// Extend triton.ir.context with a context manager protocol for ascend tests
// and examples, without modifying upstream python/src/ir.cc.
void installTritonContextManager() {
  static bool installed = false;
  if (installed) {
    return;
  }
  installed = true;

  py::module_ tritonIr = py::module_::import("triton._C.libtriton.ir");
  py::object ctxClass = tritonIr.attr("context");
  if (py::hasattr(ctxClass, "__enter__")) {
    return;
  }

  const char *patch = R"PY(
import triton._C.libtriton.ir as _triton_ir
if not hasattr(_triton_ir.context, '__enter__'):
    def _mlir_context_enter(self):
        return self
    def _mlir_context_exit(self, exc_type, exc_val, exc_tb):
        return False
    _triton_ir.context.__enter__ = _mlir_context_enter
    _triton_ir.context.__exit__ = _mlir_context_exit
)PY";
  PyGILState_STATE gil = PyGILState_Ensure();
  if (PyRun_SimpleString(patch) != 0) {
    PyGILState_Release(gil);
    throw std::runtime_error("failed to install MLIRContext context manager");
  }
  PyGILState_Release(gil);
}

} // namespace

void init_ascend_ir(py::module &&m) {
  auto affineExprClass =
      py::class_<AffineExpr>(m, "affine_expr", py::module_local());
  affineExprClass
      .def("__str__",
           [](AffineExpr self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("__repr__",
           [](AffineExpr self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return "<affine_expr " + os.str() + ">";
           })
      .def("is_symbolic_or_constant", &AffineExpr::isSymbolicOrConstant)
      .def("is_pure_affine", &AffineExpr::isPureAffine)
      .def("is_function_of_dim", &AffineExpr::isFunctionOfDim)
      .def("compose",
           [](AffineExpr self, AffineMap map) { return self.compose(map); })
      .def("get_largest_known_divisor", &AffineExpr::getLargestKnownDivisor)
      .def("floordiv", [](AffineExpr self,
                          AffineExpr other) { return self.floorDiv(other); })
      .def("ceildiv", [](AffineExpr self,
                         AffineExpr other) { return self.ceilDiv(other); })
      .def("mod",
           [](AffineExpr self, AffineExpr other) { return self % other; })
      .def("__hash__",
           [](AffineExpr self) {
             return py::int_(static_cast<uint64_t>(mlir::hash_value(self)));
           })
      .def("__eq__", [](AffineExpr lhs, AffineExpr rhs) { return lhs == rhs; })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self % py::self);
  affineExprClass
      .def_static(
          "get_constant",
          [](int64_t val, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return getAffineConstantExpr(val, context);
          },
          py::arg("value"), py::arg("context") = py::none())
      .def_static(
          "get_dim",
          [](uint32_t pos, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return getAffineDimExpr(pos, context);
          },
          py::arg("pos"), py::arg("context") = py::none())
      .def_static(
          "get_symbol",
          [](uint32_t pos, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return getAffineSymbolExpr(pos, context);
          },
          py::arg("pos"), py::arg("context") = py::none());

  affineExprClass.attr("__doc__") =
      "An MLIR affine expression representing a linear combination of "
      "dimensions and symbols with a constant offset.\n\n"
      "Affine expressions model loop bounds, array indices, and memory "
      "access patterns in the Ascend NPU compiler. They support addition, "
      "subtraction, multiplication by constants, floor division, ceiling "
      "division, and modulo operations.\n\n"
      "Create via static methods: get_constant(value), get_dim(pos), "
      "get_symbol(pos).";

  py::class_<AffineConstantExpr, AffineExpr>(m, "affine_constant_expr",
                                             py::module_local())
      .def("get_value", &AffineConstantExpr::getValue)
      .attr("__doc__") =
      "An affine expression that is a constant integer value. "
      "This is the simplest form of an affine expression, "
      "representing just a numeric constant.";
  py::class_<AffineDimExpr, AffineExpr>(m, "affine_dim_expr",
                                        py::module_local())
      .def("get_position", &AffineDimExpr::getPosition)
      .attr("__doc__") =
      "An affine expression representing a single dimension variable. "
      "Dimensions typically correspond to loop induction variables.";
  py::class_<AffineSymbolExpr, AffineExpr>(m, "affine_symbol_expr",
                                           py::module_local())
      .def("get_position", &AffineSymbolExpr::getPosition)
      .attr("__doc__") =
      "An affine expression representing a single symbol variable. "
      "Symbols represent unknown but constant values (e.g., tile sizes).";
  py::class_<AffineBinaryOpExpr, AffineExpr>(m, "affine_binary_op_expr",
                                             py::module_local())
      .def("get_lhs", &AffineBinaryOpExpr::getLHS)
      .def("get_rhs", &AffineBinaryOpExpr::getRHS)
      .attr("__doc__") =
      "An affine expression composed of two sub-expressions combined by "
      "an operator (add, sub, mul, mod, floordiv, ceildiv).";

  auto affineMapClass =
      py::class_<AffineMap>(m, "affine_map", py::module_local());
  affineMapClass
      .def("__str__",
           [](AffineMap &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("__repr__",
           [](AffineMap &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return "<affine_map " + os.str() + ">";
           })
      .def("is_identity", &AffineMap::isIdentity)
      .def("is_permutation", &AffineMap::isPermutation)
      .def("get_num_dims", &AffineMap::getNumDims)
      .def("get_num_symbols", &AffineMap::getNumSymbols)
      .def("get_num_results", &AffineMap::getNumResults)
      .def("is_empty", &AffineMap::isEmpty)
      .def("is_single_constant", &AffineMap::isSingleConstant)
      .def("is_constant", &AffineMap::isConstant)
      .def("get_constant_result",
           [](AffineMap &self) -> int64_t {
             if (!self.isSingleConstant()) {
               throw std::runtime_error(
                   "affine map is not a single constant map");
             }
             return self.getSingleConstantResult();
           })
      .def("get_result",
           [](AffineMap &self, uint32_t pos) {
             if (pos >= self.getNumResults()) {
               throw py::index_error("result index out of range");
             }
             return self.getResult(pos);
           })
      .def("get_sub_map",
           [](AffineMap &self, const std::vector<unsigned> &resultPos) {
             return self.getSubMap(resultPos);
           })
      .def("replace",
           [](AffineMap &self, AffineExpr expr, AffineExpr replacement,
              uint32_t numResultDims, uint32_t numResultSymbols) {
             return self.replace(expr, replacement, numResultDims,
                                 numResultSymbols);
           })
      .def("compose",
           [](AffineMap &self, AffineMap map) { return self.compose(map); })
      .def("get_results",
           [](AffineMap &self) -> std::vector<AffineExpr> {
             auto results = self.getResults();
             return std::vector<AffineExpr>(results.begin(), results.end());
           })
      .def("__hash__",
           [](AffineMap &self) {
             return py::int_(static_cast<uint64_t>(mlir::hash_value(self)));
           })
      .def("__eq__", [](AffineMap &lhs, AffineMap &rhs) { return lhs == rhs; })
      .def("inverse_permutation",
           [](AffineMap &self) -> py::object {
             // Validate it's a permutation first
             if (!self.isPermutation()) {
               throw py::value_error(
                   "AffineMap must be a valid permutation to compute inverse");
             }

             // Returns AffineMap directly, not a pointer
             AffineMap inverse = mlir::inversePermutation(self);

             // Check if result is valid (null AffineMap)
             if (!inverse) {
               throw py::value_error("Failed to compute inverse permutation");
             }

             return py::cast(inverse);
           })
      .def("to_dict", [](AffineMap &self) -> py::dict {
        py::list results;
        for (AffineExpr result : self.getResults()) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(result)) {
            results.append(dimExpr.getPosition());
          } else {
            std::string exprStr;
            llvm::raw_string_ostream os(exprStr);
            result.print(os);
            results.append(py::str(exprStr));
          }
        }

        py::dict ret;
        ret["num_dims"] = self.getNumDims();
        ret["num_symbols"] = self.getNumSymbols();
        ret["results"] = std::move(results);
        return ret;
      });
  affineMapClass
      .def_static(
          "get",
          [](int64_t numDims, int64_t numSymbols, const py::iterable &resultsIn,
             py::object contextObj) -> AffineMap {
            MLIRContext *context = nullptr;
            if (numDims < 0 || numSymbols < 0) {
              throw std::invalid_argument(
                  "num_dims and num_symbols must be non-negative");
            }
            llvm::SmallVector<AffineExpr> results;
            for (const auto &item : resultsIn) {
              if (py::isinstance<AffineExpr>(item)) {
                auto expr = py::cast<AffineExpr>(item);
                if (!context) {
                  context = expr.getContext();
                }
                results.push_back(expr);
                continue;
              }
              if (py::isinstance<py::int_>(item)) {
                if (!context) {
                  context = resolveContext(contextObj);
                }
                int64_t pos = py::cast<int64_t>(item);
                if (pos < 0 || pos >= numDims) {
                  throw std::invalid_argument(
                      "result dim index is out of range for num_dims");
                }
                results.push_back(getAffineDimExpr(pos, context));
                continue;
              }
              throw std::invalid_argument(
                  "results must contain affine_expr or int dim indices");
            }
            if (!context) {
              context = resolveContext(contextObj);
            }
            return AffineMap::get(numDims, numSymbols, results, context);
          },
          py::arg("num_dims"), py::arg("num_symbols"), py::arg("result_dims"),
          py::arg("context") = py::none())
      .def_static(
          "get_identity",
          [](int64_t numDims, py::object contextObj) -> AffineMap {
            auto *context = resolveContext(contextObj);
            if (numDims < 0) {
              throw std::invalid_argument("num_dims must be non-negative");
            }
            return AffineMap::getMultiDimIdentityMap(numDims, context);
          },
          py::arg("num_dims"), py::arg("context") = py::none())
      .def_static(
          "get_minor_identity",
          [](int64_t dims, int64_t results, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            if (dims < 0 || results < 0) {
              throw std::invalid_argument("dims/results must be non-negative");
            }
            return AffineMap::getMinorIdentityMap(dims, results, context);
          },
          py::arg("dims"), py::arg("results"), py::arg("context") = py::none())
      .def_static(
          "get_empty",
          [](py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return AffineMap::get(0, 0, {}, context);
          },
          py::arg("context") = py::none())
      .def_static(
          "get_permutation",
          [](const std::vector<unsigned> &permutation, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return AffineMap::getPermutationMap(permutation, context);
          },
          py::arg("permutation"), py::arg("context") = py::none())
      .def_static(
          "get_constant",
          [](int64_t value, py::object contextObj) {
            auto *context = resolveContext(contextObj);
            return AffineMap::getConstantMap(value, context);
          },
          py::arg("value"), py::arg("context") = py::none());

  affineMapClass.attr("__doc__") =
      "An MLIR affine map representing a mapping from a set of "
      "dimensions and symbols to a list of affine expressions.\n\n"
      "Affine maps encode transformations like indexing, layout "
      "permutations, and memory access patterns. They are the "
      "fundamental abstraction for describing data movement and "
      "computation placement in the Ascend NPU compiler.";

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

  py::enum_<hivm::SyncEventSlotMacroSync>(m, "SYNC_HINT", py::module_local())
      .value("wait", hivm::SyncEventSlotMacroSync::wait)
      .value("set", hivm::SyncEventSlotMacroSync::set)
      .value("internal", hivm::SyncEventSlotMacroSync::internal)
      .export_values();

  py::enum_<hivm::EVENT>(m, "EVENT", py::module_local())
      .value("EVENT_ID0", hivm::EVENT::EVENT_ID0)
      .value("EVENT_ID1", hivm::EVENT::EVENT_ID1)
      .value("EVENT_ID2", hivm::EVENT::EVENT_ID2)
      .value("EVENT_ID3", hivm::EVENT::EVENT_ID3)
      .value("EVENT_ID4", hivm::EVENT::EVENT_ID4)
      .value("EVENT_ID5", hivm::EVENT::EVENT_ID5)
      .value("EVENT_ID6", hivm::EVENT::EVENT_ID6)
      .value("EVENT_ID7", hivm::EVENT::EVENT_ID7)
      .export_values();

  py::enum_<hivm::VFMode>(m, "MODE", py::module_local())
      .value("SIMD", hivm::VFMode::SIMD)
      .value("SIMT", hivm::VFMode::SIMT)
      .value("MIX", hivm::VFMode::MIX)
      .export_values();

  py::enum_<hivm::IteratorType>(m, "IteratorType", py::module_local())
      .value("Parallel", hivm::IteratorType::kParallel)
      .value("Broadcast", hivm::IteratorType::kBroadcast)
      .value("Transpose", hivm::IteratorType::kTranspose)
      .value("Reduction", hivm::IteratorType::kReduction)
      .value("Interleave", hivm::IteratorType::kInterleave)
      .value("Deinterleave", hivm::IteratorType::kDeinterleave)
      .value("Inverse", hivm::IteratorType::kInverse)
      .value("Pad", hivm::IteratorType::kPad)
      .value("Concat", hivm::IteratorType::kConcat)
      .value("Gather", hivm::IteratorType::kGather)
      .value("Cumulative", hivm::IteratorType::kCumulative)
      .value("Opaque", hivm::IteratorType::kOpaque)
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
    gDefaultAscendContext = &context;
    DialectRegistry registry;
    registry.insert<annotation::AnnotationDialect, mlir::hivm::HIVMDialect,
                    scope::ScopeDialect>();
    // Standalone callers also use this loader before running the Ascend
    // lowering pipeline.
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
  m.def("get_int_attr", [](OpState &op, std::string &name) -> py::object {
    auto ret = op->getAttrOfType<IntegerAttr>(name);
    if (!ret) {
      return py::none();
    }
    return py::cast(ret.getInt());
  });
  m.def("get_program_grid_transforms", [](OpState &op) -> py::object {
    Attribute attribute =
        op->getAttr(mlir::triton::cfg::kProgramGridTransformsAttr);
    if (!attribute)
      return py::none();
    FailureOr<mlir::triton::cfg::ProgramGridTransformContract> contract =
        mlir::triton::cfg::parseProgramGridTransformContract(attribute);
    if (failed(contract))
      throw std::runtime_error("invalid hacc.program_grid_transforms contract");

    py::dict result;
    result["version"] = contract->version;
    result["extent_source"] = "runtime_original_grid";
    py::list hiddenExtentAxes;
    hiddenExtentAxes.append(0);
    hiddenExtentAxes.append(1);
    result["hidden_extent_axes"] = std::move(hiddenExtentAxes);
    py::list hiddenArgumentOrder;
    hiddenArgumentOrder.append("originalGridX");
    hiddenArgumentOrder.append("originalGridY");
    result["hidden_argument_order"] = std::move(hiddenArgumentOrder);
    py::list hiddenArgumentTypes;
    hiddenArgumentTypes.append("i32");
    hiddenArgumentTypes.append("i32");
    result["hidden_argument_types"] = std::move(hiddenArgumentTypes);
    py::list transforms;
    for (const mlir::triton::cfg::ProgramGridTransform &transform :
         contract->transforms) {
      py::dict item;
      item["order"] = transform.order;
      item["kind"] = "ceil_div";
      item["axis"] = transform.axis;
      item["factor"] = transform.factor;
      item["persistent_coverage"] = transform.persistentCoverage;
      item["grid_stride_abi_verified"] = transform.gridStrideAbiVerified;
      transforms.append(std::move(item));
    }
    result["transforms"] = std::move(transforms);
    return std::move(result);
  });
  m.def("get_program_mapping_scalar_specialization",
        [](OpState &op) -> py::object {
          Attribute attribute = op->getAttr(
              mlir::triton::cfg::kProgramMappingScalarSpecializationAttr);
          if (!attribute)
            return py::none();
          FailureOr<mlir::triton::cfg::ProgramMappingScalarSpecialization>
              specialization =
                  mlir::triton::cfg::parseProgramMappingScalarSpecialization(
                      attribute);
          if (failed(specialization))
            throw std::runtime_error(
                "invalid hacc.program_mapping_scalar_specialization contract");

          py::dict result;
          result["version"] = specialization->version;
          py::list arguments;
          for (const mlir::triton::cfg::ProgramMappingScalarArgument &argument :
               specialization->arguments) {
            py::dict item;
            item["index"] = argument.index;
            if (specialization->version ==
                mlir::triton::cfg::kProgramMappingScalarSpecializationVersion)
              item["name"] = argument.name;
            item["value"] = argument.value;
            arguments.append(std::move(item));
          }
          result["arguments"] = std::move(arguments);
          return std::move(result);
        });
  m.def(
      "set_program_mapping_scalar_specialization",
      [](OpState &op, int64_t version, const py::iterable &arguments) {
        auto module = dyn_cast<ModuleOp>(op.getOperation());
        if (!module)
          throw std::invalid_argument("set_program_mapping_scalar_"
                                      "specialization expects an MLIR module");
        mlir::triton::cfg::ProgramMappingScalarSpecialization specialization;
        specialization.version = version;
        const bool isLegacy =
            version ==
            mlir::triton::cfg::kProgramMappingScalarSpecializationLegacyVersion;
        const bool isNamed =
            version ==
            mlir::triton::cfg::kProgramMappingScalarSpecializationVersion;
        if (!isLegacy && !isNamed)
          throw std::invalid_argument(
              "unsupported program_mapping_scalar_specialization version");
        for (const py::handle &rawArgument : arguments) {
          if (!py::isinstance<py::sequence>(rawArgument) ||
              py::isinstance<py::str>(rawArgument))
            throw std::invalid_argument("program_mapping_scalar_specialization "
                                        "argument must be a sequence");
          py::sequence argument =
              py::reinterpret_borrow<py::sequence>(rawArgument);
          const size_t expectedSize = isLegacy ? 2 : 3;
          if (argument.size() != expectedSize)
            throw std::invalid_argument("program_mapping_scalar_specialization "
                                        "argument has invalid arity");
          uint32_t index = argument[0].cast<uint32_t>();
          if (isLegacy) {
            specialization.arguments.push_back(
                {index, "", argument[1].cast<int64_t>()});
          } else {
            specialization.arguments.push_back({index,
                                                argument[1].cast<std::string>(),
                                                argument[2].cast<int64_t>()});
          }
        }
        if (failed(mlir::triton::cfg::setProgramMappingScalarSpecialization(
                module, specialization)))
          throw std::runtime_error(
              "invalid hacc.program_mapping_scalar_specialization contract");
      },
      py::arg("module"), py::arg("version"), py::arg("arguments"));
  m.def("clear_program_mapping_scalar_specialization", [](OpState &op) {
    auto module = dyn_cast<ModuleOp>(op.getOperation());
    if (!module)
      throw std::invalid_argument(
          "clear_program_mapping_scalar_specialization expects an MLIR module");
    mlir::triton::cfg::clearProgramMappingScalarSpecialization(module);
  });
  m.def(
      "analyze_program_axis_dependence",
      [](OpState &op) -> py::list {
        auto function = llvm::dyn_cast<mlir::triton::FuncOp>(op.getOperation());
        if (!function)
          throw std::invalid_argument(
              "analyze_program_axis_dependence expects a tt.func operation");

        mlir::triton::cfg::ProgramAxisDependenceAnalysis analysis(function);
        py::list axes;
        for (int32_t axis = 0; axis < 3; ++axis) {
          const mlir::triton::cfg::ProgramAxisDependence &facts =
              analysis.get(axis);
          py::dict result;
          result["axis"] = facts.axis;
          result["program_id_count"] = facts.programIds.size();
          result["escapes"] = facts.escapes;
          result["has_side_effects"] = facts.hasSideEffects;
          result["has_unsupported_side_effects"] =
              facts.hasUnsupportedSideEffects;
          result["reads_num_programs"] = facts.readsNumPrograms;
          result["is_independent_axis_transform_candidate"] =
              facts.isIndependentAxisTransformCandidate();
          result["is_program_mapping_transform_candidate"] =
              facts.isProgramMappingTransformCandidate();

          py::list closure;
          for (Operation *operation : facts.dependenceClosure)
            closure.append(operation->getName().getStringRef().str());
          result["dependence_closure"] = std::move(closure);

          py::list reductions;
          for (int32_t reductionAxis : facts.reductionAxes)
            reductions.append(reductionAxis);
          result["reduction_axes"] = std::move(reductions);

          py::list stores;
          for (const mlir::triton::cfg::StoreAddressDependence &store :
               facts.stores) {
            py::dict summary;
            summary["pointer_depends_on_axis"] = store.pointerDependsOnAxis;
            switch (store.independence) {
            case mlir::triton::cfg::StoreAddressIndependence::
                NotProgramDependent:
              summary["address_independence"] = "not_program_dependent";
              break;
            case mlir::triton::cfg::StoreAddressIndependence::ProvenDisjoint:
              summary["address_independence"] = "proven_disjoint";
              break;
            case mlir::triton::cfg::StoreAddressIndependence::
                CanonicalDynamicStride:
              summary["address_independence"] = "canonical_dynamic_stride";
              break;
            case mlir::triton::cfg::StoreAddressIndependence::Unknown:
              summary["address_independence"] = "unknown";
              break;
            }
            stores.append(std::move(summary));
          }
          result["stores"] = std::move(stores);
          axes.append(std::move(result));
        }
        return axes;
      },
      py::arg("function"),
      "Return conservative per-axis ProgramAxisDependenceAnalysis facts for a "
      "tt.func.");
  m.def("remove_attr",
        [](OpState &op, std::string &name) -> void { op->removeAttr(name); });
  py::class_<StringAttr, Attribute>(m, "str_attr", py::module_local());
  py::class_<ArrayAttr, Attribute>(m, "array_attr", py::module_local());

  py::class_<AscendNPUIROpBuilder, TritonOpBuilder>(
      m, "ascendnpu_ir_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *, std::string, std::string>(),
           py::arg("context"), py::arg("target") = "",
           py::arg("compile_mode") = "simd",
           "Create a TritonOpBuilder with optional compile_mode (simt or simd, "
           "default: simd)")
      .def("is_simt_mode", &AscendNPUIROpBuilder::isSimtMode,
           "Check if the compile mode is simt")
      .def("create_auto_overflow_assert",
           [](AscendNPUIROpBuilder &self, Value &condition,
              const std::string &message) {
             auto messageAttr = self.getBuilder().getStringAttr(message);
             auto assertOp =
                 self.create<mlir::triton::AssertOp>(condition, messageAttr);
             assertOp->setAttr("tt.auto_overflow_assert",
                               self.getBuilder().getUnitAttr());
           })
      .def("get_i64_array_attr",
           [](AscendNPUIROpBuilder &self, const std::vector<int64_t> &array) {
             return self.getBuilder().getI64ArrayAttr(array);
           })
      .def("get_buffer_ty",
           [](AscendNPUIROpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const Attribute &memorySpace) -> Type {
             return MemRefType::get(shape, elementType,
                                    MemRefLayoutAttrInterface{}, memorySpace);
           })
      .def("get_buffer_ty_with_strides",
           [](AscendNPUIROpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, const std::vector<int64_t> &strides,
              const Attribute &memorySpace) -> Type {
             // create a layout with strides, using dynamic offset
             auto layout = StridedLayoutAttr::get(
                 self.getBuilder().getContext(), ShapedType::kDynamic, strides);
             return MemRefType::get(shape, elementType, layout, memorySpace);
           })
      .def("create_extract_scalar",
           [](AscendNPUIROpBuilder &self, Value &src,
              std::vector<Value> &indices) -> Value {
             llvm::SmallVector<Value> arg_indices;
             for (const auto &i : indices) {
               auto iTy = i.getType();
               if (!iTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), i);
                 arg_indices.push_back(v);
               } else {
                 arg_indices.push_back(i);
               }
             }
             auto ret = self.create<tensor::ExtractOp>(src, arg_indices);
             return ret;
           })
      .def("create_extract_slice",
           [](AscendNPUIROpBuilder &self, Value &ful,
              std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
              std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             llvm::SmallVector<int64_t> staticOffsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
               staticOffsets.push_back(ShapedType::kDynamic);
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> staticSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               // auto v = self.create<arith::ConstantIndexOp>(s);
               // sizes.push_back(v);
               staticSizes.push_back(s);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());

             return self.create<tensor::ExtractSliceOp>(
                 retTy, ful, offsets, sizes, strides, staticOffsets,
                 staticSizes, staticStrides);
           })
      .def("create_insert_slice",
           [](AscendNPUIROpBuilder &self, Value &ful, Value &sub,
              std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
              std::vector<int> &strd_vec) -> Value {
             llvm::SmallVector<Value> offsets;
             llvm::SmallVector<int64_t> staticOffsets;
             for (const auto &o : offs_vec) {
               auto oTy = o.getType();
               if (!oTy.isIndex()) {
                 auto v = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), o);
                 offsets.push_back(v);
               } else {
                 offsets.push_back(o);
               }
               staticOffsets.push_back(ShapedType::kDynamic);
             }
             llvm::SmallVector<Value> sizes;
             llvm::SmallVector<int64_t> staticSizes;
             llvm::SmallVector<int64_t> retSizes;
             for (const auto &s : sizs_vec) {
               // auto v = self.create<arith::ConstantIndexOp>(s);
               // sizes.push_back(v);
               staticSizes.push_back(s);
               retSizes.push_back(s);
             }
             llvm::SmallVector<Value> strides;
             llvm::SmallVector<int64_t> staticStrides;
             for (const auto &s : strd_vec) {
               auto v = self.create<arith::ConstantIndexOp>(s);
               strides.push_back(v);
               staticStrides.push_back(ShapedType::kDynamic);
             }
             auto retTy = RankedTensorType::get(
                 retSizes,
                 cast<RankedTensorType>(ful.getType()).getElementType());
             auto ret = self.create<tensor::InsertSliceOp>(
                 sub, ful, offsets, sizes, strides, staticOffsets, staticSizes,
                 staticStrides);
             return ret;
           })
      .def("create_custom_op_for_inter_core_sync",
           [](AscendNPUIROpBuilder &self, std::string &op_name,
              std::string &mode_or_sender, int id) -> void {
             auto args = self.getBuilder().getArrayAttr(
                 {self.getBuilder().getStringAttr(mode_or_sender),
                  self.getBuilder().getI32IntegerAttr(id)});
             self.create<triton::ascend::CustomOp>(op_name, args, ValueRange());
           })
      .def("create_index_select_simd",
           [](AscendNPUIROpBuilder &self, Value &src, Value &index, int32_t dim,
              std::vector<Value> &srcShape, std::vector<Value> &srcOffset,
              std::vector<int32_t> &readShape,
              std::vector<int32_t> &returnShape) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             // Get element type from source pointer
             Type elemType;
             if (auto ptrTy = dyn_cast<triton::PointerType>(src.getType())) {
               elemType = ptrTy.getPointeeType();
             } else {
               llvm::report_fatal_error(
                   "index_select_simd: src must be pointer type");
             }

             // Create return tensor type
             llvm::SmallVector<int64_t> retShape;
             for (const auto &s : returnShape) {
               retShape.push_back(s);
             }
             auto retTensorType = RankedTensorType::get(retShape, elemType);

             // Convert srcShape and srcOffset values to index type if needed
             llvm::SmallVector<Value> srcShapeIndex;
             for (auto val : srcShape) {
               if (!val.getType().isIndex()) {
                 val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                       val);
               }
               srcShapeIndex.push_back(val);
             }

             llvm::SmallVector<Value> srcOffsetIndex;
             for (auto val : srcOffset) {
               if (!val.getType().isIndex()) {
                 val = self.create<arith::IndexCastOp>(builder.getIndexType(),
                                                       val);
               }
               srcOffsetIndex.push_back(val);
             }

             // Create attributes
             auto dimAttr = builder.getI32IntegerAttr(dim);
             auto readShapeAttr = builder.getDenseI32ArrayAttr(readShape);

             // Create the IndexSelectSimdOp
             // Parameter order must match TritonOps.td definition:
             // src, index, dim, src_shape, src_offset, read_shape
             auto indexSelectSimdOp =
                 builder.create<triton::ascend::IndexSelectSimdOp>(
                     loc,
                     retTensorType,  // result type
                     src,            // src pointer
                     index,          // index tensor
                     dimAttr,        // dim attribute
                     srcShapeIndex,  // src_shape (variadic, index type)
                     srcOffsetIndex, // src_offset (variadic, index type)
                     readShapeAttr   // read_shape attribute
                 );

             return indexSelectSimdOp.getResult();
           })
      .def("create_index_put",
           [](AscendNPUIROpBuilder &self, Value &ptr, Value &index,
              Value &value, const int32_t dim, const int64_t indexBoundary,
              std::vector<Value> &endOffset, std::vector<Value> &startOffset,
              std::vector<Value> &dstStride) -> void {
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(dimI32Ty, dim);
             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val =
                 self.create<arith::ConstantIntOp>(BoundI64Ty, indexBoundary);

             self.create<triton::ascend::IndexPutOp>(ptr, index, value, dim_val,
                                                     bound_val, endOffset,
                                                     startOffset, dstStride);
           })
      .def("create_gather_out_to_ub",
           [](AscendNPUIROpBuilder &self, Value &src, Value &index,
              const int64_t indexBoundary, const int32_t dim,
              std::vector<Value> &srcStride, std::vector<Value> &endOffset,
              std::vector<Value> &startOffset,
              std::optional<Value> &other) -> Value {
             auto elemTy =
                 cast<triton::PointerType>(src.getType()).getPointeeType();
             auto idxTy = cast<RankedTensorType>(index.getType());
             auto idxShape = idxTy.getShape();
             std::vector<int64_t> retShape(idxShape.begin(), idxShape.end());
             auto resType = RankedTensorType::get(retShape, elemTy);

             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val =
                 self.create<arith::ConstantIntOp>(BoundI64Ty, indexBoundary);
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(dimI32Ty, dim);
             return self.create<triton::ascend::GatherOutToUbOp>(
                 resType, src, index, bound_val, dim_val, srcStride, endOffset,
                 startOffset, other.value_or(Value()));
           })
      .def("create_scatter_ub_to_out",
           [](AscendNPUIROpBuilder &self, Value &ptr, Value &value,
              Value &index, const int64_t indexBoundary, const int32_t dim,
              std::vector<Value> &dstStride, std::vector<Value> &endOffset,
              std::vector<Value> &startOffset) -> void {
             auto idxTy = cast<RankedTensorType>(index.getType());

             // indexBoundary need to be i64 type
             auto BoundI64Ty = self.getBuilder().getI64Type();
             auto bound_val =
                 self.create<arith::ConstantIntOp>(BoundI64Ty, indexBoundary);
             // dim need to be i32 type
             auto dimI32Ty = self.getBuilder().getI32Type();
             auto dim_val = self.create<arith::ConstantIntOp>(dimI32Ty, dim);

             self.create<triton::ascend::ScatterUbToOutOp>(
                 ptr, value, index, bound_val, dim_val, dstStride, endOffset,
                 startOffset);
           })
      // conv1d operation
      .def(
          "create_conv1d",
          [](AscendNPUIROpBuilder &self, Value input, Value weight,
             py::object bias, int64_t stride, int64_t padding_size,
             int64_t dilation, int64_t groups, Type output_type) -> Value {
            Value biasValue;
            if (!bias.is_none()) {
              biasValue = bias.cast<Value>();
            } else {
              biasValue = Value();
            }
            auto &builder = self.getBuilder();
            auto strideAttr = builder.getI64IntegerAttr(stride);
            auto paddingSizeAttr = builder.getI64IntegerAttr(padding_size);
            auto dilationAttr = builder.getI64IntegerAttr(dilation);
            auto groupsAttr = builder.getI64IntegerAttr(groups);
            auto op = self.create<triton::ascend::Conv1dOp>(
                output_type, input, weight, biasValue, strideAttr,
                paddingSizeAttr, dilationAttr, groupsAttr);
            return op.getResult();
          },
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding_size"), py::arg("dilation"),
          py::arg("groups"), py::arg("output_type"))
      // dot: D = A * B with per-operand fractal (zN) flags; result type
      // inferred from A, B and the flags.
      .def(
          "create_dot",
          [](AscendNPUIROpBuilder &self, Value &a, Value &b, bool fractalA,
             bool fractalB, bool fractalC) -> Value {
            auto &builder = self.getBuilder();
            auto op = self.create<triton::ascend::DotOp>(
                a, b, builder.getBoolAttr(fractalA),
                builder.getBoolAttr(fractalB), builder.getBoolAttr(fractalC));
            return op.getResult();
          },
          py::arg("a"), py::arg("b"), py::arg("fractal_a"),
          py::arg("fractal_b"), py::arg("fractal_c"))
      // conv2d operation
      .def(
          "create_conv2d",
          [](AscendNPUIROpBuilder &self, Value input, Value weight,
             py::object bias, py::object stride, py::object padding,
             py::object dilation, int64_t groups, Type output_type) -> Value {
            Value biasValue;
            if (!bias.is_none()) {
              biasValue = bias.cast<Value>();
            } else {
              biasValue = Value();
            }
            auto &builder = self.getBuilder();
            // Keep ints as IntegerAttr and sequences as DenseI32ArrayAttr.
            // padding = [pad_top, pad_bottom, pad_left, pad_right] or a
            // 2-element [pad_h, pad_w].
            auto buildConvParamAttr = [&](py::object param,
                                          const std::string &name,
                                          bool allowFour) -> Attribute {
              if (py::isinstance<py::int_>(param)) {
                return builder.getI64IntegerAttr(param.cast<int64_t>());
              }
              py::sequence seq = param.cast<py::sequence>();
              size_t len = py::len(seq);
              SmallVector<int32_t> values;
              for (auto item : seq)
                values.push_back(item.cast<int32_t>());
              return builder.getDenseI32ArrayAttr(values);
            };
            auto strideAttr = buildConvParamAttr(stride, "stride", false);
            auto paddingAttr = buildConvParamAttr(padding, "padding", true);
            auto dilationAttr = buildConvParamAttr(dilation, "dilation", false);
            auto groupsAttr = builder.getI64IntegerAttr(groups);
            auto op = self.create<triton::ascend::Conv2dOp>(
                output_type, input, weight, biasValue, strideAttr, paddingAttr,
                dilationAttr, groupsAttr);
            return op.getResult();
          },
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"),
          py::arg("groups"), py::arg("output_type"))
      // Add sort
      .def("create_sort",
           [](AscendNPUIROpBuilder &self, Value src, int64_t dim,
              bool descending) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             auto dimAttr = builder.getI64IntegerAttr(dim);
             auto descendingAttr = builder.getBoolAttr(descending);

             auto op = builder.create<triton::ascend::SortOp>(loc, src, dimAttr,
                                                              descendingAttr);

             return op->getResult(0);
           })
      // Add flip
      .def("create_flip",
           [](AscendNPUIROpBuilder &self, Value src, int64_t dim) -> Value {
             auto &builder = self.getBuilder();
             auto loc = self.getLastLoc();

             auto dimAttr = builder.getI64IntegerAttr(dim);

             auto op =
                 builder.create<triton::ascend::FlipOp>(loc, src, dimAttr);

             return op->getResult(0);
           })
      .def("create_tanh",
           [](AscendNPUIROpBuilder &self, Value &val) -> Value {
             return self.create<math::TanhOp>(val);
           })
      // Add an annotation
      .def("create_annotation",
           [](AscendNPUIROpBuilder &self, Value &ptr,
              const std::string &attrKey, Attribute &attrVal) {
             auto annotationOp = self.create<triton::ascend::AnnotationOp>(ptr);
             annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                   attrVal);
           })
      .def("get_int_attr",
           [](AscendNPUIROpBuilder &self, int64_t value) -> Attribute {
             return IntegerAttr::get(self.getBuilder().getI64Type(), value);
           })
      .def("get_type_array_attr",
           [](AscendNPUIROpBuilder &self,
              const std::vector<Type> &array) -> Attribute {
             return self.getBuilder().getTypeArrayAttr(array);
           })
      .def("get_core_type_attr",
           [](AscendNPUIROpBuilder &self,
              hivm::TCoreType core_type) -> Attribute {
             return self.getBuilder().getAttr<hivm::TCoreTypeAttr>(core_type);
           })
      .def("get_pipe_attr",
           [](AscendNPUIROpBuilder &self, hivm::PIPE pipe) -> Attribute {
             return self.getBuilder().getAttr<hivm::PipeAttr>(pipe);
           })
      .def("get_event_attr",
           [](AscendNPUIROpBuilder &self, hivm::EVENT event) -> Attribute {
             return hivm::EventAttr::get(self.getBuilder().getContext(), event);
           })
      .def(
          "get_sync_event_slot_attr",
          [](AscendNPUIROpBuilder &self, py::object setPipe,
             py::object waitPipe, hivm::SyncEventSlotMacroSync macroSync,
             py::object event) -> Attribute {
            auto *ctx = self.getBuilder().getContext();
            hivm::PipeAttr setPipeAttr;
            hivm::PipeAttr waitPipeAttr;
            hivm::EventAttr eventAttr;
            if (!setPipe.is_none())
              setPipeAttr =
                  hivm::PipeAttr::get(ctx, py::cast<hivm::PIPE>(setPipe));
            if (!waitPipe.is_none())
              waitPipeAttr =
                  hivm::PipeAttr::get(ctx, py::cast<hivm::PIPE>(waitPipe));
            if (!event.is_none())
              eventAttr =
                  hivm::EventAttr::get(ctx, py::cast<hivm::EVENT>(event));
            return hivm::SyncEventSlotAttr::get(ctx, setPipeAttr, waitPipeAttr,
                                                macroSync, eventAttr);
          },
          py::arg("set_pipe") = py::none(), py::arg("wait_pipe") = py::none(),
          py::arg("macro_sync") = hivm::SyncEventSlotMacroSync::wait,
          py::arg("event") = py::none())
      .def("get_vf_mode_attr",
           [](AscendNPUIROpBuilder &self, hivm::VFMode mode) -> Attribute {
             return self.getBuilder().getAttr<hivm::VFModeAttr>(mode);
           })
      .def("get_iterator_types_attr",
           [](AscendNPUIROpBuilder &self,
              const std::vector<hivm::IteratorType> &array) {
             auto attrs = llvm::to_vector(
                 llvm::map_range(array, [&self](hivm::IteratorType type) {
                   return cast<Attribute>(
                       self.getBuilder().getAttr<hivm::IteratorTypeAttr>(type));
                 }));
             return self.getBuilder().getArrayAttr(attrs);
           })
      .def("get_array_attr",
           [](AscendNPUIROpBuilder &self, const std::vector<Attribute> &attrs)
               -> Attribute { return self.getBuilder().getArrayAttr(attrs); })
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
             // Enable parsing of HACC attributes by allowing unregistered
             // dialects.
             ctx->allowUnregisteredDialects();
             return mlir::parseAttribute(value, ctx);
           })
      .def("get_affine_map_attr",
           [](AscendNPUIROpBuilder &self, AffineMap affineMap) -> Attribute {
             return AffineMapAttr::get(affineMap);
           })
      .def("get_affine_map_array_attr",
           [](AscendNPUIROpBuilder &self,
              const std::vector<AffineMap> &affineMaps) -> Attribute {
             auto *ctx = self.getBuilder().getContext();
             llvm::SmallVector<Attribute> attrs;
             attrs.reserve(affineMaps.size());
             for (const auto &map : affineMaps) {
               attrs.push_back(AffineMapAttr::get(map));
             }
             return ArrayAttr::get(ctx, attrs);
           })
      .def("get_buffer_ty_with_affine_map",
           [](AscendNPUIROpBuilder &self, std::vector<int64_t> &shape,
              Type &elementType, AffineMap affineMap,
              const Attribute &memorySpace) -> Type {
             auto layout = AffineMapAttr::get(affineMap);
             return MemRefType::get(shape, elementType, layout, memorySpace);
           })
      .def("create_fixpipe",
           [](AscendNPUIROpBuilder &self, Value src, py::object dst_obj,
              hivm::FixpipeDMAMode dma_mode,
              hivm::FixpipeDualDstMode dual_dst_mode,
              hivm::FixpipePreQuantMode pre_quant_mode,
              hivm::FixpipePreReluMode pre_relu_mode) -> py::object {
             if (!dyn_cast<RankedTensorType>(src.getType())) {
               llvm_unreachable("src is not of RankedTensorType");
             }

             auto *ctx = self.getBuilder().getContext();
             auto loc = self.getLastLoc();

             Value dstValue;
             bool needCreateDst = dst_obj.is_none();

             if (needCreateDst) {
               auto srcType = dyn_cast<RankedTensorType>(src.getType());
               auto srcShape = srcType.getShape();

               llvm::SmallVector<int64_t> dstShape(srcShape.begin(),
                                                   srcShape.end());

               if (dual_dst_mode == hivm::FixpipeDualDstMode::ROW_SPLIT) {
                 if (dstShape.size() >= 1 && dstShape[0] > 0) {
                   dstShape[0] = dstShape[0] / 2;
                 }
               } else if (dual_dst_mode ==
                          hivm::FixpipeDualDstMode::COLUMN_SPLIT) {
                 if (dstShape.size() >= 2 && dstShape[1] > 0) {
                   dstShape[1] = dstShape[1] / 2;
                 }
               }

               auto dstType =
                   RankedTensorType::get(dstShape, srcType.getElementType());
               auto emptyTensor = self.create<tensor::EmptyOp>(
                   dstType.getShape(), dstType.getElementType());
               dstValue = emptyTensor.getResult();
             } else {
               dstValue = py::cast<Value>(dst_obj);
               if (!dyn_cast<ShapedType>(dstValue.getType())) {
                 llvm_unreachable("dst is not of ShapedType");
               }
             }

             auto dma_mode_attr =
                 mlir::hivm::FixpipeDMAModeAttr::get(ctx, dma_mode);
             auto dual_dst_mode_attr =
                 mlir::hivm::FixpipeDualDstModeAttr::get(ctx, dual_dst_mode);
             auto pre_quant_mode_attr =
                 mlir::hivm::FixpipePreQuantModeAttr::get(ctx, pre_quant_mode);
             auto pre_relu_mode_attr =
                 mlir::hivm::FixpipePreReluModeAttr::get(ctx, pre_relu_mode);
             auto channel_split = BoolAttr::get(ctx, false);
             if (needCreateDst) {
               return py::cast<Value>(
                   self.create<hivm::FixpipeOp>(
                           mlir::TypeRange{dstValue.getType()}, src, dstValue,
                           dma_mode_attr, dual_dst_mode_attr, nullptr,
                           pre_quant_mode_attr, pre_relu_mode_attr,
                           channel_split)
                       .getResult(0));

             } else {
               self.create<hivm::FixpipeOp>(mlir::TypeRange{}, src, dstValue,
                                            dma_mode_attr, dual_dst_mode_attr,
                                            nullptr, pre_quant_mode_attr,
                                            pre_relu_mode_attr, channel_split);
               return py::none();
             }
           })
      .def("create_annotation_mark",
           [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
              Attribute &attrVal) {
             auto annotationOp = self.create<annotation::MarkOp>(ptr);
             annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                                   attrVal);
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
              const std::vector<Value> &outs,
              const std::vector<py::dict> &arg_attrs) -> std::vector<Value> {
             ValueRange inputs{ins};
             ValueRange outputs{outs};
             ValueRange temp_buffers{};
             TypeRange res_types{outputs};
             auto op = self.create<hivm::CustomOp>(res_types, name, inputs,
                                                   outputs, temp_buffers);
             for (auto &attr : attrs) {
               std::string attr_name = py::cast<std::string>(attr.first);
               Attribute attr_value = py::cast<Attribute>(attr.second);
               op->setAttr(attr_name, attr_value);
             }

             SmallVector<Attribute> dictAttrs(arg_attrs.size());
             Attribute emptyDict = self.getBuilder().getDictionaryAttr({});
             for (const auto &[idx, attrs] : llvm::enumerate(arg_attrs)) {
               if (idx >= op.getNumOperands())
                 continue;

               if (attrs.is_none()) {
                 dictAttrs[idx] = emptyDict;
                 continue;
               }

               llvm::SmallVector<NamedAttribute> namedAttrs;
               for (const auto &attr : attrs) {
                 std::string attr_name = py::cast<std::string>(attr.first);
                 Attribute attr_value = py::cast<Attribute>(attr.second);
                 namedAttrs.push_back(NamedAttribute(
                     self.getBuilder().getStringAttr(attr_name), attr_value));
               }

               dictAttrs[idx] = self.getBuilder().getDictionaryAttr(namedAttrs);
             }

             ArrayAttr arg_attrs_array =
                 self.getBuilder().getArrayAttr(dictAttrs);
             op->setAttr("arg_attrs", arg_attrs_array);

             auto results = op->getResults();
             return std::vector<Value>(results.begin(), results.end());
           })
      .def("create_custom_macro_op",
           [](AscendNPUIROpBuilder &self, const std::string &name,
              const py::dict &attrs, const std::vector<Value> &ins,
              const std::vector<Value> &outs,
              const std::vector<py::dict> &arg_attrs) -> std::vector<Value> {
             ValueRange inputs{ins};
             ValueRange outputs{outs};
             ValueRange temp_buffers{};
             ValueRange syncArgs{};
             TypeRange res_types{outputs};
             auto op = self.create<hivm::CustomMacroOp>(
                 res_types, name, inputs, outputs, temp_buffers, syncArgs);
             for (auto &attr : attrs) {
               std::string attr_name = py::cast<std::string>(attr.first);
               Attribute attr_value = py::cast<Attribute>(attr.second);
               op->setAttr(attr_name, attr_value);
             }

             SmallVector<Attribute> dictAttrs(arg_attrs.size());
             Attribute emptyDict = self.getBuilder().getDictionaryAttr({});
             for (const auto &[idx, attrs] : llvm::enumerate(arg_attrs)) {
               if (idx >= op.getNumOperands())
                 continue;

               if (attrs.is_none()) {
                 dictAttrs[idx] = emptyDict;
                 continue;
               }

               llvm::SmallVector<NamedAttribute> namedAttrs;
               for (const auto &attr : attrs) {
                 std::string attr_name = py::cast<std::string>(attr.first);
                 Attribute attr_value = py::cast<Attribute>(attr.second);
                 namedAttrs.push_back(NamedAttribute(
                     self.getBuilder().getStringAttr(attr_name), attr_value));
               }

               dictAttrs[idx] = self.getBuilder().getDictionaryAttr(namedAttrs);
             }

             ArrayAttr arg_attrs_array =
                 self.getBuilder().getArrayAttr(dictAttrs);
             op->setAttr("arg_attrs", arg_attrs_array);

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

  installTritonContextManager();
}
