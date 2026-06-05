#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/STLExtras.h"
#include <iostream>
#include <optional>
#include <stdexcept>

namespace py = pybind11;
using LinearLayout = mlir::triton::LinearLayout;

namespace {

mlir::MLIRContext *getLinearLayoutContext() {
  static PyObject *ctxObject = []() {
    py::module irMod = py::module::import("triton._C.libtriton.ir");
    // Keep the Python object alive for the life of the process without running
    // its destructor during interpreter shutdown (avoids segfaults).
    py::object ctx = irMod.attr("context")();
    return ctx.release().ptr();
  }();
  return py::cast<mlir::MLIRContext *>(py::handle(ctxObject));
}

} // namespace

void init_linear_layout(py::module &&m) {
  py::class_<LinearLayout>(m, "LinearLayout", py::module_local(false))
      .def(py::init<>())
      .def_static(
          "identity_1d",
          [](int32_t size, std::string inDim, std::string outDim) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::identity1D(size,
                                            mlir::StringAttr::get(ctx, inDim),
                                            mlir::StringAttr::get(ctx, outDim));
          },
          py::arg("size"), py::arg("inDim"), py::arg("outDim"))
      .def_static(
          "strided_1d",
          [](int32_t size, int32_t stride, std::string inDim,
             std::string outDim) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::strided1D(size, stride,
                                           mlir::StringAttr::get(ctx, inDim),
                                           mlir::StringAttr::get(ctx, outDim));
          },
          py::arg("size"), py::arg("stride"), py::arg("inDim"),
          py::arg("outDim"))
      .def_static(
          "zeros_1d",
          [](int32_t size, std::string inDim, std::string outDim,
             int32_t outDimSize) {
            auto *ctx = getLinearLayoutContext();
            return LinearLayout::zeros1D(
                size, mlir::StringAttr::get(ctx, inDim),
                mlir::StringAttr::get(ctx, outDim), outDimSize);
          },
          py::arg("size"), py::arg("inDim"), py::arg("outDim"),
          py::arg("outDimSize") = 1)
      .def_static(
          "from_bases",
          [](const std::vector<std::pair<
                 std::string, std::vector<std::vector<int32_t>>>> &bases,
             const std::vector<std::string> &outDimNames,
             std::optional<std::vector<int32_t>> outDimSizes,
             bool requireSurjective) {
            auto *ctx = getLinearLayoutContext();

            std::vector<
                std::pair<mlir::StringAttr, std::vector<std::vector<int32_t>>>>
                convertedBases;
            convertedBases.reserve(bases.size());
            for (const auto &entry : bases) {
              std::vector<std::vector<int32_t>> converted;
              converted.reserve(entry.second.size());
              for (const auto &vec : entry.second)
                converted.emplace_back(vec.begin(), vec.end());
              convertedBases.emplace_back(
                  mlir::StringAttr::get(ctx, entry.first),
                  std::move(converted));
            }

            if (outDimSizes) {
              if (outDimSizes->size() != outDimNames.size())
                throw std::invalid_argument("out_dim_names and out_dim_sizes "
                                            "must have the same length");
              std::vector<std::pair<mlir::StringAttr, int32_t>> outDims;
              outDims.reserve(outDimNames.size());
              for (auto it : llvm::enumerate(outDimNames))
                outDims.emplace_back(mlir::StringAttr::get(ctx, it.value()),
                                     (*outDimSizes)[it.index()]);
              return LinearLayout(convertedBases, outDims, requireSurjective);
            }

            if (!requireSurjective)
              throw std::invalid_argument("out_dim_sizes must be provided when "
                                          "require_surjective is false");

            std::vector<mlir::StringAttr> convertedNames;
            convertedNames.reserve(outDimNames.size());
            for (const auto &name : outDimNames)
              convertedNames.push_back(mlir::StringAttr::get(ctx, name));
            return LinearLayout(convertedBases, convertedNames);
          },
          py::arg("bases"), py::arg("out_dim_names"),
          py::arg("out_dim_sizes") = py::none(),
          py::arg("require_surjective") = true)
      .def("compose", &LinearLayout::compose)
      .def("invert_and_compose", &LinearLayout::invertAndCompose)
      .def("invert", &LinearLayout::invert)
      .def("pseudoinvert", &LinearLayout::pseudoinvert)
      .def("is_surjective", &LinearLayout::isSurjective)
      .def("is_injective", &LinearLayout::isInjective)
      .def("is_invertible", &LinearLayout::isInvertible)
      .def("get_in_dim_names",
           [](const LinearLayout &self) {
             std::vector<std::string> dims;
             dims.reserve(self.getNumInDims());
             for (mlir::StringAttr dim : self.getInDimNames())
               dims.push_back(dim.str());
             return dims;
           })
      .def("get_out_dim_names",
           [](const LinearLayout &self) {
             std::vector<std::string> dims;
             dims.reserve(self.getNumOutDims());
             for (mlir::StringAttr dim : self.getOutDimNames())
               dims.push_back(dim.str());
             return dims;
           })
      .def_property_readonly(
          "bases",
          [](const LinearLayout &self) {
            auto bases = self.getBases();
            pybind11::list result;
            for (const auto &it : bases) {
              pybind11::list dimBases;
              for (const auto &vec : it.second)
                dimBases.append(pybind11::cast(
                    std::vector<int32_t>(vec.begin(), vec.end())));
              result.append(pybind11::make_tuple(it.first.str(), dimBases));
            }
            return result;
          })
      .def_property_readonly(
          "out_dims",
          [](const LinearLayout &self) {
            pybind11::list result;
            for (const auto &it : self.getOutDims()) {
              result.append(pybind11::make_tuple(it.first.str(), it.second));
            }
            return result;
          })
      .def_property_readonly("num_in_dims", &LinearLayout::getNumInDims)
      .def_property_readonly("num_out_dims", &LinearLayout::getNumOutDims)
      .def("__mul__", [](const LinearLayout &lhs,
                         const LinearLayout &rhs) { return lhs * rhs; })
      .def(
          "__imul__",
          [](LinearLayout &lhs, const LinearLayout &rhs) -> LinearLayout & {
            lhs *= rhs;
            return lhs;
          },
          py::return_value_policy::reference_internal)
      .def("__eq__", [](const LinearLayout &lhs,
                        const LinearLayout &rhs) { return lhs == rhs; })
      .def("__ne__", [](const LinearLayout &lhs,
                        const LinearLayout &rhs) { return lhs != rhs; })
      .def("__repr__", [](const LinearLayout &self) { return self.toString(); })
      .def("__str__", [](const LinearLayout &self) { return self.toString(); })
      .def("get_shared_view",
           [](const LinearLayout &self, bool useHWPointOfView) {
             return mlir::triton::gpu::getSharedLayoutStr(
                 const_cast<LinearLayout &>(self), useHWPointOfView);
           })
      .def("get_distributed_view",
           [](const LinearLayout &self, bool useHWPointOfView) {
             return mlir::triton::gpu::getDistributedLayoutStr(
                 const_cast<LinearLayout &>(self), useHWPointOfView);
           })
      .def(
          "apply",
          [](const LinearLayout &self, py::dict inputsDict) {
            std::vector<std::pair<std::string, int32_t>> inputs;
            inputs.reserve(inputsDict.size());
            for (auto item : inputsDict) {
              inputs.emplace_back(py::cast<std::string>(item.first),
                                  py::cast<int32_t>(item.second));
            }
            auto *ctx = getLinearLayoutContext();
            std::vector<std::pair<mlir::StringAttr, int32_t>> converted;
            converted.reserve(inputs.size());
            for (const auto &it : inputs) {
              converted.emplace_back(mlir::StringAttr::get(ctx, it.first),
                                     it.second);
            }
            auto outputs = self.apply(converted);
            py::dict result;
            for (const auto &out : outputs) {
              result[py::str(out.first.str())] = out.second;
            }
            return result;
          },
          py::arg("inputs"))
      .def("get_matrix_view",
           [](const LinearLayout &self) {
             std::unique_ptr<uint64_t[]> matrix = mlir::triton::getMatrix(self);
             auto nRows = self.getTotalOutDimSizeLog2();
             auto nCols = self.getTotalInDimSizeLog2();
             std::vector<std::vector<int>> result(nRows,
                                                  std::vector<int>(nCols));
             for (size_t i = 0; i < nRows; ++i) {
               for (size_t j = 0; j < nCols; ++j) {
                 result[i][j] = (matrix[i] >> j) & 1;
               }
             }
             return result;
           })
      .def(
          "get_2d_matrix_view",
          [](const LinearLayout &self,
             const std::optional<std::pair<std::string, std::string>>
                 &rowColDims,
             const std::optional<std::vector<bool>> &printInfo,
             int32_t threadsPerWarp) -> std::vector<std::vector<std::string>> {
            auto *ctx = getLinearLayoutContext();

            // Get output dimension names and sizes
            auto outDims = self.getOutDims();
            if (outDims.empty()) {
              return {};
            }

            // Determine row and column dimensions
            std::string rowDimName, colDimName;
            if (rowColDims.has_value()) {
              rowDimName = rowColDims->first;
              colDimName = rowColDims->second;
            } else if (outDims.size() >= 2) {
              // Default to first two output dimensions
              auto it = outDims.begin();
              rowDimName = it->first.str();
              ++it;
              colDimName = it->first.str();
            } else if (outDims.size() == 1) {
              // Single dimension - treat as 1D layout
              rowDimName = outDims.begin()->first.str();
              colDimName = "";
            }

            // Validate dimensions exist
            auto rowDimAttr = mlir::StringAttr::get(ctx, rowDimName);
            auto colDimAttr = mlir::StringAttr::get(ctx, colDimName);

            if (!self.hasOutDim(rowDimAttr)) {
              throw std::invalid_argument("Row dimension '" + rowDimName +
                                          "' not found in layout");
            }
            if (!colDimName.empty() && !self.hasOutDim(colDimAttr)) {
              throw std::invalid_argument("Column dimension '" + colDimName +
                                          "' not found in layout");
            }

            // Get dimension sizes
            int32_t nRows = self.getOutDimSize(rowDimAttr);
            int32_t nCols =
                colDimName.empty() ? 1 : self.getOutDimSize(colDimAttr);

            // Initialize result matrix with empty strings
            std::vector<std::vector<std::string>> result(
                nRows, std::vector<std::string>(nCols));

            // Get input dimension information
            auto inDims = self.getInDims();
            int32_t numRegisters = 1;
            int32_t numLanes = 1;
            int32_t numWarps = 1;
            int32_t numBlocks = 1;

            bool enablePrintInfo = printInfo.has_value();
            int numPrint = enablePrintInfo ? printInfo->size() : -1;
            bool printRegisters = numPrint > 0 ? printInfo->at(0) : true;
            bool printLanes = numPrint > 1 ? printInfo->at(1) : true;
            bool printWarps = numPrint > 2 ? printInfo->at(2) : true;
            bool printBlocks = numPrint > 3 ? printInfo->at(3) : true;

            for (const auto &[dimName, dimSize] : inDims) {
              std::string name = dimName.str();
              if (name == "register") {
                numRegisters = dimSize;
              } else if (name == "lane") {
                numLanes = dimSize;
              } else if (name == "warp") {
                numWarps = dimSize;
              } else if (name == "block") {
                numBlocks = dimSize;
              }
            }

            // Iterate over all hardware positions and fill the matrix
            for (int32_t block = 0; block < numBlocks; ++block) {
              for (int32_t warp = 0; warp < numWarps; ++warp) {
                for (int32_t lane = 0; lane < numLanes; ++lane) {
                  for (int32_t reg = 0; reg < numRegisters; ++reg) {
                    // Build input dictionary
                    std::vector<std::pair<mlir::StringAttr, int32_t>> inputs;
                    for (const auto &[dimName, _] : inDims) {
                      std::string name = dimName.str();
                      int32_t value = 0;
                      if (name == "register")
                        value = reg;
                      else if (name == "lane")
                        value = lane;
                      else if (name == "warp")
                        value = warp;
                      else if (name == "block")
                        value = block;
                      inputs.emplace_back(dimName, value);
                    }

                    // Apply layout to get output coordinates
                    auto outputs = self.apply(inputs);

                    // Find row and column indices
                    int32_t rowIdx = 0;
                    int32_t colIdx = 0;
                    bool foundRow = false, foundCol = false;

                    for (const auto &[outDim, outVal] : outputs) {
                      if (outDim == rowDimAttr) {
                        rowIdx = outVal;
                        foundRow = true;
                      }
                      if (!colDimName.empty() && outDim == colDimAttr) {
                        colIdx = outVal;
                        foundCol = true;
                      }
                    }

                    if (!foundRow || (!colDimName.empty() && !foundCol)) {
                      continue;
                    }

                    // Validate indices
                    if (rowIdx < 0 || rowIdx >= nRows || colIdx < 0 ||
                        colIdx >= nCols) {
                      continue;
                    }

                    // Build hardware parameter label
                    std::string label;
                    if (numRegisters > 1 && printRegisters) {
                      label += "r" + std::to_string(reg);
                    }

                    if (numLanes > 1 && printLanes) {
                      label += "t" + std::to_string(lane);
                    }

                    if (numWarps > 1 && printWarps) {
                      label += "w" + std::to_string(warp);
                    }

                    if (numBlocks > 1 && printBlocks) {
                      label += "b" + std::to_string(block);
                    }

                    // Add to matrix cell (append if multiple entries)
                    if (!result[rowIdx][colIdx].empty()) {
                      result[rowIdx][colIdx] += ",";
                    }
                    result[rowIdx][colIdx] += label;
                  }
                }
              }
            }

            // Mark empty cells
            for (int32_t i = 0; i < nRows; ++i) {
              for (int32_t j = 0; j < nCols; ++j) {
                if (result[i][j].empty()) {
                  result[i][j] = ".";
                }
              }
            }

            return result;
          },
          py::arg("row_col_dims") = py::none(),
          py::arg("print_info") = py::none(), py::arg("threads_per_warp") = 64,
          R"doc(
          Returns a 2D matrix view of the layout showing hardware parameter distribution.

          Each cell in the matrix represents a tensor element (identified by its output
          coordinates), and the cell value shows which hardware location (register,
          thread, warp, block) maps to that tensor element.

          Args:
              row_col_dims: Optional tuple of (row_dimension, column_dimension) output
                  dimension names. If None, uses the first two output dimensions.
              threads_per_warp: Number of threads per warp.

          Returns:
              2D list of strings where each cell contains hardware parameter labels
              in format "r{register}t{thread}w{warp}b{block}".
          )doc");
}
