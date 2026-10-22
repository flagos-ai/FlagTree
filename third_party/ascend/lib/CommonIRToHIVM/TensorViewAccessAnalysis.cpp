/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

//===----------------------------------------------------------------------===//
// TensorViewAccessAnalysis
//===----------------------------------------------------------------------===//

#include "ascend/include/CommonIRToHIVM/TensorViewAccessAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace mlir {
namespace triton {
namespace {

static std::optional<int64_t> checkedMul(int64_t a, int64_t b) {
  if (a == 0 || b == 0)
    return int64_t{0};
  if (a > 0 && b > 0 && a > std::numeric_limits<int64_t>::max() / b)
    return std::nullopt;
  return a * b;
}

static int64_t getElementByteWidth(Type type) {
  unsigned bits = 0;
  if (auto intTy = dyn_cast<IntegerType>(type))
    bits = intTy.getWidth();
  else if (auto floatTy = dyn_cast<FloatType>(type))
    bits = floatTy.getWidth();
  else
    return -1;
  return static_cast<int64_t>((bits + 7) / 8);
}

static IndexPatternInfo
makeStepPattern(Value value, std::optional<int64_t> step,
                std::optional<int64_t> constantBase = std::nullopt,
                bool runtimeLoaded = false) {
  IndexPatternInfo result;
  result.value = value;
  result.step = step;
  result.constantBase = constantBase;
  result.runtimeLoaded = runtimeLoaded;

  if (!step)
    result.kind = IndexPatternKind::Unknown;
  else if (*step == 0)
    result.kind = IndexPatternKind::Broadcast;
  else if (*step == 1)
    result.kind = IndexPatternKind::Contiguous;
  else
    result.kind = IndexPatternKind::AffineStrided;

  return result;
}

static std::optional<int64_t> getSplatInteger(Value value, unsigned depth = 0) {
  if (!value || depth > 8)
    return std::nullopt;

  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    Attribute attr = cst.getValue();

    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return intAttr.getInt();

    if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    }
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;

  StringRef name = def->getName().getStringRef();

  if ((name == "tt.splat" || name == "tt.broadcast") &&
      def->getNumOperands() == 1)
    return getSplatInteger(def->getOperand(0), depth + 1);

  if ((name == "arith.index_cast" || name == "arith.extsi" ||
       name == "arith.extui" || name == "arith.trunci") &&
      def->getNumOperands() == 1)
    return getSplatInteger(def->getOperand(0), depth + 1);

  return std::nullopt;
}

static IndexPatternInfo
analyzeDenseIntegerConstant(Value value, DenseIntElementsAttr dense) {
  if (dense.isSplat()) {
    int64_t c = dense.getSplatValue<APInt>().getSExtValue();
    return makeStepPattern(value, 0, c);
  }

  auto shaped = dyn_cast<ShapedType>(dense.getType());
  if (!shaped || !shaped.hasStaticShape() || shaped.getRank() != 1)
    return makeStepPattern(value, std::nullopt);

  auto vals = dense.getValues<APInt>();
  if (vals.empty())
    return makeStepPattern(value, std::nullopt);

  // DenseElementsAttr's element iterators yield APInt *by value*, so their
  // PointerT is APInt rather than APInt*, which makes `it->` ill-formed
  // (iterator_facade_base::PointerProxy::operator-> returns `&R`). Dereference
  // with `*it` instead.
  auto it = vals.begin();
  int64_t first = (*it).getSExtValue();
  ++it;

  if (it == vals.end())
    return makeStepPattern(value, 0, first);

  int64_t second = (*it).getSExtValue();
  int64_t step = second - first;
  int64_t prev = second;
  ++it;

  for (; it != vals.end(); ++it) {
    int64_t cur = (*it).getSExtValue();
    if (cur - prev != step)
      return makeStepPattern(value, std::nullopt);
    prev = cur;
  }

  return makeStepPattern(value, step, first);
}

static IndexPatternInfo analyzeIndexProducerImpl(Value value,
                                                 unsigned depth = 0) {
  if (!value || depth > 16)
    return makeStepPattern(value, std::nullopt);

  Operation *def = value.getDefiningOp();
  if (!def)
    return makeStepPattern(value, std::nullopt);

  StringRef name = def->getName().getStringRef();

  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
      return analyzeDenseIntegerConstant(value, dense);

    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return makeStepPattern(value, 0, intAttr.getInt());
  }

  if (name == "tt.make_range") {
    auto start = def->getAttrOfType<IntegerAttr>("start");
    if (start)
      return makeStepPattern(value, 1, start.getInt());
    return makeStepPattern(value, 1);
  }

  if ((name == "tt.splat" || name == "tt.broadcast") &&
      def->getNumOperands() == 1) {
    Value src = def->getOperand(0);
    if (!isa<ShapedType>(src.getType()))
      return makeStepPattern(value, 0, getSplatInteger(src));
    return makeStepPattern(value, std::nullopt);
  }

  if ((name == "arith.index_cast" || name == "arith.extsi" ||
       name == "arith.extui" || name == "arith.trunci") &&
      def->getNumOperands() == 1) {
    IndexPatternInfo inner =
        analyzeIndexProducerImpl(def->getOperand(0), depth + 1);
    inner.value = value;
    return inner;
  }

  if ((name == "arith.addi" || name == "arith.subi") &&
      def->getNumOperands() == 2) {
    auto lhs = analyzeIndexProducerImpl(def->getOperand(0), depth + 1);
    auto rhs = analyzeIndexProducerImpl(def->getOperand(1), depth + 1);

    if (lhs.step && rhs.step) {
      int64_t step = name == "arith.addi" ? (*lhs.step + *rhs.step)
                                          : (*lhs.step - *rhs.step);

      std::optional<int64_t> base;
      if (lhs.constantBase && rhs.constantBase) {
        base = name == "arith.addi" ? (*lhs.constantBase + *rhs.constantBase)
                                    : (*lhs.constantBase - *rhs.constantBase);
      }

      return makeStepPattern(value, step, base,
                             lhs.runtimeLoaded || rhs.runtimeLoaded);
    }

    return makeStepPattern(value, std::nullopt, std::nullopt,
                           lhs.runtimeLoaded || rhs.runtimeLoaded);
  }

  if (name == "arith.muli" && def->getNumOperands() == 2) {
    Value lhsValue = def->getOperand(0);
    Value rhsValue = def->getOperand(1);

    if (auto c = getSplatInteger(rhsValue)) {
      auto lhs = analyzeIndexProducerImpl(lhsValue, depth + 1);
      if (lhs.step) {
        std::optional<int64_t> base;
        if (lhs.constantBase)
          base = *lhs.constantBase * *c;
        return makeStepPattern(value, *lhs.step * *c, base, lhs.runtimeLoaded);
      }
    }

    if (auto c = getSplatInteger(lhsValue)) {
      auto rhs = analyzeIndexProducerImpl(rhsValue, depth + 1);
      if (rhs.step) {
        std::optional<int64_t> base;
        if (rhs.constantBase)
          base = *rhs.constantBase * *c;
        return makeStepPattern(value, *rhs.step * *c, base, rhs.runtimeLoaded);
      }
    }

    return makeStepPattern(value, std::nullopt);
  }

  if (name == "tt.load")
    return makeStepPattern(value, std::nullopt, std::nullopt,
                           /*runtimeLoaded=*/true);

  return makeStepPattern(value, std::nullopt);
}

static FailureOr<DensePayloadInfo>
analyzeDensePayload(const TensorViewAccessInfo &view) {
  const unsigned rank = view.tile.size();

  if (rank == 0 || view.sparseDims.empty())
    return failure();

  unsigned lastSparse = 0;
  for (int64_t dim : view.sparseDims) {
    if (dim < 0 || static_cast<unsigned>(dim) >= rank)
      return failure();
    lastSparse = std::max(lastSparse, static_cast<unsigned>(dim));
  }

  DensePayloadInfo payload;
  // The payload is the dense suffix after the last sparse dimension.
  payload.firstDenseDim = lastSparse + 1;

  const int64_t elemBytes = getElementByteWidth(view.elementType);

  // No trailing dense dimensions means scalar transfer.
  if (payload.firstDenseDim >= rank) {
    payload.elements = 1;
    payload.bytes = elemBytes;
    payload.staticallyContiguous = true;
    payload.staticallyStrideable = true;
    return payload;
  }

  int64_t elements = 1;

  for (unsigned dim = payload.firstDenseDim; dim < rank; ++dim) {
    unsigned baseDim = static_cast<unsigned>(view.dimMap[dim]);

    if (baseDim >= view.staticStrides.size())
      return failure();

    int64_t extent = view.tile[dim];
    if (extent <= 0)
      return failure();

    payload.denseDims.push_back(dim);
    payload.shape.push_back(extent);

    int64_t stride = view.staticStrides[baseDim];

    payload.staticStrides.push_back(stride);

    if (stride == ShapedType::kDynamic)
      payload.hasDynamicStride = true;

    auto next = checkedMul(elements, extent);
    if (!next)
      return failure();

    elements = *next;
  }

  payload.elements = elements;

  if (elemBytes > 0) {
    if (auto bytes = checkedMul(elements, elemBytes))
      payload.bytes = *bytes;
  }

  payload.affine = true;

  if (!payload.hasDynamicStride && !payload.staticStrides.empty()) {
    bool contiguous = payload.staticStrides.back() == 1;

    for (int i = static_cast<int>(payload.staticStrides.size()) - 2;
         contiguous && i >= 0; --i) {
      int64_t expected = payload.shape[i + 1] * payload.staticStrides[i + 1];

      contiguous = payload.staticStrides[i] == expected;
    }

    payload.staticallyContiguous = contiguous;

    // A payload that is not densely packed can still be one strided DMA, as
    // long as each dim's stride does not run backwards over the next dim's
    // span: that is exactly a memref with a StridedLayoutAttr, which
    // emitPayloadGmView already builds and hivm.hir.load already accepts. The
    // cases this rejects are a transposed dim_map (innermost stride != 1, so
    // the logical block is a transpose of the memory layout rather than a skip)
    // and dims that overlap.
    bool strideable = payload.staticStrides.back() == 1;

    for (int i = static_cast<int>(payload.staticStrides.size()) - 2;
         strideable && i >= 0; --i) {
      int64_t span = payload.shape[i + 1] * payload.staticStrides[i + 1];

      strideable = payload.staticStrides[i] >= span;
    }

    payload.staticallyStrideable = strideable;
  }

  return payload;
}

} // namespace

//===----------------------------------------------------------------------===//
// Printing
//
// Several fields use a sentinel for "not known": `bytes` and the two cost
// counters are -1, `staticStrides` entries may be ShapedType::kDynamic, and the
// optionals may be empty. All of those print as `?` so a dump never shows a
// sentinel as if it were a real value.
//===----------------------------------------------------------------------===//

namespace {

void printIndent(llvm::raw_ostream &os, unsigned indent) { os.indent(indent); }

/// `-1` is the "unknown" sentinel for byte sizes and access counts.
void printCount(llvm::raw_ostream &os, int64_t value) {
  if (value < 0)
    os << "?";
  else
    os << value;
}

void printOptional(llvm::raw_ostream &os, std::optional<int64_t> value) {
  if (value)
    os << *value;
  else
    os << "?";
}

/// Renders ShapedType::kDynamic as `?`, matching how MLIR prints shapes.
/// Used for both static shapes and static strides.
void printDynamicAwareList(llvm::raw_ostream &os, ArrayRef<int64_t> values) {
  os << "[";
  llvm::interleaveComma(values, os, [&](int64_t s) {
    if (ShapedType::isDynamic(s))
      os << "?";
    else
      os << s;
  });
  os << "]";
}

template <typename RangeT>
void printList(llvm::raw_ostream &os, const RangeT &range) {
  os << "[";
  llvm::interleaveComma(range, os, [&](auto v) { os << v; });
  os << "]";
}

/// Shared by every toString() below: print into a string via
/// raw_string_ostream.
template <typename FnT> std::string captureToString(FnT &&emit) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  emit(os);
  return os.str();
}

} // namespace

void IndexPatternInfo::print(llvm::raw_ostream &os, unsigned indent,
                             bool multiline) const {
  if (!multiline) {
    printIndent(os, indent);
    os << "dim" << viewDim << ": " << stringifyIndexPatternKind(kind)
       << " step=";
    printOptional(os, step);
    os << " base=";
    printOptional(os, constantBase);
    if (runtimeLoaded)
      os << " runtime-loaded";
    if (!value)
      os << " <null-value>";
    return;
  }

  printIndent(os, indent);
  os << "index dim" << viewDim << ":\n";
  printIndent(os, indent + 2);
  os << "kind            = " << stringifyIndexPatternKind(kind) << "\n";
  printIndent(os, indent + 2);
  os << "step            = ";
  printOptional(os, step);
  os << "\n";
  printIndent(os, indent + 2);
  os << "constantBase    = ";
  printOptional(os, constantBase);
  os << "\n";
  printIndent(os, indent + 2);
  os << "runtimeLoaded   = " << (runtimeLoaded ? "true" : "false") << "\n";
  printIndent(os, indent + 2);
  os << "value           = ";
  if (value)
    os << value;
  else
    os << "<null>";
  os << "\n";
}

std::string IndexPatternInfo::toString(bool multiline) const {
  return captureToString(
      [&](llvm::raw_ostream &os) { print(os, /*indent=*/0, multiline); });
}

void IndexPatternInfo::dump() const {
  print(llvm::errs(), /*indent=*/0, /*multiline=*/true);
  llvm::errs() << "\n";
}

void DensePayloadInfo::print(llvm::raw_ostream &os, unsigned indent) const {
  printIndent(os, indent);
  os << "dense payload:\n";
  printIndent(os, indent + 2);
  os << "firstDenseDim        = " << firstDenseDim << "\n";
  printIndent(os, indent + 2);
  os << "denseDims            = ";
  printList(os, denseDims);
  os << "\n";
  printIndent(os, indent + 2);
  os << "shape               = ";
  printList(os, shape);
  os << "\n";
  printIndent(os, indent + 2);
  os << "staticStrides       = ";
  printDynamicAwareList(os, staticStrides);
  os << "\n";
  printIndent(os, indent + 2);
  os << "elements            = " << elements << "\n";
  printIndent(os, indent + 2);
  os << "bytes               = ";
  printCount(os, bytes);
  os << "\n";
  printIndent(os, indent + 2);
  os << "affine              = " << (affine ? "true" : "false") << "\n";
  printIndent(os, indent + 2);
  os << "hasDynamicStride    = " << (hasDynamicStride ? "true" : "false")
     << "\n";
  printIndent(os, indent + 2);
  os << "staticallyContiguous= " << (staticallyContiguous ? "true" : "false")
     << "\n";
  printIndent(os, indent + 2);
  os << "staticallyStrideable= " << (staticallyStrideable ? "true" : "false")
     << "\n";
  printIndent(os, indent + 2);
  os << "isScalar            = " << (isScalar() ? "true" : "false") << "\n";
}

std::string DensePayloadInfo::toString() const {
  return captureToString([&](llvm::raw_ostream &os) { print(os); });
}

void DensePayloadInfo::dump() const { print(llvm::errs()); }

void TensorViewAccessInfo::print(llvm::raw_ostream &os, unsigned indent) const {
  printIndent(os, indent);
  os << "TensorView:\n";
  printIndent(os, indent + 2);
  os << "kind          = " << stringifyTensorViewKind(kind) << "\n";
  printIndent(os, indent + 2);
  os << "elementType   = ";
  if (elementType)
    os << elementType;
  else
    os << "<null>";
  os << "\n";
  printIndent(os, indent + 2);
  os << "tile          = ";
  printList(os, tile);
  os << "\n";
  printIndent(os, indent + 2);
  os << "traversal     = ";
  printList(os, traversal);
  os << "\n";
  printIndent(os, indent + 2);
  os << "dimMap        = ";
  printList(os, dimMap);
  os << "\n";
  printIndent(os, indent + 2);
  os << "sparseDims    = ";
  printList(os, sparseDims);
  os << "\n";
  // shape/strides hold Values, so they print as the defining op. staticShape
  // and staticStrides may hold ShapedType::kDynamic and go through the
  // sentinel-aware printer.
  printIndent(os, indent + 2);
  os << "staticShape   = ";
  printDynamicAwareList(os, staticShape);
  os << "\n";
  printIndent(os, indent + 2);
  os << "staticStrides = ";
  printDynamicAwareList(os, staticStrides);
  os << "\n";
  printIndent(os, indent + 2);
  os << "shape         = ";
  printList(os, shape);
  os << "\n";
  printIndent(os, indent + 2);
  os << "strides       = ";
  printList(os, strides);
  os << "\n";
}

std::string TensorViewAccessInfo::toString() const {
  return captureToString([&](llvm::raw_ostream &os) { print(os); });
}

void TensorViewAccessInfo::dump() const { print(llvm::errs()); }

void GatherScatterAccessInfo::print(llvm::raw_ostream &os,
                                    unsigned indent) const {
  printIndent(os, indent);
  os << "gather_scatter access:\n";
  printIndent(os, indent + 2);
  os << "sparseDims            = ";
  printList(os, sparseDims);
  os << "\n";
  printIndent(os, indent + 2);
  os << "denseDims             = ";
  printList(os, denseDims);
  os << "\n";
  printIndent(os, indent + 2);
  os << "scalarElementAccesses = ";
  printCount(os, scalarElementAccesses);
  os << "\n";
  printIndent(os, indent + 2);
  os << "payloadTransferCount  = ";
  printCount(os, payloadTransferCount);
  os << "\n";

  // The ratio is the point of the dump: it is how many scalar GM accesses one
  // payload transfer replaces, i.e. what the row-DMA fast path buys.
  if (scalarElementAccesses > 0 && payloadTransferCount > 0) {
    printIndent(os, indent + 2);
    os << "accessesPerTransfer   = "
       << (scalarElementAccesses / payloadTransferCount) << "\n";
  }

  payload.print(os, indent + 2);

  printIndent(os, indent + 2);
  os << "sparse indices (" << sparseIndices.size() << "):\n";
  for (const IndexPatternInfo &index : sparseIndices) {
    index.print(os, indent + 4, /*multiline=*/false);
    os << "\n";
  }
}

std::string GatherScatterAccessInfo::toString() const {
  return captureToString([&](llvm::raw_ostream &os) { print(os); });
}

void GatherScatterAccessInfo::dump() const { print(llvm::errs()); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const IndexPatternInfo &info) {
  info.print(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const DensePayloadInfo &info) {
  info.print(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const TensorViewAccessInfo &info) {
  info.print(os);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GatherScatterAccessInfo &info) {
  info.print(os);
  return os;
}

StringRef stringifyTensorViewKind(TensorViewKind kind) {
  switch (kind) {
  case TensorViewKind::Unknown:
    return "unknown";
  case TensorViewKind::Partition:
    return "partition";
  case TensorViewKind::Strided:
    return "strided";
  case TensorViewKind::GatherScatter:
    return "gather_scatter";
  }
  llvm_unreachable("unknown TensorViewKind");
}

StringRef stringifyIndexPatternKind(IndexPatternKind kind) {
  switch (kind) {
  case IndexPatternKind::Unknown:
    return "unknown";
  case IndexPatternKind::Broadcast:
    return "broadcast";
  case IndexPatternKind::Contiguous:
    return "contiguous";
  case IndexPatternKind::AffineStrided:
    return "affine-strided";
  case IndexPatternKind::Segmented:
    return "segmented";
  case IndexPatternKind::General:
    return "general";
  }
  llvm_unreachable("unknown IndexPatternKind");
}

FailureOr<GatherScatterAccessInfo>
analyzeGatherScatterAccess(const TensorViewAccessInfo &view,
                           ValueRange indices) {
  const unsigned rank = view.rank();

  if (rank == 0 || view.dimMap.size() != rank || indices.size() != rank ||
      view.sparseDims.empty())
    return failure();

  for (int64_t extent : view.tile)
    if (extent <= 0)
      return failure();

  for (int64_t dim : view.sparseDims)
    if (dim < 0 || static_cast<unsigned>(dim) >= rank)
      return failure();

  GatherScatterAccessInfo result;

  // check sparse dims and dense dims.
  for (unsigned d = 0; d < rank; ++d) {
    bool sparse = llvm::is_contained(view.sparseDims, d);

    bool tensorIndex = isa<RankedTensorType>(indices[d].getType());

    if (sparse != tensorIndex)
      return failure();

    if (sparse)
      result.sparseDims.push_back(d);
    else
      result.denseDims.push_back(d);
  }

  auto payload = analyzeDensePayload(view);

  if (failed(payload))
    return failure();

  result.payload = *payload;

  for (unsigned d : result.sparseDims) {
    IndexPatternInfo indexInfo = analyzeIndexProducerImpl(indices[d]);

    indexInfo.viewDim = d;

    result.sparseIndices.push_back(indexInfo);
  }

  int64_t scalarAccesses = 1;

  for (int64_t extent : view.tile) {
    auto next = checkedMul(scalarAccesses, extent);

    if (!next)
      return failure();

    scalarAccesses = *next;
  }

  result.scalarElementAccesses = scalarAccesses;

  if (result.payload.isScalar()) {
    result.payloadTransferCount = scalarAccesses;
  } else {
    int64_t transfers = 1;

    for (unsigned d = 0; d < result.payload.firstDenseDim; ++d) {
      auto next = checkedMul(transfers, view.tile[d]);

      if (!next)
        return failure();

      transfers = *next;
    }

    result.payloadTransferCount = transfers;
  }

  return result;
}

} // namespace triton
} // namespace mlir
