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

#ifndef COMMONIR_TO_HIVM_TENSORVIEW_ACCESS_ANALYSIS_H
#define COMMONIR_TO_HIVM_TENSORVIEW_ACCESS_ANALYSIS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace triton {
/*
 * Broadcast → load once + broadcast
 * Contiguous → one DMA
 * AffineStrided → strided DMA
 * Segmented → several DMA
 * General → native gather / scalar transfer
 */
enum class IndexPatternKind {
  Unknown,
  // index[i] = C
  Broadcast,
  // index[i] = base + i
  Contiguous,
  // index[i] = base + i * stride
  AffineStrided,
  // several contiguous/strided runs
  Segmented,
  // arbitrary indirect indexing
  General
};

enum class TensorViewKind {
  Unknown,
  Partition,
  Strided,
  GatherScatter,
};

enum class PaddingValue {
  ZERO,
  NAN_VALUE,
  POS_INF,
  NEG_INF,
};

struct IndexPatternInfo {
  unsigned viewDim = 0;
  Value value;
  IndexPatternKind kind = IndexPatternKind::Unknown;
  std::optional<int64_t> step;
  std::optional<int64_t> constantBase;
  bool runtimeLoaded = false;

  void print(raw_ostream &os, unsigned indent = 0,
             bool multiline = false) const;
  std::string toString(bool multiline = false) const;
  void dump() const;
};

struct DensePayloadInfo {
  unsigned firstDenseDim = 0;

  SmallVector<unsigned> denseDims;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> staticStrides;

  int64_t elements = 1;
  int64_t bytes = -1;

  bool affine = true;
  bool hasDynamicStride = false;
  bool staticallyContiguous = false;
  /// Expressible as one strided DMA, a superset of staticallyContiguous: the
  /// block may skip between dims as long as it never runs backwards. A
  /// transposed dim_map is not strideable.
  bool staticallyStrideable = false;

  bool isScalar() const { return denseDims.empty(); }

  void print(raw_ostream &os, unsigned indent = 0) const;
  std::string toString() const;
  void dump() const;
};

struct TensorViewAccessInfo {
  /// Defaulted so a partially built info (traceView's early returns) never
  /// carries an indeterminate kind.
  TensorViewKind kind = TensorViewKind::Unknown;

  Type elementType;

  // tile shape
  SmallVector<int64_t> tile;
  SmallVector<int64_t> traversal;
  SmallVector<int64_t> dimMap;
  SmallVector<int64_t> sparseDims;

  // base info
  Value base;
  PaddingValue padding = PaddingValue::ZERO;
  /// Whether sparse indices are range-checked. Only a GatherScatterView can
  /// turn this on.
  bool boundsCheck = false;
  SmallVector<Value> shape;
  SmallVector<int64_t> staticShape;

  SmallVector<Value> strides;
  SmallVector<int64_t> staticStrides;

  bool ok = false;
  unsigned rank() const { return tile.size(); }

  bool isSparse() const { return kind == TensorViewKind::GatherScatter; }

  bool isDense() const { return sparseDims.empty(); }

  void print(raw_ostream &os, unsigned indent = 0) const;
  std::string toString() const;
  void dump() const;
};

struct GatherScatterAccessInfo {
  SmallVector<unsigned> sparseDims;
  SmallVector<unsigned> denseDims;

  DensePayloadInfo payload;
  SmallVector<IndexPatternInfo> sparseIndices;

  int64_t scalarElementAccesses = -1;
  int64_t payloadTransferCount = -1;

  void print(raw_ostream &os, unsigned indent = 0) const;
  std::string toString() const;
  void dump() const;
};

raw_ostream &operator<<(raw_ostream &os, const IndexPatternInfo &info);
raw_ostream &operator<<(raw_ostream &os, const DensePayloadInfo &info);
raw_ostream &operator<<(raw_ostream &os, const TensorViewAccessInfo &info);
raw_ostream &operator<<(raw_ostream &os, const GatherScatterAccessInfo &info);

FailureOr<GatherScatterAccessInfo>
analyzeGatherScatterAccess(const TensorViewAccessInfo &view,
                           ValueRange indices);

StringRef stringifyIndexPatternKind(IndexPatternKind kind);

StringRef stringifyTensorViewKind(TensorViewKind kind);

} // namespace triton
} // namespace mlir

#endif // COMMONIR_TO_HIVM_TENSORVIEW_ACCESS_ANALYSIS_H
