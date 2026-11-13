#ifndef TRITON_HCU_MATRIX_SHARED_COPY_H
#define TRITON_HCU_MATRIX_SHARED_COPY_H

#include "Analysis/MatrixCopyGeometry.h"
#include "PhysicalSharedRegions.h"
#include "ScalarMemoryAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include <array>
#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton::HCU {

// A hardware 16B/lane, wave64 G2L copies contiguous physical rows. The ordinary
// SharedLinear chart is the complete source-coordinate map; no role, protocol
// version, reserved half-buffer, or frontend-provided source basis is needed.
struct MatrixSharedCopyPlan {
  unsigned rowAxis, columnAxis, waves, copies;
  int64_t rows, columns, pitch;
  bool structured, wholeColumns, fullRows, columnInBounds, wholeRows, cacheSwizzle;
  uint64_t sourceBytes, tileSpanBytes;
  SmallVector<std::array<int32_t, 2>> bases;
};

class MatrixSharedCopies {
  llvm::DenseMap<Operation *, MatrixSharedCopyPlan> plans;
  bool valid = true;
public:
  MatrixSharedCopies(ModuleOp mod, PhysicalSharedRegions &regions,
                         StringRef arch) {
    mod.walk([&](hcugpu::MatrixLoadToLocalOp op) {
      auto type = op.getDest().getType();
      auto shared = dyn_cast<gpu::SharedLinearEncodingAttr>(type.getEncoding());
      if (!shared) return; // Preserve unrelated legacy MLS lowering.
      auto fail = [&](StringRef why) {
        op.emitError("matrix shared copy: ") << why;
        valid = false;
      };
      if (arch != "gfx936" || gpu::TritonGPUDialect::getNumCTAs(mod) != 1 ||
          gpu::TritonGPUDialect::getThreadsPerWarp(mod) != 64 ||
          type.getRank() != 2 || type.getShape() != type.getAllocShape() ||
          !(type.getElementType().isF16() || type.getElementType().isBF16()) ||
          op.getShape().size() != 2 || op.getStrides().size() != 2 ||
          op.getIndices().size() != 3 ||
          !llvm::equal(op.getTensorShape(), type.getShape()) ||
          op.getCache() != triton::CacheModifier::NONE ||
          op.getEvict() != triton::EvictionPolicy::NORMAL || op.getIsVolatile() ||
          !op.getBoundaryCheck().empty()) {
        fail("requires rank-two f16/bf16 SharedLinear, wave64 single CTA and default cache policy");
        return;
      }
      auto pointer = dyn_cast<triton::PointerType>(op.getBase().getType());
      auto arg = dyn_cast<BlockArgument>(resolveWarpSpecializeCapture(op.getBase()));
      auto fn = op->getParentOfType<triton::FuncOp>();
      auto alignment = arg && fn ? fn.getArgAttrOfType<IntegerAttr>(arg.getArgNumber(), "tt.divisibility") : IntegerAttr{};
      if (!pointer || pointer.getAddressSpace() != 1 ||
          pointer.getPointeeType() != type.getElementType() || !arg || !fn ||
          arg.getOwner() != &fn.getBody().front() || !alignment ||
          alignment.getInt() < 16 || alignment.getInt() % 16 ||
          !regions.contiguousAligned(op.getDest(), 16)) {
        fail("requires actual aligned source argument and contiguous aligned allocation-backed destination views");
        return;
      }
      SmallVector<int64_t, 2> sizes, strides;
      for (auto [size, stride] : llvm::zip(op.getShape(), op.getStrides())) {
        auto extent = memory_analysis::constant(size);
        auto pitch = memory_analysis::constant(stride);
        if (!extent || !pitch) {
          fail("requires static extents and row pitch");
          return;
        }
        sizes.push_back(*extent);
        strides.push_back(*pitch);
      }
      unsigned waves = gpu::lookupNumWarps(op);
      auto geometry =
          getMatrixCopyGeometry(sizes, strides, type.getShape(), waves);
      if (!geometry) {
        fail("unsupported chart, row pitch or incomplete wave payload");
        return;
      }
      auto [row, col, rows, cols, pitch, count, tileSpan] = *geometry;
      if (!memory_analysis::multipleOf(op.getIndices()[col], 8) ||
          !llvm::all_of(op.getIndices(), [&](Value v) { return memory_analysis::uniform(v,op); })) {
        fail("unproven uniform origins, vector-aligned columns or representable aligned row pitch");
        return;
      }
      auto ll = shared.getLinearLayout();
      auto offset = StringAttr::get(mod.getContext(), "offset");
      auto dims = standardOutDimNames(mod.getContext(), 2);
      if (ll.getNumOutDims() != 2 ||
          !llvm::equal(ll.getOutDimSizes(), type.getShape()) ||
          ll.getInDimSize(StringAttr::get(mod.getContext(), "block")) != 1) {
        fail("unsupported chart or incomplete wave payload; use ordinary buffer copy");
        return;
      }
      ll = ll.sublayout({offset}, dims);
      if (!ll.isSurjective() || ll.getTotalInDimSize() != count ||
          ll.getTotalOutDimSize() != count) {
        fail("shared chart must biject the complete active tile"); return;
      }
      if (!isMatrixCopyVectorLayout(ll, offset, row, col)) {
        fail("physical vectors are not consecutive aligned global elements");
        return;
      }
      // Inspect the actual source vectors of each complete wave. With at most
      // four 16B vectors per source row, dropping IDXEN must preserve the
      // pitched cache-bank distribution; broad rows use byte-cache addressing.
      // This gfx936 cost policy is derived from the chart, never the operand
      // role, matrix dimensions, or a frontend layout tag. It is a measured
      // heuristic, not a claim that one policy dominates all input shapes.
      llvm::DenseMap<int32_t, unsigned> vectorsPerRow;
      bool narrowSourceRows = true;
      // Classify the row distribution of the already proven 16B vectors.
      for (uint64_t i = 0; i < count; i += 8) {
        auto begin = ll.apply({{offset, int32_t(i)}});
        if (++vectorsPerRow[begin[row].second] > 4)
          narrowSourceRows = false;
        if ((i + 8) % 512 == 0)
          vectorsPerRow.clear();
      }
      // Structured descriptors encode a byte pitch below bit14. Wider
      // pitches use an ordinary byte-bounded resource. The sentinel must stay
      // outside that resource, and every local relative offset must fit its
      // signed31-bit range before conversion to the hardware i32 offset.
      uint64_t sourceBytes = 2 * (uint64_t(rows - 1) * pitch + cols);
      auto rowRange = memory_analysis::indexBounds(op.getIndices()[row], op);
      bool fullRows = rowRange && rowRange->lo >= 0 &&
          __int128(rowRange->hi) + type.getShape()[row] <= rows;
      auto colRange = memory_analysis::indexBounds(op.getIndices()[col], op);
      bool columnInBounds = colRange && colRange->lo >= 0 &&
          __int128(colRange->hi) + type.getShape()[col] <= cols;
      bool wholeColumns = cols % type.getShape()[col] == 0 &&
          memory_analysis::multipleOf(op.getIndices()[col], type.getShape()[col]);
      // A proved complete rectangle needs no descriptor row-index clamp or
      // hardware pitch. Prefer the affine byte-bounded resource when its local
      // span fits; retain structured zero-fill for unproved row/column bounds.
      // Dimension and origin alignment make every row tile either entirely
      // valid or entirely invalid. A uniform NR0 predicate preserves zero-fill
      // without a launch bound or per-vector row-index descriptor.
      bool wholeRows = rows % type.getShape()[row] == 0 &&
          memory_analysis::multipleOf(op.getIndices()[row], type.getShape()[row]);
      bool completeRectangle = (fullRows || wholeRows) &&
          (wholeColumns || columnInBounds);
      bool structured = pitch < 0x2000 &&
          !(completeRectangle && tileSpan < (uint64_t(1) << 31));
      if (!hasAddressableMatrixCopyTile(*geometry)) {
        fail("raw local span exceeds the proven signed31-bit zero-fill domain");
        return;
      }
      MatrixSharedCopyPlan plan{row,col,waves,unsigned(count/(512*waves)),
          rows,cols,pitch,structured,wholeColumns,
          fullRows,columnInBounds,wholeRows,
          pitch < 0x2000 && (structured || narrowSourceRows),
          sourceBytes,tileSpan,{}};
      for (unsigned bit = 0; bit < ll.getInDimSizeLog2(offset); ++bit) {
        auto xy = ll.getBasis(offset,bit);
        plan.bases.push_back({xy[0],xy[1]});
      }
      plans[op] = std::move(plan);
    });
  }
  bool isValid() const { return valid; }
  const MatrixSharedCopyPlan *get(Operation *op) const {
    auto it=plans.find(op);return it==plans.end()?nullptr:&it->second;
  }
};
} // namespace mlir::triton::HCU
#endif
