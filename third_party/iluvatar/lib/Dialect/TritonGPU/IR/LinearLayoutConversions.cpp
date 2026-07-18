#include <vector>

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using mlir::triton::nvidia_gpu::TensorMemoryEncodingAttr;
using mlir::triton::nvidia_gpu::TensorMemoryScalesEncodingAttr;

namespace mlir::triton::gpu {
namespace {

// We use the following nomenclature in this file.
//
//  - ctaLayout: A layout for one block, i.e. input dims [register, lane, warp]
//    for register layouts, and input dims [offset] for shared layouts.
//  - cgaLayout: Arrangement of multiple blocks, i.e. input dims [block].
//
// Note that this is inconsistent with the type name CTAEncodingAttr. That type
// is equivalent to our cgaLayout.
//
// IMO the name CTAEncodingAttr is wrong.  If we tried to be consistent anyway,
// then we'd have to rename ctaLayout to "warpLayout".  I think that's more
// confusing than being inconsistent about "cgaLayout", especially when we have
// to consider the size of the warpLayout (surely that's not the "warpSize").

#define S(v) StringAttr::get(ctx, (v))

SmallVector<unsigned> getDefaultMmaOrder(MmaEncodingTrait layout) {
  auto rank = layout.getRepOrderForOperand(0).size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

// TODO Have order be a mandatory argument of standardOutDimNames.
SmallVector<StringAttr> permuteDimNames(const SmallVector<StringAttr> &names,
                                        const SmallVector<unsigned> &order) {
  assert(names.size() == order.size());
  SmallVector<StringAttr> ret;
  for (unsigned i : order) {
    ret.push_back(names[i]);
  }
  return ret;
}

LinearLayout swizzledSharedToLinearLayout(ArrayRef<int64_t> shape,
                                          SwizzledSharedEncodingAttr shared) {
  MLIRContext *ctx = shared.getContext();

  auto shapePerCTA = getShapePerCTA(shared, shape);

  int rank = shape.size();
  if (rank == 1) {
    return combineCtaCgaWithShape(
        LinearLayout::identity1D(shapePerCTA[0], S("offset"), S("dim0")),
        shared.getCTALayout(), shape);
  }

  auto outDimNames = standardOutDimNames(ctx, rank);

  // Construct bases for the 2 most minor dimensions of the layout.  These are
  // the dims that get swizzled.
  assert(shape.size() >= 2);
  int colDim = shared.getOrder()[0];
  int rowDim = shared.getOrder()[1];
  int numCols = shapePerCTA[colDim];
  int numRows = shapePerCTA[rowDim];
  StringAttr colDimName = outDimNames[colDim];
  StringAttr rowDimName = outDimNames[rowDim];

  std::vector<std::vector<int>> bases2D;
#ifdef __ILUVATAR__
  unsigned elemBitWidth =
      (shared.getUseTcu() && shared.getVec()) ? 512u / shared.getVec() : 0;
  // IMPORTANT: useTcu SME shared layouts are element-bit-width specific.
  //  * fp32 (rowxfb32-unsuffixed / colxfb32) is handled in the dedicated
  //    `elemBitWidth == 32` branch above.
  //  * The 16-bit branch below (fp16/bf16, rowxfb16/colxfb16) is hand-derived
  //    from the 16-bit SME hardware dump. The 2x4->4x2 block-transpose
  //    granularity and the `smeContigElems = 16` constant are specific to the
  //    rowxfb16/colxfb16 intrinsics and are NOT valid for other widths.
  //  * int8 col-major (colxfb8) is GF(2)-linear and handled in the dedicated
  //    `elemBitWidth == 8` branch below. int8 row-major (rowxfb8) is a
  //    bijection but NOT GF(2)-linear, so it cannot be represented as a
  //    LinearLayout and remains disabled in jit.py::get_corex_sme (see sme.cu /
  //    docs).
  //
  // SME rowxfb16/colxfb16 write each 16 x 64B tile with a 2x4 -> 4x2
  // block-transpose at the bf16x2/f16x2 granularity.
  //
  // The 64B row is further viewed as two 32B fp16 subgroups by the TCU path.
  // The second subgroup is swizzled with row bit2: offset bit8 maps to
  // (row bit2, col bit4), so combining it with offset bit6 XORs row bit2 back
  // to zero. This matches the row-major SME dump:
  //   col16 rows 0/1/2/3 -> shared offsets 320/321/352/353
  //   col16 rows 4/5/6/7 -> shared offsets 256/257/288/289
  //
  // Therefore the offset bit order is:
  //   bit order: row0, col[0:4], row1, row2, row3, row2^col4,
  //              col[5:], row[4:].
  if (shared.getUseTcu() && elemBitWidth == 8 && shared.getOrder()[0] == 0 &&
      numCols >= 64 && numRows >= 16) {
    // int8 col-major (colxfb8). The first 10 bases are the 16x64 hardware tile
    // dump; inter-tile bits follow lowerSmeStore's col-major placement: dim0
    // (colDim/contiguous) first, then dim1 (rowDim/strided).
    //
    // The hardcoded bases encode the full 16x64 SME tile, so they require a
    // contiguous (colDim) extent of at least 64 (col basis up to 32) and a
    // strided (rowDim) extent of at least 16 (row basis up to 8). A useTcu int8
    // shared layout is also created for non-SME (useSme=0) dot operands and for
    // tiles smaller than the SME granularity; those would emit out-of-range
    // bases here (e.g. col basis 32 with numCols==32 -> "Invalid basis 32 for
    // out-dim dim0"). Guard on the tile size and let such cases fall through to
    // the generic useTcu branch below (loop-bounded, never out-of-range).
    bases2D.push_back({0, 1});
    bases2D.push_back({0, 2});
    bases2D.push_back({1, 0});
    bases2D.push_back({2, 0});
    bases2D.push_back({0, 16});
    bases2D.push_back({0, 32});
    bases2D.push_back({0, 4});
    bases2D.push_back({0, 8});
    bases2D.push_back({4, 16});
    bases2D.push_back({8, 32});
    for (int b = 64; b < numCols; b *= 2)
      bases2D.push_back({0, b});
    for (int b = 16; b < numRows; b *= 2)
      bases2D.push_back({b, 0});
  } else if (shared.getUseTcu() && elemBitWidth == 32) {
    // fp32 SME (rowxfb32-unsuffixed / colxfb32). These bases were derived and
    // GF(2)-verified from hardware single-tile dumps composed with the
    // lowerSmeStore tile/warp placement (see sme_fp32_derive.py / docs). The
    // 16x16 fp32 tile is plain row-major; inter-tile offset bits follow the
    // row-major store order.
    if (shared.getOrder()[0] != 0) {
      // row-major: within-tile (col bits then row bits), then inter-tile col,
      // then inter-tile row.
      for (int b = 1; b < numCols && b < 16; b *= 2)
        bases2D.push_back({0, b});
      for (int b = 1; b < numRows && b < 16; b *= 2)
        bases2D.push_back({b, 0});
      for (int b = 16; b < numCols; b *= 2)
        bases2D.push_back({0, b});
      for (int b = 16; b < numRows; b *= 2)
        bases2D.push_back({b, 0});
    } else {
      // col-major (colxfb32): fixed 16x16 within-tile bases (including the
      // {4,4}/{8,8} composite swizzle), then inter-tile col, then row. The full
      // 16x16 per-CTA tile always holds for SME-eligible fp32 (M/N/K >= 32).
      // NOTE: lowerSmeStore's col-major loop advances the contiguous dim0
      // (colDim) with the lower offset bits and the strided dim1 (rowDim) with
      // the higher bits, so inter-tile col bases must precede inter-tile row.
      bases2D.push_back({1, 0});
      bases2D.push_back({2, 0});
      bases2D.push_back({0, 4});
      bases2D.push_back({0, 8});
      bases2D.push_back({0, 1});
      bases2D.push_back({0, 2});
      bases2D.push_back({4, 4});
      bases2D.push_back({8, 8});
      for (int b = 16; b < numCols; b *= 2)
        bases2D.push_back({0, b});
      for (int b = 16; b < numRows; b *= 2)
        bases2D.push_back({b, 0});
    }
  } else if (shared.getUseTcu()) {
    if (shared.getOrder()[0] != 0) {
      constexpr int smeContigElems = 16;
      int lowCols = numCols < smeContigElems ? numCols : smeContigElems;
      int lowRows = numRows < 16 ? numRows : 16;
      bases2D.push_back({1, 0});
      for (int b = 1; b < lowCols; b *= 2)
        bases2D.push_back({0, b});
      for (int b = 2; b < lowRows; b *= 2)
        bases2D.push_back({b, 0});
      if (numCols > smeContigElems)
        bases2D.push_back({numRows > 4 ? 4 : 0, smeContigElems});
      for (int b = 2 * smeContigElems; b < numCols; b *= 2)
        bases2D.push_back({0, b});
      for (int b = 16; b < numRows; b *= 2)
        bases2D.push_back({b, 0});
    } else {
      // colxfb16 is the transposed hardware form.  In this function,
      // rowDimName is order[1] and colDimName is order[0], so these bases are
      // written as {dim1, dim0}.  The two composite bases match the observed
      // col-major SME dump for logical coords value = dim1 * 32 + dim0:
      //   offset bit7 -> (dim1 bit2, dim0 bit3)
      //   offset bit8 -> (dim1 bit3, dim0 bit4)
      if (numCols > 1)
        bases2D.push_back({0, 1});
      for (int b = 1; b < numRows && b < 4; b *= 2)
        bases2D.push_back({b, 0});
      for (int b = 8; b < numCols && b < 32; b *= 2)
        bases2D.push_back({0, b});
      for (int b = 2; b < numCols && b < 8; b *= 2)
        bases2D.push_back({0, b});
      // Skinny tiles may not have the col bits used by the composite swizzle;
      // keep the corresponding row bits so the layout remains surjective.
      if (numRows > 4 && numCols > 8)
        bases2D.push_back({4, 8});
      else if (numRows > 4)
        bases2D.push_back({4, 0});
      if (numRows > 8 && numCols > 16)
        bases2D.push_back({8, 16});
      else if (numRows > 8)
        bases2D.push_back({8, 0});
      for (int b = 32; b < numCols; b *= 2)
        bases2D.push_back({0, b});
      for (int b = 16; b < numRows; b *= 2)
        bases2D.push_back({b, 0});
    }
  } else
#endif
  {
    for (int col = 1; col < numCols; col *= 2) {
      bases2D.push_back({0, col});
    }
    for (int row = 1; row < numRows; row *= 2) {
      int vec = shared.getVec();
      int perPhase = shared.getPerPhase();
      int maxPhase = shared.getMaxPhase();
      bases2D.push_back({row, (vec * ((row / perPhase) % maxPhase)) % numCols});
    }
  }
  LinearLayout ctaLayout =
      LinearLayout({{S("offset"), bases2D}}, {rowDimName, colDimName});

  // Add the remaining dimensions.
  for (int i = 2; i < rank; i++) {
    int dim = shared.getOrder()[i];
    ctaLayout *= LinearLayout::identity1D(shapePerCTA[dim], S("offset"),
                                          outDimNames[dim]);
  }

  return combineCtaCgaWithShape(ctaLayout, shared.getCTALayout(), shape);
}

} // namespace

// Returns the layout of a single core matrix which tiles the nvmma layout
LinearLayout getCoreMatrixLinearLayout(NVMMASharedEncodingAttr shared,
                                       bool disableSwizzle) {
  auto *ctx = shared.getContext();

  int elemBitWidth = shared.getElementBitWidth();
  int tileWidthBytes = shared.getSwizzlingByteWidth();
  int vec = shared.getVec();
  int perPhase = shared.getPerPhase();
  int maxPhase = shared.getMaxPhase();

  int tileRows = 8;
  int tileCols = 8 * std::max(16, tileWidthBytes) / elemBitWidth;
  bool isFp4Padded = shared.getFp4Padded();

  std::vector<std::vector<int>> bases2D;
  for (int col = 1; col < tileCols; col *= 2) {
    if (isFp4Padded) {
      // Each group of 16 offsets consists of 8 "real" and 8 "padded" offsets.
      // We represent the padded layout by mapping 8 padded offsets to the same
      // coordinates as the real ones. When computing the inverse of this LL,
      // the offsets correspoding to the real ones are picked in the image by
      // invertAndCompose.
      int colPacked = col / 16 * 8 + col % 8;
      bases2D.push_back({0, colPacked});
    } else {
      bases2D.push_back({0, col});
    }
  }
  for (int row = 1; row < tileRows; row *= 2) {
    if (disableSwizzle) {
      bases2D.push_back({row, 0});
    } else if (isFp4Padded) {
      int colPadded = vec * ((row / perPhase) % maxPhase);
      int colPacked = colPadded / 16 * 8 + colPadded % 8;
      bases2D.push_back({row, colPacked});
    } else {
      bases2D.push_back({row, vec * ((row / perPhase) % maxPhase)});
    }
  }
  auto outDimNames = standardOutDimNames(ctx, 2);
  return LinearLayout({{S("offset"), bases2D}}, outDimNames);
}

LinearLayout nvmmaSharedToLinearLayout(ArrayRef<int64_t> shape,
                                       NVMMASharedEncodingAttr shared,
                                       bool disableSwizzle) {
  MLIRContext *ctx = shared.getContext();
  int rank = shape.size();
  auto shapePerCTA = getShapePerCTA(shared, shape);
  auto kOffset = S("offset");
  auto tmaShape = triton::nvidia_gpu::getTMABlockShape(shared, shapePerCTA,
                                                       /*packedSize=*/true);
  if (shared.getSwizzlingByteWidth() == 0) {
    auto outDimNames = standardOutDimNames(ctx, rank);
    LinearLayout layout = LinearLayout::identity1D(tmaShape[rank - 1], kOffset,
                                                   outDimNames[rank - 1]);
    for (int i = rank - 2; i >= 0; --i) {
      layout *= LinearLayout::identity1D(tmaShape[i], kOffset, outDimNames[i]);
    }
    layout = ensureLayoutNotSmallerThan(layout, outDimNames, shapePerCTA);
    return combineCtaCgaWithShape(layout, shared.getCTALayout(), shape);
  }
  assert(rank >= 2);

  // Collapse all the outer dim into one. We will then create a layout for this
  // shape and reshape it to the original shape.
  std::array<int64_t, 2> collapsedTmaShape{1, tmaShape.back()};
  for (int i = 0; i + 1 < rank; i++)
    collapsedTmaShape[0] *= tmaShape[i];
  if (shared.getTransposed()) {
    std::swap(collapsedTmaShape[0], collapsedTmaShape[1]);
  }

  auto tileLayout = getCoreMatrixLinearLayout(shared, disableSwizzle);
  auto outDimNames = standardOutDimNames(ctx, 2);
  auto kRow = outDimNames[0];
  auto kCol = outDimNames[1];
  auto tileRows = tileLayout.getOutDimSize(kRow);
  auto tileCols = tileLayout.getOutDimSize(kCol);

  int packingFactor = shared.getFp4Padded() ? 2 : 1;
  if (collapsedTmaShape[1] * packingFactor < tileCols ||
      collapsedTmaShape[0] < tileRows) {
    llvm::errs() << "Illegal shared layout; expected collapsed shapePerCTA to "
                    "be at least ["
                 << tileRows << ", " << (tileCols / packingFactor)
                 << "], collapsedTmaShape: [" << collapsedTmaShape[0] << ", "
                 << collapsedTmaShape[1] << "]\n";
    llvm::report_fatal_error("Illegal shared layout");
  }

  // Distribute the remaining rows and cols.
  auto layout =
      ensureLayoutNotSmallerThan(tileLayout, outDimNames, collapsedTmaShape);

  // Reshape the layout to the N-D pre-transposed shape per CTA.
  SmallVector<int64_t> maybeTransposedTmaShape = tmaShape;
  if (shared.getTransposed()) {
    // Move the outer dim to the inner position.
    // TODO: we should move back to using `order` instead of transposed to make
    // the order more explicit.
    std::rotate(maybeTransposedTmaShape.begin(),
                maybeTransposedTmaShape.begin() + 1,
                maybeTransposedTmaShape.end());
  }
  auto reshapedLayout = reshapeLayout(ctx, layout, maybeTransposedTmaShape);

  if (shared.getTransposed()) {
    SmallVector<int> order = {rank - 1};
    for (int i = 0; i < rank - 1; i++) {
      order.push_back(i);
    }
    reshapedLayout = transposeLinearLayout(reshapedLayout, order);
  }

  reshapedLayout = ensureLayoutNotSmallerThan(
      reshapedLayout, standardOutDimNames(ctx, shapePerCTA.size()),
      shapePerCTA);
  return combineCtaCgaWithShape(reshapedLayout, shared.getCTALayout(), shape);
}

/// Function to generate lane and warp layout for dot operands.
static LinearLayout broadcastedDotOperandLayout(MLIRContext *ctx,
                                                ArrayRef<unsigned> shape,
                                                ArrayRef<unsigned> order,
                                                unsigned kDim,
                                                StringAttr inDimName) {
  // Let warpsPerCTAMma = {2, 2}, then
  // warpsPerCTA = {2, 1} for opA and warpsPerCTA = {1, 2} for opB
  // assume warpOrder = {1, 0}
  // Assume that C is tiled by 2x2 tiles. Since warpOrder={1, 0}, we have that
  // the C is owned as per the following layout:
  // C: 0 | 1
  //    - | -
  //    2 | 3
  // In order to be able to compute C, we need the following warp tiling of
  // A and B:
  // A: 0 1 | 0 1    B: 0 2 | 1 3
  //    - - | - -       - - | - -
  //    2 3 | 2 3       0 2 | 1 3
  // In other words, we need to broadcast along K
  auto rank = shape.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  LinearLayout layout = LinearLayout::empty();

  // We have to broadcast along the inner dimension
  // For A, when moving along M we go from 0 to 2.
  // For B, when moving along N we go from 0 to 1.
  // As such, choosing the order of A {1, 0}, gives us the correct broadcasting
  // Same happens if the warpOrder is {0, 1}, like in Hopper
  for (auto d : order) {
    if (d == kDim) {
      layout *= LinearLayout::zeros1D(shape[d], inDimName, dimNames[d]);
    } else {
      layout *= LinearLayout::identity1D(shape[d], inDimName, dimNames[d]);
    }
  }
  return layout;
}

LinearLayout
BlockedEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();
  auto order = getOrder();
  LinearLayout ctaLayout =
      identityStandardND(S("register"), getSizePerThread(), order) *
      identityStandardND(S("lane"), getThreadsPerWarp(), order) *
      identityStandardND(S("warp"), getWarpsPerCTA(), order);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout fmaDotToLinearLayout(DotOperandEncodingAttr operandLayout,
                                  ArrayRef<int64_t> shape) {
  int rank = shape.size();
  auto blocked = cast<BlockedEncodingAttr>(operandLayout.getParent());
  MLIRContext *ctx = operandLayout.getContext();

  // TODO: introduce registerOrder or use getDefaultOrder(operandLayout)
  // Currently this order is used in legacy converter, because we do not
  // have access to full dot operand layout, only parent part.
  auto regOrder = blocked.getOrder();
  auto threadOrder = blocked.getOrder();
  auto warpOrder = blocked.getOrder();
  auto repOrder = blocked.getRepOrder();

  StringAttr kReg = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  auto threadSize = llvm::to_vector(blocked.getSizePerThread());
  auto kDimIdx = operandLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;
  threadSize[kDimIdx] = shape[kDimIdx];
  auto threadShape = blocked.getThreadsPerWarp();
  auto warpShape = blocked.getWarpsPerCTA();

  SmallVector<StringAttr> repDimNames =
      permuteDimNames(standardOutDimNames(ctx, rank), repOrder);

  auto registersLayout = identityStandardND(kReg, threadSize, regOrder);
  auto lanesLayout = broadcastedDotOperandLayout(ctx, threadShape, threadOrder,
                                                 kDimIdx, kLane);
  auto warpsLayout =
      broadcastedDotOperandLayout(ctx, warpShape, warpOrder, kDimIdx, kWarp);

  LinearLayout ctaLayout = registersLayout.transposeOuts(repDimNames) *
                           lanesLayout.transposeOuts(repDimNames) *
                           warpsLayout.transposeOuts(repDimNames);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(operandLayout), shape);
}

LinearLayout nvidiaMmaTile(MLIRContext *ctx, ArrayRef<unsigned> tileShape,
                           unsigned kWidth, ArrayRef<unsigned> order,
                           ArrayRef<unsigned> repOrder) {
  // Trivial layout mapping 0 -> (0, 0), but we set the order to repOrder
  // Like LinearLayout::empty() but with a rank and an order
  int rank = repOrder.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  auto trivialShape = SmallVector<unsigned>(rank, 1);
  LinearLayout ctaLayout =
      identityStandardND(S("register"), trivialShape, repOrder);

  assert(rank >= 2);
  auto inner = order[0];
  auto outer = order[1];

  assert(tileShape.size() == rank);
  int m = tileShape[outer];
  int n = tileShape[inner];

  // The relative order of registers and lanes is given by:
  // - Inner dim: kWidth registers
  // - Inner dim: 4 lanes
  // - Outer dim: 8 lanes
  // - Outer dim: repeat m / 8 times
  // - Inner dim: repeat n / (kWidth * 4) times
  assert(m % 8 == 0);
  assert(n % (kWidth * 4) == 0);
  // There is at least one subtile on the inner-most dimension
  // FIXME. We should implement operator* in terms of operator*=
  // and chain *= instead of using *
  auto outDimNames = llvm::to_vector(ctaLayout.getOutDimNames());
  ctaLayout = ctaLayout *
              LinearLayout::identity1D(kWidth, S("register"), dimNames[inner]) *
              LinearLayout::identity1D(4, S("lane"), dimNames[inner]) *
              LinearLayout::identity1D(8, S("lane"), dimNames[outer]) *
              LinearLayout::identity1D(m / 8, S("register"), dimNames[outer]) *
              LinearLayout::identity1D(n / (kWidth * 4), S("register"),
                                       dimNames[inner]);
  return ctaLayout;
}

LinearLayout
NvidiaMmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ctx = getContext();
  int rank = shape.size();
  assert(rank == getRank());

  SmallVector<unsigned> tileShape;
  if (isAmpere()) {
    // Ampere.getInstrShape() returns the tile shape
    tileShape = SmallVector<unsigned>(getInstrShape());
  } else {
    assert(isHopper());
    auto instrShapeMNK = getInstrShape();
    tileShape = SmallVector<unsigned>({instrShapeMNK[0], instrShapeMNK[1]});
  }
  // nvidiamma layout always assumes kWidth = 2
  constexpr auto kWidth = 2;
  auto order = getDefaultMmaOrder(*this);
  auto ctaLayout = nvidiaMmaTile(ctx, tileShape, kWidth, order, getRepOrder());

  auto warpOrder = getMatrixOrder(rank, /*rowMajor*/ !isHopper());
  ctaLayout *= identityStandardND(S("warp"), getWarpsPerCTA(), warpOrder)
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout nvidiaDotToLinearLayout(ArrayRef<int64_t> shape,
                                     DotOperandEncodingAttr dot) {
  int rank = shape.size();
  auto mma = cast<NvidiaMmaEncodingAttr>(dot.getParent());
  int kWidth = dot.getKWidth();
  bool isA = dot.getOpIdx() == 0;
  MLIRContext *ctx = mma.getContext();

  SmallVector<unsigned> tileShape(rank, 1);
  if (isA) {
    tileShape[rank - 2] = 16;
    tileShape[rank - 1] = kWidth * 8;
  } else {
    // Hopper takes the rhs via shared memory
    assert(mma.isAmpere());
    tileShape[rank - 2] = kWidth * 8;
    tileShape[rank - 1] = 8;
  }
  auto order = getOrderForDotOperand(dot.getOpIdx(), rank, /*kContig*/ true);
  auto ctaLayout =
      nvidiaMmaTile(ctx, tileShape, kWidth, order, dot.getRepOrder());
  auto kDim = isA ? rank - 1 : rank - 2;
  auto warpOrder = getMatrixOrder(rank, /*rowMajor*/ !mma.isHopper());
  ctaLayout *= broadcastedDotOperandLayout(ctx, mma.getWarpsPerCTA(), warpOrder,
                                           kDim, S("warp"))
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(dot), shape);
}

#ifdef __ILUVATAR__
static LinearLayout iluvatarMmaTile(MLIRContext *ctx, StringAttr rowDim,
                                    StringAttr colDim) {
  // Iluvatar TCU results map lane bits to contiguous columns first, then to the
  // low row offsets. Register bits hold high row offsets.
  return LinearLayout(
      {{S("register"), {{4, 0}, {8, 0}}},
       {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}}}},
      {rowDim, colDim});
}

static LinearLayout iluvatarDotTile(MLIRContext *ctx,
                                    DotOperandEncodingAttr dot,
                                    StringAttr rowDim, StringAttr colDim) {
  if (dot.getKWidth() == 1) {
    return iluvatarMmaTile(ctx, rowDim, colDim);
  } else if (dot.getKWidth() == 4) {
    if (dot.getOpIdx() == 0)
      // ALayout: thread strides (16, 4), value strides (1, 256) on an MxK
      // tile.
      return LinearLayout(
          {{S("register"), {{1, 0}, {2, 0}, {0, 16}}},
           {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}}},
          {rowDim, colDim});
    // BLayout: thread strides (1, 64), value strides (16, 256) on a KxN tile.
    return LinearLayout(
        {{S("register"), {{1, 0}, {2, 0}, {16, 0}}},
         {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {4, 0}, {8, 0}}}},
        {rowDim, colDim});
  } else if (dot.getKWidth() == 2) {
    return LinearLayout(
        {{S("register"), {{1, 0}, {8, 0}}},
         {S("lane"), {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {2, 0}, {4, 0}}}},
        {rowDim, colDim});
  } else {
    assert(false && "unsupported Iluvatar TCU dot operand kWidth");
    return LinearLayout::empty();
  }
}

LinearLayout
IluvatarMmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ctx = getContext();
  int rank = shape.size();
  assert(rank == getRank());

  auto dimNames = standardOutDimNames(ctx, rank);
  auto dimM = dimNames[rank - 2];
  auto dimN = dimNames[rank - 1];

  // Iluvatar TCU maps each warp to a 16x16 result tile. The f32x4 result
  // vector uses register bits for the high M offsets, while consecutive lane
  // bits first cover N and then the low M offsets.
  LinearLayout ctaLayout = iluvatarMmaTile(ctx, dimM, dimN);

  auto warpOrder = getDefaultMmaOrder(*this);
  ctaLayout *= identityStandardND(S("warp"), getWarpsPerCTA(), warpOrder)
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout iluvatarDotToLinearLayout(ArrayRef<int64_t> shape,
                                       DotOperandEncodingAttr dot) {
  int rank = shape.size();
  auto mma = cast<IluvatarMmaEncodingAttr>(dot.getParent());
  MLIRContext *ctx = mma.getContext();

  SmallVector<StringAttr> dimNames = standardOutDimNames(ctx, rank);

  auto order = getOrderForDotOperand(dot.getOpIdx(), rank, /*kContig*/ true);
  auto dimK = dimNames[order[0]];
  auto dimNonK = dimNames[order[1]];
  // TCU dot operands A and B use the same per-warp row/column layout shown in
  // the hardware diagram. A is shaped as (M, K), while B is shaped as (K, N),
  // so the same base vectors are attached to different logical dimensions.
  auto rowDim = dot.getOpIdx() == 0 ? dimNonK : dimK;
  auto colDim = dot.getOpIdx() == 0 ? dimK : dimNonK;
  // kWidth is derived from the operand dtype: fp32 uses 1, while fp16/bf16 use
  // 2 and keep the existing packed dot operand layout.
  LinearLayout ctaLayout = iluvatarDotTile(ctx, dot, rowDim, colDim);

  auto repOrder = mma.getRepOrderForOperand(dot.getOpIdx());
  SmallVector<StringAttr> repDimNames;
  for (auto dim : repOrder)
    repDimNames.push_back(dimNames[dim]);
  ctaLayout = ctaLayout.transposeOuts(repDimNames);

  auto kDim = dot.getOpIdx() == 0 ? rank - 1 : rank - 2;
  auto warpOrder = getDefaultMmaOrder(mma);
  ctaLayout *= broadcastedDotOperandLayout(ctx, mma.getWarpsPerCTA(), warpOrder,
                                           kDim, S("warp"))
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(dot), shape);
}
#endif

LinearLayout
DotOperandEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto parent = getParent();
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(parent)) {
    return fmaDotToLinearLayout(*this, shape);
#ifdef __ILUVATAR__
  } else if (mlir::isa<IluvatarMmaEncodingAttr>(parent)) {
    return iluvatarDotToLinearLayout(shape, *this);
#endif
  } else {
    auto mma = mlir::cast<NvidiaMmaEncodingAttr>(parent);
    return nvidiaDotToLinearLayout(shape, *this);
  }
}

LinearLayout SliceEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();

  // First compute the linear layout for this layout's parent.
  SmallVector<int64_t> parentShape(shape);
  parentShape.insert(parentShape.begin() + getDim(), 1);
  LinearLayout parentLL = triton::gpu::toLinearLayout(parentShape, getParent());

  auto sliceLL = removeStandardDim(parentLL, getDim());

  // Step 3: Along the "register" dim, remove any all-zero bases.
  auto bases = sliceLL.getBases();
  std::vector<std::vector<int>> newRegBases;
  for (const auto &basis : bases[S("register")]) {
    if (llvm::any_of(basis, [](int b) { return b != 0; })) {
      newRegBases.push_back(basis);
    }
  }
  bases[S("register")] = newRegBases;

  return LinearLayout(std::move(bases),
                      llvm::to_vector(sliceLL.getOutDimNames()));
}

LinearLayout tensorMemoryToLinearLayout(ArrayRef<int64_t> shape,
                                        TensorMemoryEncodingAttr encoding) {
  // [Zeros in TMEM LinearLayouts]
  // If there is a zero in bases rows=32,64 this means that there is
  // broadcasting, i.e. the same tensor element is duplicated in different
  // addressable blocks If the zero is in any other row/col (i.e. within a given
  // warp-addressable tmem space) it means it is not defined

  // We model packed layouts as having the rows/cols dimensions of bitWidth=16
  // This means that a layout with unpacked=True is the same as one with
  // unpacked=False
  assert(shape.size() == 2);
  auto *ctx = encoding.getContext();
  auto kRow = S("row");
  auto kCol = S("col");
  auto dims = standardOutDimNames(ctx, 2);
  // The CTAOrder = [0, 1] so se start by N so that it ends up as
  // ((tile * splitM) * splitN)
  if (encoding.getCTASplitN() > 1) {
    auto split =
        LinearLayout::identity1D(encoding.getCTASplitN(), kCol, dims[1]);
    auto newEncoding = TensorMemoryEncodingAttr::get(
        ctx, encoding.getBlockM(), encoding.getBlockN(),
        encoding.getColStride(), encoding.getCTASplitM(), 1,
        encoding.getTwoCTAs());
    return tensorMemoryToLinearLayout(
               {shape[0], shape[1] / encoding.getCTASplitN()}, newEncoding) *
           split;
  }
  if (encoding.getCTASplitM() > 1) {
    auto splitM = encoding.getCTASplitM();
    auto blockM = encoding.getBlockM();
    bool isM64TwoCTA = blockM == 64 && encoding.getTwoCTAs();
    if (isM64TwoCTA) {
      // blockM == 64 and twoCTAs is laid out as the transpose of 128xblockN
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
      blockM *= 2;
      splitM /= 2;
    }
    auto split = LinearLayout::identity1D(splitM, kCol, dims[0]);
    auto newEncoding = TensorMemoryEncodingAttr::get(
        ctx, blockM, encoding.getBlockN(), encoding.getColStride(), 1,
        encoding.getCTASplitN(), encoding.getTwoCTAs());
    auto ret =
        tensorMemoryToLinearLayout({shape[0] / splitM, shape[1]}, newEncoding) *
        split;
    // In this case, we swap the basis of the last row and last column as per
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-bny
    if (isM64TwoCTA) {
      auto bases = ret.getBases();
      auto &rowBases = bases[kRow];
      auto &colBases = bases[kCol];
      std::swap(rowBases[rowBases.size() - 1], colBases[colBases.size() - 1]);
      ret = LinearLayout(bases, ret.getOutDims(), ret.isSurjective());
    }
    return ret;
  }
  assert(encoding.getCTASplitM() == 1 && encoding.getCTASplitN() == 1);

  auto blockM = encoding.getBlockM();
  auto blockN = std::min<int32_t>(encoding.getBlockN(), shape[1]);
  assert(blockM == 64 || blockM == 128);
  LinearLayout tile =
      LinearLayout::zeros1D(encoding.getColStride(), kCol, dims[1]);
  if (blockM == 64) {
    tile *= LinearLayout::identity1D(16, kRow, dims[0]) *
            LinearLayout::identity1D(blockN, kCol, dims[1]);
    auto bases = tile.getBases();
    if (shape[0] > blockM) {
      bases[kRow].push_back({64, 0});
    } else if (shape[1] > blockN) {
      bases[kRow].push_back({0, blockN});
    } else {
      // Empty, meaning the element is not defined
      bases[kRow].push_back({0, 0});
    }
    bases[kRow].push_back({16, 0});
    bases[kRow].push_back({32, 0});
    tile = LinearLayout(bases, dims);
  } else {
    tile *= LinearLayout::identity1D(blockM, kRow, dims[0]) *
            LinearLayout::identity1D(blockN, kCol, dims[1]);
  }
  auto repsM = shape[0] / tile.getOutDimSize(dims[0]);
  auto repsN = shape[1] / tile.getOutDimSize(dims[1]);
  assert(repsM >= 1 && repsN >= 1);
  // Broadcast the remaining dimensions in order [0, 1]
  tile = tile * LinearLayout::identity1D(repsM, kCol, dims[0]) *
         LinearLayout::identity1D(repsN, kCol, dims[1]);
  return tile;
}

LinearLayout
tensorMemoryScalesToLinearLayout(ArrayRef<int64_t> shape,
                                 TensorMemoryScalesEncodingAttr encoding) {
  assert(shape.size() == 2);
  auto *ctx = encoding.getContext();
  auto kRow = S("row");
  auto kCol = S("col");
  auto dims = standardOutDimNames(ctx, 2);

  // The CTAOrder = [0, 1] so se start by N so that it ends up as
  // ((tile * splitM) * splitN)
  if (encoding.getCTASplitN() > 1) {
    auto split =
        LinearLayout::identity1D(encoding.getCTASplitN(), kCol, dims[1]);
    auto newEncoding =
        TensorMemoryScalesEncodingAttr::get(ctx, encoding.getCTASplitM(), 1);
    return tensorMemoryScalesToLinearLayout(
               {shape[0], shape[1] / encoding.getCTASplitN()}, newEncoding) *
           split;
  }
  if (encoding.getCTASplitM() > 1) {
    auto split =
        LinearLayout::identity1D(encoding.getCTASplitM(), kCol, dims[0]);
    auto newEncoding =
        TensorMemoryScalesEncodingAttr::get(ctx, 1, encoding.getCTASplitN());
    return tensorMemoryScalesToLinearLayout(
               {shape[0] / encoding.getCTASplitM(), shape[1]}, newEncoding) *
           split;
  }
  assert(encoding.getCTASplitM() == 1 && encoding.getCTASplitN() == 1);

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
  auto tile = LinearLayout::identity1D(32, kRow, dims[0]) *
              // Broadcasting along 'warps'
              LinearLayout::zeros1D(4, kRow, dims[0]) *
              LinearLayout::identity1D(4, kCol, dims[1]) *
              LinearLayout::identity1D(2, kCol, dims[0]);
  // We choose repOrder = [0, 1]
  tile *= LinearLayout::identity1D(
              llvm::divideCeil(shape[0], tile.getOutDimSize(dims[0])), kCol,
              dims[0]) *
          LinearLayout::identity1D(
              llvm::divideCeil(shape[1], tile.getOutDimSize(dims[1])), kCol,
              dims[1]);
  // See [Zeros in TMEM LinearLayouts]
  // Set some rows/cols to 0 if shape is smaller than 64 x 4
  llvm::SmallDenseMap<StringAttr, int64_t> shapeMap;
  for (auto [dim, size] : llvm::zip(dims, shape)) {
    shapeMap[dim] = size;
  }
  return ensureLayoutNotLargerThan(tile, shapeMap);
}

LinearLayout TritonGPUDialect::toLinearLayout(ArrayRef<int64_t> shape,
                                              Attribute layout) {
  CacheKey key{std::vector<int64_t>(shape.begin(), shape.end()), layout};
  if (auto result = llCache.get(key)) {
    return *result;
  }

  // Layouts are distributed or shared in triton core
  // To add a new layout add an else-if clause
  LinearLayout result = LinearLayout::empty();
  if (auto distributed = dyn_cast<DistributedEncodingTrait>(layout)) {
    result = distributed.toLinearLayout(shape);
  } else {
    assert(llvm::all_of(shape,
                        [](int64_t dim) {
                          return llvm::isPowerOf2_32(dim) && dim >= 1;
                        }) &&
           "shape must be a postive power of 2");
    if (auto shared = dyn_cast<SwizzledSharedEncodingAttr>(layout)) {
      result = swizzledSharedToLinearLayout(shape, shared);
    } else if (auto shared = dyn_cast<SharedLinearEncodingAttr>(layout)) {
      result = shared.toLinearLayout(shape);
    } else if (auto shared = dyn_cast<NVMMASharedEncodingAttr>(layout)) {
      result = nvmmaSharedToLinearLayout(shape, shared);
    } else if (auto tensorMemoryEncoding =
                   dyn_cast<TensorMemoryEncodingAttr>(layout)) {
      result = tensorMemoryToLinearLayout(shape, tensorMemoryEncoding);
    } else if (auto tensorMemoryScalesEncoding =
                   dyn_cast<TensorMemoryScalesEncodingAttr>(layout)) {
      result =
          tensorMemoryScalesToLinearLayout(shape, tensorMemoryScalesEncoding);
    } else {
      assert(0 && "unknown layout");
    }
  }

  llCache.set(std::move(key), result);
  return result;
}

LinearLayout toLinearLayout(RankedTensorType type) {
  return toLinearLayout(type.getShape(), type.getEncoding());
}

LinearLayout toLinearLayout(MemDescType type) {
  // Pass in the allocation shape. Then when using invertAndCompose it will
  // trim the allocationShape to the shape if they are different.
  // We also remove the first dimension of the allocationShape if there was a
  // call to memdesc_index
  auto shape = type.getAllocShape().take_back(type.getRank());
  return toLinearLayout(shape, type.getEncoding());
}

LinearLayout toLinearLayout(TensorOrMemDesc type) {
  if (auto ranked = dyn_cast<RankedTensorType>(type)) {
    return toLinearLayout(ranked);
  } else {
    auto memDesc = cast<MemDescType>(type);
    return toLinearLayout(memDesc);
  }
}

// UNSAFE OVERLOAD!
// If you call this with a SharedMemoryEncodingAttr, you should call it
// with the allocShape as the shape, otherwise the layout will be incorrect!
LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout) {
  auto *ctx = layout.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearLayout(shape,
                                                                   layout);
}

LinearLayout getLayoutWithinBlock(const LinearLayout &layout) {
  assert(!layout.getInDimNames().empty());
  MLIRContext *ctx = layout.getInDimNames().begin()->getContext();

  StringAttr kBlock = S("block");
  assert(layout.hasInDim(kBlock));
  auto bases = layout.getBases();
  bases[kBlock] = {};
  return LinearLayout(bases, llvm::to_vector<4>(layout.getOutDimNames()));
}

LinearLayout combineCtaCgaWithShape(LinearLayout ctaLayout,
                                    CTAEncodingAttr cgaLayoutAttr,
                                    ArrayRef<int64_t> shape) {
  int rank = shape.size();
  assert(ctaLayout.getNumOutDims() == rank);
  assert(cgaLayoutAttr.getCTAOrder().size() == rank);
  MLIRContext *ctx = cgaLayoutAttr.getContext();

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  llvm::SmallDenseMap<StringAttr, int64_t> labeledShape;
  for (auto [dim, size] : llvm::zip(outDimNames, shape)) {
    labeledShape[dim] = size;
  }

  LinearLayout cgaLayout =
      ensureLayoutNotLargerThan(cgaLayoutAttr.getLinearLayout(), labeledShape)
          .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  // Calculate the shape of the ctaLayout, which is `shape` divided by the
  // cgaLayout's size.
  llvm::SmallDenseMap<StringAttr, int64_t> ctaShape;
  assert(llvm::to_vector(ctaLayout.getOutDimNames()) ==
         llvm::to_vector(cgaLayout.getOutDimNames()));
  for (auto dim : ctaLayout.getOutDimNames()) {
    ctaShape[dim] =
        std::max(int64_t{1}, labeledShape[dim] / cgaLayout.getOutDimSize(dim));
  }

  ctaLayout = ensureLayoutNotSmallerThan(ctaLayout, ctaShape);
  ctaLayout = ensureLayoutNotLargerThan(ctaLayout, ctaShape);

  LinearLayout ret = (ctaLayout * cgaLayout).transposeOuts(outDimNames);
  for (auto dim : ret.getOutDimNames()) {
    assert(ret.getOutDimSize(dim) == labeledShape[dim]);
  }
  return ret;
}

LinearLayout chooseShemLayoutForRegToRegConversion(
    MLIRContext *ctx, ArrayRef<unsigned> tensorShape,
    ArrayRef<unsigned> repShape, ArrayRef<unsigned> order) {
  auto outDimNames = standardOutDimNames(ctx, tensorShape.size());
  LinearLayout layout = LinearLayout::empty();
  SmallVector<StringAttr> kRepDims;
  SmallVector<StringAttr> kOffsetDims;
  auto totalIters = 1;
  auto totalOffsets = 1;
  for (int i = 0; i < tensorShape.size(); i++) {
    int dim = order[i];
    StringAttr kIteration = S("iteration" + std::to_string(dim));
    StringAttr kOffset = S("offset" + std::to_string(dim));
    kRepDims.push_back(kIteration);
    kOffsetDims.push_back(kOffset);
    assert(llvm::isPowerOf2_32(repShape[dim]));
    assert(llvm::isPowerOf2_32(tensorShape[dim]));
    auto numIters = tensorShape[dim] / repShape[dim];
    layout *=
        LinearLayout::identity1D(repShape[dim], kOffset, outDimNames[dim]);
    layout *= LinearLayout::identity1D(numIters, kIteration, outDimNames[dim]);
    totalIters *= numIters;
    totalOffsets *= repShape[dim];
  }
  StringAttr kOffset = S("offset");
  StringAttr kIteration = S("iteration");
  StringAttr kBlock = S("block");
  SmallVector<StringAttr> newDims;
  newDims.append(kOffsetDims.begin(), kOffsetDims.end());
  newDims.append(kRepDims.begin(), kRepDims.end());
  // Transpose layout from [offset0, rep0, offset1, rep1, ...] to
  // [offset0, offset1, ..., rep0, rep1, ...]
  auto ret = layout.transposeIns(newDims);
  // Reshape layout from [offset0, offset1, ..., rep0, rep1, ...] to
  // [offset, rep, block]
  return ret.reshapeIns(
      {{kOffset, totalOffsets}, {kIteration, totalIters}, {kBlock, 1}});
}

LinearLayout chooseScaledWmmaScaleLayout(MLIRContext *ctx, int dotOperandIdx,
                                         ArrayRef<int64_t> dotOperandShape,
                                         unsigned wmmaMDim,
                                         ArrayRef<unsigned> tilesPerWarp,
                                         ArrayRef<unsigned> warpsPerCTA) {
  using basisT = std::vector<std::vector<int32_t>>;
  unsigned rank = dotOperandShape.size();
  auto order = mlir::triton::gpu::getMatrixOrder(rank, /*rowMajor=*/true);
  auto outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  // In scaled dot, the shapes of operands(without batch dimension) are,
  // respectively:
  // - A: [M, K]
  // - B: [K, N]
  // - aScale: [M, K / 32 or 16]
  // - bScale: [N, K / 32 or 16]
  auto dimK = outDimNames[order[0]];
  auto dimNonK = outDimNames[order[1]];

  // Each lane holds kWidth=4 consecutive values along the K dim.
  // The first 16 lanes are distributed along the nonK dim.
  unsigned scaleKWidth = 4;
  auto kSize = dotOperandShape[1];
  LinearLayout tileLayout =
      LinearLayout::identity1D(scaleKWidth, kRegister, dimK) *
      LinearLayout::identity1D(16, kLane, dimNonK);

  // If there's 1 tile per warp, we are not using the remaining 16 lanes, so
  // just let them duplicate values of the first 16 lanes.
  // Otherwise, we put consecutive values along the nonK dim in the remaining
  // 16 lanes.
  unsigned mnDim = dotOperandIdx == 0 ? rank - 2 : rank - 1;
  unsigned tilePerWarpMN = tilesPerWarp[mnDim];
  if (tilePerWarpMN > 1) {
    assert(tilePerWarpMN == 2 && "TilesPerWarp > 2 is not supported.");
    tileLayout *= LinearLayout::identity1D(tilePerWarpMN, kLane, dimNonK);
  } else {
    tileLayout *= LinearLayout::zeros1D(2, kLane, dimNonK);
  }

  // If the shape along the K dim is larger than kWidth, repeat this
  // pattern to fill the K dim.
  tileLayout *= LinearLayout::identity1D(kSize / scaleKWidth, kRegister, dimK);

  auto warpsPerCTANew = (dotOperandIdx == 1)
                            ? SmallVector{warpsPerCTA[1], warpsPerCTA[0]}
                            : SmallVector{warpsPerCTA[0], warpsPerCTA[1]};

  auto warpOrder = (dotOperandIdx == 1) ? SmallVector<unsigned>{0, 1}
                                        : SmallVector<unsigned>{1, 0};
  LinearLayout warpLayout =
      identityStandardND(kWarp, warpsPerCTANew, warpOrder);
  LinearLayout ctaLayout = tileLayout.transposeOuts(outDimNames) *
                           warpLayout.transposeOuts(outDimNames);

  return combineCtaCgaWithShape(
      ctaLayout, CTAEncodingAttr::getDefault(ctx, /*rank=*/2), dotOperandShape);
}

// PTX ISA - Warp-level MMA Block Scaling
//   https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
// This function generates layouts for scale tensors used in scaled dot
// operations.
// Implementation notes:
//   - We choose a fixed provider for A (thread-id-a = 0) and B (thread-id-b =
//   0)
//   - We choose a fixed byte selector for A (byte-id-a = 0) and B (byte-id-b =
//   0)
//   - Each lane in a quad has the same scale factor.
LinearLayout getSM120DotScaledScaleLayout(MLIRContext *ctx,
                                          ArrayRef<int64_t> shape, int opIdx,
                                          ArrayRef<unsigned> warpsPerCTA,
                                          CTAEncodingAttr ctaLayout) {
  unsigned rank = shape.size();
  auto outDims = standardOutDimNames(ctx, rank);
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  // - A: [M, K]
  // - B: [K, N]
  // - aScale: [M, K / K_GROUP_SIZE]
  // - bScale: [N, K / K_GROUP_SIZE]
  const unsigned kIdx = 1;
  const unsigned mnIdx = 0;

  std::vector<std::vector<int32_t>> laneBase;
  SmallVector<unsigned> order;
  SmallVector<unsigned> mmaWarpsPerCTA;
  if (opIdx == 0) {
    laneBase = {{8, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}};
    order = SmallVector<unsigned>{1u, 0u};
    mmaWarpsPerCTA = SmallVector<unsigned>{warpsPerCTA[0], warpsPerCTA[1]};
  } else {
    laneBase = {{0, 0}, {0, 0}, {1, 0}, {2, 0}, {4, 0}};
    order = SmallVector<unsigned>{0u, 1u};
    mmaWarpsPerCTA = SmallVector<unsigned>{warpsPerCTA[1], warpsPerCTA[0]};
  }
  LinearLayout LL =
      LinearLayout::identity1D(shape[1], kRegister, outDims[kIdx]) *
      LinearLayout({{kLane, laneBase}}, {outDims[mnIdx], outDims[kIdx]}) *
      broadcastedDotOperandLayout(ctx, mmaWarpsPerCTA, order, 1u, kWarp);
  return combineCtaCgaWithShape(LL, ctaLayout, shape);
}

LinearLayout chooseScaledMfmaScaleLayout(MLIRContext *ctx, int dotOperandIdx,
                                         ArrayRef<int64_t> dotOperandShape,
                                         unsigned mfmaMDim,
                                         ArrayRef<unsigned> tilesPerWarp,
                                         ArrayRef<unsigned> warpsPerCTA) {
  using basisT = std::vector<std::vector<int32_t>>;
  unsigned rank = dotOperandShape.size();
  auto order = mlir::triton::gpu::getMatrixOrder(rank, /*rowMajor=*/true);
  auto standardOutDims = standardOutDimNames(ctx, rank);
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  // Fetch the tilesPerWarp value in the M dimension for operand A, or in the N
  // dimension for operand B.
  unsigned mnDim = dotOperandIdx == 0 ? rank - 2 : rank - 1;
  unsigned tilePerWarpMN = tilesPerWarp[mnDim];

  // In scaled dot, the shapes of operands(without batch dimension) are,
  // respectively:
  // - A: [M, K]
  // - B: [K, N]
  // - aScale: [M, K / 32]
  // - bScale: [N, K / 32]
  //
  // In general, for both 32x32 and 16x16 scaled mfma, and no matter what
  // data type the A/B operand is, each lane takes 32 elements from A/B
  // alone K dim, and 1 or 2 elements from scale accordingly. The number of
  // scale's elements in a lane varies because the 32 elements from A/B may
  // not be consecutive.
  //
  // For mxfp4, these 32 elements are consecutive, so only 1 scale element
  // is required. But for mxfp6/mxfp8, there are 2 16-consecutive elements
  // blocks, so 2 scale elements are required.
  int32_t kSize = dotOperandShape[1];

  std::vector<std::vector<int32_t>> registerBase;
  std::vector<std::vector<int32_t>> laneBase;

  auto threadsInKDim = mfmaMDim == 32 ? 2 : 4;
  for (int32_t elem = threadsInKDim; elem < kSize; elem *= 2)
    registerBase.emplace_back(std::vector<int32_t>{elem, 0});

  for (int32_t elem = mfmaMDim; elem < tilePerWarpMN * mfmaMDim; elem *= 2)
    registerBase.emplace_back(std::vector<int32_t>{0, elem});

  if (mfmaMDim == 32) {
    // For ROCDL::mfma_scale_f32_32x32x64_f8f6f4 with fp4 input, each lane
    // takes 32 consecutive elements from A alone K dimension. The first
    // 32 lanes collectively handle A[0:32][0:32], and the other 32 lanes
    // collectively handle A[0:32][32:64]. Each lane take 1 scale element
    // accordingly. Similar to B and bScale.
    laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}};
  } else {
    assert(mfmaMDim == 16);
    // For ROCDL::mfma_scale_f32_16x16x128_f8f6f4 with fp4 input, each lane
    // takes 32 consecutive elements from A alone K dimension. The first
    // 16 lanes collectively handle A[0:16][0:32], and another 16 lanes
    // collectively handle A[0:16][32:64] and so on. Each lane take 1 scale
    // element accordingly. Similar to B and bScale.
    laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}};
  }

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);
  LinearLayout tileLayout({{kRegister, registerBase}, {kLane, laneBase}},
                          {outDimNames[order[0]], outDimNames[order[1]]});

  SmallVector<unsigned> warpsPerCTANew =
      (dotOperandIdx == 1)
          ? SmallVector<unsigned>{warpsPerCTA[1], warpsPerCTA[0]}
          : SmallVector<unsigned>{warpsPerCTA[0], warpsPerCTA[1]};

  SmallVector<unsigned> warpOrder = (dotOperandIdx == 1)
                                        ? SmallVector<unsigned>{0, 1}
                                        : SmallVector<unsigned>{1, 0};

  LinearLayout warpLayout =
      identityStandardND(kWarp, warpsPerCTANew, warpOrder);
  LinearLayout ctaLayout = tileLayout.transposeOuts(outDimNames) *
                           warpLayout.transposeOuts(outDimNames);

  auto ctaLay = CTAEncodingAttr::getDefault(ctx, 2);
  auto finalLay = combineCtaCgaWithShape(ctaLayout, ctaLay, dotOperandShape);
  return finalLay;
}

#ifdef __ILUVATAR__
// Store-friendly relayout of the TCU 16x16 result tile.
//
// The native iluvatarMmaTile maps a thread's 4 result registers to the SAME
// column at 4 different rows (register bits drive M):
//   register = {{4,0},{8,0}}      (M+4, M+8)
//   lane     = {{0,1},{0,2},{0,4},{0,8},{1,0},{2,0}}
// After truncation to a 16-bit element type that makes every global store a
// per-element 2-byte write, even though consecutive lanes already cover
// consecutive columns.
//
// This tile makes each thread hold 2 CONSECUTIVE columns (n, n+1) WHILE keeping
// the global store coalesced across lanes:
//   register = {{0,1},{8,0}}                       (N+1, M+8)
//   lane     = {{0,2},{0,4},{0,8},{1,0},{2,0},{4,0}}
// The lowest lane bit stays an N offset (N+2), so adjacent lanes still write
// adjacent columns (coalesced), and register bit 0 (N+1) gives each thread a
// contiguous 2-element (32-bit) store. It covers the same 16x16 element set and
// keeps the warp/block assignment identical to iluvatarMmaTile.
//
// Relative to iluvatarMmaTile this requires TWO register<->lane bit
// transpositions (N+1 moves lane->register, M+4 moves register->lane) plus a
// lane permutation. The generic transferWithinWarp path implements exactly this
// (multiple disjoint transpositions + lane permutation) entirely with warp
// shuffles and register selects, so no shared-memory round-trip is needed. This
// is gated by relaxing cvtNeedsWarpShuffle for Iluvatar (allowing two mixed
// transpositions), and is the v3.6-idiomatic equivalent of the v3.2 lib's
// hand-written mma->mma1 (lowerMmaToMma) store path.
static LinearLayout iluvatarStoreTile(MLIRContext *ctx, StringAttr rowDim,
                                      StringAttr colDim) {
  return LinearLayout(
      {{S("register"), {{0, 1}, {8, 0}}},
       {S("lane"), {{0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}, {4, 0}}}},
      {rowDim, colDim});
}

// 2-TCU (64B) scanline tile: a single 16x32 (two adjacent 16x16 TCU tiles)
// block, matching the v3.2 versionMinor=1 layout. Relative to the mma 16x32
// tile it differs by exactly ONE register<->lane transposition (N+1 <-> N+16):
// N+1 moves into register (so each thread holds 2 CONSECUTIVE columns -> 32-bit
// store) and the tile-rep bit N+16 moves into lane, while lane bit0 stays N+2
// (adjacent lanes write adjacent columns -> coalesced).
//
// Because it is a single mixed transposition, the mma->store convert stays on
// the warp-shuffle path even under the default cvtNeedsWarpShuffle gate (<2),
// and it keeps register pressure close to the mma baseline -- unlike the
// single-16x16-tile iluvatarStoreTile, which must evict an M register bit to
// lane (TWO transpositions, higher register pressure, needs the relaxed gate).
//
// Only valid when the warp does NOT split N (warpsPerCTA[N] == 1); otherwise
// the adjacent 16-col tile (N+16) lives in another warp and the swap is
// cross-warp.
static LinearLayout iluvatarStoreTile2TCU(MLIRContext *ctx, StringAttr rowDim,
                                          StringAttr colDim) {
  return LinearLayout(
      {{S("register"), {{0, 1}, {4, 0}, {8, 0}}}, // N+1, M+4, M+8
       {S("lane"),
        {{0, 2},
         {0, 4},
         {0, 8},
         {0, 16},
         {1, 0},
         {2, 0}}}}, // N+2,N+4,N+8,N+16,M+1,M+2
      {rowDim, colDim});
}

std::optional<LinearLayout>
chooseIluvatarStoreLayout(RankedTensorType valType) {
  auto mma = mlir::dyn_cast<IluvatarMmaEncodingAttr>(valType.getEncoding());
  if (!mma)
    return std::nullopt;

  // Wide stores only pay off for the 16-bit dot output dtypes, and the tile is
  // a full 16x16 so the shape must be a multiple of 16 along both dims.
  Type elemType = valType.getElementType();
  if (valType.getRank() != 2 || !(elemType.isF16() || elemType.isBF16()))
    return std::nullopt;
  auto shape = valType.getShape();
  if (shape[0] % 16 != 0 || shape[1] % 16 != 0)
    return std::nullopt;

  auto ctx = mma.getContext();
  auto dimNames = standardOutDimNames(ctx, 2);
  auto dimM = dimNames[0];
  auto dimN = dimNames[1];

  // When the warp does not split N, use the 2-TCU (16x32) scanline tile: it
  // reaches a coalesced 32-bit store with a single transposition and near-mma
  // register pressure. Otherwise fall back to the single-16x16-tile layout.
  auto warpsPerCTA = mma.getWarpsPerCTA();
  bool canUse2TCU =
      warpsPerCTA.size() == 2 && warpsPerCTA[1] == 1 && shape[1] % 32 == 0;
  LinearLayout ctaLayout = canUse2TCU ? iluvatarStoreTile2TCU(ctx, dimM, dimN)
                                      : iluvatarStoreTile(ctx, dimM, dimN);
  auto warpOrder = getDefaultMmaOrder(mma);
  ctaLayout *= identityStandardND(S("warp"), mma.getWarpsPerCTA(), warpOrder)
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, mma.getCTALayout(), shape);
}
#endif

} // namespace mlir::triton::gpu
