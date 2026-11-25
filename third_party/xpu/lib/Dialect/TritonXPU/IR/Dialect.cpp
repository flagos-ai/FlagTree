#include "triton/Dialect/Triton/IR/Dialect.h" // triton::PointType [Ops.cpp.inc]
#include "mlir/IR/DialectImplementation.h"    // DialectAsmParser

// clang-format off
#include "triton/Dialect/TritonXPU/IR/Dialect.h" // before cpp.inc
#include "triton/Dialect/TritonXPU/IR/TritonXPUEnums.cpp.inc"
#include "triton/Dialect/TritonXPU/IR/Dialect.cpp.inc"
// clang-format on

#include "triton/Dialect/Triton/IR/Utility.h" // ceil
#include "triton/Tools/LayoutUtils.h" // identityStandardND, ensureLayoutNotSmallerThan
#include "triton/Tools/LinearLayout.h" // LinearLayout (DistributedEncodingTrait)
#include "llvm/ADT/TypeSwitch.h"       // TypeSwitch

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace xpu {

SmallVector<unsigned> getSizePerCore(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getSizePerCoreInterface();
  }

  llvm::report_fatal_error("getSizePerCoreInterface not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> getCoresPerGroup(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getCoresPerGroupInterface();
  }

  llvm::report_fatal_error("getCoresPerGroupInterface not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> getGroupsPerCluster(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getGroupsPerClusterInterface();
  }

  llvm::report_fatal_error("getGroupsPerClusterInterface not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned> getCoresPerCluster(Attribute layout) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getCoresPerClusterInterface();
  }

  llvm::report_fatal_error("getCoresPerClusterInterface not implemented");
  return SmallVector<unsigned>();
}

unsigned getTotalElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || isa<triton::PointerType>(type))
    return 1;
  auto tensorType = cast<RankedTensorType>(type);
  return getTotalElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
                                tensorType.getElementType());
}

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getTotalElemsPerThread(shape, eltTy);
  } else if (auto sliceLayout =
                 mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    // XPU-backed slice: recurse into the parent XPU layout using the padded
    // parent shape, then drop the sliced dim. Matches the 3.0 fork's
    // SliceEncodingAttr::getTotalElemsPerThread(shape, eltTy) recursion and
    // avoids the generic DistributedEncodingTrait/LinearEncoding path, whose
    // toLinearLayout asserts on non-power-of-two shapes.
    return product<unsigned>(getElemsPerThread(layout, shape, eltTy));
  } else if (auto tritonGPUAttr =
                 mlir::dyn_cast<triton::gpu::DistributedEncodingTrait>(
                     layout)) {
    return tritonGPUAttr.getTotalElemsPerThread(shape);
  } else {
    llvm::report_fatal_error("getTotalElemsPerThread not implemented");
    return 0;
  }
}

SmallVector<unsigned> getElemsPerThread(Attribute layout,
                                        ArrayRef<int64_t> shape, Type eltTy) {
  if (auto tritonXPUAttr = mlir::dyn_cast<TritonXPU_AttrTrait>(layout)) {
    return tritonXPUAttr.getElemsPerThread(shape, eltTy);
  } else if (auto sliceLayout =
                 mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    // See getTotalElemsPerThread above: recurse into the parent XPU layout
    // with the padded parent shape, then erase the sliced dim (3.0 behavior).
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentElemsPerThread =
        getElemsPerThread(parentLayout, parentShape, eltTy);
    parentElemsPerThread.erase(parentElemsPerThread.begin() +
                               sliceLayout.getDim());
    return parentElemsPerThread;
  } else if (auto tritonGPUAttr =
                 mlir::dyn_cast<triton::gpu::DistributedEncodingTrait>(
                     layout)) {
    return tritonGPUAttr.getElemsPerThread(shape);
  } else {
    llvm::report_fatal_error("getElemsPerThread not implemented");
    return SmallVector<unsigned>();
  }
}

unsigned getGroupSize(Attribute layout) {
  unsigned size = 1;
  auto coresPerGroup = getCoresPerGroup(layout);
  for (auto e : coresPerGroup) {
    size *= e;
  }
  return size;
}

// 1 element per thread
// order = reverse(arange(rank))
triton::xpu::ClusterLayoutAttr
getDefaultClusterEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          uint32_t buffer_size, uint32_t core_num) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  // TODO[dyq]: why blockEncoding reverse order in triton 3.0
  triton::xpu::ClusterLayoutAttr encoding = triton::xpu::ClusterLayoutAttr::get(
      context, shape, order, buffer_size, core_num);
  return encoding;
}

// See the contract on the declaration in Dialect.h: this is the one place the
// unencoded-TLE-tensor reading lives, and it has to stay the same reading
// `TypeConverter` uses when it lowers one.
Type withTLEDefaultEncoding(Type type) {
  auto tensorTy = mlir::dyn_cast<RankedTensorType>(type);
  if (!tensorTy || tensorTy.getEncoding())
    return type;
  auto encoding =
      getDefaultClusterEncoding(type.getContext(), tensorTy.getShape(),
                                /*buffer_size=*/128,
                                /*core_num=*/64);
  return RankedTensorType::get(tensorTy.getShape(), tensorTy.getElementType(),
                               encoding);
}

// See the contract on the declaration in Dialect.h. Only a cluster-layout
// sibling of the same shape and element type counts: anything else is a real
// layout change, not the encoding-stripping cvt this looks through.
static bool isTLESibling(RankedTensorType candidate, RankedTensorType ty) {
  return candidate &&
         mlir::isa_and_nonnull<triton::xpu::ClusterLayoutAttr>(
             candidate.getEncoding()) &&
         candidate.getShape() == ty.getShape() &&
         candidate.getElementType() == ty.getElementType();
}

Type tleEncodedFacingTypeOrNull(Value value) {
  auto ty = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!ty)
    return Type();
  if (ty.getEncoding())
    return value.getType();
  if (auto cvt = value.getDefiningOp<triton::xpu::ConvertLayoutOp>()) {
    auto srcTy = mlir::dyn_cast<RankedTensorType>(cvt.getOperand().getType());
    if (isTLESibling(srcTy, ty))
      return srcTy;
  }
  for (Operation *user : value.getUsers()) {
    if (auto cvt = mlir::dyn_cast<triton::xpu::ConvertLayoutOp>(user)) {
      auto resTy = mlir::dyn_cast<RankedTensorType>(cvt.getResult().getType());
      if (isTLESibling(resTy, ty))
        return resTy;
    }
  }
  return Type();
}

Type tleEncodedFacingType(Value value) {
  if (Type sibling = tleEncodedFacingTypeOrNull(value))
    return sibling;
  return withTLEDefaultEncoding(value.getType());
}

// The `tle_local_ptr` behind a TLE `tt.load`, looking through the identity
// `convert_layout` the TLE type conversion leaves on SM pointers (addrspace 2):
// the canonicalizer that would fold it runs later, so a plain `getDefiningOp`
// is null there and every SM weight read stays scalar. Only a cvt whose result
// type equals its operand type is skipped -- a real relayout would change the
// per-core element distribution `tle_vload` assumes.
triton::xpu::TLELocalPtrOp getTLELocalPtrThroughCvt(Value ptr) {
  while (auto cvt = mlir::dyn_cast_or_null<triton::xpu::ConvertLayoutOp>(
             ptr.getDefiningOp())) {
    if (cvt.getResult().getType() != cvt.getOperand().getType())
      return nullptr;
    ptr = cvt.getOperand();
  }
  return ptr.getDefiningOp<triton::xpu::TLELocalPtrOp>();
}

// Whether a TLE buffer read is expressible as whole vector loads, and if so its
// uniform element addend.
//
// An LM buffer is per-core and its staging DMA already dropped this core's
// slice at the buffer base, so the lowering ignores the index tensor entirely:
// always admissible, `off` stays null. A `scope=smem` buffer instead holds the
// FULL array once per cluster, so the index tensor is what selects the elements
// and a vector read has to reproduce it. `tle_vload` expresses exactly
// `buffer + off + <this core's contiguous block> + i`, i.e. only
// `[roff +] tl.arange(0, n)`; anything permuted / gathered / strided has no
// such form and has to stay on the scalar SM path, which honors the indices
// element by element.
bool tleBufferIsSmem(Value buffer) {
  Operation *alloc = buffer.getDefiningOp();
  if (!alloc)
    return false;
  auto scope = alloc->getAttrOfType<StringAttr>("xpu.mem_scope");
  return scope && scope.getValue() == "smem";
}

bool tleSmemSliceOffset(triton::xpu::TLELocalPtrOp lp, Value &off,
                        bool allowSmem) {
  off = Value();
  Operation *alloc = lp.getBuffer().getDefiningOp();
  auto scope =
      alloc ? alloc->getAttrOfType<StringAttr>("xpu.mem_scope") : StringAttr();
  if (!scope || scope.getValue() != "smem")
    return true;
  // The escape hatch: refuse the whole scope, index or no index.
  if (!allowSmem)
    return false;
  if (lp.getIndices().empty())
    return false;
  Value idx = lp.getIndices().back();
  while (Operation *d = idx.getDefiningOp()) {
    // Layout casts and the rank change do not alter the index VALUES.
    if (auto cvt = mlir::dyn_cast<triton::xpu::ConvertLayoutOp>(d)) {
      idx = cvt.getOperand();
      continue;
    }
    if (auto ex = mlir::dyn_cast<triton::ExpandDimsOp>(d)) {
      idx = ex.getSrc();
      continue;
    }
    if (auto addOp = mlir::dyn_cast<arith::AddIOp>(d)) {
      // Exactly one splatted addend is representable, as the element offset.
      auto splatSrc = [](Value v) -> Value {
        if (auto s = v.getDefiningOp<triton::SplatOp>())
          return s.getSrc();
        return Value();
      };
      Value uniform = splatSrc(addOp.getLhs());
      Value other = addOp.getRhs();
      if (!uniform) {
        uniform = splatSrc(addOp.getRhs());
        other = addOp.getLhs();
      }
      if (!uniform || off)
        return false;
      off = uniform;
      idx = other;
      continue;
    }
    break;
  }
  auto mr = idx.getDefiningOp<triton::MakeRangeOp>();
  return mr && mr.getStart() == 0;
}

bool tleSmemGather(triton::xpu::TLELocalPtrOp lp) {
  Operation *alloc = lp.getBuffer().getDefiningOp();
  auto scope =
      alloc ? alloc->getAttrOfType<StringAttr>("xpu.mem_scope") : StringAttr();
  if (!scope || scope.getValue() != "smem")
    return false;
  if (lp.getIndices().empty())
    return false;
  // bf16 is supported via the HF vgather (16-bit lanes) + a bf16->f32 fold that
  // turns TWO 32-lane gathers into f32 result vectors, so the per-core element
  // count must be a whole number of 32-lane gathers. Otherwise keep bf16 on the
  // scalar SM path. Check the BUFFER element type, not the local_ptr pointee:
  // tritonxpu-tle-dtype-convert promotes the bf16 load's pointee to f32.
  if (mlir::getElementTypeOrSelf(lp.getBuffer().getType()).isBF16()) {
    auto ptrTensorTy =
        mlir::dyn_cast<RankedTensorType>(lp.getResult().getType());
    auto cl = ptrTensorTy
                  ? mlir::dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
                        ptrTensorTy.getEncoding())
                  : nullptr;
    if (!cl)
      return false;
    int64_t perCore = 1;
    for (auto s : cl.getSizePerCore())
      perCore *= s;
    if (perCore % 32 != 0)
      return false;
  }
  // A sliceable index (`[uniform +] arange(0,n)`) is a contiguous read handled
  // by tle_vload; only a genuinely non-slice index needs the gather.
  Value off;
  return !tleSmemSliceOffset(lp, off, /*allowSmem=*/true);
}

SmallVector<unsigned>
getCoresPerClusterWithUniqueData(Attribute layout,
                                 ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout =
          mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentCoresPerCluster =
        getCoresPerClusterWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> coresPerCluster = parentCoresPerCluster;
    coresPerCluster.erase(coresPerCluster.begin() + sliceLayout.getDim());
    return coresPerCluster;
  }
  auto coresPerCluster = getCoresPerCluster(layout);
  assert(coresPerCluster.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < coresPerCluster.size(); i++) {
    auto sizePerCore = getSizePerCore(layout)[i];
    auto maxCoresPerDim = ceil<unsigned>(tensorShape[i], sizePerCore);
    coresPerCluster[i] = std::min<unsigned>(coresPerCluster[i], maxCoresPerDim);
  }

  return coresPerCluster;
}

SmallVector<unsigned>
getCoresPerGroupWithUniqueData(Attribute layout,
                               ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout =
          mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentCoresPerGroup =
        getCoresPerGroupWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> coresPerGroup = parentCoresPerGroup;
    coresPerGroup.erase(coresPerGroup.begin() + sliceLayout.getDim());
    return coresPerGroup;
  }
  auto coresPerGroup = getCoresPerGroup(layout);
  assert(coresPerGroup.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < coresPerGroup.size(); i++) {
    coresPerGroup[i] = std::min<unsigned>(coresPerGroup[i], tensorShape[i]);
  }

  return coresPerGroup;
}

SmallVector<unsigned> getUniqueContigPerCore(Attribute layout,
                                             ArrayRef<int64_t> shape) {
  // If slice layout, call recursively on parent layout, and drop
  // sliced dim
  if (auto sliceLayout =
          mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentUniqueContigPerCore =
        triton::xpu::getUniqueContigPerCore(parentLayout, parentShape);
    parentUniqueContigPerCore.erase(parentUniqueContigPerCore.begin() +
                                    sliceLayout.getDim());
    return parentUniqueContigPerCore;
  }
  // Base case
  auto rank = shape.size();
  SmallVector<unsigned> ret(rank);
  auto contigPerCore = getSizePerCore(layout);
  assert(contigPerCore.size() == rank && "Unexpected contigPerCore size");
  for (int d = 0; d < rank; ++d) {
    ret[d] = std::min<unsigned>(shape[d], contigPerCore[d]);
  }
  return ret;
}

LogicalResult ClusterLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    ArrayRef<unsigned> sizePerCore, ArrayRef<unsigned> coresPerGroup,
    ArrayRef<unsigned> groupsPerCluster, ArrayRef<unsigned> order) {
  if (sizePerCore.size() != coresPerGroup.size() ||
      coresPerGroup.size() != groupsPerCluster.size() ||
      groupsPerCluster.size() != order.size()) {
    return emitError() << "sizePerCore, coresPerGroup, groupsPerCluster, and "
                          "order must all have the same rank.";
  }

  if (!isPermutationOfIota(order)) {
    return emitError()
           << "order must be a permutation of 0..(rank-1), but was [" << order
           << "]";
  }
  return success();
}

} // namespace xpu
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Parse Utility
//===----------------------------------------------------------------------===//

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = mlir::dyn_cast<IntegerAttr>(attr);
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

static LogicalResult parseBoolAttrValue(AsmParser &parser, Attribute attr,
                                        bool &value, StringRef desc) {
  auto boolAttr = mlir::dyn_cast<BoolAttr>(attr);
  if (!boolAttr) {
    parser.emitError(parser.getNameLoc(), "expected an bool type in ") << desc;
    return failure();
  }
  value = boolAttr.getValue();
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

static LogicalResult parseBool(AsmParser &parser, const NamedAttribute &attr,
                               bool &value, StringRef desc) {
  return parseBoolAttrValue(parser, attr.getValue(), value, desc);
};

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.cpp.inc"

SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getSizePerCoreInterface() const {
  return SmallVector<unsigned>(getSizePerCore());
}
SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getCoresPerGroupInterface() const {
  return SmallVector<unsigned>(getCoresPerGroup());
}
SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getGroupsPerClusterInterface() const {
  return SmallVector<unsigned>(getGroupsPerCluster());
}
SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getCoresPerClusterInterface() const {
  SmallVector<unsigned> coresPerCluster;
  for (unsigned d = 0, n = getOrder().size(); d < n; ++d)
    coresPerCluster.push_back(getCoresPerGroup()[d] * getGroupsPerCluster()[d]);
  return coresPerCluster;
}

SmallVector<unsigned>
triton::xpu::ClusterLayoutAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                                  Type eltTy) const {
  size_t rank = shape.size();
  auto sizePerCore = getSizePerCore();
  auto coresPerGroup = getCoresPerGroup();
  auto groupsPerCluster = getGroupsPerCluster();
  assert(rank == sizePerCore.size() &&
         "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
  SmallVector<unsigned> elemsPerThread(rank);
  for (size_t i = 0; i < rank; ++i) {
    unsigned t = groupsPerCluster[i] * coresPerGroup[i];
    elemsPerThread[i] = ceil<unsigned>(shape[i], t);
  }
  return elemsPerThread;
}

unsigned
triton::xpu::ClusterLayoutAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                       Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

// DistributedEncodingTrait stubs (LLVM22 upgrade). The XPU lowering pipeline
// does not exercise the LinearLayout-based code paths upstream uses for
// generic distributed encodings; these are provided solely so ClusterLayoutAttr
// satisfies the trait contract that SliceEncodingAttr::get now requires for
// its `parent` parameter.
SmallVector<unsigned> triton::xpu::ClusterLayoutAttr::getRepOrder() const {
  return SmallVector<unsigned>(getOrder().begin(), getOrder().end());
}

// XPU has no inter-CTA ("block") concept. Return a degenerate default
// CTA layout so that LayoutEncodingTrait consumers (e.g. SliceEncodingAttr's
// verifier calling ::getCTALayout(parent)) work uniformly.
::mlir::triton::gpu::CTAEncodingAttr
triton::xpu::ClusterLayoutAttr::getCTALayout() const {
  return ::mlir::triton::gpu::CTAEncodingAttr::getDefault(
      getContext(), static_cast<int>(getOrder().size()));
}

SmallVector<unsigned> triton::xpu::ClusterLayoutAttr::getElemsPerThread(
    ArrayRef<int64_t> shape) const {
  return getElemsPerThread(shape, /*eltTy=*/Type());
}

unsigned triton::xpu::ClusterLayoutAttr::getTotalElemsPerThread(
    ArrayRef<int64_t> shape) const {
  return product<unsigned>(getElemsPerThread(shape));
}

::mlir::triton::LinearLayout
triton::xpu::ClusterLayoutAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  // Map XPU's 3-level hierarchy (sizePerCore / coresPerGroup /
  // groupsPerCluster) onto the standard register/lane/warp dimensions used by
  // upstream LinearLayout consumers. XPU has no inter-CTA ("block") concept, so
  // we add a degenerate block dim of size 1.
  //
  // This mirrors BlockedEncodingAttr::toLinearLayout, minus the CTA layer,
  // and lets shared upstream helpers (e.g. ElementwiseOpConversionBase's
  // maybeDeduplicate) work uniformly when a tensor is encoded with a
  // ClusterLayoutAttr.
  MLIRContext *ctx = getContext();
  auto order = llvm::to_vector(getOrder());
  int rank = shape.size();

  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  // XPU allows non-power-of-two per-level sizes, but upstream LinearLayout
  // building blocks (identity1D -> strided1D, LayoutUtils::ensure*) assert
  // power-of-two sizes. This LinearLayout is only a *compatibility view* for
  // generic upstream helpers that need a LinearLayout (getFreeVariableMasks,
  // isExpensiveView, ElementwiseOpConversionBase::maybeDeduplicate); the
  // authoritative per-thread arity comes from getElemsPerThread (ceil-based),
  // and XPU offset emission uses the ceil-based emitOffsetForClusterLayout
  // (see third_party/xpu/lib/Conversion/TritonXPUToLLVM/Utility.cpp), NOT this
  // LinearLayout's register in-dim. So it is safe to round each level up to
  // the next power of two before feeding it to identityStandardND.
  auto padVec = [](ArrayRef<unsigned> v) {
    SmallVector<unsigned> out(v.begin(), v.end());
    for (auto &e : out)
      e = e <= 1 ? e : llvm::NextPowerOf2(e - 1);
    return out;
  };
  SmallVector<unsigned> paddedSizePerCore = padVec(getSizePerCore());
  SmallVector<unsigned> paddedCoresPerGroup = padVec(getCoresPerGroup());
  SmallVector<unsigned> paddedGroupsPerCluster = padVec(getGroupsPerCluster());

  ::mlir::triton::LinearLayout ctaLayout =
      ::mlir::triton::identityStandardND(kRegister, paddedSizePerCore, order) *
      ::mlir::triton::identityStandardND(kLane, paddedCoresPerGroup, order) *
      ::mlir::triton::identityStandardND(kWarp, paddedGroupsPerCluster, order);

  // Append a degenerate "block" dim so consumers that expect this dim
  // (upstream code does) don't fail.
  SmallVector<StringAttr> outDimNames =
      ::mlir::triton::standardOutDimNames(ctx, rank);
  ::mlir::triton::LinearLayout::BasesT blockBases;
  blockBases[kBlock] = std::vector<std::vector<int32_t>>{};
  ::mlir::triton::LinearLayout blockLayout(std::move(blockBases), outDimNames);
  ctaLayout = ctaLayout * blockLayout;

  // Resize to match `shape`. Use the padded (power-of-two) per-level product as
  // the desired out-dim size so ensureLayoutNotLargerThan's
  // actualSize % desiredSize == 0 assert holds for non-power-of-two shapes.
  llvm::SmallDenseMap<StringAttr, int64_t> labeledShape;
  for (int i = 0; i < rank; ++i) {
    int64_t actualSize = int64_t(paddedSizePerCore[i]) *
                         paddedCoresPerGroup[i] * paddedGroupsPerCluster[i];
    int64_t rawShape = shape[i];
    int64_t desired;
    if (actualSize >= rawShape) {
      desired =
          actualSize; // pad up; keeps register arity == ceil(shape,cpg*gpc)
    } else {
      // registers must cover the remainder; round up to a multiple of
      // actualSize
      desired = ((rawShape + actualSize - 1) / actualSize) * actualSize;
    }
    labeledShape[outDimNames[i]] = desired;
  }
  ctaLayout =
      ::mlir::triton::ensureLayoutNotSmallerThan(ctaLayout, labeledShape);
  ctaLayout =
      ::mlir::triton::ensureLayoutNotLargerThan(ctaLayout, labeledShape);
  return ctaLayout.transposeOuts(outDimNames);
}

Attribute triton::xpu::ClusterLayoutAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> sizePerCore;
  SmallVector<unsigned> coresPerGroup;
  SmallVector<unsigned> groupsPerCluster;
  SmallVector<unsigned> order;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "sizePerCore") {
      if (parseIntArrayAttr(parser, attr, sizePerCore,
                            "number of elements per core")
              .failed())
        return {};
    } else if (attr.getName() == "coresPerGroup") {
      if (parseIntArrayAttr(parser, attr, coresPerGroup,
                            "number of cores per group")
              .failed())
        return {};
    } else if (attr.getName() == "groupsPerCluster") {
      if (parseIntArrayAttr(parser, attr, groupsPerCluster,
                            "number of groups per cluster")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  return parser.getChecked<ClusterLayoutAttr>(
      parser.getContext(), sizePerCore, coresPerGroup, groupsPerCluster, order);
}

void triton::xpu::ClusterLayoutAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerCore = [" << ArrayRef(getSizePerCore()) << "]"
          << ", coresPerGroup = [" << ArrayRef(getCoresPerGroup()) << "]"
          << ", groupsPerCluster = [" << ArrayRef(getGroupsPerCluster()) << "]"
          << ", order = [" << getOrder() << "]" << "}>";
}

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//

class TritonXPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto clusterAttr =
            mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(attr)) {
      os << "cluster";
      return AliasResult::FinalAlias;
    } /* else if (auto sdnnAttr = mlir::dyn_cast<SDNNEncodingAttr>(attr)) {
      os << "sdnn";
      return AliasResult::FinalAlias;
    } */
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

struct TritonXPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    resultEncoding = triton::gpu::SliceEncodingAttr::get(
        getDialect()->getContext(), axis,
        cast<triton::gpu::DistributedEncodingTrait>(operandEncoding));
    return success();
  }

  // Infer the encoding of a tt.trans(x) given the encoding of x.
  //
  // Our goal is to choose an encoding so that the trans is a "nop".  For
  // example, in a blocked encoding, the same GPU threads hold the same
  // elements, they're just "renamed" -- what was element [i,j] of the tensor is
  // now element [j,i], but that element is held by the same GPU thread.
  //
  // For most properties of the encoding, we let
  //   outputEnc.prop = inputEnc.prop * trans.order,
  // where `x * y` means we apply permutation y to x.
  //
  // This works because prop[i] tells you something about the i'th dimension of
  // the tensor. (For example, sizePerThread[2] == 4 means that one GPU thread
  // contains 4 elements along dim 2 of the tensor.) The transpose reorders the
  // dimensions according to the perm trans.order, so we achieve our goal of
  // having a "nop" transpose by reordering the values in the prop the same way.
  //
  // The big exception to this is the encoding's `order`.
  //
  // An encoding's order is a list of dimensions, from fastest moving (most
  // minor) to slowest moving.  Thus enc.order[i] does not tell you something
  // about the i'th dimension of the tensor, and it would be disasterously
  // incorrect to do enc.order * trans.order.
  //
  // But!  If we invert enc.order, it *does* meet this criterion.  For example,
  // if enc.order = [2,0,1], inverse(enc.order) = [1,2,0].  If you stare at it,
  // you'll see that inverse(enc.order)[i] == j means that dimension i is the
  // j'th most minor.  Therefore we can safely permute *this* by trans.order.
  //
  // Thus we have
  //
  //   outputEnc.order = inverse(inverse(inputEnc.order) * trans.order)
  //                   = inverse(trans.order) * inputEnc.order.
  //
  // Give a transpose the layout that makes it a register rename.
  //
  // Why: a transpose only renames the element each core holds. Permuting the
  // per-dimension split by the transpose order keeps every core on exactly the
  // elements it had, so the LLVM lowering is a slot-wise copy instead of a
  // data movement. The layout order needs the inverse permutation (see the
  // derivation above): outputEnc.order = inverse(trans.order) * inputEnc.order.
  //
  // How:
  //   - permute sizePerCore / coresPerGroup / groupsPerCluster by the order
  //     (prop[i] describes dimension i, so the property moves with the axis);
  //   - compose the layout order with the inverse permutation;
  //   - reject a non-cluster operand: the XPU backend has no other layout it
  //     could reason about here.
  //
  // \param operandEncoding layout of the source tensor.
  // \param shape source shape (unused: the permutation is shape independent).
  // \param order the transpose permutation.
  // \param resultEncoding receives the permuted layout.
  // \param loc diagnostic anchor.
  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    auto cluster =
        mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(operandEncoding);
    if (!cluster)
      return emitOptionalError(loc,
                               "tt.trans requires a cluster layout "
                               "operand, got ",
                               operandEncoding);
    // Every index below -- dims[axis], inverse[order[i]], inverse[axis] --
    // comes straight out of the attributes, so a malformed tt.trans reads or
    // writes past the end. Reject it here rather than relying on an assert a
    // release build drops.
    unsigned rank = cluster.getSizePerCore().size();
    if (order.size() != rank || cluster.getOrder().size() != rank)
      return emitOptionalError(loc, "tt.trans order rank ", order.size(),
                               " does not match the cluster layout rank ",
                               rank);
    llvm::SmallVector<bool> seen(rank, false);
    for (int32_t axis : order) {
      if (axis < 0 || static_cast<unsigned>(axis) >= rank || seen[axis])
        return emitOptionalError(loc,
                                 "tt.trans order is not a permutation of "
                                 "[0, ",
                                 rank, ")");
      seen[axis] = true;
    }
    for (unsigned axis : cluster.getOrder())
      if (axis >= rank)
        return emitOptionalError(loc, "cluster layout order entry ", axis,
                                 " is out of range for rank ", rank);

    auto permute = [&](ArrayRef<unsigned> dims) {
      llvm::SmallVector<unsigned> out;
      out.reserve(dims.size());
      for (auto axis : order)
        out.push_back(dims[axis]);
      return out;
    };
    llvm::SmallVector<unsigned> inverse(order.size());
    for (unsigned i = 0; i < order.size(); ++i)
      inverse[order[i]] = i;
    llvm::SmallVector<unsigned> retOrder;
    for (unsigned axis : cluster.getOrder())
      retOrder.push_back(inverse[axis]);

    resultEncoding = triton::xpu::ClusterLayoutAttr::get(
        getContext(), permute(cluster.getSizePerCore()),
        permute(cluster.getCoresPerGroup()),
        permute(cluster.getGroupsPerCluster()), retOrder);
    return success();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    auto sliceEncoding =
        mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(operandEncoding);
    if (!sliceEncoding)
      return emitOptionalError(
          location, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    if (sliceEncoding.getDim() != axis)
      return emitOptionalError(
          location, "Incompatible slice dimension for ExpandDimsOp operand");
    resultEncoding = sliceEncoding.getParent();
    return success();
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const override {
    llvm_unreachable("TODO[dyq]: XPUSDNN-CHECK Add "
                     "triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    llvm_unreachable("TODO[dyq]: XPUSDNN-CHECK Add "
                     "triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }

  // Given a src shape + encoding and a dst shape, our goal is to compute a dst
  // encoding that makes the reshape a "nop".  That is, if GPU thread [x,y,z]
  // contains elements [a,b,c,d] before the reshape, it contains those same
  // elements after the reshape, they're just "renamed".
  //
  // A dst encoding that satisfies this property does not exist for all inputs.
  // Here are some positive and negative examples.
  //
  //   - NOT OK: 4x4 order=[0,1] -> 16.  Reshape merges elements so
  //     dim 1 is the fastest-changing in the dst, but the src has the opposite
  //     order.
  //   - OK: 2x2x32 order=[1,0,2] -> 4x32.  We choose dst order [0,1].
  //     What's important is that the 2x2 dimensions appear in major-to-minor
  //     order.
  //   - NOT OK: 32x32 sizePerThread=[2,2] -> 1024.  Thread 0 in the src
  //     contains elements [(0,0), (0,1), (1,0), and (1,1)].  We cannot express
  //     this with an encoding based on the dst shape.
  //   - OK: 32x4 sizePerThread=[4,4] -> 128.  dst with sizePerThread=[16] will
  //     contain the same elements as before.
  //
  // Users of this function require that it is symmetrical: if
  // (srcShape,srcEnc,dstShape) => dstEnc, then (dstShape,dstEnc,srcShape) =>
  // srcEnc.
  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    auto src = mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(srcEnc);
    if (!src)
      return emitOptionalError(loc,
                               "tt.reshape requires a cluster layout "
                               "operand, got ",
                               srcEnc);
    auto sizePerCore = src.getSizePerCore();
    auto coresPerGroup = src.getCoresPerGroup();
    auto groupsPerCluster = src.getGroupsPerCluster();
    auto order = src.getOrder();

    // Factor the trailing dimension: [.., L] -> [.., L/F, F]. The new last
    // dim F must come from elements that are already contiguous WITHIN a core
    // (sizePerCore[last] a multiple of F), so no data crosses a core and the
    // reshape is a nop. This is the shape isreal needs:
    //   [128] (sizePerCore=[2], cores=[64]) -> [64, 2] (sizePerCore=[1,2],
    //   cores=[64,1]) -- each core keeps its (real, imag) pair, so the
    //   following tt.split finds both halves in-core.
    auto factorTrailing = [&](unsigned F) -> bool {
      unsigned last = sizePerCore.size() - 1;
      if (sizePerCore[last] % F != 0)
        return false;
      llvm::SmallVector<unsigned> sz(sizePerCore.begin(), sizePerCore.end());
      llvm::SmallVector<unsigned> cg(coresPerGroup.begin(),
                                     coresPerGroup.end());
      llvm::SmallVector<unsigned> gc(groupsPerCluster.begin(),
                                     groupsPerCluster.end());
      sz[last] /= F;
      sz.push_back(F);
      cg.push_back(1);
      gc.push_back(1);
      // The new trailing dim is the fastest-varying (the contiguous pair);
      // every existing axis keeps its relative order behind it.
      llvm::SmallVector<unsigned> ord;
      ord.push_back(sz.size() - 1);
      for (unsigned a : order)
        ord.push_back(a);
      dstEnc =
          triton::xpu::ClusterLayoutAttr::get(getContext(), sz, cg, gc, ord);
      return true;
    };

    // Merge the trailing two dimensions: [.., P, F] -> [.., P*F]. Inverse of
    // the factor case, kept so the interface stays symmetrical.
    auto mergeTrailing = [&]() -> bool {
      unsigned last = sizePerCore.size() - 1;
      if (last == 0)
        return false;
      // Only when the last dim is entirely in-core (one core, one group) --
      // the mirror of how factorTrailing created it.
      if (coresPerGroup[last] != 1 || groupsPerCluster[last] != 1)
        return false;
      llvm::SmallVector<unsigned> sz(sizePerCore.begin(),
                                     sizePerCore.end() - 1);
      llvm::SmallVector<unsigned> cg(coresPerGroup.begin(),
                                     coresPerGroup.end() - 1);
      llvm::SmallVector<unsigned> gc(groupsPerCluster.begin(),
                                     groupsPerCluster.end() - 1);
      sz[last - 1] *= sizePerCore[last];
      llvm::SmallVector<unsigned> ord;
      for (unsigned a : order)
        if (a != last)
          ord.push_back(a > last ? a - 1 : a);
      dstEnc =
          triton::xpu::ClusterLayoutAttr::get(getContext(), sz, cg, gc, ord);
      return true;
    };

    // [.., L] -> [.., L/F, F]
    if (dstShape.size() == srcShape.size() + 1 &&
        srcShape.drop_back().equals(dstShape.drop_back(2)) &&
        dstShape[dstShape.size() - 2] * dstShape[dstShape.size() - 1] ==
            srcShape.back()) {
      if (factorTrailing(dstShape.back()))
        return success();
    }
    // [.., P, F] -> [.., P*F]
    if (srcShape.size() == dstShape.size() + 1 &&
        dstShape.drop_back().equals(srcShape.drop_back(2)) &&
        srcShape[srcShape.size() - 2] * srcShape[srcShape.size() - 1] ==
            dstShape.back()) {
      if (mergeTrailing())
        return success();
    }
    return emitOptionalError(
        loc, "tt.reshape on a cluster layout is only supported when it factors "
             "or merges the trailing dimension within a core; got srcShape/"
             "dstShape that would move data across cores");
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    return success();
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    llvm_unreachable(
        "TODO[dyq]: Add triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }

  // Layout of the two halves tt.split returns.
  //
  // Why: split removes the trailing dim of size 2 and keeps the elements, so
  // the halves carry the source layout with that dimension dropped -- no data
  // movement. tt.trans and tt.split both inherit
  // InferTypeOpWithLayoutEquivalence, whose verifier calls this, so the
  // conversion cannot simply set the result types by hand.
  //
  // How:
  //   - require a cluster layout of rank >= 2;
  //   - drop the last entry of each per-dimension array;
  //   - remove the last axis from `order` and renumber the axes above it.
  //
  // \param srcEnc layout of the tensor being split.
  // \param dstEnc receives the layout of each half.
  // \param shape source shape (its trailing dim must be the split dim).
  // \param loc diagnostic anchor.
  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    auto cluster = mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(srcEnc);
    if (!cluster || cluster.getSizePerCore().size() < 2)
      return emitOptionalError(loc,
                               "tt.split requires a cluster layout of "
                               "rank >= 2, got ",
                               srcEnc);
    const unsigned lastDim = cluster.getSizePerCore().size() - 1;
    auto dropLast = [&](ArrayRef<unsigned> dims) {
      return llvm::SmallVector<unsigned>(dims.begin(), dims.end() - 1);
    };
    llvm::SmallVector<unsigned> order;
    for (unsigned axis : cluster.getOrder())
      if (axis != lastDim)
        order.push_back(axis < lastDim ? axis : axis - 1);
    dstEnc = triton::xpu::ClusterLayoutAttr::get(
        getContext(), dropLast(cluster.getSizePerCore()),
        dropLast(cluster.getCoresPerGroup()),
        dropLast(cluster.getGroupsPerCluster()), order);
    return success();
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Dialect Initialization
//===----------------------------------------------------------------------===//

void mlir::triton::xpu::TritonXPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST // declare
#include "triton/Dialect/TritonXPU/IR/Ops.cpp.inc"
      >();

  addInterfaces<TritonXPUOpAsmInterface>();
  addInterfaces<TritonXPUInferLayoutInterface>();
}

#define GET_OP_CLASSES // define
#include "triton/Dialect/TritonXPU/IR/Ops.cpp.inc"

// verify TritonXPU ops
LogicalResult
triton::xpu::TritonXPUDialect::verifyOperationAttribute(Operation *op,
                                                        NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
