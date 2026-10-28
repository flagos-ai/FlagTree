//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

// clang-format off
#include "triton/Tools/Sys/GetEnv.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Analysis/TileAnalysis.h"
#include "triton/Analysis/VectorizabilityAnalysis.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <map>
// clang-format on

#define DEBUG_TYPE "tritonxpu-vectorize"

namespace mlir {
namespace triton {
namespace xpu {

template <typename OP> struct VOp;

#define VOP(SrcType, DstType)                                                  \
  template <> struct VOp<SrcType> {                                            \
    typedef DstType type;                                                      \
  };

// The entries live in VectorizabilityAnalysis.h so the analysis can expand the
// same list into `hasVectorForm` (step 2.1). Two copies would drift, and this
// dispatch ends in llvm_unreachable rather than falling back.
TTX_SCALAR_TO_VECTOR_OPS(VOP)

template <typename OP> struct VV2SVOp;

#define VV2SVOp(SrcType, DstType)                                              \
  template <> struct VV2SVOp<SrcType> {                                        \
    typedef DstType type;                                                      \
  };

VV2SVOp(triton::xpu::VvaddFOp, triton::xpu::SvaddFOp);
VV2SVOp(triton::xpu::VvmulFOp, triton::xpu::SvmulFOp);
VV2SVOp(triton::xpu::VvsubFOp, triton::xpu::SvsubFOp);
VV2SVOp(triton::xpu::VvmaxFOp, triton::xpu::SvmaxFOp);
VV2SVOp(triton::xpu::VvxorIOp, triton::xpu::SvxorIOp);

} // namespace xpu
} // namespace triton
} // namespace mlir

namespace mlir {

namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUVECTORIZE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

template <typename OpTy>
struct BitwiseCastToI32Pattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto originalTy = dyn_cast<RankedTensorType>(op.getType());
    if (!originalTy)
      return failure();

    auto innerVecTy = dyn_cast<VectorType>(originalTy.getElementType());
    if (!innerVecTy)
      return failure();

    if (innerVecTy.getElementType().isInteger(32))
      return failure();

    int totalBits =
        innerVecTy.getNumElements() * innerVecTy.getElementTypeBitWidth();
    if (totalBits % 32 != 0)
      return failure();

    int newNumElements = totalBits / 32;
    auto i32Ty = rewriter.getI32Type();
    auto newVecTy = VectorType::get({newNumElements}, i32Ty);
    auto newTensorTy = RankedTensorType::get(originalTy.getShape(), newVecTy,
                                             originalTy.getEncoding());

    SmallVector<Value, 2> newOperands;
    for (Value operand : op->getOperands()) {
      auto elemTy = getElementTypeOrSelf(operand);
      if (isa<VectorType>(elemTy))
        newOperands.push_back(
            rewriter.create<triton::BitcastOp>(loc, newTensorTy, operand));
      else if (auto rankedTy = dyn_cast<RankedTensorType>(operand.getType())) {
        auto newType = RankedTensorType::get(rankedTy.getShape(), i32Ty,
                                             rankedTy.getEncoding());
        newOperands.push_back(
            rewriter.create<arith::ExtSIOp>(loc, newType, operand));
      } else {
        newOperands.push_back(operand);
      }
    }

    auto newOp =
        rewriter.create<OpTy>(loc, newTensorTy, newOperands, op->getAttrs());

    auto finalCast =
        rewriter.create<triton::BitcastOp>(loc, originalTy, newOp.getResult());
    rewriter.replaceOp(op, finalCast.getResult());

    return success();
  }
};

struct TritonXPUVectorizePass
    : public impl::TritonXPUVectorizeBase<TritonXPUVectorizePass> {

  using impl::TritonXPUVectorizeBase<
      TritonXPUVectorizePass>::TritonXPUVectorizeBase;

  template <typename T>
  static decltype(auto) createBinVectorizedOp(T op, Type vectorizedTensorTy) {
    OpBuilder builder(op);
    return builder.create<typename VOp<T>::type>(
        op.getLoc(), vectorizedTensorTy, op.getLhs(), op.getRhs());
  }

  template <typename T>
  static decltype(auto) createUnaryVectorizedOp(T op, Type vectorizedTensorTy) {
    OpBuilder builder(op);
    return builder.create<typename VOp<T>::type>(
        op.getLoc(), vectorizedTensorTy, op.getOperand());
  }

  static decltype(auto) createLibdeviceOp(triton::ExternElementwiseOp &op,
                                          const llvm::StringRef &symbol,
                                          Type vectorizedTensorTy) {
    OpBuilder builder(op);
    return builder.create<triton::ExternElementwiseOp>(
        op.getLoc(), vectorizedTensorTy, op.getOperands(), op.getLibname(),
        op.getLibpath(), symbol, op.getPure());
  }

  // TODO[dyq]: open isMultipleOfBank
  // bool isMultipleOfBank(ModuleOp &mod) {
  //   bool res = false;
  //   mod.walk([&](arith::CmpIOp cmpiOp) {
  //     auto lhs = cmpiOp.getLhs();
  //     auto rhs = cmpiOp.getRhs();

  //     if (cmpiOp.predicate() == arith::CmpIPredicate::slt) {
  //       auto lhsShape = lhs.getType().cast<RankedTensorType>().getShape();

  //       if (lhsShape.size() == 2 && lhsShape[0] == 1) { // inner Cmp
  //       Calculation
  //         if (auto rhsOp =
  //                 rhs.getDefiningOp<arith::ConstantOp>()) { // Static Rnumel
  //           auto denseAttr = rhsOp.getValue().dyn_cast<DenseElementsAttr>();
  //           auto elemPerCore =
  //               *denseAttr.getValues<int>().begin();     // get rnumel int
  //           res = (elemPerCore & (bufferSize - 1)) == 0; // check multiple?
  //         }
  //       }
  //     }
  //   });
  //   return res;
  // }

  RankedTensorType getVectorType(Type tensorType, unsigned _elemWidth = 0,
                                 bool useElemTy = false) {
    // A TLE tensor can arrive with no encoding at all; `withTLEDefaultEncoding`
    // supplies the reading the lowering gives it, so the count asked for here
    // is the same one `TypeConverter` will hand the emitted code.
    unsigned numElems =
        getTotalElemsPerThread(withTLEDefaultEncoding(tensorType));
    Type elemTy = getElementTypeOrSelf(tensorType);
    auto elemWidth =
        _elemWidth == 0 ? elemTy.getIntOrFloatBitWidth() : _elemWidth;
    auto vectorWidth = 512 / elemWidth;

    RankedTensorType newTensorTy;

    if (numElems % vectorWidth == 0 &&
        numElems != 0) { // normal vector<16xf32>/vector<32xf16>
      // Step 1. getVectorType
      VectorType newVectorType = mlir::VectorType::get(vectorWidth, elemTy);

      // Step 2. getShape
      RankedTensorType oriTensorTy = mlir::cast<RankedTensorType>(tensorType);
      auto oriShape = oriTensorTy.getShape();
      llvm::SmallVector<int64_t, 4> newShape(oriShape.begin(), oriShape.end());
      auto rank = oriShape.size();
      newShape[rank - 1] /= vectorWidth;

      // Step 3. getEncoding
      //
      // TLE IR carries unencoded tensors: the TLE type conversion leaves LM
      // ptr-tensors unencoded, so the value crossing into a `tt.store` gets a
      // `convert_layout tensor<64x256xf16, #cluster1> -> tensor<64x256xf16>`.
      // There is no per-core split to rewrite in that case -- the shape math
      // above is a tensor-level property -- so the absent encoding stays absent
      // rather than being invented.
      Attribute newEncoding;
      if (auto oriEncoding =
              mlir::dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
                  oriTensorTy.getEncoding())) {
        auto sizePerCore = oriEncoding.getSizePerCore().vec();
        auto corePerGroup = oriEncoding.getCoresPerGroup().vec();
        auto groupsPerCluster = oriEncoding.getGroupsPerCluster().vec();
        auto order = oriEncoding.getOrder().vec();

        sizePerCore[rank - 1] =
            std::max(1, int(sizePerCore[rank - 1] / vectorWidth));

        newEncoding = triton::xpu::ClusterLayoutAttr::get(
            tensorType.getContext(), sizePerCore, corePerGroup,
            groupsPerCluster, order);
      }

      // Step 4. create RankedTensorType
      newTensorTy =
          useElemTy
              ? RankedTensorType::get(newShape, elemTy, newEncoding)
              : RankedTensorType::get(newShape, newVectorType, newEncoding);
    } else if (numElems == 1) { // special vector<1xf32>
      // Step 1. getVectorType
      VectorType newVectorType = mlir::VectorType::get(1, elemTy);
      // Step 2. getEncoding
      auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
          tensorType.getContext(), ArrayRef<unsigned>{1}, ArrayRef<unsigned>{4},
          ArrayRef<unsigned>{16}, ArrayRef<unsigned>{0});
      // Step 3. create RankedTensorType
      newTensorTy = useElemTy
                        ? RankedTensorType::get(1, elemTy, newEncoding)
                        : RankedTensorType::get(1, newVectorType, newEncoding);
    } else {
      llvm_unreachable(
          "Only Supported vector<32xTy> or vector<16xTy> or vector<1xTy>");
    }
    return newTensorTy;
  }

  // `getVectorType`'s precondition, asked *before* calling it. That function
  // reads the element type's bit width unconditionally, so a `!tt.ptr<f16>`
  // element -- which reaches the boundary check as a loop-carried `addptr`
  // result, findings 1.69 -- trips an MLIR assertion, and a per-core count that
  // is neither 1 nor a multiple of the lane count ends in its
  // `llvm_unreachable`. Both mean "no vector form", which is exactly what the
  // callers want to reject, so none of them may be reached by making the call.
  bool vectorFormExists(RankedTensorType tensorTy) {
    if (tensorTy.getRank() == 0)
      return false;
    Type elemTy = getElementTypeOrSelf(tensorTy);
    if (!elemTy.isIntOrFloat())
      return false;
    unsigned width = elemTy.getIntOrFloatBitWidth();
    if (width == 0 || width > 512)
      return false;
    unsigned numElems =
        getTotalElemsPerThread(withTLEDefaultEncoding(Type(tensorTy)));
    if (numElems == 1)
      return true;
    if (numElems == 0 || numElems % (512 / width) != 0)
      return false;
    // An absent encoding is admissible -- `getVectorType` leaves it absent and
    // reads the count through `withTLEDefaultEncoding`, which is how a TLE
    // store-facing tensor is carried. A *present* encoding that is not a
    // cluster layout is not: `getVectorType` would drop it silently.
    Attribute encoding = tensorTy.getEncoding();
    return !encoding ||
           !!mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(encoding);
  }

  // Returns false without touching the IR when some member of the set has no
  // vector form; true once the whole set has been retyped.
  bool processOpVecTy(OperationTree &vectorizedOps, ModuleOp &mod) {
    // "Terminal state Scalar => don't convert", checked before any mutation.
    // The dispatch below ends in llvm_unreachable, so bailing out part way
    // through would be worse than crashing: the ops already rewritten would
    // feed scalar consumers with no materialisation edge between them. The
    // whole set therefore has to be admissible up front, and `coverage == None`
    // is exactly the `.Default` arm mirrored in one place
    // (VectorizabilityAnalysis.cpp:59 processOpVecTyCoverage).
    //
    // Unreachable on today's 13 probes -- 1.53/1.56 measured `result{none=0}`
    // at every site, and the per-op cut additionally requires `Full` for every
    // segment member -- so this guard trades a crash for a lost vectorization
    // only on kinds the pass cannot emit anyway. If the mirror ever drifts
    // *narrow* (predicate says none, dispatch does handle it) that shows up as
    // this report plus scalar code rather than as wrong code; drift the other
    // way still reaches the `.Default` assertion.
    for (Operation *op : vectorizedOps) {
      if (processOpVecTyCoverage(op, this->tleSmemVec, this->tleVec) !=
          VecTyCoverage::None)
        continue;
      if (vecReportEnabled())
        llvm::errs() << "[VecTyRefuse] set=" << vectorizedOps.size()
                     << " op=" << op->getName() << " loc=" << op->getLoc()
                     << "\n";
      return false;
    }

    for (auto *op : vectorizedOps) {
      TypeSwitch<Operation *>(op)
          .Case<triton::xpu::GM2LMOp>([&](auto gm2lmOp) { (void)gm2lmOp; })
          .Case<triton::xpu::GM2LMMaskOp>(
              [&](auto gm2lmmaskOp) { (void)gm2lmmaskOp; })
          .Case<triton::xpu::LoadOp>([&](auto loadOp) {
            auto newVectorizedTensorTy =
                getVectorType(loadOp.getResult().getType());
            loadOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<triton::xpu::LM2GMOp>([&](auto lm2gmOp) { (void)lm2gmOp; })
          .Case<triton::xpu::LM2GMMaskOp>(
              [&](auto lm2gmmaskOp) { (void)lm2gmmaskOp; })
          .Case<triton::xpu::StoreOp>([&](auto storeOp) { (void)storeOp; })
          // The TLE leaf load. Its vector form addresses the whole per-core LM
          // buffer, so the per-element pointer tensor goes away with the scalar
          // load -- slot i of the result is LM[i*W : (i+1)*W], the contract
          // XPUTLELocalPtrOpConversion already hands out (`lmBase + i`, indices
          // ignored) and `test_tle_vector_add_1d` guards numerically. The
          // `tle_local_ptr` is left alone: the ams kernels point several loads
          // and a store at one buffer, so canonicalize drops it once the other
          // uses are gone.
          .Case<triton::LoadOp>([&](triton::LoadOp loadOp) {
            auto lp = getTLELocalPtrThroughCvt(loadOp.getPtr());
            if (!lp)
              llvm::report_fatal_error(
                  "tt.load reached processOpVecTy without a tle_local_ptr "
                  "producer; the coverage predicate and this dispatch "
                  "disagree");
            // SM gather (non-slice index into a cluster-shared buffer): keep it
            // a vector tt.load with a marker, like the bf16 leaf below, and let
            // XPUTLETritonLoadOpConversion emit one SM vgather (`vgathers`) per
            // result vector with the index as a per-lane offset. This keeps the
            // affine chain vector -- the gather was the one op forcing it
            // scalar. The index stays scalar per register; the lowering builds
            // the offset vector from it.
            if (tleSmemGather(lp)) {
              Type facingSegTy = tleEncodedFacingTypeOrNull(loadOp.getResult());
              if (!facingSegTy)
                llvm::report_fatal_error(
                    "TLE SM gather has no encoded sibling to size it");
              auto segVecTy = getVectorType(facingSegTy);
              OpBuilder segBuilder(loadOp);
              Value segPtr = makeVectorTLELocalPtr(
                  segBuilder, lp, cast<RankedTensorType>(segVecTy));
              auto segLd = segBuilder.create<triton::LoadOp>(
                  loadOp.getLoc(), segPtr, loadOp.getCache(), loadOp.getEvict(),
                  loadOp.getIsVolatile());
              segLd->setAttr("xpu.sm_gather", segBuilder.getUnitAttr());
              loadOp.getResult().replaceAllUsesWith(segLd.getResult());
              loadOp.erase();
              return;
            }
            // An SM buffer holds the whole array once per cluster, so its index
            // tensor is what selects the elements: `tleSmemSliceOffset` is the
            // shared admission check (the coverage predicate asks the same one)
            // and hands back the uniform addend, which becomes the op's element
            // offset. LM buffers get a null offset -- the lowering ignores
            // their indices because the staging DMA already placed this core's
            // slice.
            Value smOff;
            if (!tleSmemSliceOffset(lp, smOff, this->tleSmemVec))
              llvm::report_fatal_error(
                  "TLE vload on a scope=smem buffer whose index is not "
                  "[uniform +] arange(0, n); the coverage predicate and this "
                  "dispatch disagree");
            // Mirror of the store side below: an output buffer's
            // `tle_local_ptr` is unencoded, so `tt.load` off it has an
            // unencoded result, and the default reading of that splits the
            // *last* dim across the cores -- once that dim has been divided by
            // the vector width it reports the whole tile per core. The encoding
            // on the cvt the TLE type conversion left next to the load is what
            // sizes the `tle_vload` (XPUTLEVLoadOpConversion reads the
            // converted type); no sibling means no size to trust, so stop
            // rather than let `withTLEDefaultEncoding` guess one.
            Type facingTy = tleEncodedFacingTypeOrNull(loadOp.getResult());
            if (!facingTy)
              llvm::report_fatal_error(
                  "TLE vload has no encoded sibling to size it: the TLE type "
                  "conversion materializes a convert_layout next to every "
                  "unencoded TLE tensor, and the vector width is read off it");
            auto newVectorizedTensorTy = getVectorType(facingTy);
            OpBuilder builder(loadOp);
            // A bf16 buffer read into f32 registers (`xpu.lm_bf16`, stamped by
            // tritonxpu-tle-dtype-convert) has to stay a tt.load, retyped per
            // vector slot: the placement is bit arithmetic XPU3 has no
            // instruction for, so XPUTLETritonLoadOpConversion fuses it into
            // the load with VecBF16ToFP32{,Unordered}. `tle_vload` cannot
            // express it -- its result type is the buffer's, i.e. bf16 -- so
            // converting one would read the bf16 bytes as f32 and be silently
            // wrong. smem buffers keep the tle_vload form.
            if (loadOp->hasAttr("xpu.lm_bf16") &&
                !tleBufferIsSmem(lp.getBuffer())) {
              Value vecPtr = makeVectorTLELocalPtr(
                  builder, lp, cast<RankedTensorType>(newVectorizedTensorTy));
              auto newLd = builder.create<triton::LoadOp>(
                  loadOp.getLoc(), vecPtr, loadOp.getCache(), loadOp.getEvict(),
                  loadOp.getIsVolatile());
              newLd->setAttr("xpu.lm_bf16", builder.getUnitAttr());
              loadOp.getResult().replaceAllUsesWith(newLd.getResult());
              loadOp.erase();
              return;
            }
            auto vload = builder.create<triton::xpu::TLEVLoadOp>(
                loadOp.getLoc(), newVectorizedTensorTy, lp.getBuffer(),
                /*loopIndex=*/Value(), /*smElemOffset=*/smOff);
            loadOp.getResult().replaceAllUsesWith(vload.getResult());
            loadOp.erase();
          })
          // The TLE store, mirror of the leaf load above. Unlike
          // `xpu::StoreOp`, which already is the final op and simply takes a
          // vector operand, this one has to be rewritten: only `tle_vstore`
          // addresses the buffer as whole vectors, writing slot i to LM[i*W :
          // (i+1)*W] -- the same slot numbering `tle_vload` reads by, which
          // `tleSlotIndex` is the single place to turn into an LM index.
          .Case<triton::StoreOp>([&](triton::StoreOp storeOp) {
            auto lp =
                storeOp.getPtr().getDefiningOp<triton::xpu::TLELocalPtrOp>();
            if (!lp || tleBufferIsSmem(lp.getBuffer()))
              llvm::report_fatal_error(
                  "tt.store reached processOpVecTy without a per-core "
                  "tle_local_ptr producer; the coverage predicate and this "
                  "dispatch disagree (a scope=smem buffer has no store form)");
            // Past the encoding-stripping convert_layout:
            // XPUTLEVStoreOpConversion sizes the write from the value's own
            // type, and only the encoded one carries the per-core split. Taking
            // the unencoded result of that cvt would make it write the whole
            // tile per core.
            Value value = storeOp.getValue();
            while (auto cvt = dyn_cast_or_null<triton::xpu::ConvertLayoutOp>(
                       value.getDefiningOp())) {
              if (!cast<RankedTensorType>(cvt.getResult().getType())
                       .getEncoding())
                value = cvt.getOperand();
              else
                break;
            }
            OpBuilder builder(storeOp);
            // Symmetric to the bf16 load leaf: keep a vector tt.store so
            // XPUTLETritonStoreOpConversion can fuse the f32 -> bf16 placement
            // with VecFP32ToBF16{,Unordered,Slow}. The coverage predicate
            // already refused the odd register counts the paired forms cannot
            // express, so no store reaching here needs a second check.
            if (storeOp->hasAttr("xpu.lm_bf16")) {
              auto vecTy = cast<RankedTensorType>(value.getType());
              Value vecPtr = makeVectorTLELocalPtr(builder, lp, vecTy);
              auto newSt = builder.create<triton::StoreOp>(
                  storeOp.getLoc(), vecPtr, value, storeOp.getCache(),
                  storeOp.getEvict());
              newSt->setAttr("xpu.lm_bf16", builder.getUnitAttr());
              storeOp.erase();
              return;
            }
            builder.create<triton::xpu::TLEVStoreOp>(storeOp.getLoc(),
                                                     lp.getBuffer(), value,
                                                     /*loopIndex=*/Value());
            storeOp.erase();
          })
          .Case<ARITH_BINARY_FLOAT_OP>([&](auto binOp) {
            auto newVectorizedTensorTy =
                getVectorType(binOp.getResult().getType());
            auto newBinOp = createBinVectorizedOp(binOp, newVectorizedTensorTy);
            binOp.replaceAllUsesWith(newBinOp.getResult());
            binOp.erase();
          })
          .Case<ARITH_BINARY_INT_OP>([&](auto binOp) {
            auto newVectorizedTensorTy =
                getVectorType(binOp.getResult().getType());
            auto newBinOp = createBinVectorizedOp(binOp, newVectorizedTensorTy);
            binOp.replaceAllUsesWith(newBinOp.getResult());
            binOp.erase();
          })
          .Case<MATH_UNARY_OP>([&](auto unaryOp) {
            auto newVectorizedTensorTy =
                getVectorType(unaryOp.getResult().getType());
            auto newUnaryOp =
                createUnaryVectorizedOp(unaryOp, newVectorizedTensorTy);
            unaryOp.replaceAllUsesWith(newUnaryOp.getResult());
            unaryOp.erase();
          })
          .Case<arith::SIToFPOp>([&](auto unaryOp) {
            auto newVectorizedTensorTy =
                getVectorType(unaryOp.getResult().getType());
            auto newUnaryOp =
                createUnaryVectorizedOp(unaryOp, newVectorizedTensorTy);
            unaryOp.replaceAllUsesWith(newUnaryOp.getResult());
            unaryOp.erase();
          })
          // Same body as SIToFPOp, but worth its own Case for the way the type
          // is derived: `getVectorType` is applied to this op's own result
          // type, so an f16 -> f32 extf becomes tensor<Nxvector<16xf32>> from a
          // tensor<N/2xvector<32xf16>> operand. The lane counts differ on
          // purpose -- VExtFOpConversion splits each input vector into a
          // low/high pair -- and only the scalar element count survives the
          // cast. Without a Case here both would land on the `.Default`
          // llvm_unreachable, since `VOp<arith::ExtFOp>` is in
          // TTX_SCALAR_TO_VECTOR_OPS and the coverage guard at the top of this
          // function therefore lets them through.
          .Case<arith::ExtFOp, arith::TruncFOp>([&](auto castOp) {
            auto newVectorizedTensorTy =
                getVectorType(castOp.getResult().getType());
            auto newCastOp =
                createUnaryVectorizedOp(castOp, newVectorizedTensorTy);
            castOp.replaceAllUsesWith(newCastOp.getResult());
            castOp.erase();
          })
          .Case<arith::ConstantOp>([&](auto constOp) {
            auto newVectorizedTensorTy =
                getVectorType(constOp.getResult().getType());
            OpBuilder builder(constOp);
            auto newConstOp = builder.create<triton::xpu::VConstOp>(
                constOp.getLoc(), newVectorizedTensorTy, constOp.getValue());
            constOp.replaceAllUsesWith(newConstOp.getResult());
            constOp.erase();
          })
          .Case<triton::SplatOp>([&](auto splatOp) {
            auto newVectorizedTensorTy =
                getVectorType(splatOp.getResult().getType());
            OpBuilder builder(splatOp);
            auto newSplatOp = builder.create<triton::xpu::VSplatOp>(
                splatOp.getLoc(), newVectorizedTensorTy, splatOp.getOperand());
            splatOp.replaceAllUsesWith(newSplatOp.getResult());
            splatOp.erase();
          })
          .Case<scf::ForOp>([&](auto forOp) {
            auto forBody = forOp.getBody();
            auto forArgs = forBody->getArguments();
            // TODO[dyq]: check getIterOperands -> getInitArgs
            auto iterArgsInitValues = forOp.getInitArgs();
            for (int i = 0; i < iterArgsInitValues.size(); ++i) {
              Value iterArgInitValue = iterArgsInitValues[i];
              auto newVectorizedTensorTy = iterArgInitValue.getType();

              // 1. Change Input Iter Args Type
              forArgs[i + 1].setType(newVectorizedTensorTy);

              // 2. Change Output Type
              forOp.getResult(i).setType(newVectorizedTensorTy);
            }
          })
          .Case<scf::IfOp>([&](auto ifOp) {
            // 1. Get Terminator Type
            Region &thenRegion = ifOp.getThenRegion();
            Block &thenBlock = thenRegion.front();
            Operation *thenTerminator = thenBlock.getTerminator();

            // 2. Change Output Type
            if (auto yieldOp = dyn_cast<scf::YieldOp>(thenTerminator)) {
              for (int i = 0; i < yieldOp.getOperands().size(); ++i) {
                if (auto prevOp = yieldOp.getOperands()[i].getDefiningOp()) {
                  auto resType = prevOp->getResult(0).getType();
                  ifOp.getResult(i).setType(resType);
                }
              }
            }
          })
          .Case<scf::YieldOp>([&](auto yieldOp) { (void)yieldOp; })
          .Case<triton::xpu::ConvertLayoutOp>([&](auto cvtOp) {
            auto newVectorizedTensorTy =
                getVectorType(cvtOp.getResult().getType());
            cvtOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<arith::SelectOp>([&](auto selectOp) {
            auto newVectorizedTensorTy =
                getVectorType(selectOp.getResult().getType());
            OpBuilder builder(selectOp);
            auto newSelectOp = builder.create<triton::xpu::VSelectOp>(
                selectOp.getLoc(), newVectorizedTensorTy,
                selectOp.getCondition(), selectOp.getTrueValue(),
                selectOp.getFalseValue());
            selectOp.replaceAllUsesWith(newSelectOp.getResult());
            selectOp.erase();
          })
          .Case<arith::CmpFOp>([&](auto cmpFOp) {
            auto rhsTy = cmpFOp.getRhs().getType();
            Type elemTy = getElementTypeOrSelf(getElementTypeOrSelf(rhsTy));
            auto newVectorizedTensorTy = getVectorType(
                cmpFOp.getResult().getType(), elemTy.getIntOrFloatBitWidth());
            OpBuilder builder(cmpFOp);
            auto newCmpFOp = builder.create<triton::xpu::VCmpFOp>(
                cmpFOp.getLoc(), newVectorizedTensorTy, cmpFOp.getPredicate(),
                cmpFOp.getLhs(), cmpFOp.getRhs());
            cmpFOp.replaceAllUsesWith(newCmpFOp.getResult());
            cmpFOp.erase();
          })
          .Case<arith::TruncIOp>([&](auto truncIOp) {
            auto extElemwiseOp = cast<triton::ExternElementwiseOp>(
                truncIOp.getIn().getDefiningOp());

            auto newVectorizedTensorTy =
                getVectorType(truncIOp.getResult().getType(), 32);

            OpBuilder builder(extElemwiseOp);

            auto newExtElemwiseOp = builder.create<triton::ExternElementwiseOp>(
                extElemwiseOp.getLoc(), newVectorizedTensorTy,
                extElemwiseOp.getOperands().front(), extElemwiseOp.getLibname(),
                extElemwiseOp.getLibpath(), "_ZN3xpu6visnanEDv16_f",
                extElemwiseOp.getPure());

            truncIOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
            truncIOp.erase();
            extElemwiseOp.erase();
          })
          .Case<triton::xpu::CmpFOp>([&](auto cmpFOp) {
            auto rhsTy = cmpFOp.getRhs().getType();
            Type elemTy = getElementTypeOrSelf(getElementTypeOrSelf(rhsTy));
            auto newVectorizedTensorTy =
                getVectorType(cmpFOp.getResult().getType(),
                              elemTy.getIntOrFloatBitWidth(), true);
            OpBuilder builder(cmpFOp);
            auto newCmpFOp = builder.create<triton::xpu::VCmpFOp>(
                cmpFOp.getLoc(), newVectorizedTensorTy, cmpFOp.getPredicate(),
                cmpFOp.getLhs(), cmpFOp.getRhs());
            cmpFOp.replaceAllUsesWith(newCmpFOp.getResult());
            cmpFOp.erase();
          })
          .Case<triton::xpu::BroadcastOp>([&](auto broadCastOp) {
            auto newVectorizedTensorTy =
                getVectorType(broadCastOp.getResult().getType());
            broadCastOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<triton::ExpandDimsOp>([&](auto expandOp) {
            auto newVectorizedTensorTy =
                getVectorType(expandOp.getResult().getType());
            expandOp.getResult().setType(newVectorizedTensorTy);
          })
          .Case<triton::ExternElementwiseOp>([&](auto extElemwiseOp) {
            auto symbol = extElemwiseOp.getSymbol();
            OpBuilder builder(extElemwiseOp);
            auto newVectorizedTensorTy =
                getVectorType(extElemwiseOp.getResult().getType());
            if (symbol == "_ZN3xpu5tanhfEf") {
              auto newExtElemwiseOp =
                  createLibdeviceOp(extElemwiseOp, "_ZN3xpu6vtanhfEDv16_f",
                                    newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu4tanfEf") {
              auto newExtElemwiseOp = createLibdeviceOp(
                  extElemwiseOp, "_ZN3xpu5vtanfEDv16_f", newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu3erfEf") {
              auto newExtElemwiseOp = createLibdeviceOp(
                  extElemwiseOp, "_ZN3xpu4verfEDv16_f", newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu5atanfEf") {
              auto newExtElemwiseOp =
                  createLibdeviceOp(extElemwiseOp, "_ZN3xpu6vatanfEDv16_f",
                                    newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu5isinfEf") {
              auto newExtElemwiseOp =
                  createLibdeviceOp(extElemwiseOp, "_ZN3xpu6visinfEDv16_f",
                                    newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu5isnanEf") {
              auto inTy = extElemwiseOp.getOperands().front().getType();
              Type elemTy = getElementTypeOrSelf(getElementTypeOrSelf(inTy));
              auto newVectorizedTensorTy =
                  getVectorType(extElemwiseOp.getResult().getType(),
                                elemTy.getIntOrFloatBitWidth(), true);

              auto newExtElemwiseOp =
                  createLibdeviceOp(extElemwiseOp, "_ZN3xpu6visnanEDv16_f",
                                    newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else if (symbol == "_ZN3xpu6rsqrtfEf") {
              auto newExtElemwiseOp = createLibdeviceOp(
                  extElemwiseOp, "_ZN3xpu12vrsqrtf_fastEDv16_f",
                  newVectorizedTensorTy);
              extElemwiseOp.replaceAllUsesWith(newExtElemwiseOp.getResult());
              extElemwiseOp.erase();
            } else {
              LLVM_DEBUG(llvm::dbgs()
                         << "[Vectorization]: Can not Convert Symbol " << symbol
                         << " to Vfunc\n");
            }
          })
          // Kept as a last-resort assertion: the scan above should have made
          // this arm unreachable by construction.
          .Default([&](auto &op) {
            LLVM_DEBUG(op->dump());
            llvm_unreachable(
                "[Vectorization]: Unsupported Operation Type To VecType");
          });
    }
    return true;
  }

  // Default off: the model can only ever refuse a vectorization the pass would
  // otherwise perform, so leaving it off keeps the emitted code unchanged.
  bool vecCostModelEnabled() {
    return mlir::triton::tools::getBoolEnv("TRITONXPU_VEC_COST");
  }

  // Whether retyping `vectorizedOps` to vectors executes fewer operations than
  // it costs. Counted dynamically, per core:
  //
  //   scalarCost = sum over segment values of elements per core
  //   vectorCost = sum over segment values of ceil(elements per core / width)
  //              + (combine region ops + outputs) * width   [reduce roots only]
  //
  // The reduce term prices the horizontal collapse, which the region-
  // interpreting lowering emits as one scalar replay of the whole combine per
  // lane (ReduceOpToLLVM::collapseVectorsJointly) and is the only part of a
  // vectorized reduce that does not shrink with the width.
  //
  // Measured on the five golden probes (2026-08-03): this rejects nothing. It
  // approves welford (scalar=512 vector=272) even though welford's vector
  // variant measures 14 vector spills and +1110 static instructions, because
  // the collapse it charges 240 for replaces 12 region ops * 64 elements of
  // serial scalar combining -- dynamically the vector variant really does do
  // less work, and its regression is spills, not operation count.
  //
  // The trip-count-weighted estimate golden.py now reports (`dyn`) agrees with
  // the model: welford 4200 scalar against 1614 vectorized, i.e. 2.6x less work
  // executed, against the model's 512 vs 272. So approving welford is right on
  // operation count and the veto that welford needs is a register-pressure one,
  // which this model does not attempt.
  //
  // Static instruction count cannot be the objective here: with vectorization
  // vetoed everywhere the probes measure add 79, softmax 400, layernorm 523
  // against 99 / 1774 / 677 vectorized, i.e. the scalar variants are smaller in
  // every case because UnrollControl leaves them as rerolled loops. A model
  // fitted to static size would forbid all vectorization.
  //
  // Kept default-off. A veto can only refuse a rewrite, so with the flag off
  // the emitted code is unchanged (verified byte-identical on all five probes).
  bool vectorizationIsProfitable(const OperationTree &vectorizedOps,
                                 triton::xpu::ReduceOp redOp) {
    int64_t scalarCost = 0;
    int64_t vectorCost = 0;
    int64_t collapseWidth = 0;
    for (Operation *op : vectorizedOps) {
      for (Value res : op->getResults()) {
        auto tensorTy = dyn_cast<RankedTensorType>(res.getType());
        if (!tensorTy)
          continue;
        Type elemTy = getElementTypeOrSelf(tensorTy);
        // Values that keep their scalar element type either way cost the same
        // on both sides, so they cannot move the comparison.
        if (!vectorizedTyValid(elemTy))
          continue;
        int64_t elems = getNumRegs(tensorTy);
        if (elems == 0)
          continue; // pointers and other non-data values
        int64_t width = getVectorWidth(elemTy);
        scalarCost += elems;
        vectorCost += (elems + width - 1) / width;
        collapseWidth = std::max(collapseWidth, width);
      }
    }

    if (redOp && collapseWidth > 0) {
      int64_t regionOps = 0;
      for (Block &block : redOp.getCombineOp().getBlocks())
        for (Operation &op : block)
          if (!isa<triton::xpu::ReduceReturnOp>(op))
            ++regionOps;
      int64_t numOutputs = redOp.getNumResults();
      vectorCost += (regionOps + numOutputs) * collapseWidth;
    }

    LLVM_DEBUG(llvm::dbgs() << "[Vectorization]: cost scalar=" << scalarCost
                            << " vector=" << vectorCost << "\n");
    return vectorCost < scalarCost;
  }

  // One closure shared by several roots. The reduce path below otherwise calls
  // vectorizeAndProcessOpVecTy once per operand, each with a fresh
  // visited/vectorizedOps, so an op reachable from two operands gets retyped
  // twice -- welford's three reduce operands share the mask-zero constant, and
  // the second retype turns its element type into vector<16xvector<16xf32>>,
  // which asserts in getVectorWidth. Scoped to the region path
  // (TRITONXPU_REDUCE_REGION) so single-operand reduces stay byte-identical.
  void vectorizeAndProcessOpVecTyShared(ModuleOp &mod,
                                        ArrayRef<Operation *> rootOps,
                                        Type rootOpTy, std::string logMessage,
                                        triton::xpu::ReduceOp redOp = {}) {
    VectorizabilityAnalysis analysis(ReduceVec, dumpFlag,
                                     vectorFitsReduceOperand, vectorFitsValue,
                                     this->tleSmemVec, this->tleVec);
    if (!vectorFitsRoot(rootOpTy)) {
      if (vecReportEnabled() && !rootOps.empty())
        reportVecRoot("in-vectorize", "reduce-operand-shared", rootOps.front(),
                      rootOpTy, /*eligible=*/false, /*closureSize=*/0);
      return;
    }

    OperationTree visited;
    OperationTree vectorizedOps;
    for (Operation *rootOp : rootOps)
      if (!analysis.getVectorizableClosure(rootOp, visited, vectorizedOps)) {
        if (vecReportEnabled())
          reportVecRoot("in-vectorize", "reduce-operand-shared", rootOp,
                        rootOpTy, /*eligible=*/true, /*closureSize=*/0);
        return; // all-or-nothing: the reduce needs every operand in vector form
      }

    if (vecReportEnabled() && !rootOps.empty())
      reportVecRoot("in-vectorize", "reduce-operand-shared", rootOps.front(),
                    rootOpTy, /*eligible=*/true, vectorizedOps.size());

    if (vecCostModelEnabled() &&
        !vectorizationIsProfitable(vectorizedOps, redOp)) {
      LLVM_DEBUG(llvm::dbgs() << "[Vectorization]: not profitable, keeping the "
                                 "segment scalar\n");
      return;
    }

    LLVM_DEBUG(llvm::errs() << logMessage << "\n");
    processOpVecTy(vectorizedOps, mod);
  }

  void vectorizeAndProcessOpVecTy(ModuleOp &mod, Operation *rootOp,
                                  Type rootOpTy, std::string logMessage,
                                  triton::xpu::ReduceOp redOp = {}) {
    // Derived, not passed in: the only two call sites are the store walk and
    // the reduce-operand loop, and the root op kind already tells them apart.
    // The TLE store walk joins the first: same site, different store op.
    const char *site =
        isa_and_nonnull<triton::xpu::StoreOp, triton::StoreOp>(rootOp)
            ? "store"
            : "reduce-operand";
    VectorizabilityAnalysis analysis(ReduceVec, dumpFlag,
                                     vectorFitsReduceOperand, vectorFitsValue,
                                     this->tleSmemVec, this->tleVec);
    if (!vectorFitsRoot(rootOpTy)) {
      if (vecReportEnabled() && rootOp) {
        reportVecRoot("in-vectorize", site, rootOp, rootOpTy,
                      /*eligible=*/false, /*closureSize=*/0);
        reportOpenGate(rootOp, site, "root-ineligible", /*partial=*/0);
      }
      return;
    }

    OperationTree visited;
    OperationTree vectorizedOps;

    bool closed =
        analysis.getVectorizableClosure(rootOp, visited, vectorizedOps);
    if (vecReportEnabled() && rootOp)
      reportVecRoot("in-vectorize", site, rootOp, rootOpTy, /*eligible=*/true,
                    closed ? int64_t(vectorizedOps.size()) : 0);
    if (!closed) {
      reportOpenGate(rootOp, site, "walk-vetoed", vectorizedOps.size());
      cutVectorSegment(rootOp, site, mod);
      return;
    }

    LLVM_DEBUG({
      llvm::errs() << logMessage << "\n";
      if (dumpFlag) {
        for (auto vecOp : vectorizedOps)
          vecOp->dump();
      }
    });

    auto encoding = mlir::cast<RankedTensorType>(rootOpTy).getEncoding();

    if (vecCostModelEnabled() &&
        !vectorizationIsProfitable(vectorizedOps, redOp)) {
      LLVM_DEBUG(llvm::dbgs() << "[Vectorization]: not profitable, keeping the "
                                 "segment scalar\n");
      return;
    }

    processOpVecTy(perOpDecisionSet(rootOp, site, vectorizedOps), mod);
  }

  // Step 3.2, stage 1. The set handed to the rewrite comes from the closure
  // walk today; with `per-op-decision` it comes from the Vector-Flow
  // partition's terminal state instead (`VecSetDomain::term`), which is what
  // makes processOpVecTy's `.Default` reachable at all.
  //
  // Stage 1 only switches where the two sets rewrite the same ops -- differing
  // in `rewriteNoopKind` members alone, which the rewrite cannot act on. That
  // makes the switch an equivalence, checkable by byte identity, and confines
  // the sites where the sets genuinely disagree (the 10 with `onlyTerm>0` in
  // findings 1.56) to stage 2, where boundary materialisation has to be
  // handled rather than assumed away.
  //
  // The partition is rebuilt per root, not once per function: processOpVecTy
  // erases and replaces ops, so a partition built ahead of the root loop would
  // answer `stateOf` on dangling Values.
  // The value whose class the root's verdict is about: the store's value
  // operand, or the reduce operand's result.
  static Value decisionKeyValue(Operation *rootOp) {
    if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(rootOp))
      return storeOp.getValue();
    if (rootOp->getNumResults() > 0)
      return rootOp->getResult(0);
    return {};
  }

  // Non-const because processOpVecTy takes its set by mutable reference.
  OperationTree &perOpDecisionSet(Operation *rootOp, const char *site,
                                  OperationTree &closure) {
    if (!perOpDecision || !rootOp)
      return closure;
    auto funcOp = rootOp->getParentOfType<triton::FuncOp>();
    if (!funcOp)
      return closure;
    Value keyValue = decisionKeyValue(rootOp);
    if (!keyValue)
      return closure;

    VectorFlowAnalysis vflow(vectorFitsValue, this->tleSmemVec, this->tleVec);
    vflow.run(funcOp);
    decisionDomain = buildVecSetDomain(keyValue, vflow, closure);
    bool equivalent = sameActingSet(closure, decisionDomain.term);
    if (vecReportEnabled())
      llvm::errs() << "[VecDecide] " << funcOp.getName() << " site=" << site
                   << " root=" << rootOp->getName()
                   << " closure=" << closure.size()
                   << " term=" << decisionDomain.term.size()
                   << " source=" << (equivalent ? "term" : "closure-deferred")
                   << (equivalent ? "" : " reason=sets-differ")
                   << " loc=" << rootOp->getLoc() << "\n";
    if (!equivalent)
      return closure;

    // Handed in *closure order*, not in the order the domain walk found them.
    // processOpVecTy rewrites in insertion order and reads each op's already
    // rewritten operands, so a consumer that arrives before its producer is
    // built against a still-scalar operand: handing the same set in
    // cone-walk (root-first) order fails layernorm's verifier with
    // "'triton_xpu.vvaddf' op operand #0 must be fixed-length vector ...".
    // The set is the decision; the order is the rewrite's own precondition.
    decisionSet.clear();
    for (Operation *op : closure)
      if (decisionDomain.term.count(op))
        decisionSet.insert(op);
    for (Operation *op : decisionDomain.term)
      decisionSet.insert(op);
    return decisionSet;
  }

  // Own the storage the reference returned above points into; one root is
  // fully rewritten before the next is decided, so a single slot is enough.
  VecSetDomain decisionDomain;
  OperationTree decisionSet;

  // P4's unit price, in the unit P4 was calibrated in -- static instructions,
  // see findings.md 1.33 / 1.35. One materialisation edge costs
  // `N + ceil(N/W) + 1`: N scalar LM accesses, one vector LM access per whole
  // or residual vector, one LM mfence. `N` is the per-core scalar count and `W`
  // the lane count the rewrite would pick, both read off the value's own type
  // -- never `maxVecWidth`, which is a row width in elements and silently
  // collapses `N/W` to 1 (1.35's first bug). Returns 0 where the price is
  // undefined: a non-tensor, an already-vectorized type, or a shape
  // `getVectorType` would reject; callers count those separately so a 0 is
  // never read as "free". The figure is the un-tiled one (iterNum=1), which is
  // the case the calibration self-checked against (a pair at N=64, W=16 ->
  // 138); a real segment scales it by iterNum per 1.35's formula.
  int64_t p4EdgePrice(Value v, int64_t &nOut, int64_t &wOut) {
    nOut = wOut = 0;
    auto tensorTy = mlir::dyn_cast<RankedTensorType>(v.getType());
    if (!tensorTy)
      return 0;
    Type elemTy = getElementTypeOrSelf(tensorTy);
    if (mlir::isa<mlir::VectorType>(elemTy) || !elemTy.isIntOrFloat())
      return 0;
    int64_t n = getTotalElemsPerThread(withTLEDefaultEncoding(Type(tensorTy)));
    int64_t w = 512 / elemTy.getIntOrFloatBitWidth();
    if (n == 0 || w == 0 || n % w != 0)
      return 0;
    nOut = n;
    wOut = w;
    return n + n / w + 1;
  }

  // §1.53's store-side rule, reported as a *fact* about the op before anything
  // decides on it. Two independent things have to hold for a store to take a
  // vector value:
  //
  //   LM side  the store lowering computes exactly two strides -- 1 when
  //            `tensorColSize == -1`, `ceil(colSize/W)` under CoreDealMultiRows
  //            (LoadStoreOpToLLVM.cpp:1874-1890) -- and only its `isVectorized`
  //            branch (1924) writes whole vectors; everything else walks the
  //            elements as scalars.
  //   GM side  the `lm2gm` that drains the same LM buffer switches on
  //            `offsetState` and ends in `llvm_unreachable("Unknown offset
  //            state")` (2516), so Discrete / DiscreteSame have no lowering at
  //            all -- they are not slow, they are absent.
  //
  // So the two states the capability table calls Continuous / CoreDealMultiRows
  // are `Continuous` and `LocallyContinuous`, and Unknown is a reject even
  // though it lowers, because its per-pointer path is not a vector store.
  static const char *offsetStateName(int64_t state) {
    switch (state) {
    case -1:
      return "unknown";
    case 0:
      return "discrete-same";
    case 1:
      return "continuous";
    case 2:
      return "discrete";
    case 3:
      return "locally-continuous";
    }
    return "?";
  }

  // The `lm2gm` draining the buffer this store writes, or null when the pairing
  // is not a single hop off the store's pointer.
  Operation *gmSideOf(Operation *storeOp) {
    Value ptr;
    if (auto st = dyn_cast<triton::xpu::StoreOp>(storeOp))
      ptr = st.getPtr();
    if (!ptr)
      return nullptr;
    for (Operation *user : ptr.getUsers())
      if (isa<triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp>(user))
        return user;
    return nullptr;
  }

  // Which named operand of a store / lm2gm this value is, so a report can say
  // *what* would turn into a vector at the boundary. The three ops carry
  // AttrSizedOperandSegments, so an index alone does not name a role.
  // Read off `operandSegmentSizes` rather than through the generated accessors:
  // an `lm2gm` in practice carries sizes [1,0,1,1], i.e. **no** `value` operand
  // even though the ODS declares it non-optional, so `getValue()` there is
  // `getODSOperands(1).front()` on an empty range -- it silently hands back the
  // *next* operand (the `len`) and would make this report lie.
  static const char *storeOperandRole(Operation *op, unsigned idx) {
    static const char *storeRoles[] = {"ptr", "value", "mask", "index"};
    static const char *lm2gmRoles[] = {"ptr", "value", "len", "bufPtr"};
    static const char *lm2gmMaskRoles[] = {"ptr", "value", "mask", "len",
                                           "bufPtr"};
    const char **roles = nullptr;
    unsigned numRoles = 0;
    if (isa<triton::xpu::StoreOp>(op)) {
      roles = storeRoles;
      numRoles = 4;
    } else if (isa<triton::xpu::LM2GMOp>(op)) {
      roles = lm2gmRoles;
      numRoles = 4;
    } else if (isa<triton::xpu::LM2GMMaskOp>(op)) {
      roles = lm2gmMaskRoles;
      numRoles = 5;
    }
    auto sizes = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    if (!roles || !sizes || sizes.size() != (int)numRoles)
      return "?";
    unsigned seen = 0;
    for (unsigned i = 0; i < numRoles; ++i) {
      seen += sizes[i];
      if (idx < seen)
        return roles[i];
    }
    return "?";
  }

  // The operand that decides *whether* the write happens, printed alongside the
  // one that crosses the boundary: `store` takes a bool mask it never lowers,
  // `lm2gm` takes a byte `len` instead and has no mask at all.
  static OpOperand *storeGuardOperand(Operation *op) {
    for (OpOperand &use : op->getOpOperands()) {
      const char *role = storeOperandRole(op, use.getOperandNumber());
      if (!strcmp(role, "mask") || !strcmp(role, "len"))
        return &use;
    }
    return nullptr;
  }

  // The store-side rule of findings 1.53, single-sourced so the report and any
  // relaxation cannot drift: only a GM side whose `offsetState` is Continuous
  // or LocallyContinuous writes whole vectors; Discrete / DiscreteSame have no
  // lowering at all and Unknown lowers per pointer.
  bool storeRuleAllows(Operation *storeOp) {
    Operation *gm = isa<triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp>(storeOp)
                        ? storeOp
                        : gmSideOf(storeOp);
    int64_t gmState = -1;
    if (auto g = dyn_cast_or_null<triton::xpu::LM2GMOp>(gm))
      gmState = g.getOffsetState();
    else if (auto g = dyn_cast_or_null<triton::xpu::LM2GMMaskOp>(gm))
      gmState = g.getOffsetState();
    return gm && (gmState == 1 || gmState == 3);
  }

  // Escape hatch, not a feature gate: letting the store take the vector operand
  // is the default since 3.2l, and `=refuse` puts the guard back. Measured on
  // both states the rule accepts -- continuous (findings 1.63: 251.7 -> 176.6
  // us) and locally-continuous (1.64: 247.9 -> 175.7 us), both `max_abs=0`.
  //
  // The other candidate, keeping the store scalar behind a `triton_xpu.unpack`,
  // is gone rather than kept as a mode: it does not compile at all
  // (`TritonXPUAlloca` loses the pairing it uses to find the `lm2gm`'s LM
  // buffer), so there was never a choice to make here.
  bool storeBoundaryRefused() {
    static const char *mode = std::getenv("TRITONXPU_STORE_BOUNDARY");
    return mode && !strcmp(mode, "refuse");
  }

  // Calibration hatch for P4, not a feature gate: `=priced-out` lets a segment
  // the price model refused go through anyway. The 3.8 census (findings 1.69)
  // found `cut` never fires on real operators and `priced-out` is the dominant
  // verdict, so the model's refusals have never been checked against hardware
  // -- this is how a "should have cut, price blocked it" site gets measured.
  //
  // Only `priced-out` can be overridden. The other verdicts refuse because the
  // rewrite would be wrong or untyped, not because it would be unprofitable.
  bool pricedOutForced() {
    static const char *mode = std::getenv("TRITONXPU_VEC_CUT_FORCE");
    return mode && !strcmp(mode, "priced-out");
  }

  // Report-only for now (step 3.2i+): says what the rule *would* answer at
  // every store the cut had to refuse, so the criterion can be written against
  // measured states instead of against the capability table alone.
  void reportStoreRule(triton::FuncOp funcOp, const char *site,
                       const char *where, Operation *storeOp,
                       OpOperand *crossing = nullptr) {
    if (!vecReportEnabled())
      return;
    int64_t colSize = -1;
    if (auto st = dyn_cast<triton::xpu::StoreOp>(storeOp))
      colSize = st.getTensorColSize();
    // A `lm2gm` *is* the GM side; only a `store` needs the pairing hop.
    Operation *gm = isa<triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp>(storeOp)
                        ? storeOp
                        : gmSideOf(storeOp);
    int64_t gmState = -1;
    if (auto g = dyn_cast_or_null<triton::xpu::LM2GMOp>(gm))
      gmState = g.getOffsetState();
    else if (auto g = dyn_cast_or_null<triton::xpu::LM2GMMaskOp>(gm))
      gmState = g.getOffsetState();
    bool allow = storeRuleAllows(storeOp);
    llvm::errs() << "[VecStore] " << funcOp.getName() << " site=" << site
                 << " where=" << where << " kind=" << storeOp->getName()
                 << " colSize=" << colSize
                 << " gm=" << (gm ? gm->getName().getStringRef() : "none")
                 << " offsetState=" << offsetStateName(gmState)
                 << " rule=" << (allow ? "allow" : "reject");
    if (crossing) {
      Value v = crossing->get();
      Operation *def = v.getDefiningOp();
      llvm::errs() << " crossing="
                   << storeOperandRole(storeOp, crossing->getOperandNumber())
                   << "#" << crossing->getOperandNumber() << ":" << v.getType()
                   << " crossingDef="
                   << (def ? def->getName().getStringRef() : "block-arg");
    }
    if (OpOperand *guard = storeGuardOperand(storeOp))
      llvm::errs() << " guard="
                   << storeOperandRole(storeOp, guard->getOperandNumber())
                   << "#" << guard->getOperandNumber() << ":"
                   << guard->get().getType();
    else
      llvm::errs() << " guard=none";
    llvm::errs() << " loc=" << storeOp->getLoc() << "\n";
  }

  // Step 3.2 stage 2: the first change here that alters emitted code, so it
  // sits behind `per-op-decision` and the default path is untouched.
  //
  // The closure walk is all-or-nothing: when it vetoes, the whole site stays
  // scalar even though M1's partition may hold a Vector class *inside* the
  // Conflict class the store's unconditional pin creates (findings 1.58 found
  // four such sites). This cuts that class out and materialises its exit edge
  // with an UnpackOp instead of dropping it.
  //
  // Every guard below can refuse, and the reason is reported rather than
  // silently swallowed -- refusing keeps today's behaviour, so a wrong guard
  // costs an opportunity, never correctness:
  //   priced-out       P4 says net <= 0 (1.59: only truncint of the four pays)
  //   loads-only       the segment is loads and constants, i.e. load and
  //                    materialise straight back out -- measured net negative
  //                    in 2.2, priced at -17 per edge in 1.59
  //   coverage         some term op has no Full processOpVecTy Case, which is
  //                    where 1.53's `conditional` silent failures live
  //   store-in-segment / store-at-boundary
  //                    a store is involved, so 1.53's store-side rule
  //                    (Continuous / CoreDealMultiRows only, reject Discrete
  //                    and unknown) would have to be implemented first
  //   not-invertible   some boundary value has no exact scalar<->vector type
  //                    round trip, so the UnpackOp could not be typed
  bool cutVectorSegment(Operation *rootOp, const char *site, ModuleOp &mod) {
    if (!perOpDecision || !rootOp)
      return false;
    auto funcOp = rootOp->getParentOfType<triton::FuncOp>();
    Value keyValue = decisionKeyValue(rootOp);
    if (!funcOp || !keyValue)
      return false;

    VectorFlowAnalysis vflow(vectorFitsValue, this->tleSmemVec, this->tleVec);
    vflow.run(funcOp);
    OperationTree noClosure;
    VecSetDomain domain = buildVecSetDomain(keyValue, vflow, noClosure);
    SegmentPrice sp = priceSegment(domain, vflow, /*unpricedTys=*/nullptr);

    const char *reject = domain.term.empty() ? "empty"
                         : sp.net() <= 0     ? "priced-out"
                                             : nullptr;
    bool forced = false;
    if (reject && !strcmp(reject, "priced-out") && pricedOutForced()) {
      reject = nullptr;
      forced = true;
    }

    // Program order, not cone order: processOpVecTy rewrites in iteration order
    // and reads already-rewritten operands (findings 1.57).
    OperationTree segment;
    if (!reject) {
      funcOp.walk([&](Operation *op) {
        if (domain.term.count(op))
          segment.insert(op);
      });
      bool onlyLoads = true;
      for (Operation *op : segment)
        onlyLoads &= isa<triton::xpu::LoadOp>(op) ||
                     op->hasTrait<mlir::OpTrait::ConstantLike>();
      if (onlyLoads)
        reject = "loads-only";
    }
    for (Operation *op : segment) {
      if (reject)
        break;
      if (processOpVecTyCoverage(op, this->tleSmemVec, this->tleVec) !=
          VecTyCoverage::Full)
        reject = "coverage";
      else if (isa<triton::xpu::StoreOp, triton::xpu::LM2GMOp,
                   triton::xpu::LM2GMMaskOp>(op)) {
        reject = "store-in-segment";
        reportStoreRule(funcOp, site, "in-segment", op);
      }
    }

    // The boundary, recorded as (consumer, operand index, scalar type) rather
    // than as OpOperand handles: processOpVecTy replaces the producers, so the
    // values move, while the consumers stay put because they are outside the
    // segment.
    SmallVector<std::tuple<Operation *, unsigned, RankedTensorType>> boundary;
    for (Operation *op : segment) {
      if (reject)
        break;
      for (Value res : op->getResults()) {
        auto scalarTy = mlir::dyn_cast<RankedTensorType>(res.getType());
        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          if (segment.count(user))
            continue;
          if (isa<triton::xpu::StoreOp, triton::xpu::LM2GMOp,
                  triton::xpu::LM2GMMaskOp>(user)) {
            reportStoreRule(funcOp, site, "at-boundary", user, &use);
            if (storeBoundaryRefused() || !storeRuleAllows(user)) {
              reject = "store-at-boundary";
              break;
            }
            // The store takes the vector operand, so there is no edge to
            // materialise: its `mask` stays a scalar tensor and nothing reads
            // it
            // (`XPUStoreOpConversion` never calls `getMask()`; what bounds the
            // write is the paired gm2lm/lm2gm `len`). findings 1.63.
            continue;
          }
          if (!scalarTy || !vectorFormExists(scalarTy) ||
              getScalarTypeOrNull(getVectorType(scalarTy)) != scalarTy) {
            reject = "not-invertible";
            break;
          }
          boundary.emplace_back(user, use.getOperandNumber(), scalarTy);
        }
      }
    }

    if (vecReportEnabled())
      llvm::errs() << "[VecCut] " << funcOp.getName() << " site=" << site
                   << " root=" << rootOp->getName()
                   << " segment=" << segment.size()
                   << " boundary=" << boundary.size() << " price=" << sp.price
                   << " save=" << sp.save << " net=" << sp.net()
                   << " verdict=" << (reject ? reject : "cut")
                   << (forced ? " forced=priced-out" : "")
                   << " loc=" << rootOp->getLoc() << "\n";
    if (reject)
      return false;

    // The `coverage` guard above already required Full for every member, so
    // this cannot refuse today; honouring it keeps the two checks from drifting
    // apart, and a refusal leaves the IR untouched so returning false is safe.
    if (!processOpVecTy(segment, mod)) {
      if (vecReportEnabled())
        llvm::errs() << "[VecCut] " << funcOp.getName() << " site=" << site
                     << " verdict=refused-by-retype\n";
      return false;
    }

    // One UnpackOp per boundary operand, which is also how the price counts
    // them, so `boundary` and `price` stay comparable.
    for (auto &[user, idx, scalarTy] : boundary) {
      Value v = user->getOperand(idx);
      if (v.getType() == scalarTy)
        continue; // producer kept its scalar type (an inert Case)
      OpBuilder builder(user);
      Value sv = builder.create<triton::xpu::UnpackOp>(user->getLoc(), scalarTy,
                                                       v, /*bufPtr=*/Value());
      user->setOperand(idx, sv);
    }
    return true;
  }

  // Both sides of the P4 arithmetic for one candidate segment, in static
  // instructions. Single-sourced: the report and the cut decision must not be
  // able to disagree about whether a segment pays.
  struct SegmentPrice {
    int64_t price = 0;        // what the materialisation edges cost
    int64_t save = 0;         // what the term ops stop paying
    int64_t matIn = 0;        // Scalar operand entering the segment
    int64_t matOut = 0;       // Vector operand leaving it
    int64_t unpricedEdge = 0; // edge P4's formula does not apply to
    int64_t unpricedTerm = 0; // term op whose own width is unpriceable
    int64_t free = 0;         // splat constants: one materialisation either way
    int64_t net() const { return save - price; }
  };

  // The out-edges of a segment, enumerated the way `cutVectorSegment` actually
  // materialises them: one per use of a segment result by a non-member, minus
  // the store users that take the vector operand as-is (no edge to materialise,
  // findings 1.63). Single-sourced on purpose -- pricing them from the cone's
  // operands instead silently missed *sibling* consumers, which are not the
  // root's ancestors and so never enter the cone, while the rewrite still emits
  // an UnpackOp for each of them (findings 1.66: `boundary=2` against a price
  // of one edge, net reported 2.6x too optimistic).
  void forEachSegmentOutEdge(const OperationTree &members,
                             llvm::function_ref<void(OpOperand &)> fn) {
    for (Operation *op : members)
      for (Value res : op->getResults())
        for (OpOperand &use : res.getUses()) {
          Operation *user = use.getOwner();
          if (members.count(user))
            continue;
          if (isa<triton::xpu::StoreOp, triton::xpu::LM2GMOp,
                  triton::xpu::LM2GMMaskOp>(user))
            continue;
          fn(use);
        }
  }

  SegmentPrice priceSegment(const VecSetDomain &domain,
                            const VectorFlowAnalysis &vflow,
                            std::map<std::string, int64_t> *unpricedTys) {
    SegmentPrice sp;
    auto priceEdge = [&](Value v) {
      int64_t n = 0, w = 0;
      int64_t edge = p4EdgePrice(v, n, w);
      sp.price += edge;
      if (edge)
        return;
      ++sp.unpricedEdge;
      if (!unpricedTys)
        return;
      std::string ty;
      llvm::raw_string_ostream os(ty);
      v.getType().print(os);
      ++(*unpricedTys)[ty];
    };
    forEachSegmentOutEdge(domain.term, [&](OpOperand &use) {
      ++sp.matOut;
      priceEdge(use.get());
    });
    for (Operation *op : domain.cone) {
      bool inTerm = domain.term.count(op);
      if (!inTerm)
        continue;
      for (Value operand : op->getOperands()) {
        if (vflow.stateOf(operand) != VState::Scalar)
          continue;
        ++sp.matIn;
        priceEdge(operand);
      }
      // A splat constant is one materialisation either way, so counting it as
      // `N -> N/W` would inflate the saving of exactly the segments worth
      // cutting; it goes in `free` instead.
      if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
        ++sp.free;
        continue;
      }
      int64_t n = 0, w = 0;
      Value repr = decisionKeyValue(op);
      if (repr && p4EdgePrice(repr, n, w))
        sp.save += n - n / w;
      else
        ++sp.unpricedTerm;
    }
    return sp;
  }

  // Step 3.2 stage 2, report only (TRITONXPU_VEC_REPORT=1). Stage 1 could only
  // switch the roots the closure walk had already closed; every other root
  // leaves this pass through one of two earlier gates -- `vectorFitsRoot`
  // refusing the root type, or the walk vetoing inside. Which of those is worth
  // opening depends on what the partition says at the same place:
  //
  //   keyState=Scalar  -- the gate is not a lost opportunity, M1 reaches the
  //     same answer, just later. Opening it would gain nothing.
  //   keyState=Vector  -- the gate is refusing a root the partition would
  //     retype, so it is a real candidate, and `term` is what it would hand
  //     over. `p4{}` prices that hand-over both ways -- what the edges cost
  //     against what the term ops stop paying -- and `coverage` says whether
  //     `.Default` becomes reachable there.
  //
  // `partial` is what the walk had already collected when it vetoed: the walk
  // is all-or-nothing, so that set is dropped today, and its size says how far
  // it got before the veto.
  void reportOpenGate(Operation *rootOp, const char *site, const char *gate,
                      size_t partial) {
    if (!vecReportEnabled() || !rootOp)
      return;
    auto funcOp = rootOp->getParentOfType<triton::FuncOp>();
    Value keyValue = decisionKeyValue(rootOp);
    if (!funcOp || !keyValue)
      return;
    VectorFlowAnalysis vflow(vectorFitsValue, this->tleSmemVec, this->tleVec);
    vflow.run(funcOp);
    OperationTree noClosure;
    VecSetDomain domain = buildVecSetDomain(keyValue, vflow, noClosure);

    int64_t inert = 0;
    int64_t cover[3] = {0, 0, 0};
    std::map<std::string, int64_t> termKinds;
    // An unpriced edge is only harmless if its type says no materialisation is
    // needed there (an i1 mask, a pointer tensor), so the types go in the
    // report -- otherwise a 0 would hide a real cost.
    std::map<std::string, int64_t> unpricedTys;
    SegmentPrice sp = priceSegment(domain, vflow, &unpricedTys);
    for (Operation *op : domain.term) {
      ++termKinds[op->getName().getStringRef().str()];
      ++cover[unsigned(processOpVecTyCoverage(op))];
      inert += rewriteNoopKind(op);
    }

    llvm::errs() << "[VecOpen] " << funcOp.getName() << " site=" << site
                 << " gate=" << gate << " root=" << rootOp->getName()
                 << " keyState=" << toString(vflow.stateOf(keyValue))
                 << " tracked=" << vflow.isTracked(keyValue)
                 << " cone=" << domain.cone.size()
                 << " term=" << domain.term.size() << " termInert=" << inert
                 << " partial=" << partial
                 << " coverage{full=" << cover[unsigned(VecTyCoverage::Full)]
                 << ",cond=" << cover[unsigned(VecTyCoverage::Conditional)]
                 << ",none=" << cover[unsigned(VecTyCoverage::None)] << "}"
                 << " mat{in=" << sp.matIn << ",out=" << sp.matOut << "}"
                 << " p4{price=" << sp.price << ",save=" << sp.save
                 << ",net=" << sp.net() << ",unpriced=" << sp.unpricedEdge
                 << "/" << sp.unpricedTerm << ",free=" << sp.free << "}"
                 << " unpricedEdgeTys={";
    const char *usep = "";
    for (auto &entry : unpricedTys) {
      llvm::errs() << usep << entry.first << ":" << entry.second;
      usep = ",";
    }
    llvm::errs() << "} termKinds={";
    const char *sep = "";
    for (auto &entry : termKinds) {
      llvm::errs() << sep << entry.first << ":" << entry.second;
      sep = ",";
    }
    llvm::errs() << "} loc=" << rootOp->getLoc() << "\n";
  }

  // void doCompareCastI8Fusion(arith::ExtUIOp extUIOp) {
  //   if (auto cmpFOp = extUIOp.getIn().getDefiningOp<arith::CmpFOp>()) {
  //     // Only Vectorize Do Fusion
  //     auto rowsPerCore = 1;
  //     auto inputTy = cmpFOp.getLhs().getType();
  //     if (auto inputTensorTy = mlir::dyn_cast<RankedTensorType>(inputTy)) {
  //       auto rank = inputTensorTy.getShape().size();
  //       if (rank > 1) {
  //         rowsPerCore = mlir::cast<triton::xpu::ClusterLayoutAttr>(
  //                           inputTensorTy.getEncoding())
  //                           .getSizePerCore()[0];
  //       }
  //     }
  //     unsigned numElems = getTotalElemsPerThread(inputTy) / rowsPerCore;
  //     Type vecTy = getElementTypeOrSelf(inputTy);
  //     Type elemTy = getElementTypeOrSelf(vecTy);
  //     auto elemWidth = elemTy.getIntOrFloatBitWidth();
  //     auto vectorWidth = 512 / elemWidth;
  //     if (numElems < vectorWidth || numElems % vectorWidth > 0 ||
  //         !vectorizedTyValid(elemTy))
  //       return;
  //     // Fuse CmpFOp + ExtUIOp
  //     if (cmpFOp.getResult().hasOneUse()) {
  //       auto resTy = extUIOp.getOut().getType();
  //       OpBuilder builder(cmpFOp);
  //       auto newCmpFOp = builder.create<triton::xpu::CmpFCastOp>(
  //           cmpFOp.getLoc(), extUIOp.getType(), cmpFOp.getPredicate(),
  //           cmpFOp.getLhs(), cmpFOp.getRhs());
  //       extUIOp.replaceAllUsesWith(newCmpFOp.getResult());
  //       extUIOp.erase();
  //       cmpFOp->erase();
  //     }
  //   }
  // }

  bool isLoadVectorized(triton::xpu::LoadOp loadOp) {
    Type resTy = loadOp.getType();
    Type resElemTy = getElementTypeOrSelf(resTy);
    return mlir::isa<mlir::VectorType>(resElemTy);
  }

  bool SVOptimization_Cond(Operation *op) {
    bool canSVOpt = false;
    // TODO: Check block Argument
    if (!op)
      return canSVOpt;

    TypeSwitch<const Operation *>(op)
        .Case<triton::xpu::LoadOp>([&](auto loadOp) {
          if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(
                  loadOp.getPtr().getDefiningOp())) {
            OffsetState offsetState =
                static_cast<OffsetState>(gm2lmOp.getOffsetState());
            if (offsetState == OffsetState::DiscreteSame &&
                isLoadVectorized(loadOp))
              canSVOpt = true;
          } else if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(
                         loadOp.getPtr().getDefiningOp())) {
            OffsetState offsetState =
                static_cast<OffsetState>(gm2lmOp.getOffsetState());
            if (offsetState == OffsetState::DiscreteSame &&
                isLoadVectorized(loadOp))
              canSVOpt = true;
          }
        })
        .Case<triton::xpu::BroadcastOp>([&](auto bcOp) {
          auto src = bcOp.getSrc();
          if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
            auto srcShape = srcTy.getShape();
            if (srcShape.size() == 2 && srcShape[1] == 1) {
              canSVOpt = true;
            }
          }
        })
        .Case<triton::xpu::VConstOp>([&](auto vConstOp) { canSVOpt = true; })
        .Case<triton::xpu::ConvertLayoutOp>([&](auto convertOp) {
          auto srcDefOp = convertOp.getSrc().getDefiningOp();
          canSVOpt = SVOptimization_Cond(srcDefOp);
        })
        .Default([&](auto &op) { canSVOpt = false; });

    return canSVOpt;
  }

  void getUsers(SetVector<Operation *> &users, Operation *op) {
    SetVector<Operation *> visited;
    SmallVector<Operation *> worklist = {op};

    while (!worklist.empty()) {
      Operation *currentOp = worklist.pop_back_val();
      for (Operation *userOp : currentOp->getUsers()) {
        if (visited.contains(userOp))
          continue;
        visited.insert(userOp);

        if (isa<triton::xpu::ConvertLayoutOp>(userOp)) {
          worklist.push_back(userOp);
        } else {
          users.insert(userOp);
        }
      }
    }
  }

  bool collectVUser(Operation *op, DenseMap<Operation *, ElemState> &vBinOps) {
    // To check if the collection was successful.
    bool canSVOpt = true;
    SetVector<Operation *> users;
    getUsers(users, op);
    if (users.empty()) {
      canSVOpt = false;
    }
    for (auto user : users) {
      TypeSwitch<const Operation *>(user)
          .Case<XPU_VVECTORIZED_BINARY_OP>([&](auto vBinOp) {
            // The SV form is an inline asm string picked per element type, and
            // XPUSVBinaryOpConversion only has f32/i32 (`SVOp2Str`) and f16
            // (`SVOp2StrFP16`); everything else reaches its `llvm_unreachable`.
            // svxori is the exception: BitwiseCastToI32Pattern repacks it into
            // i32 lanes at the end of this pass, so a narrower integer is fine
            // there once the vector is a whole number of i32 lanes. Declining
            // is not free -- a VV integer op with a constant operand reaches
            // LLVM as a plain `xor`, instcombine folds the bitcasts back into
            // it, and 16-bit lanes are then miscompiled (flaggems bitwise_not
            // on i16 read every other element) -- so decline only what really
            // cannot be lowered.
            using VBinTy = std::decay_t<decltype(vBinOp)>;
            constexpr bool isXorI =
                std::is_same_v<VBinTy, triton::xpu::VvxorIOp>;
            Type resTy = vBinOp.getType();
            auto svVecTy = dyn_cast<VectorType>(getElementTypeOrSelf(resTy));
            Type svElemTy = getElementTypeOrSelf(getElementTypeOrSelf(resTy));
            bool svLowerable = svElemTy.isF32() || svElemTy.isF16() ||
                               svElemTy.isSignlessInteger(32);
            if (!svLowerable && isXorI && svVecTy &&
                svElemTy.isSignlessInteger() &&
                (svVecTy.getNumElements() * svElemTy.getIntOrFloatBitWidth()) %
                        32 ==
                    0)
              svLowerable = true;
            if (!svLowerable) {
              canSVOpt = false;
              return;
            }
            auto lDefineOp =
                vBinOp.getLhs().getDefiningOp(); // getLhs define op
            auto rDefineOp = vBinOp.getRhs().getDefiningOp();

            bool lCond = SVOptimization_Cond(lDefineOp);
            bool rCond = SVOptimization_Cond(rDefineOp);

            bool opIsLhs = lDefineOp == op;

            if ((opIsLhs ? lCond : rCond) && (lCond != rCond)) {
              vBinOps[vBinOp] = opIsLhs ? ElemState::SV : ElemState::VS;
            } else {
              canSVOpt = false;
            }
          })
          .Default([&](auto &user) { canSVOpt = false; });

      if (!canSVOpt)
        break;
    }
    return canSVOpt;
  }

  void SVOptimization_Modify(triton::xpu::LoadOp loadOp) {
    // Get Information
    Type tensorType = loadOp.getType();
    Type vecElemTy = getElementTypeOrSelf(tensorType);

    // vecNums / numElems  (all vector<16xTy> use one same Ty)
    unsigned vecNums =
        mlir::cast<RankedTensorType>(tensorType).getNumElements();
    unsigned numElems = getTotalElemsPerThread(tensorType);

    // elem type
    Type elemTy = getElementTypeOrSelf(vecElemTy);

    // encoding
    auto encoding = mlir::cast<triton::xpu::ClusterLayoutAttr>(
        mlir::cast<RankedTensorType>(tensorType).getEncoding());

    std::vector<unsigned> sizePerCore = {1}; // 1 for scalar
    Attribute newEncoding = triton::xpu::ClusterLayoutAttr::get(
        encoding.getContext(), sizePerCore, encoding.getCoresPerGroup(),
        encoding.getGroupsPerCluster(), encoding.getOrder());

    Type newTensorType = RankedTensorType::get(
        ceil<unsigned>(vecNums, numElems), elemTy, newEncoding);

    // Replace Origin Op
    OpBuilder builder(loadOp);
    loadOp->setAttr("SVOpt", builder.getBoolAttr(true));
    loadOp->getResult(0).setType(newTensorType);
  }

  // To check if the SVOptimization(Own) was successful.
  void SVOptimization_Modify(triton::xpu::BroadcastOp vBCOp) {
    auto src = vBCOp.getSrc();
    SetVector<Operation *> users;
    getUsers(users, vBCOp);
    for (auto user : users) {
      for (auto operand : user->getOperands()) {
        auto defOp = operand.getDefiningOp();
        // Only rewrite operands that actually derive from the broadcast being
        // optimized (directly, or through a convert_layout chain rooted at
        // vBCOp). A user may also consume an unrelated sibling broadcast (e.g.
        // `x - mean`, where both `x` and `mean` are broadcasts); replacing that
        // sibling with vBCOp's scalar src would silently drop it and leave a
        // scalar-operand SV op with a vector result type, which fails LLVM
        // translation on the residual unrealized_conversion_cast.
        if (isa_and_nonnull<triton::xpu::ConvertLayoutOp,
                            triton::xpu::BroadcastOp>(defOp) &&
            operand != src && tracesToBroadcast(operand, vBCOp)) {
          operand.replaceAllUsesWith(src);
        }
      }
    }
  }

  // Returns true if `operand` is produced by `vBCOp`, either directly or
  // through a chain of convert_layout ops.
  bool tracesToBroadcast(Value operand, triton::xpu::BroadcastOp vBCOp) {
    Operation *defOp = operand.getDefiningOp();
    while (defOp) {
      if (defOp == vBCOp.getOperation())
        return true;
      if (auto cvtOp = dyn_cast<triton::xpu::ConvertLayoutOp>(defOp)) {
        defOp = cvtOp.getSrc().getDefiningOp();
        continue;
      }
      break;
    }
    return false;
  }

  // To check if the SVOptimization(Own) was successful.
  void SVOptimization_Modify(triton::xpu::VConstOp vConstOp) {
    auto res = vConstOp.getResult();
    auto resTy = mlir::cast<RankedTensorType>(res.getType());
    auto resShape = resTy.getShape();
    triton::xpu::ClusterLayoutAttr vConstOpEncoding =
        mlir::cast<triton::xpu::ClusterLayoutAttr>(resTy.getEncoding());
    auto order = vConstOpEncoding.getOrder();
    auto groupsPerCluster = vConstOpEncoding.getGroupsPerCluster();
    auto CoresPerGroup = vConstOpEncoding.getCoresPerGroup();
    auto sizePerCore = vConstOpEncoding.getSizePerCore();
    auto groupSize = product(CoresPerGroup);
    auto nGroup = product(groupsPerCluster);
    auto nCore = groupSize * nGroup;
    unsigned rank = resTy.getRank();

    auto elemTy = getElementTypeOrSelf(vConstOp.getType());
    auto _elemTy = getElementTypeOrSelf(elemTy);
    RankedTensorType newSrcTy;
    if (rank == 1) {
      auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
          context, {1}, CoresPerGroup, groupsPerCluster, order);
      newSrcTy = RankedTensorType::get({nCore}, _elemTy, newEncoding);
    } else if (rank == 2) {
      unsigned newSizePerCore =
          ceil(resShape.front(), static_cast<int64_t>(nCore));
      auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
          context, {newSizePerCore, 1}, CoresPerGroup, groupsPerCluster, order);
      newSrcTy =
          RankedTensorType::get({resShape.front(), 1}, _elemTy, newEncoding);
    } else {
      llvm_unreachable("Got Unsupport Rank");
    }

    // TODO[dyq]: dyn_cast -> cast
    auto oriDenseAttr =
        mlir::dyn_cast<mlir::DenseElementsAttr>(vConstOp.getValue());
    auto initValue = DenseElementsAttr::getFromRawBuffer(
        newSrcTy, oriDenseAttr.getRawData());

    OpBuilder builder(vConstOp);
    auto newConstOp = builder.create<arith::ConstantOp>(vConstOp.getLoc(),
                                                        newSrcTy, initValue);
    vConstOp.replaceAllUsesWith(newConstOp.getResult());
    vConstOp.erase();
  }

  template <typename T> void createSVBinOp(T vBinOp, ElemState elemStateInt) {
    if (elemStateInt == ElemState::VS) {
      // SVSUB Has A Strict Order Of Operations.
      // V-S -> -S+V
      if constexpr (std::is_same_v<T, triton::xpu::VvsubFOp>) {
        OpBuilder builder(vBinOp);
        auto negFOp =
            builder.create<arith::NegFOp>(vBinOp.getLoc(), vBinOp.getRhs());
        auto svBinFOp = builder.create<triton::xpu::SvaddFOp>(
            vBinOp.getLoc(), vBinOp.getType(), vBinOp.getLhs(), negFOp,
            static_cast<int32_t>(elemStateInt));
        vBinOp.replaceAllUsesWith(svBinFOp.getResult());
        vBinOp.erase();
        LLVM_DEBUG(llvm::dbgs()
                   << "[Vectorization]: Apply VSSUB -> SVADD Optimization.\n");
        return;
      }
    }

    OpBuilder builder(vBinOp);
    auto svBinFOp = builder.create<typename VV2SVOp<T>::type>(
        vBinOp.getLoc(), vBinOp.getType(), vBinOp.getLhs(), vBinOp.getRhs(),
        static_cast<int32_t>(elemStateInt));
    vBinOp.replaceAllUsesWith(svBinFOp.getResult());
    vBinOp.erase();
  }

  void VvOpToSvOp(DenseMap<Operation *, ElemState> &vBinOps,
                  std::string logMessage) {
    for (auto &pair : vBinOps) {
      auto op = pair.first;
      auto elemStateInt = pair.second;
      TypeSwitch<const Operation *>(op)
          .Case<XPU_VVECTORIZED_BINARY_OP>(
              [&](auto vBinOp) { createSVBinOp(vBinOp, elemStateInt); })
          .Default([&](auto &op) {
            llvm_unreachable(
                "[Vectorization]: Got An Unexpected SV Operation Type");
          });
    }
    LLVM_DEBUG(llvm::dbgs() << logMessage);
  }

  template <typename T> void SVOptimization(T op, std::string logMessage) {
    // Step 1. collect all vUser
    DenseMap<Operation *, ElemState> vBinOps;
    if (!collectVUser(op, vBinOps))
      return;

    // Step 2. Deal Input Op Own Modification
    SVOptimization_Modify(op);

    // Step 3. Deal Input Op's User Modification
    VvOpToSvOp(vBinOps, logMessage);
  }

  // Simpify Mod Graph
  // TODO[dyq]: use canonicalizer
  void cvtOpclean(triton::xpu::ConvertLayoutOp cvtOp) {
    auto src = cvtOp.getSrc();
    auto res = cvtOp.getResult();

    if (src.getType() != res.getType())
      return;

    cvtOp.replaceAllUsesWith(src);
    cvtOp.erase();
  }

  void VvdivToVvmul(triton::xpu::VvdivFOp vvdivOp) {
    // Only can be optimized to vvmul when the denominator is a scalar, it can
    // be further optimized to svmul
    if (auto bcOp =
            vvdivOp.getRhs().getDefiningOp<triton::xpu::BroadcastOp>()) {
      auto src = bcOp.getSrc();
      auto res = bcOp.getResult();

      // Check 1. Src Shape Must Be 64x1xf32
      if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
        auto srcShape = srcTy.getShape();
        if (srcShape.size() != 2 || !(srcShape[0] == 64 && srcShape[1] == 1)) {
          return;
        } else {
          // Step 2. Create DivOp For Rhs
          OpBuilder builder(bcOp);
          SmallVector<Attribute, 4> intValues(srcShape[1],
                                              builder.getF32FloatAttr(1));
          DenseElementsAttr denseAttr =
              DenseFPElementsAttr::get(srcTy, intValues);
          auto ones =
              builder.create<arith::ConstantOp>(bcOp.getLoc(), denseAttr);

          auto oneDivByRhs =
              builder.create<arith::DivFOp>(bcOp.getLoc(), srcTy, ones, src);

          bcOp->setOperand(0, oneDivByRhs);

          // Step 3. Change vvdiv by vvmul
          OpBuilder builder_tmp(vvdivOp);
          auto vvmulOp = builder_tmp.create<triton::xpu::VvmulFOp>(
              vvdivOp.getLoc(), vvdivOp.getType(), vvdivOp.getLhs(),
              vvdivOp.getRhs());
          vvdivOp.replaceAllUsesWith(vvmulOp->getResult(0));
          vvdivOp.erase();
          LLVM_DEBUG(
              llvm::dbgs()
              << "[Vectorization]: Apply VVDIV -> VVMUL Optimization.\n");
        }
      } else {
        return;
      }
    }
  }

  void VVMacOpFusion(triton::xpu::VvmulFOp mulOp) {
    for (auto nextOp : mulOp->getUsers()) {
      if (auto addOp = dyn_cast<triton::xpu::VvaddFOp>(nextOp)) {
        auto lDefineOp = addOp.getLhs().getDefiningOp(); // getLhs define op
        OpBuilder builder(addOp);
        auto newMacOp = builder.create<triton::xpu::VMacFOp>(
            mulOp.getLoc(), mulOp.getType(), mulOp.getLhs(), mulOp.getRhs(),
            lDefineOp == mulOp ? addOp.getRhs() : addOp.getLhs());

        addOp->replaceAllUsesWith(newMacOp);
        addOp->erase();
        LLVM_DEBUG(llvm::dbgs()
                   << "[Vectorization]: Apply VVMacOp Fusion Optimization.\n");
      }
    }
  }

  void BF16ToFP32VecOptimize(ModuleOp &mod) {
    // bf16Tofp32Unordered could only used in order-independent cases
    bool bf16Tofp32Unordered = true;
    int load_cnt = 0;
    mod.walk([&](triton::xpu::LoadOp loadOp) {
      load_cnt++;
      Type ptrTy = loadOp.getPtr().getType();
      Type ptrElemTy = getElementTypeOrSelf(ptrTy);
      Type ptrDataTy = mlir::cast<PointerType>(ptrElemTy).getPointeeType();
      Type resTy = loadOp.getResult().getType();
      Type resElemTy = getElementTypeOrSelf(resTy);
      Type resScalarTy = getElementTypeOrSelf(resElemTy);

      if (resScalarTy.isF32() && ptrDataTy.isBF16()) {
        auto stride = loadOp.getStride();
        auto tensorColSize = loadOp.getTensorColSize();
        bool isVector = mlir::isa<VectorType>(resElemTy);
        bool isSvOpt = loadOp.getSVOpt();
        bool isDiscreteSame = stride == 0;
        bool isContiguous = stride == 1;
        bool notCoreDealMultiRows = tensorColSize == -1;
        bf16Tofp32Unordered &=
            (isVector && isContiguous && notCoreDealMultiRows) || isSvOpt ||
            isDiscreteSame;
      } else {
        bf16Tofp32Unordered &= false;
      }
    });

    bf16Tofp32Unordered = load_cnt == 0 ? false : bf16Tofp32Unordered;

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      Value val = storeOp.getValue();
      if (!bf16Tofp32Unordered)
        return;
      // The check must not filter on the store value's type: a
      // reduce's final store carries the de-vectorized result, but the
      // permutation happens at the LOAD and is consumed long before the
      // store. Relaxation: only a chain carrying a bf16 -> f32 load can
      // hold permuted registers, so only those need the arange check --
      // an unrelated index-derived store no longer vetoes the module. A
      // f32->bf16 store whose chain holds NO such load has nothing to
      // cancel its own permutation, so it turns the placement off.
      Type ptrElemTy = getElementTypeOrSelf(storeOp.getPtr().getType());
      Type ptrDataTy = mlir::cast<PointerType>(ptrElemTy).getPointeeType();
      Type valElemTy = getElementTypeOrSelf(val.getType());
      Type valScalarTy = getElementTypeOrSelf(valElemTy);
      DenseSet<Operation *> visitedLoad;
      if (!sliceHasBf16ToF32Load(val, visitedLoad)) {
        if (valScalarTy.isF32() && ptrDataTy.isBF16())
          bf16Tofp32Unordered &= false;
        return;
      }
      DenseSet<Operation *> visited;
      if (tleDependsOnLaneIndex(val, visited)) {
        bf16Tofp32Unordered &= false;
      }
    });

    mod.walk([&](triton::xpu::LoadOp loadOp) {
      OpBuilder builder(loadOp);
      loadOp->setAttr("bf16Tofp32Unordered",
                      builder.getBoolAttr(bf16Tofp32Unordered));
    });

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      OpBuilder builder(storeOp);
      storeOp->setAttr("bf16Tofp32Unordered",
                       builder.getBoolAttr(bf16Tofp32Unordered));
    });

    if (bf16Tofp32Unordered) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "[Vectorization]: Apply BF16ToFP32VecUnordered Optimization.\n");
    }
  }

  // P4 (boundary unit price), measurement only.
  //
  //   TRITONXPU_PROBE_BOUNDARY=<k>      filler units to insert; 0 / unset = off
  //   TRITONXPU_PROBE_SIDE=vec|scalar   which side of the boundary they run on
  //                                     (default scalar)
  //
  // The first shape of this probe -- k identity `unpack -> pack` round-trips,
  // price read off the slope in k -- measured nothing. `pack(unpack(v))` is a
  // no-op the optimizer sees through: in the emitted llir the unpack's N scalar
  // loads came back as `extractelement` off the source vector register
  // (store-to-load forwarding on the boundary's own private LM buffer), leaving
  // a single real load. The slope in k priced the residue, not a boundary.
  //
  // Hardening the round-trip against that forwarding would have been worse than
  // useless. An unpack whose source is live in vregs *legitimately* costs
  // extractelements, so a probe built to defeat the forwarding would price a
  // boundary that production code never pays.
  //
  // So this measures the quantity the model actually consumes instead. The same
  // value is computed two ways, with identical arithmetic on either side:
  //
  //   PROBE_SIDE=vec     v' = f^k(v)                 0 boundaries, k * N/W ops
  //   PROBE_SIDE=scalar  v' = pack(f^k(unpack(v)))   1 boundary pair, k * N ops
  //
  //   dt(k) = t_scalar(k) - t_vec(k) = boundaryPair + k * (N*s - (N/W)*v)
  //
  // Fit over k >= 1: the intercept is the boundary pair price that
  // redesign-v2.md:595 charges every Vector<->Scalar edge, and the slope is the
  // per-op scalar penalty 3.4's trade-off term needs. k = 0 is deliberately not
  // part of the fit -- that is the degenerate identity case above.
  //
  // f is one unit of `x -> (x * 2.0) * 0.5`. Exact for every normal f32 (both
  // factors are powers of two), so the two variants store bit-identical results
  // and the existing accuracy gate still covers them; and not foldable without
  // reassociation, which nothing licenses here because no fastmath flag is set.
  // If the llir ever shows the pair folded away the measurement is void, so
  // check that before trusting a number.
  //
  // Why a knob and not hand-written IR: triton-opt registers no XPU pass, so a
  // hand-written .ttxir cannot be driven through the rest of the pipeline.
  // PROBE_SIDE is the activation switch, not PROBE_BOUNDARY, so that k = 0 is a
  // usable data point: `scalar` with k = 0 is the bare boundary pair with no
  // filler at all, which is the direct measurement of the intercept the fit
  // extrapolates to. Both unset leaves the IR untouched.
  bool probeActive() {
    std::string side = mlir::triton::tools::getStrEnv("TRITONXPU_PROBE_SIDE");
    return side == "vec" || side == "scalar";
  }

  unsigned probeFillerCount() {
    std::string s = mlir::triton::tools::getStrEnv("TRITONXPU_PROBE_BOUNDARY");
    unsigned n = 0;
    if (s.empty() || llvm::StringRef(s).getAsInteger(10, n))
      return 0;
    return n;
  }

  bool probeScalarSide() {
    return mlir::triton::tools::getStrEnv("TRITONXPU_PROBE_SIDE") == "scalar";
  }

  // Inverse of getVectorType for the types this pass produces: undo the
  // trailing-dim and sizePerCore division. The forward direction is not
  // injective -- it clamps sizePerCore with max(1, ...) and special-cases
  // numElems==1 -- so the candidate is accepted only if getVectorType maps it
  // back to the input. Returns null otherwise; fabricating an inverse here
  // would put a wrong scalar shape into the calibration.
  RankedTensorType getScalarTypeOrNull(RankedTensorType vecTensorTy) {
    auto vecElemTy = mlir::dyn_cast<VectorType>(vecTensorTy.getElementType());
    auto enc = mlir::dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
        vecTensorTy.getEncoding());
    if (!vecElemTy || vecElemTy.getRank() != 1 || !enc)
      return {};
    int64_t width = vecElemTy.getNumElements();
    SmallVector<int64_t> shape(vecTensorTy.getShape());
    auto sizePerCore = enc.getSizePerCore().vec();
    auto rank = shape.size();
    if (rank == 0 || sizePerCore.size() != rank)
      return {};
    shape[rank - 1] *= width;
    sizePerCore[rank - 1] *= width;
    auto scalarTy = RankedTensorType::get(
        shape, vecElemTy.getElementType(),
        triton::xpu::ClusterLayoutAttr::get(
            vecTensorTy.getContext(), sizePerCore, enc.getCoresPerGroup().vec(),
            enc.getGroupsPerCluster().vec(), enc.getOrder().vec()));
    if (getVectorType(scalarTy) != vecTensorTy)
      return {};
    return scalarTy;
  }

  void insertProbeFiller(ModuleOp &mod, unsigned k, bool scalarSide) {
    // One site only, so the boundary count is exactly one pair and the filler
    // count exactly k, not k * sites.
    triton::xpu::StoreOp site;
    RankedTensorType vecTy, scalarTy;
    mod.walk([&](triton::xpu::StoreOp storeOp) {
      if (site)
        return;
      auto ty = mlir::dyn_cast<RankedTensorType>(storeOp.getValue().getType());
      if (!ty)
        return;
      // f32 only: the filler is exact because 2.0 and 0.5 are powers of two,
      // and vvmulf on the narrower float types goes through width handling this
      // probe has no reason to drag in.
      auto vecElemTy = mlir::dyn_cast<VectorType>(ty.getElementType());
      if (!vecElemTy || !vecElemTy.getElementType().isF32())
        return;
      auto sTy = getScalarTypeOrNull(ty);
      if (!sTy)
        return;
      site = storeOp;
      vecTy = ty;
      scalarTy = sTy;
    });
    if (!site) {
      mod->emitRemark(
          "[ProbeFiller] no vectorized f32 store site with an exact "
          "scalar inverse; probe inactive");
      return;
    }

    OpBuilder builder(site);
    auto loc = site.getLoc();
    // Splat attributes on the *scalar* tensor type. VConstOp takes the scalar
    // form of the attribute together with the vector result type, exactly as
    // processOpVecTy builds it from an arith::ConstantOp.
    auto twoAttr = DenseElementsAttr::get(
        scalarTy, ArrayRef<Attribute>{builder.getF32FloatAttr(2.0f)});
    auto halfAttr = DenseElementsAttr::get(
        scalarTy, ArrayRef<Attribute>{builder.getF32FloatAttr(0.5f)});

    Value v = site.getValue();
    if (scalarSide) {
      Value sv = builder.create<triton::xpu::UnpackOp>(loc, scalarTy, v,
                                                       /*bufPtr=*/Value());
      Value two = builder.create<arith::ConstantOp>(loc, scalarTy, twoAttr);
      Value half = builder.create<arith::ConstantOp>(loc, scalarTy, halfAttr);
      for (unsigned i = 0; i < k; ++i) {
        sv = builder.create<arith::MulFOp>(loc, sv, two);
        sv = builder.create<arith::MulFOp>(loc, sv, half);
      }
      v = builder.create<triton::xpu::PackOp>(loc, vecTy, sv,
                                              /*bufPtr=*/Value());
    } else {
      Value two = builder.create<triton::xpu::VConstOp>(loc, vecTy, twoAttr);
      Value half = builder.create<triton::xpu::VConstOp>(loc, vecTy, halfAttr);
      for (unsigned i = 0; i < k; ++i) {
        v = builder.create<triton::xpu::VvmulFOp>(loc, vecTy, v, two);
        v = builder.create<triton::xpu::VvmulFOp>(loc, vecTy, v, half);
      }
    }
    site.getValueMutable().assign(v);

    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "[ProbeFiller] k=" << k << " side=" << (scalarSide ? "scalar" : "vec")
       << " boundaryPairs=" << (scalarSide ? 1 : 0)
       << " scalarElems=" << getTotalElemsPerThread(Type(scalarTy))
       << " vecSlots=" << getTotalElemsPerThread(Type(vecTy));
    mod->emitRemark(msg);
  }

  // Collapse a scalar TLE parameter chain into the vector load it already is.
  //
  // triton_xpu.broadcast is the vector/scalar boundary (processOpVecTy retypes
  // only its result), so XPUBroadcastOpConversion packs each output vector lane
  // by lane, a vmmov + vor.u.mh apiece because xpu3 has no cross-lane permute.
  // The scalar source is a contiguous LM read -- tt.load on a tle_local_ptr,
  // optionally widened by arith.extf -- which is one tle_vload, so the
  // broadcast goes away with it. Runs after the store walk, where a vectorized
  // result over a scalar source is exactly that boundary.
  // A `tle_local_ptr` retyped one pointer per VECTOR slot. tt.load/tt.store
  // require "value type == pointee type", and a bf16 memory boundary keeps the
  // memory op as a tt.load/tt.store (see the `xpu.lm_bf16` branches), so the
  // pointer tensor has to be widened together with the value.
  Value makeVectorTLELocalPtr(OpBuilder &b, triton::xpu::TLELocalPtrOp lp,
                              RankedTensorType vecTy) {
    auto oriTy = cast<RankedTensorType>(lp.getResult().getType());
    auto oriPtrTy = cast<triton::PointerType>(oriTy.getElementType());
    Type newPtrTy = triton::PointerType::get(vecTy.getElementType(),
                                             oriPtrTy.getAddressSpace());
    auto newTy =
        RankedTensorType::get(vecTy.getShape(), newPtrTy, vecTy.getEncoding());
    return b
        .create<triton::xpu::TLELocalPtrOp>(lp.getLoc(), newTy, lp.getBuffer(),
                                            lp.getIndices(), lp.getLoopIndex())
        .getResult();
  }

  // What a region can feed the enclosing op's results flows out through
  // block terminators (scf yields), so walking the terminator operands is
  // sufficient for structured control flow -- and avoids dragging in values
  // that never reach the result (e.g. an arange used only to compute load
  // addresses inside the loop). Region ops outside the modeled set keep the
  // conservative full scan, so the walk never wrongly permits.
  template <typename Recurse>
  static bool walkRegionDeps(Operation *def, DenseSet<Operation *> &visited,
                             Recurse &&recurse) {
    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ExecuteRegionOp>(def)) {
      for (Region &region : def->getRegions())
        for (Block &block : region)
          if (Operation *term = block.getTerminator())
            for (Value operand : term->getOperands())
              if (recurse(operand, visited))
                return true;
      return false;
    }
    for (Region &region : def->getRegions())
      for (Block &block : region)
        for (Operation &op : block)
          for (Value operand : op.getOperands())
            if (recurse(operand, visited))
              return true;
    return false;
  }

  // Does a stored value depend on which lane an element sits in? Ported from
  // the private chain: the cheap bf16 placement permutes elements inside each
  // register pair, and load and store cancel only if nothing in between cares
  // about the lane order.
  static bool tleDependsOnLaneIndex(Value v, DenseSet<Operation *> &visited) {
    Operation *def = v.getDefiningOp();
    if (!def || isa<triton::LoadOp, triton::xpu::TLEVLoadOp>(def))
      return false;
    if (isa<triton::xpu::MakeRangeOp, triton::MakeRangeOp>(def))
      return true;
    if (!visited.insert(def).second)
      return false;
    // A loop's result comes from its BODY (the yield chain), not just from
    // the iter-arg init values the plain operand walk would follow. A
    // k29-style `acc = scf.for(...)` hides the whole compute tree --
    // including the lane-index-derived mask feeding a select -- inside the
    // region, and skipping it wrongly let the bf16 unordered (lane-
    // permuting) placement through: the mask's bits no longer lined up
    // with the permuted lanes, so `where` replaced 4 valid lanes (max
    // column 56-59 of 60) with the constant. Iter-arg back edges are
    // covered because the yield operands are walked themselves.
    if (walkRegionDeps(def, visited, tleDependsOnLaneIndex))
      return true;
    for (Value operand : def->getOperands())
      if (tleDependsOnLaneIndex(operand, visited))
        return true;
    return false;
  }

  // A load whose result registers the unordered placement permutes (f32
  // result over bf16 memory). A chain without one of these keeps original
  // lane order throughout.
  static bool isBf16ToF32Load(Operation *def) {
    auto loadOp = dyn_cast<triton::xpu::LoadOp>(def);
    if (!loadOp)
      return false;
    Type ptrElemTy = getElementTypeOrSelf(loadOp.getPtr().getType());
    Type ptrDataTy = mlir::cast<PointerType>(ptrElemTy).getPointeeType();
    Type resElemTy = getElementTypeOrSelf(loadOp.getResult().getType());
    Type resScalarTy = getElementTypeOrSelf(resElemTy);
    return resScalarTy.isF32() && ptrDataTy.isBF16();
  }

  // Does the chain carry registers converted at a bf16 -> f32 load? Only
  // those registers are lane-permuted, so only such chains can pair a
  // lane-ordered value with permuted data.
  static bool sliceHasBf16ToF32Load(Value v, DenseSet<Operation *> &visited) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    if (isBf16ToF32Load(def))
      return true;
    if (isa<triton::LoadOp, triton::xpu::TLEVLoadOp>(def))
      return false;
    if (!visited.insert(def).second)
      return false;
    if (walkRegionDeps(def, visited, sliceHasBf16ToF32Load))
      return true;
    for (Value operand : def->getOperands())
      if (sliceHasBf16ToF32Load(operand, visited))
        return true;
    return false;
  }

  // The bf16 <-> f32 placement has a cheaper form (masked loads / masked stores
  // instead of the vmerge and vscatter pairs) that permutes the elements inside
  // each register pair. Load and store permute the same way, so the permutation
  // cancels in memory -- but only if nothing in between depends on which lane
  // an element sits in. Mirrors BF16ToFP32VecOptimize on the GM staging path,
  // with the TLE-specific conditions:
  //   * no plain (non-bf16) vectorized TLE load/store is left. Those registers
  //     are NOT permuted, so mixing one in -- e.g. an f32 gamma column
  //     broadcast against a permuted bf16 tile -- would pair up the wrong
  //     elements.
  //   * no tl.arange-derived value reaches a stored value: a lane index means
  //     the lane order is the result.
  void decideTLEBf16Unordered(ModuleOp &mod) {
    bool unordered = true;
    unsigned bf16Ops = 0;
    mod.walk([&](Operation *op) {
      if (isa<triton::xpu::TLEVLoadOp, triton::xpu::TLEVStoreOp>(op)) {
        unordered = false;
        return;
      }
      if (!op->hasAttr("xpu.lm_bf16"))
        return;
      ++bf16Ops;
      if (auto st = dyn_cast<triton::StoreOp>(op)) {
        DenseSet<Operation *> visited;
        if (tleDependsOnLaneIndex(st.getValue(), visited))
          unordered = false;
      }
    });
    if (!unordered || bf16Ops == 0)
      return;
    OpBuilder b(mod);
    mod.walk([&](Operation *op) {
      if (op->hasAttr("xpu.lm_bf16"))
        op->setAttr("xpu.bf16_unordered", b.getUnitAttr());
    });
    LLVM_DEBUG(llvm::dbgs() << "[Vectorization]: Apply TLE BF16<->FP32 "
                               "unordered placement.\n");
  }

  void collapseTLEBroadcastSource(ModuleOp &mod) {
    SmallVector<triton::xpu::BroadcastOp> bcOps;
    mod.walk([&](triton::xpu::BroadcastOp bcOp) { bcOps.push_back(bcOp); });

    SmallVector<Operation *> dead;
    for (auto bcOp : bcOps) {
      auto resTy = dyn_cast<RankedTensorType>(bcOp.getResult().getType());
      if (!resTy || !isa<VectorType>(resTy.getElementType()))
        continue;
      auto srcTy = dyn_cast<RankedTensorType>(bcOp.getSrc().getType());
      if (!srcTy || isa<VectorType>(srcTy.getElementType()))
        continue;
      // Only [1xN] -> [MxN]. The [Mx1] form broadcasts one scalar per row and
      // has no vector in the buffer to read: that one is a genuine splat.
      if (srcTy.getRank() != 2 || srcTy.getShape()[0] != 1)
        continue;

      // expand_dims and convert_layout are reshapes of the same values, and
      // they appear on both sides of the widening cast: the TLE type conversion
      // leaves one between the load and the extf, and the expand_dims path adds
      // another between the extf and the broadcast. So the strip runs on both
      // sides.
      auto stripReshapes = [&](Value v) {
        while (true) {
          Operation *def = v.getDefiningOp();
          if (auto expandOp = dyn_cast_or_null<triton::ExpandDimsOp>(def)) {
            dead.push_back(expandOp);
            v = expandOp.getSrc();
          } else if (auto cvtOp =
                         dyn_cast_or_null<triton::xpu::ConvertLayoutOp>(def)) {
            dead.push_back(cvtOp);
            v = cvtOp.getOperand();
          } else {
            return v;
          }
        }
      };
      Value cur = stripReshapes(bcOp.getSrc());

      // The buffer may hold f16 that the compute chain wants as f32, which is
      // what vextf does on the 2-D input path.
      bool needsExtF = false;
      if (auto extFOp = dyn_cast_or_null<arith::ExtFOp>(cur.getDefiningOp())) {
        needsExtF = true;
        dead.push_back(extFOp);
        cur = stripReshapes(extFOp.getIn());
      }
      auto loadOp = dyn_cast_or_null<triton::LoadOp>(cur.getDefiningOp());
      if (!loadOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[TLE Broadcast Source] not a tt.load after the "
                      "reshapes: "
                   << (cur.getDefiningOp()
                           ? cur.getDefiningOp()->getName().getStringRef()
                           : llvm::StringRef("block-arg"))
                   << " needsExtF=" << needsExtF << " res=" << resTy << "\n");
        continue;
      }
      auto lpOp = getTLELocalPtrThroughCvt(loadOp.getPtr());
      if (!lpOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[TLE Broadcast Source] tt.load ptr is not a "
                      "tle_local_ptr, res="
                   << resTy << "\n");
        continue;
      }
      // A `scope=smem` buffer is shared by the whole cluster, so its index
      // tensor selects the elements and only `[uniform +] arange(0, n)` has a
      // vector form (same admission check as the leaf-load case). Refusing here
      // leaves the scalar SM read, which honors the indices element by element.
      Value smOff;
      if (!tleSmemSliceOffset(lpOp, smOff, this->tleSmemVec)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[TLE Broadcast Source] smem index has no vector form, "
                      "res="
                   << resTy << "\n");
        continue;
      }

      // Derive the loaded type through getVectorType so the slot width and the
      // trailing-dim division stay this pass's own convention rather than a
      // second copy of it: undo the vectorization of the broadcast's result to
      // get the scalar shape, put the buffer's element type on it, vectorize
      // that. Without a cast the result type is already what to read.
      auto memDescTy =
          cast<triton::gpu::MemDescType>(lpOp.getBuffer().getType());
      Type bufElemTy = memDescTy.getElementType();
      RankedTensorType loadVecTy = resTy;
      if (needsExtF) {
        auto scalarResTy = getScalarTypeOrNull(resTy);
        if (!scalarResTy) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[TLE Broadcast Source] no exact scalar inverse for "
                     << resTy << "\n");
          continue;
        }
        auto narrowTy = RankedTensorType::get(scalarResTy.getShape(), bufElemTy,
                                              scalarResTy.getEncoding());
        loadVecTy = dyn_cast<RankedTensorType>(getVectorType(narrowTy));
        if (!loadVecTy) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[TLE Broadcast Source] no vector type for " << narrowTy
                     << "\n");
          continue;
        }
        if (!isa<VectorType>(loadVecTy.getElementType())) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[TLE Broadcast Source] getVectorType left " << narrowTy
                     << " scalar\n");
          continue;
        }
      }
      // A bf16 buffer whose registers are f32 is the fused memory boundary, not
      // "some other conversion": tritonxpu-tle-dtype-convert already promoted
      // the chain and marked the load. The load then keeps its own f32 type,
      // and a VExtF on top of it would double the conversion, so a marked load
      // that still carries an extf is left alone.
      bool bf16Boundary =
          loadOp->hasAttr("xpu.lm_bf16") && !tleBufferIsSmem(lpOp.getBuffer());
      if (bf16Boundary && needsExtF) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[TLE Broadcast Source] bf16 boundary still carries an "
                      "extf, leaving it alone, res="
                   << resTy << "\n");
        continue;
      }
      if (!bf16Boundary && !needsExtF &&
          bufElemTy != getElementTypeOrSelf(resTy.getElementType())) {
        // Some other conversion sits in between; leave the chain alone.
        continue;
      }

      OpBuilder builder(bcOp);
      auto loc = bcOp.getLoc();
      // Two distinct quantities, two operands: the tile index the local_ptr
      // already carries stays $loopIndex, the SM index's uniform addend goes to
      // $smElemOffset. Sharing one slot read another core's columns.
      Value loaded;
      if (bf16Boundary) {
        Value vecPtr = makeVectorTLELocalPtr(builder, lpOp, loadVecTy);
        auto newLd = builder.create<triton::LoadOp>(
            loc, vecPtr, loadOp.getCache(), loadOp.getEvict(),
            loadOp.getIsVolatile());
        newLd->setAttr("xpu.lm_bf16", builder.getUnitAttr());
        loaded = newLd.getResult();
      } else {
        loaded = builder
                     .create<triton::xpu::TLEVLoadOp>(
                         loc, loadVecTy, lpOp.getBuffer(), lpOp.getLoopIndex(),
                         /*smElemOffset=*/smOff)
                     .getResult();
      }
      if (needsExtF)
        loaded = builder.create<triton::xpu::VExtFOp>(loc, resTy, loaded);
      bcOp.getResult().replaceAllUsesWith(loaded);
      dead.push_back(bcOp);
      dead.push_back(loadOp);
      dead.push_back(lpOp);

      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "[Vectorization]: [TLE Broadcast Source] collapsed to "
         << (bf16Boundary ? "bf16 tt.load"
                          : (needsExtF ? "vextf(tle_vload)" : "tle_vload"))
         << " for " << resTy;
      LLVM_DEBUG(llvm::dbgs() << msg << "\n");
    }

    // One tle_local_ptr / convert_layout can feed several broadcast chains: a
    // 2x-unrolled TLE body reads the same [W] weight buffer once per half beat,
    // so the same Operation is pushed here more than once. The loop below nulls
    // only the slot it erased, leaving the duplicate dangling, and erasing it
    // again is a double free -- glibc aborts inside this pass, and MLIR's crash
    // reproducer then deadlocks in malloc, so it presents as a hang. Seen on
    // xuchen18's layernorm at XBLOCK=32 ROWITERS=4.
    llvm::SetVector<Operation *> uniqueDead(dead.begin(), dead.end());
    dead.assign(uniqueDead.begin(), uniqueDead.end());

    // Erase to fixpoint and only what is genuinely unused: the same pointer op
    // can also feed a mask chain, and convert_layout results are shared.
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &op : dead) {
        if (op && op->use_empty()) {
          op->erase();
          op = nullptr;
          changed = true;
        }
      }
    }
  }

  void runOnOperation() override {
    context = &getContext();
    ModuleOp mod = getOperation();

    LLVM_DEBUG(llvm::dbgs() << __FILE__ << " START\n" << mod << "\n");

    // The prologue rewrites that used to sit here (erf lowering, maximum
    // fusion, the two compare fusions, the i1 -> i8 pattern set) now run in
    // tritonxpu-normalize, immediately before this pass. They were not
    // vectorization, and keeping them here meant no analysis could see the IR
    // this pass actually vectorizes (step 1.5).
    //
    // Their only output besides the rewritten IR is the marker below.
    if (mod->hasAttr(kBF16ToFP32VecOptOffAttrName)) {
      BF16ToFP32VecOpt = false;
      mod->removeAttr(kBF16ToFP32VecOptOffAttrName);
    }

    // Eliminate SelectOp For bufferSize X Col Size
    // TODO[dyq]: open isMultipleOfBank
    // if (isMultipleOfBank(mod)) {
    //   mod.walk([&](arith::SelectOp selectOp) {
    //     // Have Only One User(ReduceOp)
    //     if (selectOp.getResult().hasOneUse()) {
    //       auto userOp = *selectOp->user_begin();
    //       if (auto redOp = dyn_cast<triton::xpu::ReduceOp>(userOp)) {
    //         auto trueVal = selectOp.getTrueValue();
    //         auto trueValOp = trueVal.getDefiningOp();

    //         selectOp->replaceAllUsesWith(trueValOp->getResults());
    //         selectOp->erase();
    //         LLVM_DEBUG(llvm::dbgs() << "[Vectorization]: Eliminate SelectOp
    //         For "
    //                         "bufferSize X Col Size.\n");
    //       }
    //     }
    //   });
    // }

    if (ReduceVec) {
      // For [Load -> Reduce] || [Broadcast -> Reduce]
      llvm::SetVector<triton::xpu::ReduceOp> reduceOps;
      mod.walk([&](triton::xpu::ReduceOp redOp) { reduceOps.insert(redOp); });

      for (auto redOp : reduceOps) {
        // A combine region that cannot be retyped keeps the whole reduce, and
        // its producer chain, scalar. Not because of the check here -- the
        // closure walk already reaches this reduce as a *user* of the producer
        // and vetoes there (VectorizabilityAnalysis.cpp:157-181 feeding
        // :400-408 via the user loop at :414-431). Measured: disabling this
        // branch leaves the emitted code bit-identical.
        //
        // The check is here as the hook for the boundary: M3.2 replaces the
        // `continue` with an inserted triton_xpu.unpack, which is what breaks
        // that veto -- the walk then terminates at the unpack instead of at the
        // reduce, so the producer chain vectorizes while the reduce stays
        // scalar.
        if (!reduceCombineIsVectorizable(redOp)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "[Vectorization]: Reduce combine region is not "
                        "vector-representable, leaving the reduce scalar\n");
          continue;
        }

        if (reduceCombineRegionEnabled()) {
          SmallVector<Operation *> rootOps;
          for (int i = 0; i < redOp.getOperands().size() - 1; ++i)
            rootOps.push_back(redOp.getOperands()[i].getDefiningOp());
          vectorizeAndProcessOpVecTyShared(
              mod, rootOps, redOp.getOperands()[0].getType(),
              "[Vectorization]: [Load -> Reduce] || [Broadcast -> Reduce] Hit.",
              redOp);
        } else {
          for (int i = 0; i < redOp.getOperands().size() - 1; ++i) {
            auto reduceOperand = redOp.getOperands()[i];
            auto reduceOperandOp = reduceOperand.getDefiningOp();
            auto reduceOperandTy = reduceOperand.getType();
            vectorizeAndProcessOpVecTy(mod, reduceOperandOp, reduceOperandTy,
                                       "[Vectorization]: [Load -> "
                                       "Reduce] || [Broadcast -> Reduce] Hit.",
                                       redOp);
          }
        }

        ReduceOpHelper help(redOp);
        if (help.isVectorized()) {
          // reduceop's correct encoding should be inferd by its input type.
          auto srcLayout = help.getSrcLayout();
          for (Value redRes : redOp.getResults()) {
            if (auto resTy = dyn_cast<RankedTensorType>(redRes.getType())) {
              auto resSliceEncoding =
                  cast<triton::gpu::SliceEncodingAttr>(resTy.getEncoding());
              auto srcClusterEncoding =
                  cast<triton::xpu::ClusterLayoutAttr>(srcLayout);
              auto newEncoding = triton::gpu::SliceEncodingAttr::get(
                  redOp.getContext(), resSliceEncoding.getDim(),
                  srcClusterEncoding);
              auto newResTy = RankedTensorType::get(
                  resTy.getShape(), resTy.getElementType(), newEncoding);

              redRes.setType(newResTy);
            }
          }

          for (Block &block : redOp.getCombineOp().getBlocks()) {
            // Set Arg's Type to VecType
            auto inputTypes = redOp.getInputTypes();
            auto inputSize = inputTypes.size();
            int vecSize = 16;
            for (int i = 0; i < inputSize; ++i) {
              auto vecTy = getElementTypeOrSelf(inputTypes[i]);
              vecSize = cast<VectorType>(vecTy).getNumElements();
              auto arg1 = block.getArguments()[i];
              auto arg2 = block.getArguments()[inputSize + i];
              arg1.setType(vecTy);
              arg2.setType(vecTy);
            }
            // Every value the region's ops consume has to be vector by the
            // time the block is retyped. Scalar constants are not: welford's
            // combine reads a captured `arith.constant 0.0 : f32`, and a
            // constant sunk into the region keeps its scalar type because the
            // retype below leaves constants alone. Either way the result is a
            // cmpf/select with one vector and one scalar operand, which does
            // not verify. Rematerialize them as in-region splats.
            {
              OpBuilder builder(&block, block.begin());
              for (Operation &op : block) {
                for (OpOperand &use : op.getOpOperands()) {
                  Value v = use.get();
                  if (isa<VectorType>(v.getType()))
                    continue;
                  auto cstOp = v.getDefiningOp<arith::ConstantOp>();
                  if (!cstOp)
                    continue; // excluded by reduceCombineIsVectorizable
                  auto vecTy = VectorType::get(vecSize, v.getType());
                  auto splat = DenseElementsAttr::get(
                      cast<ShapedType>(Type(vecTy)),
                      ArrayRef<Attribute>{cstOp.getValue()});
                  use.set(builder.create<arith::ConstantOp>(cstOp.getLoc(),
                                                            vecTy, splat));
                }
              }
            }

            // Set CombineOp's Type to VecType
            for (auto &op : block) {
              TypeSwitch<Operation *>(&op)
                  .Case<REDUCE_COMBINE_OP, arith::SubFOp, arith::DivFOp,
                        arith::SelectOp, COMBINE_OP_TLE_EXT>(
                      [&](auto redComOp) {
                        // The non-COMBINE_OP kinds are only admitted by
                        // reduceCombineIsVectorizable while the region lowering
                        // is on (TRITONXPU_REDUCE_REGION unset or 1), so
                        // reaching them here implies the region-interpreting
                        // lowering is in use.
                        for (auto res : redComOp->getResults()) {
                          auto elemTy = res.getType();
                          VectorType vecType = VectorType::get(vecSize, elemTy);
                          res.setType(vecType);
                        }
                      })
                  .Case<arith::ConstantOp>([&](auto cstOp) {
                    // Left scalar on purpose: emitCombineOp splats it to the
                    // vector shape of whichever op consumes it.
                  })
                  .Default([&](auto defaultOp) {
                    LLVM_DEBUG(defaultOp->dump());
                    llvm_unreachable(
                        "[Vectorization]: Unsupported Operation Type "
                        "To VecType in Reduce");
                  });
            }
          }
        }
      }
    }

    // For [Broadcast -> Store]
    mod.walk([&](triton::xpu::StoreOp storeOp) {
      auto storeOpValueTy = storeOp.getValue().getType();
      vectorizeAndProcessOpVecTy(mod, storeOp, storeOpValueTy,
                                 "[Vectorization]: [Broadcast -> Store] Hit.");
    });

    // The TLE store root. Same shape as the walk above, with the root type
    // taken from behind the convert_layout the TLE type conversion left on the
    // value crossing into the store (`convert_layout tensor<64x256xf16,
    // #cluster1> -> tensor<64x256xf16>`), and every gate downstream reads
    // sizePerCore off that encoding -- `vectorFitsRoot` casts it
    // unconditionally. The encoded producer is what the chain actually is, so
    // that is the type the decision has to see.
    mod.walk([&](triton::StoreOp storeOp) {
      if (!storeOp.getPtr().getDefiningOp<triton::xpu::TLELocalPtrOp>())
        return;
      Value encoded = storeOp.getValue();
      while (auto cvt = dyn_cast_or_null<triton::xpu::ConvertLayoutOp>(
                 encoded.getDefiningOp())) {
        // Step back only while the encoding is still missing: the nearest
        // encoded ancestor is the root, not the oldest one.
        if (cast<RankedTensorType>(cvt.getResult().getType()).getEncoding())
          break;
        encoded = cvt.getOperand();
      }
      auto rootTy = dyn_cast<RankedTensorType>(encoded.getType());
      if (!rootTy || !rootTy.getEncoding())
        return;
      vectorizeAndProcessOpVecTy(mod, storeOp, rootTy,
                                 "[Vectorization]: [TLE Store] Hit.");
    });

    // The chains feeding the TLE segment's [1xN] broadcasts are still scalar at
    // this point; pull them into vector form so the broadcast does not have to
    // pack its source lane by lane.
    collapseTLEBroadcastSource(mod);
    // After every TLE memory op has its final form: the decision needs to see
    // whether any plain tle_vload/tle_vstore is left.
    decideTLEBf16Unordered(mod);

    // P4 probe: last stage that can still see fully vectorized store values,
    // before the optional cleanup/fusion stages rewrite them.
    if (probeActive())
      insertProbeFiller(mod, probeFillerCount(), probeScalarSide());

    // Eliminate CvtOp in VVOp Path
    if (cvtOp_clean) {
      mod.walk([&](triton::xpu::ConvertLayoutOp cvtOp) { cvtOpclean(cvtOp); });
    }

    // Div -> Mul
    if (Div2Mul) {
      mod.walk([&](triton::xpu::VvdivFOp vvdivFOp) { VvdivToVvmul(vvdivFOp); });
    }

    // SV Optimization offline
    if (SV_Fusion) {
      // SVOptimization For LoadOp
      mod.walk([&](triton::xpu::LoadOp vLoadOp) {
        vectorizedLoadOps.insert(vLoadOp);
      });
      for (auto vLoadOp : vectorizedLoadOps) {
        SVOptimization(vLoadOp,
                       "[Vectorization]: Apply SV Optimization For LoadOp.\n");
      }

      // SVOptimization For BroadcastOp
      mod.walk([&](triton::xpu::BroadcastOp vBCOp) {
        vectorizedBcOps.insert(vBCOp);
      });
      for (auto vBCOp : vectorizedBcOps) {
        SVOptimization(
            vBCOp, "[Vectorization]: Apply SV Optimization For BroadcastOp.\n");
      }

      // SVOptimization For ConstOp
      mod.walk([&](triton::xpu::VConstOp vConstOp) {
        vectorizedConstOps.insert(vConstOp);
      });
      for (auto vConstOp : vectorizedConstOps) {
        SVOptimization(
            vConstOp, "[Vectorization]: Apply SV Optimization For VConstOp.\n");
      }
    }

    // Sigmoid Fusion: vvdivf(1, vvaddf(vexpf(svsubf(0, x)), 1)) -> vsigmoidf(x)
    // Must run after SV Optimization (svsubf is created by SV opt from vvsubf)
    // and before VMAC Fusion.
    mod.walk([&](triton::xpu::VvdivFOp divOp) {
      // Check: divOp.lhs is splat(1.0)
      auto lhsOp = divOp.getLhs().getDefiningOp();
      if (!lhsOp)
        return;
      bool lhsIsOne = false;
      if (auto vconstOp = dyn_cast<triton::xpu::VConstOp>(lhsOp)) {
        if (auto denseAttr =
                dyn_cast<DenseFPElementsAttr>(vconstOp.getValue())) {
          if (denseAttr.isSplat() &&
              denseAttr.getSplatValue<APFloat>().convertToFloat() == 1.0f)
            lhsIsOne = true;
        }
      }
      if (!lhsIsOne)
        return;

      // Check: divOp.rhs is vvaddf
      auto addOp = divOp.getRhs().getDefiningOp<triton::xpu::VvaddFOp>();
      if (!addOp)
        return;

      // Check: one operand of add is VExpFOp, the other is the same splat(1.0)
      Value expVal = nullptr;
      auto addLhsOp = addOp.getLhs().getDefiningOp();
      auto addRhsOp = addOp.getRhs().getDefiningOp();
      if (isa_and_nonnull<triton::xpu::VExpFOp>(addLhsOp) &&
          addRhsOp == lhsOp) {
        expVal = addOp.getLhs();
      } else if (isa_and_nonnull<triton::xpu::VExpFOp>(addRhsOp) &&
                 addLhsOp == lhsOp) {
        expVal = addOp.getRhs();
      }
      if (!expVal)
        return;

      // Check: exp input is neg(x*log2e) — svsubf(0, x*log2e)
      // In XPU backend, math.exp(x) is lowered as VExpFOp which maps to
      // LLVM::Exp2Op. The input already contains the log2e multiplication.
      // We pass the exp's input directly (which is -x*log2e) to VSigmoidFOp.
      auto expOp = expVal.getDefiningOp<triton::xpu::VExpFOp>();
      Value negXLog2e = expOp.getValue();
      // Verify it's from svsubf (negation pattern)
      if (!negXLog2e.getDefiningOp<triton::xpu::SvsubFOp>())
        return;

      // Pattern matched! Replace with VSigmoidFOp
      // VSigmoidFOp input is -x*log2e (ready for exp2)
      OpBuilder builder(divOp);
      auto sigmoidOp = builder.create<triton::xpu::VSigmoidFOp>(
          divOp.getLoc(), divOp.getType(), negXLog2e);
      divOp.replaceAllUsesWith(sigmoidOp.getResult());
      divOp.erase();
      if (addOp->use_empty())
        addOp.erase();
      if (expOp->use_empty())
        expOp.erase();
      LLVM_DEBUG(llvm::dbgs() << "[Vectorization]: Sigmoid Fusion applied.\n");
    });

    // MAC Optimization offline
    if (VMAC_Fusion) {
      mod.walk([&](triton::xpu::VvmulFOp vvmulFOp) { // must walk after svOpt
        vvmulFOps.insert(vvmulFOp);
      });

      for (auto vvmulFOp : vvmulFOps) {
        VVMacOpFusion(vvmulFOp);
      }
    }

    // bfloat16 -> float32 Vector Optimization
    if (BF16ToFP32VecOpt) {
      BF16ToFP32VecOptimize(mod);
    }

    // Move BitWise Op From Other Type To int32x16
    {
      RewritePatternSet patterns(context);
      patterns.add<BitwiseCastToI32Pattern<xpu::VvxorIOp>,
                   BitwiseCastToI32Pattern<xpu::VvandIOp>,
                   BitwiseCastToI32Pattern<xpu::VvorIOp>,
                   BitwiseCastToI32Pattern<xpu::SvxorIOp>>(context);
      if (failed(applyPatternsGreedily(mod, std::move(patterns))))
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << __FILE__ << " END\n" << mod << "\n");
  }

private:
  MLIRContext *context;
  llvm::SetVector<triton::xpu::VvmulFOp> vvmulFOps;
  llvm::SetVector<triton::xpu::BroadcastOp> vectorizedBcOps;
  llvm::SetVector<triton::xpu::LoadOp> vectorizedLoadOps;
  llvm::SetVector<triton::xpu::VConstOp> vectorizedConstOps;
  bool SV_Fusion = true;
  bool VMAC_Fusion = true;
  bool cvtOp_clean = true;
  bool Div2Mul = true;
  bool ReduceVec = true;
  bool BF16ToFP32VecOpt = true;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
