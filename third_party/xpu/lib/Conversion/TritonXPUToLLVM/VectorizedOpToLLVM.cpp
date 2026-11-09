// clang-format off
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "PatternTritonXPUOpToLLVM.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h"  // LLVM22 dragon-style macros for XPU only
// clang-format on

namespace {
// TODO[dyq]: add to head file
enum class ElemState {
  SS = 0, /*00*/
  SV = 1, /*01*/
  VS = 2, /*10*/
  VV = 3  /*11*/
};

template <typename OP> struct SVOp2Str;

#define SVOp2ASMStr(SrcType, ASM_STR)                                          \
  template <> struct SVOp2Str<SrcType> {                                       \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef SVOp2Str<SrcType>::value = ASM_STR;

SVOp2ASMStr(triton::xpu::SvaddFOp, "vadd.f.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStr(triton::xpu::SvmulFOp, "vmul.f.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStr(triton::xpu::SvsubFOp, "vsub.f.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStr(triton::xpu::SvmaxFOp, "vmax.f.mz $0{mr1}, $1, $2");
SVOp2ASMStr(triton::xpu::SvxorIOp, "vxor.s.mz $0{mr1}, $1, $2");

template <typename OP> struct SVOp2StrFP16;

#define SVOp2ASMStrFP16(SrcType, ASM_STR)                                      \
  template <> struct SVOp2StrFP16<SrcType> {                                   \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef SVOp2StrFP16<SrcType>::value = ASM_STR;

SVOp2ASMStrFP16(triton::xpu::SvaddFOp, "vadd.hf.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStrFP16(triton::xpu::SvmulFOp, "vmul.hf.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStrFP16(triton::xpu::SvsubFOp, "vsub.hf.mz.rn $0{mr1}, $1, $2");
SVOp2ASMStrFP16(triton::xpu::SvmaxFOp, "vmax.hf.mz $0{mr1}, $1, $2");
SVOp2ASMStrFP16(triton::xpu::SvxorIOp, "");

template <typename OP> struct VLibOp;

#define VLibOp2DevCall(SrcType, ASM_STR)                                       \
  template <> struct VLibOp<SrcType> {                                         \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef VLibOp<SrcType>::value = ASM_STR;

VLibOp2DevCall(triton::xpu::VSinFOp, "_ZN3xpu5vsinfEDv16_f");
VLibOp2DevCall(triton::xpu::VCosFOp, "_ZN3xpu5vcosfEDv16_f");
VLibOp2DevCall(triton::xpu::VSigmoidFOp, "_ZN3xpu9vsigmoidfEDv16_f");

template <typename OP, int ARCH> struct VLibOpFP16;

#define VLibOpFP162DevCall(SrcType, ARCH, ASM_STR)                             \
  template <> struct VLibOpFP16<SrcType, ARCH> {                               \
    static const llvm::StringRef value;                                        \
  };                                                                           \
  const llvm::StringRef VLibOpFP16<SrcType, ARCH>::value = ASM_STR;

VLibOpFP162DevCall(triton::xpu::VSinFOp, 2, "_ZN3xpu5vsinfEDv32_t");
VLibOpFP162DevCall(triton::xpu::VCosFOp, 2, "_ZN3xpu5vcosfEDv32_t");
VLibOpFP162DevCall(triton::xpu::VSinFOp, 3, "_ZN3xpu5vsinfEDv32_DF16_");
VLibOpFP162DevCall(triton::xpu::VCosFOp, 3, "_ZN3xpu5vcosfEDv32_DF16_");
VLibOpFP162DevCall(triton::xpu::VSigmoidFOp, 2, "_ZN3xpu9vsigmoidfEDv32_t");
VLibOpFP162DevCall(triton::xpu::VSigmoidFOp, 3, "_ZN3xpu9vsigmoidfEDv32_DF16_");

} // namespace

namespace {

using namespace mlir;
using namespace mlir::triton;
using ::mlir::triton::gpu::getTotalElemsPerThread;

struct XPUVectorizedOpsConversionBase {

  explicit XPUVectorizedOpsConversionBase(
      const triton::xpu::TargetInfo &targetInfo) {
    switch (static_cast<XPUArch>(targetInfo.getXPUArch())) {
    case XPUArch::XPU2: {
      xpuArch = 2;
      break;
    }
    case XPUArch::XPU3: {
      xpuArch = 3;
      break;
    }
    default:
      // Pattern constructors also run for SDNN-only modules on newer
      // architectures.
      xpuArch = targetInfo.getXPUArch();
      break;
    }
  }

  unsigned getVectorSize(Type type) const {
    auto vectorTy = mlir::dyn_cast<mlir::VectorType>(type);
    if (!vectorTy)
      return 1;
    auto elemTy = vectorTy.getElementType();
    auto width = elemTy.getIntOrFloatBitWidth();

    auto shape = vectorTy.getShape();
    if (shape[0] != 16) { // return vecSize = numElems for vector<numElemsxf32>
      return shape[0];
    }

    return 512 / width;
  }

  Type convertVectorType(Type type) const {
    auto vectorType = mlir::cast<mlir::VectorType>(type);
    auto ctx = vectorType.getContext();
    auto elemTy = vectorType.getElementType();
    if (elemTy.isF16())
      return LLVM::getVectorType(LLVM::type::f16Ty(ctx), getVectorSize(type));
    else if (elemTy.isF32())
      return LLVM::getVectorType(LLVM::type::f32Ty(ctx), getVectorSize(type));
    else if (elemTy.isInteger(8))
      return LLVM::getVectorType(LLVM::type::i8Ty(ctx), getVectorSize(type));
    else if (elemTy.isInteger(16))
      return LLVM::getVectorType(LLVM::type::i16Ty(ctx), getVectorSize(type));
    else if (elemTy.isInteger(32))
      return LLVM::getVectorType(LLVM::type::i32Ty(ctx), getVectorSize(type));
    else if (elemTy.isBF16())
      return LLVM::getVectorType(LLVM::type::bf16Ty(ctx), getVectorSize(type));

    llvm_unreachable("Not implemented.");
  }

protected:
  int xpuArch = 3;
};

template <typename SrcOp, typename DstOp>
struct VVBinOpsConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VVBinOpsConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    Value lllhs = adaptor.getLhs();
    Value llrhs = adaptor.getRhs();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = lhs.getType();

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(valueTy));
    unsigned numElems = getTotalElemsPerThread(valueTy);

    auto lhsElems = unpackLLElements(loc, lllhs, rewriter);
    auto rhsElems = unpackLLElements(loc, llrhs, rewriter);
    assert(lhsElems.size() == rhsElems.size());

    SmallVector<Value> calculatedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      Value vaddOp =
          rewriter.create<DstOp>(loc, convertVectorType(valueElemTy),
                                 lhsElems[vecStart], rhsElems[vecStart]);
      calculatedVals.push_back(vaddOp);
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp>
struct SVBinOpsConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  SVBinOpsConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    int32_t elemStateInt = op.getElemState();
    ElemState elemState = static_cast<ElemState>(elemStateInt);

    Value lllhs = adaptor.getLhs();
    Value llrhs = adaptor.getRhs();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Type valueTy;
    Type scalarValTy;

    if (elemState == ElemState::SV) {
      valueTy = rhs.getType();
      scalarValTy = lhs.getType();
    } else if (elemState == ElemState::VS) {
      valueTy = lhs.getType();
      scalarValTy = rhs.getType();
    }

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(valueTy));
    unsigned numElems = getTotalElemsPerThread(valueTy);
    unsigned rowNum = 1;
    if (auto scalarTensorTy = dyn_cast<RankedTensorType>(scalarValTy)) {
      auto encoding =
          cast<triton::xpu::ClusterLayoutAttr>(scalarTensorTy.getEncoding());
      rowNum = product(encoding.getSizePerCore());
    }
    unsigned colNum = ceil(numElems, rowNum);

    // Get data from a struct
    auto lhsElems = unpackLLElements(loc, lllhs, rewriter);
    auto rhsElems = unpackLLElements(loc, llrhs, rewriter);

    // Create LLVM Op
    SmallVector<Value> calculatedVals;
    Type vecTy = getElementTypeOrSelf(valueTy);
    Type elemTy = getElementTypeOrSelf(vecTy);
    StringRef asm_string;
    if (elemTy.isF32() || elemTy.isSignlessInteger(32)) {
      asm_string = SVOp2Str<SrcOp>::value;
    } else if (elemTy.isF16()) {
      asm_string = SVOp2StrFP16<SrcOp>::value;
    } else {
      llvm_unreachable("Only FP16/FP32/I32 are supported in SVBinary!");
    }
    StringRef constraints = "=v,r,v";
    for (int i = 0; i < rowNum; ++i) {
      for (int j = 0; j < colNum; ++j) {
        if (elemState == ElemState::SV) {
          SmallVector<Value, 4> operands(
              {lhsElems[i], rhsElems[i * colNum + j]});
          auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
              loc, valueElemTy, operands, asm_string, constraints,
              /*has_side_effects=*/false,
              /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
              LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
              ArrayAttr());
          calculatedVals.push_back(asmOp.getRes());
        } else if (elemState == ElemState::VS) {
          SmallVector<Value, 4> operands(
              {rhsElems[i], lhsElems[i * colNum + j]});
          auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
              loc, valueElemTy, operands, asm_string, constraints,
              /*has_side_effects=*/false,
              /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
              LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
              ArrayAttr());
          calculatedVals.push_back(asmOp.getRes());
        }
      }
    }

    // Wrap data into a struct
    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp, typename DstOp>
struct UnaryOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  UnaryOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value value = op.getValue();
    Value result = op.getResult();

    Value llvalue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto resultTy = result.getType();
    Type resultElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(resultTy));
    unsigned numElems = getTotalElemsPerThread(value.getType());

    auto valueElems = unpackLLElements(loc, llvalue, rewriter);

    SmallVector<Value> calculatedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      Value vexpOp = rewriter.create<DstOp>(
          loc, convertVectorType(resultElemTy), valueElems[vecStart]);
      calculatedVals.push_back(vexpOp);
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resultTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp>
struct VOpConversionLibCall : public ConvertOpToLLVMPattern<SrcOp>,
                              public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VOpConversionLibCall(LLVMTypeConverter &converter, PatternBenefit benefit,
                       const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    unsigned vecElems = getTotalElemsPerThread(resultTy);
    auto resultVecTy = getElementTypeOrSelf(resultTy);
    Type vecTy = this->getTypeConverter()->convertType(resultVecTy);
    auto elemTy = getElementTypeOrSelf(vecTy);
    SmallVector<Type> types(vecElems, vecTy);
    Type structTy = this->getTypeConverter()->convertType(resultTy);

    auto operands = getOperands(rewriter, adaptor, vecElems, loc);
    SmallVector<Value> resultVals(vecElems);
    for (unsigned i = 0; i < vecElems; ++i) {
      ValueRange singleOperandRange(operands[i]);
      if (elemTy.isF32()) {
        Value devCall = mlir::LLVM::XPU::createDeviceCall(
            VLibOp<SrcOp>::value, rewriter, op, vecTy, singleOperandRange, loc);
        resultVals[i] = devCall;
      } else if (elemTy.isF16()) {
        Value devCall;
        switch (xpuArch) {
        case 2:
          devCall = mlir::LLVM::XPU::createDeviceCall(
              VLibOpFP16<SrcOp, 2>::value, rewriter, op, vecTy,
              singleOperandRange, loc);
          break;
        case 3:
          devCall = mlir::LLVM::XPU::createDeviceCall(
              VLibOpFP16<SrcOp, 3>::value, rewriter, op, vecTy,
              singleOperandRange, loc);
          break;
        default:
          llvm_unreachable("Failed to create device call with unsupported xpu "
                           "architecture.");
        }
        resultVals[i] = devCall;
      } else {
        llvm_unreachable("Only FP16 and FP32 are supported in LibDevice!");
      }
      if (!bool(resultVals[i]))
        return failure();
    }
    Value view =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, structTy);
    rewriter.replaceOp(op, view);

    return success();
  }

private:
  SmallVector<SmallVector<Value>>
  getOperands(ConversionPatternRewriter &rewriter, OpAdaptor adaptor,
              const unsigned elems, Location loc) const {
    SmallVector<SmallVector<Value>> operands(elems);
    for (auto operand : adaptor.getOperands()) {
      auto sub_operands = unpackLLElements(loc, operand, rewriter);
      for (size_t i = 0; i < elems; ++i) {
        operands[i].push_back(sub_operands[i]);
      }
    }
    return operands;
  }
};

template <typename SrcOp, typename DstOp>
struct VConstOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VConstOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value res = op.getResult();
    Attribute attr = adaptor.getValueAttr();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto resTy = res.getType();
    Type resElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(resTy));
    unsigned numElems = getTotalElemsPerThread(res.getType());

    // LLVM22: LLVM::ConstantOp requires the DenseElementsAttr to match the
    // result vector type's element count. The source op carries a tensor-shaped
    // splat attr (e.g. tensor<1x8192xf32>), but the LLVM constant must be
    // shaped like vector<16xf32>. Reshape the splat attr to the vector type.
    auto vecTy = convertVectorType(resElemTy);
    Attribute vecAttr = attr;
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
      if (denseAttr.isSplat()) {
        vecAttr = DenseElementsAttr::get(cast<ShapedType>(vecTy),
                                         denseAttr.getSplatValue<Attribute>());
      } else {
        vecAttr = DenseElementsAttr::getFromRawBuffer(cast<ShapedType>(vecTy),
                                                      denseAttr.getRawData());
      }
    }

    SmallVector<Value> calculatedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      Value vconstOp = rewriter.create<DstOp>(loc, vecTy, vecAttr);
      calculatedVals.push_back(vconstOp);
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

template <typename SrcOp>
struct VSplatOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                            public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VSplatOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                     const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  Value convertSplatLikeOp(Type resTy, Value llsrc, Type llvmResultStructTy,
                           ConversionPatternRewriter &rewriter,
                           Location loc) const {
    auto resElemTy =
        this->getTypeConverter()->convertType(getElementTypeOrSelf(resTy));
    size_t elemsPerThread = getTotalElemsPerThread(resTy);

    auto valueElems = unpackLLElements(loc, llsrc, rewriter);

    Value vector_1xTy = rewriter.create<LLVM::UndefOp>(loc, resElemTy);
    vector_1xTy =
        insert_element(resElemTy, vector_1xTy, valueElems[0], i32_val(0));

    int32_t vecSize = cast<mlir::VectorType>(resElemTy).getNumElements();
    SmallVector<int32_t, 16> zeroValues(vecSize, 0);
    // TODO[dyq]: check getI32ArrayAttr -> getDenseI32ArrayAttr
    auto zeroAttrs = rewriter.getDenseI32ArrayAttr(zeroValues);
    Value shuffleVectorOp = rewriter.create<LLVM::ShuffleVectorOp>(
        loc, resElemTy, vector_1xTy, vector_1xTy, zeroAttrs);

    llvm::SmallVector<Value> elems(elemsPerThread, shuffleVectorOp);

    return packLLElements(loc, this->getTypeConverter(), elems, rewriter,
                          llvmResultStructTy);
  }

  LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto llsrc = adaptor.getSrc();
    auto llvmResultStructTy =
        this->getTypeConverter()->convertType(op.getType());
    auto llStruct = convertSplatLikeOp(op.getType(), llsrc, llvmResultStructTy,
                                       rewriter, loc);

    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};

template <typename SrcOp>
struct VSelectOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                             public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VSelectOpConversion(LLVMTypeConverter &converter,

                      PatternBenefit benefit,
                      const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // original values
    Value condition = op.getCondition();
    Value true_value = op.getTrueValue();
    Value false_value = op.getFalseValue();

    // adaptor values
    Value llCondition = adaptor.getCondition();
    Value llTrue_value = adaptor.getTrueValue();
    Value llFalse_value = adaptor.getFalseValue();

    MLIRContext *ctx = rewriter.getContext();
    auto loc = op->getLoc();
    auto resTy = op.getType();

    Type resElemTy = getTypeConverter()->convertType(
        getElementTypeOrSelf(resTy)); // vector<16xf32>
    unsigned numElems = getTotalElemsPerThread(resTy);

    // Get data from a struct
    auto conditionElems = unpackLLElements(loc, llCondition, rewriter);
    auto trueValElems = unpackLLElements(loc, llTrue_value, rewriter);
    auto falseValElems = unpackLLElements(loc, llFalse_value, rewriter);

    // Create LLVM Op
    Type elemTy = getElementTypeOrSelf(resElemTy);
    unsigned elemBits = elemTy.getIntOrFloatBitWidth();
    unsigned vecSize = mlir::cast<VectorType>(resElemTy).getNumElements();
    SmallVector<Value> resVals;

    for (size_t elemIter = 0; elemIter < numElems; ++elemIter) {
      // Step 1. Convert Condition To v32i1/v16i1 Mask
      VectorType maskTy = VectorType::get(32, i1_ty);
      Value maskV;

      if (isa<VectorType>(conditionElems[elemIter].getType())) {
        auto condTy =
            mlir::cast<VectorType>(conditionElems[elemIter].getType());
        unsigned condSize = condTy.getNumElements();
        if (condSize == 32) {
          // Already vector<32xi1>, can use directly
          maskV = conditionElems[elemIter];
        } else {
          // Optimized: bitcast vector<Nxi1> -> iN -> i32 -> vector<32xi1>
          // Replaces 4*N scalar ops (extractelement+zext+shl+or) with 3 ops
          IntegerType intTy = IntegerType::get(ctx, condSize);
          Value intVal = bitcast(conditionElems[elemIter], intTy);
          Value i32Val = zext(i32_ty, intVal);
          maskV = bitcast(i32Val, maskTy);
        }
      } else {
        // Fallback: scalar i1 packing for non-vector conditions
        Value orV = i32_val(0);
        for (size_t conditionIter = 0; conditionIter < vecSize;
             ++conditionIter) {
          Value boolVal = conditionElems[elemIter * vecSize + conditionIter];
          Value extV = zext(i32_ty, boolVal);
          Value shlV = shl(extV, i32_val(conditionIter));
          orV = or_(orV, shlV);
        }
        maskV = bitcast(orV, maskTy);
      }

      if (elemTy.isF32()) {
        // Step 2. vset_zero()
        StringRef xor_asm_string = "vxor.s.mz $0{mr1}, $0, $0";
        StringRef xor_constraints = "=v";
        SmallVector<Value, 4> xor_operands({});
        auto zerosIAsmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, resElemTy, xor_operands, xor_asm_string, xor_constraints,
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        Value zerosFAsmOp = bitcast(zerosIAsmOp.getRes(), resElemTy);
        // Step 3. vvor_float32x16_mh(mask, zero, a, b)
        Value vvorFOp = rewriter.create<mlir::LLVM::XPU::VVOR_F_MHOp>(
            loc, resElemTy, maskV, zerosFAsmOp, trueValElems[elemIter],
            falseValElems[elemIter]);
        resVals.push_back(vvorFOp);
      } else if (elemTy.isInteger(32)) {
        // Step 2. vset_zero()
        StringRef xor_asm_string = "vxor.s.mz $0{mr1}, $0, $0";
        StringRef xor_constraints = "=v";
        SmallVector<Value, 4> xor_operands({});
        auto zerosIAsmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, resElemTy, xor_operands, xor_asm_string, xor_constraints,
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        Value zerosFAsmOp = bitcast(zerosIAsmOp.getRes(), resElemTy);
        // Step 3. vvor_int32x16_mh(mask, zero, a, b)
        Value vvorFOp = rewriter.create<mlir::LLVM::XPU::VVOR_S_MHOp>(
            loc, resElemTy, maskV, zerosFAsmOp, trueValElems[elemIter],
            falseValElems[elemIter]);
        resVals.push_back(vvorFOp);
      } else if (elemTy.isF16()) {
        // Step 2. vset_zero()
        StringRef xor_asm_string = "vxor.hf.mz $0{mr1}, $0, $0";
        StringRef xor_constraints = "=v";
        SmallVector<Value, 4> xor_operands({});
        auto zerosFAsmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, resElemTy, xor_operands, xor_asm_string, xor_constraints,
            /*has_side_effects=*/false,
            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
            LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
            ArrayAttr());
        // Step 3. vvor_float16x32_mh(mask, zero, a, b)
        Value vvorFOp = rewriter.create<mlir::LLVM::XPU::VVOR_HF_MHOp>(
            loc, resElemTy, maskV, zerosFAsmOp.getRes(), trueValElems[elemIter],
            falseValElems[elemIter]);
        resVals.push_back(vvorFOp);
      } else {
        llvm_unreachable("Only FP16 and FP32 are supported in VSelect!");
      }
    }

    // Wrap data into a struct
    auto llvmResultStructTy = getTypeConverter()->convertType(resTy);
    auto llStruct = packLLElements(loc, getTypeConverter(), resVals, rewriter,
                                   llvmResultStructTy);
    rewriter.replaceOp(op, {llStruct});

    return success();
  }
};

// Integer twin of VCmpFOpConversion, for the vcmpi UnrollControl rewrites an
// inlined select(cmpi) combine into. The result elements are i1 (packed into
// the caller's mask by VSelectOpConversion), so one vector ICmpOp per slice
// is the whole lowering.
struct VCmpIOpConversion : public ConvertOpToLLVMPattern<triton::xpu::VCmpIOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VCmpIOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VCmpIOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VCmpIOp::Adaptor;

  VCmpIOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VCmpIOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult matchAndRewrite(triton::xpu::VCmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value res = op.getResult();

    Value lllhs = adaptor.getLhs();
    Value llrhs = adaptor.getRhs();

    auto loc = op->getLoc();

    auto lhsTy = lhs.getType();
    auto rhsTy = rhs.getType();
    assert(lhsTy == rhsTy);
    auto lhsElemTy = getElementTypeOrSelf(getElementTypeOrSelf(lhsTy));
    (void)lhsElemTy;
    auto resTy = res.getType();
    auto resVecTy = getElementTypeOrSelf(resTy);
    auto resElemTy = getElementTypeOrSelf(resVecTy);
    assert(resElemTy.isInteger(1) &&
           "vcmpi result must be i1 lanes for vselect's mask packing");
    auto llResVecTy = getTypeConverter()->convertType(resVecTy);

    unsigned numVecs = getTotalElemsPerThread(lhsTy);
    auto lhsVecs = unpackLLElements(loc, lllhs, rewriter);
    auto rhsVecs = unpackLLElements(loc, llrhs, rewriter);
    assert(lhsVecs.size() == rhsVecs.size());

    SmallVector<Value> calculatedVals;
    for (size_t i = 0; i < numVecs; ++i) {
      calculatedVals.push_back(rewriter.create<LLVM::ICmpOp>(
          loc, llResVecTy, ArithCmpIPredicateToLLVM(op.getPredicate()),
          lhsVecs[i], rhsVecs[i]));
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item1__

      __PRED_ENUM(eq, eq);
      __PRED_ENUM(ne, ne);
      __PRED_ENUM(sgt, sgt);
      __PRED_ENUM(sge, sge);
      __PRED_ENUM(slt, slt);
      __PRED_ENUM(sle, sle);
      __PRED_ENUM(ugt, ugt);
      __PRED_ENUM(uge, uge);
      __PRED_ENUM(ult, ult);
      __PRED_ENUM(ule, ule);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

template <typename SrcOp>
struct VMacFOpConversion : public ConvertOpToLLVMPattern<SrcOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<SrcOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<SrcOp>::getTypeConverter;
  using OpAdaptor = typename SrcOp::Adaptor;

  VMacFOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<SrcOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

    // original values
    Value value = op.getValue();
    Value mulData = op.getMulData();
    Value addData = op.getAddData();

    // adaptor values
    Value llValue = adaptor.getValue();
    Value llMulData = adaptor.getMulData();
    Value llAddData = adaptor.getAddData();
    auto attrs = adaptor.getAttributes();

    MLIRContext *ctx = rewriter.getContext();
    auto loc = op->getLoc();
    auto resTy = op.getType();

    auto resElemTy = getTypeConverter()->convertType(
        getElementTypeOrSelf(resTy)); // vector<16xf32>
    unsigned numElems = getTotalElemsPerThread(resTy);

    // Get data from a struct
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    auto mulElems = unpackLLElements(loc, llMulData, rewriter);
    auto addElems = unpackLLElements(loc, llAddData, rewriter);

    // Create LLVM Op
    SmallVector<Value> calculatedVals;
    auto elemTy = getElementTypeOrSelf(resElemTy);
    StringRef asm_string;
    if (elemTy.isF32()) {
      asm_string = "vmac.f.mz.rn $0{mr1}, $1, $2";
    } else if (elemTy.isF16()) {
      asm_string = "vmac.hf.mz.rn $0{mr1}, $1, $2";
    } else {
      llvm_unreachable("Only FP16 and FP32 are supported in VMac!");
    }
    StringRef constraints = "=v,v,v,0";
    for (size_t vecStart = 0; vecStart < numElems; vecStart += 1) {
      SmallVector<Value, 4> operands(
          {valueElems[vecStart], mulElems[vecStart], addElems[vecStart]});
      auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, operands, asm_string, constraints,
          /*has_side_effects=*/false,
          /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr());
      calculatedVals.push_back(asmOp.getRes());
    }

    // Wrap data into a struct
    auto llvmResultStructTy = getTypeConverter()->convertType(resTy);
    auto llStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                   rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {llStruct});

    return success();
  }
};

struct VExtFOpConversion : public ConvertOpToLLVMPattern<triton::xpu::VExtFOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VExtFOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VExtFOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VExtFOp::Adaptor;

  VExtFOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VExtFOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  Value convertFp16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                          Value val, Value res, Type valElemTy, Type resElemTy,
                          Value llVal, Type llvmResultStructTy) const {
    auto ctx = rewriter.getContext();
    auto llVals = unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = getTotalElemsPerThread(val.getType());
    // One 512-bit VREG holds 32 f16 lanes but only 16 f32 lanes, so a f16
    // register usually expands into two f32 registers (low + high half). When
    // the vectorization factor is 16 the source register only carries valid
    // data in its low half and the result is a single register, so emit exactly
    // as many halves as the result type asks for.
    unsigned numResElems = getTotalElemsPerThread(res.getType());

    SmallVector<Value, 8> fp32x16Vecs;
    for (int i = 0; i < numElems && fp32x16Vecs.size() < numResElems; ++i) {
      auto asml = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, ValueRange{llVals[i]}, // operands
          "vfp162float_l.rn $0, $1",             // asm_string
          "=&v,v",                               // constraints
          false,                                 // has_size_effects
          false,                                 // is_align_stack
          LLVM::tailcallkind::TailCallKind::None,
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr::get(ctx, {}));
      fp32x16Vecs.emplace_back(asml.getRes());
      if (fp32x16Vecs.size() == numResElems)
        break;
      auto asmh = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, ValueRange{llVals[i]}, // operands
          "vfp162float_h.rn $0, $1",             // asm_string
          "=&v,v",                               // constraints
          false,                                 // has_size_effects
          false,                                 // is_align_stack
          LLVM::tailcallkind::TailCallKind::None,
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr::get(ctx, {}));
      fp32x16Vecs.emplace_back(asmh.getRes());
    }

    Value resultStruct = packLLElements(loc, getTypeConverter(), fp32x16Vecs,
                                        rewriter, llvmResultStructTy);
    return resultStruct;
  }

  Value convertBf16ToFp32(Location loc, ConversionPatternRewriter &rewriter,
                          Value val, Value res, Type valElemTy, Type resElemTy,
                          Value llVal, Type llvmResultStructTy) const {
    auto ctx = rewriter.getContext();
    auto llVals = unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = getTotalElemsPerThread(val.getType());
    // Same low/high-half expansion caveat as convertFp16ToFp32: a bf16 register
    // normally becomes two f32 registers, but at vectorization factor 16 only
    // the low half carries valid data and the result is a single register.
    // Emitting an unconditional 2x here overruns llvmResultStructTy.
    unsigned numResElems = getTotalElemsPerThread(res.getType());

    VectorType vecFp16Ty = VectorType::get(32, f16_ty);
    Value padVec = rewriter.create<LLVM::UndefOp>(loc, vecFp16Ty);
    for (size_t elemIdx = 0; elemIdx < 32; ++elemIdx) {
      padVec = insert_element(vecFp16Ty, padVec, f16_val(0), i16_val(elemIdx));
    }

    SmallVector<Value, 8> fp32x16Vecs;
    for (int i = 0; i < numElems && fp32x16Vecs.size() < numResElems; ++i) {
      Value val = bitcast(llVals[i], vecFp16Ty);
      Value vl = rewriter.create<mlir::LLVM::XPU::VMERGE_L_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vl = bitcast(vl, resElemTy);
      fp32x16Vecs.emplace_back(vl);
      if (fp32x16Vecs.size() == numResElems)
        break;
      Value vh = rewriter.create<mlir::LLVM::XPU::VMERGE_H_HFOp>(loc, vecFp16Ty,
                                                                 padVec, val);
      vh = bitcast(vh, resElemTy);
      fp32x16Vecs.emplace_back(vh);
    }

    Value resultStruct = packLLElements(loc, getTypeConverter(), fp32x16Vecs,
                                        rewriter, llvmResultStructTy);
    return resultStruct;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::VExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto val = op.getValue();
    auto res = op.getResult();
    auto llVal = adaptor.getValue();

    Type valTy = val.getType();
    Type resTy = res.getType();
    auto valElemTy = getElementTypeOrSelf(valTy);
    auto _valElemTy = getElementTypeOrSelf(valElemTy);
    auto resElemTy = getElementTypeOrSelf(resTy);
    auto _resElemTy = getElementTypeOrSelf(resElemTy);
    auto llValElemTy = typeConverter->convertType(valElemTy);
    auto llResElemTy = typeConverter->convertType(resElemTy);
    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    assert(_resElemTy.isF32() && "Only support F32 as target dtype inVExtF!");
    if (_valElemTy.isF16()) {
      auto result = convertFp16ToFp32(loc, rewriter, val, res, valElemTy,
                                      resElemTy, llVal, llvmResultStructTy);
      rewriter.replaceOp(op, {result});
    } else if (_valElemTy.isBF16()) {
      auto result = convertBf16ToFp32(loc, rewriter, val, res, valElemTy,
                                      resElemTy, llVal, llvmResultStructTy);
      rewriter.replaceOp(op, {result});
    } else {
      assert(0 && "Only support FP16 as source dtype in VExtF!");
    }
    return success();
  }
};

struct VTruncFOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>,
      public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VTruncFOp::Adaptor;

  VTruncFOpConversion(LLVMTypeConverter &converter,

                      PatternBenefit benefit,
                      const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VTruncFOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  Value convertFp32ToFp16(Location loc, ConversionPatternRewriter &rewriter,
                          Value val, Value res, Type valElemTy, Type resElemTy,
                          Value llVal, Type llvmResultStructTy) const {
    auto ctx = rewriter.getContext();
    auto llVals = unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = getTotalElemsPerThread(val.getType());

    SmallVector<Value, 8> fp16x32Vecs;
    for (int i = 0; i < numElems; i += 2) {
      auto asmlh = rewriter.create<LLVM::InlineAsmOp>(
          loc, resElemTy, ValueRange{llVals[i], llVals[i + 1]}, // operands
          "vfloat2fp16_l.rn $0, $1\nvfloat2fp16_h.rn $0, $2",   // asm_string
          "=&v,v,v",                                            // constraints
          false, // has_size_effects
          false, // is_align_stack
          LLVM::tailcallkind::TailCallKind::None,
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
          ArrayAttr::get(ctx, {}));
      fp16x32Vecs.push_back(asmlh.getRes());
    }

    Value resultStruct = packLLElements(loc, getTypeConverter(), fp16x32Vecs,
                                        rewriter, llvmResultStructTy);
    return resultStruct;
  }

  LogicalResult
  matchAndRewrite(triton::xpu::VTruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto val = op.getValue();
    auto res = op.getResult();
    auto llVal = adaptor.getValue();

    Type valTy = val.getType();
    Type resTy = res.getType();
    auto valElemTy = getElementTypeOrSelf(valTy);
    auto _valElemTy = getElementTypeOrSelf(valElemTy);
    auto resElemTy = getElementTypeOrSelf(resTy);
    auto _resElemTy = getElementTypeOrSelf(resElemTy);
    auto llValElemTy = typeConverter->convertType(valElemTy);
    auto llResElemTy = typeConverter->convertType(resElemTy);
    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    assert(_valElemTy.isF32() &&
           "Only support F32 as source dtype in VTruncF!");
    if (_resElemTy.isF16()) {
      auto result = convertFp32ToFp16(loc, rewriter, val, res, valElemTy,
                                      resElemTy, llVal, llvmResultStructTy);
      rewriter.replaceOp(op, {result});
    } else {
      assert(0 && "Only support FP16 as target dtype in VTruncF!");
    }
    return success();
  }
};

// vsitofp widens an integer element into a wider float element. When the
// source is narrower than f32 the byte count expands (i8 -> f32 is 1:4), so a
// single source VREG (512 bits) must be spread over several result VREGs: an
// i8 register holds 64 lanes while an f32 register holds 16, so the four
// 16-lane segments of the source become four f32 registers. The generic
// UnaryOpConversion assumes a 1:1 element-count mapping and packs only
// getTotalElemsPerThread(value) results into a struct sized by the result
// type -- that mismatch is the crash this pattern replaces. llc has no i8
// vector pattern for a bare LLVM::SIToFPOp (it scalarizes), so each segment
// goes through the dedicated vfix82float_{ll,lh,hl,hh} instructions; the lane
// order mirrors the device-side primitive_cast in
// xpu/kernel/cluster_primitive.h (ll = lanes 0-15 ... hh = lanes 48-63).
struct VSIToFPOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::VSIToFPOp>,
      public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VSIToFPOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VSIToFPOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VSIToFPOp::Adaptor;

  VSIToFPOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                      const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VSIToFPOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::xpu::VSIToFPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto val = op.getValue();
    auto res = op.getResult();
    auto llVal = adaptor.getValue();

    Type valTy = val.getType();
    Type resTy = res.getType();
    auto valElemTy = getElementTypeOrSelf(valTy);
    auto _valElemTy = getElementTypeOrSelf(valElemTy);
    auto resElemTy = getElementTypeOrSelf(resTy);
    auto ctx = rewriter.getContext();

    auto llVals = unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = getTotalElemsPerThread(valTy);
    unsigned numResElems = getTotalElemsPerThread(resTy);

    SmallVector<Value, 8> fp32Vecs;
    if (_valElemTy.isInteger(8)) {
      // i8 -> f32 expands 1:4: each source VREG carries four 16-lane segments
      // that map one-to-one onto four f32 VREGs.
      static const char *kSegAsm[4] = {
          "vfix82float_ll.rn $0, $1", "vfix82float_lh.rn $0, $1",
          "vfix82float_hl.rn $0, $1", "vfix82float_hh.rn $0, $1"};
      for (unsigned i = 0; i < numElems && fp32Vecs.size() < numResElems; ++i) {
        for (unsigned j = 0; j < 4 && fp32Vecs.size() < numResElems; ++j) {
          auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
              loc, resElemTy, ValueRange{llVals[i]}, kSegAsm[j], "=&v,v",
              /*has_side_effects=*/false, /*is_align_stack=*/false,
              LLVM::tailcallkind::TailCallKind::None,
              LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT),
              ArrayAttr::get(ctx, {}));
          fp32Vecs.push_back(asmOp.getRes());
        }
      }
    } else if (_valElemTy.isInteger(32)) {
      // i32 -> f32 keeps the lane count (both sides are 4 bytes wide) and llc
      // selects vfix2float.rn for the plain op.
      for (unsigned i = 0; i < numElems && fp32Vecs.size() < numResElems; ++i) {
        auto cast = rewriter.create<LLVM::SIToFPOp>(
            loc, convertVectorType(resElemTy), llVals[i]);
        fp32Vecs.push_back(cast);
      }
    } else {
      assert(0 && "Only support i8/i32 as source dtype in VSIToFPOp!");
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), fp32Vecs,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

struct VCmpFOpConversion : public ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>,
                           public XPUVectorizedOpsConversionBase {

  using ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>::getTypeConverter;
  using OpAdaptor = typename triton::xpu::VCmpFOp::Adaptor;

  VCmpFOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit,
                    const triton::xpu::TargetInfo &targetInfo)
      : ConvertOpToLLVMPattern<triton::xpu::VCmpFOp>(converter, benefit),
        XPUVectorizedOpsConversionBase(targetInfo) {}

  LogicalResult matchAndRewrite(triton::xpu::VCmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value res = op.getResult();

    Value lllhs = adaptor.getLhs();
    Value llrhs = adaptor.getRhs();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto lhsTy = lhs.getType();
    auto rhsTy = rhs.getType();
    assert(lhsTy == rhsTy);
    auto lhsVecTy = getElementTypeOrSelf(lhsTy);
    auto rhsVecTy = getElementTypeOrSelf(rhsTy);
    assert(lhsVecTy == rhsVecTy);
    auto lhsElemTy = getElementTypeOrSelf(lhsVecTy);
    auto rhsElemTy = getElementTypeOrSelf(rhsVecTy);
    assert(lhsElemTy == rhsElemTy);
    auto resTy = res.getType();
    auto resVecTy = getElementTypeOrSelf(resTy);
    auto resElemTy = getElementTypeOrSelf(resVecTy);
    auto llResVecTy = getTypeConverter()->convertType(resVecTy);

    unsigned numVecs = getTotalElemsPerThread(lhsTy);
    auto lhsVecs = unpackLLElements(loc, lllhs, rewriter);
    auto rhsVecs = unpackLLElements(loc, llrhs, rewriter);
    assert(lhsVecs.size() == rhsVecs.size());

    SmallVector<Value> calculatedVals;
    if (resElemTy.isInteger(32)) {
      if (lhsElemTy.isF16()) {
        for (size_t i = 0; i < numVecs; ++i) {
          ValueRange args({lhsVecs[i], rhsVecs[i]});
          Type resTy = rewriter.getI32Type();
          StringRef asmString = ArithCmpFPredicateToASMFP16(op.getPredicate());
          auto res = LLVM::XPU::createDeviceCall(asmString, rewriter, op, resTy,
                                                 args, loc);
          calculatedVals.push_back(res);
        }
      } else if (lhsElemTy.isF32()) {
        for (size_t i = 0; i < numVecs; ++i) {
          ValueRange args({lhsVecs[i], rhsVecs[i]});
          Type resTy = rewriter.getI32Type();
          StringRef asmString = ArithCmpFPredicateToASMFP32(op.getPredicate());
          auto res = LLVM::XPU::createDeviceCall(asmString, rewriter, op, resTy,
                                                 args, loc);
          calculatedVals.push_back(res);
        }
      } else {
        llvm_unreachable("[VectorizedOpToLLVM]: CmpF Only Support F16 or F32 "
                         "as Input Type");
      }
    } else {
      // Self-compare NaN checks (lhs == rhs) must NOT become a raw vector
      // llvm.fcmp: LLVM canonicalizes `fcmp une x, x` into `fcmp uno x, 0`
      // and `fcmp oeq x, x` into `fcmp ord x, 0`, and the XPU backend has no
      // ISel pattern for vector SETUO/SETORD -> "Cannot select" abort
      // (flaggems xlogy.py:32 `y != y`, measured 2026-09-11). The six common
      // predicates (oeq/une/ogt/oge/olt/ole, non-self) select fine (they
      // lower to vneq.f.mz & friends), so only the self-compare family needs
      // a rewrite. Express the NaN test with ops from proven-selectable
      // families -- vector bitcast, llvm.and, and the integer icmp path
      // validated end-to-end by vcmpi:
      //   isNaN(x)  <=> (bitcast<i32>(x) & 0x7fffffff) >s 0x7f800000
      //   !isNaN(x) <=> (bitcast<i32>(x) & 0x7fffffff) <=s 0x7f800000
      // (the mask clears the sign bit, so signed and unsigned agree on the
      // non-negative remainder). Python's `x != x` / `x == x` are the only
      // Triton-reachable producers of this family. F16/BF16 self-compares
      // keep the raw fcmp below and remain unsupported (no known kernel:
      // frontends convert to f32 first, as flaggems does).
      arith::CmpFPredicate pred = op.getPredicate();
      bool selfCmp = (op.getLhs() == op.getRhs());
      bool nanCheck = selfCmp && (pred == arith::CmpFPredicate::UNE ||
                                  pred == arith::CmpFPredicate::UNO ||
                                  pred == arith::CmpFPredicate::UEQ);
      bool notNanCheck = selfCmp && (pred == arith::CmpFPredicate::OEQ ||
                                     pred == arith::CmpFPredicate::ORD);
      if ((nanCheck || notNanCheck) && lhsElemTy.isF32() && numVecs > 0) {
        auto srcVecTy = mlir::cast<mlir::VectorType>(lhsVecs[0].getType());
        auto i32VecTy =
            mlir::VectorType::get(srcVecTy.getShape(), rewriter.getI32Type());
        Value absMask = rewriter.create<LLVM::ConstantOp>(
            loc, i32VecTy,
            DenseElementsAttr::get(cast<ShapedType>(i32VecTy),
                                   rewriter.getI32IntegerAttr(0x7fffffff)));
        Value nanBound = rewriter.create<LLVM::ConstantOp>(
            loc, i32VecTy,
            DenseElementsAttr::get(cast<ShapedType>(i32VecTy),
                                   rewriter.getI32IntegerAttr(0x7f800000)));
        for (size_t i = 0; i < numVecs; ++i) {
          Value bits =
              rewriter.create<LLVM::BitcastOp>(loc, i32VecTy, lhsVecs[i]);
          Value absBits =
              rewriter.create<LLVM::AndOp>(loc, i32VecTy, bits, absMask);
          Value nanMask = rewriter.create<LLVM::ICmpOp>(
              loc, llResVecTy,
              nanCheck ? LLVM::ICmpPredicate::sgt : LLVM::ICmpPredicate::sle,
              absBits, nanBound);
          calculatedVals.push_back(nanMask);
        }
      } else {
        for (size_t vecStart = 0; vecStart < numVecs; vecStart += 1) {
          Value vcmpfOp = rewriter.create<LLVM::FCmpOp>(
              loc, llResVecTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
              lhsVecs[vecStart], rhsVecs[vecStart]);
          calculatedVals.push_back(vcmpfOp);
        }
      }
    }

    Type llvmResultStructTy = getTypeConverter()->convertType(resTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), calculatedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }

  static StringRef ArithCmpFPredicateToASMFP32(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __VASM_FP32_PRED_ENUM(item__, item1__)                                 \
  case arith::CmpFPredicate::item__:                                           \
    return item1__

      __VASM_FP32_PRED_ENUM(OEQ, "_ZN3xpu8vveqfp32EDv16_fS0_");
      __VASM_FP32_PRED_ENUM(UNE, "_ZN3xpu8vvnefp32EDv16_fS0_");
      __VASM_FP32_PRED_ENUM(OGT, "_ZN3xpu8vvgtfp32EDv16_fS0_");
      __VASM_FP32_PRED_ENUM(OGE, "_ZN3xpu8vvgefp32EDv16_fS0_");
      __VASM_FP32_PRED_ENUM(OLT, "_ZN3xpu8vvltfp32EDv16_fS0_");
      __VASM_FP32_PRED_ENUM(OLE, "_ZN3xpu8vvlefp32EDv16_fS0_");

#undef __VASM_FP32_PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }

  static StringRef ArithCmpFPredicateToASMFP16(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __VASM_FP16_PRED_ENUM(item__, item1__)                                 \
  case arith::CmpFPredicate::item__:                                           \
    return item1__

      __VASM_FP16_PRED_ENUM(OEQ, "_ZN3xpu8vveqfp16EDv32_DF16_S0_");
      __VASM_FP16_PRED_ENUM(UNE, "_ZN3xpu8vvnefp16EDv32_DF16_S0_");
      __VASM_FP16_PRED_ENUM(OGT, "_ZN3xpu8vvgtfp16EDv32_DF16_S0_");
      __VASM_FP16_PRED_ENUM(OGE, "_ZN3xpu8vvgefp16EDv32_DF16_S0_");
      __VASM_FP16_PRED_ENUM(OLT, "_ZN3xpu8vvltfp16EDv32_DF16_S0_");
      __VASM_FP16_PRED_ENUM(OLE, "_ZN3xpu8vvlefp16EDv32_DF16_S0_");

#undef __VASM_FP16_PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

} // namespace

void mlir::triton::xpu::populateTTXPUVectorizedOpToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, const triton::xpu::TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<VVBinOpsConversion<triton::xpu::VvaddFOp, LLVM::FAddOp>,
               VVBinOpsConversion<triton::xpu::VvsubFOp, LLVM::FSubOp>,
               VVBinOpsConversion<triton::xpu::VvmulFOp, LLVM::FMulOp>,
               VVBinOpsConversion<triton::xpu::VvdivFOp, LLVM::FDivOp>,
               VVBinOpsConversion<triton::xpu::VvmaxFOp, LLVM::MaximumOp>,
               VVBinOpsConversion<triton::xpu::VvminFOp, LLVM::MinimumOp>,
               VVBinOpsConversion<triton::xpu::VvmaxNumFOp, LLVM::MaxNumOp>,
               VVBinOpsConversion<triton::xpu::VvminNumFOp, LLVM::MinNumOp>,
               VVBinOpsConversion<triton::xpu::VvorIOp, LLVM::OrOp>,
               VVBinOpsConversion<triton::xpu::VvxorIOp, LLVM::XOrOp>,
               VVBinOpsConversion<triton::xpu::VvandIOp, LLVM::AndOp>,
               VVBinOpsConversion<triton::xpu::VvaddIOp, LLVM::AddOp>,
               VVBinOpsConversion<triton::xpu::VvsubIOp, LLVM::SubOp>,
               VVBinOpsConversion<triton::xpu::VvmulIOp, LLVM::MulOp>,
               VVBinOpsConversion<triton::xpu::VvmaxSIOp, LLVM::SMaxOp>,
               VVBinOpsConversion<triton::xpu::VvminSIOp, LLVM::SMinOp>,
               VVBinOpsConversion<triton::xpu::VvmaxUIOp, LLVM::UMaxOp>,
               VVBinOpsConversion<triton::xpu::VvminUIOp, LLVM::UMinOp>,
               VVBinOpsConversion<triton::xpu::VvdivSIOp, LLVM::SDivOp>,
               VVBinOpsConversion<triton::xpu::VvdivUIOp, LLVM::UDivOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<SVBinOpsConversion<triton::xpu::SvaddFOp>,
               SVBinOpsConversion<triton::xpu::SvmulFOp>,
               SVBinOpsConversion<triton::xpu::SvsubFOp>,
               SVBinOpsConversion<triton::xpu::SvmaxFOp>,
               SVBinOpsConversion<triton::xpu::SvxorIOp>>(typeConverter,
                                                          benefit, targetInfo);
  patterns.add<UnaryOpConversion<triton::xpu::VExpFOp, LLVM::Exp2Op>,
               UnaryOpConversion<triton::xpu::VSqrtFOp, LLVM::SqrtOp>,
               UnaryOpConversion<triton::xpu::VAbsFOp, LLVM::FAbsOp>,
               UnaryOpConversion<triton::xpu::VLogFOp, LLVM::Log2Op>>(
      typeConverter, benefit, targetInfo);
  // vsitofp needs its own pattern: it is the only unary op above whose
  // element count can change across the op (narrow -> wide), so the generic
  // 1:1 UnaryOpConversion cannot pack its result.
  patterns.add<VSIToFPOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<VOpConversionLibCall<triton::xpu::VSinFOp>,
               VOpConversionLibCall<triton::xpu::VCosFOp>,
               VOpConversionLibCall<triton::xpu::VSigmoidFOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<VConstOpConversion<triton::xpu::VConstOp, LLVM::ConstantOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<VSplatOpConversion<triton::xpu::VSplatOp>>(typeConverter,
                                                          benefit, targetInfo);
  patterns.add<VSelectOpConversion<triton::xpu::VSelectOp>>(
      typeConverter, benefit, targetInfo);
  patterns.add<VMacFOpConversion<triton::xpu::VMacFOp>>(typeConverter, benefit,
                                                        targetInfo);
  patterns.add<VExtFOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<VTruncFOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<VCmpFOpConversion>(typeConverter, benefit, targetInfo);
  patterns.add<VCmpIOpConversion>(typeConverter, benefit, targetInfo);
}
