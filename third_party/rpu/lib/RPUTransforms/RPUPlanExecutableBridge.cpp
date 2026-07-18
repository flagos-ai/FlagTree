#include "RPUPlanExecutableBridge.h"

#include "RPU/IR/Dialect.h"
#include "RPU/IR/ExecutableKind.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace rpu {
namespace {

static std::optional<int64_t> getIntegerAttrValue(DictionaryAttr dict,
                                                  llvm::StringRef name) {
  if (!dict)
    return std::nullopt;
  auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

static std::optional<bool> getBoolAttrValue(DictionaryAttr dict,
                                            llvm::StringRef name) {
  if (!dict)
    return std::nullopt;
  auto attr = dyn_cast_or_null<BoolAttr>(dict.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static std::optional<llvm::StringRef> getStringAttrValue(DictionaryAttr dict,
                                                         llvm::StringRef name) {
  if (!dict)
    return std::nullopt;
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

struct PlanKernelAttrView {
  explicit PlanKernelAttrView(plan::KernelOp op)
      : kernelName_(op.getKernelName()), pattern_(op.getPattern()),
        emission_(op.getEmission()), layout_(op.getLayout()),
        mask_(op.getMask()) {}

  llvm::StringRef kernelName() const { return kernelName_; }

  bool isPattern(llvm::StringRef expected) const {
    return pattern_ == expected;
  }

  std::optional<int64_t> emissionInteger(llvm::StringRef name) const {
    return getIntegerAttrValue(emission_, name);
  }

  std::optional<bool> emissionBool(llvm::StringRef name) const {
    return getBoolAttrValue(emission_, name);
  }

  std::optional<llvm::StringRef> emissionString(llvm::StringRef name) const {
    return getStringAttrValue(emission_, name);
  }

  std::optional<llvm::StringRef> layoutString(llvm::StringRef name) const {
    return getStringAttrValue(layout_, name);
  }

  std::optional<bool> maskBool(llvm::StringRef name) const {
    return getBoolAttrValue(mask_, name);
  }

  std::optional<int64_t> maskInteger(llvm::StringRef name) const {
    return getIntegerAttrValue(mask_, name);
  }

private:
  llvm::StringRef kernelName_;
  llvm::StringRef pattern_;
  DictionaryAttr emission_;
  DictionaryAttr layout_;
  DictionaryAttr mask_;
};

static bool isPositiveMultipleOf16(int64_t value) {
  return value > 0 && value % 16 == 0;
}

static void setMatrixOffsets(OpBuilder &builder, Operation *op, int64_t row,
                             int64_t col = 0) {
  op->setAttr("row_offset", builder.getI32IntegerAttr(row));
  op->setAttr("col_offset", builder.getI32IntegerAttr(col));
}

static void loadExecutableDialects(MLIRContext &context) {
  context.getOrLoadDialect<triton::TritonDialect>();
  context.getOrLoadDialect<exec::RPUDialect>();
}

static exec::KernelOp createExecutableKernel(PatternRewriter &rewriter,
                                             ModuleOp executableModule,
                                             Location loc,
                                             llvm::StringRef kernelName,
                                             llvm::StringRef kind, Type ptrType,
                                             unsigned numArgs = 3) {
  rewriter.setInsertionPointToEnd(executableModule.getBody());
  exec::KernelOp kernel =
      rewriter.create<exec::KernelOp>(loc, kernelName, kind);

  Region &body = kernel.getBody();
  body.push_back(new Block());
  Block &entry = body.front();
  llvm::SmallVector<Type, 3> argTypes(numArgs, ptrType);
  llvm::SmallVector<Location, 3> argLocs(numArgs, loc);
  entry.addArguments(argTypes, argLocs);
  rewriter.setInsertionPointToStart(&entry);
  return kernel;
}

static void buildAddExecutableIntoModule(PatternRewriter &rewriter,
                                         ModuleOp executableModule,
                                         llvm::StringRef kernelName, int64_t n,
                                         int64_t logicalN, bool masked) {
  MLIRContext *context = executableModule.getContext();
  loadExecutableDialects(*context);
  Location loc = UnknownLoc::get(context);

  Type f16 = rewriter.getF16Type();
  Type ptrType = triton::PointerType::get(f16, 1);
  llvm::SmallVector<int64_t, 1> shape{n};
  Type tile = exec::TileType::get(context, shape, f16);

  exec::KernelOp kernel =
      n <= 128 ? createExecutableKernel(rewriter, executableModule, loc,
                                        kernelName, "generic", ptrType)
               : createExecutableKernel(rewriter, executableModule, loc,
                                        kernelName, "add", ptrType);
  Block &entry = kernel.getBody().front();
  Value out = entry.getArgument(0);
  Value lhs = entry.getArgument(1);
  Value rhs = entry.getArgument(2);

  auto lhsTile = rewriter.create<exec::LoadContigOp>(loc, tile, lhs, n);
  auto rhsTile = rewriter.create<exec::LoadContigOp>(loc, tile, rhs, n);
  auto sum = rewriter.create<exec::AddOp>(loc, tile, lhsTile.getResult(),
                                          rhsTile.getResult());
  auto store =
      rewriter.create<exec::StoreContigOp>(loc, out, sum.getResult(), n);
  if (masked) {
    lhsTile->setAttr("masked", rewriter.getBoolAttr(true));
    lhsTile->setAttr("logical_n", rewriter.getI32IntegerAttr(logicalN));
    lhsTile->setAttr("block_n", rewriter.getI32IntegerAttr(n));
    rhsTile->setAttr("masked", rewriter.getBoolAttr(true));
    rhsTile->setAttr("logical_n", rewriter.getI32IntegerAttr(logicalN));
    rhsTile->setAttr("block_n", rewriter.getI32IntegerAttr(n));
    store->setAttr("masked", rewriter.getBoolAttr(true));
    store->setAttr("logical_n", rewriter.getI32IntegerAttr(logicalN));
    store->setAttr("block_n", rewriter.getI32IntegerAttr(n));
  }
  rewriter.create<exec::ReturnOp>(loc);
}

struct ExecutableGemmShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

static void buildGemmExecutableIntoModule(PatternRewriter &rewriter,
                                          ModuleOp executableModule,
                                          llvm::StringRef kernelName,
                                          ExecutableGemmShape shape) {
  MLIRContext *context = executableModule.getContext();
  loadExecutableDialects(*context);
  Location loc = UnknownLoc::get(context);

  Type f16 = rewriter.getF16Type();
  Type ptrType = triton::PointerType::get(f16, 1);
  llvm::SmallVector<int64_t, 2> lhsShape{shape.m, shape.k};
  llvm::SmallVector<int64_t, 2> rhsShape{shape.k, shape.n};
  llvm::SmallVector<int64_t, 2> accShape{shape.m, shape.n};
  Type tileMK = exec::TileType::get(context, lhsShape, f16);
  Type tileKN = exec::TileType::get(context, rhsShape, f16);
  Type tileMN = exec::TileType::get(context, accShape, f16);

  exec::KernelOp kernel = createExecutableKernel(
      rewriter, executableModule, loc, kernelName, "gemm", ptrType);
  Block &entry = kernel.getBody().front();
  Value out = entry.getArgument(0);
  Value lhs = entry.getArgument(1);
  Value rhs = entry.getArgument(2);

  auto a =
      rewriter.create<exec::LoadMatrixOp>(loc, tileMK, lhs, shape.m, shape.k);
  auto b =
      rewriter.create<exec::LoadMatrixOp>(loc, tileKN, rhs, shape.k, shape.n);
  auto acc0 = rewriter.create<exec::ZeroOp>(loc, tileMN, shape.m, shape.n);
  auto acc = rewriter.create<exec::MmaOp>(loc, tileMN, a.getResult(),
                                          b.getResult(), acc0.getResult());
  rewriter.create<exec::StoreMatrixOp>(loc, out, acc.getResult(), shape.m,
                                       shape.n);
  rewriter.create<exec::ReturnOp>(loc);
}

static void buildSoftmaxExecutableIntoModule(PatternRewriter &rewriter,
                                             ModuleOp executableModule,
                                             llvm::StringRef kernelName,
                                             int64_t nvec) {
  MLIRContext *context = executableModule.getContext();
  loadExecutableDialects(*context);
  Location loc = UnknownLoc::get(context);

  Type f16 = rewriter.getF16Type();
  Type ptrType = triton::PointerType::get(f16, 1);
  llvm::SmallVector<int64_t, 1> shape{nvec};
  Type tile = exec::TileType::get(context, shape, f16);

  exec::KernelOp kernel =
      createExecutableKernel(rewriter, executableModule, loc, kernelName,
                             "softmax", ptrType, /*numArgs=*/2);
  Block &entry = kernel.getBody().front();
  Value out = entry.getArgument(0);
  Value input = entry.getArgument(1);

  auto x = rewriter.create<exec::LoadContigOp>(loc, tile, input, nvec);
  auto m = rewriter.create<exec::ReduceMaxAllOp>(loc, f16, x.getResult());
  auto shifted = rewriter.create<exec::SubScalarOp>(loc, tile, x.getResult(),
                                                    m.getResult());
  auto e = rewriter.create<exec::ExpOp>(loc, tile, shifted.getResult());
  auto s = rewriter.create<exec::ReduceSumAllOp>(loc, f16, e.getResult());
  auto invS = rewriter.create<exec::ReciprocalOp>(loc, f16, s.getResult());
  auto y = rewriter.create<exec::MulScalarOp>(loc, tile, e.getResult(),
                                              invS.getResult());
  rewriter.create<exec::StoreContigOp>(loc, out, y.getResult(), nvec);
  rewriter.create<exec::ReturnOp>(loc);
}

static void buildConvKxKExecutableIntoModule(PatternRewriter &rewriter,
                                             ModuleOp executableModule,
                                             llvm::StringRef kernelName,
                                             int64_t kernelSize) {
  MLIRContext *context = executableModule.getContext();
  loadExecutableDialects(*context);
  Location loc = UnknownLoc::get(context);

  Type f16 = rewriter.getF16Type();
  Type ptrType = triton::PointerType::get(f16, 1);
  llvm::SmallVector<int64_t, 2> shape{16, 16};
  Type tile16x16 = exec::TileType::get(context, shape, f16);

  exec::KernelOp kernel = createExecutableKernel(
      rewriter, executableModule, loc, kernelName, "convkxk", ptrType);
  Block &entry = kernel.getBody().front();
  Value out = entry.getArgument(0);
  Value input = entry.getArgument(1);
  Value weight = entry.getArgument(2);

  Value acc = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16).getResult();
  for (int64_t ky = 0; ky < kernelSize; ++ky) {
    for (int64_t kx = 0; kx < kernelSize; ++kx) {
      int64_t kernelIndex = ky * kernelSize + kx;
      int64_t xRow = ky * 16 + kx;
      int64_t wRow = kernelIndex * 16;
      auto x =
          rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, input, 16, 16);
      setMatrixOffsets(rewriter, x.getOperation(), xRow);
      auto w =
          rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, weight, 16, 16);
      setMatrixOffsets(rewriter, w.getOperation(), wRow);
      acc = rewriter
                .create<exec::MmaOp>(loc, tile16x16, x.getResult(),
                                     w.getResult(), acc)
                .getResult();
    }
  }
  rewriter.create<exec::StoreMatrixOp>(loc, out, acc, 16, 16);
  rewriter.create<exec::ReturnOp>(loc);
}

static void buildResNetBlock16ExecutableIntoModule(PatternRewriter &rewriter,
                                                   ModuleOp executableModule,
                                                   llvm::StringRef kernelName,
                                                   int64_t hidden) {
  MLIRContext *context = executableModule.getContext();
  loadExecutableDialects(*context);
  Location loc = UnknownLoc::get(context);
  Type f16 = rewriter.getF16Type();
  Type ptrType = triton::PointerType::get(f16, 1);
  llvm::SmallVector<int64_t, 2> shape16x16{16, 16};
  llvm::SmallVector<int64_t, 2> shape16xHidden{16, hidden};
  llvm::SmallVector<int64_t, 2> shapeHiddenx16{hidden, 16};
  Type tile16x16 = exec::TileType::get(context, shape16x16, f16);
  Type tile16xHidden = exec::TileType::get(context, shape16xHidden, f16);
  Type tileHiddenx16 = exec::TileType::get(context, shapeHiddenx16, f16);

  exec::KernelOp kernel =
      createExecutableKernel(rewriter, executableModule, loc, kernelName,
                             "resnet_block", ptrType, /*numArgs=*/4);
  Block &entry = kernel.getBody().front();
  Value out = entry.getArgument(0);
  Value xPtr = entry.getArgument(1);
  Value w1Ptr = entry.getArgument(2);
  Value w2Ptr = entry.getArgument(3);

  auto x = rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
  if (hidden == 32) {
    auto w1Lo =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setMatrixOffsets(rewriter, w1Lo.getOperation(), 0, 0);
    auto conv1LoZero = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto conv1Lo =
        rewriter.create<exec::MmaOp>(loc, tile16x16, x.getResult(),
                                     w1Lo.getResult(), conv1LoZero.getResult());
    auto zero1Lo = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto relu1Lo = rewriter.create<exec::MaxOp>(
        loc, tile16x16, conv1Lo.getResult(), zero1Lo.getResult());

    auto w1Hi =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setMatrixOffsets(rewriter, w1Hi.getOperation(), 0, 16);
    auto conv1HiZero = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto conv1Hi =
        rewriter.create<exec::MmaOp>(loc, tile16x16, x.getResult(),
                                     w1Hi.getResult(), conv1HiZero.getResult());
    auto zero1Hi = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto relu1Hi = rewriter.create<exec::MaxOp>(
        loc, tile16x16, conv1Hi.getResult(), zero1Hi.getResult());

    auto w2Lo =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
    setMatrixOffsets(rewriter, w2Lo.getOperation(), 0, 0);
    Value conv2Acc =
        rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16).getResult();
    conv2Acc = rewriter
                   .create<exec::MmaOp>(loc, tile16x16, relu1Lo.getResult(),
                                        w2Lo.getResult(), conv2Acc)
                   .getResult();
    auto w2Hi =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
    setMatrixOffsets(rewriter, w2Hi.getOperation(), 16, 0);
    conv2Acc = rewriter
                   .create<exec::MmaOp>(loc, tile16x16, relu1Hi.getResult(),
                                        w2Hi.getResult(), conv2Acc)
                   .getResult();

    auto residual =
        rewriter.create<exec::AddOp>(loc, tile16x16, conv2Acc, x.getResult());
    auto zero2 = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto outTile = rewriter.create<exec::MaxOp>(
        loc, tile16x16, residual.getResult(), zero2.getResult());
    rewriter.create<exec::StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
    rewriter.create<exec::ReturnOp>(loc);
    return;
  }

  auto w1 = rewriter.create<exec::LoadMatrixOp>(loc, tile16xHidden, w1Ptr, 16,
                                                hidden);
  auto conv1Zero =
      rewriter.create<exec::ZeroOp>(loc, tile16xHidden, 16, hidden);
  auto conv1 = rewriter.create<exec::MmaOp>(
      loc, tile16xHidden, x.getResult(), w1.getResult(), conv1Zero.getResult());
  auto zero1 = rewriter.create<exec::ZeroOp>(loc, tile16xHidden, 16, hidden);
  auto relu1 = rewriter.create<exec::MaxOp>(
      loc, tile16xHidden, conv1.getResult(), zero1.getResult());
  auto w2 = rewriter.create<exec::LoadMatrixOp>(loc, tileHiddenx16, w2Ptr,
                                                hidden, 16);
  auto conv2Zero = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
  auto conv2 = rewriter.create<exec::MmaOp>(
      loc, tile16x16, relu1.getResult(), w2.getResult(), conv2Zero.getResult());
  auto residual = rewriter.create<exec::AddOp>(
      loc, tile16x16, conv2.getResult(), x.getResult());
  auto zero2 = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
  auto outTile = rewriter.create<exec::MaxOp>(
      loc, tile16x16, residual.getResult(), zero2.getResult());
  rewriter.create<exec::StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
  rewriter.create<exec::ReturnOp>(loc);
}

static void buildResNet50Bottleneck16ExecutableIntoModule(
    PatternRewriter &rewriter, ModuleOp executableModule,
    llvm::StringRef kernelName, int64_t bottleneck) {
  MLIRContext *context = executableModule.getContext();
  loadExecutableDialects(*context);
  Location loc = UnknownLoc::get(context);
  Type f16 = rewriter.getF16Type();
  Type ptrType = triton::PointerType::get(f16, 1);
  llvm::SmallVector<int64_t, 2> shape{16, 16};
  Type tile16x16 = exec::TileType::get(context, shape, f16);

  exec::KernelOp kernel =
      createExecutableKernel(rewriter, executableModule, loc, kernelName,
                             "resnet50_bottleneck", ptrType, /*numArgs=*/5);
  Block &entry = kernel.getBody().front();
  Value out = entry.getArgument(0);
  Value xPtr = entry.getArgument(1);
  Value w1Ptr = entry.getArgument(2);
  Value w2Ptr = entry.getArgument(3);
  Value w3Ptr = entry.getArgument(4);

  auto xSkip =
      rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
  setMatrixOffsets(rewriter, xSkip.getOperation(), 0);
  if (bottleneck == 32) {
    auto w1Lo =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setMatrixOffsets(rewriter, w1Lo.getOperation(), 0, 0);
    auto w1Hi =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setMatrixOffsets(rewriter, w1Hi.getOperation(), 0, 16);
    Value conv2AccLo =
        rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16).getResult();
    Value conv2AccHi =
        rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16).getResult();

    for (int64_t ky = 0; ky < 3; ++ky) {
      for (int64_t kx = 0; kx < 3; ++kx) {
        int64_t kernelIndex = ky * 3 + kx;
        int64_t xRow = ky * 16 + kx;
        int64_t w2Row = kernelIndex * 32;
        auto x =
            rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
        setMatrixOffsets(rewriter, x.getOperation(), xRow);
        auto conv1LoZero =
            rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
        auto conv1Lo = rewriter.create<exec::MmaOp>(
            loc, tile16x16, x.getResult(), w1Lo.getResult(),
            conv1LoZero.getResult());
        auto zero1Lo = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
        auto relu1Lo = rewriter.create<exec::MaxOp>(
            loc, tile16x16, conv1Lo.getResult(), zero1Lo.getResult());
        auto conv1HiZero =
            rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
        auto conv1Hi = rewriter.create<exec::MmaOp>(
            loc, tile16x16, x.getResult(), w1Hi.getResult(),
            conv1HiZero.getResult());
        auto zero1Hi = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
        auto relu1Hi = rewriter.create<exec::MaxOp>(
            loc, tile16x16, conv1Hi.getResult(), zero1Hi.getResult());

        auto w2LoLo =
            rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setMatrixOffsets(rewriter, w2LoLo.getOperation(), w2Row, 0);
        conv2AccLo =
            rewriter
                .create<exec::MmaOp>(loc, tile16x16, relu1Lo.getResult(),
                                     w2LoLo.getResult(), conv2AccLo)
                .getResult();
        auto w2HiLo =
            rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setMatrixOffsets(rewriter, w2HiLo.getOperation(), w2Row + 16, 0);
        conv2AccLo =
            rewriter
                .create<exec::MmaOp>(loc, tile16x16, relu1Hi.getResult(),
                                     w2HiLo.getResult(), conv2AccLo)
                .getResult();
        auto w2LoHi =
            rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setMatrixOffsets(rewriter, w2LoHi.getOperation(), w2Row, 16);
        conv2AccHi =
            rewriter
                .create<exec::MmaOp>(loc, tile16x16, relu1Lo.getResult(),
                                     w2LoHi.getResult(), conv2AccHi)
                .getResult();
        auto w2HiHi =
            rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setMatrixOffsets(rewriter, w2HiHi.getOperation(), w2Row + 16, 16);
        conv2AccHi =
            rewriter
                .create<exec::MmaOp>(loc, tile16x16, relu1Hi.getResult(),
                                     w2HiHi.getResult(), conv2AccHi)
                .getResult();
      }
    }

    auto zero2Lo = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto relu2Lo = rewriter.create<exec::MaxOp>(loc, tile16x16, conv2AccLo,
                                                zero2Lo.getResult());
    auto zero2Hi = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto relu2Hi = rewriter.create<exec::MaxOp>(loc, tile16x16, conv2AccHi,
                                                zero2Hi.getResult());
    auto w3Lo =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w3Ptr, 16, 16);
    setMatrixOffsets(rewriter, w3Lo.getOperation(), 0, 0);
    auto conv3Zero = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    Value conv3 =
        rewriter
            .create<exec::MmaOp>(loc, tile16x16, relu2Lo.getResult(),
                                 w3Lo.getResult(), conv3Zero.getResult())
            .getResult();
    auto w3Hi =
        rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w3Ptr, 16, 16);
    setMatrixOffsets(rewriter, w3Hi.getOperation(), 16, 0);
    conv3 = rewriter
                .create<exec::MmaOp>(loc, tile16x16, relu2Hi.getResult(),
                                     w3Hi.getResult(), conv3)
                .getResult();
    auto residual =
        rewriter.create<exec::AddOp>(loc, tile16x16, conv3, xSkip.getResult());
    auto zero3 = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
    auto outTile = rewriter.create<exec::MaxOp>(
        loc, tile16x16, residual.getResult(), zero3.getResult());
    rewriter.create<exec::StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
    rewriter.create<exec::ReturnOp>(loc);
    return;
  }

  auto w1 = rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
  Value conv2Acc =
      rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16).getResult();

  for (int64_t ky = 0; ky < 3; ++ky) {
    for (int64_t kx = 0; kx < 3; ++kx) {
      int64_t kernelIndex = ky * 3 + kx;
      int64_t xRow = ky * 16 + kx;
      int64_t w2Row = kernelIndex * 16;
      auto x =
          rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
      setMatrixOffsets(rewriter, x.getOperation(), xRow);
      auto conv1Zero = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
      auto conv1 = rewriter.create<exec::MmaOp>(
          loc, tile16x16, x.getResult(), w1.getResult(), conv1Zero.getResult());
      auto zero1 = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
      auto relu1 = rewriter.create<exec::MaxOp>(
          loc, tile16x16, conv1.getResult(), zero1.getResult());
      auto w2 =
          rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
      setMatrixOffsets(rewriter, w2.getOperation(), w2Row);
      conv2Acc = rewriter
                     .create<exec::MmaOp>(loc, tile16x16, relu1.getResult(),
                                          w2.getResult(), conv2Acc)
                     .getResult();
    }
  }

  auto zero2 = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
  auto relu2 =
      rewriter.create<exec::MaxOp>(loc, tile16x16, conv2Acc, zero2.getResult());
  auto w3 = rewriter.create<exec::LoadMatrixOp>(loc, tile16x16, w3Ptr, 16, 16);
  auto conv3Zero = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
  auto conv3 = rewriter.create<exec::MmaOp>(
      loc, tile16x16, relu2.getResult(), w3.getResult(), conv3Zero.getResult());
  auto residual = rewriter.create<exec::AddOp>(
      loc, tile16x16, conv3.getResult(), xSkip.getResult());
  auto zero3 = rewriter.create<exec::ZeroOp>(loc, tile16x16, 16, 16);
  auto outTile = rewriter.create<exec::MaxOp>(
      loc, tile16x16, residual.getResult(), zero3.getResult());
  rewriter.create<exec::StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
  rewriter.create<exec::ReturnOp>(loc);
}

static void eraseModuleBodyExceptPlanRoot(ModuleOp module, plan::KernelOp root,
                                          PatternRewriter &rewriter) {
  llvm::SmallVector<Operation *> ops;
  for (Operation &op : module.getBody()->getOperations()) {
    if (&op != root.getOperation())
      ops.push_back(&op);
  }
  for (Operation *op : ops)
    rewriter.eraseOp(op);
}

struct AddExecutableConfig {
  std::string kernelName;
  int64_t n;
  int64_t logicalN;
  bool masked;
};

static std::optional<AddExecutableConfig>
getAddContiguousExecutableSpec(plan::KernelOp op) {
  PlanKernelAttrView view(op);

  std::optional<llvm::StringRef> kind = view.emissionString("kind");
  std::optional<int64_t> n = view.emissionInteger("n");
  std::optional<int64_t> logicalN = view.emissionInteger("logical_n");
  std::optional<bool> masked = view.emissionBool("masked");
  std::optional<int64_t> out = view.emissionInteger("out");
  std::optional<int64_t> lhs = view.emissionInteger("lhs");
  std::optional<int64_t> rhs = view.emissionInteger("rhs");
  std::optional<llvm::StringRef> access = view.layoutString("access");
  std::optional<llvm::StringRef> memory = view.layoutString("memory");
  std::optional<llvm::StringRef> order = view.layoutString("order");
  std::optional<bool> maskMasked = view.maskBool("masked");
  std::optional<int64_t> blockN = view.maskInteger("block_n");
  bool contiguousLayout =
      access && *access == "linear" && memory && *memory == "contiguous_vector";
  bool chunkedLayout = access && *access == "chunked_tile_view" && memory &&
                       *memory == "tensor" && order && *order == "row_major";
  bool maskedLayout = access && *access == "masked_tile_view" && memory &&
                      *memory == "tensor" && order && *order == "row_major";

  if (!view.isPattern("add") || !kind || *kind != "add" || !n || !logicalN ||
      !masked || !maskMasked || !out || *out != 0 || !lhs || *lhs != 1 ||
      !rhs || *rhs != 2 || !isPositiveMultipleOf16(*n))
    return std::nullopt;

  if (*masked) {
    if (!*maskMasked || !blockN || *blockN != *n || *logicalN <= 0 ||
        *logicalN > *n || !maskedLayout)
      return std::nullopt;
    return AddExecutableConfig{view.kernelName().str(), *n, *logicalN, true};
  }

  if (*maskMasked || *logicalN != *n || (!contiguousLayout && !chunkedLayout))
    return std::nullopt;
  return AddExecutableConfig{view.kernelName().str(), *n, *logicalN, false};
}

struct GemmExecutableShape {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct GemmExecutableSpec {
  std::string kernelName;
  GemmExecutableShape shape;
};

static std::optional<GemmExecutableSpec>
getGemmExecutableSpec(plan::KernelOp op) {
  PlanKernelAttrView view(op);

  std::optional<llvm::StringRef> kind = view.emissionString("kind");
  std::optional<int64_t> m = view.emissionInteger("m");
  std::optional<int64_t> n = view.emissionInteger("n");
  std::optional<int64_t> k = view.emissionInteger("k");
  std::optional<int64_t> out = view.emissionInteger("out");
  std::optional<int64_t> lhs = view.emissionInteger("lhs");
  std::optional<int64_t> rhs = view.emissionInteger("rhs");
  if (!view.isPattern("gemm") || !kind || *kind != "gemm" || !m || !n || !k ||
      !isPositiveMultipleOf16(*m) || !isPositiveMultipleOf16(*n) ||
      !isPositiveMultipleOf16(*k) || !out || *out != 0 || !lhs || *lhs != 1 ||
      !rhs || *rhs != 2)
    return std::nullopt;
  return GemmExecutableSpec{view.kernelName().str(),
                            GemmExecutableShape{*m, *n, *k}};
}

struct SoftmaxExecutableSpec {
  std::string kernelName;
  int64_t nvec;
};

static std::optional<SoftmaxExecutableSpec>
getSoftmaxExecutableSpec(plan::KernelOp op) {
  PlanKernelAttrView view(op);

  std::optional<llvm::StringRef> kind = view.emissionString("kind");
  std::optional<int64_t> n = view.emissionInteger("n");
  std::optional<int64_t> out = view.emissionInteger("out");
  std::optional<int64_t> input = view.emissionInteger("input");
  if (!view.isPattern("softmax") || !kind || *kind != "softmax" || !n ||
      !isPositiveMultipleOf16(*n) || !out || *out != 0 || !input || *input != 1)
    return std::nullopt;
  return SoftmaxExecutableSpec{view.kernelName().str(), *n / 16};
}

struct ConvKxKExecutableSpec {
  std::string kernelName;
  int64_t kernelSize;
};

static std::optional<ConvKxKExecutableSpec>
getConvKxKExecutableSpec(plan::KernelOp op) {
  PlanKernelAttrView view(op);

  std::optional<llvm::StringRef> kind = view.emissionString("kind");
  std::optional<int64_t> kernelSize = view.emissionInteger("kernel_size");
  std::optional<int64_t> m = view.emissionInteger("m");
  std::optional<int64_t> inChannels = view.emissionInteger("in_channels");
  std::optional<int64_t> outChannels = view.emissionInteger("out_channels");
  std::optional<int64_t> inputWidth = view.emissionInteger("input_width");
  std::optional<int64_t> out = view.emissionInteger("out");
  std::optional<int64_t> input = view.emissionInteger("input");
  std::optional<int64_t> weight = view.emissionInteger("weight");
  if (!view.isPattern("convkxk") || !kind || *kind != "convkxk" ||
      !kernelSize ||
      !exec::isSupportedExecutableConvKxKKernelSize(*kernelSize) || !m ||
      *m != 16 || !inChannels || *inChannels != 16 || !outChannels ||
      *outChannels != 16 || !inputWidth || *inputWidth != 16 || !out ||
      *out != 0 || !input || *input != 1 || !weight || *weight != 2)
    return std::nullopt;
  return ConvKxKExecutableSpec{view.kernelName().str(), *kernelSize};
}

struct ResNetBlockExecutableSpec {
  std::string kernelName;
  int64_t hidden;
};

static std::optional<ResNetBlockExecutableSpec>
getResNetBlock16ExecutableSpec(plan::KernelOp op) {
  PlanKernelAttrView view(op);

  std::optional<llvm::StringRef> kind = view.emissionString("kind");
  std::optional<int64_t> m = view.emissionInteger("m");
  std::optional<int64_t> channels = view.emissionInteger("channels");
  std::optional<int64_t> hidden = view.emissionInteger("hidden");
  std::optional<int64_t> out = view.emissionInteger("out");
  std::optional<int64_t> x = view.emissionInteger("x");
  std::optional<int64_t> w1 = view.emissionInteger("w1");
  std::optional<int64_t> w2 = view.emissionInteger("w2");
  if (!view.isPattern("resnet_block") || !kind || *kind != "resnet_block" ||
      !m || *m != 16 || !channels || *channels != 16 || !hidden ||
      (*hidden != 16 && *hidden != 32) || !out || *out != 0 || !x || *x != 1 ||
      !w1 || *w1 != 2 || !w2 || *w2 != 3)
    return std::nullopt;
  return ResNetBlockExecutableSpec{view.kernelName().str(), *hidden};
}

struct ResNet50BottleneckExecutableSpec {
  std::string kernelName;
  int64_t bottleneck;
};

static std::optional<ResNet50BottleneckExecutableSpec>
getResNet50Bottleneck16ExecutableSpec(plan::KernelOp op) {
  PlanKernelAttrView view(op);

  std::optional<llvm::StringRef> kind = view.emissionString("kind");
  std::optional<int64_t> kernelSize = view.emissionInteger("kernel_size");
  std::optional<int64_t> m = view.emissionInteger("m");
  std::optional<int64_t> channels = view.emissionInteger("channels");
  std::optional<int64_t> bottleneck = view.emissionInteger("bottleneck");
  std::optional<int64_t> inputWidth = view.emissionInteger("input_width");
  std::optional<int64_t> out = view.emissionInteger("out");
  std::optional<int64_t> input = view.emissionInteger("input");
  std::optional<int64_t> w1 = view.emissionInteger("w1");
  std::optional<int64_t> w2 = view.emissionInteger("w2");
  std::optional<int64_t> w3 = view.emissionInteger("w3");
  if (!view.isPattern("resnet50_bottleneck") || !kind ||
      *kind != "resnet50_bottleneck" || !kernelSize || *kernelSize != 3 || !m ||
      *m != 16 || !channels || *channels != 16 || !bottleneck ||
      (*bottleneck != 16 && *bottleneck != 32) || !inputWidth ||
      *inputWidth != 16 || !out || *out != 0 || !input || *input != 1 || !w1 ||
      *w1 != 2 || !w2 || *w2 != 3 || !w3 || *w3 != 4)
    return std::nullopt;
  return ResNet50BottleneckExecutableSpec{view.kernelName().str(), *bottleneck};
}

enum class ExecutableBuildKind {
  Add,
  Gemm,
  Softmax,
  ConvKxK,
  ResNetBlock,
  ResNet50Bottleneck,
};

struct ExecutableBuildSpec {
  ExecutableBuildKind kind;
  std::string kernelName;
  int64_t n = 0;
  GemmExecutableShape gemmShape{0, 0, 0};
  int64_t nvec = 0;
  int64_t kernelSize = 0;
  int64_t logicalN = 0;
  bool masked = false;
  int64_t hidden = 0;
  int64_t bottleneck = 0;
};

using MatchFn = std::optional<ExecutableBuildSpec> (*)(plan::KernelOp op);
using BuildFn = LogicalResult (*)(plan::KernelOp op, PatternRewriter &rewriter,
                                  ModuleOp module,
                                  const ExecutableBuildSpec &spec);

struct ExecutableBuildEntry {
  llvm::StringLiteral kind;
  MatchFn match;
  BuildFn build;
};

static std::optional<ExecutableBuildSpec> matchAdd(plan::KernelOp op) {
  if (std::optional<AddExecutableConfig> add =
          getAddContiguousExecutableSpec(op)) {
    ExecutableBuildSpec spec{ExecutableBuildKind::Add, add->kernelName};
    spec.n = add->n;
    spec.logicalN = add->logicalN;
    spec.masked = add->masked;
    return spec;
  }
  return std::nullopt;
}

static LogicalResult buildAdd(plan::KernelOp op, PatternRewriter &rewriter,
                              ModuleOp module,
                              const ExecutableBuildSpec &spec) {
  eraseModuleBodyExceptPlanRoot(module, op, rewriter);
  buildAddExecutableIntoModule(rewriter, module, spec.kernelName, spec.n,
                               spec.logicalN, spec.masked);
  rewriter.eraseOp(op);
  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Add executable RPU module");
  return success();
}

static std::optional<ExecutableBuildSpec> matchGemm(plan::KernelOp op) {
  if (std::optional<GemmExecutableSpec> gemm = getGemmExecutableSpec(op)) {
    ExecutableBuildSpec spec{ExecutableBuildKind::Gemm, gemm->kernelName};
    spec.gemmShape = gemm->shape;
    return spec;
  }
  return std::nullopt;
}

static LogicalResult buildGemm(plan::KernelOp op, PatternRewriter &rewriter,
                               ModuleOp module,
                               const ExecutableBuildSpec &spec) {
  eraseModuleBodyExceptPlanRoot(module, op, rewriter);
  buildGemmExecutableIntoModule(rewriter, module, spec.kernelName,
                                ExecutableGemmShape{spec.gemmShape.m,
                                                    spec.gemmShape.n,
                                                    spec.gemmShape.k});
  rewriter.eraseOp(op);
  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct GEMM executable RPU module");
  return success();
}

static std::optional<ExecutableBuildSpec> matchSoftmax(plan::KernelOp op) {
  if (std::optional<SoftmaxExecutableSpec> softmax =
          getSoftmaxExecutableSpec(op)) {
    ExecutableBuildSpec spec{ExecutableBuildKind::Softmax, softmax->kernelName};
    spec.nvec = softmax->nvec;
    return spec;
  }
  return std::nullopt;
}

static LogicalResult buildSoftmax(plan::KernelOp op, PatternRewriter &rewriter,
                                  ModuleOp module,
                                  const ExecutableBuildSpec &spec) {
  eraseModuleBodyExceptPlanRoot(module, op, rewriter);
  buildSoftmaxExecutableIntoModule(rewriter, module, spec.kernelName,
                                   spec.nvec);
  rewriter.eraseOp(op);
  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct Softmax executable RPU module");
  return success();
}

static std::optional<ExecutableBuildSpec> matchConvKxK(plan::KernelOp op) {
  if (std::optional<ConvKxKExecutableSpec> conv =
          getConvKxKExecutableSpec(op)) {
    ExecutableBuildSpec spec{ExecutableBuildKind::ConvKxK, conv->kernelName};
    spec.kernelSize = conv->kernelSize;
    return spec;
  }
  return std::nullopt;
}

static LogicalResult buildConvKxK(plan::KernelOp op, PatternRewriter &rewriter,
                                  ModuleOp module,
                                  const ExecutableBuildSpec &spec) {
  eraseModuleBodyExceptPlanRoot(module, op, rewriter);
  buildConvKxKExecutableIntoModule(rewriter, module, spec.kernelName,
                                   spec.kernelSize);
  rewriter.eraseOp(op);
  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct ConvKxK executable RPU module");
  return success();
}

static std::optional<ExecutableBuildSpec> matchResNetBlock(plan::KernelOp op) {
  if (std::optional<ResNetBlockExecutableSpec> residual =
          getResNetBlock16ExecutableSpec(op)) {
    ExecutableBuildSpec spec{ExecutableBuildKind::ResNetBlock,
                             residual->kernelName};
    spec.hidden = residual->hidden;
    return spec;
  }
  return std::nullopt;
}

static LogicalResult buildResNetBlock(plan::KernelOp op,
                                      PatternRewriter &rewriter,
                                      ModuleOp module,
                                      const ExecutableBuildSpec &spec) {
  eraseModuleBodyExceptPlanRoot(module, op, rewriter);
  buildResNetBlock16ExecutableIntoModule(rewriter, module, spec.kernelName,
                                         spec.hidden);
  rewriter.eraseOp(op);
  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct residual executable RPU module");
  return success();
}

static std::optional<ExecutableBuildSpec>
matchResNet50Bottleneck(plan::KernelOp op) {
  if (std::optional<ResNet50BottleneckExecutableSpec> bottleneck =
          getResNet50Bottleneck16ExecutableSpec(op)) {
    ExecutableBuildSpec spec{ExecutableBuildKind::ResNet50Bottleneck,
                             bottleneck->kernelName};
    spec.bottleneck = bottleneck->bottleneck;
    return spec;
  }
  return std::nullopt;
}

static LogicalResult buildResNet50Bottleneck(plan::KernelOp op,
                                             PatternRewriter &rewriter,
                                             ModuleOp module,
                                             const ExecutableBuildSpec &spec) {
  eraseModuleBodyExceptPlanRoot(module, op, rewriter);
  buildResNet50Bottleneck16ExecutableIntoModule(
      rewriter, module, spec.kernelName, spec.bottleneck);
  rewriter.eraseOp(op);
  if (failed(verify(module)))
    return module.emitError(
        "failed to verify direct bottleneck executable RPU module");
  return success();
}

static constexpr ExecutableBuildEntry kExecutableBuildEntries[] = {
    {"add", matchAdd, buildAdd},
    {"gemm", matchGemm, buildGemm},
    {"softmax", matchSoftmax, buildSoftmax},
    {"convkxk", matchConvKxK, buildConvKxK},
    {"resnet_block", matchResNetBlock, buildResNetBlock},
    {"resnet50_bottleneck", matchResNet50Bottleneck, buildResNet50Bottleneck},
};

static std::optional<
    std::pair<const ExecutableBuildEntry *, ExecutableBuildSpec>>
getExecutableBuildSpec(plan::KernelOp op) {
  for (const ExecutableBuildEntry &entry : kExecutableBuildEntries) {
    if (std::optional<ExecutableBuildSpec> spec = entry.match(op))
      return std::make_pair(&entry, *spec);
  }
  return std::nullopt;
}

static std::optional<
    std::pair<const ExecutableBuildEntry *, ExecutableBuildSpec>>
getExecutableBuildSpecForKind(plan::KernelOp op, llvm::StringRef kind) {
  for (const ExecutableBuildEntry &entry : kExecutableBuildEntries) {
    if (entry.kind != kind)
      continue;
    if (std::optional<ExecutableBuildSpec> spec = entry.match(op))
      return std::make_pair(&entry, *spec);
    return std::nullopt;
  }
  return std::nullopt;
}

static bool isSupportedExecutableBuildPattern(llvm::StringRef pattern) {
  for (const ExecutableBuildEntry &entry : kExecutableBuildEntries) {
    if (entry.kind == pattern)
      return true;
  }
  return false;
}

static std::optional<llvm::StringRef> getPlanEmissionKind(plan::KernelOp op) {
  return PlanKernelAttrView(op).emissionString("kind");
}

static std::string unsupportedExecutablePattern(llvm::StringRef pattern) {
  std::string result = "unsupported executable pattern '";
  result += pattern.str();
  result += "'";
  return result;
}

static std::string executableBuildContractForPattern(llvm::StringRef pattern) {
  if (pattern == "add")
    return "add requires unmasked f16 vector attrs with out=0 lhs=1 rhs=2, "
           "logical_n=n, positive 16-aligned n, and contiguous or chunked "
           "row-major layout";
  if (pattern == "gemm")
    return "gemm requires f16 matrix attrs with out=0 lhs=1 rhs=2 and "
           "positive 16-aligned m/n/k";
  if (pattern == "softmax")
    return "softmax requires f16 contiguous vector attrs with out=0 input=1 "
           "and positive 16-aligned n";
  if (pattern == "convkxk")
    return "convkxk requires out=0 input=1 weight=2, "
           "kernel_size in {3,5,7,9}, "
           "and m/in_channels/out_channels/input_width all 16";
  if (pattern == "resnet_block")
    return "resnet_block requires out=0 x=1 w1=2 w2=3 and "
           "m/channels are 16 and hidden is in {16,32}";
  if (pattern == "resnet50_bottleneck")
    return "resnet50_bottleneck requires out=0 input=1 w1=2 w2=3 w3=4, "
           "kernel_size=3, m/channels/input_width all 16, and "
           "bottleneck in {16,32}";
  return unsupportedExecutablePattern(pattern);
}

class RPUPlanExecutableBuildPattern : public OpRewritePattern<plan::KernelOp> {
public:
  using OpRewritePattern<plan::KernelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(plan::KernelOp op,
                                PatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    std::optional<std::pair<const ExecutableBuildEntry *, ExecutableBuildSpec>>
        spec = getExecutableBuildSpec(op);
    if (!spec)
      return failure();
    return spec->first->build(op, rewriter, module, spec->second);
  }
};

} // namespace

bool supportsRPUPlanKernelExecutableBuild(plan::KernelOp op) {
  return getExecutableBuildSpec(op).has_value();
}

bool supportsRPUPlanKernelExecutableBuildKind(plan::KernelOp op,
                                              llvm::StringRef kind) {
  return getExecutableBuildSpecForKind(op, kind).has_value();
}

std::string describeRPUPlanKernelExecutableBuildFailure(plan::KernelOp op) {
  llvm::StringRef pattern = op.getPattern();
  if (!isSupportedExecutableBuildPattern(pattern))
    return unsupportedExecutablePattern(pattern);

  std::optional<llvm::StringRef> kind = getPlanEmissionKind(op);
  if (!kind)
    return pattern.str() + " requires emission.kind='" + pattern.str() + "'";
  if (*kind != pattern)
    return pattern.str() + " requires emission.kind='" + pattern.str() +
           "', got '" + kind->str() + "'";

  return executableBuildContractForPattern(pattern);
}

void populateRPUPlanToExecutablePatterns(RewritePatternSet &patterns) {
  patterns.add<RPUPlanExecutableBuildPattern>(patterns.getContext());
}

} // namespace rpu
} // namespace mlir
