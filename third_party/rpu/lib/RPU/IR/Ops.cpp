#include "RPU/IR/Dialect.h"
#include "RPU/IR/ExecutableKind.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

using namespace mlir;
using namespace mlir::rpu::exec;

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symNameAttr;
  if (parser.parseSymbolName(symNameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument, 3> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs, /*enableNameShadowing=*/true))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void KernelOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << '(';
  Block &entry = getBody().front();
  llvm::interleaveComma(entry.getArguments(), printer, [&](BlockArgument arg) {
    printer.printRegionArgument(arg);
  });
  printer << ") ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                {SymbolTable::getSymbolAttrName()});
}

bool mlir::rpu::exec::isExecutableF16PointerType(Type type) {
  auto ptr = dyn_cast<triton::PointerType>(type);
  if (!ptr)
    return false;
  return ptr.getPointeeType().isF16();
}

FailureOr<KernelOp>
mlir::rpu::exec::getSingleExecutableKernelFromModule(ModuleOp module,
                                                     llvm::StringRef consumer) {
  SmallVector<KernelOp> kernels;
  module.walk([&](KernelOp op) { kernels.push_back(op); });
  if (kernels.empty()) {
    module.emitError() << consumer << " requires one rpu.kernel, found none";
    return failure();
  }
  if (kernels.size() > 1) {
    module.emitError() << consumer
                       << " requires one rpu.kernel, found multiple";
    return failure();
  }
  return kernels.front();
}

FailureOr<KernelOp>
mlir::rpu::exec::getSingleVerifiedExecutableKernelFromModule(
    ModuleOp module, llvm::StringRef consumer) {
  if (failed(verify(module))) {
    module.emitError() << consumer << " requires verified executable module";
    return failure();
  }
  return getSingleExecutableKernelFromModule(module, consumer);
}

std::optional<unsigned>
mlir::rpu::exec::getExecutableKernelArgumentIndex(KernelOp kernel,
                                                  Value value) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg || kernel.getBody().empty() ||
      arg.getOwner() != &kernel.getBody().front())
    return std::nullopt;
  return arg.getArgNumber();
}

FailureOr<SmallVector<unsigned, 4>>
mlir::rpu::exec::getExecutableKernelArgumentIndices(
    KernelOp kernel, ArrayRef<Value> values, llvm::StringRef diagnostic) {
  SmallVector<unsigned, 4> indices;
  indices.reserve(values.size());
  for (Value value : values) {
    std::optional<unsigned> index =
        getExecutableKernelArgumentIndex(kernel, value);
    if (!index) {
      kernel.emitError(diagnostic);
      return failure();
    }
    indices.push_back(*index);
  }
  return indices;
}

FailureOr<SmallVector<ExecutableKernelArgumentInfo, 4>>
mlir::rpu::exec::getExecutableKernelArguments(KernelOp kernel,
                                              llvm::StringRef diagnostic) {
  if (kernel.getBody().empty()) {
    kernel.emitError(diagnostic);
    return failure();
  }
  Block &block = kernel.getBody().front();
  SmallVector<ExecutableKernelArgumentInfo, 4> args;
  args.reserve(block.getNumArguments());
  for (unsigned i = 0, e = block.getNumArguments(); i < e; ++i) {
    Type type = block.getArgument(i).getType();
    if (!isExecutableF16PointerType(type)) {
      kernel.emitError(diagnostic);
      return failure();
    }
    args.push_back(ExecutableKernelArgumentInfo{
        i, std::string("arg") + std::to_string(i), block.getArgument(i), type});
  }
  return args;
}

FailureOr<ExecutableKernelContractInfo>
mlir::rpu::exec::getVerifiedExecutableKernelContract(KernelOp kernel,
                                                     llvm::StringRef consumer) {
  FailureOr<SmallVector<ExecutableKernelArgumentInfo, 4>> arguments =
      getExecutableKernelArguments(kernel, consumer);
  if (failed(arguments))
    return failure();

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();

  return ExecutableKernelContractInfo{kernel, kernel.getSymName().str(),
                                      kernel.getKind().str(), *arguments,
                                      *bodyOps};
}

FailureOr<ExecutableKernelContractInfo>
mlir::rpu::exec::getVerifiedExecutableKernelContractFromModule(
    ModuleOp module, llvm::StringRef consumer) {
  FailureOr<KernelOp> kernel =
      getSingleVerifiedExecutableKernelFromModule(module, consumer);
  if (failed(kernel))
    return failure();
  return getVerifiedExecutableKernelContract(*kernel, consumer);
}

FailureOr<ExecutableKernelScaffold>
mlir::rpu::exec::buildExecutableF16PointerKernelScaffold(
    OpBuilder &builder, ModuleOp module, Location loc,
    llvm::StringRef kernelName, llvm::StringRef kind, unsigned numArgs,
    llvm::StringRef consumer) {
  if (!module)
    return failure();
  if (kernelName.empty())
    return module.emitError() << consumer << " requires non-empty kernel name";
  if (kind.empty())
    return module.emitError() << consumer << " requires non-empty kernel kind";
  if (numArgs == 0)
    return module.emitError() << consumer << " requires at least one argument";
  if (!module.getBody())
    return module.emitError() << consumer << " requires module body";

  MLIRContext *context = module.getContext();
  context->getOrLoadDialect<triton::TritonDialect>();
  context->getOrLoadDialect<RPUDialect>();

  Type f16 = builder.getF16Type();
  Type ptrType = triton::PointerType::get(f16, 1);

  builder.setInsertionPointToEnd(module.getBody());
  KernelOp kernel = builder.create<KernelOp>(loc, kernelName, kind);

  Region &body = kernel.getBody();
  body.push_back(new Block());
  Block &entry = body.front();
  SmallVector<Type, 5> argTypes(numArgs, ptrType);
  SmallVector<Location, 5> argLocs(numArgs, loc);
  entry.addArguments(argTypes, argLocs);
  builder.setInsertionPointToStart(&entry);

  SmallVector<Value, 5> arguments;
  arguments.reserve(entry.getNumArguments());
  for (BlockArgument arg : entry.getArguments())
    arguments.push_back(arg);

  return ExecutableKernelScaffold{kernel, arguments};
}

SmallVector<Operation *>
mlir::rpu::exec::getExecutableKernelBodyOps(KernelOp kernel) {
  SmallVector<Operation *> ops;
  for (Operation &op : kernel.getBody().front())
    ops.push_back(&op);
  return ops;
}

FailureOr<SmallVector<Operation *>>
mlir::rpu::exec::getNonEmptyExecutableKernelBodyOps(KernelOp kernel,
                                                    llvm::StringRef consumer) {
  SmallVector<Operation *> ops = getExecutableKernelBodyOps(kernel);
  if (ops.empty()) {
    kernel.emitError() << consumer << " requires executable ops";
    return failure();
  }
  return ops;
}

namespace mlir {
namespace rpu {

FailureOr<RPUExecutableKernelSummary>
getRPUExecutableKernelSummaryFromKernelOp(exec::KernelOp op) {
  return RPUExecutableKernelSummary{op.getSymName().str(), op.getKind().str()};
}

FailureOr<RPUExecutableKernelSummary>
getRPUExecutableKernelSummaryFromModule(ModuleOp module) {
  FailureOr<exec::KernelOp> kernel =
      exec::getSingleVerifiedExecutableKernelFromModule(
          module, "RPU executable summary");
  if (failed(kernel))
    return failure();
  return getRPUExecutableKernelSummaryFromKernelOp(*kernel);
}

} // namespace rpu
} // namespace mlir

static FailureOr<TileType> requireTile(Operation *op, Type type) {
  auto tile = dyn_cast<TileType>(type);
  if (!tile)
    return op->emitError("requires !rpu.tile type"), failure();
  if (tile.getShape().empty())
    return op->emitError("tile shape must be non-empty"), failure();
  for (int64_t extent : tile.getShape()) {
    if (extent <= 0)
      return op->emitError("tile extents must be positive"), failure();
  }
  if (!tile.getElementType().isF16())
    return op->emitError("only f16 tile is supported"), failure();
  return tile;
}

static LogicalResult requireF16Scalar(Operation *op, Type type) {
  if (!type.isF16())
    return op->emitError("requires f16 scalar"), failure();
  return success();
}

static LogicalResult requireSameTile(Operation *op, Type lhsType,
                                     Type resultType) {
  FailureOr<TileType> lhs = requireTile(op, lhsType);
  FailureOr<TileType> result = requireTile(op, resultType);
  if (failed(lhs) || failed(result))
    return failure();
  if ((*lhs) != (*result))
    return op->emitError("input and result tile types must match");
  return success();
}

static bool hasTileShape(TileType tile, ArrayRef<int64_t> shape) {
  return llvm::equal(tile.getShape(), shape);
}

static SmallVector<int64_t, 2> matrixShape(int64_t rows, int64_t cols) {
  return SmallVector<int64_t, 2>{rows, cols};
}

template <typename OpTy>
static bool isOpAt(ArrayRef<Operation *> ops, size_t index) {
  return index < ops.size() && isa<OpTy>(ops[index]);
}

static bool hasAddBody(ArrayRef<Operation *> ops) {
  return ops.size() == 5 && isOpAt<LoadContigOp>(ops, 0) &&
         isOpAt<LoadContigOp>(ops, 1) && isOpAt<AddOp>(ops, 2) &&
         isOpAt<StoreContigOp>(ops, 3) && isOpAt<ReturnOp>(ops, 4);
}

static bool hasGemmBody(ArrayRef<Operation *> ops) {
  return ops.size() == 6 && isOpAt<LoadMatrixOp>(ops, 0) &&
         isOpAt<LoadMatrixOp>(ops, 1) && isOpAt<ZeroOp>(ops, 2) &&
         isOpAt<MmaOp>(ops, 3) && isOpAt<StoreMatrixOp>(ops, 4) &&
         isOpAt<ReturnOp>(ops, 5);
}

static bool hasLegalizableGemmBody(ArrayRef<Operation *> ops) {
  return ops.size() == 5 && isOpAt<LoadMatrixOp>(ops, 0) &&
         isOpAt<LoadMatrixOp>(ops, 1) &&
         isHighLevelLegalizableExecutableOp(ops[2]) && isOpAt<DotOp>(ops, 2) &&
         isOpAt<StoreMatrixOp>(ops, 3) && isOpAt<ReturnOp>(ops, 4);
}

static bool hasGemmExecutableBody(ArrayRef<Operation *> ops) {
  return hasGemmBody(ops) || hasLegalizableGemmBody(ops);
}

static bool hasSoftmaxBody(ArrayRef<Operation *> ops) {
  return ops.size() == 9 && isOpAt<LoadContigOp>(ops, 0) &&
         isOpAt<ReduceMaxAllOp>(ops, 1) && isOpAt<SubScalarOp>(ops, 2) &&
         isOpAt<ExpOp>(ops, 3) && isOpAt<ReduceSumAllOp>(ops, 4) &&
         isOpAt<ReciprocalOp>(ops, 5) && isOpAt<MulScalarOp>(ops, 6) &&
         isOpAt<StoreContigOp>(ops, 7) && isOpAt<ReturnOp>(ops, 8);
}

static bool hasSqrtBody(ArrayRef<Operation *> ops) {
  return ops.size() == 4 && isOpAt<LoadContigOp>(ops, 0) &&
         isOpAt<SqrtOp>(ops, 1) && isOpAt<StoreContigOp>(ops, 2) &&
         isOpAt<ReturnOp>(ops, 3);
}

static bool hasReduceSumAllBody(ArrayRef<Operation *> ops) {
  return ops.size() == 5 && isOpAt<LoadContigOp>(ops, 0) &&
         isOpAt<ReduceSumAllOp>(ops, 1) && isOpAt<FullOp>(ops, 2) &&
         isOpAt<StoreContigOp>(ops, 3) && isOpAt<ReturnOp>(ops, 4);
}

static bool hasReluBody(ArrayRef<Operation *> ops) {
  return ops.size() == 4 && isOpAt<LoadContigOp>(ops, 0) &&
         isOpAt<ReluOp>(ops, 1) && isOpAt<StoreContigOp>(ops, 2) &&
         isOpAt<ReturnOp>(ops, 3);
}

static bool hasMaximumBody(ArrayRef<Operation *> ops) {
  return ops.size() == 5 && isOpAt<LoadContigOp>(ops, 0) &&
         isOpAt<LoadContigOp>(ops, 1) && isOpAt<MaxOp>(ops, 2) &&
         isOpAt<StoreContigOp>(ops, 3) && isOpAt<ReturnOp>(ops, 4);
}

static bool hasReduceSumAxisBody(ArrayRef<Operation *> ops) {
  return ops.size() == 4 && isOpAt<LoadMatrixOp>(ops, 0) &&
         isOpAt<ReduceSumAxisOp>(ops, 1) && isOpAt<StoreContigOp>(ops, 2) &&
         isOpAt<ReturnOp>(ops, 3);
}

static bool hasBroadcastAddBody(ArrayRef<Operation *> ops) {
  return ops.size() == 5 && isOpAt<LoadMatrixOp>(ops, 0) &&
         isOpAt<LoadContigOp>(ops, 1) && isOpAt<BroadcastAddOp>(ops, 2) &&
         isOpAt<StoreMatrixOp>(ops, 3) && isOpAt<ReturnOp>(ops, 4);
}

static bool hasLegalizableSoftmaxBody(ArrayRef<Operation *> ops) {
  return ops.size() == 4 && isOpAt<LoadContigOp>(ops, 0) &&
         isHighLevelLegalizableExecutableOp(ops[1]) &&
         isOpAt<SoftmaxOp>(ops, 1) && isOpAt<StoreContigOp>(ops, 2) &&
         isOpAt<ReturnOp>(ops, 3);
}

static bool hasSoftmaxExecutableBody(ArrayRef<Operation *> ops) {
  return hasSoftmaxBody(ops) || hasLegalizableSoftmaxBody(ops);
}

static std::optional<int64_t> getConvKxKKernelSizeFromOpCount(size_t opCount) {
  if (opCount < 6 || (opCount - 3) % 3 != 0)
    return std::nullopt;
  int64_t windows = static_cast<int64_t>((opCount - 3) / 3);
  for (int64_t kernelSize : {3, 5, 7, 9}) {
    if (windows == kernelSize * kernelSize)
      return kernelSize;
  }
  return std::nullopt;
}

static std::optional<int64_t>
getConvKxKDotSeedKernelSizeFromOpCount(size_t opCount) {
  if (opCount < 5 || (opCount - 2) % 3 != 0)
    return std::nullopt;
  int64_t windows = static_cast<int64_t>((opCount - 2) / 3);
  for (int64_t kernelSize : {3, 5, 7, 9}) {
    if (windows == kernelSize * kernelSize)
      return kernelSize;
  }
  return std::nullopt;
}

enum class ConvKxKBodyLayout {
  None,
  CanonicalPrimitive,
  LegalizableDotSeed,
  ExpandedDotSeedPrimitive,
};

static bool hasConvKxKBody(ArrayRef<Operation *> ops) {
  if (!isOpAt<ZeroOp>(ops, 0))
    return false;
  std::optional<int64_t> kernelSize =
      getConvKxKKernelSizeFromOpCount(ops.size());
  if (!kernelSize)
    return false;
  int64_t windows = *kernelSize * *kernelSize;
  for (int64_t i = 0; i < windows; ++i) {
    size_t base = static_cast<size_t>(1 + i * 3);
    if (!isOpAt<LoadMatrixOp>(ops, base) ||
        !isOpAt<LoadMatrixOp>(ops, base + 1) || !isOpAt<MmaOp>(ops, base + 2))
      return false;
  }
  return isOpAt<StoreMatrixOp>(ops, ops.size() - 2) &&
         isOpAt<ReturnOp>(ops, ops.size() - 1);
}

static bool hasLegalizableConvKxKDotSeedBody(ArrayRef<Operation *> ops) {
  std::optional<int64_t> kernelSize =
      getConvKxKDotSeedKernelSizeFromOpCount(ops.size());
  if (!kernelSize || !isOpAt<LoadMatrixOp>(ops, 0) ||
      !isOpAt<LoadMatrixOp>(ops, 1) ||
      !isHighLevelLegalizableExecutableOp(ops[2]) || !isOpAt<DotOp>(ops, 2))
    return false;

  int64_t windows = *kernelSize * *kernelSize;
  for (int64_t i = 1; i < windows; ++i) {
    size_t base = static_cast<size_t>(3 + (i - 1) * 3);
    if (!isOpAt<LoadMatrixOp>(ops, base) ||
        !isOpAt<LoadMatrixOp>(ops, base + 1) || !isOpAt<MmaOp>(ops, base + 2))
      return false;
  }

  size_t storeIndex = static_cast<size_t>(3 * windows);
  return isOpAt<StoreMatrixOp>(ops, storeIndex) &&
         isOpAt<ReturnOp>(ops, storeIndex + 1);
}

static bool hasExpandedConvKxKDotSeedBody(ArrayRef<Operation *> ops) {
  std::optional<int64_t> kernelSize =
      getConvKxKKernelSizeFromOpCount(ops.size());
  if (!kernelSize || !isOpAt<LoadMatrixOp>(ops, 0) ||
      !isOpAt<LoadMatrixOp>(ops, 1) || !isOpAt<ZeroOp>(ops, 2) ||
      !isOpAt<MmaOp>(ops, 3))
    return false;

  int64_t windows = *kernelSize * *kernelSize;
  for (int64_t i = 1; i < windows; ++i) {
    size_t base = static_cast<size_t>(4 + (i - 1) * 3);
    if (!isOpAt<LoadMatrixOp>(ops, base) ||
        !isOpAt<LoadMatrixOp>(ops, base + 1) || !isOpAt<MmaOp>(ops, base + 2))
      return false;
  }

  size_t storeIndex = static_cast<size_t>(3 * windows + 1);
  return isOpAt<StoreMatrixOp>(ops, storeIndex) &&
         isOpAt<ReturnOp>(ops, storeIndex + 1);
}

static ConvKxKBodyLayout getConvKxKBodyLayout(ArrayRef<Operation *> ops) {
  if (hasConvKxKBody(ops))
    return ConvKxKBodyLayout::CanonicalPrimitive;
  if (hasLegalizableConvKxKDotSeedBody(ops))
    return ConvKxKBodyLayout::LegalizableDotSeed;
  if (hasExpandedConvKxKDotSeedBody(ops))
    return ConvKxKBodyLayout::ExpandedDotSeedPrimitive;
  return ConvKxKBodyLayout::None;
}

static bool hasConvKxKExecutableBody(ArrayRef<Operation *> ops) {
  return getConvKxKBodyLayout(ops) != ConvKxKBodyLayout::None;
}

static bool hasResNetBlockBody(ArrayRef<Operation *> ops) {
  if (ops.size() == 14)
    return isOpAt<LoadMatrixOp>(ops, 0) && isOpAt<LoadMatrixOp>(ops, 1) &&
           isOpAt<ZeroOp>(ops, 2) && isOpAt<MmaOp>(ops, 3) &&
           isOpAt<ZeroOp>(ops, 4) && isOpAt<MaxOp>(ops, 5) &&
           isOpAt<LoadMatrixOp>(ops, 6) && isOpAt<ZeroOp>(ops, 7) &&
           isOpAt<MmaOp>(ops, 8) && isOpAt<AddOp>(ops, 9) &&
           isOpAt<ZeroOp>(ops, 10) && isOpAt<MaxOp>(ops, 11) &&
           isOpAt<StoreMatrixOp>(ops, 12) && isOpAt<ReturnOp>(ops, 13);
  return ops.size() == 21 && isOpAt<LoadMatrixOp>(ops, 0) &&
         isOpAt<LoadMatrixOp>(ops, 1) && isOpAt<ZeroOp>(ops, 2) &&
         isOpAt<MmaOp>(ops, 3) && isOpAt<ZeroOp>(ops, 4) &&
         isOpAt<MaxOp>(ops, 5) && isOpAt<LoadMatrixOp>(ops, 6) &&
         isOpAt<ZeroOp>(ops, 7) && isOpAt<MmaOp>(ops, 8) &&
         isOpAt<ZeroOp>(ops, 9) && isOpAt<MaxOp>(ops, 10) &&
         isOpAt<LoadMatrixOp>(ops, 11) && isOpAt<ZeroOp>(ops, 12) &&
         isOpAt<MmaOp>(ops, 13) && isOpAt<LoadMatrixOp>(ops, 14) &&
         isOpAt<MmaOp>(ops, 15) && isOpAt<AddOp>(ops, 16) &&
         isOpAt<ZeroOp>(ops, 17) && isOpAt<MaxOp>(ops, 18) &&
         isOpAt<StoreMatrixOp>(ops, 19) && isOpAt<ReturnOp>(ops, 20);
}

static bool hasLegalizableResNetBlockBody(ArrayRef<Operation *> ops) {
  if (ops.size() == 12)
    return isOpAt<LoadMatrixOp>(ops, 0) && isOpAt<LoadMatrixOp>(ops, 1) &&
           isHighLevelLegalizableExecutableOp(ops[2]) &&
           isOpAt<DotOp>(ops, 2) && isOpAt<ZeroOp>(ops, 3) &&
           isOpAt<MaxOp>(ops, 4) && isOpAt<LoadMatrixOp>(ops, 5) &&
           isHighLevelLegalizableExecutableOp(ops[6]) &&
           isOpAt<DotOp>(ops, 6) && isOpAt<AddOp>(ops, 7) &&
           isOpAt<ZeroOp>(ops, 8) && isOpAt<MaxOp>(ops, 9) &&
           isOpAt<StoreMatrixOp>(ops, 10) && isOpAt<ReturnOp>(ops, 11);
  return ops.size() == 18 && isOpAt<LoadMatrixOp>(ops, 0) &&
         isOpAt<LoadMatrixOp>(ops, 1) &&
         isHighLevelLegalizableExecutableOp(ops[2]) && isOpAt<DotOp>(ops, 2) &&
         isOpAt<ZeroOp>(ops, 3) && isOpAt<MaxOp>(ops, 4) &&
         isOpAt<LoadMatrixOp>(ops, 5) &&
         isHighLevelLegalizableExecutableOp(ops[6]) && isOpAt<DotOp>(ops, 6) &&
         isOpAt<ZeroOp>(ops, 7) && isOpAt<MaxOp>(ops, 8) &&
         isOpAt<LoadMatrixOp>(ops, 9) &&
         isHighLevelLegalizableExecutableOp(ops[10]) &&
         isOpAt<DotOp>(ops, 10) && isOpAt<LoadMatrixOp>(ops, 11) &&
         isOpAt<MmaOp>(ops, 12) && isOpAt<AddOp>(ops, 13) &&
         isOpAt<ZeroOp>(ops, 14) && isOpAt<MaxOp>(ops, 15) &&
         isOpAt<StoreMatrixOp>(ops, 16) && isOpAt<ReturnOp>(ops, 17);
}

static bool hasResNetBlockExecutableBody(ArrayRef<Operation *> ops) {
  return hasResNetBlockBody(ops) || hasLegalizableResNetBlockBody(ops);
}

static bool hasResNet50BottleneckBody(ArrayRef<Operation *> ops) {
  if (ops.size() == 76) {
    if (!isOpAt<LoadMatrixOp>(ops, 0) || !isOpAt<LoadMatrixOp>(ops, 1) ||
        !isOpAt<ZeroOp>(ops, 2))
      return false;
    for (size_t i = 0; i < 9; ++i) {
      size_t base = 3 + i * 7;
      if (!isOpAt<LoadMatrixOp>(ops, base) || !isOpAt<ZeroOp>(ops, base + 1) ||
          !isOpAt<MmaOp>(ops, base + 2) || !isOpAt<ZeroOp>(ops, base + 3) ||
          !isOpAt<MaxOp>(ops, base + 4) ||
          !isOpAt<LoadMatrixOp>(ops, base + 5) || !isOpAt<MmaOp>(ops, base + 6))
        return false;
    }
    size_t tail = 66;
    return isOpAt<ZeroOp>(ops, tail) && isOpAt<MaxOp>(ops, tail + 1) &&
           isOpAt<LoadMatrixOp>(ops, tail + 2) &&
           isOpAt<ZeroOp>(ops, tail + 3) && isOpAt<MmaOp>(ops, tail + 4) &&
           isOpAt<AddOp>(ops, tail + 5) && isOpAt<ZeroOp>(ops, tail + 6) &&
           isOpAt<MaxOp>(ops, tail + 7) &&
           isOpAt<StoreMatrixOp>(ops, tail + 8) &&
           isOpAt<ReturnOp>(ops, tail + 9);
  }

  if (ops.size() != 172 || !isOpAt<LoadMatrixOp>(ops, 0) ||
      !isOpAt<LoadMatrixOp>(ops, 1) || !isOpAt<LoadMatrixOp>(ops, 2) ||
      !isOpAt<ZeroOp>(ops, 3) || !isOpAt<ZeroOp>(ops, 4))
    return false;
  for (size_t i = 0; i < 9; ++i) {
    size_t base = 5 + i * 17;
    if (!isOpAt<LoadMatrixOp>(ops, base) || !isOpAt<ZeroOp>(ops, base + 1) ||
        !isOpAt<MmaOp>(ops, base + 2) || !isOpAt<ZeroOp>(ops, base + 3) ||
        !isOpAt<MaxOp>(ops, base + 4) || !isOpAt<ZeroOp>(ops, base + 5) ||
        !isOpAt<MmaOp>(ops, base + 6) || !isOpAt<ZeroOp>(ops, base + 7) ||
        !isOpAt<MaxOp>(ops, base + 8) || !isOpAt<LoadMatrixOp>(ops, base + 9) ||
        !isOpAt<MmaOp>(ops, base + 10) ||
        !isOpAt<LoadMatrixOp>(ops, base + 11) ||
        !isOpAt<MmaOp>(ops, base + 12) ||
        !isOpAt<LoadMatrixOp>(ops, base + 13) ||
        !isOpAt<MmaOp>(ops, base + 14) ||
        !isOpAt<LoadMatrixOp>(ops, base + 15) || !isOpAt<MmaOp>(ops, base + 16))
      return false;
  }
  size_t tail = 158;
  return isOpAt<ZeroOp>(ops, tail) && isOpAt<MaxOp>(ops, tail + 1) &&
         isOpAt<ZeroOp>(ops, tail + 2) && isOpAt<MaxOp>(ops, tail + 3) &&
         isOpAt<LoadMatrixOp>(ops, tail + 4) && isOpAt<ZeroOp>(ops, tail + 5) &&
         isOpAt<MmaOp>(ops, tail + 6) && isOpAt<LoadMatrixOp>(ops, tail + 7) &&
         isOpAt<MmaOp>(ops, tail + 8) && isOpAt<AddOp>(ops, tail + 9) &&
         isOpAt<ZeroOp>(ops, tail + 10) && isOpAt<MaxOp>(ops, tail + 11) &&
         isOpAt<StoreMatrixOp>(ops, tail + 12) &&
         isOpAt<ReturnOp>(ops, tail + 13);
}

static bool hasLegalizableResNet50BottleneckBody(ArrayRef<Operation *> ops) {
  if (ops.size() == 66) {
    if (!isOpAt<LoadMatrixOp>(ops, 0) || !isOpAt<LoadMatrixOp>(ops, 1) ||
        !isOpAt<ZeroOp>(ops, 2))
      return false;
    for (size_t i = 0; i < 9; ++i) {
      size_t base = 3 + i * 6;
      if (!isOpAt<LoadMatrixOp>(ops, base) ||
          !isHighLevelLegalizableExecutableOp(ops[base + 1]) ||
          !isOpAt<DotOp>(ops, base + 1) || !isOpAt<ZeroOp>(ops, base + 2) ||
          !isOpAt<MaxOp>(ops, base + 3) ||
          !isOpAt<LoadMatrixOp>(ops, base + 4) || !isOpAt<MmaOp>(ops, base + 5))
        return false;
    }
    size_t tail = 57;
    return isOpAt<ZeroOp>(ops, tail) && isOpAt<MaxOp>(ops, tail + 1) &&
           isOpAt<LoadMatrixOp>(ops, tail + 2) &&
           isHighLevelLegalizableExecutableOp(ops[tail + 3]) &&
           isOpAt<DotOp>(ops, tail + 3) && isOpAt<AddOp>(ops, tail + 4) &&
           isOpAt<ZeroOp>(ops, tail + 5) && isOpAt<MaxOp>(ops, tail + 6) &&
           isOpAt<StoreMatrixOp>(ops, tail + 7) &&
           isOpAt<ReturnOp>(ops, tail + 8);
  }

  if (ops.size() != 153 || !isOpAt<LoadMatrixOp>(ops, 0) ||
      !isOpAt<LoadMatrixOp>(ops, 1) || !isOpAt<LoadMatrixOp>(ops, 2) ||
      !isOpAt<ZeroOp>(ops, 3) || !isOpAt<ZeroOp>(ops, 4))
    return false;
  for (size_t i = 0; i < 9; ++i) {
    size_t base = 5 + i * 15;
    if (!isOpAt<LoadMatrixOp>(ops, base) ||
        !isHighLevelLegalizableExecutableOp(ops[base + 1]) ||
        !isOpAt<DotOp>(ops, base + 1) || !isOpAt<ZeroOp>(ops, base + 2) ||
        !isOpAt<MaxOp>(ops, base + 3) ||
        !isHighLevelLegalizableExecutableOp(ops[base + 4]) ||
        !isOpAt<DotOp>(ops, base + 4) || !isOpAt<ZeroOp>(ops, base + 5) ||
        !isOpAt<MaxOp>(ops, base + 6) || !isOpAt<LoadMatrixOp>(ops, base + 7) ||
        !isOpAt<MmaOp>(ops, base + 8) || !isOpAt<LoadMatrixOp>(ops, base + 9) ||
        !isOpAt<MmaOp>(ops, base + 10) ||
        !isOpAt<LoadMatrixOp>(ops, base + 11) ||
        !isOpAt<MmaOp>(ops, base + 12) ||
        !isOpAt<LoadMatrixOp>(ops, base + 13) || !isOpAt<MmaOp>(ops, base + 14))
      return false;
  }
  size_t tail = 140;
  return isOpAt<ZeroOp>(ops, tail) && isOpAt<MaxOp>(ops, tail + 1) &&
         isOpAt<ZeroOp>(ops, tail + 2) && isOpAt<MaxOp>(ops, tail + 3) &&
         isOpAt<LoadMatrixOp>(ops, tail + 4) &&
         isHighLevelLegalizableExecutableOp(ops[tail + 5]) &&
         isOpAt<DotOp>(ops, tail + 5) && isOpAt<LoadMatrixOp>(ops, tail + 6) &&
         isOpAt<MmaOp>(ops, tail + 7) && isOpAt<AddOp>(ops, tail + 8) &&
         isOpAt<ZeroOp>(ops, tail + 9) && isOpAt<MaxOp>(ops, tail + 10) &&
         isOpAt<StoreMatrixOp>(ops, tail + 11) &&
         isOpAt<ReturnOp>(ops, tail + 12);
}

static bool hasResNet50BottleneckExecutableBody(ArrayRef<Operation *> ops) {
  return hasResNet50BottleneckBody(ops) ||
         hasLegalizableResNet50BottleneckBody(ops);
}

static LogicalResult verifyAddDataflow(KernelOp op, Block &block,
                                       ArrayRef<Operation *> ops) {
  auto lhsLoad = cast<LoadContigOp>(ops[0]);
  auto rhsLoad = cast<LoadContigOp>(ops[1]);
  auto add = cast<AddOp>(ops[2]);
  auto store = cast<StoreContigOp>(ops[3]);
  if (add.getLhs() != lhsLoad.getResult() ||
      add.getRhs() != rhsLoad.getResult() ||
      store.getValue() != add.getResult() ||
      lhsLoad.getPtr() != block.getArgument(1) ||
      rhsLoad.getPtr() != block.getArgument(2) ||
      store.getPtr() != block.getArgument(0))
    return op.emitOpError("kind add requires canonical add dataflow");

  auto maskedAttr = store->getAttrOfType<BoolAttr>("masked");
  auto logicalNAttr = store->getAttrOfType<IntegerAttr>("logical_n");
  auto blockNAttr = store->getAttrOfType<IntegerAttr>("block_n");
  if (!maskedAttr || !maskedAttr.getValue()) {
    if (logicalNAttr || blockNAttr)
      return op.emitOpError(
          "kind add unmasked store must not carry logical_n or block_n");
    return success();
  }
  if (!logicalNAttr || !blockNAttr)
    return op.emitOpError(
        "kind add masked store requires logical_n and block_n");
  int64_t logicalN = logicalNAttr.getInt();
  int64_t blockN = blockNAttr.getInt();
  if (blockN != store.getN())
    return op.emitOpError("kind add masked block_n must equal store n");
  if (logicalN <= 0 || logicalN > blockN)
    return op.emitOpError("kind add masked logical_n must be in (0, block_n]");
  return success();
}

FailureOr<ExecutableAddOpSequenceInfo>
mlir::rpu::exec::getExecutableAddOpSequenceInfo(KernelOp kernel,
                                                ArrayRef<Operation *> ops,
                                                llvm::StringRef consumer) {
  if (!hasAddBody(ops)) {
    kernel.emitError() << consumer << " requires canonical add executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyAddDataflow(kernel, block, ops)))
    return failure();

  auto lhsLoad = cast<LoadContigOp>(ops[0]);
  auto rhsLoad = cast<LoadContigOp>(ops[1]);
  auto add = cast<AddOp>(ops[2]);
  auto store = cast<StoreContigOp>(ops[3]);

  int64_t n = lhsLoad.getN();
  bool masked = false;
  int64_t logicalN = n;
  if (auto maskedAttr = store->getAttrOfType<BoolAttr>("masked")) {
    masked = maskedAttr.getValue();
    if (masked)
      logicalN = store->getAttrOfType<IntegerAttr>("logical_n").getInt();
  }

  return ExecutableAddOpSequenceInfo{lhsLoad, rhsLoad,  add,   store,
                                     n,       logicalN, masked};
}

static LogicalResult verifyGemmDataflow(KernelOp op, Block &block,
                                        ArrayRef<Operation *> ops) {
  if (hasLegalizableGemmBody(ops)) {
    auto lhsLoad = cast<LoadMatrixOp>(ops[0]);
    auto rhsLoad = cast<LoadMatrixOp>(ops[1]);
    auto dot = cast<DotOp>(ops[2]);
    auto store = cast<StoreMatrixOp>(ops[3]);
    if (dot.getLhs() != lhsLoad.getResult() ||
        dot.getRhs() != rhsLoad.getResult() ||
        store.getValue() != dot.getResult() ||
        lhsLoad.getPtr() != block.getArgument(1) ||
        rhsLoad.getPtr() != block.getArgument(2) ||
        store.getPtr() != block.getArgument(0))
      return op.emitOpError(
          "kind gemm requires canonical high-level dot dataflow");
    return success();
  }

  auto lhsLoad = cast<LoadMatrixOp>(ops[0]);
  auto rhsLoad = cast<LoadMatrixOp>(ops[1]);
  auto zero = cast<ZeroOp>(ops[2]);
  auto mma = cast<MmaOp>(ops[3]);
  auto store = cast<StoreMatrixOp>(ops[4]);
  if (mma.getLhs() != lhsLoad.getResult() ||
      mma.getRhs() != rhsLoad.getResult() || mma.getAcc() != zero.getResult() ||
      store.getValue() != mma.getResult() ||
      lhsLoad.getPtr() != block.getArgument(1) ||
      rhsLoad.getPtr() != block.getArgument(2) ||
      store.getPtr() != block.getArgument(0))
    return op.emitOpError("kind gemm requires canonical gemm dataflow");
  return success();
}

FailureOr<ExecutableGemmOpSequenceInfo>
mlir::rpu::exec::getExecutableGemmOpSequenceInfo(KernelOp kernel,
                                                 ArrayRef<Operation *> ops,
                                                 llvm::StringRef consumer) {
  if (!hasGemmBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical gemm executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyGemmDataflow(kernel, block, ops)))
    return failure();

  auto lhsLoad = cast<LoadMatrixOp>(ops[0]);
  auto rhsLoad = cast<LoadMatrixOp>(ops[1]);
  auto zero = cast<ZeroOp>(ops[2]);
  auto mma = cast<MmaOp>(ops[3]);
  auto store = cast<StoreMatrixOp>(ops[4]);

  int64_t m = lhsLoad.getRows();
  int64_t k = lhsLoad.getCols();
  int64_t n = rhsLoad.getCols();
  return ExecutableGemmOpSequenceInfo{lhsLoad, rhsLoad, zero, mma,
                                      store,   m,       n,    k};
}

static LogicalResult verifySoftmaxDataflow(KernelOp op, Block &block,
                                           ArrayRef<Operation *> ops) {
  if (hasLegalizableSoftmaxBody(ops)) {
    auto load = cast<LoadContigOp>(ops[0]);
    auto softmax = cast<SoftmaxOp>(ops[1]);
    auto store = cast<StoreContigOp>(ops[2]);
    if (load.getPtr() != block.getArgument(1) ||
        store.getPtr() != block.getArgument(0) ||
        softmax.getInput() != load.getResult() ||
        store.getValue() != softmax.getResult())
      return op.emitOpError(
          "kind softmax requires canonical high-level softmax dataflow");
    return success();
  }

  auto load = cast<LoadContigOp>(ops[0]);
  auto reduceMax = cast<ReduceMaxAllOp>(ops[1]);
  auto sub = cast<SubScalarOp>(ops[2]);
  auto exp = cast<ExpOp>(ops[3]);
  auto reduceSum = cast<ReduceSumAllOp>(ops[4]);
  auto reciprocal = cast<ReciprocalOp>(ops[5]);
  auto mul = cast<MulScalarOp>(ops[6]);
  auto store = cast<StoreContigOp>(ops[7]);
  if (load.getPtr() != block.getArgument(1) ||
      store.getPtr() != block.getArgument(0) ||
      reduceMax.getInput() != load.getResult() ||
      sub.getLhs() != load.getResult() ||
      sub.getRhs() != reduceMax.getResult() ||
      exp.getInput() != sub.getResult() ||
      reduceSum.getInput() != exp.getResult() ||
      reciprocal.getInput() != reduceSum.getResult() ||
      mul.getLhs() != exp.getResult() ||
      mul.getRhs() != reciprocal.getResult() ||
      store.getValue() != mul.getResult())
    return op.emitOpError("kind softmax requires canonical softmax dataflow");
  return success();
}

FailureOr<ExecutableSoftmaxOpSequenceInfo>
mlir::rpu::exec::getExecutableSoftmaxOpSequenceInfo(KernelOp kernel,
                                                    ArrayRef<Operation *> ops,
                                                    llvm::StringRef consumer) {
  if (!hasSoftmaxBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical softmax executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifySoftmaxDataflow(kernel, block, ops)))
    return failure();

  auto load = cast<LoadContigOp>(ops[0]);
  auto reduceMax = cast<ReduceMaxAllOp>(ops[1]);
  auto sub = cast<SubScalarOp>(ops[2]);
  auto exp = cast<ExpOp>(ops[3]);
  auto reduceSum = cast<ReduceSumAllOp>(ops[4]);
  auto reciprocal = cast<ReciprocalOp>(ops[5]);
  auto mul = cast<MulScalarOp>(ops[6]);
  auto store = cast<StoreContigOp>(ops[7]);
  return ExecutableSoftmaxOpSequenceInfo{load, reduceMax, sub,
                                         exp,  reduceSum, reciprocal,
                                         mul,  store,     load.getN()};
}

static LogicalResult verifySqrtDataflow(KernelOp op, Block &block,
                                        ArrayRef<Operation *> ops) {
  auto load = cast<LoadContigOp>(ops[0]);
  auto sqrt = cast<SqrtOp>(ops[1]);
  auto store = cast<StoreContigOp>(ops[2]);
  if (load.getPtr() != block.getArgument(1) ||
      store.getPtr() != block.getArgument(0) ||
      sqrt.getInput() != load.getResult() ||
      store.getValue() != sqrt.getResult())
    return op.emitOpError("kind sqrt requires canonical sqrt dataflow");
  return success();
}

static LogicalResult verifyReduceSumAllDataflow(KernelOp op, Block &block,
                                                ArrayRef<Operation *> ops) {
  auto load = cast<LoadContigOp>(ops[0]);
  auto reduceSum = cast<ReduceSumAllOp>(ops[1]);
  auto full = cast<FullOp>(ops[2]);
  auto store = cast<StoreContigOp>(ops[3]);
  if (load.getPtr() != block.getArgument(1) ||
      store.getPtr() != block.getArgument(0) ||
      reduceSum.getInput() != load.getResult() ||
      full.getValue() != reduceSum.getResult() ||
      store.getValue() != full.getResult())
    return op.emitOpError(
        "kind reduce_sum_all requires canonical reduce_sum_all dataflow");
  return success();
}

FailureOr<ExecutableSqrtOpSequenceInfo>
mlir::rpu::exec::getExecutableSqrtOpSequenceInfo(KernelOp kernel,
                                                 ArrayRef<Operation *> ops,
                                                 llvm::StringRef consumer) {
  if (!hasSqrtBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical sqrt executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifySqrtDataflow(kernel, block, ops)))
    return failure();
  auto load = cast<LoadContigOp>(ops[0]);
  auto sqrt = cast<SqrtOp>(ops[1]);
  auto store = cast<StoreContigOp>(ops[2]);
  return ExecutableSqrtOpSequenceInfo{load, sqrt, store, load.getN()};
}

FailureOr<ExecutableReduceSumAllOpSequenceInfo>
mlir::rpu::exec::getExecutableReduceSumAllOpSequenceInfo(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer) {
  if (!hasReduceSumAllBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical reduce_sum_all executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyReduceSumAllDataflow(kernel, block, ops)))
    return failure();
  auto load = cast<LoadContigOp>(ops[0]);
  auto reduceSum = cast<ReduceSumAllOp>(ops[1]);
  auto full = cast<FullOp>(ops[2]);
  auto store = cast<StoreContigOp>(ops[3]);
  return ExecutableReduceSumAllOpSequenceInfo{load, reduceSum, full, store,
                                              load.getN()};
}

static LogicalResult verifyReluDataflow(KernelOp op, Block &block,
                                        ArrayRef<Operation *> ops) {
  auto load = cast<LoadContigOp>(ops[0]);
  auto relu = cast<ReluOp>(ops[1]);
  auto store = cast<StoreContigOp>(ops[2]);
  if (load.getPtr() != block.getArgument(1) ||
      store.getPtr() != block.getArgument(0) ||
      relu.getInput() != load.getResult() ||
      store.getValue() != relu.getResult())
    return op.emitOpError("kind relu requires canonical relu dataflow");
  return success();
}

static LogicalResult verifyMaximumDataflow(KernelOp op, Block &block,
                                           ArrayRef<Operation *> ops) {
  auto lhsLoad = cast<LoadContigOp>(ops[0]);
  auto rhsLoad = cast<LoadContigOp>(ops[1]);
  auto max = cast<MaxOp>(ops[2]);
  auto store = cast<StoreContigOp>(ops[3]);
  if (lhsLoad.getPtr() != block.getArgument(1) ||
      rhsLoad.getPtr() != block.getArgument(2) ||
      store.getPtr() != block.getArgument(0) ||
      max.getLhs() != lhsLoad.getResult() ||
      max.getRhs() != rhsLoad.getResult() ||
      store.getValue() != max.getResult())
    return op.emitOpError("kind maximum requires canonical maximum dataflow");
  return success();
}

static LogicalResult verifyReduceSumAxisDataflow(KernelOp op, Block &block,
                                                 ArrayRef<Operation *> ops) {
  auto load = cast<LoadMatrixOp>(ops[0]);
  auto reduce = cast<ReduceSumAxisOp>(ops[1]);
  auto store = cast<StoreContigOp>(ops[2]);
  if (load.getPtr() != block.getArgument(1) ||
      store.getPtr() != block.getArgument(0) ||
      reduce.getInput() != load.getResult() ||
      store.getValue() != reduce.getResult())
    return op.emitOpError(
        "kind reduce_sum_axis requires canonical reduce_sum_axis dataflow");
  return success();
}

static LogicalResult verifyBroadcastAddDataflow(KernelOp op, Block &block,
                                                ArrayRef<Operation *> ops) {
  auto lhsLoad = cast<LoadMatrixOp>(ops[0]);
  auto rhsLoad = cast<LoadContigOp>(ops[1]);
  auto add = cast<BroadcastAddOp>(ops[2]);
  auto store = cast<StoreMatrixOp>(ops[3]);
  if (lhsLoad.getPtr() != block.getArgument(1) ||
      rhsLoad.getPtr() != block.getArgument(2) ||
      store.getPtr() != block.getArgument(0) ||
      add.getLhs() != lhsLoad.getResult() ||
      add.getRhs() != rhsLoad.getResult() ||
      store.getValue() != add.getResult())
    return op.emitOpError(
        "kind broadcast_add requires canonical broadcast_add dataflow");
  return success();
}

FailureOr<ExecutableReluOpSequenceInfo>
mlir::rpu::exec::getExecutableReluOpSequenceInfo(KernelOp kernel,
                                                 ArrayRef<Operation *> ops,
                                                 llvm::StringRef consumer) {
  if (!hasReluBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical relu executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyReluDataflow(kernel, block, ops)))
    return failure();
  auto load = cast<LoadContigOp>(ops[0]);
  auto relu = cast<ReluOp>(ops[1]);
  auto store = cast<StoreContigOp>(ops[2]);
  return ExecutableReluOpSequenceInfo{load, relu, store, load.getN()};
}

FailureOr<ExecutableMaximumOpSequenceInfo>
mlir::rpu::exec::getExecutableMaximumOpSequenceInfo(KernelOp kernel,
                                                    ArrayRef<Operation *> ops,
                                                    llvm::StringRef consumer) {
  if (!hasMaximumBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical maximum executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyMaximumDataflow(kernel, block, ops)))
    return failure();
  auto lhsLoad = cast<LoadContigOp>(ops[0]);
  auto rhsLoad = cast<LoadContigOp>(ops[1]);
  auto max = cast<MaxOp>(ops[2]);
  auto store = cast<StoreContigOp>(ops[3]);
  return ExecutableMaximumOpSequenceInfo{lhsLoad, rhsLoad, max, store,
                                         lhsLoad.getN()};
}

FailureOr<ExecutableReduceSumAxisOpSequenceInfo>
mlir::rpu::exec::getExecutableReduceSumAxisOpSequenceInfo(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer) {
  if (!hasReduceSumAxisBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical reduce_sum_axis executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyReduceSumAxisDataflow(kernel, block, ops)))
    return failure();
  auto load = cast<LoadMatrixOp>(ops[0]);
  auto reduce = cast<ReduceSumAxisOp>(ops[1]);
  auto store = cast<StoreContigOp>(ops[2]);
  return ExecutableReduceSumAxisOpSequenceInfo{
      load,           reduce,         store,
      load.getRows(), load.getCols(), static_cast<int64_t>(reduce.getAxis())};
}

FailureOr<ExecutableBroadcastAddOpSequenceInfo>
mlir::rpu::exec::getExecutableBroadcastAddOpSequenceInfo(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer) {
  if (!hasBroadcastAddBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical broadcast_add executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyBroadcastAddDataflow(kernel, block, ops)))
    return failure();
  auto lhsLoad = cast<LoadMatrixOp>(ops[0]);
  auto rhsLoad = cast<LoadContigOp>(ops[1]);
  auto add = cast<BroadcastAddOp>(ops[2]);
  auto store = cast<StoreMatrixOp>(ops[3]);
  return ExecutableBroadcastAddOpSequenceInfo{
      lhsLoad, rhsLoad, add, store, lhsLoad.getRows(), lhsLoad.getCols()};
}

static int64_t getI32AttrOrZero(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return 0;
  return attr.getInt();
}

static bool hasInvalidOptionalI32Attr(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return false;
  return !attr.getType().isInteger(32) || attr.getInt() < 0;
}

static bool hasInvalidMatrixOffsetAttr(Operation *op) {
  return hasInvalidOptionalI32Attr(op, "row_offset") ||
         hasInvalidOptionalI32Attr(op, "col_offset");
}

static LogicalResult verifyConvKxKDataflow(KernelOp op, Block &block,
                                           ArrayRef<Operation *> ops) {
  auto fail = [&]() {
    return op.emitOpError("kind convkxk requires canonical convkxk dataflow");
  };

  ConvKxKBodyLayout layout = getConvKxKBodyLayout(ops);
  if (layout == ConvKxKBodyLayout::None)
    return fail();

  std::optional<int64_t> kernelSize;
  if (layout == ConvKxKBodyLayout::LegalizableDotSeed)
    kernelSize = getConvKxKDotSeedKernelSizeFromOpCount(ops.size());
  else
    kernelSize = getConvKxKKernelSizeFromOpCount(ops.size());
  if (!kernelSize)
    return fail();

  Value acc;
  int64_t windows = *kernelSize * *kernelSize;
  auto checkLoadPair = [&](LoadMatrixOp x, LoadMatrixOp w,
                           int64_t window) -> LogicalResult {
    int64_t ky = window / *kernelSize;
    int64_t kx = window % *kernelSize;
    int64_t expectedXRow = ky * 16 + kx;
    int64_t expectedWRow = window * 16;
    if (hasInvalidMatrixOffsetAttr(x.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w.getOperation()))
      return fail();
    if (x.getPtr() != block.getArgument(1) ||
        w.getPtr() != block.getArgument(2) || x.getRows() != 16 ||
        x.getCols() != 16 || w.getRows() != 16 || w.getCols() != 16 ||
        getI32AttrOrZero(x.getOperation(), "row_offset") != expectedXRow ||
        getI32AttrOrZero(w.getOperation(), "row_offset") != expectedWRow ||
        getI32AttrOrZero(x.getOperation(), "col_offset") != 0 ||
        getI32AttrOrZero(w.getOperation(), "col_offset") != 0)
      return fail();
    return success();
  };

  if (layout == ConvKxKBodyLayout::CanonicalPrimitive) {
    auto zero = cast<ZeroOp>(ops[0]);
    if (zero.getRows() != 16 || zero.getCols() != 16)
      return fail();
    acc = zero.getResult();
  } else if (layout == ConvKxKBodyLayout::LegalizableDotSeed) {
    auto x = cast<LoadMatrixOp>(ops[0]);
    auto w = cast<LoadMatrixOp>(ops[1]);
    auto dot = cast<DotOp>(ops[2]);
    if (failed(checkLoadPair(x, w, /*window=*/0)) ||
        dot.getLhs() != x.getResult() || dot.getRhs() != w.getResult())
      return fail();
    acc = dot.getResult();
  } else {
    auto x = cast<LoadMatrixOp>(ops[0]);
    auto w = cast<LoadMatrixOp>(ops[1]);
    auto zero = cast<ZeroOp>(ops[2]);
    auto mma = cast<MmaOp>(ops[3]);
    if (zero.getRows() != 16 || zero.getCols() != 16 ||
        failed(checkLoadPair(x, w, /*window=*/0)) ||
        mma.getLhs() != x.getResult() || mma.getRhs() != w.getResult() ||
        mma.getAcc() != zero.getResult())
      return fail();
    acc = mma.getResult();
  }

  for (int64_t i = 0; i < windows; ++i) {
    size_t base = 0;
    if (layout == ConvKxKBodyLayout::CanonicalPrimitive)
      base = static_cast<size_t>(1 + i * 3);
    else {
      if (i == 0)
        continue;
      base = layout == ConvKxKBodyLayout::LegalizableDotSeed
                 ? static_cast<size_t>(3 + (i - 1) * 3)
                 : static_cast<size_t>(4 + (i - 1) * 3);
    }
    auto x = cast<LoadMatrixOp>(ops[base]);
    auto w = cast<LoadMatrixOp>(ops[base + 1]);
    auto mma = cast<MmaOp>(ops[base + 2]);
    if (failed(checkLoadPair(x, w, i)) || mma.getLhs() != x.getResult() ||
        mma.getRhs() != w.getResult() || mma.getAcc() != acc)
      return fail();
    acc = mma.getResult();
  }

  size_t storeIndex = static_cast<size_t>(1 + windows * 3);
  if (layout == ConvKxKBodyLayout::LegalizableDotSeed)
    storeIndex = static_cast<size_t>(3 * windows);
  else if (layout == ConvKxKBodyLayout::ExpandedDotSeedPrimitive)
    storeIndex = static_cast<size_t>(3 * windows + 1);
  auto store = cast<StoreMatrixOp>(ops[storeIndex]);
  if (store.getPtr() != block.getArgument(0) || store.getValue() != acc ||
      store.getRows() != 16 || store.getCols() != 16)
    return fail();
  return success();
}

FailureOr<ExecutableConvKxKOpSequenceInfo>
mlir::rpu::exec::getExecutableConvKxKOpSequenceInfo(KernelOp kernel,
                                                    ArrayRef<Operation *> ops,
                                                    llvm::StringRef consumer) {
  ConvKxKBodyLayout layout = getConvKxKBodyLayout(ops);
  if (layout != ConvKxKBodyLayout::CanonicalPrimitive &&
      layout != ConvKxKBodyLayout::ExpandedDotSeedPrimitive) {
    kernel.emitError() << consumer
                       << " requires primitive convkxk executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyConvKxKDataflow(kernel, block, ops)))
    return failure();

  std::optional<int64_t> kernelSize =
      getConvKxKKernelSizeFromOpCount(ops.size());
  if (!kernelSize) {
    kernel.emitError() << consumer << " requires supported convkxk op count";
    return failure();
  }

  size_t zeroIndex = layout == ConvKxKBodyLayout::CanonicalPrimitive ? 0 : 2;
  auto zero = cast<ZeroOp>(ops[zeroIndex]);
  SmallVector<LoadMatrixOp, 16> inputLoads;
  SmallVector<LoadMatrixOp, 16> weightLoads;
  SmallVector<MmaOp, 16> mmas;
  int64_t windows = *kernelSize * *kernelSize;
  for (int64_t i = 0; i < windows; ++i) {
    size_t base = layout == ConvKxKBodyLayout::CanonicalPrimitive
                      ? static_cast<size_t>(1 + i * 3)
                      : (i == 0 ? 0 : static_cast<size_t>(4 + (i - 1) * 3));
    inputLoads.push_back(cast<LoadMatrixOp>(ops[base]));
    weightLoads.push_back(cast<LoadMatrixOp>(ops[base + 1]));
    size_t mmaIndex =
        layout == ConvKxKBodyLayout::ExpandedDotSeedPrimitive && i == 0
            ? 3
            : base + 2;
    mmas.push_back(cast<MmaOp>(ops[mmaIndex]));
  }

  size_t storeIndex = layout == ConvKxKBodyLayout::CanonicalPrimitive
                          ? static_cast<size_t>(1 + windows * 3)
                          : static_cast<size_t>(3 * windows + 1);
  auto store = cast<StoreMatrixOp>(ops[storeIndex]);
  return ExecutableConvKxKOpSequenceInfo{zero, inputLoads, weightLoads,
                                         mmas, store,      *kernelSize};
}

static LogicalResult verifyResNetBlockDataflow(KernelOp op, Block &block,
                                               ArrayRef<Operation *> ops) {
  auto fail = [&]() {
    return op.emitOpError(
        "kind resnet_block requires canonical residual block dataflow");
  };
  auto is16x16Load = [](LoadMatrixOp load) {
    return load.getRows() == 16 && load.getCols() == 16;
  };
  auto is16x16Zero = [](ZeroOp zero) {
    return zero.getRows() == 16 && zero.getCols() == 16;
  };
  auto hasOffsets = [](LoadMatrixOp load, int64_t row, int64_t col) {
    Operation *op = load.getOperation();
    return getI32AttrOrZero(op, "row_offset") == row &&
           getI32AttrOrZero(op, "col_offset") == col;
  };

  auto x = cast<LoadMatrixOp>(ops[0]);
  if (hasLegalizableResNetBlockBody(ops)) {
    if (ops.size() == 18) {
      auto w1Lo = cast<LoadMatrixOp>(ops[1]);
      auto conv1Lo = cast<DotOp>(ops[2]);
      auto zero1Lo = cast<ZeroOp>(ops[3]);
      auto relu1Lo = cast<MaxOp>(ops[4]);
      auto w1Hi = cast<LoadMatrixOp>(ops[5]);
      auto conv1Hi = cast<DotOp>(ops[6]);
      auto zero1Hi = cast<ZeroOp>(ops[7]);
      auto relu1Hi = cast<MaxOp>(ops[8]);
      auto w2Lo = cast<LoadMatrixOp>(ops[9]);
      auto conv2Lo = cast<DotOp>(ops[10]);
      auto w2Hi = cast<LoadMatrixOp>(ops[11]);
      auto conv2 = cast<MmaOp>(ops[12]);
      auto residual = cast<AddOp>(ops[13]);
      auto zero2 = cast<ZeroOp>(ops[14]);
      auto outTile = cast<MaxOp>(ops[15]);
      auto store = cast<StoreMatrixOp>(ops[16]);

      if (hasInvalidMatrixOffsetAttr(w1Lo.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w1Hi.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w2Lo.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w2Hi.getOperation()))
        return fail();
      if (!is16x16Load(x) || !is16x16Load(w1Lo) || !is16x16Load(w1Hi) ||
          !is16x16Load(w2Lo) || !is16x16Load(w2Hi) || !is16x16Zero(zero1Lo) ||
          !is16x16Zero(zero1Hi) || !is16x16Zero(zero2) ||
          x.getPtr() != block.getArgument(1) ||
          w1Lo.getPtr() != block.getArgument(2) ||
          w1Hi.getPtr() != block.getArgument(2) ||
          w2Lo.getPtr() != block.getArgument(3) ||
          w2Hi.getPtr() != block.getArgument(3) ||
          store.getPtr() != block.getArgument(0) || !hasOffsets(w1Lo, 0, 0) ||
          !hasOffsets(w1Hi, 0, 16) || !hasOffsets(w2Lo, 0, 0) ||
          !hasOffsets(w2Hi, 16, 0) || store.getRows() != 16 ||
          store.getCols() != 16 || conv1Lo.getLhs() != x.getResult() ||
          conv1Lo.getRhs() != w1Lo.getResult() ||
          relu1Lo.getLhs() != conv1Lo.getResult() ||
          relu1Lo.getRhs() != zero1Lo.getResult() ||
          conv1Hi.getLhs() != x.getResult() ||
          conv1Hi.getRhs() != w1Hi.getResult() ||
          relu1Hi.getLhs() != conv1Hi.getResult() ||
          relu1Hi.getRhs() != zero1Hi.getResult() ||
          conv2Lo.getLhs() != relu1Lo.getResult() ||
          conv2Lo.getRhs() != w2Lo.getResult() ||
          conv2.getLhs() != relu1Hi.getResult() ||
          conv2.getRhs() != w2Hi.getResult() ||
          conv2.getAcc() != conv2Lo.getResult() ||
          residual.getLhs() != conv2.getResult() ||
          residual.getRhs() != x.getResult() ||
          outTile.getLhs() != residual.getResult() ||
          outTile.getRhs() != zero2.getResult() ||
          store.getValue() != outTile.getResult())
        return fail();
      return success();
    }

    auto w1 = cast<LoadMatrixOp>(ops[1]);
    auto conv1 = cast<DotOp>(ops[2]);
    auto zero1 = cast<ZeroOp>(ops[3]);
    auto relu1 = cast<MaxOp>(ops[4]);
    auto w2 = cast<LoadMatrixOp>(ops[5]);
    auto conv2 = cast<DotOp>(ops[6]);
    auto residual = cast<AddOp>(ops[7]);
    auto zero2 = cast<ZeroOp>(ops[8]);
    auto outTile = cast<MaxOp>(ops[9]);
    auto store = cast<StoreMatrixOp>(ops[10]);
    int64_t hidden = w1.getCols();

    if (x.getPtr() != block.getArgument(1) ||
        w1.getPtr() != block.getArgument(2) ||
        w2.getPtr() != block.getArgument(3) ||
        store.getPtr() != block.getArgument(0) || x.getRows() != 16 ||
        x.getCols() != 16 || w1.getRows() != 16 || hidden != 16 ||
        w2.getRows() != hidden || w2.getCols() != 16 || zero1.getRows() != 16 ||
        zero1.getCols() != hidden || zero2.getRows() != 16 ||
        zero2.getCols() != 16 || store.getRows() != 16 ||
        store.getCols() != 16 || conv1.getLhs() != x.getResult() ||
        conv1.getRhs() != w1.getResult() ||
        relu1.getLhs() != conv1.getResult() ||
        relu1.getRhs() != zero1.getResult() ||
        conv2.getLhs() != relu1.getResult() ||
        conv2.getRhs() != w2.getResult() ||
        residual.getLhs() != conv2.getResult() ||
        residual.getRhs() != x.getResult() ||
        outTile.getLhs() != residual.getResult() ||
        outTile.getRhs() != zero2.getResult() ||
        store.getValue() != outTile.getResult())
      return fail();
    return success();
  }

  if (ops.size() == 21) {
    auto w1Lo = cast<LoadMatrixOp>(ops[1]);
    auto conv1LoZero = cast<ZeroOp>(ops[2]);
    auto conv1Lo = cast<MmaOp>(ops[3]);
    auto zero1Lo = cast<ZeroOp>(ops[4]);
    auto relu1Lo = cast<MaxOp>(ops[5]);
    auto w1Hi = cast<LoadMatrixOp>(ops[6]);
    auto conv1HiZero = cast<ZeroOp>(ops[7]);
    auto conv1Hi = cast<MmaOp>(ops[8]);
    auto zero1Hi = cast<ZeroOp>(ops[9]);
    auto relu1Hi = cast<MaxOp>(ops[10]);
    auto w2Lo = cast<LoadMatrixOp>(ops[11]);
    auto conv2Zero = cast<ZeroOp>(ops[12]);
    auto conv2Lo = cast<MmaOp>(ops[13]);
    auto w2Hi = cast<LoadMatrixOp>(ops[14]);
    auto conv2 = cast<MmaOp>(ops[15]);
    auto residual = cast<AddOp>(ops[16]);
    auto zero2 = cast<ZeroOp>(ops[17]);
    auto outTile = cast<MaxOp>(ops[18]);
    auto store = cast<StoreMatrixOp>(ops[19]);

    if (hasInvalidMatrixOffsetAttr(w1Lo.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w1Hi.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w2Lo.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w2Hi.getOperation()))
      return fail();
    if (!is16x16Load(x) || !is16x16Load(w1Lo) || !is16x16Load(w1Hi) ||
        !is16x16Load(w2Lo) || !is16x16Load(w2Hi) || !is16x16Zero(conv1LoZero) ||
        !is16x16Zero(zero1Lo) || !is16x16Zero(conv1HiZero) ||
        !is16x16Zero(zero1Hi) || !is16x16Zero(conv2Zero) ||
        !is16x16Zero(zero2) || x.getPtr() != block.getArgument(1) ||
        w1Lo.getPtr() != block.getArgument(2) ||
        w1Hi.getPtr() != block.getArgument(2) ||
        w2Lo.getPtr() != block.getArgument(3) ||
        w2Hi.getPtr() != block.getArgument(3) ||
        store.getPtr() != block.getArgument(0) || !hasOffsets(w1Lo, 0, 0) ||
        !hasOffsets(w1Hi, 0, 16) || !hasOffsets(w2Lo, 0, 0) ||
        !hasOffsets(w2Hi, 16, 0) || store.getRows() != 16 ||
        store.getCols() != 16 || conv1Lo.getLhs() != x.getResult() ||
        conv1Lo.getRhs() != w1Lo.getResult() ||
        conv1Lo.getAcc() != conv1LoZero.getResult() ||
        relu1Lo.getLhs() != conv1Lo.getResult() ||
        relu1Lo.getRhs() != zero1Lo.getResult() ||
        conv1Hi.getLhs() != x.getResult() ||
        conv1Hi.getRhs() != w1Hi.getResult() ||
        conv1Hi.getAcc() != conv1HiZero.getResult() ||
        relu1Hi.getLhs() != conv1Hi.getResult() ||
        relu1Hi.getRhs() != zero1Hi.getResult() ||
        conv2Lo.getLhs() != relu1Lo.getResult() ||
        conv2Lo.getRhs() != w2Lo.getResult() ||
        conv2Lo.getAcc() != conv2Zero.getResult() ||
        conv2.getLhs() != relu1Hi.getResult() ||
        conv2.getRhs() != w2Hi.getResult() ||
        conv2.getAcc() != conv2Lo.getResult() ||
        residual.getLhs() != conv2.getResult() ||
        residual.getRhs() != x.getResult() ||
        outTile.getLhs() != residual.getResult() ||
        outTile.getRhs() != zero2.getResult() ||
        store.getValue() != outTile.getResult())
      return fail();
    return success();
  }

  auto w1 = cast<LoadMatrixOp>(ops[1]);
  auto conv1Zero = cast<ZeroOp>(ops[2]);
  auto conv1 = cast<MmaOp>(ops[3]);
  auto zero1 = cast<ZeroOp>(ops[4]);
  auto relu1 = cast<MaxOp>(ops[5]);
  auto w2 = cast<LoadMatrixOp>(ops[6]);
  auto conv2Zero = cast<ZeroOp>(ops[7]);
  auto conv2 = cast<MmaOp>(ops[8]);
  auto residual = cast<AddOp>(ops[9]);
  auto zero2 = cast<ZeroOp>(ops[10]);
  auto outTile = cast<MaxOp>(ops[11]);
  auto store = cast<StoreMatrixOp>(ops[12]);
  int64_t hidden = w1.getCols();

  if (x.getPtr() != block.getArgument(1) ||
      w1.getPtr() != block.getArgument(2) ||
      w2.getPtr() != block.getArgument(3) ||
      store.getPtr() != block.getArgument(0) || x.getRows() != 16 ||
      x.getCols() != 16 || w1.getRows() != 16 || hidden != 16 ||
      w2.getRows() != hidden || w2.getCols() != 16 ||
      conv1Zero.getRows() != 16 || conv1Zero.getCols() != hidden ||
      zero1.getRows() != 16 || zero1.getCols() != hidden ||
      conv2Zero.getRows() != 16 || conv2Zero.getCols() != 16 ||
      zero2.getRows() != 16 || zero2.getCols() != 16 || store.getRows() != 16 ||
      store.getCols() != 16 || conv1.getLhs() != x.getResult() ||
      conv1.getRhs() != w1.getResult() ||
      conv1.getAcc() != conv1Zero.getResult() ||
      relu1.getLhs() != conv1.getResult() ||
      relu1.getRhs() != zero1.getResult() ||
      conv2.getLhs() != relu1.getResult() || conv2.getRhs() != w2.getResult() ||
      conv2.getAcc() != conv2Zero.getResult() ||
      residual.getLhs() != conv2.getResult() ||
      residual.getRhs() != x.getResult() ||
      outTile.getLhs() != residual.getResult() ||
      outTile.getRhs() != zero2.getResult() ||
      store.getValue() != outTile.getResult())
    return fail();
  return success();
}

FailureOr<ExecutableResNetBlockOpSequenceInfo>
mlir::rpu::exec::getExecutableResNetBlockOpSequenceInfo(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer) {
  if (!hasResNetBlockBody(ops)) {
    kernel.emitError() << consumer
                       << " requires canonical residual executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyResNetBlockDataflow(kernel, block, ops)))
    return failure();

  auto xLoad = cast<LoadMatrixOp>(ops[0]);
  if (ops.size() == 21) {
    SmallVector<LoadMatrixOp, 2> w1Loads{cast<LoadMatrixOp>(ops[1]),
                                         cast<LoadMatrixOp>(ops[6])};
    SmallVector<LoadMatrixOp, 2> w2Loads{cast<LoadMatrixOp>(ops[11]),
                                         cast<LoadMatrixOp>(ops[14])};
    SmallVector<ZeroOp, 5> zeros{cast<ZeroOp>(ops[2]),  cast<ZeroOp>(ops[4]),
                                 cast<ZeroOp>(ops[7]),  cast<ZeroOp>(ops[9]),
                                 cast<ZeroOp>(ops[12]), cast<ZeroOp>(ops[17])};
    SmallVector<MmaOp, 4> mmas{cast<MmaOp>(ops[3]), cast<MmaOp>(ops[8]),
                               cast<MmaOp>(ops[13]), cast<MmaOp>(ops[15])};
    SmallVector<MaxOp, 3> relus{cast<MaxOp>(ops[5]), cast<MaxOp>(ops[10]),
                                cast<MaxOp>(ops[18])};
    auto residualAdd = cast<AddOp>(ops[16]);
    auto store = cast<StoreMatrixOp>(ops[19]);
    int64_t hidden = 32;
    return ExecutableResNetBlockOpSequenceInfo{xLoad,       w1Loads, w2Loads,
                                               zeros,       mmas,    relus,
                                               residualAdd, store,   hidden};
  }

  SmallVector<LoadMatrixOp, 2> w1Loads{cast<LoadMatrixOp>(ops[1])};
  SmallVector<LoadMatrixOp, 2> w2Loads{cast<LoadMatrixOp>(ops[6])};
  SmallVector<ZeroOp, 5> zeros{cast<ZeroOp>(ops[2]), cast<ZeroOp>(ops[4]),
                               cast<ZeroOp>(ops[7]), cast<ZeroOp>(ops[10])};
  SmallVector<MmaOp, 4> mmas{cast<MmaOp>(ops[3]), cast<MmaOp>(ops[8])};
  SmallVector<MaxOp, 3> relus{cast<MaxOp>(ops[5]), cast<MaxOp>(ops[11])};
  auto residualAdd = cast<AddOp>(ops[9]);
  auto store = cast<StoreMatrixOp>(ops[12]);
  int64_t hidden = 16;
  return ExecutableResNetBlockOpSequenceInfo{
      xLoad, w1Loads, w2Loads, zeros, mmas, relus, residualAdd, store, hidden};
}

static LogicalResult
verifyResNet50BottleneckDataflow(KernelOp op, Block &block,
                                 ArrayRef<Operation *> ops) {
  auto fail = [&]() {
    return op.emitOpError("kind resnet50_bottleneck requires canonical "
                          "resnet50 bottleneck dataflow");
  };
  auto is16x16Load = [](LoadMatrixOp load) {
    return load.getRows() == 16 && load.getCols() == 16;
  };
  auto is16x16Zero = [](ZeroOp zero) {
    return zero.getRows() == 16 && zero.getCols() == 16;
  };
  auto hasZeroOffsets = [](LoadMatrixOp load) {
    Operation *op = load.getOperation();
    return getI32AttrOrZero(op, "row_offset") == 0 &&
           getI32AttrOrZero(op, "col_offset") == 0;
  };
  auto hasOffsets = [](LoadMatrixOp load, int64_t row, int64_t col) {
    Operation *op = load.getOperation();
    return getI32AttrOrZero(op, "row_offset") == row &&
           getI32AttrOrZero(op, "col_offset") == col;
  };

  auto xSkip = cast<LoadMatrixOp>(ops[0]);
  if (hasLegalizableResNet50BottleneckBody(ops)) {
    if (ops.size() == 153) {
      auto w1Lo = cast<LoadMatrixOp>(ops[1]);
      auto w1Hi = cast<LoadMatrixOp>(ops[2]);
      auto conv2LoZero = cast<ZeroOp>(ops[3]);
      auto conv2HiZero = cast<ZeroOp>(ops[4]);
      if (hasInvalidMatrixOffsetAttr(xSkip.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w1Lo.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w1Hi.getOperation()))
        return fail();
      if (!is16x16Load(xSkip) || !is16x16Load(w1Lo) || !is16x16Load(w1Hi) ||
          !is16x16Zero(conv2LoZero) || !is16x16Zero(conv2HiZero) ||
          xSkip.getPtr() != block.getArgument(1) ||
          w1Lo.getPtr() != block.getArgument(2) ||
          w1Hi.getPtr() != block.getArgument(2) || !hasOffsets(xSkip, 0, 0) ||
          !hasOffsets(w1Lo, 0, 0) || !hasOffsets(w1Hi, 0, 16))
        return fail();

      Value conv2AccLo = conv2LoZero.getResult();
      Value conv2AccHi = conv2HiZero.getResult();
      for (int64_t i = 0; i < 9; ++i) {
        size_t base = static_cast<size_t>(5 + i * 15);
        int64_t ky = i / 3;
        int64_t kx = i % 3;
        int64_t expectedXRow = ky * 16 + kx;
        int64_t expectedW2Row = i * 32;

        auto x = cast<LoadMatrixOp>(ops[base]);
        auto conv1Lo = cast<DotOp>(ops[base + 1]);
        auto zero1Lo = cast<ZeroOp>(ops[base + 2]);
        auto relu1Lo = cast<MaxOp>(ops[base + 3]);
        auto conv1Hi = cast<DotOp>(ops[base + 4]);
        auto zero1Hi = cast<ZeroOp>(ops[base + 5]);
        auto relu1Hi = cast<MaxOp>(ops[base + 6]);
        auto w2LoLo = cast<LoadMatrixOp>(ops[base + 7]);
        auto conv2LoFirst = cast<MmaOp>(ops[base + 8]);
        auto w2HiLo = cast<LoadMatrixOp>(ops[base + 9]);
        auto conv2Lo = cast<MmaOp>(ops[base + 10]);
        auto w2LoHi = cast<LoadMatrixOp>(ops[base + 11]);
        auto conv2HiFirst = cast<MmaOp>(ops[base + 12]);
        auto w2HiHi = cast<LoadMatrixOp>(ops[base + 13]);
        auto conv2Hi = cast<MmaOp>(ops[base + 14]);
        if (hasInvalidMatrixOffsetAttr(x.getOperation()) ||
            hasInvalidMatrixOffsetAttr(w2LoLo.getOperation()) ||
            hasInvalidMatrixOffsetAttr(w2HiLo.getOperation()) ||
            hasInvalidMatrixOffsetAttr(w2LoHi.getOperation()) ||
            hasInvalidMatrixOffsetAttr(w2HiHi.getOperation()))
          return fail();
        if (!is16x16Load(x) || !is16x16Zero(zero1Lo) || !is16x16Zero(zero1Hi) ||
            !is16x16Load(w2LoLo) || !is16x16Load(w2HiLo) ||
            !is16x16Load(w2LoHi) || !is16x16Load(w2HiHi) ||
            x.getPtr() != block.getArgument(1) ||
            w2LoLo.getPtr() != block.getArgument(3) ||
            w2HiLo.getPtr() != block.getArgument(3) ||
            w2LoHi.getPtr() != block.getArgument(3) ||
            w2HiHi.getPtr() != block.getArgument(3) ||
            !hasOffsets(x, expectedXRow, 0) ||
            !hasOffsets(w2LoLo, expectedW2Row, 0) ||
            !hasOffsets(w2HiLo, expectedW2Row + 16, 0) ||
            !hasOffsets(w2LoHi, expectedW2Row, 16) ||
            !hasOffsets(w2HiHi, expectedW2Row + 16, 16) ||
            conv1Lo.getLhs() != x.getResult() ||
            conv1Lo.getRhs() != w1Lo.getResult() ||
            relu1Lo.getLhs() != conv1Lo.getResult() ||
            relu1Lo.getRhs() != zero1Lo.getResult() ||
            conv1Hi.getLhs() != x.getResult() ||
            conv1Hi.getRhs() != w1Hi.getResult() ||
            relu1Hi.getLhs() != conv1Hi.getResult() ||
            relu1Hi.getRhs() != zero1Hi.getResult() ||
            conv2LoFirst.getLhs() != relu1Lo.getResult() ||
            conv2LoFirst.getRhs() != w2LoLo.getResult() ||
            conv2LoFirst.getAcc() != conv2AccLo ||
            conv2Lo.getLhs() != relu1Hi.getResult() ||
            conv2Lo.getRhs() != w2HiLo.getResult() ||
            conv2Lo.getAcc() != conv2LoFirst.getResult() ||
            conv2HiFirst.getLhs() != relu1Lo.getResult() ||
            conv2HiFirst.getRhs() != w2LoHi.getResult() ||
            conv2HiFirst.getAcc() != conv2AccHi ||
            conv2Hi.getLhs() != relu1Hi.getResult() ||
            conv2Hi.getRhs() != w2HiHi.getResult() ||
            conv2Hi.getAcc() != conv2HiFirst.getResult())
          return fail();
        conv2AccLo = conv2Lo.getResult();
        conv2AccHi = conv2Hi.getResult();
      }

      size_t tail = 140;
      auto zero2Lo = cast<ZeroOp>(ops[tail]);
      auto relu2Lo = cast<MaxOp>(ops[tail + 1]);
      auto zero2Hi = cast<ZeroOp>(ops[tail + 2]);
      auto relu2Hi = cast<MaxOp>(ops[tail + 3]);
      auto w3Lo = cast<LoadMatrixOp>(ops[tail + 4]);
      auto conv3Lo = cast<DotOp>(ops[tail + 5]);
      auto w3Hi = cast<LoadMatrixOp>(ops[tail + 6]);
      auto conv3 = cast<MmaOp>(ops[tail + 7]);
      auto residual = cast<AddOp>(ops[tail + 8]);
      auto zero3 = cast<ZeroOp>(ops[tail + 9]);
      auto outTile = cast<MaxOp>(ops[tail + 10]);
      auto store = cast<StoreMatrixOp>(ops[tail + 11]);
      if (hasInvalidMatrixOffsetAttr(w3Lo.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w3Hi.getOperation()))
        return fail();
      if (!is16x16Zero(zero2Lo) || !is16x16Zero(zero2Hi) ||
          !is16x16Load(w3Lo) || !is16x16Load(w3Hi) || !is16x16Zero(zero3) ||
          w3Lo.getPtr() != block.getArgument(4) ||
          w3Hi.getPtr() != block.getArgument(4) || !hasOffsets(w3Lo, 0, 0) ||
          !hasOffsets(w3Hi, 16, 0) || relu2Lo.getLhs() != conv2AccLo ||
          relu2Lo.getRhs() != zero2Lo.getResult() ||
          relu2Hi.getLhs() != conv2AccHi ||
          relu2Hi.getRhs() != zero2Hi.getResult() ||
          conv3Lo.getLhs() != relu2Lo.getResult() ||
          conv3Lo.getRhs() != w3Lo.getResult() ||
          conv3.getLhs() != relu2Hi.getResult() ||
          conv3.getRhs() != w3Hi.getResult() ||
          conv3.getAcc() != conv3Lo.getResult() ||
          residual.getLhs() != conv3.getResult() ||
          residual.getRhs() != xSkip.getResult() ||
          outTile.getLhs() != residual.getResult() ||
          outTile.getRhs() != zero3.getResult() ||
          store.getPtr() != block.getArgument(0) ||
          store.getValue() != outTile.getResult() || store.getRows() != 16 ||
          store.getCols() != 16)
        return fail();
      return success();
    }

    auto w1 = cast<LoadMatrixOp>(ops[1]);
    auto conv2Zero = cast<ZeroOp>(ops[2]);
    if (hasInvalidMatrixOffsetAttr(xSkip.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w1.getOperation()))
      return fail();
    if (!is16x16Load(xSkip) || !is16x16Load(w1) || !is16x16Zero(conv2Zero) ||
        xSkip.getPtr() != block.getArgument(1) ||
        w1.getPtr() != block.getArgument(2) || !hasZeroOffsets(xSkip) ||
        !hasZeroOffsets(w1))
      return fail();

    Value conv2Acc = conv2Zero.getResult();
    for (int64_t i = 0; i < 9; ++i) {
      size_t base = static_cast<size_t>(3 + i * 6);
      int64_t ky = i / 3;
      int64_t kx = i % 3;
      int64_t expectedXRow = ky * 16 + kx;
      int64_t expectedW2Row = i * 16;

      auto x = cast<LoadMatrixOp>(ops[base]);
      auto conv1 = cast<DotOp>(ops[base + 1]);
      auto zero1 = cast<ZeroOp>(ops[base + 2]);
      auto relu1 = cast<MaxOp>(ops[base + 3]);
      auto w2 = cast<LoadMatrixOp>(ops[base + 4]);
      auto conv2 = cast<MmaOp>(ops[base + 5]);
      if (hasInvalidMatrixOffsetAttr(x.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w2.getOperation()))
        return fail();
      if (!is16x16Load(x) || !is16x16Zero(zero1) || !is16x16Load(w2) ||
          x.getPtr() != block.getArgument(1) ||
          w2.getPtr() != block.getArgument(3) ||
          getI32AttrOrZero(x.getOperation(), "row_offset") != expectedXRow ||
          getI32AttrOrZero(x.getOperation(), "col_offset") != 0 ||
          getI32AttrOrZero(w2.getOperation(), "row_offset") != expectedW2Row ||
          getI32AttrOrZero(w2.getOperation(), "col_offset") != 0 ||
          conv1.getLhs() != x.getResult() || conv1.getRhs() != w1.getResult() ||
          relu1.getLhs() != conv1.getResult() ||
          relu1.getRhs() != zero1.getResult() ||
          conv2.getLhs() != relu1.getResult() ||
          conv2.getRhs() != w2.getResult() || conv2.getAcc() != conv2Acc)
        return fail();
      conv2Acc = conv2.getResult();
    }

    size_t tail = 57;
    auto zero2 = cast<ZeroOp>(ops[tail]);
    auto relu2 = cast<MaxOp>(ops[tail + 1]);
    auto w3 = cast<LoadMatrixOp>(ops[tail + 2]);
    auto conv3 = cast<DotOp>(ops[tail + 3]);
    auto residual = cast<AddOp>(ops[tail + 4]);
    auto zero3 = cast<ZeroOp>(ops[tail + 5]);
    auto outTile = cast<MaxOp>(ops[tail + 6]);
    auto store = cast<StoreMatrixOp>(ops[tail + 7]);
    if (hasInvalidMatrixOffsetAttr(w3.getOperation()))
      return fail();
    if (!is16x16Zero(zero2) || !is16x16Load(w3) || !is16x16Zero(zero3) ||
        w3.getPtr() != block.getArgument(4) || !hasZeroOffsets(w3) ||
        relu2.getLhs() != conv2Acc || relu2.getRhs() != zero2.getResult() ||
        conv3.getLhs() != relu2.getResult() ||
        conv3.getRhs() != w3.getResult() ||
        residual.getLhs() != conv3.getResult() ||
        residual.getRhs() != xSkip.getResult() ||
        outTile.getLhs() != residual.getResult() ||
        outTile.getRhs() != zero3.getResult() ||
        store.getPtr() != block.getArgument(0) ||
        store.getValue() != outTile.getResult() || store.getRows() != 16 ||
        store.getCols() != 16)
      return fail();
    return success();
  }

  if (ops.size() == 172) {
    auto w1Lo = cast<LoadMatrixOp>(ops[1]);
    auto w1Hi = cast<LoadMatrixOp>(ops[2]);
    auto conv2LoZero = cast<ZeroOp>(ops[3]);
    auto conv2HiZero = cast<ZeroOp>(ops[4]);
    if (hasInvalidMatrixOffsetAttr(xSkip.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w1Lo.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w1Hi.getOperation()))
      return fail();
    if (!is16x16Load(xSkip) || !is16x16Load(w1Lo) || !is16x16Load(w1Hi) ||
        !is16x16Zero(conv2LoZero) || !is16x16Zero(conv2HiZero) ||
        xSkip.getPtr() != block.getArgument(1) ||
        w1Lo.getPtr() != block.getArgument(2) ||
        w1Hi.getPtr() != block.getArgument(2) || !hasOffsets(xSkip, 0, 0) ||
        !hasOffsets(w1Lo, 0, 0) || !hasOffsets(w1Hi, 0, 16))
      return fail();

    Value conv2AccLo = conv2LoZero.getResult();
    Value conv2AccHi = conv2HiZero.getResult();
    for (int64_t i = 0; i < 9; ++i) {
      size_t base = static_cast<size_t>(5 + i * 17);
      int64_t ky = i / 3;
      int64_t kx = i % 3;
      int64_t expectedXRow = ky * 16 + kx;
      int64_t expectedW2Row = i * 32;

      auto x = cast<LoadMatrixOp>(ops[base]);
      auto conv1LoZero = cast<ZeroOp>(ops[base + 1]);
      auto conv1Lo = cast<MmaOp>(ops[base + 2]);
      auto zero1Lo = cast<ZeroOp>(ops[base + 3]);
      auto relu1Lo = cast<MaxOp>(ops[base + 4]);
      auto conv1HiZero = cast<ZeroOp>(ops[base + 5]);
      auto conv1Hi = cast<MmaOp>(ops[base + 6]);
      auto zero1Hi = cast<ZeroOp>(ops[base + 7]);
      auto relu1Hi = cast<MaxOp>(ops[base + 8]);
      auto w2LoLo = cast<LoadMatrixOp>(ops[base + 9]);
      auto conv2LoFirst = cast<MmaOp>(ops[base + 10]);
      auto w2HiLo = cast<LoadMatrixOp>(ops[base + 11]);
      auto conv2Lo = cast<MmaOp>(ops[base + 12]);
      auto w2LoHi = cast<LoadMatrixOp>(ops[base + 13]);
      auto conv2HiFirst = cast<MmaOp>(ops[base + 14]);
      auto w2HiHi = cast<LoadMatrixOp>(ops[base + 15]);
      auto conv2Hi = cast<MmaOp>(ops[base + 16]);
      if (hasInvalidMatrixOffsetAttr(x.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w2LoLo.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w2HiLo.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w2LoHi.getOperation()) ||
          hasInvalidMatrixOffsetAttr(w2HiHi.getOperation()))
        return fail();
      if (!is16x16Load(x) || !is16x16Zero(conv1LoZero) ||
          !is16x16Zero(zero1Lo) || !is16x16Zero(conv1HiZero) ||
          !is16x16Zero(zero1Hi) || !is16x16Load(w2LoLo) ||
          !is16x16Load(w2HiLo) || !is16x16Load(w2LoHi) ||
          !is16x16Load(w2HiHi) || x.getPtr() != block.getArgument(1) ||
          w2LoLo.getPtr() != block.getArgument(3) ||
          w2HiLo.getPtr() != block.getArgument(3) ||
          w2LoHi.getPtr() != block.getArgument(3) ||
          w2HiHi.getPtr() != block.getArgument(3) ||
          !hasOffsets(x, expectedXRow, 0) ||
          !hasOffsets(w2LoLo, expectedW2Row, 0) ||
          !hasOffsets(w2HiLo, expectedW2Row + 16, 0) ||
          !hasOffsets(w2LoHi, expectedW2Row, 16) ||
          !hasOffsets(w2HiHi, expectedW2Row + 16, 16) ||
          conv1Lo.getLhs() != x.getResult() ||
          conv1Lo.getRhs() != w1Lo.getResult() ||
          conv1Lo.getAcc() != conv1LoZero.getResult() ||
          relu1Lo.getLhs() != conv1Lo.getResult() ||
          relu1Lo.getRhs() != zero1Lo.getResult() ||
          conv1Hi.getLhs() != x.getResult() ||
          conv1Hi.getRhs() != w1Hi.getResult() ||
          conv1Hi.getAcc() != conv1HiZero.getResult() ||
          relu1Hi.getLhs() != conv1Hi.getResult() ||
          relu1Hi.getRhs() != zero1Hi.getResult() ||
          conv2LoFirst.getLhs() != relu1Lo.getResult() ||
          conv2LoFirst.getRhs() != w2LoLo.getResult() ||
          conv2LoFirst.getAcc() != conv2AccLo ||
          conv2Lo.getLhs() != relu1Hi.getResult() ||
          conv2Lo.getRhs() != w2HiLo.getResult() ||
          conv2Lo.getAcc() != conv2LoFirst.getResult() ||
          conv2HiFirst.getLhs() != relu1Lo.getResult() ||
          conv2HiFirst.getRhs() != w2LoHi.getResult() ||
          conv2HiFirst.getAcc() != conv2AccHi ||
          conv2Hi.getLhs() != relu1Hi.getResult() ||
          conv2Hi.getRhs() != w2HiHi.getResult() ||
          conv2Hi.getAcc() != conv2HiFirst.getResult())
        return fail();
      conv2AccLo = conv2Lo.getResult();
      conv2AccHi = conv2Hi.getResult();
    }

    size_t tail = 158;
    auto zero2Lo = cast<ZeroOp>(ops[tail]);
    auto relu2Lo = cast<MaxOp>(ops[tail + 1]);
    auto zero2Hi = cast<ZeroOp>(ops[tail + 2]);
    auto relu2Hi = cast<MaxOp>(ops[tail + 3]);
    auto w3Lo = cast<LoadMatrixOp>(ops[tail + 4]);
    auto conv3Zero = cast<ZeroOp>(ops[tail + 5]);
    auto conv3Lo = cast<MmaOp>(ops[tail + 6]);
    auto w3Hi = cast<LoadMatrixOp>(ops[tail + 7]);
    auto conv3 = cast<MmaOp>(ops[tail + 8]);
    auto residual = cast<AddOp>(ops[tail + 9]);
    auto zero3 = cast<ZeroOp>(ops[tail + 10]);
    auto outTile = cast<MaxOp>(ops[tail + 11]);
    auto store = cast<StoreMatrixOp>(ops[tail + 12]);
    if (hasInvalidMatrixOffsetAttr(w3Lo.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w3Hi.getOperation()))
      return fail();
    if (!is16x16Zero(zero2Lo) || !is16x16Zero(zero2Hi) || !is16x16Load(w3Lo) ||
        !is16x16Zero(conv3Zero) || !is16x16Load(w3Hi) || !is16x16Zero(zero3) ||
        w3Lo.getPtr() != block.getArgument(4) ||
        w3Hi.getPtr() != block.getArgument(4) || !hasOffsets(w3Lo, 0, 0) ||
        !hasOffsets(w3Hi, 16, 0) || relu2Lo.getLhs() != conv2AccLo ||
        relu2Lo.getRhs() != zero2Lo.getResult() ||
        relu2Hi.getLhs() != conv2AccHi ||
        relu2Hi.getRhs() != zero2Hi.getResult() ||
        conv3Lo.getLhs() != relu2Lo.getResult() ||
        conv3Lo.getRhs() != w3Lo.getResult() ||
        conv3Lo.getAcc() != conv3Zero.getResult() ||
        conv3.getLhs() != relu2Hi.getResult() ||
        conv3.getRhs() != w3Hi.getResult() ||
        conv3.getAcc() != conv3Lo.getResult() ||
        residual.getLhs() != conv3.getResult() ||
        residual.getRhs() != xSkip.getResult() ||
        outTile.getLhs() != residual.getResult() ||
        outTile.getRhs() != zero3.getResult() ||
        store.getPtr() != block.getArgument(0) ||
        store.getValue() != outTile.getResult() || store.getRows() != 16 ||
        store.getCols() != 16)
      return fail();
    return success();
  }

  auto w1 = cast<LoadMatrixOp>(ops[1]);
  auto conv2Zero = cast<ZeroOp>(ops[2]);
  if (hasInvalidMatrixOffsetAttr(xSkip.getOperation()) ||
      hasInvalidMatrixOffsetAttr(w1.getOperation()))
    return fail();
  if (!is16x16Load(xSkip) || !is16x16Load(w1) || !is16x16Zero(conv2Zero) ||
      xSkip.getPtr() != block.getArgument(1) ||
      w1.getPtr() != block.getArgument(2) || !hasZeroOffsets(xSkip) ||
      !hasZeroOffsets(w1))
    return fail();

  Value conv2Acc = conv2Zero.getResult();
  for (int64_t i = 0; i < 9; ++i) {
    size_t base = static_cast<size_t>(3 + i * 7);
    int64_t ky = i / 3;
    int64_t kx = i % 3;
    int64_t expectedXRow = ky * 16 + kx;
    int64_t expectedW2Row = i * 16;

    auto x = cast<LoadMatrixOp>(ops[base]);
    auto conv1Zero = cast<ZeroOp>(ops[base + 1]);
    auto conv1 = cast<MmaOp>(ops[base + 2]);
    auto zero1 = cast<ZeroOp>(ops[base + 3]);
    auto relu1 = cast<MaxOp>(ops[base + 4]);
    auto w2 = cast<LoadMatrixOp>(ops[base + 5]);
    auto conv2 = cast<MmaOp>(ops[base + 6]);
    if (hasInvalidMatrixOffsetAttr(x.getOperation()) ||
        hasInvalidMatrixOffsetAttr(w2.getOperation()))
      return fail();
    if (!is16x16Load(x) || !is16x16Zero(conv1Zero) || !is16x16Zero(zero1) ||
        !is16x16Load(w2) || x.getPtr() != block.getArgument(1) ||
        w2.getPtr() != block.getArgument(3) ||
        getI32AttrOrZero(x.getOperation(), "row_offset") != expectedXRow ||
        getI32AttrOrZero(x.getOperation(), "col_offset") != 0 ||
        getI32AttrOrZero(w2.getOperation(), "row_offset") != expectedW2Row ||
        getI32AttrOrZero(w2.getOperation(), "col_offset") != 0 ||
        conv1.getLhs() != x.getResult() || conv1.getRhs() != w1.getResult() ||
        conv1.getAcc() != conv1Zero.getResult() ||
        relu1.getLhs() != conv1.getResult() ||
        relu1.getRhs() != zero1.getResult() ||
        conv2.getLhs() != relu1.getResult() ||
        conv2.getRhs() != w2.getResult() || conv2.getAcc() != conv2Acc)
      return fail();
    conv2Acc = conv2.getResult();
  }

  size_t tail = 66;
  auto zero2 = cast<ZeroOp>(ops[tail]);
  auto relu2 = cast<MaxOp>(ops[tail + 1]);
  auto w3 = cast<LoadMatrixOp>(ops[tail + 2]);
  auto conv3Zero = cast<ZeroOp>(ops[tail + 3]);
  auto conv3 = cast<MmaOp>(ops[tail + 4]);
  auto residual = cast<AddOp>(ops[tail + 5]);
  auto zero3 = cast<ZeroOp>(ops[tail + 6]);
  auto outTile = cast<MaxOp>(ops[tail + 7]);
  auto store = cast<StoreMatrixOp>(ops[tail + 8]);
  if (hasInvalidMatrixOffsetAttr(w3.getOperation()))
    return fail();
  if (!is16x16Zero(zero2) || !is16x16Load(w3) || !is16x16Zero(conv3Zero) ||
      !is16x16Zero(zero3) || w3.getPtr() != block.getArgument(4) ||
      !hasZeroOffsets(w3) || relu2.getLhs() != conv2Acc ||
      relu2.getRhs() != zero2.getResult() ||
      conv3.getLhs() != relu2.getResult() || conv3.getRhs() != w3.getResult() ||
      conv3.getAcc() != conv3Zero.getResult() ||
      residual.getLhs() != conv3.getResult() ||
      residual.getRhs() != xSkip.getResult() ||
      outTile.getLhs() != residual.getResult() ||
      outTile.getRhs() != zero3.getResult() ||
      store.getPtr() != block.getArgument(0) ||
      store.getValue() != outTile.getResult() || store.getRows() != 16 ||
      store.getCols() != 16)
    return fail();
  return success();
}

FailureOr<ExecutableResNet50BottleneckOpSequenceInfo>
mlir::rpu::exec::getExecutableResNet50BottleneckOpSequenceInfo(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer) {
  if (!hasResNet50BottleneckBody(ops)) {
    kernel.emitError()
        << consumer
        << " requires canonical resnet50 bottleneck executable body";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  Block &block = kernel.getBody().front();
  if (failed(verifyResNet50BottleneckDataflow(kernel, block, ops)))
    return failure();

  auto skipLoad = cast<LoadMatrixOp>(ops[0]);
  if (ops.size() == 172) {
    SmallVector<LoadMatrixOp, 2> w1Loads{cast<LoadMatrixOp>(ops[1]),
                                         cast<LoadMatrixOp>(ops[2])};
    SmallVector<LoadMatrixOp, 9> xWindowLoads;
    SmallVector<LoadMatrixOp, 36> w2Loads;
    SmallVector<LoadMatrixOp, 2> w3Loads;
    SmallVector<ZeroOp, 42> zeros{cast<ZeroOp>(ops[3]), cast<ZeroOp>(ops[4])};
    SmallVector<MmaOp, 56> mmas;
    SmallVector<MaxOp, 21> relus;

    for (int64_t i = 0; i < 9; ++i) {
      size_t base = static_cast<size_t>(5 + i * 17);
      xWindowLoads.push_back(cast<LoadMatrixOp>(ops[base]));
      zeros.push_back(cast<ZeroOp>(ops[base + 1]));
      zeros.push_back(cast<ZeroOp>(ops[base + 3]));
      zeros.push_back(cast<ZeroOp>(ops[base + 5]));
      zeros.push_back(cast<ZeroOp>(ops[base + 7]));
      mmas.push_back(cast<MmaOp>(ops[base + 2]));
      mmas.push_back(cast<MmaOp>(ops[base + 6]));
      mmas.push_back(cast<MmaOp>(ops[base + 10]));
      mmas.push_back(cast<MmaOp>(ops[base + 12]));
      mmas.push_back(cast<MmaOp>(ops[base + 14]));
      mmas.push_back(cast<MmaOp>(ops[base + 16]));
      relus.push_back(cast<MaxOp>(ops[base + 4]));
      relus.push_back(cast<MaxOp>(ops[base + 8]));
      w2Loads.push_back(cast<LoadMatrixOp>(ops[base + 9]));
      w2Loads.push_back(cast<LoadMatrixOp>(ops[base + 11]));
      w2Loads.push_back(cast<LoadMatrixOp>(ops[base + 13]));
      w2Loads.push_back(cast<LoadMatrixOp>(ops[base + 15]));
    }

    size_t tail = 158;
    zeros.push_back(cast<ZeroOp>(ops[tail]));
    zeros.push_back(cast<ZeroOp>(ops[tail + 2]));
    zeros.push_back(cast<ZeroOp>(ops[tail + 5]));
    zeros.push_back(cast<ZeroOp>(ops[tail + 10]));
    relus.push_back(cast<MaxOp>(ops[tail + 1]));
    relus.push_back(cast<MaxOp>(ops[tail + 3]));
    relus.push_back(cast<MaxOp>(ops[tail + 11]));
    w3Loads.push_back(cast<LoadMatrixOp>(ops[tail + 4]));
    w3Loads.push_back(cast<LoadMatrixOp>(ops[tail + 7]));
    mmas.push_back(cast<MmaOp>(ops[tail + 6]));
    mmas.push_back(cast<MmaOp>(ops[tail + 8]));
    auto residualAdd = cast<AddOp>(ops[tail + 9]);
    auto store = cast<StoreMatrixOp>(ops[tail + 12]);
    int64_t bottleneck = 32;
    return ExecutableResNet50BottleneckOpSequenceInfo{
        skipLoad, w1Loads, xWindowLoads, w2Loads, w3Loads,   zeros,
        mmas,     relus,   residualAdd,  store,   bottleneck};
  }

  SmallVector<LoadMatrixOp, 2> w1Loads{cast<LoadMatrixOp>(ops[1])};
  SmallVector<LoadMatrixOp, 9> xWindowLoads;
  SmallVector<LoadMatrixOp, 36> w2Loads;
  SmallVector<LoadMatrixOp, 2> w3Loads;
  SmallVector<ZeroOp, 42> zeros{cast<ZeroOp>(ops[2])};
  SmallVector<MmaOp, 56> mmas;
  SmallVector<MaxOp, 21> relus;

  for (int64_t i = 0; i < 9; ++i) {
    size_t base = static_cast<size_t>(3 + i * 7);
    xWindowLoads.push_back(cast<LoadMatrixOp>(ops[base]));
    zeros.push_back(cast<ZeroOp>(ops[base + 1]));
    zeros.push_back(cast<ZeroOp>(ops[base + 3]));
    mmas.push_back(cast<MmaOp>(ops[base + 2]));
    mmas.push_back(cast<MmaOp>(ops[base + 6]));
    relus.push_back(cast<MaxOp>(ops[base + 4]));
    w2Loads.push_back(cast<LoadMatrixOp>(ops[base + 5]));
  }

  size_t tail = 66;
  zeros.push_back(cast<ZeroOp>(ops[tail]));
  zeros.push_back(cast<ZeroOp>(ops[tail + 3]));
  zeros.push_back(cast<ZeroOp>(ops[tail + 6]));
  relus.push_back(cast<MaxOp>(ops[tail + 1]));
  relus.push_back(cast<MaxOp>(ops[tail + 7]));
  w3Loads.push_back(cast<LoadMatrixOp>(ops[tail + 2]));
  mmas.push_back(cast<MmaOp>(ops[tail + 4]));
  auto residualAdd = cast<AddOp>(ops[tail + 5]);
  auto store = cast<StoreMatrixOp>(ops[tail + 8]);
  int64_t bottleneck = 16;
  return ExecutableResNet50BottleneckOpSequenceInfo{
      skipLoad, w1Loads, xWindowLoads, w2Loads, w3Loads,   zeros,
      mmas,     relus,   residualAdd,  store,   bottleneck};
}

using BodyPredicateFn = bool (*)(ArrayRef<Operation *> ops);
using DataflowVerifierFn = LogicalResult (*)(KernelOp op, Block &block,
                                             ArrayRef<Operation *> ops);

struct ExecutableVerifierEntry {
  llvm::StringLiteral kind;
  BodyPredicateFn hasBody;
  llvm::StringLiteral bodyDiagnostic;
  DataflowVerifierFn verifyDataflow;
};

static constexpr ExecutableVerifierEntry kExecutableVerifierEntries[] = {
    {"add", hasAddBody, "kind add requires canonical add executable body",
     verifyAddDataflow},
    {"gemm", hasGemmExecutableBody,
     "kind gemm requires canonical or legalizable gemm executable body",
     verifyGemmDataflow},
    {"softmax", hasSoftmaxExecutableBody,
     "kind softmax requires canonical or legalizable softmax executable body",
     verifySoftmaxDataflow},
    {"convkxk", hasConvKxKExecutableBody,
     "kind convkxk requires canonical or legalizable convkxk executable body",
     verifyConvKxKDataflow},
    {"resnet_block", hasResNetBlockExecutableBody,
     "kind resnet_block requires canonical or legalizable residual block "
     "executable body",
     verifyResNetBlockDataflow},
    {"resnet50_bottleneck", hasResNet50BottleneckExecutableBody,
     "kind resnet50_bottleneck requires canonical or legalizable resnet50 "
     "bottleneck executable body",
     verifyResNet50BottleneckDataflow},
    {"sqrt", hasSqrtBody, "kind sqrt requires canonical sqrt executable body",
     verifySqrtDataflow},
    {"reduce_sum_all", hasReduceSumAllBody,
     "kind reduce_sum_all requires canonical reduce_sum_all executable body",
     verifyReduceSumAllDataflow},
    {"relu", hasReluBody, "kind relu requires canonical relu executable body",
     verifyReluDataflow},
    {"maximum", hasMaximumBody,
     "kind maximum requires canonical maximum executable body",
     verifyMaximumDataflow},
    {"reduce_sum_axis0", hasReduceSumAxisBody,
     "kind reduce_sum_axis0 requires canonical reduce_sum_axis executable body",
     verifyReduceSumAxisDataflow},
    {"reduce_sum_axis1", hasReduceSumAxisBody,
     "kind reduce_sum_axis1 requires canonical reduce_sum_axis executable body",
     verifyReduceSumAxisDataflow},
    {"broadcast_add", hasBroadcastAddBody,
     "kind broadcast_add requires canonical broadcast_add executable body",
     verifyBroadcastAddDataflow},
};

static const ExecutableVerifierEntry *
lookupExecutableVerifierEntry(StringRef kind) {
  for (const ExecutableVerifierEntry &entry : kExecutableVerifierEntries) {
    if (kind == entry.kind)
      return &entry;
  }
  return nullptr;
}

static constexpr llvm::StringLiteral kHighLevelLegalizableExecutableOpNames[] =
    {
        "rpu.compact_elementwise1d",
        "rpu.dot",
        "rpu.elementwise16_value_map",
        "rpu.softmax",
};

ArrayRef<llvm::StringLiteral>
mlir::rpu::exec::getHighLevelLegalizableExecutableOpNames() {
  return kHighLevelLegalizableExecutableOpNames;
}

bool mlir::rpu::exec::isHighLevelLegalizableExecutableOpName(
    llvm::StringRef opName) {
  for (llvm::StringLiteral name : getHighLevelLegalizableExecutableOpNames()) {
    if (opName == name)
      return true;
  }
  return false;
}

ExecutableOpLoweringClass
mlir::rpu::exec::getExecutableOpLoweringClass(Operation *op) {
  if (isa<LoadContigOp, StoreContigOp, LoadMatrixOp, StoreMatrixOp, ZeroOp,
          MmaOp, AddOp, MulOp, MaxOp, ReduceMaxAllOp, SubScalarOp, ExpOp,
          ReduceSumAllOp, ReciprocalOp, MulScalarOp, SqrtOp, FullOp, ReluOp,
          ReduceSumAxisOp, BroadcastAddOp, ReturnOp>(op))
    return ExecutableOpLoweringClass::Renderable;
  if (isHighLevelLegalizableExecutableOpName(op->getName().getStringRef()))
    return ExecutableOpLoweringClass::HighLevelLegalizable;
  return ExecutableOpLoweringClass::Unsupported;
}

bool mlir::rpu::exec::isGenericRenderableExecutableOp(Operation *op) {
  return getExecutableOpLoweringClass(op) ==
         ExecutableOpLoweringClass::Renderable;
}

bool mlir::rpu::exec::isHighLevelLegalizableExecutableOp(Operation *op) {
  return getExecutableOpLoweringClass(op) ==
         ExecutableOpLoweringClass::HighLevelLegalizable;
}

bool mlir::rpu::exec::isGenericLegalizableExecutableOp(Operation *op) {
  return isGenericRenderableExecutableOp(op) ||
         isa<CompactElementwise1DOp, Elementwise16ValueMapOp>(op);
}

LogicalResult
mlir::rpu::exec::verifyExecutableModuleRenderable(ModuleOp module) {
  LogicalResult result = success();
  module.walk([&](KernelOp kernel) {
    if (kernel.getBody().empty())
      return;
    Block &entry = kernel.getBody().front();
    for (Operation &bodyOp : entry.getOperations()) {
      if (getExecutableOpLoweringClass(&bodyOp) ==
          ExecutableOpLoweringClass::Renderable)
        continue;
      bodyOp.emitError(
          "RPU executable RPURC emission requires renderable executable ops; "
          "run high-level executable legalization before emission")
          << " (" << bodyOp.getName() << ")";
      result = failure();
    }
  });
  return result;
}

static LogicalResult verifyGenericBodyStructure(KernelOp op,
                                                ArrayRef<Operation *> ops) {
  if (ops.empty())
    return op.emitOpError("kind generic requires executable ops");
  if (!isa<ReturnOp>(ops.back()))
    return op.emitOpError("kind generic requires rpu.return terminator");
  for (size_t i = 0, e = ops.size(); i < e; ++i) {
    Operation *bodyOp = ops[i];
    if (!isGenericLegalizableExecutableOp(bodyOp))
      return bodyOp->emitError(
          "kind generic requires legalizable generic executable ops");
    if (i + 1 != e && isa<ReturnOp>(bodyOp))
      return bodyOp->emitError("kind generic rpu.return must terminate body");
  }
  return success();
}

static LogicalResult verifyKindBodySkeleton(KernelOp op, StringRef kind,
                                            ArrayRef<Operation *> ops) {
  if (kind == "generic")
    return verifyGenericBodyStructure(op, ops);
  if (const ExecutableVerifierEntry *entry =
          lookupExecutableVerifierEntry(kind)) {
    if (!entry->hasBody(ops))
      return op.emitOpError(entry->bodyDiagnostic);
  }
  return success();
}

FailureOr<SmallVector<Operation *>>
mlir::rpu::exec::getCanonicalExecutableKernelBodyOps(KernelOp kernel,
                                                     llvm::StringRef consumer) {
  FailureOr<SmallVector<Operation *>> ops =
      getNonEmptyExecutableKernelBodyOps(kernel, consumer);
  if (failed(ops))
    return failure();

  auto kindAttr = kernel->getAttrOfType<StringAttr>("kind");
  if (!kindAttr) {
    kernel.emitError() << consumer << " requires executable kind";
    return failure();
  }
  if (!isSupportedExecutableKernelKind(kindAttr.getValue())) {
    kernel.emitError() << consumer << " requires supported executable kind";
    return failure();
  }
  if (failed(verifyKindBodySkeleton(kernel, kindAttr.getValue(), *ops)))
    return failure();
  return *ops;
}

bool mlir::rpu::exec::isGenericRenderableExecutableKernelKind(StringRef kind) {
  return kind == "generic" || kind == "add" || kind == "gemm" ||
         kind == "softmax" || kind == "convkxk" || kind == "resnet_block" ||
         kind == "resnet50_bottleneck" || kind == "sqrt" ||
         kind == "reduce_sum_all" || kind == "relu" || kind == "maximum" ||
         kind == "reduce_sum_axis0" || kind == "reduce_sum_axis1" ||
         kind == "broadcast_add";
}

LogicalResult mlir::rpu::exec::verifyGenericRenderableExecutableOpSequence(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer) {
  auto kindAttr = kernel->getAttrOfType<StringAttr>("kind");
  if (!kindAttr)
    return kernel.emitError() << consumer << " requires executable kind";
  if (!isGenericRenderableExecutableKernelKind(kindAttr.getValue()))
    return kernel.emitError()
           << consumer << " requires generic-renderable executable kind";
  for (Operation *op : ops) {
    if (!isGenericRenderableExecutableOp(op))
      return op->emitError()
             << consumer << " requires generic-renderable executable ops";
  }
  return success();
}

bool mlir::rpu::exec::isExecutableCompactVectorBinaryOp(Operation *op) {
  return isa<AddOp, MulOp>(op);
}

static constexpr int64_t kElementwise16ValueMapOpcodeAdd = 0;
static constexpr int64_t kElementwise16ValueMapOpcodeMul = 1;

static LogicalResult
verifyCompactElementwise1DContract(Operation *op, OperandRange inputs,
                                   Type resultType, ArrayRef<int64_t> opcodes,
                                   ArrayRef<int64_t> lhsSlots,
                                   ArrayRef<int64_t> rhsSlots) {
  FailureOr<TileType> resultTile = requireTile(op, resultType);
  if (failed(resultTile))
    return failure();
  if ((*resultTile).getRank() != 1 || !(*resultTile).getElementType().isF16())
    return op->emitOpError("requires 1D f16 tile result");
  int64_t n = (*resultTile).getExtent();
  if (n <= 0 || n > 256 || n % 16 != 0)
    return op->emitOpError("requires positive 16-aligned tile extent <= 256");
  if (inputs.size() < 2 || inputs.size() > 4)
    return op->emitOpError("requires two to four inputs");
  if (opcodes.empty() || opcodes.size() > 3)
    return op->emitOpError("requires one to three opcodes");
  if (opcodes.size() != lhsSlots.size() || opcodes.size() != rhsSlots.size())
    return op->emitOpError("requires opcode/lhs/rhs arrays of equal length");
  for (Value input : inputs) {
    FailureOr<TileType> inputTile = requireTile(op, input.getType());
    if (failed(inputTile))
      return failure();
    if (*inputTile != *resultTile)
      return op->emitOpError("input and result tile types must match");
  }

  int64_t availableSlots = static_cast<int64_t>(inputs.size());
  for (size_t index = 0, end = opcodes.size(); index < end; ++index) {
    const int64_t opcode = opcodes[index];
    if (opcode != kElementwise16ValueMapOpcodeAdd &&
        opcode != kElementwise16ValueMapOpcodeMul)
      return op->emitOpError("opcode must be 0 add or 1 mul");
    const int64_t lhs = lhsSlots[index];
    const int64_t rhs = rhsSlots[index];
    if (lhs < 0 || rhs < 0 || lhs >= availableSlots || rhs >= availableSlots)
      return op->emitOpError("value slot is out of range");
    ++availableSlots;
  }
  return success();
}

static Value getElementwise1DBinaryLhs(Operation *op) {
  if (auto add = dyn_cast<AddOp>(op))
    return add.getLhs();
  return cast<MulOp>(op).getLhs();
}

static Value getElementwise1DBinaryRhs(Operation *op) {
  if (auto add = dyn_cast<AddOp>(op))
    return add.getRhs();
  return cast<MulOp>(op).getRhs();
}

static Value getElementwise1DBinaryResult(Operation *op) {
  if (auto add = dyn_cast<AddOp>(op))
    return add.getResult();
  return cast<MulOp>(op).getResult();
}

static bool sameVectorMaskInfo(const ExecutableVectorMaskInfo &lhs,
                               const ExecutableVectorMaskInfo &rhs) {
  return lhs.masked == rhs.masked && lhs.logicalN == rhs.logicalN &&
         lhs.blockN == rhs.blockN;
}

static void setExecutableVectorMaskAttrs(OpBuilder &builder, Operation *op,
                                         int64_t logicalN, int64_t blockN) {
  op->setAttr("masked", builder.getBoolAttr(true));
  op->setAttr("logical_n", builder.getI32IntegerAttr(logicalN));
  op->setAttr("block_n", builder.getI32IntegerAttr(blockN));
}

static void setExecutableMatrixOffsets(OpBuilder &builder, Operation *op,
                                       int64_t row, int64_t col = 0) {
  op->setAttr("row_offset", builder.getI32IntegerAttr(row));
  op->setAttr("col_offset", builder.getI32IntegerAttr(col));
}

FailureOr<ExecutableVectorMaskInfo>
mlir::rpu::exec::getExecutableVectorMaskInfo(Operation *op, int64_t n,
                                             llvm::StringRef consumer) {
  auto maskedAttr = op->getAttrOfType<BoolAttr>("masked");
  auto logicalNAttr = op->getAttrOfType<IntegerAttr>("logical_n");
  auto blockNAttr = op->getAttrOfType<IntegerAttr>("block_n");
  if (!maskedAttr || !maskedAttr.getValue()) {
    if (logicalNAttr || blockNAttr) {
      op->emitError() << consumer
                      << " unmasked op must not carry logical_n or block_n";
      return failure();
    }
    return ExecutableVectorMaskInfo{/*masked=*/false, n, n};
  }
  if (!logicalNAttr || !blockNAttr) {
    op->emitError() << consumer << " masked op requires logical_n and block_n";
    return failure();
  }
  if (!logicalNAttr.getType().isInteger(32) ||
      !blockNAttr.getType().isInteger(32)) {
    op->emitError() << consumer << " masked logical_n and block_n must be i32";
    return failure();
  }
  int64_t logicalN = logicalNAttr.getInt();
  int64_t blockN = blockNAttr.getInt();
  if (blockN != n) {
    op->emitError() << consumer << " masked block_n must equal n";
    return failure();
  }
  if (logicalN <= 0 || logicalN > blockN) {
    op->emitError() << consumer << " masked logical_n must be in (0, block_n]";
    return failure();
  }
  return ExecutableVectorMaskInfo{/*masked=*/true, logicalN, blockN};
}

LogicalResult mlir::rpu::exec::verifyExecutableCompactVectorLoad(
    LoadContigOp load, Value expectedPtr, int64_t expectedN,
    const ExecutableVectorMaskInfo &expectedMask, llvm::StringRef consumer) {
  if (load.getPtr() != expectedPtr)
    return load.emitError()
           << consumer << " requires expected vector load pointer";
  if (load.getN() != expectedN)
    return load.emitError() << consumer << " requires matching vector n";
  FailureOr<TileType> tile =
      requireTile(load.getOperation(), load.getResult().getType());
  if (failed(tile))
    return failure();
  if ((*tile).getRank() != 1 || (*tile).getExtent() != expectedN)
    return load.emitError() << consumer << " requires compact vector load tile";
  FailureOr<ExecutableVectorMaskInfo> mask =
      getExecutableVectorMaskInfo(load.getOperation(), expectedN, consumer);
  if (failed(mask) || !sameVectorMaskInfo(*mask, expectedMask))
    return load.emitError() << consumer << " requires matching vector masks";
  return success();
}

LogicalResult mlir::rpu::exec::verifyExecutableCompactVectorBinaryOp(
    Operation *op, int64_t expectedN, llvm::StringRef consumer) {
  if (!isExecutableCompactVectorBinaryOp(op))
    return op->emitError() << consumer << " requires compact vector Add/Mul op";
  Value lhs = getElementwise1DBinaryLhs(op);
  Value rhs = getElementwise1DBinaryRhs(op);
  Value result = getElementwise1DBinaryResult(op);
  FailureOr<TileType> lhsTile = requireTile(op, lhs.getType());
  FailureOr<TileType> rhsTile = requireTile(op, rhs.getType());
  FailureOr<TileType> resultTile = requireTile(op, result.getType());
  if (failed(lhsTile) || failed(rhsTile) || failed(resultTile))
    return failure();
  auto isExpected = [&](TileType tile) {
    return tile.getRank() == 1 && tile.getExtent() == expectedN;
  };
  if (!isExpected(*lhsTile) || !isExpected(*rhsTile) ||
      !isExpected(*resultTile))
    return op->emitError() << consumer
                           << " requires compact vector binary tile";
  return success();
}

LogicalResult mlir::rpu::exec::verifyExecutableCompactVectorStore(
    StoreContigOp store, Value expectedPtr, Value expectedValue,
    int64_t expectedN, const ExecutableVectorMaskInfo &expectedMask,
    llvm::StringRef consumer) {
  if (store.getPtr() != expectedPtr)
    return store.emitError()
           << consumer << " requires expected vector store pointer";
  if (store.getValue() != expectedValue)
    return store.emitError()
           << consumer << " requires store consumes final binary result";
  if (store.getN() != expectedN)
    return store.emitError() << consumer << " requires matching store n";
  FailureOr<TileType> tile =
      requireTile(store.getOperation(), store.getValue().getType());
  if (failed(tile))
    return failure();
  if ((*tile).getRank() != 1 || (*tile).getExtent() != expectedN)
    return store.emitError()
           << consumer << " requires compact vector store tile";
  FailureOr<ExecutableVectorMaskInfo> mask =
      getExecutableVectorMaskInfo(store.getOperation(), expectedN, consumer);
  if (failed(mask) || !sameVectorMaskInfo(*mask, expectedMask))
    return store.emitError() << consumer << " requires matching store mask";
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableConvKxKBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableConvKxKBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (!isSupportedExecutableConvKxKKernelSize(spec.kernelSize))
    return kernel.emitError() << consumer << " requires supported kernel_size";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";

  unsigned argIndices[] = {spec.outputArgIndex, spec.inputArgIndex,
                           spec.weightArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 2> tileShape{16, 16};
  Type tile16x16 = TileType::get(builder.getContext(), tileShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  Value acc;
  bool hasAcc = false;
  for (int64_t ky = 0; ky < spec.kernelSize; ++ky) {
    for (int64_t kx = 0; kx < spec.kernelSize; ++kx) {
      int64_t kernelIndex = ky * spec.kernelSize + kx;
      int64_t inputRow = ky * 16 + kx;
      int64_t weightRow = kernelIndex * 16;
      auto x = builder.create<LoadMatrixOp>(
          loc, tile16x16, entry.getArgument(spec.inputArgIndex), 16, 16);
      setExecutableMatrixOffsets(builder, x.getOperation(), inputRow);
      auto w = builder.create<LoadMatrixOp>(
          loc, tile16x16, entry.getArgument(spec.weightArgIndex), 16, 16);
      setExecutableMatrixOffsets(builder, w.getOperation(), weightRow);
      if (!hasAcc) {
        auto dot =
            builder.create<DotOp>(loc, tile16x16, x.getResult(), w.getResult());
        acc = dot.getResult();
        hasAcc = true;
        continue;
      }
      auto mma = builder.create<MmaOp>(loc, tile16x16, x.getResult(),
                                       w.getResult(), acc);
      acc = mma.getResult();
    }
  }
  builder.create<StoreMatrixOp>(loc, entry.getArgument(spec.outputArgIndex),
                                acc, 16, 16);
  builder.create<ReturnOp>(loc);

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableResNetBlockBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableResNetBlockBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.hidden != 16 && spec.hidden != 32)
    return kernel.emitError() << consumer << " requires hidden in {16,32}";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";

  unsigned argIndices[] = {spec.outputArgIndex, spec.xArgIndex, spec.w1ArgIndex,
                           spec.w2ArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 2> shape16x16{16, 16};
  llvm::SmallVector<int64_t, 2> shape16xHidden{16, spec.hidden};
  llvm::SmallVector<int64_t, 2> shapeHiddenx16{spec.hidden, 16};
  Type tile16x16 = TileType::get(builder.getContext(), shape16x16, f16);
  Type tile16xHidden = TileType::get(builder.getContext(), shape16xHidden, f16);
  Type tileHiddenx16 = TileType::get(builder.getContext(), shapeHiddenx16, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  Value out = entry.getArgument(spec.outputArgIndex);
  Value xPtr = entry.getArgument(spec.xArgIndex);
  Value w1Ptr = entry.getArgument(spec.w1ArgIndex);
  Value w2Ptr = entry.getArgument(spec.w2ArgIndex);

  auto x = builder.create<LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
  if (spec.hidden == 32) {
    auto w1Lo = builder.create<LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w1Lo.getOperation(), 0, 0);
    auto conv1Lo =
        builder.create<DotOp>(loc, tile16x16, x.getResult(), w1Lo.getResult());
    auto zero1Lo = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto relu1Lo = builder.create<MaxOp>(loc, tile16x16, conv1Lo.getResult(),
                                         zero1Lo.getResult());

    auto w1Hi = builder.create<LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w1Hi.getOperation(), 0, 16);
    auto conv1Hi =
        builder.create<DotOp>(loc, tile16x16, x.getResult(), w1Hi.getResult());
    auto zero1Hi = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto relu1Hi = builder.create<MaxOp>(loc, tile16x16, conv1Hi.getResult(),
                                         zero1Hi.getResult());

    auto w2Lo = builder.create<LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w2Lo.getOperation(), 0, 0);
    Value conv2Acc = builder
                         .create<DotOp>(loc, tile16x16, relu1Lo.getResult(),
                                        w2Lo.getResult())
                         .getResult();
    auto w2Hi = builder.create<LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w2Hi.getOperation(), 16, 0);
    conv2Acc = builder
                   .create<MmaOp>(loc, tile16x16, relu1Hi.getResult(),
                                  w2Hi.getResult(), conv2Acc)
                   .getResult();

    auto residual =
        builder.create<AddOp>(loc, tile16x16, conv2Acc, x.getResult());
    auto zero2 = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto outTile = builder.create<MaxOp>(loc, tile16x16, residual.getResult(),
                                         zero2.getResult());
    builder.create<StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
    builder.create<ReturnOp>(loc);
  } else {
    auto w1 = builder.create<LoadMatrixOp>(loc, tile16xHidden, w1Ptr, 16,
                                           spec.hidden);
    auto conv1 = builder.create<DotOp>(loc, tile16xHidden, x.getResult(),
                                       w1.getResult());
    auto zero1 = builder.create<ZeroOp>(loc, tile16xHidden, 16, spec.hidden);
    auto relu1 = builder.create<MaxOp>(loc, tile16xHidden, conv1.getResult(),
                                       zero1.getResult());
    auto w2 = builder.create<LoadMatrixOp>(loc, tileHiddenx16, w2Ptr,
                                           spec.hidden, 16);
    auto conv2 = builder.create<DotOp>(loc, tile16x16, relu1.getResult(),
                                       w2.getResult());
    auto residual =
        builder.create<AddOp>(loc, tile16x16, conv2.getResult(), x.getResult());
    auto zero2 = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto outTile = builder.create<MaxOp>(loc, tile16x16, residual.getResult(),
                                         zero2.getResult());
    builder.create<StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
    builder.create<ReturnOp>(loc);
  }

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableResNet50BottleneckBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableResNet50BottleneckBodyBuildSpec &spec,
    llvm::StringRef consumer) {
  if (spec.bottleneck != 16 && spec.bottleneck != 32)
    return kernel.emitError() << consumer << " requires bottleneck in {16,32}";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";

  unsigned argIndices[] = {spec.outputArgIndex, spec.inputArgIndex,
                           spec.w1ArgIndex, spec.w2ArgIndex, spec.w3ArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 2> shape16x16{16, 16};
  Type tile16x16 = TileType::get(builder.getContext(), shape16x16, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  Value out = entry.getArgument(spec.outputArgIndex);
  Value xPtr = entry.getArgument(spec.inputArgIndex);
  Value w1Ptr = entry.getArgument(spec.w1ArgIndex);
  Value w2Ptr = entry.getArgument(spec.w2ArgIndex);
  Value w3Ptr = entry.getArgument(spec.w3ArgIndex);

  auto xSkip = builder.create<LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
  setExecutableMatrixOffsets(builder, xSkip.getOperation(), 0);
  if (spec.bottleneck == 32) {
    auto w1Lo = builder.create<LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w1Lo.getOperation(), 0, 0);
    auto w1Hi = builder.create<LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w1Hi.getOperation(), 0, 16);
    Value conv2AccLo =
        builder.create<ZeroOp>(loc, tile16x16, 16, 16).getResult();
    Value conv2AccHi =
        builder.create<ZeroOp>(loc, tile16x16, 16, 16).getResult();

    for (int64_t ky = 0; ky < 3; ++ky) {
      for (int64_t kx = 0; kx < 3; ++kx) {
        int64_t kernelIndex = ky * 3 + kx;
        int64_t xRow = ky * 16 + kx;
        int64_t w2Row = kernelIndex * 32;
        auto x = builder.create<LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
        setExecutableMatrixOffsets(builder, x.getOperation(), xRow);
        auto conv1Lo = builder.create<DotOp>(loc, tile16x16, x.getResult(),
                                             w1Lo.getResult());
        auto zero1Lo = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
        auto relu1Lo = builder.create<MaxOp>(
            loc, tile16x16, conv1Lo.getResult(), zero1Lo.getResult());
        auto conv1Hi = builder.create<DotOp>(loc, tile16x16, x.getResult(),
                                             w1Hi.getResult());
        auto zero1Hi = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
        auto relu1Hi = builder.create<MaxOp>(
            loc, tile16x16, conv1Hi.getResult(), zero1Hi.getResult());

        auto w2LoLo =
            builder.create<LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setExecutableMatrixOffsets(builder, w2LoLo.getOperation(), w2Row, 0);
        conv2AccLo = builder
                         .create<MmaOp>(loc, tile16x16, relu1Lo.getResult(),
                                        w2LoLo.getResult(), conv2AccLo)
                         .getResult();
        auto w2HiLo =
            builder.create<LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setExecutableMatrixOffsets(builder, w2HiLo.getOperation(), w2Row + 16,
                                   0);
        conv2AccLo = builder
                         .create<MmaOp>(loc, tile16x16, relu1Hi.getResult(),
                                        w2HiLo.getResult(), conv2AccLo)
                         .getResult();
        auto w2LoHi =
            builder.create<LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setExecutableMatrixOffsets(builder, w2LoHi.getOperation(), w2Row, 16);
        conv2AccHi = builder
                         .create<MmaOp>(loc, tile16x16, relu1Lo.getResult(),
                                        w2LoHi.getResult(), conv2AccHi)
                         .getResult();
        auto w2HiHi =
            builder.create<LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setExecutableMatrixOffsets(builder, w2HiHi.getOperation(), w2Row + 16,
                                   16);
        conv2AccHi = builder
                         .create<MmaOp>(loc, tile16x16, relu1Hi.getResult(),
                                        w2HiHi.getResult(), conv2AccHi)
                         .getResult();
      }
    }

    auto zero2Lo = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto relu2Lo =
        builder.create<MaxOp>(loc, tile16x16, conv2AccLo, zero2Lo.getResult());
    auto zero2Hi = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto relu2Hi =
        builder.create<MaxOp>(loc, tile16x16, conv2AccHi, zero2Hi.getResult());
    auto w3Lo = builder.create<LoadMatrixOp>(loc, tile16x16, w3Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w3Lo.getOperation(), 0, 0);
    Value conv3 = builder
                      .create<DotOp>(loc, tile16x16, relu2Lo.getResult(),
                                     w3Lo.getResult())
                      .getResult();
    auto w3Hi = builder.create<LoadMatrixOp>(loc, tile16x16, w3Ptr, 16, 16);
    setExecutableMatrixOffsets(builder, w3Hi.getOperation(), 16, 0);
    conv3 = builder
                .create<MmaOp>(loc, tile16x16, relu2Hi.getResult(),
                               w3Hi.getResult(), conv3)
                .getResult();
    auto residual =
        builder.create<AddOp>(loc, tile16x16, conv3, xSkip.getResult());
    auto zero3 = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto outTile = builder.create<MaxOp>(loc, tile16x16, residual.getResult(),
                                         zero3.getResult());
    builder.create<StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
    builder.create<ReturnOp>(loc);
  } else {
    auto w1 = builder.create<LoadMatrixOp>(loc, tile16x16, w1Ptr, 16, 16);
    Value conv2Acc = builder.create<ZeroOp>(loc, tile16x16, 16, 16).getResult();

    for (int64_t ky = 0; ky < 3; ++ky) {
      for (int64_t kx = 0; kx < 3; ++kx) {
        int64_t kernelIndex = ky * 3 + kx;
        int64_t xRow = ky * 16 + kx;
        int64_t w2Row = kernelIndex * 16;
        auto x = builder.create<LoadMatrixOp>(loc, tile16x16, xPtr, 16, 16);
        setExecutableMatrixOffsets(builder, x.getOperation(), xRow);
        auto conv1 = builder.create<DotOp>(loc, tile16x16, x.getResult(),
                                           w1.getResult());
        auto zero1 = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
        auto relu1 = builder.create<MaxOp>(loc, tile16x16, conv1.getResult(),
                                           zero1.getResult());
        auto w2 = builder.create<LoadMatrixOp>(loc, tile16x16, w2Ptr, 16, 16);
        setExecutableMatrixOffsets(builder, w2.getOperation(), w2Row);
        conv2Acc = builder
                       .create<MmaOp>(loc, tile16x16, relu1.getResult(),
                                      w2.getResult(), conv2Acc)
                       .getResult();
      }
    }

    auto zero2 = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto relu2 =
        builder.create<MaxOp>(loc, tile16x16, conv2Acc, zero2.getResult());
    auto w3 = builder.create<LoadMatrixOp>(loc, tile16x16, w3Ptr, 16, 16);
    auto conv3 = builder.create<DotOp>(loc, tile16x16, relu2.getResult(),
                                       w3.getResult());
    auto residual = builder.create<AddOp>(loc, tile16x16, conv3.getResult(),
                                          xSkip.getResult());
    auto zero3 = builder.create<ZeroOp>(loc, tile16x16, 16, 16);
    auto outTile = builder.create<MaxOp>(loc, tile16x16, residual.getResult(),
                                         zero3.getResult());
    builder.create<StoreMatrixOp>(loc, out, outTile.getResult(), 16, 16);
    builder.create<ReturnOp>(loc);
  }

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::expandExecutableDotOp(OpBuilder &builder,
                                                     DotOp dot,
                                                     llvm::StringRef consumer) {
  (void)consumer;
  if (failed(dot.verify()))
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(dot);
  Location loc = dot.getLoc();
  auto tileType = cast<TileType>(dot.getResult().getType());
  ArrayRef<int64_t> shape = tileType.getShape();
  auto zero = builder.create<ZeroOp>(loc, tileType, shape[0], shape[1]);
  auto mma = builder.create<MmaOp>(loc, tileType, dot.getLhs(), dot.getRhs(),
                                   zero.getResult());

  dot.getResult().replaceAllUsesWith(mma.getResult());
  dot->erase();
  return success();
}

LogicalResult mlir::rpu::exec::expandExecutableSoftmaxOp(
    OpBuilder &builder, SoftmaxOp softmax, llvm::StringRef consumer) {
  (void)consumer;
  if (failed(softmax.verify()))
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(softmax);
  Location loc = softmax.getLoc();
  Type f16 = builder.getF16Type();
  Type tileType = softmax.getResult().getType();
  Value input = softmax.getInput();

  auto m = builder.create<ReduceMaxAllOp>(loc, f16, input);
  auto shifted =
      builder.create<SubScalarOp>(loc, tileType, input, m.getResult());
  auto e = builder.create<ExpOp>(loc, tileType, shifted.getResult());
  auto s = builder.create<ReduceSumAllOp>(loc, f16, e.getResult());
  auto invS = builder.create<ReciprocalOp>(loc, f16, s.getResult());
  auto y = builder.create<MulScalarOp>(loc, tileType, e.getResult(),
                                       invS.getResult());

  softmax.getResult().replaceAllUsesWith(y.getResult());
  softmax->erase();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableSoftmaxBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableSoftmaxBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.nvec <= 0)
    return kernel.emitError() << consumer << " requires positive nvec";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";

  unsigned argIndices[] = {spec.outputArgIndex, spec.inputArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{spec.nvec};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  auto x = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.inputArgIndex), spec.nvec);
  auto y = builder.create<SoftmaxOp>(loc, tileType, x.getResult());
  builder.create<StoreContigOp>(loc, entry.getArgument(spec.outputArgIndex),
                                y.getResult(), spec.nvec);
  builder.create<ReturnOp>(loc);

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableGemmBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableGemmBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.m <= 0 || spec.n <= 0 || spec.k <= 0)
    return kernel.emitError() << consumer << " requires positive m/n/k";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";

  unsigned argIndices[] = {spec.outputArgIndex, spec.lhsArgIndex,
                           spec.rhsArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 2> lhsShape{spec.m, spec.k};
  llvm::SmallVector<int64_t, 2> rhsShape{spec.k, spec.n};
  llvm::SmallVector<int64_t, 2> accShape{spec.m, spec.n};
  Type tileMK = TileType::get(builder.getContext(), lhsShape, f16);
  Type tileKN = TileType::get(builder.getContext(), rhsShape, f16);
  Type tileMN = TileType::get(builder.getContext(), accShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  auto lhs = builder.create<LoadMatrixOp>(
      loc, tileMK, entry.getArgument(spec.lhsArgIndex), spec.m, spec.k);
  auto rhs = builder.create<LoadMatrixOp>(
      loc, tileKN, entry.getArgument(spec.rhsArgIndex), spec.k, spec.n);
  auto dot =
      builder.create<DotOp>(loc, tileMN, lhs.getResult(), rhs.getResult());
  builder.create<StoreMatrixOp>(loc, entry.getArgument(spec.outputArgIndex),
                                dot.getResult(), spec.m, spec.n);
  builder.create<ReturnOp>(loc);

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableAddBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableAddBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.n <= 0)
    return kernel.emitError() << consumer << " requires positive vector n";
  if (spec.masked) {
    if (spec.logicalN <= 0 || spec.logicalN > spec.n)
      return kernel.emitError() << consumer << " requires 0 < logical_n <= n";
  } else if (spec.logicalN != spec.n) {
    return kernel.emitError() << consumer << " requires logical_n == n";
  }
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";

  unsigned argIndices[] = {spec.outputArgIndex, spec.lhsArgIndex,
                           spec.rhsArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{spec.n};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  auto lhs = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.lhsArgIndex), spec.n);
  auto rhs = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.rhsArgIndex), spec.n);
  auto sum =
      builder.create<AddOp>(loc, tileType, lhs.getResult(), rhs.getResult());
  auto store = builder.create<StoreContigOp>(
      loc, entry.getArgument(spec.outputArgIndex), sum.getResult(), spec.n);
  if (spec.masked) {
    setExecutableVectorMaskAttrs(builder, lhs.getOperation(), spec.logicalN,
                                 spec.n);
    setExecutableVectorMaskAttrs(builder, rhs.getOperation(), spec.logicalN,
                                 spec.n);
    setExecutableVectorMaskAttrs(builder, store.getOperation(), spec.logicalN,
                                 spec.n);
  }
  builder.create<ReturnOp>(loc);

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  if (failed(getExecutableAddOpSequenceInfo(kernel, *bodyOps, consumer)))
    return failure();
  return success();
}

static LogicalResult
verifyElementwise16ValueMapContract(Operation *op, OperandRange inputs,
                                    Type resultType, ArrayRef<int64_t> opcodes,
                                    ArrayRef<int64_t> lhsSlots,
                                    ArrayRef<int64_t> rhsSlots) {
  FailureOr<TileType> resultTile = requireTile(op, resultType);
  if (failed(resultTile))
    return failure();
  if ((*resultTile).getRank() != 1 || (*resultTile).getExtent() != 16 ||
      !(*resultTile).getElementType().isF16())
    return op->emitOpError("requires !rpu.tile<16xf16> result");
  if (inputs.size() < 2 || inputs.size() > 4)
    return op->emitOpError("requires two to four inputs");
  if (opcodes.size() < 1 || opcodes.size() > 3)
    return op->emitOpError("requires one to three opcodes");
  if (opcodes.size() != lhsSlots.size() || opcodes.size() != rhsSlots.size())
    return op->emitOpError("requires opcode/lhs/rhs arrays of equal length");
  if (inputs.size() != opcodes.size() + 1)
    return op->emitOpError("requires input count equal opcode count plus one");
  for (Value input : inputs) {
    FailureOr<TileType> inputTile = requireTile(op, input.getType());
    if (failed(inputTile))
      return failure();
    if (*inputTile != *resultTile)
      return op->emitOpError("input and result tile types must match");
  }

  int64_t availableSlots = static_cast<int64_t>(inputs.size());
  for (size_t index = 0, end = opcodes.size(); index < end; ++index) {
    const int64_t opcode = opcodes[index];
    if (opcode != kElementwise16ValueMapOpcodeAdd &&
        opcode != kElementwise16ValueMapOpcodeMul)
      return op->emitOpError("opcode must be 0 add or 1 mul");
    const int64_t lhs = lhsSlots[index];
    const int64_t rhs = rhsSlots[index];
    if (lhs < 0 || rhs < 0 || lhs >= availableSlots || rhs >= availableSlots)
      return op->emitOpError("value-map slot is out of range");
    if (index > 0) {
      const int64_t previousResultSlot =
          static_cast<int64_t>(inputs.size() + index - 1);
      if (lhs != previousResultSlot && rhs != previousResultSlot)
        return op->emitOpError("requires linear previous-result chain");
    }
    ++availableSlots;
  }
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableSqrtBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableSqrtBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.nvec <= 0)
    return kernel.emitError() << consumer << " requires positive nvec";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  unsigned argIndices[] = {spec.outputArgIndex, spec.inputArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }
  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{spec.nvec};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  auto x = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.inputArgIndex), spec.nvec);
  auto y = builder.create<SqrtOp>(loc, tileType, x.getResult());
  builder.create<StoreContigOp>(loc, entry.getArgument(spec.outputArgIndex),
                                y.getResult(), spec.nvec);
  builder.create<ReturnOp>(loc);
  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableReduceSumAllBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableReduceSumAllBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.nvec <= 0)
    return kernel.emitError() << consumer << " requires positive nvec";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  unsigned argIndices[] = {spec.outputArgIndex, spec.inputArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }
  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{spec.nvec};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  auto x = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.inputArgIndex), spec.nvec);
  auto s = builder.create<ReduceSumAllOp>(loc, f16, x.getResult());
  auto v = builder.create<FullOp>(loc, tileType, s.getResult(), spec.nvec);
  builder.create<StoreContigOp>(loc, entry.getArgument(spec.outputArgIndex),
                                v.getResult(), spec.nvec);
  builder.create<ReturnOp>(loc);
  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableReluBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableReluBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.nvec <= 0)
    return kernel.emitError() << consumer << " requires positive nvec";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  unsigned argIndices[] = {spec.outputArgIndex, spec.inputArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }
  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{spec.nvec};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  auto x = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.inputArgIndex), spec.nvec);
  auto y = builder.create<ReluOp>(loc, tileType, x.getResult());
  builder.create<StoreContigOp>(loc, entry.getArgument(spec.outputArgIndex),
                                y.getResult(), spec.nvec);
  builder.create<ReturnOp>(loc);
  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableMaximumBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableMaximumBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.n <= 0)
    return kernel.emitError() << consumer << " requires positive vector n";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  unsigned argIndices[] = {spec.outputArgIndex, spec.lhsArgIndex,
                           spec.rhsArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }
  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{spec.n};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  auto lhs = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.lhsArgIndex), spec.n);
  auto rhs = builder.create<LoadContigOp>(
      loc, tileType, entry.getArgument(spec.rhsArgIndex), spec.n);
  auto m =
      builder.create<MaxOp>(loc, tileType, lhs.getResult(), rhs.getResult());
  builder.create<StoreContigOp>(loc, entry.getArgument(spec.outputArgIndex),
                                m.getResult(), spec.n);
  builder.create<ReturnOp>(loc);
  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableReduceSumAxisBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableReduceSumAxisBodyBuildSpec &spec,
    llvm::StringRef consumer) {
  if (spec.rows <= 0 || spec.cols <= 0)
    return kernel.emitError() << consumer << " requires positive rows and cols";
  if (spec.axis != 0 && spec.axis != 1)
    return kernel.emitError() << consumer << " requires axis in {0, 1}";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  unsigned argIndices[] = {spec.outputArgIndex, spec.inputArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }
  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 2> matShape{spec.rows, spec.cols};
  Type matTileType = TileType::get(builder.getContext(), matShape, f16);
  int64_t outDim = spec.axis == 0 ? spec.cols : spec.rows;
  llvm::SmallVector<int64_t, 1> outShape{outDim};
  Type vecTileType = TileType::get(builder.getContext(), outShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  auto load = builder.create<LoadMatrixOp>(
      loc, matTileType, entry.getArgument(spec.inputArgIndex), spec.rows,
      spec.cols);
  auto reduce = builder.create<ReduceSumAxisOp>(loc, vecTileType,
                                                load.getResult(), spec.axis);
  builder.create<StoreContigOp>(loc, entry.getArgument(spec.outputArgIndex),
                                reduce.getResult(), outDim);
  builder.create<ReturnOp>(loc);
  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableBroadcastAddBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableBroadcastAddBodyBuildSpec &spec, llvm::StringRef consumer) {
  if (spec.rows <= 0 || spec.cols <= 0)
    return kernel.emitError() << consumer << " requires positive rows and cols";
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  unsigned argIndices[] = {spec.outputArgIndex, spec.lhsArgIndex,
                           spec.rhsArgIndex};
  for (unsigned index : argIndices) {
    if (index >= entry.getNumArguments())
      return kernel.emitError() << consumer << " arg index is out of range";
  }
  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 2> matShape{spec.rows, spec.cols};
  Type matTileType = TileType::get(builder.getContext(), matShape, f16);
  llvm::SmallVector<int64_t, 1> vecShape{spec.rows};
  Type vecTileType = TileType::get(builder.getContext(), vecShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);
  auto lhs = builder.create<LoadMatrixOp>(loc, matTileType,
                                          entry.getArgument(spec.lhsArgIndex),
                                          spec.rows, spec.cols);
  auto rhs = builder.create<LoadContigOp>(
      loc, vecTileType, entry.getArgument(spec.rhsArgIndex), spec.rows);
  auto sum = builder.create<BroadcastAddOp>(loc, matTileType, lhs.getResult(),
                                            rhs.getResult());
  builder.create<StoreMatrixOp>(loc, entry.getArgument(spec.outputArgIndex),
                                sum.getResult(), spec.rows, spec.cols);
  builder.create<ReturnOp>(loc);
  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::expandExecutableElementwise16ValueMapOp(
    OpBuilder &builder, Elementwise16ValueMapOp map, llvm::StringRef consumer) {
  (void)consumer;
  if (failed(verifyElementwise16ValueMapContract(
          map.getOperation(), map.getInputs(), map.getResult().getType(),
          map.getOpcodes(), map.getLhsSlots(), map.getRhsSlots())))
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(map);
  llvm::SmallVector<Value, 8> values;
  for (Value input : map.getInputs())
    values.push_back(input);

  ArrayRef<int64_t> opcodes = map.getOpcodes();
  ArrayRef<int64_t> lhsSlots = map.getLhsSlots();
  ArrayRef<int64_t> rhsSlots = map.getRhsSlots();
  Type tileType = map.getResult().getType();
  Location opLoc = map.getLoc();
  for (size_t index = 0, end = opcodes.size(); index < end; ++index) {
    Value lhs = values[static_cast<unsigned>(lhsSlots[index])];
    Value rhs = values[static_cast<unsigned>(rhsSlots[index])];
    if (opcodes[index] == kElementwise16ValueMapOpcodeAdd) {
      auto result = builder.create<AddOp>(opLoc, tileType, lhs, rhs);
      values.push_back(result.getResult());
    } else {
      auto result = builder.create<MulOp>(opLoc, tileType, lhs, rhs);
      values.push_back(result.getResult());
    }
  }

  map.getResult().replaceAllUsesWith(values.back());
  map->erase();
  return success();
}

LogicalResult mlir::rpu::exec::expandExecutableCompactElementwise1DOp(
    OpBuilder &builder, CompactElementwise1DOp op, llvm::StringRef consumer) {
  (void)consumer;
  if (failed(verifyCompactElementwise1DContract(
          op.getOperation(), op.getInputs(), op.getResult().getType(),
          op.getOpcodes(), op.getLhsSlots(), op.getRhsSlots())))
    return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(op);
  llvm::SmallVector<Value, 8> values;
  for (Value input : op.getInputs())
    values.push_back(input);

  ArrayRef<int64_t> opcodes = op.getOpcodes();
  ArrayRef<int64_t> lhsSlots = op.getLhsSlots();
  ArrayRef<int64_t> rhsSlots = op.getRhsSlots();
  Type tileType = op.getResult().getType();
  Location loc = op.getLoc();
  for (size_t index = 0, end = opcodes.size(); index < end; ++index) {
    Value lhs = values[static_cast<unsigned>(lhsSlots[index])];
    Value rhs = values[static_cast<unsigned>(rhsSlots[index])];
    if (opcodes[index] == kElementwise16ValueMapOpcodeAdd) {
      auto result = builder.create<AddOp>(loc, tileType, lhs, rhs);
      values.push_back(result.getResult());
    } else {
      auto result = builder.create<MulOp>(loc, tileType, lhs, rhs);
      values.push_back(result.getResult());
    }
  }

  op.getResult().replaceAllUsesWith(values.back());
  op->erase();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableElementwise16ValueMapBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableElementwise16ValueMapBuildSpec &spec,
    llvm::StringRef consumer) {
  if (spec.outputArgIndex != 0)
    return kernel.emitError() << consumer << " requires output arg0";
  if (spec.inputArgIndices.size() < 2 || spec.inputArgIndices.size() > 4)
    return kernel.emitError()
           << consumer << " requires two to four ordered inputs";
  if (spec.ops.size() < 1 || spec.ops.size() > 3)
    return kernel.emitError() << consumer << " requires one to three ops";
  if (spec.inputArgIndices.size() != spec.ops.size() + 1)
    return kernel.emitError()
           << consumer << " requires input count equal op count plus one";
  for (size_t i = 0, e = spec.inputArgIndices.size(); i < e; ++i) {
    if (spec.inputArgIndices[i] != static_cast<unsigned>(i + 1))
      return kernel.emitError() << consumer << " requires ordered input args";
  }
  for (size_t opIndex = 1; opIndex < spec.ops.size(); ++opIndex) {
    const int64_t previousResultSlot =
        static_cast<int64_t>(spec.inputArgIndices.size() + opIndex - 1);
    const ExecutableCompactVectorBinaryBuildOp &op = spec.ops[opIndex];
    if (op.lhs != previousResultSlot && op.rhs != previousResultSlot)
      return kernel.emitError()
             << consumer << " requires linear previous-result chain";
  }
  int64_t availableSlots = static_cast<int64_t>(spec.inputArgIndices.size());
  for (const ExecutableCompactVectorBinaryBuildOp &op : spec.ops) {
    if (op.lhs < 0 || op.rhs < 0 || op.lhs >= availableSlots ||
        op.rhs >= availableSlots)
      return kernel.emitError()
             << consumer << " value-map slot is out of range";
    ++availableSlots;
  }
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  if (entry.getNumArguments() != spec.inputArgIndices.size() + 1)
    return kernel.emitError()
           << consumer << " requires output arg plus ordered input args";

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{16};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  llvm::SmallVector<Value, 8> valueSlots;
  valueSlots.reserve(spec.inputArgIndices.size() + spec.ops.size());
  for (unsigned argIndex : spec.inputArgIndices) {
    auto load = builder.create<LoadContigOp>(loc, tileType,
                                             entry.getArgument(argIndex), 16);
    valueSlots.push_back(load.getResult());
  }

  llvm::SmallVector<int64_t, 3> opcodeValues;
  llvm::SmallVector<int64_t, 3> lhsSlotValues;
  llvm::SmallVector<int64_t, 3> rhsSlotValues;
  for (const ExecutableCompactVectorBinaryBuildOp &op : spec.ops) {
    opcodeValues.push_back(op.opcode == ExecutableCompactVectorBinaryOpcode::Add
                               ? kElementwise16ValueMapOpcodeAdd
                               : kElementwise16ValueMapOpcodeMul);
    lhsSlotValues.push_back(op.lhs);
    rhsSlotValues.push_back(op.rhs);
  }

  auto valueMap = builder.create<Elementwise16ValueMapOp>(
      loc, tileType, valueSlots, builder.getDenseI64ArrayAttr(opcodeValues),
      builder.getDenseI64ArrayAttr(lhsSlotValues),
      builder.getDenseI64ArrayAttr(rhsSlotValues));

  builder.create<StoreContigOp>(loc, entry.getArgument(0), valueMap.getResult(),
                                16);
  builder.create<ReturnOp>(loc);

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::buildExecutableCompactElementwise1DBody(
    OpBuilder &builder, Location loc, KernelOp kernel,
    const ExecutableCompactElementwise1DBuildSpec &spec,
    llvm::StringRef consumer) {
  if (spec.n <= 0)
    return kernel.emitError() << consumer << " requires positive vector n";
  if (spec.inputArgIndices.size() < 2 || spec.inputArgIndices.size() > 4)
    return kernel.emitError()
           << consumer << " requires two to four compact vector inputs";
  if (spec.ops.empty() || spec.ops.size() > 3)
    return kernel.emitError()
           << consumer << " requires one to three compact vector binary ops";
  if (spec.masked) {
    if (spec.logicalN <= 0 || spec.logicalN > spec.n)
      return kernel.emitError() << consumer << " requires 0 < logical_n <= n";
  } else if (spec.logicalN != spec.n) {
    return kernel.emitError() << consumer << " requires logical_n == n";
  }
  if (kernel.getBody().empty())
    return kernel.emitError() << consumer << " requires executable body";
  Block &entry = kernel.getBody().front();
  if (!entry.empty())
    return kernel.emitError()
           << consumer << " requires empty body before materialization";
  if (spec.outputArgIndex >= entry.getNumArguments())
    return kernel.emitError()
           << consumer << " output arg index is out of range";
  for (unsigned argIndex : spec.inputArgIndices) {
    if (argIndex >= entry.getNumArguments())
      return kernel.emitError()
             << consumer << " input arg index is out of range";
  }

  Type f16 = builder.getF16Type();
  llvm::SmallVector<int64_t, 1> tileShape{spec.n};
  Type tileType = TileType::get(builder.getContext(), tileShape, f16);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&entry);

  llvm::SmallVector<Value, 8> values;
  values.reserve(spec.inputArgIndices.size() + spec.ops.size());
  for (unsigned argIndex : spec.inputArgIndices) {
    auto load = builder.create<LoadContigOp>(
        loc, tileType, entry.getArgument(argIndex), spec.n);
    if (spec.masked)
      setExecutableVectorMaskAttrs(builder, load.getOperation(), spec.logicalN,
                                   spec.n);
    values.push_back(load.getResult());
  }

  llvm::SmallVector<int64_t, 3> opcodeValues;
  llvm::SmallVector<int64_t, 3> lhsSlotValues;
  llvm::SmallVector<int64_t, 3> rhsSlotValues;
  int64_t availableSlots = static_cast<int64_t>(values.size());
  for (const ExecutableCompactVectorBinaryBuildOp &op : spec.ops) {
    if (op.lhs < 0 || op.rhs < 0 || op.lhs >= availableSlots ||
        op.rhs >= availableSlots)
      return kernel.emitError()
             << consumer << " compact vector value slot is out of range";
    opcodeValues.push_back(op.opcode == ExecutableCompactVectorBinaryOpcode::Add
                               ? kElementwise16ValueMapOpcodeAdd
                               : kElementwise16ValueMapOpcodeMul);
    lhsSlotValues.push_back(op.lhs);
    rhsSlotValues.push_back(op.rhs);
    ++availableSlots;
  }

  auto elementwise = builder.create<CompactElementwise1DOp>(
      loc, tileType, values, builder.getDenseI64ArrayAttr(opcodeValues),
      builder.getDenseI64ArrayAttr(lhsSlotValues),
      builder.getDenseI64ArrayAttr(rhsSlotValues));

  auto store =
      builder.create<StoreContigOp>(loc, entry.getArgument(spec.outputArgIndex),
                                    elementwise.getResult(), spec.n);
  if (spec.masked)
    setExecutableVectorMaskAttrs(builder, store.getOperation(), spec.logicalN,
                                 spec.n);
  builder.create<ReturnOp>(loc);

  FailureOr<SmallVector<Operation *>> bodyOps =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(bodyOps))
    return failure();
  return success();
}

LogicalResult mlir::rpu::exec::verifyExecutableElementwise1DOpSequence(
    KernelOp kernel, ArrayRef<Operation *> ops, llvm::StringRef consumer) {
  if (kernel.getKind() != "generic") {
    kernel.emitError() << consumer << " requires kind generic Elementwise1D";
    return failure();
  }
  if (kernel.getBody().empty()) {
    kernel.emitError() << consumer << " requires executable body";
    return failure();
  }
  if (ops.size() < 5 || !isa<ReturnOp>(ops.back())) {
    kernel.emitError()
        << consumer << " requires kind generic Elementwise1D return terminator";
    return failure();
  }

  SmallVector<LoadContigOp, 4> loads;
  SmallVector<Operation *, 4> binaryOps;
  StoreContigOp store;
  size_t index = 0;
  while (index < ops.size() && isa<LoadContigOp>(ops[index]))
    loads.push_back(cast<LoadContigOp>(ops[index++]));
  while (index < ops.size() && isExecutableCompactVectorBinaryOp(ops[index]))
    binaryOps.push_back(ops[index++]);
  if (index + 2 != ops.size() || !isa<StoreContigOp>(ops[index]) ||
      !isa<ReturnOp>(ops[index + 1])) {
    kernel.emitError() << consumer
                       << " requires compact load/binary/store/return body";
    return failure();
  }
  store = cast<StoreContigOp>(ops[index]);
  if (loads.size() < 2 || loads.size() > 4 || binaryOps.empty() ||
      binaryOps.size() > 3) {
    kernel.emitError()
        << consumer
        << " requires two to four loads and one to three binary ops";
    return failure();
  }

  Block &block = kernel.getBody().front();
  if (block.getNumArguments() != loads.size() + 1) {
    kernel.emitError() << consumer
                       << " requires output arg0 plus ordered input args";
    return failure();
  }
  if (store.getPtr() != block.getArgument(0)) {
    kernel.emitError() << consumer << " requires store to output arg0";
    return failure();
  }

  int64_t n = loads.front().getN();
  FailureOr<ExecutableVectorMaskInfo> firstMask =
      getExecutableVectorMaskInfo(loads.front().getOperation(), n, consumer);
  if (failed(firstMask))
    return failure();

  llvm::DenseSet<Value> availableValues;
  for (size_t i = 0, e = loads.size(); i < e; ++i) {
    if (failed(verifyExecutableCompactVectorLoad(
            loads[i], block.getArgument(i + 1), n, *firstMask, consumer))) {
      return failure();
    }
    availableValues.insert(loads[i].getResult());
  }

  Value finalResult;
  for (Operation *op : binaryOps) {
    if (failed(verifyExecutableCompactVectorBinaryOp(op, n, consumer)))
      return failure();
    Value lhs = getElementwise1DBinaryLhs(op);
    Value rhs = getElementwise1DBinaryRhs(op);
    if (!availableValues.contains(lhs) || !availableValues.contains(rhs)) {
      kernel.emitError() << consumer
                         << " requires ordered acyclic binary dataflow";
      return failure();
    }
    finalResult = getElementwise1DBinaryResult(op);
    availableValues.insert(finalResult);
  }

  if (failed(verifyExecutableCompactVectorStore(
          store, block.getArgument(0), finalResult, n, *firstMask, consumer))) {
    return failure();
  }

  return success();
}

FailureOr<SmallVector<Operation *>>
mlir::rpu::exec::getGenericRenderableExecutableKernelBodyOps(
    KernelOp kernel, llvm::StringRef consumer) {
  FailureOr<SmallVector<Operation *>> ops =
      getCanonicalExecutableKernelBodyOps(kernel, consumer);
  if (failed(ops))
    return failure();
  if (failed(
          verifyGenericRenderableExecutableOpSequence(kernel, *ops, consumer)))
    return failure();
  return *ops;
}

static LogicalResult verifyKindOperandDataflow(KernelOp op, StringRef kind,
                                               Block &block,
                                               ArrayRef<Operation *> ops) {
  if (const ExecutableVerifierEntry *entry =
          lookupExecutableVerifierEntry(kind))
    return entry->verifyDataflow(op, block, ops);
  return success();
}

static LogicalResult verifyOptionalNonNegativeI32Attr(Operation *op,
                                                      llvm::StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr)
    return success();
  if (!attr.getType().isInteger(32))
    return op->emitError("load_matrix offsets must be i32 attributes");
  if (attr.getInt() < 0)
    return op->emitError("load_matrix offsets must be non-negative");
  return success();
}

LogicalResult KernelOp::verify() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module || getOperation()->getParentOp() != module.getOperation())
    return emitOpError("must be a top-level operation in builtin.module");
  if (getSymName().empty())
    return emitOpError("requires non-empty sym_name");
  auto kindAttr = (*this)->getAttrOfType<StringAttr>("kind");
  if (!kindAttr)
    return emitOpError("requires kind");
  if (!isSupportedExecutableKernelKind(kindAttr.getValue()))
    return emitOpError("unsupported kind");
  if (getBody().empty())
    return emitOpError("requires one executable region block");
  if (!getBody().hasOneBlock())
    return emitOpError("requires one executable region block");
  Block &block = getBody().front();
  for (Type argType : block.getArgumentTypes()) {
    if (!isExecutableF16PointerType(argType))
      return emitOpError("block arguments must be !tt.ptr<f16>");
  }
  std::optional<unsigned> expectedArgCount =
      expectedExecutableKernelArgCount(kindAttr.getValue());
  if (expectedArgCount && block.getNumArguments() != *expectedArgCount)
    return emitOpError("kind ")
           << kindAttr.getValue() << " requires " << *expectedArgCount
           << " f16 pointer arguments";
  FailureOr<SmallVector<Operation *>> ops =
      getCanonicalExecutableKernelBodyOps(*this, "rpu.kernel verifier");
  if (failed(ops))
    return failure();
  return verifyKindOperandDataflow(*this, kindAttr.getValue(), block, *ops);
}

LogicalResult LoadContigOp::verify() {
  if (getN() <= 0)
    return emitOpError("requires positive n");
  if (!isExecutableF16PointerType(getPtr().getType()))
    return emitOpError("requires !tt.ptr<f16> input");
  FailureOr<TileType> tile = requireTile(getOperation(), getResult().getType());
  if (failed(tile))
    return failure();
  if ((*tile).getRank() != 1 || (*tile).getExtent() != getN())
    return emitOpError("result tile extent must equal n");
  if (failed(mlir::rpu::exec::getExecutableVectorMaskInfo(
          getOperation(), getN(), "rpu.load_contig")))
    return failure();
  return success();
}

LogicalResult AddOp::verify() {
  FailureOr<TileType> lhs = requireTile(getOperation(), getLhs().getType());
  FailureOr<TileType> rhs = requireTile(getOperation(), getRhs().getType());
  FailureOr<TileType> result =
      requireTile(getOperation(), getResult().getType());
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  if ((*lhs) != (*rhs) || (*lhs) != (*result))
    return emitOpError("operand and result tile types must match");
  return success();
}

LogicalResult MulOp::verify() {
  FailureOr<TileType> lhs = requireTile(getOperation(), getLhs().getType());
  FailureOr<TileType> rhs = requireTile(getOperation(), getRhs().getType());
  FailureOr<TileType> result =
      requireTile(getOperation(), getResult().getType());
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  if ((*lhs) != (*rhs) || (*lhs) != (*result))
    return emitOpError("operand and result tile types must match");
  return success();
}

LogicalResult Elementwise16ValueMapOp::verify() {
  return verifyElementwise16ValueMapContract(
      getOperation(), getInputs(), getResult().getType(), getOpcodes(),
      getLhsSlots(), getRhsSlots());
}

LogicalResult CompactElementwise1DOp::verify() {
  return verifyCompactElementwise1DContract(getOperation(), getInputs(),
                                            getResult().getType(), getOpcodes(),
                                            getLhsSlots(), getRhsSlots());
}

LogicalResult SoftmaxOp::verify() {
  if (failed(requireSameTile(getOperation(), getInput().getType(),
                             getResult().getType())))
    return failure();
  FailureOr<TileType> tile = requireTile(getOperation(), getResult().getType());
  if (failed(tile))
    return failure();
  if ((*tile).getRank() != 1)
    return emitOpError("requires 1D f16 tile");
  return success();
}

LogicalResult MaxOp::verify() {
  FailureOr<TileType> lhs = requireTile(getOperation(), getLhs().getType());
  FailureOr<TileType> rhs = requireTile(getOperation(), getRhs().getType());
  FailureOr<TileType> result =
      requireTile(getOperation(), getResult().getType());
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  if ((*lhs) != (*rhs) || (*lhs) != (*result))
    return emitOpError("operand and result tile types must match");
  return success();
}

LogicalResult StoreContigOp::verify() {
  if (getN() <= 0)
    return emitOpError("requires positive n");
  if (!isExecutableF16PointerType(getPtr().getType()))
    return emitOpError("requires !tt.ptr<f16> output pointer");
  FailureOr<TileType> tile = requireTile(getOperation(), getValue().getType());
  if (failed(tile))
    return failure();
  if ((*tile).getRank() != 1 || (*tile).getExtent() != getN())
    return emitOpError("stored tile extent must equal n");
  if (failed(mlir::rpu::exec::getExecutableVectorMaskInfo(
          getOperation(), getN(), "rpu.store_contig")))
    return failure();
  return success();
}

LogicalResult LoadMatrixOp::verify() {
  if (getRows() <= 0 || getCols() <= 0)
    return emitOpError("requires positive rows and cols");
  if (!isExecutableF16PointerType(getPtr().getType()))
    return emitOpError("requires !tt.ptr<f16> input");
  FailureOr<TileType> tile = requireTile(getOperation(), getResult().getType());
  if (failed(tile))
    return failure();
  SmallVector<int64_t, 2> shape = matrixShape(getRows(), getCols());
  if (!hasTileShape(*tile, shape))
    return emitOpError("result tile shape must equal rows x cols");
  if (failed(verifyOptionalNonNegativeI32Attr(getOperation(), "row_offset")) ||
      failed(verifyOptionalNonNegativeI32Attr(getOperation(), "col_offset")))
    return failure();
  return success();
}

LogicalResult ZeroOp::verify() {
  if (getRows() <= 0 || getCols() <= 0)
    return emitOpError("requires positive rows and cols");
  FailureOr<TileType> tile = requireTile(getOperation(), getResult().getType());
  if (failed(tile))
    return failure();
  SmallVector<int64_t, 2> shape = matrixShape(getRows(), getCols());
  if (!hasTileShape(*tile, shape))
    return emitOpError("result tile shape must equal rows x cols");
  return success();
}

LogicalResult MmaOp::verify() {
  FailureOr<TileType> lhs = requireTile(getOperation(), getLhs().getType());
  FailureOr<TileType> rhs = requireTile(getOperation(), getRhs().getType());
  FailureOr<TileType> acc = requireTile(getOperation(), getAcc().getType());
  FailureOr<TileType> result =
      requireTile(getOperation(), getResult().getType());
  if (failed(lhs) || failed(rhs) || failed(acc) || failed(result))
    return failure();
  if ((*lhs).getRank() != 2 || (*rhs).getRank() != 2 || (*acc).getRank() != 2 ||
      (*result).getRank() != 2)
    return emitOpError(
        "mma tile shapes must satisfy lhs MxK, rhs KxN, acc/result MxN");

  ArrayRef<int64_t> lhsShape = (*lhs).getShape();
  ArrayRef<int64_t> rhsShape = (*rhs).getShape();
  ArrayRef<int64_t> accShape = (*acc).getShape();
  ArrayRef<int64_t> resultShape = (*result).getShape();
  int64_t m = lhsShape[0];
  int64_t k = lhsShape[1];
  int64_t n = rhsShape[1];
  if (rhsShape[0] != k || accShape[0] != m || accShape[1] != n ||
      resultShape[0] != m || resultShape[1] != n)
    return emitOpError(
        "mma tile shapes must satisfy lhs MxK, rhs KxN, acc/result MxN");
  return success();
}

LogicalResult DotOp::verify() {
  FailureOr<TileType> lhs = requireTile(getOperation(), getLhs().getType());
  FailureOr<TileType> rhs = requireTile(getOperation(), getRhs().getType());
  FailureOr<TileType> result =
      requireTile(getOperation(), getResult().getType());
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  if ((*lhs).getRank() != 2 || (*rhs).getRank() != 2 ||
      (*result).getRank() != 2)
    return emitOpError(
        "dot tile shapes must satisfy lhs MxK, rhs KxN, result MxN");

  ArrayRef<int64_t> lhsShape = (*lhs).getShape();
  ArrayRef<int64_t> rhsShape = (*rhs).getShape();
  ArrayRef<int64_t> resultShape = (*result).getShape();
  int64_t m = lhsShape[0];
  int64_t k = lhsShape[1];
  int64_t n = rhsShape[1];
  if (rhsShape[0] != k || resultShape[0] != m || resultShape[1] != n)
    return emitOpError(
        "dot tile shapes must satisfy lhs MxK, rhs KxN, result MxN");
  return success();
}

LogicalResult StoreMatrixOp::verify() {
  if (getRows() <= 0 || getCols() <= 0)
    return emitOpError("requires positive rows and cols");
  if (!isExecutableF16PointerType(getPtr().getType()))
    return emitOpError("requires !tt.ptr<f16> output pointer");
  FailureOr<TileType> tile = requireTile(getOperation(), getValue().getType());
  if (failed(tile))
    return failure();
  SmallVector<int64_t, 2> shape = matrixShape(getRows(), getCols());
  if (!hasTileShape(*tile, shape))
    return emitOpError("stored tile shape must equal rows x cols");
  return success();
}

LogicalResult ReduceMaxAllOp::verify() {
  if (failed(requireTile(getOperation(), getInput().getType())))
    return failure();
  return requireF16Scalar(getOperation(), getResult().getType());
}

LogicalResult SubScalarOp::verify() {
  if (failed(requireF16Scalar(getOperation(), getRhs().getType())))
    return failure();
  return requireSameTile(getOperation(), getLhs().getType(),
                         getResult().getType());
}

LogicalResult ExpOp::verify() {
  return requireSameTile(getOperation(), getInput().getType(),
                         getResult().getType());
}

LogicalResult SqrtOp::verify() {
  return requireSameTile(getOperation(), getInput().getType(),
                         getResult().getType());
}

LogicalResult ReluOp::verify() {
  return requireSameTile(getOperation(), getInput().getType(),
                         getResult().getType());
}

LogicalResult ReduceSumAxisOp::verify() {
  FailureOr<TileType> input = requireTile(getOperation(), getInput().getType());
  FailureOr<TileType> result =
      requireTile(getOperation(), getResult().getType());
  if (failed(input) || failed(result))
    return failure();
  if ((*input).getRank() != 2)
    return emitOpError("input must be a rank-2 f16 tile");
  if ((*result).getRank() != 1)
    return emitOpError("result must be a rank-1 f16 tile");
  int64_t axis = getAxis();
  if (axis != 0 && axis != 1)
    return emitOpError("axis must be 0 or 1");
  int64_t expected =
      axis == 0 ? (*input).getShape()[1] : (*input).getShape()[0];
  if ((*result).getShape()[0] != expected)
    return emitOpError("result extent must equal input non-reduce dim");
  return success();
}

LogicalResult BroadcastAddOp::verify() {
  FailureOr<TileType> lhs = requireTile(getOperation(), getLhs().getType());
  FailureOr<TileType> rhs = requireTile(getOperation(), getRhs().getType());
  FailureOr<TileType> result =
      requireTile(getOperation(), getResult().getType());
  if (failed(lhs) || failed(rhs) || failed(result))
    return failure();
  if ((*lhs).getRank() != 2 || (*rhs).getRank() != 1 ||
      (*result).getRank() != 2)
    return emitOpError(
        "broadcast_add requires rank-2 lhs, rank-1 rhs, rank-2 result");
  if (!llvm::equal((*lhs).getShape(), (*result).getShape()))
    return emitOpError("result shape must equal lhs shape");
  if ((*rhs).getShape()[0] != (*lhs).getShape()[0])
    return emitOpError("rhs extent must equal lhs rows");
  return success();
}

LogicalResult FullOp::verify() {
  if (failed(requireF16Scalar(getOperation(), getValue().getType())))
    return failure();
  return requireTile(getOperation(), getResult().getType());
}

LogicalResult ReduceSumAllOp::verify() {
  if (failed(requireTile(getOperation(), getInput().getType())))
    return failure();
  return requireF16Scalar(getOperation(), getResult().getType());
}

LogicalResult ReciprocalOp::verify() {
  if (failed(requireF16Scalar(getOperation(), getInput().getType())))
    return failure();
  return requireF16Scalar(getOperation(), getResult().getType());
}

LogicalResult MulScalarOp::verify() {
  if (failed(requireF16Scalar(getOperation(), getRhs().getType())))
    return failure();
  return requireSameTile(getOperation(), getLhs().getType(),
                         getResult().getType());
}

#define GET_OP_CLASSES
#include "RPU/IR/RPUOps.cpp.inc"
