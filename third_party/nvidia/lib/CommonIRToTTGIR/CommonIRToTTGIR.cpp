// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "nvidia/include/CommonIRToTTGIR/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir-ext/Dialect/CommonIR/IR/CommonIRDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>

namespace mlir::triton {

#define GEN_PASS_DEF_COMMONIRTOTTGIR
#include "nvidia/include/CommonIRToTTGIR/Passes.h.inc"

namespace {

namespace tile = mlir::triton::tile;
namespace ttg = mlir::triton::gpu;

static FailureOr<ttg::MemDescType> getMemDescType(Operation *op,
                                                  tile::BufType bufferType) {
  if (bufferType.getMemorySpace() != tile::MemorySpace::Shared) {
    op->emitError("GPU TileIR conversion currently requires #tile.shared");
    return failure();
  }

  Attribute layout = op->getAttr("tle.gpu_layout");
  if (!layout) {
    op->emitError("is missing the preserved tle.gpu_layout attribute");
    return failure();
  }
  if (!isa<ttg::SharedEncodingTrait>(layout)) {
    op->emitError("requires a preserved TritonGPU shared-memory layout");
    return failure();
  }

  auto memorySpace = ttg::SharedMemorySpaceAttr::get(op->getContext());
  return ttg::MemDescType::get(bufferType.getShape(),
                               bufferType.getElementType(), layout, memorySpace,
                               /*mutableMemory=*/true);
}

static FailureOr<Value>
lookupBuffer(Operation *op, Value buffer,
             const llvm::DenseMap<Value, Value> &mapped) {
  auto it = mapped.find(buffer);
  if (it == mapped.end()) {
    op->emitError("uses a TileIR buffer without a converted allocation");
    return failure();
  }
  return it->second;
}

static FailureOr<ttg::MemDescType>
inferMemDescType(ModuleOp module, triton::FuncOp func, BlockArgument argument,
                 const llvm::DenseMap<Value, Value> &mapped) {
  ttg::MemDescType inferred;
  for (Operation *user : argument.getUsers()) {
    auto bridge = dyn_cast<UnrealizedConversionCastOp>(user);
    if (!bridge || bridge.getInputs().size() != 1 ||
        bridge.getInputs().front() != argument || bridge->getNumResults() != 1)
      continue;
    auto type = dyn_cast<ttg::MemDescType>(bridge.getResult(0).getType());
    if (!type)
      continue;
    if (inferred && inferred != type) {
      bridge.emitError(
          "conflicting descriptor types for the same TileIR buffer");
      return failure();
    }
    inferred = type;
  }
  if (!inferred) {
    unsigned argumentNumber = argument.getArgNumber();
    module.walk([&](triton::CallOp call) {
      if (inferred ||
          module.lookupSymbol<triton::FuncOp>(call.getCallee()) != func ||
          argumentNumber >= call.getNumOperands())
        return;
      Value operand = call.getOperand(argumentNumber);
      auto it = mapped.find(operand);
      if (it != mapped.end())
        operand = it->second;
      inferred = dyn_cast<ttg::MemDescType>(operand.getType());
    });
  }
  if (!inferred) {
    emitError(argument.getLoc())
        << "cannot infer a TritonGPU descriptor type for TileIR block argument";
    return failure();
  }
  return inferred;
}

static LogicalResult convertFunctionArguments(
    ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
    llvm::DenseMap<Value, Type> &convertedArgumentTypes,
    SmallVectorImpl<UnrealizedConversionCastOp> &argumentCasts) {
  for (auto func : module.getOps<triton::FuncOp>()) {
    if (func.getBody().empty())
      continue;
    Block &entry = func.getBody().front();
    for (BlockArgument argument : entry.getArguments()) {
      if (!isa<tile::BufType>(argument.getType()))
        continue;
      auto type = inferMemDescType(module, func, argument, mapped);
      if (failed(type))
        return failure();
      OpBuilder builder(func.getContext());
      builder.setInsertionPointToStart(&entry);
      auto cast = builder.create<UnrealizedConversionCastOp>(
          argument.getLoc(), TypeRange{*type}, ValueRange{argument});
      mapped[argument] = cast.getResult(0);
      convertedArgumentTypes[argument] = *type;
      argumentCasts.push_back(cast);
    }
  }
  return success();
}

static LogicalResult convertWarpSpecializeCaptures(
    ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
    llvm::DenseMap<Value, Type> &convertedArgumentTypes,
    SmallVectorImpl<UnrealizedConversionCastOp> &argumentCasts) {
  bool failedConversion = false;
  module.walk([&](ttg::WarpSpecializeOp op) {
    if (failedConversion)
      return;
    auto captures = op.getExplicitCaptures();
    for (Region *region : op.getPartitionRegions()) {
      for (auto [index, capture] : llvm::enumerate(captures)) {
        BlockArgument argument = region->getArgument(index);
        if (!isa<tile::BufType>(argument.getType()))
          continue;
        auto converted = lookupBuffer(op, capture, mapped);
        if (failed(converted)) {
          failedConversion = true;
          return;
        }
        OpBuilder builder(op.getContext());
        builder.setInsertionPointToStart(&region->front());
        auto cast = builder.create<UnrealizedConversionCastOp>(
            argument.getLoc(), TypeRange{(*converted).getType()},
            ValueRange{argument});
        mapped[argument] = cast.getResult(0);
        convertedArgumentTypes[argument] = (*converted).getType();
        argumentCasts.push_back(cast);
      }
    }
  });
  return failure(failedConversion);
}

static bool isPointerLike(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

static Type getPointeeValueType(Type pointerType) {
  if (auto ptr = dyn_cast<triton::PointerType>(pointerType))
    return ptr.getPointeeType();
  auto tensorType = dyn_cast<RankedTensorType>(pointerType);
  if (!tensorType)
    return {};
  auto ptr = dyn_cast<triton::PointerType>(tensorType.getElementType());
  if (!ptr)
    return {};
  return RankedTensorType::get(tensorType.getShape(), ptr.getPointeeType(),
                               tensorType.getEncoding());
}

static FailureOr<Value> castToI32(OpBuilder &builder, Location loc,
                                  Value value) {
  if (value.getType().isInteger(32))
    return value;
  if (value.getType().isIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), value)
        .getResult();
  emitError(loc) << "expected an index or i32 TileIR subview offset";
  return failure();
}

static FailureOr<int64_t> getConstantOffset(Value value) {
  if (auto indexCast = value.getDefiningOp<arith::IndexCastOp>())
    value = indexCast.getIn();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return failure();
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer)
    return failure();
  return integer.getInt();
}

static LogicalResult checkTensorType(Operation *op, Type type,
                                     ttg::MemDescType bufferType) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getShape() != bufferType.getShape() ||
      tensor.getElementType() != bufferType.getElementType())
    return op->emitError(
        "requires a full-buffer tensor with matching shape and element type");
  return success();
}

static LogicalResult checkCopy(tile::CopyOp op, Value pointers,
                               ttg::MemDescType bufferType) {
  // The frozen CommonIR interface carries more copy modes than this backend
  // supports. Reject their semantics rather than silently doing a plain copy.
  if (op->hasAttr("engine") || op->hasAttr("transpose") ||
      op->hasAttr("dst_nz_layout") ||
      (op.getSrcLayout() && *op.getSrcLayout() != tile::Layout::ND))
    return op.emitError("GPU tile.copy supports only plain ND copies without "
                        "engine, transpose or NZ layout attributes");
  auto tensor = dyn_cast<RankedTensorType>(pointers.getType());
  auto pointer = tensor ? dyn_cast<triton::PointerType>(tensor.getElementType())
                        : triton::PointerType();
  if (!pointer || pointer.getAddressSpace() != 1)
    return op.emitError("GPU tile.copy requires a tensor of global pointers");
  return checkTensorType(op, getPointeeValueType(pointers.getType()),
                         bufferType);
}

static LogicalResult
convertAllocations(ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
                   SmallVectorImpl<Operation *> &eraseOps) {
  WalkResult result = module.walk([&](tile::AllocOp op) {
    auto memDescType =
        getMemDescType(op, cast<tile::BufType>(op.getResult().getType()));
    if (failed(memDescType))
      return WalkResult::interrupt();
    OpBuilder builder(op);
    mapped[op.getResult()] =
        builder.create<ttg::LocalAllocOp>(op.getLoc(), *memDescType);
    eraseOps.push_back(op);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static void
rewriteConvertedOperands(ModuleOp module,
                         const llvm::DenseMap<Value, Value> &mapped) {
  module.walk([&](Operation *op) {
    if (isa<tile::AllocOp, tile::SubViewOp, tile::CopyOp, tile::ToTensorOp,
            tile::StoreTensorOp, UnrealizedConversionCastOp>(op))
      return;
    for (OpOperand &operand : op->getOpOperands()) {
      auto it = mapped.find(operand.get());
      if (it != mapped.end())
        operand.set(it->second);
    }
  });
}

static LogicalResult
convertBufferBridges(ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
                     ArrayRef<UnrealizedConversionCastOp> argumentCasts,
                     SmallVectorImpl<Operation *> &eraseOps) {
  WalkResult result = module.walk([&](UnrealizedConversionCastOp op) {
    if (llvm::is_contained(argumentCasts, op) || op.getInputs().size() != 1 ||
        op->getNumResults() != 1 ||
        !isa<tile::BufType>(op.getInputs().front().getType()) ||
        !isa<ttg::MemDescType>(op.getResult(0).getType()))
      return WalkResult::advance();

    auto source = lookupBuffer(op, op.getInputs().front(), mapped);
    if (failed(source))
      return WalkResult::interrupt();
    if ((*source).getType() != op.getResult(0).getType()) {
      op.emitError("CommonIR buffer bridge descriptor type does not match the "
                   "converted buffer");
      return WalkResult::interrupt();
    }
    mapped[op.getResult(0)] = *source;
    eraseOps.push_back(op);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static void finalizeConvertedArguments(
    ModuleOp module, const llvm::DenseMap<Value, Type> &convertedArgumentTypes,
    ArrayRef<UnrealizedConversionCastOp> argumentCasts) {
  for (auto [value, type] : convertedArgumentTypes)
    cast<BlockArgument>(value).setType(type);

  for (UnrealizedConversionCastOp cast : argumentCasts) {
    cast.getResult(0).replaceAllUsesWith(cast.getInputs().front());
    cast.erase();
  }

  for (auto func : module.getOps<triton::FuncOp>()) {
    if (func.getBody().empty())
      continue;
    SmallVector<Type> inputTypes(func.getBody().front().getArgumentTypes());
    SmallVector<Type> resultTypes(func.getResultTypes());
    func.setFunctionType(
        FunctionType::get(func.getContext(), inputTypes, resultTypes));
  }
}

static LogicalResult convertSubviews(ModuleOp module,
                                     llvm::DenseMap<Value, Value> &mapped,
                                     SmallVectorImpl<Operation *> &eraseOps) {
  WalkResult result = module.walk([&](tile::SubViewOp op) {
    auto source = lookupBuffer(op, op.getSource(), mapped);
    auto resultType =
        getMemDescType(op, cast<tile::BufType>(op.getResult().getType()));
    if (failed(source) || failed(resultType))
      return WalkResult::interrupt();

    OpBuilder builder(op);
    auto sourceType = cast<ttg::MemDescType>((*source).getType());
    if (op.getOffsets().size() != sourceType.getRank() ||
        op.getSizes().size() != resultType->getRank() ||
        op.getStrides().size() != resultType->getRank()) {
      op.emitError("GPU tile.subview requires source-rank offsets and "
                   "result-rank sizes and strides");
      return WalkResult::interrupt();
    }
    for (auto [size, dim] : llvm::zip(op.getSizes(), resultType->getShape())) {
      if (cast<IntegerAttr>(size).getInt() != dim) {
        op.emitError("tile.subview sizes must match its result shape");
        return WalkResult::interrupt();
      }
    }
    if (llvm::any_of(op.getStrides(), [](Attribute stride) {
          return cast<IntegerAttr>(stride).getInt() != 1;
        })) {
      op.emitError("GPU tile.subview requires unit strides");
      return WalkResult::interrupt();
    }
    Value converted;
    if (sourceType.getRank() == resultType->getRank() + 1) {
      if (sourceType.getShape().drop_front() != resultType->getShape()) {
        op.emitError("rank-reducing tile.subview must preserve trailing shape");
        return WalkResult::interrupt();
      }
      for (Value offset : op.getOffsets().drop_front()) {
        auto constant = getConstantOffset(offset);
        if (failed(constant) || *constant != 0) {
          op.emitError(
              "rank-reducing tile.subview requires zero trailing offsets");
          return WalkResult::interrupt();
        }
      }
      auto leading = getConstantOffset(op.getOffsets().front());
      if (succeeded(leading) &&
          (*leading < 0 || *leading >= sourceType.getShape().front())) {
        op.emitError("tile.subview offset is outside the source buffer");
        return WalkResult::interrupt();
      }
      auto index = castToI32(builder, op.getLoc(), op.getOffsets().front());
      if (failed(index))
        return WalkResult::interrupt();
      converted = builder.create<ttg::MemDescIndexOp>(op.getLoc(), *resultType,
                                                      *source, *index);
    } else if (sourceType.getRank() == resultType->getRank()) {
      SmallVector<int32_t> offsets;
      offsets.reserve(op.getOffsets().size());
      for (auto [axis, offset] : llvm::enumerate(op.getOffsets())) {
        auto constant = getConstantOffset(offset);
        if (failed(constant)) {
          op.emitError(
              "same-rank tile.subview currently requires static offsets");
          return WalkResult::interrupt();
        }
        if (*constant < 0 || *constant > std::numeric_limits<int32_t>::max() ||
            *constant >
                sourceType.getShape()[axis] - resultType->getShape()[axis]) {
          op.emitError("tile.subview offset is outside the source buffer");
          return WalkResult::interrupt();
        }
        offsets.push_back(static_cast<int32_t>(*constant));
      }
      converted = builder.create<ttg::MemDescSubsliceOp>(
          op.getLoc(), *resultType, *source,
          builder.getDenseI32ArrayAttr(offsets));
    } else {
      op.emitError("unsupported tile.subview rank change");
      return WalkResult::interrupt();
    }

    mapped[op.getResult()] = converted;
    eraseOps.push_back(op);
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static LogicalResult
convertTileUsers(ModuleOp module, llvm::DenseMap<Value, Value> &mapped,
                 SmallVectorImpl<Operation *> &eraseOps, bool enableAsyncCopy,
                 SmallVectorImpl<ttg::AsyncCopyGlobalToLocalOp> &copies) {
  bool failedConversion = false;
  OpBuilder builder(module.getContext());

  module.walk([&](Operation *operation) {
    if (failedConversion)
      return;

    if (auto op = dyn_cast<tile::ToTensorOp>(operation)) {
      auto source = lookupBuffer(op, op.getSrc(), mapped);
      if (failed(source)) {
        failedConversion = true;
        return;
      }
      if (failed(
              checkTensorType(op, op.getResult().getType(),
                              cast<ttg::MemDescType>((*source).getType())))) {
        failedConversion = true;
        return;
      }
      builder.setInsertionPoint(op);
      Value replacement = builder.create<ttg::LocalLoadOp>(
          op.getLoc(), op.getResult().getType(), *source);
      op.getResult().replaceAllUsesWith(replacement);
      eraseOps.push_back(op);
      return;
    }

    if (auto op = dyn_cast<tile::StoreTensorOp>(operation)) {
      auto destination = lookupBuffer(op, op.getDst(), mapped);
      if (failed(destination)) {
        failedConversion = true;
        return;
      }
      if (failed(checkTensorType(
              op, op.getSrc().getType(),
              cast<ttg::MemDescType>((*destination).getType())))) {
        failedConversion = true;
        return;
      }
      builder.setInsertionPoint(op);
      builder.create<ttg::LocalStoreOp>(op.getLoc(), op.getSrc(), *destination);
      eraseOps.push_back(op);
      return;
    }

    if (auto op = dyn_cast<tile::CopyOp>(operation)) {
      Value src = op.getSrc();
      Value dst = op.getDst();
      if (isa<tile::BufType>(src.getType())) {
        auto converted = lookupBuffer(op, src, mapped);
        if (failed(converted)) {
          failedConversion = true;
          return;
        }
        src = *converted;
      }
      if (isa<tile::BufType>(dst.getType())) {
        auto converted = lookupBuffer(op, dst, mapped);
        if (failed(converted)) {
          failedConversion = true;
          return;
        }
        dst = *converted;
      }

      builder.setInsertionPoint(op);
      if (isa<ttg::MemDescType>(dst.getType())) {
        if (failed(checkCopy(op, src, cast<ttg::MemDescType>(dst.getType())))) {
          failedConversion = true;
          return;
        }
        if (enableAsyncCopy) {
          auto copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
              op.getLoc(), src, dst, Value(), Value(),
              triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL,
              false);
          auto commit = builder.create<ttg::AsyncCommitGroupOp>(
              op.getLoc(), ValueRange{copy.getToken()});
          copies.push_back(copy);
          // tile.copy is synchronous to its users. Keep the completion token
          // explicit so later GPU passes can reason about shared-memory reads.
          builder.create<ttg::AsyncWaitOp>(op.getLoc(), commit.getResult(), 0);
        } else {
          Value value = builder.create<triton::LoadOp>(
              op.getLoc(), src, triton::CacheModifier::NONE,
              triton::EvictionPolicy::NORMAL, false);
          builder.create<ttg::LocalStoreOp>(op.getLoc(), value, dst);
        }
      } else if (isa<ttg::MemDescType>(src.getType()) &&
                 isPointerLike(dst.getType())) {
        if (failed(checkCopy(op, dst, cast<ttg::MemDescType>(src.getType())))) {
          failedConversion = true;
          return;
        }
        Type valueType = getPointeeValueType(dst.getType());
        if (!valueType) {
          op.emitError("cannot determine tile.copy destination value type");
          failedConversion = true;
          return;
        }
        Value value =
            builder.create<ttg::LocalLoadOp>(op.getLoc(), valueType, src);
        builder.create<triton::StoreOp>(op.getLoc(), dst, value,
                                        triton::CacheModifier::NONE,
                                        triton::EvictionPolicy::NORMAL);
      } else {
        op.emitError("unsupported operands for GPU tile.copy conversion");
        failedConversion = true;
        return;
      }
      eraseOps.push_back(op);
      return;
    }

    if (isa<tile::SetFlagOp, tile::WaitFlagOp, tile::PipeBarrierOp>(
            operation)) {
      operation->emitError("GPU CommonIR conversion does not support tile "
                           "engine/event synchronization; use the existing "
                           "GPU synchronization path");
      failedConversion = true;
    }
  });

  return failure(failedConversion);
}

static Value getSharedAllocation(Value value) {
  while (value) {
    if (value.getDefiningOp<ttg::LocalAllocOp>())
      return value;
    if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>())
      value = index.getSrc();
    else if (auto slice = value.getDefiningOp<ttg::MemDescSubsliceOp>())
      value = slice.getSrc();
    else
      return {};
  }
  return {};
}

static void combineCopyWaits(ArrayRef<ttg::AsyncCopyGlobalToLocalOp> copies) {
  llvm::SmallPtrSet<Operation *, 16> generated;
  for (auto copy : copies)
    generated.insert(copy);
  for (auto copy : copies) {
    auto commit =
        dyn_cast_or_null<ttg::AsyncCommitGroupOp>(copy->getNextNode());
    if (!commit || commit.getInputTokens().size() != 1)
      continue;
    auto wait = dyn_cast_or_null<ttg::AsyncWaitOp>(commit->getNextNode());
    Value allocation = getSharedAllocation(copy.getResult());
    if (!wait || !wait.getResult().use_empty() || !allocation)
      continue;
    SmallVector<Value> allocations{allocation};
    while (true) {
      Operation *next = wait->getNextNode();
      // Only cross pure, region-free address/value computations. A read,
      // write, barrier or control-flow boundary must observe the earlier copy.
      while (next && !next->getNumRegions() && isMemoryEffectFree(next))
        next = next->getNextNode();
      if (!next || !generated.contains(next))
        break;
      auto nextCopy = cast<ttg::AsyncCopyGlobalToLocalOp>(next);
      Value nextAllocation = getSharedAllocation(nextCopy.getResult());
      if (!nextAllocation || llvm::is_contained(allocations, nextAllocation))
        break;
      auto nextCommit =
          dyn_cast_or_null<ttg::AsyncCommitGroupOp>(next->getNextNode());
      auto nextWait =
          nextCommit
              ? dyn_cast_or_null<ttg::AsyncWaitOp>(nextCommit->getNextNode())
              : ttg::AsyncWaitOp();
      if (!nextWait || !nextWait.getResult().use_empty())
        break;
      SmallVector<Value> tokens(commit.getInputTokens());
      llvm::append_range(tokens, nextCommit.getInputTokens());
      nextCommit.getInputTokensMutable().assign(tokens);
      wait.erase();
      commit.erase();
      commit = nextCommit;
      wait = nextWait;
      allocations.push_back(nextAllocation);
    }
  }
}

class CommonIRToTTGIRPass
    : public impl::CommonIRToTTGIRBase<CommonIRToTTGIRPass> {
public:
  using CommonIRToTTGIRBase::CommonIRToTTGIRBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::DenseMap<Value, Value> mapped;
    llvm::DenseMap<Value, Type> convertedArgumentTypes;
    SmallVector<Operation *> eraseOps;
    SmallVector<ttg::AsyncCopyGlobalToLocalOp> copies;
    SmallVector<UnrealizedConversionCastOp> argumentCasts;

    if (failed(convertAllocations(module, mapped, eraseOps)) ||
        failed(convertWarpSpecializeCaptures(
            module, mapped, convertedArgumentTypes, argumentCasts)) ||
        failed(convertFunctionArguments(module, mapped, convertedArgumentTypes,
                                        argumentCasts)) ||
        failed(convertSubviews(module, mapped, eraseOps)) ||
        failed(convertBufferBridges(module, mapped, argumentCasts, eraseOps)) ||
        failed(convertTileUsers(module, mapped, eraseOps, enableAsyncCopy,
                                copies))) {
      signalPassFailure();
      return;
    }

    rewriteConvertedOperands(module, mapped);

    for (Operation *op : llvm::reverse(eraseOps)) {
      if (llvm::any_of(op->getResults(),
                       [](Value result) { return !result.use_empty(); })) {
        op->emitError("still has users after GPU TileIR conversion");
        signalPassFailure();
        return;
      }
      op->erase();
    }

    finalizeConvertedArguments(module, convertedArgumentTypes, argumentCasts);
    combineCopyWaits(copies);

    bool hasRemainingGpuTileOps = false;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "tile") {
        op->emitError("was not eliminated by GPU TileIR conversion");
        hasRemainingGpuTileOps = true;
        return;
      }
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op);
          cast && cast.getInputs().size() == 1 &&
          isa<tile::BufType>(cast.getInputs().front().getType())) {
        cast.emitError("CommonIR buffer bridge was not eliminated");
        hasRemainingGpuTileOps = true;
      }
    });
    if (hasRemainingGpuTileOps)
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton
