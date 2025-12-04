//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
//
////////////////////////////////////////////////////////////////////////////////
// Overview
////////////////////////////////////////////////////////////////////////////////
//
// This pass attempts to lower all loads and stores of unstructured pointers to
// tts.gather or tts.scatter that take a single base, a tensor of offsets, an
// optional tensor of mask values, and a default value in case of load.
//
// In addition, all pointer-producing ops will be eliminated and replaced by
// offset-producing ops. tts.gather and tts.scatter will use the pointer
// directly from the kernel arguments as opposed to pointer produced by ops such
// as tt.addptr and tt.splat.
//
// Example:
//
// %12 = tts.gather %arg0[%10] : (<f32>, tensor<64xi64>) -> tensor<64xf32>
// tts.scatter %12 into %arg1[%arg3] : tensor<64xf32> into (<f32>,
// tensor<64xi32>)
//
// Current assumptions and limitations:
//   - For simplicity, the pass assumes that gather / scatter operations load /
//   store from / to a single base with a tensor of random offsets. As a
//   result, the following triton program would not work:
//
//    @triton.jit
//    def gather_simple(in0, in1, out0):
//        offs = tl.arange(0, 8)
//        in0_ptrs = in0 + offs
//        in1_ptrs = in1 + offs
//        ptrs = tl.cat(in0_ptrs, in1_ptrs, can_reorder=True)
//        c = tl.load(ptrs)
//        out_offs = tl.arange(0, 16)
//        tl.store(out0 + out_offs, c)
//
//   In the above program, `ptrs` contains 2 bases: `in0` and `in1` after the
//   `cat` operation.
//
////////////////////////////////////////////////////////////////////////////////
// Future work
////////////////////////////////////////////////////////////////////////////////
//
// Future work may include scaling the algorithm to support such cases -- one
// possible solution is to let tts.gather and tts.scatter take in an additional
// tensor of base pointers corresponding to the tensor of offsets. But because
// we do not want pointer-producing ops to be present after this pass, we can
// use a tensor of index where each element indicates the index of the pointer
// argument to be used. The drawback is a gather or scatter operation now needs
// one extract lookup to get the base which will affect performance.
//
////////////////////////////////////////////////////////////////////////////////
// Algorithm
////////////////////////////////////////////////////////////////////////////////
//
// Because the goal of triton-shared is to eventually lower all triton ops and
// types to mlir, we want to transform the IR such that the usages of triton
// pointers are as limited as possible. Doing so will help simplify conversion
// to mlir dialects in subsequent passes. In a familiar fashion to the
// triton-to-structured pass, we want triton pointers to only appear in
// tts.gather and tts.scatter only.
//
// With that goal in mind, we want to revisit the triton pointer type.
//
// Triton pointers are created and manipulated through a sequence of ops such as
// tt.addptr, tt.splat, or tt.broadcast. If a triton pointer is created
// through `tt.addptr %ptr %offset`, the new pointer will contain the same base
// pointer as the original pointer; its offset will also be accumulated.
//
// Triton pointers created through tt.splat and tt.broadcast retain their base
// pointers and offsets. Tensors of pointers, however, may have different bases
// when tl.cat is present. For simplicity, we assume tl.cat isn't present as
// mentioned in the overview section.
//
// Therefore, a single triton pointer (tt.ptr) has two pieces of info that is
// implicit:
//   - a base pointer which comes from the kernel arguments
//   - an offset which could be either a tensor of offset or a single integer
//   offset
//
// Leveraging this insight, in order to limit the usages of triton pointer, we
// can explicitly compute and split the above two pieces of info. So chains of
// tt.addptr, tt.splat, and tt.broadcast which produce triton pointers can be
// transformed to sequences of offset (of integer type) manipulation ops and a
// base pointer which comes from the kernel arguments. With this approach, only
// tts.gather and tts.scatter need to be aware of the pointer type.
//
// In essence, this pass transforms all sequences of tt.addptr into sequences of
// offset accumulation ops which are then fed into a single op
// tts.gather or tts.scatter that takes:
//
//   - a base pointer from the kernel arguments
//   - a tensor of offsets (or single offset) that indicates the offsets from
//   the base pointer
//
// All intermediate tt.addptr ops are converted to arith.addi ops that compute
// the offsets. Offsets start at 0 with the provided bit-width. All pointer
// shape manipulation ops such as tt.splat and tt.broadcast will instead operate
// on the offsets and will be converted to linalg in triton-arith-to-linalg.
//
// By default, the pass uses i32 for the initial offsets of all pointers
// (configurable via offset-bit-width=width). If any intermediate tt.addptr
// introduces a larger bitwidth offset, the offsets will be sign-extended to the
// larger bitwidth.
//
////////////////////////////////////////////////////////////////////////////////
// Algorithm
////////////////////////////////////////////////////////////////////////////////
//
// This pass uses a standard worklist-based algorithm to walk the use-def chains
// of all pointer arguments and create replacement ops that operate on offsets
// instead of tt.ptr types.
//
// In cases such as tt.addptr, tt.splat, and tt.broadcast, we create
// corresponding replacement ops which will then be used to map the results
// at the end of the algorithm. We do not want to modify these ops in-place
// because the use-def chains may be changed. In special cases like scf.for, we
// also set the type of the iter-arg and result directly which is usually frown
// upon (but justified).
//
// This approach is used in favor of the traditional ConversionPatternRewriter
// which converts all pointer type into an offset integer type because
// TypeConverter does not support dynamic type based on value. This limitation
// means we have to decide the same bitwidth for all tt.addptr sequences which
// is not ideal.
//
// For instance, assuming we have two sequences of tt.addptr: one operates on
// 32-bit offsets while the other operates on 64-bit offsets. If we set the
// default bitwidth to 64, the 32-bit sequence will require unncessary
// sign-extending when computing the offsets. Contrast this with the manual
// approach, we will only sign-extend where necessary.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "utils/utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <queue>

#define DEBUG_TYPE "triton-to-unstructured"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToUnstructured/Passes.h.inc"

namespace {

// Given a type, return the offset type corresponding to that type with the
// specified width.
// If the type is a tensor, return a tensor of offsets of the same shape. If the
// type is a pointer, return a single offset type.
static Type getPtrOffsetType(Type type, unsigned int bitWidth) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto ptrType =
            dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      return RankedTensorType::get(
          tensorType.getShape(), IntegerType::get(type.getContext(), bitWidth));
    }
  }

  if (auto ptrType = dyn_cast<triton::PointerType>(type)) {
    return IntegerType::get(type.getContext(), bitWidth);
  }

  llvm_unreachable("unexpected type");
  return nullptr;
}

static unsigned int getBitWidth(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    if (auto integerType = dyn_cast<IntegerType>(tensorType.getElementType())) {
      return integerType.getWidth();
    }
  } else if (auto integerType = dyn_cast<IntegerType>(type)) {
    return integerType.getWidth();
  }

  llvm_unreachable("unexpected type");
  return 0;
}

class TritonToUnstructuredPass
    : public TritonToUnstructuredBase<TritonToUnstructuredPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, math::MathDialect, affine::AffineDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                tts::TritonStructuredDialect>();
  }

  struct PtrOffset {
    // the source pointer which comes from the kernel argument
    Value ptr;
    // the pointer type that corresponds to this offset; used when
    // creating tts.make_unstructured_tptr
    Type ptrType;
    // bitwidth that is used for this offset, used to track if sign-extension is
    // necessary
    unsigned int bitWidth;
    // the offset value
    Value offset;
  };

  LogicalResult processUnstructuredPtrs(unsigned int defaultBitWidth = 32) {
    llvm::SmallDenseSet<Value> ptrArgs;
    llvm::DenseMap<Value, PtrOffset> offsetMap;
    std::queue<Value> workList;

    getOperation().walk([&](FunctionOpInterface func) {
      for (auto arg : func.getArguments()) {
        if (!triton::utils::isPtrTypeLike(arg.getType())) {
          continue;
        }

        OpBuilder b(func->getRegion(0));
        Value zero = b.create<arith::ConstantOp>(
            arg.getLoc(),
            b.getIntegerAttr(IntegerType::get(&getContext(), defaultBitWidth),
                             0));

        ptrArgs.insert(arg);
        offsetMap.insert({arg, {arg, arg.getType(), defaultBitWidth, zero}});
        workList.push(arg);
      }
    });

    getOperation().walk([&](triton::IntToPtrOp op) {
      // We only want to handle single source pointer,
      // skip if this op produces tensor of pointers
      if (isa<RankedTensorType>(op.getType())) {
        return;
      }
      auto res = op.getResult();
      OpBuilder b(op);
      Value zero = b.create<arith::ConstantOp>(
          op.getLoc(),
          b.getIntegerAttr(IntegerType::get(&getContext(), defaultBitWidth),
                           0));

      offsetMap.insert({res, {res, res.getType(), defaultBitWidth, zero}});
      workList.push(res);
    });

    llvm::SmallVector<Operation *> toDelete;
    llvm::SmallVector<Operation *> ptrUsers;

    while (!workList.empty()) {
      auto val = workList.front();
      workList.pop();

      for (auto &use : val.getUses()) {
        auto user = use.getOwner();

        auto res =
            llvm::TypeSwitch<Operation *, LogicalResult>(user)

                .Case<arith::SelectOp>([&](arith::SelectOp op) {
                  auto ptr = op->getOperand(0);

                  if (!offsetMap.contains(op.getTrueValue()) ||
                      !offsetMap.contains(op.getFalseValue()))
                    return success();
                  auto TrueValue = offsetMap.at(op.getTrueValue());
                  auto FalseValue = offsetMap.at(op.getFalseValue());
                  assert(TrueValue.bitWidth == FalseValue.bitWidth &&
                         "arith.select op should have the same bitwidth "
                         "for both true and false values");
                  auto res = op.getResult();
                  auto resType = op.getType();

                  OpBuilder b{op};
                  auto newOffset = b.create<arith::SelectOp>(
                      op->getLoc(),
                      getPtrOffsetType(resType, TrueValue.bitWidth),
                      op.getCondition(), TrueValue.offset, FalseValue.offset);
                  PtrOffset newOffsetInfo{res, resType, TrueValue.bitWidth,
                                          newOffset};

                  offsetMap.insert({
                      res,
                      newOffsetInfo,
                  });
                  workList.push(res);
                  return success();
                })
                .Case<triton::PtrToIntOp>([&](triton::PtrToIntOp op) {
                  auto offsetInfo = offsetMap.at(op.getSrc());

                  OpBuilder b{op};
                  // We are converting a pointer to an integer here,
                  // materialized the pointer using the accumulated offset
                  // that we have stored so far.
                  auto materializedAddPtr = b.create<triton::AddPtrOp>(
                      op->getLoc(), offsetInfo.ptrType, offsetInfo.ptr,
                      offsetInfo.offset);

                  // Change the op to use the "simplified" pointer above.
                  // This should not affect the traversal of uses, but hacky.
                  // We will need to revisit how we process the IRs in this pass
                  // later.
                  op->setOperand(0, materializedAddPtr);

                  return success();
                })
                .Case<triton::BitcastOp>([&](triton::BitcastOp bitcast) {
                  OpBuilder b{bitcast};
                  auto loc = bitcast->getLoc();

                  auto offsetInfo = offsetMap.at(bitcast.getOperand());

                  auto newBitcast = b.create<triton::BitcastOp>(
                      loc, bitcast.getType(), offsetInfo.ptr);
                  bitcast->replaceAllUsesWith(newBitcast);

                  PtrOffset newOffsetInfo{newBitcast, offsetInfo.ptrType,
                                          offsetInfo.bitWidth,
                                          offsetInfo.offset};
                  offsetMap.insert({newBitcast, newOffsetInfo});
                  workList.push(newBitcast);

                  return success();
                })
                .Case<triton::AddPtrOp>([&](triton::AddPtrOp addptr) {
                  OpBuilder b{addptr};
                  auto loc = addptr->getLoc();

                  auto offsetInfo = offsetMap.at(addptr.getPtr());

                  auto prevOff = offsetInfo.offset;
                  auto off = addptr.getOffset();

                  auto lhsWidth = offsetInfo.bitWidth;
                  auto rhsWidth = getBitWidth(off.getType());
                  auto resWidth = std::max(lhsWidth, rhsWidth);

                  if (lhsWidth < resWidth) {
                    prevOff = b.create<arith::ExtSIOp>(
                        loc, getPtrOffsetType(offsetInfo.ptrType, resWidth),
                        prevOff);
                  }

                  if (rhsWidth < resWidth) {
                    off = b.create<arith::ExtSIOp>(
                        loc, getPtrOffsetType(offsetInfo.ptrType, resWidth),
                        off);
                  }

                  auto accumulatedOff = b.create<arith::AddIOp>(
                      loc, getPtrOffsetType(addptr.getType(), resWidth),
                      prevOff, off);

                  PtrOffset newOffsetInfo{offsetInfo.ptr, addptr.getType(),
                                          resWidth, accumulatedOff};

                  offsetMap.insert({addptr, newOffsetInfo});
                  workList.push(addptr);
                  toDelete.push_back(addptr);

                  return success();
                })
                .Case<triton::SplatOp, triton::BroadcastOp,
                      triton::ExpandDimsOp>([&](Operation *op) {
                  auto res = op->getResult(0);
                  auto resType = res.getType();

                  if (!triton::utils::isPtrTypeLike(resType)) {
                    return success();
                  }

                  auto ptr = op->getOperand(0);
                  auto offsetInfo = offsetMap.at(ptr);

                  OpBuilder b{op};
                  auto clone =
                      b.create(op->getLoc(), op->getName().getIdentifier(),
                               ValueRange{offsetInfo.offset},
                               TypeRange{getPtrOffsetType(
                                   resType, offsetInfo.bitWidth)});

                  PtrOffset newOffsetInfo{offsetInfo.ptr, resType,
                                          offsetInfo.bitWidth,
                                          clone->getResult(0)};

                  offsetMap.insert({
                      res,
                      newOffsetInfo,
                  });
                  workList.push(res);
                  toDelete.push_back(op);

                  return success();
                })
                .Case<triton::LoadOp, triton::StoreOp, triton::MakeTensorPtrOp,
                      tts::MakeTensorPtrOp>([&](Operation *op) {
                  // Special case:
                  // We do not want to create "unstructured tensor pointer" into
                  // tts.make_tptr if the base pointer is directly from the
                  // kernel arguments.
                  if (auto makeTensorPtr = dyn_cast<tts::MakeTensorPtrOp>(op)) {
                    if (ptrArgs.contains(makeTensorPtr.getBase())) {
                      return success();
                    }
                    if (auto bitcast = dyn_cast<triton::BitcastOp>(
                            makeTensorPtr.getBase().getDefiningOp())) {
                      if (ptrArgs.contains(bitcast.getSrc())) {
                        return success();
                      }
                    }
                  }

                  ptrUsers.push_back(op);
                  return success();
                })
                .Case<scf::ForOp>([&](scf::ForOp forOp) {
                  // Index of the init-arg corresponding to this use, note that
                  // we have to subtract by 3 from the operand number because
                  // scf.for ops always have 3 leading operands for start, end,
                  // and step.
                  auto argIndex = use.getOperandNumber() - 3;
                  auto init = forOp.getInitArgs()[argIndex];

                  auto offsetInfo = offsetMap.at(init);

                  auto offsetType =
                      getPtrOffsetType(offsetInfo.ptrType, offsetInfo.bitWidth);

                  // We're setting both the types of the iter-arg and the
                  // corresponding result directly to the offset type.
                  // At this point, the IR is in an invalid state because the
                  // init-args still have tt.ptr. But at the end, we will
                  // replace all uses of the tt.ptr to offset values.
                  auto iterArg = forOp.getRegionIterArg(argIndex);
                  iterArg.setType(offsetType);

                  auto res = forOp.getResult(argIndex);
                  res.setType(offsetType);

                  // For other ops, we only need to push the result into the
                  // worklist. But for scf.for, the iter-arg corresponding to
                  // the init-arg is used in the op's body instead, we have to
                  // process uses of the iter-arg.
                  PtrOffset iterArgOffset{offsetInfo.ptr, offsetInfo.ptrType,
                                          offsetInfo.bitWidth, iterArg};
                  offsetMap.insert({
                      iterArg,
                      iterArgOffset,
                  });

                  PtrOffset resOffset{offsetInfo.ptr, offsetInfo.ptrType,
                                      offsetInfo.bitWidth, res};
                  offsetMap.insert({
                      res,
                      resOffset,
                  });
                  workList.push(iterArg);
                  workList.push(res);

                  return success();
                })
                .Case<scf::YieldOp>([](auto) { return success(); })
                .Case<triton::CatOp>([](triton::CatOp op) {
                  op->emitError("Do not support gather / scatter with multiple "
                                "bases yet");
                  return failure();
                })
                .Default([&](Operation *op) {
                  op->emitError("unexpected op in ptr sequence");
                  return failure();
                });

        if (failed(res)) {
          return failure();
        }
      }
    }

    for (auto op : ptrUsers) {
      OpBuilder b{op};
      auto loc = op->getLoc();
      auto res =
          llvm::TypeSwitch<Operation *, LogicalResult>(op)
              .Case<triton::LoadOp>([&](triton::LoadOp load) {
                auto offsetInfo = offsetMap.at(load.getPtr());

                auto other = load.getOther();

                if (other) {
                  other = triton::utils::getScalarValue(other, loc, b);
                  if (!other) {
                    load->emitError("cannot parse `other` value for load");
                    return failure();
                  }
                }

                auto gather = b.create<tts::GatherOp>(
                    loc, load.getType(), offsetInfo.ptr, offsetInfo.offset,
                    load.getMask(), other);

                load->replaceAllUsesWith(gather->getResults());
                load->erase();
                return success();
              })
              .Case<triton::StoreOp>([&](triton::StoreOp store) {
                auto offsetInfo = offsetMap.at(store.getPtr());
                b.create<tts::ScatterOp>(loc, offsetInfo.ptr, offsetInfo.offset,
                                         store.getValue(), store.getMask());
                store->erase();
                return success();
              })
              .Case<triton::MakeTensorPtrOp,
                    tts::MakeTensorPtrOp>([&](auto makeTensorPtr) {
                // For block pointers, the base could come from a sequence of
                // `tt.addptr`. Accumulate the target offset with the offset
                // we have saved.
                auto offsetInfo = offsetMap.at(makeTensorPtr.getBase());
                auto baseOffset = offsetInfo.offset;

                makeTensorPtr.getBaseMutable().set(offsetInfo.ptr);

                // Add the existing offset from the base to the offset
                // operand in the ops.
                auto &offsetOpnd = makeTensorPtr.getOffsetsMutable()[0];
                auto currOffset = offsetOpnd.get();

                auto baseOffType = baseOffset.getType();
                auto currOffType = currOffset.getType();

                if (baseOffType != currOffType) {
                  if (currOffType.isIndex()) {
                    baseOffset = b.create<arith::IndexCastOp>(
                        loc, b.getIndexType(), baseOffset);
                  } else if (currOffType.isInteger()) {
                    if (baseOffType.getIntOrFloatBitWidth() <
                        currOffType.getIntOrFloatBitWidth()) {
                      baseOffset = b.create<arith::ExtSIOp>(loc, currOffType,
                                                            baseOffset);
                    } else {
                      // MakeTensorPtrOp only takes i32 offsets, so we need
                      // to truncate if the offsets were already in i64
                      makeTensorPtr.emitWarning(
                          "truncating offsets which may result in data loss");
                      baseOffset = b.create<arith::TruncIOp>(loc, currOffType,
                                                             baseOffset);
                    }
                  }
                }

                auto accumulatedOffset = b.create<arith::AddIOp>(
                    loc, currOffset.getType(), baseOffset, currOffset);

                offsetOpnd.set(accumulatedOffset);

                return success();
              })

              .Default([&](Operation *op) {
                op->emitError("unexpected op in ptr sequence");
                return failure();
              });

      if (failed(res)) {
        return failure();
      }
    }

    for (auto op : toDelete) {
      auto ptrInfo = offsetMap.at(op->getResult(0));
      op->replaceAllUsesWith(ValueRange{ptrInfo.offset});
      op->erase();
    }

    return success();
  }

  void runOnOperation() override {
    if (failed(processUnstructuredPtrs(offsetBitWidth))) {
      getOperation()->emitWarning(
          "Cannot transform tensor of pointers into a single base pointer "
          "with tensor of offsets");
      return;
    }

    PassManager pm(&getContext(), getOperation().getOperationName());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToUnstructuredPass() {
  return std::make_unique<TritonToUnstructuredPass>();
}
