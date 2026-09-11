#include "triton/Analysis/VectorizabilityAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonxpu-vectorizability"

namespace mlir {
namespace triton {
namespace xpu {

bool vectorizedTyValid(Type elemTy) {
  if (elemTy.isF16() || elemTy.isF32() || elemTy.isBF16() ||
      elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32))
    return true;
  return false;
}

unsigned getVectorWidth(Type elemTy) {
  return 512 / elemTy.getIntOrFloatBitWidth();
}

bool reduceCombineIsVectorizable(triton::xpu::ReduceOp redOp) {
  // The region lowering emits the combine region op by op
  // (ReduceOpToLLVM::interpretCombine) instead of applying the single defining
  // op of each output, so a wider set of ops has a vector form. This predicate
  // and Vectorize's retyping must widen together: that TypeSwitch ends in
  // llvm_unreachable, it does not fall back.
  bool region = reduceCombineRegionEnabled();
  for (Block &block : redOp.getCombineOp().getBlocks())
    for (auto &op : block) {
      if (region ? !isa<REDUCE_COMBINE_REGION_OP>(op)
                 : !isa<REDUCE_COMBINE_OP>(op))
        return false;
      // A value captured from outside the region is not retyped when Vectorize
      // retypes the region, so it would leave a vector op with a scalar
      // operand. Constants are the exception: Vectorize rematerializes those as
      // in-region splats.
      for (Value operand : op.getOperands())
        if (operand.getParentBlock() != &block &&
            !operand.getDefiningOp<arith::ConstantOp>())
          return false;
    }
  return true;
}

bool hasVectorForm(Operation *op) {
  if (!op)
    return false;
#define TTX_VF_HAS_CASE(SrcType, DstType)                                      \
  if (isa<SrcType>(op))                                                        \
    return true;
  TTX_SCALAR_TO_VECTOR_OPS(TTX_VF_HAS_CASE)
#undef TTX_VF_HAS_CASE
  return false;
}

VecTyCoverage processOpVecTyCoverage(Operation *op) {
  if (!op)
    return VecTyCoverage::None;
  // Order matters: the two conditional kinds must be tested before the blanket
  // lists, or `arith::TruncIOp` would read as uncovered and
  // ExternElementwiseOp as fully covered.
  if (isa<triton::ExternElementwiseOp, arith::TruncIOp>(op))
    return VecTyCoverage::Conditional;
  // The one instance-aware arm. The arith::SelectOp Case builds a VSelectOp for
  // any element type, but VSelect only lowers f32/i32/f16 and is
  // `llvm_unreachable` otherwise (VectorizedOpToLLVM.cpp). The growth walk
  // never reached that because its own select gate (`vectorize()` below)
  // already requires f16/f32; the segment cut has no such gate, so dropout's i8
  // mask select was admitted and crashed ConvertTritonXPUToLLVM
  // (findings 1.71). Mirror the growth gate rather than VSelect's wider i32
  // support: f16/f32 is what has actually been exercised.
  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    Type elemTy = getElementTypeOrSelf(selectOp.getTrueValue().getType());
    return (elemTy.isF16() || elemTy.isF32()) ? VecTyCoverage::Full
                                              : VecTyCoverage::Conditional;
  }
  if (hasVectorForm(op) || isa<TTX_VECTORIZE_RETYPE_OPS>(op))
    return VecTyCoverage::Full;
  return VecTyCoverage::None;
}

const char *toString(VecTyCoverage coverage) {
  switch (coverage) {
  case VecTyCoverage::None:
    return "none";
  case VecTyCoverage::Full:
    return "full";
  case VecTyCoverage::Conditional:
    return "conditional";
  }
  return "?";
}

bool vecReportEnabled() {
  return mlir::triton::tools::getBoolEnvXPU("TRITONXPU_VEC_REPORT");
}

void reportVecRoot(const char *stage, const char *site, Operation *root,
                   Type rootOpTy, bool eligible, int64_t closureSize,
                   int64_t cands) {
  StringRef kernel = "<unknown>";
  if (auto funcOp = root->getParentOfType<triton::FuncOp>())
    kernel = funcOp.getName();

  // Element count per core rather than the tensor type: it is the quantity
  // `vectorFitsRoot` divides by the vector width, so a report that printed the
  // type would hide why a root was rejected.
  int64_t elemsPerCore = 0;
  int64_t width = 0;
  if (isa<RankedTensorType>(rootOpTy)) {
    elemsPerCore = getTotalElemsPerThread(rootOpTy);
    Type elemTy = getElementTypeOrSelf(rootOpTy);
    if (!isa<VectorType>(elemTy) && vectorizedTyValid(elemTy))
      width = getVectorWidth(elemTy);
  }

  llvm::errs() << "[VecAnalysis] " << kernel << " stage=" << stage
               << " site=" << site << " root=" << root->getName()
               << " elemsPerCore=" << elemsPerCore << " vecWidth=" << width
               << " eligible=" << eligible << " closure=" << closureSize;
  if (cands >= 0)
    llvm::errs() << " cands=" << cands;
  llvm::errs() << " loc=" << root->getLoc() << "\n";
}

Fit vectorFitUnknown(Value value, FitQuery query, unsigned wantWidth) {
  return Fit::Unknown;
}

Fit VectorizabilityAnalysis::askFit(Value value, FitQuery query,
                                    unsigned wantWidth) {
  Fit fit = fitOracle(value, query, wantWidth);
  if (fit == Fit::Unknown)
    fitCandidates.push_back({value, query, wantWidth});
  return fit;
}

Operation *VectorizabilityAnalysis::getBlockArgumentOp(Value arg) {
  BlockArgument blockArg = mlir::dyn_cast<BlockArgument>(arg);
  Block *block = blockArg.getOwner();
  unsigned argIndex = blockArg.getArgNumber();

  if (auto forOp = dyn_cast<mlir::scf::ForOp>(block->getParentOp())) {
    // TODO[dyq]: check getIterOperands -> getInitArgs
    Value initValue =
        forOp.getInitArgs()[argIndex - forOp.getNumInductionVars()];
    return initValue.getDefiningOp();
  }
  llvm_unreachable(
      "[Vectorization]: Operand is Not a BlockArgument of scf::for.");
  return nullptr;
}

bool VectorizabilityAnalysis::binLikeOpVectorize(Value lhs, Value rhs,
                                                 OperationTree &visited,
                                                 OperationTree &vectorizedOps) {
  bool isFP32Ty = getElementTypeOrSelf(lhs.getType()).isF32() &&
                  getElementTypeOrSelf(rhs.getType()).isF32();
  bool isFP16Ty = getElementTypeOrSelf(lhs.getType()).isF16() &&
                  getElementTypeOrSelf(rhs.getType()).isF16();
  bool isINT32Ty = getElementTypeOrSelf(lhs.getType()).isInteger(32) &&
                   getElementTypeOrSelf(rhs.getType()).isInteger(32);
  bool isINT16Ty = getElementTypeOrSelf(lhs.getType()).isInteger(16) &&
                   getElementTypeOrSelf(rhs.getType()).isInteger(16);
  bool isINT8Ty = getElementTypeOrSelf(lhs.getType()).isInteger(8) &&
                  getElementTypeOrSelf(rhs.getType()).isInteger(8);
  if (!isFP32Ty && !isFP16Ty && !isINT32Ty && !isINT16Ty && !isINT8Ty) {
    return false;
  }

  bool isVectorized = false;

  Operation *lhsOp = lhs.getDefiningOp();
  Operation *rhsOp = rhs.getDefiningOp();

  Operation *lhsLoopInitOp = nullptr;
  Operation *rhsLoopInitOp = nullptr;

  if (mlir::isa<BlockArgument>(lhs)) {
    lhsLoopInitOp = getBlockArgumentOp(lhs);
  }

  if (mlir::isa<BlockArgument>(rhs)) {
    rhsLoopInitOp = getBlockArgumentOp(rhs);
  }

  bool lhsVectorized = lhsOp ? vectorize(lhsOp, visited, vectorizedOps)
                             : vectorize(lhsLoopInitOp, visited, vectorizedOps);
  bool rhsVectorized = rhsOp ? vectorize(rhsOp, visited, vectorizedOps)
                             : vectorize(rhsLoopInitOp, visited, vectorizedOps);

  isVectorized = lhsVectorized && rhsVectorized;
  return isVectorized;
}

bool VectorizabilityAnalysis::vectorize(Operation *op, OperationTree &visited,
                                        OperationTree &vectorizedOps) {
  if (!op) {
    return false;
  }
  visited.insert(op);

  if (vectorizedOps.contains(op))
    return true;

  bool isVectorized = false;
  TypeSwitch<const Operation *>(op)
      .Case<triton::xpu::GM2LMOp>([&](auto gm2lmOp) { isVectorized = true; })
      .Case<triton::xpu::GM2LMMaskOp>(
          [&](auto gm2lmmaskOp) { isVectorized = true; })
      .Case<triton::xpu::LM2GMOp>([&](auto lm2gmOp) { isVectorized = true; })
      .Case<triton::xpu::LM2GMMaskOp>(
          [&](auto lm2gmmaskOp) { isVectorized = true; })
      .Case<triton::xpu::GetCoreIdOp>(
          [&](auto coreIdOp) { isVectorized = true; })
      .Case<triton::GetProgramIdOp>(
          [&](auto programIdOp) { isVectorized = true; })
      .Case<arith::ConstantOp>([&](auto constOp) { isVectorized = true; })
      .Case<arith::IndexCastOp>([&](auto unaryOp) { isVectorized = true; })
      .Case<triton::xpu::LoadOp>([&](auto loadOp) {
        Type elemTy = getElementTypeOrSelf(loadOp.getType());
        // Pointers and an already-vectorized element type have no bitwidth.
        if (!elemTy.isIntOrFloat())
          return;
        auto vectorWidth = 512 / elemTy.getIntOrFloatBitWidth();
        // This case has no E-independent half (§2.1.1): the footprint question
        // is all there is, so it goes to the oracle whole. Note it does not
        // consult `vectorizedTyValid` and does not factor out `rowsPerCore`,
        // unlike `vectorFitsRoot` -- inconsistencies #1 and #2, kept verbatim.
        isVectorized = askFit(loadOp.getResult(), FitQuery::WholeVectors,
                              vectorWidth) != Fit::No;
      })
      .Case<triton::xpu::StoreOp>([&](auto storeOp) {
        isVectorized = vectorize(storeOp.getValue().getDefiningOp(), visited,
                                 vectorizedOps);
      })
      .Case<triton::xpu::ReduceOp>([&](auto reduceOp) {
        if (ReduceVec) {
          isVectorized = true;

          // The trailing operand is the loop index, so `size() - 1` counts the
          // inputs -- unsigned, hence the guard instead of a wrap.
          if (reduceOp.getOperands().size() < 2) {
            isVectorized = false;
            return;
          }

          for (int i = 0; i < reduceOp.getOperands().size() - 1; ++i) {
            auto reduceOperand = reduceOp.getOperands()[i];
            auto reduceOperandTy = reduceOperand.getType();

            if (!reduceOperandFits(reduceOp, reduceOperandTy)) {
              isVectorized = false;
            }
          }

          if (!reduceCombineIsVectorizable(reduceOp))
            isVectorized = false;
        } else {
          isVectorized = false;
        }
      })
      .Case<triton::xpu::ExtractOp>([&](auto extractOp) {
        isVectorized = vectorize(extractOp.getTensor().getDefiningOp(), visited,
                                 vectorizedOps);
      })
      .Case<triton::SplatOp>([&](auto splatOp) {
        auto defineOp = splatOp.getSrc().getDefiningOp();
        if (!defineOp) { // some splatOp deal in_ptr
          isVectorized = true;
        } else if (!mlir::isa<RankedTensorType>(splatOp.getSrc().getType())) {
          // A scalar source holds one element per thread by construction
          // (ClusterLayoutAttr aside, Dialect.cpp:140-141), no distribution
          // involved -- that is this case's E-independent half.
          isVectorized = true;
        } else { // some splatOp deal tensor
          isVectorized = askFit(splatOp.getSrc(), FitQuery::SingleElem,
                                /*wantWidth=*/1) != Fit::No;
        }
      })
      .Case<triton::xpu::BroadcastOp>([&](auto broadCastOp) {
        // Some BroadcastOp From ReduceOp
        auto srcTy =
            mlir::dyn_cast<RankedTensorType>(broadCastOp.getSrc().getType());
        auto resTy =
            mlir::dyn_cast<RankedTensorType>(broadCastOp.getResult().getType());

        // Unranked or non-tensor operands cannot be reasoned about here; the
        // conservative answer is the `isVectorized = false` this case starts
        // with.
        if (!srcTy || !resTy)
          return;

        auto srcShape = srcTy.getShape();
        auto resShape = resTy.getShape();

        auto rank = srcTy.getRank();

        // The rank and the two shape relations are the E-independent half; the
        // element count is the oracle's. Testing them in this order only skips
        // oracle calls, both predicates are pure.
        if (rank == 2) {
          // srcShape[0] > 32: Scalar Calculations Perform Better than Vector
          // Calculations When The Data Size is Small. (Why > 32? Which Op?)
          if ((srcShape[0] == resShape[0] && srcShape[1] == 1) ||
              (srcShape[0] == 1 && srcShape[1] == resShape[1])) {
            // 16 regardless of element type: right for f32, wrong for f16/i8
            // (should be 32/64). Inconsistency #3 of §2.1.1, kept verbatim and
            // deliberately left at the call site rather than in the oracle.
            isVectorized =
                askFit(broadCastOp.getResult(), FitQuery::AtLeastWidth,
                       /*wantWidth=*/16) != Fit::No;
          }
        }
      })
      .Case<triton::ExpandDimsOp>([&](auto expandDimsOp) {
        isVectorized = vectorize(expandDimsOp.getOperand().getDefiningOp(),
                                 visited, vectorizedOps);
      })
      .Case<triton::AddPtrOp>([&](auto addPtrOp) {
        isVectorized = vectorize(addPtrOp.getPtr().getDefiningOp(), visited,
                                 vectorizedOps) &&
                       vectorize(addPtrOp.getOffset().getDefiningOp(), visited,
                                 vectorizedOps);
      })
      .Case<triton::xpu::ConvertLayoutOp>([&](auto cvtOp) {
        auto cvtResTy =
            mlir::dyn_cast<RankedTensorType>(cvtOp.getResult().getType());
        if (!cvtResTy)
          return;
        auto cvtOpResEncoding = cvtResTy.getEncoding();
        if (isa<triton::xpu::ClusterLayoutAttr>(cvtOpResEncoding)) {
          isVectorized = vectorize(cvtOp.getOperand().getDefiningOp(), visited,
                                   vectorizedOps);
        }
      })
      .Case<arith::SelectOp>([&](auto selectOp) {
        auto tv = selectOp.getTrueValue();
        auto fv = selectOp.getFalseValue();
        auto tType = getElementTypeOrSelf(tv.getType());
        auto fType = getElementTypeOrSelf(fv.getType());
        isVectorized = (tType == fType && (tType.isF16() || tType.isF32()) &&
                        binLikeOpVectorize(tv, fv, visited, vectorizedOps));
      })
      .Case<arith::CmpIOp>([&](auto cmpIOp) {
        isVectorized = false;
        // TODO: Add vCmpIOp Support
        //   auto lhs = cmpIOp.getLhs();
        //   auto rhs = cmpIOp.getRhs();
        //   isVectorized = binLikeOpVectorize(lhs, rhs, visited,
        //   vectorizedOps);
      })
      .Case<arith::CmpFOp>([&](auto cmpFOp) {
        auto lhs = cmpFOp.getLhs();
        auto rhs = cmpFOp.getRhs();
        isVectorized = binLikeOpVectorize(lhs, rhs, visited, vectorizedOps);
      })
      .Case<arith::TruncIOp>([&](auto truncIOp) {
        isVectorized = false;
        if (auto extElemwiseOp = dyn_cast_or_null<triton::ExternElementwiseOp>(
                truncIOp.getIn().getDefiningOp())) {
          isVectorized =
              vectorize(extElemwiseOp.getOperands().front().getDefiningOp(),
                        visited, vectorizedOps);
        }
      })
      .Case<triton::xpu::CmpFOp>([&](auto cmpFOp) {
        auto lhs = cmpFOp.getLhs();
        auto rhs = cmpFOp.getRhs();
        isVectorized = binLikeOpVectorize(lhs, rhs, visited, vectorizedOps);
      })
      .Case<scf::IfOp>([&](auto ifOp) {
        // For then Region
        Region &thenRegion = ifOp.getThenRegion();
        // getTerminator() asserts on an empty region or a block that does not
        // end in a terminator, so a malformed region answers "not vectorizable"
        // rather than tripping the assert.
        if (thenRegion.empty() || !thenRegion.front().mightHaveTerminator())
          return;
        Block &thenBlock = thenRegion.front();
        Operation *thenTerminator = thenBlock.getTerminator();
        isVectorized = true;
        if (auto yieldOp = dyn_cast<scf::YieldOp>(thenTerminator)) {
          for (int i = 0; i < yieldOp.getOperands().size(); ++i) {
            if (auto yieldDef = yieldOp.getOperands()[i].getDefiningOp()) {
              isVectorized &= vectorize(yieldDef, visited, vectorizedOps);
            }
          }
        }

        // For Else Region
        if (!ifOp.getElseRegion().empty() &&
            ifOp.getElseRegion().front().mightHaveTerminator()) {
          Region &elseRegion = ifOp.getElseRegion();
          Block &elseBlock = elseRegion.front();
          Operation *elseTerminator = elseBlock.getTerminator();
          if (auto yieldOp = dyn_cast<scf::YieldOp>(elseTerminator)) {
            for (int i = 0; i < yieldOp.getOperands().size(); ++i) {
              if (auto yieldDef = yieldOp.getOperands()[i].getDefiningOp()) {
                isVectorized &= vectorize(yieldDef, visited, vectorizedOps);
              }
            }
          }
        }
      })
      .Case<scf::ForOp>([&](auto forOp) {
        // TODO[dyq]: check getIterOperands -> getInitArgs
        auto iterArgsInitValues = forOp.getInitArgs();
        Region &region = forOp.getRegion();
        if (region.empty() || !region.front().mightHaveTerminator())
          return;
        Block &block = region.front();
        Operation *terminator = block.getTerminator();
        isVectorized = true;
        if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
          for (int i = 0; i < yieldOp.getOperands().size(); ++i) {
            if (auto yieldDef = yieldOp.getOperands()[i].getDefiningOp()) {
              isVectorized &= vectorize(yieldDef, visited, vectorizedOps);
            }
          }
        }
      })
      .Case<scf::YieldOp>([&](auto yieldOp) {
        isVectorized = true;
        for (int i = 0; i < yieldOp.getOperands().size(); ++i) {
          if (auto yieldDef = yieldOp.getOperands()[i].getDefiningOp()) {
            isVectorized &= vectorize(yieldDef, visited, vectorizedOps);
          }
        }
      })
      .Case<triton::ExternElementwiseOp>([&](auto extElemwiseOp) {
        auto symbol = extElemwiseOp.getSymbol();
        // front() below reads the first operand, so check before, not after.
        if (extElemwiseOp.getOperands().empty()) {
          isVectorized = false;
          return;
        }
        auto prevOp = extElemwiseOp.getOperands().front().getDefiningOp();
        if (symbol == "_ZN3xpu5tanhfEf") {
          isVectorized = true;
          for (auto operand : extElemwiseOp.getOperands()) {
            isVectorized =
                isVectorized && vectorize(prevOp, visited, vectorizedOps);
          }
        } else if (symbol == "_ZN3xpu4tanfEf") {
          isVectorized = true;
          for (auto operand : extElemwiseOp.getOperands()) {
            isVectorized =
                isVectorized && vectorize(prevOp, visited, vectorizedOps);
          }
        } else if (symbol == "_ZN3xpu3erfEf") {
          isVectorized = true;
          for (auto operand : extElemwiseOp.getOperands()) {
            isVectorized =
                isVectorized && vectorize(prevOp, visited, vectorizedOps);
          }
        } else if (symbol == "_ZN3xpu5atanfEf") {
          isVectorized = true;
          for (auto operand : extElemwiseOp.getOperands()) {
            isVectorized =
                isVectorized && vectorize(prevOp, visited, vectorizedOps);
          }
        } else if (symbol == "_ZN3xpu5isinfEf") {
          isVectorized = false;
          // TODO: check visinf logic
          // isVectorized = true;
          // for (auto operand : extElemwiseOp.getOperands()) {
          //   isVectorized =
          //       isVectorized && vectorize(prevOp, visited, vectorizedOps);
          // }
        } else if (symbol == "_ZN3xpu5isnanEf") {
          isVectorized = true;
          for (auto operand : extElemwiseOp.getOperands()) {
            isVectorized = vectorize(prevOp, visited, vectorizedOps);
          }
        } else if (symbol == "_ZN3xpu6rsqrtfEf") {
          auto outType =
              getElementTypeOrSelf(extElemwiseOp.getResult().getType());
          for (auto operand : extElemwiseOp.getOperands()) {
            isVectorized =
                outType.isF32() && vectorize(prevOp, visited, vectorizedOps);
          }
        } else {
          isVectorized = false;
          LLVM_DEBUG(llvm::dbgs()
                     << "[Vectorization]: Unsupported LibDeviceOp Symbol"
                     << symbol << "\n");
        }
      })
      .Case<arith::SIToFPOp>([&](arith::SIToFPOp unaryOp) {
        auto inType = getElementTypeOrSelf(unaryOp.getIn().getType());
        isVectorized = inType.isInteger(32) &&
                       vectorize(unaryOp.getOperand().getDefiningOp(), visited,
                                 vectorizedOps);
      })
      .Case<ARITH_BINARY_FLOAT_OP>([&](auto binOp) {
        auto lhs = binOp.getLhs();
        auto rhs = binOp.getRhs();
        isVectorized = binLikeOpVectorize(lhs, rhs, visited, vectorizedOps);
      })
      .Case<ARITH_BINARY_INT_OP>([&](auto binOp) {
        auto lhs = binOp.getLhs();
        auto rhs = binOp.getRhs();
        isVectorized = binLikeOpVectorize(lhs, rhs, visited, vectorizedOps);
      })
      .Case<MATH_UNARY_OP>([&](auto unaryOp) {
        isVectorized = vectorize(unaryOp.getOperand().getDefiningOp(), visited,
                                 vectorizedOps);
      });

  if (!isVectorized) {
    if (dumpFlag) {
      LLVM_DEBUG({
        op->dump();
        llvm_unreachable("[Vectorization]: Unsupported Operation");
      });
    }
    return false;
  }

  // Dont Need To Vectorize ReduceOp's Result
  if (auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(op))
    return true;

  for (Operation *user : op->getUsers()) {
    if (visited.contains(user))
      continue;

    // FIXME: We've omitted the `other` value of LoadOp when create GM2LMOp
    // in the past. However, `other` value comes back as we are about to
    // separate GM2LMOp and LoadOp, and it will lead to a user LoadOp be in
    // the vectorization path. Actions should be taken to handle this case.
    // Here we workaround to skip LoadOp's `other` value.
    if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(user)) {
      if (op == loadOp.getOther().getDefiningOp()) {
        continue;
      }
    }

    if (!vectorize(user, visited, vectorizedOps))
      return false;
  }

  vectorizedOps.insert(op);
  return true;
}

//===----------------------------------------------------------------------===//
// Vector-Flow analysis (step 2.1). See the header for why this is a partition
// and not a lattice climb.
//===----------------------------------------------------------------------===//

const char *toString(VState state) {
  switch (state) {
  case VState::Unset:
    return "Unset";
  case VState::Vector:
    return "Vector";
  case VState::Scalar:
    return "Scalar";
  case VState::Conflict:
    return "Conflict";
  }
  return "?";
}

bool vflowReportEnabled() {
  return mlir::triton::tools::getBoolEnvXPU("TRITONXPU_VFLOW_REPORT");
}

namespace {
// Unset is the identity; two different pins are a boundary, not an error.
VState joinState(VState a, VState b) {
  if (a == b)
    return a;
  if (a == VState::Unset)
    return b;
  if (b == VState::Unset)
    return a;
  return VState::Conflict;
}

// Values that carry data whose representation is the question. Pointers, index
// and i1 plumbing are not: nothing retypes them, so putting them in classes
// would only inflate the counts.
bool isDataValue(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return false;
  Type elemTy = tensorTy.getElementType();
  return isa<VectorType>(elemTy) || elemTy.isIntOrFloat();
}
} // namespace

unsigned VectorFlowAnalysis::idOf(Value value) {
  auto it = ids.find(value);
  if (it != ids.end())
    return it->second;
  unsigned id = parent.size();
  ids.insert({value, id});
  parent.push_back(id);
  pins.push_back(VState::Unset);
  ++stats.values;
  return id;
}

unsigned VectorFlowAnalysis::find(unsigned id) {
  while (parent[id] != id) {
    parent[id] = parent[parent[id]];
    id = parent[id];
  }
  return id;
}

void VectorFlowAnalysis::unite(Value a, Value b) {
  if (!isDataValue(a) || !isDataValue(b))
    return;
  unsigned ra = find(idOf(a));
  unsigned rb = find(idOf(b));
  if (ra == rb)
    return;
  parent[rb] = ra;
  pins[ra] = joinState(pins[ra], pins[rb]);
  ++stats.unions;
}

void VectorFlowAnalysis::pin(Value value, VState state) {
  if (!isDataValue(value))
    return;
  unsigned root = find(idOf(value));
  pins[root] = joinState(pins[root], state);
  if (state == VState::Vector)
    ++stats.vectorPins;
  else if (state == VState::Scalar)
    ++stats.scalarPins;
}

VState VectorFlowAnalysis::stateOf(Value value) const {
  auto it = ids.find(value);
  if (it == ids.end())
    return VState::Unset;
  // `find` compresses, so do the walk read-only here.
  unsigned id = it->second;
  while (parent[id] != id)
    id = parent[id];
  return pins[id];
}

// Mirrors the closure walk's own gate (this file, the
// `.Case<triton::ExternElementwiseOp>` at ~:427), which is the source of truth
// for whether a libdevice call has a vector form: processOpVecTy's table
// (Vectorize.cpp:404-455) is wider -- it also rewrites isinf -- but the gate
// rejects isinf, so those entries are unreachable and admitting them here would
// pin Vector on a chain that cannot vectorize.
//
// Element type is checked the same way as for the element-wise table above: i1
// has no vector form, which drops isnan to Scalar. isnan is the one entry whose
// rewrite retypes to a vector of the *input* bitwidth (Vectorize.cpp:436-440),
// fused with the trunci that consumes it; that fused boundary is its own step
// (the `truncint` probe is its minimal shape), so keep the conservative answer.
static bool externSymbolHasVectorForm(triton::ExternElementwiseOp extOp) {
  if (extOp->getNumResults() != 1)
    return false;
  Type elemTy = getElementTypeOrSelf(extOp.getResult().getType());
  if (!isa<VectorType>(elemTy) && !vectorizedTyValid(elemTy))
    return false;
  StringRef symbol = extOp.getSymbol();
  if (symbol == "_ZN3xpu6rsqrtfEf")
    return elemTy.isF32(); // guarded by outType.isF32() in the gate
  return symbol == "_ZN3xpu5tanhfEf" || symbol == "_ZN3xpu4tanfEf" ||
         symbol == "_ZN3xpu3erfEf" || symbol == "_ZN3xpu5atanfEf" ||
         symbol == "_ZN3xpu5isnanEf";
}

void VectorFlowAnalysis::visit(Operation *op) {
  // Element-wise, per the single-sourced table. `select` and the two compares
  // ride along: they are retyped as one unit with their operands even though
  // the compare's result element type differs.
  if (hasVectorForm(op) ||
      isa<arith::SelectOp, arith::CmpFOp, triton::xpu::CmpFOp>(op)) {
    if (op->getNumResults() != 1)
      return;
    Value result = op->getResult(0);
    // A select's condition is not an equality edge. Vectorize builds VSelectOp
    // with `selectOp.getCondition()` passed straight through, unretyped
    // (Vectorize.cpp:332-335), so the mask follows its own producer: welford
    // holds both shapes in one kernel, a vcmpf-fed vselect taking
    // vector<16xi1> (welford.ttxir:110) and a mask-fed one taking a plain
    // tensor<1x4096xi1> (:103). Uniting it dragged welford's whole f32
    // accumulator class Scalar through that i1 mask -- the two disagreeing
    // roots step 2.1 measured. CmpFOp keeps the union: its result *is* retyped
    // to vector<16xi1> together with its operands (Vectorize.cpp:339-348).
    auto selectOp = dyn_cast<arith::SelectOp>(op);
    for (Value operand : op->getOperands()) {
      if (selectOp && operand == selectOp.getCondition())
        continue;
      unite(result, operand);
    }
    // Op kind is the table's business; element type is not. i1 has no vector
    // form, which is why a bool chain ends up Scalar here rather than Unset.
    Type elemTy = getElementTypeOrSelf(result.getType());
    if (!isa<VectorType>(elemTy) && !vectorizedTyValid(elemTy))
      pin(result, VState::Scalar);
    return;
  }

  TypeSwitch<Operation *>(op)
      .Case<triton::xpu::LoadOp>([&](auto loadOp) {
        // Seed, but only the E-independent half is decided here: the footprint
        // question goes to the oracle, and `Unknown` seeds nothing rather than
        // guessing (this analysis may end up running before CoreTiling).
        Value result = loadOp.getResult();
        Type elemTy = getElementTypeOrSelf(result.getType());
        if (!elemTy.isIntOrFloat() || !vectorizedTyValid(elemTy)) {
          pin(result, VState::Scalar);
          return;
        }
        unsigned width = getVectorWidth(elemTy);
        Fit fit = fitOracle(result, FitQuery::WholeVectors, width);
        if (fit == Fit::Yes)
          pin(result, VState::Vector);
        else if (fit == Fit::No)
          pin(result, VState::Scalar);
      })
      .Case<triton::xpu::StoreOp>(
          [&](auto storeOp) { pin(storeOp.getValue(), VState::Vector); })
      .Case<triton::xpu::ReduceOp>([&](auto reduceOp) {
        // The transfer function of step 2.2 (§3.1): operands may be Vector, the
        // entry is a boundary, the results are Scalar.
        //
        // Results, unconditionally: an XPU reduce yields one element per row
        // (`tensor<1xf32>`, welford.ttxir:135) under either lowering, so there
        // is no vector form for it to want. Note this is a pin on the *result*
        // class alone -- no edge joins it to the operands, and that absence is
        // what makes the entry a boundary instead of a Conflict.
        for (Value result : reduceOp.getResults())
          pin(result, VState::Scalar);

        // No pin on the data operands, in either direction.
        //
        // Step 2.1 pinned them Scalar whenever the combine region could not be
        // retyped, so that the partition would agree with the all-or-nothing
        // closure walk. That is the pin 2.2 removes: it asserts "the producer
        // chain must stay scalar", where the truth is "the chain may be Vector
        // and the reduce unpacks at its entry" -- the boundary this analysis
        // exists to report, and the one the closure walk structurally cannot
        // express. Pinning Vector instead would be just as wrong: a scalar
        // island producer (i1 chain, extern_elementwise) legitimately arrives
        // Scalar. So each operand keeps its own class, and the boundaries are
        // counted in `run` once the partition is final.
        //
        // Consequence to expect in the report, not to paper over: on a reduce
        // whose combine cannot be retyped, the walk says 0 and this says
        // Vector. That disagreement is the modelled boundary; VFlow's per-root
        // line classifies it as `kind=boundary` rather than folding it into
        // agree=1.
      })
      .Case<triton::xpu::BroadcastOp>([&](auto bcOp) {
        // A broadcast that replicates along the innermost (vectorized) axis is
        // a Scalar -> Vector boundary, not an equality edge: processOpVecTy
        // retypes only the *result* (Vectorize.cpp:383-387), leaving the source
        // scalar, and SVOptimization_Cond recognises exactly this shape
        // (`srcShape[1] == 1`, Vectorize.cpp:693-701). Uniting here is what
        // made every reduce-consuming kernel report Conflict: the Scalar-pinned
        // reduce result reached the Vector-pinned elementwise chain through the
        // broadcast.
        auto srcTy = mlir::dyn_cast<RankedTensorType>(bcOp.getSrc().getType());
        auto resTy =
            mlir::dyn_cast<RankedTensorType>(bcOp.getResult().getType());
        if (srcTy && resTy && !srcTy.getShape().empty() &&
            !resTy.getShape().empty() && srcTy.getShape().back() == 1 &&
            resTy.getShape().back() > 1)
          return;
        unite(bcOp.getResult(), bcOp.getSrc());
      })
      .Case<triton::ExpandDimsOp, triton::xpu::ConvertLayoutOp,
            triton::xpu::ExtractOp>([&](Operation *passThrough) {
        // Retyped in place, so source and result share a state.
        if (passThrough->getNumResults() == 1 &&
            passThrough->getNumOperands() >= 1)
          unite(passThrough->getResult(0), passThrough->getOperand(0));
      })
      .Case<triton::SplatOp, arith::ConstantOp>([&](Operation *materialized) {
        // Rebuilt as VSplatOp / VConstOp with a scalar source, so the result is
        // free and no edge crosses into the source.
      })
      .Case<scf::ForOp>([&](auto forOp) {
        // init <-> iter_arg <-> yield <-> result, per index. These four are the
        // edges the closure walk cannot express, and the reason a loop-carried
        // value currently vetoes on the way in.
        auto yieldOp =
            dyn_cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
        for (unsigned i = 0, e = forOp.getInitArgs().size(); i < e; ++i) {
          Value iterArg = forOp.getRegionIterArgs()[i];
          unite(iterArg, forOp.getInitArgs()[i]);
          unite(iterArg, forOp.getResult(i));
          if (yieldOp && i < yieldOp.getOperands().size())
            unite(iterArg, yieldOp.getOperands()[i]);
        }
      })
      .Case<scf::IfOp>([&](auto ifOp) {
        for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
          if (region->empty() || !region->front().mightHaveTerminator())
            continue;
          auto yieldOp =
              dyn_cast<scf::YieldOp>(region->front().getTerminator());
          if (!yieldOp)
            continue;
          for (unsigned i = 0, e = std::min(yieldOp.getOperands().size(),
                                            ifOp.getResults().size());
               i < e; ++i)
            unite(ifOp.getResult(i), yieldOp.getOperands()[i]);
        }
      })
      .Case<triton::ExternElementwiseOp>([&](auto extOp) {
        // The pin follows the symbol whitelist, not the op kind. Pinning every
        // extern Scalar (what step 2.1 did) mis-states a chain that vectorizes
        // *today*: `libdev` (tanhf) becomes vtanhf on
        // tensor<512xvector<16xf32>>, so the unconditional pin turned the store
        // class Conflict and would make a per-op decision either cut a boundary
        // mid-chain or drop the chain to scalar -- both regressions. Measured
        // in findings §1.54.
        bool vectorSymbol = externSymbolHasVectorForm(extOp);
        for (Value result : extOp->getResults()) {
          pin(result, vectorSymbol ? VState::Vector : VState::Scalar);
          ++stats.externPins;
        }
        // Equality edge, only on the shape the rewrite actually rewires: it
        // hands the vector symbol `extElemwiseOp.getOperands().front()`
        // (Vectorize.cpp:370-372) and the vector symbols are unary
        // (`...EDv16_f`), so a multi-operand extern gets the pin without the
        // edge rather than an edge the rewrite does not make.
        if (vectorSymbol && extOp->getNumOperands() == 1)
          unite(extOp->getResult(0), extOp->getOperand(0));
      })
      .Case<scf::YieldOp, triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp,
            triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp>(
          [&](Operation *handledElsewhere) {})
      .Default([&](Operation *other) {
        // Not modelled: pin whatever data it produces Scalar. A local scalar
        // island is the conservative answer and, unlike the closure walk's
        // veto, it stays local instead of propagating to the root.
        for (Value result : other->getResults()) {
          if (!isDataValue(result))
            continue;
          pin(result, VState::Scalar);
          ++stats.unknownPins;
        }
      });
}

bool rewriteNoopKind(Operation *op) {
  return isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp,
             triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp,
             triton::xpu::StoreOp>(op);
}

bool sameActingSet(const OperationTree &lhs, const OperationTree &rhs) {
  auto acting = [](const OperationTree &set) {
    llvm::SmallVector<Operation *> ops;
    for (Operation *op : set)
      if (!rewriteNoopKind(op))
        ops.push_back(op);
    return ops;
  };
  llvm::SmallVector<Operation *> a = acting(lhs), b = acting(rhs);
  if (a.size() != b.size())
    return false;
  llvm::SmallPtrSet<Operation *, 16> seen(a.begin(), a.end());
  for (Operation *op : b)
    if (!seen.count(op))
      return false;
  return true;
}

VecSetDomain buildVecSetDomain(Value keyValue, const VectorFlowAnalysis &vflow,
                               const OperationTree &closure) {
  VecSetDomain domain;
  if (!keyValue)
    return domain;

  llvm::SmallVector<Value> worklist{keyValue};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *def = value.getDefiningOp();
    if (!def || !domain.cone.insert(def))
      continue;
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
    // Region-carrying ops contribute their terminator operands, matching the
    // closure walk's ForOp / YieldOp cases.
    for (Region &region : def->getRegions())
      for (Block &block : region)
        if (Operation *terminator = block.getTerminator())
          for (Value operand : terminator->getOperands())
            worklist.push_back(operand);
  }
  for (Operation *op : closure)
    if (!domain.cone.count(op)) {
      domain.offCone.insert(op);
      domain.cone.insert(op);
    }

  for (Operation *op : domain.cone) {
    bool wantsVector = false;
    for (Value result : op->getResults())
      wantsVector |= vflow.stateOf(result) == VState::Vector;
    if (op->getNumResults() == 0)
      for (Value operand : op->getOperands())
        wantsVector |= vflow.stateOf(operand) == VState::Vector;
    if (wantsVector)
      domain.term.insert(op);
  }
  return domain;
}

void VectorFlowAnalysis::run(triton::FuncOp func) {
  // Two passes so that the edges are all in place before the pins are counted
  // per class: pinning is order-independent only once the partition is final.
  func.walk([&](Operation *op) {
    if (hasVectorForm(op) ||
        isa<arith::SelectOp, arith::CmpFOp, triton::xpu::CmpFOp, scf::ForOp,
            scf::IfOp, triton::xpu::BroadcastOp, triton::ExpandDimsOp,
            triton::xpu::ConvertLayoutOp, triton::xpu::ExtractOp>(op))
      visit(op);
  });
  func.walk([&](Operation *op) {
    if (!(hasVectorForm(op) ||
          isa<arith::SelectOp, arith::CmpFOp, triton::xpu::CmpFOp, scf::ForOp,
              scf::IfOp, triton::xpu::BroadcastOp, triton::ExpandDimsOp,
              triton::xpu::ConvertLayoutOp, triton::xpu::ExtractOp>(op)))
      visit(op);
  });

  // Step 2.2: how many reduce entries actually are boundaries. Counted here
  // rather than in `visit` because a class is not final while the walk runs --
  // welford's accumulator operands only become Vector after the loop edges are
  // in place. `combineVec` decides whether the boundary costs a real unpack:
  // the region lowering consumes the vector operands inside the combine region,
  // the legacy one-op-per-output lowering cannot and needs the unpack in front.
  func.walk([&](triton::xpu::ReduceOp redOp) {
    if (redOp.getOperands().size() < 2)
      return;
    ++stats.reduceOps;
    bool combineVec = reduceCombineIsVectorizable(redOp);
    for (unsigned i = 0, e = redOp.getOperands().size() - 1; i < e; ++i) {
      if (stateOf(redOp.getOperands()[i]) != VState::Vector)
        continue;
      ++stats.reduceVectorEntries;
      if (!combineVec)
        ++stats.reduceEntryUnpacks;
    }
  });

  llvm::DenseSet<unsigned> roots;
  for (auto &entry : ids)
    roots.insert(find(entry.second));
  stats.classes = roots.size();
  for (unsigned root : roots) {
    switch (pins[root]) {
    case VState::Vector:
      ++stats.vectorClasses;
      break;
    case VState::Scalar:
      ++stats.scalarClasses;
      break;
    case VState::Conflict:
      ++stats.conflictClasses;
      break;
    case VState::Unset:
      ++stats.unsetClasses;
      break;
    }
  }
}

} // namespace xpu
} // namespace triton
} // namespace mlir
