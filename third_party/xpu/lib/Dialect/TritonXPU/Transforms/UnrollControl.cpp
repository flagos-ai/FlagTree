#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-unroll-control"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUUNROLLCONTROL
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

template <typename OP> struct COMOp;

#define COMOP(SrcType, DstType)                                                \
  template <> struct COMOp<SrcType> {                                          \
    typedef DstType type;                                                      \
  };

COMOP(arith::AddFOp, triton::xpu::VvaddFOp);
COMOP(arith::MulFOp, triton::xpu::VvmulFOp);
COMOP(arith::MaxNumFOp, triton::xpu::VvmaxNumFOp);
COMOP(arith::MinNumFOp, triton::xpu::VvminNumFOp);
COMOP(arith::OrIOp, triton::xpu::VvorIOp);
COMOP(arith::XOrIOp, triton::xpu::VvxorIOp);
COMOP(arith::AndIOp, triton::xpu::VvandIOp);

struct TritonXPUUnrollControl
    : public impl::TritonXPUUnrollControlBase<TritonXPUUnrollControl> {

public:
  using impl::TritonXPUUnrollControlBase<
      TritonXPUUnrollControl>::TritonXPUUnrollControlBase;

  TritonXPUUnrollControl() = default;
  TritonXPUUnrollControl(unsigned bufferSize, unsigned coreNum,
                         unsigned unrollNum) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
    this->unrollNum = unrollNum;
  }

  template <typename T> static decltype(auto) createCombineVectorizedOp(T op) {
    OpBuilder builder(op);
    return builder.create<typename COMOp<T>::type>(
        op.getLoc(), op.getResult().getType(), op.getLhs(), op.getRhs());
  }

  void processOpVecTy(ModuleOp &m) {
    m.walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<COMBINE_BINARY_OP>([&](auto combineBinaryOp) {
            if (auto tensorTy = dyn_cast<RankedTensorType>(
                    combineBinaryOp.getResult().getType())) {
              if (isa<VectorType>(getElementTypeOrSelf(tensorTy))) {
                auto vecOp = createCombineVectorizedOp(combineBinaryOp);
                combineBinaryOp.replaceAllUsesWith(vecOp.getResult());
                combineBinaryOp.erase();
              }
            }
          })
          .Case<arith::CmpFOp>([&](auto cmpFOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(cmpFOp.getResult().getType())) {
              if (isa<VectorType>(getElementTypeOrSelf(tensorTy))) {
                OpBuilder builder(cmpFOp);
                auto vecOp = builder.create<triton::xpu::VCmpFOp>(
                    cmpFOp.getLoc(), cmpFOp.getResult().getType(),
                    cmpFOp.getPredicate(), cmpFOp.getLhs(), cmpFOp.getRhs());
                ;
                cmpFOp.replaceAllUsesWith(vecOp.getResult());
                cmpFOp.erase();
              }
            }
          });
    });
  }

  bool isAncestorOf(Operation *op1, Operation *op2, bool needBefore = false) {
    Block *block1 = op1->getBlock();
    for (Block *block2 = op2->getBlock(); block2 != nullptr;) {
      if (block1 == block2) {
        if (needBefore && !op1->isBeforeInBlock(op2)) {
          return false;
        }
        return true;
      }
      op2 = block2->getParentOp();
      if (op2 == nullptr) {
        break;
      }
      block2 = op2->getBlock();
    }
    return false;
  }

  bool isForBlockSizeArgument(Operation *op, Value operand) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Block *block = forOp.getBody();
      for (BlockArgument arg : block->getArguments()) {
        if (arg == operand) {
          return true;
        }
      }
    }
    return false;
  }

  void getUnrollTree(Operation *op, SetVector<Operation *> &opTree,
                     SetVector<Operation *> &visitedOps,
                     SetVector<Operation *> &excludeChainOps, Operation *rootOp,
                     bool isTop2Bottom = true, bool needBefore = false) {
    if (!op || visitedOps.count(op) ||
        isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp,
            triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp, scf::YieldOp,
            triton::xpu::ReduceOp, triton::xpu::ReduceReturnOp,
            triton::xpu::ScanOp, triton::xpu::ScanReturnOp>(op)) {
      return;
    }

    visitedOps.insert(op);
    if (isAncestorOf(op, rootOp, needBefore) ||
        op->getBlock() == rootOp->getBlock()) {
      opTree.insert(op);
    }

    // Search definedOp of childOp
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Then
      auto &ifThenBlock = ifOp.getThenRegion().front();
      for (auto &inBlockOp : ifThenBlock) {
        getUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps, rootOp,
                      isTop2Bottom, needBefore);
      }
      // Else
      auto &ifElseRegion = ifOp.getElseRegion();
      if (!ifElseRegion.empty()) {
        auto &ifElseBlock = ifElseRegion.front();
        for (auto &inBlockOp : ifElseBlock) {
          getUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps, rootOp,
                        isTop2Bottom, needBefore);
        }
      }
    }

    // from bottom to top
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      Block *body = forOp.getBody();
      for (auto &op : body->getOperations()) {
        if (isa<triton::xpu::LoadOp, arith::ConstantOp, triton::xpu::VConstOp>(
                &op)) {
        } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(&op)) {
          auto defOp = storeOp.getValue().getDefiningOp();
          if (!isAncestorOf(forOp.getOperation(), defOp)) {
            getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                          isTop2Bottom, needBefore);
          }
        } else {
          for (auto operand : op.getOperands()) {
            if (isForBlockSizeArgument(forOp.getOperation(), operand))
              continue;
            auto defOp = operand.getDefiningOp();
            if (!isAncestorOf(forOp.getOperation(), defOp)) {
              getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                            isTop2Bottom, needBefore);
            }
          }
        }
      }
    } else if (isa<triton::xpu::LoadOp, arith::ConstantOp,
                   triton::xpu::VConstOp>(op)) {
    } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
      auto defOp = storeOp.getValue().getDefiningOp();
      getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                    isTop2Bottom, needBefore);
    } else {
      for (auto operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        getUnrollTree(defOp, opTree, visitedOps, excludeChainOps, rootOp,
                      isTop2Bottom, needBefore);
      }
    }

    if (isTop2Bottom) {
      // from top to bottom
      if (excludeChainOps.count(op) ||
          isa<arith::ConstantOp, triton::xpu::VConstOp>(op)) {
      } else {
        for (auto userOp : op->getUsers()) {
          getUnrollTree(userOp, opTree, visitedOps, excludeChainOps, rootOp,
                        isTop2Bottom, needBefore);
        }
      }
    }
    return;
  }

  bool isOuterBroadcast(Operation *op) {
    if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
      auto src = broadcastOp.getSrc();
      auto result = broadcastOp.getResult();
      if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
        if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
          int64_t srcElemNum = 1;
          if (auto vecTy = dyn_cast<VectorType>(getElementTypeOrSelf(srcTy))) {
            srcElemNum = vecTy.getNumElements();
          }
          int64_t resElemNum = 1;
          if (auto vecTy = dyn_cast<VectorType>(getElementTypeOrSelf(resTy))) {
            resElemNum = vecTy.getNumElements();
          }
          auto srcShape = srcTy.getShape();
          auto resShape = resTy.getShape();
          int64_t srcInnerNum = srcElemNum * srcShape.back();
          int64_t resInnerNum = resElemNum * resShape.back();
          if (srcInnerNum != resInnerNum) { // unequal dim 1 shape means in
                                            // the inner axis op chain
            assert(srcShape.front() == resShape.front() && "Invalid BroadCast");
            return true;
          }
        }
      }
    }
    return false;
  }

  template <typename T> Operation *getVdefOp(T op) {
    Operation *vDefOp;
    auto elemState = static_cast<ElemState>(op.getElemState());
    if (elemState == ElemState::SV) {
      vDefOp = op.getRhs().getDefiningOp();
    } else if (elemState == ElemState::VS) {
      vDefOp = op.getLhs().getDefiningOp();
    } else {
      llvm_unreachable(
          "[Unroll Control]: ElemState the SVOp Only Could be SV/VS.");
    }
    return vDefOp;
  }

  void getPostReduceUnrollTree(Operation *op, SetVector<Operation *> &opTree,
                               SetVector<Operation *> &visitedOps,
                               SetVector<Operation *> &excludeChainOps,
                               Operation *rootOp) {
    if (!op || visitedOps.count(op) ||
        isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp,
            triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp, scf::YieldOp,
            triton::xpu::ReduceOp, triton::xpu::ReduceReturnOp,
            triton::xpu::ScanOp, triton::xpu::ScanReturnOp>(op)) {
      return;
    }

    visitedOps.insert(op);
    if (isAncestorOf(op, rootOp) || op->getBlock() == rootOp->getBlock()) {
      opTree.insert(op);
    }

    // Search definedOp of childOp
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // Then
      auto &ifThenBlock = ifOp.getThenRegion().front();
      for (auto &inBlockOp : ifThenBlock) {
        getPostReduceUnrollTree(&inBlockOp, opTree, visitedOps, excludeChainOps,
                                rootOp);
      }
      // Else
      auto &ifElseRegion = ifOp.getElseRegion();
      if (!ifElseRegion.empty()) {
        auto &ifElseBlock = ifElseRegion.front();
        for (auto &inBlockOp : ifElseBlock) {
          getPostReduceUnrollTree(&inBlockOp, opTree, visitedOps,
                                  excludeChainOps, rootOp);
        }
      }
    }

    // from bottom to top
    if (isa<triton::xpu::LoadOp, arith::ConstantOp, triton::xpu::VConstOp>(
            op) ||
        isOuterBroadcast(op)) {
    } else if (isa<XPU_SVECTORIZED_BINARY_OP>(op)) {
      TypeSwitch<Operation *>(op).Case<XPU_SVECTORIZED_BINARY_OP>(
          [&](auto vBinOp) {
            Operation *vDefOp = getVdefOp(vBinOp);
            getPostReduceUnrollTree(vDefOp, opTree, visitedOps, excludeChainOps,
                                    rootOp);
          });
    } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
      auto defOp = storeOp.getValue().getDefiningOp();
      getPostReduceUnrollTree(defOp, opTree, visitedOps, excludeChainOps,
                              rootOp);
    } else {
      for (auto operand : op->getOperands()) {
        auto defOp = operand.getDefiningOp();
        getPostReduceUnrollTree(defOp, opTree, visitedOps, excludeChainOps,
                                rootOp);
      }
    }

    return;
  }

  int64_t getNumCol(Type type) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(type))
      return tensorTy.getShape().back();
    else
      return 1;
  }

  int64_t getNumInVector(Type type) {
    if (auto vecType = dyn_cast<VectorType>(type))
      return vecType.getNumElements();
    else
      return 1;
  }

  int64_t getNumUnroll(Type type) {
    int64_t numUnroll = this->unrollNum * this->coreNum;
    if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
      auto clusterEncoding =
          cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
      numUnroll = this->unrollNum * clusterEncoding.getCoresPerGroup().back();
    }
    return numUnroll;
  }

  Type createPointerType(Type type, int64_t vecSize) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      Type elemType = getElementTypeOrSelf(tensorType);
      Type elemScalarType = getElementTypeOrSelf(elemType);
      Type pointerType = triton::PointerType::get(elemScalarType, 0);
      auto shape = tensorType.getShape().vec();
      shape[shape.size() - 1] = shape.back() * vecSize;
      return RankedTensorType::get(shape, pointerType,
                                   tensorType.getEncoding());
    } else {
      return triton::PointerType::get(type, 0);
    }
  }

  triton::xpu::ClusterLayoutAttr
  createEncoding(MLIRContext *context, triton::xpu::ClusterLayoutAttr &encoding,
                 int64_t iterNum) const {
    auto sizePerCore = encoding.getSizePerCore().vec();
    sizePerCore[sizePerCore.size() - 1] =
        ceil<int64_t>(sizePerCore.back(), iterNum);
    auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
        context, sizePerCore, encoding.getCoresPerGroup(),
        encoding.getGroupsPerCluster(), encoding.getOrder());
    return newEncoding;
  }

  void setTensorType(MLIRContext *context, Operation *op, int64_t iterNum,
                     bool isOuter, bool sliceShape = true) const {
    for (auto [i, resTy] : llvm::enumerate(op->getResultTypes())) {
      if (isa<RankedTensorType>(resTy) && !isOuter) {
        auto tensorTy = cast<RankedTensorType>(resTy);
        auto shape = tensorTy.getShape().vec();
        if (sliceShape) {
          shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
        }
        RankedTensorType controledTensorTy;
        if (auto sliceEncoding = dyn_cast<triton::gpu::SliceEncodingAttr>(
                tensorTy.getEncoding())) {
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(sliceEncoding.getParent());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          auto newEncoding = triton::gpu::SliceEncodingAttr::get(
              context, sliceEncoding.getDim(), newClusterEncoding);
          controledTensorTy = RankedTensorType::get(
              shape, tensorTy.getElementType(), newEncoding);
        } else {
          auto clusterEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
          auto newClusterEncoding =
              createEncoding(context, clusterEncoding, iterNum);
          controledTensorTy = RankedTensorType::get(
              shape, tensorTy.getElementType(), newClusterEncoding);
        }
        op->getResult(i).setType(controledTensorTy);
      }
    }
  }

  void setHoistedOperand(MLIRContext *context, OpBuilder &builder,
                         Location &loc, mlir::Block &block, scf::IfOp &ifOp,
                         int64_t iterNum) {
    for (auto &inBlockOp : block) {
      if (auto yieldOp = llvm::dyn_cast<scf::YieldOp>(&inBlockOp)) {
        unsigned numifOpResults = ifOp.getNumResults();
        unsigned numyieldOpOperands = yieldOp.getNumOperands();
        // isOperandValidInSameForBlock denotes two points:
        // 1. Extraction is required if the operand of YieldOp
        //    does not match the type expected by the result of IfOp
        // 2. whether the operand of YieldOp is in the same ForBlock as IfOp.
        SmallVector<bool, 4> isOperandValidInSameForBlock(numyieldOpOperands);
        assert((numifOpResults == numyieldOpOperands) &&
               "The number of IfOp results and YieldOp operands must match.");
        for (unsigned i = 0; i < numyieldOpOperands; ++i) {
          Type ifOpResTy = ifOp.getResult(i).getType();
          isOperandValidInSameForBlock[i] =
              isOperandOperationInSameForBlock(&inBlockOp, i) ||
              (inBlockOp.getOperand(i).getType() == ifOpResTy);
          if (!isOperandValidInSameForBlock[i]) {
            assert(isa<arith::ConstantOp>(
                       inBlockOp.getOperand(i).getDefiningOp()) &&
                   "Unable to extract the non-constant operand.");
            auto extractSliceOp =
                getExtractedOperand(context, builder, loc, yieldOp, i, iterNum);
            extractSliceOp->moveBefore(ifOp);
            inBlockOp.setOperand(i, extractSliceOp->getResult(0));
          }
        }
      } else if (((&inBlockOp)->hasTrait<OpTrait::SameTypeOperands>() ||
                  (&inBlockOp)
                      ->hasTrait<OpTrait::SameOperandsAndResultType>())) {
        // 1. setOperandTensorType
        if ((&inBlockOp)->hasTrait<OpTrait::NOperands<2>::Impl>()) {
          unsigned numOperands = inBlockOp.getNumOperands();
          SmallVector<bool, 4> isOperandValidInSameForBlock(numOperands);
          for (size_t i = 0; i < numOperands; ++i) {
            isOperandValidInSameForBlock[i] =
                isOperandOperationInSameForBlock(&inBlockOp, i) ||
                (inBlockOp.getOperand(i).getType() ==
                 inBlockOp.getOperand(i ^ 1).getType());
            if (!isOperandValidInSameForBlock[i]) {
              assert(isa<arith::ConstantOp>(
                         inBlockOp.getOperand(i).getDefiningOp()) &&
                     "Unable to extract the non-constant operand.");
              auto extractSliceOp = getExtractedOperand(context, builder, loc,
                                                        &inBlockOp, i, iterNum);
              extractSliceOp->moveBefore(ifOp);
              inBlockOp.setOperand(i, extractSliceOp->getResult(0));
            }
          }
        }
      }
    }
  }

  triton::xpu::ExtractSliceOp
  getExtractedOperand(MLIRContext *context, OpBuilder &builder, Location &loc,
                      mlir::Operation *op, unsigned operandIndex,
                      int64_t iterNum) const {
    auto resTy = op->getOperand(operandIndex).getType();
    RankedTensorType tensorTy;
    if (isa<RankedTensorType>(resTy)) {
      tensorTy = cast<RankedTensorType>(resTy);
    }
    auto shape = tensorTy.getShape().vec();
    shape[shape.size() - 1] = ceil<int64_t>(shape.back(), iterNum);
    auto clusterEncoding =
        cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
    auto newClusterEncoding = createEncoding(context, clusterEncoding, iterNum);

    RankedTensorType controledTensorTy = RankedTensorType::get(
        shape, tensorTy.getElementType(), newClusterEncoding);
    triton::xpu::ExtractSliceOp extractSliceOp =
        builder.create<triton::xpu::ExtractSliceOp>(
            loc, controledTensorTy, op->getOperand(operandIndex));
    return extractSliceOp;
  }

  // Determine whether the operand has been hoisted
  bool isOperandOperationInSameForBlock(mlir::Operation *op,
                                        unsigned operandIndex) {
    auto *parentOp = op->getParentOp();
    while (parentOp && !llvm::isa<mlir::scf::ForOp>(parentOp)) {
      parentOp = parentOp->getParentOp();
    }
    if (!parentOp)
      return false;

    auto forOp = llvm::cast<mlir::scf::ForOp>(parentOp);
    mlir::Value operand = op->getOperand(operandIndex);
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
      mlir::Block *block = forOp.getBody()->front().getBlock();
      return blockArg.getOwner() == block;
    } else {
      mlir::Operation *definingOp = operand.getDefiningOp();
      if (definingOp) {
        return definingOp->getBlock()->getParentOp() == forOp.getOperation();
      }
    }
    return false;
  }

  void insertIndex(Operation *op, Value idxVar) {
    OpBuilder builder(op);
    auto operandSegmentSizesAttr =
        op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    SmallVector<int, 4> operandSegmentSizes(
        operandSegmentSizesAttr.asArrayRef());
    // LoadOp: 0: ptr, 1: mask, 2: other, 3: index
    // StoreOp: 0: ptr, 1: value, 2: mask, 3: index
    // MakeRangeOp: 0: loopIndex, 1: unrollIndex
    // InterleaveOp: 0: loopIndex, 1: unrollIndex
    ++operandSegmentSizes[operandSegmentSizes.size() - 1];
    op->setAttr("operandSegmentSizes",
                builder.getDenseI32ArrayAttr(operandSegmentSizes));
    op->insertOperands(op->getNumOperands(), {idxVar});
  }

  void getOpChainBwdPostReduce(llvm::SetVector<Operation *> &opChain,
                               Operation *op) {
    if (!op) {
      return;
    }
    opChain.insert(op);

    int noDefCnt = 0;
    for (auto operand : op->getOperands()) {
      if (!operand.getDefiningOp()) {
        noDefCnt++;
      }
    }

    if (isa<arith::ConstantOp, triton::xpu::VConstOp, triton::xpu::StoreOp,
            triton::xpu::ReduceOp>(op) ||
        noDefCnt == op->getNumOperands()) {
      return;
    }

    for (auto operand : op->getOperands()) {
      getOpChainBwdPostReduce(opChain, operand.getDefiningOp());
    }
  }

  void getOuterChain(llvm::SetVector<Operation *> &allOpTree,
                     llvm::SetVector<Operation *> &outerChain,
                     bool postReduce = false) {
    for (auto op : allOpTree) {
      if (auto expandDimOp = dyn_cast<triton::ExpandDimsOp>(op)) {
        auto src = expandDimOp.getSrc();
        auto result = expandDimOp.getResult();
        if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
            if (expandDimOp.getAxis() == 1) {
              if (postReduce) {
                getOpChainBwdPostReduce(outerChain, expandDimOp);
              } else {
                getOpChainBwd(outerChain, expandDimOp);
              }
              outerChain.remove(expandDimOp);
            }
          }
        }
      }
      if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
        auto src = broadcastOp.getSrc();
        auto result = broadcastOp.getResult();
        if (auto srcTy = dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = dyn_cast<RankedTensorType>(result.getType())) {
            int64_t srcElemNum = 1;
            if (auto vecTy =
                    dyn_cast<VectorType>(getElementTypeOrSelf(srcTy))) {
              srcElemNum = vecTy.getNumElements();
            }
            int64_t resElemNum = 1;
            if (auto vecTy =
                    dyn_cast<VectorType>(getElementTypeOrSelf(resTy))) {
              resElemNum = vecTy.getNumElements();
            }
            auto srcShape = srcTy.getShape();
            auto resShape = resTy.getShape();
            int64_t srcInnerNum = srcElemNum * srcShape.back();
            int64_t resInnerNum = resElemNum * resShape.back();
            if (srcInnerNum != resInnerNum) { // unequal dim 1 shape means in
                                              // the inner axis op chain
              assert(srcShape.front() == resShape.front() &&
                     "Invalid BroadCast");
              if (postReduce) {
                getOpChainBwdPostReduce(outerChain, broadcastOp);
              } else {
                getOpChainBwd(outerChain, broadcastOp);
              }
              outerChain.remove(broadcastOp);
            }
          }
        }
      }
    }
  }

  void
  getOuterChains(const SmallVector<llvm::SetVector<Operation *>> &allOpTrees,
                 SmallVector<llvm::SetVector<Operation *>> &outerChains,
                 bool postReduce = false) {
    for (auto allOpTree : allOpTrees) {
      SetVector<Operation *> outerChain;
      getOuterChain(allOpTree, outerChain, postReduce);
      outerChains.emplace_back(outerChain);
    }
  }

  void getDAG(Operation *op, SetVector<Operation *> &visitedOps,
              SmallVector<SetVector<Operation *>> &unrollOpTrees,
              SetVector<Operation *> &excludeChainOps, bool isTop2Bottom = true,
              bool needBefore = false) {
    SetVector<Operation *> opTree;
    getUnrollTree(op, opTree, visitedOps, excludeChainOps, op, isTop2Bottom,
                  needBefore);
    if (!opTree.empty()) {
      SetVector<Operation *> sortedOpTree = sortOpTree(opTree);
      unrollOpTrees.emplace_back(sortedOpTree);
    }
  }

  void getPostReduceDAG(Operation *op, SetVector<Operation *> &visitedOps,
                        SmallVector<SetVector<Operation *>> &unrollOpTrees,
                        SetVector<Operation *> &excludeChainOps) {
    SetVector<Operation *> opTree;
    getPostReduceUnrollTree(op, opTree, visitedOps, excludeChainOps, op);
    if (!opTree.empty()) {
      SetVector<Operation *> sortedOpTree = sortOpTree(opTree);
      unrollOpTrees.emplace_back(sortedOpTree);
    }
  }

  void createFor(OpBuilder &builder, Location &loc, int64_t start,
                 int64_t iterNum, scf::ForOp &forOp, arith::IndexCastOp &idxVar,
                 ValueRange &iterArgs) {
    auto lower = builder.create<arith::ConstantIndexOp>(loc, start);
    auto upper = builder.create<arith::ConstantIndexOp>(loc, iterNum);
    auto step = builder.create<arith::ConstantIndexOp>(loc, 1);
    if (iterArgs.empty()) {
      forOp = builder.create<scf::ForOp>(loc, lower, upper, step);
    } else {
      forOp = builder.create<scf::ForOp>(loc, lower, upper, step, iterArgs);
    }
    builder.setInsertionPointToStart(forOp.getBody());

    idxVar = builder.create<arith::IndexCastOp>(loc, builder.getI32Type(),
                                                forOp.getInductionVar());
  }

  void createLoopBody(MLIRContext *context, OpBuilder &builder, Location &loc,
                      int64_t iterNum, SetVector<Operation *> &unrollOpTree,
                      SetVector<Operation *> &outerChain,
                      arith::IndexCastOp &idxVar, IRMapping &mapping) {
    for (auto op : unrollOpTree) {
      bool isOuter = inOpChain(outerChain, op);
      auto newOp = builder.clone(*op, mapping);
      setTensorType(context, newOp, iterNum, isOuter);
      TypeSwitch<Operation *>(newOp)
          .Case<triton::xpu::LoadOp>([&](auto loadOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(loadOp.getPtr().getType())) {
              auto shape = tensorTy.getShape();
              bool isOuter = (shape.size() == 2 && shape.back() == 1);
              if (!isOuter && !loadOp.getSVOpt() && !loadOp.getIsDiscrete()) {
                insertIndex(newOp, idxVar);
              }
            }
          })
          .Case<triton::xpu::StoreOp>([&](auto storeOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(storeOp.getPtr().getType())) {
              auto shape = tensorTy.getShape();
              bool isOuter = (shape.size() == 2 && shape.back() == 1);
              if (!isOuter) {
                insertIndex(newOp, idxVar);
              }
            }
          })
          .Case<triton::xpu::MakeRangeOp>([&](auto makeRangeOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(op->getResults()[0].getType())) {
              insertIndex(newOp, idxVar);
            }
          })
          .Case<triton::xpu::InterleaveOp>([&](auto interleaveOp) {
            if (auto tensorTy =
                    dyn_cast<RankedTensorType>(op->getResults()[0].getType())) {
              insertIndex(newOp, idxVar);
            }
          })
          .Case<XPUPrintOp>([&](auto xpuprintOp) {
            Value idxVar64 = builder.create<arith::ExtSIOp>(
                loc, builder.getI64Type(), idxVar);
            Value ucBound = builder.create<arith::ConstantIntOp>(
                loc, iterNum, builder.getI64Type());
            auto NewOp = builder.create<XPUPrintOp>(
                xpuprintOp.getLoc(), xpuprintOp.getPidx(), xpuprintOp.getPidy(),
                xpuprintOp.getPidz(), xpuprintOp.getOuterIndex(),
                xpuprintOp.getInnerIndex(), idxVar64,
                xpuprintOp.getInnerBound(), ucBound, xpuprintOp.getPrefixAttr(),
                xpuprintOp.getHexAttr(), xpuprintOp.getArgs());
            newOp->erase();
          })
          .Case<triton::AddPtrOp>([&](auto addPtrOp) {
            auto ptr = addPtrOp.getPtr();
            auto offset = addPtrOp.getOffset();

            if (mlir::dyn_cast<mlir::BlockArgument>(ptr)) {
              // For the time being,
              // it seems that no additional processing
              // is needed for this addPtrOp here
            } else {
              auto ptrTensorTy = dyn_cast<RankedTensorType>(ptr.getType());
              auto offsetTensorTy =
                  dyn_cast<RankedTensorType>(offset.getType());
              if (ptrTensorTy && offsetTensorTy &&
                  ptrTensorTy.getShape() != offsetTensorTy.getShape()) {
                auto extractOp = builder.create<triton::xpu::ExtractOp>(
                    loc, getElementTypeOrSelf(ptr),
                    builder.getI32IntegerAttr(0), ptr);
                auto splatTy = RankedTensorType::get(
                    offsetTensorTy.getShape(), getElementTypeOrSelf(ptr),
                    offsetTensorTy.getEncoding());
                auto splatOp =
                    builder.create<triton::SplatOp>(loc, splatTy, extractOp);
                addPtrOp.setOperand(0, splatOp);
                addPtrOp->moveAfter(splatOp);
              }
            }
          })
          .Case<arith::ConstantOp>([&](auto constantOp) {
            auto value = constantOp.getValue();
            if (auto attr = dyn_cast<DenseElementsAttr>(value)) {
              value = DenseElementsAttr::getFromRawBuffer(
                  cast<ShapedType>(constantOp.getType()), attr.getRawData());
            }
            constantOp.setValueAttr(value);
          })
          .Case<scf::IfOp>([&](auto ifOp) {
            // process ifOp recursively to handle nested ifOp
            auto processIfOp = [&](auto &self, scf::IfOp ifOp) -> void {
              auto &thenRegion = ifOp.getThenRegion();
              if (!thenRegion.empty()) {

                auto &thenBlock = thenRegion.front();
                for (auto &op : thenBlock) {
                  if (auto nestedIfOp = dyn_cast<scf::IfOp>(op)) {
                    self(self, nestedIfOp);
                  } else {
                    setTensorType(context, &op, iterNum, isOuter);
                  }
                }
                setHoistedOperand(context, builder, loc, thenBlock, ifOp,
                                  iterNum);
              }
              auto &elseRegion = ifOp.getElseRegion();
              if (!elseRegion.empty()) {
                auto &elseBlock = elseRegion.front();

                for (auto &op : elseBlock) {
                  if (auto nestedIfOp = dyn_cast<scf::IfOp>(op)) {
                    self(self, nestedIfOp);
                  } else {
                    setTensorType(context, &op, iterNum, isOuter);
                  }
                }
                setHoistedOperand(context, builder, loc, elseBlock, ifOp,
                                  iterNum);
              }
            };
            processIfOp(processIfOp, ifOp);
          })
          .Case<scf::ForOp>([&](auto forOp) {
            // step 1 : set iter arg type.
            unsigned numInitArgs =
                forOp.getNumOperands() - 3; // 减去初始值、上界和步长
            Block &entryBlock = forOp.getBodyRegion().front();
            if (numInitArgs > 0 && entryBlock.getNumArguments() > 1) {
              for (unsigned i = 0; i < numInitArgs; ++i) {
                Type initIterArgType = forOp.getOperand(3 + i).getType();
                Type regionIterArgType =
                    entryBlock.getArgument(i + 1).getType();
                if (initIterArgType != regionIterArgType) {
                  entryBlock.getArgument(i + 1).setType(initIterArgType);
                }
              }
            }

            // step 2 : set ops' type in loop body.
            Block *body = forOp.getBody();
            for (auto &op : body->getOperations()) {
              setTensorType(context, &op, iterNum, isOuter);
            }
          });
    }
  }

  void eraseDAG(SetVector<Operation *> &unrollOpTree) {
    SetVector<Operation *> eraseOpTree(unrollOpTree.rbegin(),
                                       unrollOpTree.rend());
    for (auto op : eraseOpTree) {
      SetVector<Operation *> users;
      for (auto user : op->getUsers()) {
        if (isa<triton::xpu::ReduceReturnOp>(user)) {
          users.insert(user);
        }
      }
      for (auto user : users) {
        user->erase();
      }
      if (op->use_empty()) {
        op->erase();
      }
    }
  }

  void moveAllocaAndGM2LM(scf::ForOp forOp,
                          SetVector<Operation *> &unrollOpTree) {
    ModuleOp m = getOperation();
    DenseMap<mlir::Operation *, unsigned> op2Line;
    getOpLine(m, op2Line);

    SmallVector<Operation *> gm2lmOps;
    SmallVector<Operation *> allocaOps;
    for (auto op : unrollOpTree) {
      if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
        auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMOp>(loadOp.getPtr());
        if (gm2lmOp) {
          gm2lmOps.emplace_back(gm2lmOp);
        }
        auto gm2lmmaskOp =
            findDefOpBwd<triton::xpu::GM2LMMaskOp>(loadOp.getPtr());
        if (gm2lmmaskOp) {
          gm2lmOps.emplace_back(gm2lmmaskOp);
        }
      }
      if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
        auto alloca = findDefOpBwd<triton::xpu::AllocaOp>(storeOp.getPtr());
        if (alloca) {
          allocaOps.emplace_back(alloca);
        }
      }
    }

    // move alloca when merge store
    for (auto allocaOp : allocaOps) {
      if (allocaOp->getBlock() == forOp->getBlock() &&
          forOp->isBeforeInBlock(allocaOp)) {
        allocaOp->moveBefore(forOp);
      }
    }

    for (auto gm2lmOp : gm2lmOps) {
      if (gm2lmOp->getBlock() != forOp->getBlock())
        continue;

      if (gm2lmOp->isBeforeInBlock(forOp))
        continue;

      for (auto operand : gm2lmOp->getOperands()) {
        auto op = operand.getDefiningOp();
        if (!op)
          continue;
        if (op2Line[op] > op2Line[forOp]) {
          op->moveBefore(forOp);
        }
      }
      gm2lmOp->moveBefore(forOp);
    }
  }

  void unrollControl(MLIRContext *context,
                     SmallVector<SetVector<Operation *>> &unrollOpTrees,
                     bool postReduce = false) {
    // Get outerChains
    SmallVector<SetVector<Operation *>> outerChains;
    getOuterChains(unrollOpTrees, outerChains, postReduce);
    for (int i = 0; i < unrollOpTrees.size(); ++i) {
      auto outerChain = outerChains[i];
      auto unrollOpTree = unrollOpTrees[i];
      // 1. Prepare for unroll control
      int64_t numCol = 1;
      int64_t numUnroll = 1;
      triton::xpu::StoreOp insertPt;
      SmallVector<triton::xpu::StoreOp> allStoreOps;
      for (auto op : unrollOpTree) {
        // 1.1 Get insertPt and tensor num
        if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(op)) {
          auto type = storeOp.getValue().getType();
          numUnroll = numUnroll == 1 ? getNumUnroll(type)
                                     : std::min(numUnroll, getNumCol(type));
          numCol =
              numCol == 1 ? getNumCol(type) : std::min(numCol, getNumCol(type));
          allStoreOps.emplace_back(storeOp);
          //[TODO] To deal with the case that storeOps are in more than one
          // block
          if (insertPt && insertPt->getBlock() != storeOp->getBlock()) {
            return;
          }
          if (!insertPt || storeOp->isBeforeInBlock(insertPt)) {
            insertPt = storeOp;
          }
        }
      }
      if (insertPt) {
        auto loc = insertPt.getLoc();
        int64_t iterNum = ceil<int64_t>(numCol, numUnroll);
        if (iterNum <= 1)
          return;
        LLVM_DEBUG(llvm::dbgs()
                   << "[Unroll Control] Hit Unroll Control Pointwise\n");
        // 2. Unroll control
        // 2.1 Create forOp
        OpBuilder builder(insertPt);
        scf::ForOp forOp;
        arith::IndexCastOp idxVar;
        ValueRange iterArgs = {};
        createFor(builder, loc, 0, iterNum, forOp, idxVar, iterArgs);
        // 2.2 Move Alloca & GM2LM Op before ForOp
        moveAllocaAndGM2LM(forOp, unrollOpTree);
        // 2.3 Set Tensor Type
        IRMapping mapping;
        createLoopBody(context, builder, loc, iterNum, unrollOpTree, outerChain,
                       idxVar, mapping);

        // 3. Erase old DAG
        eraseDAG(unrollOpTree);
      }
    }
  }

  void unrollControlReduce(MLIRContext *context,
                           SetVector<Operation *> &unrollOpTree,
                           Operation *insertPt, ValueRange &iterArgs,
                           ValueRange &returnOperands) {
    SetVector<Operation *> outerChain;
    getOuterChain(unrollOpTree, outerChain);
    if (auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(insertPt)) {
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      int64_t iterNum = ceil<int64_t>(numCol, numUnroll);
      if (iterNum <= 1)
        return;
      OpBuilder builder(reduceOp);
      auto loc = reduceOp.getLoc();
      // 1. Prepare for unroll control
      // Insert ExtractSliceOp for TensorType
      SmallVector<Value> newIterArgs(iterArgs.size());
      for (int i = 0; i < iterArgs.size(); ++i) {
        auto iterArgDefOp = iterArgs[i].getDefiningOp();
        bool isOuter = inOpChain(outerChain, iterArgDefOp);
        auto extractSliceOp = builder.create<triton::xpu::ExtractSliceOp>(
            loc, iterArgs[i].getType(), iterArgs[i]);
        setTensorType(context, extractSliceOp, iterNum, isOuter);
        auto inUnrollOpTree = [&](OpOperand &operand) {
          return unrollOpTree.count(operand.getOwner());
        };
        iterArgs[i].replaceUsesWithIf(extractSliceOp.getResult(),
                                      inUnrollOpTree);
        newIterArgs[i] = extractSliceOp.getResult();
      }
      // 2. Unroll control
      // 2.1 Create forOp
      scf::ForOp forOp;
      arith::IndexCastOp idxVar;
      ValueRange newIterArgsRange(newIterArgs);
      createFor(builder, loc, 1, iterNum, forOp, idxVar, newIterArgsRange);
      // 2.2 Set Tensor Type
      IRMapping mapping;
      createLoopBody(context, builder, loc, iterNum, unrollOpTree, outerChain,
                     idxVar, mapping);
      bool isOuterReduce = inOpChain(outerChain, reduceOp);
      setTensorType(context, reduceOp, iterNum, isOuterReduce, false);
      // 2.3 Modify users and defs
      // replace initArgs with iterArgs
      auto inForOp = [&](OpOperand &operand) {
        return forOp == operand.getOwner()->getBlock()->getParentOp();
      };
      auto forBody = forOp.getBody();
      auto forArgs = forBody->getArguments();
      for (int i = 0; i < forOp.getInitArgs().size(); ++i) {
        forOp.getInitArgs()[i].replaceUsesWithIf(forArgs[i + 1], inForOp);
      }
      SmallVector<Value> mapRes;
      for (int i = 0; i < returnOperands.size(); ++i) {
        mapRes.emplace_back(mapping.lookup(returnOperands[i]));
      }
      builder.create<scf::YieldOp>(loc, mapRes);
      auto isReduceOp = [&](OpOperand &operand) {
        return reduceOp == operand.getOwner();
      };
      for (int i = 0; i < forOp.getResults().size(); ++i) {
        reduceOp.getOperands()[i].replaceUsesWithIf(forOp.getResults()[i],
                                                    isReduceOp);
      }
      // 3. Erase old DAG
      eraseDAG(unrollOpTree);
    }
  }

  void getExcludeChainOps(ModuleOp &m,
                          SetVector<Operation *> &excludeChainOps) {
    m.walk([&](Operation *op) {
      TypeSwitch<const Operation *>(op)
          .Case<XPU_MEMORY_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<XPU_MEMORY_MASK_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getMask()) {
              getOpChainBwd(excludeChainOps,
                            memoryOp.getMask().getDefiningOp());
            }
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<triton::xpu::LoadOp, triton::xpu::StoreOp>([&](auto acessOp) {
            if (acessOp.getMask()) {
              getOpChainBwd(excludeChainOps, acessOp.getMask().getDefiningOp());
            }
          });
    });
  }

  void
  getExcludeChainOpsforUnrollControl(ModuleOp &m,
                                     SetVector<Operation *> &excludeChainOps) {
    m.walk([&](Operation *op) {
      TypeSwitch<const Operation *>(op)
          .Case<XPU_MEMORY_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<XPU_MEMORY_MASK_OP>([&](auto memoryOp) {
            getOpChainBwd(excludeChainOps, memoryOp.getPtr().getDefiningOp());
            if (memoryOp.getMask()) {
              getOpChainBwd(excludeChainOps,
                            memoryOp.getMask().getDefiningOp());
            }
            if (memoryOp.getLen()) {
              getOpChainBwd(excludeChainOps, memoryOp.getLen().getDefiningOp());
            }
          })
          .Case<triton::xpu::StoreOp>([&](auto storeOp) {
            if (storeOp.getMask()) {
              getOpChainBwd(excludeChainOps, storeOp.getMask().getDefiningOp());
            }
          })
          .Case<triton::xpu::LoadOp>([&](auto loadOp) {
            if (loadOp.getMask()) {
              auto op = loadOp.getMask().getDefiningOp();
              auto userNum =
                  std::distance(op->getUsers().begin(), op->getUsers().end());
              decltype(userNum) loadNum = 0;
              for (auto user : op->getUsers()) {
                if (isa<triton::xpu::LoadOp>(user)) {
                  loadNum++;
                }
              }
              if (userNum == loadNum) {
                getOpChainBwd(excludeChainOps,
                              loadOp.getMask().getDefiningOp());
              }
            }
          });
    });
  }

  void findDiscretePtrChain(SetVector<Operation *> &unrollOpTree,
                            SetVector<Operation *> &newUnrollOpTree) {
    for (auto op : unrollOpTree) {
      if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
        bool isDiscrete = loadOp.getIsDiscrete();
        if (isDiscrete) {
          OpBuilder builder(loadOp);
          auto loc = loadOp.getLoc();
          auto resType = loadOp.getResult().getType();
          int64_t numCol = getNumCol(resType);
          int64_t numUnroll = getNumUnroll(resType);
          if (numCol > numUnroll && numCol % numUnroll == 0) {
            auto lmPtr = loadOp.getPtr();
            if (auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMOp>(lmPtr)) {
              auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
              auto offset = gmPtrOp.getOffset();
              auto newLmPtr = builder.create<triton::AddPtrOp>(
                  loc, lmPtr.getType(), lmPtr, offset);
              SetVector<Operation *> ptrVisitedOps;
              SetVector<Operation *> ptrExcludeChainOps;
              getUnrollTree(newLmPtr, newUnrollOpTree, ptrVisitedOps,
                            ptrExcludeChainOps, newLmPtr, false);
              if (!newUnrollOpTree.empty()) {
                newUnrollOpTree = sortOpTree(newUnrollOpTree);
              }
              gm2lmOp->setAttr("offsetState",
                               builder.getSI32IntegerAttr(static_cast<int32_t>(
                                   OffsetState::Continuous)));
              loadOp.setOperand(0, newLmPtr);

            } else if (auto gm2lmOp =
                           findDefOpBwd<triton::xpu::GM2LMMaskOp>(lmPtr)) {
              auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
              auto offset = gmPtrOp.getOffset();
              auto newLmPtr = builder.create<triton::AddPtrOp>(
                  loc, lmPtr.getType(), lmPtr, offset);
              SetVector<Operation *> ptrVisitedOps;
              SetVector<Operation *> ptrExcludeChainOps;
              getUnrollTree(newLmPtr, newUnrollOpTree, ptrVisitedOps,
                            ptrExcludeChainOps, newLmPtr, false);
              if (!newUnrollOpTree.empty()) {
                newUnrollOpTree = sortOpTree(newUnrollOpTree);
              }
              gm2lmOp->setAttr("offsetState",
                               builder.getSI32IntegerAttr(static_cast<int32_t>(
                                   OffsetState::Continuous)));
              loadOp.setOperand(0, newLmPtr);
            }
          }
        }
      }
    }
  }

  void
  findDiscretePtrChains(SmallVector<SetVector<Operation *>> &unrollOpTrees,
                        SmallVector<SetVector<Operation *>> &newUnrollOpTrees) {
    for (auto [i, unrollOpTree] : llvm::enumerate(unrollOpTrees)) {
      findDiscretePtrChain(unrollOpTree, newUnrollOpTrees[i]);
    }
  }

  void createDiscreteOffset(ModuleOp &m) {
    m.walk([&](triton::xpu::LoadOp loadOp) {
      bool isDiscrete = loadOp.getIsDiscrete();
      if (isDiscrete) {
        OpBuilder builder(loadOp);
        auto loc = builder.getUnknownLoc();
        auto lmPtr = loadOp.getPtr();
        auto lmAddPtr =
            cast<triton::AddPtrOp>(findDefOpBwd<triton::AddPtrOp>(lmPtr));
        auto lmOffset = lmAddPtr.getOffset();
        if (auto gm2lmOp = findDefOpBwd<triton::xpu::GM2LMOp>(lmPtr)) {
          auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
          auto gmOffset = gmPtrOp.getOffset();
          auto extractOp = builder.create<triton::xpu::ExtractOp>(
              loc, getElementTypeOrSelf(gmOffset), builder.getI32IntegerAttr(0),
              gmOffset);
          auto splatOp = builder.create<triton::SplatOp>(
              loc, lmOffset.getType(), extractOp);
          auto offset = builder.create<arith::SubIOp>(loc, lmOffset.getType(),
                                                      lmOffset, splatOp);
          lmAddPtr.setOperand(1, offset);
          lmAddPtr->moveAfter(offset);
          if (gm2lmOp->getOperand(0) == lmAddPtr.getResult())
            gm2lmOp->moveAfter(lmAddPtr);
        } else if (auto gm2lmOp =
                       findDefOpBwd<triton::xpu::GM2LMMaskOp>(lmPtr)) {
          auto gmPtrOp = findDefOpBwd<triton::AddPtrOp>(gm2lmOp.getPtr());
          auto gmOffset = gmPtrOp.getOffset();
          auto extractOp = builder.create<triton::xpu::ExtractOp>(
              loc, getElementTypeOrSelf(gmOffset), builder.getI32IntegerAttr(0),
              gmOffset);
          auto splatOp = builder.create<triton::SplatOp>(
              loc, lmOffset.getType(), extractOp);
          auto offset = builder.create<arith::SubIOp>(loc, lmOffset.getType(),
                                                      lmOffset, splatOp);
          lmAddPtr.setOperand(1, offset);
          lmAddPtr->moveAfter(offset);
          if (gm2lmOp->getOperand(0) == lmAddPtr.getResult())
            gm2lmOp->moveAfter(lmAddPtr);
        }
      }
    });
  }

  void pointwiseUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Data-flow Analysis: get load -> store DAG
    //    (op in ptrChain/lenChain/maskChain will not walk from top to down)
    // 1.1 Get excludeChainOps
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    // 1.2 Get load -> store DAG
    SetVector<Operation *> visitedOps;
    SmallVector<SetVector<Operation *>> unrollOpTrees;
    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto valType = storeOp.getValue().getType();
      int64_t numCol = getNumCol(valType);
      int64_t numUnroll = getNumUnroll(valType);
      if (numCol > numUnroll && numCol % numUnroll == 0) {
        getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps);
      }
      for (auto visitedOp : visitedOps) {
        if (isa<arith::ConstantOp>(visitedOp)) {
          visitedOps.remove(visitedOp);
        }
      }
    });
    if (unrollOpTrees.size() == 0)
      return;

    // 1.3 Find ptr chain of discrete for moving to loop body
    SmallVector<SetVector<Operation *>> newUnrollOpTrees(unrollOpTrees);
    findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees);

    // 2. Deal with unroll opTrees
    unrollControl(context, newUnrollOpTrees);

    // 3. Calculate discrete offset in the runtime
    createDiscreteOffset(m);
  }

  void createLoadStore(scf::ForOp &forOp, scf::YieldOp &yieldOp, Value &yield,
                       int i, Block &block,
                       SmallVector<Operation *> &storeOps) {
    OpBuilder builder(yieldOp);
    auto loc = yieldOp->getLoc();
    Type yieldType = yield.getType();
    Type yieldElemType = getElementTypeOrSelf(yieldType);
    int64_t vecSize = getNumInVector(yieldElemType);
    Type ptrTy = createPointerType(yieldType, vecSize);
    int64_t tensorSize = getTensorSize(yieldType);
    if (!forOp.getResults()[i].use_empty()) {
      // Create Alloca Store for Init Args
      auto initForArg = forOp.getInitArgs()[i];
      auto newAllocaOp = builder.create<triton::xpu::AllocaOp>(
          loc, ptrTy, tensorSize * vecSize);
      auto initStoreOp = builder.create<triton::xpu::StoreOp>(
          loc, newAllocaOp, initForArg, Value(), Value(), -1, false,
          Dtype::UNKNOWN, MemorySyncMode::SYNC);
      newAllocaOp->moveBefore(forOp);
      initStoreOp->moveBefore(forOp);
      // Create Load for Input
      auto inputLoadOp = builder.create<triton::xpu::LoadOp>(
          loc, yieldType, newAllocaOp, Value(), Value(), Value(), 1, -1, false,
          false, false, MemorySyncMode::SYNC);
      auto notUsedForYield = [&](OpOperand &operand) {
        return !isa<scf::YieldOp>(operand.getOwner());
      };
      auto forArg = forOp.getRegionIterArgs()[i];
      forArg.replaceUsesWithIf(inputLoadOp, notUsedForYield);
      inputLoadOp->moveBefore(&block.front());
      // Create Store for Output
      auto outputStoreOp = builder.create<triton::xpu::StoreOp>(
          loc, newAllocaOp, yield, Value(), Value(), -1, false, Dtype::UNKNOWN,
          MemorySyncMode::SYNC);
      outputStoreOp->moveBefore(yieldOp);
      storeOps.emplace_back(outputStoreOp);
      // Create Load for Reduce
      auto reduceLoadOp = builder.create<triton::xpu::LoadOp>(
          loc, yieldType, newAllocaOp, Value(), Value(), Value(), 1, -1, false,
          false, false, MemorySyncMode::SYNC);

      // Replace For Result with Load
      auto notReduceLoadOp = [&](OpOperand &operand) {
        return reduceLoadOp != operand.getOwner();
      };
      forOp.getResults()[i].replaceUsesWithIf(reduceLoadOp, notReduceLoadOp);

      // Move Load closed to For user
      reduceLoadOp->moveAfter(forOp);
      Operation *insertPt = nullptr;
      for (auto user : forOp.getResults()[i].getUsers()) {
        if (!insertPt) {
          insertPt = user;
        } else {
          if (insertPt->getBlock() == user->getBlock()) {
            if (user->isBeforeInBlock(insertPt)) {
              insertPt = user;
            }
          }
        }
      }
      if (insertPt) {
        reduceLoadOp->moveBefore(insertPt);
      }

      // Discard Yield by setting initForArg to operand
      yieldOp->setOperand(i, initForArg);
    }
  }

  void getUnrollInfoReduce(triton::xpu::ReduceOp &reduceOp, int64_t &numCol,
                           int64_t &numUnroll) {
    auto types = reduceOp.getOperandTypes();
    assert(types.size() > 1);
    for (int i = 0; i < types.size() - 1; ++i) {
      if (i == 0) {
        numCol = getNumCol(types[i]);
        numUnroll = getNumUnroll(types[i]);
      } else {
        assert(numCol == getNumCol(types[i]));
        assert(numUnroll == getNumUnroll(types[i]));
      }
    }
  }

  void forUnrollControl(ModuleOp &m, MLIRContext *context) {
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOpsforUnrollControl(m, excludeChainOps);
    SetVector<Operation *> vistedForOps;
    // 1. Create Store Load
    m.walk([&](triton::xpu::ReduceOp reduceOp) {
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      if (numCol > numUnroll && numCol % numUnroll == 0) {
        llvm::SetVector<Operation *> reduceOpDefsBwd;
        getOpChainBwd(reduceOpDefsBwd, reduceOp);
        for (auto operand : reduceOpDefsBwd) {
          if (auto forOp = dyn_cast<scf::ForOp>(operand)) {
            if (!vistedForOps.count(forOp)) {
              LLVM_DEBUG(llvm::dbgs()
                         << "[Unroll Control] Hit Unroll Control For\n");
              vistedForOps.insert(forOp);
              auto &forBlock = forOp.getRegion().front();
              bool hasIf = false;
              SetVector<Operation *> visitedOps;
              for (auto &inForBlockOp : forBlock) {
                if (auto ifOp = dyn_cast<scf::IfOp>(inForBlockOp)) {
                  SmallVector<Operation *> storeOps;
                  auto &ifBlock = ifOp.getThenRegion().front();
                  auto yieldOp = cast<scf::YieldOp>(ifBlock.getTerminator());
                  for (auto [i, yield] :
                       llvm::enumerate(yieldOp.getOperands())) {
                    createLoadStore(forOp, yieldOp, yield, i, ifBlock,
                                    storeOps);
                  }
                  // Unroll control
                  for (auto storeOp : storeOps) {
                    if (visitedOps.count(storeOp))
                      continue;
                    SmallVector<SetVector<Operation *>> unrollOpTrees;
                    getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps,
                           true, true);
                    // Find ptr chain of discrete for moving to loop body
                    SmallVector<SetVector<Operation *>> newUnrollOpTrees(
                        unrollOpTrees);
                    findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees);
                    unrollControl(context, newUnrollOpTrees);
                  }
                  hasIf = true;
                }
              }
              if (!hasIf) {
                SmallVector<Operation *> storeOps;
                auto yieldOp = cast<scf::YieldOp>(forBlock.getTerminator());
                for (auto [i, yield] : llvm::enumerate(yieldOp.getOperands())) {
                  createLoadStore(forOp, yieldOp, yield, i, forBlock, storeOps);
                }
                // Unroll control
                for (auto storeOp : storeOps) {
                  if (visitedOps.count(storeOp))
                    continue;
                  SmallVector<SetVector<Operation *>> unrollOpTrees;
                  getDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps,
                         true, true);
                  // Find ptr chain of discrete for moving to loop body
                  SmallVector<SetVector<Operation *>> newUnrollOpTrees(
                      unrollOpTrees);
                  findDiscretePtrChains(unrollOpTrees, newUnrollOpTrees);
                  unrollControl(context, newUnrollOpTrees);
                }
              }
            }
          }
        }
      }
    });
  }

  void getInlineInfo(SetVector<Operation *> &inlineOps, Operation *startOp,
                     ValueRange &returnOperands) {
    Operation *op = startOp;
    while (!isa<triton::xpu::ReduceReturnOp>(op)) {
      inlineOps.insert(op);
      op = op->getNextNode();
    }
    returnOperands = op->getOperands();
  }

  void createReduceWithinCore(ModuleOp &m, MLIRContext *context) {
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    m.walk([&](triton::xpu::ReduceOp reduceOp) {
      ReduceOpHelper helper(reduceOp);
      OpBuilder builder(reduceOp);
      auto loc = reduceOp->getLoc();
      SetVector<Operation *> visitedOps;
      auto reduceOperandNum = reduceOp.getNumOperands() - 1;
      SmallVector<SetVector<Operation *>> copyOpTrees;
      SetVector<Operation *> unrollOpTree;
      int64_t numCol = 1, numUnroll = 1;
      getUnrollInfoReduce(reduceOp, numCol, numUnroll);
      if (numCol > numUnroll && numCol % numUnroll == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[Unroll Control] Hit Unroll Control Reduction\n");
        for (int i = 0; i < reduceOperandNum; ++i) {
          if (auto reduceDefOp = reduceOp.getOperands()[i].getDefiningOp()) {
            getDAG(reduceDefOp, visitedOps, copyOpTrees, excludeChainOps,
                   false);
          }
        }
        // 1. Copy Defined Op Chain of Reduce Operand for InitArgs
        IRMapping mapping;
        for (auto &copyOpTree : copyOpTrees) {
          for (auto &copyOp : copyOpTree) {
            auto newOp = builder.clone(*copyOp, mapping);
            unrollOpTree.insert(newOp);
          }
        }
        // 2. Inline Combine Op of Reduce
        // Clone Region
        IRRewriter rewriter(builder);
        Block *currentBlock = rewriter.getBlock();
        Region &parent = *currentBlock->getParent();
        rewriter.cloneRegionBefore(reduceOp.getCombineOp(), &parent.front());
        auto &newReduce = parent.front();
        // Set Type for Cloned Ops
        auto tensorTy = reduceOp.getInputTypes()[0];
        auto shape = tensorTy.getShape();
        for (auto &op : newReduce) {
          if (isa<arith::CmpFOp>(op) || isa<arith::CmpIOp>(op)) {
            auto tensorTy0 = op.getOperand(0).getType();
            auto tensorTy1 = op.getOperand(1).getType();
            int operandIndexNeedModify;
            mlir::Type operandNeedReserved;
            if (tensorTy0 != tensorTy1) {
              if ((mlir::isa<mlir::FloatType>(tensorTy0) ||
                   mlir::isa<mlir::IntegerType>(tensorTy0)) &&
                  mlir::isa<mlir::TensorType>(tensorTy1)) {
                operandIndexNeedModify = 0;
                operandNeedReserved = tensorTy1;
              } else if ((mlir::isa<mlir::FloatType>(tensorTy1) ||
                          mlir::isa<mlir::IntegerType>(tensorTy1)) &&
                         mlir::isa<mlir::TensorType>(tensorTy0)) {
                operandIndexNeedModify = 1;
                operandNeedReserved = tensorTy0;
              }
              assert(
                  isa<arith::ConstantOp>(
                      op.getOperand(operandIndexNeedModify).getDefiningOp()) &&
                  "Unable to extract the non-constant operand.");
              auto splatOp = builder.create<triton::SplatOp>(
                  loc, operandNeedReserved,
                  op.getOperand(operandIndexNeedModify));
              splatOp->moveBefore(&op);
              op.setOperand(operandIndexNeedModify, splatOp.getResult());
            }
          } else if (auto selOp = dyn_cast<arith::SelectOp>(op)) {
            auto tensorTy1 = selOp.getODSOperands(1)[0].getType();
            auto tensorTy2 = selOp.getODSOperands(2)[0].getType();
            int operandIndexNeedModify;
            mlir::Type operandNeedReserved;
            if (tensorTy1 != tensorTy2) {
              if ((mlir::isa<mlir::FloatType>(tensorTy1) ||
                   mlir::isa<mlir::IntegerType>(tensorTy1)) &&
                  mlir::isa<mlir::TensorType>(tensorTy2)) {
                operandIndexNeedModify = 1;
                operandNeedReserved = tensorTy2;
              } else if ((mlir::isa<mlir::FloatType>(tensorTy2) ||
                          mlir::isa<mlir::IntegerType>(tensorTy2)) &&
                         mlir::isa<mlir::TensorType>(tensorTy1)) {
                operandIndexNeedModify = 2;
                operandNeedReserved = tensorTy1;
              }
              assert(isa<arith::ConstantOp>(
                         selOp.getOperand(operandIndexNeedModify)
                             .getDefiningOp()) &&
                     "Unable to extract the non-constant operand.");

              auto splatOp = builder.create<triton::SplatOp>(
                  loc, operandNeedReserved,
                  selOp.getOperand(operandIndexNeedModify));
              splatOp->moveBefore(&op);
              selOp.setOperand(operandIndexNeedModify, splatOp.getResult());
            }
          }
          for (auto [i, resTy] : llvm::enumerate(op.getResultTypes())) {
            auto inlineTensorTy =
                RankedTensorType::get(shape, resTy, tensorTy.getEncoding());
            op.getResult(i).setType(inlineTensorTy);
          }
        }
        // Inline Ops
        llvm::SmallVector<Value> combineArgs(2 * reduceOperandNum);
        for (unsigned i = 0; i < reduceOperandNum; ++i) {
          combineArgs[i] = reduceOp.getOperands()[i];
          combineArgs[reduceOperandNum + i] =
              mapping.lookup(reduceOp.getOperands()[i]);
        }
        auto currOp = &*rewriter.getInsertionPoint();
        auto insertOp = currOp->getPrevNode();
        rewriter.inlineBlockBefore(&newReduce, currOp, combineArgs);
        ValueRange returnOperands;
        getInlineInfo(unrollOpTree, insertOp, returnOperands);

        auto isReduceOp = [&](OpOperand &operand) {
          return reduceOp == operand.getOwner();
        };
        llvm::SmallVector<Value> iterArgs(reduceOperandNum);
        for (auto [i, returnOperand] : llvm::enumerate(returnOperands)) {
          iterArgs[i] = reduceOp.getOperands()[i];
          reduceOp.getOperands()[i].replaceUsesWithIf(returnOperand,
                                                      isReduceOp);
        }
        // Find ptr chain of discrete for moving to loop body
        SetVector<Operation *> newUnrollOpTree(unrollOpTree);
        findDiscretePtrChain(unrollOpTree, newUnrollOpTree);
        // 3. Create Loop for ReduceWithinCore
        ValueRange iterArgsRange(iterArgs);
        unrollControlReduce(context, newUnrollOpTree, reduceOp, iterArgsRange,
                            returnOperands);
        // 4. For Vectorize: triton.addf->triton_xpu.vvaddf
        processOpVecTy(m);
      }
    });
  }

  bool isPostReduceStore(triton::xpu::StoreOp storeOp) {
    bool _isPostReduceStore = false;
    if (auto valTy = dyn_cast<RankedTensorType>(storeOp.getValue().getType())) {
      auto shape = valTy.getShape();
      if (shape.size() > 1 && shape.back() > 1) {
        _isPostReduceStore = true;
      }
    }
    return _isPostReduceStore;
  }

  void mergeSets(SmallVector<SetVector<Operation *>> &unrollOpTrees) {
    // Create Mapping of All Sets
    DenseMap<Operation *, SmallVector<SetVector<Operation *> *>> opToSets;
    for (auto &set : unrollOpTrees) {
      for (Operation *op : set) {
        opToSets[op].push_back(&set);
      }
    }
    // Merge unrollOpTrees that has common nodes
    DenseSet<SetVector<Operation *> *> processedSets;
    for (auto &currentSet : unrollOpTrees) {
      if (processedSets.count(&currentSet))
        continue;
      SetVector<Operation *> mergedSet = currentSet;
      bool hasMerged = true;
      while (hasMerged) {
        hasMerged = false;
        for (Operation *op : mergedSet) {
          auto &relatedSets = opToSets[op];
          for (auto *relatedSet : relatedSets) {
            if (relatedSet == &mergedSet || processedSets.count(relatedSet))
              continue;

            mergedSet.insert(relatedSet->begin(), relatedSet->end());
            relatedSet->clear();
            processedSets.insert(relatedSet);
            hasMerged = true;
          }
        }
      }
      if (mergedSet.size() > currentSet.size()) {
        mergedSet = sortOpTree(mergedSet);
        currentSet = mergedSet;
      }
      // Remove Empty Sets
      unrollOpTrees.erase(
          llvm::remove_if(
              unrollOpTrees,
              [](const SetVector<Operation *> &set) { return set.empty(); }),
          unrollOpTrees.end());
    }
  }

  void postReduceUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Data-flow Analysis: get post reduce -> store DAG
    //    (op in ptrChain/lenChain/maskChain will not walk from top to down)
    // 1.1 Get excludeChainOps
    SetVector<Operation *> excludeChainOps;
    getExcludeChainOps(m, excludeChainOps);
    // 1.2 Get load -> store DAG
    SmallVector<SetVector<Operation *>> unrollOpTrees;
    m.walk([&](triton::xpu::StoreOp storeOp) {
      SetVector<Operation *> visitedOps;
      auto valType = storeOp.getValue().getType();
      int64_t numCol = getNumCol(valType);
      int64_t numUnroll = getNumUnroll(valType);
      bool _isPostReduceStore = isPostReduceStore(storeOp);
      if (numCol > numUnroll && numCol % numUnroll == 0 && _isPostReduceStore) {
        getPostReduceDAG(storeOp, visitedOps, unrollOpTrees, excludeChainOps);
      }
    });
    if (unrollOpTrees.size() == 0)
      return;

    // 2. Merge unrollOpTrees that has common nodes
    mergeSets(unrollOpTrees);

    // 3. Deal with unroll opTrees
    LLVM_DEBUG(llvm::dbgs()
               << "[Unroll Control] Hit Unroll Control Post Reduction\n");
    unrollControl(context, unrollOpTrees, true);
  }

  void reductionUnrollControl(ModuleOp &m, MLIRContext *context) {
    // 1. Unroll Control for Reduce For
    forUnrollControl(m, context);
    // 2. Create For for ReduceWithinCore
    createReduceWithinCore(m, context);
    // 3. Deal with BroadCastOp/ReduceOp to StoreOp
    postReduceUnrollControl(m, context);
    // 4. Calculate discrete offset in the runtime
    createDiscreteOffset(m);
    // 5. Check Def-Use Shape Match
    checkDefUseShapeMatch(m, context);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    bool isScan = false;
    m.walk([&](triton::xpu::ScanOp scanOp) { isScan = true; });
    if (isScan) {
      return;
    }

    m.walk([&](triton::xpu::StoreOp storeOp) {
      auto dtype = storeOp.getDtype();
      auto valTy = storeOp.getValue().getType();
      auto ptrTy = storeOp.getPtr().getType();
      auto valElemTy = getElementTypeOrSelf(getElementTypeOrSelf(valTy));
      auto ptrElemTy = getElementTypeOrSelf(getElementTypeOrSelf(ptrTy));
      if (dtype == Dtype::FP32 && valElemTy.isInteger(32) &&
          cast<triton::PointerType>(ptrElemTy).getPointeeType().isInteger(8)) {
        this->unrollNum = 4;
      }
    });

    bool isReduce = false;
    m.walk([&](triton::xpu::ReduceOp redOp) {
      isReduce = true;
      // Set this->unrollNum=1 When coreDealMultiRows
      RankedTensorType operandType = redOp.getInputTypes()[0];
      auto shape = operandType.getShape();
      auto layout =
          cast<triton::xpu::ClusterLayoutAttr>(operandType.getEncoding());
      unsigned rowsPerCore = layout.getSizePerCore()[0];
      this->unrollNum =
          (shape.size() == 2 && rowsPerCore > 1) ? 1 : this->unrollNum;
    });

    if (isReduce) {
      reductionUnrollControl(m, context);
    } else {
      pointwiseUnrollControl(m, context);
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
