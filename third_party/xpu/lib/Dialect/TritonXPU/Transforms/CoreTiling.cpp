//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-core-tiling"

namespace mlir {

namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUCORETILING
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUCoreTilingPass
    : public impl::TritonXPUCoreTilingBase<TritonXPUCoreTilingPass> {

  using impl::TritonXPUCoreTilingBase<
      TritonXPUCoreTilingPass>::TritonXPUCoreTilingBase;

  TritonXPUCoreTilingPass() = default;
  TritonXPUCoreTilingPass(bool dumpFlag, unsigned bufferSize, unsigned coreNum,
                          unsigned groupsPerCluster) {
    this->dumpFlag = dumpFlag;
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
    this->groupsPerCluster = groupsPerCluster;
  }

  inline bool isAxisNone(triton::ReduceOp &reduceOp) {
    ReduceOpHelper helper(reduceOp);
    auto reduceOpTensorShape = helper.getSrcShape();
    for (auto src : reduceOp.getSrcs()) {
      if (auto defOp = src.getDefiningOp()) {
        if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(defOp)) {
          if (auto reshapeResTy =
                  dyn_cast<RankedTensorType>(reshapeOp.getResult().getType())) {
            if (reshapeResTy.getShape().size() == 1) {
              assert(reduceOp.getAxis() == 0);
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  inline bool ifInChain(SetVector<Operation *> chain, Operation *iOp) {
    return std::find(chain.begin(), chain.end(), iOp) != chain.end();
  }

  Attribute getOptimizedGEncoding(MLIRContext *context, RankedTensorType type,
                                  SetVector<Operation *> &innerChain,
                                  SetVector<Operation *> &outerChain,
                                  Operation *op, unsigned ngroup,
                                  unsigned groupsize) {
    Attribute newEncoding;
    auto shape = type.getShape();
    unsigned rank = shape.size();
    if (auto globalEncoding =
            dyn_cast<triton::xpu::ClusterLayoutAttr>(type.getEncoding())) {
      std::vector<unsigned> newSizePerCore;
      std::vector<unsigned> newCoresPerGroup;
      std::vector<unsigned> newGroupsPerCluster;
      std::vector<unsigned> order;

      if (rank == 1) {
        order = {0};
        if (!ifInChain(outerChain, op) || ifInChain(innerChain, op)) {
          newSizePerCore = {ceil<unsigned>(shape[0], groupsize)};
          newCoresPerGroup = {groupsize};
          newGroupsPerCluster = {1};
        } else {
          newSizePerCore = {ceil<unsigned>(shape[0], ngroup)};
          newCoresPerGroup = {1};
          newGroupsPerCluster = {ngroup};
        }
      } else if (rank == 2) {
        newCoresPerGroup = {1, groupsize};
        newGroupsPerCluster = {ngroup, 1};
        order = {0, 1};
        if (rowsPerCore > 1 && shape[0] != 1) {
          newSizePerCore = {rowsPerCore, ceil<unsigned>(shape[1], groupsize)};
        } else {
          newSizePerCore = {1, ceil<unsigned>(shape[1], groupsize)};
        }
        if (rowsPerCore > 1) {
          if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
            // BroadcastOp specifies the element size directly using the
            // sizePerThread value.
            order = {1, 0};
          }
        }
      } else {
        llvm_unreachable("Reduce Optimization With Rank > 2 Unsupported");
      }
      newEncoding = triton::xpu::ClusterLayoutAttr::get(
          context, newSizePerCore, newCoresPerGroup, newGroupsPerCluster,
          order);
    }

    return newEncoding;
  }

  bool getTensorColSize(ModuleOp &mod) {
    mod.walk([&](arith::CmpIOp cmpiOp) {
      auto lhs = cmpiOp.getLhs();
      auto rhs = cmpiOp.getRhs();

      if (auto lhsTensorTy = dyn_cast<RankedTensorType>(lhs.getType())) {
        auto lhsShape = lhsTensorTy.getShape();
        if (cmpiOp.getPredicate() == arith::CmpIPredicate::slt &&
            lhsShape.size() == 2 && lhsShape[0] == 1) { // inner Cmp Calculation
          if (auto rhsOp = rhs.getDefiningOp<arith::ConstantOp>()) {
            if (auto denseAttr =
                    mlir::dyn_cast<DenseElementsAttr>(rhsOp.getValue())) {
              auto values = denseAttr.getValues<mlir::APInt>();
              if (!values.empty()) {
                rawColSize = values[0].getZExtValue();
              }
            }
          }
        }
      }
    });

    return rawColSize ? true : false;
  }

  int roundupPow2(int n) {
    int ret = 1;
    while (n > ret) {
      ret *= 2;
    }
    return ret;
  };

  // Get ReduceOps' Shape To Check If Can Be Optimized
  bool canBeOptimized(ModuleOp &mod) {
    bool canBeOpt = false;
    int colSize = 0;

    auto checkGroupInfo = [&](RankedTensorType &tensorType) {
      if (auto globalEncoding = dyn_cast<triton::xpu::ClusterLayoutAttr>(
              tensorType.getEncoding())) {
        auto shape = tensorType.getShape();
        size_t _ngroup = product(globalEncoding.getGroupsPerCluster());
        size_t _groupsize = product(globalEncoding.getCoresPerGroup());
        size_t _sizePerCore = product(globalEncoding.getSizePerCore());
        size_t ncore = _ngroup * _groupsize;
        size_t m = shape.front();
        size_t n = shape.back();
        size_t newgroupsize = ceil<size_t>(n, static_cast<size_t>(bufferSize));
        newgroupsize = roundupPow2(newgroupsize);
        // min is for not using the whole 64 cores case
        size_t newngroup = std::min(ceil<size_t>(ncore, newgroupsize), m);
        newgroupsize = std::min(ceil<size_t>(ncore, newngroup), n);

        if (newngroup == 1 && newgroupsize == 64)
          canBeOpt = false;
      }
    };

    mod.walk([&](triton::ReduceOp reduceOp) {
      ReduceOpHelper helper(reduceOp);
      auto reduceOpTensorShape = helper.getSrcShape();

      if (reduceOpTensorShape.size() == 2) {
        if (reduceOp.getAxis() == 1) {
          colSize = std::max(colSize, static_cast<int>(reduceOpTensorShape[1]));
          canBeOpt = true;

          // rowsPerCore Upper = [128 / 16, 128 / 32, 128 / 64]
          unsigned rowsPerCoreUpper = this->bufferSize / reduceOpTensorShape[1];
          unsigned rowsPerCoreLower = 1;

          unsigned rowsPerCoreCal;
          for (rowsPerCoreCal = rowsPerCoreUpper;
               rowsPerCoreCal > rowsPerCoreLower; rowsPerCoreCal /= 2) {
            if (reduceOpTensorShape[0] % (rowsPerCoreCal * core_num) == 0)
              break;
          }

          rowsPerCore = std::min<unsigned>(rowsPerCoreUpper, rowsPerCoreCal);
          rowsPerCore = std::max<unsigned>(rowsPerCore, rowsPerCoreLower);
          if (!getTensorColSize(mod) || colSize < rawColSize)
            rowsPerCore = 1;

          auto tensorType =
              cast<RankedTensorType>(reduceOp.getOperandTypes()[0]);
          checkGroupInfo(tensorType);
        }
      } else if (isAxisNone(reduceOp)) {
        canBeOpt = true;
      } else if (canBeOpt) {
        llvm_unreachable("Not All Reduce Op can be Optimized");
      }
    });

    if (!getTensorColSize(mod) || colSize < rawColSize)
      rowsPerCore = 1;

    return canBeOpt;
  }

  mlir::Operation *findRootOp(mlir::Operation *op) {
    mlir::Operation *rootOp = op;
    while (rootOp->getParentOp()) {
      rootOp = rootOp->getParentOp();
      if (rootOp->getParentOp() && isa<triton::FuncOp>(rootOp->getParentOp())) {
        return rootOp;
      }
    }
    return op;
  }

  void getOpUserChainFwd(llvm::SetVector<Operation *> &opChain, Operation *op,
                         Operation *curentOp) {
    opChain.insert(op);

    if (isa<triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp,
            triton::xpu::BroadcastOp, triton::ExpandDimsOp, triton::SplatOp>(
            op) &&
        op != curentOp) {
      return;
    }

    for (auto userOp : op->getUsers()) {
      if (!opChain.contains(userOp)) {
        getOpUserChainFwd(opChain, userOp, curentOp);
      }
    }
  }

  void getOpDefChainBwd(llvm::SetVector<Operation *> &opChain, Operation *op,
                        Operation *curentOp) {
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

    if ((isa<mlir::arith::ConstantOp, triton::xpu::BroadcastOp,
             triton::ExpandDimsOp, triton::SplatOp, triton::ReduceOp>(op) ||
         noDefCnt == op->getNumOperands()) &&
        op != curentOp) {
      return;
    }

    for (auto operand : op->getOperands()) {
      getOpDefChainBwd(opChain, operand.getDefiningOp(), curentOp);
    }
  }

  void dumpOpChain(bool dumpFlag, Operation *op,
                   SetVector<Operation *> &innerChain,
                   SetVector<Operation *> &outerChain) {
    if (dumpFlag) {
      LLVM_DEBUG({
        llvm::dbgs() << "\n[Current Op]:\n";
        op->dump();
        llvm::dbgs() << "\n[innerChain]:\n";
        for (auto innerOp : innerChain) {
          innerOp->dump();
        }
        llvm::dbgs() << "\n[outerChain]:\n";
        for (auto outerOp : outerChain) {
          outerOp->dump();
        }
      });
    }
  }

  void getChain(ModuleOp &mod, SetVector<Operation *> &innerChain,
                SetVector<Operation *> &outerChain) {
    mod.walk([&](mlir::Operation *op) {
      if (auto expandDimOp = dyn_cast<triton::ExpandDimsOp>(op)) {
        auto src = expandDimOp.getSrc();
        auto result = expandDimOp.getResult();
        if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = mlir::dyn_cast<RankedTensorType>(result.getType())) {
            if (expandDimOp.getAxis() == 0) {
              getOpDefChainBwd(innerChain, expandDimOp, expandDimOp);
              innerChain.remove(expandDimOp);
            } else if (expandDimOp.getAxis() == 1) {
              getOpDefChainBwd(outerChain, expandDimOp, expandDimOp);
              outerChain.remove(expandDimOp);
            } else {
              llvm_unreachable("expand dim axis must be 0 or 1");
            }
            dumpOpChain(dumpFlag, op, innerChain, outerChain);
          }
        }
      } else if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
        auto src = broadcastOp.getSrc();
        auto result = broadcastOp.getResult();
        if (auto srcTy = mlir::dyn_cast<RankedTensorType>(src.getType())) {
          if (auto resTy = mlir::dyn_cast<RankedTensorType>(result.getType())) {
            auto srcShape = srcTy.getShape();
            auto resShape = resTy.getShape();
            assert(srcShape.size() <= 2);
            assert(resShape.size() <= 2);
            assert(srcShape.size() == resShape.size());
            if (srcShape[0] != resShape[0]) { // unequal dim 0 shape means
                                              // in the inner axis op chain
              getOpDefChainBwd(innerChain, broadcastOp, broadcastOp);
              innerChain.remove(broadcastOp);
            } else {
              getOpDefChainBwd(outerChain, broadcastOp, broadcastOp);
              outerChain.remove(broadcastOp);
            }
          }
        }
        dumpOpChain(dumpFlag, op, innerChain, outerChain);
      } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMOp>(op)) {
        if (auto _reduceOp =
                findDefOpBwd<triton::ReduceOp>(lm2gmOp.getValue())) {
          auto reduceOp = cast<triton::ReduceOp>(_reduceOp);
          if (reduceOp.getAxis() == 0) {
            getOpDefChainBwd(innerChain, lm2gmOp, lm2gmOp);
          } else if (reduceOp.getAxis() == 1) {
            getOpDefChainBwd(outerChain, lm2gmOp, lm2gmOp);
          } else {
            llvm_unreachable("reduce axis must be 0 or 1");
          }
        }
      } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMMaskOp>(op)) {
        if (auto _reduceOp =
                findDefOpBwd<triton::ReduceOp>(lm2gmOp.getValue())) {
          auto reduceOp = cast<triton::ReduceOp>(_reduceOp);
          if (reduceOp.getAxis() == 0) {
            getOpDefChainBwd(innerChain, lm2gmOp, lm2gmOp);
          } else if (reduceOp.getAxis() == 1) {
            getOpDefChainBwd(outerChain, lm2gmOp, lm2gmOp);
          } else {
            llvm_unreachable("reduce axis must be 0 or 1");
          }
        }
      }
    });
    mod.walk([&](mlir::Operation *op) {
      if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMOp>(op)) {
        if (auto _rangeOp =
                findDefOpBwd<triton::MakeRangeOp>(lm2gmOp.getValue())) {
          if (innerChain.contains(_rangeOp)) {
            getOpDefChainBwd(innerChain, lm2gmOp, lm2gmOp);
          } else if (outerChain.contains(_rangeOp)) {
            getOpDefChainBwd(outerChain, lm2gmOp, lm2gmOp);
          }
          dumpOpChain(dumpFlag, op, innerChain, outerChain);
        }
      } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMMaskOp>(op)) {
        if (auto _rangeOp =
                findDefOpBwd<triton::MakeRangeOp>(lm2gmOp.getValue())) {
          if (innerChain.contains(_rangeOp)) {
            getOpDefChainBwd(innerChain, lm2gmOp, lm2gmOp);
          } else if (outerChain.contains(_rangeOp)) {
            getOpDefChainBwd(outerChain, lm2gmOp, lm2gmOp);
          }
          dumpOpChain(dumpFlag, op, innerChain, outerChain);
        }
      }
    });
  }

  // The common mrOp will be shared while row_size = col_size.
  // In this case, we need to create a new mrOp for innerChain.
  // The two mrOp will be modified with different [inner/outer] encodings.
  void recoverMakeRange(ModuleOp &mod) {
    mod.walk([&](mlir::Operation *op) {
      if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
        OpBuilder builder(rangeOp);
        auto loc = builder.getUnknownLoc();
        // Get the Value (the result of the MakeRangeOp)
        mlir::Value rangeValue = rangeOp.getResult();
        // Use a list to hold the operands to modify. Iterating over users
        // while modifying is generally unsafe/tricky.
        llvm::SmallVector<mlir::OpOperand *> usesToChange;
        // Collect all uses (mlir::OpOperand*) except the first one (i=0)
        int i = 0;
        for (mlir::OpOperand &use : rangeValue.getUses()) {
          if (i++ > 0) {
            usesToChange.push_back(&use);
          }
        }
        // Now, iterate over the collected OpOperands and perform the fix
        for (mlir::OpOperand *operandToChange : usesToChange) {
          // 1. Clone the operation
          auto newRangeOp = builder.create<triton::MakeRangeOp>(
              loc, rangeOp.getType(), rangeOp.getStart(), rangeOp.getEnd());
          // 2. Set the operand to use the new operation's result
          operandToChange->set(newRangeOp.getResult());
        }
      }
    });
  }

  // Modify All Op Encoding
  void modifyOpEncoding(ModuleOp &mod, MLIRContext *context,
                        SetVector<Operation *> &innerChain,
                        SetVector<Operation *> &outerChain) {
    size_t ngroup = 1;
    size_t groupsize = 64;
    bool isFirst = true;

    auto getGroupInfo = [&](RankedTensorType &tensorType) {
      if (auto globalEncoding = dyn_cast<triton::xpu::ClusterLayoutAttr>(
              tensorType.getEncoding())) {
        auto shape = tensorType.getShape();
        size_t _ngroup = product(globalEncoding.getGroupsPerCluster());
        size_t _groupsize = product(globalEncoding.getCoresPerGroup());
        size_t _sizePerCore = product(globalEncoding.getSizePerCore());
        size_t ncore = _ngroup * _groupsize;
        size_t m = shape.front();
        size_t n = shape.back();
        size_t newgroupsize =
            ceil<size_t>(n, static_cast<size_t>(this->bufferSize));
        newgroupsize = roundupPow2(newgroupsize);
        // min is for not using the whole 64 cores case
        size_t newngroup = std::min(ceil<size_t>(ncore, newgroupsize), m);
        newgroupsize = std::min(ceil<size_t>(ncore, newngroup), n);
        if (isFirst) {
          ngroup = newngroup;
          groupsize = newgroupsize;
        } else {
          assert(ngroup == newngroup && "reduce ngroup is not consistent");
          assert(groupsize == newgroupsize &&
                 "reduce groupsize is not consistent");
        }
        isFirst = false;
      }
    };

    // Step 0. Get Group Info
    if (this->groupsPerCluster > 1) {
      ngroup = this->groupsPerCluster;
      assert(this->coreNum % ngroup == 0 &&
             "groups_per_cluster only could be 1, 2, 4, 8, 16, 32, 64");
      groupsize = ceil<size_t>(this->coreNum, ngroup);
    } else {
      mod.walk([&](mlir::Operation *op) {
        if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
          if (auto tensorType =
                  dyn_cast<RankedTensorType>(reduceOp.getOperandTypes()[0])) {
            if (tensorType.getShape().size() == 2) {
              getGroupInfo(tensorType);
            } else if (isAxisNone(reduceOp)) {
              auto defOp = reduceOp.getSrcs()[0].getDefiningOp();
              if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(defOp)) {
                if (auto reshapeResTy = dyn_cast<RankedTensorType>(
                        reshapeOp.getResult().getType())) {
                  if (reshapeResTy.getShape().size() == 1) {
                    auto reshapeSrcTy = cast<RankedTensorType>(
                        reshapeOp.getOperand().getType());
                    getGroupInfo(reshapeSrcTy);
                  }
                }
              }
            }
          }
        }
      });
    }
    LLVM_DEBUG(llvm::dbgs() << "[Reduction SoftGroup]: "
                            << "GroupNum = " << ngroup
                            << ", GroupSize = " << groupsize << "\n");
    // Step 1. Modify All Op Encoding
    mod.walk([&](mlir::Operation *op) {
      auto opResults = op->getResults();
      for (auto opResult : opResults) {
        if (auto resTy = dyn_cast<RankedTensorType>(opResult.getType())) {
          auto shape = resTy.getShape();
          auto elemTy = resTy.getElementType();
          auto encoding = resTy.getEncoding();
          Attribute newEncoding; // newEncoding

          if (auto globalEncoding =
                  dyn_cast<triton::xpu::ClusterLayoutAttr>(encoding)) {
            newEncoding = getOptimizedGEncoding(
                context, resTy, innerChain, outerChain, op, ngroup, groupsize);
          } else if (auto sliceEncoding =
                         dyn_cast<triton::gpu::SliceEncodingAttr>(encoding)) {
            // must be globalEncoding
            if (auto parentEncoding = dyn_cast<triton::xpu::ClusterLayoutAttr>(
                    sliceEncoding.getParent())) {
              auto newParentEncoding =
                  getOptimizedGEncoding(context, resTy, innerChain, outerChain,
                                        op, ngroup, groupsize);
              newEncoding = triton::gpu::SliceEncodingAttr::get(
                  context, sliceEncoding.getDim(), newParentEncoding);
            } else {
              llvm_unreachable("Unsupported SliceEncoding's Parent Attribute");
            }
          } else {
            llvm_unreachable("Unsupported Encoding Attribute");
          }
          auto newResTy = RankedTensorType::get(shape, elemTy, newEncoding);
          opResult.setType(newResTy);
        }
      }
    });

    if (dumpFlag) {
      bool dump_module = true;
      LLVM_DEBUG(bool dump_module = true;
                 llvm::dbgs() << "\n after modify encoding module \n";
                 mod.walk([&](mlir::Operation *op) {
                   if (dump_module) {
                     mlir::ModuleOp module =
                         op->getParentOfType<mlir::ModuleOp>();
                     module.dump();
                     dump_module = false;
                     llvm::dbgs() << "\n";
                   }
                 }););
    }

    // Step 2. Special Modification For [constOp, expandDimsOp, reduceOp,
    // forOp] Step 2.1. ConstOp: value's encoding is not modified before
    // this walk
    mod.walk([&](arith::ConstantOp constOp) {
      auto newValue = constOp.getValue();
      if (auto attr = dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
        newValue = DenseElementsAttr::getFromRawBuffer(
            mlir::cast<ShapedType>(constOp.getType()), attr.getRawData());
      }
      OpBuilder builder(constOp);
      auto loc = constOp.getLoc();
      auto newConstOp = builder.create<mlir::arith::ConstantOp>(
          loc, constOp.getType(), newValue);

      constOp.replaceAllUsesWith(newConstOp.getResult());
      constOp.erase();
    });

    // Step 2.2. ExpandDimsOp: it expands the data dimension, so its prev
    // cvtOp's correct encoding should be inferd by its operand. cvtOp is
    // actually generated after expandDimsOp, so we need to modify the
    // encoding of the previous cvtOp after determining the shape of
    // expandDimsOp.
    mod.walk([&](triton::ExpandDimsOp expandOp) {
      auto expandOpType = cast<RankedTensorType>(expandOp.getType());
      auto globalEncoding =
          cast<triton::xpu::ClusterLayoutAttr>(expandOpType.getEncoding());

      if (auto cvtOp =
              expandOp.getSrc().getDefiningOp<triton::xpu::ConvertLayoutOp>()) {
        auto cvtOpType = cast<RankedTensorType>(cvtOp.getType());
        auto sliceEncoding =
            cast<triton::gpu::SliceEncodingAttr>(cvtOpType.getEncoding());

        auto newSliceEncoding = triton::gpu::SliceEncodingAttr::get(
            context, sliceEncoding.getDim(), globalEncoding);
        auto newResTy = RankedTensorType::get(
            cvtOpType.getShape(), cvtOpType.getElementType(), newSliceEncoding);

        cvtOp->getResult(0).setType(newResTy);
      } else {
        llvm_unreachable("ExpandDimsOp With Error Operand");
      }
    });

    // Step 2.3. ForOp: we need to modify forOp's argTy, args can't be
    // walked.
    mod.walk([&](scf::ForOp forOp) {
      auto forBody = forOp.getBody();
      // modify forOp's argTy
      auto forArgs = forBody->getArguments();
      for (auto forArg : forArgs) {
        if (auto argTy = dyn_cast<RankedTensorType>(forArg.getType())) {
          auto shape = argTy.getShape();
          auto elemTy = argTy.getElementType();
          auto argEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(argTy.getEncoding());

          auto newArgEncoding = getOptimizedGEncoding(
              context, argTy, innerChain, outerChain, forOp, ngroup, groupsize);
          auto newArgTy = RankedTensorType::get(shape, elemTy, newArgEncoding);

          forArg.setType(newArgTy);
        }
      }

      // modify forOp's resTy
      auto forResults = forOp->getResults();
      for (auto forRes : forResults) {
        if (auto argTy = dyn_cast<RankedTensorType>(forRes.getType())) {
          auto shape = argTy.getShape();
          auto elemTy = argTy.getElementType();
          auto argEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(argTy.getEncoding());

          auto newArgEncoding = getOptimizedGEncoding(
              context, argTy, innerChain, outerChain, forOp, ngroup, groupsize);
          auto newArgTy = RankedTensorType::get(shape, elemTy, newArgEncoding);

          forRes.setType(newArgTy);
        }
      }
    });

    // Step 2.4. ReduceOp: it reduces the data dimension, so its correct
    // encoding should be inferd by its input type.
    mod.walk([&](triton::ReduceOp redOp) {
      assert(redOp->getNumResults() == redOp->getNumOperands());
      for (int i = 0; i < redOp->getNumResults(); ++i) {
        if (auto resTy =
                dyn_cast<RankedTensorType>(redOp.getResult()[i].getType())) {
          // auto resTy =
          // cast<RankedTensorType>(redOp.getResult()[i].getType());
          auto srcTy = cast<RankedTensorType>(redOp.getOperandTypes()[i]);

          auto resSliceEncoding =
              cast<triton::gpu::SliceEncodingAttr>(resTy.getEncoding());
          auto srcGlobalEncoding =
              cast<triton::xpu::ClusterLayoutAttr>(srcTy.getEncoding());

          auto newEncoding = triton::gpu::SliceEncodingAttr::get(
              context, resSliceEncoding.getDim(), srcGlobalEncoding);
          auto newResTy = RankedTensorType::get(
              resTy.getShape(), resTy.getElementType(), newEncoding);

          redOp->getResult(i).setType(newResTy);
        }
      }
    });

    // Step 2.5. ReshapeOp: it changes the data dimension, so its correct
    // encoding should be inferd by its input type.
    mod.walk([&](triton::ReshapeOp reshapeOp) {
      if (auto reshapeResTy =
              dyn_cast<RankedTensorType>(reshapeOp.getResult().getType())) {
        auto reshapeResShape = reshapeResTy.getShape();
        if (reshapeResShape.size() == 1) {
          unsigned ncore = ngroup * groupsize;
          std::vector<unsigned> newSizePerCore = {
              ceil<unsigned>(reshapeResShape[0], ncore)};
          std::vector<unsigned> newCoresPerGroup = {ncore};
          std::vector<unsigned> newGroupsPerCluster = {1};
          std::vector<unsigned> order = {0};
          Attribute newReshapeResEncoding = triton::xpu::ClusterLayoutAttr::get(
              context, newSizePerCore, newCoresPerGroup, newGroupsPerCluster,
              order);
          auto newReshapeResTy = RankedTensorType::get(
              reshapeResShape, reshapeResTy.getElementType(),
              newReshapeResEncoding);
          reshapeOp.getResult().setType(newReshapeResTy);
        }
      }
    });
  }

  // Add ConvertLayout For Braoadcast
  void addCvtForBCOp(ModuleOp &mod, MLIRContext *context) {
    mod.walk([&](triton::xpu::BroadcastOp bcOp) {
      auto resTy = cast<RankedTensorType>(bcOp.getResult().getType());
      auto resEncoding =
          cast<triton::xpu::ClusterLayoutAttr>(resTy.getEncoding());
      auto finEncoding = triton::xpu::ClusterLayoutAttr::get(
          context, resEncoding.getSizePerCore(), resEncoding.getCoresPerGroup(),
          resEncoding.getGroupsPerCluster(), {0, 1});
      auto finTy = RankedTensorType::get(
          resTy.getShape(), getElementTypeOrSelf(resTy), finEncoding);

      OpBuilder builder(bcOp);
      auto newBCOp = builder.create<triton::xpu::BroadcastOp>(
          bcOp->getLoc(), resTy, bcOp.getSrc());
      auto cvt = builder.create<triton::xpu::ConvertLayoutOp>(bcOp->getLoc(),
                                                              finTy, newBCOp);

      bcOp.replaceAllUsesWith(cvt.getResult());
      bcOp->erase();
    });
  }

  void addTensorColSizeForMemoryOp(ModuleOp &mod, MLIRContext *context) {
    mod.walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      auto resTy = cast<RankedTensorType>(gm2lmOp.getResult().getType());
      auto resShape = resTy.getShape();

      if (resShape.size() == 2 && resShape[0] > core_num) {
        OpBuilder builder(gm2lmOp);
        gm2lmOp->setAttr("tensorColSize",
                         builder.getSI32IntegerAttr(
                             std::min((unsigned)resShape[1], rawColSize)));
      }
    });

    mod.walk([&](triton::xpu::LM2GMOp lm2gmOp) {
      auto resTy = cast<RankedTensorType>(lm2gmOp.getValue().getType());
      auto resShape = resTy.getShape();

      if (resShape.size() == 2 && resShape[0] > core_num) {
        OpBuilder builder(lm2gmOp);
        lm2gmOp->setAttr("tensorColSize",
                         builder.getSI32IntegerAttr(
                             std::min((unsigned)resShape[1], rawColSize)));
      }
    });

    mod.walk([&](triton::xpu::GM2LMMaskOp gm2lmOp) {
      auto resTy = cast<RankedTensorType>(gm2lmOp.getResult().getType());
      auto resShape = resTy.getShape();

      if (resShape.size() == 2 && resShape[0] > core_num) {
        OpBuilder builder(gm2lmOp);
        gm2lmOp->setAttr("tensorColSize",
                         builder.getSI32IntegerAttr(
                             std::min((unsigned)resShape[1], rawColSize)));
      }
    });

    mod.walk([&](triton::xpu::LM2GMMaskOp lm2gmOp) {
      auto resTy = cast<RankedTensorType>(lm2gmOp.getValue().getType());
      auto resShape = resTy.getShape();

      if (resShape.size() == 2 && resShape[0] > core_num) {
        OpBuilder builder(lm2gmOp);
        lm2gmOp->setAttr("tensorColSize",
                         builder.getSI32IntegerAttr(
                             std::min((unsigned)resShape[1], rawColSize)));
      }
    });

    mod.walk([&](triton::xpu::LoadOp loadOp) {
      auto resTy = cast<RankedTensorType>(loadOp.getResult().getType());
      auto resShape = resTy.getShape();

      if (resShape.size() == 2 && resShape[0] > core_num) {
        OpBuilder builder(loadOp);
        loadOp->setAttr("tensorColSize",
                        builder.getSI32IntegerAttr(resShape[1]));
      }
    });

    mod.walk([&](triton::xpu::StoreOp storeOp) {
      auto resTy = cast<RankedTensorType>(storeOp.getValue().getType());
      auto resShape = resTy.getShape();

      if (resShape.size() == 2 && resShape[0] > core_num) {
        OpBuilder builder(storeOp);
        storeOp->setAttr("tensorColSize",
                         builder.getSI32IntegerAttr(resShape[1]));
      }
    });
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    // Step 1. Check If Can Be Optimized
    if (!canBeOptimized(mod) && this->groupsPerCluster == 1)
      return;

    // Step 2. Recover MakeRange If It's A Common Op
    recoverMakeRange(mod);

    // Step . Collect innerChain && outerChain
    SetVector<Operation *> innerChain;
    SetVector<Operation *> outerChain;
    getChain(mod, innerChain, outerChain);

    // Step 3. Modify All Op Encoding With Optimization Rules
    modifyOpEncoding(mod, context, innerChain, outerChain);

    // Step 4. Add ConvertLayout For Braoadcast
    // This step can be eliminated if we set sizePerBank with its shape
    if (rowsPerCore > 1) {
      addCvtForBCOp(mod, context);
    }

    // Step 5. Add tensorColSize Attr for GM2LMOp
    // This step can be eliminated if we set sizePerBank with its shape
    if (rowsPerCore > 1) {
      addTensorColSizeForMemoryOp(mod, context);
    }

    if (rowsPerCore == 1)
      LLVM_DEBUG(llvm::dbgs() << "Core Tiling M-ColSize Opt Hit!\n");
    else if (rowsPerCore > 1)
      LLVM_DEBUG(llvm::dbgs() << "Core Tiling S-ColSize Opt Hit!\n");
  }

private:
  unsigned rowsPerCore = 1;
  unsigned rawColSize = 0;
  unsigned core_num = 64;
};

} // namespace xpu
} // namespace triton

} // namespace mlir
