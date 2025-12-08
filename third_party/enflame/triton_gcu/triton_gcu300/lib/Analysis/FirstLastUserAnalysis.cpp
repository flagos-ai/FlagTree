/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Analysis/FirstLastUserAnalysis.h"

#include <algorithm>
#include <vector>

#include "Conversion/TritonToGCU/Utils.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "first-last-user-analysis"

namespace mlir {
namespace triton {
namespace gcu {

namespace {

bool isMustAliasOp(mlir::Operation *op) {
  if (llvm::isa<triton::AddPtrOp, triton::PtrToIntOp, triton::IntToPtrOp,
                triton::BitcastOp, triton::gcu::PtrToIntOp,
                triton::gcu::IntToPtrOp, triton::IntToPtrOp,
                triton::PtrToIntOp>(op)) {
    return true;
  } else if (llvm::isa<triton::gpu::ConvertLayoutOp>(op)) {
    auto convertLayout = cast<triton::gpu::ConvertLayoutOp>(op);
    auto src = convertLayout.getSrc();
    auto srcNumElems = triton::gcu::getElemsPerThread(src.getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(convertLayout.getType());

    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    auto dstTy = dyn_cast<RankedTensorType>(convertLayout.getType());
    if ((!srcTy) || (!dstTy)) {
      assert(false && "srcTy or dstTy not a RankedTensorType");
    }

    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();
    if (srcLayout == dstLayout) {
      return true;
    }
    if (srcNumElems == dstNumElems &&
        src.getType().getShape() == convertLayout.getType().getShape()) {
      if (!mlir::isa<triton::gpu::SharedEncodingTrait>(srcLayout)) {
        return true;
      } else if (isa<triton::gpu::SliceEncodingAttr>(srcLayout) &&
                 isa<triton::gpu::SliceEncodingAttr>(dstLayout)) {
        if (cast<triton::gpu::SliceEncodingAttr>(srcLayout).getDim() ==
            cast<triton::gpu::SliceEncodingAttr>(dstLayout).getDim()) {
          return true;
        }
      }
    }
    return false;
  } else if (isa<triton::ExpandDimsOp>(op)) {
    auto expandDimOp = cast<triton::ExpandDimsOp>(op);
    auto srcNumElems =
        triton::gcu::getElemsPerThread(expandDimOp.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(expandDimOp.getType());
    srcNumElems.insert(srcNumElems.begin() + expandDimOp.getAxis(), 1);
    if (srcNumElems == dstNumElems) {
      return true;
    }
    return false;
  } else if (isa<triton::ReshapeOp>(op)) {
    auto reshapeOp = cast<triton::ReshapeOp>(op);
    auto srcNumElems =
        triton::gcu::getElemsPerThread(reshapeOp.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(reshapeOp.getType());
    if (srcNumElems == dstNumElems) {
      return true;
    }
    return false;
  } else if (isa<triton::BroadcastOp>(op)) {
    auto broastOp = cast<triton::BroadcastOp>(op);
    auto srcNumElems =
        triton::gcu::getElemsPerThread(broastOp.getSrc().getType());
    auto dstNumElems = triton::gcu::getElemsPerThread(broastOp.getType());
    if (srcNumElems == dstNumElems) {
      return true;
    }
    return false;
  } else {
    return false;
  }
}

template <typename DomInfoT, typename FuncT>
void getUsersWithAlias(mlir::Operation *op, DomInfoT &domInfo, FuncT &funcInfo,
                       std::vector<mlir::Operation *> &userList,
                       std::vector<mlir::Block *> &blockList,
                       std::vector<mlir::Operation *> &aliasList) {
  auto opRegion = op->getParentRegion();
  for (auto user : op->getUsers()) {
    if (llvm::isa<memref::DeallocOp>(user)) {
      llvm::report_fatal_error("double free please checek IR");
    }

    if (user->getParentRegion() == opRegion) {
      userList.push_back(user);
      blockList.push_back(user->getBlock());
      if (isMustAliasOp(user) || llvm::isa<scf::ForOp, scf::WhileOp>(user)) {
        aliasList.push_back(user);
      }
    } else {
      auto parent = user->getParentOp();
      auto curUser = user;
      bool mayAlias =
          llvm::isa<scf::IfOp, scf::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
              parent) &&
          llvm::isa<scf::YieldOp>(curUser);

      while ((!isa<triton::FuncOp>(parent)) &&
             (!isa<mlir::func::FuncOp>(parent))) {
        if (parent->getParentRegion() == opRegion)
          break;

        curUser = mayAlias ? funcInfo(parent, domInfo) : nullptr;
        parent = parent->getParentOp();
        mayAlias =
            llvm::isa<scf::IfOp, scf::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
                parent) &&
            llvm::isa_and_nonnull<scf::YieldOp>(curUser);
      }

      if (parent->getParentRegion() != opRegion) {
        parent->dump();
        op->dump();
        llvm::report_fatal_error("invalid user please checek IR");
      }
      userList.push_back(parent);
      blockList.push_back(parent->getBlock());
      if (mayAlias) {
        aliasList.push_back(parent);
      }
    }
  }
}

mlir::Operation *getLastUserOfOp(mlir::Operation *op,
                                 PostDominanceInfo &postDomInfo) {
  auto opRegion = op->getParentRegion();
  std::vector<mlir::Operation *> userList;
  std::vector<mlir::Block *> blockList;
  std::vector<mlir::Operation *> aliasList;

  getUsersWithAlias(op, postDomInfo, getLastUserOfOp, userList, blockList,
                    aliasList);

  // Analysis alias op
  while (!aliasList.empty()) {
    std::vector<mlir::Operation *> tmpList(aliasList.size());
    std::copy(aliasList.begin(), aliasList.end(), tmpList.begin());
    aliasList.clear();
    for (auto tmp : tmpList) {
      getUsersWithAlias(tmp, postDomInfo, getLastUserOfOp, userList, blockList,
                        aliasList);
    }
  }

  if (blockList.empty())
    return nullptr;

  Block *dom = postDomInfo.findNearestCommonDominator(blockList);

  /**        B0
  //        /  \
  //       v    v
  //      B1 <- B2
  //             |
  //             v
  //            B3
  //   1). B1 and B3 has a "return" op.
  //   2). B2 and B3 has a use for a alloc op which locate in B0. At the time,
  //   the "postDomInfo.findNearestCommonDominator"  return nullptr
  **/
  if (dom == nullptr) {
    auto lastBlock = blockList[0];
    for (auto iter = opRegion->rbegin(); iter != opRegion->rend(); iter++) {
      auto index = std::find(blockList.begin(), blockList.end(), &(*iter));
      if (index != blockList.end()) {
        lastBlock = &(*iter);
        break;
      }
    }
    dom = lastBlock;
  }

  if (dom->empty()) {
    llvm::report_fatal_error("dominator block is empty");
  }

  mlir::Operation *lastUser = nullptr;

  // Block dom maybe is not in the blockList
  auto lastBlockIter = std::find(blockList.begin(), blockList.end(), dom);
  if (lastBlockIter == blockList.end()) {
    lastUser = &(dom->front());
  } else {
    for (size_t i = 0; i < userList.size(); ++i) {
      if (dom != userList[i]->getBlock()) {
        continue;
      }

      if (lastUser == nullptr) {
        lastUser = userList[i];
        continue;
      }

      if (lastUser->isBeforeInBlock(userList[i])) {
        lastUser = userList[i];
      }
    }
  }

  if (lastUser && isMustAliasOp(lastUser)) {
    lastUser = nullptr;
  }
  return lastUser;
}

mlir::Operation *getFirstUserOfOp(mlir::Operation *op, DominanceInfo &domInfo) {
  auto opRegion = op->getParentRegion();

  std::vector<mlir::Operation *> userList;
  std::vector<mlir::Block *> blockList;
  std::vector<mlir::Operation *> aliasList;

  getUsersWithAlias(op, domInfo, getFirstUserOfOp, userList, blockList,
                    aliasList);

  // Analysis alias op
  while (!aliasList.empty()) {
    std::vector<mlir::Operation *> tmpList(aliasList.size());
    std::copy(aliasList.begin(), aliasList.end(), tmpList.begin());
    aliasList.clear();
    for (auto tmp : tmpList) {
      getUsersWithAlias(tmp, domInfo, getFirstUserOfOp, userList, blockList,
                        aliasList);
    }
  }

  if (blockList.empty())
    return nullptr;

  Block *dom = domInfo.findNearestCommonDominator(blockList);
  if (dom == nullptr) {
    llvm::report_fatal_error("cannot find nearest common dominator block");
  }

  if (dom->empty()) {
    llvm::report_fatal_error("dominator block is empty");
  }

  mlir::Operation *firstUser = nullptr;
  auto firstBlockIter = std::find(blockList.begin(), blockList.end(), dom);
  if (firstBlockIter == blockList.end()) {
    firstUser = &(dom->back());
  } else {
    for (size_t i = 0; i < userList.size(); ++i) {
      if (dom == userList[i]->getBlock()) {
        if (firstUser == nullptr) {
          firstUser = userList[i];
          continue;
        }
        if (userList[i]->isBeforeInBlock(firstUser)) {
          firstUser = userList[i];
        }
      }
    }
  }

  if (firstUser && isMustAliasOp(firstUser)) {
    firstUser = nullptr;
  }

  if (firstUser == nullptr) {
    return nullptr;
  }

  auto nextOp = op->getNextNode();
  while (isa<memref::DeallocOp>(nextOp)) {
    nextOp = nextOp->getNextNode();
  }
  if (nextOp == firstUser) {
    firstUser = nullptr;
  }
  return firstUser;
}

} // namespace

mlir::Operation *FirstLastUserAnalysis::getLastUserOp(mlir::Value value,
                                                      mlir::Region *opRegion) {
  std::vector<mlir::Operation *> userList;
  std::vector<mlir::Block *> blockList;
  std::vector<mlir::Operation *> aliasList;

  for (auto user : value.getUsers()) {
    if (llvm::isa<memref::DeallocOp>(user)) {
      llvm::report_fatal_error("double free please checek IR");
    }

    if (user->getParentRegion() == opRegion) {
      userList.push_back(user);
      blockList.push_back(user->getBlock());
      if (isMustAliasOp(user)) {
        aliasList.push_back(user);
      }
    } else {
      auto parent = user->getParentOp();
      auto curUser = user;
      bool mayAlias = llvm::isa<scf::IfOp, scf::IndexSwitchOp>(parent) &&
                      llvm::isa<scf::YieldOp>(curUser);

      while ((!isa<triton::FuncOp>(parent)) &&
             (!isa<mlir::func::FuncOp>(parent))) {
        if (parent->getParentRegion() == opRegion)
          break;

        curUser = mayAlias ? getLastUserOfOp(parent, postDominators) : nullptr;
        parent = parent->getParentOp();
        mayAlias = llvm::isa<scf::IfOp, scf::IndexSwitchOp>(parent) &&
                   llvm::isa_and_nonnull<scf::YieldOp>(curUser);
      }

      if (parent->getParentRegion() != opRegion) {
        parent->dump();
        value.dump();
        llvm_unreachable("invalid user please checek IR 1");
      }
      userList.push_back(parent);
      blockList.push_back(parent->getBlock());
      if (mayAlias) {
        aliasList.push_back(parent);
      }
    }
  }

  // Analysis alias op
  while (!aliasList.empty()) {
    std::vector<mlir::Operation *> tmpList(aliasList.size());
    std::copy(aliasList.begin(), aliasList.end(), tmpList.begin());
    aliasList.clear();
    for (auto tmp : tmpList) {
      getUsersWithAlias(tmp, postDominators, getLastUserOfOp, userList,
                        blockList, aliasList);
    }
  }

  if (blockList.empty())
    return nullptr;

  Block *dom = postDominators.findNearestCommonDominator(blockList);

  /**        B0
  //        /  \
  //      B1 <- B2
  //             |
  //            B3
  //   1). B1 and B3 has a "return" op.
  //   2). B2 and B3 has a use for a alloc op which locate in B0. At the time,
  //   the "postDominators.findNearestCommonDominator"  return nullptr
  **/
  if (dom == nullptr) {
    auto lastBlock = blockList[0];
    for (auto iter = opRegion->rbegin(); iter != opRegion->rend(); iter++) {
      auto index = std::find(blockList.begin(), blockList.end(), &(*iter));
      if (index != blockList.end()) {
        lastBlock = &(*iter);
        break;
      }
    }
    dom = lastBlock;
  }

  if (dom->empty()) {
    llvm::report_fatal_error("dominator block is empty");
  }

  mlir::Operation *lastUser = nullptr;

  // Block dom maybe is not in the blockList
  auto lastBlockIter = std::find(blockList.begin(), blockList.end(), dom);
  if (lastBlockIter == blockList.end()) {
    lastUser = &(dom->front());
  } else {
    for (size_t i = 0; i < userList.size(); ++i) {
      if (dom != userList[i]->getBlock()) {
        continue;
      }

      if (lastUser == nullptr) {
        lastUser = userList[i];
        continue;
      }

      if (lastUser->isBeforeInBlock(userList[i])) {
        lastUser = userList[i];
      }
    }
  }

  if (lastUser && isMustAliasOp(lastUser)) {
    lastUser = nullptr;
  }
  return lastUser;
}

void FirstLastUserAnalysis::start() {
  assert(llvm::isa<mlir::gpu::GPUModuleOp>(moduleOp) &&
         "The input operation is not a gpu module");
  moduleOp->walk<WalkOrder::PreOrder>([&](mlir::Operation *_op) {
    if (_op->getResults().empty())
      return;

    if (llvm::isa<arith::ConstantOp>(_op) &&
        llvm::any_of(_op->getResultTypes(), llvm::IsaPred<RankedTensorType>)) {
      lastUserMap[_op] = getLastUserOfOp(_op, postDominators);
    } else if (llvm::isa<
                   scf::IfOp, scf::IndexSwitchOp, scf::WhileOp, scf::ForOp,
                   triton::SplatOp, arith::ConstantOp, triton::AddPtrOp,
                   triton::PtrToIntOp, triton::IntToPtrOp,
                   triton::gcu::PtrToIntOp, triton::gcu::IntToPtrOp,
                   triton::MulhiUIOp, triton::ScanOp, triton::HistogramOp,
                   triton::gcu::LoadOp, triton::LoadOp, triton::BroadcastOp,
                   triton::ExpandDimsOp, triton::ReshapeOp, triton::SplitOp,
                   triton::JoinOp, triton::CatOp, triton::gcu::MatmulOp,
                   triton::DotOp, triton::ReduceOp, triton::MakeRangeOp,
                   triton::BitcastOp, triton::gcu::ElementwiseFusionRegionOp>(
                   _op)) {
      lastUserMap[_op] = getLastUserOfOp(_op, postDominators);
    } else if (llvm::isa<triton::TransOp, triton::gpu::ConvertLayoutOp,
                         triton::gcu::LocalLoadOp>(_op)) {
      lastUserMap[_op] = getLastUserOfOp(_op, postDominators);
      firstUserMap[_op] = getFirstUserOfOp(_op, dominators);
    }
  });
}
} // namespace gcu
} // namespace triton
} // namespace mlir
