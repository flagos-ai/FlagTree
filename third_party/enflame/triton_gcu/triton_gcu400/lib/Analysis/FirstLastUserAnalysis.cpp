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
#include <utility>
#include <vector>

#include "Conversion/TritonToGCU/Utility.h"
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

// Need to deal with 'maybe alias'
void FirstLastUserAnalysis::getUsersForLast(
    mlir::Value value, mlir::Region *opRegion, PostDominanceInfo &postDomInfo,
    llvm::SetVector<std::pair<Operation *, int>> &userList,
    llvm::SetVector<Block *> &blockList,
    llvm::SetVector<std::pair<Operation *, int>> &aliasList) {
  for (auto &use : value.getUses()) {
    Operation *user = use.getOwner();
    auto number = use.getOperandNumber();

    if (user->getParentRegion() == opRegion) {
      blockList.insert(user->getBlock());
      if (isMustAliasOp(user) || llvm::isa<scf::WhileOp>(user)) {
        userList.insert(std::make_pair(user, number));
        aliasList.insert(std::make_pair(user, number));
      } else if (llvm::isa<scf::ForOp>(user)) {
        auto forOp = dyn_cast<scf::ForOp>(user);
        auto numControl = forOp.getNumControlOperands();
        if (number < numControl) {
          userList.insert(std::make_pair(user, number));
        } else {
          userList.insert(std::make_pair(user, number - numControl));
          aliasList.insert(std::make_pair(user, number - numControl));
        }
      } else {
        userList.insert(std::make_pair(user, number));
      }
    } else {
      auto parent = user->getParentOp();

      std::pair<Operation *, int> curUser = std::make_pair(nullptr, -1);
      if (isMustAliasOp(user) || llvm::isa<scf::WhileOp>(user)) {
        auto result = user->getResults()[number];
        curUser = getLastUserOfValue(result, postDomInfo);
        if (lastUserMap.count(result) == 0)
          lastUserMap[result] = curUser;
      } else if (llvm::isa<scf::ForOp>(user)) {
        auto forOp = dyn_cast<scf::ForOp>(user);
        auto numControl = forOp.getNumControlOperands();
        if (number < numControl) {
          curUser = std::make_pair(user, number);
        } else {
          auto result = user->getResults()[number - numControl];
          curUser = getLastUserOfValue(result, postDomInfo);
          if (lastUserMap.count(result) == 0)
            lastUserMap[result] = curUser;
        }
      } else {
        curUser = std::make_pair(user, number);
      }
      bool mayAlias =
          llvm::isa<scf::IfOp, scf::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
              parent) &&
          llvm::isa_and_nonnull<scf::YieldOp>(curUser.first);

      while ((!isa<triton::FuncOp>(parent)) &&
             (!isa<mlir::func::FuncOp>(parent))) {
        if (parent->getParentRegion() == opRegion)
          break;

        if (mayAlias) {
          auto result = parent->getResults()[curUser.second];
          curUser = getLastUserOfValue(result, postDomInfo);
          if (lastUserMap.count(result) == 0)
            lastUserMap[result] = curUser;

        } else {
          curUser = std::make_pair(nullptr, -1);
        }
        parent = parent->getParentOp();
        mayAlias =
            llvm::isa<scf::IfOp, scf::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
                parent) &&
            llvm::isa_and_nonnull<scf::YieldOp>(curUser.first);
      }

      if (parent->getParentRegion() != opRegion) {
        parent->dump();
        value.getDefiningOp()->dump();
        llvm::report_fatal_error("invalid user please checek IR");
      }
      userList.insert(std::make_pair(parent, curUser.second));
      blockList.insert(parent->getBlock());
      if (mayAlias) {
        aliasList.insert(std::make_pair(parent, curUser.second));
      }
    }
  }
}

// Get Last User in value's region
std::pair<Operation *, int>
FirstLastUserAnalysis::getLastUserOfValue(mlir::Value value,
                                          PostDominanceInfo &postDomInfo) {
  if (lastUserMap.count(value) != 0)
    return lastUserMap[value];

  Region *opRegion = value.getParentRegion();
  if (!opRegion)
    llvm::report_fatal_error("can't analysis block argument");

  llvm::SetVector<std::pair<Operation *, int>> userList;
  llvm::SetVector<mlir::Block *> blockList;
  llvm::SetVector<std::pair<Operation *, int>> aliasList;
  getUsersForLast(value, opRegion, postDomInfo, userList, blockList, aliasList);

  // Analysis alias op
  llvm::SetVector<std::pair<Operation *, int>> allAliasList;
  while (!aliasList.empty()) {
    llvm::SetVector<std::pair<Operation *, int>> tmpList;
    tmpList.insert(aliasList.begin(), aliasList.end());
    allAliasList.insert(aliasList.begin(), aliasList.end());
    aliasList.clear();
    for (auto tmp : tmpList) {
      if (llvm::isa<scf::IfOp, scf::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
              tmp.first)) {
        getUsersForLast(tmp.first->getResults()[tmp.second], opRegion,
                        postDomInfo, userList, blockList, aliasList);
      } else {
        getUsersForLast(tmp.first->getResults()[0], opRegion, postDomInfo,
                        userList, blockList, aliasList);
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "userList:" << userList.size() << "\n";
    llvm::dbgs() << "blockList:" << blockList.size() << "\n";
  });

  if (blockList.empty())
    return std::make_pair(nullptr, -1);

  std::vector<mlir::Block *> tmpBlocks(blockList.begin(), blockList.end());
  Block *dom = postDomInfo.findNearestCommonDominator(tmpBlocks);

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

  std::pair<Operation *, int> lastUser = std::make_pair(nullptr, -1);

  // auto getUserOp = [&](const std::pair<Operation *, int> &user) {
  //   return user.first;
  // };
  // auto mappedBegin = llvm::map_iterator(allAliasList.begin(), getUserOp);
  // auto mappedEnd = llvm::map_iterator(allAliasList.end(), getUserOp);

  // Block dom maybe is not in the blockList
  auto lastBlockIter = std::find(blockList.begin(), blockList.end(), dom);
  if (lastBlockIter == blockList.end()) {
    lastUser = std::make_pair(&(dom->front()), -1);
  } else {
    for (size_t i = 0; i < userList.size(); ++i) {
      // auto aliasIter = std::find(mappedBegin, mappedEnd, userList[i].first);
      // LLVM_DEBUG({
      //   llvm::dbgs() << "aliasIter:"
      //                << (aliasIter == mappedEnd) << "\n";
      // });

      // if (dom != userList[i].first->getBlock() ||
      //     aliasIter != mappedEnd) {
      //   continue;
      // }
      if (dom != userList[i].first->getBlock()) {
        continue;
      }

      if (lastUser.first == nullptr) {
        lastUser = userList[i];
        continue;
      }

      if (lastUser.first->isBeforeInBlock(userList[i].first)) {
        lastUser = userList[i];
      }
    }
  }

  return lastUser;
}

// No Need to deal with 'maybe alias'
void FirstLastUserAnalysis::getUsersForFisrt(
    mlir::Value value, mlir::Region *opRegion,
    llvm::SetVector<std::pair<Operation *, int>> &userList,
    llvm::SetVector<mlir::Block *> &blockList,
    llvm::SetVector<std::pair<Operation *, int>> &aliasList) {
  for (auto &use : value.getUses()) {
    Operation *user = use.getOwner();
    auto number = use.getOperandNumber();

    if (user->getParentRegion() == opRegion) {
      userList.insert(std::make_pair(user, number));
      blockList.insert(user->getBlock());
      if (isMustAliasOp(user)) {
        aliasList.insert(std::make_pair(user, number));
      }
    } else {
      auto parent = user->getParentOp();

      while ((!isa<triton::FuncOp>(parent)) &&
             (!isa<mlir::func::FuncOp>(parent))) {
        if (parent->getParentRegion() == opRegion)
          break;

        parent = parent->getParentOp();
      }

      if (parent->getParentRegion() != opRegion) {
        parent->dump();
        value.getDefiningOp()->dump();
        llvm::report_fatal_error("invalid user please checek IR");
      }

      userList.insert(std::make_pair(parent, -1));
      blockList.insert(parent->getBlock());
    }
  }
}

std::pair<Operation *, int>
FirstLastUserAnalysis::getFirstUserOfValue(mlir::Value value,
                                           DominanceInfo &domInfo) {
  Region *opRegion = value.getParentRegion();
  if (!opRegion) {
    llvm::report_fatal_error("can't analysis block argument");
  }

  llvm::SetVector<std::pair<Operation *, int>> userList;
  llvm::SetVector<mlir::Block *> blockList;
  llvm::SetVector<std::pair<Operation *, int>> aliasList;

  getUsersForFisrt(value, opRegion, userList, blockList, aliasList);

  // Analysis alias op
  llvm::SetVector<std::pair<Operation *, int>> allAliasList;
  while (!aliasList.empty()) {
    llvm::SetVector<std::pair<Operation *, int>> tmpList;
    tmpList.insert(aliasList.begin(), aliasList.end());
    allAliasList.insert(aliasList.begin(), aliasList.end());
    aliasList.clear();
    for (auto tmp : tmpList) {
      getUsersForFisrt(tmp.first->getResults()[0], opRegion, userList,
                       blockList, aliasList);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "userList:" << userList.size() << "\n";
    llvm::dbgs() << "blockList:" << blockList.size() << "\n";
  });

  if (blockList.empty())
    return std::make_pair(nullptr, -1);

  std::vector<mlir::Block *> tmpBlocks(blockList.begin(), blockList.end());
  Block *dom = domInfo.findNearestCommonDominator(tmpBlocks);
  if (!dom) {
    llvm::report_fatal_error("cannot find nearest common dominator block");
  }
  if (dom->empty()) {
    llvm::report_fatal_error("dominator block is empty");
  }

  std::pair<Operation *, int> firstUser = std::make_pair(nullptr, -1);

  auto getUserOp = [&](const std::pair<Operation *, int> &user) {
    return user.first;
  };
  auto mappedBegin = llvm::map_iterator(allAliasList.begin(), getUserOp);
  auto mappedEnd = llvm::map_iterator(allAliasList.end(), getUserOp);

  auto firstBlockIter = std::find(blockList.begin(), blockList.end(), dom);
  if (firstBlockIter == blockList.end()) {
    firstUser = std::make_pair(&(dom->back()), -1);
  } else {
    for (size_t i = 0; i < userList.size(); ++i) {
      if (dom != userList[i].first->getBlock() ||
          std::find(mappedBegin, mappedEnd, userList[i].first) != mappedEnd) {
        continue;
      }

      if (firstUser.first == nullptr) {
        firstUser = userList[i];
        continue;
      }

      if (userList[i].first->isBeforeInBlock(firstUser.first)) {
        firstUser = userList[i];
      }
    }
  }

  if (firstUser.first == nullptr) {
    return std::make_pair(nullptr, -1);
  }

  if (value.getDefiningOp()->getNextNode() == firstUser.first) {
    firstUser = std::make_pair(nullptr, -1);
  }

  return firstUser;
}

std::pair<Operation *, int>
FirstLastUserAnalysis::getLastUser(mlir::Value value, mlir::Region *opRegion) {
  llvm::SetVector<std::pair<Operation *, int>> userList;
  llvm::SetVector<mlir::Block *> blockList;
  llvm::SetVector<std::pair<Operation *, int>> aliasList;

  getUsersForLast(value, opRegion, postDominators, userList, blockList,
                  aliasList);

  // Analysis alias op
  llvm::SetVector<std::pair<Operation *, int>> allAliasList;
  while (!aliasList.empty()) {
    llvm::SetVector<std::pair<Operation *, int>> tmpList;
    tmpList.insert(aliasList.begin(), aliasList.end());
    allAliasList.insert(aliasList.begin(), aliasList.end());
    aliasList.clear();
    for (auto tmp : tmpList) {
      if (llvm::isa<scf::IfOp, scf::IndexSwitchOp, scf::ForOp, scf::WhileOp>(
              tmp.first)) {
        getUsersForLast(tmp.first->getResults()[tmp.second], opRegion,
                        postDominators, userList, blockList, aliasList);
      } else {
        getUsersForLast(tmp.first->getResults()[0], opRegion, postDominators,
                        userList, blockList, aliasList);
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "userList:" << userList.size() << "\n";
    llvm::dbgs() << "blockList:" << blockList.size() << "\n";
  });

  if (blockList.empty())
    return std::make_pair(nullptr, -1);

  std::vector<mlir::Block *> tmpBlocks(blockList.begin(), blockList.end());
  Block *dom = postDominators.findNearestCommonDominator(tmpBlocks);

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

  std::pair<Operation *, int> lastUser = std::make_pair(nullptr, -1);

  // auto getUserOp = [&](const std::pair<Operation *, int> &user) {
  //   return user.first;
  // };
  // auto mappedBegin = llvm::map_iterator(allAliasList.begin(), getUserOp);
  // auto mappedEnd = llvm::map_iterator(allAliasList.end(), getUserOp);

  // Block dom maybe is not in the blockList
  auto lastBlockIter = std::find(blockList.begin(), blockList.end(), dom);
  if (lastBlockIter == blockList.end()) {
    lastUser = std::make_pair(&(dom->front()), -1);
  } else {
    for (size_t i = 0; i < userList.size(); ++i) {
      // auto aliasIter = std::find(mappedBegin, mappedEnd, userList[i].first);
      // LLVM_DEBUG({
      //   llvm::dbgs() << "aliasIter:"
      //                << (aliasIter == mappedEnd) << "\n";
      // });

      // if (dom != userList[i].first->getBlock() ||
      //     aliasIter != mappedEnd) {
      //   continue;
      // }
      if (dom != userList[i].first->getBlock()) {
        continue;
      }

      if (lastUser.first == nullptr) {
        lastUser = userList[i];
        continue;
      }

      if (lastUser.first->isBeforeInBlock(userList[i].first)) {
        lastUser = userList[i];
      }
    }
  }

  return lastUser;
}

void FirstLastUserAnalysis::start() {
  assert(llvm::isa<mlir::gpu::GPUModuleOp>(moduleOp) &&
         "The input operation is not a gpu module");
  moduleOp->walk<WalkOrder::PreOrder>([&](mlir::Operation *_op) {
    if (_op->getResults().empty())
      return;

    if (llvm::isa<arith::ConstantOp, triton::PtrToIntOp, triton::IntToPtrOp,
                  triton::gcu::PtrToIntOp, triton::gcu::IntToPtrOp,
                  triton::AddPtrOp, triton::LoadOp>(_op) &&
        llvm::any_of(_op->getResultTypes(), llvm::IsaPred<RankedTensorType>)) {
      LLVM_DEBUG({ llvm::dbgs() << "_op:" << *_op << "\n"; });
      int i = 0;
      for (auto v : _op->getResults()) {
        LLVM_DEBUG({ llvm::dbgs() << "i:" << i++ << "\n"; });
        if (lastUserMap.count(v) == 0)
          lastUserMap[v] = getLastUserOfValue(v, postDominators);

        LLVM_DEBUG({
          if (lastUserMap[v].first) {
            llvm::dbgs() << "lastUserMap[v].first :" << *lastUserMap[v].first
                         << "\n";
          } else {
            llvm::dbgs() << "lastUserMap[v].first is nullptr" << "\n";
          }
        });
      }
    } else if (llvm::isa<scf::IfOp, scf::IndexSwitchOp, scf::WhileOp,
                         scf::ForOp, triton::SplatOp, triton::AtomicRMWOp,
                         triton::MulhiUIOp, triton::ScanOp, triton::HistogramOp,
                         triton::BroadcastOp, triton::ExpandDimsOp,
                         triton::ReshapeOp, triton::SplitOp, triton::JoinOp,
                         triton::CatOp, triton::gcu::MatmulOp, triton::DotOp,
                         triton::ReduceOp, triton::MakeRangeOp,
                         triton::gcu::ElementwiseFusionRegionOp>(_op)) {
      LLVM_DEBUG({ llvm::dbgs() << "_op:" << *_op << "\n"; });
      int i = 0;
      for (auto v : _op->getResults()) {
        LLVM_DEBUG({ llvm::dbgs() << "i:" << i++ << "\n"; });
        if (lastUserMap.count(v) == 0)
          lastUserMap[v] = getLastUserOfValue(v, postDominators);

        LLVM_DEBUG({
          if (lastUserMap[v].first) {
            llvm::dbgs() << "lastUserMap[v].first :" << *lastUserMap[v].first
                         << "\n";
          } else {
            llvm::dbgs() << "lastUserMap[v].first is nullptr" << "\n";
          }
        });
      }
    } else if (llvm::isa<triton::TransOp, triton::gpu::ConvertLayoutOp,
                         triton::gcu::LoadOp, triton::gcu::LocalLoadOp>(_op)) {
      LLVM_DEBUG({ llvm::dbgs() << "_op:" << *_op << "\n"; });
      int i = 0;
      for (auto v : _op->getResults()) {
        LLVM_DEBUG({ llvm::dbgs() << "i:" << i++ << "\n"; });
        if (lastUserMap.count(v) == 0)
          lastUserMap[v] = getLastUserOfValue(v, postDominators);

        firstUserMap[v] = getFirstUserOfValue(v, dominators);

        LLVM_DEBUG({
          if (lastUserMap[v].first) {
            llvm::dbgs() << "lastUserMap[v].first :" << *lastUserMap[v].first
                         << "\n";
          } else {
            llvm::dbgs() << "lastUserMap[v].first is nullptr\n";
          }

          if (firstUserMap[v].first) {
            llvm::dbgs() << "firstUserMap[v].first :" << *firstUserMap[v].first
                         << "\n";
          } else {
            llvm::dbgs() << "firstUserMap[v].first is nullptr\n";
          }
        });
      }
    }
  });
}
} // namespace gcu
} // namespace triton
} // namespace mlir
