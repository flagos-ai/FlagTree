/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

static constexpr const char *DEBUG_TYPE = "BufferCountManager";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

namespace mlir {
namespace triton {

namespace {

constexpr int kDefaultIntraBufferCount = 3;
constexpr int kDefaultInterBufferCount = 2;
constexpr int kDefaultLoadBufferCount = 1;
constexpr int kBufferCountWarningThreshold = 3;

inline llvm::StringLiteral getAttrName(BufferCountManager::DepType type) {
  switch (type) {
  case BufferCountManager::DepType::IntraCore:
    return CVPipeline::kIntraBufCount;
  case BufferCountManager::DepType::InterCore:
    return CVPipeline::kInterCoreBufCount;
  case BufferCountManager::DepType::LoadStore:
    return CVPipeline::kLoadStoreBufCount;
  }
  llvm_unreachable("unknown BufferCountManager::DepType");
}

inline int getDefaultCount(BufferCountManager::DepType type) {
  switch (type) {
  case BufferCountManager::DepType::IntraCore:
    return kDefaultIntraBufferCount;
  case BufferCountManager::DepType::InterCore:
    return kDefaultInterBufferCount;
  case BufferCountManager::DepType::LoadStore:
    return kDefaultLoadBufferCount;
  }
  llvm_unreachable("unknown BufferCountManager::DepType");
}

} // namespace

BufferCountManager::BufferCountManager(Operation *root)
    : module_(root ? root->getParentOfType<ModuleOp>() : ModuleOp()) {
  initFromModule();
}

BufferCountManager::BufferCountManager(ModuleOp module) : module_(module) {
  initFromModule();
}

void BufferCountManager::initFromModule() {
  if (!module_) {
    LOG_DEBUG("initFromModule: module_ is null, skip");
    return;
  }
  OpBuilder builder(module_.getContext());
  LOG_DEBUG("initFromModule: getContext OK, about to loop");
  for (auto type :
       {DepType::IntraCore, DepType::InterCore, DepType::LoadStore}) {
    if (module_->getAttrOfType<IntegerAttr>(getAttrName(type))) {
      LOG_DEBUG("initFromModule: type=" << (int)type << " already set, skip");
      continue;
    }
    module_->setAttr(getAttrName(type),
                     builder.getI32IntegerAttr(getDefaultCount(type)));
    LOG_DEBUG("initFromModule: type=" << (int)type << " set to "
                                      << getDefaultCount(type));
  }
}

void BufferCountManager::setBufferCount(DepType type, int count) {
  if (count <= 0) {
    LOG_DEBUG("Invalid buffer count: " << count << " (must be > 0)");
    return;
  }
  if (count > kBufferCountWarningThreshold) {
    LOG_DEBUG("Warning: buffer count " << count << " > "
                                       << kBufferCountWarningThreshold
                                       << " is not recommended");
  }
  OpBuilder builder(module_.getContext());
  switch (type) {
  case DepType::IntraCore:
    module_->setAttr(CVPipeline::kIntraBufCount,
                     builder.getI32IntegerAttr(count));
    LOG_DEBUG("IntraBufferCount set to " << count);
    break;
  case DepType::InterCore:
    module_->setAttr(CVPipeline::kInterCoreBufCount,
                     builder.getI32IntegerAttr(count));
    LOG_DEBUG("InterBufferCount set to " << count);
    break;
  case DepType::LoadStore:
    module_->setAttr(CVPipeline::kLoadStoreBufCount,
                     builder.getI32IntegerAttr(count));
    LOG_DEBUG("LoadBufferCount set to " << count);
    break;
  default:
    LOG_DEBUG("Unknown DepType: " << static_cast<int>(type));
    break;
  }
}

void BufferCountManager::buildBufferCountMap(
    llvm::DenseMap<Value, std::vector<Value>> &depValueMap,
    llvm::DenseMap<Value, int> &bufferCountMap, DepType type) {
  int bufCount = getBufferCountByType(type);

  for (auto &p : depValueMap) {
    for (Value depVal : p.second) {
      if (isa<BlockArgument>(depVal) || !depVal.getDefiningOp())
        continue;
      bufferCountMap[depVal] = bufCount;
    }
  }
}

int BufferCountManager::getBufferCountByType(DepType type) const {
  auto attr = module_->getAttrOfType<IntegerAttr>(getAttrName(type));
  if (!attr) {
    int defaultCount = getBufferCountByType(type);
    LOG_DEBUG("getBufferCountByType(" << static_cast<int>(type)
                                      << ") returns default buffer count "
                                      << defaultCount);
    return defaultCount;
  }
  int count = static_cast<int>(attr.getInt());
  LOG_DEBUG("getBufferCountByType(" << static_cast<int>(type)
                                    << ") = " << count);
  return count;
}

} // namespace triton
} // namespace mlir
