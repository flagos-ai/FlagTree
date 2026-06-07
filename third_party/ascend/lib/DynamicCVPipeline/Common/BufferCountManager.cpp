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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

static constexpr const char *DEBUG_TYPE = "BufferCountManager";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

constexpr int kDefaultIntraBufferCount = 2;
constexpr int kDefaultInterBufferCount = 1;
constexpr int kDefaultLoadBufferCount = 2;
constexpr int kBufferCountWarningThreshold = 3;

namespace mlir {
namespace triton {

BufferCountManager& BufferCountManager::getInstance()
{
    static BufferCountManager instance;
    return instance;
}

BufferCountManager::BufferCountManager()
    : intraBufferCount_(kDefaultIntraBufferCount),
      interCoreBufferCount_(kDefaultInterBufferCount),
      loadStoreBufferCount_(kDefaultLoadBufferCount)
{
    LOG_DEBUG("Default initialized: "
          << "IntraBufferCount=" << intraBufferCount_
          << ", InterBufferCount=" << interCoreBufferCount_
          << ", LoadBufferCount=" << loadStoreBufferCount_);
}

void BufferCountManager::setBufferCount(DepType type, int count)
{
    if (count <= 0) {
        LOG_DEBUG("Invalid buffer count: " << count << " (must be > 0)");
        return;
    }
    if (count >= kBufferCountWarningThreshold) {
        LOG_DEBUG("Warning: buffer count " << count << " >= "
              << kBufferCountWarningThreshold << " is not recommended");
    }
    switch (type) {
        case DepType::IntraCore:
            intraBufferCount_ = count;
            LOG_DEBUG("IntraBufferCount set to " << count);
            break;
        case DepType::InterCore:
            interCoreBufferCount_ = count;
            LOG_DEBUG("InterBufferCount set to " << count);
            break;
        case DepType::LoadStore:
            loadStoreBufferCount_ = count;
            LOG_DEBUG("LoadBufferCount set to " << count);
            break;
        default:
            LOG_DEBUG("Unknown DepType: " << static_cast<int>(type));
            break;
    }
}

void BufferCountManager::buildBufferCountMap(
    llvm::DenseMap<Value, std::vector<Value>> &depValueMap,
    llvm::DenseMap<Value, int> &bufferCountMap,
    DepType type)
{
    int bufCount = getBufferCountByType(type);

    for (auto &p : depValueMap) {
        for (Value depVal : p.second) {
            if (isa<BlockArgument>(depVal) || !depVal.getDefiningOp())
                continue;
            bufferCountMap[depVal] = bufCount;
        }
    }
}

int BufferCountManager::getBufferCountByType(DepType type) const
{
    int count = 1;
    switch (type) {
        case DepType::IntraCore:
            count = intraBufferCount_;
            break;
        case DepType::InterCore:
            count = interCoreBufferCount_;
            break;
        case DepType::LoadStore:
            count = loadStoreBufferCount_;
            break;
        default:
            LOG_DEBUG("Unknown DepType: " << static_cast<int>(type));
            break;
    }
    LOG_DEBUG("getBufferCountByType(" << static_cast<int>(type) << ") = " << count
          << " (IntraCore=0, InterCore=1, LoadStore=2)");
    return count;
}

} // namespace triton
} // namespace mlir