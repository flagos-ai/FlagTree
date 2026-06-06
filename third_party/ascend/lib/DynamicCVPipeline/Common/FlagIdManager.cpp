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

#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "flag-id-manager";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;
using namespace hivm;

static constexpr int kMaxOps = 1024;

FlagIdManager::FlagIdManager(ModuleOp module)
{
    this->module = module;
    scanExistingFlags(module);
    LOG_DEBUG("FlagIdManager: Initialized with max_id = " << currentMaxId << "\n");
}

void FlagIdManager::scanExistingFlags(ModuleOp module)
{
    module.walk([&](Operation *op) {
        if (isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) {
            int flag = -1;
            if (auto intAttr = op->getAttrOfType<IntegerAttr>("static_flag_id")) {
                flag = (int)intAttr.getInt();
            } else if (auto intAttr = op->getAttrOfType<IntegerAttr>("flag")) {
                flag = (int)intAttr.getInt();
            }
            if (flag >= 0) {
                currentMaxId = std::max(currentMaxId, (int64_t)flag);
            }
        }
    });
}

int FlagIdManager::acquireId(Operation* insertionPoint)
{
    if (currentMaxId < MAX_FLAG_ID) {
        return ++currentMaxId;
    }

    return INVALID_FLAG_ID;
}
