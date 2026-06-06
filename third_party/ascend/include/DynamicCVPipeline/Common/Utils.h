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

#ifndef ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#define ADD_AUTO_SCHEDULING_COMMON_UTILS_H
#include <string_view>
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace CVPipeline {

inline constexpr llvm::StringLiteral kCoreType = "ssbuffer.core_type";
inline constexpr llvm::StringLiteral kBlockId = "ssbuffer.block_id";

enum CoreType {
    UNDETERMINED = 0,
    VECTOR_ONLY = 1 << 0,
    CUBE_ONLY = 1 << 1,
    CUBE_AND_VECTOR = VECTOR_ONLY | CUBE_ONLY,
};

inline constexpr CoreType fromStrCoreType(std::string_view s)
{
    if (s == "VECTOR") {
        return CoreType::VECTOR_ONLY;
    }
    if (s == "CUBE") {
        return CoreType::CUBE_ONLY;
    }

    return CoreType::UNDETERMINED;
}

// Functions for managing core types
CoreType getOpCoreType(Operation *op);

llvm::LogicalResult verifyOpBlockId(Operation *op);
std::optional<int64_t> getOpBlockId(Operation *op);

} // namespace CVPipeline
} // namespace mlir

#endif
