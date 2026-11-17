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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SSBUFFER_MANAGER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SSBUFFER_MANAGER_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace mlir {
namespace triton {

static constexpr llvm::StringRef kMemrefExtVolatile = "memref_ext.volatile";

// SSBuffer Manager for managing SSBuffer address allocation and type tracking
// Purpose: Globally manage SSBuffer addresses across the entire pass pipeline
// This class maintains a single mapping table: Value -> address (int64_t)
// Type information is retrieved from the Value itself
class SSBufferManager {
public:
  // SSBuffer address space and constants
  static constexpr int ADDR_INT_TYPE = 64;
  static constexpr int SSBUF_BASE_ADDR = 2048; // Base address for SSBuffer
  static constexpr int SSBUF_ADDR_OFFSET =
      8; // Address offset for each allocation
  static constexpr int SSBUF_ADDR_MAX = 6072; // Maximum allowed address

  // Constructor
  SSBufferManager() = default;

  std::optional<int64_t> allocateAddr(Value value);

  std::optional<std::pair<Value, Type>> findValueByAddr(int64_t addr);

  std::optional<int64_t>
  writeToSSBuffer(Value value, OpBuilder &builder,
                  SmallVectorImpl<Operation *> &createdOps);

  std::optional<Value>
  readFromSSBuffer(int64_t addr, OpBuilder &builder,
                   SmallVectorImpl<Operation *> &createdOps);

  // Get the number of allocated addresses
  size_t getAllocatedCount() const { return valueToAddrMap.size(); }

  // Clear all mappings (for testing or reset)
  void clear() {
    valueToAddrMap.clear();
    addrToValueMap.clear();
  }

private:
  static bool isScalarType(Type type);

  // Forward mapping: Value -> address (int64_t)
  // Used for address allocation and reuse
  llvm::DenseMap<Value, int64_t> valueToAddrMap;

  // Reverse mapping: address (int64_t) -> Value
  // Used for fast lookup when reading from SSBuffer
  // This avoids O(n) traversal in findValueByAddr
  llvm::DenseMap<int64_t, Value> addrToValueMap;
};

static constexpr int ADDR_INT_TYPE = 64;
static constexpr int CONST_INT_TYPE = 32;

inline MemRefType getSsbufMemrefType(Builder &builder) {
  auto i32Type = builder.getIntegerType(CONST_INT_TYPE);
  auto addressSpaceAttr =
      builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::SSBUF);
  return MemRefType::get({}, i32Type, nullptr, addressSpaceAttr);
}

inline std::pair<arith::ConstantOp, hivm::PointerCastOp>
getSsbufConstAndPointerCast(OpBuilder &builder, Location loc, uint64_t addr,
                            Type elemType) {
  auto i64Type = builder.getIntegerType(ADDR_INT_TYPE);
  auto addrAttr = builder.getIntegerAttr(i64Type, addr);
  auto addrConst = builder.create<arith::ConstantOp>(loc, i64Type, addrAttr);
  auto addressSpaceAttr =
      builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::SSBUF);
  auto memrefType = MemRefType::get({}, elemType, nullptr, addressSpaceAttr);

  return {addrConst, builder.create<hivm::PointerCastOp>(
                         loc, memrefType, addrConst.getResult())};
}

inline hivm::PointerCastOp createPointerCastOp(OpBuilder &builder, Location loc,
                                               uint64_t addr) {
  return getSsbufConstAndPointerCast(builder, loc, addr,
                                     builder.getIntegerType(CONST_INT_TYPE))
      .second;
}

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SSBUFFER_MANAGER_H
