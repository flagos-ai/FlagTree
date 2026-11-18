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

#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace triton;

// Helper function to check if a type is a scalar type
// Scalar types include: IntegerType (i1, i8, i16, i32, i64, etc.)
// and FloatType (f16, f32, f64, bf16, f8, etc.)
bool SSBufferManager::isScalarType(Type type) {
  // Check if it's a scalar type
  if (isa<IntegerType>(type)) {
    return true;
  } else if (isa<FloatType>(type)) {
    return true;
  } else {
    // Not a scalar type
    return false;
  }
}

// Allocate SSBuffer address for a value
std::optional<int64_t> SSBufferManager::allocateAddr(Value value) {
  // Check if the value's type is a scalar type
  Type valueType = value.getType();
  if (!isScalarType(valueType)) {
    // Not a scalar type, return error
    return std::nullopt;
  }

  // Check if this value already has an allocated SSBuffer address
  auto it = valueToAddrMap.find(value);
  if (it != valueToAddrMap.end()) {
    // Reuse the existing SSBuffer address
    return it->second;
  }

  // Allocate a new SSBuffer address based on current map size
  // Address = base_addr + map_size * offset
  int64_t addrValue =
      SSBUF_BASE_ADDR + valueToAddrMap.size() * SSBUF_ADDR_OFFSET;

  // Check if address exceeds maximum limit
  if (addrValue > SSBUF_ADDR_MAX) {
    // Address out of range, return error
    return std::nullopt;
  }

  valueToAddrMap[value] = addrValue;
  addrToValueMap[addrValue] = value;
  return addrValue;
}

// Find the Value and its type for a given address
std::optional<std::pair<Value, Type>>
SSBufferManager::findValueByAddr(int64_t addr) {
  // Use reverse mapping table for fast lookup
  auto it = addrToValueMap.find(addr);
  if (it == addrToValueMap.end()) {
    // Address not found in mapping table
    return std::nullopt;
  }

  // Found the address in the reverse mapping table
  Value foundValue = it->second;
  Type dataType = foundValue.getType();
  return std::make_pair(foundValue, dataType);
}

// Write a value to SSBuffer and return the SSBuffer address (int64_t)
std::optional<int64_t>
SSBufferManager::writeToSSBuffer(Value value, OpBuilder &builder,
                                 SmallVectorImpl<Operation *> &createdOps) {
  // Allocate or get existing SSBuffer address
  auto addrResult = allocateAddr(value);
  if (!addrResult) {
    // Address allocation failed (either not scalar type or out of range)
    return std::nullopt;
  }

  int64_t addrValue = addrResult.value();
  Location loc = builder.getUnknownLoc();
  auto [constOp, pointerCastOp] =
      getSsbufConstAndPointerCast(builder, loc, addrValue, value.getType());
  createdOps.push_back(constOp);
  createdOps.push_back(pointerCastOp);

  // Store the value to SSBuffer
  auto storeOp =
      builder.create<memref::StoreOp>(loc, value, pointerCastOp.getResult());
  createdOps.push_back(storeOp); // Record the created operation

  // Return the address value (int64_t), not the pointer
  return addrValue;
}

// Read a value from SSBuffer based on the given address (int64_t)
std::optional<Value>
SSBufferManager::readFromSSBuffer(int64_t addr, OpBuilder &builder,
                                  SmallVectorImpl<Operation *> &createdOps) {
  // Find the Value and its type for the given address
  auto findResult = findValueByAddr(addr);
  if (!findResult) {
    // Address not found in manager, return error
    return std::nullopt;
  }

  // Load the value back with the original scalar type, not a hardcoded i32.
  Type dataType = findResult.value().second;
  Location loc = builder.getUnknownLoc();
  auto [constOp, pointerCastOp] =
      getSsbufConstAndPointerCast(builder, loc, addr, dataType);
  createdOps.push_back(constOp);
  createdOps.push_back(pointerCastOp);

  // Load the value from SSBuffer address with the retrieved data type
  auto loadOp = builder.create<memref::LoadOp>(loc, pointerCastOp.getResult(),
                                               ValueRange{});
  createdOps.push_back(loadOp); // Record the created operation

  auto markOp = builder.create<annotation::MarkOp>(loc, loadOp.getResult());
  markOp->setAttr(kMemrefExtVolatile, builder.getUnitAttr());
  createdOps.push_back(markOp);

  return loadOp.getResult();
}
