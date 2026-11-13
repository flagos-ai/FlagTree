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

#ifndef TRITON_TO_CFG_TENSOR_H
#define TRITON_TO_CFG_TENSOR_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace triton {
namespace cfg {

// Compute类型（用于指令分类）
enum class ComputeType {
  CUBE,   // shape为2维
  VECTOR, // shape为1维
  SCALAR  // 其他标量
};

// TensorObject - Tensor对象定义
class TensorObject {
public:
  // Tensor类型分类
  enum class TensorKind {
    GLOBAL_MEMORY, // 全局内存tensor（如gm_obj）
    L2,            // L2 Cache
    L1,            // L1 Cache
    UB             // Unified Buffer
  };

  // 构造函数
  TensorObject(StringRef name, ArrayRef<int64_t> shape, Type type,
               Type elementType, TensorKind kind = TensorKind::GLOBAL_MEMORY)
      : name(name.str()), shape(shape.begin(), shape.end()), type(type),
        elementType(elementType), kind(kind) {}

  // 获取tensor名称
  const std::string &getName() const { return name; }

  // 设置tensor名称
  void setName(StringRef newName) { name = newName.str(); }

  // 获取shape
  ArrayRef<int64_t> getShape() const { return shape; }

  // 获取MLIR类型
  Type getType() const { return type; }

  // 获取tensor种类
  TensorKind getKind() const { return kind; }

  // 设置tensor种类
  void setKind(TensorKind newKind) { kind = newKind; }

  // 获取元素数据类型
  Type getElementType() const { return elementType; }

  // 设置元素数据类型
  void setElementType(Type newElementType) { elementType = newElementType; }

  // 获取维度数
  size_t getRank() const { return shape.size(); }

  // 获取tensor大小（元素总数）
  int64_t getSize() const {
    int64_t size = 1;
    for (int64_t dim : shape) {
      size *= dim;
    }
    return size;
  }

  // 判断是否是一维tensor
  bool isVector() const { return shape.size() == 1; }

  // 判断是否是二维tensor
  bool isMatrix() const { return shape.size() == 2; }

  // 判断是否是标量
  bool isScalar() const { return shape.empty(); }

  // 打印信息
  void print(llvm::raw_ostream &os) const {
    os << "Tensor[" << name << ", shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << shape[i];
    }
    os << "], element=" << elementType << ", kind=" << getKindString() << "]";
  }

  // 获取kind的字符串表示
  std::string getKindString() const {
    switch (kind) {
    case TensorKind::GLOBAL_MEMORY:
      return "GLOBAL_MEMORY";
    case TensorKind::L2:
      return "L2";
    case TensorKind::L1:
      return "L1";
    case TensorKind::UB:
      return "UB";
    }
    return "UNKNOWN";
  }

private:
  std::string name; // Tensor名称，如"gm_obj_0"
  SmallVector<int64_t> shape;
  Type type;        // 完整类型（如tensor<64x64xf32>）
  Type elementType; // 元素数据类型（如f32, i8, f16等）
  TensorKind kind;
};

// 从类型中提取shape和element type
inline void extractShapeAndElementType(Type type,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType) {
  if (auto rankedType = mlir::dyn_cast<RankedTensorType>(type)) {
    shape.append(rankedType.getShape().begin(), rankedType.getShape().end());
    elementType = rankedType.getElementType();
  } else if (auto ptrType = mlir::dyn_cast<triton::PointerType>(type)) {
    Type pointeeType = ptrType.getPointeeType();
    if (auto rankedType = mlir::dyn_cast<RankedTensorType>(pointeeType)) {
      shape.append(rankedType.getShape().begin(), rankedType.getShape().end());
      elementType = rankedType.getElementType();
    } else {
      // 标量指针
      elementType = pointeeType;
    }
  } else {
    // 默认值
    elementType = type;
  }
}

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_TENSOR_H
