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

#ifndef TRITON_TO_CFG_MEMORY_SSA_H
#define TRITON_TO_CFG_MEMORY_SSA_H

#include "TritonToGraph/tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace cfg {

// Forward declarations
class MemorySSADef;
class MemorySSAUse;

class BasicBlock;
class ControlFlowGraph;

// Memory SSA Definition - 表示tensor/pointer的定义
class MemorySSADef {
public:
  // 构造函数
  MemorySSADef(TensorObject *tensor, Operation *defOp, unsigned version = 0)
      : tensor(tensor), defOp(defOp), version(version) {}

  // 获取tensor对象
  TensorObject *getTensor() const { return tensor; }

  // 获取创建该definition的操作
  // - 对于入参，返回nullptr
  // - 对于其他操作，返回对应的Operation指针
  Operation *getDefOp() const { return defOp; }

  // 获取版本号（函数内全局唯一序数）
  unsigned getVersion() const { return version; }

  // 获取唯一标识: tensor_name,version 或 tensor_name,param
  std::string getId() const {
    return tensor->getName() + "," +
           (defOp ? std::to_string(version) : std::to_string(0));
  }

  // 判断是否是入参（函数参数）
  bool isParameter() const { return defOp == nullptr; }

  // 判断是否是phi节点（控制流合并）
  bool isPhi() const {
    return defOp && (isa<scf::IfOp>(defOp) || isa<scf::ForOp>(defOp) ||
                     isa<scf::WhileOp>(defOp));
  }

  // 打印信息
  void print(llvm::raw_ostream &os) const {
    os << "Definition[" << getId() << ", tensor=" << tensor->getName();
    if (defOp) {
      os << ", op=" << defOp->getName();
    } else {
      os << ", param";
    }
    os << "]";
  }

private:
  TensorObject *tensor; // 对应的tensor对象
  Operation *defOp;     // 创建该definition的操作
  unsigned version;     // 版本号（函数内全局递增）
};

// Memory SSA Use - 表示tensor/pointer的使用
class MemorySSAUse {
public:
  // 构造函数
  MemorySSAUse(MemorySSADef *definition, Operation *userOp, unsigned operandIdx)
      : definition(definition), userOp(userOp), operandIdx(operandIdx) {
    // 缓存value以提高查询性能
    operandValue = userOp->getOperand(operandIdx);
  }

  // 获取使用的definition
  MemorySSADef *getDefinition() const { return definition; }

  // 获取使用该definition的操作
  Operation *getUserOp() const { return userOp; }

  // 获取operand序号
  unsigned getOperandIdx() const { return operandIdx; }

  // 获取Value（从userOp的operand）
  Value getValue() const { return operandValue; }

  // 获取用户操作的名称
  std::string getUserOpName() const {
    return userOp->getName().getStringRef().str();
  }

  // 打印信息
  void print(llvm::raw_ostream &os) const {
    os << "Use[";
    if (definition) {
      os << definition->getId();
    } else {
      os << "null";
    }
    os << " in " << userOp->getName() << ", operand #" << operandIdx << "]";
  }

private:
  MemorySSADef *definition; // 使用的definition
  Operation *userOp;        // 使用该definition的操作
  unsigned operandIdx;      // operand序号
  Value operandValue;       // 缓存的operand value
};

// PhiInfo - 循环Phi信息
struct PhiInfo {
  // Phi类型
  enum Type {
    ITER_ARG,  // scf.for的iter_arg
    IF_RESULT, // scf.if的result
    WHILE_ARG  // scf.while的arg
  };

  Type type;
  BasicBlock *loopHeader; // 循环头基本块

  // Phi值的来源
  struct {
    MemorySSADef *initialValue; // 初始值（初始iteration）
    MemorySSADef *yieldValue;   // yield的值（后续iteration）
  } comingFrom;

  // 是否是第一次迭代
  bool isInitial() const { return comingFrom.yieldValue == nullptr; }

  // 获取当前definition（根据上下文决定）
  MemorySSADef *getCurrentDefinition(int iteration) const {
    return (iteration == 0) ? comingFrom.initialValue : comingFrom.yieldValue;
  }
};

// MemorySSAInfo - 指令的Memory SSA信息
struct MemorySSAInfo {
  // 指令的operands使用的definitions
  SmallVector<MemorySSAUse> uses;

  // 指令的results创建的definitions
  SmallVector<MemorySSADef *> definitions;

  // Alias信息（仅对pointer相关操作）
  struct AliasInfo {
    Value aliasee;            // 别名的源value
    TensorObject *baseTensor; // 对应的tensor对象
  };
  std::optional<AliasInfo> aliasInfo;

  // 快速查询接口
  bool hasDefinition(Value value) const {
    // 通过 getDefiningOp() 获取定义该 value 的操作
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return false;

    // 获取该操作的所有 results
    auto results = defOp->getResults();
    for (size_t i = 0; i < definitions.size() && i < results.size(); ++i) {
      if (results[i] == value) {
        return definitions[i] != nullptr;
      }
    }
    return false;
  }

  MemorySSADef *getDefinition(Value value) const {
    // 查找该value在results中的索引
    for (auto result : llvm::enumerate(value.getDefiningOp()->getResults())) {
      if (result.value() == value) {
        size_t idx = result.index();
        if (idx < definitions.size()) {
          return definitions[idx];
        }
        break;
      }
    }
    return nullptr;
  }

  bool hasUse(Value value) const {
    for (const MemorySSAUse &use : uses) {
      if (use.getValue() == value) {
        return true;
      }
    }
    return false;
  }

  SmallVector<MemorySSAUse> getUses(Value value) const {
    SmallVector<MemorySSAUse> result;
    for (const MemorySSAUse &use : uses) {
      if (use.getValue() == value) {
        result.push_back(use);
      }
    }
    return result;
  }

  // 判断是否创建了新的definition
  bool hasNewDefinitions() const { return !definitions.empty(); }

  // 判断是否是写入操作
  bool isMemoryWriter() const {
    for (MemorySSADef *def : definitions) {
      if (def && !def->isParameter()) {
        return true;
      }
    }
    return false;
  }

  // 遍历定义
  void forEachDefinition(llvm::function_ref<void(MemorySSADef *)> func) const {
    for (MemorySSADef *def : definitions) {
      if (def)
        func(def);
    }
  }

  void forEachUse(llvm::function_ref<void(const MemorySSAUse &)> func) const {
    for (const MemorySSAUse &use : uses) {
      if (use.getDefinition())
        func(use);
    }
  }

  // 清空所有信息
  void clear() {
    uses.clear();
    definitions.clear();
    aliasInfo.reset();
  }

  // 打印信息
  void print(llvm::raw_ostream &os) const {
    os << "MemorySSAInfo[\n";
    os << "  Uses: " << uses.size() << "\n";
    for (const MemorySSAUse &use : uses) {
      os << "    ";
      use.print(os);
      os << "\n";
    }
    os << "  Definitions: " << definitions.size() << "\n";
    for (MemorySSADef *def : definitions) {
      if (def) {
        os << "    ";
        def->print(os);
        os << "\n";
      }
    }
    if (aliasInfo) {
      os << "  Alias: " << aliasInfo->aliasee << " -> "
         << aliasInfo->baseTensor->getName() << "\n";
    }
    os << "]";
  }
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_CFG_MEMORY_SSA_H
