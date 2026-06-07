#ifndef AffinityDAGDEF
#define AffinityDAGDEF
#include "Utils.hpp"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <sys/types.h>
#include <type_traits>

namespace mlir {
namespace AffinityDAG {

enum class OpAbility {
  PREFER_VECTOR = 1 << 0,
  CUBE_ONLY = 1 << 1,
  CUBE_AND_VECTOR = PREFER_VECTOR | CUBE_ONLY

};

enum CoreType {
  UNDETERMINED = 0,
  VECTOR_ONLY = 1 << 0,
  CUBE_ONLY = 1 << 1,
  CUBE_AND_VECTOR = VECTOR_ONLY | CUBE_ONLY
};

inline constexpr CoreType toCoreType(OpAbility ct) {
  using U = std::underlying_type_t<OpAbility>;
  return static_cast<CoreType>(static_cast<U>(ct));
}

constexpr inline CoreType operator|(CoreType lhs, CoreType rhs) {
  return enumOp(std::bit_or<>(), lhs, rhs);
}

inline CoreType operator&(CoreType lhs, CoreType rhs) {
  return enumOp(std::bit_and<>(), lhs, rhs);
}

inline bool intersects(CoreType lhs, CoreType rhs) {
  return (lhs & rhs) != CoreType::UNDETERMINED;
}

inline CoreType operator&(OpAbility lhs, CoreType rhs) {
  return toCoreType(lhs) & rhs;
}

inline CoreType operator!(CoreType ct) {
  CoreType newCt = UNDETERMINED;
  if ((ct & CoreType::CUBE_ONLY) == UNDETERMINED) {
    newCt = newCt | CoreType::CUBE_ONLY;
  }

  if ((ct & CoreType::VECTOR_ONLY) == UNDETERMINED) {
    newCt = newCt | CoreType::VECTOR_ONLY;
  }

  return newCt;
}

inline hivm::TCoreType toHivm(CoreType ct) {
  switch (ct) {
  case UNDETERMINED:
    return hivm::TCoreType::CUBE_OR_VECTOR;
  case CUBE_ONLY:
    return hivm::TCoreType::CUBE;
  case VECTOR_ONLY:
    return hivm::TCoreType::VECTOR;
  case CUBE_AND_VECTOR:
    return hivm::TCoreType::CUBE_AND_VECTOR;
  default:
    llvm_unreachable("Invalid CoreType that cannot convert to hivm");
  }
}

inline bool intersects(OpAbility lhs, CoreType rhs) {
  return (lhs & rhs) != CoreType::UNDETERMINED;
}

inline bool exactlyOneType(CoreType ct) {
  return (ct == CUBE_ONLY) || (ct == VECTOR_ONLY);
}

const char *literalCoreType(CoreType ct);

class MoveOnly {
protected:
  MoveOnly() = default;
  ~MoveOnly() = default;

  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;

  MoveOnly(MoveOnly &&) = default;
  MoveOnly &operator=(MoveOnly &&) = default;
};

class Node;
class OpNode;
class ValueNode;

ValueNode *getDataSource(OpNode *op);

class Graph : MoveOnly {
public:
  using OpMapRaw = llvm::DenseMap<Operation *, std::unique_ptr<OpNode>>;
  using ValueMapRaw = llvm::DenseMap<Value, std::unique_ptr<ValueNode>>;
  using OpMap = std::shared_ptr<OpMapRaw>;
  using ValueMap = std::shared_ptr<ValueMapRaw>;

  Graph(Block *block, Graph *parent = nullptr, OpMap opMap = nullptr,
        ValueMap valueMap = nullptr, bool inheritParent = true);

  static std::unique_ptr<Graph> fromMultiBlockFunc(triton::FuncOp funcOp);

  OpMapRaw &getOpMap() const { return *opMap; }

  ValueMapRaw &getValueMap() const { return *valueMap; }

  // [DEBUG] start
  std::unique_ptr<llvm::DenseMap<Operation *, OpNode *>> legacyOpMap = nullptr;
  std::unique_ptr<llvm::DenseMap<Value, CoreType>> legacyValueTypes = nullptr;

  inline llvm::DenseMap<Operation *, OpNode *> &getOpMapLegacy() {
    if (!legacyOpMap) {
      legacyOpMap =
          std::move(std::make_unique<llvm::DenseMap<Operation *, OpNode *>>());
      for (auto &[key, val] : *opMap) {
        (*legacyOpMap)[key] = val.get();
      }
    }

    return *legacyOpMap;
  }

  llvm::DenseMap<Value, CoreType> &getValueTypes();

  // [DEBUG] end

private:
  friend class Node;
  friend class OpNode;
  OpMap opMap;
  ValueMap valueMap;
  Block *block;
  Graph *parent;
  OpNode *terminator = nullptr;
  size_t opCount = 0;
  llvm::SmallVector<ValueNode *, 4> blockArgs;
};

class Node : MoveOnly {
protected:
  friend class Graph;
  friend class ValueNode;
  bool isUpstreamOfCubeMem = false;
  virtual CoreType absorbImpl() = 0;
  llvm::SmallVector<Node *, 4> outputs;

public:
  CoreType isOnPrivate = UNDETERMINED;

  enum NodeKind { NK_Op, NK_Value };

  inline CoreType isOn() const { return isOnPrivate; }

  bool absorb() {
    auto newCoreType = absorbImpl();
    auto changed = newCoreType != isOnPrivate;
    isOnPrivate = newCoreType;

    return changed;
  };

  virtual llvm::SmallVector<Node *, 4> getAffected() const = 0;
  virtual OpNode *getSourceOpNode() = 0;

  ArrayRef<Node *> getOutputs() const { return outputs; }

  CoreType absorbCommon();

private:
  const NodeKind kind;

public:
  NodeKind getKind() const { return kind; }

protected:
  Node(NodeKind kind) : kind(kind) {}
};

class OpNode : public Node {
  friend class Graph;
  friend class ValueNode;
  llvm::SmallVector<ValueNode *, 4> inputs;
  llvm::SmallVector<Graph, 2> subgraphs;
  virtual CoreType absorbImpl() override;

public:
  Operation *op;

  OpNode(Operation *op, Graph *graph);
  OpAbility canRunOn() const;
  inline ArrayRef<ValueNode *> getInputs() const { return inputs; }

  static bool classof(const Node *node) { return node->getKind() == NK_Op; }

  virtual llvm::SmallVector<Node *, 4> getAffected() const override {
    llvm::SmallVector<Node *, 4> result(inputs.begin(), inputs.end());
    result.append(outputs.begin(), outputs.end());

    return result;
  }

  virtual OpNode *getSourceOpNode() override { return this; }
};

class ValueNode : public Node {
  friend class Graph;
  friend class OpNode;
  virtual CoreType absorbImpl() override;

public:
  Node *source = nullptr;
  Value value;
  // ValueNode(OpResult value);
  // ValueNode(BlockArgument value);

  ValueNode(Value value) : Node(NK_Value), value(value) {};
  virtual OpNode *getSourceOpNode() override {
    if (!source) {
      return nullptr;
    }

    return source->getSourceOpNode();
  }
  static bool classof(const Node *node) { return node->getKind() == NK_Value; }

  virtual llvm::SmallVector<Node *, 4> getAffected() const override {
    llvm::SmallVector<Node *, 4> result(outputs.begin(), outputs.end());
    if (source)
      result.push_back(source);

    return result;
  }
};

class GraphManager {
private:
  llvm::DenseMap<llvm::StringRef, std::shared_ptr<AffinityDAG::Graph>> graphs;

public:
  static GraphManager &getInstance() {
    static GraphManager instance;
    return instance;
  }

  void registerGraph(llvm::StringRef funcName,
                     std::shared_ptr<AffinityDAG::Graph> graph) {
    graphs[funcName] = graph;
  }

  AffinityDAG::Graph *getGraph(llvm::StringRef funcName) {
    auto it = graphs.find(funcName);
    return it != graphs.end() ? it->second.get() : nullptr;
  }

  void removeGraph(llvm::StringRef funcName) { graphs.erase(funcName); }
};

inline llvm::DenseMap<Value, CoreType> &Graph::getValueTypes() {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  if (!legacyValueTypes) {
    legacyValueTypes =
        std::move(std::make_unique<llvm::DenseMap<Value, CoreType>>());
    for (auto &[key, val] : *valueMap) {
      llvm::dbgs() << key << "\n";
      llvm::dbgs().flush();
      (*legacyValueTypes)[key] = val.get()->isOn();
    }
  }

  return *legacyValueTypes;
}

} // namespace AffinityDAG
} // namespace mlir
#endif
