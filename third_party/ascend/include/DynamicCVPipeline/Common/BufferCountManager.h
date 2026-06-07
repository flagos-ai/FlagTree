#ifndef TRITON_DYNAMIC_CV_PIPELINE_ADDMULTIBUFFERCONTROL_BUFFER_COUNT_MANAGER_H
#define TRITON_DYNAMIC_CV_PIPELINE_ADDMULTIBUFFERCONTROL_BUFFER_COUNT_MANAGER_H

#include <vector>
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"

namespace mlir {

class Operation;

namespace triton {

class BufferCountManager {
public:
    static BufferCountManager& getInstance();

    enum class DepType { IntraCore, InterCore, LoadStore };

    void setBufferCount(DepType type, int count);

    void buildBufferCountMap(
        llvm::DenseMap<Value, std::vector<Value>> &depValueMap,
        llvm::DenseMap<Value, int> &bufferCountMap,
        DepType type);

    int getBufferCountByType(DepType type) const;

private:
    BufferCountManager();
    BufferCountManager(const BufferCountManager&) = delete;
    BufferCountManager& operator=(const BufferCountManager&) = delete;

    int intraBufferCount_;
    int interCoreBufferCount_;
    int loadStoreBufferCount_;
};

#define BUFFER_COUNT (BufferCountManager::getInstance())

} // namespace triton
} // namespace mlir

#endif // TRITON_DYNAMIC_CV_PIPELINE_ADDMULTIBUFFERCONTROL_BUFFER_COUNT_MANAGER_H