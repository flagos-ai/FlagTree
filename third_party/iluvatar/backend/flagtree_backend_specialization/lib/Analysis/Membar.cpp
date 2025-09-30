#include "triton/Analysis/Membar.h"

namespace mlir {

// type: 0 all | 1 del W from other R |2 del R from other W
void BlockInfo::erase(BlockInfo &other, int type) {
  if (type == 0) {
    for (auto &sri : other.syncReadIntervals)
      syncReadIntervals.erase(sri);
    for (auto &swi : other.syncWriteIntervals)
      syncWriteIntervals.erase(swi);
  } else if (type == 1) {
    for (auto &sri : other.syncReadIntervals)
      syncWriteIntervals.erase(sri);
  } else if (type == 2) {
    for (auto &swi : other.syncWriteIntervals)
      syncReadIntervals.erase(swi);
  }
}

// for debug
void BlockInfo::printIntervals() {
  if (syncReadIntervals.size() > 0 || syncWriteIntervals.size() > 0) {
    std::cout << " syncReadIntervals";
    for (auto &lhs : syncReadIntervals)
      std::cout << " [" << lhs.start() << ", " << lhs.end() << "] ";
    std::cout << "" << std::endl;
    std::cout << " syncWriteIntervals";
    for (auto &lhs : syncWriteIntervals)
      std::cout << " [" << lhs.start() << ", " << lhs.end() << "] ";
    std::cout << "" << std::endl;
  }
}

}