#ifndef TRITON_TOOLS_SWIZZLEDEBUGDUMP_H_
#define TRITON_TOOLS_SWIZZLEDEBUGDUMP_H_

#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace mlir::triton {

// One row in the exported access table.
// Each row represents one load/store access issued by one logical thread.
struct AccessRecord {
  std::string op;
  int warpId = 0;
  int laneId = 0;
  int regId = 0;
  int threadId = 0;
  int accessOrd = 0;

  int vec = 0;
  int bank = 0;
  int segment = 0;

  // Logical coordinates produced by the optional logical layout.
  std::vector<int> logicalCoords;
};

// Enumerate all accesses in a layout and convert them into CSV-friendly rows.
// The input space is assumed to be based on register/lane/warp dimensions.
std::vector<AccessRecord>
enumerateLayoutAccesses(llvm::StringRef opName, const LinearLayout &smemLayout,
                        const LinearLayout *logicalLayout = nullptr);

// Write access records to a CSV file.
void writeAccessRecordsCsv(llvm::StringRef filePath,
                           llvm::ArrayRef<AccessRecord> rows, int logicalRank);

// Dump both store and load access tables to one CSV file.
// If both logical layouts are provided, this function validates that the
// logical coordinates attached to the same (vec, bank, segment) are identical
// between store and load.
void dumpSwizzleAccessCsv(llvm::StringRef accessCsvPath,
                          const LinearLayout &stsToSmem,
                          const LinearLayout &ldsToSmem,
                          const LinearLayout *stsLogicalLayout = nullptr,
                          const LinearLayout *ldsLogicalLayout = nullptr);

} // namespace mlir::triton

#endif // TRITON_TOOLS_SWIZZLEDEBUGDUMP_H_
