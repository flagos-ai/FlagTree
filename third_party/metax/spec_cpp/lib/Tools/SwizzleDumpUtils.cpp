#include "triton/Tools/SwizzleDumpUtils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <tuple>
#include <utility>

namespace mlir::triton {
namespace {

int getDimValue(llvm::ArrayRef<std::pair<StringAttr, int32_t>> vals,
                StringAttr dim, int defaultValue = 0) {
  for (auto [d, v] : vals) {
    if (d == dim)
      return v;
  }
  return defaultValue;
}

llvm::SmallVector<std::pair<StringAttr, int32_t>>
buildInputPoint(const LinearLayout &layout, int regId, int laneId, int warpId) {
  auto *ctx = layout.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");

  llvm::SmallVector<std::pair<StringAttr, int32_t>> ins;
  for (StringAttr inDim : layout.getInDimNames()) {
    if (inDim == kReg) {
      ins.push_back({inDim, regId});
    } else if (inDim == kLane) {
      ins.push_back({inDim, laneId});
    } else if (inDim == kWarp) {
      ins.push_back({inDim, warpId});
    } else {
      ins.push_back({inDim, 0});
    }
  }
  return ins;
}

std::string logicalCoordsToString(llvm::ArrayRef<int> coords) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "(";
  for (size_t i = 0; i < coords.size(); ++i) {
    os << coords[i];
    if (i + 1 != coords.size())
      os << ",";
  }
  os << ")";
  return os.str();
}

void validateSmemLogicalConsistency(llvm::ArrayRef<AccessRecord> stsRows,
                                    llvm::ArrayRef<AccessRecord> ldsRows) {
  using SmemKey = std::tuple<int, int, int>;

  std::map<SmemKey, std::vector<int>> stsMap;
  std::map<SmemKey, std::vector<int>> ldsMap;

  auto buildMap = [](llvm::ArrayRef<AccessRecord> rows,
                     std::map<SmemKey, std::vector<int>> &outMap,
                     llvm::StringRef tag) {
    for (const auto &r : rows) {
      SmemKey key{r.vec, r.bank, r.segment};
      std::vector<int> coords(r.logicalCoords.begin(), r.logicalCoords.end());

      auto it = outMap.find(key);
      if (it == outMap.end()) {
        outMap.emplace(key, coords);
      } else if (it->second != coords) {
        llvm::errs() << "[swizzle-csv] inconsistent logical coords within "
                     << tag << " for (vec=" << r.vec << ", bank=" << r.bank
                     << ", segment=" << r.segment
                     << "): existing=" << logicalCoordsToString(it->second)
                     << ", new=" << logicalCoordsToString(coords) << "\n";
        llvm_unreachable("inconsistent logical coords within one op");
      }
    }
  };

  buildMap(stsRows, stsMap, "sts");
  buildMap(ldsRows, ldsMap, "lds");

  for (const auto &[key, stsCoords] : stsMap) {
    auto found = ldsMap.find(key);
    if (found == ldsMap.end()) {
      auto [vec, bank, segment] = key;
      llvm::errs() << "[swizzle-csv] smem position missing in lds: (vec=" << vec
                   << ", bank=" << bank << ", segment=" << segment << ")\n";
      llvm_unreachable("lds missing smem position");
    }
    if (found->second != stsCoords) {
      auto [vec, bank, segment] = key;
      llvm::errs() << "[swizzle-csv] sts/lds logical coords mismatch for "
                   << "(vec=" << vec << ", bank=" << bank
                   << ", segment=" << segment
                   << "): sts=" << logicalCoordsToString(stsCoords)
                   << ", lds=" << logicalCoordsToString(found->second) << "\n";
      llvm_unreachable("sts/lds logical coords mismatch");
    }
  }

  for (const auto &[key, _] : ldsMap) {
    if (!stsMap.count(key)) {
      auto [vec, bank, segment] = key;
      llvm::errs() << "[swizzle-csv] smem position missing in sts: (vec=" << vec
                   << ", bank=" << bank << ", segment=" << segment << ")\n";
      llvm_unreachable("sts missing smem position");
    }
  }
}

} // namespace

std::vector<AccessRecord>
enumerateLayoutAccesses(llvm::StringRef opName, const LinearLayout &smemLayout,
                        const LinearLayout *logicalLayout) {
  auto *ctx = smemLayout.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");

  auto kVec = StringAttr::get(ctx, "vec");
  auto kBank = StringAttr::get(ctx, "bank");
  auto kSegment = StringAttr::get(ctx, "segment");

  int numRegs = smemLayout.hasInDim(kReg) ? smemLayout.getInDimSize(kReg) : 1;
  int numLanes =
      smemLayout.hasInDim(kLane) ? smemLayout.getInDimSize(kLane) : 1;
  int numWarps =
      smemLayout.hasInDim(kWarp) ? smemLayout.getInDimSize(kWarp) : 1;

  std::vector<AccessRecord> rows;
  rows.reserve(numRegs * numLanes * numWarps);

  for (int warp = 0; warp < numWarps; ++warp) {
    for (int lane = 0; lane < numLanes; ++lane) {
      for (int reg = 0; reg < numRegs; ++reg) {
        auto ins = buildInputPoint(smemLayout, reg, lane, warp);
        auto smemOuts = smemLayout.apply(ins);

        AccessRecord r;
        r.op = std::string(opName);
        r.warpId = warp;
        r.laneId = lane;
        r.regId = reg;
        r.threadId = warp * numLanes + lane;

        r.vec = getDimValue(smemOuts, kVec, 0);
        r.bank = getDimValue(smemOuts, kBank, 0);
        r.segment = getDimValue(smemOuts, kSegment, 0);

        if (logicalLayout) {
          auto logicalOuts = logicalLayout->apply(ins);
          for (auto [dim, value] : logicalOuts)
            r.logicalCoords.push_back(value);
        }

        rows.push_back(std::move(r));
      }
    }
  }

  std::stable_sort(rows.begin(), rows.end(),
                   [](const AccessRecord &a, const AccessRecord &b) {
                     return std::tie(a.op, a.threadId, a.regId) <
                            std::tie(b.op, b.threadId, b.regId);
                   });

  std::map<std::pair<std::string, int>, int> nextOrd;
  for (auto &r : rows) {
    auto key = std::make_pair(r.op, r.threadId);
    r.accessOrd = nextOrd[key]++;
  }

  return rows;
}

void writeAccessRecordsCsv(llvm::StringRef filePath,
                           llvm::ArrayRef<AccessRecord> rows, int logicalRank) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filePath, ec);
  if (ec) {
    llvm::errs() << "failed to open csv file: " << filePath
                 << ", error = " << ec.message() << "\n";
    return;
  }

  os << "op,warp_id,lane_id,reg_id,thread_id,access_ord,vec,bank,segment";
  for (int i = 0; i < logicalRank; ++i)
    os << ",coord" << i;
  os << "\n";

  for (const auto &r : rows) {
    os << r.op << "," << r.warpId << "," << r.laneId << "," << r.regId << ","
       << r.threadId << "," << r.accessOrd << "," << r.vec << "," << r.bank
       << "," << r.segment;

    for (int i = 0; i < logicalRank; ++i) {
      os << ",";
      if (i < static_cast<int>(r.logicalCoords.size()))
        os << r.logicalCoords[i];
    }
    os << "\n";
  }
}

void dumpSwizzleAccessCsv(llvm::StringRef accessCsvPath,
                          const LinearLayout &stsToSmem,
                          const LinearLayout &ldsToSmem,
                          const LinearLayout *stsLogicalLayout,
                          const LinearLayout *ldsLogicalLayout) {
  auto stsRows = enumerateLayoutAccesses("sts", stsToSmem, stsLogicalLayout);
  auto ldsRows = enumerateLayoutAccesses("lds", ldsToSmem, ldsLogicalLayout);

  if (stsLogicalLayout && ldsLogicalLayout)
    validateSmemLogicalConsistency(stsRows, ldsRows);

  int logicalRank = 0;
  for (const auto &r : stsRows)
    logicalRank =
        std::max(logicalRank, static_cast<int>(r.logicalCoords.size()));
  for (const auto &r : ldsRows)
    logicalRank =
        std::max(logicalRank, static_cast<int>(r.logicalCoords.size()));

  std::vector<AccessRecord> allRows;
  allRows.reserve(stsRows.size() + ldsRows.size());
  allRows.insert(allRows.end(), stsRows.begin(), stsRows.end());
  allRows.insert(allRows.end(), ldsRows.begin(), ldsRows.end());

  writeAccessRecordsCsv(accessCsvPath, allRows, logicalRank);

  llvm::dbgs() << "[swizzle-csv] access dumped to: " << accessCsvPath << "\n";
}

} // namespace mlir::triton
