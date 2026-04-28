#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_MLSGROUP_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_MLSGROUP_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// HCUGPU MLS instruction selection utilities
//===----------------------------------------------------------------------===//

// flags: lsb: elem3 | row3 << 8 | col3 << 16 | alt2 << 24  msb
#define MLS_DS_FLAGS_GET_ELEM3(flags) ((flags) & 0xFF)
#define MLS_DS_FLAGS_GET_ROW3(flags) (((flags) >> 8) & 0xFF)
#define MLS_DS_FLAGS_GET_COL3(flags) (((flags) >> 16) & 0xFF)
#define MLS_DS_FLAGS_GET_ALT2(flags) (((flags) >> 24) & 0xFF)

#define MLS_DS_FLAGS_PACK(elem3, row3, col3, alt2) \
  (((elem3) & 0xFF) | \
   (((row3) & 0xFF) << 8) | \
   (((col3) & 0xFF) << 16) | \
   (((alt2) & 0xFF) << 24))

#define MLS_INST_DS_CNT_MAX  (4)

enum class MlsInterleaveKind : unsigned {
  InterleaveNone = 0,
  Interleave2    = 1,
  Interleave4    = 2,
  Interleave8    = 3,
};


struct MlsInsnAttr {
  unsigned opIdx;                  // or nonKTileIdx;
  std::array<unsigned, 2> mlsTile; // A: [m, k] or B: [k, n]
  bool kMajor;
  unsigned elemBitWidth;           // 4, 6, 8, 16, 32
  MlsInterleaveKind interleaveKind;
  unsigned mlsVersion;

  struct Mlnsn{
    std::array<unsigned, 2> instrShape;
    std::array<unsigned, 2> instrsPerWarp;
    std::array<unsigned, 2> instrOrder;
    llvm::StringRef insn;
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } mlInsn;

  struct DsInsn {
    std::array<unsigned, 2> instrShape;
    std::array<unsigned, 2> instrsPerWarp;
    std::array<unsigned, 2> instrOrder;
    llvm::StringRef insn;
    unsigned flags;
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } dsInsn;
};

using MatrixLoadInsnAttr   = MlsInsnAttr::Mlnsn;
using DsReadMatrixInsnAttr = MlsInsnAttr::DsInsn;

class MlsInsn {
private:
  MlsInsnAttr attr;

public:
  static FailureOr<MlsInsn> selectOrGetMlsInsn(unsigned nonKTile, unsigned kTile, unsigned elemBitWidth,
                                               unsigned opIdx, bool kMajor,
                                               MlsInterleaveKind altKind = MlsInterleaveKind::InterleaveNone,
                                               unsigned version = 1);
  MlsInsn(const MlsInsnAttr &attr) : attr(attr) {}

  unsigned getOpIdx() { return attr.opIdx; }
  bool isKMajor() { return attr.kMajor; }
  bool isTranspose() { return attr.kMajor; }
  bool isRowMajor() { return attr.opIdx == 0 ? attr.kMajor : !attr.kMajor; }
  unsigned getMajorDimIndex() { return attr.opIdx == 0 ? attr.kMajor : !attr.kMajor; }
  unsigned getDotLayoutKWidth() { return attr.elemBitWidth == 16 ? 4 : (attr.elemBitWidth == 8 ? 8 : 8); }
  SmallVector<unsigned> getMlsTile() { return SmallVector<unsigned>(attr.mlsTile.begin(), attr.mlsTile.end()); }
  unsigned getMlsNonKTile() { return attr.opIdx == 0 ? attr.mlsTile[0] : attr.mlsTile[1]; }
  unsigned getMlsKTile() { return attr.opIdx == 0 ? attr.mlsTile[1] : attr.mlsTile[0]; }
  unsigned getElemBitWidth() { return attr.elemBitWidth; }
  MlsInterleaveKind getInterleaveKind() { return attr.interleaveKind; }
  unsigned getAlt2Kind() { return static_cast<unsigned>(getInterleaveKind()); }
  unsigned getMlsVersion() { return attr.mlsVersion; }
  const MatrixLoadInsnAttr &getMatrixLoadInsnAttr() { return attr.mlInsn; }
  const DsReadMatrixInsnAttr &getDsReadMatrixInsnAttr() { return attr.dsInsn; }
};

} // namespace mlir

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTRANSFORMS_MLSGROUP_H_
