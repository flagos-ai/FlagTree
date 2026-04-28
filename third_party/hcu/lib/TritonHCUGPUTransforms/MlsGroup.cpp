#include "TritonHCUGPUTransforms/MlsGroup.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

struct MlsInsnGroupSelectKey {
  unsigned nonKTile;
  unsigned kTile;
  bool transpose;
  unsigned elemBitWidth;
  MlsInterleaveKind alt2Kind;
  unsigned version;
};

struct iMlsInsnAttr {
  struct {
    unsigned instNonKDim;
    unsigned instKDim;
    llvm::StringRef insn;
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } mlInfo;
  struct {
    unsigned instNonKDim;
    unsigned instKDim;
    llvm::StringRef insn;
    unsigned flags; // lsb: elem3 | row3 << 8 | col3 << 16 | alt2 << 24  msb
    std::array<unsigned, MLS_INST_DS_CNT_MAX> dsByteOffsets;
  } dsInfo;
};


template <typename T>
constexpr typename std::underlying_type<T>::type cast_as_underlying2(T t) {
  return static_cast<typename std::underlying_type<T>::type>(t);
}
struct MlsInsnGroupSelectKeyInfo
    : public llvm::DenseMapInfo<MlsInsnGroupSelectKey> {
  static inline MlsInsnGroupSelectKey getEmptyKey() {
    return {0, 0, false, 16, MlsInterleaveKind::InterleaveNone, 0};
  }

  static inline MlsInsnGroupSelectKey getTombstoneKey() {
    return {~0U, ~0U, true, ~0U, MlsInterleaveKind::InterleaveNone, ~0U};
  }

  static inline bool isEqual(const MlsInsnGroupSelectKey &lhs,
                             const MlsInsnGroupSelectKey &rhs) {
    return lhs.nonKTile == rhs.nonKTile && lhs.kTile == rhs.kTile &&
           lhs.transpose == rhs.transpose && lhs.elemBitWidth == rhs.elemBitWidth &&
           lhs.version == rhs.version && lhs.alt2Kind == rhs.alt2Kind;
  }

  static unsigned getHashValue(const MlsInsnGroupSelectKey &key) {
    auto nonKTileHash = llvm::detail::combineHashValue(key.nonKTile, key.kTile);
    auto transposeHash = llvm::detail::combineHashValue(nonKTileHash, key.transpose);
    auto elemBitWidthHash = llvm::detail::combineHashValue(transposeHash, key.elemBitWidth);
    auto alt2KindHash = cast_as_underlying2(key.alt2Kind);
    auto verHash = llvm::detail::combineHashValue(elemBitWidthHash, alt2KindHash);
    return llvm::detail::combineHashValue(verHash, key.version);
  }
};


using MlsInsnGroupMap = llvm::DenseMap<MlsInsnGroupSelectKey, iMlsInsnAttr,
                                        MlsInsnGroupSelectKeyInfo>;

auto getMlsInsnGroupAttrMap = []() -> const MlsInsnGroupMap & {
  static MlsInsnGroupMap MlsInsnMap {
    // b16:
    // matrix_load_32x16_b16: AT/BN(transpose = 1), interleave none,  mnk required least [16, 16, 32]
    {{16, 32, true, 16, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 32, ROCDL::hcu_matrix_load_32X16_b16::getOperationName(), {0, 0, 0, 0}},
        {16, 32, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 0, 0, 0}}
      }
    },

    // matrix_load_32x16_b16: AN/BT(transpose = 0), interleave none,  mnk required least [32, 32, 16]
    {{32, 16, false, 16, MlsInterleaveKind::InterleaveNone, 1},
      {{32, 16, ROCDL::hcu_matrix_load_32X16_b16::getOperationName(), {0, 0, 0, 0}},
        {32, 16, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 0, 0, 0}}
      }
    },

    // matrix_load_64x16_b16: AT/BN(transpose = 1), interleave none,  mnk required least [16, 16, 64]
    {{16, 64, true, 16, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_b16::getOperationName(), {0, 0, 0, 0}},
        {16, 32, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 1024, 0, 0}}
      }
    },

    // matrix_load_64x16_b16: AN/BT(transpose = 0), interleave none,  mnk required least [64, 64, 16]
    {{64, 16, false, 16, MlsInterleaveKind::InterleaveNone, 1},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b16::getOperationName(), {0, 0, 0, 0}},
        {32, 16, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(2, 2, 1, 0), {0, 1024, 0, 0}}
      }
    },

    // b8:
    // matrix_load_64x16_b8: AT/BN(transpose = 1), interleave none,  mnk required least [16, 16, 64]
    {{16, 64, true, 8, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 64, ROCDL::hcu_matrix_load_64X16_b8::getOperationName(), {0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 0, 0, 0}}
      }
    },
    // matrix_load_64x16_b8: AN/BT(transpose = 0), interleave none,  mnk required least [64, 64, 16]
    {{64, 32, false, 8, MlsInterleaveKind::InterleaveNone, 1},
      {{64, 16, ROCDL::hcu_matrix_load_64X16_b8::getOperationName(), {0, 1024, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 0, 0}}
      }
    },
    // matrix_load_128x16_b8: AT/BN(transpose = 1), interleave none,  mnk required least [16, 16, 128]
    {{16, 128, true, 8, MlsInterleaveKind::InterleaveNone, 1},
      {{16, 128, ROCDL::hcu_matrix_load_128X16_b8::getOperationName(), {0, 0, 0, 0}},
        {16, 64, ROCDL::hcu_ds_read_matrix_trans_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 3, 1, 0), {0, 1024, 0, 0}}
      }
    },
    // matrix_load_128x16_b8: AN/BT(transpose = 0), interleave none,  mnk required least [128, 128, 32]
    {{128, 32, false, 8, MlsInterleaveKind::InterleaveNone, 1},
      {{128, 16, ROCDL::hcu_matrix_load_128X16_b8::getOperationName(), {0, 1024, 0, 0}},
        {32, 32, ROCDL::hcu_ds_read_matrix_format::getOperationName(),
                MLS_DS_FLAGS_PACK(1, 2, 2, 0), {0, 32, 2048, 2080}}
      }
    },
  };
  return MlsInsnMap;
};

FailureOr<MlsInsn> MlsInsn::selectOrGetMlsInsn(unsigned nonKTile, unsigned kTile, unsigned elemBitWidth,
                                               unsigned opIdx, bool kMajor,
                                               MlsInterleaveKind altKind, unsigned version) {
  bool transpose = kMajor;  // HCU if not AN/BT, set transpose to true.
  auto MlsInsnAttrMap = getMlsInsnGroupAttrMap();

  MlsInsnGroupSelectKey key = {nonKTile, kTile, transpose,
                               elemBitWidth, altKind, version};
  auto it = MlsInsnAttrMap.find(key);
  if (it == MlsInsnAttrMap.end())
    return failure();


  // fill the MlsInsnAttr
  std::array<unsigned, 2> order = opIdx == 0 ? (kMajor ? std::array<unsigned, 2>{1, 0}
                                                       : std::array<unsigned, 2>{0, 1})
                                             : (kMajor ? std::array<unsigned, 2>{0, 1}
                                                       : std::array<unsigned, 2>{1, 0});
  iMlsInsnAttr iAttr = it->second;

  MlsInsnAttr oAttr;

  oAttr.opIdx = opIdx;
  oAttr.mlsTile = opIdx == 0 ? std::array<unsigned, 2>{nonKTile, kTile}
                             : std::array<unsigned, 2>{kTile, nonKTile};
  oAttr.kMajor = kMajor;
  oAttr.elemBitWidth = elemBitWidth;
  oAttr.interleaveKind = altKind;
  oAttr.mlsVersion = version;
  assert(oAttr.interleaveKind == static_cast<MlsInterleaveKind>(MLS_DS_FLAGS_GET_ALT2(iAttr.dsInfo.flags)));

  if (opIdx == 0) {
    oAttr.mlInsn.instrShape   = {iAttr.mlInfo.instNonKDim, iAttr.mlInfo.instKDim};
    oAttr.mlInsn.instrsPerWarp= {nonKTile/iAttr.mlInfo.instNonKDim, kTile/iAttr.mlInfo.instKDim};
    oAttr.mlInsn.instrOrder   = order;
    oAttr.mlInsn.insn         = iAttr.mlInfo.insn;
    oAttr.mlInsn.dsByteOffsets= iAttr.mlInfo.dsByteOffsets;

    oAttr.dsInsn.instrShape   = {iAttr.dsInfo.instNonKDim, iAttr.dsInfo.instKDim};
    oAttr.dsInsn.instrsPerWarp= {nonKTile/iAttr.dsInfo.instNonKDim, kTile/iAttr.dsInfo.instKDim};
    oAttr.dsInsn.instrOrder   = order;  // ds_read_matrix instrOrder is always for tensor order.
    oAttr.dsInsn.insn         = iAttr.dsInfo.insn;
    oAttr.dsInsn.flags        = iAttr.dsInfo.flags;
    oAttr.dsInsn.dsByteOffsets= iAttr.dsInfo.dsByteOffsets;
  } else {
    oAttr.mlInsn.instrShape   = {iAttr.mlInfo.instKDim, iAttr.mlInfo.instNonKDim};
    oAttr.mlInsn.instrsPerWarp= {kTile/iAttr.mlInfo.instKDim, nonKTile/iAttr.mlInfo.instNonKDim};
    oAttr.mlInsn.instrOrder   = order;
    oAttr.mlInsn.insn         = iAttr.mlInfo.insn;
    oAttr.mlInsn.dsByteOffsets= iAttr.mlInfo.dsByteOffsets;

    oAttr.dsInsn.instrShape   = {iAttr.dsInfo.instKDim, iAttr.dsInfo.instNonKDim};
    oAttr.dsInsn.instrsPerWarp= {kTile/iAttr.dsInfo.instKDim, nonKTile/iAttr.dsInfo.instNonKDim};
    oAttr.dsInsn.instrOrder   = order;
    oAttr.dsInsn.insn         = iAttr.dsInfo.insn;
    oAttr.dsInsn.flags        = iAttr.dsInfo.flags;
    oAttr.dsInsn.dsByteOffsets= iAttr.dsInfo.dsByteOffsets;
  }

  // Note: current matrix_load tile seems only exist [1, 1], [xxx, 1], [1, xxx] and no [xxx, xxx] case
  // so cur instrOrder set to tensor order now.
  assert(oAttr.mlInsn.instrsPerWarp[0] == 1 || oAttr.mlInsn.instrsPerWarp[1] == 1);

  unsigned dsInstsPerWarp = oAttr.dsInsn.instrsPerWarp[0] * oAttr.dsInsn.instrsPerWarp[1];
  unsigned mlInstsPerWarp = oAttr.mlInsn.instrsPerWarp[0] * oAttr.mlInsn.instrsPerWarp[1];
  assert(dsInstsPerWarp <= MLS_INST_DS_CNT_MAX && mlInstsPerWarp <= MLS_INST_DS_CNT_MAX);

  return MlsInsn(oAttr);
}


} // namespace mlir
