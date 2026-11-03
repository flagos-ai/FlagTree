#ifndef TRITON_HCU_WHOLE_PACKET_CONSUMER_PLAN_H
#define TRITON_HCU_WHOLE_PACKET_CONSUMER_PLAN_H
#include "Dialect/TritonHCUGPU/IR/WholePacketLayout.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::triton::HCU {
namespace whole_packet_consumer_detail {
namespace wp = triton::hcugpu::wholepacket;
struct Plan {
  using MatrixGroup = SmallVector<unsigned,8>;
  using MatrixGroups = SmallVector<MatrixGroup>;
  unsigned registers = 0, width = 1;
  bool matrix = false, matrixAlt = false, repartition = false;
  // Only the optional instruction/group optimization budget can fall back to
  // scalar reads. Compact preflight rejects that fallback instead of granting
  // authority from a partially explored matrix plan.
  bool optimizationBudgetExceeded = false;
  std::array<unsigned,3> registerAxes{0,1,2};
  MatrixGroups matrixByAnchor;
  SmallVector<unsigned> groups, representatives, deltas, unique;
};
} // namespace whole_packet_consumer_detail

// Pure and immutable: no Operation mutation, LLVM builder, rw, allocation,
// imported witness, shape-name selector or emission. The same plan feeds the
// owner preflight and consumer lowering. All addresses are role-relative words.
inline std::optional<whole_packet_consumer_detail::Plan>
planWholePacketConsumer(RankedTensorType resultType,
    const triton::hcugpu::wholepacket::PacketSemantics &semantics,
    const triton::hcugpu::wholepacket::SourceLayout &source,
    unsigned role, unsigned packetVersion, bool partitionRequested,
    uint64_t optimizationBudget = uint64_t(1) << 26) {
  namespace wp = triton::hcugpu::wholepacket;
  using Plan = whole_packet_consumer_detail::Plan;
  if (!resultType || resultType.getRank()!=2 || !resultType.hasStaticShape() ||
      !resultType.getElementType().isIntOrFloat() ||
      resultType.getElementType().getIntOrFloatBitWidth()!=16 || role>1 ||
      (packetVersion!=256 && packetVersion!=257 && packetVersion!=258) ||
      !source.rows || source.rows>wp::Elements || !source.columns ||
      source.columns>wp::Elements || !source.warps || source.warps>8 || (source.warps & (source.warps-1)) ||
      (source.rows & (source.rows-1)) || (source.columns & (source.columns-1)) ||
      source.activeElements()>wp::Elements ||
      semantics.storageRows(resultType.getShape()[0],resultType.getShape()[1])!=source.rows ||
      semantics.storageColumns(resultType.getShape()[0],resultType.getShape()[1])!=source.columns)
    return std::nullopt;
  auto encoding=resultType.getEncoding();
  if (!encoding || !isa<triton::gpu::DotOperandEncodingAttr,
                         triton::gpu::LinearEncodingAttr>(encoding))
    return std::nullopt;
  auto *ctx=resultType.getContext();
  auto reg=StringAttr::get(ctx,"register"),lane=StringAttr::get(ctx,"lane"),warp=StringAttr::get(ctx,"warp");
  auto logical=triton::gpu::toLinearLayout(resultType);
  if (!logical.hasInDim(reg) || !logical.hasInDim(lane) ||
      !logical.hasInDim(warp) || logical.getNumOutDims()!=2)
    return std::nullopt;
  for (auto name:logical.getInDimNames())
    if (name!=reg && name!=lane && name!=warp && logical.getInDimSize(name)!=1)
      return std::nullopt;
  unsigned dim=0;
  for (auto name:logical.getOutDimNames())
    if (logical.getOutDimSize(name)!=resultType.getShape()[dim++])
      return std::nullopt;
  logical=logical.sublayout({reg,lane,warp},llvm::to_vector(logical.getOutDimNames()));
  unsigned registers=logical.getInDimSize(reg);
  if (!registers || logical.getInDimSize(lane)!=64 ||
      logical.getInDimSize(warp)!=source.warps)
    return std::nullopt;
  // Mandatory whole-result proof is independently bounded before allocating.
  // Unsupported enormous layouts retain the old owner (and fail closed if
  // later requested as an opaque consumer); they never obtain a compact view.
  uint64_t points=uint64_t(registers)*64*source.warps;
  if (points>(uint64_t(1)<<20)) return std::nullopt;
  const auto *p=&source;
  SmallVector<unsigned> physicalTable;
  physicalTable.reserve(points);
  for (unsigned r=0;r<registers;++r)for (unsigned w=0;w<p->warps;++w)
    for (unsigned l=0;l<64;++l) {
      auto xy=logical.apply({{reg,int32_t(r)},{lane,int32_t(l)},{warp,int32_t(w)}});
      if (xy.size()!=2 || xy[0].second<0 || xy[1].second<0 ||
          xy[0].second>=resultType.getShape()[0] ||
          xy[1].second>=resultType.getShape()[1]) return std::nullopt;
      unsigned row=semantics.transpose?xy[1].second:xy[0].second;
      unsigned col=semantics.transpose?xy[0].second:xy[1].second;
      unsigned coord=row*p->columns+col;
      if (coord>=p->activeElements() || p->ordinal(coord)>=p->activeElements() ||
          p->logical(p->ordinal(coord))!=coord) return std::nullopt;
      unsigned address=p->physical(semantics.transfer,coord);
      if (address>=p->activeElements()) return std::nullopt;
      physicalTable.push_back(address);
    }
  uint64_t visits=0;
  bool exhausted=false;
  auto physicalAt=[&](unsigned r,unsigned l,unsigned w)->unsigned {
    if (++visits>optimizationBudget || r>=registers || l>=64 || w>=p->warps) {
      exhausted=true;return 0x3fffffffu;
    }
    return physicalTable[(uint64_t(r)*p->warps+w)*64+l];
  };
  bool structured=semantics.transfer==wp::Role::StructuredRows;
  // Normal matrix-read's actual implicit per-lane gather. Validate the entire
  // finite destination layout; neither MFMA warp geometry nor tensor shape is
  // a selection condition. Any unsupported ownership retains tracked loads.
  auto matrixLegal=[&]() {
    if(structured || role!=1 || registers%8 || logical.getInDimSize(lane)!=64 ||
        logical.getInDimSize(warp)!=p->warps)return false;
    constexpr unsigned sources[8]={0,4,8,12,2,6,10,14};
    for(unsigned r=0;r<registers;r+=8)for(unsigned w=0;w<p->warps;++w)
      for(unsigned l=0;l<64;++l)for(unsigned j=0;j<8;++j) {
        unsigned sourceLane=(l&~15u)+sources[j]+((l>>3)&1);
        unsigned sourceReg=((sourceLane>>2)&3)|(((sourceLane>>1)&1)<<2);
        unsigned readLane=(sourceLane&48)|((sourceLane&1)<<3);
        unsigned base=physicalAt(r+sourceReg,readLane,w);
        if(base%8 || base+7>=p->activeElements() || base+(l&7)!=physicalAt(r+j,l,w))return false;
      }
    return true;
  };
  bool matrix=matrixLegal();
  // The request is not a proof: every selected partition is verified over the
  // complete register/lane/warp chart. Absent/false keeps the original path.
  bool repartition=false;
  std::array<unsigned,3> registerAxes{0,1,2};
  using MatrixGroup=SmallVector<unsigned,8>;
  using MatrixGroups=SmallVector<MatrixGroup>;
  MatrixGroups matrixByAnchor(registers);
  if(partitionRequested && !structured && role==1 && registers>=8 &&
      !(registers&(registers-1)) && logical.getInDimSize(lane)==64 &&
      logical.getInDimSize(warp)==p->warps) {
    auto groupsFor=[&](const std::array<unsigned,3> &axes) {
      MatrixGroups result;
      unsigned mask=(1u<<axes[0])|(1u<<axes[1])|(1u<<axes[2]);
      for(unsigned base=0;base<registers;++base)if(!(base&mask)) {
        MatrixGroup g;
        for(unsigned j=0;j<8;++j) {
          unsigned r=base;
          for(unsigned bit=0;bit<3;++bit)r|=((j>>bit)&1u)<<axes[bit];
          g.push_back(r);
        }
        result.push_back(std::move(g));
      }
      return result;
    };
    auto groupReadPhysicalAt=[&](const MatrixGroup &g,unsigned l,unsigned w) {
      unsigned sr=((l>>2)&3)|(((l>>1)&1)<<2);
      unsigned rl=(l&48)|((l&1)<<3);
      return physicalAt(g[sr],rl,w);
    };
    auto groupsLegal=[&](const MatrixGroups &groups) {
      if(groups.size()*8!=registers)return false;
      SmallVector<bool> seen(registers,false);
      for(const auto &g:groups) {
        if(g.size()!=8)return false;
        for(unsigned r:g) {
          if(r>=registers || seen[r])return false;
          seen[r]=true;
        }
        constexpr unsigned sourceLanes[8]={0,4,8,12,2,6,10,14};
        for(unsigned w=0;w<p->warps;++w)for(unsigned l=0;l<64;++l)
          for(unsigned j=0;j<8;++j) {
            unsigned sl=(l&~15u)+sourceLanes[j]+((l>>3)&1);
            unsigned base=groupReadPhysicalAt(g,sl,w);
            if(base%8 || base+7>=p->activeElements() ||
                base+(l&7)!=physicalAt(g[j],l,w))return false;
          }
      }
      return true;
    };
    // Structural rank, not a hardware latency/conflict guarantee. Model eight
    // issuing lanes, each16B/four32-bit words across32banks. Deduplicate exact
    // same-word broadcasts, then count distinct-word bank fan-in.
    auto bankScore=[&](const MatrixGroups &groups) {
      uint64_t excess=0;unsigned maximum=0;
      for(const auto &g:groups)for(unsigned w=0;w<p->warps;++w)
        for(unsigned first=0;first<64;first+=8) {
          std::array<SmallVector<unsigned,4>,32> banks;
          for(unsigned l=first;l<first+8;++l) {
            unsigned word=groupReadPhysicalAt(g,l,w)/2;
            for(unsigned q=0;q<4;++q) {
              unsigned address=word+q;
              auto &bank=banks[address%32];
              if(!llvm::is_contained(bank,address))bank.push_back(address);
            }
          }
          for(const auto &bank:banks)if(!bank.empty()) {
            excess+=bank.size()-1;
            maximum=std::max(maximum,unsigned(bank.size()));
          }
        }
      return std::make_pair(excess,maximum);
    };
    unsigned bits=0;for(unsigned n=registers;n>1;n>>=1)++bits;
    MatrixGroups bestGroups;
    std::pair<uint64_t,unsigned> bestScore{uint64_t(-1),unsigned(-1)};
    std::array<unsigned,3> bestAxes{0,1,2};
    // Baseline wins ties. Otherwise first strict improvement in generic
    // lexicographic bit-axis order is stable; no shape/Torch register gate.
    if(matrix) {bestGroups=groupsFor(bestAxes);bestScore=bankScore(bestGroups);}
    for(unsigned a=0;a<bits;++a)for(unsigned b=a+1;b<bits;++b)
      for(unsigned c=b+1;c<bits;++c) {
        std::array<unsigned,3> axes{a,b,c};auto groups=groupsFor(axes);
        if(!groupsLegal(groups))continue;
        auto score=bankScore(groups);
        if(score<bestScore) {bestGroups=std::move(groups);bestScore=score;bestAxes=axes;}
      }
    if(!bestGroups.empty()) {
      matrix=true;registerAxes=bestAxes;
      repartition=registerAxes!=std::array<unsigned,3>{0,1,2};
      if(repartition)for(auto &g:bestGroups)matrixByAnchor[g[0]]=std::move(g);
    }
  }
  // ALT's gather is independent of a dot parent or a requested tile. Prove
  // all physical addresses, including every lane/warp/register and the byte
  // alignment and bounds of each issuing lane. Normal matrix reads keep
  // their existing selection and register partition unchanged.
  bool matrixAlt=false;
  if(!matrix && !structured && role==1 && registers>=8 &&
      !(registers&(registers-1)) && logical.getInDimSize(lane)==64 &&
      logical.getInDimSize(warp)==p->warps) {
    unsigned bits=logical.getInDimSizeLog2(reg);
    for(unsigned chosen=2;chosen<bits && !matrixAlt;++chosen) {
      unsigned mask=3u|(1u<<chosen);
      MatrixGroups candidates;
      bool legal=true;
      for(unsigned base=0;base<registers && legal;++base)if(!(base&mask)) {
        MatrixGroup g;
        for(unsigned j=0;j<8;++j)
          g.push_back(base|(j&3u)|((j>>2)<<chosen));
        for(unsigned w=0;w<p->warps && legal;++w)
          for(unsigned l=0;l<64 && legal;++l)
            for(unsigned j=0;j<8;++j) {
              unsigned address=physicalAt(g[j&3u],l&~3u,w);
              if(address%8 || address+7>=p->activeElements() ||
                  address+(j>>2)+2*(l&3u)!=physicalAt(g[j],l,w)) {
                legal=false;break;
              }
            }
        candidates.push_back(std::move(g));
      }
      if(legal && candidates.size()*8==registers) {
        matrix=matrixAlt=true;repartition=true;
        registerAxes={0,1,chosen};
        for(auto &g:candidates)matrixByAnchor[g[0]]=std::move(g);
      }
    }
  }
  auto vectorLegal=[&](unsigned width) {
    if(registers%width)return false;
    for(unsigned r=0;r<registers;r+=width)for(unsigned w=0;w<p->warps;++w)
      for(unsigned l=0;l<64;++l) {
        unsigned base=physicalAt(r,l,w);
        if(base%width || base+width>p->activeElements())return false;
        for(unsigned j=1;j<width;++j)if(physicalAt(r+j,l,w)!=base+j)return false;
      }
    return true;
  };
  unsigned width=matrix?8:(packetVersion==258 && vectorLegal(8))?8:vectorLegal(4)?4:vectorLegal(2)?2:1;
  auto readPhysicalAt=[&](unsigned r,unsigned l,unsigned w) {
    if(!matrix)return physicalAt(r,l,w);
    unsigned sourceReg=matrixAlt?((l>>2)&3):((l>>2)&3)|(((l>>1)&1)<<2);
    unsigned sourceLane=(l&48)|(matrixAlt?((l&3)<<2):((l&1)<<3));
    return physicalAt(repartition?matrixByAnchor[r][sourceReg]:r+sourceReg,sourceLane,w);
  };
  SmallVector<unsigned> groups;
  if(repartition) {
    for(unsigned r=0;r<registers;++r)if(!matrixByAnchor[r].empty())groups.push_back(r);
  } else for(unsigned r=0;r<registers;r+=width)groups.push_back(r);
  llvm::sort(groups,[&](unsigned x,unsigned y){return readPhysicalAt(x,0,0)<readPhysicalAt(y,0,0);});
  SmallVector<unsigned> representatives(registers),deltas(registers);
  SmallVector<unsigned> unique;
  for(unsigned r:groups) {
    bool found=false;
    for(unsigned rep:unique) {
      int delta=int(readPhysicalAt(r,0,0))-int(readPhysicalAt(rep,0,0));
      if(delta<0 || 2*delta>65535)continue;
      bool equal=true;
      for(unsigned w=0;w<p->warps && equal;++w)for(unsigned l=0;l<64;++l)
        equal&=int(readPhysicalAt(r,l,w))-int(readPhysicalAt(rep,l,w))==delta;
      if(equal){representatives[r]=rep;deltas[r]=delta;found=true;break;}
    }
    if(!found){representatives[r]=r;deltas[r]=0;unique.push_back(r);}
  }
  if (exhausted) {
    Plan fallback;
    fallback.registers=registers;
    fallback.optimizationBudgetExceeded=true;
    fallback.matrixByAnchor.resize(registers);
    fallback.representatives.resize(registers);
    fallback.deltas.resize(registers,0);
    for(unsigned r=0;r<registers;++r) {
      fallback.groups.push_back(r);fallback.unique.push_back(r);
      fallback.representatives[r]=r;
    }
    return fallback;
  }
  // Prove the FULL issuing footprint of the selected read and its final
  // representative+immediate form, independently of the selection heuristics.
  if (!width || registers%width || groups.size()!=registers/width)
    return std::nullopt;
  SmallVector<bool> covered(registers,false);
  for (unsigned r:groups) {
    if (r>=registers || representatives[r]>=registers ||
        !llvm::is_contained(unique,representatives[r]) ||
        (repartition && (matrixByAnchor[r].size()!=8 || width!=8)))
      return std::nullopt;
    for (unsigned j=0;j<width;++j) {
      unsigned output=repartition?matrixByAnchor[r][j]:r+j;
      if(output>=registers || covered[output]) return std::nullopt;
      covered[output]=true;
    }
    for (unsigned w=0;w<p->warps;++w)for(unsigned l=0;l<64;++l) {
      // Lookup is over the same complete result chart; do not consume any
      // optional-search budget when checking the final safety obligation.
      auto at=[&](unsigned rr,unsigned ll)->unsigned {
        if(rr>=registers || ll>=64) return 0x3fffffffu;
        return physicalTable[(uint64_t(rr)*p->warps+w)*64+ll];
      };
      auto issuing=[&](unsigned anchor)->unsigned {
        if(!matrix) return at(anchor,l);
        unsigned sr=matrixAlt?((l>>2)&3):((l>>2)&3)|(((l>>1)&1)<<2);
        unsigned sl=(l&48)|(matrixAlt?((l&3)<<2):((l&1)<<3));
        if(repartition && (anchor>=matrixByAnchor.size() || matrixByAnchor[anchor].size()!=8))
          return 0x3fffffffu;
        return at(repartition?matrixByAnchor[anchor][sr]:anchor+sr,sl);
      };
      unsigned address=issuing(r),representative=issuing(representatives[r]);
      if(address%width || uint64_t(address)+width>p->activeElements() ||
          uint64_t(representative)+deltas[r]!=address ||
          uint64_t(representative)+deltas[r]+width>p->activeElements())
        return std::nullopt;
    }
  }
  if(llvm::is_contained(covered,false)) return std::nullopt;
  Plan result;
  result.registers=registers;result.width=width;result.matrix=matrix;
  result.matrixAlt=matrixAlt;result.repartition=repartition;
  result.registerAxes=registerAxes;result.matrixByAnchor=std::move(matrixByAnchor);
  result.groups=std::move(groups);result.representatives=std::move(representatives);
  result.deltas=std::move(deltas);result.unique=std::move(unique);
  return result;
}
} // namespace mlir::triton::HCU
#endif
