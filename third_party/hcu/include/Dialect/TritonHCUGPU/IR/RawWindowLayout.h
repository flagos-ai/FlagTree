#ifndef TRITON_HCU_RAW_WINDOW_LAYOUT_H_
#define TRITON_HCU_RAW_WINDOW_LAYOUT_H_
#include <array>
#include <cstdint>
#include <optional>
namespace mlir::triton::hcugpu::rawwindow {
constexpr unsigned Elements=16384, Bytes=32768, Lanes=64, Waves=4;
// Arithmetic physical packet rotation measured independently on gfx936.
// This is not a GF(2) LinearLayout or an ordinary packed-m0 LDS address.
inline unsigned physical(unsigned ordinal) {
  unsigned copy=ordinal>>11, wave=(ordinal>>9)&3;
  return 2048*copy+512*wave+((ordinal+16*wave)&511);
}
struct SourceLayout {
  unsigned rows,columns;
  std::array<unsigned,14> basis,inverseBasis;
  static unsigned apply(const std::array<unsigned,14>& b,unsigned v) {
    unsigned out=0;for(unsigned i=0;i<14;++i)if(v&(1u<<i))out^=b[i];return out;
  }
  unsigned logical(unsigned ordinal) const {return apply(basis,ordinal);}
  unsigned ordinal(unsigned logical) const {return apply(inverseBasis,logical);}
  unsigned address(unsigned logical) const {return physical(ordinal(logical));}
  static std::optional<SourceLayout> get(unsigned rows,unsigned columns,
                                        const std::array<unsigned,14>& basis) {
    auto power=[](unsigned v){return v&&!(v&(v-1));};
    if(!power(rows)||!power(columns)||columns<8||uint64_t(rows)*columns!=Elements)return {};
    SourceLayout out{rows,columns,basis,{}};
    std::array<unsigned,Elements> inverse;inverse.fill(Elements);
    for(unsigned u=0;u<Elements;++u) {
      unsigned c=out.logical(u),base=out.logical(u&~7u);
      if(c>=Elements||inverse[c]!=Elements||base%8||c!=base+(u&7)||c/columns!=base/columns)return {};
      inverse[c]=u;
    }
    for(unsigned i=0;i<14;++i)out.inverseBasis[i]=inverse[1u<<i];
    for(unsigned c=0;c<Elements;++c)if(out.ordinal(c)!=inverse[c])return {};
    return out;
  }
};
}
#endif
