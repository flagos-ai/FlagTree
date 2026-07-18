#ifndef TRITON_SDNN_COMBINE_PATTERNS_H
#define TRITON_SDNN_COMBINE_PATTERNS_H

namespace mlir {
namespace triton {
namespace sdnn {

void populateMMAScaleCombinePatterns(RewritePatternSet &patterns,
                                     uint32_t xpu_arch);
void populateCombinePatterns(RewritePatternSet &patterns, uint32_t xpu_arch);
void populatePostCombinePatterns(RewritePatternSet &patterns,
                                 uint32_t xpu_arch);
void populateMaskEWPatterns(RewritePatternSet &patterns, uint32_t xpu_arch);
void populateDMAPatterns(RewritePatternSet &patterns, uint32_t xpu_arch);
void polulateEwBufferSeperate(RewritePatternSet &patterns);

} // namespace sdnn
} // namespace triton
} // namespace mlir

#endif // TRITON_SDNN_COMBINE_PATTERNS_H
