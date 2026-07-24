from pathlib import Path


# Keep this check device-free so backend CI can validate header selection early.
def test_enflame_elementwise_dedup_fallback_is_vendored():
    root = Path(__file__).resolve().parents[5]
    main_header = root / "include/triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
    enflame_header = (
        root / "third_party/enflame/backend/spec/include/triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h")
    main_source = main_header.read_text()
    enflame_source = enflame_header.read_text()

    assert ("for (auto [c, d] : llvm::zip(constancy, dims)) {\n"
            "      assert(llvm::isPowerOf2_32(c));" in main_source)
    assert "if (!llvm::isPowerOf2_32(c))\n        return resultVals;" in enflame_source
    assert "assert(llvm::isPowerOf2_32(c));" in enflame_source
