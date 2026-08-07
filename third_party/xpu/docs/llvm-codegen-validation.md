# XPU LLVM Codegen Cross-Toolchain Validation

## Conclusion

Do not disable TritonXPU vectorization to work around the BF16 broadcast `mul`
failure. FlagTree and the internal Triton reference both enable
`add_tritonxpu_vectorize_pass` when `isCloseVectorization` is `false`, and that
is the intended implementation.

The observed failure was isolated to the XPU LLVM backend in the
`llvm_trust` toolchain. For the same optimized LLVM IR and the same runtime
inputs:

- the internal `xtdk-llvm22` toolchain generated a correct XPUBIN;
- `llvm_trust` generated an XPUBIN with 1,048,481 mismatches out of 33,554,432
  BF16 values (3.1%);
- disabling TritonXPU vectorization avoided the failure, but was only a
  diagnostic experiment and is not an accepted fix.

The reproducer was FlagGems test
`tests/test_mul.py::test_mul_broadcast_shape[dtype2-shape_a2-shape_b2]`, with
shapes `(1048576, 1)` and `(1, 32)`, BF16 dtype, and random seed 0.

## Evidence

The FlagTree and internal compiler pipelines produced LLVM IR that differed
only in the PID embedded in `DIFile` debug metadata. Instructions, types,
address calculations, target triple, data layout, and kernel annotations were
identical. The IR had already passed external-library linking, target data
layout attachment, O3 optimization, and `amend_func` before comparison.

Both LLVM toolchains reported LLVM 22.1.8 and accepted the same input with:

```bash
llc -mtriple=xpu3-baidu-none-gnu -mcpu=xpu3 -filetype=asm \
  -o kernel.s kernel.llir
llc -mtriple=xpu3-baidu-none-gnu -mcpu=xpu3 -filetype=obj \
  -o kernel.o kernel.llir
xpu3-elfconv-triton kernel.o kernel.xpubin TOOLCHAIN_BIN_DIR
```

The outputs differed materially in stack size, BF16 vector construction, mask
register handling, `vmul`, `vscatter`, and loop code. The artifacts from the
2026-07-19 investigation had these SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| `llvm_trust/lib/libLLVMXPUCodeGen.a` | `6a2fcedbeb272946aa51809556b38edd93663d6cd2ecf9dd07132c7bdb4fd2c1` |
| internal `lib/libLLVMXPUCodeGen.a` | `d52cdc7684f9c122fbf3e57824d47d8816a84ab3f6ade4eb350dcf4c8850f874` |
| `llvm_trust` XPUBIN | `726b50cfe59c014bd0a37018f6981a76484df75135c5a5d53267fb721c71c0ca` |
| internal XPUBIN | `bc2f394d235f801cc00891eb26835ed42786577bf601fcf806c4ffd9ef5d2ae1` |

The two XPUBINs were then injected separately into the same FlagTree Python
and XPU runtime process by temporarily replacing `XPUBackend.make_xpubin` with
a function that returned the selected binary. The internal XPUBIN passed on
XPU 6. The `llvm_trust` XPUBIN failed on XPU 7 with the same deterministic
3.1% mismatch. This controls for the Triton frontend, MLIR pipeline, LLVM IR,
FlagGems kernel, runtime, input, and seed.

A crossed `llc`/`elfconv` matrix isolated the failing stage further:

| Object producer | XPUBIN converter | Device result |
| --- | --- | --- |
| internal `xtdk-llvm22` `llc` | internal `elfconv` | Pass |
| `llvm_trust` `llc` | `llvm_trust` `elfconv` | Fail, 3.1% mismatch |
| internal `xtdk-llvm22` `llc` | `llvm_trust` `elfconv` | Pass |
| `llvm_trust` `llc` | internal `elfconv` | Fail, 3.1% mismatch |

Correctness follows the object producer, not the XPUBIN converter. The defect
is therefore in the `llvm_trust` XPU LLVM code-generation path before
`elfconv`, rather than in XPUBIN packaging.

## Validation Procedure

Use this procedure when two XPU Triton builds produce different numerical
results and a compiler backend issue is suspected:

1. Use the same kernel, shapes, dtype, launch configuration, environment, and
   fixed random seed in both builds.
2. Save the final optimized LLVM IR immediately before
   `xpu.llvm.translate_to_asm`.
3. Normalize or ignore only non-semantic debug metadata such as generated
   source paths and PIDs. Confirm that all executable IR is identical.
4. Feed one selected LLVM IR file to each toolchain's own `llc`, using the same
   target triple, CPU, and flags. Produce both assembly and object files.
5. Convert each object with the matching toolchain's
   `xpu3-elfconv-triton`. Do not mix `llc` and `elfconv` installations in this
   initial comparison.
6. Compare toolchain versions, CodeGen library hashes, assembly, object files,
   and XPUBIN hashes. Assembly differences alone prove codegen divergence, not
   which output is correct.
7. Inject each XPUBIN into one otherwise identical runtime and execute the
   same fixed-seed correctness test. Override only the final XPUBIN-producing
   stage; do not recompile the kernel between runs.
8. Attribute the failure to a toolchain only when one injected binary passes
   and the other deterministically fails. Record exact commands, hashes,
   hardware, driver/runtime versions, mismatch count, and test identifier.
9. If `elfconv` remains a possible cause, cross the matrix: convert each object
   with both toolchains' `elfconv` and inject all four XPUBINs. If correctness
   follows the object producer, the defect is in LLVM code generation; if it
   follows the converter, investigate XPUBIN packaging instead.

If the LLVM IR differs semantically, stop at step 3 and investigate the
frontend or pass pipeline instead. If both injected binaries behave the same,
continue with ELF conversion, loading, launch ABI, or runtime investigation.

## Implementation Policy

- Keep the FlagTree pass pipeline aligned with internal Triton, including BF16
  vectorization.
- Do not add dtype-specific or global vectorization bypasses for this issue.
- Treat a different final assembly as evidence requiring device execution, not
  as sufficient proof of a compiler defect.
- Reuse the same-IR, dual-toolchain, injected-XPUBIN experiment for future
  suspected XPU backend miscompilations.
