# flagtree.patch

Root-tree adaptations the spacemit backend needs on top of FlagTree main.
Applied at install time by `scripts/install_flagtree_plugin.sh`.

## What's in it

| File | Why |
|------|-----|
| `CMakeLists.txt` | Gate NVPTX/AMDGPU target libs on `LLVM_TARGETS_TO_BUILD` so the RISC-V-only prebuilt LLVM links cleanly. |
| `include/triton/Dialect/Triton/IR/TritonAttrDefs.td` | Add `PAD_NEG_INF` / `PAD_INF` to the padding-option enum. |
| `python/src/ir.cc` | Bind the new `PADDING_OPTION` enum values. |
| `python/triton/compiler/code_generator.py` | Recognize `smt.parallel` iterator; propagate `bind_sub_block` attr. |
| `python/triton/language/{semantic,standard}.py` | `neg_inf`/`inf` padding branches. |
| `python/triton/runtime/interpreter.py` | Interpreter support for the new padding options. |
| `python/setup_tools/utils/spacemit.py` | **New file.** Build hooks: cmake args, `get_package_data` (ships `driver.c` + `bin/` + `lib/` + `include/` in the wheel), `install_extension` (copy `spine-triton-opt`). |
| `setup.py` | Spacemit plugin path: build against root triton via `TRITON_PLUGIN_DIRS`, hide `FLAGTREE_BACKEND` from CMake, skip CUDA dep download, gate UT off. |
| `third_party/proton/Dialect/*.td` | LLVM22 dialect fixes (dropped `let llvmOp` field). |
| `third_party/tle/{CMakeLists.txt,dialect/include/IR/TleDialect.td}` | LLVM22 compat. |

## Application: 3-way fallback

The install script tries, in order:

1. `git apply -R --check` — already applied, skip.
2. `git apply --check` — clean forward apply.
3. `git apply --3way` — **3-way merge fallback.** Uses blob hashes in the
   patch's `index` line to merge around context drift. This is what makes
   the patch resilient to FlagTree main updates: small changes in
   surrounding lines no longer require a regen.
4. Fail with a regenerate hint.

Conflict markers (`<<<<<<<`/`=======`/`>>>>>>>`) are checked after a 3-way
apply; if present, the script errors out so they're not silently shipped.

## When to regenerate

The 3-way fallback handles **context drift** (surrounding lines changed)
but not **semantic drift** (the lines the patch itself touches changed).
Regenerate when:

- A hunk's target lines no longer exist (function renamed, block deleted).
- The patch semantically conflicts with a main change (e.g. main reworked
  the same `CodeGenerator.visit_For` block).
- CI fails on patch application even after 3-way.

## How to regenerate

From the worktree root (spacemit-backend-plugin branch):

```bash
# 1. Apply the current patch (resolves what it can, leaves .rej for the rest)
git apply --reject third_party/spacemit/patch/flagtree.patch

# 2. Manually apply the rejected hunks (see *.rej files)
#    Edit each file to add the spacemit changes around the new main code.

# 3. Clean up reject files
find . -name '*.rej' -delete

# 4. Regenerate the patch (intent-to-add new files first)
git add -N python/setup_tools/utils/spacemit.py   # if it's a new file
git diff > third_party/spacemit/patch/flagtree.patch

# 5. Revert the working tree
git checkout -- CMakeLists.txt include/ python/ setup.py \
                 third_party/proton/ third_party/tle/
rm -f python/setup_tools/utils/spacemit.py

# 6. Verify the new patch applies cleanly
git apply --check third_party/spacemit/patch/flagtree.patch

# 7. Local smoke build
source ~/triton/bin/activate
SPINE_MLIR_INSTALL_DIR=... SPINE_RUNTIME_INSTALL_DIR=... \
  bash third_party/spacemit/scripts/install_flagtree_plugin.sh
```

## Tips for a resilient patch

- **Small hunks beat big ones.** Each hunk's context is a fingerprint; the
  fewer lines it spans, the less likely main drifts inside it.
- **Don't touch lines you don't need to.** If a hunk only needs to add 2
  lines, don't reformat the surrounding 10.
- **Prefer `git diff` output.** Hand-edited patches lose blob hashes, which
  breaks the `--3way` fallback.
