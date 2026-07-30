```bash
cd /.../FlagTree
git apply third_party/spacemit/patch/flagtree.patch
FLAGTREE_BACKEND=spacemit \
LLVM_SYSPATH=/.../llvm-f6ded0be897e2878612dd903f7e8bb85448269e5-build-x86-release \
TRITON_BUILD_PROTON=OFF \
MAX_JOBS=2 \
pip install . --no-build-isolation -v
```