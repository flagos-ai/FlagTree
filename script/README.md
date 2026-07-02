# FlagTree Ascend 后端编译步骤

## 步骤 1：克隆仓库并设置环境变量

```bash
git clone --recursive https://github.com/KernelLLM/FlagTree
export FLAG_TREE_DIR=/mnt/data01/yuansheng/workspace/flagos/CommonIR/FlagTree
cd $FLAG_TREE_DIR/third_party
git clone https://github.com/kateyijian/flir
git checkout origin/CommonIR
```

## 步骤 2：执行构建脚本

```bash
bash $FLAG_TREE_DIR/script/flagtree_build.sh
```

## 步骤 3：运行测试

```bash
python $FLAG_TREE_DIR/test/CommonIR/fa_triton_arch.py --dump-mlir=$FLAG_TREE_DIR/test/CommonIR/fa_triton.mlir
```

## 步骤 4：测试算子

```bash
python $FLAG_TREE_DIR/test/CommonIR/fa_triton_arch.py --dump-linalg=fa_linalg.mlir
export TLE_REPLACE_IR_FILE=fa_linalg.mlir
python dummy_fa_kernel.py
```
