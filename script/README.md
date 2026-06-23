# FlagTree Ascend 后端编译步骤

## 步骤 1：进入工程目录并设置环境变量

```bash
export FLAG_TREE_DIR=/mnt/data01/yuansheng/workspace/flagos/CommonIR/FlagTree/script
cd $FLAG_TREE_DIR
```

## 步骤 2：执行构建脚本

```bash
bash /mnt/data01/yuansheng/workspace/flagos/CommonIR/FlagTree/script/flagtree_build.sh
```

## 步骤 3：运行测试

```bash
python /mnt/data01/yuansheng/workspace/flagos/CommonIR/FlagTree/test/CommonIR/fa_triton_arch.py --dump-dir=$FLAG_TREE_DIR/test/CommonIR/fa_triton.mlir
```