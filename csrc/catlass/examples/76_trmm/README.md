# trmm

## 功能说明

本样例演示基于 CATLASS GEMM kernel 组件实现 TRMM（triangular matrix multiply）：

- `side = 0`：`C = alpha * op(A) * B`，其中 `A` 为三角矩阵。
- `side = 1`：`C = alpha * A * op(B)`，其中 `B` 为三角矩阵。
- `uplo = 0` 表示 lower triangular，`uplo = 1` 表示 upper triangular。
- `trans = 0` 表示不转置三角矩阵，`trans = 1` 表示转置三角矩阵。

当前样例使用 `float32` 输入输出，`diag = 0`（non-unit diagonal）。

## 构建

```bash
bash scripts/build.sh 76_trmm
```

## 运行

命令格式：

```bash
./output/bin/76_trmm m n side uplo trans diag alpha [device_id]
```

示例：

```bash
./output/bin/76_trmm 256 512 0 0 0 0 1.0
./output/bin/76_trmm 512 256 1 1 1 0 1.0
```

程序会在 host 侧构造三角矩阵与 dense 矩阵，运行 NPU kernel，并与 CPU reference 结果做精度比对。
