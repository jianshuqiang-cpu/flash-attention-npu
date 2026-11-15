# D512 实现说明与设计必要补充（实现工程师记录）

对应设计文档 `DESIGN_FA3_950_D512.md`，本次实现（Phase 1，未运行 NPU）。

## 所有权说明

- 按设计清单修改 11 个生产文件 + 新增 `fused_task.h`。
- **设计必要补充（主审认可）**：`init_outputs.hpp` 加入所有权。
  原因：空区间（emptySpan）输出初始化原固定偏移 `OUTPUT_UB_OFFSET=6*32768=196608`，
  D512 一个 AIV 最多 64 行零输出需要 `64*512*2=65536` B，结束于 262144，超出
  每 AIV 248 KiB 预算；原 LSE 固定偏移也落在新 D512 GO 范围内。新增
  `InitOutputs950(Resource&, const fa950_plan::UbPlan&)` 重载：D512 计划
  （`ub.lseBytes != 0`）使用 plan.go / plan.lse；旧计划保持原固定偏移，行为不变。
  DMA 防护（MTE3_V 6/7）与 PIPE_ALL 收尾原样保留。

## 关键实现决策

1. **D512 plan 单独选出**（`fwd_resource_plan.h` 的 `MakeUbPlanD512()`）：
   UB=248064 B，LO 32768 B（64×128 FP32），GO 131072 B（64×512 FP32），
   P UB 单槽（16640 B，pStages=1），DM 三槽，LSE scratch 256 B，无 mask。
   `UbPlan` 新增 `goBytes/lseBytes/pStages` 字段；旧计划 goBytes==loBytes、
   lseBytes=0、pStages=2，全部旧 static_assert 与 Host 测试保持通过。
   `KernelUbPlan(512,128)` 返回 D512 plan；L1 公式 D512=491520 B。

2. **Q tile 选择**：`SelectQueryTile/QueryTile/QueryTileForTasks` 对精确
   D512 一律返回 128，避开 Q64 旧固定 UB layout；旧维度分支不变
   （legacy Host 测试 `test_flash_attn_npu_v3_950_resource_plan.cpp` 通过）。

3. **共享几何**（新 `fused_task.h`，仅 D512 使用）：
   `MakeShape`：fullHeadTile=1（完整 Q 块单 head 任务）、
   `tailHeadTile = LegacyHeadTile(r, G, 128)`、`totalTasks = fullTasks + tailHeadTasks`；
   不含跨 KV group 融合、segment、fixedLayout。调用方统一：
   Host `FillSplitCoreTilingData`、AICPU `ComputeFAMetadata`、
   AICore batch 扫描与 decode、Host `FillFlashDecodeTilingData` base-task 枚举、
   FD combine `RunUniform`（D512 时用共享几何 decode，避免旧 large-D head cap
   与主 kernel 几何不一致）。FD 形状域 q≤16 时新旧公式数值相等，
   Host 测试穷举验证 task 级等价。

4. **位置式 mask**：从 0867a03 摘取 `ComputeScaleAndMaxPositions(/64)` 与
   位置式 operator()，D512 专用（P 单槽、固定 MTE3_V(3) 事件，与 Score
   双槽解耦）；causal/SWA 的 tile 边界判定与窗口裁剪保留当前逻辑，仅 mask
   机制换成寄存器比较；causal KV 裁剪改用 qSBlockSize（tail 有效长度）。
   SWA left/right 裁剪到 Sk-1/Sq-1（与 donor 一致；在有效位置上等价于
   无限窗口）。两处 Host `MakeDeviceTriuMask` 对 D512 跳过。

5. **PV 四片交付**：`pv_matmul`（AIC）每 128 列片 FixPipe 前等两个 AIV free、
   后发 ready（不再只首片等/末片发）；LO 目的列偏移恒 0。`rescale_o`
   （AIV）新增 `SubCoreComputeD512`：4 轮 ready/free（零行 AIV 同样 4 轮），
   首片等 MTE3_V(4)（GO DMA 防护），逐片 FP32 更新 GO[:, c:c+128]
   （首 tile 复制，否则 dm*GO+LO），4 片完成后统一 GL 归一化 →
   zeroRowCount 清零 → 整块 Cast → 原有 scatter/LSE/partial 路径。
   FD partial 维持 FP32。新 VF：`CopyPieceD512/RescalePieceD512/NormalizeD512`
   （LO stride 128 / GO stride 512，独立）。无列片 in-place 融合 Cast。
   `operator()` 尾部新增 `bool d512Mode=false`（在 vDequantScale 之后），
   普通路径显式传 `1.0f, d512`；FP8 调用点仍只传 vDequantScale，语义不变。

6. **P UB 单槽事件**：无 mask 路径新增 `pStages_` 判断——D512 用 P 槽 0 与
   固定 MTE3_V(3)（init 已有 Set、release 已有 Wait，配对数量一致）；
   旧路径仍按 ubSBufId 旋转。K/V 单槽沿用事件 1/3（loop 内仅触达 1/3，
   init/release 的 1,2,3,4 保持成对）。PV 通道 5 复用原 flag：
   D512 每 KV tile 4 次 ready/free（otmpStages=1，不触达通道 6），
   init 预置一次、release 等待 5/21，余量与旧协议同形。

7. **Q copy**：`loadQGM` 仅 `embed==512 && qNBlockSize==1` 走整块 strided
   copy（与 67de7a4/bd74814 donor 一致，L1 pitch-1 布局 + 2 参 ND2NZ）；
   D≤256 与 qN>1 保持原多 ND copy 分支。L1→L0A 读取只使用布局外层 stride
   （row 256 / col 2048），与 qSBlockSize 参数无关；该组合的 NPU 证据来自
   donor 分支验证，本实现保留其精确形态并限制在 D512。

8. **FD combine**：`MAX_HEAD_DIM=512`；保留 32 KiB partial 输入窗口
   （8 行 D512 每轮 2 个 split，8×512 FP32 accumulator 16 KiB）；
   新增 static_assert 实际末端容量（tile 累加器、D512 行块、总 UB 预算）。
   输入批次按实际 D 计算（原逻辑），未改名义常量。
   `RunUniform` 对 headDim==512 使用共享几何 decode（见 3）。

9. **接口**：`mha_fwd.cpp` 仅放开精确 (512,512)；257..511 与不对称组合
   仍被 [1,256] 检查拒绝；两处 mask 缓存创建对 D512 跳过；enableDN 仍限
   ≤256（D512 不进入 DN）；FP8 128 限制不变。

## 验证记录

- Phase 1（Host/编译，详见 PHASE1_REPORT.md）：
  - Host：`tests/test_flash_attn_npu_v3_950_d512_host.cpp`（g++ 编译并运行通过）：
    D512 plan 偏移/容量/对齐、旧 plan 兼容、tile 分支、资源字段与设备 L1
    地址推演、共享几何穷举（7494200 query/head 元素，无重复/遗漏/跨组，
    物理行 ≤128）、已知用例、batch 扫描一致性、FD 形状域 task 级等价、
    旧几何回归。
  - 旧 Host 测试 `test_flash_attn_npu_v3_950_resource_plan.cpp` 与
    `test_attention_window_950_host.cpp` 重跑通过（旧行为未破坏）。
  - CPU 参考器自检 6/6（exit 0）；950 v3 完整构建 exit 0（.so SHA256
    `8d6e4030…c3992`）。
- Phase 2（NPU 功能，详见 PHASE2_REPORT.md）：
  - 最小 smoke 8/8；D512 API 全套 168/168（exit 0）。
  - 旧 forward 回归：q128 108、migration 前向 91、FP8 125、graph 7、
    init_outputs 26、stride 16 通过；17 个旧套件失败定性为既有行为差异
    （证据与最小复现保留在 Phase2 报告 §4），不计入 D512 结果。
  - 未运行 sanitizer；无性能结论。
