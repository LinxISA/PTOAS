# PTOAS CV Preload 优化设计

## 1. FA CV 分离 Kernel：为什么需要 Preload

PTOAS 的 CV preload 优化首先要解决 FA 这类已经手写 C/V 分离的 kernel。以本仓库 `fa_perf.pto` 的形态为例，FA 的四个主要 stage 分布在两个 kernel 中：

| stage | kernel | 主要计算 | TPipe |
| --- | --- | --- | --- |
| `compute_qk` | Cube | `Q * K^T` matmul，产生 QK | `tpush id=25`，C2V |
| `compute_p` | Vector | softmax/exp，产生 P | `tpop id=25`，再 `tpush id=30`，V2C |
| `compute_pv` | Cube | `P * V` matmul，产生 PV | `tpop id=30`，再 `tpush id=27`，C2V |
| `compute_gu` | Vector | 累加 PV 并做最终归一化 | `tpop id=27` |

这个结构与 pto-isa manual FA README 中的描述一致：手写 FA kernel 将计算拆成 `compute_qk`、`compute_p`、`compute_pv`、`compute_gu`，通过 inter-CV FIFO 连接 Cube 和 Vector pipeline；README 也用 `qkPreloadNum` 表示 QK 预执行深度，并让 QK/P/PV FIFO 深度随 preload 增加。README 的 tuning notes 中，`qkPreloadNum` 默认 4，`qkp_tile_fifo_size` 按 `1 + qkPreloadNum` 派生，`pv_tile_fifo_size` 也镜像这个深度。参考：[pto-isa flash_atten README](https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/common/flash_atten/README.md)。

### 1.1 优化前：按同一个逻辑 tile 串行推进

未做 preload 时，虽然 Cube 和 Vector 已经分成两个 kernel，但同一个 S1 tile 的跨核依赖仍然形成如下链条：

```text
Cube   compute_qk(i)  -> push QK(i)
Vector pop QK(i)      -> compute_p(i)  -> push P(i)
Cube   pop P(i)       -> compute_pv(i) -> push PV(i)
Vector pop PV(i)      -> compute_gu(i)
```

用 PTOAS 的 CV 分离 IR 可以抽象成：

```mlir
func.func @cube_kernel(...) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  scf.for %i = %lb to %ub step %step {
    // QK producer: Cube -> Vector, pipe id=25
    %qk = pto.talloc_to_aiv {id = 25, split = 0}
    ... compute_qk(%i) ...
    pto.tstore ... outs(%qk ...)
    pto.tpush_to_aiv(%qk) {id = 25, split = 0}

    // PV relay: Vector -> Cube -> Vector, pipe id=30 then id=27
    %p = pto.tpop_from_aiv {id = 30, split = 0}
    ... compute_pv(%i, %p) ...
    pto.tfree_from_aiv(%p) {id = 30, split = 0}
    %pv = pto.talloc_to_aiv {id = 27, split = 0}
    pto.tstore ... outs(%pv ...)
    pto.tpush_to_aiv(%pv) {id = 27, split = 0}
  }
}

func.func @vector_kernel(...) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  scf.for %i = %lb to %ub step %step {
    // P relay: Cube -> Vector -> Cube, pipe id=25 then id=30
    %qk = pto.tpop_from_aic {id = 25, split = 0}
    ... compute_p(%i, %qk) ...
    %p = pto.talloc_to_aic {id = 30, split = 0}
    pto.tstore ... outs(%p ...)
    pto.tfree_from_aic(%qk) {id = 25, split = 0}
    pto.tpush_to_aic(%p) {id = 30, split = 0}

    // GU consumer: Cube -> Vector, pipe id=27
    %pv = pto.tpop_from_aic {id = 27, split = 1}
    ... compute_gu(%i, %pv) ...
    pto.tfree_from_aic(%pv) {id = 27, split = 1}
  }
}
```

这种写法的问题不是功能错误，而是 C/V 资源会在等待对端 stage 时产生空泡：

```text
tile i:
  Cube   QK(i)  ---- wait P(i) ---- PV(i) ----
  Vector ---- wait QK(i) ---- P(i) ---- wait PV(i) ---- GU(i)
```

TPipe 的 `tpush/tpop/tfree/talloc` 已经能保证跨核同步正确，但如果 compiler 不改变 scope 的 logical iteration，两个核仍然倾向于围绕同一个 tile 做握手，难以让 QK、P、PV、GU 同时处于稳态流水中。

### 1.2 优化后：scope 级 preload 展开

preload 的核心是把四个 stage 标成不同的 `preload_num`，让 producer 在更靠前的 logical tile 上运行，让 consumer drain 更旧的 logical tile：

| scope | kernel | role | TPipe | `preload_num` |
| --- | --- | --- | --- | --- |
| `S3_QK` | Cube | producer | `push id=25` | 3 |
| `S2_P` | Vector | relay | `pop id=25 -> push id=30` | 2 |
| `S1_PV` | Cube | relay | `pop id=30 -> push id=27` | 1 |
| `S0_GU` | Vector | consumer | `pop id=27` | 0 |

`max_preload_num = 4` 时，`pto-cv-create-preload` 将原 loop 扩成 physical loop，并为每个 scope 使用不同的 stage IV：

```text
stage_iv(p) = physical_iv - (max_preload_num - 1 - p) * step
```

展开后的概念写法如下。实际 IR 中 Cube 和 Vector 仍是两个函数，且每个函数内部保持原 loop body 的词法顺序。

```mlir
// cube_kernel
scf.for %t = %lb to (%ub + 4 * %step) step %step {
  %i_pv = %t - 2 * %step
  scf.if (%lb <= %i_pv && %i_pv < %ub) {
    // S1_PV: consume P(i_pv), produce PV(i_pv)
    %p = pto.tpop_from_aiv {id = 30, split = 0}
    ... compute_pv(%i_pv, %p) ...
    pto.tfree_from_aiv(%p) {id = 30, split = 0}
    %pv = pto.talloc_to_aiv {id = 27, split = 0}
    pto.tpush_to_aiv(%pv) {id = 27, split = 0}
  }

  %i_qk = %t
  scf.if (%lb <= %i_qk && %i_qk < %ub) {
    // S3_QK: produce QK(i_qk)
    %qk = pto.talloc_to_aiv {id = 25, split = 0}
    ... compute_qk(%i_qk) ...
    pto.tpush_to_aiv(%qk) {id = 25, split = 0}
  }
}

// vector_kernel
scf.for %t = %lb to (%ub + 4 * %step) step %step {
  %i_gu = %t - 3 * %step
  scf.if (%lb <= %i_gu && %i_gu < %ub) {
    // S0_GU: consume PV(i_gu)
    %pv = pto.tpop_from_aic {id = 27, split = 1}
    ... compute_gu(%i_gu, %pv) ...
    pto.tfree_from_aic(%pv) {id = 27, split = 1}
  }

  %i_p = %t - %step
  scf.if (%lb <= %i_p && %i_p < %ub) {
    // S2_P: consume QK(i_p), produce P(i_p)
    %qk = pto.tpop_from_aic {id = 25, split = 0}
    ... compute_p(%i_p, %qk) ...
    %p = pto.talloc_to_aic {id = 30, split = 0}
    pto.tstore ... outs(%p ...)
    pto.tfree_from_aic(%qk) {id = 25, split = 0}
    pto.tpush_to_aic(%p) {id = 30, split = 0}
  }
}
```

稳态下，同一个 physical iteration 会同时推进多个 logical tile：

```text
physical t:
  Cube   PV(t - 2)                    QK(t)
  Vector GU(t - 3)                    P(t - 1)
```

这样 Cube 可以提前生产未来 tile 的 QK，Vector 可以消费上一批 QK 生成 P，Cube 又消费更旧的 P 生成 PV，Vector drain 最旧的 PV。TPipe 内部同步仍然负责“数据是否真的可读/slot 是否可复用”，compiler 只负责把不同 scope 映射到不同 logical iteration，并为 local/workspace buffer 准备足够的 stage 存储。

### 1.3 优化效果和必要性

pto-isa manual FA README 用无软件流水、`qkPreloadNum=2/4` 以及仿真 pipeline 的对比说明了同一个方向：preload 的直接目标是解除跨 stage 数据依赖带来的串行化，让 Vector 计算资源尽量保持忙碌；当流水形态形成后，瓶颈会进一步暴露到 Cube 侧 TSTORE、FIFO 深度、UB/L1 buffer 等资源上。

preload 的效果可以从四个层面理解：

- 减少跨核等待空泡：`tpush/tpop` 的等待不消失，但等待更容易被另一个 stage 的计算覆盖。
- 保持 C/V 资源忙碌：QK、P、PV、GU 不再围绕同一个 tile 串行握手，而是在不同 tile 上形成稳态流水。
- 暴露内存规划需求：同一 physical iteration 同时存在多个 logical tile 的中间结果，`plan-memory` 必须为 preload local buffer 分配多份地址，并由 create-preload 按 stage 轮转。
- 受资源约束：更大的 preload 需要更深的 FIFO、更多 UB/L1 local buffer 和更多 workspace slot。manual FA 中 `qkPreloadNum` 会影响 QK/P/PV FIFO 深度，PTOAS 的 `max_preload_num` 必须受相同资源约束。

因此，PTOAS 的 CV preload pass 不只是“复制 loop body”。它需要把 TPipe transaction 识别成 scope，生成跨 C/V 一致的 `preload_num`，让 plan-memory 看到 multibuffer 需求，最后再做 loop 扩界和 scope 展开。

## 2. 目标和边界

本文描述 PTOAS 中 CV 分离代码的第一版 preload 优化设计，重点覆盖四件事：

1. 如何从 TPipe 通信识别 CV scope。
2. 如何根据 C/V 核之间的 producer/consumer 关系生成 `preload_num`。
3. `plan-memory` 如何消费 preload 标注并影响 local buffer 地址规划。
4. 如何参考 NPU IR `CreatePreload.cpp` 展开已标记的 preload loop。

设计参考：

- `C:\Users\rdp\Documents\AscendNPU-IR\bishengir\lib\Dialect\HIVM\Transforms\CreatePreload.cpp`
- `C:\Users\rdp\Documents\AscendNPU-IR\bishengir\lib\Dialect\HIVM\Transforms\PlanMemory.cpp`

本文中的 `preload` 表示 CV scope/stage 级提前执行，不限定为单个 buffer load；相关 buffer 标注和 memory plan 只是支撑这种 stage 提前执行的实现手段。

PTOAS 输入中 C 核和 V 核函数已经分离，跨核通信由用户显式写成 `talloc`、`tpush`、`tpop`、`tfree`。这些 TPipe API 内部天然包含跨核同步动作。因此，PTOAS 的 CV 优化不需要重新发明跨核同步原语，但必须把这些同步动作作为 scope 边界和 preload 调度约束的一部分。

V1 不做 subtiling，不拆分 pipe entry，不改变用户定义的 TPipe 粒度。当前实现已经完成自动 scope 识别、`preload_num` 生成、no-result `pto.cv.scope` 标注，以及 opt-in 的 `pto-cv-create-preload` 展开。create-preload 默认关闭，只有打开 `--enable-cv-create-preload` 时才让 scope 保留到 plan-memory 之后并执行展开。

## 3. IR 标注模型

第一版使用一个编译器内部的 no-result `pto.cv.scope` op 表达 CV preload scope。它是单 block region 容器，不带 operand/result，也不带 `yield` terminator。这个选择和 PTOAS 的输入形态有关：FA 这类 CV 分离 kernel 的跨 scope 数据主要通过 TPipe FIFO、外提 tilebuf/memref 或 workspace 传递，而不是通过 tensor SSA result 在 scope 之间直接传值。

```mlir
pto.cv.scope {
  pto.tpop(...)
  ...
  pto.tpush(...)
  pto.tfree(...)
} {
  pto.cv.scope_id = 7,
  pto.cv.group_id = 0,
  pto.cv.preload_num = 2,
  pto.cv.max_preload_num = 4,
  pto.cv.core = "vector",
  pto.cv.role = "relay"
}
```

因此，第一版 `pto.cv.scope` 有两个约束：

- scope 内定义的 SSA value 不允许被 scope 外直接使用。若一个值需要跨 scope 使用，要么把定义外提到 scope 之外，要么通过 TPipe/workspace/local buffer 这样的显式 side effect 传递。
- scope 间的 producer/consumer 关系不依赖 SSA def-use，而是由 `pto.cv.input_pipe`、`pto.cv.output_pipe`、`pto.cv.group_id`、`pto.cv.preload_num` 等 metadata 保留。

这与 NPU IR 的 tensor-level `scope.scope -> result` 方案不同。NPU IR 在 bufferization 前用 scope result 和 `scope.return` 保留跨 region tensor SSA 数据流；PTOAS 第一版选择 no-result scope，是因为 tile 多为 tilebuf/memref 或 TPipe borrowed entry，且可要求普通 tile alloc 外提。未来如果出现 scope 内生成的 SSA tile 必须被外部直接使用，再扩展 `pto.cv.scope` 的 result/yield 形式。

关键标注：

| 标注 | 语义 |
| --- | --- |
| `pto.cv.scope_id` | 单个 scope 的稳定编号，用于诊断和调试。 |
| `pto.cv.group_id` | 同一个 preload 展开组。一个 group 对应一个可展开的 `scf.for` pipeline。 |
| `pto.cv.preload_num` | scope 在展开 loop 中的 stage 编号。数值越大，逻辑迭代越靠前。 |
| `pto.cv.max_preload_num` | 当前 preload group 的 stage 数。 |
| `pto.cv.core` | `cube` 或 `vector`。 |
| `pto.cv.role` | `producer`、`consumer` 或 `relay`。 |
| `pto.cv.input_pipe` | scope 消费的上游逻辑 pipe，例如 `c2v:25`；producer scope 为空。 |
| `pto.cv.output_pipe` | scope 生产的下游逻辑 pipe，例如 `v2c:30`；consumer scope 为空。 |

buffer 侧需要额外标注：

| 标注 | 语义 |
| --- | --- |
| `pto.multi_buffer` | 该 root local buffer 需要多份物理存储。CV preload 自动标注时数值取当前跨 C/V stage 链的 `max_preload_num`。 |
| `pto.cv.preload_workspace` | workspace/subview 需要按 stage 改写 slot 维度。 |

当前实现中，`pto-cv-auto-mark-multi-buffer` 在 `pto-cv-mark-preload-scopes` 之前运行，复用同一套 TPipe stage 链识别逻辑，把 scope 内使用到的根 `pto.alloc_tile` / `memref.alloc` 标上 `pto.multi_buffer`。`PTOViewToMemref` 会把 `pto.alloc_tile` 上的标注透传到降低后的 `memref.alloc`、`pto.pointer_cast`、`pto.bind_tile`，其中非 level3 的 plan-memory 路径继续从 `memref.alloc` 读取该属性。

当前 `pto-cv-mark-preload-scopes` 生成 no-result `pto.cv.scope` 以及 scope 级 `pto.cv.*` 标注；`pto-cv-create-preload` 继续消费这些稳定语义，在 plan-memory 生成多地址 local buffer 后做 loop 扩界、guard 插入和 stage 地址旋转。

## 4. Scope 识别

### 4.1 TPipe 动作语义

TPipe op 不只是普通 IO，它们也是同步动作：

| op | producer/consumer | 同步含义 | scope 边界含义 |
| --- | --- | --- | --- |
| `talloc` | producer | 等待并获取空闲 FIFO slot | producer transaction 开始。 |
| `tpush` | producer | 提交数据，通知 consumer 可读 | producer scope 的提交边界。 |
| `tpop` | consumer | 等待 producer 数据可读并获取 slot/tile | consumer scope 的获取边界。 |
| `tfree` | consumer | 释放已消费 slot，允许 producer 复用 | consumer scope 的释放边界。 |

因此，scope 不应该只按普通计算 op 切分，而要按 TPipe transaction 的生命周期切分：

- producer scope：从产生待发送数据的计算开始，到对应 `tpush` 结束。若是 global entry，scope 内还包含 `talloc -> tstore -> tpush`。
- consumer scope：从 `tpop` 开始，到最后一次使用 pop 出来的 tile/entry 并执行对应 `tfree` 结束。
- relay scope：同一核内既消费上游 pipe，又生产下游 pipe，并且中间值或 pop 出来的 entry 不能在两个 scope 间安全切开时，使用一个 relay scope 覆盖从 `tpop` 获取到 producer `tpush` 提交和 consumer `tfree` 释放都完成的整段 transaction。`tpush` 与 `tfree` 的相对顺序保持用户 IR 的词法顺序。

`tpush` 和 `tpop` 是最重要的边界：`tpush` 是 producer 的 commit 边界，`tpop` 是 consumer 的 acquire 边界。但同一核内同时存在 producer 和 consumer 时，不能简单用所有 `tpush/tpop` 机械切段，还必须看 borrowed value 的生命周期和数据依赖。

第一版实现把识别出的 stage 作为一个连续 op range 包进 `pto.cv.scope`，scope 必须覆盖“产生/消费该 pipe 数据的计算”，而不是只覆盖 `talloc/tstore/tpush` 或 `tpop/tfree` 这几条 TPipe op：

- producer-only：从上一条 CV 边界之后的本 stage 计算开始，到匹配 `tpush` 结束。若 `talloc` 写在计算之后，scope 仍然要包含前面的 load/matmul/arith 等 producer 计算。
- consumer-only：从 `tpop` 开始，到该 stage 的尾部结束；如果 `tfree` 后还有使用已拷贝到 local buffer 的计算或 store，也属于同一个 consumer scope。
- relay：从上游 `tpop` 开始，到下游 `tpush` 提交且上游 `tfree` 释放都完成为止，中间包含必要的计算、`tfree`、`talloc` 和 `tpush`。`tpush` 与 `tfree` 的相对顺序保持用户 IR 的词法顺序。

因为当前 `pto.cv.scope` 不带 result/yield，wrap 前必须检查该 range 内定义的 SSA value 没有逃逸到 scope 外。若存在逃逸，说明该 scope 需要带 result/yield 或需要把定义外提，第一版 pass 不应强行 wrap，否则会破坏 SSA dominance。

### 4.2 单核内 producer/consumer 共存

同一个 C/V kernel loop 内可能同时有 producer scope 和 consumer scope。识别规则如下：

1. 从每个 TPipe op 建立 transaction：
   - producer transaction：同一 logical pipe 上支配 `tpush` 的 `talloc`，以及写入该 entry/tile 的计算。
   - consumer transaction：`tpop` 产生的 tile/entry，到匹配 `tfree` 之间的所有使用。
2. 如果 consumer 的输出已经物化到普通 local buffer，且该 local buffer 不依赖 pop 出来的 borrowed entry 生命周期，可以切成 consumer scope 和 producer scope。
3. 如果 producer 直接或间接依赖 `tpop` 得到的 tile/entry，或把 consumer/producer 切开会破坏 borrowed entry 生命周期、FIFO 提交/释放顺序、数据依赖，则合并成 relay scope。
4. 一个 relay scope 可以同时含有多个 pipe 动作，但它对 preload 展开来说是不可再拆的 stage 单元。

例子：

```mlir
%qk = pto.tpop_from_aic {id = 25}
%p = pto.softmax(%qk)
%p_entry = pto.talloc_to_aic {id = 30}
pto.tstore ... outs(%p_entry ...)
pto.tfree_from_aic(%qk) {id = 25}
pto.tpush_to_aic(%p_entry) {id = 30}
```

这里 vector 先消费 C2V 的 QK，再生产 V2C 的 P。如果 `%p` 的计算依赖 `%qk`，且把 consumer 与 producer 分成两个 scope 会丢失这段 transaction 的生命周期和词法顺序，则这段应标成一个 relay scope，而不是在 `tpop` 和 `tpush` 之间强拆。

### 4.3 跨 C/V 核配对

跨核关系必须以 logical pipe 为单位建立，而不是以 SSA value 为单位。logical pipe key 建议包含：

```text
(pipe_id, direction, split, entry_kind, slot_shape, dtype)
```

其中 direction 是 C2V 或 V2C：

- C2V producer：cube kernel 中的 `talloc_to_aiv/tpush_to_aiv`。
- C2V consumer：vector kernel 中的 `tpop_from_aic/tfree_from_aic`。
- V2C producer：vector kernel 中的 `talloc_to_aic/tpush_to_aic`。
- V2C consumer：cube kernel 中的 `tpop_from_aiv/tfree_from_aiv`。

识别 pass 为每个 loop 内的 TPipe 动作建立 occurrence：

```text
CVPipeOccurrence {
  func,
  core,
  parent_loop,
  loop_iv,
  lexical_index,
  logical_pipe,
  op_kind,      // talloc/tpush/tpop/tfree
  role,         // producer/consumer
  scope_id
}
```

同一 logical pipe 上 producer occurrence 和 consumer occurrence 按 FIFO 顺序配对。配对时要同时检查：

- producer 的 `tpush` 和 consumer 的 `tpop` 类型、shape、dtype、split 一致。
- consumer 的 `tfree` 和 producer 的 slot 复用关系合法。
- 同一个 loop 内不同 pipe 的 occurrence 顺序被记录下来，后续生成 `preload_num` 时不能丢失。
- 包成 no-result `pto.cv.scope` 后，scope 间原本可能存在的 SSA def-use 不再作为主要依赖来源；跨 C/V pipeline 依赖由 `input_pipe/output_pipe` 建边，loop 内 side-effect 顺序由 scope 的词法顺序保留。

## 5. Preload Num 生成

### 5.1 基本语义

PTOAS 的 `preload_num` 语义与 NPU IR `CreatePreload.cpp` 对齐：

```text
stage_iv(preload_num p) =
  physical_iv - (max_preload_num - 1 - p) * step
```

因此：

- `preload_num = max_preload_num - 1` 表示最靠前的 stage，使用当前 physical iteration。
- `preload_num = 0` 表示最靠后的 drain stage，使用最旧的 logical iteration。
- 数值越大，scope 越“提前”执行。

以 `max_preload_num = 4` 为例，展开后同一个 physical iteration 中的逻辑迭代关系是：

| `preload_num` | logical iv |
| --- | --- |
| 3 | `physical_iv` |
| 2 | `physical_iv - step` |
| 1 | `physical_iv - 2 * step` |
| 0 | `physical_iv - 3 * step` |

这点非常重要：`preload_num` 不是简单的词法顺序编号，而是 scope 在 pipeline 中相对 drain stage 的提前距离。

### 5.2 依赖图

自动生成 `preload_num` 时，先建立 scope 级图：

```text
CVScopeNode {
  scope_id,
  core,
  parent_loop,
  lexical_range,
  role,
  logical_pipes,
  reads,
  writes,
  tpipe_actions
}
```

图上包含四类边：

| 边 | 来源 | 含义 |
| --- | --- | --- |
| data edge | SSA/use-def、memref alias、tile/entry 使用 | producer 的结果必须先于 consumer 使用。 |
| tpipe ready edge | `tpush -> tpop` | producer 提交的数据被对端消费。 |
| tpipe release edge | `tfree -> talloc` | consumer 释放 slot 后 producer 才能安全复用。 |
| lexical/lifetime edge | 同一 loop 内 TPipe occurrence 顺序和 borrowed value 生命周期 | 保持用户显式写出的 transaction 顺序。 |

其中 `tpipe ready edge` 是生成 preload stage 的主要依据。若希望 producer 为 consumer 预取下一轮数据，则 producer scope 的 `preload_num` 应比该 consumer scope 大 1。对一条链：

```text
producer -> relay -> relay -> consumer
```

可以得到：

```text
producer.preload_num = 3
relay1.preload_num   = 2
relay2.preload_num   = 1
consumer.preload_num = 0
max_preload_num      = 4
```

### 5.3 生成算法

对每个可展开的 loop group：

1. 收集 C/V 两侧 parent loop 中的 CV scope。
2. 按 logical pipe FIFO 顺序配对 producer/consumer occurrence。
3. 构建 scope dependency graph。
4. 找到 drain scope。drain scope 是该 pipeline 中最靠后的 consumer，通常是产生当前 loop 可见最终结果或释放最后一个输入 token 的 scope。
5. 从 drain scope 反向沿 data/tpipe ready edge 计算 stage distance。
6. 设置：

```text
preload_num(scope) = stage_distance_from_drain(scope)
max_preload_num    = max(preload_num) + 1
```

7. 对同一个 scope 中多个 pipe 动作取最大 stage distance，避免一个 scope 被分到多个 stage。
8. 检查同一 parent loop 内是否存在相同 `preload_num` 的多个不可交换 side-effect scope。若存在，需要合并 scope 或放弃自动标注。
9. 检查 `preload_num` 不超过可用 FIFO 深度或用户指定的 preload 深度。若需要的 stage 数大于 pipe slot 数，不能自动开启 preload。

### 5.4 词法顺序的影响

preload 展开 pass 不会重排原 loop body，它只在原词法顺序上为不同 scope 映射不同 logical iv。因此，PTOAS 生成 `preload_num` 时必须保留 loop 内不同 producer/consumer TPipe 的出现顺序。

例如 FA 的 CV 分离代码中可以抽象出以下跨核链：

```text
Cube:   qk_push(id=25)
Vector: qk_pop(id=25) -> p_push(id=30)
Cube:   p_pop(id=30)  -> pv_push(id=27)
Vector: pv_pop(id=27)
```

对应推荐标注：

| scope | core | role | `preload_num` |
| --- | --- | --- | --- |
| QK producer | Cube | producer | 3 |
| QK consumer + P producer | Vector | relay | 2 |
| P consumer + PV producer | Cube | relay | 1 |
| PV consumer | Vector | consumer | 0 |

如果 Cube loop 中 `p_pop/pv_push` 出现在 `qk_push` 之前，标注仍然可以是 `1` 后接 `3`。展开 pass 会保持这个词法顺序：

```text
physical iteration t:
  Cube P/PV relay uses logical iteration t - 2 * step
  Cube QK producer uses logical iteration t
```

这正是需要记录 loop 内 TPipe 顺序的原因。不能简单按 `preload_num` 对 scope 排序，否则会改变用户写出的 FIFO transaction 顺序，可能破坏 TPipe 内部同步协议。

### 5.5 第一版实现策略

当前第一版已经先落地三段前置标注逻辑：

1. `pto-cv-auto-mark-multi-buffer`
   - 在创建 `pto.cv.scope` 前扫描 C/V kernel 中的 TPipe transaction。
   - 复用 producer / relay / consumer stage 链识别，计算与 scope pass 一致的 `max_preload_num`。
   - 追溯 scope 内 tile/memref operand 的 root local alloc，给 `pto.alloc_tile` 或 `memref.alloc` 写入 `pto.multi_buffer = max_preload_num`；已有用户标注不覆盖。

2. `pto-cv-mark-preload-scopes`
   - 自动识别 scope，生成 no-result `pto.cv.scope`。
   - 生成 `group_id`、`preload_num`、`max_preload_num`、`role`、`input_pipe`、`output_pipe` 等 scope metadata。

3. `pto-inline-cv-preload-scopes`
   - 过渡期在 frontend pipe lowering 之后把 no-result scope 展回父 block，保证未接入 preload 展开时不影响现有 codegen。

后续还需要补独立的 `pto-cv-verify-preload-marks`，把 scope/preload 标注的一致性诊断从 create-preload 中拆出。`pto-cv-create-preload` 已经先实现第一版，用于将已标注 scope 展开成 stage-level preload loop。

## 6. Plan Memory 中 preload 标注的影响

NPU IR 的 `PlanMemory.cpp` 对 preload 的关键处理可以概括为三步：

1. `annotation.mark` 上的 multibuffer 属性记录到 `buffer2MultiNum`。
2. `preload_local_buffer` / preload local alloc 被加入 `preloadBuffers`，生命周期分析在 parent `for` 上显式生成 gen/kill。
3. memory plan 为 `multiBufferNum > 1` 的 storage entry 扩展多个地址，后续 pointer cast 携带地址数组。

PTOAS 需要保留同样的契约。

### 6.1 Local buffer 必须变成 multibuffer

被多个 preload stage 同时使用的 local buffer 不能继续只有一个物理地址。否则展开后不同 logical iteration 的中间数据会互相覆盖。

规则：

- scope 内使用到、且参与 CV preload stage 链的 root local buffer 标注 `pto.multi_buffer = max_preload_num`。
- 对 high-level tile IR，标注先挂在 `pto.alloc_tile` 上；`PTOViewToMemref` 负责把它透传到降低后的 `memref.alloc` 或显式地址路径的 `pto.pointer_cast` / `pto.bind_tile`。
- plan-memory 看到 `memref.alloc {pto.multi_buffer = N}` 后，为它生成 N 个可轮转地址。
- create-preload 在展开时按 `preload_num` 旋转这些地址。

### 6.2 生命周期分析的特殊处理

普通 local buffer 的 gen/kill 可以由 MLIR liveness 直接得到；preload buffer 不行，因为展开前 IR 里只有一个 loop iteration，看不到多个 stage 同时活跃。

因此，plan-memory 需要像 NPU IR 一样做特殊处理：

- 识别所有带 `pto.multi_buffer` 的 root alloc 及其 alias。
- 普通 op 的 gen 逻辑遇到 preload buffer 时先跳过，避免把它当作单 iteration buffer。
- 在包含 preload scope 的 parent `for` 上补充 gen/kill，使该 buffer 的生命周期覆盖整个 preload loop。
- alias buffer 必须一起进入 gen/kill，避免 subview/cast 绕过保护。

这样 memory plan 才不会把不同 stage 的地址错误复用给其他 live buffer。

### 6.3 StorageEntry 扩展

plan-memory 为每个 gen buffer 建立 storage entry。若该 buffer 有 `pto.multi_buffer = N`：

- 原 entry 保留第 0 个地址。
- 额外创建 `N - 1` 个等价 storage entry。
- 地址规划完成后，`buffer2Offsets[buffer]` 中包含 N 个 offset。
- lowering 到 pointer cast 或等价 PTOAS 地址表达时，需要保留这 N 个地址。

create-preload 依赖这个地址数组做 stage 旋转。如果 plan-memory 没有提前扩展，preload 展开只能复制计算，不能保证 local storage 正确。

### 6.4 Workspace 和 TPipe FIFO

TPipe FIFO 的 slot 管理由 TPipe API 负责，不应该被 plan-memory 当成普通 local temp 复制。特别是 consumer 侧的 reserved FIFO buffer 不是 preload local scratch，除非明确标注，否则不能按 stage duplicate。

但有两类 buffer 仍需要处理：

- 编译器生成的 local scratch：如果跨 stage 活跃，按 `pto.multi_buffer = max_preload_num` 处理。
- workspace/subview：如果需要由不同 stage 访问不同 workspace slot，标注 `pto.cv.preload_workspace`，create-preload 展开时改写 slot 维度。

workspace slot 的推荐计算方式：

```text
slot = ((stage_iv - loop_lb) / loop_step) % max_preload_num
```

NPU IR 参考实现中使用的是 `stage_iv / step % max_preload_num`，隐含 loop lower bound 已规范化到 0。PTOAS 若不能保证 loop lb 为 0，应使用带 `loop_lb` 的形式，或在 create-preload 前先规范化 loop。

### 6.5 建议 pass 顺序

建议第一版 pipeline：

```text
frontend TPipe IR
  -> pto-cv-auto-mark-multi-buffer
  -> pto-cv-mark-preload-scopes / pto-cv-verify-preload-marks
  -> lower frontend pipe ops, preserve or consume cv scope before codegen
  -> view/tile bufferization
  -> pto-plan-memory
  -> pto-resolve-reserved-buffer
  -> pto-cv-create-preload
  -> canonicalize/cse
  -> inline/remove cv scope
  -> existing sync/codegen passes
```

关键点是：`pto-cv-auto-mark-multi-buffer` 必须在 `pto-cv-mark-preload-scopes` 之前运行，因为它要在 high-level `pto.alloc_tile` 还没有被 scope 包裹和 lowering 前标注 root local buffer；`pto-plan-memory` 必须在 `pto-cv-create-preload` 之前看到 `pto.multi_buffer` 并生成多地址；`pto-cv-create-preload` 必须在最终 codegen 前完成 scope 展开和地址旋转。

当前 pipeline 中，`pto-cv-auto-mark-multi-buffer` 会生成 `pto.multi_buffer`，`pto-cv-mark-preload-scopes` 会生成显式 `pto.cv.scope`。默认路径仍在 frontend pipe lowering 之后运行 `pto-inline-cv-preload-scopes`，保证未开启 preload 展开时不影响现有 codegen；开启 `--enable-cv-create-preload` 时，scope 会保留到 `pto-plan-memory` / `pto-resolve-reserved-buffers` 之后，再由 `pto-cv-create-preload` 展开，随后才 inline/remove scope 容器。

## 7. Preload 展开逻辑

### 7.1 输入约束

`pto-cv-create-preload` 的输入是已标注 IR：

- 每个要展开的 scope 都在某个 `scf.for` 的直接 body 中，或第一版先限制为直接 body。
- 同一 parent `for` 下的 preload scope 具有相同 `group_id` 和 `max_preload_num`。
- 单个 parent `for` 内 `preload_num` 唯一且在 `[0, max_preload_num)`；允许缺失某些 `preload_num`，缺失 stage 表示该 kernel loop 在该 stage 没有 scope。
- 跨 C/V group 的 `preload_num` 全集应覆盖 `[0, max_preload_num)`，否则 pipeline 不完整。
- scope 内的 TPipe transaction 完整，不跨 scope 泄漏 borrowed entry，relay scope 除外。
- 非 scope 的 side-effect op 第一版不允许夹在 preload group 中，除非 pass 明确知道如何按 stage clone。

### 7.2 收集 PreloadInfo

对每个 parent `scf.for` 建立：

```text
PreloadInfo {
  max_preload_num,
  lb,
  ub,
  step,
  original_iv,
  mappings[max_preload_num],
  scopes[preload_num]
}
```

其中 `mappings[p]` 表示原 IR value 在 `preload_num = p` stage 中对应的新 value。

### 7.3 改写 loop 边界

参考 NPU IR `CreatePreload.cpp`，新 loop 的 exclusive upper bound 是：

```text
new_ub = old_ub + max_preload_num * step
```

注意这里是 `max_preload_num * step`，不是 `(max_preload_num - 1) * step`。因为 `scf.for` upper bound 是 exclusive，最后一个实际 physical iv 是：

```text
old_ub + (max_preload_num - 1) * step
```

新 loop 的每个 stage IV：

```text
stage_iv[p] = physical_iv - (max_preload_num - 1 - p) * step
```

stage 的有效条件：

```text
old_lb <= stage_iv[p] && stage_iv[p] < old_ub
```

如果前置 canonicalize 已把 `old_lb` 规范成 0，可以退化成 NPU IR 里的 `0 <= stage_iv && stage_iv < old_ub`。

### 7.4 改写 loop-carried value

preload local buffer 的 loop-carried 参数要特殊处理：

- 如果 init arg 来自带 `pto.multi_buffer` 的 pointer cast/alloc，不再作为新 loop 的普通 init arg。
- 展开时为每个 `preload_num` clone 一个旋转后的 pointer cast，并写入对应 mapping。
- no-result `pto.cv.scope` 不返回 preload local buffer；若 scope 内使用该 buffer，展开时通过 stage mapping 找到旋转后的 pointer cast。
- 普通 loop-carried value 仍按原 `scf.for` 规则 yield，推荐使用最高 stage mapping，即 `mappings[max_preload_num - 1]`。

### 7.5 改写 local buffer 地址

plan-memory 产生的 pointer cast 携带 N 个地址。create-preload 对不同 stage 做旋转：

```text
new_addrs[(max_preload_num - preload_num - 1 + i) % max_preload_num]
  = old_addrs[i]
```

直观理解：

- 高 `preload_num` stage 使用更靠前的 ring slot。
- 低 `preload_num` stage 使用更旧的 ring slot。
- 物理 loop 每走一步，logical iteration 在 ring buffer 上自然前进。

这个旋转必须只应用于带 `pto.multi_buffer` 且属于 preload local buffer 的地址，不能应用于 TPipe 内部 FIFO reserved buffer。

### 7.6 改写 workspace/subview

若 op 带 `pto.cv.preload_workspace`：

1. clone 原 `memref.subview` 或等价 view op。
2. 将第 0 维 offset 改为 stage slot：

```text
slot = ((stage_iv - lb) / step) % max_preload_num
```

3. 若新 view type 与原 use type 不完全一致，可临时插入 adaptor cast。
4. 后续 canonicalize/adaptor propagation 消除临时 cast。

### 7.7 改写 scope

对每个 no-result preload scope：

```mlir
scf.if (%cond_for_stage) {
  pto.cv.scope {
    // clone old scope body with mappings[preload_num]
  }
}
```

clone scope body 时：

- 普通 op 用 `mappings[preload_num]` clone。
- nested `scf.for` 递归 clone body。
- 带 `pto.multi_buffer` 的 preload local pointer cast 使用旋转后的地址。
- `preload_workspace` subview 使用 stage slot。
- `talloc/tpush/tpop/tfree` 保持在原 scope 内，不重新排序。
- relay scope 保持整体 clone，不在展开阶段拆开。

第一版不支持 scope result/yield，所以 create-preload 不从 `pto.cv.scope` 产出 `scf.if` result。需要跨 scope 使用的值必须满足以下之一：

- TPipe 数据由 `tpush/tpop` 的 FIFO side effect 传递，依赖关系由 `input_pipe/output_pipe` 和 `group_id` 表示。
- 普通 tilebuf/memref alloc 在 scope 外定义，scope 内只读写其内容。
- workspace/subview 由 `preload_workspace` 逻辑按 stage slot 重新生成。
- loop-carried index 或其它纯 SSA value 仍然按外层 `scf.for` 的 `iter_args/scf.yield` 规则处理，不通过 `pto.cv.scope` 返回。

如果未来需要让 scope 内定义的 SSA tile 直接给外部使用，应扩展 `pto.cv.scope` 的 result/yield 形式，并像 NPU IR `scope.scope` 一样在展开时把 result 映射回各 stage mapping。

### 7.8 非 scope op

NPU IR 参考实现会对非 scope op 按每个 stage clone。PTOAS 第一版建议更保守：

- 允许 constants、index arithmetic、pure shape/view op 按 stage clone。
- 允许带 `pto.multi_buffer` 或 `preload_workspace` 标注的地址/view op 由专门逻辑 clone。
- 不允许未知 side-effect op 出现在 preload group 中。若发现，pass 报错并要求前置 scope 识别把它纳入某个 scope。

这样可以避免复制 TPipe 或 memory side effect 导致语义变化。

### 7.9 展开后的清理

展开后运行：

- adaptor propagation。
- CSE。
- canonicalize。
- inline/remove `pto.cv.scope`。
- verifier 检查 TPipe transaction 和 buffer alias。

## 8. 诊断和限制

pass 应给出明确诊断：

- scope 缺少 `preload_num` 或 `max_preload_num`。
- 跨 C/V group 的 `preload_num` 全集不连续，或单个 parent loop 内 `preload_num` 重复。
- scope 不在支持的 parent `scf.for` 形态下。
- C/V logical pipe 无法配对，或 shape/dtype/split 不一致。
- preload local buffer 缺少 `pto.multi_buffer`，或数量小于 `max_preload_num`。
- TPipe borrowed entry 生命周期跨出 scope。
- 未知 side-effect op 出现在 preload group 中但不属于任何 scope。
- 推导出的 `max_preload_num` 超过 FIFO slot 深度或用户指定上限。

第一版限制：

- 不做 subtiling。
- 不跨函数重排。
- 不自动改变 TPipe `id/split/slot_size`。
- 不自动合并不同 parent loop 的 preload group。
- 不支持无法证明 loop step/lb/ub 兼容的 C/V loop group。

## 9. 实现和测试计划

当前实现状态：

1. 已实现 `pto-cv-auto-mark-multi-buffer`，在 CV scope 标注前自动生成 `pto.multi_buffer`。
2. 已实现 `pto-cv-mark-preload-scopes`，自动识别 scope 并生成 `preload_num` / `max_preload_num`。
3. 已实现过渡期 `pto-inline-cv-preload-scopes`，保持当前 codegen pipeline 不被 no-result scope 影响。
4. 已实现 plan-memory 对 CV preload multi-buffer 的 loop 级生命周期处理：带 `pto.multi_buffer` 且被 preload scope 使用的 local buffer 在 parent `scf.for` 上生成/释放，scope 内普通 gen/kill 不再切断它。
5. 已实现 opt-in `pto-cv-create-preload`，完成无 `iter_args` loop 的扩界、stage condition、scope clone 和 local `pto.pointer_cast` 多地址旋转。
6. 待实现 `pto-cv-verify-preload-marks`。
7. 待实现 workspace/subview 的 `pto.cv.preload_workspace` slot 改写。

测试建议：

- 已有一个最小两 stage C2V preload IR，验证 `new_ub = old_ub + N * step`、stage condition、plan-memory 产生 N 个地址，以及 create-preload 按 `preload_num` 旋转。
- 一个 workspace subview 标注测试，验证第 0 维 slot 改写。
- 已有一个 FA 风格四 stage 样例，覆盖 C/V 分离、relay scope、同一核内 producer/consumer 共存、自动 `pto.multi_buffer` 标注，以及 loop 内 TPipe 词法顺序和 `preload_num` 顺序不一致的场景。
- 已有 FA level3 显式地址负例：若 local buffer 只有 1 个 planned address，即使带 `pto.multi_buffer` 标注，create-preload 也会报错而不是静默展开。
- 后续负例测试：重复 `preload_num`、缺少 `max_preload_num`、scope 外 borrowed entry、未知 side-effect op。
