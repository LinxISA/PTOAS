# GraphSyncSolver Buf-ID 同步形态设计

本文描述在 `PTOGraphSyncSolver` 上扩展一种可选的"Buf-ID"同步发射形态，
让用户在保留现有 `set_flag` / `wait_flag` 路径的前提下，按需切换到 A5 的
`get_buf` / `rls_buf` 编程模型。

---

## 1. 背景

### 1.1 当前形态：`set_flag` / `wait_flag`

PTOAS 现有的两个自动同步 pass（`PTOInsertSync` 和 `PTOGraphSyncSolver`）最终都
落到一对 `pto.set_flag` / `pto.wait_flag` 指令：

- `pto.set_flag[src_pipe, dst_pipe, event_id]` 在 producer 之后；
- `pto.wait_flag[src_pipe, dst_pipe, event_id]` 在 consumer 之前；
- 同 pipe 内部依赖用 `pto.barrier pipe`；
- ODS 在 [PTOOps.td:2271](../../include/PTO/IR/PTOOps.td) 一带。

这种形态要求**严格配对**——set 跟着 src 端的 pipe 指令走、wait 跟着 dst 端的
pipe 指令走，两者位置可以分别处于不同 scope。在动态 shape、循环可空、
条件分支不对称等场景下，求解器需要复杂的外提 / 合并优化来保证配对正确。

### 1.2 新形态：Buf-ID（A5）

A5 新增一组指令 `get_buf` / `rls_buf`，把"配对"语义从指令级别下沉到硬件
scoreboard：

```
get_buf(pipe, #id)   // 在 pipe 指令之前取号
<pipe op>            // 正常执行
rls_buf(pipe, #id)   // 在 pipe 指令之后叫号
```

硬件为每个 `id` 维护一对 global 计数器 `(get_cnt, rel_cnt)`：

- `get_buf` 进发射队列前：抢号 `get_cnt = global_get_cnt`，然后 `global_get_cnt++`；
- 被 bracket 起来的指令必须等到 `get_cnt == global_rel_cnt` 才能发射；
- 指令结束后由 `rls_buf` 推进 `global_rel_cnt++`。

不同 pipe 用相同 id 时，硬件按抢号顺序串行化这些 pipe 的 bracket。

ODS 与 EmitC 路径**已经存在**：
- ODS：[PTOOps.td:2327](../../include/PTO/IR/PTOOps.td)（`pto.get_buf`） /
  [PTOOps.td:2357](../../include/PTO/IR/PTOOps.td)（`pto.rls_buf`）；
- EmitC：[PTOToEmitC.cpp:4903](../../lib/PTO/Transforms/PTOToEmitC.cpp)
  （`PTOGetBufToEmitC`） / [PTOToEmitC.cpp:4933](../../lib/PTO/Transforms/PTOToEmitC.cpp)
  （`PTORlsBufToEmitC`）。

**目前没有 pass 会自动生成 `get_buf`/`rls_buf`**——只有手写测试样例
（[test/samples/Sync/test_a5_buf_sync.pto](../../test/samples/Sync/test_a5_buf_sync.pto)）。

### 1.3 编程约束（来自 hw 规格）

1. 同一 (pipe, id) 的 `get_buf` 与 `rls_buf` 必须配对，且顺序为先 get 后 rel；
   连续两次 get 同 id 不合法（会导致 rel 推进异常）。
2. 不同 pipe 共享同 id 时，每个 pipe 的 get/rel 必须**连续**，中间不能插入
   其他 pipe 的同 id get/rel。
3. buf-id 只用于**跨 pipe 同步**。
4. 多个 pipe 对的同步**不建议**复用同一个 id（性能损失，硬件正确性不受影响）。

### 1.4 A5 不需要 pipe barrier

buf-id 是 A5 特性。**A5 硬件天然保证同 pipe 顺序执行**，所以同 pipe 真依赖
**完全不需要**显式 barrier 指令——即 `pto.barrier` 在 A5 上不会发射。

这一点现有求解器内核已部分体现：[SyncSolver.cpp::handleBarrierConflict](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)
里对 `isRegBasedArch` （即 A5）针对 PIPE_V / PIPE_M 提前 return：

```cpp
if (options.isRegBasedArch) {
  if (corePipeSrc.pipe == pto::PIPE::PIPE_V ||
      corePipeSrc.pipe == pto::PIPE::PIPE_M) {
    return;
  }
}
```

buf-id 模式（仅 A5）下进一步扩展：**所有 pipe 的 barrier 都跳过**。详见 3.4 节。

---

## 2. 总体方案

### 2.1 目标

在 `PTOGraphSyncSolver` 上加一个**用户可选的发射形态**：
- 默认 `set-wait`：行为与现状完全一致；
- 可选 `buf-id`：在 A5 上把每对真依赖落成 `get_buf` / `rls_buf` 的 bracket；
- 两种形态共存，**求解器内核不动**（依赖识别、event-id 染色、外提 / 合并、
  barrier-all 回退），只在最后两步（in-memory IR 构造 + codegen 发射）分支。

### 2.2 关键复用点

| 阶段 | 复用程度 |
| --- | --- |
| `IRTranslator`：MLIR → 求解器 IR | 完全复用 |
| `Solver`：冲突检测 / `processOrders` / `processConflict` | 完全复用 |
| `EventIdSolver`：图染色分配 id | 完全复用（buf-id 池与 event-id 池同构） |
| `getBeforeAfterSyncMaps`：ConflictPair → 内存 SyncOp | 末段分支 |
| `SyncSolverIR`：`SetWaitOp` / `BarrierOp` | 新增 `GetBufOp` / `RlsBufOp` 类 |
| `CodeGenerator::emitSyncOp`：内存 SyncOp → MLIR | 新增 case 分支 |

### 2.3 形态对照

对单个真依赖 `(producer @ Psrc, consumer @ Pdst)`，分配 N 个 id（N≥1）后两种
形态的发射方式：

| | set-wait | buf-id |
| --- | --- | --- |
| producer 上方 | — | `get_buf[Psrc, #id_k]` ×N |
| producer 下方 | `set_flag[Psrc, Pdst, #id_k]` ×N | `rls_buf[Psrc, #id_k]` ×N |
| consumer 上方 | `wait_flag[Psrc, Pdst, #id_k]` ×N | `get_buf[Pdst, #id_k]` ×N |
| consumer 下方 | — | `rls_buf[Pdst, #id_k]` ×N |
| 同 pipe 真依赖 | `pto.barrier P` | **不发射任何指令**（A5 同 pipe 硬件保序，见 1.4） |

N 来自 `EventIdInfo::eventIdRepeatNum`，用于 multibuffer 流水的 id 轮转，
两种形态语义等价。

### 2.4 buf-id 不需要 backward-sync 外提

这是 buf-id 相对 set/wait 的核心优势之一，专门说明一下避免理解错位。

set/wait 模型下，求解器对循环间依赖（backward sync）必须做形态一 / 形态二
的转换：

- **形态一（in-loop + 守卫）**：每轮迭代里 `if i>0 wait`、`if i<N-1 set`，
  循环体内有条件分支；
- **形态二（out-of-loop 补偿）**：循环前补一个 `set`、循环后补一个 `wait`，
  循环体内成对 set/wait。

这两种形态存在的根本原因是：set/wait 是"指令级强配对"——每条 wait 必须有
恰好一条 set 与之匹配，N 次迭代里多/少一条都会卡死。
求解器里 `tryMovingOutBackwardSyncPairsToOuterLoops` /
`mergeBackwardSyncPairs` / `considerOuterBackwardSyncPairs` 这些
优化（[SyncSolver.h:407-413](../../include/PTO/Transforms/GraphSyncSolver/SyncSolver.h)）
就是在处理这类不对称配对。

**buf-id 是顺序编程模型**，硬件 scoreboard 的 `(get_cnt, rel_cnt)` 自身
就是计数器——
producer 每轮 bracket 一次推进一格，consumer 每轮 bracket 一次消费一格，
N 轮迭代自然匹配 N 轮 bracket。
唯一需要的写法就是 doc 1.2 给出的:

```text
for i = 0:N
  get_buf(MTE2, #id); MTE2(); rls_buf(MTE2, #id)   // producer 自然 bracket
  get_buf(V,    #id); Vector(); rls_buf(V,    #id) // consumer 自然 bracket
```

**结论：buf-id 模式下 backward ConflictPair 不需要单独发 bracket，但 backward 分析本身不能省**——它要用来决定 forward pair 之间是否要**复用同一个 id**。

#### 单 buffer vs 多 buffer 的 id 分配

考虑 doc §1.2 / §1.3 两个 case：

**Case 1（双 buffer，可并行）**：
```text
for i = 0:N
  load(gm -> ub0)   # MTE2, write ub0
  vadd(ub0 -> ub1)  # V,    read ub0 write ub1
  store(ub1 -> gm)  # MTE3, read ub1
```
两条 forward dep 走两个不同 buffer：load/vadd 在 ub0 上同步，vadd/store 在 ub1 上同步。两条链应该用**不同的 id** 以获得最大并行度（下一轮的 load 不需要等本轮的 store 完成）。

**Case 2（单 buffer，必须串行）**：
```text
for i = 0:N
  load(gm -> ub0)
  vadd(ub0 -> ub0)
  store(ub0 -> gm)
```
所有 forward dep 都走 ub0。这种情况下下一轮的 load 写 ub0 必须等本轮的 store 读完 ub0——必须**复用同一个 id**，让计数器单调性串起整条 load→vadd→store→next-load 链。

如果两条 forward 用不同 id（像 case 1 那样），iter i+1 的 load（在 id_A 上）只会等 iter i 的 vadd（也在 id_A 上），不会等 iter i 的 store（在 id_B 上）——结果 store 读 ub0 时被 load 覆写，data race。

#### 落实

1. `tryMovingOutBackwardSyncPairsToOuterLoops`、`considerOuterBackwardSyncPairs`、
   `mergeBackwardSyncPairs` 这些 set/wait 的 backward 外提路径在 buf-id 模式下**直接关掉**。
2. `backwardSyncEvents` / `backwardSyncEventsAfterMerge` 不会被填充。
3. 在 `getBeforeAfterSyncMaps` emit 时，对所有 forward ConflictPair 做一个 **buffer-aware union-find 预处理**：
   - 用 backward 边作为"同 buffer 桥"。
   - 同一连通分量的 forward pair 共享一个 buf id（取分量内最小的 id 作代表）。
   - 每个 `(anchor_op, pipe, id)` 三元组只发一对 bracket（dedup）——多个 ConflictPair 落到同一 (op, pipe, id) 上时合并。
4. `conflictPair->isInnerBackward` 为 true 的 pair 不发 bracket（只参与 step 3 的 union-find 桥）。

具体落实见 3.6 节。

### 2.5 不在本期范围

- **`set_flag_dyn` / `wait_flag_dyn`（动态 event id）的 buf-id 对位**。
  当前求解器不输出动态形态，暂不考虑。

---

## 3. 实现细节

### 3.1 ODS / Dialect 层

**基本不动**。两个 op 已就位：

```tablegen
def GetBufOp : PTO_Op<"get_buf"> {
  let arguments = (ins
    PTO_PipeEventTypeLikeAttr:$op_type,   // 高级 op 类型（映射到 PIPE）
    I32Attr:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );
  let hasVerifier = 1;
}
def RlsBufOp : PTO_Op<"rls_buf"> { /* 同 */ }
```

**建议增量**（可后置，不阻塞主流程）：在 `Op::verify()` 里加上轻量本地校验，
检查 `op_type` 能映射到一个具体 pipe（已在 EmitC 端校验，前移到 verifier
更友好）。本设计不依赖这一项。

### 3.2 Pass option 与 CLI

#### 3.2.1 `Passes.td`

`PTOGraphSyncSolver` 增加一个选项（[Passes.td:65](../../include/PTO/Transforms/Passes.td)）：

```tablegen
let options = [
  Option<"eventIdNumMax", "event-id-num-max", "int64_t",
         /*default=*/"8", "...">,
  Option<"syncStyle", "sync-style", "std::string",
         /*default=*/"\"set-wait\"",
         "Sync emission style: 'set-wait' (default) or 'buf-id' (A5 only).">,
];
```

#### 3.2.2 `tools/ptoas/ptoas.cpp`

在现有 [ptoas.cpp:193](../../tools/ptoas/ptoas.cpp)（`enableGraphSyncSolver`）旁加：

```cpp
static llvm::cl::opt<std::string> graphSyncSolverSyncStyle(
    "graph-sync-solver-sync-style",
    llvm::cl::desc("Sync emission style for graph sync solver: "
                   "'set-wait' (default) or 'buf-id' (A5 only)."),
    llvm::cl::init("set-wait"));
```

在 [ptoas.cpp:1173](../../tools/ptoas/ptoas.cpp) 透传到 pass option：

```cpp
PTOGraphSyncSolverOptions graphSyncOpts;
graphSyncOpts.eventIdNumMax = graphSyncSolverEventIdMax;
graphSyncOpts.syncStyle = graphSyncSolverSyncStyle;
pm.addNestedPass<func::FuncOp>(createPTOGraphSyncSolverPass(graphSyncOpts));
```

#### 3.2.3 Pass 入口校验

[PTOGraphSyncSolver.cpp:51](../../lib/PTO/Transforms/GraphSyncSolver/PTOGraphSyncSolver.cpp)
把字符串解析成枚举塞进 `SyncSolverOptions`，并做 arch 兜底：

```cpp
SyncEmitStyle emitStyle = parseEmitStyle(syncStyle); // 字符串 → enum
const bool isA5 = pto::isTargetArchA5(func.getOperation());
if (emitStyle == SyncEmitStyle::BUF_ID && !isA5) {
  func.emitError("--graph-sync-solver-sync-style=buf-id requires --pto-arch=a5");
  return signalPassFailure();
}
SyncSolverOptions opts(SyncMode::INTRA_CORE_SYNC, !isA5, isA5);
opts.eventIdNumMax = eventIdNumMax;
opts.emitStyle = emitStyle;
```

### 3.3 `SyncSolverOptions` 扩展

[Utility.h:123](../../include/PTO/Transforms/GraphSyncSolver/Utility.h) 增补：

```cpp
enum class SyncEmitStyle { SET_WAIT, BUF_ID };

struct SyncSolverOptions {
  // 现有字段不变 ...
  SyncEmitStyle emitStyle{SyncEmitStyle::SET_WAIT};
};
```

ctor 内可按 emitStyle 调整若干默认值（保守起见在 buf-id 下关掉跨 pipe-pair
的 id 复用以符合约束 4 的"建议"）：

```cpp
if (emitStyle == SyncEmitStyle::BUF_ID) {
  reuseSyncPairToSaveEventIds = false; // 约束 4：不同 pipe 对不共享 id
}
```

### 3.4 求解器内核：核心保持，关掉 backward-sync 外提族

`Solver::solve()` 全链路（[SyncSolver.cpp](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)）：

**保留不变**：
- `processConflict` / `handleSetWaitConflict` / `handleUnitFlagConflict`；
- `EventIdSolver` 的染色——buf-id pool 与 event-id pool 在 hw 上同源、容量
  同步（默认 8，由 `eventIdNumMax` 控制）；
- `reuseSyncPairToSaveEventIds`（在 set-wait 下默认开，buf-id 下默认关——
  约束 4 的"不同 pipe 对不建议共享 id"）。

**buf-id 模式下关掉的部分**：
- 同 pipe barrier 整体跳过（A5 硬件保序，见 1.4 节）。
  `handleBarrierConflict` 入口直接 return；`pickAndInsertABarrierAll` 不
  会触发 barrier 路径——但仍作为 event-id 耗尽时的兜底保留（buf-id 池耗尽
  时 fallback 到 PIPE_ALL 强同步形态需要 hw 团队确认；阶段一沿用现有 fallback
  机制，到 emit 阶段把 `BarrierOp(PIPE_ALL)` 翻译成 buf-id 等价形态或保留
  set/wait 形态作为应急）。
- backward-sync 外提族（按 2.4 节的理由）：
  - `tryMovingOutBackwardSyncPairsToOuterLoops`
    （[SyncSolver.h:413](../../include/PTO/Transforms/GraphSyncSolver/SyncSolver.h)）；
  - `considerOuterBackwardSyncPairs`
    （[SyncSolver.h:407](../../include/PTO/Transforms/GraphSyncSolver/SyncSolver.h)）；
  - `mergeBackwardSyncPairs` / `mergeBackwardSyncEventIds` /
    `insertMergedBackwardSyncPairs`
    （[SyncSolver.h:401-405](../../include/PTO/Transforms/GraphSyncSolver/SyncSolver.h)）。

具体实现方式见 3.6 节末尾。

**理由**：求解器输出的 `ConflictPair` 已经包含全部 emit 所需的信息：
`(setOp, waitOp, setCorePipeInfo, waitCorePipeInfo, eventIdNode->getEventIds())`。
buf-id 与 set-wait 的差异只是这套信息**怎么落成 IR**——外提优化是 set/wait
强配对的副产物，buf-id 不需要。

### 3.5 内存 SyncOp IR：新增类

[SyncSolverIR.h:67-91](../../include/PTO/Transforms/GraphSyncSolver/SyncSolverIR.h)
的 `OpType` enum 增补：

```cpp
enum struct OpType {
  // ...
  SYNC_OP,
    BARRIER_OP,
    SW_FLAG_OP,
      SET_FLAG_OP,
      WAIT_FLAG_OP,
    SW_FLAG_OP_END,
    BUF_OP,                  // 新增：buf-id 家族区间起点
      GET_BUF_OP,            // 新增
      RLS_BUF_OP,            // 新增
    BUF_OP_END,              // 新增
  SYNC_OP_END,
  // ...
};
```

类层级（与 `SetFlagOp`/`WaitFlagOp` 对称）：

```cpp
class BufOp : public SyncOp {
public:
  pto::PIPE pipe{pto::PIPE::PIPE_UNASSIGNED}; // 自身所在 pipe（src 或 dst）
  int64_t bufId{-1};                          // 复用 eventId 池
  BufOp(OpType opType, Operation *op, OperationBase *parentOp,
        pto::PIPE pipe, int64_t bufId)
      : SyncOp(opType, op, parentOp), pipe(pipe), bufId(bufId) {}
  static bool classof(const OperationBase *e) {
    return e->opType >= OpType::BUF_OP && e->opType < OpType::BUF_OP_END;
  }
};

class GetBufOp : public BufOp { /* OpType::GET_BUF_OP */ };
class RlsBufOp : public BufOp { /* OpType::RLS_BUF_OP */ };
```

`SetFlagOp` / `WaitFlagOp` **不删**——两条路径共存。

### 3.6 `getBeforeAfterSyncMaps` 分支

[SyncSolver.cpp:2216-2270](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)
在每个非 barrier / 非 unitFlag 的 ConflictPair 处理段加分支：

```cpp
if (conflictPair->isBarrier()) {
  if (options.emitStyle == SyncEmitStyle::BUF_ID) {
    // A5 同 pipe 硬件保序，不发射任何 barrier。
    continue;
  }
  // 现有 BarrierOp 路径，不变
} else if (options.emitStyle == SyncEmitStyle::SET_WAIT) {
  // 现有 SetFlagOp / WaitFlagOp 路径，不变
  // syncMapAfter[conflictPair->setOp].push_back(setOp);
  // syncMapBefore[conflictPair->waitOp].push_front(waitOp);
} else {
  // BUF_ID 路径
  auto setPipe = conflictPair->setCorePipeInfo.pipe;
  auto waitPipe = conflictPair->waitCorePipeInfo.pipe;
  for (int64_t bufId : conflictPair->eventIdNode->getEventIds()) {
    // producer 端 bracket（src pipe）
    syncMapBefore[conflictPair->setOp].push_back(
        make_unique<GetBufOp>(conflictPair->setOp->op,
                              conflictPair->setOp->parentOp, setPipe, bufId));
    syncMapAfter[conflictPair->setOp].push_back(
        make_unique<RlsBufOp>(conflictPair->setOp->op,
                              conflictPair->setOp->parentOp, setPipe, bufId));
    // consumer 端 bracket（dst pipe）
    syncMapBefore[conflictPair->waitOp].push_back(
        make_unique<GetBufOp>(conflictPair->waitOp->op,
                              conflictPair->waitOp->parentOp, waitPipe, bufId));
    syncMapAfter[conflictPair->waitOp].push_back(
        make_unique<RlsBufOp>(conflictPair->waitOp->op,
                              conflictPair->waitOp->parentOp, waitPipe, bufId));
  }
}
```

**约束 1/2 的天然满足**：bracket 形式下，`syncMapBefore[X]` 里某个 `get_buf(P, id)`
与 `syncMapAfter[X]` 里对应的 `rls_buf(P, id)` 之间隔着的只有 anchor op `X` 本身。
codegen 按 "before 顺序 push → 原 op → after 顺序 push" 的固定模式落盘，不会有
其他 (pipe, id) 的 bracket 插入到中间。

**backward ConflictPair 处理**：

在 `getBeforeAfterSyncMaps` 主循环的 `continue` 守卫加一条：

```cpp
if (options.isBufIdEmit() && conflictPair->isInnerBackward) {
  continue;
}
```

这样 backward ConflictPair 的 bracket 完全不发；其语义由 forward ConflictPair 的 bracket 通过计数器单调性间接覆盖。

注意：此 backward 仍然会被 solver 计数和分配 id（因为 EventIdSolver 早就跑完了），分配掉的 id 实际未使用——是浪费但不影响正确性。后续如果 id 池紧张，可以把这个过滤前移到 `processConflict`/`handleSetWaitConflict`，从源头就不创建 backward pair。

**backward-sync 路径处理**：

`getBeforeAfterSyncMaps` 末段以及 `runSolver` 内的下列调用，在 buf-id 模式下
通过提早 return / 开关位绕过：

| 调用点 | buf-id 行为 |
| --- | --- |
| `collectBackwardSyncEventIds` ([SyncSolver.cpp:1925](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)) | 直接 return，不收集 |
| `mergeBackwardSyncPairs` ([SyncSolver.cpp:2272](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp) 调用点) | 直接 return |
| `insertMergedBackwardSyncPairs` ([SyncSolver.cpp:2556](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp) 调用点) | 直接 return |
| `considerOuterBackwardSyncPairs` (`runSolver` 内调用) | 直接返回 `success`，不进入外提循环 |
| `tryMovingOutBackwardSyncPairsToOuterLoops` (`runSolver` 内调用) | 同上 |

实现上推荐统一通过 `if (options.emitStyle == SyncEmitStyle::BUF_ID) return …;`
前置守卫，集中在每个函数入口，避免散点改动。

结果是：buf-id 模式下，每个真依赖的 ConflictPair 的 `setOp`/`waitOp` 直接
等于求解器最初识别出的 producer / consumer 锚点，bracket 就在原位落，
对应 doc 1.2 buf-id 同步代码原样形态。

### 3.7 `CodeGenerator::emitSyncOp` 分支

[SyncSolverCodeGen.cpp:97](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolverCodeGen.cpp)
增加 case：

```cpp
void CodeGenerator::emitSyncOp(IRRewriter &rewriter, SyncOp *syncOp) {
  if (auto *barrier = dyn_cast<BarrierOp>(syncOp)) { /* 不变 */ }

  if (auto *getBuf = dyn_cast<GetBufOp>(syncOp)) {
    rewriter.create<pto::GetBufOp>(resolveSyncLoc(getBuf),
        pipeToPipeEventTypeAttr(rewriter.getContext(), getBuf->pipe),
        rewriter.getI32IntegerAttr(getBuf->bufId),
        rewriter.getI32IntegerAttr(0)); // mode 默认 0
    return;
  }
  if (auto *rlsBuf = dyn_cast<RlsBufOp>(syncOp)) {
    rewriter.create<pto::RlsBufOp>(resolveSyncLoc(rlsBuf),
        pipeToPipeEventTypeAttr(rewriter.getContext(), rlsBuf->pipe),
        rewriter.getI32IntegerAttr(rlsBuf->bufId),
        rewriter.getI32IntegerAttr(0));
    return;
  }

  auto *setWait = dyn_cast<SetWaitOp>(syncOp);
  if (!setWait || setWait->eventIds.empty()) return;
  // 现有 set-wait 发射路径不变
}
```

**`pipeToPipeEventTypeAttr` helper**：`pto.get_buf` 的 `op_type` 是
`PipeEventTypeLikeAttr`（高级 op 类型），不是 `PipeAttr`。EmitC 端已有
`mapSyncOpTypeToPipe`（[PTOToEmitC.cpp:4914](../../lib/PTO/Transforms/PTOToEmitC.cpp)）。
我们需要反向工具：`PIPE → PipeEventTypeAttr` 或 `PIPE → SyncOpTypeAttr`。
**实现位置**：放到 `include/PTO/IR/PTO.h` 或一个新的 `PTO/IR/SyncOpTypeMap.h`，
两个方向都暴露。

`setInsertionPoint` 已经支持 placeholder（loop 边界 / scope 起止）锚点，
buf-id 不需要新的锚点类型。

### 3.8 跨 pipe-pair 共享 id 必须分配冲突边（**正确性必须**）

> 这一节最初被列为"如果需要"的精化，实际上是 buf-id 的**硬正确性约束**。

set-wait 模式下，求解器按 (pipeSrc, pipeDst) 独立维护 `EventIdSolver` 实例
（[SyncSolver.cpp::getEventIdSolverRef](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)），
不同 pipe 对各自从 id=0 开始着色。这对 set/wait 是正确的——
`set_flag[MTE2,V,#0]` 和 `set_flag[V,MTE3,#0]` 落到两个不同的硬件 flag
寄存器（pipe 对组合是 hw 寻址的一部分），物理上隔离。

**buf-id 不行**。`get_buf[V,#0]` 不带"对方 pipe"参数，所以 `(MTE2→V)` 的
#0 和 `(V→MTE3)` 的 #0 在 hw 上是**同一个 scoreboard**。考虑下面 IR：

```text
tload   # PIPE_MTE2，pair1 setOp
tmuls   # PIPE_V，pair1 waitOp + pair2 setOp
tstore  # PIPE_MTE3，pair2 waitOp
```

按 set-wait 着色逻辑会给两个 pair 都分配 id=0。展开成 buf-id 形态：

```text
get_buf(MTE2, #0); tload; rls_buf(MTE2, #0)
get_buf(V,    #0);                              # pair1 consumer on tmuls
get_buf(V,    #0);                              # pair2 producer on tmuls (illegal!)
tmuls
rls_buf(V,    #0)
rls_buf(V,    #0)
get_buf(MTE3, #0); tstore; rls_buf(MTE3, #0)
```

`get_buf(V,#0)` 连续出现两次 → 违反 doc §1.4 约束 1（"连续两次 get 同 id
不合法"，会让 hw 计数器混乱）。

#### 修复方案

两处改动：

1. **`getEventIdSolverRef`**：buf-id 模式跟 cross-core 模式一样把 key 折成
   `(PIPE_UNASSIGNED, PIPE_UNASSIGNED)`，让**所有** pipe 对共享同一个
   `EventIdSolver`。所有 ConflictPair 都进入同一个染色图，避免不同 pipe
   对从同一 id pool 起点重复发牌。

2. **`checkIntersect`**：buf-id 模式下，两个 ConflictPair 只要共享**任一**
   pipe 就视为冲突（加冲突边），不再要求 (setPipe, waitPipe) 二元组
   完全相同。原因：buf-id bracket 锚在 RWOperation 上而不是 `(setOp, waitOp)`
   范围内的中间索引，两个 pair 各自在共享的 pipe 上的锚点可能落在同一个
   op 上，进而产生连续 get 违法形态。

完整代码改动见 [SyncSolver.cpp::getEventIdSolverRef](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)
和 [SyncSolver.cpp::checkIntersect](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)。

#### 修复后实际输出（与文档前文 §2.3 的形态表对齐）

```text
get_buf(TLOAD, #1); tload; rls_buf(TLOAD, #1)        # pair1, id=1
get_buf(TVEC,  #1);                                  # pair1 consumer on tmuls
get_buf(TVEC,  #0);                                  # pair2 producer on tmuls (different id)
tmuls
rls_buf(TVEC,  #1)
rls_buf(TVEC,  #0)
get_buf(TSTORE_VEC, #0); tstore; rls_buf(TSTORE_VEC, #0)
```

- 约束 1（同 pipe 同 id 不能重入）：TVEC 上 id=0 一对 + id=1 一对，**不同 id**，合法。
- 约束 2（不同 pipe 同 id 各自连续）：id=1 跨 TLOAD/TVEC，TLOAD 那对内部无其他 pipe 的 id=1；TVEC 那对内部只有 id=0 的 get/rel（不是 id=1），合法。

#### 保守度

`checkIntersect` 在 buf-id 下采用"共享任一 pipe → 冲突"的最保守判据，**忽略**
IR 范围 overlap。这会偶尔产出更密的冲突图，把不必要的 ConflictPair 染成不同
id，浪费 buf-id 池。在 8-id 默认池场景下实测影响不大；若后续在大型 kernel 上
出现 id 池耗尽问题，可以细化为"共享 pipe 上的锚点 RWOperation 重合"——这个
判据严格等价于真实硬件冲突，但实现更繁琐。阶段一保守为主。

### 3.9 Mirror-image ConflictPair 去重

对于循环里的同一对 producer/consumer 跨 pipe，solver 会同时检测出：

- 正向（intra-iter）：`F: tload[i] → tmuls[i]`（MTE2 → V，RAW）
- 反向（loop-carried）：`B: tmuls[i] → tload[i+1]`（V → MTE2，WAW）

set-wait 模式下两条都要发出，因为 set/wait 严格配对：F 在 iter 内成对，
B 必须横跨循环边界单独成对（form 1 或 form 2 都行）。

buf-id 模式下，F 和 B 的"bracket 集合"完全一样：

```text
F brackets: {(tload, MTE2), (tmuls, V)}
B brackets: {(tmuls, V), (tload, MTE2)}    # 相同的无序集合
```

硬件 `(get_cnt, rel_cnt)` 计数器的单调性已经同时承担了 F 和 B 两条依赖：
iter i+1 的 `get_buf(MTE2, #id)` 抢号时，号码必须等到 iter i 的
`rls_buf(V, #id)` 推进 rel_cnt 之后才能 pop——这就是 B 的反向语义。再单独
为 B 分配一对 bracket 是冗余，浪费 id。

doc §1.2 canonical 例子直接展示了这一点：

```text
for i = 0:100
  get_buf(MTE2, #id); load();   rel_buf(MTE2, #id)
  get_buf(V,    #id); Vector(); rel_buf(V,    #id)
```

单一 #id 同时覆盖正反向。

#### 修复

在 `getBeforeAfterSyncMaps` 前置一个 mirror-image 去重 pass：用无序 anchor 集合
`{(setOp, setPipe), (waitOp, waitPipe)}` 作 key，把同 key 的多个 ConflictPair
合成一条（保留 id 最小的代表，其余 mark 为 redundant 跳过 emit）。

实测效果：`tload + tmuls` canonical loop case 从 2 个 id（id 0 给 F，id 1
给 B）减到 1 个 id（id 0 给 F，B 被去重）；输出与 doc §1.2 完全一致。

#### 局限

去重发生在 `getBeforeAfterSyncMaps`，此时 `EventIdSolver` 已经分配过 id 了
——被丢弃的 B 那个 id 是"已 reserved 但没 emit"，浪费一个 slot。若后续
出现 id 池紧张，可以把 mirror dedup 提前到 `processConflict` 阶段（不创建
重复的 ConflictPair），或者在 `EventIdSolver::createNode` 时检查 mirror
共享 EventIdNode。阶段一不强制做。

---

## 4. 跨层同步检查（参考 `.claude/rules/cross-layer-sync.md`）

| 层 | 变更点 |
| --- | --- |
| ODS (`include/PTO/IR/*.td`) | `get_buf`/`rls_buf` 已存在，可选加 verifier |
| C++ IR & verifiers (`lib/PTO/IR`) | 同上 |
| 求解器内核 (`lib/PTO/Transforms/GraphSyncSolver/`) | `SyncSolverIR.h` 加 `BufOp`/`GetBufOp`/`RlsBufOp`；`SyncSolver.cpp::getBeforeAfterSyncMaps` 加分支；`SyncSolverCodeGen.cpp::emitSyncOp` 加分支；`Utility.h::SyncSolverOptions` 加 `emitStyle` |
| Pass 入口 (`PTOGraphSyncSolver.cpp`) | 解析新选项 + arch 校验 |
| Passes.td | 增加 `sync-style` option |
| CLI (`tools/ptoas/ptoas.cpp`) | 新 flag `--graph-sync-solver-sync-style` 透传 |
| Python 绑定 (`python/`) | 不变（dialect 已绑定 get_buf/rls_buf） |
| 文档 | 本文 + 更新 [docs/designs/ptoas-auto-sync-design.md](ptoas-auto-sync-design.md) sync style 章节 + [docs/PTO_IR_manual.md](../PTO_IR_manual.md) sync 章节加 buf-id 模型说明 |
| Tests | 见第 5 章 |

---

## 5. 测试方案

### 5.1 单元 / lit 测试

参考 `test/lit/pto/` 现有 `*_gss.pto` 体例（如
[graph_sync_solver_basic.pto](../../test/lit/pto/graph_sync_solver_basic.pto)），
新增：

| 测试 | 目的 | 关键 FileCheck |
| --- | --- | --- |
| `graph_sync_solver_buf_id_basic.pto` | A5 双 pipe forward dep 最小用例 | 出现 `pto.get_buf`/`pto.rls_buf`，不出现 `pto.set_flag` / `pto.wait_flag` |
| `graph_sync_solver_buf_id_multibuf.pto` | multibuffer 流水，N>1 个 id | 同一 anchor 上有 N 对 bracket，bufId 互不相同 |
| `graph_sync_solver_buf_id_no_barrier.pto` | 同 pipe 真依赖 | **不出现** `pto.barrier`、也不出现 buf-id 指令（A5 硬件保序） |
| `graph_sync_solver_buf_id_backward.pto` | 循环间依赖（backward sync） | producer/consumer 在原位 bracket，无形态二外提；不出现 `if isFirstIter` 守卫 |
| `graph_sync_solver_buf_id_arch_guard.pto` | A3 + buf-id 触发报错 | 显式 emitError |
| `graph_sync_solver_buf_id_parity.pto` | 同输入分别跑 set-wait / buf-id，编号对齐 | bufId 集合 == eventId 集合，bracket 数 == set+wait 数 |

`runop.sh`（[test/samples/runop.sh](../../test/samples/runop.sh)）的对应行加 buf-id
模式补充。

### 5.2 端到端 / sample

复用 [test/samples/Sync/](../../test/samples/Sync/) 现有 `tmatmulk_autosync*.py`
等样例，再加一个 buf-id 版本（`test/samples/Sync/tmatmulk_autosync_buf_id.py`）。
EmitC 落到 C++ 后通过 board validation（[test/samples/MatMul/](../../test/samples/MatMul/)
体例）验证 hw 仿真器上语义正确。

### 5.3 Bug 报告体例（参考 `.claude/rules/testing-and-examples.md`）

后续提交的回归测试统一按 **Before / Expected / Actual** 三段式：
- Before：最小 .pto；
- Expected：buf-id bracket 形态、对应 set-wait 等价形态、verifier 结果；
- Actual：`ptoas` 命令行 + FileCheck。

---

## 6. 实施步骤

| 步骤 | 内容 | 风险 |
| --- | --- | --- |
| S1 | `SyncSolverOptions::emitStyle` 枚举 + Passes.td option + CLI flag + Pass 入口校验 | 不改语义；纯透传，几乎无风险 |
| S2 | `SyncSolverIR.h` 增加 `BufOp`/`GetBufOp`/`RlsBufOp` + OpType；不影响现有 `SetFlagOp`/`WaitFlagOp` | 编译期/链接期 |
| S3 | `getBeforeAfterSyncMaps` 分支 + `pipeToPipeEventTypeAttr` helper + `emitSyncOp` 分支 | 主要风险点，要保证 set-wait 路径完全等价 |
| S4 | lit 测试 5 个 + 一份端到端 sample | 校准 FileCheck pattern |
| S5 | 更新 `PTO_IR_manual.md` / `ptoas-auto-sync-design.md` | 文档而已 |
| 阶段二（可选） | 约束 4 的强 / 弱模式切换；hw 团队明确后再决定是否在 `EventIdSolver` 层按 pipe-pair 打散 | 独立 PR，不阻塞阶段一 |

---

## 7. 风险与开放问题

1. **buf-id 池容量**：阶段一沿用 `eventIdNumMax = 8`。需 hw 团队确认 A5 上
   buf-id pool 是否与 event-id pool 共享物理资源、容量是否一致。若分离，
   需要把 `eventIdNumMax` 拆成两个独立选项。
2. **`mode` 字段**：`pto.get_buf` / `pto.rls_buf` 有 `I32Attr:$mode` 默认 0。
   当前文档未提非 0 模式语义，全部填 0。需 hw 团队进一步明确。
3. **约束 4（不同 pipe 对共享 id）**：早期版本把这归为"软提示"。实际验证发现
   ConflictPair 共享 pipe 时跨 pair 共享 id 会违反约束 1（连续 get 非法），
   所以阶段一已经在 `getEventIdSolverRef` + `checkIntersect` 里强制隔离，
   见 3.8。
4. **A2 / A3 行为**：阶段一在 `!isA5 && BUF_ID` 时直接报错。如未来低世代芯片
   也支持 buf-id，需放开 arch 校验。
5. **与 `PTOInsertSync` 的关系**：本设计只覆盖 `PTOGraphSyncSolver`。
   `PTOInsertSync` 是独立 pipeline，是否需要同步加 buf-id 形态另外评估
   （三个自动同步选项当前互斥，[ptoas.cpp:1094](../../tools/ptoas/ptoas.cpp)）。
6. **backward-sync**：见 2.4 节，buf-id 模型本身已是顺序模型，**不需要**
   set/wait 那套形态二外提。阶段一直接在 producer / consumer 原位 bracket
   即可。求解器里现有的若干 `*backwardSync*` 优化在 buf-id 模式下全部短路。
