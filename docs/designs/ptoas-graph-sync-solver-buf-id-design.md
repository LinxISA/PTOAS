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

**结论：buf-id 模式下 backward ConflictPair 不需要单独发 bracket**——一旦同一 SSA tile buffer 上所有 forward pair 都共享同一个 buf id，硬件 scoreboard 的计数器单调性已经把跨迭代依赖一并串起。

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

**判别准则**：两条 forward F1、F2 是否在同一 buffer chain 上？

实现里直接把"是否同 buffer"作为一等公民：在 `checkMemoryConflicts` 里把
触发冲突的 SSA `mlir::Value` 透传到 `ConflictPair::conflictBuffer`（每个
mem-info 对独立一个 ConflictPair），后续所有 buf-id 相关的着色 / 裁剪决
策都按 `conflictBuffer` 来判别。判定退化成：

> `conflictPair1->conflictBuffer == conflictPair2->conflictBuffer` ⇒ 同 buffer chain。

历史上文档里提到过的"backward 边作 buffer 桥"代理仍保留作为兜底
（`getBeforeAfterSyncMaps` 里的 union-find pre-pass，专门处理无 conflictBuffer
的特殊形态），但默认路径走 `conflictBuffer` 直接比较。

#### 落实

1. `tryMovingOutBackwardSyncPairsToOuterLoops`、`considerOuterBackwardSyncPairs`、
   `mergeBackwardSyncPairs` 这些 set/wait 的 backward 外提路径在 buf-id 模式下**直接关掉**。
2. `backwardSyncEvents` / `backwardSyncEventsAfterMerge` 不会被填充。
3. `ConflictPair` 携带 `conflictBuffer` 字段（[SyncSolverIR.h](../../include/PTO/Transforms/GraphSyncSolver/SyncSolverIR.h)）；`checkMemoryConflicts`
   返回 `SmallVector<tuple<CorePipeInfo, CorePipeInfo, Value>>`，每一项独立一个 ConflictPair。
4. **着色阶段**（`checkIntersect`）：同 `conflictBuffer` 的两 pair 不算冲突，落入
   同一 color → 同一 buf id；同时 emit-side 用 `(anchor_op, pipe, bufId)` 三元
   组 dedup，把跨 pair 落在同一锚点同一 pipe 同一 id 的 bracket 合成一对。
   这条规则同时覆盖 RAW + WAR + WAW（所有 same-buffer 依赖都搭同一条 ratchet 链）。
5. **裁剪阶段**（`checkGraphConflict`）：
   - 不同 `conflictBuffer` 的现有 pair 不能传递覆盖候选（独立 scoreboard）；
   - 互斥 scf.if 分支的现有 pair 不能传递覆盖候选（运行时根本不同路径）；
   - 同 pipe-pair 不同 consumer 的现有 pair：若候选 consumer 写了与现有
     consumer 不同的目标 buffer，则现有 pair 不能传递覆盖（独立 destination
     需要独立 bracket）；若目标 buffer 相同，则 WAW 链已经把候选串进
     ratchet，候选可被裁剪。
   - 同 pipe-pair 不同 producer 的现有 pair：不能传递覆盖（与上一条对称——
     buf-id 没有 producer-side FIFO 自动覆盖语义，每个 producer 都需要自己
     的 bracket 进入 chain；尤其要保证 cross-iter 时 first producer 也在
     ratchet 链上）。
6. `conflictPair->isInnerBackward` 为 true 的 pair 不发 bracket（计数器
   ratchet 已经把它的语义吸收掉）；同时 mirror-image dedup（前向与反向无
   序锚点集合相同的 pair）会更早一步把反向 pair 标记为 redundant。
7. **跨迭代 ping-pong 优化**：互斥 scf.if 分支位于同一循环、且写不同 SSA
   buffer 时，强制冲突（用不同 id）以让两分支跨迭代并行起来（见 §3.8.2
   规则 1 的 ping-pong 例外）。

具体落实见 3.6 / 3.8 节。

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

**buf-id 模式新增的字段 / 行为**：
- `ConflictPair::conflictBuffer`（`mlir::Value`）：记录触发依赖的 SSA tile
  buffer，由 `checkMemoryConflicts` 一并返回、经 `processConflict` /
  `handleConflict` 透传。这是 §2.4 / §3.8 所有 buf-id 决策的
  核心信息载体。同一个 (rwOp1, rwOp2) 的多个 mem-info 对，会分裂成多个独立
  的 ConflictPair（每个携带不同的 `conflictBuffer`），方便后续 buffer-aware
  着色与裁剪。

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
`(setOp, waitOp, setCorePipeInfo, waitCorePipeInfo, eventIdNode->getEventIds(), conflictBuffer)`。
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

[SyncSolver.cpp::getBeforeAfterSyncMaps](../../lib/PTO/Transforms/GraphSyncSolver/SyncSolver.cpp)
在每个非 barrier / 非 unitFlag 的 ConflictPair 处理段加分支。整体流程：

1. **Mirror-image dedup**（§3.9）：以无序 anchor 集合
   `{(setOp, setPipe), (waitOp, waitPipe)}` 为 key 把"forward + backward
   同形"的 pair 标记 redundant（保留 id 最小者）。
2. **Forward / backward 拆分**：以 `isInnerBackward` 划分，backward pair 后
   续只用作 union-find 桥（兜底）。
3. **Anchor 解析**：buf-id 模式下 bracket 直接锚在 `cp->op1` / `cp->op2`
   的 RWOperation 上（不外提到 setOp / waitOp 的 PlaceHolder LCA），见
   `anchorsOf`。理由：hoist 到外层 scope 会让 ratchet 在 if-then 路径未
   走时也推进，破坏 `(get_cnt, rel_cnt)` 与真实 pipe 活动的对应关系。
4. **Buffer-aware 共享 id**（实际实现）：因为 §3.8.2 规则 2 已经在
   `checkIntersect` 阶段把同 `conflictBuffer` 的 pair 染成同色，emit 阶段
   只需用 `(anchor_op, pipe, bufId)` 三元组对 bracket 做集合去重即可。
   保留的 union-find 预处理（backward-edge 桥）继续作为对 conflictBuffer
   为 null 情况下的兜底。
5. **发射**：对每个 forward pair 取 `eventIdNode->getEventIds().front()`
   作 bufId，调用 `emitBracket(anchor, bufId)`：

   ```cpp
   auto emitBracket = [&](AnchorEnd anchor, int64_t bufId, ConflictPair *dbg) {
     auto key = std::make_pair(anchor, bufId);
     if (emittedBefore.insert(key).second)
       syncMapBefore[anchor.first].push_back(
           std::make_unique<GetBufOp>(anchor.first->op, anchor.first->parentOp,
                                      anchor.second, bufId));
     if (emittedAfter.insert(key).second)
       syncMapAfter[anchor.first].push_back(
           std::make_unique<RlsBufOp>(anchor.first->op, anchor.first->parentOp,
                                      anchor.second, bufId));
   };
   ```

   `(anchor, bufId)` 集合 dedup 保证同 anchor / 同 pipe / 同 id 只发一对
   bracket——这对维持约束 1（同 pipe 同 id 不重入）至关重要：
   - 多条 forward pair 在同一锚点用同一 id（同 buffer chain 多次穿过同一
     op）→ dedup 合成一对；
   - 多 buffer 的 pair 在同一锚点用不同 id → 发多对 bracket，互不冲突。

**Barrier 处理**：

```cpp
if (conflictPair->isBarrier()) {
  if (options.isBufIdEmit()) {
    // A5 同 pipe 硬件保序，不发射任何 barrier
    continue;
  }
  // 现有 BarrierOp 路径，不变
}
```

**Backward ConflictPair 处理**：

```cpp
if (options.isBufIdEmit() && conflictPair->isInnerBackward) {
  continue;
}
```

backward pair 不发 bracket；语义由 forward pair 通过同 buffer 共享 id 的
ratchet 链 + mirror-image dedup 一并吸收。backward 仍会被 solver 计数和
分配 id（因为 `EventIdSolver` 早就跑完了），分配掉的 id 实际未使用——是
浪费但不影响正确性。后续若 id 池紧张，可前移到
`processConflict`/`handleSetWaitConflict` 从源头不创建。

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

### 3.8 `checkIntersect` 与 `checkGraphConflict` 的 buf-id 规则集

求解器的两条决策线分别决定 **着色**（哪些 pair 必须用不同 id）与 **裁剪**
（哪些 pair 不需要单独 emit）。buf-id 模式下两者都按 `conflictBuffer` /
mutex-branch / pipe-pair 做了多层细化。下文给出当前实现的完整规则。

#### 3.8.1 共用前提

- `EventIdSolver` 实例：buf-id 模式跟 cross-core 模式一样在
  `getEventIdSolverRef` 里把 key 折成 `(PIPE_UNASSIGNED, PIPE_UNASSIGNED)`，
  让**所有** pipe 对共享同一个着色图；否则不同 pipe 对会各自从 id=0 起
  发牌，产生跨 pipe-pair 撞 id 的隐患。
- `reuseSyncPairToSaveEventIds` 在 buf-id 下默认关掉（约束 4 的软性提示）。
- 同 pipe barrier 整体跳过（§1.4 / §3.4）。

#### 3.8.2 `checkIntersect`：着色冲突边规则（buf-id 路径）

依次判定，命中即返回：

1. **互斥 scf.if 分支** → **默认无冲突**（可共享 id，因为运行时只走一条
   分支）。

   **例外（loop ping-pong）**：两 pair 同时满足
   - 都有 `conflictBuffer`，且**不同**；
   - 共享一个 Loop 祖先

   时，强制冲突。意图：跨迭代轮流执行 if / else 分支的 ping-pong 形态，
   同 id 会让 iter k+1 的分支等 iter k 的分支 ratchet 完才能起步，浪费
   独立 memory 上的并行机会。

   典型例子（gemm.pto）：
   ```text
   for v55 {
     if (v56 % 2 == 0) {
       tload(v30); tload(v34); ...   # mat tiles at addr 0 / 262144
     } else {
       tload(v32); tload(v36); ...   # mat tiles at addr 196608 / 65536
     }
   }
   ```
   if 的 v30 与 else 的 v32 是不同 SSA `alloc_tile`，二者各拿一组独立
   id，iter k+1 的 else 分支 tload 可以在 iter k 的 if 分支 textract / matmul
   还在跑时就在 MTE2 上发出。

2. **同 `conflictBuffer`** → 无冲突（可共享同一 buf id）。

   理由：硬件 `(get_cnt, rel_cnt)` ratchet 链顺着同 id 串起 RAW + WAR + WAW
   三类同 buffer 依赖；emit-side 的 `(anchor_op, pipe, id)` 三元组 dedup
   会把同锚点重复 bracket 合并成一对，自然满足约束 1（连续两次同 pipe
   同 id `get_buf` 非法）。

3. **共享任一 pipe** → 冲突（约束 1）。

   即使 `(setPipe, waitPipe)` 二元组不完全相同，只要两 pair 出现了同一
   个 pipe，bracket 锚在 RWOperation 上时就可能落在同一个 op 上引发
   "连续两次同 pipe 同 id `get_buf`"。比 set-wait 的范围 overlap 检查更
   保守，但实测对 8-id 默认池影响不大。

4. 否则不冲突。

#### 3.8.3 `checkGraphConflict`：传递覆盖裁剪规则（buf-id 路径）

候选 pair 由 `handleConflict` 抛入 `checkGraphConflict` 之前，遍历所有
已选 / 持久化 pair，三层过滤决定哪些 pair 进入 Dijkstra 图：

1. **不同 `conflictBuffer` 的现有 pair**：buf-id 是 per-id 独立 scoreboard，
   不同 buffer 之间没有计数器关系 → 不能传递覆盖 → 丢弃。

2. **互斥 scf.if 分支的现有 pair**：候选与现有运行时不在同一路径 → 不能
   传递覆盖 → 丢弃。

3. **同 pipe-pair 但不同 consumer 的现有 pair**（candidate 在 RAW 方向：
   `candOp2` 读 `conflictBuffer`）：
   - 若候选 consumer 与现有 consumer 写**相同**目标 buffer：现有
     consumer → 它下游的 RAW peer → 候选（通过共享 id 的 WAW 链）已经
     形成 ratchet 闭环，候选可被覆盖 → 保留现有 pair 进入图。
   - 若目标 buffer 不同：没有 WAW 链 → 现有 pair 不能传递覆盖候选 → 丢弃。

   注意此规则只在候选确实读 `conflictBuffer` 时（RAW 方向）触发；对 WAR /
   WAW 方向的候选不生效，由"同 buffer 共享 id"在 checkIntersect 那一层
   通过 ratchet 链自动覆盖。

4. **同 pipe-pair 但不同 producer 的现有 pair**（与规则 3 对称）：
   `conflictPair->op1 != candOp1 && conflictPair->op2 == candOp2` 且 pipe-pair /
   conflictBuffer 都匹配 → 丢弃。

   理由：set/wait 模式下，同 producer pipe FIFO 让最晚的 producer 一次
   `set_flag` 覆盖更早的所有 producer——求解器的传递覆盖正是基于这条
   FIFO 性质。buf-id 没有这条性质：每个 producer 的 bracket 锚在自己
   的 op 上，counter ratchet 只串联**有 bracket 的**锚点。让现有 pair
   传递覆盖，会把候选 producer 排除在 chain 之外。

   典型受影响场景：循环里多个 matmul producer (writes `v46`) 喂同一个
   `tstore` consumer，外层循环跨迭代用 `v46` 形成 ping-pong 累加。
   若只保留 last producer 的 bracket，iter k+1 的 first matmul（同
   OperationBase 多次 Occurrence）在 PIPE_M FIFO 上仅等 iter k 的
   last matmul 完成，**不**等 iter k 的 tstore 完成 → iter k+1 的
   accumulator 写覆盖 iter k 仍在读的 v46，data race。规则 4 让 first
   matmul 自己拿一个 bracket，counter ratchet 自然把 tstore 串进去。

   代价：中间 producer（既不是 first 也不是 last）也会拿到 bracket，
   形成略多的 get_buf/rls_buf 序列；但中间 producer 本身就靠 PIPE_M
   FIFO 串行化，多出的 bracket 实际上是 no-op，对 intra-iter 性能无
   影响。

5. 其余 pair 走原有 Dijkstra 覆盖逻辑。

#### 3.8.4 例子

**例子 1：linear chain（doc §1.2 Case 1）**
```text
tload  # MTE2 -> %tile
tmuls  # V    -> %tile
tstore # MTE3 -> %tile
```
- 两 forward pair 都在 `%tile` 上：checkIntersect §3.8.2 第 2 条触发，
  共享同一 buf id。
- emit：`get_buf(MTE2, X) tload rls_buf(MTE2, X); get_buf(V, X) tmuls
  rls_buf(V, X); get_buf(MTE3, X) tstore rls_buf(MTE3, X)`。tmuls 上 V
  端被同 (anchor, X) dedup 成一对 bracket，串起整条 ratchet。

**例子 2：one producer, multiple consumers（gemm.pto 内层）**
```text
tload v30 -> MTE1
textract v38 := f(v30, ...) # 1st write of v38
textract v40 := g(v30, ...) # 1st write of v40
textract v38 := h(v30, ...) # 2nd write of v38
textract v40 := i(v30, ...) # 2nd write of v40
```
- 全部 4 个 RAW pair 都在 `v30` 上 → checkIntersect 同 buffer 共享 id。
- 候选 `(tload, textract v38 2nd)` 在 checkGraphConflict 里：现有
  `(tload, textract v38 1st)` 写同样的 destination `v38` → 命中 §3.8.3
  第 3 条第一种情况 → 现有 pair 保留 → Dijkstra 找到覆盖 → 候选被裁剪。
- 候选 `(tload, textract v40 1st)`：现有 `(tload, textract v38 1st)` 写
  `v38` ≠ candidate 写 `v40` → 命中 §3.8.3 第 3 条第二种情况 → 现有
  pair 丢弃 → 候选自己 emit bracket。
- 结果：1st 与 2nd 写 v38 / v40 各保留 tload-sync bracket，后续覆盖写
  靠 WAW ratchet 链继承。

**例子 3：mutex branches in loop（gemm.pto 外层 if/else）**
```text
for v55 {
  if {
    tload(v30); tload(v34); ...
  } else {
    tload(v32); tload(v36); ...
  }
}
```
- `(tload, ...)` 在 if 分支与 `(tload, ...)` 在 else 分支：互斥 + 共享
  Loop + 不同 conflictBuffer → §3.8.2 第 1 条 ping-pong 例外触发，强制
  冲突，分得不同 id。
- iter k 用 if 的 id 组，iter k+1 用 else 的 id 组，互不阻塞。

**例子 4：multiple producers, one consumer across outer loop（gemm.pto
accumulator + tstore）**
```text
for v53 {
  for v55 {
    matmul_1st (writes v46)         # in inner if/else
    matmul_ACC_2nd (reads/writes v46)
    matmul_ACC_3rd (reads/writes v46)
    matmul_ACC_4th (reads/writes v46)
  }
  tstore v66, v46                   # reads v46, in v53 body
}
```
- 4+ 个 forward pair `(matmul_X, tstore)`，全部在 `v46` 上，全部
  PIPE_M → PIPE_FIX。
- 候选 `(matmul_1st, tstore)`，现有 `(matmul_ACC_4th, tstore)`：
  - 同 conflictBuffer (v46)、同 pipe-pair (M → FIX)、同 op2 (tstore)、
    不同 op1 → 命中 §3.8.3 规则 4 → 现有 pair 丢弃 → 候选自己 emit
    bracket。
  - 其他中间 producer 同理也保留各自的 bracket。
- 结果：所有 matmul 在 PIPE_M 上拿到 id_v46 的 get/rls bracket，counter
  按 1st → 2nd → 3rd → 4th → tstore 顺序 ratchet。
- 跨外层迭代：iter k+1 的 matmul_1st（同 OperationBase 不同 Occurrence）
  get_buf 时 snap 已经在 iter k 的 tstore rls 之后，硬件等到 iter k
  tstore 读完 v46 才允许 iter k+1 写 v46，WAR 安全。

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
| 求解器内核 (`lib/PTO/Transforms/GraphSyncSolver/`) | `SyncSolverIR.h` 加 `BufOp`/`GetBufOp`/`RlsBufOp` + `ConflictPair::conflictBuffer`；`SyncSolver.cpp::checkMemoryConflicts` 返回 `(srcInfo, dstInfo, Value)` 三元组；`SyncSolver.cpp::checkIntersect` / `checkGraphConflict` 加 buf-id 路径（§3.8）；`SyncSolver.cpp::getBeforeAfterSyncMaps` 加分支；`SyncSolverCodeGen.cpp::emitSyncOp` 加分支；`Utility.h::SyncSolverOptions` 加 `emitStyle` |
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
当前已落地的测试集（[test/lit/pto/graph_sync_solver_buf_id_*.pto](../../test/lit/pto/)）：

| 测试 | 目的 | 关键 FileCheck |
| --- | --- | --- |
| `graph_sync_solver_buf_id_basic.pto` | 双 pipe forward dep + 同 pipe 真依赖共两个 case | 出现 `pto.get_buf`/`pto.rls_buf`；同 pipe case 不出现 `pto.barrier`；linear chain 上 anchor 共享 id（emit-side dedup） |
| `graph_sync_solver_buf_id_canonical_loop.pto` | doc §1.2 canonical 循环（单 buffer F+B 镜像） | 单一 id 覆盖 forward 与 backward；无 backward 外提；无 `scf.if` 守卫 |
| `graph_sync_solver_buf_id_two_buffer_loop.pto` | doc §1.2 双 buffer 循环（两条独立 buffer chain） | 两条 chain 分得不同 id；vadd 在 V 上有两对 bracket（构成跨 chain 衔接） |
| `graph_sync_solver_buf_id_cross_loop.pto` | 跨 ConflictPair 通过同 buffer 形成传递覆盖 | 后续消费者继承前序 ratchet，不重复 emit |
| `graph_sync_solver_buf_id_backward.pto` | backward ConflictPair 不发射 | 不出现 backward-only bracket；forward ratchet 自然吸收 |
| `graph_sync_solver_buf_id_if_branch.pto` | producer 位于 scf.if 单分支 | bracket 紧贴 RWOp，不外提到 LCA |
| `graph_sync_solver_buf_id_if_branch_consumer.pto` | consumer 位于 scf.if 单分支 | 同上方向对称 |
| `graph_sync_solver_buf_id_if_both_branches.pto` | mutex 分支共享 id（非循环 case） | if / else 走 §3.8.2 规则 1 共享 id（不触发 ping-pong 例外） |
| `graph_sync_solver_buf_id_arch_guard.pto` | A3 + buf-id 触发报错 | 显式 emitError |

后续可补：
- `graph_sync_solver_buf_id_ping_pong.pto`：覆盖 §3.8.2 规则 1 的 ping-pong
  例外路径（互斥分支 + 同循环 + 不同 buffer → 不同 id）。
- `graph_sync_solver_buf_id_parity.pto`：同输入分别跑 set-wait / buf-id，
  bufId 集合 == eventId 集合，bracket 数 == set+wait 数。
- `graph_sync_solver_buf_id_multibuf.pto`：multibuffer 流水（`eventIdRepeatNum > 1`），
  同一 anchor 上有 N 对 bracket。

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

| 步骤 | 内容 | 状态 |
| --- | --- | --- |
| S1 | `SyncSolverOptions::emitStyle` 枚举 + Passes.td option + CLI flag + Pass 入口校验 | ✅ 已落地（PR #665 初版） |
| S2 | `SyncSolverIR.h` 增加 `BufOp`/`GetBufOp`/`RlsBufOp` + OpType；不影响现有 `SetFlagOp`/`WaitFlagOp` | ✅ 已落地（PR #665 初版） |
| S3 | `getBeforeAfterSyncMaps` 分支 + `pipeToPipeEventTypeAttr` helper + `emitSyncOp` 分支 | ✅ 已落地（PR #665 初版） |
| S4 | lit 测试 9 个 + 一份端到端 sample | ✅ 已落地（`graph_sync_solver_buf_id_*.pto`） |
| S5 | `ConflictPair::conflictBuffer` 字段 + `checkMemoryConflicts` 返回 `(srcInfo, dstInfo, Value)` 三元组 + 透传 | ✅ 已落地（PR #665 fix commit `829945ff`） |
| S6 | `checkGraphConflict` buffer-aware / mutex-aware 裁剪 | ✅ 已落地（commits `829945ff` + `448f34e9`） |
| S7 | `checkIntersect` 同 buffer 共享 id 规则 + `checkGraphConflict` 同 pipe-pair 不同 consumer 的 WAW-chain 判据 | ✅ 已落地（commit `c3cc9b8d`） |
| S8 | `checkIntersect` 跨迭代 ping-pong 例外（同循环 + 不同 buffer + 互斥分支 → 强制冲突） | ✅ 已落地（commit `ce764e3d`） |
| S9 | `checkGraphConflict` 同 pipe-pair 不同 producer 的对称裁剪规则（避免 cross-iter 累加器 WAR 漏掉 first producer） | ✅ 已落地（commit `880ac9f7`） |
| S10 | 更新 `PTO_IR_manual.md` / `ptoas-auto-sync-design.md` | TODO |
| 阶段二（可选） | alias-aware WAW-chain（§7-7）、ping-pong 跨嵌套循环（§7-8）、`mode != 0` 支持（§7-2） | 独立 PR，不阻塞阶段一 |

---

## 7. 风险与开放问题

1. **buf-id 池容量**：阶段一沿用 `eventIdNumMax = 8`。需 hw 团队确认 A5 上
   buf-id pool 是否与 event-id pool 共享物理资源、容量是否一致。若分离，
   需要把 `eventIdNumMax` 拆成两个独立选项。
2. **`mode` 字段**：`pto.get_buf` / `pto.rls_buf` 有 `I32Attr:$mode` 默认 0。
   当前文档未提非 0 模式语义，全部填 0。需 hw 团队进一步明确。
3. ~~**约束 4（不同 pipe 对共享 id）**~~ **（已解决）**：

   早期版本把这归为"软提示"。实际验证发现 ConflictPair 共享 pipe 时
   跨 pair 共享 id 会违反约束 1（连续 get 非法）。落地方案见 §3.8：
   - `getEventIdSolverRef`：buf-id 模式所有 pipe 对共享同一 `EventIdSolver`。
   - `checkIntersect`：按 §3.8.2 的四层规则（互斥分支 / 同 buffer /
     共享 pipe / 默认）决定冲突，避免共享 pipe 上同 id 重入；同时允许
     同 SSA buffer 跨 pipe-pair 共享 id（emit-side anchor dedup 保证约束 1）。

   `reuseSyncPairToSaveEventIds` 在 buf-id 下默认仍关，以契合约束 4 的
   "不同 pipe 对不建议共享 id" 软性提示——但允许"同 buffer 跨 pipe-pair
   共享 id"是性能必要的反例。
4. **A2 / A3 行为**：阶段一在 `!isA5 && BUF_ID` 时直接报错。如未来低世代芯片
   也支持 buf-id，需放开 arch 校验。
5. **与 `PTOInsertSync` 的关系**：本设计只覆盖 `PTOGraphSyncSolver`。
   `PTOInsertSync` 是独立 pipeline，是否需要同步加 buf-id 形态另外评估
   （三个自动同步选项当前互斥，[ptoas.cpp:1094](../../tools/ptoas/ptoas.cpp)）。
6. ~~**backward-sync**~~ **（已解决）**：buf-id 模型本身已是顺序模型，
   **不需要** set/wait 那套形态二外提。求解器里现有的若干 `*backwardSync*`
   优化在 buf-id 模式下全部短路；backward ConflictPair 不发 bracket，其
   语义由 forward pair 通过同 buffer 共享 id 的 ratchet 链 + mirror-image
   dedup 一并吸收。见 §2.4 / §3.6 / §3.9。
7. **`checkGraphConflict` 同 pipe-pair 不同 consumer 的 WAW-chain 判据
   保守度**：当前实现要求候选与现有 consumer 写"同一 SSA Value"才能利用
   WAW 链覆盖。若两条 forward 通过 alias / view 写同一物理 tile 但 SSA
   Value 不同（例如经 `partition_view` / `bind_tile` 后），会被判为不同
   destination，候选保留独立 bracket。功能正确，仅可能多出冗余 bracket；
   后续若 alias 分析可靠，可放宽判据至"指向同一 tile 物理地址"。
8. **ping-pong 触发条件保守度**：当前 §3.8.2 规则 1 例外只在两 pair 的
   `op1` 共享一个 **直接** Loop 祖先时触发。嵌套循环或同分支跨多层
   循环的 ping-pong 场景需要进一步分析（罕见，阶段一不处理）。
