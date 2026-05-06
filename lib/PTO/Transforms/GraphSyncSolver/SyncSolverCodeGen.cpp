// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <utility>

#define DEBUG_TYPE "pto-graph-sync-solver-codegen"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

CodeGenerator::CodeGenerator(std::unique_ptr<Solver> solver)
    : options(solver->options) {
  funcOp = solver->funcOp;
  funcIr = std::move(solver->funcIr);
  for (auto &cp : solver->chosenConflictedPairs) {
    if (!cp || !cp->eventIdNode)
      continue;
    cp->eventIds = cp->eventIdNode->getEventIds();
    cp->eventIdNode = nullptr;
  }
  chosenConflictedPairs = std::move(solver->chosenConflictedPairs);
}

static PipeAttr makePipe(MLIRContext *ctx, PIPE p) {
  return PipeAttr::get(ctx, p);
}

static EventAttr makeEvent(MLIRContext *ctx, int64_t eid) {
  // PTO event enum currently provides EVENT_ID0..EVENT_ID7 (0..7).
  return EventAttr::get(ctx, static_cast<EVENT>(eid));
}

Operation *CodeGenerator::resolveSyncAnchor(OperationBase *opBase,
                                            bool insertAfter) {
  assert(opBase);
  // PlaceHolders may not have an `op` directly; map to the surrounding
  // loop / block / scope's MLIR op.
  if (auto *ph = dyn_cast<PlaceHolder>(opBase)) {
    if (ph->beforeOp != nullptr) {
      assert(ph->beforeOp->op);
      return ph->beforeOp->op;
    }
    if (ph->afterOp != nullptr) {
      assert(ph->afterOp->op);
      return ph->afterOp->op;
    }
    if (ph->block != nullptr) {
      // Block start/end: anchor at the block's parent op.
      Operation *parent = ph->block->getParentOp();
      return parent;
    }
    return nullptr;
  }
  return opBase->op;
}

Location CodeGenerator::resolveSyncLoc(OperationBase *opBase) {
  if (Operation *anchor = resolveSyncAnchor(opBase, /*insertAfter=*/false))
    return anchor->getLoc();
  return funcOp.getLoc();
}

void CodeGenerator::insertSetFlag(IRRewriter &rewriter, OperationBase *anchor,
                                  PIPE setPipe, PIPE waitPipe, int64_t eventId,
                                  bool insertAfter) {
  Operation *anchorOp = resolveSyncAnchor(anchor, insertAfter);
  if (!anchorOp)
    return;
  Location loc = resolveSyncLoc(anchor);
  if (insertAfter)
    rewriter.setInsertionPointAfter(anchorOp);
  else
    rewriter.setInsertionPoint(anchorOp);
  auto srcAttr = makePipe(rewriter.getContext(), setPipe);
  auto dstAttr = makePipe(rewriter.getContext(), waitPipe);
  auto eidAttr = makeEvent(rewriter.getContext(), eventId);
  rewriter.create<pto::SetFlagOp>(loc, srcAttr, dstAttr, eidAttr);
}

void CodeGenerator::insertWaitFlag(IRRewriter &rewriter, OperationBase *anchor,
                                   PIPE setPipe, PIPE waitPipe, int64_t eventId,
                                   bool insertAfter) {
  Operation *anchorOp = resolveSyncAnchor(anchor, insertAfter);
  if (!anchorOp)
    return;
  Location loc = resolveSyncLoc(anchor);
  if (insertAfter)
    rewriter.setInsertionPointAfter(anchorOp);
  else
    rewriter.setInsertionPoint(anchorOp);
  auto srcAttr = makePipe(rewriter.getContext(), setPipe);
  auto dstAttr = makePipe(rewriter.getContext(), waitPipe);
  auto eidAttr = makeEvent(rewriter.getContext(), eventId);
  rewriter.create<pto::WaitFlagOp>(loc, srcAttr, dstAttr, eidAttr);
}

void CodeGenerator::insertBarrier(IRRewriter &rewriter, OperationBase *anchor,
                                  PIPE pipe, bool insertAfter) {
  Operation *anchorOp = resolveSyncAnchor(anchor, insertAfter);
  if (!anchorOp)
    return;
  Location loc = resolveSyncLoc(anchor);
  if (insertAfter)
    rewriter.setInsertionPointAfter(anchorOp);
  else
    rewriter.setInsertionPoint(anchorOp);
  auto pipeAttr = makePipe(rewriter.getContext(), pipe);
  rewriter.create<pto::BarrierOp>(loc, pipeAttr);
}

// P0-2: Slot index = LOGICAL iteration % N = ((iv - lb) / step) % N, NOT
// `iv % N`. When step is not 1 the latter only ever yields a stride-shifted
// subset of slots (e.g. step=2,N=4 -> {0,2}), silently corrupting the rotation.
// Keep a `iv % N` fast path for the canonical step=1, lb=0 case so the
// generated IR stays minimal in the common scenario.
static bool gssIsConstantIndexEqualTo(Value v, int64_t target) {
  if (!v)
    return false;
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == target;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt() == target;
  }
  return false;
}

Value CodeGenerator::getOrCreateLoopCounter(IRRewriter &rewriter,
                                            scf::ForOp forOp, int64_t n,
                                            Location loc) {
  auto key = std::make_pair(forOp, n);
  auto it = loop2BufferCounter_.find(key);
  if (it != loop2BufferCounter_.end())
    return it->second;
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value cN = rewriter.create<arith::ConstantIndexOp>(loc, n);
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Value normalized = iv;
  if (!gssIsConstantIndexEqualTo(lb, 0))
    normalized = rewriter.create<arith::SubIOp>(loc, normalized, lb);
  if (!gssIsConstantIndexEqualTo(step, 1))
    normalized = rewriter.create<arith::DivUIOp>(loc, normalized, step);
  Value rem = rewriter.create<arith::RemUIOp>(loc, normalized, cN);
  loop2BufferCounter_[key] = rem;
  return rem;
}

// P0-3: Anchor preservation.
//
// Earlier this function unconditionally placed the dyn wait at the loop body
// start and the dyn set right before the yield. That works for the simple
// "loop body is one straight segment" case but corrupts every other shape:
//
//   * If the consumer RWOp lives under an `scf.if`, the body-start dyn wait
//     pops the queue even on iterations where the consumer is skipped, so we
//     consume primes that were never produced.
//   * If the producer RWOp lives under an `scf.if`, the before-terminator dyn
//     set pushes onto the queue every iteration regardless of whether the
//     producer ran, drifting the queue depth.
//
// The original solver already chose the actual conflicting RWOperations as
// `cp->op1`/`cp->op2` (these point to the producer/consumer MLIR ops directly,
// inside whatever control flow they live in). Use those positions when they
// are inside the rotation loop body. Fall back to body-start/before-terminator
// only when the anchor is missing or actually lives outside the loop (e.g.,
// LCA-driven PlaceHolder anchors that resolved to the loop boundary).
static Operation *getMultiBufferAnchorOpInLoop(RWOperation *rwOp,
                                               scf::ForOp loop) {
  if (!rwOp || !rwOp->op || !loop)
    return nullptr;
  Operation *op = rwOp->op;
  // The op must live strictly inside the rotation loop body for the dyn ops
  // to dominate / be dominated by the in-body selector value.
  if (op == loop.getOperation())
    return nullptr;
  if (!loop->isProperAncestor(op))
    return nullptr;
  return op;
}

void CodeGenerator::emitMultiBufferSetWait(IRRewriter &rewriter,
                                           ConflictPair *cp) {
  // Multi-buffer codegen mirrors the InsertSync output:
  //   pre-loop:   N pto.set_flag (queue prime, one per event id)
  //   in-loop:    pto.wait_flag_dyn(idx) at the consumer anchor,
  //               pto.set_flag_dyn(idx)  at the producer anchor
  //   post-loop:  N pto.wait_flag (queue drain)
  // The dyn set/wait MUST live inside the loop body so they share the
  // `iv mod N` selector's dominance. GSS's default backward-sync anchors
  // (set after loop / wait before loop) only work for single-buffer where
  // there is no per-iteration selector.
  assert(cp);
  const auto &eids = cp->eventIds;
  int64_t n = cp->eventIdInfo.eventIdNum;
  assert((int64_t)eids.size() >= n && n >= 2);
  if ((int64_t)eids.size() < n || n < 2)
    return;
  scf::ForOp loop = cp->eventIdInfo.multibufferLoop;
  assert(loop && "multi-buffer codegen needs a non-null rotation loop");
  if (!loop)
    return;
  PIPE setPipe = cp->setCorePipeInfo.pipe;
  PIPE waitPipe = cp->waitCorePipeInfo.pipe;
  Location loc = loop.getLoc();
  auto srcAttr = makePipe(rewriter.getContext(), setPipe);
  auto dstAttr = makePipe(rewriter.getContext(), waitPipe);

  // Resolve the lexical-first / lexical-last RWOp positions in the loop body.
  //
  // NOTE: cp->op1 / cp->op2 are NOT MLIR-lexical-ordered. processOrders sorts
  // by syncIrIndex, and for backward-edge processing the "second iteration"
  // copy of an op gets a LARGER syncIrIndex than the original. So for a
  // backward dep between tload and tstore, cp->op1 is typically the prev-iter
  // tstore (lexically LATER in the MLIR loop body) and cp->op2 is the
  // next-iter tload copy (whose underlying MLIR op is lexically EARLIER).
  // We need to compare actual MLIR positions and pick the right anchors.
  //
  // For backward (cross-iteration) slot rotation:
  //   - The wait_flag_dyn must precede the iteration's FIRST MLIR user of the
  //     slot, otherwise the next iteration could overwrite a slot still in
  //     flight from two iterations ago.
  //   - The set_flag_dyn must follow the iteration's LAST MLIR user of the
  //     slot, otherwise the queue would advance before the slot is free.
  // This matches the conservative body-start / before-terminator placement
  // exactly when the loop body has no other ops besides op1..op2 and is
  // strictly tighter when those ops live under nested control flow.
  Operation *anchor1 = getMultiBufferAnchorOpInLoop(cp->op1, loop);
  Operation *anchor2 = getMultiBufferAnchorOpInLoop(cp->op2, loop);
  Operation *firstUserAnchor = nullptr;
  Operation *lastUserAnchor = nullptr;
  if (anchor1 && anchor2 && anchor1->getBlock() == anchor2->getBlock()) {
    if (anchor1->isBeforeInBlock(anchor2)) {
      firstUserAnchor = anchor1;
      lastUserAnchor = anchor2;
    } else {
      firstUserAnchor = anchor2;
      lastUserAnchor = anchor1;
    }
  }
  // If anchors are in different blocks (e.g. one under scf.if), or one is
  // missing, leave them null and fall back to body-start/before-terminator
  // below. That preserves correctness at the cost of guarding fewer cases.

  // 1. Pre-loop: queue-prime with N concrete event ids.
  rewriter.setInsertionPoint(loop);
  for (int64_t i = 0; i < n; ++i) {
    auto eidAttr = makeEvent(rewriter.getContext(), eids[i]);
    rewriter.create<pto::SetFlagOp>(loc, srcAttr, dstAttr, eidAttr);
  }

  // 2. In-loop: build (or reuse) the slot counter at body start, then a
  // select chain over the assigned event ids. The chain dominates every
  // in-body anchor because it lives at body start.
  Value rem = getOrCreateLoopCounter(rewriter, loop, n, loc);
  rewriter.setInsertionPointAfter(rem.getDefiningOp());
  Value selected =
      rewriter.create<arith::ConstantIndexOp>(loc, eids[0]);
  for (int64_t i = 1; i < n; ++i) {
    Value ci = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value eq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              rem, ci);
    Value idv = rewriter.create<arith::ConstantIndexOp>(loc, eids[i]);
    selected = rewriter.create<arith::SelectOp>(loc, eq, idv, selected);
  }

  // 3a. wait_flag_dyn just before the iteration's first user of the slot.
  // Fall back to body start (right after the selector) if that anchor lives
  // outside the rotation loop body (e.g. solver chose loop-boundary
  // PlaceHolders).
  if (firstUserAnchor) {
    rewriter.setInsertionPoint(firstUserAnchor);
  } else {
    rewriter.setInsertionPointAfter(selected.getDefiningOp());
  }
  rewriter.create<pto::WaitFlagDynOp>(loc, srcAttr, dstAttr, selected);

  // 3b. set_flag_dyn right after the iteration's last user of the slot. Fall
  // back to right before the loop terminator if that anchor lives outside.
  if (lastUserAnchor) {
    rewriter.setInsertionPointAfter(lastUserAnchor);
  } else {
    Operation *terminator = loop.getBody()->getTerminator();
    if (!terminator)
      return;
    rewriter.setInsertionPoint(terminator);
  }
  rewriter.create<pto::SetFlagDynOp>(loc, srcAttr, dstAttr, selected);

  // 4. Post-loop: drain by waiting on each prime.
  rewriter.setInsertionPointAfter(loop);
  for (int64_t i = 0; i < n; ++i) {
    auto eidAttr = makeEvent(rewriter.getContext(), eids[i]);
    rewriter.create<pto::WaitFlagOp>(loc, srcAttr, dstAttr, eidAttr);
  }
}

void CodeGenerator::emitOne(IRRewriter &rewriter, ConflictPair *cp) {
  if (cp->isBarrierAll) {
    // Single PIPE_ALL barrier inserted just before the wait anchor.
    insertBarrier(rewriter, cp->waitOp, PIPE::PIPE_ALL,
                  /*insertAfter=*/false);
    return;
  }
  if (cp->isBarrier()) {
    insertBarrier(rewriter, cp->waitOp, cp->setCorePipeInfo.pipe,
                  /*insertAfter=*/false);
    return;
  }
  // Multi-buffer path: dyn set/wait + iv mod N selector.
  if (cp->eventIdInfo.isMultiBuffer()) {
    emitMultiBufferSetWait(rewriter, cp);
    return;
  }
  // Single-buffer path: classic static set/wait pair.
  const auto &eids = cp->eventIds;
  assert(!eids.empty());
  if (eids.empty())
    return;
  int64_t eid = eids[0];
  PIPE setPipe = cp->setCorePipeInfo.pipe;
  PIPE waitPipe = cp->waitCorePipeInfo.pipe;
  // SetFlag goes AFTER the producer anchor; WaitFlag goes BEFORE the
  // consumer anchor.
  insertSetFlag(rewriter, cp->setOp, setPipe, waitPipe, eid,
                /*insertAfter=*/true);
  insertWaitFlag(rewriter, cp->waitOp, setPipe, waitPipe, eid,
                 /*insertAfter=*/false);
}

void CodeGenerator::generateResultOps() {
  IRRewriter rewriter(funcOp.getContext());
  // Stable order: sort by (waitOp address, then id) just so codegen is
  // deterministic regardless of DenseMap iteration; emission order does not
  // affect correctness because each op is anchored independently.
  std::vector<ConflictPair *> ordered;
  ordered.reserve(chosenConflictedPairs.size());
  for (auto &cp : chosenConflictedPairs)
    ordered.push_back(cp.get());
  std::sort(ordered.begin(), ordered.end(),
            [](ConflictPair *a, ConflictPair *b) {
              if (a->endIndex != b->endIndex)
                return a->endIndex < b->endIndex;
              return a->id < b->id;
            });
  for (auto *cp : ordered)
    emitOne(rewriter, cp);
}
