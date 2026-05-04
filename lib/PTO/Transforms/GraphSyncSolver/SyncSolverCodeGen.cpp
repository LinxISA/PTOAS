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
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include <cassert>

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
  // Normal set/wait pair. Single event id in the minimal port.
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
