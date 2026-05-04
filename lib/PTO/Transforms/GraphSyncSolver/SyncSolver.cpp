// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/GraphSyncSolver/SyncSolver.h"

#include "PTO/Transforms/GraphSyncSolver/GraphSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cassert>
#include <set>
#include <utility>

#define DEBUG_TYPE "pto-graph-sync-solver"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

Solver::Solver(std::unique_ptr<IRTranslator> tr) : options(tr->options) {
  funcOp = tr->funcOp;
  funcIr = std::move(tr->funcIr);
  syncIr = std::move(tr->syncIr);
  opAllOccurrences = std::move(tr->opAllOccurrences);
  processingOrders = std::move(tr->processingOrders);
  buffer2MemInfoMap = std::move(tr->buffer2MemInfoMap);
}

void Solver::solve() { processOrders(); }

// ---- helpers ---------------------------------------------------------------

EventIdSolver *Solver::getEventIdSolver(PIPE setPipe, PIPE waitPipe) {
  auto key = std::make_pair(setPipe, waitPipe);
  auto it = eventIdSolvers_.find(key);
  if (it == eventIdSolvers_.end()) {
    auto s = std::make_unique<EventIdSolver>(options.eventIdNumMax);
    auto *raw = s.get();
    eventIdSolvers_[key] = std::move(s);
    return raw;
  }
  return it->second.get();
}

bool Solver::checkVisited(Occurrence *o1, Occurrence *o2) {
  auto [it, inserted] = processedOccPairs_.insert({o1, o2});
  return !inserted;
}

bool Solver::checkImpossibleOccPair(Occurrence *o1, Occurrence *o2) {
  assert(o1 && o2);
  if (o1->op == o2->op)
    return false;
  auto [par1, par2] = Occurrence::getLCAPair(o1, o2);
  // If two occurrences are the immediate trueScope/falseScope siblings of a
  // Condition, they cannot be live in the same execution path.
  return par1->parentOcc != nullptr &&
         par1->parentOcc == par2->parentOcc &&
         isa_and_present<Condition>(par1->parentOcc->op);
}

bool Solver::checkAlreadySynced(Occurrence *o1, Occurrence *o2) {
  // Mirrors HIVM's checkAlreadySynced: when the LCA loop properly contains
  // the LCA of the underlying ops AND every loop layer between them is
  // sequential (non-parallel), the loop boundary already serializes the
  // two operations, so we can skip emitting an extra sync.
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(o1, o2);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(o1->op, o2->op);
  if (!parOcc1->parentOcc || !parOp1->parentOp)
    return false;
  auto *parentLoop = OperationBase::getParentLoop(parOcc1->op);
  auto *curLoop = OperationBase::getParentLoop(parOp1);
  if (parentLoop == nullptr || parentLoop == curLoop)
    return false;
  assert(curLoop && parentLoop->isProperAncestor(curLoop));
  while (curLoop != parentLoop) {
    if (!llvm::cast<Loop>(curLoop)->isParallel)
      return true;
    curLoop = OperationBase::getParentLoop(curLoop);
    assert(curLoop);
  }
  return false;
}

bool Solver::checkSkipParallelLoop(Occurrence *o1, Occurrence *o2) {
  if (!isBackwardSync(o1, o2))
    return false;
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(o1, o2);
  auto *loopOcc = Occurrence::getParentLoop(parOcc1);
  if (!loopOcc)
    return false;
  return llvm::cast<Loop>(loopOcc->op)->isParallel;
}

bool Solver::isBackwardSync(Occurrence *o1, Occurrence *o2) {
  // Backward = the two occurrences live under the same parent occurrence
  // *Loop* in two different "iteration copies" produced by syncIrBuilder, or
  // the underlying op id ordering is reversed.
  if (o1->op->id >= o2->op->id)
    return true;
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(o1, o2);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(o1->op, o2->op);
  if (!parOcc1->parentOcc)
    return false;
  return parOcc1->parentOcc->op != parOp1->parentOp;
}

Occurrence *Solver::getBeforePlaceHolderOcc(Occurrence *loopOcc) {
  assert(loopOcc && isa<Scope>(loopOcc->op));
  int idx = loopOcc->syncIrIndex - 1;
  assert(idx >= 0 && idx < (int)syncIr.size());
  return syncIr[idx].get();
}

Occurrence *Solver::getAfterPlaceHolderOcc(Occurrence *loopOcc) {
  assert(loopOcc && isa<Scope>(loopOcc->op));
  int idx = loopOcc->syncIrEndIndex;
  assert(idx >= 0 && idx < (int)syncIr.size());
  return syncIr[idx].get();
}

std::pair<Occurrence *, Occurrence *>
Solver::getSetWaitOcc(Occurrence *o1, Occurrence *o2) {
  // Lift to the LCA siblings.
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(o1, o2);
  Occurrence *setOcc = parOcc1;
  Occurrence *waitOcc = parOcc2;
  // For backward syncs, push the anchors to the surrounding loop's
  // before / after PlaceHolder so the set lands AFTER the loop body of one
  // iteration and the wait lands BEFORE the next iteration's body.
  if (isBackwardSync(o1, o2)) {
    if (auto *loopOcc = Occurrence::getParentLoop(parOcc1)) {
      setOcc = getAfterPlaceHolderOcc(loopOcc);
      waitOcc = getBeforePlaceHolderOcc(loopOcc);
    }
  } else {
    // Forward: if either anchor itself is a Loop, snap to the surrounding
    // PlaceHolder so codegen has a stable insertion point.
    if (isa<Loop>(setOcc->op))
      setOcc = getAfterPlaceHolderOcc(setOcc);
    if (isa<Loop>(waitOcc->op))
      waitOcc = getBeforePlaceHolderOcc(waitOcc);
  }
  return {setOcc, waitOcc};
}

// ---- memory conflict enumeration -------------------------------------------

llvm::SmallVector<std::pair<CorePipeInfo, CorePipeInfo>>
Solver::checkMemoryConflicts(RWOperation *r1, RWOperation *r2) {
  assert(r1 && r2);
  auto key = std::make_pair(r1, r2);
  auto [it, inserted] = memConflictMem_.insert({key, {}});
  if (!inserted)
    return it->second;

  auto &out = it->second;

  auto checkPair = [&](const llvm::SmallVector<const BaseMemInfo *> &a,
                       const llvm::SmallVector<const BaseMemInfo *> &b) {
    DepBaseMemInfoPairVec tmp;
    return memAnalyzer_.DepBetween(a, b, tmp);
  };

  // RAW (r1 reads, r2 writes)
  if (checkPair(r1->readMemInfo, r2->writeMemInfo))
    out.push_back({CorePipeInfo(TCoreType::CUBE_OR_VECTOR, r1->pipeRead),
                   CorePipeInfo(TCoreType::CUBE_OR_VECTOR, r2->pipeWrite)});
  // WAR (r1 writes, r2 reads)
  if (checkPair(r1->writeMemInfo, r2->readMemInfo))
    out.push_back({CorePipeInfo(TCoreType::CUBE_OR_VECTOR, r1->pipeWrite),
                   CorePipeInfo(TCoreType::CUBE_OR_VECTOR, r2->pipeRead)});
  // WAW (r1 writes, r2 writes)
  if (checkPair(r1->writeMemInfo, r2->writeMemInfo))
    out.push_back({CorePipeInfo(TCoreType::CUBE_OR_VECTOR, r1->pipeWrite),
                   CorePipeInfo(TCoreType::CUBE_OR_VECTOR, r2->pipeWrite)});
  return out;
}

// ---- main loop -------------------------------------------------------------

void Solver::processOrders() {
  for (auto &po : processingOrders) {
    Occurrence *o1 = po.occ1;
    Occurrence *o2 = po.occ2;
    if (o1 == o2)
      continue;
    if (o1->syncIrIndex >= o2->syncIrIndex)
      continue;
    if (checkVisited(o1, o2))
      continue;
    if (checkImpossibleOccPair(o1, o2))
      continue;
    if (checkAlreadySynced(o1, o2))
      continue;
    if (checkSkipParallelLoop(o1, o2))
      continue;
    processConflict(o1, o2, po.rwOp1, po.rwOp2);
  }
}

void Solver::processConflict(Occurrence *o1, Occurrence *o2, RWOperation *r1,
                             RWOperation *r2) {
  // RAW/WAR/WAW can all fire between the same two ops while mapping to the
  // same (pipeSrc, pipeDst). Emit at most one sync candidate per pipe pair.
  std::set<std::pair<CorePipeInfo, CorePipeInfo>> seen;
  for (auto [src, dst] : checkMemoryConflicts(r1, r2)) {
    if (!seen.insert({src, dst}).second)
      continue;
    handleConflict(o1, o2, r1, r2, src, dst);
  }
}

void Solver::handleConflict(Occurrence *o1, Occurrence *o2, RWOperation *r1,
                            RWOperation *r2, CorePipeInfo src,
                            CorePipeInfo dst) {
  (void)r1;
  (void)r2;
  if (!checkGraphConflict(o1, o2, src, dst))
    return; // already covered transitively
  if (src == dst)
    handleBarrierConflict(o1, o2, src, dst);
  else
    handleSetWaitConflict(o1, o2, src, dst);
}

bool Solver::checkGraphConflict(Occurrence *o1, Occurrence *o2,
                                CorePipeInfo src, CorePipeInfo dst) {
  GraphSolver gs;
  llvm::DenseSet<ConflictPair *> visited;
  auto consider = [&](ConflictPair *cp) {
    if (cp->isBarrierAll || cp->isBarrier())
      return;
    if (cp->endIndex < o1->endIndex || cp->startIndex > o2->startIndex)
      return;
    if (!visited.insert(cp).second)
      return;
    gs.addConflictPair(cp);
  };
  for (auto *p : o1->getAllParents()) {
    auto it = scopeOccChosenConflicts_.find(p);
    if (it != scopeOccChosenConflicts_.end())
      for (auto *cp : it->second)
        consider(cp);
  }
  for (auto *p : o2->getAllParents()) {
    auto it = scopeOccChosenConflicts_.find(p);
    if (it != scopeOccChosenConflicts_.end())
      for (auto *cp : it->second)
        consider(cp);
  }
  auto dist = gs.runDijkstra(src, dst, o1->endIndex, o2->startIndex);
  return !dist.has_value() || dist.value() > o2->startIndex;
}

void Solver::insertBarrierAllBeforeOcc(Occurrence *occ) {
  assert(occ && occ->parentOcc);
  // Manufacture a synthetic ConflictPair flagged as barrier-all; codegen
  // will translate this to `pto.barrier <PIPE_ALL>` placed just before the
  // wait-op's anchor.
  auto cp = std::make_unique<ConflictPair>(
      /*op1=*/dyn_cast_if_present<RWOperation>(occ->op),
      /*op2=*/dyn_cast_if_present<RWOperation>(occ->op),
      /*setOp=*/occ->op, /*waitOp=*/occ->op,
      /*setOcc=*/occ, /*waitOcc=*/occ,
      CorePipeInfo(TCoreType::CUBE_OR_VECTOR, PIPE::PIPE_ALL),
      CorePipeInfo(TCoreType::CUBE_OR_VECTOR, PIPE::PIPE_ALL),
      occ->startIndex, occ->startIndex);
  cp->isBarrierAll = true;
  scopeOccChosenConflicts_[occ->parentOcc].insert(cp.get());
  chosenConflictedPairs.push_back(std::move(cp));
}

std::vector<ConflictPair *>
Solver::getIntersectingConflictPairs(ConflictPair *cp) const {
  std::vector<ConflictPair *> out;
  if (cp->isBarrier() || cp->isBarrierAll)
    return out;
  for (auto &cur : chosenConflictedPairs) {
    if (cur.get() == cp || cur->isBarrier() || cur->isBarrierAll)
      continue;
    if (cur->setCorePipeInfo != cp->setCorePipeInfo ||
        cur->waitCorePipeInfo != cp->waitCorePipeInfo)
      continue;
    if (checkRangesIntersect(cp->startIndex, cp->endIndex + 1,
                             cur->startIndex, cur->endIndex + 1))
      out.push_back(cur.get());
  }
  return out;
}

void Solver::handleBarrierConflict(Occurrence *o1, Occurrence *o2,
                                   CorePipeInfo src, CorePipeInfo dst) {
  assert(src == dst);
  if (src.pipe == PIPE::PIPE_S)
    return; // PIPE_S is naturally serialized; emitting a barrier is wasteful.
  // Anchor the barrier at o2's wait-side parent so it lands right before the
  // consumer.
  Occurrence *waitOcc = o2;
  auto cp = std::make_unique<ConflictPair>(
      dyn_cast_if_present<RWOperation>(o1->op),
      dyn_cast_if_present<RWOperation>(o2->op),
      o2->op, o2->op, o2, o2, src, dst, o2->startIndex, o2->startIndex);
  scopeOccChosenConflicts_[waitOcc->parentOcc].insert(cp.get());
  chosenConflictedPairs.push_back(std::move(cp));
}

void Solver::handleSetWaitConflict(Occurrence *o1, Occurrence *o2,
                                   CorePipeInfo src, CorePipeInfo dst) {
  auto [setOcc, waitOcc] = getSetWaitOcc(o1, o2);
  assert(setOcc && waitOcc);

  auto cp = std::make_unique<ConflictPair>(
      dyn_cast_if_present<RWOperation>(o1->op),
      dyn_cast_if_present<RWOperation>(o2->op),
      setOcc->op, waitOcc->op, setOcc, waitOcc, src, dst, setOcc->endIndex,
      waitOcc->startIndex);

  // Speculatively color: try inserting this candidate into the EventIdSolver
  // and roll back if the graph would exceed the hardware budget.
  auto *colorer = getEventIdSolver(src.pipe, dst.pipe);
  colorer->pushActionNone();
  cp->eventIdNode = colorer->createNode(cp.get(), /*eventIdNum=*/1);
  std::vector<ConflictPair *> intersecting = getIntersectingConflictPairs(cp.get());
  colorer->addConflicts(cp.get(), intersecting);

  if (!colorer->isColorable()) {
    // Fallback: this is the minimal port's "no multi-strategy retry" knob.
    // Drop the speculative coloring and emit a single PIPE_ALL barrier.
    colorer->undoActions();
    insertBarrierAllBeforeOcc(waitOcc);
    return;
  }
  colorer->clearActionStack();

  // Attach to LCA scope occurrences so future checkGraphConflict calls see it.
  auto [normSet, normWait] = OperationBase::getLCAPair(setOcc->op, waitOcc->op);
  Occurrence *normScopeOcc1 = setOcc->getParentWithOp(normSet->parentOp);
  Occurrence *normScopeOcc2 = waitOcc->getParentWithOp(normWait->parentOp);
  if (normScopeOcc1)
    scopeOccChosenConflicts_[normScopeOcc1].insert(cp.get());
  if (normScopeOcc2 && normScopeOcc2 != normScopeOcc1)
    scopeOccChosenConflicts_[normScopeOcc2].insert(cp.get());

  chosenConflictedPairs.push_back(std::move(cp));
}
