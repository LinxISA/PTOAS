//===---------- SyncSolverCodeGen.cpp ---- Graph Sync Solver --------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolverCodeGen.h"

#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

static PipeAttr makePipe(MLIRContext *ctx, PIPE pipe) {
  return PipeAttr::get(ctx, pipe);
}

static EventAttr makeEvent(MLIRContext *ctx, int64_t eventId) {
  return EventAttr::get(ctx, static_cast<EVENT>(eventId));
}

Operation *CodeGenerator::resolveSyncAnchor(OperationBase *opBase) {
  if (!opBase)
    return nullptr;
  if (auto *ph = dyn_cast<PlaceHolder>(opBase)) {
    if (ph->beforeOp)
      return ph->beforeOp->op;
    if (ph->afterOp)
      return ph->afterOp->op;
    if (ph->block)
      return ph->block->getParentOp();
    return nullptr;
  }
  return opBase->op;
}

Location CodeGenerator::resolveSyncLoc(OperationBase *opBase) {
  if (Operation *anchor = resolveSyncAnchor(opBase))
    return anchor->getLoc();
  return funcOp.getLoc();
}

void CodeGenerator::setInsertionPoint(IRRewriter &rewriter,
                                      OperationBase *opBase,
                                      bool insertAfter) {
  Operation *anchor = resolveSyncAnchor(opBase);
  if (!anchor) {
    rewriter.setInsertionPointToStart(&funcOp.getBody().front());
    return;
  }
  if (insertAfter)
    rewriter.setInsertionPointAfter(anchor);
  else
    rewriter.setInsertionPoint(anchor);
}

void CodeGenerator::emitSyncOp(IRRewriter &rewriter, SyncOp *syncOp) {
  if (auto *barrier = dyn_cast<BarrierOp>(syncOp)) {
    rewriter.create<pto::BarrierOp>(resolveSyncLoc(barrier),
                                    makePipe(rewriter.getContext(),
                                             barrier->pipe));
    return;
  }

  auto *setWait = dyn_cast<SetWaitOp>(syncOp);
  if (!setWait || setWait->eventIds.empty())
    return;

  int64_t eventId = setWait->eventIds.front();
  auto srcAttr = makePipe(rewriter.getContext(), setWait->pipeSrc);
  auto dstAttr = makePipe(rewriter.getContext(), setWait->pipeDst);
  auto eventAttr = makeEvent(rewriter.getContext(), eventId);
  Location loc = resolveSyncLoc(setWait);

  if (isa<SetFlagOp>(setWait)) {
    rewriter.create<pto::SetFlagOp>(loc, srcAttr, dstAttr, eventAttr);
  } else if (isa<WaitFlagOp>(setWait)) {
    rewriter.create<pto::WaitFlagOp>(loc, srcAttr, dstAttr, eventAttr);
  }
}

void CodeGenerator::emitSyncMap(IRRewriter &rewriter, SyncMap &syncMap,
                                bool insertAfter) {
  for (auto &[opBase, syncOps] : syncMap) {
    setInsertionPoint(rewriter, opBase, insertAfter);
    for (auto &syncOp : syncOps)
      emitSyncOp(rewriter, syncOp.get());
  }
}

void CodeGenerator::generateResultOps() {
  IRRewriter rewriter(funcOp.getContext());
  emitSyncMap(rewriter, syncMapBefore, /*insertAfter=*/false);
  emitSyncMap(rewriter, syncMapAfter, /*insertAfter=*/true);
}
