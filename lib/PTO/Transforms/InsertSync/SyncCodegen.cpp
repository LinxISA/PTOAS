// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;
 
// ==============================================================================
// 1. Helper Functions
// ==============================================================================
 
static pto::PipeAttr getPipeAttr(Builder &builder, PipelineType pipe) {
  auto odsPipeVal = static_cast<pto::PIPE>(pipe);
  return pto::PipeAttr::get(builder.getContext(), odsPipeVal);
}
 
static pto::EventAttr getEventAttr(Builder &builder, int id) {
  auto odsEventVal = static_cast<pto::EVENT>(id);
  return pto::EventAttr::get(builder.getContext(), odsEventVal);
}
 
static bool IsSameSyncSignature(const SyncOperation *existing,
                                const SyncOperation *candidate) {
  if (existing->GetType() != candidate->GetType())
    return false;
  if (existing->GetActualSrcPipe() != candidate->GetActualSrcPipe())
    return false;
  if (existing->GetActualDstPipe() != candidate->GetActualDstPipe())
    return false;
  if (existing->IsAutoSyncTailBarrier() != candidate->IsAutoSyncTailBarrier())
    return false;
  if (candidate->isSyncSetType() || candidate->isSyncWaitType())
    return existing->eventIds == candidate->eventIds;
  return true;
}

static bool IsSyncExist(const SyncOps &list, SyncOperation *newSync) {
  // Tombstone entries are soft-deleted and should never participate in
  // deduplication; otherwise they can shadow a later live sync with
  // the same signature.
  if (newSync->uselessSync)
    return true;

  for (auto *existing : list) {
    if (existing == newSync)
      return true;
    if (existing->uselessSync)
      continue;
    if (!IsSameSyncSignature(existing, newSync))
      continue;
    return true;
  }
  return false;
}
 
static void MergeSyncList(SyncOps &dstList, const SyncOps &srcList) {
  for (auto *sync : srcList) {
    if (sync->uselessSync)
      continue;
    if (!IsSyncExist(dstList, sync)) {
      dstList.push_back(sync);
    }
  }
}

static Operation *resolveSyncInsertAnchor(Operation *op, SyncOperation *sync) {
  if (!sync->isCompensation)
    return op;

  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (!ifOp.elseBlock()) {
      OpBuilder builder(ifOp.getContext());
      Block *elseBlock = new Block();
      ifOp.getElseRegion().push_back(elseBlock);
      builder.setInsertionPointToEnd(elseBlock);
      builder.create<scf::YieldOp>(ifOp.getLoc());
    }
    return ifOp.getElseRegion().front().getTerminator();
  }

  if (op->hasTrait<OpTrait::IsTerminator>())
    return op;
  return op->getBlock()->getTerminator();
}

static bool shouldInsertBefore(Operation *op, bool beforeInsert,
                               const SyncOperation *sync) {
  return beforeInsert || sync->isCompensation ||
         op->hasTrait<OpTrait::IsTerminator>();
}

static void setSyncInsertionPoint(IRRewriter &rewriter, Operation *op,
                                  bool insertBefore) {
  if (insertBefore) {
    rewriter.setInsertionPoint(op);
    return;
  }
  rewriter.setInsertionPointAfter(op);
}

static bool hasNeighborBarrier(Block *block, Block::iterator ip,
                               pto::PipeAttr pipeAttr, bool insertBefore) {
  if (insertBefore) {
    if (ip == block->begin())
      return false;
    auto prevBarrier = dyn_cast<pto::BarrierOp>(&*std::prev(ip));
    return prevBarrier && prevBarrier.getPipe() == pipeAttr;
  }

  if (ip == block->end())
    return false;
  auto nextBarrier = dyn_cast<pto::BarrierOp>(&*ip);
  return nextBarrier && nextBarrier.getPipe() == pipeAttr;
}

static void createSetOrWaitFlagOp(IRRewriter &rewriter, Operation *op,
                                  SyncOperation *sync, pto::PipeAttr srcPipe,
                                  pto::PipeAttr dstPipe,
                                  pto::EventAttr eventId) {
  if (sync->isSyncWaitType()) {
    rewriter.create<pto::WaitFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
    return;
  }
  rewriter.create<pto::SetFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
}

static void createSetOrWaitFlagDynOp(IRRewriter &rewriter, Operation *op,
                                     SyncOperation *sync, pto::PipeAttr srcPipe,
                                     pto::PipeAttr dstPipe, Value eventIndex) {
  if (sync->isSyncWaitType()) {
    rewriter.create<pto::WaitFlagDynOp>(op->getLoc(), srcPipe, dstPipe,
                                        eventIndex);
    return;
  }
  rewriter.create<pto::SetFlagDynOp>(op->getLoc(), srcPipe, dstPipe, eventIndex);
}

// ==============================================================================
// 2. SyncCodegen Implementation
// ==============================================================================
 
void SyncCodegen::Run() {
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  
  UpdateOpInsertSync(rewriter);
 
  // [Optional Debug] 这里的 Debug 打印可以保留或注释掉
  // ...
 
  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op2InsertSync.count(op)) {
      // 处理 PRE Sync
      for (auto &syncBefore : op2InsertSync[op].pipeBefore) {
        SyncInsert(rewriter, op, syncBefore, true);
      }
      // 处理 POST Sync (逆序遍历，为了保持插入后的顺序正确)
      for (auto &syncAfter : llvm::reverse(op2InsertSync[op].pipeAfter)) {
        SyncInsert(rewriter, op, syncAfter, false);
      }
    }
  });

  // Ensure the tail clean barrier is emitted at function tail, right before
  // return, instead of being interleaved with other trailing sync ops.
  AppendAutoSyncTailBarrierIfNeeded(rewriter);
}
 
void SyncCodegen::UpdateOpInsertSync(IRRewriter &rewriter) {
  for (auto &nowElement : syncIR_) {
    if (auto *compoundElement = dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      UpdateCompoundOpInsertSync(compoundElement);
    } else if (auto *placeHolder = dyn_cast<PlaceHolderInstanceElement>(nowElement.get())) {
      updatePlaceHolderOpInsertSync(placeHolder);
    } else if (auto *loopElement = dyn_cast<LoopInstanceElement>(nowElement.get())) {
      UpdateLoopOpInsertSync(loopElement);
    } else if (auto *branchElement = dyn_cast<BranchInstanceElement>(nowElement.get())) {
      UpdateBranchOpInsertSync(branchElement);
    }
  }
}
 
void SyncCodegen::UpdateCompoundOpInsertSync(CompoundInstanceElement *nowCompound) {
  auto &pipeBuild = op2InsertSync[nowCompound->elementOp];
  MergeSyncList(pipeBuild.pipeBefore, nowCompound->pipeBefore);
  MergeSyncList(pipeBuild.pipeAfter, nowCompound->pipeAfter);
}
 
void SyncCodegen::UpdateLoopOpInsertSync(LoopInstanceElement *nowElement) {
  if (nowElement->getLoopKind() == KindOfLoop::LOOP_END) {
    auto *loopBegin = dyn_cast<LoopInstanceElement>(syncIR_[nowElement->beginId].get());
    auto &pipeBuild = op2InsertSync[nowElement->elementOp];
    MergeSyncList(pipeBuild.pipeBefore, loopBegin->pipeBefore);
    MergeSyncList(pipeBuild.pipeAfter, nowElement->pipeAfter);
  }
}
 
void SyncCodegen::UpdateBranchOpInsertSync(BranchInstanceElement *nowElement) {
  if (nowElement->getBranchKind() == KindOfBranch::IF_END) {
    auto *branchBegin = dyn_cast<BranchInstanceElement>(syncIR_[nowElement->beginId].get());
    auto &pipeBuild = op2InsertSync[nowElement->elementOp];
    MergeSyncList(pipeBuild.pipeBefore, branchBegin->pipeBefore);
    MergeSyncList(pipeBuild.pipeAfter, nowElement->pipeAfter);
  }
}
 
void SyncCodegen::updatePlaceHolderOpInsertSync(PlaceHolderInstanceElement *placeHolder) {
  // 1. 处理 Virtual Else
  if (placeHolder->isVirtualElse) {
      auto ifOp = dyn_cast<scf::IfOp>(placeHolder->parentIfOp);
      if (!ifOp) return;
 
      // 如果还没有 else block，创建一个
      if (!ifOp.elseBlock()) {
          OpBuilder builder(ifOp.getContext());
          // 只有当确实有 Sync 指令需要插入时才创建
          if (!placeHolder->pipeBefore.empty() || !placeHolder->pipeAfter.empty()) {
               Region &elseRegion = ifOp.getElseRegion();
               Block *elseBlock = new Block();
               elseRegion.push_back(elseBlock);
               builder.setInsertionPointToEnd(elseBlock);
               builder.create<scf::YieldOp>(ifOp.getLoc());
          }
      }
      
      // 更新映射：将 Virtual Placeholder 映射到新创建的 Yield Op
      if (ifOp.elseBlock()) {
          placeHolder->elementOp = ifOp.getElseRegion().front().getTerminator();
      } else {
          // 依然没有 Sync 需要插入，直接返回
          return;
      }
  } 
  // 2. 处理 Normal PlaceHolder (Then End or Existing Else End)
  else if (placeHolder->elementOp == placeHolder->parentIfOp) {
      // 之前的 Translator 逻辑把 Normal Placeholder 也映射到了 ifOp
      // 我们需要修正它指向 Yield
      auto ifOp = dyn_cast<scf::IfOp>(placeHolder->elementOp);
      // 判断是 Then 还是 Else
      // 简单判断：看 index。或者 Translator 里直接存 Yield Op。
      // 这里假设 Translator 存的是 IfOp，我们需要找到对应的 Yield。
      // ... 
      // 建议在 Translator 里直接让 elementOp 指向 Yield Op（如果存在）。
  }
 
  // 执行常规的 Sync 插入
  if (!placeHolder->elementOp) return;
  auto &pipeBuild = op2InsertSync[placeHolder->elementOp];
  MergeSyncList(pipeBuild.pipeBefore, placeHolder->pipeBefore);
  MergeSyncList(pipeBuild.pipeAfter, placeHolder->pipeAfter);
}
 
void SyncCodegen::SyncInsert(IRRewriter &rewriter, Operation *op,
                             SyncOperation *sync, bool beforeInsert) {
  if (sync->uselessSync)
    return;

  Operation *insertAnchorOp = resolveSyncInsertAnchor(op, sync);
  bool forceBefore = shouldInsertBefore(insertAnchorOp, beforeInsert, sync);
  if (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER) {
    CreateBarrierOp(rewriter, insertAnchorOp, sync, forceBefore);
  } else if (sync->isSyncSetType() || sync->isSyncWaitType()) {
    if (sync->eventIdNum > 1 && sync->eventIds.size() > 1) {
      CreateSetWaitOpForMultiBuffer(rewriter, insertAnchorOp, sync, forceBefore);
    } else {
      CreateSetWaitOpForSingleBuffer(rewriter, insertAnchorOp, sync, forceBefore);
    }
  }
}
 
// [核心修改] 加强版 CreateBarrierOp
void SyncCodegen::CreateBarrierOp(IRRewriter &rewriter, Operation *op,
                                  SyncOperation *sync, bool beforeInsert) {
  // A5: PIPE_V intra-pipe ordering is guaranteed by hardware; do not emit
  // explicit vector barrier (it is also rejected by backend checks).
  if (isTargetArchA5(func_.getOperation()) &&
      sync->GetActualSrcPipe() == PipelineType::PIPE_V) {
    return;
  }

  // Only the compiler-inserted tail clean barrier is deferred to function tail.
  // Other PIPE_ALL barriers, including event-id-exhaustion fallbacks, must stay
  // at their original program point to preserve local ordering.
  if (sync->IsAutoSyncTailBarrier()) {
    pendingAutoSyncTailBarrier_ = true;
    return;
  }

  bool insertAtPos = beforeInsert || op->hasTrait<OpTrait::IsTerminator>();
  setSyncInsertionPoint(rewriter, op, insertAtPos);
  Block *block = rewriter.getInsertionBlock();
  Block::iterator ip = rewriter.getInsertionPoint();
  auto currentPipeAttr = getPipeAttr(rewriter, sync->GetActualSrcPipe());

  if (hasNeighborBarrier(block, ip, currentPipeAttr, insertAtPos))
    return;

  auto barrier = rewriter.create<pto::BarrierOp>(op->getLoc(), currentPipeAttr);

  (void)barrier;
}

void SyncCodegen::AppendAutoSyncTailBarrierIfNeeded(IRRewriter &rewriter) {
  if (!pendingAutoSyncTailBarrier_)
    return;

  SmallVector<func::ReturnOp, 4> returns;
  func_.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  if (returns.empty())
    return;

  auto pipeAllAttr = getPipeAttr(rewriter, PipelineType::PIPE_ALL);
  for (auto ret : returns) {
    rewriter.setInsertionPoint(ret);
    auto barrier = rewriter.create<pto::BarrierOp>(ret.getLoc(), pipeAllAttr);
    barrier->setAttr("pto.auto_sync_tail_barrier", rewriter.getUnitAttr());
    if (auto hintAttr =
            func_->getAttrOfType<mlir::StringAttr>("pto.auto_sync_tail_hint")) {
      barrier->setAttr("pto.auto_sync_tail_hint", hintAttr);
    }
  }

  pendingAutoSyncTailBarrier_ = false;
}
 
void SyncCodegen::CreateSetWaitOpForSingleBuffer(IRRewriter &rewriter,
                                                 Operation *op,
                                                 SyncOperation *sync,
                                                 bool beforeInsert) {
  setSyncInsertionPoint(rewriter, op,
                        beforeInsert || op->hasTrait<OpTrait::IsTerminator>());
  auto srcPipe = getPipeAttr(rewriter, sync->GetActualSrcPipe());
  auto dstPipe = getPipeAttr(rewriter, sync->GetActualDstPipe());
  auto eventId = getEventAttr(rewriter, sync->eventIds[0]);
  createSetOrWaitFlagOp(rewriter, op, sync, srcPipe, dstPipe, eventId);
}
 
void SyncCodegen::CreateSetWaitOpForMultiBuffer(IRRewriter &rewriter,
                                                Operation *op,
                                                SyncOperation *sync,
                                                bool beforeInsert) {
  Value eventIdxDyn;
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    eventIdxDyn = GetBufferSelected(rewriter, op, sync);
  }
  setSyncInsertionPoint(
      rewriter, op, beforeInsert || op->hasTrait<OpTrait::IsTerminator>());
  auto srcPipe = getPipeAttr(rewriter, sync->GetActualSrcPipe());
  auto dstPipe = getPipeAttr(rewriter, sync->GetActualDstPipe());
  if (!eventIdxDyn) {
    int id0 = sync->eventIds.empty() ? 0 : sync->eventIds[0];
    auto eventId = getEventAttr(rewriter, id0);
    createSetOrWaitFlagOp(rewriter, op, sync, srcPipe, dstPipe, eventId);
    return;
  }
  createSetOrWaitFlagDynOp(rewriter, op, sync, srcPipe, dstPipe, eventIdxDyn);
}

// P0-2: Slot index must be the LOGICAL iteration count modulo N, not `iv % N`.
// Compute `((iv - lb) / step) % N`, with a fast path for the canonical
// step=1, lb=0 case so we don't bloat IR for the common scenario.
static bool isyncIsConstantIndexEqualTo(Value v, int64_t target) {
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

Value SyncCodegen::GetBufferSelected(IRRewriter &rewriter, Operation *op,
                                     SyncOperation *sync) {
  if (SyncIndex2SelectBuffer.count(sync->GetSyncIndex())) {
    return SyncIndex2SelectBuffer[sync->GetSyncIndex()];
  }

  unsigned N = static_cast<unsigned>(sync->eventIdNum);
  if (N <= 1 || sync->eventIds.size() < N)
    return nullptr;

  auto parentLoop = op->getParentOfType<scf::ForOp>();
  if (!parentLoop)
    return nullptr;

  Value counter;
  auto loopIt = loop2BufferCounter.find(parentLoop);
  if (loopIt != loop2BufferCounter.end() && loopIt->second.second == N) {
    counter = loopIt->second.first;
  } else {
    rewriter.setInsertionPointToStart(parentLoop.getBody());
    Value iv = parentLoop.getInductionVar();
    Value cN = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), N);
    Value lb = parentLoop.getLowerBound();
    Value step = parentLoop.getStep();
    Value normalized = iv;
    if (!isyncIsConstantIndexEqualTo(lb, 0))
      normalized = rewriter.create<arith::SubIOp>(op->getLoc(), normalized, lb);
    if (!isyncIsConstantIndexEqualTo(step, 1))
      normalized = rewriter.create<arith::DivUIOp>(op->getLoc(), normalized, step);
    counter = rewriter.create<arith::RemUIOp>(op->getLoc(), normalized, cN);
    loop2BufferCounter[parentLoop] = {counter, N};
  }

  rewriter.setInsertionPointAfter(counter.getDefiningOp());
  Value selected =
      rewriter.create<arith::ConstantIndexOp>(op->getLoc(), sync->eventIds[0]);
  for (unsigned i = 1; i < N; ++i) {
    Value ci = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i);
    Value eq = rewriter.create<arith::CmpIOp>(op->getLoc(), arith::CmpIPredicate::eq,
                                              counter, ci);
    Value idv =
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), sync->eventIds[i]);
    selected = rewriter.create<arith::SelectOp>(op->getLoc(), eq, idv, selected);
  }

  SyncIndex2SelectBuffer[sync->GetSyncIndex()] = selected;
  return selected;
}
