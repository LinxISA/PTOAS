// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pto-graph-sync-solver-translator"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

IRTranslator::IRTranslator(func::FuncOp func, SyncSolverOptions opts)
    : options(opts), funcOp(func) {
  buildBufferAliasMap();

  // Build hierarchical funcIr tree.
  auto fn = std::make_unique<Function>(func.getOperation());
  auto scope = buildScopeFromRegion(func.getBody(), fn.get());
  fn->body.push_back(std::move(scope));
  funcIr = std::move(fn);

  // Build syncIr / processingOrders.
  syncIrBuilder(funcIr.get(), nullptr, /*depth=*/0, /*isUseless=*/false);
}

void IRTranslator::buildBufferAliasMap() {
  // Reuse the existing PTO InsertSync translator just to populate the alias
  // map. We discard its SyncIRs output entirely.
  SyncIRs throwawaySyncIR;
  MemoryDependentAnalyzer memAnalyzer;
  PTOIRTranslator helper(throwawaySyncIR, memAnalyzer, buffer2MemInfoMap,
                         funcOp, SyncAnalysisMode::NORMALSYNC);
  helper.Build();
}

void IRTranslator::collectMemInfo(
    Value v, llvm::SmallVector<const BaseMemInfo *> &out) const {
  if (!v)
    return;
  auto it = buffer2MemInfoMap.find(v);
  if (it == buffer2MemInfoMap.end())
    return;
  for (auto &mi : it->second)
    out.push_back(mi.get());
}

std::pair<llvm::SmallVector<const BaseMemInfo *>,
          llvm::SmallVector<const BaseMemInfo *>>
IRTranslator::getReadWriteMemInfo(Operation *op) const {
  llvm::SmallVector<const BaseMemInfo *> reads, writes;
  if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
    memEffect.getEffects(effects);
    for (auto &e : effects) {
      Value v = e.getValue();
      if (!v)
        continue;
      if (isa<MemoryEffects::Read>(e.getEffect()))
        collectMemInfo(v, reads);
      else if (isa<MemoryEffects::Write>(e.getEffect()))
        collectMemInfo(v, writes);
    }
  }
  return {std::move(reads), std::move(writes)};
}

std::unique_ptr<OperationBase>
IRTranslator::buildRWFromPipeOp(Operation *op, OperationBase *parentOp) {
  auto pipeOp = dyn_cast<pto::OpPipeInterface>(op);
  if (!pipeOp)
    return nullptr;
  PIPE pipe = pipeOp.getPipe();
  if (pipe == PIPE::PIPE_UNASSIGNED)
    return nullptr;

  auto [reads, writes] = getReadWriteMemInfo(op);
  if (reads.empty() && writes.empty()) {
    // Op has a Pipe but no tracked memory effects -> still record so the
    // GraphSolver can model pipe ordering even without memory dependencies.
  }
  return std::make_unique<RWOperation>(op, parentOp, pipe, pipe, reads, writes);
}

std::unique_ptr<Scope>
IRTranslator::buildScopeFromRegion(Region &region, OperationBase *parentOp) {
  auto scope = std::make_unique<Scope>();
  scope->parentOp = parentOp;

  bool isFunctionRegion = isa_and_present<Function>(parentOp);
  if (!isFunctionRegion && region.getBlocks().size() > 1) {
    // Conservative: arbitrary CFG inside non-function regions is not
    // supported in this minimal port. Bail out by treating it as empty.
    return scope;
  }

  for (Block &block : region.getBlocks()) {
    Scope *parScope = scope.get();
    if (isFunctionRegion) {
      auto fb = std::make_unique<FunctionBlock>();
      fb->parentOp = scope.get();
      parScope = fb.get();
      scope->body.push_back(std::move(fb));
    }

    auto blockBegin = std::make_unique<PlaceHolder>(nullptr, parScope);
    blockBegin->scopeBegin = parScope;
    blockBegin->block = &block;
    parScope->body.push_back(std::move(blockBegin));

    for (Operation &op : block.getOperations()) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        auto trueScope = buildScopeFromRegion(ifOp.getThenRegion(), nullptr);
        std::unique_ptr<Scope> falseScope;
        if (ifOp.elseBlock())
          falseScope = buildScopeFromRegion(ifOp.getElseRegion(), nullptr);
        auto cond =
            std::make_unique<Condition>(&op, parScope, std::move(trueScope),
                                        std::move(falseScope));
        parScope->body.push_back(std::move(cond));
        continue;
      }
      if (isa<LoopLikeOpInterface>(op)) {
        auto loop = std::make_unique<Loop>(&op, parScope);
        // Recognize a `pto.parallel_loop` UnitAttr as the parallel marker
        // (same idea as HIVM's `hivm.parallel_loop`).
        loop->isParallel =
            op.hasAttrOfType<UnitAttr>("pto.parallel_loop");
        for (Region &r : op.getRegions()) {
          auto innerScope = buildScopeFromRegion(r, loop.get());
          loop->body.push_back(std::move(innerScope));
        }
        auto beforePh = std::make_unique<PlaceHolder>(nullptr, loop->parentOp);
        beforePh->beforeOp = loop.get();
        auto afterPh = std::make_unique<PlaceHolder>(nullptr, loop->parentOp);
        afterPh->afterOp = loop.get();
        parScope->body.push_back(std::move(beforePh));
        parScope->body.push_back(std::move(loop));
        parScope->body.push_back(std::move(afterPh));
        continue;
      }
      if (isa<pto::OpPipeInterface>(op)) {
        if (auto rw = buildRWFromPipeOp(&op, parScope))
          parScope->body.push_back(std::move(rw));
        continue;
      }
      // Anything else (alloc_tile, make_tensor_view, arith, scf.yield, return,
      // ...) is structural metadata we don't model here.
    }

    auto blockEnd = std::make_unique<PlaceHolder>(nullptr, parScope);
    blockEnd->scopeEnd = parScope;
    blockEnd->block = &block;
    parScope->body.push_back(std::move(blockEnd));
  }

  return scope;
}

// ---- syncIr / processingOrders ---------------------------------------------

bool IRTranslator::skipLaterIterations(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 && occ2);
  auto skip = [&](Occurrence *o, Occurrence *other) {
    if (!o->parentOcc)
      return false;
    if (!isa<Loop>(o->parentOcc->op))
      return false;
    int split = o->parentOcc->loopSplitIndex;
    return o->syncIrIndex < split && split <= other->syncIrIndex;
  };
  return skip(occ1, occ2) || skip(occ2, occ1);
}

void IRTranslator::emitOrders(Occurrence *occ1, Occurrence *occ2,
                              bool isUseless) {
  assert(occ1 && occ2);
  if (skipLaterIterations(occ1, occ2))
    return;
  if (isa<Scope>(occ1->op) && isa<Scope>(occ2->op)) {
    emitOrdersForCross(occ1->childOccs, occ2->childOccs, isUseless);
    return;
  }
  if (isa<RWOperation>(occ1->op) && isa<Scope>(occ2->op)) {
    emitOrdersForCross({occ1}, occ2->childOccs, isUseless);
    return;
  }
  if (isa<Scope>(occ1->op) && isa<RWOperation>(occ2->op)) {
    emitOrdersForCross(occ1->childOccs, {occ2}, isUseless);
    return;
  }
  if (auto *rw1 = dyn_cast<RWOperation>(occ1->op)) {
    if (auto *rw2 = dyn_cast<RWOperation>(occ2->op)) {
      processingOrders.emplace_back(occ1, occ2, rw1, rw2, isUseless);
    }
  }
}

void IRTranslator::emitOrdersForList(
    const llvm::SmallVector<Occurrence *> &occs, bool isUseless) {
  int64_t n = static_cast<int64_t>(occs.size());
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i - 1; j >= 0; --j)
      emitOrders(occs[j], occs[i], isUseless);
}

void IRTranslator::emitOrdersForCross(
    const llvm::SmallVector<Occurrence *> &occs1,
    const llvm::SmallVector<Occurrence *> &occs2, bool isUseless) {
  for (auto *o2 : occs2)
    for (auto *o1 : llvm::reverse(occs1))
      emitOrders(o1, o2, isUseless);
}

void IRTranslator::syncIrBuilder(OperationBase *op, Occurrence *parentOcc,
                                 int depth, bool isUseless) {
  assert(op);
  int startIndex = ++globalIndex;
  auto occ =
      std::make_unique<Occurrence>(op, parentOcc, depth, startIndex, -1);
  occ->syncIrIndex = static_cast<int>(syncIr.size());
  syncIr.push_back(std::move(occ));
  Occurrence *occPtr = syncIr.back().get();
  opAllOccurrences[op].push_back(occPtr);
  if (parentOcc)
    parentOcc->childOccs.push_back(occPtr);

  if (auto *loopOp = dyn_cast<Loop>(op)) {
    // First "iteration" copy.
    for (auto &child : loopOp->body)
      syncIrBuilder(child.get(), occPtr, depth + 1, isUseless);
    occPtr->loopSplitIndex = static_cast<int>(syncIr.size());
    // Second "iteration" copy (marked isUseless=true so the solver knows it
    // should not generate net-new sync ops).
    for (auto &child : loopOp->body)
      syncIrBuilder(child.get(), occPtr, depth + 1, true);
    // Generate processing orders within this loop now that childOccs is full.
    int64_t childN = static_cast<int64_t>(occPtr->childOccs.size());
    if (childN > 0 && childN % 2 == 0) {
      llvm::SmallVector<Occurrence *> firstHalf(occPtr->childOccs.begin(),
                                                occPtr->childOccs.begin() +
                                                    childN / 2);
      llvm::SmallVector<Occurrence *> secondHalf(
          occPtr->childOccs.begin() + childN / 2, occPtr->childOccs.end());
      emitOrdersForList(firstHalf, isUseless);
      emitOrdersForList(secondHalf, /*isUseless=*/true);
      // Cross-iteration pairs: head of iter-2 vs tail of iter-1, using their
      // children to mirror the upstream semantics.
      for (auto *o2 : secondHalf)
        for (auto *o1 : llvm::reverse(firstHalf))
          emitOrdersForCross(o1->childOccs, o2->childOccs, isUseless);
    }
  } else if (auto *scopeOp = dyn_cast<Scope>(op)) {
    for (auto &child : scopeOp->body)
      syncIrBuilder(child.get(), occPtr, depth + 1, isUseless);
    emitOrdersForList(occPtr->childOccs, isUseless);
  }

  int endIndex = ++globalIndex;
  occPtr->endIndex = endIndex;
  occPtr->syncIrEndIndex = static_cast<int>(syncIr.size());
}
