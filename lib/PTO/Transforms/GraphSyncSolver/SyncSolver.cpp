// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===--------- SyncSolver.cpp ------- Graph Sync Solver -------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/GraphSyncSolver/SyncSolver.h"
#include "PTO/Transforms/GraphSyncSolver/GraphSolver.h"
#include "PTO/Transforms/GraphSyncSolver/MemInfo.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "PTO-gss-solver"

using namespace mlir;
using namespace pto::syncsolver;

// Reset per-pass bookkeeping to start fresh.
void Solver::reset(bool resetEventIdRanOutOpts) {
  if (resetEventIdRanOutOpts) {
    reusePairs.clear();
    disabledMultiEventIdPairs.clear();
    backwardSyncEventsAfterMerge.clear();
    moveBackwardSyncPairsToOutmostLoop = false;
    dontMoveBackwardSyncPairsToOutmostLoop = false;
  }
  skipOcc.clear();
  syncedPairs.clear();
  processedOccPairs.clear();
  chosenConflictedPairs.clear();
  scopeOccChosenConflicts.clear();
  scopeOccPairChosenConflicts.clear();
  backwardSyncEvents.clear();
  replacedWithReusableSyncedPairs.clear();
  reusedPairs.clear();
  barrierAllPairs.clear();
  insertedBarrierAllBefore.clear();
  eventIdSolver.clear();
  resetUnitFlag();
}

void Solver::resetUnitFlag() {
  for (auto *rwOp : unitFlagFeaturedOps) {
    rwOp->mergedUnitFlagInfo.reset();
    for (auto *occ : opAllOccurrences[rwOp]) {
      occ->unitFlagInfo.reset();
    }
  }
}

// Helpers to find first/last iteration occurrences relative to parent
// occurrences.
Occurrence *Solver::getFirstIterOcc(Occurrence *occ, Occurrence *parOcc) {
  assert(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getFirstIterOcc(
        occ->getNthParent(occ->depth - parOcc->depth - 1), parOcc);
    return getFirstIterOcc(occ, newParOcc);
  }
  auto *it =
      std::find_if(parOcc->childOccs.begin(), parOcc->childOccs.end(),
                   [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != parOcc->childOccs.end());
  return *it;
}

Occurrence *Solver::getLastIterOcc(Occurrence *occ, Occurrence *parOcc) {
  assert(occ != nullptr && parOcc != nullptr);
  if (parOcc->depth + 1 < occ->depth) {
    auto *newParOcc = getLastIterOcc(
        occ->getNthParent(occ->depth - parOcc->depth - 1), parOcc);
    return getLastIterOcc(occ, newParOcc);
  }
  auto it =
      std::find_if(parOcc->childOccs.rbegin(), parOcc->childOccs.rend(),
                   [occ](Occurrence *curOcc) { return occ->op == curOcc->op; });
  assert(it != parOcc->childOccs.rend());
  return *it;
}

bool Solver::checkSkipCrossCorePair(Occurrence *occ1, Occurrence *occ2) {
  if (!options.isCrossCoreMode()) {
    return false;
  }
  auto *rwOp1 = llvm::dyn_cast<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(rwOp1->coreType != pto::TCoreType::CUBE_OR_VECTOR);
  assert(rwOp2->coreType != pto::TCoreType::CUBE_OR_VECTOR);
  if (rwOp1->coreType == rwOp2->coreType) {
    return true;
  }
  if (rwOp1->coreType == pto::TCoreType::CUBE_AND_VECTOR) {
    return true;
  }
  return false;
}

bool Solver::checkSkipParallelLoop(Occurrence *occ1, Occurrence *occ2) {
  if (!isBackwardSync(occ1, occ2)) {
    return false;
  }
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  auto *parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
  assert(parentLCALoopOcc != nullptr);
  auto *parentLCALoopOp = llvm::cast<Loop>(parentLCALoopOcc->op);
  return parentLCALoopOp->isParallel;
}

// Check whether occurrences belong to impossible (if-else) pairing.
bool Solver::checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (occ1->op == occ2->op) {
    return false;
  }
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  bool isIfElseSituation =
      parOcc1->parentOcc != nullptr &&
      parOcc1->parentOcc == parOcc2->parentOcc &&
      llvm::isa_and_present<Condition>(parOcc1->parentOcc->op);
  return isIfElseSituation;
}

// Detect whether occ1 and occ2 have already been covered by an earlier sync.
bool Solver::checkAlreadySynced(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(occ1->op != nullptr && occ2->op != nullptr);

  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  assert(parOcc1->parentOcc != nullptr && parOcc2->parentOcc != nullptr);

  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  assert(parOp1 != nullptr && parOp2 != nullptr);
  assert(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);

  auto *parentLoop = OperationBase::getParentloop(parOcc1->op);
  auto *curLoop = OperationBase::getParentloop(parOp1);
  if (parentLoop == nullptr || parentLoop == curLoop) {
    return false;
  }

  assert(curLoop != nullptr);
  assert(parentLoop->isProperAncestor(curLoop));
  while (curLoop != parentLoop) {
    if (!llvm::cast<Loop>(curLoop)->isParallel) {
      return true;
    }
    curLoop = OperationBase::getParentloop(curLoop);
    assert(curLoop != nullptr);
  }
  return false;
}

// Unit-flag reuse check between two RWOperations.
bool Solver::checkAlreadySyncedWithUnitFlag(Occurrence *occ1,
                                            Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (!options.enableUnitFlagFeature) {
    return false;
  }
  if (!occ1->hasUnitFlagFeat || !occ2->hasUnitFlagFeat) {
    return false;
  }
  llvm::DenseSet<Occurrence *> visited;
  DEBUG_WITH_TYPE("gss-sync-solver-check-unit-flag", {
    llvm::dbgs() << "unit-flag-step: " << occ1->syncIrIndex << ' '
                 << occ1->op->str(0, false) << "\n";
  });
  Occurrence *curOcc = occ1->unitFlagInfo.linkedElementAsSet;
  while (curOcc != nullptr) {
    DEBUG_WITH_TYPE("gss-sync-solver-check-unit-flag", {
      llvm::dbgs() << "unit-flag-step: " << curOcc->syncIrIndex << ' '
                   << curOcc->op->str(0, false) << "\n";
    });
    auto [it, isInserted] = visited.insert(curOcc);
    if (!isInserted) {
      break;
    }
    if (curOcc == occ2) {
      return true;
    }
    curOcc = curOcc->unitFlagInfo.linkedElementAsSet;
  }
  return false;
}

bool Solver::ignoreMemoryConflict(RWOperation *rwOp1, RWOperation *rwOp2,
                                  const MemInfo &memInfo1,
                                  const MemInfo &memInfo2) {
  if (options.isIntraCoreMode()) {
    if (memInfo1.isWorkSpace && memInfo2.isWorkSpace) {
      if (options.intraCoreIgnoreWorkSpaceFunctionArguments) {
        return true;
      }
    }
  }
  return false;
}

bool Solver::checkMemInfoConflict(RWOperation *rwOp1, RWOperation *rwOp2,
                                  const MemInfo &memInfo1,
                                  const MemInfo &memInfo2,
                                  std::optional<int64_t> lcmLen,
                                  std::optional<int64_t> eventIdNum) {
  if (ignoreMemoryConflict(rwOp1, rwOp2, memInfo1, memInfo2)) {
    return false;
  }
  return MemInfo::checkConflict(memInfo1, memInfo2, lcmLen, eventIdNum);
}

bool Solver::checkMemInfoConflict(
    RWOperation *rwOp1, RWOperation *rwOp2,
    const llvm::SmallVector<MemInfo> &memInfoList1,
    const llvm::SmallVector<MemInfo> &memInfoList2,
    std::optional<int64_t> lcmLen, std::optional<int64_t> eventIdNum) {
  for (auto &memInfo1 : memInfoList1) {
    for (auto &memInfo2 : memInfoList2) {
      if (checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2, lcmLen,
                               eventIdNum)) {
        return true;
      }
    }
  }
  return false;
}

// High-level wrapper computing pipe pairs that represent memory conflicts
// between two RW ops. Each returned tuple identifies the (src pipe, dst pipe,
// underlying SSA buffer) triple for one conflict. Multiple buffers
// conflicting at the same pipe-pair yield multiple entries so downstream
// code (buf-id mode) can keep ConflictPairs distinguishable per buffer.
llvm::SmallVector<std::tuple<CorePipeInfo, CorePipeInfo, mlir::Value>>
Solver::checkMemoryConflicts(RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  auto [it, isInserted] = checkMemoryConflictsMem.insert({{rwOp1, rwOp2}, {}});
  if (!isInserted) {
    return it->second;
  }
  auto coreSrc = rwOp1->coreType;
  auto coreDst = rwOp2->coreType;
  if (options.isCrossCoreMode()) {
    if (coreDst == pto::TCoreType::CUBE_AND_VECTOR) {
      coreDst = (coreSrc == pto::TCoreType::VECTOR) ? pto::TCoreType::CUBE
                                                     : pto::TCoreType::VECTOR;
    }
    assert(coreSrc == pto::TCoreType::VECTOR ||
           coreSrc == pto::TCoreType::CUBE);
    assert(coreDst == pto::TCoreType::VECTOR ||
           coreDst == pto::TCoreType::CUBE);
  }
  // Walk each pair of MemInfo entries and record per-buffer conflicts.
  // Conflicts found between an entry pair share the producer-side MemInfo's
  // SSA Value as the "buffer id" — both ops see the same memory through
  // this value (or its alias). When that value is null the entry is still
  // recorded with a null buffer (buffer-aware filters degrade to permissive
  // behavior, which preserves the legacy set/wait emission).
  using BufferConflict = std::tuple<CorePipeInfo, CorePipeInfo, mlir::Value>;
  llvm::SetVector<BufferConflict> collectedConflictsSet;
  auto collect = [&](const llvm::SmallVector<MemInfo> &lhs,
                     const llvm::SmallVector<MemInfo> &rhs,
                     CorePipeInfo srcInfo, CorePipeInfo dstInfo) {
    for (auto &m1 : lhs) {
      for (auto &m2 : rhs) {
        if (!checkMemInfoConflict(rwOp1, rwOp2, m1, m2)) continue;
        mlir::Value buffer = m1.value ? m1.value : m2.value;
        collectedConflictsSet.insert({srcInfo, dstInfo, buffer});
      }
    }
  };
  collect(rwOp1->readMemInfo, rwOp2->writeMemInfo,
          CorePipeInfo(coreSrc, rwOp1->pipeRead),
          CorePipeInfo(coreDst, rwOp2->pipeWrite));
  collect(rwOp1->writeMemInfo, rwOp2->readMemInfo,
          CorePipeInfo(coreSrc, rwOp1->pipeWrite),
          CorePipeInfo(coreDst, rwOp2->pipeRead));
  collect(rwOp1->writeMemInfo, rwOp2->writeMemInfo,
          CorePipeInfo(coreSrc, rwOp1->pipeWrite),
          CorePipeInfo(coreDst, rwOp2->pipeWrite));
  llvm::SmallVector<BufferConflict> collectedConflicts(
      collectedConflictsSet.begin(), collectedConflictsSet.end());
  return it->second = collectedConflicts;
}

bool Solver::checkMemoryConflictBetweenOccExclusive(
    Occurrence *occ1, Occurrence *occ2,
    std::function<bool(RWOperation *)> filter) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  for (int i = occ1->syncIrEndIndex; i < occ2->syncIrIndex; i++) {
    if (auto *otherOp = llvm::dyn_cast_if_present<RWOperation>(syncIr[i]->op)) {
      if (!filter(otherOp)) {
        continue;
      }
      if (!checkMemoryConflicts(rwOp1, otherOp).empty()) {
        return true;
      }
      if (!checkMemoryConflicts(rwOp2, otherOp).empty()) {
        return true;
      }
    }
  }
  return false;
}

std::optional<LoopLikeOpInterface>
Solver::getMultiBufferLoop(RWOperation *rwOp1, RWOperation *rwOp2,
                           const llvm::SmallVector<MemInfo> &memInfoList1,
                           const llvm::SmallVector<MemInfo> &memInfoList2) {
  std::optional<LoopLikeOpInterface> multibufferLoop;
  for (auto &memInfo1 : memInfoList1) {
    for (auto &memInfo2 : memInfoList2) {
      if (checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2)) {
        if (!memInfo1.pointerLikeInfo.has_value() ||
            !memInfo2.pointerLikeInfo.has_value()) {
          return {};
        }
        auto multibufferLoop1 = memInfo1.pointerLikeInfo->parentLoop;
        auto multibufferLoop2 = memInfo2.pointerLikeInfo->parentLoop;
        if (multibufferLoop1 == nullptr ||
            multibufferLoop1 != multibufferLoop2) {
          return {};
        }
        if (multibufferLoop.has_value() &&
            multibufferLoop.value() != multibufferLoop1) {
          return {};
        }
        multibufferLoop = multibufferLoop1;
      }
    }
  }
  return multibufferLoop;
}

std::optional<LoopLikeOpInterface>
Solver::getMultiBufferLoop(RWOperation *rwOp1, RWOperation *rwOp2) {
  std::optional<LoopLikeOpInterface> multibufferLoop;
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->readMemInfo,
                           rwOp2->writeMemInfo)) {
    auto curMultibufferLoop = getMultiBufferLoop(
        rwOp1, rwOp2, rwOp1->readMemInfo, rwOp2->writeMemInfo);
    if (multibufferLoop.has_value() &&
        multibufferLoop.value() != curMultibufferLoop) {
      return {};
    }
    multibufferLoop = curMultibufferLoop;
  }
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                           rwOp2->readMemInfo)) {
    auto curMultibufferLoop = getMultiBufferLoop(
        rwOp1, rwOp2, rwOp1->writeMemInfo, rwOp2->readMemInfo);
    if (multibufferLoop.has_value() &&
        multibufferLoop.value() != curMultibufferLoop) {
      return {};
    }
    multibufferLoop = curMultibufferLoop;
  }
  if (checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                           rwOp2->writeMemInfo)) {
    auto curMultibufferLoop = getMultiBufferLoop(
        rwOp1, rwOp2, rwOp1->writeMemInfo, rwOp2->writeMemInfo);
    if (multibufferLoop.has_value() &&
        multibufferLoop.value() != curMultibufferLoop) {
      return {};
    }
    multibufferLoop = curMultibufferLoop;
  }
  return multibufferLoop;
}

std::optional<EventIdInfo>
Solver::getMultiBufferEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                  RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);

  int64_t lcm = 1;
  int64_t minWriteSize = LONG_MAX;
  LoopLikeOpInterface multibufferLoop{nullptr};

  if (options.isTestMode()) {
    auto *parLoop1 = occ1->getParentOfType<Loop>();
    auto *parLoop2 = occ2->getParentOfType<Loop>();
    if (!parLoop1 || parLoop1 != parLoop2) {
      return {};
    }
    auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
    if (!parLoop1->isProperAncestor(setOcc) ||
        !parLoop1->isProperAncestor(waitOcc)) {
      return {};
    }
  } else {
    auto multibufferLoopOpt = getMultiBufferLoop(rwOp1, rwOp2);
    if (!multibufferLoopOpt.has_value() || !multibufferLoopOpt.value()) {
      return {};
    }
    multibufferLoop = multibufferLoopOpt.value();
    assert(multibufferLoop != nullptr);
    auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
    if (!setOcc->getParentWithOp(multibufferLoop,
                                 /*assertExists=*/false) ||
        !waitOcc->getParentWithOp(multibufferLoop,
                                  /*assertExists=*/false)) {
      return {};
    }
  }

  for (auto &memInfo1 : rwOp1->readMemInfo) {
    for (auto &memInfo2 : rwOp2->writeMemInfo) {
      if (checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2)) {
        int64_t curLcm = std::lcm(memInfo1.getSz(), memInfo2.getSz());
        lcm = std::lcm(lcm, curLcm);
        minWriteSize = std::min(minWriteSize, memInfo2.getSz());
      }
    }
  }
  for (auto &memInfo1 : rwOp1->writeMemInfo) {
    for (auto &memInfo2 : rwOp2->readMemInfo) {
      if (checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2)) {
        int64_t curLcm = std::lcm(memInfo1.getSz(), memInfo2.getSz());
        lcm = std::lcm(lcm, curLcm);
        minWriteSize = std::min(minWriteSize, memInfo1.getSz());
      }
    }
  }
  for (auto &memInfo1 : rwOp1->writeMemInfo) {
    for (auto &memInfo2 : rwOp2->writeMemInfo) {
      if (checkMemInfoConflict(rwOp1, rwOp2, memInfo1, memInfo2)) {
        int64_t curLcm = std::lcm(memInfo1.getSz(), memInfo2.getSz());
        lcm = std::lcm(lcm, curLcm);
        minWriteSize = std::min(minWriteSize, memInfo1.getSz());
        minWriteSize = std::min(minWriteSize, memInfo2.getSz());
      }
    }
  }

  // In case no write sizes were positive.
  if (minWriteSize == LONG_MAX) {
    minWriteSize = 1;
    return {};
  }

  int64_t eventIdNum = minWriteSize;
  for (; eventIdNum >= 1; eventIdNum--) {
    // llvm::dbgs() << "checking event-id-num: " << eventIdNum << '\n';
    int64_t curLcm = std::lcm(lcm, eventIdNum);
    bool okRW = !checkMemInfoConflict(rwOp1, rwOp2, rwOp1->readMemInfo,
                                      rwOp2->writeMemInfo, curLcm, eventIdNum);
    bool okWR = !checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                                      rwOp2->readMemInfo, curLcm, eventIdNum);
    bool okWW = !checkMemInfoConflict(rwOp1, rwOp2, rwOp1->writeMemInfo,
                                      rwOp2->writeMemInfo, curLcm, eventIdNum);
    if (okRW && okWR && okWW) {
      break;
    }
  }
  if (eventIdNum <= 1) {
    return {};
  }
  EventIdInfo eventIdInfo(eventIdNum);
  eventIdInfo.multibufferLoop = multibufferLoop;
  return eventIdInfo;
}

std::optional<EventIdInfo>
Solver::checkMultiBufferEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                    RWOperation *rwOp1, RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!options.isTestMode()) {
    if (!checkAllParentLoopsAreForLoops(rwOp1->op) ||
        !checkAllParentLoopsAreForLoops(rwOp2->op)) {
      return {};
    }
  }
  if (auto eventIdInfo = getMultiBufferEventIdInfo(occ1, occ2, rwOp1, rwOp2)) {
    return eventIdInfo;
  }
  return {};
}

std::optional<EventIdInfo>
Solver::checkCVMultiBufferUnrollEventIdInfo(RWOperation *rwOp1,
                                            RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!options.isCrossCoreMode()) {
    return {};
  }
  auto *parentLoop1 = rwOp1->getParentOfType<Loop>();
  auto *parentLoop2 = rwOp2->getParentOfType<Loop>();
  while (parentLoop1 != nullptr && !parentLoop1->multibufferUnrollNum) {
    parentLoop1 = parentLoop1->getParentOfType<Loop>();
  }
  while (parentLoop2 != nullptr && !parentLoop2->multibufferUnrollNum) {
    parentLoop2 = parentLoop2->getParentOfType<Loop>();
  }
  if (!parentLoop1 || !parentLoop2) {
    return {};
  }
  if (auto *parCond1 = rwOp1->getParentOfType<Condition>()) {
    if (!parCond1->isProperAncestor(rwOp2)) {
      return {};
    }
  }
  if (auto *parCond2 = rwOp2->getParentOfType<Condition>()) {
    if (!parCond2->isProperAncestor(rwOp1)) {
      return {};
    }
  }
  assert(parentLoop1->multibufferUnrollNum.value() ==
         parentLoop2->multibufferUnrollNum.value());
  EventIdInfo eventIdInfo;
  eventIdInfo.eventIdNum = parentLoop1->multibufferUnrollNum.value();
  eventIdInfo.multibufferUnrollLoop1 =
      cast<LoopLikeOpInterface>(parentLoop1->op);
  eventIdInfo.multibufferUnrollLoop2 =
      cast<LoopLikeOpInterface>(parentLoop2->op);
  return eventIdInfo;
}

std::optional<EventIdInfo>
Solver::checkCVMultiBufferPreloadEventIdInfo(RWOperation *rwOp1,
                                             RWOperation *rwOp2) {
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  if (!options.isCrossCoreMode()) {
    return {};
  }
  auto *parentScope1 = rwOp1->getParentOfType<Scope>();
  auto *parentScope2 = rwOp2->getParentOfType<Scope>();
  while (parentScope1 != nullptr && !parentScope1->maxPreloadNum.has_value()) {
    parentScope1 = parentScope1->getParentOfType<Scope>();
  }
  while (parentScope2 != nullptr && !parentScope2->maxPreloadNum.has_value()) {
    parentScope2 = parentScope2->getParentOfType<Scope>();
  }
  if (!parentScope1 || !parentScope2) {
    return {};
  }
  if (auto *parCond1 = rwOp1->getParentOfType<Condition>()) {
    if (!parCond1->isProperAncestor(rwOp2)) {
      return {};
    }
  }
  if (auto *parCond2 = rwOp2->getParentOfType<Condition>()) {
    if (!parCond2->isProperAncestor(rwOp1)) {
      return {};
    }
  }

  auto *parentLoop1 = parentScope1->getParentOfType<Loop>();
  auto *parentLoop2 = parentScope2->getParentOfType<Loop>();
  if (parentLoop1 == nullptr || parentLoop1 != parentLoop2) {
    return {};
  }

  assert(parentScope1->preloadNum.has_value());
  assert(parentScope2->preloadNum.has_value());
  assert(parentScope1->maxPreloadNum.value() ==
         parentScope2->maxPreloadNum.value());

  auto parentForLoop = llvm::dyn_cast_if_present<scf::ForOp>(parentLoop1->op);
  assert(parentForLoop != nullptr);

  EventIdInfo eventIdInfo;
  eventIdInfo.eventIdNum = parentScope1->maxPreloadNum.value();
  eventIdInfo.preloadOffset1 = parentScope1->maxPreloadNum.value() -
                               parentScope1->preloadNum.value() - 1;
  eventIdInfo.preloadOffset2 = parentScope2->maxPreloadNum.value() -
                               parentScope2->preloadNum.value() - 1;
  eventIdInfo.multibufferLoop = parentForLoop;
  return eventIdInfo;
}

// Determine required event id count and optional multibuffer loop parent for
// occurrences.
EventIdInfo Solver::getEventIdInfo(Occurrence *occ1, Occurrence *occ2,
                                   RWOperation *rwOp1, RWOperation *rwOp2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst) {
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  EventIdInfo singleEventId(1);
  if (!isBackwardSync(occ1, occ2)) {
    return singleEventId;
  }
  if (auto eventIdInfo = checkCVMultiBufferUnrollEventIdInfo(rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  if (auto eventIdInfo = checkCVMultiBufferPreloadEventIdInfo(rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  if (auto eventIdInfo =
          checkMultiBufferEventIdInfo(occ1, occ2, rwOp1, rwOp2)) {
    return eventIdInfo.value();
  }
  return singleEventId;
}

// Graph-based check to determine if adding a sync between occ1 and occ2 would
// block progress. Uses GraphSolver (Dijkstra) to estimate minimal reachable
// index.
bool Solver::checkGraphConflict(
    Occurrence *occ1, Occurrence *occ2, CorePipeInfo corePipeSrc,
    CorePipeInfo corePipeDst, EventIdInfo eventIdInfo,
    mlir::Value conflictBuffer,
    std::optional<int> startIndex, std::optional<int> endIndex,
    const llvm::SmallVector<ConflictPair *> &extraConflictPairs,
    const llvm::SmallVector<ConflictPair *> &ignoreConflictPairs) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (!startIndex.has_value()) {
    startIndex = occ1->endIndex;
  }
  if (!endIndex.has_value()) {
    endIndex = occ2->startIndex;
  }
  // In buf-id mode, transitive coverage through existing pairs is only valid
  // when those pairs operate on the SAME underlying buffer as the candidate
  // (each buf-id is an independent scoreboard counter, so a same-pipe pair
  // on a different buffer does not transitively sync the candidate's
  // counter). Filter out cross-buffer existing pairs from the graph when
  // both sides carry buffer info.
  const bool bufferAware = options.isBufIdEmit() && conflictBuffer != nullptr;
  GraphSolver graphSolver(options);
  llvm::DenseSet<ConflictPair *> visited;
  auto handleConflictPair = [&](ConflictPair *conflictPair) {
    if (conflictPair->couldNotRun) {
      return;
    }
    if (conflictPair->endIndex < startIndex.value() ||
        conflictPair->startIndex > endIndex.value()) {
      return;
    }
    if (conflictPair->isInnerBackward) {
      if ((eventIdInfo.eventIdNum * eventIdInfo.eventIdRepeatNum) <
          (conflictPair->eventIdInfo.eventIdNum *
           conflictPair->eventIdInfo.eventIdRepeatNum)) {
        return;
      }
    }
    if (llvm::find(ignoreConflictPairs, conflictPair) !=
        ignoreConflictPairs.end()) {
      return;
    }
    if (bufferAware && conflictPair->conflictBuffer != nullptr &&
        conflictPair->conflictBuffer != conflictBuffer) {
      // Different-buffer existing pair: cannot transitively cover the
      // candidate in buf-id semantics.
      return;
    }
    auto [it, isInserted] = visited.insert(conflictPair);
    if (!isInserted) {
      return;
    }
    DEBUG_WITH_TYPE("gss-sync-solver-check-graph-conflict", {
      llvm::dbgs() << "add-conflict-pair: " << conflictPair->str() << '\n';
    });
    graphSolver.addConflictPair(conflictPair);
  };

  for (auto *parOcc : occ1->getAllParents()) {
    if (scopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : scopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto *parOcc : occ2->getAllParents()) {
    if (scopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : scopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto &[scopeOccPair, chosenConflicts] : scopeOccPairChosenConflicts) {
    auto [scopeOcc1, scopeOcc2] = scopeOccPair;
    if (scopeOcc1->isProperAncestor(occ1) &&
        scopeOcc2->isProperAncestor(occ2)) {
      for (auto *conflictPair : chosenConflicts) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto *parOcc : occ1->getAllParents()) {
    if (persistentScopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : persistentScopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto *parOcc : occ2->getAllParents()) {
    if (persistentScopeOccChosenConflicts.contains(parOcc)) {
      for (auto *conflictPair : persistentScopeOccChosenConflicts[parOcc]) {
        handleConflictPair(conflictPair);
      }
    }
  }
  for (auto *conflictPair : extraConflictPairs) {
    handleConflictPair(conflictPair);
  }
  std::optional<int> mnDistance;
  if (options.enableUnitFlagFeature) {
    mnDistance = graphSolver.runDijkstraUnitFlagEnabled(
        occ1, occ2, corePipeSrc, corePipeDst, startIndex.value(),
        endIndex.value());
  } else {
    mnDistance = graphSolver.runDijkstra(corePipeSrc, corePipeDst,
                                         startIndex.value(), endIndex.value());
  }
  return !mnDistance.has_value() || mnDistance.value() > endIndex.value();
}

bool Solver::checkSyncOpsConflicts(ConflictPair *conflictPair1,
                                   ConflictPair *conflictPair2) {
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->startIndex > conflictPair2->startIndex) {
    std::swap(conflictPair1, conflictPair2);
  }
  if (conflictPair1->startIndex >= conflictPair2->startIndex ||
      conflictPair1->endIndex >= conflictPair2->endIndex) {
    return true;
  }
  bool result = false;
  if (conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo) {
    auto corePipeSrc = conflictPair1->setCorePipeInfo;
    auto corePipeDst = conflictPair2->setCorePipeInfo;
    Occurrence *occ1 = conflictPair1->setOcc;
    Occurrence *occ2 = conflictPair2->setOcc;
    auto startIndex = conflictPair1->startIndex + 1;
    auto endIndex = conflictPair2->startIndex;
    conflictPair1->startIndex += 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result ||
             checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                conflictPair1->eventIdInfo,
                                /*conflictBuffer=*/nullptr, startIndex,
                                endIndex, {conflictPair1}, {conflictPair2});
    conflictPair1->startIndex -= 1;
  }
  if (conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
    auto corePipeSrc = conflictPair1->waitCorePipeInfo;
    auto corePipeDst = conflictPair2->waitCorePipeInfo;
    Occurrence *occ1 = conflictPair1->waitOcc;
    Occurrence *occ2 = conflictPair2->waitOcc;
    auto startIndex = conflictPair1->endIndex;
    auto endIndex = conflictPair2->endIndex - 1;
    conflictPair2->endIndex -= 1;
    assert(occ1 != nullptr && occ2 != nullptr);
    result = result ||
             checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst,
                                conflictPair1->eventIdInfo,
                                /*conflictBuffer=*/nullptr, startIndex,
                                endIndex, {conflictPair1}, {conflictPair2});
    conflictPair2->endIndex += 1;
  }
  DEBUG_WITH_TYPE("gss-check-sync-ops-conflicts", {
    if (result) {
      llvm::dbgs() << "sync-ops-conflict-found: " << "\n";
      llvm::dbgs() << " " << conflictPair1->str() << '\n';
      llvm::dbgs() << " " << conflictPair2->str() << '\n';
    }
  });
  return result;
}

// Check whether two ConflictPair entries conflict in pipe and time ranges.
bool Solver::checkIntersect(ConflictPair *conflictPair1,
                            ConflictPair *conflictPair2) {
  assert(conflictPair1 != nullptr && conflictPair2 != nullptr);
  if (conflictPair1 == conflictPair2) {
    return false;
  }
  if (conflictPair1->isBarrier() || conflictPair2->isBarrier()) {
    return false;
  }
  if (conflictPair1->dontCheckForConflict ||
      conflictPair2->dontCheckForConflict) {
    return false;
  }
  if (options.isCrossCoreMode()) {
    return checkSyncOpsConflicts(conflictPair1, conflictPair2);
  }
  if (options.isBufIdEmit()) {
    auto branchUnder = [](OperationBase *op,
                          Condition *condition) -> Scope * {
      if (op == nullptr || condition == nullptr)
        return nullptr;
      auto *trueScope = condition->getTrueScope();
      if (trueScope == op || trueScope->isProperAncestor(op))
        return trueScope;
      if (condition->hasFalseScope()) {
        auto *falseScope = condition->getFalseScope();
        if (falseScope == op || falseScope->isProperAncestor(op))
          return falseScope;
      }
      return nullptr;
    };
    auto requiredBranch = [&](ConflictPair *cp,
                              Condition *condition) -> Scope * {
      Scope *required = nullptr;
      for (auto *op : {static_cast<OperationBase *>(cp->op1),
                       static_cast<OperationBase *>(cp->op2)}) {
        auto *branch = branchUnder(op, condition);
        if (branch == nullptr)
          continue;
        if (required != nullptr && required != branch)
          return nullptr;
        required = branch;
      }
      return required;
    };
    auto collectParentConditions =
        [](OperationBase *op, llvm::SmallVectorImpl<Condition *> &conditions) {
          for (auto *parent = op ? op->parentOp : nullptr; parent != nullptr;
               parent = parent->parentOp) {
            if (auto *condition = dyn_cast<Condition>(parent))
              conditions.push_back(condition);
          }
        };
    auto mutuallyExclusive = [&]() {
      llvm::SmallVector<Condition *> conditions;
      collectParentConditions(conflictPair1->op1, conditions);
      collectParentConditions(conflictPair1->op2, conditions);
      collectParentConditions(conflictPair2->op1, conditions);
      collectParentConditions(conflictPair2->op2, conditions);
      llvm::sort(conditions);
      conditions.erase(std::unique(conditions.begin(), conditions.end()),
                       conditions.end());
      for (auto *condition : conditions) {
        auto *branch1 = requiredBranch(conflictPair1, condition);
        auto *branch2 = requiredBranch(conflictPair2, condition);
        if (branch1 != nullptr && branch2 != nullptr && branch1 != branch2)
          return true;
      }
      return false;
    };
    if (mutuallyExclusive())
      return false;

    // Two buf-id brackets must use different ids whenever they share any
    // pipe — same-id same-pipe re-entry is illegal per the buf-id spec
    // constraint 1 ("两次连续 get(P,#id) 非法"). Unlike set-wait, having
    // different dst pipes (or different src pipes) is NOT enough to share
    // an id: e.g. (MTE2 -> V, id 0) and (V -> MTE3, id 0) both bracket the
    // common V anchor with id 0, producing back-to-back `get_buf(V, #0)`
    // before the V op, which the spec forbids. The range-based set-wait
    // overlap check doesn't apply because buf-id brackets extend strictly
    // around the anchor RWOperation (a single index), not across the
    // [setOp.endIndex, waitOp.startIndex] gap.
    auto pipes1 = std::array{conflictPair1->setCorePipeInfo.pipe,
                             conflictPair1->waitCorePipeInfo.pipe};
    auto pipes2 = std::array{conflictPair2->setCorePipeInfo.pipe,
                             conflictPair2->waitCorePipeInfo.pipe};
    for (auto p1 : pipes1) {
      for (auto p2 : pipes2) {
        if (p1 == p2) {
          return true;
        }
      }
    }
    return false;
  }
  if (conflictPair1->setCorePipeInfo != conflictPair2->setCorePipeInfo ||
      conflictPair1->waitCorePipeInfo != conflictPair2->waitCorePipeInfo) {
    return false;
  }
  for (auto [l1, r1] : getRanges(conflictPair1)) {
    for (auto [l2, r2] : getRanges(conflictPair2)) {
      if (checkRangesIntersect(l1, r1 + 1, l2, r2 + 1)) {
        return true;
      }
    }
  }
  return false;
}

// Obtain available event ids while accounting for already chosen conflicts.
std::vector<ConflictPair *>
Solver::getIntersectingConflictPairs(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  if (conflictPair->isBarrier()) {
    return {};
  }
  if (conflictPair->dontCheckForConflict) {
    return {};
  }
  std::vector<ConflictPair *> intersectingConflictPairs;
  for (auto &curConflictPair : chosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  for (auto &curConflictPair : persistentChosenConflictedPairs) {
    if (checkIntersect(conflictPair, curConflictPair.get())) {
      intersectingConflictPairs.push_back(curConflictPair.get());
    }
  }
  return intersectingConflictPairs;
}

// Processed-pair tracking helpers.
bool Solver::checkVisited(Occurrence *occ1, Occurrence *occ2) {
  auto [it, isInserted] = processedOccPairs.insert(std::make_pair(occ1, occ2));
  return !isInserted;
}

bool Solver::checkSkippable(bool reverseOrder, Occurrence *occ) {
  return skipOcc[reverseOrder].contains(occ);
}

// Synced-pair memoization helpers.
EventIdNode *Solver::getOldEventIdNodeIfExists(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  auto oldConflictPairs = getMemorizedSyncedPairs(conflictPair);
  if (oldConflictPairs.empty()) {
    return {};
  }
  ConflictPair *oldConflictPair = *oldConflictPairs.begin();
  assert(oldConflictPair != nullptr && oldConflictPair->eventIdNode != nullptr);
  return oldConflictPair->eventIdNode;
}

llvm::DenseSet<ConflictPair *>
Solver::getMemorizedSyncedPairs(ConflictPair *conflictPair) {
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  return syncedPairs[key];
}

void Solver::memorizeSyncedPair(ConflictPair *conflictPair) {
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  syncedPairs[key].insert(conflictPair);
#ifndef NDEBUG
  for (auto *oldConflictPair : syncedPairs[key]) {
    assert(oldConflictPair->eventIdNode == conflictPair->eventIdNode);
  }
#endif
}

void Solver::forgetSyncedPair(ConflictPair *conflictPair) {
  assert(conflictPair != nullptr);
  auto key = std::make_tuple(
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo);
  syncedPairs[key].erase(conflictPair);
}

void Solver::memorizeReusedSyncedPair(ConflictPair *conflictPair,
                                      ConflictPair *reusedConflictPair) {
  assert(conflictPair != nullptr);
  replacedWithReusableSyncedPairs[{
      conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
      conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo}] =
      reusedConflictPair;
}

bool Solver::skipMMad1DecomposedLoopOpt(Occurrence *occ1, Occurrence *occ2) {
  auto *parentLoopOp1 = OperationBase::getParentloop(occ1->op);
  auto *parentLoopOp2 = OperationBase::getParentloop(occ2->op);
  if (parentLoopOp1 != nullptr && parentLoopOp2 != nullptr) {
    if (parentLoopOp1 != parentLoopOp2) {
      if (isa<MmadL1LoopOp>(parentLoopOp1) &&
          isa<MmadL1LoopOp>(parentLoopOp2)) {
        return true;
      }
    }
  }
  return false;
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::checkAndApplyMmadl0LoopOpt(ConflictPair *conflictPair, Occurrence *occ1,
                                   Occurrence *occ2, Occurrence *parOcc1,
                                   Occurrence *parOcc2) {
  if (!options.decomposeMmadl1Op) {
    return {};
  }
  if (occ1->parentOcc != nullptr && occ1->parentOcc->parentOcc != nullptr &&
      occ1->parentOcc->parentOcc->parentOcc == parOcc1 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ1->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(
          occ1->parentOcc->parentOcc->op)) {
    conflictPair->setOnLastIterOnly = true;
    return std::make_pair(occ1, parOcc2);
  }
  if (!conflictPair->isInnerBackward && occ2->parentOcc != nullptr &&
      occ2->parentOcc->parentOcc != nullptr &&
      occ2->parentOcc->parentOcc->parentOcc == parOcc2 &&
      llvm::isa_and_present<syncsolver::LoadL0AOp, syncsolver::LoadL0BOp>(
          occ2->op) &&
      llvm::isa_and_present<syncsolver::MmadL1LoopOp>(
          occ2->parentOcc->parentOcc->op)) {
    conflictPair->waitOnFirstIterOnly = true;
    return std::make_pair(parOcc1, occ2);
  }
  return {};
}

std::optional<UnitFlagInfo> Solver::checkUnitFlagPatterns(Occurrence *occ1,
                                                          Occurrence *occ2) {
  return {};
}

Occurrence *Solver::getBeforePlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrIndex - 1;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->beforeOp == occ->op);
#endif
  return placeHolderOcc;
}

Occurrence *Solver::getAfterPlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrEndIndex;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->afterOp == occ->op);
#endif
  return placeHolderOcc;
}

Occurrence *Solver::getScopeBeginPlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrIndex + 1;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->scopeBegin == occ->op);
#endif
  return placeHolderOcc;
}

Occurrence *Solver::getScopeEndPlaceHolderOcc(Occurrence *occ) {
  assert(occ != nullptr);
  assert(llvm::isa_and_present<Scope>(occ->op));
  int index = occ->syncIrEndIndex - 1;
  assert(0 <= index && index < static_cast<int>(syncIr.size()));
  auto *placeHolderOcc = syncIr[index].get();
#ifndef NDEBUG
  auto *placeHolderOp = llvm::dyn_cast<PlaceHolder>(placeHolderOcc->op);
  assert(placeHolderOp != nullptr);
  assert(placeHolderOp->scopeEnd == occ->op);
#endif
  return placeHolderOcc;
}

std::pair<Occurrence *, Occurrence *>
Solver::getSetWaitLCAPairOcc(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);

  auto [grandParOcc1, grandParOcc2] = Occurrence::getLCAPair(occ1, occ2);
  assert(grandParOcc1 != nullptr && grandParOcc2 != nullptr);
  assert(grandParOcc1->parentOcc != nullptr &&
         grandParOcc2->parentOcc != nullptr);

  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  assert(parOp1 != nullptr && parOp2 != nullptr);
  assert(parOp1->parentOp != nullptr && parOp2->parentOp != nullptr);
  assert(parOp1->parentOp == parOp2->parentOp);

  auto *parOcc1 = occ1->getParentWithOp(parOp1->parentOp);
  auto *parOcc2 = occ2->getParentWithOp(parOp2->parentOp);
  assert(parOcc1 != nullptr && parOcc2 != nullptr);
  assert(parOcc1 != occ1 && parOcc2 != occ2);

  auto *setOcc = occ1->getNthParent(occ1->depth - parOcc1->depth - 1);
  auto *waitOcc = occ2->getNthParent(occ2->depth - parOcc2->depth - 1);
  assert(setOcc != nullptr && waitOcc != nullptr);
  assert(parOcc1->isProperAncestor(setOcc));
  assert(parOcc2->isProperAncestor(waitOcc));

  auto *parLoop = Occurrence::getParentloop(setOcc);
  while (parLoop != nullptr && grandParOcc1->isProperAncestor(parLoop)) {
    setOcc = parLoop;
    waitOcc = Occurrence::getParentloop(waitOcc);
    parLoop = Occurrence::getParentloop(setOcc);
  }
  return std::make_pair(setOcc, waitOcc);
}

std::pair<Occurrence *, Occurrence *>
Solver::getFixedSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  // - get setOcc waitOcc where:
  // setOcc->op->parent = waitOcc->op->parent = lca(occ1, occ2)->op
  auto [setOcc, waitOcc] = getSetWaitLCAPairOcc(occ1, occ2);

  // - check if it's the case of while loop:
  // while{
  //  before{
  //    occ1
  //  }
  //  setOcc;
  //  waitOcc;
  //  after{
  //    occ2
  //  }
  // }
  // - and fix it to be:
  // while{
  //  before{
  //    occ1
  //    setOcc;
  //    ...
  //    waitOcc;
  //    placeHolder
  //  }
  //  after{
  //    occ2
  //  }
  // }
  if (setOcc->op != waitOcc->op) {
    if (auto *parLoopOp =
            llvm::dyn_cast_if_present<Loop>(setOcc->parentOcc->op)) {
      if (parLoopOp->body.size() > 1 && !isa<PlaceHolder>(waitOcc->op)) {
        auto *placeHolderOcc = getScopeEndPlaceHolderOcc(setOcc);
        std::tie(setOcc, waitOcc) = getSetWaitLCAPairOcc(occ1, placeHolderOcc);
      }
    }
  }

  // - check if it's the case of:
  // loop(iter-1){
  //   condition{
  //     true-scope{}
  //     setOcc()
  //     false-scope{}
  //   }
  // }
  // loop(iter-2){
  //   condition{
  //     true-scope{}
  //     waitOcc()
  //     false-scope{}
  //   }
  // }
  // - and fix it to be:
  // loop(iter-1){
  //   condition{
  //     true-scope{}
  //     false-scope{}
  //   }
  //   setOcc()
  // }
  // loop(iter-2){
  //   waitOcc()
  //   condition{
  //     true-scope{}
  //     false-scope{}
  //   }
  // }
  if (isBackwardSync(occ1, occ2)) {
    if (setOcc->parentOcc != nullptr) {
      if (llvm::isa_and_present<Condition>(setOcc->parentOcc->op)) {
        setOcc = setOcc->parentOcc;
      }
    }
    if (waitOcc->parentOcc != nullptr) {
      if (llvm::isa_and_present<Condition>(waitOcc->parentOcc->op)) {
        waitOcc = waitOcc->parentOcc;
      }
    }
  }

  // - for the case of cv-pipelining:
  // loop(){
  //   op1
  // } {unroll=x}
  // setOcc
  // waitOcc
  // loop(){
  //   op2
  // } {unroll=x}
  // - and fix it to be:
  // loop(){
  //   op1
  //   setOcc
  // } {unroll=x}
  // loop(){
  //   waitOcc
  //   op2
  // } {unroll=x}
  if (options.isCrossCoreMode()) {
    assert(setOcc->op != nullptr && waitOcc->op != nullptr);
    auto *forOp1 = llvm::dyn_cast_if_present<Loop>(setOcc->op);
    auto *forOp2 = llvm::dyn_cast_if_present<Loop>(waitOcc->op);
    if (forOp1 != nullptr && forOp2 != nullptr) {
      if (forOp1->multibufferUnrollNum && forOp2->multibufferUnrollNum) {
        assert(forOp1->multibufferUnrollNum == forOp2->multibufferUnrollNum);
        setOcc = occ1->getNthParent(occ1->depth - setOcc->depth - 2);
        waitOcc = occ2->getNthParent(occ2->depth - waitOcc->depth - 2);
      }
    }
  }

  // - for the case of cv-pipelining:
  // scope(){
  //   op1
  // } {preload=x}
  // setOcc
  // waitOcc
  // scope(){
  //   op2
  // } {preload=x}
  // - and fix it to be:
  // scope(){
  //   op1
  //   setOcc
  // } {preload=x}
  // scope(){
  //   waitOcc
  //   op2
  // } {preload=x}
  if (options.isCrossCoreMode()) {
    assert(setOcc->op != nullptr && waitOcc->op != nullptr);
    auto *scopeOp1 = llvm::dyn_cast_if_present<Scope>(setOcc->op);
    auto *scopeOp2 = llvm::dyn_cast_if_present<Scope>(waitOcc->op);
    if (scopeOp1 != nullptr && scopeOp2 != nullptr) {
      if (scopeOp1->maxPreloadNum && scopeOp2->maxPreloadNum) {
        assert(scopeOp1->maxPreloadNum == scopeOp2->maxPreloadNum);
        setOcc = occ1->getNthParent(occ1->depth - setOcc->depth - 2);
        waitOcc = occ2->getNthParent(occ2->depth - waitOcc->depth - 2);
      }
    }
  }

  // - check if it's the case of:
  // {
  //   op1
  //   setOcc
  //   ...
  //   waitOcc
  //   loop(){}
  //   setOcc
  //   ...
  //   waitOcc
  //   op2
  // }
  // - and fix it to be:
  // {
  //   op1
  //   setOcc
  //   ...
  //   waitOcc
  //   placeHolder
  //   loop(){}
  //   placeHolder
  //   setOcc
  //   ...
  //   waitOcc
  //   op2
  // }
  if (llvm::isa_and_present<Loop>(setOcc->op)) {
    setOcc = getAfterPlaceHolderOcc(setOcc);
  }
  if (llvm::isa_and_present<Loop>(waitOcc->op)) {
    waitOcc = getBeforePlaceHolderOcc(waitOcc);
  }

  return std::make_pair(setOcc, waitOcc);
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::getFunctionBlockSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *parFunctionBlock1 = occ1->getParentOfType<FunctionBlock>();
  auto *parFunctionBlock2 = occ2->getParentOfType<FunctionBlock>();
  if (parFunctionBlock1 == parFunctionBlock2) {
    return {};
  }
  auto *placeHolderOcc = getScopeBeginPlaceHolderOcc(parFunctionBlock2);
  return std::make_pair(placeHolderOcc, occ2);
}

std::optional<std::pair<Occurrence *, Occurrence *>>
Solver::getUnlikelyCondSetWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  assert(occ1 != nullptr && occ2 != nullptr);
  if (options.isCrossCoreMode() && isBackwardSync(occ1, occ2)) {
    return {};
  }
  if (auto *unlikelyParCondOcc1 =
          Occurrence::getUnlikelyParentCondition(occ1)) {
    if (!unlikelyParCondOcc1->isProperAncestor(occ2)) {
      auto *parentLoopOcc = Occurrence::getParentloop(unlikelyParCondOcc1);
      if (parentLoopOcc == nullptr || parentLoopOcc->isProperAncestor(occ2)) {
        auto *placeHolderOcc = getScopeEndPlaceHolderOcc(
            occ1->getNthParent(occ1->depth - unlikelyParCondOcc1->depth - 1));
        return std::make_pair(occ1, placeHolderOcc);
      }
    }
  }
  if (auto *unlikelyParCondOcc2 =
          Occurrence::getUnlikelyParentCondition(occ2)) {
    if (!unlikelyParCondOcc2->isProperAncestor(occ1)) {
      auto *parentLoopOcc = Occurrence::getParentloop(unlikelyParCondOcc2);
      if (parentLoopOcc == nullptr || parentLoopOcc->isProperAncestor(occ1)) {
        auto *placeHolderOcc = getScopeBeginPlaceHolderOcc(
            occ2->getNthParent(occ2->depth - unlikelyParCondOcc2->depth - 1));
        return std::make_pair(placeHolderOcc, occ2);
      }
    }
  }
  return {};
}

std::pair<Occurrence *, Occurrence *> Solver::getSetWaitOcc(Occurrence *occ1,
                                                            Occurrence *occ2) {
  if (auto functionBlockOpt = getFunctionBlockSetWaitOcc(occ1, occ2)) {
    std::tie(occ1, occ2) = functionBlockOpt.value();
  }
  if (auto unlikelyOpt = getUnlikelyCondSetWaitOcc(occ1, occ2)) {
    std::tie(occ1, occ2) = unlikelyOpt.value();
  }
  return getFixedSetWaitOcc(occ1, occ2);
}

Occurrence *Solver::getBarrierWaitOcc(Occurrence *occ1, Occurrence *occ2) {
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);
  if (!waitOcc->isProperAncestor(occ2)) {
    return waitOcc;
  }
  auto allParents = occ2->getAllParents();
  while (!allParents.empty() && allParents.back()->isProperAncestor(waitOcc)) {
    allParents.pop_back();
  }
  while (allParents.size() >= 2 &&
         llvm::isa_and_present<Condition>(allParents.back()->op)) {
    allParents.pop_back();
    assert(llvm::isa_and_present<Scope>(allParents.back()->op));
    allParents.pop_back();
  }
  waitOcc = !allParents.empty() ? allParents.back() : occ2;
  return waitOcc;
}

void Solver::insertBarrierAllBeforeOcc(Occurrence *occ, bool isUseless,
                                       bool isPersistent) {
  assert(occ != nullptr);
  auto *rwOp = llvm::dyn_cast_if_present<RWOperation>(occ->op);
  assert(rwOp != nullptr);
  auto conflictPair = std::make_unique<ConflictPair>(
      nullptr, nullptr, rwOp, rwOp, occ, occ,
      CorePipeInfo(pto::TCoreType::CUBE_OR_VECTOR, pto::PIPE::PIPE_ALL),
      CorePipeInfo(pto::TCoreType::CUBE_OR_VECTOR, pto::PIPE::PIPE_ALL),
      occ->startIndex, occ->startIndex);
  conflictPair->isUseless = isUseless;
  auto *normScopeOcc = occ->parentOcc;
  assert(normScopeOcc != nullptr);
  LLVM_DEBUG(llvm::dbgs() << (isPersistent ? "is-persistent " : "")
                          << occ->op->str(0, false) << ' '
                          << conflictPair->str() << '\n';);
  if (isPersistent) {
    persistentScopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
    persistentChosenConflictedPairs.push_back(std::move(conflictPair));
  } else {
    insertedBarrierAllBefore[occ->op].insert({occ, isUseless});
    scopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
    chosenConflictedPairs.push_back(std::move(conflictPair));
  }
}

void Solver::insertBarrierAllBeforeOp(OperationBase *op, bool isUseless,
                                      bool isPersistent) {
  assert(op != nullptr);
  for (auto *occ : opAllOccurrences[op]) {
    insertBarrierAllBeforeOcc(occ, isUseless, isPersistent);
    isUseless = true;
  }
}

// When barrier-all markers need to be chosen, insert them before all
// occurrences for the chosen op.
void Solver::pickAndInsertABarrierAll() {
  assert(!insertedBarrierAllBefore.empty());
  OperationBase *chosenOp = nullptr;
  for (auto &[op, vec] : insertedBarrierAllBefore) {
    if (vec.empty()) {
      continue;
    }
    if (chosenOp == nullptr || chosenOp->id > op->id) {
      chosenOp = op;
    }
  }
  assert(chosenOp != nullptr);
  insertBarrierAllBeforeOp(chosenOp, /*isUseless=*/false,
                           /*isPersistent=*/true);
}

bool Solver::isBackwardSync(Occurrence *occ1, Occurrence *occ2) {
  if (occ1->op->id >= occ2->op->id) {
    return true;
  }
  assert(occ1 != nullptr && occ2 != nullptr);
  assert(occ1->op != nullptr && occ2->op != nullptr);
  auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
  auto [parOp1, parOp2] = OperationBase::getLCAPair(occ1->op, occ2->op);
  return parOcc1->parentOcc->op != parOp1->parentOp;
}

bool Solver::reuseCmp(ConflictPair *conflictPair1,
                      ConflictPair *conflictPair2) {
  assert(conflictPair1 != nullptr && conflictPair2 != nullptr);
  assert(conflictPair1->op1 != nullptr && conflictPair1->op2 != nullptr);
  assert(conflictPair2->op1 != nullptr && conflictPair2->op2 != nullptr);
  if (conflictPair1->startIndex != conflictPair2->startIndex) {
    return conflictPair1->startIndex < conflictPair2->startIndex;
  }
  if (conflictPair1->endIndex != conflictPair2->endIndex) {
    return conflictPair1->endIndex > conflictPair2->endIndex;
  }
  if (conflictPair1->op1 != conflictPair2->op1) {
    return conflictPair1->op1->id > conflictPair2->op1->id;
  }
  if (conflictPair1->op2 != conflictPair2->op2) {
    return conflictPair1->op2->id > conflictPair2->op2->id;
  }
  return false;
}

ConflictPair *Solver::getReusableConflictPair(
    ConflictPair *conflictPair,
    const llvm::DenseSet<ConflictPair *> &conflictPairsSet) {
  assert(conflictPair != nullptr);
  ConflictPair *ret = nullptr;
  for (auto *curConflictPair : conflictPairsSet) {
    if (curConflictPair->isBarrier() || curConflictPair->dontReuse) {
      continue;
    }
    if (curConflictPair->op1 != conflictPair->op1 ||
        curConflictPair->op2 != conflictPair->op2 ||
        curConflictPair->setCorePipeInfo != conflictPair->setCorePipeInfo ||
        curConflictPair->waitCorePipeInfo != conflictPair->waitCorePipeInfo) {
      continue;
    }
    if (!checkIntersect(conflictPair, curConflictPair)) {
      continue;
    }
    if (curConflictPair->startIndex >= conflictPair->startIndex) {
      continue;
    }
    if (conflictPair->eventIdNode->eventIdNum <
        curConflictPair->eventIdNode->eventIdNum) {
      continue;
    }
    assert(conflictPair->eventIdNode != nullptr);
    assert(curConflictPair->eventIdNode != nullptr);
    if (conflictPair->eventIdNode->eventIdNum >
        curConflictPair->eventIdNode->eventIdNum) {
      if (conflictPair->eventIdNode->eventIdNum %
          curConflictPair->eventIdNode->eventIdNum) {
        continue;
      }
    }
    assert(conflictPair->startIndex <= curConflictPair->endIndex);
    assert(curConflictPair->endIndex <= conflictPair->endIndex);
    if (ret == nullptr || reuseCmp(ret, curConflictPair)) {
      ret = curConflictPair;
    }
  }
  return ret;
}

bool Solver::reuseConflictPair(ConflictPair *conflictPair,
                               Occurrence *scopeOcc1, Occurrence *scopeOcc2) {
  if (conflictPair->isBarrier()) {
    return false;
  }
  if (scopeOcc1->op != scopeOcc2->op) {
    return false;
  }
  if (!barrierAllPairs.empty()) {
    return false;
  }

  ConflictPair *oldReusedConflictPair = nullptr;
  if (conflictPair->isUseless) {
    auto it = replacedWithReusableSyncedPairs.find(
        {conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
         conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo});
    if (it != replacedWithReusableSyncedPairs.end()) {
      oldReusedConflictPair = it->second;
    }
  }

#ifndef NDEBUG
  if (!conflictPair->isUseless) {
    auto it = replacedWithReusableSyncedPairs.find(
        {conflictPair->backwardSyncLoopOp, conflictPair->op1, conflictPair->op2,
         conflictPair->setCorePipeInfo, conflictPair->waitCorePipeInfo});
    assert(it == replacedWithReusableSyncedPairs.end());
  }
#endif

  if (conflictPair->isUseless && oldReusedConflictPair == nullptr) {
    return false;
  }

  auto corePipeSrc = conflictPair->setCorePipeInfo;
  auto corePipeDst = conflictPair->waitCorePipeInfo;

  if (oldReusedConflictPair == nullptr) {
    if (!reusePairs.contains({corePipeSrc, corePipeDst}) ||
        reusePairs[{corePipeSrc, corePipeDst}] <=
            reusedPairs[{corePipeSrc, corePipeDst}]) {
      return false;
    }
  }

  assert(reusePairs.contains(std::make_tuple(corePipeSrc, corePipeDst)));
  assert(reusePairs[std::make_tuple(corePipeSrc, corePipeDst)] >=
         reusedPairs[std::make_tuple(corePipeSrc, corePipeDst)]);

  ConflictPair *opt1 = nullptr;
  ConflictPair *opt2 = nullptr;
  ConflictPair *opt3 = nullptr;
  ConflictPair *opt4 = nullptr;
  ConflictPair *opt5 = nullptr;

  auto it1 = scopeOccChosenConflicts.find(scopeOcc1);
  auto it2 = scopeOccChosenConflicts.find(scopeOcc2);
  auto it3 = scopeOccPairChosenConflicts.find({scopeOcc1, scopeOcc2});
  auto it4 = persistentScopeOccChosenConflicts.find(scopeOcc1);
  auto it5 = persistentScopeOccChosenConflicts.find(scopeOcc2);

  if (it1 != scopeOccChosenConflicts.end()) {
    opt1 = getReusableConflictPair(conflictPair, it1->second);
  }
  if (it2 != scopeOccChosenConflicts.end()) {
    opt2 = getReusableConflictPair(conflictPair, it2->second);
  }
  if (it3 != scopeOccPairChosenConflicts.end()) {
    opt3 = getReusableConflictPair(conflictPair, it3->second);
  }
  if (it4 != persistentScopeOccChosenConflicts.end()) {
    opt4 = getReusableConflictPair(conflictPair, it4->second);
  }
  if (it5 != persistentScopeOccChosenConflicts.end()) {
    opt5 = getReusableConflictPair(conflictPair, it5->second);
  }

  ConflictPair *reusableConflictPair = nullptr;
  for (auto *opt : {opt1, opt2, opt3, opt4, opt5}) {
    if (opt != nullptr) {
      if (reusableConflictPair == nullptr ||
          reuseCmp(reusableConflictPair, opt)) {
        reusableConflictPair = opt;
      }
    }
  }

  if (reusableConflictPair == nullptr) {
    return false;
  }

  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    llvm::dbgs() << "reuse: " << conflictPair->str() << '\n';
    llvm::dbgs() << "with: " << reusableConflictPair->str() << '\n';
  });

  assert(reusableConflictPair->startIndex < conflictPair->startIndex);
  assert(reusableConflictPair->endIndex <= conflictPair->endIndex);
  reusableConflictPair->setOp = conflictPair->setOp;
  reusableConflictPair->setOcc = conflictPair->setOcc;
  reusableConflictPair->startIndex = conflictPair->startIndex;

  if (!conflictPair->isUseless) {
    memorizeReusedSyncedPair(conflictPair, reusableConflictPair);
  }

  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    if (oldReusedConflictPair != nullptr) {
      llvm::dbgs() << "old-reuse: " << oldReusedConflictPair->str() << '\n';
    }
  });

  if (oldReusedConflictPair != nullptr) {
    assert(oldReusedConflictPair->op1 == reusableConflictPair->op1);
    assert(oldReusedConflictPair->op2 == reusableConflictPair->op2);
    assert(oldReusedConflictPair->waitOp == reusableConflictPair->waitOp);
  }

  if (!conflictPair->isUseless) {
    reusedPairs[{corePipeSrc, corePipeDst}] += 1;
  }

  return true;
}

std::unique_ptr<EventIdSolver> &
Solver::getEventIdSolverRef(pto::PIPE pipeSrc, pto::PIPE pipeDst) {
  if (options.isCrossCoreMode() || options.isBufIdEmit()) {
    // Cross-core mode shares one event-id pool across all pipe pairs.
    // Buf-id mode does the same: get_buf/rls_buf don't carry the "other"
    // pipe in the opcode, so the same numeric id on two different pipe-
    // pairs would alias to the same hw scoreboard and the resulting
    // bracketing pattern would violate constraint 1 of the buf-id spec
    // (two consecutive get(pipe, #id) before a single op is illegal).
    pipeSrc = pto::PIPE::PIPE_UNASSIGNED;
    pipeDst = pto::PIPE::PIPE_UNASSIGNED;
  }
  auto key = std::make_tuple(pipeSrc, pipeDst);
  if (!eventIdSolver.contains(key)) {
    int64_t eventIdNumMax =
        getHWAvailableEventIdNum(options.syncMode, pipeSrc, pipeDst);
    if (options.eventIdNumMax.has_value()) {
      eventIdNumMax = std::min(eventIdNumMax, options.eventIdNumMax.value());
      eventIdNumMax = std::max<int64_t>(eventIdNumMax, 1);
    }
    eventIdSolver[key] = std::make_unique<EventIdSolver>(eventIdNumMax);
  }
  return eventIdSolver[key];
}

bool Solver::checkReuseMultiBufferFlagId(ConflictPair *conflictPair) {
  if (options.useDifferentMultiBufferFlagIds) {
    return false;
  }
  if (!conflictPair->isInnerBackward ||
      conflictPair->eventIdInfo.eventIdNum <= 1 ||
      conflictPair->movedToOuterLoop) {
    return false;
  }
  auto [setOcc, waitOcc] =
      std::tie(conflictPair->setOcc, conflictPair->waitOcc);
  auto *backwardSyncLoopOcc = conflictPair->backwardSyncLoopOcc;
  assert(backwardSyncLoopOcc != nullptr);
  if (auto *parCondOcc1 = setOcc->getParentOfType<Condition>()) {
    if (!parCondOcc1->isProperAncestor(backwardSyncLoopOcc)) {
      return false;
    }
  }
  if (auto *parCondOcc2 = waitOcc->getParentOfType<Condition>()) {
    if (!parCondOcc2->isProperAncestor(backwardSyncLoopOcc)) {
      return false;
    }
  }
  return true;
}

void Solver::handleSetWaitConflict(Occurrence *occ1, Occurrence *occ2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst,
                                   EventIdInfo eventIdInfo, bool isUseless,
                                   mlir::Value conflictBuffer) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(corePipeSrc != corePipeDst);

  Loop *parentLCALoopOp{nullptr};
  Occurrence *parentLCALoopOcc{nullptr};
  Occurrence *parentLCALoopBeforePHOcc{nullptr};
  Occurrence *parentLCALoopAfterPHOcc{nullptr};
  auto [setOcc, waitOcc] = getSetWaitOcc(occ1, occ2);

  auto [lcaSetOp, lcaWaitOp] =
      OperationBase::getLCAPair(setOcc->op, waitOcc->op);
  auto *normScopeOcc1 = setOcc->getParentWithOp(lcaSetOp->parentOp);
  auto *normScopeOcc2 = waitOcc->getParentWithOp(lcaWaitOp->parentOp);
  assert(normScopeOcc1->op == normScopeOcc2->op);
  auto *normScopeOp = normScopeOcc1->op;
  assert(normScopeOp != nullptr);
  assert(normScopeOp->parentOp != nullptr);

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setOcc, waitOcc, corePipeSrc,
      corePipeDst, setOcc->endIndex, waitOcc->startIndex);
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  conflictPair->isUseless = isUseless;
  conflictPair->isInnerBackward = isBackwardSync(setOcc, waitOcc);
  conflictPair->eventIdInfo = eventIdInfo;
  conflictPair->conflictBuffer = conflictBuffer;

  if (conflictPair->isInnerBackward) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    assert(parOcc1 != nullptr && parOcc2 != nullptr);

    parentLCALoopOcc = parOcc1->getParentOfType<Loop>();
    if (moveBackwardSyncPairsToOutmostLoop) {
      while (auto *grandParentLoopOcc =
                 parentLCALoopOcc->getParentOfType<Loop>()) {
        conflictPair->movedToOuterLoop = true;
        parentLCALoopOcc = grandParentLoopOcc;
      }
    }
    assert(parentLCALoopOcc != nullptr);
    conflictPair->backwardSyncLoopOcc = parentLCALoopOcc;

    parentLCALoopOp = llvm::dyn_cast<Loop>(parentLCALoopOcc->op);
    assert(parentLCALoopOp != nullptr);
    conflictPair->backwardSyncLoopOp = parentLCALoopOp;

    parentLCALoopBeforePHOcc = getBeforePlaceHolderOcc(parentLCALoopOcc);
    assert(parentLCALoopBeforePHOcc != nullptr);
    parentLCALoopAfterPHOcc = getAfterPlaceHolderOcc(parentLCALoopOcc);
    assert(parentLCALoopAfterPHOcc != nullptr);
  }

  if (auto setWaitOccs = checkAndApplyMmadl0LoopOpt(conflictPair.get(), occ1,
                                                    occ2, setOcc, waitOcc)) {
    std::tie(setOcc, waitOcc) = setWaitOccs.value();
    conflictPair->updateSetWaitOccs(setOcc, waitOcc);
  }

  if (!conflictPair->isInnerBackward ||
      disabledMultiEventIdPairs.contains({corePipeSrc, corePipeDst})) {
    conflictPair->eventIdInfo = EventIdInfo(1);
  }
  if (checkReuseMultiBufferFlagId(conflictPair.get())) {
    conflictPair->eventIdInfo.eventIdRepeatNum =
        conflictPair->eventIdInfo.eventIdNum;
    conflictPair->eventIdInfo.eventIdNum = 1;
  }

  auto &curEventIdSolver = getEventIdSolverRef(
      conflictPair->setCorePipeInfo.pipe, conflictPair->waitCorePipeInfo.pipe);
  curEventIdSolver->pushActionNone();

  auto checkColorable = [&]() -> bool {
    if (curEventIdSolver->isColorable()) {
      return true;
    }
    LLVM_DEBUG(llvm::dbgs() << "will-be-converted-to-barrier-all "
                            << conflictPair->str() << '\n';);
    insertBarrierAllBeforeOp(occ2->op, conflictPair->isUseless,
                             /*isPersistent=*/false);
    barrierAllPairs.insert({corePipeSrc, corePipeDst});
    curEventIdSolver->undoActions();
    return false;
  };

  if (auto *oldEventIdNode = getOldEventIdNodeIfExists(conflictPair.get())) {
    conflictPair->eventIdNode = oldEventIdNode;
    curEventIdSolver->insertConflictPair(oldEventIdNode, conflictPair.get());
  } else {
    bool reversedPriority = false;
    if (conflictPair->isInnerBackward) {
      if (OperationBase::getParentloop(occ1->op) == normScopeOp->parentOp &&
          OperationBase::getParentloop(occ2->op) == normScopeOp->parentOp) {
        reversedPriority = true;
      }
    }
    conflictPair->eventIdNode = curEventIdSolver->createNode(
        conflictPair.get(), conflictPair->eventIdInfo.eventIdNum,
        reversedPriority);
  }

  if (options.reuseSyncPairToSaveEventIds) {
    if (reuseConflictPair(conflictPair.get(), normScopeOcc1, normScopeOcc2)) {
      curEventIdSolver->undoActions();
      return;
    }
  }

  auto intersectingConflictPairs =
      getIntersectingConflictPairs(conflictPair.get());
  curEventIdSolver->addConflicts(conflictPair.get(), intersectingConflictPairs);
  if (!checkColorable()) {
    return;
  }

  LLVM_DEBUG({
    llvm::dbgs() << conflictPair->str() << '\n';
    if (parentLCALoopOcc != nullptr) {
      llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
    }
  });

  llvm::SmallVector<std::pair<std::unique_ptr<ConflictPair>, Occurrence *>>
      extraConflictPairs;

  auto insertExtraConflictPair = [&](Occurrence *setOcc, Occurrence *waitOcc,
                                     Occurrence *parentScope,
                                     bool couldNotRun = false) -> bool {
    assert(setOcc != nullptr && waitOcc != nullptr && parentScope != nullptr);
    auto extraConflictPair = conflictPair->clone(setOcc, waitOcc);
    extraConflictPair->isUseless = true;
    extraConflictPair->dontReuse = true;
    if (couldNotRun || options.moveOutAndMergeBackwardSyncPairs) {
      extraConflictPair->couldNotRun = true;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "extra-conflict-pair: " << extraConflictPair->str()
                   << "\n";
    });
    curEventIdSolver->insertConflictPair(conflictPair->eventIdNode,
                                         extraConflictPair.get());
    auto intersectingConflictPairs =
        getIntersectingConflictPairs(extraConflictPair.get());
    curEventIdSolver->addConflicts(extraConflictPair.get(),
                                   intersectingConflictPairs);
    if (!checkColorable()) {
      return false;
    }
    extraConflictPairs.push_back(
        std::make_pair(std::move(extraConflictPair), parentScope));
    return true;
  };

  if (conflictPair->isInnerBackward && conflictPair->eventIdNode != nullptr) {
    bool insertOuterBwdConflictPair = false;
    if ((conflictPair->eventIdInfo.eventIdNum *
         conflictPair->eventIdInfo.eventIdRepeatNum) > 1) {
      insertOuterBwdConflictPair = true;
    } else if (options.isCrossCoreMode()) {
      if (setOcc->parentOcc == nullptr ||
          setOcc->parentOcc->parentOcc == nullptr ||
          setOcc->parentOcc->parentOcc->op != parentLCALoopOp) {
        insertOuterBwdConflictPair = true;
      } else if (waitOcc->parentOcc == nullptr ||
                 waitOcc->parentOcc->parentOcc == nullptr ||
                 waitOcc->parentOcc->parentOcc->op != parentLCALoopOp) {
        insertOuterBwdConflictPair = true;
      }
    }
    if (insertOuterBwdConflictPair) {
      // insert useless conflictPair to cover the whole loop when having
      // multi-eventid backward sync to reserve the eventIds.
      if (!insertExtraConflictPair(parentLCALoopBeforePHOcc,
                                   parentLCALoopAfterPHOcc,
                                   parentLCALoopOcc->parentOcc)) {
        return;
      }
    }
  }

  if (conflictPair->isInnerBackward && conflictPair->eventIdNode != nullptr) {
    // insert header/footer useless conflictPairs to reserve the eventIds.
    auto *loopOpOcc1 = getFirstIterOcc(waitOcc, normScopeOcc1);
    auto *loopOpOcc2 = getLastIterOcc(setOcc, normScopeOcc2);
    if (!insertExtraConflictPair(parentLCALoopBeforePHOcc, loopOpOcc1,
                                 parentLCALoopOcc, /*couldNotRun=*/true)) {
      return;
    }
    if (!insertExtraConflictPair(loopOpOcc2, parentLCALoopAfterPHOcc,
                                 parentLCALoopOcc, /*couldNotRun=*/true)) {
      return;
    }
  }

  bool dontInsert = false;
  if (conflictPair->isInnerBackward && normScopeOcc1 != normScopeOcc2) {
    auto *parCond = OperationBase::getParentCondition(conflictPair->setOp);
    if (auto *conditionOp = llvm::dyn_cast_if_present<Condition>(parCond)) {
      if (parentLCALoopOcc->op->isProperAncestor(conditionOp)) {
        scopeOccPairChosenConflicts[{normScopeOcc1, normScopeOcc2}].insert(
            conflictPair.get());
        dontInsert = true;
      }
    }
  }
  if (!dontInsert) {
    assert(parentLCALoopOcc != nullptr || normScopeOcc1 == normScopeOcc2);
    scopeOccChosenConflicts[normScopeOcc1].insert(conflictPair.get());
    scopeOccChosenConflicts[normScopeOcc2].insert(conflictPair.get());
  }

  memorizeSyncedPair(conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));

  for (auto &[extraConflictPair, parentScope] : extraConflictPairs) {
    scopeOccChosenConflicts[parentScope].insert(extraConflictPair.get());
    chosenConflictedPairs.push_back(std::move(extraConflictPair));
  }

  curEventIdSolver->clearActionStack();
}

void Solver::handleBarrierConflict(Occurrence *occ1, Occurrence *occ2,
                                   CorePipeInfo corePipeSrc,
                                   CorePipeInfo corePipeDst, bool isUseless,
                                   mlir::Value conflictBuffer) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);

  assert(corePipeSrc == corePipeDst);
  if (corePipeSrc.pipe == pto::PIPE::PIPE_S) {
    return;
  }
  if (options.isRegBasedArch) {
    if (corePipeSrc.pipe == pto::PIPE::PIPE_V ||
        corePipeSrc.pipe == pto::PIPE::PIPE_M) {
      return;
    }
  }
  auto *waitOcc = getBarrierWaitOcc(occ1, occ2);

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, waitOcc->op, waitOcc->op, waitOcc, waitOcc, corePipeSrc,
      corePipeDst, waitOcc->startIndex, waitOcc->startIndex);
  conflictPair->isUseless = isUseless;
  conflictPair->conflictBuffer = conflictBuffer;
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  LLVM_DEBUG({ llvm::dbgs() << conflictPair->str() << '\n'; });

  auto *normScopeOcc = waitOcc->parentOcc;
  scopeOccChosenConflicts[normScopeOcc].insert(conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
}

void Solver::handleUnitFlagConflict(Occurrence *occ1, Occurrence *occ2,
                                    CorePipeInfo corePipeSrc,
                                    CorePipeInfo corePipeDst,
                                    UnitFlagInfo unitFlagInfo, bool isUseless) {
  assert(occ1 != nullptr && occ2 != nullptr);
  auto *rwOp1 = llvm::dyn_cast_if_present<RWOperation>(occ1->op);
  auto *rwOp2 = llvm::dyn_cast_if_present<RWOperation>(occ2->op);
  assert(rwOp1 != nullptr && rwOp2 != nullptr);
  assert(corePipeSrc != corePipeDst);

  auto *setOcc = occ1;
  auto *waitOcc = occ2;
  auto *normScopeOcc1 = setOcc->parentOcc;
  auto *normScopeOcc2 = waitOcc->parentOcc;

  auto conflictPair = std::make_unique<ConflictPair>(
      rwOp1, rwOp2, setOcc->op, waitOcc->op, setOcc, waitOcc, corePipeSrc,
      corePipeDst, setOcc->endIndex, waitOcc->startIndex);
  assert(conflictPair->startIndex <= conflictPair->endIndex);

  conflictPair->isUseless = true;
  conflictPair->dontReuse = true;
  conflictPair->replacedWithUnitFlag = true;
  conflictPair->dontCheckForConflict = true;
  conflictPair->isInnerBackward = isBackwardSync(setOcc, waitOcc);

#ifndef NDEBUG
  Occurrence *parentLCALoopOcc{nullptr};
  if (conflictPair->isInnerBackward) {
    auto [parOcc1, parOcc2] = Occurrence::getLCAPair(occ1, occ2);
    assert(parOcc1 != nullptr && parOcc2 != nullptr);
    parentLCALoopOcc = Occurrence::getParentloop(parOcc1);
    assert(parentLCALoopOcc != nullptr);
  }

  LLVM_DEBUG({
    llvm::dbgs() << conflictPair->str() << '\n';
    if (parentLCALoopOcc != nullptr) {
      llvm::dbgs() << parentLCALoopOcc->op->str(0, false) << '\n';
    }
  });
#endif

  occ1->unitFlagInfo.merge(unitFlagInfo, occ1, occ2,
                           /*asSet=*/true, /*asWait=*/false);
  occ2->unitFlagInfo.merge(unitFlagInfo, occ1, occ2,
                           /*asSet=*/false, /*asWait=*/true);
  if (!isUseless) {
    rwOp1->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/true,
                                    /*asWait=*/false);
    rwOp2->mergedUnitFlagInfo.merge(unitFlagInfo, /*asSet=*/false,
                                    /*asWait=*/true);
  }

  scopeOccPairChosenConflicts[{normScopeOcc1, normScopeOcc2}].insert(
      conflictPair.get());
  chosenConflictedPairs.push_back(std::move(conflictPair));
}

void Solver::handleConflict(Occurrence *occ1, Occurrence *occ2,
                            RWOperation *rwOp1, RWOperation *rwOp2,
                            CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst,
                            EventIdInfo eventIdInfo, bool isUseless,
                            mlir::Value conflictBuffer) {
  if (!checkGraphConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo,
                          conflictBuffer)) {
    return;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "conflict found: " << "eventIdNum("
                 << eventIdInfo.eventIdNum << ")\n";
    llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                 << occ1->endIndex << ' ' << rwOp1->str(0, false) << '\n';
    llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                 << occ2->endIndex << ' ' << rwOp2->str(0, false) << '\n';
  });
  if (corePipeSrc == corePipeDst) {
    handleBarrierConflict(occ1, occ2, corePipeSrc, corePipeDst, isUseless,
                          conflictBuffer);
  } else if (auto unitFlagInfo = checkUnitFlagPatterns(occ1, occ2)) {
    handleUnitFlagConflict(occ1, occ2, corePipeSrc, corePipeDst,
                           unitFlagInfo.value(), isUseless);
  } else {
    handleSetWaitConflict(occ1, occ2, corePipeSrc, corePipeDst, eventIdInfo,
                          isUseless, conflictBuffer);
  }
}

void Solver::calcAllEventIds() {
  for (auto &[pipes, eventIdSolver] : eventIdSolver) {
    assert(eventIdSolver != nullptr);

    [[maybe_unused]] auto result =
        eventIdSolver->shrinkEventIdMaxToEventIdNum();
    assert(llvm::succeeded(result));
    assert(eventIdSolver->isColorable());
  }
}

void Solver::collectBackwardSyncEventIds() {
  LLVM_DEBUG(llvm::dbgs() << "collectBackwardSyncEventIds\n";);
  for (auto &conflictPair : chosenConflictedPairs) {
    if (!conflictPair->isUseless && conflictPair->isInnerBackward &&
        conflictPair->eventIdNode != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "  " << conflictPair->str() << "\n";);
      for (auto eventId : conflictPair->eventIdNode->getEventIds()) {
        auto &e = backwardSyncEvents[conflictPair->backwardSyncLoopOp]
                                    [{conflictPair->setCorePipeInfo,
                                      conflictPair->waitCorePipeInfo}][eventId];
        e = std::max(e, conflictPair->eventIdInfo.eventIdRepeatNum);
      }
    }
  }
}

void Solver::resetAndBuildSetWaitOpIndex(const SyncMap &syncMapBefore,
                                         const SyncMap &syncMapAfter) {
  globalSetWaitIndex = 0;
  setWaitStartIndex.clear();
  setWaitEndIndex.clear();
  setWaitStartIndexInclusive.clear();
  setWaitEndIndexInclusive.clear();
  setWaitFlagOpsIndex.clear();
  collectSetWaitOpsIndexes(funcIr.get(), syncMapBefore, syncMapAfter);
}

std::set<std::pair<int64_t, SetWaitOp *>> &
Solver::getSetWaitOpsIndexRef(pto::PIPE pipeSrc, pto::PIPE pipeDst,
                              int64_t eventId) {
  auto key = std::make_tuple(pipeSrc, pipeDst, eventId);
  return setWaitFlagOpsIndex[key];
}

// Collect indices for all Set/Wait ops to facilitate merging decisions.
void Solver::collectSetWaitOpsIndexes(OperationBase *op,
                                      const SyncMap &syncMapBefore,
                                      const SyncMap &syncMapAfter) {
  assert(op != nullptr);
  setWaitStartIndexInclusive[op] = globalSetWaitIndex++;
  if (syncMapBefore.count(op)) {
    auto *it = syncMapBefore.find(op);
    assert(it != syncMapBefore.end());
    for (auto &syncOp : it->second) {
      if (auto *setWaitOp = llvm::dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          auto &index = getSetWaitOpsIndexRef(setWaitOp->pipeSrc,
                                              setWaitOp->pipeDst, eventId);
          index.insert({globalSetWaitIndex++, setWaitOp});
        }
      }
    }
  }
  setWaitStartIndex[op] = globalSetWaitIndex++;
  if (auto *scopeOp = llvm::dyn_cast<Scope>(op)) {
    for (auto &childOp : scopeOp->body) {
      collectSetWaitOpsIndexes(childOp.get(), syncMapBefore, syncMapAfter);
    }
  }
  setWaitEndIndex[op] = globalSetWaitIndex++;
  if (syncMapAfter.count(op)) {
    auto *it = syncMapAfter.find(op);
    assert(it != syncMapAfter.end());
    for (auto &syncOp : it->second) {
      if (auto *setWaitOp = llvm::dyn_cast<SetWaitOp>(syncOp.get())) {
        for (auto eventId : setWaitOp->eventIds) {
          auto &index = getSetWaitOpsIndexRef(setWaitOp->pipeSrc,
                                              setWaitOp->pipeDst, eventId);
          index.insert({globalSetWaitIndex++, setWaitOp});
        }
      }
    }
  }
  setWaitEndIndexInclusive[op] = globalSetWaitIndex++;
}

bool Solver::checkBackwardSyncEventsContains(OperationBase *op,
                                             CorePipeInfo corePipeSrc,
                                             CorePipeInfo corePipeDst,
                                             int64_t eventId) {
  auto *it1 = backwardSyncEvents.find(op);
  if (it1 == backwardSyncEvents.end()) {
    return false;
  }
  auto it2 = it1->second.find({corePipeSrc, corePipeDst});
  if (it2 == it1->second.end()) {
    return false;
  }
  return it2->second.contains(eventId);
}

bool Solver::checkBackwardSyncEventsContainsAfterMerge(
    OperationBase *op, CorePipeInfo corePipeSrc, CorePipeInfo corePipeDst) {
  auto *it1 = backwardSyncEventsAfterMerge.find(op);
  if (it1 == backwardSyncEventsAfterMerge.end()) {
    return false;
  }
  return it1->second.contains({corePipeSrc, corePipeDst});
}

// Check whether a backward-sync event id can be merged at scope level.
bool Solver::checkMergeable(Scope *scopeOp, CorePipeInfo corePipeSrc,
                            CorePipeInfo corePipeDst, int64_t eventId,
                            bool shouldBeUsedAtleastOnce) {
  auto &index =
      getSetWaitOpsIndexRef(corePipeSrc.pipe, corePipeDst.pipe, eventId);
  if (shouldBeUsedAtleastOnce) {
    auto it = index.lower_bound({setWaitStartIndexInclusive[scopeOp], nullptr});
    bool usedAtleastOnce =
        it != index.end() && it->first < setWaitEndIndexInclusive[scopeOp];
    if (!usedAtleastOnce) {
      return false;
    }
  }
  {
    auto it1 =
        index.lower_bound({setWaitStartIndexInclusive[scopeOp], nullptr});
    auto it2 = index.lower_bound({setWaitEndIndex[scopeOp], nullptr});
    bool usedBefore =
        it1 != index.end() && it1->first < setWaitStartIndex[scopeOp];
    bool usedAfter =
        it2 != index.end() && it2->first < setWaitEndIndexInclusive[scopeOp];
    if (usedBefore || usedAfter) {
      return false;
    }
  }
  if (auto *conditionOp = llvm::dyn_cast<Condition>(scopeOp)) {
    if (!conditionOp->hasFalseScope()) {
      return false;
    }
    return checkMergeable(conditionOp->getTrueScope(), corePipeSrc, corePipeDst,
                          eventId, true) &&
           checkMergeable(conditionOp->getFalseScope(), corePipeSrc,
                          corePipeDst, eventId, true);
  }
  if (auto *loopOp = llvm::dyn_cast<Loop>(scopeOp)) {
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
        if (!checkMergeable(childScopeOp, corePipeSrc, corePipeDst, eventId,
                            false)) {
          return false;
        }
      }
    }
    for (auto &childOp : loopOp->body) {
      if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
        if (checkMergeable(childScopeOp, corePipeSrc, corePipeDst, eventId,
                           true)) {
          return true;
        }
      }
    }
    return false;
  }
  for (auto &childOp : scopeOp->body) {
    auto it1 =
        index.lower_bound({setWaitStartIndexInclusive[childOp.get()], nullptr});
    auto it2 = index.lower_bound({setWaitEndIndex[childOp.get()], nullptr});
    bool usedAtleastOnce = it1 != index.end() &&
                           it1->first < setWaitEndIndexInclusive[childOp.get()];
    if (!usedAtleastOnce) {
      continue;
    }
    bool before =
        it1 != index.end() && it1->first < setWaitStartIndex[childOp.get()];
    bool after = it2 != index.end() &&
                 it2->first < setWaitEndIndexInclusive[childOp.get()];
    if (before || after) {
      return false;
    }
    if (!checkBackwardSyncEventsContains(childOp.get(), corePipeSrc,
                                         corePipeDst, eventId)) {
      return false;
    }
    if (checkBackwardSyncEventsContainsAfterMerge(childOp.get(), corePipeSrc,
                                                  corePipeDst)) {
      return false;
    }
  }
  return true;
}

// Attempt to merge backward sync events across children and prune duplicates.
void Solver::mergeBackwardSyncEventIds(OperationBase *op) {
  auto *scopeOp = llvm::dyn_cast_if_present<Scope>(op);
  if (scopeOp == nullptr) {
    return;
  }
  for (auto &op : scopeOp->body) {
    mergeBackwardSyncEventIds(op.get());
  }

  if (llvm::isa_and_present<FunctionBlock>(op)) {
    return;
  }
  if (llvm::isa_and_present<Condition, Loop>(op->parentOp)) {
    return;
  }

  auto *conditionOp = llvm::dyn_cast<Condition>(op);
  if (conditionOp != nullptr) {
    if (!conditionOp->hasFalseScope()) {
      return;
    }
  }

  llvm::DenseSet<std::tuple<CorePipeInfo, CorePipeInfo, int64_t>> toBeErased;

  llvm::SmallVector<pto::TCoreType> coreTypes;
  if (options.isCrossCoreMode()) {
    coreTypes = {pto::TCoreType::VECTOR, pto::TCoreType::CUBE};
  } else {
    coreTypes = {pto::TCoreType::CUBE_OR_VECTOR};
  }
  size_t pipeNumMax = static_cast<size_t>(pto::PIPE::PIPE_NUM);
  const int64_t eventIdMax = getHWAvailableEventIdNum(options.syncMode);

  for (int64_t eventId = 0; eventId < eventIdMax; ++eventId) {
    for (auto coreSrc : coreTypes) {
      for (auto coreDst : coreTypes) {
        for (size_t pipeSrcInt = 0; pipeSrcInt < pipeNumMax; pipeSrcInt++) {
          for (size_t pipeDstInt = 0; pipeDstInt < pipeNumMax; pipeDstInt++) {
            auto pipeSrc = static_cast<pto::PIPE>(pipeSrcInt);
            auto pipeDst = static_cast<pto::PIPE>(pipeDstInt);
            auto corePipeSrc = CorePipeInfo(coreSrc, pipeSrc);
            auto corePipeDst = CorePipeInfo(coreDst, pipeDst);
            if (checkBackwardSyncEventsContains(scopeOp, corePipeSrc,
                                                corePipeDst, eventId)) {
              continue;
            }
            if (checkMergeable(scopeOp, corePipeSrc, corePipeDst, eventId)) {
              toBeErased.insert({corePipeSrc, corePipeDst, eventId});
              backwardSyncEvents[scopeOp][{corePipeSrc, corePipeDst}].insert(
                  {eventId, 1});
            }
          }
        }
      }
    }
  }

  if (isa<Condition, Loop>(scopeOp)) {
    for (auto &op : scopeOp->body) {
      if (auto *block = llvm::dyn_cast<Scope>(op.get())) {
        for (auto &childOp : block->body) {
          if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
            for (auto [corePipeSrc, corePipeDst, eventId] : toBeErased) {
              if (checkBackwardSyncEventsContains(childScopeOp, corePipeSrc,
                                                  corePipeDst, eventId)) {
                auto key = std::make_tuple(corePipeSrc, corePipeDst);
                backwardSyncEvents[childScopeOp][key].erase(eventId);
                if (backwardSyncEvents[childScopeOp][key].empty()) {
                  backwardSyncEvents[childScopeOp].erase(key);
                }
              }
            }
          }
        }
      }
    }
  } else {
    for (auto &childOp : scopeOp->body) {
      if (auto *childScopeOp = llvm::dyn_cast<Scope>(childOp.get())) {
        for (auto [corePipeSrc, corePipeDst, eventId] : toBeErased) {
          if (checkBackwardSyncEventsContains(childScopeOp, corePipeSrc,
                                              corePipeDst, eventId)) {
            auto key = std::make_tuple(corePipeSrc, corePipeDst);
            backwardSyncEvents[childScopeOp][key].erase(eventId);
            if (backwardSyncEvents[childScopeOp][key].empty()) {
              backwardSyncEvents[childScopeOp].erase(key);
            }
          }
        }
      }
    }
  }
}

void Solver::mergeBackwardSyncPairs(SyncMap &syncMapBefore,
                                    SyncMap &syncMapAfter) {
  if (!options.moveOutAndMergeBackwardSyncPairs) {
    return;
  }
  if (options.isIntraCoreMode()) {
    resetAndBuildSetWaitOpIndex(syncMapBefore, syncMapAfter);
    auto *scopeOp = llvm::dyn_cast<Scope>(funcIr.get());
    assert(scopeOp != nullptr && scopeOp->body.front() != nullptr);
    mergeBackwardSyncEventIds(scopeOp->body.front().get());
  }
}

SyncBeforeAfterMap Solver::getBeforeAfterSyncMaps() {
  calcAllEventIds();
  SyncMap syncMapBefore, syncMapAfter;
  std::vector<ConflictPair *> conflictPairs;
  for (auto &conflictPair : chosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }
  for (auto &conflictPair : persistentChosenConflictedPairs) {
    conflictPairs.push_back(conflictPair.get());
  }

  // Buf-id mirror-image deduplication.
  //
  // The forward and backward memory hazards for the same (producer op,
  // consumer op) pair across pipes generate two distinct ConflictPairs:
  //   F: setOp=A@Pa, waitOp=B@Pb   (forward,  intra-iter)
  //   B: setOp=B@Pb, waitOp=A@Pa   (backward, loop-carried)
  // In set-wait this is necessary — F's pair fires within an iteration,
  // B's pair fires across the loop boundary. In buf-id the two pairs would
  // bracket the same (op, pipe) anchor set with two different ids,
  // duplicating brackets needlessly: the scoreboard counter ordering
  // produced by one bracket pair already enforces both deps (per doc
  // §1.2's canonical "for { load; vector }" example). Drop one of the
  // mirror images so we emit a single bracket pair per anchor set.
  //
  // We keep the lower-id ConflictPair (deterministic, and tends to keep
  // the forward pair which was usually allocated first).
  using AnchorEnd = std::pair<OperationBase *, pto::PIPE>;

  llvm::DenseSet<ConflictPair *> bufIdRedundantMirror;
  if (options.isBufIdEmit()) {
    using AnchorKey = std::pair<AnchorEnd, AnchorEnd>;
    llvm::DenseMap<AnchorKey, ConflictPair *> seen;
    for (auto *cp : conflictPairs) {
      if (cp->isUseless || cp->replacedWithUnitFlag || cp->isBarrier())
        continue;
      // Key on the raw RWOperations (op1/op2), not the solver's hoisted
      // setOp/waitOp, so that mirror pairs are recognized at the actual
      // RWOperation level regardless of where the solver placed the anchor.
      AnchorEnd a{static_cast<OperationBase *>(cp->op1),
                  cp->setCorePipeInfo.pipe};
      AnchorEnd b{static_cast<OperationBase *>(cp->op2),
                  cp->waitCorePipeInfo.pipe};
      if (b < a)
        std::swap(a, b);
      AnchorKey key{a, b};
      auto [it, inserted] = seen.try_emplace(key, cp);
      if (!inserted) {
        // Pick the smaller id as the keeper so output is stable.
        if (cp->id < it->second->id) {
          bufIdRedundantMirror.insert(it->second);
          it->second = cp;
        } else {
          bufIdRedundantMirror.insert(cp);
        }
      }
    }
  }

  // Buf-id emit takes a separate path: collect unique (anchor_op, pipe, bufId)
  // brackets from the forward ConflictPair set with a buffer-aware id-grouping
  // pre-pass, then emit one bracket per unique triple. See the long comment
  // inside the if-block.
  if (options.isBufIdEmit()) {
    // === Buffer-aware id grouping ===
    //
    // Two ConflictPairs that synchronize on the same buffer must share a
    // buf-id so that the (get_cnt, rel_cnt) chain in the forward bracket
    // sequence transitively orders cross-iter conflicts (e.g. iter i+1's
    // load after iter i's store) for free. Two pairs on different buffers
    // can keep distinct ids for parallelism.
    //
    // ConflictPair doesn't carry buffer info directly. Proxy:
    //   F1, F2 are on the same buffer chain iff
    //     - they share an anchor (op, pipe), AND
    //     - some backward ConflictPair B has F1's non-shared endpoint and
    //       F2's non-shared endpoint as its two anchors.
    // Rationale: same buffer means iter i's read on one endpoint conflicts
    // with iter i+1's write on the other endpoint, which is exactly what
    // the solver records as a backward edge between them.
    //
    // We union-find forward pairs by this relation, then each group gets a
    // single shared bufId (the smallest among members' allocated ids).
    // Backward pairs are never emitted — their semantic is folded into the
    // forward chain through id sharing.

    llvm::DenseMap<ConflictPair *, ConflictPair *> ufParent;
    std::function<ConflictPair *(ConflictPair *)> ufFind =
        [&](ConflictPair *x) -> ConflictPair * {
      while (ufParent[x] != x) {
        ufParent[x] = ufParent[ufParent[x]];
        x = ufParent[x];
      }
      return x;
    };
    auto ufUnion = [&](ConflictPair *a, ConflictPair *b) {
      auto *ra = ufFind(a);
      auto *rb = ufFind(b);
      if (ra != rb)
        ufParent[ra] = rb;
    };

    llvm::SmallVector<ConflictPair *> forwardPairs;
    llvm::SmallVector<ConflictPair *> backwardPairs;
    for (auto *cp : conflictPairs) {
      if (cp->isUseless || cp->replacedWithUnitFlag || cp->isBarrier())
        continue;
      if (bufIdRedundantMirror.contains(cp))
        continue;
      if (cp->isInnerBackward) {
        backwardPairs.push_back(cp);
      } else {
        forwardPairs.push_back(cp);
        ufParent[cp] = cp;
      }
    }

    // Use the RAW producer/consumer RWOperation anchors (op1, op2) rather
    // than the solver's setOp/waitOp. The latter may have been hoisted to
    // an LCA / PlaceHolder under the scope of the producer or consumer
    // (correct for set/wait, which needs strict pairing along every CFG
    // edge), but in buf-id mode we want brackets *tightly* around the
    // actual RWOperation. Hoisting buf-id brackets above an scf.if would
    // make the counter advance even when the conditional op doesn't run,
    // which decouples the get_cnt/rel_cnt semantics from real pipe activity.
    auto anchorsOf = [](ConflictPair *cp) -> std::array<AnchorEnd, 2> {
      return {{{static_cast<OperationBase *>(cp->op1),
                cp->setCorePipeInfo.pipe},
               {static_cast<OperationBase *>(cp->op2),
                cp->waitCorePipeInfo.pipe}}};
    };

    for (size_t i = 0; i < forwardPairs.size(); ++i) {
      for (size_t j = i + 1; j < forwardPairs.size(); ++j) {
        auto *f1 = forwardPairs[i];
        auto *f2 = forwardPairs[j];
        auto a1 = anchorsOf(f1);
        auto a2 = anchorsOf(f2);
        bool bridged = false;
        for (int k = 0; k < 2 && !bridged; ++k) {
          for (int l = 0; l < 2 && !bridged; ++l) {
            if (a1[k] != a2[l])
              continue;
            AnchorEnd ns1 = a1[1 - k];
            AnchorEnd ns2 = a2[1 - l];
            if (ns1 == ns2)
              continue;
            for (auto *b : backwardPairs) {
              auto ab = anchorsOf(b);
              bool hasNs1 = (ab[0] == ns1 || ab[1] == ns1);
              bool hasNs2 = (ab[0] == ns2 || ab[1] == ns2);
              if (hasNs1 && hasNs2) {
                bridged = true;
                break;
              }
            }
          }
        }
        if (bridged)
          ufUnion(f1, f2);
      }
    }

    llvm::DenseMap<ConflictPair *, int64_t> groupId;
    for (auto *f : forwardPairs) {
      if (!f->eventIdNode || f->eventIdNode->getEventIds().empty())
        continue;
      int64_t myId = f->eventIdNode->getEventIds().front();
      auto *root = ufFind(f);
      auto it = groupId.find(root);
      if (it == groupId.end() || myId < it->second)
        groupId[root] = myId;
    }

    // === Emit unique (anchor_op, pipe, bufId) brackets from forward pairs ===
    //
    // Multiple ConflictPairs in the same group will contribute the same
    // (op, pipe, groupId) on each shared anchor; emit each unique triple
    // only once so that we don't violate spec constraint 1 (no two
    // consecutive get_buf(pipe, #id) before a single op).
    llvm::DenseSet<std::pair<AnchorEnd, int64_t>> emittedBefore;
    llvm::DenseSet<std::pair<AnchorEnd, int64_t>> emittedAfter;
    auto emitBracket = [&](AnchorEnd anchor, int64_t bufId,
                           ConflictPair *debugCp) {
      auto key = std::make_pair(anchor, bufId);
      if (emittedBefore.insert(key).second) {
        auto getBuf = std::make_unique<GetBufOp>(
            anchor.first->op, anchor.first->parentOp, anchor.second, bufId);
        LLVM_DEBUG(getBuf->debugId = debugCp->id);
        syncMapBefore[anchor.first].push_back(std::move(getBuf));
      }
      if (emittedAfter.insert(key).second) {
        auto rlsBuf = std::make_unique<RlsBufOp>(
            anchor.first->op, anchor.first->parentOp, anchor.second, bufId);
        LLVM_DEBUG(rlsBuf->debugId = debugCp->id);
        syncMapAfter[anchor.first].push_back(std::move(rlsBuf));
      }
    };

    for (auto *cp : conflictPairs) {
      if (cp->isUseless || cp->replacedWithUnitFlag || cp->isBarrier())
        continue;
      if (cp->isInnerBackward)
        continue;
      if (bufIdRedundantMirror.contains(cp))
        continue;
      if (!cp->eventIdNode || cp->eventIdNode->getEventIds().empty())
        continue;
      auto gIt = groupId.find(ufFind(cp));
      int64_t bufId = (gIt != groupId.end())
                          ? gIt->second
                          : cp->eventIdNode->getEventIds().front();
      // Anchor at the raw RWOperations (see comment on anchorsOf).
      emitBracket({static_cast<OperationBase *>(cp->op1),
                   cp->setCorePipeInfo.pipe},
                  bufId, cp);
      emitBracket({static_cast<OperationBase *>(cp->op2),
                   cp->waitCorePipeInfo.pipe},
                  bufId, cp);
    }

    return std::make_pair(std::move(syncMapBefore), std::move(syncMapAfter));
  }

  // === set-wait path (unchanged) ===
  for (auto *conflictPair : conflictPairs) {
    if (conflictPair->isUseless) {
      continue;
    }
    if (conflictPair->replacedWithUnitFlag) {
      continue;
    }
    assert(conflictPair->setOp != nullptr && conflictPair->waitOp != nullptr);
    if (conflictPair->isBarrier()) {
      auto barrierOp = std::make_unique<BarrierOp>(
          conflictPair->waitOp->op, conflictPair->waitOp->parentOp,
          conflictPair->waitCorePipeInfo.pipe);
      LLVM_DEBUG(barrierOp->debugId = conflictPair->id);
      syncMapBefore[conflictPair->waitOp].push_back(std::move(barrierOp));
    } else {
      assert(conflictPair->eventIdNode != nullptr);
      auto setOp = std::make_unique<SetFlagOp>(
          conflictPair->setOp->op, conflictPair->setOp->parentOp,
          conflictPair->eventIdNode->getEventIds(),
          conflictPair->setCorePipeInfo.pipe,
          conflictPair->waitCorePipeInfo.pipe);
      auto waitOp = std::make_unique<WaitFlagOp>(
          conflictPair->waitOp->op, conflictPair->waitOp->parentOp,
          conflictPair->eventIdNode->getEventIds(),
          conflictPair->setCorePipeInfo.pipe,
          conflictPair->waitCorePipeInfo.pipe);
      if (options.isCrossCoreMode()) {
        setOp->coreType = conflictPair->setCorePipeInfo.coreType;
        waitOp->coreType = conflictPair->waitCorePipeInfo.coreType;
      }
      setOp->eventIdInfo = conflictPair->eventIdInfo;
      waitOp->eventIdInfo = conflictPair->eventIdInfo;
      setOp->checkLastIter = conflictPair->setOnLastIterOnly;
      waitOp->checkFirstIter = conflictPair->waitOnFirstIterOnly;
      LLVM_DEBUG({
        setOp->debugId = conflictPair->id;
        waitOp->debugId = conflictPair->id;
      });
      assert(setOp != nullptr && waitOp != nullptr);
      syncMapAfter[conflictPair->setOp].push_back(std::move(setOp));
      syncMapBefore[conflictPair->waitOp].push_front(std::move(waitOp));
    }
  }

  collectBackwardSyncEventIds();
  mergeBackwardSyncPairs(syncMapBefore, syncMapAfter);

  for (auto &[op, mp] : backwardSyncEvents) {
    if (mp.empty()) {
      continue;
    }
    auto *scopeOp = llvm::dyn_cast<Scope>(op);
    assert(scopeOp != nullptr);
    for (auto [setWaitCorePipes, eventIdsMp] : mp) {
      if (eventIdsMp.empty()) {
        continue;
      }
      llvm::SmallVector<int64_t> eventIds;
      for (auto [eventId, repeatNum] : eventIdsMp) {
        llvm::SmallVector<int64_t> curEventIds(repeatNum, eventId);
        llvm::append_range(eventIds, curEventIds);
      }
      llvm::sort(eventIds);
      auto [corePipeSrc, corePipeDst] = setWaitCorePipes;
      auto setOp =
          std::make_unique<SetFlagOp>(scopeOp->op, scopeOp->parentOp, eventIds,
                                      corePipeSrc.pipe, corePipeDst.pipe);
      auto waitOp =
          std::make_unique<WaitFlagOp>(scopeOp->op, scopeOp->parentOp, eventIds,
                                       corePipeSrc.pipe, corePipeDst.pipe);
      setOp->allAtOnce = true;
      waitOp->allAtOnce = true;
      if (options.isCrossCoreMode()) {
        setOp->coreType = corePipeSrc.coreType;
        waitOp->coreType = corePipeDst.coreType;
      }
      assert(setOp != nullptr && waitOp != nullptr);
      syncMapBefore[scopeOp].push_back(std::move(setOp));
      syncMapAfter[scopeOp].push_front(std::move(waitOp));
    }
  }
  return std::make_pair(std::move(syncMapBefore), std::move(syncMapAfter));
}

void Solver::processConflict(Occurrence *occ1, Occurrence *occ2,
                             RWOperation *rwOp1, RWOperation *rwOp2,
                             bool isUseless) {
  for (auto [corePipeSrc, corePipeDst, conflictBuffer] :
       checkMemoryConflicts(rwOp1, rwOp2)) {
    if (options.alwaysUsePipeSAsWaitingPipe) {
      corePipeDst.pipe = pto::PIPE::PIPE_S;
    }
    auto eventIdInfo =
        getEventIdInfo(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst);
    handleConflict(occ1, occ2, rwOp1, rwOp2, corePipeSrc, corePipeDst,
                   eventIdInfo, isUseless, conflictBuffer);
  }
}

// Main processing loop that iterates processingOrders and attempts to
// discover and record conflicts.
void Solver::processOrders() {
  for (auto &[occ1, occ2, rwOp1, rwOp2, isUseless] : processingOrders) {
    assert(occ1 != occ2);
    assert(occ1->syncIrIndex < occ2->syncIrIndex);
    if (checkVisited(occ1, occ2)) {
      assert(false && "expected to not check a pair more than once.");
      continue;
    }
    if (checkImpossibleOccPair(occ1, occ2) || checkAlreadySynced(occ1, occ2) ||
        skipMMad1DecomposedLoopOpt(occ1, occ2) ||
        checkSkipParallelLoop(occ1, occ2) ||
        checkSkipCrossCorePair(occ1, occ2)) {
      continue;
    }
    DEBUG_WITH_TYPE("gss-sync-solver-checking", {
      llvm::dbgs() << "checking: " << (isUseless ? "is-useless\n" : "\n");
      llvm::dbgs() << occ1->syncIrIndex << ' ' << occ1->startIndex << ' '
                   << occ1->endIndex << ' ' << occ1->op->str(0, false) << '\n';
      llvm::dbgs() << occ2->syncIrIndex << ' ' << occ2->startIndex << ' '
                   << occ2->endIndex << ' ' << occ2->op->str(0, false) << '\n';
    });
    if (checkAlreadySyncedWithUnitFlag(occ1, occ2)) {
      continue;
    }
    processConflict(occ1, occ2, rwOp1, rwOp2, isUseless);
  }
}

void Solver::insertMergedBackwardSyncPairs() {
  for (auto &[scopeOp, st] : backwardSyncEventsAfterMerge) {
    for (auto &corePipeInfoPair : st) {
      auto [corePipeSrc, corePipeDst] = corePipeInfoPair;
      for (auto *scopeOcc : opAllOccurrences[scopeOp]) {
        auto *parentScopeOcc = scopeOcc->parentOcc;
        assert(parentScopeOcc != nullptr);
        Occurrence *setOcc = nullptr;
        Occurrence *waitOcc = nullptr;
        auto startIndex = scopeOcc->startIndex;
        auto endIndex = scopeOcc->endIndex;
        if (isa<Loop>(scopeOp)) {
          setOcc = getBeforePlaceHolderOcc(scopeOcc);
          waitOcc = getAfterPlaceHolderOcc(scopeOcc);
          startIndex = setOcc->endIndex;
          endIndex = waitOcc->startIndex;
        }
        auto conflictPair = std::make_unique<ConflictPair>(
            nullptr, nullptr, nullptr, nullptr, setOcc, waitOcc, corePipeSrc,
            corePipeDst, startIndex, endIndex);
        assert(conflictPair->startIndex <= conflictPair->endIndex);
        conflictPair->isUseless = true;
        conflictPair->dontReuse = true;
        conflictPair->dontCheckForConflict = true;
        conflictPair->couldNotRun = false; // notice this
        LLVM_DEBUG({
          llvm::dbgs() << "consider-merged-backward-pair: "
                       << scopeOp->str(0, false) << ' ' << conflictPair->str()
                       << "\n";
        });
        scopeOccChosenConflicts[parentScopeOcc].insert(conflictPair.get());
        chosenConflictedPairs.push_back(std::move(conflictPair));
      }
    }
  }
}

llvm::LogicalResult Solver::considerOuterBackwardSyncPairs() {
  if (!options.considerOuterBackwardSyncPairs) {
    return llvm::failure();
  }
  bool backwardPairsPositionChanged = false;
  for (auto &[scopeOp, st] : backwardSyncEventsAfterMerge) {
    SmallVector<std::tuple<CorePipeInfo, CorePipeInfo>> toBeErased;
    for (auto &corePipeInfoPair : st) {
      if (!backwardSyncEvents.contains(scopeOp) ||
          !backwardSyncEvents[scopeOp].contains(corePipeInfoPair)) {
        toBeErased.push_back(corePipeInfoPair);
      }
    }
    if (!toBeErased.empty()) {
      backwardPairsPositionChanged = true;
      for (auto &corePipeInfoPair : toBeErased) {
        st.erase(corePipeInfoPair);
      }
    }
  }
  int chosenOpsDepth = -1;
  SmallVector<OperationBase *> chosenOps;
  for (auto &[scopeOp, mp] : backwardSyncEvents) {
    if (backwardSyncEventsAfterMerge.contains(scopeOp)) {
      continue;
    }
    int scopeOpDepth = scopeOp->getDepth();
    if (chosenOpsDepth == scopeOpDepth) {
      chosenOps.push_back(scopeOp);
    } else if (chosenOpsDepth == -1 || chosenOpsDepth < scopeOpDepth) {
      chosenOps.clear();
      chosenOps.push_back(scopeOp);
      chosenOpsDepth = scopeOpDepth;
    }
  }
  if (chosenOps.empty()) {
    return llvm::failure();
  }
  bool newPairIsInserted = false;
  for (auto *chosenOp : chosenOps) {
    for (auto &[corePipeInfoPair, eventIdsMp] : backwardSyncEvents[chosenOp]) {
      assert(!eventIdsMp.empty());
      if (!eventIdsMp.empty()) {
        auto [it, isInserted] =
            backwardSyncEventsAfterMerge[chosenOp].insert(corePipeInfoPair);
        newPairIsInserted |= isInserted;
      }
    }
  }
  return llvm::success(backwardPairsPositionChanged || newPairIsInserted);
}

llvm::LogicalResult Solver::reuseSyncPairToSaveEventIds() {
  if (!options.reuseSyncPairToSaveEventIds || barrierAllPairs.empty()) {
    return llvm::failure();
  }
  bool limitReached = true;
  for (auto [corePipeSrc, corePipeDst] : barrierAllPairs) {
    if (reusePairs[{corePipeSrc, corePipeDst}] < maxReuseNum) {
      if (reusePairs[{corePipeSrc, corePipeDst}] <=
          reusedPairs[{corePipeSrc, corePipeDst}]) {
        reusePairs[{corePipeSrc, corePipeDst}] += 1;
        limitReached = false;
      }
    }
  }
  DEBUG_WITH_TYPE("gss-sync-solver-reuse", {
    llvm::dbgs() << "reusePairs: \n";
    for (auto [pipeCorePairs, cnt] : reusePairs) {
      llvm::dbgs() << get<0>(pipeCorePairs).pipe << ' '
                   << get<1>(pipeCorePairs).pipe << ' ' << cnt << '\n';
    }
  });
  return llvm::success(!limitReached);
}

llvm::LogicalResult Solver::disableMultiEventIdForBarrierAllPairs() {
  if (!options.disableMultiEventIdForBarrierAllPairs ||
      barrierAllPairs.empty()) {
    return llvm::failure();
  }
  bool newPairIsInserted = false;
  for (auto corePipeInfoPair : barrierAllPairs) {
    auto [it, isInserted] = disabledMultiEventIdPairs.insert(corePipeInfoPair);
    newPairIsInserted |= isInserted;
  }
  LLVM_DEBUG({
    if (newPairIsInserted) {
      llvm::dbgs() << "disabled-multi-event-id-pairs: \n";
      for (auto &[corePipeSrc, corePipeDst] : disabledMultiEventIdPairs) {
        llvm::dbgs() << corePipeSrc.coreType << ' ' << corePipeSrc.pipe << ' '
                     << corePipeDst.coreType << ' ' << corePipeDst.pipe << '\n';
      }
    }
  });
  return llvm::success(newPairIsInserted);
}

llvm::LogicalResult Solver::tryMovingOutBackwardSyncPairsToOuterLoops() {
  if (!options.moveOutAndMergeBackwardSyncPairs || !options.isCrossCoreMode() ||
      dontMoveBackwardSyncPairsToOutmostLoop) {
    return llvm::failure();
  }
  if (!moveBackwardSyncPairsToOutmostLoop) {
    moveBackwardSyncPairsToOutmostLoop = true;
    return llvm::success();
  }
  if (!barrierAllPairs.empty()) {
    moveBackwardSyncPairsToOutmostLoop = false;
    dontMoveBackwardSyncPairsToOutmostLoop = true;
    return llvm::success();
  }
  return llvm::failure();
}

// High-level solve orchestration with multiple passes and optional merging
// iterations.
llvm::LogicalResult Solver::runSolver(bool enableOpts1, bool enableOpts2) {
  reset(/*resetEventIdRanOutOpts=*/true);

  int64_t runNum = 0;
  while (runNum++ < maxRunNum) {
    LLVM_DEBUG(llvm::dbgs() << "runNum: " << runNum << '\n');

    reset();
    insertMergedBackwardSyncPairs();
    processOrders();

    if (llvm::succeeded(tryMovingOutBackwardSyncPairsToOuterLoops())) {
      continue;
    }

    if (enableOpts1) {
      if (options.considerOuterBackwardSyncPairs) {
        getBeforeAfterSyncMaps();
        if (llvm::succeeded(considerOuterBackwardSyncPairs())) {
          continue;
        }
        if (!barrierAllPairs.empty()) {
          backwardSyncEventsAfterMerge.clear();
        }
      }
    }

    if (enableOpts2) {
      if (!barrierAllPairs.empty()) {
        if (llvm::succeeded(reuseSyncPairToSaveEventIds())) {
          continue;
        }
        if (llvm::succeeded(disableMultiEventIdForBarrierAllPairs())) {
          continue;
        }
      }
    }

    if (!barrierAllPairs.empty()) {
      pickAndInsertABarrierAll();
      reset(/*resetEventIdRanOutOpts=*/true);
      continue;
    }
    break;
  }

  reset();
  insertMergedBackwardSyncPairs();
  processOrders();

  return llvm::success(runNum < maxRunNum);
}

void Solver::solve() {
  if (llvm::succeeded(runSolver())) {
    return;
  }
  if (!options.isTestMode()) {
    if (llvm::succeeded(runSolver(/*enableOpts1=*/false))) {
      return;
    }
    if (llvm::succeeded(
            runSolver(/*enableOpts1=*/false, /*enableOpts2=*/false))) {
      return;
    }
  }
  llvm_unreachable("GSS: runSolver() failed.");
}
