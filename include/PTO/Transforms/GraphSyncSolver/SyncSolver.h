// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- SyncSolver.h - GraphSyncSolver main loop -----------------*- C++ -*-===//
//
// Drives the "for each pair (occ1, occ2) decide whether to add a sync"
// algorithm, delegating reachability to GraphSolver and event-id assignment
// to EventIdSolver. When the latter cannot find an assignment, the candidate
// is downgraded to a `pto.barrier <PIPE_ALL>` (no multi-strategy retry).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H

#include "PTO/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "PTO/Transforms/GraphSyncSolver/SyncSolverIRTranslator.h"
#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {
namespace syncsolver {

class Solver {
public:
  SyncSolverOptions options;
  func::FuncOp funcOp;

  std::unique_ptr<OperationBase> funcIr;
  std::vector<std::unique_ptr<Occurrence>> syncIr;
  llvm::DenseMap<OperationBase *, std::vector<Occurrence *>> opAllOccurrences;
  std::vector<ProcessingOrder> processingOrders;

  // Reused PTO buffer alias map. Owned by IRTranslator originally, moved here.
  Buffer2MemInfoMap buffer2MemInfoMap;

  // Selected sync candidates (set/wait pairs and barriers).
  std::vector<std::unique_ptr<ConflictPair>> chosenConflictedPairs;

  Solver() = delete;
  explicit Solver(std::unique_ptr<IRTranslator> tr);

  // Top-level entry point. Walks `processingOrders`, fills
  // `chosenConflictedPairs`. After this returns the EventIdSolver(s) are
  // already colored.
  void solve();

private:
  MemoryDependentAnalyzer memAnalyzer_;

  llvm::DenseMap<std::pair<PIPE, PIPE>, std::unique_ptr<EventIdSolver>>
      eventIdSolvers_;

  // Conflict pairs already chosen, indexed by the LCA scope occurrence so
  // checkGraphConflict only needs to consider the relevant subset.
  llvm::DenseMap<Occurrence *, llvm::DenseSet<ConflictPair *>>
      scopeOccChosenConflicts_;

  // Memoization of the conflict-pipe enumeration between two RWOperation.
  llvm::DenseMap<std::pair<RWOperation *, RWOperation *>,
                 llvm::SmallVector<std::pair<CorePipeInfo, CorePipeInfo>>>
      memConflictMem_;

  llvm::DenseSet<std::pair<Occurrence *, Occurrence *>> processedOccPairs_;

  // ----- main pipeline -----------------------------------------------------
  void processOrders();
  void processConflict(Occurrence *occ1, Occurrence *occ2, RWOperation *rw1,
                       RWOperation *rw2);
  void handleConflict(Occurrence *occ1, Occurrence *occ2, RWOperation *rw1,
                      RWOperation *rw2, CorePipeInfo src, CorePipeInfo dst);
  void handleBarrierConflict(Occurrence *occ1, Occurrence *occ2,
                             CorePipeInfo src, CorePipeInfo dst);
  void handleSetWaitConflict(Occurrence *occ1, Occurrence *occ2,
                             CorePipeInfo src, CorePipeInfo dst);

  // Conservative memory dependence between two RWOperation; produces
  // (setCorePipeInfo, waitCorePipeInfo) tuples for each detected conflict.
  llvm::SmallVector<std::pair<CorePipeInfo, CorePipeInfo>>
  checkMemoryConflicts(RWOperation *r1, RWOperation *r2);

  // Returns true if a fresh sync between occ1/occ2 is needed, i.e. the
  // GraphSolver-based reachability says the destination pipe is unreachable
  // within the candidate interval given currently chosen pairs.
  bool checkGraphConflict(Occurrence *occ1, Occurrence *occ2, CorePipeInfo src,
                          CorePipeInfo dst);

  // Pre-pruning predicates (mirrors a subset of the upstream Solver).
  bool checkImpossibleOccPair(Occurrence *occ1, Occurrence *occ2);
  bool checkAlreadySynced(Occurrence *occ1, Occurrence *occ2);
  bool checkSkipParallelLoop(Occurrence *occ1, Occurrence *occ2);
  bool checkVisited(Occurrence *occ1, Occurrence *occ2);

  // Move the (set, wait) anchors up so that they share the same parent
  // scope, fixing the various edge-cases (loop boundary, condition
  // boundary, etc.). Minimal subset of the upstream getSetWaitOcc.
  std::pair<Occurrence *, Occurrence *> getSetWaitOcc(Occurrence *occ1,
                                                      Occurrence *occ2);

  // Backward-edge classification. Used by handleSetWaitConflict to lift the
  // anchor pair to the surrounding loop's PlaceHolder gates.
  bool isBackwardSync(Occurrence *occ1, Occurrence *occ2);

  // Candidate -> intersecting already-chosen pairs (for graph coloring).
  std::vector<ConflictPair *>
  getIntersectingConflictPairs(ConflictPair *cp) const;

  // Insert a barrier-all (PIPE_ALL) just before the wait occurrence's op.
  void insertBarrierAllBeforeOcc(Occurrence *occ);

  // EventIdSolver lookup.
  EventIdSolver *getEventIdSolver(PIPE setPipe, PIPE waitPipe);

  // Map a Loop occurrence to its before / after PlaceHolder Occurrences.
  Occurrence *getBeforePlaceHolderOcc(Occurrence *loopOcc);
  Occurrence *getAfterPlaceHolderOcc(Occurrence *loopOcc);
};

} // namespace syncsolver
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_SYNCSOLVER_H
