// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- EventIdSolver.h - Event-id graph coloring ----------------*- C++ -*-===//
//
// Welsh-Powell style coloring of an interference graph whose nodes are
// EventIdNode (one ConflictPair group) and edges are time-overlapping
// conflict pairs. Each insertion is journaled on `actionsStack` so the
// surrounding solver can speculatively try a candidate and roll it back when
// the graph cannot be colored within the hardware budget.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_EVENTIDSOLVER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_EVENTIDSOLVER_H

#include "PTO/Transforms/GraphSyncSolver/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <memory>
#include <stack>
#include <vector>

namespace mlir {
namespace pto {
namespace syncsolver {

enum class ActionType {
  NONE,
  ADD_NODE,
  ADD_EDGE,
  INSERT_CONFLICT_PAIR,
  ASSIGN_EVENT_IDS,
  ASSIGN_NEED_RECALC
};

struct Action {
  ActionType type;
  // Variant payload (only the relevant subset is read per type).
  EventIdNode *node{nullptr};
  EventIdNode *node2{nullptr};
  ConflictPair *conflictPair{nullptr};
  llvm::SmallVector<int64_t> oldEventIds;
  llvm::SmallVector<int64_t> newEventIds;
  bool oldBool{false};
  bool newBool{false};
};

class EventIdSolver {
public:
  explicit EventIdSolver(int64_t eventIdsNumMax)
      : eventIdsNumMax(eventIdsNumMax) {}

  // Coloring success: assigned colors fit into [0, eventIdsNumMax).
  bool isColorable();

  EventIdNode *getNode(ConflictPair *cp);
  EventIdNode *createNode(ConflictPair *cp, int64_t eventIdNum = 1);

  void addConflicts(ConflictPair *src,
                    const std::vector<ConflictPair *> &dsts);

  // Run the Welsh-Powell coloring. Idempotent given current graph state.
  void calcEventIds();

  // Journaling helpers used by the surrounding solver to speculatively try
  // a candidate ConflictPair.
  void pushActionNone();
  void clearActionStack();
  void undoActions();

  // Internally used but exposed for the solver's test/inspect paths.
  void insertConflictPair(EventIdNode *n, ConflictPair *cp);

private:
  // Coloring state.
  int64_t eventIdsNumMax{0};
  bool needRecalculateEventIds{false};
  llvm::SmallVector<std::unique_ptr<EventIdNode>> nodes;
  llvm::DenseMap<EventIdNode *, llvm::DenseMap<EventIdNode *, int64_t>> adjList;
  llvm::DenseMap<EventIdNode *, int64_t> sumAdjListSizes;
  llvm::DenseMap<ConflictPair *, EventIdNode *> conflictPair2Node;
  std::stack<Action> actionsStack;

  int64_t getUsedEventIdsNum(bool dontCalc = false);

  // Mutators (each pushes an Action onto actionsStack so undoActions can
  // reverse them).
  void addNode(std::unique_ptr<EventIdNode> n);
  void addEdge(EventIdNode *a, EventIdNode *b);
  void assignEventIds(EventIdNode *n, llvm::SmallVector<int64_t> eids,
                      bool pushAction = true);
  void assignNeedRecalc(bool newValue, bool pushAction = true);

  // Inverses (used by undoActions; do not push new Actions).
  void removeNode(EventIdNode *n);
  void removeEdge(EventIdNode *a, EventIdNode *b);
  void eraseConflictPair(EventIdNode *n, ConflictPair *cp);

  // Welsh-Powell helpers.
  llvm::SmallVector<int64_t> getAdjUsedEventIds(EventIdNode *n);
  llvm::SmallVector<int64_t> getChosenEventIds(EventIdNode *n);
};

} // namespace syncsolver
} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_GRAPHSYNCSOLVER_EVENTIDSOLVER_H
