// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/GraphSyncSolver/EventIdSolver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <set>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

// ---- private mutators -------------------------------------------------------

void EventIdSolver::addNode(std::unique_ptr<EventIdNode> n) {
  Action a;
  a.type = ActionType::ADD_NODE;
  a.node = n.get();
  actionsStack.push(std::move(a));
  nodes.push_back(std::move(n));
}

void EventIdSolver::removeNode(EventIdNode *n) {
  // Only valid when n has no remaining adjacency entries.
  assert(adjList[n].empty());
  assert(!sumAdjListSizes[n]);
  adjList.erase(n);
  sumAdjListSizes.erase(n);
  assert(!nodes.empty() && nodes.back().get() == n);
  nodes.pop_back();
}

void EventIdSolver::insertConflictPair(EventIdNode *n, ConflictPair *cp) {
  assert(n && cp);
  Action a;
  a.type = ActionType::INSERT_CONFLICT_PAIR;
  a.node = n;
  a.conflictPair = cp;
  actionsStack.push(std::move(a));
  conflictPair2Node[cp] = n;
  n->insertConflictPair(cp);
}

void EventIdSolver::eraseConflictPair(EventIdNode *n, ConflictPair *cp) {
  assert(n && cp);
  conflictPair2Node.erase(cp);
  n->eraseConflictPair(cp);
}

void EventIdSolver::addEdge(EventIdNode *a, EventIdNode *b) {
  assert(a && b);
  if (a == b)
    return;
  Action act;
  act.type = ActionType::ADD_EDGE;
  act.node = a;
  act.node2 = b;
  actionsStack.push(std::move(act));
  if (!adjList[a][b]++)
    sumAdjListSizes[a] += b->eventIdNum;
  if (!adjList[b][a]++)
    sumAdjListSizes[b] += a->eventIdNum;
  if (!needRecalculateEventIds) {
    if (sumAdjListSizes[a] + a->eventIdNum > eventIdsNumMax ||
        sumAdjListSizes[b] + b->eventIdNum > eventIdsNumMax) {
      assignNeedRecalc(true);
    }
  }
}

void EventIdSolver::removeEdge(EventIdNode *a, EventIdNode *b) {
  assert(a && b);
  if (--adjList[a][b] == 0) {
    adjList[a].erase(b);
    sumAdjListSizes[a] -= b->eventIdNum;
  }
  if (--adjList[b][a] == 0) {
    adjList[b].erase(a);
    sumAdjListSizes[b] -= a->eventIdNum;
  }
}

void EventIdSolver::assignEventIds(EventIdNode *n,
                                   llvm::SmallVector<int64_t> eids,
                                   bool pushAction) {
  assert(n);
  if (n->getEventIds() == eids)
    return;
  if (pushAction) {
    Action a;
    a.type = ActionType::ASSIGN_EVENT_IDS;
    a.node = n;
    a.oldEventIds = n->getEventIds();
    a.newEventIds = eids;
    actionsStack.push(std::move(a));
  }
  n->setEventIds(std::move(eids));
}

void EventIdSolver::assignNeedRecalc(bool newValue, bool pushAction) {
  if (newValue == needRecalculateEventIds)
    return;
  if (pushAction) {
    Action a;
    a.type = ActionType::ASSIGN_NEED_RECALC;
    a.oldBool = needRecalculateEventIds;
    a.newBool = newValue;
    actionsStack.push(std::move(a));
  }
  needRecalculateEventIds = newValue;
}

EventIdNode *EventIdSolver::getNode(ConflictPair *cp) {
  assert(cp);
  auto it = conflictPair2Node.find(cp);
  assert(it != conflictPair2Node.end());
  return it->second;
}

EventIdNode *EventIdSolver::createNode(ConflictPair *cp, int64_t eventIdNum) {
  assert(cp);
  assert(eventIdNum > 0);
  auto n = std::make_unique<EventIdNode>(cp, eventIdNum);
  auto *raw = n.get();
  addNode(std::move(n));
  insertConflictPair(raw, cp);
  // Pre-seed colors as [0..eventIdNum) so isColorable does the right thing
  // before calcEventIds is called (matches the upstream behavior).
  llvm::SmallVector<int64_t> initial(eventIdNum, 0);
  std::iota(initial.begin(), initial.end(), 0);
  assignEventIds(raw, std::move(initial));
  return raw;
}

void EventIdSolver::addConflicts(
    ConflictPair *src, const std::vector<ConflictPair *> &dsts) {
  assert(src);
  EventIdNode *a = getNode(src);
  for (auto *d : dsts) {
    EventIdNode *b = getNode(d);
    addEdge(a, b);
  }
}

// ---- coloring ---------------------------------------------------------------

llvm::SmallVector<int64_t>
EventIdSolver::getAdjUsedEventIds(EventIdNode *n) {
  llvm::SmallDenseSet<int64_t> used;
  for (auto [other, frq] : adjList[n]) {
    (void)frq;
    auto &ids = other->getEventIds();
    used.insert(ids.begin(), ids.end());
  }
  llvm::SmallVector<int64_t> v(used.begin(), used.end());
  llvm::sort(v);
  return v;
}

llvm::SmallVector<int64_t>
EventIdSolver::getChosenEventIds(EventIdNode *n) {
  llvm::SmallVector<int64_t> chosen;
  llvm::SmallVector<int64_t> used = getAdjUsedEventIds(n);
  int64_t cur = 0;
  auto *it = used.begin();
  while ((int64_t)chosen.size() < n->eventIdNum) {
    while (it != used.end() && *it < cur)
      ++it;
    if (it != used.end() && *it == cur)
      ++it;
    else
      chosen.push_back(cur);
    ++cur;
  }
  llvm::sort(chosen);
  return chosen;
}

void EventIdSolver::calcEventIds() {
  // Smallest-degree-first ordering, then assign colors in reverse.
  auto cmp = [](const std::pair<int64_t, EventIdNode *> &a,
                const std::pair<int64_t, EventIdNode *> &b) {
    if (a.first != b.first)
      return a.first < b.first;
    return a.second->id < b.second->id;
  };
  std::set<std::pair<int64_t, EventIdNode *>, decltype(cmp)> st(cmp);
  llvm::DenseMap<EventIdNode *, int64_t> nodeValue;

  for (auto &n : nodes) {
    assignEventIds(n.get(), {});
    int64_t v = sumAdjListSizes[n.get()] + n->eventIdNum;
    nodeValue[n.get()] = v;
    st.emplace(v, n.get());
  }

  llvm::SmallVector<EventIdNode *> ordered;
  while (!st.empty()) {
    auto *node = st.begin()->second;
    st.erase(st.begin());
    nodeValue.erase(node);
    ordered.push_back(node);
    for (auto [adj, frq] : adjList[node]) {
      (void)frq;
      if (nodeValue.contains(adj)) {
        st.erase({nodeValue[adj], adj});
        nodeValue[adj] -= node->eventIdNum;
        st.insert({nodeValue[adj], adj});
      }
    }
  }
  for (auto *n : llvm::reverse(ordered)) {
    auto eids = getChosenEventIds(n);
    assert(!eids.empty());
    assignEventIds(n, eids);
  }
  assignNeedRecalc(false);
}

int64_t EventIdSolver::getUsedEventIdsNum(bool dontCalc) {
  if (!dontCalc)
    calcEventIds();
  assert(!needRecalculateEventIds);
  llvm::SmallDenseSet<int64_t> used;
  for (auto &n : nodes) {
    auto &eids = n->getEventIds();
    if (eids.empty())
      continue;
    used.insert(eids.begin(), eids.end());
  }
  return static_cast<int64_t>(used.size());
}

bool EventIdSolver::isColorable() {
  if (needRecalculateEventIds)
    calcEventIds();
  return getUsedEventIdsNum(/*dontCalc=*/true) <= eventIdsNumMax;
}

// ---- journaling -------------------------------------------------------------

void EventIdSolver::pushActionNone() {
  Action a;
  a.type = ActionType::NONE;
  actionsStack.push(std::move(a));
}

void EventIdSolver::clearActionStack() {
  while (!actionsStack.empty())
    actionsStack.pop();
}

void EventIdSolver::undoActions() {
  while (!actionsStack.empty() && actionsStack.top().type != ActionType::NONE) {
    Action a = std::move(actionsStack.top());
    actionsStack.pop();
    switch (a.type) {
    case ActionType::ADD_NODE:
      removeNode(a.node);
      break;
    case ActionType::ADD_EDGE:
      removeEdge(a.node, a.node2);
      break;
    case ActionType::INSERT_CONFLICT_PAIR:
      eraseConflictPair(a.node, a.conflictPair);
      break;
    case ActionType::ASSIGN_EVENT_IDS:
      assignEventIds(a.node, a.oldEventIds, /*pushAction=*/false);
      break;
    case ActionType::ASSIGN_NEED_RECALC:
      assignNeedRecalc(a.oldBool, /*pushAction=*/false);
      break;
    default:
      break;
    }
  }
  if (!actionsStack.empty()) {
    assert(actionsStack.top().type == ActionType::NONE);
    actionsStack.pop();
  }
}
