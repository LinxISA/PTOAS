// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/GraphSyncSolver/GraphSolver.h"
#include <cassert>
#include <queue>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

bool GraphSolver::Edge::operator<(const Edge &o) const {
  // The set is per-(src, dst) bucket so endpoints must match by construction.
  assert(corePipeSrc == o.corePipeSrc);
  assert(corePipeDst == o.corePipeDst);
  if (startIndex != o.startIndex)
    return startIndex < o.startIndex;
  return endIndex < o.endIndex;
}

void GraphSolver::addPair(ConflictPair *cp, CorePipeInfo src, CorePipeInfo dst,
                          int sIdx, int eIdx) {
  Edge edge(cp, src, dst, sIdx, eIdx);
  adjacencyList[src][dst].insert(edge);
}

void GraphSolver::addConflictPair(ConflictPair *cp) {
  assert(cp);
  // Minimal port: no PIPE_ALL fanout, no PIPE_S broadcast, no unit-flag.
  addPair(cp, cp->setCorePipeInfo, cp->waitCorePipeInfo, cp->startIndex,
          cp->endIndex);
}

std::optional<int> GraphSolver::runDijkstra(CorePipeInfo corePipeSrc,
                                            CorePipeInfo corePipeDst,
                                            int startIndex, int endIndex) {
  llvm::DenseMap<CorePipeInfo, int> distance;
  std::priority_queue<std::pair<int, CorePipeInfo>,
                      std::vector<std::pair<int, CorePipeInfo>>,
                      std::greater<std::pair<int, CorePipeInfo>>>
      que;
  que.emplace(startIndex, corePipeSrc);
  auto [coreDst, pipeDst] = corePipeDst;

  while (!que.empty()) {
    auto [curIndex, curCorePipe] = que.top();
    que.pop();

    if (curCorePipe == corePipeDst && distance.count(corePipeDst))
      break;
    if (distance.count(curCorePipe) && distance[curCorePipe] < curIndex)
      continue;
    if (distance.count(curCorePipe) && distance[curCorePipe] > endIndex)
      break;

    auto [curCore, curPipe] = curCorePipe;
    // PIPE_S / PIPE_ALL act as soft "any pipe" sinks on the destination core
    // (mirrors the upstream HIVM semantics): once we reach them, we can
    // claim arrival at corePipeDst at the same index.
    if (curCore == coreDst &&
        ((curIndex != startIndex && curPipe == PIPE::PIPE_S) ||
         curPipe == PIPE::PIPE_ALL)) {
      distance[corePipeDst] = curIndex;
      break;
    }

    for (auto &[endCorePipe, edges] : adjacencyList[curCorePipe]) {
      auto it = edges.lower_bound(Edge(nullptr, curCorePipe, endCorePipe,
                                       curIndex, /*eIdx=*/-1));
      for (; it != edges.end(); ++it) {
        if (!distance.count(endCorePipe) ||
            distance[endCorePipe] > it->endIndex) {
          distance[endCorePipe] = it->endIndex;
          que.emplace(it->endIndex, endCorePipe);
        }
      }
    }
  }

  if (distance.count(corePipeDst))
    return distance[corePipeDst];
  return std::nullopt;
}
