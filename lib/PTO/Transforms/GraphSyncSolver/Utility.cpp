// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/GraphSyncSolver/Utility.h"

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include <cassert>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::syncsolver;

int ConflictPair::globalIdCounter = 0;
int EventIdNode::globalIdCounter = 0;

// ---- Occurrence -------------------------------------------------------------

bool Occurrence::sameScope(Occurrence *a, Occurrence *b) {
  assert(a && b && a->parentOcc && b->parentOcc);
  return a->parentOcc == b->parentOcc;
}

int Occurrence::getDepth(Occurrence *occ) {
  int d = 0;
  while (occ != nullptr) {
    occ = occ->parentOcc;
    ++d;
  }
  return d;
}

Occurrence *Occurrence::getNthParent(int dist) {
  Occurrence *cur = this;
  while (dist--) {
    assert(cur);
    cur = cur->parentOcc;
  }
  assert(cur);
  return cur;
}

Occurrence *Occurrence::getParentWithOp(OperationBase *op, bool assertExists) {
  assert(op);
  Occurrence *cur = this;
  while (cur != nullptr) {
    if (cur->op == op)
      return cur;
    cur = cur->parentOcc;
  }
  assert(!assertExists);
  return nullptr;
}

bool Occurrence::isProperAncestor(Occurrence *occ) {
  assert(occ);
  int d1 = getDepth(this);
  int d2 = getDepth(occ);
  if (d1 >= d2)
    return false;
  return occ->getNthParent(d2 - d1) == this;
}

llvm::SmallVector<Occurrence *> Occurrence::getAllParents() {
  llvm::SmallVector<Occurrence *> parents;
  Occurrence *cur = this->parentOcc;
  while (cur != nullptr) {
    parents.push_back(cur);
    cur = cur->parentOcc;
  }
  return parents;
}

std::pair<Occurrence *, Occurrence *>
Occurrence::getLCAPair(Occurrence *a, Occurrence *b) {
  assert(a && b);
  int da = getDepth(a);
  int db = getDepth(b);
  if (da < db)
    b = b->getNthParent(db - da);
  else if (da > db)
    a = a->getNthParent(da - db);
  while (a->parentOcc != b->parentOcc) {
    a = a->parentOcc;
    b = b->parentOcc;
  }
  assert(a != b);
  return {a, b};
}

Occurrence *Occurrence::getParentLoop(Occurrence *occ) {
  assert(occ);
  Occurrence *cur = occ->parentOcc;
  while (cur != nullptr && !llvm::isa<Loop>(cur->op))
    cur = cur->parentOcc;
  return cur;
}

// ---- range helpers ----------------------------------------------------------

bool mlir::pto::syncsolver::checkRangesIntersect(int l1, int r1, int l2,
                                                 int r2) {
  return r1 > l2 && r2 > l1;
}
