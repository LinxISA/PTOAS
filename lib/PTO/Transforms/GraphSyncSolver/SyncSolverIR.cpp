// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/GraphSyncSolver/SyncSolverIR.h"
#include <cassert>

using namespace mlir;
using namespace mlir::pto::syncsolver;

int OperationBase::globalIndex = 0;

bool OperationBase::sameScope(OperationBase *a, OperationBase *b) {
  assert(a && b && a->parentOp && b->parentOp);
  return a->parentOp == b->parentOp;
}

int OperationBase::getDepth() const {
  int d = 0;
  const OperationBase *cur = this;
  while (cur != nullptr) {
    cur = cur->parentOp;
    ++d;
  }
  return d;
}

OperationBase *OperationBase::getNthParent(int dist) {
  OperationBase *cur = this;
  while (dist--) {
    assert(cur);
    cur = cur->parentOp;
  }
  return cur;
}

std::pair<OperationBase *, OperationBase *>
OperationBase::getLCAPair(OperationBase *a, OperationBase *b) {
  assert(a && b);
  int da = a->getDepth();
  int db = b->getDepth();
  if (da < db)
    b = b->getNthParent(db - da);
  else if (da > db)
    a = a->getNthParent(da - db);
  while (a->parentOp != b->parentOp) {
    a = a->parentOp;
    b = b->parentOp;
  }
  assert(a && b && a->parentOp == b->parentOp);
  return {a, b};
}

OperationBase *OperationBase::getParentLoop(OperationBase *op) {
  assert(op);
  OperationBase *cur = op->parentOp;
  while (cur != nullptr && !llvm::isa<Loop>(cur))
    cur = cur->parentOp;
  return cur;
}

bool OperationBase::isProperAncestor(OperationBase *op) {
  assert(op);
  int da = this->getDepth();
  int db = op->getDepth();
  if (da >= db)
    return false;
  return op->getNthParent(db - da) == this;
}

llvm::SmallVector<OperationBase *> OperationBase::getAllParents() {
  llvm::SmallVector<OperationBase *> parents;
  OperationBase *cur = this->parentOp;
  while (cur != nullptr) {
    parents.push_back(cur);
    cur = cur->parentOp;
  }
  return parents;
}
