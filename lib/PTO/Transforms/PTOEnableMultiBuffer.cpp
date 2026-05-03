// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOENABLEMULTIBUFFER
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static LogicalResult lowerMultiBufferPointerCast(IRRewriter &rewriter,
                                                 PointerCastOp op,
                                                 scf::ForOp forOp) {
  ValueRange addrs = op.getAddrs();
  unsigned n = static_cast<unsigned>(addrs.size());
  assert(n >= 2);

  Location loc = op.getLoc();
  MemRefType resTy = op.getType();
  Value validRow = op.getValidRow();
  Value validCol = op.getValidCol();
  std::optional<TileBufConfigAttr> config = op.getConfig();

  rewriter.setInsertionPoint(forOp);
  SmallVector<Value> slotBufs;
  slotBufs.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    auto oneAddr = addrs.slice(i, 1);
    PointerCastOp slot = rewriter.create<PointerCastOp>(
        loc, resTy, oneAddr, validRow, validCol,
        config.has_value()
            ? static_cast<Attribute>(*config)
            : Attribute());
    slotBufs.push_back(slot.getResult());
  }

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value cN = rewriter.create<arith::ConstantIndexOp>(loc, n);
  Value rem = rewriter.create<arith::RemUIOp>(loc, iv, cN);

  Value selected = slotBufs[0];
  for (unsigned i = 1; i < n; ++i) {
    Value ci = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value eq = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rem, ci);
    selected =
        rewriter.create<arith::SelectOp>(loc, eq, slotBufs[i], selected);
  }

  rewriter.replaceOp(op, selected);
  return success();
}

// Multi-buffer is a local-memory optimisation. GM pointer casts may also be
// multi-address (e.g., a future workspace path) but the iv-mod-N selector is
// not meaningful for GM, so we skip them here.
static bool isLocalScopePointerCast(PointerCastOp op) {
  auto memrefTy = dyn_cast<MemRefType>(op.getType());
  if (!memrefTy)
    return false;
  auto attr = memrefTy.getMemorySpace();
  if (!attr)
    return false;
  auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(attr);
  if (!ptoAttr)
    return false;
  AddressSpace as = ptoAttr.getAddressSpace();
  return as == AddressSpace::VEC || as == AddressSpace::MAT;
}

struct PTOEnableMultiBufferPass
    : public mlir::pto::impl::PTOEnableMultiBufferBase<
          PTOEnableMultiBufferPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<PointerCastOp> work;
    func.walk([&](PointerCastOp op) {
      if (op.getAddrs().size() > 1)
        work.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (PointerCastOp op : work) {
      // D2: scope guard. Multi-buffer slot selection only makes sense for
      // local memory (VEC/MAT). Multi-address casts in GM (e.g., reserved
      // workspaces) must keep their original semantics.
      if (!isLocalScopePointerCast(op)) {
        op.emitWarning() << "pto-enable-multi-buffer: skipping non-local "
                            "pointer_cast (multi-buffer is VEC/MAT-only)";
        continue;
      }

      auto forOp = op->getParentOfType<scf::ForOp>();
      if (!forOp) {
        op.emitWarning()
            << "pto-enable-multi-buffer: expected enclosing scf.for; skipping";
        continue;
      }

      // D1: loop-invariance guard. The pass hoists each addr operand and the
      // resulting single-address pto.pointer_cast above `forOp`. SSA dominance
      // requires every addr to be defined outside the loop. Today PlanMemory
      // emits constant i64 offsets so this always holds, but a future
      // dynamic-address path (e.g., workspace double-buffer) would silently
      // violate dominance without this check.
      bool addrsAreLoopInvariant = true;
      for (Value addr : op.getAddrs()) {
        if (!forOp.isDefinedOutsideOfLoop(addr)) {
          addrsAreLoopInvariant = false;
          break;
        }
      }
      if (!addrsAreLoopInvariant) {
        op.emitWarning() << "pto-enable-multi-buffer: addr operand is not "
                            "loop-invariant; skipping (would break SSA on "
                            "hoist)";
        continue;
      }

      if (failed(lowerMultiBufferPointerCast(rewriter, op, forOp))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOEnableMultiBufferPass() {
  return std::make_unique<PTOEnableMultiBufferPass>();
}
