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

// (loop, factor N) -> shared `((iv - lb) / step) mod N` counter inside that
// loop. B5: when several multi-buffer pointer_casts share the same enclosing
// scf.for and the same N, they should all read from the same counter rather
// than each inserting its own arith.remui + N constant ops. This mirrors
// SyncCodegen::loop2BufferCounter so the two passes emit consistent counters.
//
// P0-2 (slot index): The slot must be the LOGICAL iteration index modulo N,
// not the physical induction variable modulo N. When `step != 1` or `lb != 0`,
// `iv mod N` skips slots whenever gcd(step, N) > 1 (e.g. step=2,N=4 only ever
// produces 0,2). Compute `((iv - lb) / step) mod N` explicitly; degenerate to
// the cheaper `iv mod N` when `lb == 0` and `step == 1`.
using LoopFactorKey = std::pair<scf::ForOp, unsigned>;

static bool isConstantIndexEqualTo(Value v, int64_t target) {
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

static Value getOrCreateLoopCounter(
    IRRewriter &rewriter,
    llvm::DenseMap<LoopFactorKey, Value> &cache,
    scf::ForOp forOp, unsigned n, Location loc) {
  auto key = std::make_pair(forOp, n);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value cN = rewriter.create<arith::ConstantIndexOp>(loc, n);
  Value lb = forOp.getLowerBound();
  Value step = forOp.getStep();
  Value normalized = iv;
  if (!isConstantIndexEqualTo(lb, 0))
    normalized = rewriter.create<arith::SubIOp>(loc, normalized, lb);
  if (!isConstantIndexEqualTo(step, 1))
    normalized = rewriter.create<arith::DivUIOp>(loc, normalized, step);
  Value rem = rewriter.create<arith::RemUIOp>(loc, normalized, cN);
  cache[key] = rem;
  return rem;
}

static LogicalResult lowerMultiBufferPointerCast(
    IRRewriter &rewriter, PointerCastOp op, scf::ForOp forOp,
    llvm::DenseMap<LoopFactorKey, Value> &counterCache) {
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

  Value rem = getOrCreateLoopCounter(rewriter, counterCache, forOp, n, loc);
  // Insertion point for the select chain: right after the cached counter.
  // (For a freshly created counter this lands at the same position as before;
  // for cached ones it lands wherever that earlier remui sits, which is still
  // at loop-body start so the chain dominates all uses inside the loop.)
  rewriter.setInsertionPointAfter(rem.getDefiningOp());

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

// CV-create-preload hoists per-stage rotated pointer_casts ABOVE the rotation
// loop, so a multi-address pointer_cast may have NO enclosing scf.for of its
// own even though its users still rotate over an enclosing loop. When the cast
// itself is loop-invariant, infer the rotation loop from its users: every use
// must live inside the SAME scf.for, otherwise a single `iv mod N` selector
// can't be valid. Returns nullptr to mean "skip with a warning".
static scf::ForOp inferRotationLoopFromUses(PointerCastOp op) {
  scf::ForOp common;
  for (Operation *user : op->getUsers()) {
    auto userFor = user->getParentOfType<scf::ForOp>();
    if (!userFor)
      return nullptr;
    if (!common)
      common = userFor;
    else if (common != userFor)
      return nullptr;
  }
  return common;
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
    // Per-(loop, factor) counter cache so multiple multi-buffer pointer_casts
    // sharing a loop and N reuse one `iv mod N` (B5).
    llvm::DenseMap<LoopFactorKey, Value> counterCache;
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
        // Hoisted-by-CVCreatePreload case: the rotated multi-address cast lives
        // at function level, but its users still rotate over an inner scf.for.
        // Infer that loop and reuse the regular lowering path; if uses span
        // multiple loops (or none), fall through to the warning + skip.
        forOp = inferRotationLoopFromUses(op);
      }
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

      if (failed(lowerMultiBufferPointerCast(rewriter, op, forOp,
                                             counterCache))) {
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
