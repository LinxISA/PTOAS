// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PlanMemory.cpp ----Plan Buffer Memory Address ----------------------===//
//===----------------------------------------------------------------------===//

#include "PTOPlanMemory.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/DialectConversion.h"
#include "AllocToPointerCast.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#define DEBUG_TYPE "pto-plan-memory"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X)

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PLANMEMORY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace pto;

namespace {

constexpr int64_t kBitsPerByte = 8;
constexpr unsigned kI32BitWidth = 32;
constexpr unsigned kMemoryEffectReserveSize = 8;
constexpr int kSingleBufferCount = 1;
constexpr int kDoubleBufferCount = 2;
constexpr int64_t kA5VecLocalMemBits = 2031616;
constexpr int64_t kA3VecLocalMemBits = 1572864;
constexpr int64_t kMatLocalMemBits = 4194304;
constexpr int64_t kLocalMemAlignmentBytes = 256;

struct LocalMemSpec {
  int64_t capacityBits = 0;
  int64_t alignBytes = 1;
};

static int64_t ceilDivBitsToBytes(int64_t bits) {
  return (bits + kBitsPerByte - 1) / kBitsPerByte;
}

static int64_t alignUpBytes(int64_t value, int64_t align) {
  int64_t safeAlign = std::max<int64_t>(align, 1);
  if (safeAlign == 1)
    return value;
  int64_t rem = value % safeAlign;
  if (rem == 0)
    return value;
  return value + (safeAlign - rem);
}

static LocalMemSpec getLocalMemSpec(Operation *op, AddressSpace as) {
  switch (as) {
  case AddressSpace::VEC:
    return isTargetArchA5(op)
               ? LocalMemSpec{kA5VecLocalMemBits, kLocalMemAlignmentBytes}
               : LocalMemSpec{kA3VecLocalMemBits, kLocalMemAlignmentBytes};
  case AddressSpace::MAT:
    return LocalMemSpec{kMatLocalMemBits, kLocalMemAlignmentBytes};
  default:
    return LocalMemSpec{};
  }
}

static void collectStableValueOrder(Region &region, AsmState &asmState,
                                    DenseMap<Value, std::string> &stableValueKeys,
                                    SmallVectorImpl<Value> &seenValues) {
  auto recordValue = [&](Value value) {
    if (stableValueKeys.find(value) != stableValueKeys.end())
      return;
    std::string key;
    llvm::raw_string_ostream os(key);
    value.printAsOperand(os, asmState);
    stableValueKeys[value] = os.str();
    seenValues.push_back(value);
  };

  for (Block &block : region) {
    for (BlockArgument blockArg : block.getArguments())
      recordValue(blockArg);
    for (Operation &op : block) {
      for (Value result : op.getResults())
        recordValue(result);
      for (Region &nestedRegion : op.getRegions())
        collectStableValueOrder(nestedRegion, asmState, stableValueKeys,
                                seenValues);
    }
  }
}

static StableValueOrderMap buildStableValueOrder(func::FuncOp func) {
  DenseMap<Value, std::string> stableValueKeys;
  SmallVector<Value> seenValues;
  AsmState asmState(func);
  collectStableValueOrder(func.getBody(), asmState, stableValueKeys, seenValues);

  llvm::sort(seenValues, [&](Value lhs, Value rhs) {
    const std::string &lhsKey = stableValueKeys.find(lhs)->second;
    const std::string &rhsKey = stableValueKeys.find(rhs)->second;
    if (lhsKey != rhsKey)
      return lhsKey < rhsKey;
    return isLessValue(lhs, rhs);
  });

  StableValueOrderMap stableValueOrder;
  for (auto [index, value] : llvm::enumerate(seenValues))
    stableValueOrder[value] = index;
  return stableValueOrder;
}

static uint32_t lookupStableValueOrder(
    Value value, const StableValueOrderMap &stableValueOrder) {
  auto it = stableValueOrder.find(value);
  if (it != stableValueOrder.end())
    return it->second;
  return std::numeric_limits<uint32_t>::max();
}

static void sortValuesByStableOrder(
    SmallVectorImpl<Value> &values,
    const StableValueOrderMap &stableValueOrder) {
  llvm::sort(values, [&](Value lhs, Value rhs) {
    uint32_t lhsOrder = lookupStableValueOrder(lhs, stableValueOrder);
    uint32_t rhsOrder = lookupStableValueOrder(rhs, stableValueOrder);
    if (lhsOrder != rhsOrder)
      return lhsOrder < rhsOrder;
    return isLessValue(lhs, rhs);
  });
}

static SmallVector<Value> getScratchBuffersFromEffects(Operation *op,
                                                       ValueRange dpsInits,
                                                       const StableValueOrderMap &stableValueOrder) {
  SmallVector<Value> scratchBuffers;
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect)
    return scratchBuffers;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
              kMemoryEffectReserveSize>
      effects;
  memEffect.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    Value value = effect.getValue();
    if (!value)
      continue;
    if (!llvm::is_contained(op->getOperands(), value))
      continue;
    if (llvm::is_contained(dpsInits, value))
      continue;
    if (!llvm::is_contained(scratchBuffers, value))
      scratchBuffers.push_back(value);
  }
  sortValuesByStableOrder(scratchBuffers, stableValueOrder);
  return scratchBuffers;
}

static SmallVector<ValuePair>
getScratchConflictPairsFromEffects(Operation *op, ValueRange dpsInits,
                                   const StableValueOrderMap &stableValueOrder) {
  SmallVector<ValuePair> conflictPairs;
  SmallVector<Value> scratchBuffers =
      getScratchBuffersFromEffects(op, dpsInits, stableValueOrder);
  for (Value scratch : scratchBuffers) {
    for (Value dst : dpsInits) {
      if (!scratch || !dst || scratch == dst)
        continue;
      conflictPairs.emplace_back(scratch, dst);
    }
  }
  return conflictPairs;
}

enum class ReserveBufferMode {
  None,
  Auto,
  Manual,
};

struct ReserveBufferPlan {
  ReserveBufferMode mode = ReserveBufferMode::None;
  ReserveBufferOp reserveOp;
  AddressSpace addressSpace = AddressSpace::Zero;
  int64_t sizeBytes = 0;
  int64_t capacityBytes = 0;
  int64_t alignBytes = 1;
};

using ReserveBufferPlans = SmallVector<ReserveBufferPlan>;

static LogicalResult analyzeReserveBufferPlans(func::FuncOp funcOp,
                                               ReserveBufferPlans &plans) {
  SmallVector<ReserveBufferOp> reserveOps;
  funcOp.walk(
      [&](ReserveBufferOp reserveOp) { reserveOps.push_back(reserveOp); });

  if (reserveOps.empty())
    return success();

  for (ReserveBufferOp reserveOp : reserveOps) {
    AddressSpace as = reserveOp.getLocation().getAddressSpace();
    auto spec = getLocalMemSpec(reserveOp.getOperation(), as);
    if (spec.capacityBits <= 0 || spec.alignBytes <= 0)
      return reserveOp.emitOpError("unsupported reserve_buffer location");

    int64_t capacityBytes = spec.capacityBits / kBitsPerByte;
    int64_t sizeBytes = reserveOp.getSize();
    bool autoAlloc = reserveOp.getAutoAlloc();

    ReserveBufferPlan &plan = plans.emplace_back();
    plan.mode = autoAlloc ? ReserveBufferMode::Auto : ReserveBufferMode::Manual;
    plan.reserveOp = reserveOp;
    plan.addressSpace = as;
    plan.sizeBytes = sizeBytes;
    plan.capacityBytes = capacityBytes;
    plan.alignBytes = spec.alignBytes;

    // Auto mode only declares that one contiguous region must be reserved.
    // The concrete base is filled later from a hole in the target local space.
    if (autoAlloc) {
      if (reserveOp.getBaseAttr()) {
        return reserveOp.emitOpError(
            "expects 'base' to be absent when 'auto' is true");
      }
      continue;
    }

    // In manual mode, reserve_buffer.base is already fixed by the frontend or
    // an earlier stage. Only basic validation is needed here.
    auto baseAttr = reserveOp.getBaseAttr();
    if (!baseAttr)
      return reserveOp.emitOpError("expects 'base' when 'auto' is false");

    int64_t baseBytes = baseAttr.getInt();
    if (baseBytes % spec.alignBytes != 0) {
      return reserveOp.emitOpError(
          "expects 'base' to satisfy the address-space alignment");
    }
    if (baseBytes + sizeBytes > capacityBytes) {
      return reserveOp.emitOpError("exceeds available local memory capacity");
    }
  }

  return success();
}

struct OccupiedByteRange {
  int64_t begin = 0;
  int64_t end = 0;
};

static LogicalResult assignAutoReserveBufferBases(
    ReserveBufferPlans &plans,
    const std::map<Value, BufferInfo, ValueComparator> &bufferInfos,
    const DenseMap<Value, SmallVector<uint64_t>> &buffer2Offsets) {
  std::map<AddressSpace, SmallVector<OccupiedByteRange>> occupiedByAddressSpace;
  for (const auto &it : bufferInfos) {
    Value buffer = it.first;
    const BufferInfo &bufferInfo = it.second;

    auto offsetsIt = buffer2Offsets.find(buffer);
    if (offsetsIt == buffer2Offsets.end())
      continue;

    // Reserve-buffer allocation intentionally happens after normal MemPlan.
    // Reconstruct the already occupied byte ranges from the planned local
    // buffers, then place reserve_buffer into the first aligned hole.
    int64_t occupiedSizeBytes =
        alignUpBytes(ceilDivBitsToBytes(bufferInfo.constBits), /*align=*/1);
    for (uint64_t offsetBytes : offsetsIt->second) {
      occupiedByAddressSpace[bufferInfo.bufferScope].push_back(
          OccupiedByteRange{static_cast<int64_t>(offsetBytes),
                            static_cast<int64_t>(offsetBytes) + occupiedSizeBytes});
    }
  }

  auto normalizeRanges = [](SmallVector<OccupiedByteRange> &ranges) {
    llvm::sort(ranges,
               [](const OccupiedByteRange &lhs, const OccupiedByteRange &rhs) {
                 return lhs.begin < rhs.begin;
               });

    SmallVector<OccupiedByteRange> merged;
    for (const OccupiedByteRange &range : ranges) {
      if (merged.empty() || range.begin > merged.back().end) {
        merged.push_back(range);
        continue;
      }
      merged.back().end = std::max(merged.back().end, range.end);
    }
    ranges.swap(merged);
  };

  for (auto &it : occupiedByAddressSpace)
    normalizeRanges(it.second);

  for (ReserveBufferPlan &plan : plans) {
    if (plan.mode != ReserveBufferMode::Auto || !plan.reserveOp)
      continue;

    SmallVector<OccupiedByteRange> &occupied =
        occupiedByAddressSpace[plan.addressSpace];

    // First-fit search: try address 0 first, then keep moving the candidate to
    // the end of the current occupied interval until a large-enough aligned
    // hole is found.
    int64_t candidateBase = 0;
    for (const OccupiedByteRange &range : occupied) {
      candidateBase = alignUpBytes(candidateBase, plan.alignBytes);
      if (candidateBase + plan.sizeBytes <= range.begin)
        break;
      candidateBase = std::max(candidateBase, range.end);
    }
    candidateBase = alignUpBytes(candidateBase, plan.alignBytes);

    if (candidateBase + plan.sizeBytes > plan.capacityBytes) {
      return plan.reserveOp.emitOpError(
          "failed to allocate local memory hole for reserve_buffer");
    }

    plan.reserveOp->setAttr(
        "base",
        IntegerAttr::get(
            IntegerType::get(plan.reserveOp.getContext(), kI32BitWidth),
            candidateBase));
    occupied.push_back(
        OccupiedByteRange{candidateBase, candidateBase + plan.sizeBytes});
    normalizeRanges(occupied);
  }
  return success();
}

} // namespace
