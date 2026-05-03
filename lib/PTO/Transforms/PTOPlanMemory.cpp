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

#include "PTO/Transforms/MultiBuffer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/DialectConversion.h"
#include "AllocToPointerCast.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallPtrSet.h"
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

static void collectStableValueOrder(Region &region,
                                    AsmState &asmState,
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

static void appendUniqueValue(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value))
    values.push_back(value);
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

static bool isReserveBufferAddr(Value value, ReserveBufferOp reserveOp) {
  return value && value == reserveOp.getAddr();
}

static LogicalResult computeFifoLocalBufferSizeBytes(Operation *op,
                                                     int64_t slotSizeBytes,
                                                     IntegerAttr localSlotNumAttr,
                                                     int64_t &sizeBytes) {
  if (!localSlotNumAttr)
    return success();

  int64_t localSlotNum = localSlotNumAttr.getInt();
  if (slotSizeBytes <= 0)
    return op->emitOpError("expects FIFO slot_size to be positive");
  if (localSlotNum <= 0)
    return op->emitOpError("expects FIFO local_slot_num to be positive");
  if (slotSizeBytes > std::numeric_limits<int64_t>::max() / localSlotNum)
    return op->emitOpError("FIFO local buffer size overflows int64_t");

  sizeBytes = slotSizeBytes * localSlotNum;
  if (sizeBytes > std::numeric_limits<int32_t>::max())
    return op->emitOpError(
        "FIFO local buffer size exceeds reserve_buffer size attribute range");
  return success();
}

static FailureOr<int64_t>
computeAutoReserveBufferSizeBytes(func::FuncOp funcOp,
                                  ReserveBufferOp reserveOp) {
  std::optional<int64_t> fifoLocalSizeBytes;
  bool failedToCompute = false;

  auto updateFromFifo = [&](Operation *op, int64_t slotSizeBytes,
                            IntegerAttr localSlotNumAttr) -> LogicalResult {
    if (!localSlotNumAttr)
      return success();

    int64_t currentSizeBytes = 0;
    if (failed(computeFifoLocalBufferSizeBytes(
            op, slotSizeBytes, localSlotNumAttr, currentSizeBytes)))
      return failure();

    // One reserve_buffer normally feeds one FIFO. If the IR shares it across
    // multiple pipe init ops, reserve enough for the largest local buffer.
    fifoLocalSizeBytes =
        fifoLocalSizeBytes ? std::max(*fifoLocalSizeBytes, currentSizeBytes)
                           : currentSizeBytes;
    return success();
  };

  WalkResult walkResult = funcOp.walk([&](Operation *op) -> WalkResult {
    if (auto initOp = dyn_cast<InitializeL2G2LPipeOp>(op)) {
      if (!isReserveBufferAddr(initOp.getLocalAddr(), reserveOp) &&
          !isReserveBufferAddr(initOp.getPeerLocalAddr(), reserveOp))
        return WalkResult::advance();
      if (failed(updateFromFifo(initOp.getOperation(), initOp.getSlotSize(),
                                initOp.getLocalSlotNumAttr()))) {
        failedToCompute = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (auto initOp = dyn_cast<AicInitializePipeOp>(op)) {
      if (!isReserveBufferAddr(initOp.getC2vConsumerBuf(), reserveOp) &&
          !isReserveBufferAddr(initOp.getV2cConsumerBuf(), reserveOp))
        return WalkResult::advance();
      if (failed(updateFromFifo(initOp.getOperation(), initOp.getSlotSize(),
                                initOp.getLocalSlotNumAttr()))) {
        failedToCompute = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (auto initOp = dyn_cast<AivInitializePipeOp>(op)) {
      if (!isReserveBufferAddr(initOp.getC2vConsumerBuf(), reserveOp) &&
          !isReserveBufferAddr(initOp.getV2cConsumerBuf(), reserveOp))
        return WalkResult::advance();
      if (failed(updateFromFifo(initOp.getOperation(), initOp.getSlotSize(),
                                initOp.getLocalSlotNumAttr()))) {
        failedToCompute = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  (void)walkResult;
  if (failedToCompute)
    return failure();
  return fifoLocalSizeBytes.value_or(reserveOp.getSize());
}

static void setReserveBufferSizeBytes(ReserveBufferOp reserveOp,
                                      int64_t sizeBytes) {
  reserveOp->setAttr(
      "size",
      IntegerAttr::get(IntegerType::get(reserveOp.getContext(), kI32BitWidth),
                       sizeBytes));
}

static LogicalResult
validateAutoReserveBufferCapacity(ReserveBufferPlans &plans) {
  DenseMap<AddressSpace, int64_t> reservedBytesByAddressSpace;

  for (ReserveBufferPlan &plan : plans) {
    if (plan.mode != ReserveBufferMode::Auto)
      continue;

    int64_t alignedSizeBytes = alignUpBytes(plan.sizeBytes, plan.alignBytes);
    if (alignedSizeBytes > plan.capacityBytes)
      return plan.reserveOp.emitOpError(
          "exceeds available local memory capacity");

    int64_t usedBytes = reservedBytesByAddressSpace[plan.addressSpace];
    if (usedBytes > plan.capacityBytes - alignedSizeBytes) {
      return plan.reserveOp.emitOpError(
          "cumulative auto reserve_buffer size exceeds available local "
          "memory capacity");
    }
    reservedBytesByAddressSpace[plan.addressSpace] =
        usedBytes + alignedSizeBytes;
  }

  return success();
}

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
    if (autoAlloc) {
      auto computedSizeBytes =
          computeAutoReserveBufferSizeBytes(funcOp, reserveOp);
      if (failed(computedSizeBytes))
        return failure();
      sizeBytes = *computedSizeBytes;
      if (sizeBytes != reserveOp.getSize())
        setReserveBufferSizeBytes(reserveOp, sizeBytes);
    }

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

  if (failed(validateAutoReserveBufferCapacity(plans)))
    return failure();

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

static DenseMap<AddressSpace, uint64_t>
collectAutoReserveBufferBitsByAddressSpace(const ReserveBufferPlans &plans) {
  DenseMap<AddressSpace, uint64_t> reservedBitsByAddressSpace;
  for (const ReserveBufferPlan &plan : plans) {
    if (plan.mode != ReserveBufferMode::Auto)
      continue;
    int64_t alignedSizeBytes = alignUpBytes(plan.sizeBytes, plan.alignBytes);
    reservedBitsByAddressSpace[plan.addressSpace] +=
        static_cast<uint64_t>(alignedSizeBytes) * kBitsPerByte;
  }
  return reservedBitsByAddressSpace;
}

} // namespace

void MemLivenessAnalysis::build() {
  Region &funcRegion = func_.getBody();
  stableValueOrder = buildStableValueOrder(func_);
  Liveness live(func_);
  // Recursively obtaining IR information.
  RecursionIR(&funcRegion, live);
  // the lifetime of the buffer.
  GenerateBufferLife();
  collectMultiBufferAnnotations();
}

void MemLivenessAnalysis::collectMultiBufferAnnotations() {
  func_.walk([&](memref::AllocOp alloc) {
    auto attr = alloc->getAttrOfType<IntegerAttr>(kPtoMultiBufferAttrName);
    if (!attr)
      return;
    uint64_t n = attr.getValue().getZExtValue();
    if (n <= 1 || n > kPtoMultiBufferMaxNum)
      return;
    buffer2MultiNum[alloc.getResult()] = static_cast<uint32_t>(n);
  });
}

bool MemLivenessAnalysis::isLocalMemPlan() const {
  return planMode == MemPlanMode::LOCAL_MEM_PLAN;
}

bool MemLivenessAnalysis::isGlobalWorkSpaceMemPlan() const {
  return planMode == MemPlanMode::GLOBAL_WORKSPACE_PLAN;
}

void MemLivenessAnalysis::RecursionIR(Region *region, Liveness live) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    // recursive control flow
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      RecursiveIfOp(ifOp, live);
      return WalkResult::skip();
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      RecursiveForOp(forOp, live);
      return WalkResult::skip();
    }

    // process operation
    auto curOpInfo = UpdateLinearOperation(op);
    auto mayAliasOp = getOperationAliasInfo(op);
    if (mayAliasOp.has_value()) {
      auto aliasPair = mayAliasOp.value();
      UpdateBufferAlias(aliasPair.first, aliasPair.second);
    } else if (isa<pto::DeclareTileMemRefOp>(op)) {
      // Internal placeholder for a tile whose runtime address is assigned by
      // pipe operations such as tpop. This op does not allocate local storage
      // and should not participate in memory planning.
      return WalkResult::advance();
    } else if (auto bindOp = dyn_cast<pto::BindTileOp>(op)) {
      // BindTile result is only an alias of the source buffer. Treat every use
      // of the result as a use of the source in liveness analysis.
      UpdateBufferAlias(bindOp.getResult(), bindOp.getSource());
      return WalkResult::advance();
    } else if (isLocalMemPlan() && dyn_cast<memref::AllocOp>(op)) {
      if (failed(CheckLocalBufferAllocOp(op))) {
        return WalkResult::interrupt();
      }
      UpdateOpBufferInfo(op, op->getResults());
      return WalkResult::advance();
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto tprintOp = dyn_cast<pto::TPrintOp>(op)) {
      // TPrintOp only reads from buffer, similar to LoadOp
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto tgetvalOp = dyn_cast<pto::TGetValOp>(op)) {
      (void)tgetvalOp;
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(op->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto setValidShapeOp = dyn_cast<pto::SetValidShapeOp>(op)) {
      (void)setValidShapeOp;
      // Metadata-only update on an existing tile handle. Keep the source buffer
      // alive through this operation, but do not model it as producing a new
      // alias/result buffer.
      UpdateOpGenInfo(curOpInfo, ValueRange{op->getOperand(0)});
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      UpdateStoreOpInfo(curOpInfo, storeOp.getMemRef(), live);
    } else if (auto ptoDpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
      // PTO ops with destination (tile_buf, partition_view, etc.); no
      // tensor/memref-only verification.
      SmallVector<Value> genBuffers = llvm::to_vector(ptoDpsOp.getDpsInits());
      auto scratchBuffers = getScratchBuffersFromEffects(
          op, ptoDpsOp.getDpsInits(), stableValueOrder);
      genBuffers.append(scratchBuffers.begin(), scratchBuffers.end());
      UpdateOpGenInfo(curOpInfo, genBuffers);
      for (const auto &conflictPair :
           getScratchConflictPairsFromEffects(op, ptoDpsOp.getDpsInits(),
                                              stableValueOrder)) {
        RecordSemanticConflict(conflictPair.first, conflictPair.second);
      }
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto dstStyleOp = dyn_cast<DestinationStyleOpInterface>(op)) {
      // Process the operation of pto instructions as follows:
      // pto.hir.copy ins(%0 : memref<16xf16, #pto.address_space<gm>>)
      //              outs(%1 : memref<16xxf16, #pto.address_space<ub>>)
      // need to handle kill buffer.
      UpdateInitAndResAlias(dstStyleOp);
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(dstStyleOp.getDpsInits()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      UpdateBufferAlias(selectOp.getResult(), selectOp.getTrueValue(), true);
      UpdateBufferAlias(selectOp.getResult(), selectOp.getFalseValue(), true);
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(callOp->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (isa<pto::TAllocOp, pto::TPushOp, pto::TFreeOp,
                   pto::InitializeL2LPipeOp, pto::InitializeL2G2LPipeOp,
                   pto::BuildAsyncSessionOp,
                   pto::TPutAsyncOp, pto::TGetAsyncOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(op->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto gpuLaunchOp = dyn_cast<gpu::LaunchFuncOp>(op)) {
      UpdateOpGenInfo(curOpInfo, llvm::to_vector(gpuLaunchOp->getOperands()));
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (failed(CheckIfUnknownOpTouchBuffer(op))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("PlanMemory Traverse IR Failed! ");
  }
}

void MemLivenessAnalysis::UpdateInitAndResAlias(
    DestinationStyleOpInterface dstStyleOp) {
  auto results = dstStyleOp->getResults();
  if (results.empty()) {
    return;
  }
  for (auto [res, init] :
       llvm::zip(dstStyleOp->getResults(), dstStyleOp.getDpsInits())) {
    auto iter = buffer2AliasVec.find(init);
    if (iter == buffer2AliasVec.end()) {
      continue;
    }
    auto tensorType = dyn_cast_or_null<TensorType>(res.getType());
    if (tensorType) {
      UpdateBufferAlias(res, init);
    }
  }
}

OpInfo *MemLivenessAnalysis::UpdateLinearOperation(Operation *op) {
  auto opInfo = std::make_unique<OpInfo>(op, seqIndex++);
  auto curOpInfo = opInfo.get();
  linearOperation.push_back(std::move(opInfo));
  return curOpInfo;
}

void MemLivenessAnalysis::UpdateForOpBufferAlias(scf::ForOp forOp) {
  if (forOp.getResults().empty()) {
    return;
  }
  if (!forOp.getRegionIterArgs().empty()) {
    if (forOp.getYieldedValues().size() != forOp.getRegionIterArgs().size() ||
        forOp.getInitArgs().size() != forOp.getRegionIterArgs().size()) {
      llvm::report_fatal_error("scf.for alias sizes are inconsistent");
    }
    for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      // yielded values alias region iter args.
      UpdateBufferAlias(forOp.getYieldedValues()[i], arg);
    }
  }
  if (forOp->getResults().size() != forOp.getYieldedValues().size())
    llvm::report_fatal_error("scf.for result/yield sizes are inconsistent");
  for (auto [i, arg] : llvm::enumerate(forOp.getYieldedValues())) {
    // forOp result values alias region iter yielded values.
    UpdateBufferAlias(forOp->getResult(i), arg);
  }
}

void MemLivenessAnalysis::RecursiveForOp(scf::ForOp forOp, Liveness live) {
  // Process the operation of ForOp as follows:
  // alloca %allocA
  // %0 = scf.for %arg4 = %c0 to %c1024 step %c128 iter_args(%arg5 = %4)->
  //      (memref<16x16x16xf16, #pto.address_space<ub>>):
  //          def(allocA)
  //          ...
  //          scf.yield %alloc0 : memref<16xf16,#pto.address_space<ub>>
  // need to handle kill buffer.
  auto forBeginSeq = UpdateLinearOperation(forOp.getOperation());
  UpdateOpGenInfo(forBeginSeq, GetLiveBuffersInLoop(forOp, live));
  UpdateForOpInitArgsAlias(forOp);
  RecursionIR(&forOp.getRegion(), live);
  UpdateForOpBufferAlias(forOp);
  auto forEndSeq = UpdateLinearOperation(forOp.getOperation());
  OpKillHandle(forEndSeq, live, forOp->getBlock());
}

void MemLivenessAnalysis::UpdateForOpInitArgsAlias(scf::ForOp forOp) {
  if (forOp.getInitArgs().empty()) {
    return;
  }
  if (forOp.getInitArgs().size() != forOp.getRegionIterArgs().size())
    llvm::report_fatal_error("scf.for init/iter-arg sizes are inconsistent");
  for (auto [i, arg] : llvm::enumerate(forOp.getInitArgs())) {
    // init args alias region iter args.
    UpdateBufferAlias(forOp.getRegionIterArgs()[i], arg);
  }
}

void MemLivenessAnalysis::UpdateIfOpBufferAlias(scf::IfOp ifOp,
                                                scf::YieldOp yieldOp) {
  if (ifOp.getResults().empty()) {
    return;
  }
  if (ifOp->getResults().size() != yieldOp->getOperands().size())
    llvm::report_fatal_error("scf.if result/yield sizes are inconsistent");
  for (auto [i, arg] : llvm::enumerate(yieldOp->getOperands())) {
    // Multiple buffers involved, requiring one-to-one correspondence.
    UpdateBufferAlias(ifOp->getResult(i), arg);
  }
}

void MemLivenessAnalysis::RecursiveIfOp(scf::IfOp ifOp, Liveness live) {
  // Process the operation of IfOp as follows:
  // %0 = scf.if %cond -> (memref<16xf16, #pto.address_space<ub>>)
  //        scf.yield %alloc0: memref<16xf16, #pto.address_space<ub>>
  //      else:
  //        scf.yield %alloc1 : memref<16xf16, #pto.address_space<ub>>
  auto curIfThen = UpdateLinearOperation(ifOp.getOperation());
  RecursionIR(&ifOp.getThenRegion(), live);
  auto curIfElse = UpdateLinearOperation(ifOp.getOperation());
  UpdateIfOpBufferAlias(ifOp, ifOp.thenYield());

  auto curIfEnd = curIfElse;
  if (ifOp.elseBlock()) {
    RecursionIR(&ifOp.getElseRegion(), live);
    curIfEnd = UpdateLinearOperation(ifOp.getOperation());
    UpdateIfOpBufferAlias(ifOp, ifOp.elseYield());
  }
  OpKillHandle(curIfEnd, live, ifOp->getBlock());
}

SmallVector<Value> MemLivenessAnalysis::GetLiveBuffersInLoop(scf::ForOp forOp,
                                                             Liveness live) {
  const auto *liveBlockInfo = live.getLiveness(forOp->getBlock());
  auto currentLiveValues =
      liveBlockInfo->currentlyLiveValues(forOp.getOperation());
  if (currentLiveValues.empty()) {
    return {};
  }
  // The gen buffer of the same operation must ensure the order of priority.
  SetVector<Value> currentLiveValuesOrder;
  for (auto buffer : currentLiveValues) {
    currentLiveValuesOrder.insert(buffer);
  }
  SetVector<Value> allocBeforeLoopBufferSet;
  for (const Value &operand : currentLiveValuesOrder) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto Buffer : aliasBuffers) {
      auto iter = buffer2status.find(Buffer);
      if (iter == buffer2status.end())
        continue;
      if ((iter->second == BufferStatus::DEFFINED ||
           iter->second == BufferStatus::KILLED) &&
          CanDelayLoopEntryGenUntilFirstWrite(forOp, Buffer)) {
        delayedLoopEntryGenBuffers[Buffer] = true;
        continue;
      }
      allocBeforeLoopBufferSet.insert(Buffer);
    }
  }
  SmallVector<Value> allocBeforeLoopBuffers(allocBeforeLoopBufferSet.begin(),
                                            allocBeforeLoopBufferSet.end());
  sortValuesByStableOrder(allocBeforeLoopBuffers, stableValueOrder);
  return allocBeforeLoopBuffers;
}

bool MemLivenessAnalysis::CanDelayLoopEntryGenUntilFirstWrite(
    scf::ForOp forOp, Value buffer) {
  SetVector<Value> aliasBuffers = GetAliasBuffers(buffer);
  aliasBuffers.insert(buffer);
  Block *body = forOp.getBody();
  if (!body)
    return false;

  for (Operation &op : body->without_terminator()) {
    if (!OperationOrNestedRegionTouchesAnyAlias(&op, aliasBuffers))
      continue;
    if (auto nestedForOp = dyn_cast<scf::ForOp>(&op)) {
      return llvm::any_of(aliasBuffers, [&](Value alias) {
        return CanDelayLoopEntryGenUntilFirstWrite(nestedForOp, alias);
      });
    }
    return IsWriteOnlyDpsInitForAlias(&op, aliasBuffers);
  }
  return false;
}

bool MemLivenessAnalysis::OperationDirectlyTouchesAnyAlias(
    Operation *op, const SetVector<Value> &aliasBuffers) const {
  auto touchesValue = [&](Value value) {
    return value && llvm::is_contained(aliasBuffers, value);
  };
  for (Value operand : op->getOperands()) {
    if (touchesValue(operand))
      return true;
  }
  for (Value result : op->getResults()) {
    if (touchesValue(result))
      return true;
  }

  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
              kMemoryEffectReserveSize>
      effects;
  memEffect.getEffects(effects);
  for (const auto &effect : effects) {
    if (touchesValue(effect.getValue()))
      return true;
  }
  return false;
}

bool MemLivenessAnalysis::OperationOrNestedRegionTouchesAnyAlias(
    Operation *op, const SetVector<Value> &aliasBuffers) const {
  if (OperationDirectlyTouchesAnyAlias(op, aliasBuffers))
    return true;
  if (op->getNumRegions() == 0)
    return false;

  bool touches = false;
  op->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (nestedOp == op)
      return WalkResult::advance();
    if (OperationDirectlyTouchesAnyAlias(nestedOp, aliasBuffers)) {
      touches = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return touches;
}

bool MemLivenessAnalysis::OperationReadsAnyAlias(
    Operation *op, const SetVector<Value> &aliasBuffers) const {
  auto touchesValue = [&](Value value) {
    return value && llvm::is_contained(aliasBuffers, value);
  };

  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect) {
    return llvm::any_of(op->getOperands(), touchesValue);
  }

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
              kMemoryEffectReserveSize>
      effects;
  memEffect.getEffects(effects);
  return llvm::any_of(effects, [&](const auto &effect) {
    return isa<MemoryEffects::Read>(effect.getEffect()) &&
           touchesValue(effect.getValue());
  });
}

bool MemLivenessAnalysis::IsWriteOnlyDpsInitForAlias(
    Operation *op, const SetVector<Value> &aliasBuffers) const {
  auto ptoDpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(op);
  if (!ptoDpsOp)
    return false;

  bool hasAliasDpsInit = false;
  for (Value init : ptoDpsOp.getDpsInits()) {
    if (llvm::is_contained(aliasBuffers, init)) {
      hasAliasDpsInit = true;
      break;
    }
  }
  if (!hasAliasDpsInit)
    return false;

  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>,
              kMemoryEffectReserveSize>
      effects;
  memEffect.getEffects(effects);

  bool hasWrite = false;
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    if (!value || !llvm::is_contained(aliasBuffers, value))
      continue;
    if (isa<MemoryEffects::Read>(effect.getEffect()))
      return false;
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      hasWrite = true;
      continue;
    }
    return false;
  }
  return hasWrite;
}

bool MemLivenessAnalysis::CanKillBeforeNextOverwrite(
    Operation *op, const SetVector<Value> &aliasBuffers) {
  for (Operation *nextOp = op->getNextNode(); nextOp;
       nextOp = nextOp->getNextNode()) {
    if (!OperationOrNestedRegionTouchesAnyAlias(nextOp, aliasBuffers))
      continue;

    if (auto forOp = dyn_cast<scf::ForOp>(nextOp)) {
      return llvm::any_of(aliasBuffers, [&](Value alias) {
        return CanDelayLoopEntryGenUntilFirstWrite(forOp, alias);
      });
    }

    return IsWriteOnlyDpsInitForAlias(nextOp, aliasBuffers);
  }
  return false;
}

bool MemLivenessAnalysis::CanRegenerateBufferAtOp(Operation *op,
                                                  Value buffer) {
  SetVector<Value> aliasBuffers = GetAliasBuffers(buffer);
  aliasBuffers.insert(buffer);
  return IsWriteOnlyDpsInitForAlias(op, aliasBuffers);
}

LogicalResult
MemLivenessAnalysis::CheckLocalBufferAllocOp(Operation *op) const {
  auto allocOp = dyn_cast<memref::AllocOp>(op);
  if (!allocOp)
    return op->emitError("must be alloc op"), failure();
  auto memorySpaceAttr = GetBufferSpaceAttr(allocOp.getResult());
  if (isLocalBuffer(memorySpaceAttr)) {
    return success();
  }
  allocOp.getOperation()->emitError("Alloc buffer not at UB space! ");
  return failure();
}

bool MemLivenessAnalysis::isSkippableOp(Operation *op) const {
  // Call-like ops are still modeled explicitly. Only pure terminators and
  // dim queries are skipped here.
  return isa<func::ReturnOp, scf::YieldOp, memref::DimOp>(op);
}

LogicalResult
MemLivenessAnalysis::CheckIfUnknownOpTouchBuffer(Operation *op) const {
  if (isSkippableOp(op) || isGlobalWorkSpaceMemPlan()) {
    // This scene can be ignored.
    return success();
  }
  if (isOpTouchLocalBuffer(op)) {
    op->emitError("PlanMemory Fail : Unrecognized type of Operation touches "
                  "local buffer!");
    return failure();
  }
  return success();
}

void MemLivenessAnalysis::UpdateBufferAlias(Value buffer, Value aliasBuffer,
                                            bool isIgnoreInplace) {
  // union all alias buffers about `aliasBuffer` and `buffer`
  auto unionAliasSet =
      Union(GetAliasBuffers(aliasBuffer), GetAliasBuffers(buffer));
  unionAliasSet.insert(buffer);
  unionAliasSet.insert(aliasBuffer);

  // update alias map info for each buffer
  // e.g. if A alias B, C alias D, now update:
  // A alias B,C,D; B alias A,C,D; C alias A,B,D; D alias A,B,C
  for (auto buf : unionAliasSet) {
    // remove buf self from union alias set
    auto clonedAliasSet = unionAliasSet;
    clonedAliasSet.remove(buf);

    buffer2AliasVec[buf] = clonedAliasSet;
  }

  // mark the alias buffer as ignoring Inplace if it is not generated by
  // memref.alloc.
  auto it = bufferInfos.find(aliasBuffer);
  if (isIgnoreInplace && it != bufferInfos.end()) {
    it->second.ignoreInplace = true;
  }
}

SetVector<Value> MemLivenessAnalysis::Union(SetVector<Value> set1,
                                            SetVector<Value> set2) {
  SetVector<Value> unionSet;
  unionSet.insert(set1.begin(), set1.end());
  unionSet.insert(set2.begin(), set2.end());
  return unionSet;
}

SetVector<Value> MemLivenessAnalysis::GetAliasBuffers(Value aliasBuffer) {
  auto trueVar = buffer2AliasVec.find(aliasBuffer);
  if (trueVar != buffer2AliasVec.end()) {
    return trueVar->second;
  }
  return {};
}

void MemLivenessAnalysis::UpdateStoreOpInfo(OpInfo *opInfo,
                                            const Value storeValue,
                                            Liveness live) {
  // The src of memref store may also serve as a gen buffer.
  SmallVector<Value, 1> storeValues;
  storeValues.push_back(storeValue);
  UpdateOpGenInfo(opInfo, storeValues);
  // Collect kill buffers corresponding to operation.
  OpKillHandle(opInfo, live, opInfo->operation->getBlock());
}

void MemLivenessAnalysis::UpdateOpBufferInfo(Operation *op,
                                             const ValueRange &results) {
  for (const Value &operand : results) {
    auto it = buffer2status.find(operand);
    if (it != buffer2status.end()) {
      continue;
    }
    bufferInfos[operand] = GenerateBufferInfo(op, operand);
    buffer2status[operand] = BufferStatus::DEFFINED;
  }
}

void MemLivenessAnalysis::UpdateOpGenInfo(OpInfo *opInfo,
                                          const ValueRange &results) {
  if (results.empty()) {
    return;
  }
  for (Value operand : results) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto buffer : aliasBuffers) {
      UpdateOperandGenInfo(opInfo, buffer);
    }
  }
}

void MemLivenessAnalysis::UpdateOperandGenInfo(OpInfo *opInfo, Value operand) {
  auto iter_buffer = buffer2status.find(operand);
  if (iter_buffer == buffer2status.end())
    return;
  if (iter_buffer->second == BufferStatus::DEFFINED) {
    appendUniqueValue(genKillMap[opInfo].gen, operand);
    buffer2status[iter_buffer->first] = BufferStatus::GENED;
    buffer2GenOp[iter_buffer->first] = opInfo->operation;
  } else if (iter_buffer->second == BufferStatus::KILLED) {
    if (!CanRegenerateBufferAtOp(opInfo->operation, operand)) {
      llvm_unreachable("The buffer memory has been released and cannot be "
                       "used again before it is redefined! ");
    }
    appendUniqueValue(genKillMap[opInfo].gen, operand);
    buffer2status[iter_buffer->first] = BufferStatus::GENED;
    buffer2GenOp[iter_buffer->first] = opInfo->operation;
  } else if (iter_buffer->second == BufferStatus::GENED) {
    SetVector<Value> aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    if (IsWriteOnlyDpsInitForAlias(opInfo->operation, aliasBuffers)) {
      appendUniqueValue(genKillMap[opInfo].kill, operand);
      appendUniqueValue(genKillMap[opInfo].gen, operand);
      buffer2GenOp[iter_buffer->first] = opInfo->operation;
    }
  }
}

void MemLivenessAnalysis::OpKillHandle(OpInfo *opInfo, Liveness live,
                                       Block *block) {
  const auto *liveBlockInfo = live.getLiveness(block);
  auto currentLiveValues =
      liveBlockInfo->currentlyLiveValues(opInfo->operation);
  if (currentLiveValues.empty()) {
    return;
  }
  SmallVector<Value> liveValues(currentLiveValues.begin(),
                                currentLiveValues.end());
  // Always begin from a stable ordering so seed=0 reproduces the original
  // single-shot PTO behavior. When the PlanMemoryPass retry loop drives a
  // non-zero seed, getShuffledRange permutes the candidates to expose
  // alternative gen/kill orderings - the search dimension that lets a later
  // attempt succeed where the first one wedged on a pathological order.
  sortValuesByStableOrder(liveValues, stableValueOrder);
  liveValues = getShuffledRange(liveValues);
  for (const Value &operand : liveValues) {
    UpdateOpKillInfo(opInfo, operand, live);
  }
}

void MemLivenessAnalysis::UpdateOpKillInfo(OpInfo *opInfo, Value operand,
                                           Liveness live) {
  auto aliasBuffers = GetAliasBuffers(operand);
  aliasBuffers.insert(operand);
  for (Value aliasBuffer : aliasBuffers) {
    auto iterBuffer = buffer2status.find(aliasBuffer);
    if (iterBuffer == buffer2status.end())
      return;
    Operation *defOp = iterBuffer->first.getDefiningOp();
    bool canKillInThisBlock =
        defOp && IsInSameBlock(defOp, opInfo->operation);
    auto delayedGen = delayedLoopEntryGenBuffers.find(iterBuffer->first);
    if (!canKillInThisBlock && delayedGen != delayedLoopEntryGenBuffers.end() &&
        delayedGen->second) {
      Operation *genOp = GetBufferGenOp(iterBuffer->first);
      canKillInThisBlock = genOp && IsInSameBlock(genOp, opInfo->operation);
    }
    bool canKillCurrentValue =
        AllDeadAfter(opInfo->operation, aliasBuffers, live) ||
        (OperationReadsAnyAlias(opInfo->operation, aliasBuffers) &&
         CanKillBeforeNextOverwrite(opInfo->operation, aliasBuffers));
    if (iterBuffer->second == BufferStatus::GENED && canKillInThisBlock &&
        canKillCurrentValue) {
      appendUniqueValue(genKillMap[opInfo].kill, aliasBuffer);
      buffer2status[iterBuffer->first] = BufferStatus::KILLED;
      buffer2GenOp.erase(iterBuffer->first);
    }
  }
}

Operation *MemLivenessAnalysis::GetBufferGenOp(Value buffer) const {
  auto it = buffer2GenOp.find(buffer);
  if (it != buffer2GenOp.end())
    return it->second;
  return nullptr;
}

bool MemLivenessAnalysis::IsInSameBlock(Operation *op1, Operation *op2) const {
  return op1->getBlock() == op2->getBlock();
}

bool MemLivenessAnalysis::AllDeadAfter(Operation *op, SetVector<Value> aliasVec,
                                       Liveness live) const {
  for (auto aliasBuffer : aliasVec) {
    if (!live.isDeadAfter(aliasBuffer, op)) {
      return false;
    }
  }
  return true;
}

void MemLivenessAnalysis::RecordSemanticConflict(Value lhs, Value rhs) {
  SetVector<Value> lhsAliases = GetAliasBuffers(lhs);
  lhsAliases.insert(lhs);
  SetVector<Value> rhsAliases = GetAliasBuffers(rhs);
  rhsAliases.insert(rhs);

  auto appendUniquePair = [&](Value a, Value b) {
    if (!a || !b || a == b)
      return;
    ValuePair pair = isLessValue(a, b) ? ValuePair(a, b) : ValuePair(b, a);
    if (!llvm::is_contained(semanticConflictPairs, pair))
      semanticConflictPairs.push_back(pair);
  };

  for (Value a : lhsAliases)
    for (Value b : rhsAliases)
      appendUniquePair(a, b);
}

BufferInfo MemLivenessAnalysis::GenerateBufferInfo(Operation *op,
                                                   Value operand) {
  auto memorySpaceAttr = GetBufferSpaceAttr(operand);
  if (isLocalMemPlan() && isLocalBuffer(memorySpaceAttr)) {
    if (!memorySpaceAttr.has_value())
      llvm::report_fatal_error("local buffer must have memory space");
    return GetBufferInfo(op, operand,
                         memorySpaceAttr.value().getAddressSpace());
  }
  llvm_unreachable("buffer must has BufferInfo !");
}

BufferInfo MemLivenessAnalysis::GetBufferInfo(Operation *op, Value operand,
                                              pto::AddressSpace bufferScope) {
  BufferInfo bufferInfo;
  bufferInfo.operation = op;
  bufferInfo.bufferScope = bufferScope;
  // get buffer size, now for static shape
  Value traceValue = tracebackMemRef(operand);
  auto memRefType = cast<MemRefType>(traceValue.getType());
  bufferInfo.bufferType = memRefType.getElementType();
  std::optional<int64_t> totalStaticSize =
      getStaticTotalSize(memRefType.getShape());
  if (!totalStaticSize.has_value())
    llvm::report_fatal_error("failed to obtain buffer static shape size");
  bufferInfo.constBits =
      totalStaticSize.value() *
      static_cast<int64_t>(memRefType.getElementTypeBitWidth());
  return bufferInfo;
}

void MemLivenessAnalysis::GenerateBufferLife() {
  int scopeTime = 0;
  DenseMap<Value, std::shared_ptr<BufferLife>> openLives;
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    auto it = genKillMap.find(linearOperation[i].get());
    if (it == genKillMap.end()) {
      scopeTime++;
      continue;
    }

    SmallVector<Value> postGenKills;
    for (const Value &killBuffer : it->second.kill) {
      auto iter = openLives.find(killBuffer);
      if (iter != openLives.end()) {
        iter->second->freeTime = scopeTime;
        openLives.erase(iter);
        continue;
      }
      if (!llvm::is_contained(it->second.gen, killBuffer))
        llvm::report_fatal_error("buffer lifetime killed before generation");
      appendUniqueValue(postGenKills, killBuffer);
    }

    for (const Value &genBuffer : it->second.gen) {
      if (openLives.find(genBuffer) != openLives.end())
        llvm::report_fatal_error("buffer lifetime generated before release");
      std::shared_ptr<BufferLife> bufferLife =
          std::make_shared<BufferLife>(genBuffer);
      bufferLife->allocTime = scopeTime;
      buffer2Life[genBuffer].push_back(bufferLife);
      openLives[genBuffer] = std::move(bufferLife);
    }

    for (const Value &killBuffer : postGenKills) {
      auto iter = openLives.find(killBuffer);
      if (iter == openLives.end())
        llvm::report_fatal_error("buffer lifetime generated after release");
      iter->second->freeTime = scopeTime;
      openLives.erase(iter);
    }
    scopeTime++;
  }
}

std::shared_ptr<BufferLife>
StorageEntry::GetBufferLifeByValue(const Value v) const {
  auto find = std::find_if(
      bufferLifeVec.begin(), bufferLifeVec.end(),
      [v](std::shared_ptr<BufferLife> life) { return life->buffer == v; });
  if (find != bufferLifeVec.end()) {
    return *find;
  }
  return nullptr;
}

bool MemPlan::IsReusePTOOp(Operation *op) const {
  if (restrictInplaceAsISA)
    return false;

  // not in ISA but confirmed with hardware developers:
  // elementwise ops with the same shape and the same bitwidth operands can also
  // do memory inplace for src and dst
  return false;
}

SmallVector<ValuePair> MemPlan::GenerateInplaceList() {
  SmallVector<ValuePair> inplaceList;
  DenseMap<Operation *, bool> hasTouchOp;
  inplaceList.insert(inplaceList.end(), inplacePairList.begin(),
                     inplacePairList.end());
  for (auto &operationSeq : linearOperation) {
    auto it = genKillMap.find(operationSeq.get());
    if (it == genKillMap.end())
      continue;
    if (hasTouchOp[operationSeq->operation]) {
      continue;
    }

    SmallVector<Value> genBuffers(it->second.gen.begin(), it->second.gen.end());
    SmallVector<Value> killBuffers(it->second.kill.begin(), it->second.kill.end());
    sortValuesByStableOrder(genBuffers, stableValueOrder);
    sortValuesByStableOrder(killBuffers, stableValueOrder);

    for (const Value &genBuffer : genBuffers) {
      auto genBufferIter = bufferInfos.find(genBuffer);
      if (genBufferIter == bufferInfos.end())
        llvm::report_fatal_error("gen buffer missing from buffer info map");
      if (genBufferIter->second.ignoreInplace) {
        continue;
      }

      for (const Value &killBuffer : killBuffers) {
        auto killBufferIter = bufferInfos.find(killBuffer);
        if (killBufferIter == bufferInfos.end())
          llvm::report_fatal_error("kill buffer missing from buffer info map");
        if (killBufferIter->second.ignoreInplace) {
          continue;
        }

        bool bufferSizeMatch =
            killBufferIter->second.constBits >= genBufferIter->second.constBits;
        bool isResuableOp = IsReusePTOOp(it->first->operation);
        bool canInplace = bufferSizeMatch && isResuableOp;
        if (canInplace) {
          inplaceList.emplace_back(std::make_pair(genBuffer, killBuffer));
          break;
        }
      }
    }
    // Nodes in inplace are only processed once.
    hasTouchOp[operationSeq->operation] = true;
  }
  return inplaceList;
}

void MemPlan::EmitPlanMemoryFailureInfo() {
  if (failApplyBufferInfo.empty())
    return;
  for (auto &iter : failApplyBufferInfo) {
    AddressSpace space = iter.first;
    auto bufferSpaceInfo = GetPlannableBufferSpaceInfo(space);
    func_.emitError() << stringifyEnum(space) << " overflow, requires "
                      << iter.second << " bits while " << bufferSpaceInfo.second
                      << " bits avaliable!";
  }
}

bool MemPlan::RecordOverflowIfAny() {
  if (!failApplyBufferInfo.empty()) {
    return true;
  }
  if (planMode != MemPlanMode::LOCAL_MEM_PLAN ||
      memscope2rootStorageEntry.empty()) {
    return false;
  }

  for (auto &it : memscope2rootStorageEntry) {
    auto *rootStorageEntry = it.second;
    if (!rootStorageEntry) {
      continue;
    }
    auto bufferSpaceInfo =
        GetPlannableBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
    size_t maxBits = bufferSpaceInfo.second;
    uint64_t maxAllocBits = rootStorageEntry->alignedConstBits;
    for (auto *child : rootStorageEntry->mergedChildren) {
      maxAllocBits =
          std::max(maxAllocBits, child->bitsOffset + child->alignedConstBits);
    }
    if (maxAllocBits > maxBits) {
      failApplyBufferInfo[rootStorageEntry->bufInfo->bufferScope] =
          maxAllocBits;
    }
  }

  return !failApplyBufferInfo.empty();
}

bool MemPlan::HasSemanticConflict(const StorageEntry *entry,
                                  const BufferLifeVec &bufferLives) const {
  if (!entry || semanticConflictPairs.empty() || bufferLives.empty())
    return false;

  auto containsPair = [&](Value lhs, Value rhs) {
    ValuePair pair = isLessValue(lhs, rhs) ? ValuePair(lhs, rhs)
                                           : ValuePair(rhs, lhs);
    return llvm::is_contained(semanticConflictPairs, pair);
  };

  for (Value entryBuffer : entry->inplaceBuffers) {
    for (const auto &life : bufferLives) {
      if (!life)
        continue;
      Value otherBuffer = life->buffer;
      if (!otherBuffer || entryBuffer == otherBuffer)
        continue;
      if (containsPair(entryBuffer, otherBuffer))
        return true;
    }
  }
  return false;
}

// Plan Memory algorithm.
LogicalResult MemPlan::plan(bool emitErrors) {
  // Construct StorageEntry structure.
  GenerateStorageEntry();
  // Plan memory address.
  PlanStatus as = planMode == MemPlanMode::LOCAL_MEM_PLAN
                      ? PlanLocalMemAddress()
                      : PlanWorkSpaceMemAddress();
  if (as == PlanStatus::PLAN_FAILED) {
    if (emitErrors)
      EmitPlanMemoryFailureInfo();
    return failure();
  }
  if (RecordOverflowIfAny()) {
    if (emitErrors)
      EmitPlanMemoryFailureInfo();
    return failure();
  }
  auto hasAddressOverlap = [](const StorageEntry *lhs, const StorageEntry *rhs) {
    uint64_t lhsBegin = lhs->bitsOffset;
    uint64_t lhsEnd = lhs->bitsOffset + lhs->alignedConstBits;
    uint64_t rhsBegin = rhs->bitsOffset;
    uint64_t rhsEnd = rhs->bitsOffset + rhs->alignedConstBits;
    return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
  };
  SmallVector<const StorageEntry *> plannedEntries;
  plannedEntries.reserve(StorageEntryVec.size() + pingEntry2RelationPongEntry.size());
  for (const auto &entry : StorageEntryVec) {
    plannedEntries.push_back(entry.get());
  }
  for (const auto &entry : pingEntry2RelationPongEntry) {
    plannedEntries.push_back(entry.second.get());
  }
  for (size_t i = 0; i < plannedEntries.size(); ++i) {
    for (size_t j = i + 1; j < plannedEntries.size(); ++j) {
      const StorageEntry *lhs = plannedEntries[i];
      const StorageEntry *rhs = plannedEntries[j];
      if (!lhs || !rhs) {
        continue;
      }
      if (lhs->bufInfo->bufferScope != rhs->bufInfo->bufferScope) {
        continue;
      }
      if (!hasAddressOverlap(lhs, rhs)) {
        continue;
      }
      bool lifeOverlap =
          !GetOverlapBufferLife(lhs->bufferLifeVec, rhs->bufferLifeVec).empty();
      bool semanticConflict = HasSemanticConflict(lhs, rhs->bufferLifeVec);
      if (!lifeOverlap && !semanticConflict) {
        continue;
      }
      if (emitErrors) {
        func_.emitError()
            << "PlanMemory produced overlapping local buffers in "
            << stringifyEnum(lhs->bufInfo->bufferScope)
            << " at offsets " << lhs->bitsOffset << " and "
            << rhs->bitsOffset;
      }
      return failure();
    }
  }
  // Update the address information of each buffer after memory buffer.
  UpdateBuffer2Offsets();
  if (enablePrintMemoryAllocatedSize) {
    PrintSuccessfulAllocatedMaxBits();
  }
  return success();
}

void MemPlan::GenerateStorageEntry() {
  // create new storage entry.
  SetVector<Value> seenBuffers;
  for (auto &operation : linearOperation) {
    auto it = genKillMap.find(operation.get());
    if (it == genKillMap.end())
      continue;
    SmallVector<Value> genBuffers(it->second.gen.begin(), it->second.gen.end());
    sortValuesByStableOrder(genBuffers, stableValueOrder);
    for (const Value &genBuffer : genBuffers) {
      if (llvm::is_contained(seenBuffers, genBuffer))
        continue;
      auto iter = bufferInfos.find(genBuffer);
      if (iter == bufferInfos.end()) {
        continue;
      }
      seenBuffers.insert(genBuffer);
      const BufferLifeVec &bufLives = buffer2Life.at(genBuffer);
      std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
      entry->bufInfo = &iter->second;
      entry->bufferLifeVec.append(bufLives.begin(), bufLives.end());
      entry->inplaceBuffers.emplace_back(iter->first);
      auto multiBuffer = buffer2MultiNum.find(genBuffer);
      if (multiBuffer != buffer2MultiNum.end()) {
        entry->multiBufferNum = multiBuffer->second;
      }
      buffer2storageEntry[genBuffer] = entry.get();
      // Verify the validity of parameters after initialization.
      ValidateParameters(entry);
      StorageEntryVec.emplace_back(std::move(entry));
    }
  }
}

void MemPlan::PrintSuccessfulAllocatedMaxBits() {
  auto it = memscope2rootStorageEntry.find(pto::AddressSpace::VEC);
  if (it != memscope2rootStorageEntry.end()) {
    if (!it->second)
      llvm::report_fatal_error("missing root storage entry for VEC scope");
    uint64_t ubAllocBits = it->second->alignedConstBits + it->second->bitsOffset;
    for (auto& child : it->second->mergedChildren) {
      ubAllocBits = std::max(ubAllocBits, child->bitsOffset + child->alignedConstBits);
    }
    llvm::outs() << "[PTOPlanMemory] Allocated UB size = " << ubAllocBits
                 << " bits\n";
  }
}

void MemPlan::ValidateParameters(std::unique_ptr<StorageEntry> &e) const {
  if (!e->bufInfo->operation)
    llvm::report_fatal_error("storage entry missing defining operation");
  if (e->bufInfo->constBits < 0U)
    llvm::report_fatal_error("storage entry has invalid memory size");
  if (e->bufferLifeVec.empty())
    llvm::report_fatal_error("storage entry missing lifetime information");
}

void MemPlan::UpdateBuffer2Offsets() {
  // Slot ordering contract (relied on by EnableMultiBuffer / AllocToPointerCast):
  //   buffer2Offsets[buf][i] is the byte offset of physical slot `i`,
  //   selected at runtime by `iv mod N == i`. Slot 0 is the primary entry's
  //   bitsOffset; slots 1..N-1 are `relationOtherBuffers[0..N-2]` in order.
  // To preserve this we walk via primaries and explicitly emit slots, and
  // skip non-primary slot entries that share `inplaceBuffers` with the
  // primary (they would otherwise be double-pushed by a naive linear walk).
  auto bitsToBytes = [](uint64_t bits) {
    return (bits + kBitsToByte - 1) / kBitsToByte;
  };

  for (auto &e : StorageEntryVec) {
    if (e->isMultiBufferSlot)
      continue; // handled via its primary below

    if (e->multiBufferNum > 1) {
      // Multi-buffer primary: emit slot 0 then slots 1..N-1 in declared order.
      for (Value &buffer : e->inplaceBuffers) {
        buffer2Offsets[buffer].push_back(bitsToBytes(e->bitsOffset));
        for (StorageEntry *slot : e->relationOtherBuffers) {
          if (!slot)
            llvm::report_fatal_error(
                "multi-buffer primary has null relation slot");
          buffer2Offsets[buffer].push_back(bitsToBytes(slot->bitsOffset));
        }
      }
      // Defensive invariant: each multi-buffered buffer must end up with
      // exactly multiBufferNum offsets after this call (modulo the
      // SPEC_LEVEL_1 single-reuse-db append below, which only fires for
      // single-buffer entries).
      for (Value &buffer : e->inplaceBuffers) {
        if (buffer2Offsets[buffer].size() != e->multiBufferNum) {
          llvm::report_fatal_error(
              "multi-buffer offset count mismatch in UpdateBuffer2Offsets");
        }
      }
      continue;
    }

    // Single-buffer entry: classic single-offset push.
    for (Value &buffer : e->inplaceBuffers) {
      buffer2Offsets[buffer].push_back(bitsToBytes(e->bitsOffset));
    }
  }
  // In the MultiBuffer scenario, single reuse db will result in additional
  // storageEntry. Only fires for single-buffer primaries that took a DB slot.
  UpdateMultiBufferReuseExtraOffset();
}

void MemPlan::UpdateMultiBufferReuseExtraOffset() {
  if (pingEntry2RelationPongEntry.empty()) {
    return;
  }

  for (auto &relationEntry : pingEntry2RelationPongEntry) {
    for (Value &buffer : relationEntry.second->inplaceBuffers) {
      // MultiBuffer can cause multiple addrs.
      buffer2Offsets[buffer].push_back(
          (relationEntry.second->bitsOffset + kBitsToByte - 1) /
          kBitsToByte);
    }
  }
}

void MemPlan::MergeInplaceSE() {
  // get the list of inplace value pair.
  SmallVector<ValuePair> inplaceList = GenerateInplaceList();
  // try to merge storage entries. genSE is replaced by KillSE.
  for (const auto &pairIter : inplaceList) {
    const StorageEntry *genSE = buffer2storageEntry[pairIter.first];
    StorageEntry *killSE = buffer2storageEntry[pairIter.second];
    if (genSE == killSE) {
      // already same storageEntry, no need to inplace.
      continue;
    }
    if (genSE == nullptr || killSE == nullptr)
      llvm::report_fatal_error("invalid storage entry during inplace merge");
    BufferLifeVec mergedBufferLifeVec;
    mergedBufferLifeVec.insert(mergedBufferLifeVec.end(),
                               genSE->bufferLifeVec.begin(),
                               genSE->bufferLifeVec.end());
    mergedBufferLifeVec.insert(mergedBufferLifeVec.end(),
                               killSE->bufferLifeVec.begin(),
                               killSE->bufferLifeVec.end());
    MergeBufferVec(mergedBufferLifeVec);
    killSE->bufferLifeVec.swap(mergedBufferLifeVec);
    //  merge allocs of two storage entry and update inplaceBuffers.
    killSE->inplaceBuffers.insert(killSE->inplaceBuffers.begin(),
                                  genSE->inplaceBuffers.begin(),
                                  genSE->inplaceBuffers.end());

    // Take the maximum value for the inplace scene.
    killSE->multiBufferNum =
        std::max(genSE->multiBufferNum, killSE->multiBufferNum);

    // all buffers have same storage entry after merging
    for (auto &buffer : genSE->inplaceBuffers) {
      buffer2storageEntry[buffer] = killSE;
    }
    // remove the alloc info of dst after successful merging
    auto e = std::find_if(StorageEntryVec.begin(), StorageEntryVec.end(),
                          [genSE](std::unique_ptr<StorageEntry> &se) {
                            return se.get() == genSE;
                          });
    StorageEntryVec.erase(e);
  }
}

PlanStatus MemPlan::PlanLocalMemAddress() {
  // merge from the first storage entry
  MergeInplaceSE();
  dmaFirstPipelineOpt.build(func_);
  ExpandMultiBufferStorageEntry();
  MergeSameScopeSE();
  return PlanMemAddressOfWholeLocalBuffer();
}

PlanStatus MemPlan::PlanWorkSpaceMemAddress() {
  // merge from the first storage entry
  MergeInplaceSE();
  ExpandMultiBufferStorageEntry();
  return PlanMemOffsetOfWholeWorkSpace();
}

PlanStatus MemPlan::PlanMemOffsetOfWholeWorkSpace() {
  for (auto &it : workSpaceArg2rootStorageEntry) {
    StorageEntry *rootStorageEntry = it.second;
    if (!enableGlobalReuse) {
      GlobalWorkspaceNoReuse(rootStorageEntry);
      continue;
    }
    MemBoundList outline;
    PlanRecHis history;
    SpecInfo si;
    // Can be reuse without conflicting life intervals.
    si.specLevel = si.minLevel;
    int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
    outline.push_back(std::make_shared<MemoryBound>(
        BufferLifeVec(), 0, std::numeric_limits<uint64_t>::max(), nullptr));

    // The initial value is rootStorageEntry.
    StorageEntry *curEntry = rootStorageEntry;
    while (si.childIdx < childrenNum) {
      curEntry->alignedConstBits =
          static_cast<uint64_t>(curEntry->bufInfo->constBits);
      curEntry->childIdx = si.childIdx;
      LogicalResult planResult = MultiSpecPlan(si, outline, history, curEntry);
      if (failed(planResult)) {
        return PlanStatus::PLAN_FAILED;
      }
      if (si.childIdx >= childrenNum) {
        break;
      }
      curEntry = rootStorageEntry->mergedChildren[si.childIdx];
    }
  }
  planStatus = PlanStatus::PLAN_SUCCESS;
  return planStatus;
}

void MemPlan::GlobalWorkspaceNoReuse(StorageEntry *rootStorageEntry) {
  rootStorageEntry->bitsOffset = 0;
  uint64_t offset = static_cast<uint64_t>(rootStorageEntry->bufInfo->constBits);
  for (StorageEntry *child : rootStorageEntry->mergedChildren) {
    child->bitsOffset = offset;
    offset += static_cast<uint64_t>(child->bufInfo->constBits);
  }
}

void MemPlan::ExpandMultiBufferStorageEntry() {
  // StorageEntry that needs to be expanded.
  size_t size = StorageEntryVec.size();
  for (size_t i = 0; i < size; i++) {
    if (StorageEntryVec[i]->multiBufferNum <= 1)
      continue;
    StorageEntry *primary = StorageEntryVec[i].get();
    primary->relationOtherBuffers.clear();
    uint32_t n = primary->multiBufferNum;
    for (uint32_t k = 1; k < n; ++k) {
      std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
      entry->bufInfo = primary->bufInfo;
      entry->bufferLifeVec = primary->bufferLifeVec;
      entry->alignedConstBits = primary->alignedConstBits;
      entry->inplaceBuffers = primary->inplaceBuffers;
      entry->multiBufferNum = primary->multiBufferNum;
      // Mark this as a non-primary slot. UpdateBuffer2Offsets uses this flag
      // to enforce primary-first slot ordering.
      entry->isMultiBufferSlot = true;
      StorageEntry *raw = entry.get();
      primary->relationOtherBuffers.push_back(raw);
      StorageEntryVec.push_back(std::move(entry));
    }
  }
}

bool MemPlan::IsEnoughForBuffersNoReuse(StorageEntry *rootStorageEntry,
                                        size_t restBufferSize,
                                        size_t alignUnit) {
  auto iter =
      bufferScope2RequiredSize.find(rootStorageEntry->bufInfo->bufferScope);
  if (iter == bufferScope2RequiredSize.end())
    llvm::report_fatal_error("missing required-size entry for buffer scope");
  if (iter->second < restBufferSize) {
    PlanBuffersWithoutReuse(rootStorageEntry, alignUnit);
    return true;
  }
  return false;
}

void MemPlan::PlanBuffersWithoutReuse(StorageEntry *rootStorageEntry,
                                      size_t alignUnit) {
  uint offset = 0;
  rootStorageEntry->bitsOffset = offset;
  offset = AlignUp(rootStorageEntry->bufInfo->constBits, alignUnit);
  rootStorageEntry->alignedConstBits = offset;
  for (StorageEntry *child : rootStorageEntry->mergedChildren) {
    child->bitsOffset = offset;
    uint64_t alignedBits = AlignUp(child->bufInfo->constBits, alignUnit);
    offset += alignedBits;
    child->alignedConstBits = alignedBits;
  }
}

void MemPlan::MergeSameScopeSE() {
  // Construct root StorageEntry and collect the same scope StorageEntry
  for (auto &iter : StorageEntryVec) {
    auto iter_scope =
        memscope2rootStorageEntry.find(iter->bufInfo->bufferScope);
    if (iter_scope == memscope2rootStorageEntry.end()) {
      memscope2rootStorageEntry[iter->bufInfo->bufferScope] = iter.get();
    } else {
      iter_scope->second->mergedChildren.push_back(iter.get());
    }
  }

  // set bufferScope2RequiredSize for all StorageEntry
  for (auto &rootStorageEntry : memscope2rootStorageEntry) {
    auto bufferSpaceInfo = GetPlannableBufferSpaceInfo(rootStorageEntry.first);
    size_t accumulateSize = AlignUp(rootStorageEntry.second->bufInfo->constBits,
                                    bufferSpaceInfo.first);
    for (auto &childrenStorageEntry : rootStorageEntry.second->mergedChildren) {
      size_t curStorageSize = AlignUp(childrenStorageEntry->bufInfo->constBits,
                                      bufferSpaceInfo.first);
      accumulateSize = accumulateSize + curStorageSize;
    }
    bufferScope2RequiredSize[rootStorageEntry.first] = accumulateSize;
  }
}

void MemPlan::PlanMemAddressForLevel0(
    StorageEntry *rootStorageEntry) {
  // get the buffer info for a given scope.
  auto bufferSpaceInfo =
      GetBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
  size_t align = bufferSpaceInfo.first;
  size_t maxBits = UINT64_MAX;
  rootStorageEntry = GetReorderRootStorageEntry(rootStorageEntry);
  // memory outline in a given buffer scope.
  MemBoundList outline;
  PlanRecHis history;
  SpecInfo si;
  si.specLevel = SPEC_LEVEL_0;
  si.maxLevel = SPEC_LEVEL_0;
  int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
  outline.push_back(
      std::make_shared<MemoryBound>(BufferLifeVec(), 0, maxBits, nullptr));

  // The initial value is rootStorageEntry.
  StorageEntry *curEntry = rootStorageEntry;
  while (si.childIdx < childrenNum) {
    uint64_t needBits = static_cast<uint64_t>(curEntry->bufInfo->constBits);
    curEntry->alignedConstBits = AlignUp(needBits, align);
    curEntry->childIdx = si.childIdx;
    (void)MultiSpecPlan(si, outline, history, curEntry);
    if (si.childIdx >= childrenNum) {
      break;
    }
    curEntry = rootStorageEntry->mergedChildren[si.childIdx];
  }
  // Find the max appled bits from all children and root, which is the max
  // memory applied in this buffer space.
  uint64_t maxAllocBits = rootStorageEntry->alignedConstBits;
  auto children = rootStorageEntry->mergedChildren;
  for (auto *child : children) {
    maxAllocBits =
        std::max(maxAllocBits, child->bitsOffset + child->alignedConstBits);
  }
  failApplyBufferInfo[rootStorageEntry->bufInfo->bufferScope] = maxAllocBits;
}

PlanStatus MemPlan::PlanMemAddressOfWholeLocalBuffer() {
  // Start plan
  for (auto &it : memscope2rootStorageEntry) {
    StorageEntry *rootStorageEntry = it.second;
    // get the buffer info for a given scope.
    auto bufferSpaceInfo =
        GetPlannableBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
    size_t align = bufferSpaceInfo.first;
    size_t maxBits = bufferSpaceInfo.second;
    if (rootStorageEntry->mergedChildren.empty()) {
      PlanStatus status = PlanSingleLocalBuffer(rootStorageEntry, align, maxBits);
      if (status != PlanStatus::PLAN_SUCCESS)
        return status;
      continue;
    }
    if (IsEnoughForBuffersNoReuse(rootStorageEntry, maxBits, align)) {
      continue;
    }
    PlanStatus status = PlanReusableLocalBuffer(rootStorageEntry, align, maxBits);
    if (status != PlanStatus::PLAN_SUCCESS)
      return status;
  }
  planStatus = PlanStatus::PLAN_SUCCESS;
  return planStatus;
}

PlanStatus MemPlan::PlanSingleLocalBuffer(StorageEntry *rootStorageEntry,
                                          size_t align, size_t maxBits) {
  uint64_t needAlignedBits = AlignUp(rootStorageEntry->bufInfo->constBits, align);
  if (needAlignedBits > maxBits) {
    failApplyBufferInfo[rootStorageEntry->bufInfo->bufferScope] =
        needAlignedBits;
    return PlanStatus::PLAN_FAILED;
  }
  rootStorageEntry->bitsOffset = 0;
  rootStorageEntry->alignedConstBits = needAlignedBits;
  return PlanStatus::PLAN_SUCCESS;
}

PlanStatus MemPlan::PlanReusableLocalBuffer(StorageEntry *rootStorageEntry,
                                            size_t align, size_t maxBits) {
  rootStorageEntry = GetReorderRootStorageEntry(rootStorageEntry);
  ReportMemLifeDebugInfo(rootStorageEntry);

  MemBoundList outline;
  PlanRecHis history;
  SpecInfo si;
  si.specLevel = si.maxLevel;
  int childrenNum = static_cast<int>(rootStorageEntry->mergedChildren.size());
  outline.push_back(
      std::make_shared<MemoryBound>(BufferLifeVec(), 0, maxBits, nullptr));

  StorageEntry *curEntry = rootStorageEntry;
  while (si.childIdx < childrenNum) {
    uint64_t needBits = static_cast<uint64_t>(curEntry->bufInfo->constBits);
    curEntry->alignedConstBits = AlignUp(needBits, align);
    curEntry->childIdx = si.childIdx;
    LDBG("\n");
    LDBG("----------Need-Plan-CurEntry---------\n");
    ReportCurEntryDebugInfo(curEntry);
    LDBG("\n");
    LogicalResult planResult = MultiSpecPlan(si, outline, history, curEntry);
    if (failed(planResult)) {
      StatusWrapper statusWrapper = {false,   curEntry->alignedConstBits,
                                     &si,     outline,
                                     history, rootStorageEntry};
      LDBG("\n");
      LDBG("----------ApplyFailStrategy---------\n");
      ReportCurEntryDebugInfo(curEntry);
      LDBG("\n");
      PlanStatus status = ApplyFailStrategy(statusWrapper, maxBits);
      if (status == PlanStatus::RESTART_NEW_PLAN) {
        si = SpecInfo();
        curEntry = rootStorageEntry;
        continue;
      }
      if (status == PlanStatus::PLAN_FAILED) {
        ReportAllocatedEntryDebugInfo(rootStorageEntry);
        PlanMemAddressForLevel0(rootStorageEntry);
        return status;
      }
    }
    if (si.childIdx >= childrenNum)
      break;
    curEntry = rootStorageEntry->mergedChildren[si.childIdx];
  }
  return PlanStatus::PLAN_SUCCESS;
}

void MemPlan::ReportMemLifeDebugInfo(StorageEntry *rootStorageEntry) {
  LDBG("-------------------------- Buffer2Life --------------------------\n");
  MemLifeDebugInfo(rootStorageEntry);
  for (auto &StorageEntry : rootStorageEntry->mergedChildren) {
    MemLifeDebugInfo(StorageEntry);
  }
}

void MemPlan::MemLifeDebugInfo(StorageEntry *storageEntry) {
  for (auto &buffer : storageEntry->inplaceBuffers) {
    if (buffer.getDefiningOp()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(buffer.getDefiningOp())) {
        LDBG("Buffer : " << allocOp.getResult() << "\n");
      }
    }
  }
  for (auto &bufferLife : storageEntry->bufferLifeVec) {
    LDBG("bufferLife : "
         << "allocTime : " << bufferLife->allocTime
         << " , freeTime : " << bufferLife->freeTime << "\n");
  }
  LDBG("\n");
}

void MemPlan::ReportCurEntryDebugInfo(const StorageEntry *curEntry) {
  for (auto &buffer : curEntry->inplaceBuffers) {
    if (buffer.getDefiningOp()) {
      if (auto allocOp = dyn_cast<memref::AllocOp>(buffer.getDefiningOp())) {
        LDBG("buffer : ");
        LDBG(allocOp.getResult());
      }
    }
  }
}

StorageEntry *
MemPlan::GetReorderRootStorageEntry(StorageEntry *rootStorageEntry) {
  if (rootStorageEntry->bufInfo->bufferScope != pto::AddressSpace::VEC) {
    return rootStorageEntry;
  }
  SmallVector<StorageEntry *> origStorageEntryVec = {rootStorageEntry};
  origStorageEntryVec.insert(origStorageEntryVec.end(),
                             rootStorageEntry->mergedChildren.begin(),
                             rootStorageEntry->mergedChildren.end());

  // reorder storage entrys: dma touched buffers + other buffers + scalar
  // touched buffers
  SmallVector<StorageEntry *> reorderedStorageEntryVec;
  SmallVector<StorageEntry *> touchPipeScalarStorageEntryVec;
  for (auto &storageEntry : origStorageEntryVec) {
    for (auto &buffer : storageEntry->inplaceBuffers) {
      if (dmaFirstPipelineOpt.IsDmaBuffer(buffer)) {
        reorderedStorageEntryVec.push_back(storageEntry);
        break;
      }
      if (dmaFirstPipelineOpt.IsScalarBuffer(buffer)) {
        touchPipeScalarStorageEntryVec.push_back(storageEntry);
        break;
      }
    }
  }
  for (auto &storageEntry : origStorageEntryVec) {
    auto it1 = std::find(reorderedStorageEntryVec.begin(),
                         reorderedStorageEntryVec.end(), storageEntry);
    auto it2 = std::find(touchPipeScalarStorageEntryVec.begin(),
                         touchPipeScalarStorageEntryVec.end(), storageEntry);
    if (it1 == reorderedStorageEntryVec.end() &&
        it2 == touchPipeScalarStorageEntryVec.end()) {
      reorderedStorageEntryVec.push_back(storageEntry);
    }
  }

  reorderedStorageEntryVec.insert(reorderedStorageEntryVec.end(),
                                  touchPipeScalarStorageEntryVec.begin(),
                                  touchPipeScalarStorageEntryVec.end());

  // Ensure that ping pong is continuously plan mem in the multi buffer.
  ReorderContinuousPingPongEntry(reorderedStorageEntryVec);
  StorageEntry *reorderedRootStorageEntry = reorderedStorageEntryVec[0];
  reorderedRootStorageEntry->mergedChildren.clear();
  for (size_t j = 1; j < reorderedStorageEntryVec.size(); ++j) {
    reorderedRootStorageEntry->mergedChildren.push_back(
        reorderedStorageEntryVec[j]);
  }
  return reorderedRootStorageEntry;
}

void MemPlan::ReorderContinuousPingPongEntry(
    SmallVector<StorageEntry *> &storageEntryVec) {
  SmallVector<StorageEntry *> reorderedStorageEntryVec;
  llvm::SmallPtrSet<StorageEntry *, 16> seen;
  auto pushOnce = [&](StorageEntry *e) {
    if (e && seen.insert(e).second)
      reorderedStorageEntryVec.push_back(e);
  };
  for (auto &storageEntry : storageEntryVec) {
    if (seen.count(storageEntry))
      continue;
    pushOnce(storageEntry);
    // Keep the N-buffer siblings adjacent to the primary so spec-level
    // planning and rollback can reason about them as one contiguous region.
    if (storageEntry->multiBufferNum > 1 &&
        !storageEntry->relationOtherBuffers.empty()) {
      for (StorageEntry *rel : storageEntry->relationOtherBuffers)
        pushOnce(rel);
    }
  }
  reorderedStorageEntryVec.swap(storageEntryVec);
}

std::pair<size_t, size_t>
MemPlan::GetBufferSpaceInfo(pto::AddressSpace &space) const {
  switch (space) {
  case pto::AddressSpace::VEC:
    return std::make_pair(ubAlignSize, ubSpaceSize);
  case pto::AddressSpace::MAT:
    return std::make_pair(l1AlignSize, l1SpaceSize);
  case pto::AddressSpace::ACC:
    return std::make_pair(l0cAlignSize, l0cSpaceSize);
  case pto::AddressSpace::LEFT:
    return std::make_pair(l0aAlignSize, l0aSpaceSize);
  case pto::AddressSpace::RIGHT:
    return std::make_pair(l0bAlignSize, l0bSpaceSize);
  case pto::AddressSpace::BIAS:
    return std::make_pair(biasAlignSize, biasSpaceSize);
  case pto::AddressSpace::SCALING:
    return std::make_pair(scalingAlignSize, scalingSpaceSize);
  }

  llvm_unreachable("Temporarily unsupported memory buffer space !");
}

std::pair<size_t, size_t>
MemPlan::GetPlannableBufferSpaceInfo(pto::AddressSpace &space) const {
  auto bufferSpaceInfo = GetBufferSpaceInfo(space);
  auto it = reservedBufferBitsByScope.find(space);
  if (it == reservedBufferBitsByScope.end()) {
    return bufferSpaceInfo;
  }
  if (it->second >= bufferSpaceInfo.second) {
    return std::make_pair(bufferSpaceInfo.first, size_t{0});
  }
  return std::make_pair(bufferSpaceInfo.first,
                        bufferSpaceInfo.second - it->second);
}

LogicalResult MemPlan::MultiSpecPlan(SpecInfo &si, MemBoundList &outline,
                                     PlanRecHis &history, StorageEntry *entry) {
  LogicalResult planResult = failure();
  for (int i = si.specLevel; i >= si.minLevel; i--) {
    planResult = SpecAlloc(outline, history, entry, si, i);
    if (succeeded(planResult)) {
      if (si.childIdx == si.specStartIdx) {
        // In roll back plan, when the specified specStartIdx is reached,
        // the subsequent plan still adopts the maxLevel strategy.
        si.specLevel = si.maxLevel;
      }
      si.childIdx++;
      break;
    }
  }
  return planResult;
}

LogicalResult MemPlan::SpecAlloc(MemBoundList &outline, PlanRecHis &his,
                                 StorageEntry *e, const SpecInfo &si,
                                 int localLevel) {
  if (e == nullptr) {
    // Defensive: a null entry would otherwise crash later when reading
    // `e->alignedConstBits` / `e->bufferLifeVec`.
    return failure();
  }
  if (std::any_of(his.begin(), his.end(),
                  [e](PlanRecord &r) { return r.entry && r.entry == e; })) {
    // If the plan has already been completed, return success directly.
    return success();
  }
  // Zero-sized entries (e.g. degenerate / dynamically-shaped buffers that were
  // statically resolved to 0 bits) cannot meaningfully consume an outline
  // bound. Pin them at offset 0 and report success so the rest of the planner
  // can advance. Mirrors HIVM PlanMemory.cpp behavior.
  if (e->alignedConstBits == 0) {
    e->bitsOffset = 0;
    return success();
  }
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    uint64_t allocOffset = (*start)->offset;
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      // if index & addr are as same as last rollback result,
      // continue to find next result
      if (IsSamePlanAsLastRollBack(allocOffset, e->childIdx, si) ||
          VerifyConflictStage0(e, last)) {
        start = end;
        break;
      }
      if (size < e->alignedConstBits) {
        continue;
      }
      // If SPEC_LEVEL_1, then the address of pong Offset address needs to be
      // allocated.
      uint64_t pongOffset{0};
      if (localLevel == SPEC_LEVEL_1 &&
          VerifyConflictStage1(outline, his, e,
                               OutlineSectionInfo(start, end, size, false),
                               pongOffset)) {
        break;
      }

      if (VerifyConflictStage2(his, e, localLevel, start, outline)) {
        break;
      }
      if (VerifyConflictStage3(his, e, localLevel, start, outline)) {
        break;
      }
      e->bitsOffset = allocOffset;
      UpdateOutline(outline, his, e,
                    OutlineSectionInfo(start, end, size, false), localLevel);

      if (localLevel == SPEC_LEVEL_1) {
        // There is no conflict with the historical plan of buffer life, and
        // the address of the Pong can be assigned.
        PlanRelationPongEntryAddress(pongOffset, e);
        SpecAllocRelationPongEntry(outline, his, e, pongOffset);
      }
      LDBG("APPLY_SPEC_LEVEL:  " << localLevel << "\n");
      bool needRecord =
          allocatedEntry.end() ==
          std::find(allocatedEntry.begin(), allocatedEntry.end(), e);
      if (needRecord) {
        allocatedEntry.push_back(e);
      }
      return success();
    }
  }
  return failure();
}

LoopLikeOpInterface
MemPlan::GetBufferParentLoop(const SmallVector<Value> &buffers) {
  llvm::SmallSet<LoopLikeOpInterface, 1> parentLoopVec;
  for (auto buffer : buffers) {
    if (!buffer.getDefiningOp()) {
      if (!isa<scf::ForOp>(buffer.getParentBlock()->getParentOp()))
        llvm::report_fatal_error("expected loop-carried block argument");
      // Init args and region iter arg are inplace, ignore Region Iter Arg
      // without DefineOp.
      continue;
    }
    LoopLikeOpInterface bufferParentLoop = getParentLoop(buffer);
    if (bufferParentLoop) {
      parentLoopVec.insert(bufferParentLoop);
    } else {
      return nullptr;
    }
  }
  if (parentLoopVec.size() == 1) {
    return *parentLoopVec.begin();
  }
  return nullptr;
}

bool MemPlan::VerifyConflictStage1(MemBoundList &outline, PlanRecHis &his,
                                   StorageEntry *e,
                                   const OutlineSectionInfo &outlineInfo,
                                   uint64_t &pongOffset) {
  if (outlineInfo.mem_start != outlineInfo.mem_end) {
    return true;
  }
  auto reuseBoundStorageEntry = (*outlineInfo.mem_start)->lastStorageEntry;
  if (!reuseBoundStorageEntry) {
    // This area has not been planed, so there is no need to consider it.
    return true;
  }

  StorageEntry *multiRelationPongEntry =
      GetMultiRelationPongEntry(reuseBoundStorageEntry);
  if (multiRelationPongEntry) {
    bool hasRelationSlots =
        !e->relationOtherBuffers.empty() &&
        llvm::all_of(e->relationOtherBuffers, [](StorageEntry *s) {
          return s->bitsOffset != 0;
        });
    if (e->multiBufferNum == kSingleBufferCount ||
        (e->multiBufferNum > 1 && hasRelationSlots)) {
      auto parentLoop1 = GetBufferParentLoop(e->inplaceBuffers);
      auto parentLoop2 =
          GetBufferParentLoop(reuseBoundStorageEntry->inplaceBuffers);
      if (!(parentLoop1 != nullptr && parentLoop2 != nullptr &&
            parentLoop1 == parentLoop2)) {
        // Cannot be reused under the same for.
        return true;
      }
      // There are two situations:
      // 1. Single reuse DB.
      // 2. DB reuse DB.
      pongOffset = multiRelationPongEntry->bitsOffset;
      bool conflict = std::any_of(
          his.begin(), his.end(), [pongOffset, e, this](PlanRecord &r) {
            return this->IsBufferLifeVecConflict(r, pongOffset, e);
          });
      if (!conflict) {
        return false;
      }
    }
  }
  return true;
}

StorageEntry *
MemPlan::GetMultiRelationPongEntry(const StorageEntry *reuseBoundStorageEntry) {
  if (reuseBoundStorageEntry->multiBufferNum > 1 &&
      !reuseBoundStorageEntry->relationOtherBuffers.empty()) {
    StorageEntry *last =
        reuseBoundStorageEntry->relationOtherBuffers.back();
    if (last->bitsOffset != 0) {
      return last;
    }
  }
  auto iter = pingEntry2RelationPongEntry.find(reuseBoundStorageEntry);
  if (iter != pingEntry2RelationPongEntry.end()) {
    // If the reuseBoundStorageEntry itself is single, but has already been
    // reused with db and has an extra pong StorageEntry is added.
    return iter->second.get();
  }
  return nullptr;
}

void MemPlan::SpecAllocRelationPongEntry(MemBoundList &outline, PlanRecHis &his,
                                         StorageEntry *e, uint64_t offset) {
  SmallVector<StorageEntry *, 8> targets;
  if (e->multiBufferNum > 1 && !e->relationOtherBuffers.empty()) {
    for (StorageEntry *rel : e->relationOtherBuffers)
      if (rel->bitsOffset != 0)
        targets.push_back(rel);
  } else {
    auto iter = pingEntry2RelationPongEntry.find(e);
    if (iter != pingEntry2RelationPongEntry.end())
      targets.push_back(iter->second.get());
  }

  for (StorageEntry *pongStorageEntry : targets) {
    uint64_t slotOffset = pongStorageEntry->bitsOffset;
    bool placed = false;
    for (MemBoundListConstIter start = outline.begin();
         start != outline.end() && !placed; ++start) {
      if ((*start)->offset != slotOffset)
        continue;
      uint64_t size = 0;
      for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
        std::shared_ptr<MemoryBound> last = *end;
        size += last->extent;
        if (size < pongStorageEntry->alignedConstBits)
          continue;
        UpdateOutline(outline, his, pongStorageEntry,
                      OutlineSectionInfo(start, end, size, true), SPEC_LEVEL_1);
        placed = true;
        break;
      }
    }
    if (!placed)
      llvm::report_fatal_error("pong storage entry outline not found");
  }
}

bool MemPlan::IsBufferLifeVecConflict(PlanRecord &r, uint64_t offset,
                                      const StorageEntry *e) const {
  if ((r.firstMemBound->offset + r.allExtent > offset) &&
      (r.firstMemBound->offset < offset + e->alignedConstBits)) {
    if (HasSemanticConflict(e, r.firstMemBound->bufferLifeVec))
      return true;
    DenseMap<ValuePair, BufferLife> intersection =
        GetOverlapBufferLife(r.entry->bufferLifeVec, e->bufferLifeVec);
    return !intersection.empty();
  }
  return false;
}

void MemPlan::PlanRelationPongEntryAddress(uint64_t offset, StorageEntry *e) {
  if (e->multiBufferNum == kSingleBufferCount) {
    std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
    entry->bufInfo = e->bufInfo;
    entry->bufferLifeVec = e->bufferLifeVec;
    entry->alignedConstBits = e->alignedConstBits;
    entry->inplaceBuffers = e->inplaceBuffers;
    entry->multiBufferNum = e->multiBufferNum;
    entry->bitsOffset = offset;
    pingEntry2RelationPongEntry[e] = std::move(entry);
    return;
  }
  if (e->relationOtherBuffers.empty())
    return;
  pto::AddressSpace scope = e->bufInfo->bufferScope;
  size_t alignUnit = GetPlannableBufferSpaceInfo(scope).first;

  e->relationOtherBuffers[0]->bitsOffset = offset;
  uint64_t cur = offset;
  for (size_t i = 1; i < e->relationOtherBuffers.size(); ++i) {
    cur = AlignUp(cur + e->relationOtherBuffers[i - 1]->alignedConstBits,
                  alignUnit);
    e->relationOtherBuffers[i]->bitsOffset = cur;
  }
}

bool MemPlan::VerifyConflictStageCommon(
    PlanRecHis &his, const StorageEntry *e, MemBoundListConstIter &start,
    const MemBoundList &outline,
    std::function<bool(const StorageEntry *, const StorageEntry *)>
        conflictChecker) {
  bool touchMemCanUse = false;
  MemBoundListConstIter foundMem;

  for (auto iter = start; iter != outline.end(); ++iter) {
    uint64_t offset = (*iter)->offset;
    bool conflict = std::any_of(
        his.begin(), his.end(), [offset, e, &conflictChecker](PlanRecord &r) {
          return (r.firstMemBound->offset + r.allExtent > offset) &&
                 (r.firstMemBound->offset < offset + e->alignedConstBits) &&
                 conflictChecker(r.entry, e);
        });
    // if conflict, continue finding the first bound that has no conflict
    // if last bound do not meet the size, continue
    if (conflict ||
        ((*iter == outline.back()) && (*iter)->extent < e->alignedConstBits)) {
      continue;
    }
    touchMemCanUse = true;
    foundMem = iter;
    break;
  }

  if (touchMemCanUse) {
    bool conflict = (foundMem != start);
    start = conflict ? --foundMem : start;
    return conflict;
  }
  // if cannot find a bound that has no conflict with current entry,
  return true;
}

bool MemPlan::VerifyConflictStage2(PlanRecHis &his, const StorageEntry *e,
                                   int specLevel, MemBoundListConstIter &start,
                                   const MemBoundList &outline) {
  if (specLevel != SPEC_LEVEL_2) {
    return false;
  }
  // SPEC_LEVEL_2 only blocks reuse for buffers that pipe-conflict *and* live
  // in the same parent loop. Buffers in different loop nests are allowed to
  // share an offset even if they would conflict on a pipe basis - the looser
  // policy compared to SPEC_LEVEL_3.
  return VerifyConflictStageCommon(
      his, e, start, outline,
      [this](const StorageEntry *e1, const StorageEntry *e2) {
        return this->PipeConflictInSameLoop(e1, e2);
      });
}

bool MemPlan::VerifyConflictStage3(PlanRecHis &his, const StorageEntry *e,
                                   int specLevel, MemBoundListConstIter &start,
                                   const MemBoundList &outline) {
  if (specLevel != SPEC_LEVEL_3) {
    return false;
  }
  // SPEC_LEVEL_3 forbids reuse whenever any pipe conflict exists, regardless
  // of loop scope - the most conservative pipe policy and the level
  // MultiSpecPlan attempts first.
  return VerifyConflictStageCommon(
      his, e, start, outline,
      [this](const StorageEntry *e1, const StorageEntry *e2) {
        return this->PipeConflict(e1, e2, this->pipeDmaConflictMap);
      });
}

bool MemPlan::PipeConflictInSameLoop(const StorageEntry *e1,
                                     const StorageEntry *e2) {
  if (e1 == nullptr || e2 == nullptr) {
    return false;
  }
  // Only treat the conflict as fatal when both entries hang off the same
  // parent loop. Distinct loops (or top-level buffers) are deliberately
  // permitted to share an offset under SPEC_LEVEL_2.
  auto parentLoop1 = GetBufferParentLoop(e1->inplaceBuffers);
  auto parentLoop2 = GetBufferParentLoop(e2->inplaceBuffers);
  if (parentLoop1 != parentLoop2) {
    return false;
  }
  return true;
}

bool MemPlan::PipeConflict(const StorageEntry *e1, const StorageEntry *e2,
                           DenseMap<StorageEntryPair, bool> &conflictMap) {
  if (e1 == nullptr || e2 == nullptr) {
    return false;
  }
  auto sePair = std::make_pair(e1, e2);
  auto [iter, isInserted] = conflictMap.try_emplace(sePair, false);
  if (!isInserted) {
    return iter->second;
  }

  for (const Value var1 : e1->inplaceBuffers) {
    for (const Value var2 : e2->inplaceBuffers) {
      bool conflict = dmaFirstPipelineOpt.BufferPipeConflict(var1, var2);
      if (conflict) {
        iter->second = true;
        return true;
      }
    }
  }
  return false;
}

void MemPlan::UpdateOutline(MemBoundList &outline, PlanRecHis &his,
                            StorageEntry *e,
                            const OutlineSectionInfo &outlineInfo,
                            int localLevel) const {
  if (e == nullptr) {
    // Defensive: skip outline mutation when the caller passed a null entry
    // (mirrors HIVM PlanMemory.cpp).
    return;
  }
  auto start = outlineInfo.mem_start;
  MemBoundListConstIter end = outlineInfo.mem_end;
  // outline:
  // |-------start+end-------------|
  // |--head--|--split e--|--tail--|
  uint64_t res = outlineInfo.size - e->alignedConstBits;
  std::shared_ptr<MemoryBound> last = *end;
  ++end;
  std::shared_ptr<MemoryBound> bound;
  SmallVector<std::shared_ptr<MemoryBound>> splitBound;
  // split e, to get Boundbound
  if (splitOutline) {
    // add splitBound by splitting e to section
    AddMemBoundInSectionalWay(e, start, end, splitBound);
  } else {
    // origin outline
    BufferLifeVec life(e->bufferLifeVec.begin(), e->bufferLifeVec.end());
    MergeBufferLife(start, end, life);
    splitBound.emplace_back(std::make_shared<MemoryBound>(
        life, e->bitsOffset, e->alignedConstBits, e));
  }

  // insert tail memory bound
  if (res > 0) {
    bound = std::make_shared<MemoryBound>(last->bufferLifeVec,
                                          last->offset + last->extent - res,
                                          res, last->lastStorageEntry);
    end = outline.insert(end, bound);
  }
  // insert split e memory bound
  for (int i = static_cast<int>(splitBound.size()) - 1; i >= 0; --i) {
    end = outline.insert(end, splitBound[i]);
  }
  // record the current plan of first split entry in his
  his.emplace_back(PlanRecord{localLevel,
                              e->childIdx,
                              res > 0,
                              false,
                              splitBound.size(),
                              e,
                              e->alignedConstBits,
                              splitBound[0],
                              {},
                              outlineInfo.isDirectlyRollback});
  PlanRecord &r = his.back();
  r.replaced.splice(r.replaced.begin(), outline, start, end);
}

void MemPlan::AddMemBoundInSectionalWay(
    StorageEntry *e, MemBoundListConstIter start, MemBoundListConstIter end,
    SmallVector<std::shared_ptr<MemoryBound>> &splitBound) const {
  // |--outline1--|--outline2--|--outline3--|
  // |---------e------------ |
  // |--split e1 -|-split e2-|
  for (auto iter = start; iter != end; ++iter) {
    BufferLifeVec life(e->bufferLifeVec.begin(), e->bufferLifeVec.end());
    life.insert(life.end(), (*iter)->bufferLifeVec.begin(),
                (*iter)->bufferLifeVec.end());
    // merge the buffer life
    MergeBufferVec(life);
    // get the extent
    uint64_t size = 0;
    if (std::distance(start, iter) == std::distance(start, end) - 1) {
      // deal with the last split e2
      size = e->bitsOffset + e->alignedConstBits - (*iter)->offset;
    } else {
      size = (*iter)->extent;
    }
    splitBound.emplace_back(
        std::make_shared<MemoryBound>(life, (*iter)->offset, size, e));
  }
}

inline void MemPlan::MergeBufferLife(MemBoundList::const_iterator start,
                                     MemBoundList::const_iterator end,
                                     BufferLifeVec &newLife) const {
  size_t size = 0;
  for (auto it = start; it != end; ++it) {
    size += (*it)->bufferLifeVec.size();
  }
  newLife.reserve(size);
  for (auto it = start; it != end; ++it) {
    newLife.insert(newLife.end(), (*it)->bufferLifeVec.begin(),
                   (*it)->bufferLifeVec.end());
  }
  MergeBufferVec(newLife);
}

void MemPlan::MergeBufferVec(BufferLifeVec &bufferLife) const {
  if (bufferLife.empty()) {
    return;
  }
  BufferLifeVec mergedLife;
  mergedLife.reserve(bufferLife.size());
  // sort life by alloc and free time
  std::sort(bufferLife.begin(), bufferLife.end(), CompareBufferLife());
  int start = bufferLife[0]->allocTime;
  int end = bufferLife[0]->freeTime;
  auto buffer = bufferLife[0]->buffer;
  // merge life
  for (size_t i = 1; i < bufferLife.size(); i++) {
    auto &life = bufferLife[i];
    if (life->allocTime <= end + 1) {
      end = end < life->freeTime ? life->freeTime : end;
    } else {
      mergedLife.emplace_back(std::make_unique<BufferLife>(buffer, start, end));
      buffer = life->buffer;
      start = life->allocTime;
      end = life->freeTime;
    }
  }
  mergedLife.emplace_back(std::make_unique<BufferLife>(buffer, start, end));
  bufferLife.swap(mergedLife);
}

bool MemPlan::IsSamePlanAsLastRollBack(uint64_t allocOffset, int curChildIdx,
                                       const SpecInfo &si) const {
  return curChildIdx == si.rollbackIdx && allocOffset == si.rollbackAddr;
}

// spec_level == SPEC_LEVEL_0
inline bool
MemPlan::VerifyConflictStage0(StorageEntry *e,
                              const std::shared_ptr<MemoryBound> &last) {
  if (HasSemanticConflict(e, last->bufferLifeVec))
    return true;
  // level_0: offset = 0, offset means life distance
  DenseMap<ValuePair, BufferLife> intersection =
      GetOverlapBufferLife(e->bufferLifeVec, last->bufferLifeVec);
  return !intersection.empty();
}

// verify two buffer life vectors is conflict or not
// The key pair looks like the following diagram
// indicate that var1 is generated later than var2.
//                                     buffer2
//                               --- PLAN_time
//   buffer1       intersected   | |
//                 buffer_life   | |
// PLAN_time ---    lo ---      ---
//            | |       ---      | |
//            | |       ---      | |
//            ---    hi ---      --- free_time
//            | |
//            | |
// free_time  ---
// Meantime, the overlap is the intersected buffer_life.
DenseMap<ValuePair, BufferLife>
MemPlan::GetOverlapBufferLife(const BufferLifeVec &b1,
                              const BufferLifeVec &b2) const {
  DenseMap<ValuePair, BufferLife> intersection;
  size_t i = 0;
  size_t j = 0;
  size_t b1Len = b1.size();
  size_t b2Len = b2.size();
  if (b1Len == 0 || b2Len == 0) {
    return intersection;
  }
  while (i < b1Len && j < b2Len) {
    auto lo = std::max(b1[i]->allocTime, b2[j]->allocTime);
    auto hi = std::min(b1[i]->freeTime, b2[j]->freeTime);
    if (lo <= hi) {
      BufferLife bufferLife(nullptr, lo, hi);
      ValuePair key =
          lo == b1[i]->allocTime && hi == b2[j]->freeTime
              ? std::make_pair(b1[i]->buffer,
                               b2[j]->buffer) // case in the diagram
              : std::make_pair(b2[j]->buffer, b1[i]->buffer); // opposing case
      intersection.try_emplace(key, bufferLife);
    }
    if (b1[i]->freeTime < b2[j]->freeTime) {
      i += 1;
    } else {
      j += 1;
    }
  }
  return intersection;
}

PlanStatus MemPlan::ApplyFailStrategy(StatusWrapper &statusWrapper,
                                      const size_t maxBits) {
  RollBackForAllocFail(statusWrapper, maxBits);
  // second class rollback, level 1 --> 0
  if (statusWrapper.si->specLevel > SPEC_LEVEL_0 &&
      statusWrapper.si->childIdx >= 0) {
    statusWrapper.si->specLevel--;
    return PlanStatus::CONTINUE_PLAN;
  }
  if (!splitOutline) {
    // roll back to origin again, enable split outline.
    splitOutline = true;
    return PlanStatus::RESTART_NEW_PLAN;
  }
  return PlanStatus::PLAN_FAILED;
}

void MemPlan::ReportAllocatedEntryDebugInfo(StorageEntry *rootStorageEntry) {
  auto printRecord = [this](const StorageEntry *entry) {
    uint64_t needByte =
        (entry->alignedConstBits + kBitsToByte - 1) / kBitsToByte;
    uint64_t offsetByte =
        (entry->bitsOffset + kBitsToByte - 1) / kBitsToByte;
    ReportCurEntryDebugInfo(entry);
    LDBG(", offset: " << offsetByte);
    LDBG(", extent: " << needByte);
    LDBG(", buffer life: ");
    for (auto &bufferLife : entry->bufferLifeVec) {
      LDBG("[" << bufferLife->allocTime << "-" << bufferLife->freeTime
               << "], ");
    }
  };
  LDBG("--------------------------BUFFER ALLOCATE "
       "START-------------------------- "
       << "\n"
       << "\n");
  LDBG("  BUFFER ALLOCATE START: UB"
       << "\n");
  if (!allocatedEntry.empty()) {
    for (auto &entry : allocatedEntry) {
      printRecord(entry);
      LDBG("\n");
    }
    size_t num = allocatedEntry.size() - 1;
    if (rootStorageEntry->mergedChildren.size() <= num)
      return;
    const StorageEntry *failedSe = rootStorageEntry->mergedChildren[num];
    printRecord(failedSe);
    LDBG("alloc fail,because exceed bound of memory \n"
         << "  BUFFER ALLOCATE END \n");
    LDBG("\n"
         << "--------------------------BUFFER ALLOCATE "
            "END-------------------------- "
         << "\n");
  }
}

LogicalResult MemPlan::InitMemSpecsFromModule(func::FuncOp funcOp) {
  struct MemSpec {
    int ubSpaceSize;
    int l1SpaceSize;
    int l0aSpaceSize;
    int l0bSpaceSize;
    int l0cSpaceSize;
    int ubAlignSize;
    int l1AlignSize;
    int l0cAlignSize;
    int l0aAlignSize;
    int l0bAlignSize;
    int biasAlignSize;
    int biasSpaceSize;
    int scalingAlignSize;
    int scalingSpaceSize;
  };

  const MemSpec kA3 = {
      1572864, 4194304, 524288, 524288, 1048576, 256, 256,
      4096,    4096,    4096,   256,    524288, 256, 1572864};
  const MemSpec kA5 = {
      2031616, 4194304, 524288, 524288, 2097152, 256, 256,
      4096,    4096,    4096,   256,    524288, 256, 2031616};

  auto applySpec = [this](const MemSpec &spec) {
    ubSpaceSize = spec.ubSpaceSize;
    l1SpaceSize = spec.l1SpaceSize;
    l0aSpaceSize = spec.l0aSpaceSize;
    l0bSpaceSize = spec.l0bSpaceSize;
    l0cSpaceSize = spec.l0cSpaceSize;
    ubAlignSize = spec.ubAlignSize;
    l1AlignSize = spec.l1AlignSize;
    l0cAlignSize = spec.l0cAlignSize;
    l0aAlignSize = spec.l0aAlignSize;
    l0bAlignSize = spec.l0bAlignSize;
    biasAlignSize = spec.biasAlignSize;
    biasSpaceSize = spec.biasSpaceSize;
    scalingAlignSize = spec.scalingAlignSize;
    scalingSpaceSize = spec.scalingSpaceSize;
  };

  // Default to a3.
  applySpec(kA3);

  // --pto-arch options:
  // a3 -> default memory spec
  // a5 -> override memory spec
  if (isTargetArchA5(getTopLevelModuleOp(funcOp))) {
    applySpec(kA5);
  }
  return success();
}

void MemPlan::RollBackForAllocFail(StatusWrapper &statusWrapper,
                                   const size_t maxBits) {
  while (ContinueRollBack(statusWrapper)) {
    RollBackForAllocFailInner(statusWrapper, maxBits);
  }
}

bool MemPlan::ContinueRollBack(const StatusWrapper &statusWrapper) const {
  return (!statusWrapper.hasEnoughRollBackSize) &&
         (!statusWrapper.history.empty() && (!statusWrapper.outline.empty()));
}

void MemPlan::RollBackForAllocFailInner(StatusWrapper &statusWrapper,
                                        const size_t maxBits) {
  auto &si = statusWrapper.si;
  if (si->childIdx > si->specStartIdx) {
    si->specStartIdx = si->childIdx;
  }
  // Check whether the container is empty before accessing "history"
  while (!statusWrapper.history.empty()) {
    PlanRecord r =
        RollbackOutline(statusWrapper.history, statusWrapper.outline);
    auto iter = pingEntry2RelationPongEntry.find(r.entry);
    if (iter != pingEntry2RelationPongEntry.end()) {
      pingEntry2RelationPongEntry.erase(iter);
    }
    if (r.isDirectlyRollback ||
        (r.entry->multiBufferNum > 1 &&
         r.entry->relationOtherBuffers.empty())) {
      continue;
    }
    si->childIdx = r.childIdx;
    si->specLevel = r.specLevel;
    if (si->specLevel > si->minLevel) {
      // record rollback info: index and address
      si->rollbackAddr =
          si->childIdx == -1
              ? UINT64_MAX
              : statusWrapper.RootE->mergedChildren[si->childIdx]->bitsOffset;
      si->rollbackIdx = si->childIdx;
      if (statusWrapper.si->rollbackAddr + statusWrapper.alignedConstBits >
          maxBits) {
        continue;
      }
      statusWrapper.hasEnoughRollBackSize = true;
      break;
    }
  }
}

PlanRecord MemPlan::RollbackOutline(PlanRecHis &history,
                                    MemBoundList &outline) const {
  auto r = history.back();
  auto it = std::find(outline.begin(), outline.end(), r.firstMemBound);
  // |--head--|--split entry--|--tail--|
  // erase head
  if (r.headed) {
    it--;
    it = outline.erase(it);
  }
  // erase split entry
  for (size_t i = 0; i < r.splitNums; i++) {
    it = outline.erase(it);
  }
  // erase tail
  if (r.tailed) {
    it = outline.erase(it);
  }
  // restore outline and replaced
  outline.splice(it, r.replaced);
  history.pop_back();
  return r;
}

namespace {
struct PlanMemoryPass : public mlir::pto::impl::PlanMemoryBase<PlanMemoryPass> {
public:
  explicit PlanMemoryPass(const mlir::pto::PlanMemoryOptions &planMemoryOption)
      : PlanMemoryBase(planMemoryOption) {}

  void runOnOperation() override;

private:
  void populateBufferAddressToAllocOp(
      RewritePatternSet &patterns,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets) {
    if (this->memMode == MemPlanMode::LOCAL_MEM_PLAN) {
      patterns.add<MemrefAllocaOpToPointerCastOpPattern>(patterns.getContext(),
                                                         buffer2Offsets);
    }
  }
};
} // namespace

void PlanMemoryPass::runOnOperation() {
  // The plan-memory algorithm is sensitive to the order in which liveness
  // candidates are visited. To dampen that sensitivity (and avoid spurious
  // overflows on order-dependent corner cases) the pass retries planning up
  // to `kPlanRetryCount` times with deterministic but distinct shuffle seeds.
  // - The first attempt (seed=0) preserves the original PTO single-shot
  //   behavior: stable sort, no shuffle.
  // - Subsequent attempts use seed=attempt to permute candidates inside
  //   `MemLivenessAnalysis::OpKillHandle`, exposing alternate gen/kill
  //   orderings.
  // - Diagnostics from `MemPlan::plan` are suppressed on every attempt
  //   except the last, so a recoverable failure on attempt N does not pollute
  //   the user's error output when attempt N+1 succeeds.
  constexpr int kPlanRetryCount = 20;

  ModuleOp moduleOp = getOperation();
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    ReserveBufferPlans reservePlans;
    if (this->memMode == MemPlanMode::LOCAL_MEM_PLAN &&
        failed(analyzeReserveBufferPlans(funcOp, reservePlans))) {
      return signalPassFailure();
    }
    if (this->memMode == MemPlanMode::LOCAL_MEM_PLAN) {
      for (ReserveBufferPlan &reservePlan : reservePlans) {
        if (reservePlan.mode != ReserveBufferMode::Manual)
          continue;
        reservePlan.reserveOp.emitOpError(
            "pto.reserve_buffer with explicit 'base' (auto = false) is not "
            "supported in PlanMemory; use --pto-level=level3 or set auto = true");
        return signalPassFailure();
      }
    }

    DenseMap<Value, SmallVector<uint64_t>> plannedBuffer2Offsets;
    std::map<Value, BufferInfo, ValueComparator> plannedBufferInfos;
    bool planSucceeded = false;

    for (int attempt = 0; attempt < kPlanRetryCount; ++attempt) {
      LDBG("Memory planning attempt " << attempt + 1 << "/" << kPlanRetryCount
                                      << "\n");

      MemLivenessAnalysis memLiveness(funcOp, this->memMode,
                                      /*randomSeed=*/static_cast<uint32_t>(attempt));
      memLiveness.build();

      MemPlan memPlan(this->memMode, this->enableGlobalReuse,
                      this->enablePrintMemoryAllocatedSize,
                      this->restrictInplaceAsISA);
      if (failed(memPlan.InitMemSpecsFromModule(funcOp))) {
        return signalPassFailure();
      }
      memPlan.func_ = funcOp;
      memPlan.SetLinearOperation(memLiveness.linearOperation);
      // Snapshot bufferInfos before SetBufferInfos copies it into memPlan, so
      // that on success we can hand them to assignAutoReserveBufferBases
      // without keeping `memLiveness` alive past the loop iteration.
      auto bufferInfosSnapshot = memLiveness.bufferInfos;
      memPlan.SetBufferInfos(memLiveness.bufferInfos);
      memPlan.SetBuffer2Life(memLiveness.buffer2Life);
      memPlan.SetGenKillMap(memLiveness.genKillMap);
      memPlan.SetBuffer2MultiNum(memLiveness.buffer2MultiNum);
      memPlan.SetInplacePairList(memLiveness.inplacePairList);
      memPlan.SetSemanticConflictPairs(memLiveness.semanticConflictPairs);
      memPlan.SetStableValueOrder(std::move(memLiveness.stableValueOrder));
      memPlan.SetReservedBufferBitsByScope(
          collectAutoReserveBufferBitsByAddressSpace(reservePlans));

      const bool isLastAttempt = attempt == kPlanRetryCount - 1;
      if (succeeded(memPlan.plan(/*emitErrors=*/isLastAttempt))) {
        plannedBuffer2Offsets = memPlan.GetBuffer2Offsets();
        plannedBufferInfos = std::move(bufferInfosSnapshot);
        planSucceeded = true;
        break;
      }
      if (isLastAttempt) {
        // Errors were already emitted by the final memPlan.plan() call.
        return signalPassFailure();
      }
    }

    if (!planSucceeded) {
      // Defensive: should be unreachable because the loop above either breaks
      // on success or signals failure on the last attempt.
      return signalPassFailure();
    }

    // Keep reserve_buffer allocation outside the core MemPlan algorithm:
    // normal local buffers are planned first, then reserve_buffer claims one
    // aligned hole in its target address space.
    if (this->memMode == MemPlanMode::LOCAL_MEM_PLAN &&
        failed(assignAutoReserveBufferBases(reservePlans, plannedBufferInfos,
                                            plannedBuffer2Offsets))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    populateBufferAddressToAllocOp(patterns, plannedBuffer2Offsets);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass>
mlir::pto::createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption) {
  return std::make_unique<PlanMemoryPass>(planMemoryOption);
}
