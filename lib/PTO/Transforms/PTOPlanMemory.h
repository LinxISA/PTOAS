// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PlanMemory.h ----Plan Buffer Memory Address ------------------------===//
//===----------------------------------------------------------------------===//
#ifndef PTO_PLAN_MEMORY_H
#define PTO_PLAN_MEMORY_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/MultiBuffer.h"
#include "OptMemPlanForPipeline.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallSet.h"



#include <functional>
#include <list>
#include <random>

namespace mlir {
namespace pto {

// Value comparator for std::map
inline bool isLessValue(const Value &a, const Value &b) {
  return a.getImpl() < b.getImpl();
}

struct ValueComparator {
  bool operator()(const Value &a, const Value &b) const {
    return isLessValue(a, b);
  }
};

using StableValueOrderMap = DenseMap<Value, uint32_t>;

/// Various states when collecting gen-kill.
enum BufferStatus { UNDEFFINED = 0, DEFFINED, GENED, KILLED };

/// Pair of inplace Value.
using ValuePair = std::pair<Value, Value>;

/// Result status after plan memory.
enum PlanStatus {
  PLAN_SUCCESS = 0,
  RESTART_NEW_PLAN,
  CONTINUE_PLAN,
  PLAN_FAILED
};

/// Memory reuse plan mode can be achieved without conflicting life
/// intervals, offset = 0.
constexpr const int SPEC_LEVEL_0 = 0;

/// By increasing the lifespan by 1 without conflict,
/// memory reuse plan mode can be implemented to avoid dependency on
/// continuous instructions caused by plan, offset = 1.
constexpr const int SPEC_LEVEL_1 = 1;

/// pipe conflict opt for buffers in the same parent loop only.
/// Less restrictive than SPEC_LEVEL_3: only blocks reuse when the conflicting
/// buffers share a parent loop, allowing reuse across distinct loop nests.
constexpr const int SPEC_LEVEL_2 = 2;

/// pipe conflict opt for any pipe conflict (most conservative reuse policy).
/// This is the initial level the planner attempts; it falls back to less
/// restrictive levels (2 -> 1 -> 0) when allocation fails.
constexpr const int SPEC_LEVEL_3 = 3;

/// plan information of alloc buffer.
struct BufferInfo {
  /// Alloc operation of buffer.
  Operation *operation{nullptr};
  /// Space corresponding to buffer.
  pto::AddressSpace bufferScope;
  /// The size required for the buffer.
  int64_t constBits{0};
  /// The type of element in the buffer.
  Type bufferType;
  /// Alias buffer does not participate in inplace.
  /// e.g :
  ///  alloc A
  ///  for(arg = A) :
  ///    alloc B
  ///    ...
  ///    alloc C
  ///    vadd ins(B, D), outs(C)
  ///    scf.yield C
  /// Put (A, C) in inplacePairList and inplace them together in next plan
  /// memory. Because here do not union the lifetime of A and C, just set the
  /// ignoreInplace of A and C to be true so that A will not be inplaced with
  /// other buffer due to wrong lifetime.
  /// TODO: Modify the lifetime of A and C and allow them to be inplaced further
  bool ignoreInplace{false};
};

/// linear operation info.
struct OpInfo {
  OpInfo(Operation *operation, int index)
      : operation(operation), index(index) {}
  Operation *operation{nullptr};
  int index{0};
};

struct GenKillEntry {
  /// record the gen operands, namely the operand buffer that is firstly written
  /// by operation.
  SmallVector<Value> gen;

  /// record the kill operands, namely the operand buffer that is last read by
  /// operation.
  SmallVector<Value> kill;
};

/// Record buffer life interval information.
struct BufferLife {
  BufferLife(Value buffer, int64_t start, int64_t end)
      : buffer(buffer), allocTime(start), freeTime(end) {}
  BufferLife(Value buffer) : buffer(buffer) {}
  /// buffer value.
  Value buffer;
  /// the buffer allocate time.
  int64_t allocTime{-1};
  /// the buffer free time.
  int64_t freeTime{-1};
};

/// a list of buffer life for a given storage entry
using BufferLifeVec = SmallVector<std::shared_ptr<BufferLife>>;

struct StorageEntry {
  /// The the buffer plan info.
  BufferInfo *bufInfo{nullptr};

  /// The lifespan of a buffer.
  BufferLifeVec bufferLifeVec;

  /// The children of this entry, not including itself.
  SmallVector<StorageEntry *> mergedChildren;

  /// The current entry needs to be planed an aligned size.
  uint64_t alignedConstBits{0};

  /// The current entry's child index.
  int childIdx;

  /// The starting address after the current entry allocation.
  uint64_t bitsOffset{0};

  /// Allocs that inplace buffer this entry.
  SmallVector<Value> inplaceBuffers;

  /// Extra physical slots for multi-buffering (N-1 entries; slot 0 is `this`).
  /// Pointers reference sibling `StorageEntry` objects owned in `MemPlan`.
  SmallVector<StorageEntry *, 8> relationOtherBuffers;

  /// The number of multibuffer optimization.
  /// note: default 1 which means single buffer and does not do multibuffer
  /// optimization.
  uint32_t multiBufferNum{1};

  /// Get Bufferlife by vaule
  std::shared_ptr<BufferLife> GetBufferLifeByValue(const Value v) const;
};

struct MemoryBound {
  MemoryBound(BufferLifeVec life, uint64_t o, uint64_t e, const StorageEntry *s)
      : bufferLifeVec(std::move(life)), offset(o), extent(e),
        lastStorageEntry(s) {}
  /// collection of buffer plan and free time which use this Memory
  BufferLifeVec bufferLifeVec;
  /// offset of tagged memory
  uint64_t offset;
  /// extent of this bound
  uint64_t extent;
  /// always record storage entry of last plan
  const StorageEntry *lastStorageEntry;
};
using MemBoundList = std::list<std::shared_ptr<MemoryBound>>;
using MemBoundListConstIter = MemBoundList::const_iterator;

/// record of buffer plan. used for speculative rollback
struct PlanRecord {
  /// speculative level of this plan
  int specLevel;
  /// child index
  int childIdx;
  /// if this plan split last memory bound
  bool tailed;
  /// if this plan has bank offset memory bound
  bool headed;
  /// split number of entry
  size_t splitNums;
  /// record the entry for bank info
  StorageEntry *entry;
  /// the whole extent,add all the split e together
  uint64_t allExtent;
  /// inserted memory bound node
  std::shared_ptr<MemoryBound> firstMemBound;
  /// replaced memory bound node
  MemBoundList replaced;
  /// When the current PlanRecord is rolled back, it must be rolled back
  /// directly.
  bool isDirectlyRollback;
};

using PlanRecHis = SmallVector<PlanRecord>;

struct SpecInfo {
  /// Initial / "ceiling" level. Defaults to SPEC_LEVEL_3 so the planner starts
  /// with the most conservative pipe-conflict policy and degrades to lower
  /// levels (2 -> 1 -> 0) on failure, matching the HIVM reference behavior.
  int maxLevel = SPEC_LEVEL_3;
  int minLevel = SPEC_LEVEL_0;
  int specLevel = SPEC_LEVEL_3;
  int childIdx = -1;
  int specStartIdx = 0;
  int rollbackIdx = -1;
  uint64_t rollbackAddr = UINT64_MAX;
};

struct OutlineSectionInfo {
  OutlineSectionInfo() = default;
  OutlineSectionInfo(MemBoundListConstIter &start, MemBoundListConstIter &end,
                     uint64_t s, bool isDirectlyRollback)
      : mem_start(start), mem_end(end), size(s),
        isDirectlyRollback(isDirectlyRollback) {}
  /// The start of memory plan
  MemBoundListConstIter mem_start;
  /// The end of memory plan
  MemBoundListConstIter mem_end;
  /// The size of memory plan
  uint64_t size{0};
  /// When the current PlanRecord is rolled back, it must be rolled back
  /// directly.
  bool isDirectlyRollback;
};

/// comparator of buffer life
struct CompareBufferLife {
  bool operator()(const std::shared_ptr<BufferLife> &lhs,
                  const std::shared_ptr<BufferLife> &rhs) const {
    if (lhs->allocTime == rhs->allocTime) {
      return lhs->freeTime < rhs->freeTime;
    }
    return lhs->allocTime < rhs->allocTime;
  }
};

struct StatusWrapper {
  /// Is it enough to roll back
  bool hasEnoughRollBackSize;
  /// The size required for the buffer
  uint64_t alignedConstBits;
  /// spec info
  SpecInfo *si;
  /// current outline info
  MemBoundList &outline;
  /// current history plan info
  PlanRecHis &history;
  /// for origin e StorageEntry
  StorageEntry *RootE;
};

class MemLivenessAnalysis {
public:
  /// `randomSeed` controls the deterministic shuffle of liveness candidates
  /// during gen/kill collection. The default seed of 0 preserves the
  /// stable-sort ordering of the original PTO behavior; non-zero seeds
  /// (driven by the retry loop in PlanMemoryPass) explore alternative
  /// orderings to recover from order-sensitive planning failures.
  MemLivenessAnalysis(func::FuncOp func, MemPlanMode planMode,
                      uint32_t randomSeed = 0)
      : func_(func), planMode(planMode), randomSeed(randomSeed),
        randomGenerator(randomSeed) {}

  void build();

  /// linear operation info.
  SmallVector<std::unique_ptr<OpInfo>> linearOperation;

  /// map from buffer value to its buffer information.
  std::map<Value, BufferInfo, ValueComparator> bufferInfos;

  /// stable IR order for Values used to keep memory planning deterministic.
  StableValueOrderMap stableValueOrder;

  /// map from buffer to its lifetime intervals.
  DenseMap<Value, BufferLifeVec> buffer2Life;

  /// map from operation to its gen and kill buffer.
  DenseMap<OpInfo *, GenKillEntry> genKillMap;

  /// record the map from the buffer to its number of buffer if it does
  /// multibuffer optimization.
  /// note: the map only record the buffer which do multi buffer
  /// optimization and ignore single buffer.
  DenseMap<Value, uint32_t> buffer2MultiNum;

  /// record inplace pair list.
  SmallVector<ValuePair> inplacePairList;

  /// record semantic conflict pair list.
  SmallVector<ValuePair> semanticConflictPairs;

  /// now plan mode is LOCAL_MEM_PLAN.
  bool isLocalMemPlan() const;

  /// now plan mode is GLOBAL_WORKSPACE_PLAN.
  bool isGlobalWorkSpaceMemPlan() const;

private:
  /// Read `pto.multi_buffer` on memref.alloc and fill `buffer2MultiNum`.
  void collectMultiBufferAnnotations();

  void RecursionIR(Region *region, Liveness live);

  /// Get the buffer used within the loop and defined outside the loop.
  SmallVector<Value> GetLiveBuffersInLoop(scf::ForOp forOp, Liveness live);

  /// Check whether a loop-live buffer can be generated by its first overwrite
  /// inside the loop instead of being generated at the loop entry.
  bool CanDelayLoopEntryGenUntilFirstWrite(scf::ForOp forOp, Value buffer);

  /// Return true if op directly uses or effects any alias buffer.
  bool OperationDirectlyTouchesAnyAlias(Operation *op,
                                        const SetVector<Value> &aliasBuffers) const;

  /// Return true if op or any nested op touches any alias buffer.
  bool OperationOrNestedRegionTouchesAnyAlias(
      Operation *op, const SetVector<Value> &aliasBuffers) const;

  /// Return true if op directly reads any alias buffer.
  bool OperationReadsAnyAlias(Operation *op,
                              const SetVector<Value> &aliasBuffers) const;

  /// Return true if op fully overwrites one alias as a DPS init without first
  /// reading any alias in the same alias set.
  bool IsWriteOnlyDpsInitForAlias(Operation *op,
                                  const SetVector<Value> &aliasBuffers) const;

  /// Return true if the next touch after op redefines the buffer before read.
  bool CanKillBeforeNextOverwrite(Operation *op,
                                  const SetVector<Value> &aliasBuffers);

  /// Return true if a killed buffer can start a new lifetime at op.
  bool CanRegenerateBufferAtOp(Operation *op, Value buffer);

  /// Return the operation that actually generated the buffer lifetime.
  Operation *GetBufferGenOp(Value buffer) const;

  /// Update for Op tensor init args and tensor result args alias info.
  void UpdateInitAndResAlias(DestinationStyleOpInterface dstStyleOp);

  /// Recursive operation for.
  void RecursiveForOp(scf::ForOp forOp, Liveness live);

  /// Update for Op init args region iter args alias info.
  void UpdateForOpInitArgsAlias(scf::ForOp forOp);

  /// Update forOp result buffer/region iter arg/yielded buffer args alias info.
  void UpdateForOpBufferAlias(scf::ForOp forOp);

  /// Recursive operation if.
  void RecursiveIfOp(scf::IfOp ifOp, Liveness live);

  /// Update buffer alias information for ifop.
  void UpdateIfOpBufferAlias(scf::IfOp ifOp, scf::YieldOp yieldOp);

  /// Update and obtain op info information.
  OpInfo *UpdateLinearOperation(Operation *op);

  /// Obtain all information about the buffer.
  void UpdateOpBufferInfo(Operation *op, const ValueRange &results);

  /// Generate buffer info.
  BufferInfo GenerateBufferInfo(Operation *op, Value operand);

  /// Obtain the buffer info of plan operation.
  BufferInfo GetBufferInfo(Operation *op, Value operand,
                           pto::AddressSpace bufferScope);

  /// Process gen buffer based on the result value of op.
  void UpdateOpGenInfo(OpInfo *opInfo, const ValueRange &results);

  /// Update normal operand gen information on buffer.
  void UpdateOperandGenInfo(OpInfo *opInfo, Value operand);

  /// Update temp buffer for DestinationStyleOpInterface op.
  void UpdateOpTempGenInfo(OpInfo *opInfo);

  /// Update the relationship of buffer aliases.
  void UpdateBufferAlias(Value buffer, Value aliasBuffer,
                         bool isIgnoreInplace = false);

  /// Return the union of set1 and set2.
  SetVector<Value> Union(SetVector<Value> set1, SetVector<Value> set2);

  /// Get alias buffer information.
  SetVector<Value> GetAliasBuffers(Value aliasBuffer);

  /// Check whether there is an unknown operation with buffer
  /// information.
  LogicalResult CheckIfUnknownOpTouchBuffer(Operation *op) const;

  /// Determine whether the current operation can be skipped.
  bool isSkippableOp(Operation *op) const;

  /// Update store op information.
  void UpdateStoreOpInfo(OpInfo *opInfo, const Value storeValue, Liveness live);

  /// Check if it is local buffer with memory space
  LogicalResult CheckLocalBufferAllocOp(Operation *op) const;

  /// kill buffer handle.
  void OpKillHandle(OpInfo *opInfo, Liveness live, Block *block);

  /// Process kill buffer based on the result live of op.
  void UpdateOpKillInfo(OpInfo *opInfo, Value operand, Liveness live);

  /// Have all alias buffer been killed.
  bool AllDeadAfter(Operation *op, SetVector<Value> aliasVec,
                    Liveness live) const;

  /// Determine whether two operation are in the same block.
  bool IsInSameBlock(Operation *op1, Operation *op2) const;

  /// Generate buffer's life time.
  void GenerateBufferLife();

  /// initialize the buffers that must be inplaced together
  /// namely, the alias buffers of memref.alloc,
  /// e.g. for iter arg and for yield.
  void InitializeInplacePairList();

  /// Record semantic non-reuse pairs for buffers that may be used
  /// simultaneously inside one instruction, such as scratch and dst.
  void RecordSemanticConflict(Value lhs, Value rhs);

  func::FuncOp func_;

  /// different mode for mem plan.
  MemPlanMode planMode;

  /// Gen-kill status corresponding to buffer.
  DenseMap<Value, BufferStatus> buffer2status;

  /// Operation where the current buffer lifetime was generated.
  DenseMap<Value, Operation *> buffer2GenOp;

  /// Buffers whose loop-entry generation was delayed to their first write in
  /// the loop body.
  DenseMap<Value, bool> delayedLoopEntryGenBuffers;

  /// map on buffer alias
  DenseMap<Value, SetVector<Value>> buffer2AliasVec;

  int seqIndex{0};

  /// Deterministic shuffle seed forwarded into liveness collection so the
  /// PlanMemory retry loop can sample different gen/kill orderings.
  uint32_t randomSeed{0};

  /// Random generator used by `getShuffledRange` for deterministic shuffles.
  std::mt19937 randomGenerator;

public:
  /// Return a copy of `range` deterministically shuffled with
  /// `randomGenerator`. When `randomSeed == 0` the shuffle is skipped so the
  /// first attempt preserves the stable order used by single-shot PTO runs.
  template <typename RangeT> RangeT getShuffledRange(const RangeT &range) {
    RangeT rangeClone = range;
    if (randomSeed == 0)
      return rangeClone;
    std::shuffle(rangeClone.begin(), rangeClone.end(), randomGenerator);
    return rangeClone;
  }
};

/// Pair of StorageEntry.
using StorageEntryPair = std::pair<const StorageEntry *, const StorageEntry *>;

class MemPlan {
public:
  MemPlan(MemPlanMode planMode, bool enableGlobalReuse, bool enablePrintMemoryAllocatedSize,
          bool restrictInplaceAsISA)
      : planMode(planMode), enableGlobalReuse(enableGlobalReuse),
        enablePrintMemoryAllocatedSize(enablePrintMemoryAllocatedSize),
        restrictInplaceAsISA(restrictInplaceAsISA) {}

  /// Run the memory-planning algorithm. When `emitErrors` is false, failure
  /// diagnostics are suppressed; this lets the PlanMemoryPass retry loop swallow
  /// intermediate failures and only surface errors on the final attempt.
  LogicalResult plan(bool emitErrors = true);

  /// Get buffer2Offsets
  inline DenseMap<Value, SmallVector<uint64_t>> GetBuffer2Offsets() {
    return buffer2Offsets;
  }

  inline void
  SetLinearOperation(SmallVector<std::unique_ptr<OpInfo>> &linearOp) {
    linearOperation = std::move(linearOp);
  };

  inline void
  SetBufferInfos(std::map<Value, BufferInfo, ValueComparator> bufsInfo) {
    bufferInfos = bufsInfo;
  }

  inline void SetBuffer2Life(DenseMap<Value, BufferLifeVec> buf2Life) {
    buffer2Life = std::move(buf2Life);
  }

  inline void SetGenKillMap(DenseMap<OpInfo *, GenKillEntry> gkMap) {
    genKillMap = gkMap;
  }

  inline void SetBuffer2MultiNum(DenseMap<Value, uint32_t> buf2MulBufNum) {
    buffer2MultiNum = buf2MulBufNum;
  }

  inline void SetInplacePairList(SmallVector<ValuePair> inplaceList) {
    inplacePairList = inplaceList;
  }

  inline void SetSemanticConflictPairs(SmallVector<ValuePair> conflictPairs) {
    semanticConflictPairs = std::move(conflictPairs);
  }

  inline void SetStableValueOrder(StableValueOrderMap valueOrder) {
    stableValueOrder = std::move(valueOrder);
  }

  inline void
  SetReservedBufferBitsByScope(DenseMap<pto::AddressSpace, uint64_t> reservedBits) {
    reservedBufferBitsByScope = std::move(reservedBits);
  }

  /// Setup the device's storage specs
  LogicalResult InitMemSpecsFromModule(func::FuncOp funcOp);

  func::FuncOp func_;

private:
  /// different mode for mem plan.
  MemPlanMode planMode;

  /// Enable global workspace reuse.
  bool enableGlobalReuse;

  /// Enable print memory allocated size.
  bool enablePrintMemoryAllocatedSize;

  /// enable PTO op plan memory inplace
  bool restrictInplaceAsISA;

  /// StorageEntry generate.
  void GenerateStorageEntry();

  /// Print successful memory alloc.
  void PrintSuccessfulAllocatedMaxBits();

  /// Post-plan sanity check for local memory overflow.
  bool RecordOverflowIfAny();

  /// Prepare the memref.alloc plan.
  PlanStatus PlanLocalMemAddress();

  /// Prepare the memrefExt.alloc_workspace plan.
  PlanStatus PlanWorkSpaceMemAddress();

  /// merge all storage entry to the first storage entry for WorkSpaceArg.
  void MergeSameWorkSpaceArgSE();

  /// Start plan for same work space arg offset.
  PlanStatus PlanMemOffsetOfWholeWorkSpace();

  /// Enable global workspace no reuse.
  void GlobalWorkspaceNoReuse(StorageEntry *rootStorageEntry);

  /// Verify that constBits is legal.
  void ValidateParameters(std::unique_ptr<StorageEntry> &e) const;

  /// Expanding the Storage Entry due to the addition of MultiBuffer.
  void ExpandMultiBufferStorageEntry();

  /// merge all storage entry to the first storage entry.
  void MergeSameScopeSE();

  /// merge all storage entry which can be inplaced.
  void MergeInplaceSE();

  /// Start plan.
  PlanStatus PlanMemAddressOfWholeLocalBuffer();

  /// Plan a single local buffer without reuse.
  PlanStatus PlanSingleLocalBuffer(StorageEntry *rootStorageEntry, size_t align,
                                   size_t maxBits);

  /// Plan a reusable local buffer scope.
  PlanStatus PlanReusableLocalBuffer(StorageEntry *rootStorageEntry,
                                     size_t align, size_t maxBits);

  /// Plan memory only by level0 to report failure info.
  void PlanMemAddressForLevel0(StorageEntry *rootStorageEntry);

  /// Determine if the current space is enough to allocate all buffers.
  bool IsEnoughForBuffersNoReuse(StorageEntry *rootStorageEntry,
                                 size_t restBufferSize, size_t alignUnit);

  /// Adjust the allocation order of rootStoreEntry to prioritize the allocation
  /// of buffers corresponding to DMA.
  StorageEntry *GetReorderRootStorageEntry(StorageEntry *rootStorageEntry);

  /// Assign addresses without reuse.
  void PlanBuffersWithoutReuse(StorageEntry *rootStorageEntry,
                               size_t alignUnit);

  /// Obtain buffer space size and alignment information.
  std::pair<size_t, size_t> GetBufferSpaceInfo(pto::AddressSpace &space) const;

  /// Obtain effective buffer space size after accounting for reserve_buffer.
  std::pair<size_t, size_t>
  GetPlannableBufferSpaceInfo(pto::AddressSpace &space) const;

  /// Emit buffer applied failure message.
  void EmitPlanMemoryFailureInfo();

  /// Multi level plan strategy.
  LogicalResult MultiSpecPlan(SpecInfo &si, MemBoundList &outline,
                              PlanRecHis &history, StorageEntry *entry);

  /// plan buffer in speculative ways.
  LogicalResult SpecAlloc(MemBoundList &outline, PlanRecHis &his,
                          StorageEntry *e, const SpecInfo &si, int localLevel);

  /// spec_level == SPEC_LEVEL_2: pipe conflict only blocks reuse when the
  /// conflicting buffers share the same parent loop (less restrictive
  /// fallback below SPEC_LEVEL_3).
  bool VerifyConflictStage2(PlanRecHis &his, const StorageEntry *e,
                            int specLevel, MemBoundListConstIter &start,
                            const MemBoundList &outline);

  /// spec_level == SPEC_LEVEL_3: any pipe conflict between buffers blocks
  /// reuse. This is the most conservative pipe-conflict policy and is the
  /// initial stage attempted by MultiSpecPlan.
  bool VerifyConflictStage3(PlanRecHis &his, const StorageEntry *e,
                            int specLevel, MemBoundListConstIter &start,
                            const MemBoundList &outline);

  /// Shared scaffold for stage-2 / stage-3 conflict verification: parameterized
  /// by `conflictChecker` so each level can plug in its own pipe-conflict
  /// predicate while reusing the outline-walk and fallback logic.
  bool VerifyConflictStageCommon(
      PlanRecHis &his, const StorageEntry *e, MemBoundListConstIter &start,
      const MemBoundList &outline,
      std::function<bool(const StorageEntry *, const StorageEntry *)>
          conflictChecker);

  /// spec_level == SPEC_LEVEL_1, pure single can reuse with db.
  bool VerifyConflictStage1(MemBoundList &outline, PlanRecHis &his,
                            StorageEntry *e,
                            const OutlineSectionInfo &outlineInfo,
                            uint64_t &pongOffset);

  /// Check if e1 and e2 have any pipe conflict, regardless of loop scope.
  /// Cached in `conflictMap` to avoid recomputing the cartesian product of
  /// inplace buffers on each query.
  bool PipeConflict(const StorageEntry *e1, const StorageEntry *e2,
                    DenseMap<StorageEntryPair, bool> &conflictMap);

  /// Check if e1 and e2 have a pipe conflict that occurs within the same
  /// parent loop. Used by SPEC_LEVEL_2 to permit cross-loop reuse that
  /// SPEC_LEVEL_3 would forbid.
  bool PipeConflictInSameLoop(const StorageEntry *e1, const StorageEntry *e2);

  /// spec_level == SPEC_LEVEL_2, MTE2/MTE3 is pipe conflict with all existing
  /// allocation. check if current entry has OptDmaPipe-conflict with buffers
  /// already allocate at current position. if conflict exists, continue loop
  /// until first not-conflict iter is found. Then update start as the first
  /// bound right before the not-conflict one.
  bool VerifyDmaPipeConflict(const StorageEntry *e, int specLevel,
                             MemBoundListConstIter &start,
                             MemBoundListConstIter &end);

  /// Check if it matches the previous rollback result.
  bool IsSamePlanAsLastRollBack(uint64_t allocOffset, int curChildIdx,
                                const SpecInfo &si) const;

  /// spec_level == SPEC_LEVEL_0, life time reuse.
  inline bool VerifyConflictStage0(StorageEntry *e,
                                   const std::shared_ptr<MemoryBound> &last);

  /// Update the outline information and record history
  void UpdateOutline(MemBoundList &outline, PlanRecHis &his, StorageEntry *e,
                     const OutlineSectionInfo &outlineInfo,
                     int localLevel) const;

  /// plan strategy is achieved through split method.
  void AddMemBoundInSectionalWay(
      StorageEntry *e, MemBoundListConstIter start, MemBoundListConstIter end,
      SmallVector<std::shared_ptr<MemoryBound>> &splitBound) const;

  /// merge the buffer life between start and end.
  inline void MergeBufferLife(MemBoundList::const_iterator start,
                              MemBoundList::const_iterator end,
                              BufferLifeVec &newLife) const;

  /// merge buffers in a vector.
  void MergeBufferVec(BufferLifeVec &bufferLife) const;

  /// Judge if need to restart plan memory with other strategy after
  /// plan failed.
  PlanStatus ApplyFailStrategy(StatusWrapper &statusWrapper,
                               const size_t maxBits);

  void RollBackForAllocFail(StatusWrapper &statusWrapper, const size_t maxBits);

  /// Check if memory plan can be rolled back.
  bool ContinueRollBack(const StatusWrapper &statusWrapper) const;

  /// Memory plan fallback information processing.
  void RollBackForAllocFailInner(StatusWrapper &statusWrapper,
                                 const size_t maxBits);

  /// Fallback outline plan.
  PlanRecord RollbackOutline(PlanRecHis &history, MemBoundList &outline) const;

  /// Update the plan memory address corresponding to mem buffer.
  void UpdateBuffer2Offsets();

  /// Update extra addresses offset caused by multi buffer reuse.
  void UpdateMultiBufferReuseExtraOffset();

  /// generate inplace list by some rules
  SmallVector<ValuePair> GenerateInplaceList();

  /// the ptoop that can reuse dst address and src address in limited situation
  bool IsReusePTOOp(Operation *op) const;

  /// Get overlap buffer life.
  DenseMap<ValuePair, BufferLife>
  GetOverlapBufferLife(const BufferLifeVec &b1, const BufferLifeVec &b2) const;

  bool HasSemanticConflict(const StorageEntry *entry,
                           const BufferLifeVec &bufferLives) const;

  /// Reorder and make the storage entries of ping and pong continuous.
  void
  ReorderContinuousPingPongEntry(SmallVector<StorageEntry *> &storageEntryVec);

  /// Determine if the current buffer life of the Storage Entry conflicts with
  /// the memory that has already been allocated in history.
  bool IsBufferLifeVecConflict(PlanRecord &r, uint64_t offset,
                               const StorageEntry *e) const;

  /// Assign pong storage entry's address.
  void PlanRelationPongEntryAddress(uint64_t offset, StorageEntry *e);

  /// Processing Pong Storage Entry Information.
  void SpecAllocRelationPongEntry(MemBoundList &outline, PlanRecHis &his,
                                  StorageEntry *e, uint64_t offset);

  /// Get relative pong storage entry when the current reuse bound storage entry
  /// is of type db.
  StorageEntry *
  GetMultiRelationPongEntry(const StorageEntry *reuseBoundStorageEntry);

  /// Get the innermost for loop of buffer definition.
  LoopLikeOpInterface GetBufferParentLoop(const SmallVector<Value> &buffers);

  /// Report all tensors life time info.
  void ReportMemLifeDebugInfo(StorageEntry *rootStorageEntry);

  /// Report tensor life time for debug.
  void MemLifeDebugInfo(StorageEntry *storageEntry);

  /// Report tensor which is defined by memref allco.
  void ReportCurEntryDebugInfo(const StorageEntry *curEntry);

  /// Report tensor allocate info.
  void ReportAllocatedEntryDebugInfo(StorageEntry *rootStorageEntry);

private:
  /// The buffer corresponding to each operation.
  SmallVector<std::unique_ptr<OpInfo>> linearOperation;

  /// map from buffer value to its buffer information.
  std::map<Value, BufferInfo, ValueComparator> bufferInfos;

  /// map from buffer to its lifetime intervals.
  DenseMap<Value, BufferLifeVec> buffer2Life;

  /// record the map from the buffer to its number of buffer if it does
  /// multibuffer optimization.
  /// note: the map only record the buffer which do multi buffer optimization
  /// and ignore single buffer.
  DenseMap<Value, uint32_t> buffer2MultiNum;

  /// map from operation to its gen and kill buffer.
  DenseMap<OpInfo *, GenKillEntry> genKillMap;

  /// record all storage entry to be plan address.
  SmallVector<std::unique_ptr<StorageEntry>> StorageEntryVec;

  /// The current status of memory plan.
  PlanStatus planStatus{PlanStatus::PLAN_SUCCESS};

  /// Whether to adopt a split strategy.
  bool splitOutline{false};

  /// map from memref buffer to plan memory address.
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;

  /// map from each scope to its root StorageEntry.
  DenseMap<pto::AddressSpace, StorageEntry *> memscope2rootStorageEntry;

  /// map from workspace arg to its root StorageEntry.
  DenseMap<Value, StorageEntry *> workSpaceArg2rootStorageEntry;

  /// map from buffer scope to its required size to plan rest memory without any
  /// reuse.
  DenseMap<pto::AddressSpace, size_t> bufferScope2RequiredSize;

  /// total aligned auto-reserved capacity per local address space, in bits.
  DenseMap<pto::AddressSpace, uint64_t> reservedBufferBitsByScope;

  /// map from buffer value to its storage entry info
  DenseMap<Value, StorageEntry *> buffer2storageEntry;

  /// stable IR order for Values used to keep memory planning deterministic.
  StableValueOrderMap stableValueOrder;

  /// Memory dma pipe first plan optimization.
  OptMemPlanForDma dmaFirstPipelineOpt;

  /// Map from the storage entry pair to its pipeDma conflict info.
  DenseMap<StorageEntryPair, bool> pipeDmaConflictMap;

  /// Ping storage entry corresponding to reused additional Pong entry.
  DenseMap<StorageEntry *, std::unique_ptr<StorageEntry>>
      pingEntry2RelationPongEntry;

  SmallVector<const StorageEntry *> allocatedEntry;

  /// record inplace pair list.
  SmallVector<ValuePair> inplacePairList;

  /// record semantic conflict pair list.
  SmallVector<ValuePair> semanticConflictPairs;

  /// inplace-reuse info for the vf call.
  //VFCallInplaceReuseInfo *vfInplaceReuseInfo;

  /// The scope of the buffer applied memory fail and the max bits it applied.
  std::map<pto::AddressSpace, uint64_t> failApplyBufferInfo;

  /// The device's UB storage size
  int ubSpaceSize{0};

  /// The device's L1 storage size
  int l1SpaceSize{0};

  /// The device's L0A storage size
  int l0aSpaceSize{0};

  /// The device's L0B storage size
  int l0bSpaceSize{0};

  /// The device's L0C storage size
  int l0cSpaceSize{0};

  /// The device's UB align size
  int ubAlignSize{0};

  /// The device's L1 align size
  int l1AlignSize{0};

  /// The device's L0C align size
  int l0cAlignSize{0};

  int l0aAlignSize{0};

  int l0bAlignSize{0};

  int biasAlignSize{0};

  int biasSpaceSize{0};

  /// The device's SCALING align size
  int scalingAlignSize{0};

  /// The device's SCALING storage size
  int scalingSpaceSize{0};

};

} // namespace pto
} // namespace mlir

#endif // BISHENG_DIALECT_PTO_TRANSFORMS_PLAN_MEMORY_H
