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
    func_.emitError() << stringifyEnum(space) << " overflow, requires "
                      << iter.second << " bits while "
                      << GetBufferSpaceInfo(space).second << " bits avaliable!";
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
        GetBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
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
LogicalResult MemPlan::plan() {
  // Construct StorageEntry structure.
  GenerateStorageEntry();
  // Plan memory address.
  PlanStatus as = planMode == MemPlanMode::LOCAL_MEM_PLAN
                      ? PlanLocalMemAddress()
                      : PlanWorkSpaceMemAddress();
  if (as == PlanStatus::PLAN_FAILED) {
    EmitPlanMemoryFailureInfo();
    return failure();
  }
  if (RecordOverflowIfAny()) {
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
      func_.emitError()
          << "PlanMemory produced overlapping local buffers in "
          << stringifyEnum(lhs->bufInfo->bufferScope)
          << " at offsets " << lhs->bitsOffset << " and " << rhs->bitsOffset;
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
  for (auto &operation : linearOperation) {
    auto it = genKillMap.find(operation.get());
    if (it == genKillMap.end())
      continue;
    SmallVector<Value> genBuffers(it->second.gen.begin(), it->second.gen.end());
    sortValuesByStableOrder(genBuffers, stableValueOrder);
    for (const Value &genBuffer : genBuffers) {
      auto iter = bufferInfos.find(genBuffer);
      if (iter == bufferInfos.end()) {
        continue;
      }
      const std::shared_ptr<BufferLife> &bufLife = buffer2Life.at(genBuffer);
      std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
      entry->bufInfo = &iter->second;
      entry->bufferLifeVec.emplace_back(bufLife);
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
  for (auto &e : StorageEntryVec) {
    for (Value &buffer : e->inplaceBuffers) {
      // MultiBuffer can cause multiple addrs.
      buffer2Offsets[buffer].push_back(
          (e->bitsOffset + kBitsToByte - 1) / kBitsToByte);
    }
  }
  // In the MultiBuffer scenario, single reuse db will result in additional
  // storageEntry.
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
    if (StorageEntryVec[i]->multiBufferNum > 1) {
      std::unique_ptr<StorageEntry> entry = std::make_unique<StorageEntry>();
      entry->bufInfo = StorageEntryVec[i]->bufInfo;
      entry->bufferLifeVec = StorageEntryVec[i]->bufferLifeVec;
      entry->alignedConstBits = StorageEntryVec[i]->alignedConstBits;
      entry->inplaceBuffers = StorageEntryVec[i]->inplaceBuffers;
      entry->multiBufferNum = StorageEntryVec[i]->multiBufferNum;
      // Ping saves information related to Pong.
      StorageEntryVec[i]->relationPongEntry = entry.get();
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
    auto bufferSpaceInfo = GetBufferSpaceInfo(rootStorageEntry.first);
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
        GetBufferSpaceInfo(rootStorageEntry->bufInfo->bufferScope);
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
  for (auto &storageEntry : storageEntryVec) {
    auto it = std::find(reorderedStorageEntryVec.begin(),
                        reorderedStorageEntryVec.end(), storageEntry);
    if (it == reorderedStorageEntryVec.end()) {
      reorderedStorageEntryVec.push_back(storageEntry);
      if (storageEntry->multiBufferNum == kDoubleBufferCount &&
          storageEntry->relationPongEntry) {
        // Ping Pong continuous save.
        reorderedStorageEntryVec.push_back(storageEntry->relationPongEntry);
      }
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

