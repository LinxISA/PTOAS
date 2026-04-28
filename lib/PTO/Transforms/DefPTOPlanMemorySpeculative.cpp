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
  if (std::any_of(his.begin(), his.end(),
                  [e](PlanRecord &r) { return r.entry && r.entry == e; })) {
    // If the plan has already been completed, return success directly.
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
    if (e->multiBufferNum == kSingleBufferCount ||
        (e->multiBufferNum == kDoubleBufferCount && e->relationPongEntry &&
         (e->relationPongEntry->bitsOffset != 0))) {
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
  if (reuseBoundStorageEntry->multiBufferNum == kDoubleBufferCount &&
      reuseBoundStorageEntry->relationPongEntry &&
      (reuseBoundStorageEntry->relationPongEntry->bitsOffset != 0)) {
    // If the reuseBoundStorageEntry itself requires db, directly match and
    // return relationPongEntry.
    return reuseBoundStorageEntry->relationPongEntry;
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
  for (MemBoundListConstIter start = outline.begin(); start != outline.end();
       ++start) {
    uint64_t size = 0;
    // Find the MemBound corresponding to the Pong offset.
    if ((*start)->offset != offset) {
      continue;
    }
    for (MemBoundListConstIter end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      size += last->extent;
      if (size < e->alignedConstBits) {
        continue;
      }
      StorageEntry *pongStorageEntry = nullptr;
      auto iter = pingEntry2RelationPongEntry.find(e);
      if (iter != pingEntry2RelationPongEntry.end()) {
        pongStorageEntry = iter->second.get();
      }
      if (e->multiBufferNum == kDoubleBufferCount && e->relationPongEntry) {
        pongStorageEntry = e->relationPongEntry;
      }
      if (!pongStorageEntry)
        llvm::report_fatal_error("pong storage entry not found");
      UpdateOutline(outline, his, pongStorageEntry,
                    OutlineSectionInfo(start, end, size, true), SPEC_LEVEL_1);
      return;
    }
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
  } else if (e->multiBufferNum == kDoubleBufferCount) {
    e->relationPongEntry->bitsOffset = offset;
  } else {
    llvm_unreachable("Does not support multi buffer num greater than 2 !");
  }
}

bool MemPlan::VerifyConflictStage2(PlanRecHis &his, const StorageEntry *e,
                                   int specLevel, MemBoundListConstIter &start,
                                   const MemBoundList &outline) {
  if (specLevel != SPEC_LEVEL_2) {
    return false;
  }
  bool touchMemCanUse = false;
  MemBoundListConstIter foundMem;

  for (auto iter = start; iter != outline.end(); ++iter) {
    uint64_t offset = (*iter)->offset;
    bool conflict =
        std::any_of(his.begin(), his.end(), [offset, e, this](PlanRecord &r) {
          return (r.firstMemBound->offset + r.allExtent > offset) &&
                 (r.firstMemBound->offset < offset + e->alignedConstBits) &&
                 this->PipeConflict(r.entry, e, this->pipeDmaConflictMap);
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
      llvm::report_fatal_error("missing failed storage entry");
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
        (r.entry->multiBufferNum == kDoubleBufferCount &&
         !r.entry->relationPongEntry)) {
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

