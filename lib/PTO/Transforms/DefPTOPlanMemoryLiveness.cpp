// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

void MemLivenessAnalysis::build() {
  Region &funcRegion = func_.getBody();
  stableValueOrder = buildStableValueOrder(func_);
  Liveness live(func_);
  // Recursively obtaining IR information.
  RecursionIR(&funcRegion, live);
  // the lifetime of the buffer.
  GenerateBufferLife();
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
    } else if (isa<pto::TPushOp, pto::TFreeOp, pto::InitializeL2LPipeOp,
                   pto::InitializeL2G2LPipeOp, pto::BuildAsyncSessionOp,
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
  SmallVector<Value> allocBeforeLoopBuffers;
  const auto *liveBlockInfo = live.getLiveness(forOp->getBlock());
  auto currentLiveValues =
      liveBlockInfo->currentlyLiveValues(forOp.getOperation());
  if (currentLiveValues.empty()) {
    return allocBeforeLoopBuffers;
  }
  // The gen buffer of the same operation must ensure the order of priority.
  SetVector<Value> currentLiveValuesOrder;
  for (auto buffer : currentLiveValues) {
    currentLiveValuesOrder.insert(buffer);
  }
  for (const Value &operand : currentLiveValuesOrder) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto Buffer : aliasBuffers) {
      auto iter = buffer2status.find(Buffer);
      if (iter != buffer2status.end())
        allocBeforeLoopBuffers.push_back(Buffer);
    }
  }
  sortValuesByStableOrder(allocBeforeLoopBuffers, stableValueOrder);
  return allocBeforeLoopBuffers;
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
    genKillMap[opInfo].gen.push_back(operand);
    buffer2status[iter_buffer->first] = BufferStatus::GENED;
  } else if (iter_buffer->second == BufferStatus::KILLED) {
    llvm_unreachable("The buffer memory has been released and cannot be used "
                     "again! ");
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
  sortValuesByStableOrder(liveValues, stableValueOrder);
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
    if (iterBuffer->second == BufferStatus::GENED &&
        IsInSameBlock(iterBuffer->first.getDefiningOp(), opInfo->operation) &&
        AllDeadAfter(opInfo->operation, aliasBuffers, live)) {
      genKillMap[opInfo].kill.push_back(aliasBuffer);
      buffer2status[iterBuffer->first] = BufferStatus::KILLED;
    }
  }
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
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    auto it = genKillMap.find(linearOperation[i].get());
    if (it == genKillMap.end()) {
      scopeTime++;
      continue;
    }
    // Time given to buffer start.
    for (const Value &genBuffer : it->second.gen) {
      std::unique_ptr<BufferLife> bufferLife =
          std::make_unique<BufferLife>(genBuffer);
      bufferLife->allocTime = scopeTime;
      buffer2Life[genBuffer] = std::move(bufferLife);
    }
    // Time given to buffer end.
    for (const Value &killBuffer : it->second.kill) {
      auto iter = buffer2Life.find(killBuffer);
      if (iter == buffer2Life.end())
        llvm::report_fatal_error("buffer lifetime killed before generation");
      iter->second->freeTime = scopeTime;
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
