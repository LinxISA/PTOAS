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

    MemLivenessAnalysis memLiveness(funcOp, this->memMode);
    memLiveness.build();

    MemPlan memPlan(this->memMode, this->enableGlobalReuse,
                    this->enablePrintMemoryAllocatedSize,
                    this->restrictInplaceAsISA);
    if (failed(memPlan.InitMemSpecsFromModule(funcOp))) {
      return signalPassFailure();
    }
    memPlan.func_ = funcOp;
    memPlan.SetLinearOperation(memLiveness.linearOperation);
    memPlan.SetBufferInfos(memLiveness.bufferInfos);
    memPlan.SetBuffer2Life(memLiveness.buffer2Life);
    memPlan.SetGenKillMap(memLiveness.genKillMap);
    memPlan.SetBuffer2MultiNum(memLiveness.buffer2MultiNum);
    memPlan.SetInplacePairList(memLiveness.inplacePairList);
    memPlan.SetSemanticConflictPairs(memLiveness.semanticConflictPairs);
    memPlan.SetStableValueOrder(std::move(memLiveness.stableValueOrder));
    if (failed(memPlan.plan())) {
      return signalPassFailure();
    }
    // Keep reserve_buffer allocation outside the core MemPlan algorithm:
    // normal local buffers are planned first, then reserve_buffer claims one
    // aligned hole in its target address space.
    if (this->memMode == MemPlanMode::LOCAL_MEM_PLAN &&
        failed(assignAutoReserveBufferBases(reservePlans, memLiveness.bufferInfos,
                                            memPlan.GetBuffer2Offsets()))) {
      return signalPassFailure();
    }

    RewritePatternSet patterns(&getContext());
    populateBufferAddressToAllocOp(patterns, memPlan.GetBuffer2Offsets());
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass>
mlir::pto::createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption) {
  return std::make_unique<PlanMemoryPass>(planMemoryOption);
}
