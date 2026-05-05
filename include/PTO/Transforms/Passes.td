// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

// Define PTOInsertSync Pass
def PTOInsertSync : Pass<"pto-insert-sync", "func::FuncOp"> {
  let summary = "Insert synchronization flags for PTO dialect";
  let description = [{
    Analyzes data dependencies between Cube, Vector and MTE pipelines
    and inserts explicit synchronization instructions.
  }];
 
  // Declare the constructor function name
  let constructor = "mlir::pto::createPTOInsertSyncPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect" 
  ];
}

// Define PTOGraphSyncSolver Pass (graph-based intra-core sync solver).
// Coexists with PTOInsertSync; selectable via the driver flag
// `--enable-graph-sync-solver`. Minimal port of bishengir's GraphSyncSolver:
// Dijkstra-based reachability pruning + Welsh-Powell event-id coloring with
// fallback to a single PIPE_ALL barrier when the budget is exceeded.
def PTOGraphSyncSolver : Pass<"pto-graph-sync-solver", "func::FuncOp"> {
  let summary = "Graph-based intra-core synchronization solver for PTO";
  let description = [{
    Alternative to PTOInsertSync. Builds a hierarchical SyncSolver IR and
    decides which set/wait/barrier ops to emit by combining a reachability
    analysis (so transitively-covered candidates are skipped) with graph
    coloring over the pool of EVENT_IDs.
  }];

  let constructor = "mlir::pto::createPTOGraphSyncSolverPass()";

  let options = [
    Option<"eventIdNumMax", "event-id-num-max", "int64_t",
           /*default=*/"8",
           "Maximum EVENT_ID slots usable by Welsh-Powell coloring; exceeding "
           "the budget falls back to a single PIPE_ALL barrier.">,
  ];

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def ConvertToPTOOp : Pass<"convert-to-pto-op"> {
  let summary = "Convert Ops from other dialects to PTO Ops";
  let constructor = "mlir::pto::createConvertToPTOOpPass()";
  let dependentDialects = ["arith::ArithDialect", "pto::PTODialect",
                           "tensor::TensorDialect",
  ];
}

def InferPTOLayout : Pass<"pto-infer-layout", "func::FuncOp"> {
  let summary = "Infer GlobalTensor layout (ND/DN/NZ) for make_tensor_view";
  let description = [{
    For each pto.make_tensor_view with fully static shape/stride, compute layout
    (ND/DN/NZ) per design rules and attach the optional `layout` attribute.
    Dynamic stride/shape are left unset.
  }];
  let constructor = "mlir::pto::createInferPTOLayoutPass()";
  let dependentDialects = ["pto::PTODialect", "arith::ArithDialect"];
}

def PTOA5NormalizeTMov : Pass<"pto-a5-normalize-tmov", "func::FuncOp"> {
  let summary = "Normalize risky A5 vec->vec col_major TMOV into row-major reinterpret + TMOV";
  let description = [{
    Rewrites A5 `pto.tmov` patterns that use vec->vec col_major/none_box tiles:
      src(col_major) -> pto.treshape(row_major) -> pto.tmov(row_major->row_major)
      dst(col_major) -> pto.treshape(row_major)
    This avoids unsupported A5 PTO-ISA vec->vec col_major TMOV paths while
    preserving alias semantics via SSA treshape (no real data movement).
  }];
  let constructor = "mlir::pto::createPTOA5NormalizeTMovPass()";
  let dependentDialects = ["pto::PTODialect", "func::FuncDialect"];
}


def InferPTOMemScope : Pass<"pto-infer-mem-scope"> {
  let summary = "Infer memory scope for PTO Ops";
  let constructor = "mlir::pto::createInferPTOMemScopePass()";
  let dependentDialects = ["pto::PTODialect", "memref::MemRefDialect", ];
}

def PlanMemory : Pass<"pto-plan-memory", "ModuleOp"> {
  let summary = "Plan memory for PTO Ops";
  let constructor = "mlir::pto::createPlanMemoryPass()";
  let dependentDialects = ["pto::PTODialect", ];
  let options = [
    Option<"memMode", "mem-plan-mode", "pto::MemPlanMode",
           "pto::MemPlanMode::LOCAL_MEM_PLAN",
           "plan mem mode (default is LOCAL_MEM_PLAN)",
           [{::llvm::cl::values(
               clEnumValN(pto::MemPlanMode::LOCAL_MEM_PLAN, "local-mem-plan",
                          "plan mem mode is for memref.alloc"),
               clEnumValN(
                   pto::MemPlanMode::GLOBAL_WORKSPACE_PLAN,
                   "global-work-space-plan",
                   "plan mem mode is for memref_ext.alloc_workspace"))}]>,
    Option<"enableGlobalReuse", "enable-global-workspace-reuse", "bool",
           /*default=*/"false",
           "Enable global workspace reuse ,default : false">,
    Option<"enablePrintMemoryAllocatedSize", "enable-print-memory-allocated-size", "bool",
           /*default=*/"false",
           "Enable print memory allocated size, default : false">,
    Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
           /*default=*/"false",
           "restrict memory inplace as isa, default : false">,
  ];
}
def PTOLoweringSyncToPipe : Pass<"pto-lowering-sync-to-pipe", "func::FuncOp"> {
  let summary = "Lower high-level sync ops to low-level pipe ops";
  let description = [{
    Converts high-level `pto.record_event` and `pto.wait_event` operations
    into low-level `pto.set_flag` and `pto.wait_flag` operations based on
    hardware pipe mapping logic.
  }];

  let constructor = "mlir::pto::createLoweringSyncToPipePass()";
  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOWrapFunctionsInSections : Pass<"pto-wrap-functions-in-sections", "func::FuncOp"> {
  let summary = "Wrap attributed single-core functions in PTO cube/vector sections";
  let description = [{
    Rewrites frontend-marked functions into the existing PTO section form:
    - `pto.kernel_kind = #pto.kernel_kind<cube>` -> wrap the function body in
      `pto.section.cube`
    - `pto.kernel_kind = #pto.kernel_kind<vector>` -> wrap the function body
      in `pto.section.vector`
  }];

  let constructor = "mlir::pto::createPTOWrapFunctionsInSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOLowerFrontendPipeOps : Pass<"pto-lower-frontend-pipe-ops", "func::FuncOp"> {
  let summary = "Lower frontend TPUSH/TPOP pipe ops to internal pipe ops";
  let description = [{
    Converts frontend `aic/aiv_initialize_pipe`, `tpush_to_*`,
    `tpop_from_*`, and `tfree_from_*` ops into the internal unified pipe IR:
    `initialize_l2l_pipe` / `initialize_l2g2l_pipe`, `tpush`,
    `declare_tile`, `tpop`, and `tfree`.
  }];

  let constructor = "mlir::pto::createPTOLowerFrontendPipeOpsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOAssignDefaultFrontendPipeId : Pass<"pto-assign-default-frontend-pipe-id", "func::FuncOp"> {
  let summary = "Materialize default id = 0 on frontend TPUSH/TPOP pipe ops";
  let description = [{
    Preserves backward compatibility with older frontend pipe syntax that
    omitted `id`. This pass rewrites missing `id` attrs on
    `aic/aiv_initialize_pipe`, `tpush_to_*`, `tpop_from_*`, and
    `tfree_from_*` to an explicit `id = 0` before frontend pipe lowering.
  }];

  let constructor = "mlir::pto::createPTOAssignDefaultFrontendPipeIdPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOInferValidatePipeInit : Pass<"pto-infer-validate-pipe-init", "ModuleOp"> {
  let summary = "Infer and validate internal pipe init nosplit configuration";
  let description = [{
    Runs after frontend pipe lowering and before memory planning. For each
    logical pipe, this pass:
    - validates that downstream `pto.tpush` / `pto.tpop` / `pto.tfree` users
      do not mix `split = 0` with `split = 1/2`
    - preserves explicit `nosplit` attrs on internal pipe init ops and rejects
      conflicts across peer pipe init pairs
    - infers missing `nosplit` attrs from downstream split usage for backward
      compatibility with older IR that omitted init-level `nosplit`
    - propagates the resolved `nosplit` value across peer pipe init pairs so
      both ends of one logical pipe agree before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOInferValidatePipeInitPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOCVAutoMarkMultiBuffer : Pass<"pto-cv-auto-mark-multi-buffer", "ModuleOp"> {
  let summary = "Auto-mark CV preload local buffers as multi-buffered";
  let description = [{
    Scans Cube/Vector frontend TPipe transactions before CV scope materialization,
    pairs producer/relay/consumer stages across logical C2V/V2C pipes, derives the
    same max preload depth used by `pto-cv-mark-preload-scopes`, and annotates the
    root local buffers used by those stages with `pto.multi_buffer`.

    The pass intentionally runs before scope creation so the marker is attached to
    existing `pto.alloc_tile` / `memref.alloc` anchors and can be forwarded to plan
    memory after tile-buffer lowering.
  }];

  let constructor = "mlir::pto::createPTOCVAutoMarkMultiBufferPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];
}

def PTOCVMarkPreloadScopes : Pass<"pto-cv-mark-preload-scopes", "ModuleOp"> {
  let summary = "Create CV preload scopes from frontend TPipe transactions";
  let description = [{
    Scans Cube/Vector kernel_kind functions before frontend pipe lowering,
    recognizes producer/consumer/relay TPipe transaction scopes, wraps each
    transaction in a no-result `pto.cv.scope`, pairs them across logical
    C2V/V2C pipes, and annotates the scope op with pto.cv.* preload metadata.
  }];

  let constructor = "mlir::pto::createPTOCVMarkPreloadScopesPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect"
  ];
}

def PTOCVCreatePreload : Pass<"pto-cv-create-preload", "ModuleOp"> {
  let summary = "Expand marked CV preload scopes into stage-guarded loops";
  let description = [{
    Consumes no-result `pto.cv.scope` operations annotated with
    `pto.cv.preload_num` / `pto.cv.max_preload_num`, expands the parent
    `scf.for` upper bound by `max_preload_num * step`, clones each scope under
    a stage-valid `scf.if`, remaps the loop induction variable to the stage IV,
    and rotates multi-address `pto.pointer_cast` values prepared by plan memory.

    This pass expects local preload buffers to have already been annotated with
    `pto.multi_buffer` and lowered/planned into variadic `pto.pointer_cast`
    values. It deliberately does not synthesize storage addresses itself.
  }];

  let constructor = "mlir::pto::createPTOCVCreatePreloadPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def PTOInlineCVPreloadScopes : Pass<"pto-inline-cv-preload-scopes", "ModuleOp"> {
  let summary = "Inline no-result CV preload scope containers";
  let description = [{
    Inlines `pto.cv.scope` bodies back into their parent blocks. This keeps the
    current codegen pipeline unchanged while leaving an explicit scope IR point
    for debug printing and future preload expansion passes.
  }];

  let constructor = "mlir::pto::createPTOInlineCVPreloadScopesPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOResolveReservedBuffers : Pass<"pto-resolve-reserved-buffers", "ModuleOp"> {
  let summary = "Resolve reserved local buffer addresses and peer pipe flag bases";
  let description = [{
    Runs after `pto-plan-memory`. Assumes `pto.reserve_buffer` base addresses
    have already been planned, then:
    - aligns missing `flag_base` attrs for peer internal pipe init ops
    - rejects internal pipe init ops without explicit `flag_base` when their
      `local_addr` cannot be traced back to `pto.reserve_buffer` /
      `pto.import_reserved_buffer`
    - replaces `pto.reserve_buffer` / `pto.import_reserved_buffer` results with
      resolved constant local addresses
    - erases the frontend-only reserve/import ops before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOResolveReservedBuffersPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOVerifyTFree : Pass<"pto-verify-tfree", "func::FuncOp"> {
  let summary = "Verify explicit matching pto.tfree placement for pto.tpop";
  let description = [{
    For each `pto.tpop` in section.cube / section.vector, verify there is a
    matching later `pto.tfree(pipe_handle)` in the same block, the borrowed
    tile is not used after the free, and the same pipe does not accumulate
    multiple outstanding pops before the matching free.
  }];

  let constructor = "mlir::pto::createPTOVerifyTFreePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOEnableMultiBuffer : Pass<"pto-enable-multi-buffer", "func::FuncOp"> {
  let summary = "Lower variadic pto.pointer_cast with multi-buffer addrs into "
                "single-address casts plus a per-iteration arith.select";
  let description = [{
    Mirrors HIVM's `EnableMultiBuffer` lowering: takes a `pto.pointer_cast` with
    N>1 address operands, hoists each address into its own single-address
    `pto.pointer_cast` outside the parent `scf.for`, then replaces the original
    multi-address cast with an N-way `arith.select` chain driven by `iv mod N`.
    Runs after `pto-insert-sync` so the multi-address `pto.pointer_cast` stays
    visible to dependency analysis.
  }];
  let constructor = "mlir::pto::createPTOEnableMultiBufferPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOViewToMemref : Pass<"pto-view-to-memref", "ModuleOp"> {
  let summary = "Lower PTO views to memref with Metadata Binding";
  let description = [{
    Lower PTO tile/view operations to memref-based IR while preserving tile
    metadata through binding ops and SSA backtracking.
  }];

  let constructor = "mlir::pto::createPTOViewToMemrefPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // MLIR_DIALECT_PTO_PASSES
