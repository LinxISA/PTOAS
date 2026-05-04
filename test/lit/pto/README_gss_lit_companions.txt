GraphSyncSolver companion lit files (*_gss.pto)
================================================

For each PTO lit test that originally used `--enable-insert-sync`, a sibling file
named `<stem>_gss.pto` duplicates the same module IR but runs:

  --enable-graph-sync-solver [--graph-sync-solver-event-id-max=64]

(except `tassign_level3_loop_rebind_gss.pto`, which adds a `not` RUN expecting the
`tassign`/sync mutual exclusion error when GraphSyncSolver is enabled).

Why separate files?
  InsertSync tests pin exact EmitC ordering (`CHECK-NEXT`) and concrete EVENT_ID
  numbers. GraphSyncSolver uses a different solver/codegen path; expectations are
  relaxed where needed (`CHECK-DAG`, `EVENT_ID{{[0-9]+}}`, or shortened guards).

Judging whether generated C++ sync is "reasonable"
----------------------------------------------------
  - Memory hazards between pipes should have *some* conservative serialization:
    paired `set_flag`/`wait_flag` on the producer/consumer pipes, and/or
    `pipe_barrier(PIPE_ALL)` fallbacks when coloring/event ids fail.
  - Level3 GM/partition `tload`/`tstore` paths should still show ordering between
    MTE2/MTE3 (or equivalent barriers) before touching vec tiles.
  - Cross-core FIFO smoke (`tpush_tpop_frontend_lowering_a3_gss.pto`) should keep a
    handshake between remote producers (`TPOP`) and local consumers (`TMOV`,
    `TNEG`) via MTE2-directed waits analogous to InsertSync.
  These companions encode minimal FileCheck anchors for those properties; they do
  not prove simulator-level correctness—only regression snapshots.

Known gap: `issue564_k_loop_mte1_mte2_wait_regression_gss.pto` was removed because
`ptoas --enable-graph-sync-solver` currently segfaults on that IR (Level3 + dual kernel module).

Regenerating stricter checks (optional)
---------------------------------------
  Run `ptoas` with the same flags as the RUN line, inspect stdout C++, and tighten
  CHECKs when GSS output stabilizes.
