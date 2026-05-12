# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, IndexType, InsertionPoint, IntegerType, Location, Module
from mlir.dialects import arith, func, pto, scf


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            i32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)
            tv_i32 = pto.TensorViewType.get([1], i32, ctx)
            pv_i32 = pto.PartitionTensorViewType.get([1], i32, ctx)
            pipe_all = pto.PipeAttr.get(pto.PIPE.PIPE_ALL, ctx)

            fn_ty = func.FunctionType.get([ptr_i32, ptr_i32, i32, i32, i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("TWaitAtomicKernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                local_counter_ptr, remote_counter_ptr, threshold, iters, my_rank = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c0_i32 = arith.ConstantOp(i32, 0).result
                c1_i32 = arith.ConstantOp(i32, 1).result

                local_view = pto.MakeTensorViewOp(tv_i32, local_counter_ptr, [c1], [c1]).result
                remote_view = pto.MakeTensorViewOp(tv_i32, remote_counter_ptr, [c1], [c1]).result
                local_counter = pto.PartitionViewOp(
                    pv_i32, local_view, offsets=[c0], sizes=[c1]
                ).result
                remote_counter = pto.PartitionViewOp(
                    pv_i32, remote_view, offsets=[c0], sizes=[c1]
                ).result

                is_non_root = arith.CmpIOp(arith.CmpIPredicate.ne, my_rank, c0_i32).result
                branch = scf.IfOp(is_non_root, [], hasElse=True)

                with InsertionPoint(branch.then_block):
                    iters_idx = arith.IndexCastOp(idx, iters).result
                    loop = scf.ForOp(c0, iters_idx, c1, [])
                    with InsertionPoint(loop.body):
                        pto.TNotifyOp(
                            remote_counter,
                            c1_i32,
                            pto.NotifyOpAttr.get(pto.NotifyOp.AtomicAdd, ctx),
                        )
                        scf.YieldOp([])
                    pto.barrier(pipe_all)
                    scf.YieldOp([])

                with InsertionPoint(branch.else_block):
                    pto.TWaitOp(
                        local_counter,
                        threshold,
                        pto.WaitCmpAttr.get(pto.WaitCmp.GE, ctx),
                    )
                    pto.barrier(pipe_all)
                    scf.YieldOp([])

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
