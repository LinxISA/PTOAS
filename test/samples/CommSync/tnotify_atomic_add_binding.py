# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, IndexType, InsertionPoint, IntegerType, Location, Module
from mlir.dialects import arith, func, pto


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

            fn_ty = func.FunctionType.get([ptr_i32], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("TNotifyAtomicAddKernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                signal_ptr = entry.arguments[0]
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                one_i32 = arith.ConstantOp(i32, 1).result

                signal_view = pto.MakeTensorViewOp(tv_i32, signal_ptr, [c1], [c1]).result
                signal = pto.PartitionViewOp(
                    pv_i32, signal_view, offsets=[c0], sizes=[c1]
                ).result

                pto.TNotifyOp(
                    signal,
                    one_i32,
                    pto.NotifyOpAttr.get(pto.NotifyOp.AtomicAdd, ctx),
                )

                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
