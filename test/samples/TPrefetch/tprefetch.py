# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, InsertionPoint, IntegerType, Location, Module, IndexType
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()
            i64 = IntegerType.get_signless(64, ctx)
            idx = IndexType.get(ctx)
            fn_ty = func.FunctionType.get([i64, idx, idx, idx, idx], [])
            with InsertionPoint(module.body):
                fn = func.FuncOp("tprefetch_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                pto.TPrefetchOp(*entry.arguments)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
