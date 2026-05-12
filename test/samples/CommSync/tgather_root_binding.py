# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module
from mlir.dialects import arith, func, pto, scf


def _make_part(ptr, tv_ty, pv_ty, c0, c1, count):
    tv = pto.MakeTensorViewOp(tv_ty, ptr, [count], [c1]).result
    return pto.PartitionViewOp(pv_ty, tv, offsets=[c0], sizes=[count]).result


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            module = Module.create()

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv_f32 = pto.TensorViewType.get([256], f32, ctx)
            pv_f32 = pto.PartitionTensorViewType.get([256], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            pipe_all = pto.PipeAttr.get(pto.PIPE.PIPE_ALL, ctx)
            tb_f32 = pto.TileBufType.get([1, 256], f32, vec, [1, 256], None, ctx)

            fn_ty = func.FunctionType.get(
                [ptr_f32, ptr_f32, ptr_f32, ptr_f32, ptr_f32, i32, i32], []
            )
            with InsertionPoint(module.body):
                fn = func.FuncOp("TGatherKernelImpl", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                dst_ptr, src0_ptr, src1_ptr, src2_ptr, src3_ptr, my_rank, nranks = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c256 = arith.ConstantOp(idx, 256).result
                c1_i32 = arith.ConstantOp(i32, 1).result
                _ = nranks

                dst = _make_part(dst_ptr, tv_f32, pv_f32, c0, c1, c256)
                src0 = _make_part(src0_ptr, tv_f32, pv_f32, c0, c1, c256)
                src1 = _make_part(src1_ptr, tv_f32, pv_f32, c0, c1, c256)
                src2 = _make_part(src2_ptr, tv_f32, pv_f32, c0, c1, c256)
                src3 = _make_part(src3_ptr, tv_f32, pv_f32, c0, c1, c256)
                group = [src0, src1, src2, src3]

                ping = pto.AllocTileOp(tb_f32).result

                is_root = arith.CmpIOp(arith.CmpIPredicate.eq, my_rank, c1_i32).result
                root_if = scf.IfOp(is_root, [], hasElse=False)
                with InsertionPoint(root_if.then_block):
                    pto.CommTGatherOp(dst, ping, group, 1)
                    scf.YieldOp([])

                pto.barrier(pipe_all)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
