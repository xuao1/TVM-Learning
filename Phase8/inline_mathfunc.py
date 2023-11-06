from __future__ import absolute_import, print_function

import numpy as np
import tvm
from tvm import te
from tvm.ir import register_op_attr, register_intrin_lowering

######################################################################
# 直接声明外部数学调用
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: tvm.tir.call_pure_extern("float32", "__expf", A[i]), name="B")
s = te.create_schedule(B.op)
# print(tvm.lower(s, [A, B], name="expf", simple_mode=True))
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
f = tvm.build(s, [A, B], "cuda", name="myexp")
# print(f.imported_modules[0].get_source())

######################################################################
# 统一内联调用
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: te.exp(A[i]), name="B")
s = te.create_schedule(B.op)
# print(tvm.lower(s, [A, B], name="exp", simple_mode=True))
num_thread = 64
bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
fcuda = tvm.build(s, [A, B], "cuda", name="myexp")
# print(fcuda.imported_modules[0].get_source())
fopencl = tvm.build(s, [A, B], "opencl", name="myexp")
# print(fopencl.imported_modules[0].get_source())

######################################################################
# 自定义 CUDA 内联函数降级规则
def my_cuda_math_rule(op):
    assert isinstance(op, tvm.tir.Call)
    name = op.op.name
    assert name.startswith("tir.")
    dispatch_name = name[4:]
    if op.dtype == "float32":
        return tvm.tir.call_pure_extern("float32", "%sf" % dispatch_name, op.args[0])
    elif op.dtype == "float64":
        return tvm.tir.call_pure_extren("float32", dispatch_name, op.args[0])
    else:
        return op

register_intrin_lowering("tir.exp", target="cuda", f=my_cuda_math_rule, level=99)
