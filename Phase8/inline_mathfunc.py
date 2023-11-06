from __future__ import absolute_import, print_function

import numpy as np
import tvm
from tvm import te
from tvm.ir import register_op_attr, register_intrin_lowering

# 直接声明外部数学调用
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: tvm.tir.call_pure_extern("float32", "__expf", A[i]), name="B")
s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], name="expf", simple_mode=True))