from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np

# 声明变量
n = te.var("n")
m = te.var("m")

##########################################################################
# reduce
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
print(tvm.lower(s, [A, B], simple_mode=True))