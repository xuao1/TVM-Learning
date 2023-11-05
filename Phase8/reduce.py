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
BF = s.rfactor(B, ki)
print(tvm.lower(s, [A, B], simple_mode=True))

for k_inner, i in T.grid(16, n):
    B_rf_1[k_inner * n + i] = T.float32(0)
    for k_outer in range((m + 15) // 16):
        if T.likely(k_outer * 16 + k_inner < m):
            A_2 = T.Buffer((A_1.strides[0] * n,), data=A_1.data, buffer_type="auto")
            B_rf_1[k_inner * n + i] = B_rf_1[k_inner * n + i] + A_2[i * A_1.strides[0] + (k_outer * 16 + k_inner) * A_1.strides[1]]