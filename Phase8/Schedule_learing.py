from __future__ import absolute_import, print_function

import tvm
from tvm import te
import numpy as np

# 声明变量
n = te.var("n")
m = te.var("m")

# 声明一个矩阵元素乘法
A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))