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

# print(tvm.lower(s, [A, B, C], simple_mode=True))

##########################################################################
# split
# use factor
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] * 2, name="B")
s = te.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
# print(tvm.lower(s, [A, B], simple_mode=True))

# use nparts
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] * 2, name="B")
s = te.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], nparts=32)
# print(tvm.lower(s, [A, B], simple_mode=True))

##########################################################################
# tile
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j] * 2, name="B")
s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# print(tvm.lower(s, [A, B], simple_mode=True))

##########################################################################
# fuse
# s[B].fuse(xi, yi)
# print(tvm.lower(s, [A, B], simple_mode=True))

##########################################################################
# reorder
s[B].reorder(xi, yo, xo, yi)
print(tvm.lower(s, [A, B], simple_mode=True))