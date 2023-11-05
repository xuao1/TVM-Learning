from __future__ import absolute_import, print_function

import tvm
import tvm.testing
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
# print(tvm.lower(s, [A, B], simple_mode=True))

##########################################################################
# bind
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] * 2, name="B")
s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=64)
# s[B].bind(bx, te.thread_axis("blockIdx.x"))
# s[B].bind(tx, te.thread_axis("threadIdx.x"))
# print(tvm.lower(s, [A, B], simple_mode=True))

##########################################################################
# compute_at
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")
s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
# print(tvm.lower(s, [A, B, C], simple_mode=True))

##########################################################################
# compute_inline
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")
s = te.create_schedule(C.op)
s[B].compute_inline()
# print(tvm.lower(s, [A, B, C], simple_mode=True))

##########################################################################
# compute_root
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")
s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
# print(tvm.lower(s, [A, B, C], simple_mode=True))
s[B].compute_root()
# print(tvm.lower(s, [A, B, C], simple_mode=True))

##########################################################################
# reduce
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
s = te.create_schedule(B.op)
print(tvm.lower(s, [A, B], simple_mode=True))