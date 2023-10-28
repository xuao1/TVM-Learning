import tvm
import tvm.testing
from tvm import te
import numpy as np

target = tvm.target.Target(target="llvm -mcpu=tigerlake", host="llvm -mcpu=tigerlake")

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

for (int i = 0; i < n; ++i) {
  C[i] = A[i] + B[i];
}
s = te.create_schedule(C.op)

fadd = tvm.build(s, [A, B, C], target, name="myadd")