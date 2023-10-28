import tvm
import tvm.testing
from tvm import te
import numpy as np

target = tvm.target.Target(target="llvm -mcpu=tigerlake", host="llvm -mcpu=tigerlake")

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

factor = 4	# 和 CPU 的线程数量匹配

outer, inner = s[C].split(C.op.axis[0], factor=factor)

s[C].parallel(outer)
s[C].vectorize(inner)

fadd_vector = tvm.build(s, [A, B, C], target, name="myadd_vector")

dev = tvm.device(target.kind.name, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

fadd_vector(a, b, c)

tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())       # 验证结果