# 使用张量表达式和 Schedule

## 1 TVM 中的 Schedule 原语

通过 TVM 提供的各种原语来调度计算。

```python
from tvm import te
```

Scheduel 是一组计算转换，可用于转换程序中的循环计算。

声明变量：`n = te.vat("n")`

Scheduel 可由算子列表创建，它默认以行优先的方式串行计算张量

`lower` 会将计算从定义转换为实际可调用的函数

**一个 Schedule 由多个 Stage 组成**，一个 Stage 代表一个操作的 schedule。每个 stage 的调度都有多种方法：

### 1.1 split

`split` 可根据 factor 将指定 axis 拆分为两个 axis

```python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] * 2, name="B")

s = te.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
print(tvm.lower(s, [A, B], simple_mode=True))		
```

算子 B 进行了 split，实际在计算的时候，会将原本该 axis 上的运算拆分为两层 for 循环，内层 for 循环是从 0 到 31（即 factor 个元素）。

注意到，拆分需要指定算子的 `.op.axis[0]`，同时指定每次内层循环的计算数目 factor.

