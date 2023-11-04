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

`split` 可根据 **factor** 将指定 axis 拆分为两个 axis

```python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] * 2, name="B")

s = te.create_schedule(B.op)
xo, xi = s[B].split(B.op.axis[0], factor=32)
```

算子 B 进行了 split，实际在计算的时候，会将原本该 axis 上的运算拆分为两层 for 循环，内层 for 循环是从 0 到 31（即 factor 个元素）。以上述代码为例，可以**理解为拆分成若干个小组，每组 32 个元素**。

注意到，**拆分需要指定算子的 `.op.axis[0]`，同时指定每次内层循环的计算数目 factor.**

也可以用 **nparts** 来拆分 axis，它拆分 axis 的方式与 factor 相反。即外层循环的执行次数 nparts 个，可以**理解为拆分成 32 个小组，每个小组若干个元素**（平均分）。

### 1.2 tile

`tile` 可在两个 axis 上逐块执行计算。

```python
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j]*2, name="B")

s = te.create_schedule(B.op)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
```

即两个维度均进行 split，而且是 factor 的 split

### 1.3 fuse

`fuse` 可将一个计算的两个连续 axis 融合。

对于上面的 `tile`，可以将 `i.inner` 和 `j.inner` （即 xi 和 yi）融合：

```python
s[B].fuse(xi, yi)	 
```

在这个例子中，原本 xi 是从 0 到 9，yi 是从 0 到 5，那么融合以后就是 0 到 50.

### 1.4 reorder

`reorder` 可按指定的顺序对 axis 重新排序

对于上面的 `tile`，对 axis 重新排序：

```python
s[B].reorder(xi, yo, xo, yi)
```

### 1.5 bind

`bind` 可将指定的 axis 与线程 axis 绑定，常用于 GPU 编程。

```python
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] * 2, name="B")
s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=64)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))
```

首先进行 `split`，将计算分成若干个组，每组计算 64 个元素。

将组绑定到 block 上，将组内的元素绑定到 thread 上。













