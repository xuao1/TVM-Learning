# 使用张量表达式和 Schedule

## 1 TVM 中的 Schedule 原语

通过 TVM 提供的各种原语来调度计算。

```python
from tvm import te
```

Scheduel 是一组计算转换，用于指定执行特定操作的策略和顺序。

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

会导致连续性的访问

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

> 可以注意到，这时生成的可执行代码没有了原本的两层循环：
>
> 在第二个代码片段中，通过  `bind`  操作将这些循环维度绑定到了 GPU 的线程块（`blockIdx.x`）和线程（`threadIdx.x`）上。在这种情况下，循环不再是显式的嵌套 `for` 循环，而是将执行分布在 GPU 的线程块和线程上，这种执行方式是并行的。
>
> 当执行这段代码时，GPU 会为每个线程块和每个线程块中的每个线程分配实际的执行实例

### 1.6 compute_at

对于包含多个算子的 schedule，TVM 默认会分别计算。

```python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")
s = te.create_schedule(C.op)
```

执行顺序是，先计算完全部的 B，再计算 C，即会有两个 for 循环：

![image-20231105104805286](..\img\image-20231105104805286.png)

而加入以下 `compute_at` 语句：

```python
s[B].compute_at(s[C], C.op.axis[0])
```

这句话的字面意思是将 B 的计算移动到 C 计算的首个 axis 中，实际效果就是将 B 的计算合并到 C 的计算中，即对于每个 i，先计算 `B[i]`，接着计算 `C[i]`：

![image-20231105104927117](..\img\image-20231105104927117.png)

### 1.7 compute_inline

`compute_inline` 可将 stage 标记为 inline，然后扩展计算体，并将其插入到需要张量的地址。

```python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")
s = te.create_schedule(C.op)
s[B].compute_inline()
```

该操作会将算子 B 的计算放在 C 中：

![image-20231105105839271](..\img\image-20231105105839271.png)

没有显式的 B 了

### 1.8 compute_root

`compute_root` 可将一个 stage 的计算移动到 root，更具体地说，`compute_root` 会使得计算在一个单独的循环中执行，而不是嵌入到其他计算中。

> 目前看来，是和 `compute_root` 以及 `compute_inline` 是逆操作

```python
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")
s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
s[B].compute_root()
```

即，原本执行完 `compute_at` 或者 `compute_inline` 后，算子 B 会和 C 一起运算，再执行 `cpmpute_root` 后，又会单独在一个 for 循环中计算。

### 1.9 总结

上述是若干 Scheduel 原语，可以看到这几个特点：

+ create_scheduel 需要的参数是算子的 `.op`
+ 执行 Scheduel 原语的是 `s[B]`，即 Schedule 的算子



## 2 归约 reduce

关联归约算子（如 sum/max/min）是线性代数运算的典型构造块。

求矩阵的行和：

首先用 `re.reduce_axis` 声明一个归约轴，然后 `te.sum` 接收要归约的表达式和规约轴。

```python
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), name="k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
```

###  2.1 Schedule 归约

```python
s = te.create_schedule(B.op)
```

![image-20231105133652642](..\img\image-20231105133652642.png)

reduction 轴类似于普通轴，可以拆分。

```python
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
```

![image-20231105134052514](..\img\image-20231105134052514.png)

即将原本的内层循环求行和（归约操作）拆分为两层循环。

### 2.2 归约因式分解和并行化

构建归约时不能简单地在 reduction 轴上并行化，需要划分归约，将局部规约结果存储在数组中，然后再对临时数组进行归约。

> 归约操作通常需要将多个元素的值组合成一个单一的结果。
>
> 所以对归约操作并行化时需要小心处理，因为这涉及到多个线程同时更新同一个输出位置。
>
> 为了安全地并行化规约操作，可以采取分治策略：
>
> 局部规约，合并局部结果，全局规约
>
> 核心在于对于规约轴上的数据，**局部规约就是跨步规约**

`rfactor` 原语对计算进行了上述重写，在下面的调度中，B 的结果被写入一个临时结果 B.rf，分解后的维度成为 B.rf 的第一个维度。

```python
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
```

![image-20231105135950171](C:\Users\15370\AppData\Roaming\Typora\typora-user-images\image-20231105135950171.png)

首先看第一个循环：

![image-20231105140730237](C:\Users\15370\AppData\Roaming\Typora\typora-user-images\image-20231105140730237.png)

最外层循环：k_inner 从 0 到 15，首先 k 即为归约轴，即列数，分成了若干组，每组 16 个元素

次外层循环：i 从 0 到 n-1，即每行

`B_rf_1` 存储局部规约结果，首先初始化 k_inner * n + i，即可以视为二维数组 `B_rf_1[16][n]`，`B_rf_1[1][2]` 存储的是第 2 行，所有组的第一个元素的和，即先跨步归约（求和）

最内层循环就是做跨步求和的.

实际做的事情就是：

**将数组 `A[n][m]` split 后是为三维数组 `A[n][k_outer][16]`，那么就是对 `A[n][][16]` 的第二维求和，结果是 `B_rf_1[16][n]`**，对 A 数组的取数是跨步取数

再看第二个循环：

![image-20231105140716422](C:\Users\15370\AppData\Roaming\Typora\typora-user-images\image-20231105140716422.png)

最外层循环是每行，最终的规约结果存储在 `B_2` 数组中

内层循环则是对于每行（比如说，第 ax0 行），**对 `B_rf_i[][ax0]` 归约，结果存到 `B_2[ax0]`** ，可以看到，对于数组 `B_rf_1`，仍然是跨步取数。

`rfactor` 用于对归约计算进行变换，目的是优化归约操作的并行执行。

当一个归约操作在GPU等并行硬件上执行时，如果直接进行归约，那么所有的并行线程必须等待所有的输入都计算完成才能开始归约过程，这会导致不必要的同步和潜在的资源闲置。通过 `rfactor`，TVM 能够将归约拆分为两个阶段：

1. **局部归约**（Local Reduction）：每个线程或线程块计算归约操作的一个部分，这可以并行进行而不需要任何同步。
2. **全局归约**（Global Reduction）：然后将所有局部归约的结果合并起来以得到最终的归约结果。

在这个特定的例子中，`rfactor` 被用来将原始的 `B` 张量计算拆分成两个部分，通过创建一个新的中间表示 `BF`。这个中间表示 `BF` 是对原始张量 `A` 进行部分归约的结果。每个 `BF` 元素都是输入 `A` 对应行的**部分和**（由 `ki` 定义的部分）。

### 2.3 跨线程归约

接下来可以在因子轴上进行并行化，这里 B 的 reduction 轴被标记为线程，如果唯一的 reduction 轴在设备中可以进行跨线程归约，则 TVM 允许将 reduction 轴标记为 thread。

```
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
xo, xi = s[B].split(s[B].op.axis[0], factor=32)
s[B].bind(xo, te.thread_axis("blockIdx.x"))
s[B].bind(xi, te.thread_axis("threadIdx.y"))
tx = te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
s[B].set_store_predicate(tx.var.equal(0))
fcuda = tvm.build(s, [A, B], "cuda")
```

可以看到，第三行将 B 的第一个维度拆分成两部分，分别绑定到 `blockIdx.x` 和 `threadIdx.y`. 在每个线程块内部，将有一个 `threadIdx.y` 的维度用于执行这 32 个迭代。

之后定义了一个线程轴 `tx`，表示 CUDA 中的线程索引 `threadIdx.x`，将 B 的归约轴绑定到线程轴 `tx`.

之后指定 BF 在 B 的归约轴上计算。

倒数第二行代码设置了一个存储谓词，确保只有当 `tx` 为 0 时，才会执行对 B 的写操作。

最后，调用 `tvm.build` 将 Scheduel 编译成 CUDA 代码。

编译结果查看：`print(fcuda.imported_modules[0].get_source())`

+ 如果不加 `s[BF].compute_at(s[B], s[B].op.reduce_axis[0])`：

  ![image-20231106104607691](C:\Users\15370\AppData\Roaming\Typora\typora-user-images\image-20231106104607691.png)

  可以看到，计算 `B_rf` 的部分并没有体现出 thraedId，这个操作将在 CUDA 的全局内存上进行。

  当 `B_rf` 的计算不绑定到线程上时，它是按照 TVM 内部默认的串行或并行策略执行的。在生成的代码中，这部分循环可能默认为串行执行或者由 TVM 的 runtime 自动分配线程执行，具体取决于调度策略。

+ 加上 `s[BF].compute_at(s[B], s[B].op.reduce_axis[0])`：































