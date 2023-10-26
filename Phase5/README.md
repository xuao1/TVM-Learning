# 5 使用 Python 接口（AutoTVM）编译和优化模型

本节使用 TVM 的 Python API 来实现，完成：

+ 使用 TVM runtime 编译模型，运行模型进行预测
+ 使用 TVM 进行调优，使用调优数据重新编译模型，运行优化模型进行预测

```python
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
```

两个文件，均为完整执行了从下载模型、下载和预处理侧视图像、编译模型、运行、输出数据后处理：

+ `tvmpython.py`：未调优版本，包含上述全部过程
+ `records.json`：自动调优（autoschedule）版本，额外加入了调优过程，此外，它的编译是依赖调优结果进行的

