# 4 使用 TVMC Python 入门：TVM 的高级 API

这一部分是使用 TVM 的 Python API，对应为

```python
from tvm.driver import tvmc
```

完成：

1. 使用 TVM runtime 编译模型，运行模型进行预测

2. 使用 TVM 进行调优，使用调优数据重新编译模型，运行优化模型进行预测

包含的文件包括：

+ `tvmcpythonintro.py`：调用并执行 TVM 功能的 Python 脚本文件
+ `records.json`：自动调优（autoschedule）生成的 调优记录文件

但是目前对于调优记录文件的加载仍有问题，待解决

