from tvm.driver import tvmc
import numpy as np

# 加载模型
model = tvmc.load("my_model.onnx")

# 调优
log_file = "records.json"
# tvmc.tune(model, target="llvm -mcpu=broadwell", enable_autoscheduler = True, tuning_records=log_file)
# tvmc.tune(model, target="llvm -mcpu=broadwell", prior_records=log_file)	# 会使用先前的 tuning 结果，并在此基础上继续优化

# 编译模型
# package = tvmc.compile(model, target="llvm")    # 未使用调优
package = tvmc.compile(model, target="llvm", tuning_records=log_file) # 使用调优

# 运行模型
input_data = np.load("imagenet_cat.npz")
result = tvmc.run(package, device="cpu", inputs=input_data)
np.savez("predictions.npz", output_0=result.outputs['output_0'])

