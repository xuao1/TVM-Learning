import onnx
from tvm.contrib.download import download_testdata
from PIL import Image
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor


###################################################################
# 下载和加载 ONNX 模型
model_url = (
    "https://github.com/onnx/models/raw/main/"
    "vision/classification/resnet/model/"
    "resnet50-v2-7.onnx"
)

model_path = download_testdata(model_url, "resnet50-v2-7.onnx", module="onnx")
onnx_model = onnx.load(model_path)

# 为 numpy 的 RNG 设置 seed，得到一致的结果
np.random.seed(0)

print("The model is downloaded from %s!" % (model_url))


###################################################################
# 下载、预处理和加载测试图像
img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

# 重设大小为 224x224
resized_image = Image.open(img_path).resize((224, 224))
img_data = np.asarray(resized_image).astype("float32")

# 输入图像是 HWC 布局，而 ONNX 需要 CHW 输入，所以转换数组
img_data = np.transpose(img_data, (2, 0, 1))

# 根据 ImageNet 输入规范进行归一化
imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

# 添加 batch 维度，期望 4 维输入：NCHW。
img_data = np.expand_dims(norm_img_data, axis=0)

print("The image is downloaded from %s!" % (img_url))


###################################################################
# 使用 Relay 编译模型
input_name = "data" # 输入名称可能因模型类型而异，可用 Netron 工具检查输入名称
shape_dict = {input_name: img_data.shape}

# 将模型导入到 Relay
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# 用标准优化，将模型构建到 TVM 库中
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# 从库中创建一个 TVM 计算图 runtime 模块
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

print("Relay frontend is done.")


###################################################################
# 在 TVM Runtime 执行
# dtype = "float32"
module.set_input(input_name, img_data)
module.run()
output_shape = (1, 1000)
tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

print("TVM runtime is done.")


###################################################################
# 输出后处理
from scipy.special import softmax

# 下载标签列表
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

# 打开输出文件并读取输出张量
scores = softmax(tvm_output)
scores = np.squeeze(scores)
ranks = np.argsort(scores)[::-1]
for rank in ranks[0:5]:
    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))