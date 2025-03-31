import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2

# 定义 TensorRT 引擎路径
trt_engine_path = "EfficientStereo_fp32.engine"

# 创建 Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 反序列化引擎
def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

# 创建 TensorRT 运行时并加载引擎
runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(runtime, trt_engine_path)

print("✅ TensorRT 引擎加载完成！")

context = engine.create_execution_context()

# 获取输入和输出张量的名称
input_name_1 = engine.get_tensor_name(0)  # 第一个输入张量的名称
input_name_2 = engine.get_tensor_name(1)  # 第二个输入张量的名称
output_name = engine.get_tensor_name(2)   # 输出张量的名称

# 获取输入和输出张量的形状
input_shape_1 = engine.get_tensor_shape(input_name_1)
input_shape_2 = engine.get_tensor_shape(input_name_2)
output_shape = engine.get_tensor_shape(output_name)

# 计算所需的内存大小（假设数据类型为 float32）
input_size_1 = int(np.prod(input_shape_1) * np.dtype(np.float32).itemsize)
input_size_2 = int(np.prod(input_shape_2) * np.dtype(np.float32).itemsize)
output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)

# 分配 GPU 内存
d_input_1 = cuda.mem_alloc(input_size_1)
d_input_2 = cuda.mem_alloc(input_size_2)
d_output = cuda.mem_alloc(output_size)

# 创建 CUDA 流
stream = cuda.Stream()

print("✅ GPU 内存分配完成！")

# 准备输入数据（根据实际情况替换为您的数据）
def preprocess_image(image_path, input_shape):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 你的数据是彩色图
    img = cv2.resize(img, (input_shape[3], input_shape[2]))  # 调整大小，确保宽度是 input_shape[3]，高度是 input_shape[2]
    img = img.astype(np.float32) / 255.0  # 归一化到 [0,1]
    img = np.transpose(img, (2, 0, 1))  # 变换通道顺序 (H, W, C) → (C, H, W)
    img = np.expand_dims(img, axis=0)  # 添加批次维度 (1, H, W, C)
    return img

input_shape = (1, 3, 384, 1280)  # 修改为你的模型输入尺寸
input_data_1 = preprocess_image(r'G:\EfficientStereo\ONNX\Test_image\000001_10.png', input_shape)
input_data_2 = preprocess_image(r'G:\EfficientStereo\ONNX\Test_image2\000001_10.png', input_shape)
input_data_1 = np.ascontiguousarray(input_data_1)
input_data_2 = np.ascontiguousarray(input_data_2)

print(input_data_1.shape)  # 应该是 (1, 3, 384, 1280)
print(input_data_2.shape)  # 应该是 (1, 3, 384, 1280)

# 将输入数据传输到 GPU
cuda.memcpy_htod_async(d_input_1, input_data_1, stream)
cuda.memcpy_htod_async(d_input_2, input_data_2, stream)

# 设置输入和输出张量的地址
context.set_tensor_address(input_name_1, int(d_input_1))
context.set_tensor_address(input_name_2, int(d_input_2))
context.set_tensor_address(output_name, int(d_output))

# 执行推理
context.execute_async_v3(stream.handle)

# 分配用于存储输出数据的主机内存
output_data = np.empty(output_shape, dtype=np.float32)

# 将输出数据从 GPU 传输回主机
cuda.memcpy_dtoh_async(output_data, d_output, stream)

# 同步流以确保所有操作完成
stream.synchronize()

print("✅ 推理完成！")
print("输出结果：", output_data)
output_data_uint8 = (output_data).astype(np.uint8).squeeze()
cv2.imshow('Output Data Visualization', output_data_uint8)
cv2.waitKey(0)  # 等待按键
cv2.destroyAllWindows()  # 关闭窗口