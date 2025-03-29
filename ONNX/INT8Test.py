import onnxruntime as ort
import numpy as np
import cv2
# 载入量化后的 INT8 模型
model_int8 = "EfficientStereo_quantized.onnx"


sess_options = ort.SessionOptions()
sess_options.log_severity_level = 0  # 让日志更加详细
session = ort.InferenceSession(model_int8, providers=["CUDAExecutionProvider"])
# 打印模型的输入信息
input_details = session.get_inputs()
for inp in input_details:
    print(f"Input Name: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")

# 打印模型的输出信息
output_details = session.get_outputs()
for out in output_details:
    print(f"Output Name: {out.name}, Shape: {out.shape}, Type: {out.type}")


left,right = r'G:\EfficientStereo\ONNX\Test_image\000001_10.png', r'G:\EfficientStereo\ONNX\Test_image2\000001_10.png'
# 读取 1.png 并转换格式
def pad_to_size(image, target_width, target_height):
    """填充图片到指定尺寸 (target_width, target_height)"""
    h, w, c = image.shape

    # 计算需要填充的像素
    top = (target_height - h) // 2
    bottom = target_height - h - top
    left = (target_width - w) // 2
    right = target_width - w - left

    # 进行填充（使用黑色填充，也可以改为 (255, 255, 255) 白色）
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image
def read(image_path):
    image = cv2.imread(image_path)  # 读取图片 (默认是 BGR 格式)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    target_width, target_height = 1280, 384
    h, w, _ = image.shape
    if w < target_width or h < target_height:
        image = pad_to_size(image, target_width, target_height)
    # image = cv2.resize(image, (1280, 384))  # 调整到 (宽=1280, 高=384)
    image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    image = np.transpose(image, (2, 0, 1))  # 变换通道顺序 (H, W, C) → (C, H, W)
    input_data = np.expand_dims(image, axis=0)  # 增加 batch 维度，变成 (1, 3, 384, 1280)
    return input_data

input_l = read(left)
input_r = read(right)


# 运行推理
outputs = session.run(None, {"input": input_l, "input.31": input_r})
print("Available providers:", session.get_providers())
print("Current provider:", session.get_provider_options())
# 打印输出信息
for i, output in enumerate(outputs):
    print(f"Output {i} - Shape: {output.shape}, Dtype: {output.dtype}")

output_np = np.array(outputs[0]).squeeze()
output_np = cv2.normalize(output_np, None, 0, 255, cv2.NORM_MINMAX)
output_np = np.uint8(output_np)

# 显示图片
cv2.imshow("Model Output", output_np)
cv2.waitKey(0)
cv2.destroyAllWindows()







model_fp32 = "EfficientStereo_fused.onnx"
session_fp32 = ort.InferenceSession(model_fp32, providers=["CUDAExecutionProvider"])

# 运行 FP32 模型
output_fp32 = session_fp32.run(None, {"input": input_l, "input.31": input_r})

# 计算 INT8 和 FP32 结果的误差
for i in range(len(outputs)):
    diff = np.abs(output_fp32[i] - outputs[i]).mean()
    print(f"Output {i} - Mean Absolute Difference: {diff:.6f}")