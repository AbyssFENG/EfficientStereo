import tensorrt as trt

# 定义 ONNX 和 TensorRT 引擎的路径
onnx_model_path = "EfficientStereo_fused.onnx"
trt_engine_path = "EfficientStereo_fp32.engine"

# 创建 TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 创建 Builder、Network 和 Parser
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# 读取 ONNX 并解析
with open(onnx_model_path, "rb") as f:
    if not parser.parse(f.read()):
        print("❌ ONNX 解析失败！")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

print("✅ ONNX 解析成功！")

# 创建 BuilderConfig
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 设置 1GB 工作区内存
config.flags = 0  # FP32 不需要特殊 flag

# **使用新方法 build_serialized_network**
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    print("❌ TensorRT Engine 构建失败！")
    exit()

# **反序列化为可用的 TensorRT Engine**
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)

# **保存 TensorRT Engine 到文件**
with open(trt_engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"✅ TensorRT Engine 生成完成：{trt_engine_path}")
