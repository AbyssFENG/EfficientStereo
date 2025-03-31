import tensorrt as trt

# 定义 ONNX 和 TensorRT 引擎的路径
onnx_model_path = "EfficientStereo.onnx"
trt_engine_path = "EfficientStereo_fp16.engine"

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
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB 工作区内存

# **启用 FP16 计算**
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)  # ✅ 修正的 FP16 代码
    print("✅ 已启用 FP16 模式")
else:
    print("⚠️ 你的 GPU 可能不支持 FP16，将使用 FP32")

# **使用 build_serialized_network（TensorRT 10.9 适配）**
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    print("❌ TensorRT Engine 构建失败！")
    exit()

# **反序列化 Engine**
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)

# **保存 Engine**
with open(trt_engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"✅ TensorRT FP16 Engine 生成完成：{trt_engine_path}")
