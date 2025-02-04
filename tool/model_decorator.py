import torch
import torch.nn as nn

from functools import wraps
from time import time
from torch.profiler import profile, ProfilerActivity

from fvcore.nn import FlopCountAnalysis
from contextlib import contextmanager

profiling = False


@contextmanager
def enable_profiling():
    global profiling
    profiling = True
    try:
        yield
    finally:
        profiling = False


# 修饰器模板：用于计时功能
def profile_decorator(name, profile_start=0, profile_end=1):
    def profile_wrapper(func):
        @wraps(func)
        def profiled_func(*args, **kwargs):
            # 判断是否在监控范围内
            if profiled_func.call_time < profile_end:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time()
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    print(f"Exception during '{name}': {e}")
                    raise e
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt = time() - start_time
                if profiled_func.call_time >= profile_start:
                    print(f"{name} takes {dt:.6f} seconds at call time {profiled_func.call_time}")
                profiled_func.call_time += 1
            else:
                result = func(*args, **kwargs)
            return result
        # 初始化调用次数为零
        profiled_func.call_time = 0
        return profiled_func
    return profile_wrapper


def disp_model_info(module: nn.Module, name: str, opt: str, input=None):
    if opt == 'print_params':
        total_params = sum(p.numel() for p in module.parameters())
        print(f"Number of parameters in {name} module: {total_params}")
        return
    elif opt == 'print_Gflops':
        if input is None:
            print(f"No input provided for {name} module FLOPs calculation.")
            return
        try:
            with enable_profiling():
                # 使用 torch.profiler 统计 FLOPs
                for _ in range(10):  # 进行多次前向传播以热身
                    module(*input)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False) as prof:
                    # 调用原始的 forward 方法，避免递归
                    module.forward(*input)
                # 打印统计结果

                flops = FlopCountAnalysis(module, input).total()
            gflops = flops / 1e9
            print(f"Number of FLOPs in {name} module: {flops} ({gflops:.3f} GFLOPs)")

            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        except Exception as e:
            print(f"Exception during FLOPs profiling in '{name}': {e}")


# 类修饰器：用于在初始化和前向传播时调用 disp_model_info
def model_decorator(model_cls):
    original_init = model_cls.__init__
    original_forward = model_cls.forward

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # 执行原始的初始化方法
        original_init(self, *args, **kwargs)
        # 保存原始的 forward 方法，避免被装饰后的递归调用
        self._original_forward = original_forward.__get__(self, model_cls)
        # 初始化时调用 print_params
        disp_model_info(self, self.__class__.__name__, 'print_params')

    model_cls.__init__ = new_init

    @wraps(original_forward)
    @profile_decorator(name=f'{model_cls.__name__} forward', profile_start=10, profile_end=11)
    def new_forward(self, *args, **kwargs):
        # 检查是否是第一次前向传播
        if not profiling and not hasattr(self, '_flops_printed'):
            # 获取输入数据，用于 FLOPs 计算
            input_data = tuple(args) if len(args) > 0 else kwargs.get('input', None)
            if input_data is not None:
                disp_model_info(self, self.__class__.__name__, 'print_Gflops', input=input_data)
            self._flops_printed = True
        # 调用原始的 forward 方法
        return self._original_forward(*args, **kwargs)

    model_cls.forward = new_forward
    return model_cls
