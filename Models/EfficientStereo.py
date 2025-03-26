import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Models.Aggregate import HourGlassDisp
from Models.CostVolume import CostVolumeGWC, CostVolumeConcat
from Models.Feature import LightFeature, ResNet18_34, ResNet50_152, MobileNet, EfficientNet, StarNet, RepVit, ViT
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis



class DisparityRegression(nn.Module):
    # 关键点：把概率转换为视差
    def __init__(self, max_disp):
        super(DisparityRegression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(max_disp)), [1, max_disp, 1, 1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data, 1, keepdim=True)
        return out


class Classifier(nn.Module):
    def __init__(self, inplanes):
        super(Classifier, self).__init__()
        self.Classify = nn.Sequential(nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(inplanes),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(inplanes, 1, kernel_size=3, padding=1, stride=1, bias=False))

    def forward(self, x):
        return self.Classify(x)


# @model_decorator
class EfficientStereo(nn.Module):
    def __init__(self, max_disp):
        super(EfficientStereo, self).__init__()
        self.max_disp = max_disp
        # LightFeature, ResNet18_34, ResNet50_152, MobileNet, EfficientNet, StarNet, RepVit, ViT
        self.FeatureExtraction = LightFeature.FeatureExtract()
        self.BuildVolume = CostVolumeGWC.CostVolumeGWC(int(max_disp / 8), 32)
        # self.BuildVolume = CostVolumeConcat.CostVolumeConcat(max_disp, 8)
        self.CostAgg = HourGlassDisp.HourGlass(32)
        
        self.Classify = Classifier(32)
        self.Regression = DisparityRegression(max_disp)
    def forward(self, left, right):
        left_feature = self.FeatureExtraction(left)

        # import matplotlib.pyplot as plt
        # # 选择第一个 batch 和第一个通道的特征图
        # single_feature_map = left_feature[0, 0, :, :].cpu().numpy()  # 转换为 numpy 数组
        # # 使用 matplotlib 显示特征图
        # plt.figure(figsize=(single_feature_map.shape[1] / 20, single_feature_map.shape[0] / 20), dpi=100)
        # plt.imshow(single_feature_map, cmap='viridis')
        #
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 去除空白填充
        #
        # plt.savefig('../Feature/RepVit30.png')
        # plt.show()

        right_feature = self.FeatureExtraction(right)

        cost = self.BuildVolume(left_feature, right_feature)

        cost_aggregated = self.CostAgg(cost)
        cost_classified = self.Classify(cost_aggregated)

        cost_up_sampled = F.interpolate(cost_classified, size=[self.max_disp, left.size(2), left.size(3)],
                                        mode='trilinear', align_corners=False)
        cost_squeezed = torch.squeeze(cost_up_sampled, 1)
        pred = F.softmax(cost_squeezed, dim=1)
        pred = self.Regression(pred)

        return pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.rand(1, 3, 384, 1280)
    img = img.to(device)
    max_disp = 192
    print("input = ", img.size())
    test = EfficientStereo(max_disp)
    test.eval()
    # test = nn.DataParallel(test)
    test = test.to(device)

    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    input_data = (img, img)
    # torchinfo
    summary(test, input_data=input_data)
    torch.cuda.empty_cache()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = test(img,img)

    # 创建 CUDA 事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 测量推理时间
    with torch.no_grad():
        torch.cuda.synchronize()  # 同步 GPU 操作
        start_event.record()
        _ = test(img,img)
        end_event.record()
        torch.cuda.synchronize()  # 同步 GPU 操作

    # 计算推理时间（毫秒）
    inference_time = start_event.elapsed_time(end_event)
    print(f"Inference time: {inference_time:.4f} MS")

    torch.cuda.empty_cache()
    # fvcore
    flops = FlopCountAnalysis(test, input_data)
    print(f"FLOPs: {flops.total() / 1e9:.3f} GFLOPs")
    # parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in test.parameters()])))

    # 记录峰值显存占用
    peak_memory = torch.cuda.max_memory_allocated()
    # 计算显存占用
    print(f"Initial memory: {initial_memory / (1024 ** 3):.2f} GB")
    print(f"Peak memory: {peak_memory / (1024 ** 3):.2f} GB")
    print(f"Total memory: {(peak_memory - initial_memory) / (1024 ** 3):.2f} GB")

    print("Current memory: {:.4g}G".format(torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0))
    output = test(img, img)
    print("output = ", output.size())


if __name__ == '__main__':
    main()
