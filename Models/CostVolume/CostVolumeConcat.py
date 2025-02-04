import torch
from torch import nn


class CostVolumeConcat(nn.Module):
    def __init__(self, max_disp, down):
        super(CostVolumeConcat, self).__init__()
        self.max_disp = max_disp
        self.down = down

    def forward(self, left, right):
        """
        构建视差相关的代价体积（Cost Volume）。
        参数:
            left (torch.Tensor): 左视图张量，形状为 (batch_size, channels, height, width)
            right (torch.Tensor): 右视图张量，形状为 (batch_size, channels, height, width)
        返回:
            cost (torch.Tensor): 构建好的代价体积，形状为 (batch_size, channels * 2, max_disp//down, height, width)
        """
        batch_size, channels, height, width = left.size()
        disp_levels = self.max_disp // self.down

        # 确定设备和数据类型
        device = left.device
        dtype = left.dtype

        # 根据训练模式决定是否追踪梯度
        if self.training:
            # 训练模式下，允许追踪梯度
            cost = torch.zeros(
                batch_size,
                channels * 2,
                disp_levels,
                height,
                width,
                device=device,
                dtype=dtype
            )
            for i in range(disp_levels):
                if i > 0:
                    # 左视图右移 i 像素，右视图左移 i 像素
                    cost[:, :channels, i, :, i:] = left[:, :, :, i:]
                    cost[:, channels:, i, :, i:] = right[:, :, :, :-i]
                else:
                    # 视差为零时，不需要移动
                    cost[:, :channels, i, :, :] = left
                    cost[:, channels:, i, :, :] = right
        else:
            # 评估模式下，不追踪梯度
            with torch.no_grad():
                cost = torch.zeros(
                    batch_size,
                    channels * 2,
                    disp_levels,
                    height,
                    width,
                    device=device,
                    dtype=dtype
                )
                for i in range(disp_levels):
                    if i > 0:
                        cost[:, :channels, i, :, i:] = left[:, :, :, i:]
                        cost[:, channels:, i, :, i:] = right[:, :, :, :-i]
                    else:
                        cost[:, :channels, i, :, :] = left
                        cost[:, channels:, i, :, :] = right

        # 确保内存连续
        cost = cost.contiguous()

        return cost
