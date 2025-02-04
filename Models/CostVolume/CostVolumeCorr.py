from torch import nn


class CostVolume(nn.Module):
    def __init__(self, max_disp, down):
        super(CostVolume, self).__init__()
        self.volume_size = int(max_disp / down)

    def forward(self, left_feature, right_feature):
        b, c, h, w = left_feature.size()
        cost_volume = left_feature.new_zeros(b, self.volume_size, h, w)

        for i in range(self.volume_size):
            if i > 0:
                cost_volume[:, i, :, i:] = (left_feature[:, :, :, i:] *
                                            right_feature[:, :, :, :-i]).mean(dim=1)
            else:
                cost_volume[:, i, :, :] = (left_feature * right_feature).mean(dim=1)
        cost_volume = cost_volume.contiguous()
        return cost_volume
