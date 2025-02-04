from torch import nn


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


class CostVolumeGWC(nn.Module):
    def __init__(self, maxdisp, num_group):
        super(CostVolumeGWC, self).__init__()
        self.maxdisp = maxdisp
        self.num_group = num_group

    def forward(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, self.num_group, self.maxdisp, H, W])
        for i in range(self.maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                               self.num_group)
            else:
                volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, self.num_group)
        volume = volume.contiguous()
        return volume
