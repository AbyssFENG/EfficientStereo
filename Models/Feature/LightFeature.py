import torch
from torch import nn
import time


def conv_bn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


# ConvTranspose2D + BatchNormalization
def trans_conv_bn(in_planes, out_planes, kernel_size, stride, pad, out_pad):

    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                                            stride=stride, padding=pad, output_padding=out_pad),
                         nn.BatchNorm2d(out_planes))


class FeatureExtract(nn.Module):
    def __init__(self):
        super(FeatureExtract, self).__init__()
        # size=1/2  3层
        self.first_conv = nn.Sequential(conv_bn(in_planes=3, out_planes=32, kernel_size=3, stride=2, pad=1, dilation=1),
                                        nn.ReLU(inplace=True),
                                        conv_bn(32, 32, 3, 1, 1, 1),
                                        nn.ReLU(),
                                        conv_bn(32, 32, 3, 1, 1, 1),
                                        nn.ReLU()
                                        )

        # size=1/4
        self.conv1 = nn.Sequential(conv_bn(32, 64, 3, 2, 1, 1),
                                   nn.ReLU())
        # size=1/8
        self.conv2 = nn.Sequential(conv_bn(64, 128, 3, 2, 1, 1),
                                   nn.ReLU())

    def forward(self, x):
        out = self.first_conv(x)
        out1 = self.conv1(out)
        out1 = self.conv2(out1)
        return out1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.rand(1, 3, 384, 1280)
    img = img.to(device)
    print("input = ", img.size())
    test = FeatureExtract()
    test.eval()
    test = nn.DataParallel(test)
    test = test.to(device)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in test.parameters()])))
    output = []
    All_time = 0
    for i in range(100):
        time2 = time.time()
        output = test(img)
        time1 = time.time() - time2
        if i > 1:
            All_time += time1
        print(i, "FPS = ", 1 / (time1 + 0.00001))
        if i == 99:
            print("{:.3g}G".format(torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0))
    print("Average_FPS = ", 1 / (All_time / 98))
    print("output =", output.size())


if __name__ == '__main__':
    main()
