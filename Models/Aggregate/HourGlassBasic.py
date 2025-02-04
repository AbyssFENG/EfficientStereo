import torch.nn as nn
import torch.nn.functional as F
import torch
import time


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
                                   padding=pad, stride=stride, bias=False),
                         nn.BatchNorm3d(out_planes))


class HourGlass(nn.Module):
    def __init__(self, inplanes):
        super(HourGlass, self).__init__()

        self.first = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.BatchNorm3d(inplanes),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.BatchNorm3d(inplanes),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=(1, 2, 2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 4, kernel_size=3, stride=(1, 2, 2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 4, inplanes * 4, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 4, inplanes * 2, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2),
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2),
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x):

        out1 = self.first(x)
        out2 = self.conv1(out1)
        out3 = F.relu(self.conv2(out2), inplace=True)

        out4 = self.conv3(out3)
        out5 = self.conv4(out4)

        out6 = F.relu(self.conv5(out5) + out3, inplace=True)

        out = self.conv6(out6) + out1

        return out


# test
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.rand(1, 32, 24, 32, 64)
    img = img.to(device)
    print("input = ", img.size())
    test = HourGlass(32)
    test.eval()
    test = nn.DataParallel(test)
    test = test.to(device)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in test.parameters()])))
    All_time = 0
    output = []
    for i in range(100):
        time2 = time.time()
        output = test(img)
        time1 = time.time() - time2
        if i > 1:
            All_time += time1
        print('index', i+1, "FPS = ", 1 / (time1 + 0.00001))
        if i == 99:
            print("{:.3g}G".format(torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0))
    print("Average_FPS = ", 1 / (All_time / 98))
    print("output =", output.size())


if __name__ == '__main__':
    main()
