from PIL import Image
from torch.utils import data
import random
import numpy as np
from torchvision import transforms
from Dataloader import readpfm

"""
    用来加载PFM格式的标签
    Used to load disparity maps in PFM format

    使用训练集使用随机裁切 256*512
    Use training set with random cropping of 256 * 512

    返回Tensor格式
    Return the format of Tensor
"""


imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def disparity_loader(path):
    return readpfm.readPFM(path)


def scale_crop(normalize):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    return transforms.Compose(t_list)


def get_transform():
    normalize = imagenet_stats

    return scale_crop(normalize=normalize)


class MyImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, training, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left_path = self.left[index]
        right_path = self.right[index]
        disp_L_path = self.disp_L[index]
        # left

        left = Image.open(left_path).convert('RGB')

        right = Image.open(right_path).convert('RGB')

        disp_L, scaleL = self.dploader(disp_L_path)
        disp_L = np.ascontiguousarray(disp_L, dtype=np.float32)

        # 训练使用
        if self.training:
            w, h = left.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right.crop((x1, y1, x1 + tw, y1 + th))

            dataL = disp_L[y1:y1 + th, x1:x1 + tw]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:

            processed = get_transform()
            left_img = processed(left)
            right_img = processed(right)

            return left_img, right_img, disp_L

    def __len__(self):
        return len(self.left)
