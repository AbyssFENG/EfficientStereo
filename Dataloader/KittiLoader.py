import os
import os.path
from PIL import Image
from torch.utils import data
import random
import numpy as np
from torchvision import transforms
import cv2

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def dataloader(filepath):

    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'

    image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

    val_indices = [1, 3, 6, 20, 26, 35, 38, 41, 43, 44, 49, 60, 67, 70, 81, 84, 89, 97, 109, 119, 122, 123, 129, 130,
                   132, 134, 141, 144, 152, 158, 159, 165, 171, 174, 179, 182, 184, 186, 187, 196]

    # 从 image 中提取验证集
    val = [image[i] for i in val_indices]

    # 计算出所有 indices，然后去除验证集的 indices 得到训练集的 indices
    all_indices = set(range(len(image)))
    train_indices = list(all_indices - set(val_indices))

    # 从 image 中提取训练集
    train = [image[i] for i in train_indices]
    # train = image[:160]  # 前160张作为训练
    # val = image[160:]

    left_train = [filepath+left_fold+img for img in train]
    right_train = [filepath+right_fold+img for img in train]
    disp_train_L = [filepath+disp_L+img for img in train]

    left_val = [filepath+left_fold+img for img in val]
    right_val = [filepath+right_fold+img for img in val]
    disp_val_L = [filepath+disp_L+img for img in val]


    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')


def opencv_loader(path):
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def disparity_loader(path):
    return Image.open(path)


def disparity_opencv_loader(path):
    return cv2.imread(path)


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    return transforms.Compose(t_list)


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None):
    normalize = __imagenet_stats
    input_size = 256

    return scale_crop(input_size=input_size,
                      scale_size=scale_size, normalize=normalize)


class MyImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader, openloader=opencv_loader, opendisploader=disparity_opencv_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

        self.cache1 = {}
        self.cache2 = {}
        self.cache3 = {}

    def __getitem__(self, index):
        left_path = self.left[index]
        right_path = self.right[index]
        disp_L_path = self.disp_L[index]

        if left_path not in self.cache1:
            # 如果图像不在缓存中，则加载它
            left = Image.open(left_path).convert('RGB')

            self.cache1[left_path] = left
        else:
            # 如果图像已经在缓存中，则直接返回它
            left = self.cache1[left_path]

        if right_path not in self.cache2:
            # 如果图像不在缓存中，则加载它
            right = Image.open(right_path).convert('RGB')

            self.cache2[right_path] = right
        else:
            # 如果图像已经在缓存中，则直接返回它
            right = self.cache2[right_path]

        if disp_L_path not in self.cache3:
            # 如果图像不在缓存中，则加载它
            disp_L = Image.open(disp_L_path)

            self.cache3[disp_L_path] = disp_L
        else:
            # 如果图像已经在缓存中，则直接返回它
            disp_L = self.cache3[disp_L_path]

        # 训练使用
        if self.training:
            w, h = left.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right.crop((x1, y1, x1 + tw, y1 + th))

            dataL = np.ascontiguousarray(disp_L, dtype=np.float32) / 256
            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:

            dataL = np.ascontiguousarray(disp_L, dtype=np.float32) / 256

            processed = get_transform()
            left_img = processed(left)
            right_img = processed(right)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
