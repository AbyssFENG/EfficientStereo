from torchvision import transforms
import torch
import random

left_img = torch.rand(1, 256, 512)
right_img = torch.rand(1, 256, 512)
# 随机颜色抖动

color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
left_img = color_jitter(left_img)
right_img = color_jitter(right_img)


def random_occlusion(image, occlusion_size=50):
    h, w = image.shape[1], image.shape[2]
    x1 = random.randint(0, w - occlusion_size)
    y1 = random.randint(0, h - occlusion_size)
    image[:, y1:y1+occlusion_size, x1:x1+occlusion_size] = 0
    return image

# 增加高斯噪声
def add_gaussian_noise(image, mean=0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)


left_img = add_gaussian_noise(left_img)
right_img = add_gaussian_noise(right_img)
