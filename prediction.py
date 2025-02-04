import torch
import torch.nn as nn
import time
from Models import EfficientStereo
import torch.nn.functional as F
import numpy as np
from Dataloader import FileLoader
from Dataloader import DataloaderPNG, DataloaderPFM
from torch.utils import data


load_model = '../trained/your_weights.tar'
fold_path = '../Dataset/Test/'

seed = 1
padding = 32
torch.cuda.manual_seed(seed)

model = EfficientStereo.EfficientStereo(192)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if load_model is not None:
    print('load Model')
    state_dict = torch.load(load_model)
    model.load_state_dict(state_dict['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


def compute_error(pred_disp, gt_disp):
    """计算预测视差图与ground truth之间的误差，并返回差异图"""
    gt_disp = gt_disp.numpy()
    gt_disp = np.squeeze(gt_disp)
    # 创建有效区域的掩码（gt_disp > 0）
    mask = gt_disp > 0  # 所有大于0的像素为有效区域

    # 初始化差异图，将所有像素初始化为0
    difference = np.zeros_like(pred_disp)

    # 计算有效区域内的差异
    difference[mask] = pred_disp[mask] - gt_disp[mask]

    # 计算绝对误差
    error = np.abs(difference)
    avg_error = np.mean(error[mask])

    mask = mask.astype(np.uint8)
    valid_mask = mask > 0
    img_np = pred_disp[valid_mask]
    disp_np = gt_disp[valid_mask]
    np_error = np.abs(img_np - disp_np)
    bad_pixel_mask = (np_error > 3) & (np_error > disp_np * 0.05)
    bad_pixel_rate = np.mean(bad_pixel_mask)

    # 打印误差统计
    print("Average Error: %.3f" % avg_error)
    print("Bad 3.0 Error Rate: %.3f%%" % (bad_pixel_rate * 100))

    # 首先，初始化为黑色
    error_image = np.zeros((pred_disp.shape[0], pred_disp.shape[1], 3), dtype=np.uint8)

    # 定义颜色
    # white = [255, 255, 255]  # RGB
    # blue = [0, 0, 255]
    # red = [255, 100, 100]
    # error_image[np.where(((error > 3) | (error > gt_disp * 0.05)) & mask)] = blue
    # error_image[np.where((error > 3) & (error > gt_disp * 0.05) & mask)] = red
    # error_image[np.where(((error <= 3) | (error <= gt_disp * 0.05)) & mask)] = white

    # 返回平均误差、Bad 3.0 错误率以及差异图
    return avg_error, bad_pixel_rate, error_image


def test(imgL, imgR):
    model.eval()

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    start_time = time.time()
    with torch.no_grad():
        disp = model(imgL, imgR)
    elapsed_time = time.time() - start_time
    disp = torch.squeeze(disp)
    disp = disp.data.cpu().numpy()

    return disp, elapsed_time


def main():

    # loader
    left_val, right_val, disp_val = FileLoader.pathloader(fold_path)
    TestImgLoader = data.DataLoader(DataloaderPNG.MyImageLoader(left_val, right_val, disp_val, False),
                                    batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    # count
    total_avg_error = 0
    total_bad_pixel_rate = 0
    processed_count = 0
    for idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TestImgLoader):

        disp_L = disp_crop_L
        imgL = imgL_crop
        imgR = imgR_crop

        if imgL.shape[2] % padding != 0:
            times = imgL.shape[2] // padding
            top_pad = (times + 1) * padding - imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % padding != 0:
            times = imgL.shape[3] // padding
            right_pad = (times + 1) * padding - imgL.shape[3]
        else:
            right_pad = 0

        imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
        imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

        # 调用测试函数进行推理

        pred_disp, elapsed_time = test(imgL, imgR)  # 确保`test`函数已定义

        # remove pad
        if top_pad != 0 and right_pad != 0:
            img = pred_disp[top_pad:, :-right_pad]
        elif top_pad == 0 and right_pad != 0:
            img = pred_disp[:, :-right_pad]
        elif top_pad != 0 and right_pad == 0:
            img = pred_disp[top_pad:, :]
        else:
            img = pred_disp

        print("time for index %d = %.3f seconds" % (idx, elapsed_time))

        avg_error, bad_pixel_rate, difference = compute_error(img, disp_L)

        total_avg_error += avg_error
        total_bad_pixel_rate += bad_pixel_rate
        processed_count += 1

    print("All Average Error = %.3f " % (total_avg_error/processed_count))
    print("All Bad3 Error = %.3f %%" % (total_bad_pixel_rate/processed_count * 100))


if __name__ == '__main__':
    main()
