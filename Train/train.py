import torch
from torch import nn
from Models import EfficientStereo
import time
import torch.nn.functional as F
from tool import RenewFilename
import numpy as np
from Dataloader import FileLoader, DataloaderPNG, DataloaderPFM
from torch.utils import data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

finetune = False
if finetune:
    epochs = 500
else:
    epochs = 1000

padding = 32
logs = '../logs/log'
seed = 1
Train_path = '../dataset/kitti2015/'
Test_path = '../dataset/kitti2015/'
maxdisp = 192
savemodel_path = '../trained/train1/'
pre_dir = '../trained/finetune_000.tar'

torch.cuda.manual_seed(seed)

# 加载模型
model = EfficientStereo.EfficientStereo(maxdisp)
model = nn.DataParallel(model)
model.cuda()

if finetune:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=0.0001)

else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=0.0001)
    

def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    output = model(imgL, imgR)
    output = torch.squeeze(output, 1)

    loss = 1.0 * F.smooth_l1_loss(output[mask], disp_true[mask], reduction='mean')

    loss.backward()
    optimizer.step()

    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))

    imgL, imgR = imgL.cuda(), imgR.cuda()

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

    with torch.no_grad():
        output3 = model(imgL, imgR)

    if top_pad != 0 and right_pad != 0:
        img = output3[:, :, top_pad:, :-right_pad]
    elif top_pad == 0 and right_pad != 0:
        img = output3[:, :, :, :-right_pad]
    elif top_pad != 0 and right_pad == 0:
        img = output3[:, :, top_pad:, :]
    else:
        img = output3

        # numpy
    img = torch.squeeze(img, dim=1).data.cpu()
    img_np = img.numpy()
    disp_np = disp_true.numpy()
    mask = disp_np > 0  # 所有大于0的像素为有效区域

    # 初始化差异图，将所有像素初始化为0
    difference = np.zeros_like(img_np)

    # 计算有效区域内的差异
    difference[mask] = img_np[mask] - disp_np[mask]

    # 计算绝对误差
    error = np.abs(difference)

    bad_pixel_mask = (error > 3) & (error > disp_np * 0.05)

    if np.sum(mask) > 0:
        bad_pixel_rate = np.sum(bad_pixel_mask) / np.sum(mask)
    else:
        bad_pixel_rate = 0

    # 计算平均绝对误差，仅在有效区域内计算
    avg_error = float(np.mean(error[mask]))

    torch.cuda.empty_cache()

    # 返回错误率
    return bad_pixel_rate, avg_error


def main():
    # 加载数据
    all_left_img, all_right_img, all_left_d = FileLoader.TrainLoader(Train_path)
    test_left_img, test_right_img, test_left_d = FileLoader.TestLoader(Test_path)
    TrainImgLoader = data.DataLoader(
        DataloaderPNG.MyImageLoader(all_left_img, all_right_img, all_left_d, True),
        batch_size=16, shuffle=True, num_workers=16, drop_last=False)
    TestImgLoader = data.DataLoader(
        DataloaderPNG.MyImageLoader(test_left_img, test_right_img, test_left_d, False),
        batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    updated_params_count = 0
    updated_count = 0
    # 判断预训练模型是否存在
    if os.path.exists(pre_dir):
        print("加载预训练权重文件:", pre_dir)
        checkpoint = torch.load(pre_dir, map_location='cpu')  # 根据需要调整 map_location

        # 假设 checkpoint 中保存的权重在 'state_dict' 键下
        old_state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        updated_params_count = 0
        updated_count = 0
        skipped_params = []

        for k, v in old_state_dict.items():
            # 如果键名包含 'model.' 前缀，则移除它
            if k.startswith('model.'):
                new_k = k.replace('model.', '', 1)
            else:
                new_k = k

            if new_k in model.state_dict():
                new_state_dict[new_k] = v
                updated_params_count += 1
            else:
                skipped_params.append(new_k)
            updated_count += 1

        # 加载新 state dict，使用 strict=False 以允许部分加载
        model.load_state_dict(new_state_dict, strict=False)
        print("已更新 %d / %d 个参数。" % (updated_params_count, updated_count))

        if skipped_params:
            print("未能匹配的参数数量: %d" % len(skipped_params))

    else:
        print("权重文件不存在:", pre_dir)

    print("更新的参数数量:", updated_params_count)
    print("总参数数量:", updated_count)

    # 检查Logs路径是否存在
    new_logs = RenewFilename.get_unique_path(logs)
    print("logs 保存路径: ", new_logs)
    writer = SummaryWriter(new_logs)

    # 检查保存路径是否存在
    savemodel = RenewFilename.get_unique_path(savemodel_path)
    if not os.path.exists(savemodel):
        try:
            # 如果路径不存在，则创建该路径
            os.makedirs(savemodel)
            print("保存路径 ", savemodel, "已成功创建。")
        except OSError as e:
            print("创建保存路径时出错:", e)
    else:
        print("保存路径 ", savemodel, "已经存在。")

    min_error = 100
    min_avg = 100
    min_epo = 0
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    start_full_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f'Epoch [{epoch}/1000], LR: {scheduler.get_last_lr()[0]:.6f}')
        total_train_loss = 0
        total_test_loss = 0
        total_test_avg = 0

        # training
        train_time = time.time()
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):

            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        writer.add_scalar("train loss", total_train_loss / len(TrainImgLoader), epoch)
        print("本轮训练花费：", time.time()-train_time)
        scheduler.step()
        # Test

        test_time = time.time()
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            bad3error, avg_error = test(imgL, imgR, disp_L)
            print('Iter %d 3-px error = %.3f,avg error = %.3f' % (batch_idx, bad3error * 100, avg_error))
            total_test_avg += avg_error
            total_test_loss += bad3error
        print("本轮测试花费：", time.time() - test_time)
        print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_loss / len(TestImgLoader) * 100))
        print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_avg / len(TestImgLoader)))
        writer.add_scalar("bad3 loss", (total_test_loss / len(TestImgLoader)) * 100, epoch)
        writer.add_scalar("avg error", total_test_avg / len(TestImgLoader), epoch)
        if total_test_loss / len(TestImgLoader) * 100 < min_error:
            min_error = total_test_loss / len(TestImgLoader) * 100
            min_avg = total_test_avg / len(TestImgLoader)
            min_epo = epoch
            save_time = time.time()
            # SAVE
            if epoch > 500 or (finetune and epoch > 200):
                savefilename = savemodel + 'finetune_' + str(epoch) + '.tar'
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss / len(TrainImgLoader),
                    'test_loss': total_test_loss / len(TestImgLoader) * 100,
                }, savefilename)
                print("保存花费：", time.time() - save_time)
        print('Min epoch %d total test error = %.3f' % (min_epo, min_error))
        print('Min epoch %d total average error = %.3f' % (min_epo, min_avg))
    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    print("##### Train over #####")
    print("Min Bad3 error = ", min_error)
    print("Min average error", min_avg)
    print("min error epoch = ", min_epo)
    writer.close()


if __name__ == '__main__':
    main()
