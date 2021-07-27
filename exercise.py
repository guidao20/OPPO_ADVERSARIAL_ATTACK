import imageio
import numpy as np
import torch
import pytorch_msssim


# 图像质量评分score_2，范围[0, 1]，越高越好
def getscore2(ori_img, adv_img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = pytorch_msssim.MSSSIM()
    img1 = ori_img / 255
    img2 = adv_img / 255
    score2 = m(img1, img2).item()
    return score2


# 扰动大小评分规则score_1，范围[0, 1]，越高越好
def getscore1(ori_img, adv_img):
    ori_img = ori_img.cpu().detach().numpy().astype(int)  # 图像数组，（height, weight, channels）
    adv_img = adv_img.cpu().detach().numpy().astype(int)
    # dif = np.clip((adv_img - ori_img), -20, 20)  # 扰动限制在[-20, 20]的区间范围内
    dif = adv_img - ori_img
    # score1 = 1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    score1 = (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    return score1

if __name__ == '__main__':
    ori_path = './data/demo/images/0/0_2.jpg'
    adv_path = './advSamples_images/0/0_2.jpg'
    ori = imageio.imread(ori_path)
    adv = imageio.imread(adv_path)
    # ori_label = '1'
    # adv_label = '3'
    score1 = getscore1(ori, adv)
    # print(score)
    score2 = getscore2(ori, adv)
    print(score1, score2)