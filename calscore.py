import imageio
import numpy as np
import torch
import pytorch_msssim
import csv
import os

# 图像质量评分score_2，范围[0, 1]，越高越好
def getscore2(ori_img, adv_img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = pytorch_msssim.MSSSIM()
    img1 = torch.from_numpy(ori_img.transpose((2, 0, 1)) / 255).float().unsqueeze(0)
    img2 = torch.from_numpy(adv_img.transpose((2, 0, 1)) / 255).float().unsqueeze(0)
    score2 = m(img1, img2).item()
    return score2


# 扰动大小评分规则score_1，范围[0, 1]，越高越好
def getscore1(ori_img, adv_img):
    ori_img = ori_img.astype(int)  # 图像数组，（height, weight, channels）
    adv_img = adv_img.astype(int)
    dif = np.clip((adv_img - ori_img), -20, 20)  # 扰动限制在[-20, 20]的区间范围内
    # dif = adv_img - ori_img
    # score1 = 1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    score1 = 1 - (dif[:, :, 0].max() + dif[:, :, 1].max() + dif[:, :, 2].max()) / 60
    return score1


def cal_imgs(ori_dir,adv_dir,output_path):
    with open(output_path,'w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['image','score1','score2'])
        labels=os.listdir(ori_dir)
        for label in labels:
            imgs=os.listdir(os.path.join(ori_dir,label))
            for img in imgs:
                ori_img=imageio.imread(os.path.join(ori_dir,label,img))
                adv_img=imageio.imread(os.path.join(adv_dir,label,img))
                score1=getscore1(ori_img,adv_img)
                score2=getscore2(ori_img,adv_img)
                writer.writerow([label+'/'+img,score1,score2])
                print(label+'/'+img+' score1: '+str(score1)+' score2: '+str(score2))


if __name__ == '__main__':
    # ori_path = './data/demo/images/0/0_2.jpg'
    # adv_path = './advSamples_images/0/0_2.jpg'
    # ori = imageio.imread(ori_path)
    # adv = imageio.imread(adv_path)
    # # ori_label = '1'
    # # adv_label = '3'
    # score1 = getscore1(ori, adv)
    # # print(score)
    # score2 = getscore2(ori, adv)
    # print(score1, score2)
    cal_imgs('./data/demo/images/',
             './advSamples_images/',
             './score721.csv')