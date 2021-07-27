'''
    attribute:
    1 Multiprocessing operation
    2 Remove small interference noise
    3 Leverage the four ensemble models to calculate the top three images most similar to the original image and save to likelihood.json
    4 The stop condition is scores < break_threshold(-0.25) or iteration steps >300
    5 momentum = 0.5
    6 guass kernel size = 3
    7 learning rate = 8
'''

import json
import random
from guass import *
import os
from PIL import Image
import torchvision
import warnings
import torch.multiprocessing
from model_irse import IR_50, IR_101, IR_152
from exercise import getscore1, getscore2

warnings.filterwarnings("ignore")
to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


# TV loss
def tv_loss(input_t):
    temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
    temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
    temp = (input_t - temp1)**2 + (input_t - temp2)**2
    return temp.sum()


# 初始化噪声
def get_init_noise(device):
    noise = torch.Tensor(1, 3, 112, 112)
    noise = torch.nn.init.xavier_normal(noise, gain=1)
    return noise.to(device)


# 返回人脸识别模型（主干网络）（输出512维的向量）
def get_model(model, param, device, proportion, kind):
    if kind != '-1,1' and kind != '0,1':
        raise "no this kind model!"
    m = model([112, 112])
    m.eval()
    m.to(device)
    m.load_state_dict(torch.load(param, map_location=device))
    model_dict = {'model': m, 'proportion': proportion, 'kind': kind}
    return model_dict


# 返回模型池
def get_model_pool(device):
    model_pool = []
    model_pool.append(get_model(IR_50, 'models/backbone_ir50_ms1m_epoch120.pth', device, 2, '-1,1'))
    model_pool.append(get_model(IR_50, 'models/Backbone_IR_50_LFW.pth', device, 1, '-1,1'))
    model_pool.append(get_model(IR_101, 'models/Backbone_IR_101_Batch_108320.pth', device, 1, '-1,1'))
    model_pool.append(get_model(IR_152, 'models/Backbone_IR_152_MS1M_Epoch_112.pth', device, 1, '-1,1'))
    return model_pool

# Normalization
def normal_model_proportion(model_pool):
    sum1 = 0
    for model_dict in model_pool:
        sum1 += model_dict['proportion']
    for model_dict in model_pool:
        model_dict['proportion'] /= sum1
    return model_pool


# 随机选择一个模型
def random_choose_model(model_pool):
    s = len(model_pool)
    index = random.randint(0, s - 1)
    return model_pool[index]


# Return the torch.tensor image pool
def get_img_pool(person_list, device):
    person_pool = []
    for el in person_list:
        person_pool.append(to_torch_tensor(Image.open(el)).unsqueeze_(0).to(device))
    return person_pool


# 单步迭代
def iter_step(tmp_noise, origin_img, target_img, mask, gaussian_blur, model_pool, index, loss1_v, momentum=0, lr=1):
    tmp_noise.requires_grad = True
    noise = gaussian_blur(tmp_noise)
    noise *= mask
    loss1 = 0
    score1 = 0
    score2 = 0
    for model_dict in model_pool:
        model = model_dict['model']
        proportion = model_dict['proportion']
        v1 = l2_norm(model(origin_img + noise))  # 对抗样本向量
        v2_1 = l2_norm(model(origin_img)).detach_()  # 原图向量
        v2_2 = l2_norm(model(target_img)).detach_()  # 目标图片向量
        tmp1 = (v1*v2_1).sum()  # 对抗样本 与 原图 的向量的内积
        tmp2 = (v1*v2_2).sum()  # 对抗样本 与 目标图片 的向量的内积
        r1 = 1
        r2 = 1
        if tmp1 < 0.2:  # 如果与原图的相似度足够小了，loss不再考虑原图的因素 原: if tmp1 < 0.3
            r1 = 0
        if tmp2 > 0.8:  # 如果与目标图片的相似度足够大了，loss不再考虑目标图片的因素 原: if tem2 > 0.8
            r2 = 0
        loss1 += (r1 * tmp1 - r2 * tmp2) * proportion  # Cos Loss
        score1 += tmp1.item() * proportion
        score2 += tmp2.item() * proportion

    loss1.backward(retain_graph=True)  # 反向传播更新噪声
    loss1_v = tmp_noise.grad.detach() * (1 - momentum) + loss1_v * momentum  # 加入动量项，利用loss1更新扰动
    tmp_noise.grad.data.zero_()

    r3 = 1
    if index > 100:
        r3 *= 0.1
    if index > 200:
        r3 *= 0.1

    loss2 = (noise**2).sum().sqrt()  # L2 loss
    loss3 = tv_loss(noise)  # TV loss
    loss = r3 * 0.025 * loss2 + r3 * 0.004 * loss3
    loss.backward()
    tmp_noise = tmp_noise.detach() - lr * (tmp_noise.grad.detach() + loss1_v)
    tmp_noise = (tmp_noise + origin_img).clamp_(-1, 1) - origin_img
    tmp_noise = tmp_noise.clamp_(-0.2, 0.2)
    return tmp_noise, score1, score2, loss1_v


# 多步迭代, 调用单步迭代函数
def noise_iter(model_pool, origin_img, target_img, mask, gaussian_blur, device):
    learning_rate = 8
    momentum = 0.5
    tmp_noise = get_init_noise(device)
    index = 0
    loss1_v = 0
    while True:
        index += 1
        tmp_noise, socre1, socre2, loss1_v = iter_step(tmp_noise, origin_img, target_img, mask, gaussian_blur, model_pool, index, loss1_v, momentum, lr=learning_rate)
        yield tmp_noise, socre1, socre2


# 计算单个对抗样本
def cal_adv(origin_name, target_name, mask_name, model_pool, gaussian_blur, device):
    break_threshold = -0.25
    origin_img = to_torch_tensor(Image.open(origin_name).convert('RGB'))
    origin_img = origin_img.unsqueeze_(0).to(device)
    target_img = to_torch_tensor(Image.open(target_name))
    target_img = target_img.unsqueeze_(0).to(device)
    mask = torchvision.transforms.ToTensor()(Image.open(mask_name))
    mask = mask.unsqueeze_(0).to(device)
    generator = noise_iter(model_pool, origin_img, target_img, mask, gaussian_blur, device)
    scores = 0
    i = 0
    while True:
        tmp_noise, socre1, socre2 = next(generator)
        socre = socre1 - socre2
        if socre < -0.4:
            socre = -0.4
        scores = 0.5 * socre + 0.5 * scores
        i += 1
        if i > 300:
            f = open('hard.txt', 'a')
            f.write(origin_name + ';' + target_name + ';' + str(scores) + '\n')
            f.close()
            print('origin img is %s, target img is %s, iter %d, socre is %0.3f'
                % (origin_name.split('\\')[1], target_name.split('\\')[1], i, scores))
            break
        if scores < break_threshold:
            print('origin img is %s, target img is %s, iter %d, socre is %0.3f'
                % (origin_name.split('\\')[1], target_name.split('\\')[1], i, scores))
            break;

    return gaussian_blur(tmp_noise) * mask, origin_img, i


# 单进程运行
def one_process_run(face_path, mask_path, name):
    if os.path.exists('hard.txt'):
        os.remove('hard.txt')
    if os.path.exists('iter_num.txt'):
        os.remove('iter_num.txt')

    device = torch.device('cuda:1')
    model_pool = normal_model_proportion(get_model_pool(device))

    people_list = os.listdir(face_path)
    likelihood = json.load(open("likelihood_" + name + ".json"))
    gaussian_blur = get_gaussian_blur(kernel_size=3, device=device)  # 高斯滤波：消除高斯噪声，在图像处理的降噪、平滑中应用较多
    l2 = []
    for origin_name in people_list:
        name_index = origin_name.split('__')[0] + '.png'  # target图像的索引没有‘_0001’的编号
        tar_path = os.path.join('./data/112/' + name, likelihood[name_index][2][:-4])
        tar_name = os.listdir(tar_path)[0]
        tar = os.path.join(tar_path, tar_name)
        # 生成当前图像的对抗样本
        noise, _, iter_num = cal_adv(os.path.join(face_path, origin_name),
                                     tar,
                                     os.path.join(mask_path, origin_name),
                                     model_pool,
                                     gaussian_blur,
                                     device)
        THRESHOLD = 2 + 0.00001  # eps
        noise = torch.round(noise * 127.5)[0].cpu().numpy()
        noise = noise.swapaxes(0, 1).swapaxes(1, 2)
        noise = noise.clip(-25.5, 25.5)
        noise = np.where((noise > -THRESHOLD) & (noise < THRESHOLD), 0, noise)
        origin_img = np.array(Image.open(os.path.join(face_path, origin_name)), dtype=float)
        numpy_adv_sample = (origin_img + noise).clip(0, 255)
        adv_sample = Image.fromarray(np.uint8(numpy_adv_sample))

        # Enumerate L2 norm
        noise = (numpy_adv_sample - origin_img)
        noise_l2_norm = np.sqrt(np.sum(noise ** 2, axis=2)).mean()
        l2.append(noise_l2_norm)
        print('%s noise l2_norm is %.4f' % (origin_name, noise_l2_norm))
        f = open('iter_num.txt', 'a')
        f.write('%s %d %.4f\n' % (origin_name, iter_num, noise_l2_norm))
        f.close()

        # 保存图像
        if os.path.exists('advSamples_' + name + '/') is False:
            os.mkdir('advSamples_' + name + '/')
        jpg_img = 'advSamples_' + name + '/' + origin_name
        png_img = jpg_img.replace('jpg', 'png')
        adv_sample.save(png_img)
        # os.rename(png_img, jpg_img)
    return l2


# Utilize one GPU to generate an adversarial sample with two processing
def one_device_run(p_pool, people_list, device):
    double_process = True
    model_pool = get_model_pool(device)
    model_pool = normal_model_proportion(model_pool)
    print('----model load over----')
    res = []
    if double_process:
        for model_dict in model_pool:
            model_dict['model'].share_memory()
        res.append(p_pool.apply_async(one_process_run, args=(people_list[:len(people_list) // 2], model_pool, device)))
        res.append(p_pool.apply_async(one_process_run, args=(people_list[len(people_list) // 2:], model_pool, device)))
    else:
        res.append(p_pool.apply_async(one_process_run, args=(people_list, model_pool, device)))
    return res


if __name__ == '__main__':
    # one_process_run(path, model_pool, device)
    exit()
