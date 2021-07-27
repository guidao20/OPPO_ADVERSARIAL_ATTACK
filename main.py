'''
1. align_dataset_mtcnn.py :进行人脸检测、人脸对齐、图像裁剪 : input_dir output_dir  文件 112
   cal_likehood.py: 计算相似度
2. merge_dir.py: merge_1, 生成计算相似度文件需要的人脸数据集 merge_2: 将人脸数据集合并成单级文件夹结构 输入：步骤1的output_dir 输出:  likelihood_imagess.json
3. crop_image: 生成mask 输出    mask
4. target_iteration.py: 生成对抗样本 输入: 单级目录结构人脸数据集
'''
import os
from align.align_dataset_mtcnn import main
from merge_dir import merge_1, merge_2
from cal_likehood import callike
from crop_image import gen_mask
from target_iteration import one_process_run
import cv2
import os
import shutil


dataset = ['CASIAdemo', 'VGGdemo', 'LFWdemo', 'images']
num = 3

input = './data/demo/' + dataset[num]
output = './data/112/' + dataset[num]

src = output
des1 = './data/single_pic/' + dataset[num]

des2 = './data/single_dir/' + dataset[num]

mask_path = './data/mask/' + dataset[num]


# mtcnn
# main(input, output)

# 对人脸数据集进行规范化
# people = os.listdir(output)
# for i in people:
#     path_p = os.path.join(output, i)
#     if os.path.isfile(path_p):
#         os.remove(path_p)
#     else:
#         faces = os.listdir(path_p)
#         for face in faces:
#             NewName = os.path.join(path_p, i+'__'+face)
#             OldName = os.path.join(path_p, face)
#             os.rename(OldName, NewName)
#
# merge_1(src, des1)  # 计算相似度用
# merge_2(src, des2)  # 生成对抗样本用
#
# # 计算相似度
# callike(des1, dataset[num])
#
# pic_path = des2
#
# gen_mask(pic_path, mask_path)

one_process_run(des2, mask_path, dataset[num])
exit()
