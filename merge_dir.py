# coding=utf-8
import os
import shutil


# 每人抽取1张图片，用于生成likehood.json
def merge_1(src1, des1):
    folders = os.listdir(src1)
    determination = des1
    if not os.path.exists(determination):
        os.makedirs(determination)

    for folder in folders:
        dir = src1 + '\\' + str(folder)
        files = os.listdir(dir)
        source = dir + '\\' + str(files[0])
        name = files[0].split('__')[0]
        deter = determination + '\\' + name + '.png'
        shutil.copyfile(source, deter)


# 将lfw合并成单级目录结构，用于target_iteration生成对抗样本
def merge_2(src2, des2):
    folders = os.listdir(src2)
    determination = des2
    if not os.path.exists(determination):
        os.makedirs(determination)

    for folder in folders:
        dir = src2 + '\\' + str(folder)
        files = os.listdir(dir)
        for file in files:
            source = dir + '\\' + str(file)
            deter = determination + '\\' + str(file)
            shutil.copyfile(source, deter)
