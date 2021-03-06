# -*- coding: utf-8 -*
import os
import dlib
import numpy as np
import cv2


predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def get_landmarks(img):
    dets = detector(img, 1)
    landmarks = np.zeros((17,2))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmarks[0] = (shape.part(48).x, shape.part(48).y)
        for i in range(6):
            landmarks[1+i] = (shape.part(59-i).x, shape.part(59-i).y)    
        for i in range(10):
            landmarks[7+i] = (shape.part(26-i).x, shape.part(26-i).y)
    return landmarks


def inside(X,Y,Region): 
    j=len(Region)-1
    flag=False
    for i in range(len(Region)):
        if (Region[i][1]<Y and Region[j][1]>=Y or Region[j][1]<Y and Region[i][1]>=Y):  
            if (Region[i][0] + (Y - Region[i][1]) / (Region[j][1] - Region[i][1]) * (Region[j][0] - Region[i][0]) < X):
                flag =not flag
        j=i
    return flag


def gen_mask(picpath, savepath):
    paths = []
    print('图片的路径：', picpath)
    for root, dirs, files in os.walk(picpath):
        for f in files:
            paths.append(os.path.join(root, f))
    num = 1
    for path in paths:
        print("processing image  =========>")
        print(num)
        img = cv2.imread(path)
        region = get_landmarks(img)
        shape = list(img.shape)
        img1 = img.copy()
        for i in range(shape[0]):
            for j in range(shape[1]):
                if not inside(j, i, region):
                    img1[i, j] = (0, 0, 0)
                else:
                    img1[i, j] = (255, 255, 255)

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        cv2.imwrite(savepath + '\\' + path.split('\\')[-1], img1)
        num += 1

if __name__ == '__main__':
    gen_mask()
