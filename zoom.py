"""
OPPO  Competion
This file is used to recover the size of dataset image.

"""
import os
import cv2
import shutil

adv_dir='H:\\FaceRecognition\\Attack\\TIANCHI_BlackboxAdversial-master\\advSamples_images'
data_dir='H:\\FaceRecognition\\Attack\\TIANCHI_BlackboxAdversial-master\\data\\demo\\images'

labels=os.listdir(data_dir)
for label in labels:
    print('Processing: '+label)
    imgs=os.listdir(os.path.join(data_dir,label))
    adv_imgs=os.listdir(os.path.join(adv_dir,label))

    for img in imgs:
        if img in adv_imgs:
            # resize
            adv_img_data=cv2.imread(os.path.join(data_dir,label,img))
            h,w,_=adv_img_data.shape

            img_data=cv2.imread(os.path.join(adv_dir,label,img))
            img_data=cv2.resize(img_data,(w,h))

            cv2.imwrite(os.path.join(adv_dir,label,img),img_data)
        else:
            # copy the img from data_dir to adv_dir
            shutil.copyfile(os.path.join(data_dir,label,img),os.path.join(adv_dir,label,img))
