#!/usr/bin/env python
# coding: utf-8


import random
import torch
import utils
from model_irse import IR_50, IR_101, IR_152
import os
from PIL import Image


def init_model(model, param, device):
    m = model([112, 112])
    m.eval()
    m.to(device)
    m.load_state_dict(torch.load(param, map_location=torch.device('cuda:1')))
    return m

def get_model_pool(device):
    model_pool = []
    # double
    model_pool.append(init_model(IR_50, 'models/backbone_ir50_ms1m_epoch120.pth', device))
    model_pool.append(init_model(IR_50, 'models/backbone_ir50_ms1m_epoch120.pth', device))
    
    model_pool.append(init_model(IR_50, 'models/Backbone_IR_50_LFW.pth', device))
    model_pool.append(init_model(IR_101, 'models/Backbone_IR_101_Batch_108320.pth', device))
    model_pool.append(init_model(IR_152, 'models/Backbone_IR_152_MS1M_Epoch_112.pth', device))
    return model_pool


# path为数据集的单极目录结构
def callike(path, name):
    device = torch.device('cuda:1')
    model_pool = get_model_pool(device)
    print('----models load over----')
    # faces = os.listdir('.\\data\\lfw_single')
    faces = os.listdir(path)

    vectors_list = []
    for model in model_pool:
        vectors = []
        for face in faces:
            try:
                face = utils.to_torch_tensor(Image.open(os.path.join(path, face)).convert('RGB'))
            except OSError:
                continue
            face = face.unsqueeze_(0).to(device)
            vectors.append(utils.l2_norm(model(face)).detach_())
        vectors_list.append(vectors)

    print('----vectors calculate over----')
    confusion_matrixes = []
    for vectors in vectors_list:
        print("len(vectors)", len(vectors))
        s = torch.FloatTensor(len(vectors), len(vectors))
        for i, vector1 in enumerate(vectors):
            for j, vector2 in enumerate(vectors[i + 1:]):
                tmp = (vector1 * vector2).sum().item()
                s[i, j + i + 1] = tmp
                s[j + i + 1, i] = tmp
        # for i in range(712)
        print("len(vectors_list)", len(vectors_list))
        for i in range(len(vectors_list)):
            s[i, i] = 0
        confusion_matrixes.append(s)

    # In[5]:
    confusion_matrix = confusion_matrixes[0].clone()
    for tmp in confusion_matrixes[1:]:
        confusion_matrix += tmp

    # In[6]:
    import json
    value1, index_like1 = torch.max(confusion_matrix, 1)
    for i, j in enumerate(index_like1):
        confusion_matrix[i, j] = 0
    value2, index_like2 = torch.max(confusion_matrix, 1)
    for i, j in enumerate(index_like2):
        confusion_matrix[i, j] = 0
    value3, index_like3 = torch.max(confusion_matrix, 1)
    a = {}
    for i, face in enumerate(faces):
        a[face] = [faces[index_like1[i]], value1[i].item()/5, faces[index_like2[i]], value2[i].item()/5, faces[index_like3[i]], value3[i].item()/5]
    f = open("likelihood_" + name + ".json", "w")
    f.write(json.dumps(a))
    f.close()

if __name__=='__main__':
    callike()


