import os
import shutil


def softdir(path):
    current_path = path
    filename_list = os.listdir(current_path)

    print('正在分类整理进文件夹ing...')
    for filename in filename_list:
        try:
            name1, name2 = filename.split('.')
            if name2 == 'jpg' or name2 == 'png':
                try:
                    dir_name = filename.split('__')[0]
                    face_name = filename.split('__')[1]
                    os.mkdir(current_path + '\\' + dir_name)
                    print('创建文件夹'+dir_name)
                except:
                    pass
                try:
                    shutil.move(current_path+'\\'+filename, current_path+'\\'+dir_name)
                    print(filename+'转移成功！')
                except Exception as e:
                    print('移动失败:' + e)
        except:
            pass

    print('整理完毕！')


if __name__ == '__main__':
    p = '.\\advSamples_images'
    faces = os.listdir(p)
    for face in faces:
        houzhui = face.split(".")
        NewName = os.path.join(p, houzhui[0]+'.jpg')
        OldName = os.path.join(p, face)
        os.rename(OldName, NewName)

    softdir('.\\advSamples_images')

    output = '.\\advSamples_images'
    people = os.listdir(output)
    for i in people:
        path_p = os.path.join(output, i)
        if os.path.isfile(path_p):
            os.remove(path_p)
        else:
            faces = os.listdir(path_p)
            for face in faces:
                name = face.split("__")
                NewName = os.path.join(path_p, name[1])
                OldName = os.path.join(path_p, face)
                os.rename(OldName, NewName)
