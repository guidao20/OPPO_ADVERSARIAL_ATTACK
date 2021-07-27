import os
import shutil

current_path = '.\\data\\multi'
filename_list = os.listdir(current_path)

print('正在分类整理进文件夹ing...')
for filename in filename_list:
    try:
        name1, name2 = filename.split('.')
        if name2 == 'jpg' or name2 == 'png':
            try:
                os.mkdir(current_path + '\\' + name1[:-4])
                print('创建文件夹'+name1[:-4])
            except:
                pass
            try:
                shutil.move(current_path+'\\'+filename, current_path+'\\'+name1[:-4])
                print(filename+'转移成功！')
            except Exception as e:
                print('移动失败:' + e)
    except:
        pass

print('整理完毕！')



