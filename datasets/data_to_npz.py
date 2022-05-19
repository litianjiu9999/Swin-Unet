import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import os  #通过os模块调用系统命令
import torch
def CHASEDB12npz():
    #转化数据集代码

    path = r'D:\\codeanddata\\datasets\\CHASEDB1\\images\\*.jpg'
    path2 = r'D:\\codeanddata\\code\\TransUNet\\TransUNet_origin\\data\\CHASEDB1\\'
    for i,img_path in enumerate(glob.glob(path)):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        label_path = img_path.replace('images','1st_label')
        label_path = label_path.replace('.jpg','_1stHO.png')
        print(label_path)
        # label_path = path1
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        label[label>1] = 1
        # label = torch.from_numpy(label)
        # label = label.unsqueeze(2)
        # label = label.numpy()
        print(label.shape)
        print(label)        
        np.savetxt("label.txt", label,fmt='%f',delimiter=',')
        np.savez(path2+str(i),image=image,label=label)
        print('------------',i)



    # 生成txt文件
    #----------------------------------------------------------------------------------------------------------#
    # file_path = "D:\\codeanddata\\code\\TransUNet\\TransUNet_origin\\data\\CHASEDB1\\train\\"  #文件路径
    # path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表
    # path_name = []#定义一个空列表
    # for i in path_list:
    #     path_name.append(i.split(".")[0]) #若带有后缀名，利用循环遍历path_list列表，split去掉后缀名
    # # path_name = path_name.sort() #排序
    # for file_name in path_name:
    #     # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    #     with open("D:\\codeanddata\\code\\TransUNet\\TransUNet_origin\\data\\CHASEDB1\\train.txt", "a") as file:
    #         file.write(file_name + "\n")
    #         print(file_name)
    #     file.close()

    # file_path = "D:\\codeanddata\\code\\TransUNet\\TransUNet_origin\\data\\CHASEDB1\\test\\"  #文件路径
    # path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表
    # path_name = []#定义一个空列表
    # for i in path_list:
    #     path_name.append(i.split(".")[0]) #若带有后缀名，利用循环遍历path_list列表，split去掉后缀名
    # # path_name = path_name.sort() #排序
    # for file_name in path_name:
    #     # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    #     with open("D:\\codeanddata\\code\\TransUNet\\TransUNet_origin\\data\\CHASEDB1\\test.txt", "a") as file:
    #         file.write(file_name + "\n")
    #         print(file_name)
    #     file.close()

    #----------------------------------------------------------------------------------------------------------#


    print('dataset <CHASEDB1> is ready')

def DRIVE2npz():
    #转化数据集代码 训练集
    path = r'D:\\codeanddata\\datasets\\DRIVE\\training\\images\\*.tif'
    path2 = r'D:\\codeanddata\\code\\TransUNet\\TransUNet_origin\\data\\DRIVE\\train\\'
    for i,img_path in enumerate(glob.glob(path)):
        
        image = imageio.imread(img_path)
        image = image.astype('uint8')
        label_path = img_path.replace('images','1st_manual')
        label_path = label_path.replace('training.tif','manual1.gif')
        label = imageio.imread(label_path)
        label = label.astype('uint8')
        print(label.shape)
        np.savez(path2+str(i),image=image,label=label)
        print('------------',i)

    ###测试集   
    path = r'D:\\codeanddata\\datasets\\DRIVE\\test\\images\\*.tif'
    path2 = r'D:\\codeanddata\\code\\TransUNet\\TransUNet_origin\\data\\DRIVE\\test\\'
    for i,img_path in enumerate(glob.glob(path)):
        image = imageio.imread(img_path)
        image = image.astype('uint8')
        label_path = img_path.replace('images','1st_manual')
        label_path = label_path.replace('test.tif','manual1.gif')
        label = imageio.imread(label_path)
        label = label.astype('uint8')
        print(label.shape)
        np.savez(path2+str(i),image=image,label=label)
        print('------------',i)

    print('dataset <DRIVE> is ready')

# DRIVE2npz()
CHASEDB12npz()
