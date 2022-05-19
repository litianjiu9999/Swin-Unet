import glob
from operator import index
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import os  #通过os模块调用系统命令
import torch

def cut_save_image(image,ilm,savepath):#ilm代表输入的是image/label/mask #path 表示裁剪完成后要存储的路径
	width,height=image.size
	item_width=int(width/4)
	box_list=[]
	count=0
	nofimagepatchs = 1
    
	for j in range(0,4):
		for i in range(0,4):
			count+=1
			box=(i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width)
			box_list.append(box)
	print(count)
	image_list = [image.crop(box) for box in box_list]
	for image in image_list:
		image.save(savepath+str(ilm)+'_'+str(nofimagepatchs)+'.png')
		nofimagepatchs+=1 

def CHASEDB12npz():
    #转化数据集代码

    path = r'D:\\codeanddata\\datasets\\CHASEDB1\\images\\*.jpg'
    path2 = r'D:\\codeanddata\\code\\Swin-Unet\\data\\CHASEDB1\\'
    path3 = r'D:\\codeanddata\\datasets\\NPZ\\CHASEDB1\\image\\*.png'
    # for i,img_path in enumerate(glob.glob(path)):#将图片切成16份，每个224*224，然后存到path3

    #     image = Image.open(img_path)
    #     image=image.resize((896,896))
    #     ilm = "image"+str(i)
    #     cut_save_image(image,ilm,path3)
        
    #     label_path = img_path.replace('images','1st_label')
    #     label_path = label_path.replace('.jpg','_1stHO.png')
    #     temppath = path3.replace('image','label')
    #     label = Image.open(label_path)
    #     label = label.resize((896,896))
    #     ilm = "label"+str(i)
    #     cut_save_image(label,ilm,temppath)
        
    #     mask_path = img_path.replace('images','mask')
    #     mask_path = mask_path.replace('.jpg','.png')
    #     print(mask_path)
    #     temppath = path3.replace('image','mask')
    #     mask = Image.open(mask_path)
    #     mask = mask.resize((896,896))
    #     ilm = "mask"+str(i)
    #     cut_save_image(mask,ilm,temppath)
    #     print('------------',i)

    # for i,img_path in enumerate(glob.glob(path3)):#将切好的图片转换成npz格式
        
    #     # print(img_path)
    #     npzname = img_path[48:-4]
    #     # print(npzname)
    #     image = cv2.imread(img_path)
    #     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #     label_path = img_path.replace('image','label')
    #     # label_path = label_path.replace('.jpg','_1stHO.png')
    #     print(label_path)
    #     label = cv2.imread(label_path)
    #     label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    #     label[label>1] = 1
    #     print(label.shape)
    #     print(label)        
    #     np.savetxt("label.txt", label,fmt='%f',delimiter=',')
    #     np.savez(path2+str(npzname),image=image,label=label)
    #     print('------------',i)

    # 生成txt文件
    #----------------------------------------------------------------------------------------------------------#
    # file_path = "D:\\codeanddata\\code\\Swin-Unet\\data\\CHASEDB1\\train"  #文件路径
    # path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表
    # path_name = []#定义一个空列表
    # for i in path_list:
    #     print(i)
    #     path_name.append(i.split(".")[0]) #若带有后缀名，利用循环遍历path_list列表，split去掉后缀名
    # # path_name = path_name.sort() #排序
    # for file_name in path_name:
    #     # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
    #     with open("D:\\codeanddata\\code\\Swin-Unet\\lists\\lists_Synapse\\train.txt", "a") as file:
    #         file.write(file_name + "\n")
    #         print(file_name)
    #     file.close()

    file_path = "D:\\codeanddata\\code\\Swin-Unet\\data\\CHASEDB1\\test"  #文件路径
    path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表
    numofimg = len(path_list)
    cycletime = int(numofimg/16)
    ilm = "image"
    path_name = []#定义一个空列表
    for i in range(cycletime):
        j=1
        for j in range(1,17):
            savename = ilm + str(i) + "_" + str(j)
            # print(savename)
            path_name.append(savename)

    # path_name = path_name.sort() #排序
    print(path_name)
    # path_name = list(map(int, path_name))
    # path_name = sorted(path_name) #排序
    # path_name = list(map(str, path_name))
    for file_name in path_name:
        # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
        with open("D:\\codeanddata\\code\\Swin-Unet\\lists\\lists_Synapse\\test_vol.txt", "a") as file:
            file.write(file_name + "\n")
            print(file_name)
        file.close()

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
