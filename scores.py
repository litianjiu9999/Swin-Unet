import numpy as py
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import os  #通过os模块调用系统命令
import torch
# import PIL.Image as Image
# import os
# ###############################将16个图像拼接回一个大图##########################################
# IMAGES_PATH = r'D:\\codeanddata\\code\\outputs\\swinunet\\predictions\\'  # 图片集地址
# IMAGES_FORMAT = ['.jpg', '.JPG','.png']  # 图片格式
# IMAGE_SIZE = 224  # 每张小图片的大小
# IMAGE_ROW = 4  # 图片间隔，也就是合并成一张图后，一共有几行
# IMAGE_COLUMN = 4  # 图片间隔，也就是合并成一张图后，一共有几列
# IMAGE_SAVE_PATH = '.\\predictions'  # 图片转换后的地址

# # 获取图片集地址下的所有图片名称
# # image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
# #                os.path.splitext(name)[1] == item]
# file_path = "D:\\codeanddata\\code\\outputs\\swinunet\\predictions"  #文件路径
# path_list = os.listdir(file_path) #遍历整个文件夹下的文件name并返回一个列表
# numofimg = len(path_list)
# cycletime = int(numofimg/16)
# ilm = "image"
# image_names = []#定义一个空列表
# for i in range(cycletime):
#     j=1
#     for j in range(1,17):
#         savename = ilm + str(i) + "_" + str(j) + ".png"
#         # print(savename)
#         image_names.append(savename)

# # 定义图像拼接函数
# def image_compose(image_names,i):
#     to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
#     # to_image.show()
#     # 循环遍历，把每张图片按顺序粘贴到对应位置上
#     for y in range(1, IMAGE_ROW + 1):
#         for x in range(1, IMAGE_COLUMN + 1):
#             from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
#                 (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
#             to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
#             # to_image.show()
#     to_image.save(IMAGE_SAVE_PATH+'/'+str(i)+'.png')  # 保存新图
#     to_image.show()

# for i in range(cycletime):
#     temp_image_names = image_names[i*16:(i+1)*16]
#     print(temp_image_names)
#     image_compose(temp_image_names,i)
# ###################################################################################################

from sklearn.metrics import f1_score
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import os  #通过os模块调用系统命令
import torch
from sklearn.metrics import confusion_matrix,precision_recall_curve,f1_score,roc_auc_score,auc,recall_score, auc,roc_curve,jaccard_score
from skimage.metrics import structural_similarity as ssim
# from libtiff import TIFF
import time

label_path = r'D:\\codeanddata\\code\\Swin-Unet\\label\\*.png'
pred_path = 'D:\\codeanddata\\code\\Swin-Unet\\predictions\\*.png'
#---------------------------------------------------------------------------#
filenames = glob.glob(label_path)
limit = len(filenames)
print("limit:",limit)
# y_true = np.zeros((limit,896,896),dtype=np.float32)
# y_pred = np.zeros((limit,896,896),dtype=np.float32)
# y_pred_auc = np.zeros((limit,896,896),dtype=np.float32)
y_true = np.zeros((limit,224,224),dtype=np.float32)
y_pred = np.zeros((limit,224,224),dtype=np.float32)
y_pred_auc = np.zeros((limit,224,224),dtype=np.float32)
c = 0
#---------------------------------------------------------------------------#
for i,img_path in enumerate(glob.glob(label_path)):
    print(i,img_path)
    label = cv2.imread(img_path)
    label = cv2.resize(label,dsize=(224,224),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("image", label)
    label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)    
    # image = cv2.resize(image,(pred_path,pred_path)) 
    label[label>=0.5] = 1
    label[label<0.5] = 0
    print(label.shape)

    pred_path = img_path[0:34] + 'predictions\\' + str(i) + '.png'
    # print(pred_path)
    pred = cv2.imread(pred_path)
    pred = cv2.resize(pred,dsize=(224,224),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    # cv2.imshow("image", pred)

    pred = cv2.cvtColor(pred,cv2.COLOR_BGR2GRAY) 
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    print(pred.shape)
    # print(f1_score(label, pred, average='binary'))

    # confusion = confusion_matrix(label, pred)
    # print(confusion)

    # tn, fp, fn, tp = confusion.ravel() 
    # F1_score = 2*tp/(2*tp+fn+fp) 
    # print("F1 score (F-measure): " + str(F1_score))
    #---------------------------------------------------------------------------#
    label_arr = np.asarray(label,dtype=np.float32)
    y_true[c,:,:] = label_arr 
    y_pred[c,:,:] = pred
    c = c +1
    
y_true = y_true.flatten()
y_pred = y_pred.flatten()
y_pred_auc = y_pred_auc.flatten()
confusion = confusion_matrix(y_true, y_pred)
print(confusion)

tn, fp, fn, tp = confusion.ravel()   
metric_cal = time.time()
if float(np.sum(confusion)) != 0:
    accuracy =  float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
print("Global Accuracy: " + str(accuracy))
specificity = 0
if float(confusion[0, 0] + confusion[0, 1]) != 0:
    specificity = tn / (tn + fp)
print("Specificity: " + str(specificity))
sensitivity = 0
if float(confusion[1, 1] + confusion[1, 0]) != 0:
    sensitivity = tp / (tp + fn) 
print("Sensitivity: " + str(sensitivity))

precision = 0
if float(confusion[1, 1] + confusion[0, 1]) != 0:
    precision = tp / (tp + fp) 
print("Precision: " + str(precision))


F1_score = 2*tp/(2*tp+fn+fp) 
print("F1 score (F-measure): " + str(F1_score))

AUC_ROC = roc_auc_score(y_true, y_pred)
print("AUC_ROC: " + str(AUC_ROC))

ssim = ssim(y_true, y_pred, data_range=y_true.max()-y_true.min())
print("SSIM: " + str(ssim))

meanIOU = jaccard_score(y_true,y_pred)
print("meanIOU: " + str(meanIOU))    


 
# y_pred = [0, 1, 1, 1, 2, 2]
# y_true = [0, 1, 0, 2, 1, 1]

# print(f1_score(y_true, y_pred, average='macro'))  
# print(f1_score(y_true, y_pred, average='weighted'))  