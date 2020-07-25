import os
import cv2
import numpy as np
import tkinter as tk
import random

''' 读取图片 ''' 
# Example: 
#       imread(r"C:\\imgfiles",rgb2bgr = False)
def imread(filePath, rgb2bgr = False):
    try:
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        if(rgb2bgr is True):
            cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img
    except Exception:
        print(Exception)
        return None

''' 显示图片,  '''
# Example: 
#       imshow("origin", origin_img)
def imshow(winname='default', img = None, mode = 'CENTER'):
    if(img is None):
        print("Image is None!")
        return 
    # 获取屏幕分辨率
    window = tk.Tk()
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    window.destroy()
    # 计算显示
    if mode == 'CENTER':
        Location = [int((screenwidth - img.shape[1] )/2),int((screenheight - img.shape[0] )/2)]
    elif mode == 'LTOP':
        Location = [0,0]
    elif mode == 'RTOP':
        Location = [screenwidth - img.shape[1],0]
    elif mode == 'LDOWN':
        Location = [0,screenheight - img.shape[0]]
    elif mode == 'RDOWN':
        Location = [screenwidth - img.shape[1],screenheight - img.shape[0]]
    else:
        print("Please Choose Location.")
        return
    cv2.imshow(winname,img)
    cv2.moveWindow(winname,Location[0],Location[1])

''' 裁剪图片 '''
# Example: 
# (1) Given specified size: 
#               resize_ing = resize(image, size = (width,height))
#      size为缩放尺寸,先宽后高
# (2) Given scale: 
#               resize_ing = resize(image, size = (scale_W,scale_H))
#      size为缩放比例,先宽后高
def resize(img,size= (0.1,0.1)):
    try:
        if(size[0] <= 1 and size[1] <= 1):
            resize_img = cv2.resize(img,dsize= (0,0),fx= size[0],fy= size[1])
        elif(size[0] >= 1 and size[1] >= 1):
            resize_img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        else:
            print("Illegal Size!")
            resize_img = None
        return resize_img
    except Exception:
        print(Exception)
        return None

''' 颜色空间转换'''
# Example: 
#       trans_img = transColor(origin_img,type= 'gray')
def transColor(img,type= 'gray'):
    try:
        if (type == 'gray'):
            trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif(type == 'rgb'):
            trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif(type == 'hsv'):
            # h: 色调/相 (Hue) s:饱和度 (Saturation) v: 明度 (Value)
            trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif(type == 'lab'):
            # l: 亮度  a和b为对立色，如 a: 红色到绿色，则b:黄色到蓝色
            trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif(type == 'yuv'):
            # y: 明亮度 (Luminance或Luma)  u、v: 色度(Chrominance或Chroma)
            trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return trans_img
    except Exception:
        print(Exception)
        return None


''' 颜色通道分离 '''
def split(img):
    channels = []
    try:
        for i in range(img.shape[2]):
            channels.append(cv2.split(img)[i])
    except:
        # 单通道 or 图像为None 直接进队返回
        channels.append(img)
    finally:
        return channels
    

''' 二值化 '''
# Example: 
#       gray_img = cv_black(gray_img, threshold = 110)
def cv_black(img, threshold = 110, mode= 'normal'):
    try:
        if(mode == 'otsu'):
            # Otsu阈值法
            thres, black_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        else:
            # 固定阈值
            thres, black_img = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
        return thres, black_img
    except Exception:
        print(Exception)
        return 0,None 

''' 获取ROI区域 '''
# Example: 
#       roi_img = cv_black(image, center = (rows,cols), size = (width,height) )
#  center 为待裁剪区域的中心坐标，先行后列
#   size  为待裁剪区域的尺寸, 先宽后高
def cv_roi(img, center = (100,100), size = (100,100)):
    try:
        roi_img = img[int((center[1] - size[1])/2):int((center[1] + size[1])/2),
                      int((center[0] - size[0])/2):int((center[0] + size[0])/2) ]
        return roi_img
    except Exception:
        print(Exception)
        return None 

''' 批量导入图片 '''
# Example: 
#       img_list,img_name_list = load_images(r"C:\\imgfile\\", 'jpg',100)
def load_images(path, format ='jpg',amout= 1):
    img_list = []
    img_name_list = []
    for filename in os.listdir(path):
          if(os.path.splitext(filename)[-1] == '.'+format):
                img_name_list.append(os.path.splitext(filename)[0] )
                img = cv_imread(os.path.join(path, filename))
                img_list.append(img)
                amout -= 1
                if amout <= 0:  # 控制读取图片的数量
                    break
    return img_list,img_name_list

''' 存储图片 '''
# Example: 
#       save_name = save(image, r"C:\\","name","jpg") '''
def save_images(img, path, name,format):
    number  = -1
    format = '.' + format
    filename = name + format
    file_name_tmp = os.path.join(path, filename)
    while(os.path.isfile(file_name_tmp)):
        number +=  1
        filename = name +str(number) + format
        file_name_tmp = os.path.join(path, filename)
    cv2.imwrite(file_name_tmp,img)
    return name +str(number)

'''椒盐噪声'''
# 
# prob:噪声比例,[0,1.0]
#
def sp_noise(image,prob):
    
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

'''椒盐噪声'''
# 
# mean:均值,sigma标准差
#
def gauss_noise(image, means=0, sigma=0.01):
    image = np.array(image/255, dtype=float) # 图像归一化
    noise = np.random.normal(means, sigma, image.shape)  # 产生高斯分布
    out_img = image + noise  # 叠加噪声
    # 检查是否越界
    if out_img.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    # 限幅
    out_img = np.clip(out_img, low_clip, 1.0)
    # 恢复灰度值范围为[0:255]
    out_img = np.uint8(out_img*255)
    return out_img