import numpy as np
import cv2
import copy
import cv_joke as cv_jk
import scipy.signal as sci
import matrix as mt
import random 
import time

'''将默认值为0的内核的某一区域全置为1'''
def ones_kernel(kernel,size=(1,1),loc=(0,0),value = 1):
    tmp = np.ones(size)
    kernel_tmp = copy.deepcopy(kernel)
    kernel_tmp[loc[0]:(loc[0]+size[0]),loc[1]:(loc[1]+size[1])] = tmp
    return kernel_tmp

'''Side Mean Filter实现'''
def s_meanfilter(img,radius,iteration = 1):
    r = radius
    zero_kernel = np.zeros([2*r+1,2*r+1])
    k_L = ones_kernel(zero_kernel,size= (2*r+1,r+1),loc= (0,0))/((2*r+1)*(r+1))
    k_R = ones_kernel(zero_kernel,size= (2*r+1,r+1),loc= (0,r))/((2*r+1)*(r+1))
    K_U = k_L.T
    k_D = K_U[::-1]
    k_NW = ones_kernel(zero_kernel,size= (r+1,r+1),loc= (0,0))/((r+1)*(r+1))
    k_NE = ones_kernel(zero_kernel,size= (r+1,r+1),loc= (0,r))/((r+1)*(r+1))
    k_SW = k_NW[::-1]
    k_SE = k_NE[::-1]
    kernels = [k_L,k_R,K_U,k_D,k_NW,k_NE,k_SW,k_SE]

    m = img.shape[0]+2*r
    n = img.shape[1]+2*r
    dis = np.zeros([8,m,n]);
    result = copy.deepcopy(img)
    
    for ch in range(img.shape[2]):
        U = np.pad(img[:,:,ch],(r,r),'edge');
        for i in range(iteration):
            for id,kernel in enumerate(kernels):
                conv2 = sci.correlate2d(U,kernel,'same')
                dis[id] = conv2 - U
            U = U + mt.mat_absmin(dis)
        result[:,:,ch] = U[r:-r,r:-r]
    return result

'''滑动内核与矩阵星乘取中值'''
def mid_mult(img,kernel,r,start_offset,end_offset):
    result = []
    for row in range(start_offset[0],img.shape[0]-kernel.shape[0]+1-end_offset[0]):
        for col in range(start_offset[1],img.shape[1]-kernel.shape[1]+1-end_offset[1]):
            #img_roi = copy.deepcopy(img[row:row+kernel.shape[0],col:col+kernel.shape[1]])
            img_roi = img[row:row+kernel.shape[0],col:col+kernel.shape[1]]
            mid_tmp = np.median(img_roi*kernel)
            result.append(mid_tmp)
    result = np.reshape(np.array(result),(img.shape[0]-2*r,img.shape[1]-2*r))
    result = np.pad(result,(r,r),'edge');
    return result

'''Side Median Filter实现'''
def s_medianfilter(img,radius,iteration = 1):
    r = radius
    # 异型内核
    k_L = np.ones((2*r+1,r+1))
    k_R = k_L
    k_U = k_L.T
    k_D = k_U
    k_NW = np.ones((r+1,r+1))
    k_NE = k_NW
    k_SW = k_NW
    k_SE = k_NW
    kernels = [k_L,k_R,k_U,k_D,k_NW,k_NE,k_SW,k_SE]
    start_offsets = [(0,0),(0,r),(0,0),(r,0),(0,0),(0,r),(r,0),(r,r)]
    end_offsets = [(0,r),(0,0),(r,0),(0,0),(r,r),(r,0),(0,r),(0,0)]
    
    m = img.shape[0]+2*r
    n = img.shape[1]+2*r
    dis = np.zeros([8,m,n]);
    result = copy.deepcopy(img)
    for ch in range(img.shape[2]):
        U = np.pad(img[:,:,ch],(r,r),'edge');
        for i in range(iteration):
            for id in range(len(kernels)): 
                mid_result = mid_mult(U,kernels[id],r,start_offsets[id],end_offsets[id])
                dis[id] = mid_result - U
            U = U + mt.mat_absmin(dis)
        result[:,:,ch] = U[r:-r,r:-r]
    return result

'''将内核的某一区域保留,其余置0'''
def zeros_kernel(kernel,size=(1,1),loc=(0,0)):
    kernel_tmp = np.zeros(kernel.shape)
    kernel_tmp[loc[0]:(loc[0]+size[0]),loc[1]:(loc[1]+size[1])] = kernel[loc[0]:(loc[0]+size[0]),loc[1]:(loc[1]+size[1])]
    kernel_tmp = kernel_tmp/np.sum(kernel_tmp)
    return kernel_tmp

'''Side Gaussian Filter实现'''
def s_gausfilter(img,radius,sigma = 0,iteration = 1):
    r = radius
    gaus_kernel = cv2.getGaussianKernel(2*r+1,sigma) # sigma = ((n-1)*0.5 - 1)*0.3 + 0.8
    gaus_kernel = gaus_kernel.dot(gaus_kernel.T)
    gaus_kernel = gaus_kernel.astype(np.float)
    k_L = zeros_kernel(gaus_kernel,size= (2*r+1,r+1),loc= (0,0))
    k_R = zeros_kernel(gaus_kernel,size= (2*r+1,r+1),loc= (0,r))
    K_U = k_L.T
    k_D = K_U[::-1]
    k_NW = zeros_kernel(gaus_kernel,size= (r+1,r+1),loc= (0,0))
    k_NE = zeros_kernel(gaus_kernel,size= (r+1,r+1),loc= (0,r))
    k_SW = k_NW[::-1]
    k_SE = k_NE[::-1]
    kernels = [k_L,k_R,K_U,k_D,k_NW,k_NE,k_SW,k_SE]
    m = img.shape[0]+2*r
    n = img.shape[1]+2*r
    dis = np.zeros([8,m,n]);
    result = copy.deepcopy(img)
    for ch in range(img.shape[2]):
        U = np.pad(img[:,:,ch],(r,r),'edge');
        for i in range(iteration):
            for id,kernel in enumerate(kernels):
                conv2 = sci.correlate2d(U,kernel,'same')
                dis[id] = conv2 - U
            U = U + mt.mat_absmin(dis)
        result[:,:,ch] = U[r:-r,r:-r]
    return result

'''测试1'''
# image = cv_jk.imread('image\cat.jpg')
# resizeimg = cv_jk.resize(image,(0.8,0.8))
# noise_img = cv_jk.sp_noise(resizeimg,0.02)
# # 均值滤波
# img_mean = cv2.blur(noise_img, (7,7))
# img_mean_5 = noise_img
# img_mean_10 = noise_img
# start = time.process_time()
# for i in range(5):
#     img_mean_5 = cv2.blur(img_mean_5, (7,7))
# for i in range(10):
#     img_mean_10 = cv2.blur(img_mean_10, (7,7))
# print((time.process_time()-start)/16)
# start = time.process_time()
# # S-均值滤波
# result = s_meanfilter(noise_img,3,1)
# result_5 = s_meanfilter(noise_img,3,5)
# result_10 = s_meanfilter(noise_img,3,10)
# print((time.process_time()-start)/16)
# # cv_jk.imshow('noise_img',noise_img)
# # cv_jk.imshow('img_mean',img_mean)
# # cv_jk.imshow('result',result)

# cv_jk.save_images(resizeimg,r'.\result','source_img','jpg')
# cv_jk.save_images(noise_img,r'.\result','noise_img','jpg')
# cv_jk.save_images(img_mean,r'.\result','img_mean','jpg')
# cv_jk.save_images(img_mean_5,r'.\result','img_mean_5','jpg')
# cv_jk.save_images(img_mean_10,r'.\result','img_mean_10','jpg')
# cv_jk.save_images(result,r'.\result','result','jpg')
# cv_jk.save_images(result_5,r'.\result','result_5','jpg')
# cv_jk.save_images(result_10,r'.\result','result_10','jpg')

# cv2.waitKey(0)

# '''测试2'''
# image = cv_jk.imread('image\cat.jpg')
# resizeimg = cv_jk.resize(image,(0.8,0.8))
# noise_img = cv_jk.sp_noise(resizeimg,0.02)
# # 中值滤波
# img_median = cv2.medianBlur(noise_img, 7)
# img_median_5 = noise_img
# img_median_10 = noise_img
# start = time.process_time()
# for i in range(5):
#     img_median_5 = cv2.medianBlur(img_median_5, 7)
# for i in range(10):
#     img_median_10 = cv2.medianBlur(img_median_10, 7)
# print((time.process_time()-start)/16)
# start = time.process_time()
# # S-中值滤波
# result = s_medianfilter(noise_img,3,1)
# print('1')
# result_5 = s_medianfilter(noise_img,3,5)
# print('5')
# result_10 = s_medianfilter(noise_img,3,10)
# print('10')
# print((time.process_time()-start)/16)
# # cv_jk.imshow('noise_img',noise_img)
# # cv_jk.imshow('img_median',img_median)
# # cv_jk.imshow('result',result)

# # cv_jk.save_images(resizeimg,r'.\result','source_img','jpg')
# # cv_jk.save_images(noise_img,r'.\result','noise_img','jpg')
# cv_jk.save_images(img_median,r'.\result','trad_median','jpg')
# cv_jk.save_images(img_median_5,r'.\result','trad_median_5','jpg')
# cv_jk.save_images(img_median_10,r'.\result','trad_median_10','jpg')
# cv_jk.save_images(result,r'.\result','s-median_result','jpg')
# cv_jk.save_images(result_5,r'.\result','s-median_result_5','jpg')
# cv_jk.save_images(result_10,r'.\result','s-median_result_10','jpg')

'''测试3'''
image = cv_jk.imread('image\moon.jpg')
resizeimg = cv_jk.resize(image,(0.3,0.3))
noise_img = cv_jk.sp_noise(resizeimg,0.02)
# 高斯滤波
img_gaussion = cv2.GaussianBlur(noise_img,(15,15),0.01)
# S-高斯滤波
result = s_gausfilter(noise_img,7)
cv_jk.imshow('source_img',noise_img)
cv_jk.imshow('img_gaussion',img_gaussion)
cv_jk.imshow('result',result)
cv2.waitKey(0)


'''调试'''
# img_array = np.array([[[0],[1],[2]],[[0],[1],[2]],[[0],[1],[2]]])
# s_gausfilter(img_array,1)




# img_array = np.array([[[0,0,0],[1,1,1]],[[0,0,0],[1,1,1]]])
# result = s_boxfilter(img_array,1,1)

# Hello~你好请问可以请教一下关于您发表的边窗滤波的一点问题吗?
# 就是我用 python 复现了一下文中提到的 S-BOX filter, 虽然最后出来的结果还可以, 但是运行速度很慢
# 在这个过程中其实我怀疑我对这个算法的思想还是拿捏的不准确, 所以想请教你一下~
# 我的理解是生成的 Kernel 大小是(2r+1,2r+1),采取不同方向的时候, 需要对Kernel内容进行调整
# 比如 k_L 就需要将右半部分置0,剩下的左半部分内核取均值，而 k_NW 这样斜向的就只保留角落的一块区域并取均值。
# 然后用这8个(2r+1,2r+1)的核去与图像作卷积，取 dis 最小的卷积结果作为最后该像素点的像素值
