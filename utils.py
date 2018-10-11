import os
import time
import numpy as np
import operator
import math
from scipy import optimize
from scipy.stats import gmean
from skimage.util import random_noise
from skimage import transform
import random
from skimage.io import imread
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
import cv2
import json
from glob import glob

def prcocess_tiff(s_dir, d_dir_train, d_dir_val, bl = 200, wl = 3840):
    file_names = glob(os.path.join(s_dir, '*.tiff'))
    for file_name in file_names:
        print ('processing file: ', file_name)
        img = cv2.imread(file_name, -1)
        img = (img - bl) / (wl - bl)
        img *= 255.0
        coin = random.random()
        if coin > 0.3:
            save_name = os.path.join(d_dir_train, file_name.split('/')[-1][:-5] + '.png')
        else:
            save_name = os.path.join(d_dir_val, file_name.split('/')[-1][:-5] + '.png')
        # print (save_name)
        cv2.imwrite(save_name, img)


def batch_stable_process(img_batch, use_crop, use_clip, use_flip, use_rotate, use_noise):
    b,h,w,_ = img_batch.shape
    img_batch_after = []
    for index in range(b):
        img = img_batch[index,...]
        if use_crop:
            img = random_crop(img)
        if use_clip:
            img = random_clip(img)
        if use_flip:
            img =  random_flip(img)
        if use_rotate:
            img = random_rotate(img)
        if use_noise:
            img = random_add_noise(img)
        img_batch_after.append(img)
    img_batch_after = np.stack(img_batch_after, axis = 0)
    return img_batch_after

def random_crop(img, size = None):
    h,w, _ = img.shape
    if size is None:
        size = int(np.random.uniform(0.6, 1) * h)
    start_y = np.random.randint(0, h-size)
    start_x = np.random.randint(0, w-size)
    img_tmp = img[start_y: size+ start_y, start_x: start_x + size]
    return cv2.resize(img_tmp, (h,w))

def random_clip(img, rate = None):
    if rate is None:
        rate = np.random.uniform(0,0.2)
    img_max = img.max()
    img_min = img.min()
    img = np.clip(img, img_min * (1+rate), img_max * (1 - rate))
    return img

def random_flip(img, flag = None):
    if flag is None:
        flag = np.random.randint(0,2)
    if flag > 0:
        img = img[:, ::-1]
    return img

def random_rotate(img, angle = None):
    if angle is None:
        angle = np.random.randint(10,60)
    h,w,_ = img.shape
    img = transform.rotate(img, angle)
    return img

def random_add_noise(img, var = None):
    # in skiiamge_utils
    if var is None:
        var = 0.005
    mean = 0.0
    gau_img = random_noise(img, mean = mean, var = var) # defalu mode is gaussian
    pos_img = random_noise(gau_img, mode = 'poisson')
    return pos_img


def np_convolve(input, filts, final_K, final_W, spatial=True):
    kpad = final_K//2
    sh = input.shape
    ch = sh[-1]
    initial_W = ch
    h, w = sh[1], sh[2]
    input = np.pad(input, [[0, 0], [kpad, kpad], [
                   kpad, kpad], [0, 0]], mode='constant')
    img_stack = []
    for i in range(final_K):
        for j in range(final_K):
            img_stack.append(input[:, i:h+i, j:w+j, :])
    img_stack = np.stack(img_stack, axis=-2)  # [batch, h, w, K**2, ch]

    A = np.reshape(img_stack, [sh[0], h, w, final_K**2 * ch, 1])

    fsh = filts.shape
    x = np.reshape(filts, [fsh[0], fsh[1] if spatial else 1, fsh[2] if spatial else 1,
                           final_K ** 2 * initial_W, final_W])

    return np.sum(A * x, axis=-2)


def special_downsampling(img, scale):
    h,w,c = img.shape
    h_down, w_down = h // scale, w // scale
    img_down = np.ones([h_down, w_down, c])
    for j in range(h_down):
        for i in range(w_down):
            cut_out = img[j* scale: (j+1) *scale, i*scale:(i+1)*scale,:]
            value = np.mean(np.mean(cut_out, axis = 0), axis = 0)
            img_down[j,i,:] = value
    return img_down



if __name__ == '__main__':
    s_dir = '../Downloads/Sony'
    d_dir_train = './data/sony/train'
    d_dir_val = './data/sony/val'
    if not os.path.exists(d_dir_train):
        os.mkdir(d_dir_train)
    if not os.path.exists(d_dir_val):
        os.mkdir(d_dir_val)
    prcocess_tiff(s_dir, d_dir_train, d_dir_val)
