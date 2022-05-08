#from jittor import transform
#import jittor as jt
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageEnhance
import os
import random
import numpy as np
import copy

origin_name = 'dataset/'
dst_name = 'class_dataset/'

def save_photo(dst_dir_path, img2save):
    global transform_count
    # 保存图片
    if type(img2save) == np.ndarray:
        img2save = Image.fromarray(img2save)
    
    img2save.save(dst_dir_path + "/{}.png".format(transform_count), quality = 0.95)
    
    transform_count += 1
    pass

if __name__ == '__main__':
    random.seed(666)
    np.random.seed(233)
    
    for i in tqdm(range(5001)):
        global transform_count
        transform_count = 0
        # 新建class文件夹
        class_dir_path = dst_name + '{}'.format(i)
        os.mkdir(class_dir_path)
        #-------------------------------------------------------------------------------------
        # 对img应用各个变换，每次变换完之后保存
        # 遮盖相关
        # 遮盖50%全通道
        # 粒度最小
        img = np.array(Image.open(origin_name + '{}.png'.format(i)))
        img_change = copy.deepcopy(img)
        for j in range(224):
            ran_grid = random.sample(range(1, 224), 112)         # 每行遮盖一半，使遮盖不集中在某些区域
            for k in ran_grid:
                img_change[j][k] = [0, 0, 0]
        save_photo(class_dir_path, img_change)
        
        # 粒度大一倍
        img_change = copy.deepcopy(img)
        for j in range(112):
            ran_grid = random.sample(range(1, 112), 66)         # 每两行遮盖一半，使遮盖不集中在某些区域
            for k in ran_grid:
                img_change[2*j][2*k] = [0, 0, 0]
                img_change[2*j+1][2*k] = [0, 0, 0]
                img_change[2*j][2*k+1] = [0, 0, 0]
                img_change[2*j+1][2*k+1] = [0, 0, 0]
        save_photo(class_dir_path, img_change)
        
        # 粒度再大一倍
        img_change = copy.deepcopy(img)
        for j in range(66):
            ran_grid = random.sample(range(1, 66), 33)         # 每两行遮盖一半，使遮盖不集中在某些区域
            for k in ran_grid:
                for row_bia in range(4):
                    for col_bia in range(4):
                        if 4*j + row_bia >= 224 or 4*k + col_bia >= 224:
                            continue
                        img_change[4*j + row_bia][4*k + col_bia] = [0, 0, 0]
        save_photo(class_dir_path, img_change)
        
        # 依次遮盖三个通道
        for channel in range(3):
            img_change = copy.deepcopy(img)
            for j in range(224):
                for k in range(224):
                    img_change[j][k][channel] = 0
            save_photo(class_dir_path, img_change)
        
        # 旋转随机角度
        img = Image.open(origin_name + '{}.png'.format(i))
        rotate_angle = random.randint(1, 3)
        img_change = copy.deepcopy(img)
        img_change = img_change.rotate(rotate_angle * 90, expand = 1)
        save_photo(class_dir_path, img_change)
        
        # 随机上下或左右对称翻转
        flip_direc = random.randint(1, 2)
        img_change = copy.deepcopy(img)
        if flip_direc == 1:
            img_change = img_change.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            img_change = img_change.transpose(Image.FLIP_LEFT_RIGHT)
        save_photo(class_dir_path, img_change)
        
        # 在每个维度上增加噪声
        # 椒盐噪声
        salt = 0.9
        pepper = 0.1
        img_change = np.array(Image.open(origin_name + '{}.png'.format(i)))
        for channel in range(3):
            for j in range(224):
                for k in range(224):
                    r = random.random()
                    if r > salt:
                        img_change[j][k][channel] = 255
                    elif r < pepper:
                        img_change[j][k][channel] = 0
        save_photo(class_dir_path, img_change)
        
        # 白噪声
        img_change = np.array(Image.open(origin_name + '{}.png'.format(i)))
        noise = 255 * np.random.normal(0, 0.15, size=(224, 224, 3))
        for channel in range(3):
            for j in range(224):
                for k in range(224):
                    color_after = img_change[j][k][channel] + noise[j][k][channel]
                    if color_after > 255:
                        color_after = 255
                    if color_after < 0:
                        color_after = 0
                    img_change[j][k][channel] = color_after
        save_photo(class_dir_path, img_change)
        
        # 亮度
        img = Image.open(origin_name + '{}.png'.format(i))
        light = 1.25
        imageLight = (ImageEnhance.Brightness(img)).enhance(light)
        save_photo(class_dir_path, imageLight)
        dark = 0.75
        imageDark = (ImageEnhance.Brightness(img)).enhance(dark)
        save_photo(class_dir_path, imageDark)
        
        #饱和度
        light = 3
        imageLight = (ImageEnhance.Color(img)).enhance(light)
        save_photo(class_dir_path, imageLight)
        dark = 0.5
        imageDark = (ImageEnhance.Color(img)).enhance(dark)
        save_photo(class_dir_path, imageDark)
        
        # 锐化程度
        light = 100
        imageLight = (ImageEnhance.Sharpness(img)).enhance(light)
        save_photo(class_dir_path, imageLight)
        dark = 0
        imageDark = (ImageEnhance.Sharpness(img)).enhance(dark)
        save_photo(class_dir_path, imageDark)
