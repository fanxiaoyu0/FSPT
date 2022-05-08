from jittor import transform
import jittor as jt
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

filename = ['data/1.tif', 'data/2.tif', 'data/3.tif', 'data/4.tif', 'data/5.jpg']
num_eachpic = 1000
num_base = 0

if __name__ == '__main__':
    jt.set_global_seed(648)
    
    for aPic in filename:
        # 数据输入
        img = Image.open(aPic)
        '''
        print(img.size)        #(1760 x 2640)      (1904, 4096)
        input()
        '''
        # 定义并应用变换，得到一个新图片
        data_transforms_1 = transform.Compose([
            transform.RandomResizedCrop(size = (224, 224), scale = (0.05, 1.0))       
        ])
        
        count = 0
        for i in range(int(img.size[1]/112)):
            for j in range(int(img.size[0]/112)):
                height = 224
                width = 224
                if i*112 + height > img.size[1]:
                    height = img.size[1] - i*112
                if j*112 + width > img.size[0]:
                    width = img.size[0] - j*112
                tmp = transform.crop(img, top = i*112, left = j*112, height = height, width = width) 
                tmp = transform.resize(tmp, size=(224, 224))
                tmp.save("result/{}.png".format(count + num_base),quality = 0.95)
                count += 1
        
        for i in range(num_eachpic):
            tmp = data_transforms_1(img)
            tmp.save("result/{}.png".format(i + count + num_base),quality = 0.95)
            if i + count >= 1000:
                break
        
        num_base += 1000
