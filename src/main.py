from audioop import mul
import os
from tracemalloc import stop
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR, MultiStepLR
import sys
# import argparse
import numpy as np
# from PIL import Image
from MLPMixer import MLPMixerForImageClassification
from ConvMixer import ConvMixer
from ViT import VisionTransformer
from jittor.models.resnet import Resnet50
import shutil
from tqdm import tqdm
from jittor import Module
import pdb
import random
import math
from PIL import Image
from tensorboardX import SummaryWriter
import time
import smtplib, ssl
from trycourier import Courier

writer = SummaryWriter()
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
jt.flags.use_cuda=1
jt.set_global_seed(648)

def send_email_to_myself():
    # Install Courier SDK: pip install trycourier
    client = Courier(auth_token="pk_prod_FRV8652N0ZMXW1KPF236PV7ADNV1")
    resp = client.send_message(
        message={
            "to": {
            "email": "fan-xy19@mails.tsinghua.edu.cn"
            },
            "content": {
            "title": "Welcome to Courier!",
            "body": "Want to hear a joke? {{joke}}"
            },
            "data":{
            "joke": "Why does Python live on land? Because it is above C level"
            }
        }
    )

class SoftLabelCrossEntropyLoss(Module):
    def __init__(self):
        self.epsilon=0.1
        self.n=1020
        
    def execute(self, output:jt.Var, target:jt.Var):
        if target.ndim == 1:
            target = target.reshape((-1, ))
            target = target.broadcast(output, [1])
            target = target.index(1) == target
        target_weight = jt.ones(target.shape[0], dtype='float32')
        softTarget=target*(1-self.epsilon-self.epsilon/(self.n-1))+self.epsilon/(self.n-1)
        output = output - output.max([1], keepdims=True)
        logsum = output.exp().sum(1).log()
        loss:jt.Var = (logsum - (output*softTarget).sum(1)) * target_weight
        return loss.sum() / target_weight.sum()

def cutmix(batch:jt.Var, target:jt.Var,num_classes,p=0.5,alpha=1.0):
    if target.ndim == 1:
        target = np.eye(num_classes, dtype=np.float32)[target]
    if random.random() >= p:
        return batch, target
        
    batch_rolled = np.roll(batch, 1, 0)
    target_rolled = np.roll(target, 1,0)
    
    lambda_param =np.random.beta(alpha, alpha)
    W, H = batch.shape[-1], batch.shape[-2]
    # print(W,H)
    # pdb.set_trace()
    
    r_x = np.random.randint(W)
    r_y = np.random.randint(H)
    
    r = 0.5 * math.sqrt(1. - lambda_param)
    r_w_half = int(r * W)
    r_h_half = int(r * H)
    
    x1 = int(max(r_x - r_w_half, 0))
    y1 = int(max(r_y - r_h_half, 0))
    x2 = int(min(r_x + r_w_half, W))
    y2 = int(min(r_y + r_h_half, H))
    
    batch[:,:,y1:y2,x1:x2] = batch_rolled[:,:,y1:y2,x1:x2]
    lambda_param = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    
    target_rolled *= 1 - lambda_param
    target = target * lambda_param + target_rolled
    target=jt.float32(target)
    # print(type(target))
    # print(target[0][0:100])
    # pdb.set_trace()
    return batch, target

class RandomCutmix:
    def __init__(self, num_classes, p=0.5, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        
    def __call__(self, batch, target:jt.Var):
        if target.ndim == 1:
            target = np.eye(self.num_classes, dtype=np.float32)[target]
            
        if random.random() >= self.p:
            return batch, target
            
        batch_rolled = np.roll(batch, 1, 0)
        target_rolled = np.roll(target, 1,0)
        
        lambda_param =0.5# np.random.beta(self.alpha, self.alpha)
        W, H = batch.shape[-1], batch.shape[-2]
        
        r_x = np.random.randint(W)
        r_y = np.random.randint(H)
        
        r = 0.5 * math.sqrt(1. - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)
        
        x1 = int(max(r_x - r_w_half, 0))
        y1 = int(max(r_y - r_h_half, 0))
        x2 = int(min(r_x + r_w_half, W))
        y2 = int(min(r_y + r_h_half, H))
        
        batch[:,:,y1:y2,x1:x2] = batch_rolled[:,:,y1:y2,x1:x2]
        lambda_param = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        
        target_rolled *= 1 - lambda_param
        target = target * lambda_param + target_rolled
        
        return batch, target

def pretrain_one_epoch(model:nn.Module, train_loader, criterion, optimizer, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []
    for i, (images, labels) in enumerate(train_loader):
        images,labels=cutmix(images,labels,num_classes=1020)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == np.argmax(labels.data, axis=1))
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])
    scheduler.step()
    return round(total_acc/total_num,4),round(sum(losses)/len(losses),4)

def train_one_epoch(model:nn.Module, train_loader, criterion, optimizer, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []
    for i, (images, labels) in enumerate(train_loader):
        # images,labels=cutmix(images,labels,num_classes=1020)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        # acc = np.sum(pred == np.argmax(labels.data, axis=1))
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])
        if i%100==0:
            print(i,loss.data[0])
    scheduler.step()
    return round(total_acc/total_num,4),round(sum(losses)/len(losses),4)

def valid_one_epoch(model:nn.Module, val_loader):
    model.eval()
    total_acc = 0
    total_num = 0
    for i, (images, labels) in enumerate(val_loader):
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
    acc = total_acc / total_num
    return round(acc,4)

def calculate_test_set_accuracy(model:nn.Module, testLoader):
    model.eval()

    total_acc = 0
    total_num = 0
    # pbar = tqdm(testLoader, desc="calculate_test_set_accuracy")
    for index,(images, labels) in enumerate(testLoader):        
        output = model(images)
        pred = np.argmax(output.data, axis=1)
        acc = np.sum(pred == labels.data)
        total_acc += acc
        total_num += labels.shape[0]
        # pbar.set_description(f'acc={total_acc / total_num:.4f}')
        # if index%200==0:
            # print(index,"test accuracy",round(total_acc/total_num,4))
    acc = total_acc / total_num
    return round(acc,4)

def train(model:nn.Module,modelName,learningRate,epochs,etaMin,imageSize,savedName):
    startTime=time.time()
    # region Processing data 
    # resizedImageSize=imageSize
    # croppedImagesize=imageSize-32
    data_transforms = {
        'train': transform.Compose([
            transform.Resize((imageSize,imageSize)),
            # transform.RandomCrop((croppedImagesize, croppedImagesize)),       # 从中心开始裁剪
            transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transform.RandomRotation(90),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transform.Compose([
            transform.Resize((imageSize, imageSize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize((imageSize, imageSize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }
    print("Loading data...")
    batch_size = 64
    data_dir = '../data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                    ['train', 'valid', 'test']}
    traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
    validdataset = image_datasets['valid'].set_attrs(batch_size=batch_size, shuffle=False)
    testDataSet = image_datasets['test'].set_attrs(batch_size=batch_size, shuffle=False)
    # endregion

    # region model and optimizer
    # criterion = SoftLabelCrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, 15, eta_min=etaMin)
    # endregion

    # region train and valid
    print("----------------- A new trial ---------------------")
    print("modelName",modelName,"learning rate:",learningRate,"etaMin",etaMin,"imgSize",imageSize,"savedName",savedName,\
        "criterion","CrossEntropyLoss","weight_decay",1e-3,"TMax",15)
    with open("../result/summary/"+modelName+".txt","a") as f:
        f.write("modelName: "+modelName+"  learning rate:"+str(learningRate)+"  etaMin:"+str(etaMin)+"  imgSize:"+str(imageSize)+\
            "  savedName:"+savedName+"  criterion:"+"  CrossEntropyLoss"+"  weight_decay:"+str(1e-3)+"  TMax:"+str(15)+"\n")
    maxBearableEpochs=50
    noProgressEpochs=0
    stopEpoch=0
    currentBestAccuracy=0.0
    currentBestEpoch=0
    for epoch in range(epochs):
        trainAccuracy,trainLoss=train_one_epoch(model, traindataset, criterion, optimizer, 1, scheduler)
        endTime=time.time()
        print("Time after train one epoch:",endTime-startTime)
        validAccuracy=valid_one_epoch(model, validdataset)
        endTime=time.time()
        print("Time after valid one epoch:",endTime-startTime)
        print("epoch:",epoch,"validAccuracy:",validAccuracy,"trainAccuracy:",trainAccuracy,"delta:",round(validAccuracy-currentBestAccuracy,4))
        writer.add_scalars(modelName,{'trainAccuracy':trainAccuracy,'validAccuracy':validAccuracy,"trainLoss":trainLoss}, epoch)
        if validAccuracy > currentBestAccuracy:
            currentBestAccuracy=validAccuracy
            currentBestEpoch=epoch
            model.save("../weight/"+modelName+"/"+savedName)
            noProgressEpochs=0
        else:
            noProgressEpochs+=1
            if noProgressEpochs>=maxBearableEpochs:
                stopEpoch=epoch
                break
        stopEpoch=epoch
    testAccuracy=valid_one_epoch(model,testDataSet)
    endTime=time.time()
    print("Time used:",endTime-startTime)
    print("==========================================================================================")
    print("testAccuracy",testAccuracy,"validAccuracy",currentBestAccuracy,"bestEpoch",currentBestEpoch,"stopEpoch",stopEpoch)
    print("==========================================================================================")
    send_email_to_myself()
    # endregion

def pretrain(model:nn.Module,modelName,learningRate,epochs,etaMin,savedName):
    # region Processing data 
    resizedImageSize = 256
    croppedImagesize=resizedImageSize-32
    data_transforms = {
        'pretrain': transform.Compose([
            transform.Resize((resizedImageSize,resizedImageSize)),
            transform.RandomCrop((croppedImagesize, croppedImagesize)),       # 从中心开始裁剪
            transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transform.RandomRotation(90),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize((croppedImagesize, croppedImagesize)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }
    batch_size = 64
    data_dir = '../data'
    image_datasets = {x: jt.dataset.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                    ['pretrain', 'valid', 'test']}
    traindataset = image_datasets['pretrain'].set_attrs(batch_size=batch_size, shuffle=True)
    # endregion
    
    # region hyper parameters
    criterion = SoftLabelCrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)    
    scheduler = CosineAnnealingLR(optimizer, 15, eta_min=etaMin)
    # endregion

    # region pretrain
    print("----------------- A new trial ---------------------")
    print("modelName",modelName,"learning rate:",learningRate,"epochs",epochs,"etaMin",etaMin,"savedName",savedName)
    currentBestAccuracy=0.0
    for epoch in range(epochs):
        trainAccuracy,trainLoss=pretrain_one_epoch(model, traindataset, criterion, optimizer, 1, scheduler)
        print("epoch:",epoch,"trainAccuracy:",trainAccuracy,"delta:",round(trainAccuracy-currentBestAccuracy,4))
        writer.add_scalars(modelName,{'trainAccuracy':trainAccuracy,"trainLoss":trainLoss}, epoch)
        if trainAccuracy > currentBestAccuracy:
            currentBestAccuracy=trainAccuracy
        if epoch%20==0:    
            model.save("../model/"+modelName+"/"+savedName)
    # endregion


if __name__=="__main__":
    # send_email_to_myself()
    # vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=5001)
    # train(model=vitModel,modelName="ViT",learningRate=5e-5,epochs=300,etaMin=1e-5,imageSize=224,savedName="1.pkl",)

    mlpMixerModel=MLPMixerForImageClassification(
        in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=5001,image_size=224,dropout=0)
    train(model=mlpMixerModel,modelName="MLPMixer",learningRate=2.3e-5,epochs=200,etaMin=1e-5,imageSize=224,savedName="1.pkl")
    # print(sys.argv)
    # if sys.argv[1]=="ViT":
    #     print("ViT")
    #     vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=5001)
    #     train(model=vitModel,modelName="ViT",learningRate=5e-5,epochs=300,etaMin=1e-5,imageSize=224,savedName=sys.argv[2]+".pkl",)
    # elif sys.argv[1]=="MLPMixer":
    #     mlpMixerModel=MLPMixerForImageClassification(
    #         in_channels=3,patch_size=16, d_model=512, depth=12, num_classes=5001,image_size=224,dropout=0)
    #     # print(mlpMixerModel)
    #     # print(mlpMixerModel._modules.keys())
    #     train(model=mlpMixerModel,modelName="MLPMixer",learningRate=2.3e-5,epochs=200,etaMin=1e-5,imageSize=224,savedName=sys.argv[2]+".pkl")
    # elif sys.argv[1]=="ConvMixer":
    #     convMixerModel=ConvMixer(dim = 768, depth = 32, kernel_size=7, patch_size=7,n_classes=5001)
    #     train(model=convMixerModel,modelName="ConvMixer",learningRate=1e-4,epochs=500,etaMin=1e-7,imageSize=224,savedName=sys.argv[2]+".pkl")
    # elif sys.argv[1]=="ResNet":
    #     resnetModel=Resnet50(num_classes=5001)
    #     train(model=resnetModel,modelName="ResNet",learningRate=1e-4,epochs=1600,etaMin=1e-7,imageSize=224,savedName=sys.argv[2]+".pkl")
    
    writer.close()
