import os
import numpy as np
import csv
import pandas as pd
import torch
import torch.utils.data as data
import skimage
# import scipy.misc
import imageio
import cv2
import random
from numpy import *
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, cfg, train_mode=True):
        self.img_folder = cfg.img_folder
        self.csv_name = cfg.csv_name 
        self.is_train = train_mode
        self.inp_res = cfg.data_shape
        self.num_class = cfg.num_class
        self.cfg = cfg

        self.csv_folder=os.path.join(self.img_folder,'..')
        self.csv_path=os.path.join(self.csv_folder,self.csv_name)
        data = pd.read_csv(self.csv_path)

        self.img_path=data['image_path']
        self.labels=data['labels']
        if(self.is_train):
            print('trainset len = '.format(len(self.labels)))
        else:
            print('valset len = '.format(len(self.labels)))


    # def data_augmentation(self, img):
    #     flag=random.uniform(0, 4)
    #     if(flag>=1 and flag<2):
    #         img = cv2.flip(img , 1)

    #     if(flag>=2 and flag<3):
    #         height, width = img.shape[0], img.shape[1]
    #         center = (width / 2., height / 2.)
    #         angle = random.uniform(0, 45)
    #         if random.randint(0, 1):
    #                 angle *= -1
    #             rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    #             img = cv2.warpAffine(img, rotMat, (width, height))

    #     if(flag>=3 and flag<4):
    #         img = cv2.flip(img , 1)
    #         height, width = img.shape[0], img.shape[1]
    #         center = (width / 2., height / 2.)
    #         angle = random.uniform(0, 45)
    #         if random.randint(0, 1):
    #                 angle *= -1
    #             rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    #             img = cv2.warpAffine(img, rotMat, (width, height))
    #     return img
    
    def data_augmentation(self, img):
        flag=random.uniform(0, 2)
        if(flag>=1 and flag<2):
            height, width = img.shape[0], img.shape[1]
            center = (width / 2., height / 2.)
            angle = random.uniform(0, 15)
            if random.randint(0, 1):
                angle *= -1
                rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rotMat, (width, height))
        return img

    def __getitem__(self, index):
        a=self.img_path[index]
        img_path=os.path.join(self.csv_folder,a)
        #image = scipy.misc.imread(img_path, mode='RGB')
        image = imageio.imread(img_path,as_gray = True)
        label=self.labels[index]
        if self.is_train:
            image = self.data_augmentation(image) 
        image = np.array(image,'float32')
        image /= 255   

        image = cv2.resize(image,(self.inp_res))  

        #train dataset has (224,224) and (224,224,4),so we need to change them to (3,224,224) to fit the resnet   
        if len(image.shape) != 3:
            img=np.array([image for i in range(3)])
        else:
            img=image[:,:,0]
            img=np.array([image for i in range(3)])
            
        img = torch.from_numpy(img)
        
        return img,label

    def __len__(self):
        return len(self.labels)


        


