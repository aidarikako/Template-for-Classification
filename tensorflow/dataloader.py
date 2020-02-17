import os
import tensorflow as tf 
import pandas as pd
import random

class MyDataset():
    def __init__(self, cfg, train_mode=0):
        self.img_folder = cfg.img_folder
        self.csv_name = cfg.csv_name 
        self.is_train = train_mode
        self.inp_res = cfg.data_shape
        self.num_class = cfg.num_class
        self.cfg = cfg

        self.csv_folder=os.path.join(self.img_folder,'..')
        self.csv_path=os.path.join(self.csv_folder,self.csv_name)
        self.data = pd.read_csv(self.csv_path)
       
        z=self.data['image_path'].values.tolist()
        self.data_list=[os.path.join(self.csv_folder,z1) for z1 in z]

    
    def preprocess(self,x,y):
        x = tf.io.read_file(x)
        x = tf.image.decode_jpeg(x,channels=3)
        x = tf.image.resize(x,self.inp_res)
        x = tf.cast(x,dtype=tf.float32)/255
       #y = tf.one_hot(y,depth=self.num_class)
        y = tf.convert_to_tensor(y)           
        return x,y

    def get_dataset(self):

        if(self.is_train==0):
            print('trainset len:{}'.format(len(self.data_list)))
            self.label_list = self.data['labels'].values.tolist()
            train_dataset = tf.data.Dataset.from_tensor_slices((self.data_list,self.label_list))
            train_dataset = train_dataset.shuffle(1024).map(self.preprocess).batch(self.cfg.batch_size).repeat()
            return train_dataset,len(self.data_list)

        elif(self.is_train==1):
            print('valset len:{}'.format(len(self.data_list)))
            self.label_list = self.data['labels'].values.tolist()
            val_dataset = tf.data.Dataset.from_tensor_slices((self.data_list,self.label_list))
            val_dataset = val_dataset.map(self.preprocess).batch(self.cfg.batch_size)
            return val_dataset,len(self.data_list)

        else:
            print('testset len:{}'.format(len(self.data_list)))
            self.label_list = self.data['labels'].values.tolist()
            test_dataset = tf.data.Dataset.from_tensor_slices((self.data_list,self.label_list))
            test_dataset = test_dataset.map(self.preprocess).batch(self.cfg.batch_size)
            # test_dataset = tf.data.Dataset.from_tensor_slices((self.data_list))
            # test_dataset = test_dataset.map(self.preprocess).batch(self.cfg.batch_size)
            return test_dataset,len(self.data_list)

    
