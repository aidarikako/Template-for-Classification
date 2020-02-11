import os
import os.path
import sys
import numpy as np

class train_Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    

    img_folder=os.path.join(root_dir, 'trainset', 'images')
    csv_name='train.csv'
    data_shape=(224,224)
    num_class=4
    batch_size=2
    epochs=12
    epoch_delay=4

train_cfg = train_Config()

class val_Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    

    img_folder=os.path.join(root_dir, 'valset', 'images')
    csv_name='test.csv'
    data_shape=(224,224)
    num_class=4
    batch_size=1

val_cfg = val_Config()

class test_Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')
    
    checkpoint_path=os.path.join(cur_dir,'checkpoint')
    img_folder=os.path.join(root_dir, 'testset', 'images')
    csv_name='upload.csv'
    data_shape=(224,224)
    num_class=4
    batch_size=4

test_cfg = test_Config()